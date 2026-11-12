"""A conservative, reduced-order paddy water kernel for research scenarios.

This is an explicit modelling proposal, not a calibrated Hongmiao field model
and not an exact implementation of the FAO-56 dual crop-coefficient method.

State B is the combined root-zone and surface store, expressed as an equivalent
water depth over field area. The assumed instantaneous homogeneous equilibrium
fills a root zone to saturation before retaining a surface layer. This removes
an independent infiltration timescale: it is a declared reduced-order closure,
not a claim that real fields equilibrate immediately. Soil moisture may fall
below wilting point; transpiration then stops, while the assumed surface
evaporation closure may continue to withdraw remaining water.

The nonlinear percolation and evaporation functions, theta_crit, and default
parameter values require sensitivity analysis or local calibration. Percolation
and overflow leave the represented field. Neither is automatically returned to
a local pond or another area. A larger environment must account for such flows
explicitly if it chooses to route them.

Within each substep: add supplied rain/irrigation, overflow, evaluate stress and
wetness, withdraw transpiration, withdraw evaporation, then drain. Rain and
irrigation arguments are integrated depths for that substep; ET0 is a daily
rate. No action-selection or allocation policy is implemented here.
"""

from dataclasses import dataclass
import math
from numbers import Real


def _number(name: str, value: float, *, minimum: float = 0.0,
            strict_minimum: bool = False) -> None:
    # The inner kernel mostly receives built-in numbers. Avoid millions of
    # abstract-number subclass checks while retaining the same validation for
    # NumPy scalars, other Real implementations and forbidden booleans.
    if type(value) not in (int, float) and (isinstance(value, bool) or not isinstance(value, Real)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < minimum or (strict_minimum and value == minimum):
        comparison = ">" if strict_minimum else ">="
        raise ValueError(f"{name} must be {comparison} {minimum}")


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class FieldParams:
    area_ha: float
    soil_depth_mm: float = 300.0
    theta_sat: float = 0.50
    theta_fc: float = 0.39
    theta_wp: float = 0.23
    theta_crit: float = 0.40
    h_max_mm: float = 80.0
    percolation_sat_mm_day: float = 2.0
    drain_power: float = 1.0
    kcb: float = 1.0
    ke_wet: float = 0.03
    evap_power: float = 1.0

    def __post_init__(self) -> None:
        for name in ("area_ha", "soil_depth_mm", "drain_power", "evap_power"):
            _number(name, getattr(self, name), strict_minimum=True)
        for name in ("theta_sat", "theta_fc", "theta_wp", "theta_crit",
                     "h_max_mm", "percolation_sat_mm_day", "kcb", "ke_wet"):
            _number(name, getattr(self, name))
        if not 0 <= self.theta_wp < self.theta_fc < self.theta_sat <= 1:
            raise ValueError("require 0 <= theta_wp < theta_fc < theta_sat <= 1")
        if not self.theta_wp < self.theta_crit <= self.theta_sat:
            raise ValueError("require theta_wp < theta_crit <= theta_sat")
        # Products of individually finite inputs must also stay finite.
        if not math.isfinite(self.max_store_mm):
            raise ValueError("derived maximum store must be finite")

    @property
    def root_saturation_mm(self) -> float:
        return self.soil_depth_mm * self.theta_sat

    @property
    def root_field_capacity_mm(self) -> float:
        return self.soil_depth_mm * self.theta_fc

    @property
    def root_wilting_mm(self) -> float:
        return self.soil_depth_mm * self.theta_wp

    @property
    def root_available_water_mm(self) -> float:
        """Root-zone water between field capacity and wilting, excluding ponding.

        This capacity is not current available water and is not total storage.
        FAO-56 TAW has this form; the stress closure below remains a declared
        reduced-order model rather than an exact FAO rice implementation.
        """
        return self.soil_depth_mm * (self.theta_fc - self.theta_wp)

    @property
    def max_store_mm(self) -> float:
        return self.root_saturation_mm + self.h_max_mm


@dataclass(frozen=True)
class FieldState:
    B_mm: float

    def __post_init__(self) -> None:
        _number("B_mm", self.B_mm)


def _check_state(params: FieldParams, state: FieldState) -> None:
    if not isinstance(params, FieldParams):
        raise TypeError("params must be FieldParams")
    if not isinstance(state, FieldState):
        raise TypeError("state must be FieldState")
    if state.B_mm > params.max_store_mm:
        raise ValueError("initial B_mm exceeds root saturation plus surface capacity")


def partition(params: FieldParams, state: FieldState) -> dict[str, float]:
    """Return root storage, surface depth, root moisture and transpiration Ks."""
    _check_state(params, state)
    W = min(state.B_mm, params.root_saturation_mm)
    h = max(state.B_mm - params.root_saturation_mm, 0.0)
    theta = W / params.soil_depth_mm
    ks = _clip01((W - params.root_wilting_mm) /
                 (params.soil_depth_mm * (params.theta_crit - params.theta_wp)))
    return {"W_mm": W, "h_mm": h, "theta": theta, "Ks": ks}


def mm_to_m3(depth_mm: float, area_ha: float) -> float:
    """One millimetre over one hectare equals 10 cubic metres."""
    _number("depth_mm", depth_mm)
    _number("area_ha", area_ha, strict_minimum=True)
    result = 10.0 * area_ha * depth_mm
    if not math.isfinite(result):
        raise ValueError("converted volume must be finite")
    return result


def field_step(
    params: FieldParams,
    state: FieldState,
    rain_mm: float,
    irrigation_mm: float,
    et0_mm_day: float,
    dt_day: float = 1.0 / 96.0,
) -> tuple[FieldState, dict[str, float]]:
    """Advance one substep and return state plus actual depth fluxes in mm.

    Fluxes Tpot, Ta, Ea, D and Q are potential transpiration, actual
    transpiration, actual evaporation, deep percolation and overflow. The
    diagnostic stress_deficit is Tpot - Ta. water_residual is the signed balance
    error, which should be zero to floating-point tolerance. It is not silently
    corrected by changing any flux. Ks and evaporation_wetness are evaluated
    after inflows/overflow and before either ET withdrawal. E_demand is reported
    separately from actual Ea. The declared sequential withdrawal gives
    transpiration priority when the substep contains insufficient water.
    """
    _check_state(params, state)
    for name, value in (("rain_mm", rain_mm), ("irrigation_mm", irrigation_mm),
                        ("et0_mm_day", et0_mm_day)):
        _number(name, value)
    _number("dt_day", dt_day, strict_minimum=True)
    if dt_day > 1.0:
        raise ValueError("dt_day must be <= 1; subdivide longer intervals")

    before_overflow = math.fsum((state.B_mm, rain_mm, irrigation_mm))
    if not math.isfinite(before_overflow):
        raise ValueError("combined water input must be finite")
    Q = max(before_overflow - params.max_store_mm, 0.0)
    B = min(before_overflow, params.max_store_mm)
    initial_partition = partition(params, FieldState(B))
    ks = initial_partition["Ks"]
    W = initial_partition["W_mm"]
    h = initial_partition["h_mm"]

    Tpot = params.kcb * et0_mm_day * dt_day
    wetness = (1.0 if h > 0.0 else
               (W / params.root_saturation_mm) ** params.evap_power)
    E_demand = params.ke_wet * et0_mm_day * dt_day * wetness
    if not math.isfinite(Tpot) or not math.isfinite(E_demand):
        raise ValueError("derived potential ET must be finite")

    Ta = min(ks * Tpot, max(B - params.root_wilting_mm, 0.0))
    B -= Ta
    Ea = min(E_demand, B)
    B -= Ea

    drain_fraction = _clip01(
        (min(B, params.root_saturation_mm) - params.root_field_capacity_mm) /
        (params.root_saturation_mm - params.root_field_capacity_mm)
    ) ** params.drain_power
    drain_demand = params.percolation_sat_mm_day * dt_day * drain_fraction
    if not math.isfinite(drain_demand):
        raise ValueError("derived percolation must be finite")
    D = min(max(B - params.root_field_capacity_mm, 0.0), drain_demand)
    B -= D

    new_state = FieldState(B)
    residual = math.fsum((state.B_mm, rain_mm, irrigation_mm,
                          -Ta, -Ea, -D, -Q, -new_state.B_mm))
    fluxes = {
        "rain_mm": float(rain_mm),
        "irrigation_mm": float(irrigation_mm),
        "Tpot": Tpot,
        "Ta": Ta,
        "Ea": Ea,
        "E_demand": E_demand,
        "D": D,
        "Q": Q,
        "Ks": ks,
        "evaporation_wetness": wetness,
        "stress_deficit": max(Tpot - Ta, 0.0),
        "water_residual": residual,
    }
    return new_state, fluxes
