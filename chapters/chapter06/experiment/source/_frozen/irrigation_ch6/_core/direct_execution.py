"""Pure daily field execution for the direct-to-field allocation environment.

No regional controller, pond, source account or conveyance network participates
in this kernel. Allocated water is uniformly irrigated during the day. The
field kernel conserves water, retains soil/surface water and computes actual
ET, deep loss and overflow. Calendar durations remain research assumptions.
"""
from __future__ import annotations

from dataclasses import fields, replace
from datetime import date
from functools import lru_cache
import math
from numbers import Integral

import numpy as np

from .field import FieldParams, FieldState, field_step
from ._direct_kernel import field_substeps


FIELD_PARAMETER_NAMES = (
    "soil_depth_mm", "theta_sat", "theta_fc", "theta_wp", "theta_crit",
    "h_max_mm", "percolation_sat_mm_day", "drain_power", "ke_wet", "evap_power",
)
STAGE_NAMES = ("vegetative", "booting", "flowering", "grain_filling", "ripening", "mature")
_FIELD_INIT_NAMES = tuple(item.name for item in fields(FieldParams))
_FLOAT_FIELD_TYPES = (float,) * len(_FIELD_INIT_NAMES)


def _immutable_column(values):
    # A bytes backing store cannot be made writable with setflags. These
    # process-local cached inputs are never returned as public diagnostics.
    return np.frombuffer(np.asarray(values, dtype=np.float64).tobytes(), dtype=np.float64)


def _pack_field_parameters(parameters, steps):
    column = lambda name: _immutable_column([getattr(p, name) for p in parameters])
    return (column("area_ha"), column("root_saturation_mm"),
            column("root_field_capacity_mm"), column("root_wilting_mm"),
            column("max_store_mm"),
            _immutable_column(column("soil_depth_mm") *
                              (column("theta_crit") - column("theta_wp"))),
            column("kcb"), column("ke_wet"),
            _immutable_column(column("percolation_sat_mm_day") * (1. / steps)),
            column("drain_power"), column("evap_power"))


@lru_cache(maxsize=128)
def _cached_field_parameters(parameter_rows, coefficient_bytes, steps):
    coefficients = np.frombuffer(coefficient_bytes, dtype=np.float64).reshape(-1, 2)
    parameters = []
    for (_, row), (kcb, ke) in zip(parameter_rows, coefficients):
        values = dict(zip(_FIELD_INIT_NAMES, row))
        values.update(kcb=kcb, ke_wet=ke)
        parameters.append(FieldParams(**values))
    return _pack_field_parameters(parameters, steps)


def _daily_field_parameters(field_params, coefficients, steps):
    """Reuse immutable numeric packs, not caller-owned dataclass instances.

    Value snapshots prevent stale hits if a caller replaces parameters between
    days. Scalar types are part of the key, so Python and NumPy arithmetic keep
    their validation semantics. Custom dataclass subclasses retain the original
    replace/validation path, including unhashable extra fields.
    """
    rows = []
    for original in field_params:
        if type(original) is not FieldParams:
            break
        values = tuple(getattr(original, name) for name in _FIELD_INIT_NAMES)
        types = tuple(map(type, values))
        if types != _FLOAT_FIELD_TYPES and any(
                kind not in (int, float) and not issubclass(kind, (np.integer, np.floating))
                for kind in types):
            break
        rows.append((types, values))
    else:
        return _cached_field_parameters(tuple(rows), coefficients.tobytes(), steps)
    parameters = [replace(original, kcb=kcb, ke_wet=ke)
                  for original, (kcb, ke) in zip(field_params, coefficients)]
    return _pack_field_parameters(parameters, steps)


def calendar_profile(day: date, heading: date, maturity_offset_days: int = 35):
    """Return public stage index, basal coefficient and outcome stage index.

    Maturity is a fixed calendar boundary, not a threshold in supplied water.
    Ripening ET is retained in the water balance but omitted by the declared
    four-stage Jensen proxy. Varying maturity changes the ripening duration.
    """
    if (isinstance(maturity_offset_days, (bool, np.bool_))
            or not isinstance(maturity_offset_days, Integral) or maturity_offset_days <= 25):
        raise ValueError("maturity_offset_days must be an integer greater than 25")
    age = (day - heading).days
    if age < -20:
        return 0, 1.0, 0
    if age < 0:
        return 1, 1.15, 1
    if age < 10:
        return 2, 1.15, 2
    if age < 25:
        return 3, 1.15 - .35 * (age - 10) / 15., 3
    if age < maturity_offset_days:
        return 4, .80 - .20 * (age - 25) / (maturity_offset_days - 25), -1
    return 5, 0., -1


def advance_direct_day_reference(*, field_params, storage_mm, heading_dates, day,
                       maturity_offset_days, action_m3, rain_mm, eto_mm_day,
                       substeps_per_day=96, crop_profiles=None):
    """Compute one day without modifying inputs or drawing random numbers.

    ``rain_mm`` and ``eto_mm_day`` are realized per-region forcings. All arrays
    have shape (N,). The caller checks action budgets and mature-region masks.
    Fluxes are in mm over each field; the aggregate balance residual is m3.
    """
    n = len(field_params)
    if (isinstance(substeps_per_day, (bool, np.bool_))
            or not isinstance(substeps_per_day, Integral) or substeps_per_day < 1):
        raise ValueError("substeps_per_day must be a positive integer")
    arrays = {}
    for name, value in (("storage_mm", storage_mm), ("action_m3", action_m3),
                        ("rain_mm", rain_mm), ("eto_mm_day", eto_mm_day)):
        raw = np.asarray(value)
        if (raw.shape != (n,) or raw.dtype.kind not in "iuf"
                or any(isinstance(x, (bool, np.bool_))
                       for x in np.asarray(value, dtype=object).flat)):
            raise ValueError(f"{name} must be a real vector of length {n}")
        a = np.array(raw, dtype=np.float64, copy=True)
        if np.any(~np.isfinite(a)) or np.any(a < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
        arrays[name] = a
    if len(heading_dates) != n:
        raise ValueError("each field needs one heading date")
    if crop_profiles is not None:
        profiles = np.asarray(crop_profiles, dtype=np.float64)
        if (profiles.shape != (n, 4) or np.any(~np.isfinite(profiles))
                or np.any(profiles[:, 1:3] < 0)
                or np.any(profiles[:, 0] != profiles[:, 0].astype(int))
                or np.any(profiles[:, 3] != profiles[:, 3].astype(int))
                or np.any((profiles[:, 0] < 0) | (profiles[:, 0] > 5))
                or np.any((profiles[:, 3] < -1) | (profiles[:, 3] > 3))):
            raise ValueError("crop_profiles must be N rows of stage, Kcb, Ke, outcome stage")
    steps = int(substeps_per_day)
    dt = 1. / steps
    areas = np.array([p.area_ha for p in field_params], dtype=np.float64)
    irrigation_mm = arrays["action_m3"] / (10. * areas)
    next_storage = np.empty(n)
    daily = {key: np.zeros(n) for key in ("Ta", "Ea", "D", "Q", "reference_et")}
    outcome_stage = np.empty(n, dtype=np.int64)
    public_stage = np.empty(n, dtype=np.int64)
    for i, original in enumerate(field_params):
        if crop_profiles is None:
            stage, kcb, outcome = calendar_profile(day, heading_dates[i], maturity_offset_days)
            ke = original.ke_wet
        else:
            stage, kcb, ke, outcome = profiles[i]
        public_stage[i], outcome_stage[i] = stage, outcome
        params = replace(original, kcb=kcb, ke_wet=ke)
        state = FieldState(float(arrays["storage_mm"][i]))
        daily["reference_et"][i] = (kcb + params.ke_wet) * arrays["eto_mm_day"][i]
        for _ in range(steps):
            state, flux = field_step(params, state,
                rain_mm=float(arrays["rain_mm"][i] * dt),
                irrigation_mm=float(irrigation_mm[i] * dt),
                et0_mm_day=float(arrays["eto_mm_day"][i]), dt_day=dt)
            for key in ("Ta", "Ea", "D", "Q"):
                daily[key][i] += flux[key]
        next_storage[i] = state.B_mm
    actual_et = daily["Ta"] + daily["Ea"]
    residual = (arrays["storage_mm"] + irrigation_mm + arrays["rain_mm"]
                - actual_et - daily["D"] - daily["Q"] - next_storage)
    if np.any(np.abs(residual) > 1e-7 * np.maximum(1., arrays["storage_mm"] + irrigation_mm + arrays["rain_mm"])):
        raise ArithmeticError("direct field water balance did not close")
    if np.any(actual_et > daily["reference_et"] + 1e-8):
        raise ArithmeticError("actual ET exceeds the same-weather reference")
    return {"field_storage_mm": next_storage, "daily_et_mm": actual_et,
            "daily_transpiration_mm": daily["Ta"], "daily_evaporation_mm": daily["Ea"],
            "daily_drainage_mm": daily["D"], "daily_overflow_mm": daily["Q"],
            "daily_reference_et_mm": daily["reference_et"],
            "daily_irrigation_mm": irrigation_mm, "daily_rain_mm": arrays["rain_mm"],
            "daily_eto_mm": arrays["eto_mm_day"], "outcome_stage_index": outcome_stage,
            "stage_index": public_stage, "last_action_m3": arrays["action_m3"],
            "water_mass_residual_m3": math.fsum((residual * (10. * areas)).tolist())}


def advance_direct_day(*, field_params, storage_mm, heading_dates, day,
                       maturity_offset_days, action_m3, rain_mm, eto_mm_day,
                       substeps_per_day=96, crop_profiles=None):
    """Compute one day without modifying inputs or drawing random numbers.

    ``rain_mm`` and ``eto_mm_day`` are realized per-region forcings. All arrays
    have shape (N,). The caller checks action budgets and mature-region masks.
    Fluxes are in mm over each field; the aggregate balance residual is m3.
    """
    n = len(field_params)
    if (isinstance(substeps_per_day, (bool, np.bool_))
            or not isinstance(substeps_per_day, Integral) or substeps_per_day < 1):
        raise ValueError("substeps_per_day must be a positive integer")
    arrays = {}
    for name, value in (("storage_mm", storage_mm), ("action_m3", action_m3),
                        ("rain_mm", rain_mm), ("eto_mm_day", eto_mm_day)):
        raw = np.asarray(value)
        if (raw.shape != (n,) or raw.dtype.kind not in "iuf"
                or any(isinstance(x, (bool, np.bool_))
                       for x in np.asarray(value, dtype=object).flat)):
            raise ValueError(f"{name} must be a real vector of length {n}")
        a = np.array(raw, dtype=np.float64, copy=True)
        if np.any(~np.isfinite(a)) or np.any(a < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
        arrays[name] = a
    if len(heading_dates) != n:
        raise ValueError("each field needs one heading date")
    if crop_profiles is not None:
        profiles = np.asarray(crop_profiles, dtype=np.float64)
        if (profiles.shape != (n, 4) or np.any(~np.isfinite(profiles))
                or np.any(profiles[:, 1:3] < 0)
                or np.any(profiles[:, 0] != profiles[:, 0].astype(int))
                or np.any(profiles[:, 3] != profiles[:, 3].astype(int))
                or np.any((profiles[:, 0] < 0) | (profiles[:, 0] > 5))
                or np.any((profiles[:, 3] < -1) | (profiles[:, 3] > 3))):
            raise ValueError("crop_profiles must be N rows of stage, Kcb, Ke, outcome stage")
    steps = int(substeps_per_day)
    dt = 1. / steps
    # Vectorize across regions while retaining the same 96 sequential substeps.
    # The scalar reference above remains available for numerical regression.
    if crop_profiles is None:
        profiles = np.empty((n, 4), dtype=np.float64)
        for i, original in enumerate(field_params):
            stage, kcb, outcome = calendar_profile(day, heading_dates[i], maturity_offset_days)
            profiles[i] = (stage, kcb, original.ke_wet, outcome)
    public_stage = profiles[:, 0].astype(np.int64)
    outcome_stage = profiles[:, 3].astype(np.int64)
    (areas, saturation, capacity, wilting, maximum, denominator, kcb, ke,
     drain_rate, drain_power, evap_power) = _daily_field_parameters(
         field_params, profiles[:, 1:3], steps)
    irrigation_mm = arrays["action_m3"] / (10. * areas)
    daily = {}
    B = arrays["storage_mm"].copy()
    if np.any(B > maximum):
        raise ValueError("initial B_mm exceeds root saturation plus surface capacity")
    rain_step = arrays["rain_mm"] * dt
    irrigation_step = irrigation_mm * dt
    tpot = kcb * arrays["eto_mm_day"] * dt
    epot = ke * arrays["eto_mm_day"] * dt
    daily["reference_et"] = (kcb + ke) * arrays["eto_mm_day"]
    if not np.isfinite(tpot).all() or not np.isfinite(epot).all():
        raise ValueError("derived potential ET must be finite")
    B, daily["Ta"], daily["Ea"], daily["D"], daily["Q"] = field_substeps(
        B, rain_step, irrigation_step, maximum, saturation, wilting, denominator,
        tpot, epot, capacity, drain_rate, drain_power, evap_power, steps)
    next_storage = B
    actual_et = daily["Ta"] + daily["Ea"]
    residual = (arrays["storage_mm"] + irrigation_mm + arrays["rain_mm"]
                - actual_et - daily["D"] - daily["Q"] - next_storage)
    if np.any(np.abs(residual) > 1e-7 * np.maximum(1., arrays["storage_mm"] + irrigation_mm + arrays["rain_mm"])):
        raise ArithmeticError("direct field water balance did not close")
    if np.any(actual_et > daily["reference_et"] + 1e-8):
        raise ArithmeticError("actual ET exceeds the same-weather reference")
    return {"field_storage_mm": next_storage, "daily_et_mm": actual_et,
            "daily_transpiration_mm": daily["Ta"], "daily_evaporation_mm": daily["Ea"],
            "daily_drainage_mm": daily["D"], "daily_overflow_mm": daily["Q"],
            "daily_reference_et_mm": daily["reference_et"],
            "daily_irrigation_mm": irrigation_mm, "daily_rain_mm": arrays["rain_mm"],
            "daily_eto_mm": arrays["eto_mm_day"], "outcome_stage_index": outcome_stage,
            "stage_index": public_stage, "last_action_m3": arrays["action_m3"],
            "water_mass_residual_m3": math.fsum((residual * (10. * areas)).tolist())}
