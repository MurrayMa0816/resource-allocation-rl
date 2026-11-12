"""Joint daily weather forcing for the independent irrigation environment.

The bundled records are NASA POWER grid estimates, not local station readings.
The training driver is a declared first-order donor model, not a calibrated
forecast service. The donor years are declared by the scenario; the current
Hongmiao case uses 2018--2021, independently of historical evaluation years.
Its public forecast uses only those prior donor pools;
actual weather is sampled by the environment after action validation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
import math
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from .direct_scenarios import DirectScenario as Scenario

PARAM_NAMES = (
    "T2M", "T2M_MAX", "T2M_MIN", "T2MDEW", "RH2M", "WS2M", "PS",
    "PRECTOTCORR", "ALLSKY_SFC_SW_DWN",
)


def _readonly_vector(values: Any, size: int, label: str) -> np.ndarray:
    raw = np.asarray(values)
    bool_elements = any(isinstance(x, (bool, np.bool_))
                        for x in np.asarray(values, dtype=object).flat)
    if raw.shape != (size,) or raw.dtype.kind not in "iuf" or bool_elements:
        raise ValueError(f"{label} must be a real numeric vector of length {size}")
    vector = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be finite")
    vector.setflags(write=False)
    return vector


def _validate_rain_multiplier(value: Any) -> float:
    """Normalize an explicit research forcing scale, never a weather datum."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("rain_multiplier must be a finite nonnegative number")
    try:
        multiplier = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("rain_multiplier must be a finite nonnegative number") from exc
    if not math.isfinite(multiplier) or multiplier < 0:
        raise ValueError("rain_multiplier must be a finite nonnegative number")
    return multiplier


@dataclass(frozen=True)
class WeatherDay:
    """Nine weather variables in PARAM_NAMES order and FAO56 ET0 (mm/day).

    ``date`` is the occurrence date, never the randomly selected donor date.
    Units are degC, degC, degC, degC, percent, m/s, kPa, mm/day, MJ/m2/day.
    """

    date: date
    vector: np.ndarray
    eto_mm_day: float

    def __post_init__(self) -> None:
        if type(self.date) is not date:
            raise ValueError("weather date must be a calendar date without a time component")
        vector = _readonly_vector(self.vector, 9, "weather")
        if vector[2] > vector[1] or not 0 <= vector[4] <= 100:
            raise ValueError("weather temperature or relative humidity is invalid")
        if vector[5] < 0 or vector[6] <= 0 or vector[7] < 0 or vector[8] < 0:
            raise ValueError("wind, pressure, rain or radiation is invalid")
        if (isinstance(self.eto_mm_day, (bool, np.bool_))
            or not isinstance(self.eto_mm_day, Real)):
            raise ValueError("ET0 must be a finite nonnegative number")
        eto = float(self.eto_mm_day)
        if not math.isfinite(eto) or eto < 0:
            raise ValueError("ET0 must be a finite nonnegative number")
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "eto_mm_day", eto)

    @property
    def rain_mm(self) -> float:
        return float(self.vector[7])

    @property
    def tmax_c(self) -> float:
        return float(self.vector[1])

    @property
    def weather_class(self) -> int:
        return weather_class(self.vector)


def weather_class(vector: Any) -> int:
    """Observable one-day memory: wet>=1 mm and hot Tmax>=35 degC."""
    values = _readonly_vector(vector, 9, "previous weather")
    return int(values[7] >= 1) + 2 * int(values[1] >= 35)


def reference_et0(
    vector: Any, day: date, *, latitude: float = 32.0, elevation_m: float = 40.62,
) -> float:
    """Frozen FAO56 daily Penman--Monteith calculation, in mm/day.

    Uses dew-point vapour pressure, 2 m wind, zero daily soil heat flux,
    radiation albedo 0.23 and a declared cloudiness ratio floor 0.33. This
    reproduces the bundled preparation formula. POWER pressure is in kPa.
    This daily radiation procedure requires positive clear-sky radiation;
    polar-night dates need a different explicitly supplied radiation protocol.
    Reference: https://www.fao.org/4/x0490e/x0490e08.htm .
    """
    v = _readonly_vector(vector, 9, "weather")
    tmin, tmax, dew, u2, pressure, rs = (float(v[i]) for i in (2, 1, 3, 5, 6, 8))
    if tmin > tmax or not 0 <= v[4] <= 100:
        raise ValueError("weather temperature or relative humidity is invalid")
    if u2 < 0 or pressure <= 0 or v[7] < 0 or rs < 0:
        raise ValueError("wind, pressure, rain or radiation is invalid")
    if min(tmin, tmax, dew) <= -237.3:
        raise ValueError("temperature is outside the vapour-pressure formula domain")
    if (type(day) is not date or isinstance(latitude, (bool, np.bool_))
        or not isinstance(latitude, Real) or not -90 < latitude < 90):
        raise ValueError("ET0 needs a calendar date and latitude strictly inside the poles")
    if (isinstance(elevation_m, (bool, np.bool_)) or not isinstance(elevation_m, Real)
        or not math.isfinite(elevation_m)):
        raise ValueError("elevation must be finite")

    def es(t: float) -> float:
        return 0.6108 * math.exp(17.27 * t / (t + 237.3))

    temperature = (tmin + tmax) / 2
    ea = es(dew)
    vap = (es(tmin) + es(tmax)) / 2
    delta = 4098 * es(temperature) / (temperature + 237.3) ** 2
    gamma = 0.000665 * pressure
    j = day.timetuple().tm_yday
    phi = math.radians(latitude)
    dr = 1 + 0.033 * math.cos(2 * math.pi * j / 365)
    dec = 0.409 * math.sin(2 * math.pi * j / 365 - 1.39)
    sunset = math.acos(max(-1.0, min(1.0, -math.tan(phi) * math.tan(dec))))
    ra = (24 * 60 / math.pi) * 0.0820 * dr * (
        sunset * math.sin(phi) * math.sin(dec)
        + math.cos(phi) * math.cos(dec) * math.sin(sunset)
    )
    rso = (0.75 + 2e-5 * elevation_m) * ra
    if not math.isfinite(rso) or rso <= 0:
        raise ValueError("daily ET0 requires positive clear-sky radiation")
    ratio = max(0.33, min(1.0, rs / rso))
    rnl = 4.903e-9 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2
    rnl *= (0.34 - 0.14 * math.sqrt(ea)) * (1.35 * ratio - 0.35)
    rn = 0.77 * rs - rnl
    vpd = max(0.0, vap - ea)
    raw = (0.408 * delta * rn + gamma * 900 / (temperature + 273) * u2 * vpd)
    raw /= delta + gamma * (1 + 0.34 * u2)
    if not math.isfinite(raw):
        raise ValueError("derived daily ET0 must be finite")
    return max(0.0, raw)


class WeatherDriver:
    """Private weather transition driver, separate from policy observations.

    Markov mode draws exactly one uniform index at reset and one per sampled
    day. Calling forecast never consumes randomness. Historical mode is a
    private evaluation trajectory and does not pretend to be the training MDP.
    Its forecast still uses the frozen training table, never future test rows.
    Snapshots include private RNG state and must not enter policy ``info``.
    """

    def __init__(self, scenario: Scenario, mode: str = "markov", year: int = 2022):
        if mode not in {"markov", "historical"}:
            raise ValueError("weather mode must be 'markov' or 'historical'")
        if isinstance(year, (bool, np.bool_)) or not isinstance(year, Integral):
            raise ValueError("weather year must be an integer")
        self.scenario = scenario
        self.mode = mode
        self.year = int(year)
        self._rng: np.random.Generator | None = None
        if mode == "historical":
            for t in range(-1, scenario.T):
                self._historical_record(t)

    def _indices(self, t: int, previous_class: int) -> tuple[int, int]:
        for value, label in ((t, "day index"), (previous_class, "weather class")):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{label} must be an integer")
        if not 0 <= t < self.scenario.T or not 0 <= previous_class < 4:
            raise ValueError("day index or weather class is outside the scenario")
        return int(t), int(previous_class)

    def _historical_record(self, t: int) -> WeatherDay:
        crop_day = self.scenario.start_date + timedelta(days=t)
        try:
            occurrence = crop_day.replace(year=self.year)
            return self.scenario.weather_records[occurrence.isoformat()]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"historical weather is unavailable for day {t} in {self.year}") from exc

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, int]:
        """Return observed previous weather; None continues an existing stream."""
        if seed is not None:
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
                raise ValueError("weather seed must be a nonnegative integer or None")
            self._rng = np.random.Generator(np.random.PCG64(int(seed)))
        elif self._rng is None:
            self._rng = np.random.Generator(np.random.PCG64())
        if self.mode == "historical":
            previous = self._historical_record(-1)
        else:
            candidates = self.scenario.weather_contract["initial_donor_dates"]
            index = int(self._rng.integers(len(candidates)))
            previous = self.scenario.weather_records[candidates[index]]
        return previous.vector.copy(), previous.weather_class

    def forecast(self, t: int, previous_class: int) -> np.ndarray:
        """Expected rain, ET0, hot probability from the fixed training pool."""
        key = self._indices(t, previous_class)
        return np.array(self.scenario.forecast_table[key], dtype=np.float64, copy=True)

    def sample(self, t: int, previous_class: int) -> WeatherDay:
        """Generate today's forcing only after the environment accepts action."""
        t, previous_class = self._indices(t, previous_class)
        if self._rng is None:
            raise RuntimeError("weather driver must be reset or restored before sampling")
        if self.mode == "historical":
            record = self._historical_record(t)
            return WeatherDay(record.date, record.vector, record.eto_mm_day)
        candidates = self.scenario.donor_pools[f"{t}:{previous_class}"]
        index = int(self._rng.integers(len(candidates)))
        record = self.scenario.weather_records[candidates[index]]
        day = self.scenario.start_date + timedelta(days=t)
        grid = self.scenario.weather_contract["common_driver"]
        eto = reference_et0(record.vector, day, latitude=float(grid["latitude"]),
                            elevation_m=float(grid["grid_elevation_m"]))
        return WeatherDay(day, record.vector, eto)

    def snapshot(self) -> dict[str, Any]:
        """Private day-boundary checkpoint; seed/donor identity are never public."""
        if self._rng is None:
            raise RuntimeError("weather driver has no initialized RNG")
        return {"mode": self.mode, "year": self.year,
                "rng_state": deepcopy(self._rng.bit_generator.state)}

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore a matching driver without consuming a reset draw."""
        if not isinstance(checkpoint, Mapping):
            raise ValueError("weather checkpoint must be a mapping")
        if checkpoint.get("mode") != self.mode or checkpoint.get("year") != self.year:
            raise ValueError("weather checkpoint mode/year does not match")
        if "rng_state" not in checkpoint:
            raise ValueError("weather checkpoint lacks RNG state")
        rng = np.random.Generator(np.random.PCG64())
        try:
            rng.bit_generator.state = deepcopy(checkpoint["rng_state"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("weather checkpoint has an invalid PCG64 state") from exc
        self._rng = rng
