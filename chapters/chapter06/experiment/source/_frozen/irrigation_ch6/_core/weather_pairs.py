"""Shared daily rain/ET0 forcing from archived historical pairs.

The two public quantities are rain (mm/day) and FAO56 reference ET0 (mm/day).
Each sample retains the two values from the same historical date. Training
uses 2018--2021 records only, with a calendar window and the *preceding*
historical day's rain class. Evaluation replays an archived year. This is a
research weather generator, not a locally calibrated weather forecast. An
explicit rain multiplier transforms output rain only, keeping ET0 unchanged.
Positive multipliers preserve the original donor chain; zero rain instead uses
unconditional seasonal donors to avoid an unobservable wet/dry transition state.
The explicit persistent_drought research mode additionally restricts donors to
ET0 >= 3 mm/day. It never raises recorded ET0, conditions on allocation/state,
or presents the resulting zero-rain sequence as historical weather replay.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
import json
from numbers import Integral
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .weather import _readonly_vector, _validate_rain_multiplier

if TYPE_CHECKING:
    from .direct_scenarios import DirectScenario


PAIR_NAMES = ("rain_mm_day", "reference_et0_mm_day")
WET_THRESHOLD_MM = 1.0
SEASON_WINDOW_DAYS = 15
TRAIN_YEARS = (2018, 2019, 2020, 2021)
DROUGHT_MIN_ET0_MM = 3.0


def pair_weather_class(vector: Any, *, rain_multiplier: float = 1.0) -> int:
    """Recover original wet/dry memory from observed, scaled previous rain.

    A positive scale moves the observable threshold, not the donor threshold.
    At zero scale all observations have zero rain and there is no wet/dry state.
    """
    multiplier = _validate_rain_multiplier(rain_multiplier)
    values = _readonly_vector(vector, 2, "rain/ET0 weather")
    if np.any(values < 0):
        raise ValueError("rain and ET0 must be nonnegative")
    if multiplier == 0:
        if values[0] != 0:
            raise ValueError("zero rain_multiplier requires zero observed rain")
        return 0
    return int(values[0] >= WET_THRESHOLD_MM * multiplier)


@dataclass(frozen=True)
class PairWeatherDay:
    """One already transformed pair, dated by occurrence rather than donor day.

    ``rain_multiplier`` supplies the fixed scale needed to interpret observable
    wet/dry memory; construction does not apply the multiplier a second time.
    """

    date: date
    vector: np.ndarray
    rain_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if type(self.date) is not date:
            raise ValueError("weather date must be a calendar date without a time component")
        multiplier = _validate_rain_multiplier(self.rain_multiplier)
        vector = _readonly_vector(self.vector, 2, "rain/ET0 weather")
        pair_weather_class(vector, rain_multiplier=multiplier)
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "rain_multiplier", multiplier)

    @property
    def rain_mm(self) -> float:
        return float(self.vector[0])

    @property
    def eto_mm_day(self) -> float:
        return float(self.vector[1])

    @property
    def weather_class(self) -> int:
        return pair_weather_class(self.vector, rain_multiplier=self.rain_multiplier)


def _calendar_distance(first: date, second: date) -> int:
    """Circular month/day distance, with leap day represented consistently."""
    a = first.replace(year=2000).timetuple().tm_yday
    b = second.replace(year=2000).timetuple().tm_yday
    difference = abs(a - b)
    return min(difference, 366 - difference)


class PairWeatherDriver:
    """Generate one common rain/ET0 pair for every region on a physical day.

    Forecasts are deterministic training-pool means on the forcing scale. They neither
    inspect future evaluation rows nor consume random draws. Markov reset
    selects a previous-day pair, and each sample selects one donor pair.
    ``reset(None)`` continues the RNG stream; an explicit seed restarts it.
    Private snapshots preserve reproducibility and never belong in policy info.
    """

    vector_size = 2
    forecast_size = 2

    def __init__(self, scenario: DirectScenario, mode: str = "markov", year: int = 2022,
                 *, rain_multiplier: float = 1.0):
        if mode not in {"markov", "historical", "persistent_drought"}:
            raise ValueError("weather mode must be 'markov', 'historical' or 'persistent_drought'")
        if isinstance(year, (bool, np.bool_)) or not isinstance(year, Integral):
            raise ValueError("weather year must be an integer")
        self._rain_multiplier = _validate_rain_multiplier(rain_multiplier)
        if mode == "persistent_drought" and self.rain_multiplier != 0:
            raise ValueError("persistent_drought requires rain_multiplier=0")
        self.scenario, self.mode, self.year = scenario, mode, int(year)
        self._rng: np.random.Generator | None = None
        # Cache copied pairs, so a later caller cannot mutate a forcing record.
        self._records = {
            key: PairWeatherDay(record.date, [record.rain_mm, record.eto_mm_day])
            for key, record in scenario.weather_records.items()
        }
        # PairWeatherDay has already validated each copied vector. Retain its
        # class once; no external array can change these construction inputs.
        record_classes = {
            key: int(record.vector[0] >= WET_THRESHOLD_MM)
            for key, record in self._records.items()
        }
        # Validate all transformed inputs before any reset/sample RNG draw.
        # In particular, a finite scale must not overflow output rainfall or
        # underflow so far that the observable wet/dry class becomes ambiguous.
        for key, record in self._records.items():
            transformed = self._scaled_vector(record.vector)
            if (self.rain_multiplier > 0
                    and int(transformed[0] >= WET_THRESHOLD_MM * self.rain_multiplier)
                    != record_classes[key]):
                raise ValueError("rain_multiplier is too small to preserve the observed rain class")
        donors = tuple(sorted(
            key for key, record in self._records.items() if record.date.year in TRAIN_YEARS
        ))
        if not donors or {self._records[key].date.year for key in donors} != set(TRAIN_YEARS):
            raise ValueError("paired weather needs archived records in all 2018--2021 training years")
        previous_date = scenario.start_date - timedelta(days=1)
        self._initial_dates = tuple(
            key for key in donors
            if (self._records[key].date.month, self._records[key].date.day)
            == (previous_date.month, previous_date.day)
        )
        initial_drought_fallback = False
        if mode == "persistent_drought":
            self._initial_dates = tuple(
                key for key in self._initial_dates
                if self._records[key].eto_mm_day >= DROUGHT_MIN_ET0_MM
            )
            if not self._initial_dates:
                initial_drought_fallback = True
                self._initial_dates = tuple(
                    key for key in donors
                    if (_calendar_distance(self._records[key].date, previous_date)
                        <= SEASON_WINDOW_DAYS
                        and self._records[key].eto_mm_day >= DROUGHT_MIN_ET0_MM)
                )
            if not self._initial_dates:
                raise ValueError("persistent_drought lacks initial seasonal donors with ET0 >= 3 mm/day")
        if not self._initial_dates:
            raise ValueError("paired weather lacks training records for the initial previous date")
        # These are properties of archived donors, independent of the target
        # day. Compute them once rather than revalidating preceding pairs and
        # converting calendar dates for every day/class candidate pool.
        donor_days = np.array([
            self._records[key].date.replace(year=2000).timetuple().tm_yday
            for key in donors
        ])
        previous_classes = []
        for key in donors:
            previous_key = (self._records[key].date - timedelta(days=1)).isoformat()
            previous = self._records.get(previous_key)
            previous_classes.append(
                record_classes[previous_key]
                if previous is not None and previous.date.year in TRAIN_YEARS else -1
            )
        previous_classes = np.array(previous_classes)
        self._donor_pools: dict[tuple[int, int], tuple[str, ...]] = {}
        self._forecasts: dict[tuple[int, int], np.ndarray] = {}
        fallback_keys = []
        for t in range(scenario.T):
            target = scenario.start_date + timedelta(days=t)
            target_day = target.replace(year=2000).timetuple().tm_yday
            distance = np.abs(donor_days - target_day)
            nearby_indices = np.flatnonzero(
                np.minimum(distance, 366 - distance) <= SEASON_WINDOW_DAYS
            )
            nearby = tuple(donors[i] for i in nearby_indices)
            if not nearby:
                raise ValueError(f"paired weather lacks seasonal training donors for day {t}")
            if mode == "persistent_drought":
                nearby = tuple(key for key in nearby
                               if self._records[key].eto_mm_day >= DROUGHT_MIN_ET0_MM)
                if not nearby:
                    raise ValueError(f"persistent_drought lacks seasonal donors with ET0 >= 3 mm/day for day {t}")
            for cls in (0, 1):
                conditional = tuple(donors[i] for i in nearby_indices
                                    if previous_classes[i] == cls)
                pool = nearby if self.rain_multiplier == 0 else conditional or nearby
                if not conditional and self.rain_multiplier != 0:
                    fallback_keys.append([t, cls])
                self._donor_pools[t, cls] = pool
                forecast = self._scaled_vector(
                    np.mean([self._records[key].vector for key in pool], axis=0)
                )
                forecast.setflags(write=False)
                self._forecasts[t, cls] = forecast
        encoded_training_pairs = json.dumps(
            [[key, *self._records[key].vector.tolist()] for key in donors],
            separators=(",", ":"), allow_nan=False,
        ).encode()
        self._contract = {
            "schema": "hongmiao-paired-weather-v2",
            "quantities": list(PAIR_NAMES), "units": ["mm/day", "mm/day"],
            "train_donor_years": list(TRAIN_YEARS),
            "wet_threshold_mm": WET_THRESHOLD_MM,
            "rain_multiplier": self.rain_multiplier,
            "reference_et0_multiplier": 1.0,
            "observable_wet_threshold_mm": (WET_THRESHOLD_MM * self.rain_multiplier
                                             if self.rain_multiplier > 0 else None),
            "season_window_days": SEASON_WINDOW_DAYS,
            "conditioning": ("preceding unscaled historical donor day rain >= wet_threshold_mm"
                             if self.rain_multiplier > 0 else
                             "unconditional seasonal donors; zero rain has no observable wet/dry state"),
            "pair_sampling": "sample an original rain/ET0 donor pair, then multiply rain only",
            "preserves_original_markov_chain": self.rain_multiplier > 0,
            "forcing_kind": ("historical" if self.rain_multiplier == 1 else "rain_scaled_historical")
                            if mode == "historical" else
                            ("zero_rain_seasonal" if self.rain_multiplier == 0 else
                             "markov" if self.rain_multiplier == 1 else "rain_scaled_markov"),
            "historical_replay": ("original archived rain and ET0" if self.rain_multiplier == 1 else
                                  "archived rain multiplied by rain_multiplier; ET0 unchanged; not original historical weather"),
            "fallback": "same seasonal window without rain-class conditioning",
            "fallback_day_class_pairs": fallback_keys,
            "forecast": "training donor-pool mean with rain multiplied by rain_multiplier and ET0 unchanged",
            "initial_previous_weather": "previous-day archived pair with the same rain multiplier",
            "scope": "one pair shared by all regions; no local perturbations",
            "settings_status": "research settings, not locally calibrated rainfall parameters",
            "training_pairs_sha256": sha256(encoded_training_pairs).hexdigest(),
            "training_pair_count": len(donors),
            "common_driver": deepcopy(scenario.weather_contract.get("common_driver", {})),
        }
        if mode == "persistent_drought":
            support = {str(t): list(self._donor_pools[t, 0]) for t in range(scenario.T)}
            selected = sorted(set(self._initial_dates).union(
                *(self._donor_pools[t, 0] for t in range(scenario.T))))
            encoded_support = json.dumps(
                {"initial_dates": list(self._initial_dates), "day_donor_dates": support,
                 "pairs": [[key, *self._records[key].vector.tolist()] for key in selected]},
                sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode()
            self._contract.update({
                "schema": "hongmiao-paired-weather-persistent-drought-v1",
                "weather_mode": mode,
                "minimum_reference_et0_mm_day": DROUGHT_MIN_ET0_MM,
                "et0_selection": "retain original ET0 >= threshold; no clipping, scaling or replacement",
                "conditioning": "date-only filtered seasonal donors; independent of state, action and previous wet/dry class",
                "pair_sampling": "uniform filtered training donor; replace rain by zero and retain donor ET0 exactly",
                "preserves_original_markov_chain": False,
                "forcing_kind": "controlled_persistent_drought",
                "historical_replay": "none; controlled research sequence, not historical replay",
                "year_role": "calendar metadata only; no evaluation-year weather is sampled",
                "fallback": "initial date only: if exact month/day has no eligible donor, use its +/-15-day training window; reject empty support",
                "fallback_day_class_pairs": [],
                "initial_previous_weather": "eligible previous-month/day training donor; documented seasonal fallback only if empty",
                "initial_previous_weather_seasonal_fallback": initial_drought_fallback,
                "initial_donor_dates": list(self._initial_dates),
                "day_donor_dates": support,
                "filtered_support_sha256": sha256(encoded_support).hexdigest(),
                "forecast": "exact filtered seasonal donor mean, zero rain, no RNG draw or realized future lookup",
                "settings_status": "controlled no-rain, high-evaporative-demand research scenario; not observed local drought",
            })
        # Checkpoint matching additionally includes evaluation inputs and calendar.
        all_pairs = [[key, *record.vector.tolist()]
                     for key, record in sorted(self._records.items())]
        self._signature = sha256(json.dumps(
            {"contract": self._contract, "records": all_pairs,
             "start_date": scenario.start_date.isoformat(), "horizon": scenario.T},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        if mode == "historical":
            for t in range(-1, scenario.T):
                self._historical_record(t)

    @property
    def rain_multiplier(self) -> float:
        """Fixed output scale; donor records always retain original rainfall."""
        return self._rain_multiplier

    def _scaled_vector(self, vector: np.ndarray) -> np.ndarray:
        result = vector.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            result[0] *= self.rain_multiplier
        if not np.all(np.isfinite(result)):
            raise ValueError("rain_multiplier produces nonfinite scaled rainfall")
        return result

    def class_from_vector(self, vector: Any) -> int:
        """Derive transition memory from public previous weather and fixed scale."""
        return pair_weather_class(vector, rain_multiplier=self.rain_multiplier)

    @property
    def contract(self) -> dict[str, Any]:
        """Public generator settings and source identity, without private RNG state."""
        return deepcopy(self._contract)

    def _indices(self, t: int, previous_class: int) -> tuple[int, int]:
        for value, label in ((t, "day index"), (previous_class, "weather class")):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{label} must be an integer")
        if not 0 <= t < self.scenario.T or previous_class not in (0, 1):
            raise ValueError("day index or rain class is outside the scenario")
        return int(t), int(previous_class)

    def _historical_record(self, t: int) -> PairWeatherDay:
        day = self.scenario.start_date + timedelta(days=t)
        try:
            occurrence = day.replace(year=self.year)
            return self._records[occurrence.isoformat()]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"historical weather is unavailable for day {t} in {self.year}") from exc

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, int]:
        """Return yesterday's observed pair, continuing the stream when seed=None."""
        if seed is not None:
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
                raise ValueError("weather seed must be a nonnegative integer or None")
            self._rng = np.random.Generator(np.random.PCG64(int(seed)))
        elif self._rng is None:
            self._rng = np.random.Generator(np.random.PCG64())
        if self.mode == "historical":
            previous = self._historical_record(-1)
        else:
            index = int(self._rng.integers(len(self._initial_dates)))
            previous = self._records[self._initial_dates[index]]
        observed = self._scaled_vector(previous.vector)
        return observed, self.class_from_vector(observed)

    def forecast(self, t: int, previous_class: int) -> np.ndarray:
        """Scaled mean [rain, ET0] using only the applicable training pool."""
        return self._forecasts[self._indices(t, previous_class)].copy()

    def sample(self, t: int, previous_class: int) -> PairWeatherDay:
        """Draw today's common pair after action validation; no future observation."""
        t, previous_class = self._indices(t, previous_class)
        if self._rng is None:
            raise RuntimeError("weather driver must be reset or restored before sampling")
        if self.mode == "historical":
            record = self._historical_record(t)
        else:
            pool = self._donor_pools[t, previous_class]
            record = self._records[pool[int(self._rng.integers(len(pool)))]]
        occurrence = ((self.scenario.start_date + timedelta(days=t)).replace(year=self.year)
                      if self.mode == "historical"
                      else self.scenario.start_date + timedelta(days=t))
        return PairWeatherDay(occurrence, self._scaled_vector(record.vector), self.rain_multiplier)

    def snapshot(self) -> dict[str, Any]:
        """Private checkpoint; restoring consumes neither reset nor sample draws."""
        if self._rng is None:
            raise RuntimeError("weather driver has no initialized RNG")
        return {"schema": "paired-weather-rng-v1", "mode": self.mode, "year": self.year,
                "contract_signature": self._signature,
                "rng_state": deepcopy(self._rng.bit_generator.state)}

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        """Validate before replacing the RNG, preserving state on failure."""
        if not isinstance(checkpoint, Mapping):
            raise ValueError("weather checkpoint must be a mapping")
        expected = {"schema": "paired-weather-rng-v1", "mode": self.mode, "year": self.year,
                    "contract_signature": self._signature}
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise ValueError("weather checkpoint mode/year/contract does not match")
        if "rng_state" not in checkpoint:
            raise ValueError("weather checkpoint lacks RNG state")
        rng = np.random.Generator(np.random.PCG64())
        try:
            rng.bit_generator.state = deepcopy(checkpoint["rng_state"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("weather checkpoint has an invalid PCG64 state") from exc
        self._rng = rng
