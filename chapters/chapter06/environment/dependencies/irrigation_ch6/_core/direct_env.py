"""Direct daily irrigation allocation, shared by chapters three through six.

There is one representative crop per region and no pond or conveyance layer.
All accepted allocation is irrigated; field water is physical state, not an
unspent allocation account. This numerical core also supports explicitly
delayed reports for low-level tests. The formal Hongmiao chapter classes all
use current reports. Debug data and checkpoints are evaluator-only interfaces.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
from datetime import date, timedelta
import hashlib
import json
import math
from numbers import Integral, Real
from collections.abc import Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .direct_execution import advance_direct_day, calendar_profile, FIELD_PARAMETER_NAMES
from .direct_scenarios import DirectScenario
from .field import FieldState, partition
from .outcomes import OutcomeLedger, DEFAULT_LAMBDAS
from .weather import WeatherDriver, WeatherDay, weather_class


REPORT_KEYS = ("field_storage_mm", "stage_et_mm", "stage_reference_et_mm")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def _real(value, label, minimum=0.):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or value < minimum):
        raise ValueError(f"{label} must be finite and at least {minimum}")
    return float(value)


def _array(value, shape, label, *, minimum=0.):
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "iuf"
            or any(isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat)):
        raise ValueError(f"{label} must be a real array with shape {shape}")
    result = np.array(raw, dtype=np.float64, copy=True)
    if np.any(~np.isfinite(result)) or np.any(result < minimum):
        raise ValueError(f"{label} must be finite and at least {minimum}")
    return result


def _boolean_mask(value, size, label):
    raw = np.asarray(value)
    if raw.shape != (size,) or raw.dtype.kind != "b":
        raise ValueError(f"{label} must be a boolean vector with shape ({size},)")
    return raw.copy()


def _rng_from_state(state):
    result = np.random.Generator(np.random.PCG64())
    try:
        result.bit_generator.state = copy.deepcopy(state)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("invalid PCG64 checkpoint") from exc
    return result


class DirectAllocationSpace(spaces.Box):
    """Budget-feasible sampling, independent of field water and yield.

    ``sample()`` retains the original dense random demonstration distribution.
    Callers can supply a selection mask or explicitly use ``sample_sparse``.
    These are test/exploration helpers, not an irrigation policy or a rule
    requiring sparse actions. ``step`` never invokes either sampler.
    """

    def __init__(self, env):
        self._env = env
        high = max(env.scenario.quota_plan) if env.chapter == 3 else env._total_budget
        super().__init__(low=np.zeros(env.N), high=np.full(env.N, high), dtype=np.float64)

    def sample(self, mask=None, probability=None):
        if probability is not None:
            raise ValueError("direct allocation sampling does not accept probabilities")
        if mask is not None:
            mask = _boolean_mask(mask, self._env.N, "selection mask")
            if (self._env.chapter == 3 and self._env._allowance() > 0.
                    and not np.any(mask & self._env._active_mask())):
                raise ValueError("a positive fixed daily total requires a selected region")
        # Keep the unmasked legacy draw order and distribution unchanged.
        fraction = 1. if self._env.chapter == 3 else float(self.np_random.uniform())
        return self._env.feasible_allocation(self.np_random.uniform(size=self.shape),
                                            fraction=fraction, selected_mask=mask)

    def sample_sparse(self, active_count=None, *, fraction=None):
        """Sample a selected set and allocation for explicit tests/exploration.

        ``active_count`` specifies the selected-set size, or None samples its
        size uniformly (1..N for a positive chapter-three quota, 0..N for a
        carryover task). The full set remains possible. Eligibility uses only
        the public allocation calendar, never field moisture or hidden truth.
        No quota implies an all-zero allocation. A supplied fraction sets the
        share of the public allowance; otherwise use the legacy random share
        for carryover chapters and one for chapter three.
        """
        env = self._env
        eligible = np.flatnonzero(env._active_mask())
        if active_count is not None:
            if (isinstance(active_count, (bool, np.bool_))
                    or not isinstance(active_count, Integral)
                    or not 0 <= active_count <= len(eligible)):
                raise ValueError("active_count must be an integer within the eligible region count")
            active_count = int(active_count)
        if fraction is not None:
            fraction = _real(fraction, "fraction")
            if fraction > 1.:
                raise ValueError("fraction must be between zero and one")
            if env.chapter == 3 and env._allowance() > 0. and fraction != 1.:
                raise ValueError("chapter 3 requires fraction=1 for the fixed daily total")
        required = env.chapter == 3 and env._allowance() > 0.
        if required and (active_count == 0 or len(eligible) == 0):
            raise ValueError("a positive fixed daily total requires a selected region")
        if env._allowance() == 0. or fraction == 0.:
            return env.feasible_allocation(np.zeros(env.N), fraction=0.)
        count = (int(self.np_random.integers(1 if required else 0, len(eligible) + 1))
                 if active_count is None else active_count)
        mask = np.zeros(env.N, dtype=bool)
        if count:
            mask[self.np_random.choice(eligible, size=count, replace=False)] = True
        if fraction is None:
            fraction = 1. if env.chapter == 3 else float(self.np_random.uniform())
        weights = self.np_random.uniform(size=env.N)
        # Uniform draws may include exact zero. Selected samples should have
        # positive support; resampling here is not action repair inside step.
        zero = mask & (weights == 0.)
        while np.any(zero):
            weights[zero] = self.np_random.uniform(size=int(np.count_nonzero(zero)))
            zero = mask & (weights == 0.)
        return env.feasible_allocation(weights, fraction=fraction, selected_mask=mask)

    def contains(self, value):
        try:
            self._env._validate_action(value)
            return True
        except (ValueError, TypeError, OverflowError):
            return False


class DirectIrrigationEnv(gym.Env):
    """One daily vector of actual field-inlet m3, one crop in each region.

    Chapter classes define their task around this shared physics engine.
    Current cumulative ET is an idealized complete observation; it is not a
    claim that farms measure daily ET perfectly. Explicit report delays here
    are numerical test controls, not the formal chapter-six research problem.
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 2}

    def __init__(self, *, scenario: DirectScenario, chapter=3,
                 total_budget_m3=None, daily_limit_m3=None, report_delay_days=None,
                 local_weather_spread=.10, yield_lambdas=DEFAULT_LAMBDAS,
                 weather_mode="markov", historical_year=2022, render_mode=None):
        super().__init__()
        if isinstance(chapter, (bool, np.bool_)) or not isinstance(chapter, Integral) or chapter not in (3, 4, 5, 6):
            raise ValueError("chapter must be 3, 4, 5 or 6")
        if render_mode not in (None, "ansi", "rgb_array"):
            raise ValueError("render_mode must be None, 'ansi' or 'rgb_array'")
        self.chapter, self.render_mode = int(chapter), render_mode
        if not isinstance(scenario, DirectScenario):
            raise TypeError("DirectIrrigationEnv requires an explicit DirectScenario; use make_env for Hongmiao93")
        # Re-run the dataclass boundary: ordinary deepcopy makes readonly NumPy
        # arrays writable and can silently detach a quota from its fingerprint.
        self.scenario = replace(scenario)
        self.N, self.T, self.C = self.scenario.N, self.scenario.T, 1
        self._areas = np.asarray(self.scenario.areas_ha, dtype=np.float64).copy()
        self._initial_storage = np.asarray(self.scenario.initial_storage_mm, dtype=np.float64).copy()
        self._capacities = np.array([p.max_store_mm for p in self.scenario.field_params])
        self._field_parameters = np.array([[getattr(p, key) for key in FIELD_PARAMETER_NAMES]
                                          for p in self.scenario.field_params], dtype=np.float64)
        self._maturity_index = np.array([
            (h + timedelta(days=int(self.scenario.maturity_offset_days)) - self.scenario.start_date).days
            for h in self.scenario.heading_dates], dtype=np.int64)
        if np.any(self._maturity_index <= 0) or (self.scenario.evaluation_mode == "maturity" and np.any(self._maturity_index > self.T)):
            raise ValueError("all crops must mature after reset and within the episode; extend explicit weather and quota inputs")
        if chapter == 3 and total_budget_m3 is not None:
            raise ValueError("chapter 3 uses daily quotas, not a carryable total budget")
        self._total_budget = _real(math.fsum(self.scenario.quota_plan) if total_budget_m3 is None
                                   else total_budget_m3, "total_budget_m3")
        self._daily_limit_config = None
        if daily_limit_m3 is None:
            self._daily_limit = np.full(self.T, np.inf)
        else:
            raw = np.asarray(daily_limit_m3)
            if raw.shape not in ((), (self.T,)):
                raise ValueError("daily_limit_m3 must be a scalar or T-vector")
            self._daily_limit = _array(np.broadcast_to(raw, (self.T,)), (self.T,), "daily_limit_m3")
            self._daily_limit_config = self._daily_limit.tolist()
        if chapter == 3 and np.any(self.scenario.quota_plan > self._daily_limit):
            raise ValueError("chapter 3 daily quota exceeds the explicitly declared daily limit")
        self.local_weather_spread = _real(local_weather_spread, "local_weather_spread")
        if self.local_weather_spread >= 1:
            raise ValueError("local_weather_spread must be less than one")
        delay = 0 if report_delay_days is None else report_delay_days
        raw = np.asarray(delay)
        if (raw.dtype.kind not in "iu" or raw.shape not in ((), (self.N,))
                or np.any(raw < 0) or np.any(raw > np.iinfo(np.int64).max)):
            raise ValueError("report_delay_days must be a nonnegative integer or N-vector")
        self._report_delay = np.broadcast_to(raw, (self.N,)).astype(np.int64).copy()
        if chapter != 6 and np.any(self._report_delay):
            raise ValueError("chapters 3--5 observe current reports")
        raw_lambdas = np.asarray(yield_lambdas)
        if raw_lambdas.shape == (self.N, 4):
            yield_lambdas = raw_lambdas[:, None, :]
        self._ledger = OutcomeLedger(self._areas[:, None], yield_lambdas)
        self._yield_lambdas = self._ledger.lambdas[:, 0, :]
        self._weather = self._make_weather_driver(weather_mode, historical_year)
        self._weather_vector_size = getattr(self._weather, "vector_size", 9)
        self._forecast_size = getattr(self._weather, "forecast_size", 3)
        self._weather_mode, self._historical_year = weather_mode, int(historical_year)
        self._day, self._spent, self._return = 0, 0., 0.
        self._storage = self._initial_storage.copy()
        self._previous_weather = np.zeros(self._weather_vector_size)
        self._previous_class = 0
        self._local_rng = np.random.default_rng()
        self._history, self._actions = [], []
        self._last_info, self._diagnostics = {}, {}
        self._initialized = False
        self._fingerprint = self._scenario_fingerprint()
        self.action_space = DirectAllocationSpace(self)
        self.observation_space = self._make_observation_space()

    def _make_weather_driver(self, weather_mode, historical_year):
        """Override only the weather input protocol; preserve daily physics."""
        return WeatherDriver(self.scenario, weather_mode, historical_year)

    def _weather_class_from_vector(self, vector):
        if hasattr(self._weather, "class_from_vector"):
            return self._weather.class_from_vector(vector)
        WeatherDay(self.scenario.start_date, vector, 0.)
        return weather_class(vector)

    def _settlement_mask(self, day):
        if self.scenario.evaluation_mode == "terminal_projection":
            return np.full(self.N, day >= self.T, dtype=bool)
        return self._maturity_index <= day

    def _settle_ledger(self, ledger, day):
        mask = self._settlement_mask(day)
        self._check_settlement_coverage(ledger, mask)
        if self.scenario.evaluation_mode == "terminal_projection":
            return ledger.evaluate_terminal(mask[:, None])
        return ledger.settle(mask[:, None])

    def _scenario_fingerprint(self):
        s = self.scenario
        payload = {"schema": "direct-field-v4", "ids": s.region_ids,
            "fields": [asdict(p) for p in s.field_params], "initial": self._initial_storage,
            "headings": s.heading_dates, "maturity_offset": s.maturity_offset_days,
            "evaluation_mode": s.evaluation_mode,
            "quota": s.quota_plan, "start": s.start_date, "substeps": s.field_substeps_per_day,
            "budget_mode": "fixed_daily_total" if self.chapter == 3 else "carryover",
            "total_budget": self._total_budget, "daily_limit": self._daily_limit_config,
            "report_delay": self._report_delay, "local_spread": self.local_weather_spread,
            "lambdas": self._yield_lambdas, "weather_mode": self._weather_mode,
            "historical_year": self._historical_year, "weather_contract": s.weather_contract,
            "weather_driver_contract": getattr(self._weather, "contract", None),
            "weather": [(key, record.vector, record.eto_mm_day) for key, record in sorted(s.weather_records.items())],
            "donor_pools": s.donor_pools,
            "forecasts": sorted((k, v) for k, v in s.forecast_table.items())}
        raw = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _make_observation_space(self):
        nn = lambda shape: spaces.Box(0., np.inf, shape=shape, dtype=np.float64)
        integer = lambda low, high, shape: spaces.Box(low, high, shape=shape, dtype=np.int64)
        return spaces.Dict({
            "day_index": integer(0, self.T, ()), "horizon_days": integer(self.T, self.T, ()),
            "rice_area_ha": nn((self.N,)), "field_params": nn(self._field_parameters.shape),
            "field_storage_capacity_mm": nn((self.N,)),
            "reported_field_storage_mm": nn((self.N,)),
            "reported_stage_et_mm": nn((self.N, 4)),
            "reported_stage_reference_et_mm": nn((self.N, 4)),
            "heading_age_days": integer(-1000000, 1000000, (self.N,)),
            "days_to_maturity": integer(0, max(self.T, int(self._maturity_index.max())), (self.N,)),
            "stage_index": integer(0, 5, (self.N,)),
            "settled_mask": spaces.Box(0, 1, shape=(self.N,), dtype=np.bool_),
            "settled_relative_yield": spaces.Box(0., 1., shape=(self.N,), dtype=np.float64),
            "report_day_index": integer(0, self.T, (self.N,)),
            "report_age_days": integer(0, self.T, (self.N,)),
            "available_water_m3": nn(()), "remaining_budget_m3": nn(()),
            "daily_quota_plan_m3": nn((self.T,)), "daily_limit_plan_m3": nn((self.T,)),
            "budget_is_carryable": spaces.Box(0, 1, shape=(), dtype=np.bool_),
            "previous_weather": spaces.Box(-np.inf, np.inf, shape=(self._weather_vector_size,), dtype=np.float64),
            "weather_forecast": nn((self._forecast_size,)), "yield_lambdas": nn((self.N, 4)),
        })

    def _remaining(self):
        if self.chapter == 3:
            return float(math.fsum(self.scenario.quota_plan[self._day:]))
        return max(0., self._total_budget - self._spent)

    def _allowance(self):
        if self._day >= self.T:
            return 0.
        resource = float(self.scenario.quota_plan[self._day]) if self.chapter == 3 else self._remaining()
        return min(resource, float(self._daily_limit[self._day]))

    def _active_mask(self):
        return (self._maturity_index > self._day) & (self._day < self.T)

    def action_constraints(self):
        """Only public budget/calendar limits; no hidden field-water bounds."""
        active = self._active_mask()
        return {"lower_m3": np.zeros(self.N),
                "upper_m3": np.where(active, self._allowance(), 0.),
                "total_limit_m3": self._allowance(), "active_mask": active.copy(),
                "meter": "field_inlet",
                "sum_constraint": "equal" if self.chapter == 3 else "less_than_or_equal"}

    def _validate_action(self, action):
        a = _array(action, (self.N,), "allocation")
        if np.any(a[~self._active_mask()] != 0.):
            raise ValueError("regions that have matured or left the allocation window must receive zero new irrigation")
        total = math.fsum(a.tolist())
        # A small floating arithmetic tolerance is not physical action repair.
        tolerance = 1e-10 + 1e-12 * self._allowance()
        if total > self._allowance() + tolerance:
            raise ValueError("allocation exceeds the available field-inlet budget")
        if self.chapter == 3 and (abs(total - self._allowance()) > tolerance
                                  or (self._allowance() == 0. and total != 0.)):
            raise ValueError("chapter 3 allocation must equal the fixed daily total")
        return a, total

    def feasible_allocation(self, weights=None, *, fraction=1., selected_mask=None):
        """Convert caller-selected nonnegative weights to a physical action.

        The optional boolean mask belongs to the caller/policy. Unselected
        entries and exact zero weights stay zero; no moisture threshold,
        minimum dose, cooldown, epsilon or dense fallback is applied. Omitting
        weights explicitly requests equal weights within the selected set.
        Explicit all-zero weights yield zero, subject to the chapter budget
        check (therefore rejected for a positive fixed chapter-three quota).
        This optional constructor never modifies an action supplied to step.
        """
        fraction = _real(fraction, "fraction")
        if fraction > 1:
            raise ValueError("fraction must be between zero and one")
        if self.chapter == 3 and self._allowance() > 0 and fraction != 1.:
            raise ValueError("chapter 3 requires fraction=1 for the fixed daily total")
        w = np.ones(self.N) if weights is None else _array(weights, (self.N,), "weights")
        selected = (np.ones(self.N, dtype=bool) if selected_mask is None else
                    _boolean_mask(selected_mask, self.N, "selected_mask"))
        w = np.where(self._active_mask() & selected, w, 0.)
        if not np.any(w) or self._allowance() == 0 or fraction == 0:
            action = np.zeros(self.N)
            self._validate_action(action)
            return action
        w /= np.max(w)
        action = w / math.fsum(w.tolist()) * (self._allowance() * fraction)
        self._validate_action(action)
        return action

    def _current_report(self):
        return {"field_storage_mm": self._storage.copy(),
                "stage_et_mm": self._ledger.actual_et_mm[:, 0, :],
                "stage_reference_et_mm": self._ledger.reference_et_mm[:, 0, :]}

    def _crop_profiles(self, day):
        """Optional explicit public calendar; None uses the explicit scenario heading calendar."""
        return None

    def _public_stages(self, day):
        profiles = self._crop_profiles(day)
        if profiles is not None:
            return np.asarray(profiles[:, 0], dtype=np.int64)
        return np.array([calendar_profile(day, h, self.scenario.maturity_offset_days)[0]
                         for h in self.scenario.heading_dates], dtype=np.int64)

    def _outcome_stages(self, day):
        profiles = self._crop_profiles(day)
        if profiles is not None:
            return np.asarray(profiles[:, 3], dtype=np.int64)
        return np.array([calendar_profile(day, h, self.scenario.maturity_offset_days)[2]
                         for h in self.scenario.heading_dates], dtype=np.int64)

    def _format_settlements(self, records):
        for record in records:
            record["region_id"] = self.scenario.region_ids[record["region_index"]]
            record.pop("cohort_index", None)
        return records

    def _check_settlement_coverage(self, ledger, selected):
        """Case-specific full-season validation before committing a transition."""

    def _observation(self):
        report_days = np.maximum(0, self._day - self._report_delay)
        history = self._history or [self._current_report()]
        day = self.scenario.start_date + timedelta(days=self._day)
        obs = {"reported_" + key: np.array([history[int(d)][key][i]
                        for i, d in enumerate(report_days)]) for key in REPORT_KEYS}
        obs.update({"day_index": np.array(self._day, dtype=np.int64),
            "horizon_days": np.array(self.T, dtype=np.int64),
            "rice_area_ha": self._areas.copy(), "field_params": self._field_parameters.copy(),
            "field_storage_capacity_mm": self._capacities.copy(),
            "heading_age_days": np.array([(day - h).days for h in self.scenario.heading_dates], dtype=np.int64),
            "days_to_maturity": np.maximum(0, self._maturity_index - self._day),
            "stage_index": self._public_stages(day),
            "settled_mask": self._ledger.settled_mask[:, 0],
            "settled_relative_yield": self._ledger.settled_relative_yield[:, 0],
            "report_day_index": report_days.copy(), "report_age_days": self._day - report_days,
            "available_water_m3": np.array(self._allowance()),
            "remaining_budget_m3": np.array(self._remaining()),
            "daily_quota_plan_m3": np.array(self.scenario.quota_plan, copy=True),
            "daily_limit_plan_m3": np.minimum(self._daily_limit, self._total_budget),
            "budget_is_carryable": np.array(self.chapter != 3),
            "previous_weather": self._previous_weather.copy(),
            "weather_forecast": self._weather.forecast(self._day, self._previous_class)
                                 if self._day < self.T else np.zeros(self._forecast_size),
            "yield_lambdas": self._yield_lambdas.copy()})
        return obs

    def _episode_initial_storage(self):
        """Initial storage for this episode; derived cases may sample at reset."""
        return self._initial_storage

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("reset options are unsupported; pass a DirectScenario to the constructor")
        if seed is not None:
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
                raise ValueError("seed must be a nonnegative integer or None")
            seed = int(seed)
        # Build the new episode before changing the live one. In particular,
        # an invalid seed or weather reset cannot leave an empty outcome ledger
        # attached to report history from the previous episode.
        weather = copy.copy(self._weather)
        if self._initialized:
            weather.restore(self._weather.snapshot())
        previous, previous_class = weather.reset(seed)
        ledger = OutcomeLedger(self._areas[:, None], self._yield_lambdas[:, None, :])
        local_rng = (np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(610,)))
                     if seed is not None else copy.deepcopy(self._local_rng))
        super().reset(seed=seed)
        self._day, self._spent, self._return = 0, 0., 0.
        self._storage = self._episode_initial_storage().copy()
        self._ledger, self._weather, self._local_rng = ledger, weather, local_rng
        self._previous_weather, self._previous_class = previous, previous_class
        self._history = [self._current_report()]
        self._actions = []
        self._last_info, self._diagnostics = {}, {}
        self._initialized = True
        return self._observation(), {"environment_version": "direct-v4", "resource_meter": "field_inlet",
            "reward_identity": "area_weighted_relative_yield_" + self.scenario.evaluation_mode, "objective_discount": 1.,
            "evaluation_mode": self.scenario.evaluation_mode,
            "budget_mode": "fixed_daily_total" if self.chapter == 3 else "carryover_budget",
            "maturity_identity": "declared_heading_plus_offset_calendar",
            "report_identity": "research_historical_state_reports",
            "yield_identity": "external_Jensen_window_conditioned_not_local_calibration"}

    def _advance_physics_day(self, **kwargs):
        """Execute the default transition; explicit research cases may override it."""
        return advance_direct_day(**kwargs)

    def step(self, action):
        if not self._initialized:
            raise RuntimeError("reset must be called before step")
        if self._day >= self.T:
            raise RuntimeError("episode terminated; call reset")
        a, allocated = self._validate_action(action)
        # Work on independent stochastic/accounting objects until every check
        # passes. Rejected input and failed execution cannot consume randomness.
        weather = copy.copy(self._weather)
        weather.restore(self._weather.snapshot())
        local_rng = copy.deepcopy(self._local_rng)
        realized = weather.sample(self._day, self._previous_class)
        noise = local_rng.uniform(-self.local_weather_spread, self.local_weather_spread, (2, self.N))
        rain = realized.rain_mm * (1. + noise[0])
        eto = realized.eto_mm_day * (1. + noise[1])
        day = self.scenario.start_date + timedelta(days=self._day)
        result = self._advance_physics_day(field_params=self.scenario.field_params,
            storage_mm=self._storage, heading_dates=self.scenario.heading_dates, day=day,
            maturity_offset_days=self.scenario.maturity_offset_days, action_m3=a,
            rain_mm=rain, eto_mm_day=eto, substeps_per_day=self.scenario.field_substeps_per_day,
            crop_profiles=self._crop_profiles(day))
        ledger = copy.deepcopy(self._ledger)
        ledger.accumulate(result["outcome_stage_index"][:, None],
                          result["daily_et_mm"][:, None], result["daily_reference_et_mm"][:, None])
        reward, settlements = self._settle_ledger(ledger, self._day + 1)
        if self._day + 1 == self.T and not ledger.all_settled:
            raise ArithmeticError("unsettled crops at terminal boundary")
        settlements = self._format_settlements(settlements)
        info = {"day_index_executed": self._day, "date": day.isoformat(),
                "executed_action_m3": a.copy(), "allocated_water_m3": allocated,
                "settlements": settlements, "objective_discount": 1., "mass_balance_ok": True}
        self._weather, self._local_rng, self._ledger = weather, local_rng, ledger
        self._storage = result["field_storage_mm"]
        self._previous_weather, self._previous_class = realized.vector.copy(), realized.weather_class
        self._day += 1
        self._spent += allocated
        self._return += reward
        self._actions.append(a.copy())
        self._history.append(self._current_report())
        info["remaining_budget_m3"] = self._remaining()
        self._last_info, self._diagnostics = copy.deepcopy(info), result
        return self._observation(), float(reward), self._day == self.T, False, info

    def private_diagnostics(self):
        """Evaluator-only truth, never to be passed as a policy observation."""
        result = copy.deepcopy(self._diagnostics)
        result.update(self._current_report())
        for key in ("daily_et_mm", "daily_drainage_mm", "daily_overflow_mm",
                    "daily_irrigation_mm", "daily_rain_mm", "daily_reference_et_mm", "last_action_m3"):
            result.setdefault(key, np.zeros(self.N))
        result.setdefault("water_mass_residual_m3", 0.)
        conversion = 10. * self._areas
        totals = {name: math.fsum((np.asarray(result[key]) * conversion).tolist())
                  for name, key in (("irrigation_m3", "daily_irrigation_mm"),
                                    ("rain_m3", "daily_rain_mm"), ("et_m3", "daily_et_mm"),
                                    ("drainage_m3", "daily_drainage_mm"),
                                    ("overflow_m3", "daily_overflow_mm"))}
        totals["water_mass_residual_m3"] = result["water_mass_residual_m3"]
        result["daily_physics"] = totals
        return result

    def episode_summary(self):
        """Public outcomes/account totals only; no current hidden field state."""
        return {"days_executed": self._day, "horizon_days": self.T, "return": self._return,
                "allocated_water_m3": self._spent, "remaining_budget_m3": self._remaining(),
                "water_used_m3": self._spent, "complete": self._day == self.T, "regions": self.N,
                "settled_regions": int(np.count_nonzero(self._ledger.settled_mask)),
                "all_settled": self._ledger.all_settled, "objective_discount": 1.,
                "evaluation_mode": self.scenario.evaluation_mode,
                "resource_meter": "field_inlet"}

    def allocation_pattern(self):
        """Describe executed allocations, not eligibility for the next action.

        History comes only from actual public actions. Zero irrigation does
        not freeze soil water or ET. These display/audit statistics are not
        extra policy observations and impose no mandatory waiting period.
        A region never irrigated has days_since_last_irrigation=-1; an
        irrigation in the most recent executed step has age zero.
        """
        last = np.full(self.N, -1, dtype=np.int64)
        for index, action in enumerate(self._actions):
            last[np.asarray(action) > 0.] = index
        elapsed = len(self._actions)
        current = (np.asarray(self._actions[-1]) > 0. if elapsed else
                   np.zeros(self.N, dtype=bool))
        age = np.where(last >= 0, elapsed - 1 - last, -1)
        return {"has_step": bool(elapsed), "irrigated_mask": current.copy(),
                "irrigated_regions": int(np.count_nonzero(current)),
                "non_irrigated_regions": int(self.N - np.count_nonzero(current)),
                "consecutive_non_irrigation_days": np.where(last >= 0, age, elapsed),
                "days_since_last_irrigation": age.copy()}

    def _render_transition_start(self):
        """Physical boundary at which the most recent policy action was made."""
        return max(0, self._day - 1)

    def render_context(self):
        """Detached public context for the preceding allocation and reports.

        The two water reports are reconstructed from their delivery timestamps,
        never from current private storage. A delayed report difference is not
        presented as the effect of the most recent allocation. For chapter 3,
        ``remaining_after_m3`` is the numerical difference from the fixed daily
        total, not carryover; ``next_available_m3`` is the separate next total.
        This method adds no state and draws no random numbers.
        """
        if not self._initialized:
            raise RuntimeError("reset must be called before rendering step context")
        day = self._day
        has_step = day > 0
        start_day = self._render_transition_start()
        carryable = self.chapter != 3
        current_indices = np.maximum(0, day - self._report_delay)
        before_storage = None
        previous_indices = None
        current_transition = np.zeros(self.N, dtype=bool)
        net_other = [None] * self.N
        allocated, unused, reward, newly_settled = 0., 0., 0., 0
        before_allowance = self._allowance()
        if has_step:
            previous_indices = np.maximum(0, start_day - self._report_delay)
            before_storage = np.array([self._history[int(t)]["field_storage_mm"][i]
                                       for i, t in enumerate(previous_indices)])
            after_storage = np.array([self._history[int(t)]["field_storage_mm"][i]
                                      for i, t in enumerate(current_indices)])
            current_transition = (previous_indices == start_day) & (current_indices == day)
            action = np.asarray(self._actions[start_day], dtype=np.float64)
            allocated = math.fsum(action.tolist())
            if carryable:
                before_resource = max(0., self._total_budget - (self._spent - allocated))
            else:
                before_resource = float(self.scenario.quota_plan[start_day])
            before_allowance = min(before_resource, float(self._daily_limit[start_day]))
            unused = max(0., before_allowance - allocated)
            settlements = self._last_info["settlements"]
            reward = math.fsum(float(r["reward_contribution"]) for r in settlements)
            newly_settled = len(settlements)
            difference = after_storage - before_storage - action / (10. * self._areas)
            net_other = [float(value) if known else None
                         for value, known in zip(difference, current_transition)]
        previous_label = ("尚无执行前状态" if not has_step else
                          "执行前田地水量" if np.all(current_transition) else
                          "上次决策时收到的水分报告")
        return _jsonable({
            "day": day,
            "executed_date": (self.scenario.start_date + timedelta(days=start_day)).isoformat()
                             if has_step else None,
            "state_date": (self.scenario.start_date + timedelta(days=day)).isoformat(),
            "has_step": has_step,
            "allocation_pattern": self.allocation_pattern(),
            "episode_complete": day >= self.T,
            "executed_day_index": start_day if has_step else None,
            "physical_days_advanced": day - start_day if has_step else 0,
            "available_before_m3": before_allowance,
            "allocated_m3": allocated,
            "executed_action_m3": self._actions[start_day] if has_step else np.zeros(self.N),
            "unused_allowance_m3": unused,
            "remaining_after_m3": self._remaining() if carryable else unused,
            "carryable": carryable,
            "budget_after_label": ("回合未用总预算" if day >= self.T else "剩余可结转预算") if carryable else "本日总量差额",
            "next_available_m3": self._allowance(),
            "reward": reward,
            "cumulative_return": self._return,
            "newly_settled_count": newly_settled,
            "evaluation_mode": self.scenario.evaluation_mode,
            "settled_count": int(np.count_nonzero(self._ledger.settled_mask)),
            "previous_reported_storage_mm": before_storage,
            "previous_report_day_index": previous_indices,
            "current_report_day_index": current_indices,
            "report_age_days": day - current_indices,
            "transition_is_current": current_transition,
            "public_net_other_mm": net_other,
            "previous_storage_label": previous_label,
        })

    def get_state(self):
        """Private JSON checkpoint including RNG state and report history."""
        if not self._initialized:
            raise RuntimeError("reset must be called before checkpointing")
        return _jsonable({"schema": "direct-env-v4", "fingerprint": self._fingerprint,
            "day": self._day, "storage_mm": self._storage, "spent_m3": self._spent,
            "return": self._return, "previous_weather": self._previous_weather,
            "previous_class": self._previous_class, "weather": self._weather.snapshot(),
            "local_rng": self._local_rng.bit_generator.state,
            "gym_rng": self.np_random.bit_generator.state,
            "action_rng": self.action_space.np_random.bit_generator.state,
            "ledger": self._ledger.snapshot(), "reports": self._history,
            "actions": self._actions, "last_info": self._last_info,
            "diagnostics": self._diagnostics})

    def set_state(self, checkpoint):
        """Atomically restore a matching, validated evaluator checkpoint."""
        if not isinstance(checkpoint, Mapping):
            raise ValueError("checkpoint must be a mapping")
        state = copy.deepcopy(checkpoint)
        if state.get("schema") != "direct-env-v4" or state.get("fingerprint") != self._fingerprint:
            raise ValueError("checkpoint belongs to a different direct environment")
        day = state.get("day")
        if isinstance(day, bool) or not isinstance(day, Integral) or not 0 <= day <= self.T:
            raise ValueError("invalid checkpoint day")
        day = int(day)
        storage = _array(state.get("storage_mm"), (self.N,), "checkpoint field storage")
        for params, value in zip(self.scenario.field_params, storage):
            partition(params, FieldState(float(value)))
        spent = _real(state.get("spent_m3"), "checkpoint allocated water")
        total_return = _real(state.get("return"), "checkpoint return")
        previous = _array(state.get("previous_weather"), (self._weather_vector_size,), "previous weather", minimum=-np.inf)
        cls = state.get("previous_class")
        if isinstance(cls, bool) or not isinstance(cls, Integral) or int(cls) != self._weather_class_from_vector(previous):
            raise ValueError("checkpoint weather class does not match observed weather")
        ledger = OutcomeLedger(self._areas[:, None], self._yield_lambdas[:, None, :])
        ledger.restore(state.get("ledger"))
        if not np.array_equal(ledger.settled_mask[:, 0], self._settlement_mask(day)):
            raise ValueError("checkpoint settlement flags disagree with calendar")
        if not math.isclose(total_return, ledger.settled_return, abs_tol=1e-10, rel_tol=1e-10):
            raise ValueError("checkpoint return disagrees with settled outcomes")
        expected_counts = np.zeros((self.N, 1, 4), dtype=np.int64)
        for t in range(day):
            crop_day = self.scenario.start_date + timedelta(days=t)
            for i, stage in enumerate(self._outcome_stages(crop_day)):
                if stage >= 0:
                    expected_counts[i, 0, stage] += 1
        if not np.array_equal(ledger.stage_observations, expected_counts):
            raise ValueError("checkpoint stage counts disagree with elapsed calendar")
        self._check_settlement_coverage(ledger, self._settlement_mask(day))
        actions = state.get("actions")
        if not isinstance(actions, list) or len(actions) != day:
            raise ValueError("checkpoint action history length mismatch")
        restored_actions, amounts = [], []
        for t, raw in enumerate(actions):
            a = _array(raw, (self.N,), "checkpoint allocation")
            amount = math.fsum(a.tolist())
            if np.any(a[self._maturity_index <= t] != 0):
                raise ValueError("checkpoint irrigates a mature region")
            remaining = self._total_budget - math.fsum(amounts)
            limit = min(float(self._daily_limit[t]),
                        float(self.scenario.quota_plan[t]) if self.chapter == 3 else max(0., remaining))
            if amount > limit + 1e-10 + 1e-12 * limit:
                raise ValueError("checkpoint action exceeded its available budget")
            if self.chapter == 3 and (abs(amount - limit) > 1e-10 + 1e-12 * limit
                                      or (limit == 0. and amount != 0.)):
                raise ValueError("checkpoint chapter 3 action differs from its fixed daily total")
            amounts.append(amount)
            restored_actions.append(a)
        if not math.isclose(spent, math.fsum(amounts), abs_tol=1e-8, rel_tol=1e-12):
            raise ValueError("checkpoint resource account disagrees with action history")
        reports = state.get("reports")
        if not isinstance(reports, list) or len(reports) != day + 1:
            raise ValueError("checkpoint report history length mismatch")
        restored_reports = []
        for raw in reports:
            if not isinstance(raw, Mapping) or set(raw) != set(REPORT_KEYS):
                raise ValueError("checkpoint report fields mismatch")
            record = {key: _array(raw[key], (self.N,) if key == "field_storage_mm" else (self.N, 4),
                                  "checkpoint " + key) for key in REPORT_KEYS}
            if np.any(record["field_storage_mm"] > self._capacities):
                raise ValueError("checkpoint report exceeds field capacity")
            if np.any(record["stage_et_mm"] > record["stage_reference_et_mm"] + 1e-8):
                raise ValueError("checkpoint report actual ET exceeds reference")
            if restored_reports:
                for key in REPORT_KEYS[1:]:
                    if np.any(record[key] < restored_reports[-1][key]):
                        raise ValueError("checkpoint cumulative ET decreased")
            restored_reports.append(record)
        first, latest = restored_reports[0], restored_reports[-1]
        if (not np.array_equal(first["field_storage_mm"], self._episode_initial_storage())
                or np.any(first["stage_et_mm"]) or np.any(first["stage_reference_et_mm"])):
            raise ValueError("checkpoint initial report differs from scenario")
        if (not np.array_equal(latest["field_storage_mm"], storage)
                or not np.array_equal(latest["stage_et_mm"], ledger.actual_et_mm[:, 0, :])
                or not np.array_equal(latest["stage_reference_et_mm"], ledger.reference_et_mm[:, 0, :])):
            raise ValueError("checkpoint latest report differs from physical/ET state")
        weather = copy.copy(self._weather)
        weather.restore(state.get("weather"))
        local_rng = _rng_from_state(state.get("local_rng"))
        gym_rng = _rng_from_state(state.get("gym_rng"))
        action_rng = _rng_from_state(state.get("action_rng"))
        last_info, diagnostics = self._restore_output_records(state, day, spent, storage,
                                                              ledger, restored_reports, restored_actions)
        restored = {
            "_storage": storage, "_ledger": ledger,
            "_day": day, "_spent": spent, "_return": total_return,
            "_previous_weather": previous, "_previous_class": int(cls),
            "_weather": weather, "_local_rng": local_rng, "_np_random": gym_rng,
            "_history": restored_reports, "_actions": restored_actions,
            "_last_info": last_info, "_diagnostics": diagnostics,
            "_initialized": True,
        }
        # Derived cases can impose stronger observation invariants than this
        # numerical core (e.g. shared reference ET in Hongmiao). Validate those
        # against a candidate before replacing any live state or RNG. A bad
        # checkpoint must not be accepted only to fail at the next observation.
        candidate = copy.copy(self)
        candidate.__dict__.update(restored)
        observation = candidate._observation()
        if not self.observation_space.contains(observation):
            raise ValueError("checkpoint produces an invalid policy observation")
        self.__dict__.update(restored)
        self.action_space._np_random = action_rng

    def _restore_output_records(self, state, day, spent, storage, ledger, reports, actions):
        """Validate saved outputs as carefully as replay-critical state."""
        info, raw = state.get("last_info"), state.get("diagnostics")
        if not isinstance(info, Mapping) or not isinstance(raw, Mapping):
            raise ValueError("checkpoint output records must be mappings")
        if day == 0:
            if info or raw:
                raise ValueError("reset checkpoint must have empty last-step records")
            return {}, {}
        positive_arrays = ("field_storage_mm", "daily_et_mm", "daily_transpiration_mm",
            "daily_evaporation_mm", "daily_drainage_mm", "daily_overflow_mm",
            "daily_reference_et_mm", "daily_irrigation_mm", "daily_rain_mm", "daily_eto_mm", "last_action_m3")
        keys = set(positive_arrays) | {"stage_index", "outcome_stage_index", "water_mass_residual_m3"}
        if set(raw) != keys:
            raise ValueError("checkpoint diagnostic fields do not match the direct kernel")
        clean = {key: _array(raw[key], (self.N,), "checkpoint " + key) for key in positive_arrays}
        for key, low, high in (("stage_index", 0, 5), ("outcome_stage_index", -1, 3)):
            arr = np.asarray(raw[key])
            if (arr.dtype.kind not in "iu" or arr.shape != (self.N,)
                    or np.any(arr < low) or np.any(arr > high)):
                raise ValueError("invalid checkpoint diagnostic stage index")
            clean[key] = arr.astype(np.int64, copy=True)
        clean["water_mass_residual_m3"] = _real(raw["water_mass_residual_m3"],
                                               "checkpoint mass residual", minimum=-np.inf)
        executed_day = self.scenario.start_date + timedelta(days=day - 1)
        if (not np.array_equal(clean["stage_index"], self._public_stages(executed_day))
                or not np.array_equal(clean["outcome_stage_index"], self._outcome_stages(executed_day))):
            raise ValueError("checkpoint diagnostic stages disagree with calendar")
        if (not np.array_equal(clean["field_storage_mm"], storage)
                or not np.array_equal(clean["last_action_m3"], actions[-1])
                or not np.allclose(clean["daily_irrigation_mm"] * (10. * self._areas), actions[-1],
                                   atol=1e-8, rtol=1e-12)):
            raise ValueError("checkpoint diagnostics disagree with storage or executed allocation")
        if (not np.allclose(clean["daily_et_mm"], clean["daily_transpiration_mm"] + clean["daily_evaporation_mm"],
                            atol=1e-10, rtol=1e-12)
                or np.any(clean["daily_et_mm"] > clean["daily_reference_et_mm"] + 1e-8)):
            raise ValueError("checkpoint ET diagnostic components disagree")
        residual = (reports[-2]["field_storage_mm"] + clean["daily_irrigation_mm"] + clean["daily_rain_mm"]
                    - clean["daily_et_mm"] - clean["daily_drainage_mm"] - clean["daily_overflow_mm"] - storage)
        if np.any(np.abs(residual) > 1e-7 * np.maximum(1., reports[-2]["field_storage_mm"]
                                                       + clean["daily_irrigation_mm"] + clean["daily_rain_mm"])):
            raise ValueError("checkpoint diagnostic water balance does not close")
        expected_residual = math.fsum((residual * (10. * self._areas)).tolist())
        if not math.isclose(expected_residual, clean["water_mass_residual_m3"], abs_tol=1e-7, rel_tol=1e-8):
            raise ValueError("checkpoint diagnostic mass residual disagrees")
        # Reconstruct the last evaluation records from validated accumulators.
        prior = ledger.snapshot()
        old_mask = self._settlement_mask(day - 1)[:, None]
        prior["settled_mask"] = old_mask.tolist()
        prior["settled_relative_yield"] = np.where(old_mask, ledger.settled_relative_yield, 0.).tolist()
        prior_ledger = OutcomeLedger(self._areas[:, None], self._yield_lambdas[:, None, :])
        prior_ledger.restore(prior)
        _, records = self._settle_ledger(prior_ledger, day)
        records = self._format_settlements(records)
        remaining = (math.fsum(self.scenario.quota_plan[day:]) if self.chapter == 3
                     else max(0., self._total_budget - spent))
        expected = {"day_index_executed": day - 1, "date": executed_day.isoformat(),
                    "executed_action_m3": actions[-1].copy(),
                    "allocated_water_m3": math.fsum(actions[-1].tolist()),
                    "settlements": records, "objective_discount": 1., "mass_balance_ok": True,
                    "remaining_budget_m3": remaining}
        if _jsonable(info) != _jsonable(expected):
            raise ValueError("checkpoint last-step information disagrees with validated state")
        return expected, clean

    def render(self):
        if self.render_mode == "ansi":
            return (f"day={self._day}/{self.T} regions={self.N} return={self._return:.6f} "
                    f"available_field_water_m3={self._allowance():.0f}")
        if self.render_mode == "rgb_array":
            from .direct_visualization import render_observation
            context = self.render_context()
            return render_observation(self._observation(), self.scenario.region_ids, chapter=self.chapter,
                last_action=context["executed_action_m3"], mode="rgb_array", context=context)
        return None

    def close(self):
        pass
