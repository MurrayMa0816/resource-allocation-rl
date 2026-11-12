"""Pure public-observation interface for the chapter-six irrigation task.

The raw environment keeps physical units.  This module supplies a documented,
invertible linear encoding and an executed-action resource ledger; it never
reads a simulator, diagnostic, future weather, reward, or random generator.

The five returned state keys match the *structure* used by the existing
allocation methods.  The old two-domain Task/Spec/rollout factories do not
thereby acquire irrigation support.  No algorithm or training loop is defined
here, and the objective preference is appended separately, once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real

import numpy as np


STATE_KEYS = ("local", "global", "caps", "budget", "remaining")
OBSERVATION_KEYS = frozenset((
    "day_index", "available_water_m3", "rice_area_ha",
    "root_available_water_mm", "percolation_sat_mm_day",
    "reported_field_storage_mm", "reported_completed_yield_factor",
    "reported_current_stage_et_mm", "reported_current_stage_reference_et_mm",
    "previous_weather",
))
LOCAL_FEATURES = (
    "rice_area_ha", "percolation_sat_mm_day", "root_available_water_mm",
    "reported_field_storage_mm", "reported_completed_yield_factor",
    "reported_current_stage_et_mm",
)
GLOBAL_FEATURES = (
    "day_fraction", "remaining_budget_fraction",
    "reported_current_stage_reference_et_mm", "previous_rain_mm",
    "previous_reference_et_mm_day",
)
OBSERVATION_SCHEMA = "hongmiao89-high-yield-compact-observation-v1"
FEATURE_SCHEMA = "irrigation89-high-yield-public-six-local-five-global-v1"


def _real(value, name, *, nonnegative=True):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value))):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _numeric(value, shape, name, *, nonnegative=False):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real array with shape {shape}")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all() or (nonnegative and np.any(result < 0)):
        raise ValueError(f"{name} must contain finite" + (" nonnegative values" if nonnegative else " values"))
    return result


def _tolerance(remaining):
    # This is the frozen physical engine's validation tolerance, not a dose,
    # permission to repair an executed action, or a feature normalization.
    return 1e-10 + 1e-12 * float(remaining)


@dataclass(frozen=True)
class IrrigationSpec:
    """Immutable public geometry and interface metadata, independent of Spec.

    ``application_chapter`` identifies the dissertation chapter.  It is not
    the historical method-factory chapter code (2/3), which this specification
    deliberately does not impersonate.  ``nominal_total`` is an initial pacing
    reference only; it imposes no daily use or delivery limit.  ``caps`` in a
    state are redundant seasonal-balance bounds, not measured regional gates.

    The dictionary fields permit read-only construction of existing model
    classes.  They neither choose a learned policy nor import method code.
    Zero total supply is a valid environment/control case; existing model
    constructors requiring positive scales need a positive-budget task.
    """

    groups: int
    horizon: int
    initial_budget: float
    region_ids: tuple[str, ...]
    adapter: str = 'unit'
    area_ha: tuple = ()
    spatial_sigma_scale: float = 1.
    domain: str = field(default="hongmiao89_reasonable_depth_irrigation", init=False)
    application_chapter: int = field(default=6, init=False)
    local_dim: int = field(default=6, init=False)
    global_dim: int = field(default=5, init=False)
    per_object_cap: None = field(default=None, init=False)
    peak_total: None = field(default=None, init=False)
    action_cost_scale: float = field(default=1., init=False)
    dictionary_kind: str = field(default="raw", init=False)
    dictionary_seed: int = field(default=71331, init=False)

    def __post_init__(self):
        for name in ("groups", "horizon"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Integral) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        scale=_real(self.spatial_sigma_scale, 'spatial_sigma_scale')
        if scale not in (1.,.5,.25): raise ValueError('Only declared spatial noise scales1/.5/.25 are allowed')
        object.__setattr__(self,'spatial_sigma_scale',scale)
        if self.adapter not in ('unit','depth_area','depth_area_physical'):
            raise ValueError('Unknown declared irrigation adapter')
        object.__setattr__(self, 'local_dim', 8 if self.adapter == 'depth_area_physical' else 6)
        areas = tuple(float(v) for v in self.area_ha)
        if len(areas) != 89 or not all(math.isfinite(v) and v > 0 for v in areas):
            raise ValueError('Complete positive fixed public area geometry required')
        object.__setattr__(self, 'area_ha', areas)
        object.__setattr__(self, "initial_budget", _real(self.initial_budget, "initial_budget"))
        if self.groups != 89 or self.horizon != 81:
            raise ValueError('This specification requires exactly 89 objects and 81 days')
        ids = tuple(self.region_ids)
        if (len(ids) != self.groups or any(not isinstance(x, str) or not x for x in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("region_ids must be unique nonempty strings in public object order")
        from .constants import EXCLUDED_IDS, RETAINED_IDS
        expected = RETAINED_IDS
        if ids != expected or set(ids) & set(EXCLUDED_IDS):
            raise ValueError('Region identity/order differs from the registered reduced case')
        object.__setattr__(self, "region_ids", ids)

    @classmethod
    def from_env(cls, env):
        """Read public fixed attributes only; do not reset or inspect the core."""
        if getattr(env, 'application_domain', None) != 'hongmiao89_reasonable_depth_irrigation':
            raise ValueError('An actual reduced89 environment is required')
        return cls(groups=env.N, horizon=env.T,
                   initial_budget=env.initial_budget_m3,
                   region_ids=tuple(env.region_ids), adapter=env.adapter, spatial_sigma_scale=env.spatial_sigma_scale,
                   area_ha=tuple(float(v) for v in env.scenario.areas_ha))

    @property
    def nominal_total(self):
        return self.initial_budget / self.horizon

    @property
    def initial_budget_m3(self):
        return self.initial_budget

    @property
    def action_tolerance(self):
        return _tolerance(self.initial_budget)

    @property
    def budget_mode(self):
        return "adaptive"

    @property
    def resource_unit(self):
        return "m3"

    @property
    def action_unit(self):
        return "m3"

    @property
    def observation_version(self):
        return OBSERVATION_SCHEMA

    @property
    def feature_version(self):
        return f'reasonable89-{self.adapter}-{self.local_dim}-local-five-global-v1'

    @property
    def reward_range(self):
        return (0., 1.)

    def to_dict(self):
        """Detached metadata; no objective standard is inferred by this layer."""
        return {
            "adapter": self.adapter, "area_ha": list(self.area_ha), "spatial_sigma_scale": self.spatial_sigma_scale,
            "domain": self.domain, "application_chapter": self.application_chapter,
            "groups": self.groups, "horizon": self.horizon,
            "local_dim": self.local_dim, "global_dim": self.global_dim,
            "initial_budget": self.initial_budget, "nominal_total": self.nominal_total,
            "per_object_cap": None, "peak_total": None,
            "action_cost_scale": self.action_cost_scale,
            "action_tolerance": self.action_tolerance,
            "resource_unit": self.resource_unit, "action_unit": self.action_unit,
            "dictionary_kind": self.dictionary_kind, "dictionary_seed": self.dictionary_seed,
            "region_ids": list(self.region_ids), "budget_mode": self.budget_mode,
            "observation_version": self.observation_version,
            "feature_version": self.feature_version,
            "state_keys": list(STATE_KEYS), "local_features": list(LOCAL_FEATURES) + (['total_store_per_soil_depth','no_stress_storage_deficit_mm'] if self.local_dim==8 else []),
            "global_features": list(GLOBAL_FEATURES),
            "local_scale_divisors": ["public mean rice_area_ha at reset", 2., 48., 48., 1., 100.] + (["public TAW/(.39-.23)",48.] if self.local_dim==8 else []),
            "global_scale_divisors": ["horizon", "initial_budget (zero budget maps to zero)", 100., 10., 5.],
            "feature_transform": "original six linear inputs retained; physical adapter appends B/Z and max(.40Z-B,0)/48 only",
            "neural_balance": True, "memory_fields": ["executed_action_budget_ledger_only"],
            "preference_is_environment_observation": False,
            "method_task_factory_integrated": False,
            "terminal_geometry": "budget and caps zero; remaining retains actual unspent water",
            "nominal_total_identity": "initial_budget/horizon pacing reference, not a daily limit",
        }


class IrrigationEncoder:
    """Validate compact observations and encode only public values.

    The scalar spend ledger uses exactly the engine's sequential day sums:
    ``spent += math.fsum(action.tolist())``.  At the terminal boundary the raw
    available amount is zero because no more actions may be taken, while the
    ledger still retains actual unspent supply.  Failed reset/update calls are
    transactional: neither the current report nor the ledger changes.
    """

    def __init__(self, spec):
        if not isinstance(spec, IrrigationSpec):
            raise TypeError("IrrigationEncoder requires an IrrigationSpec")
        self.spec = spec
        self._current = None
        self._spent = 0.
        self._area_scale = None

    def _copy(self, observation):
        if not isinstance(observation, dict) or set(observation) != OBSERVATION_KEYS:
            raise ValueError("Expected exactly the ten compact public observation fields")
        raw_day = np.asarray(observation["day_index"])
        if (raw_day.shape != () or raw_day.dtype.kind not in "iu"
                or not 0 <= int(raw_day) <= self.spec.horizon):
            raise ValueError("day_index must be an integer within the decision horizon")
        current = {"day_index": np.array(int(raw_day), dtype=np.int64)}
        for name in LOCAL_FEATURES:
            current[name] = _numeric(observation[name], (self.spec.groups,), name, nonnegative=True)
        for name in ("available_water_m3", "reported_current_stage_reference_et_mm"):
            current[name] = _numeric(observation[name], (), name, nonnegative=True)
        current["previous_weather"] = _numeric(
            observation["previous_weather"], (2,), "previous_weather", nonnegative=True)
        if (np.any(current["rice_area_ha"] <= 0)
                or np.any(current["root_available_water_mm"] <= 0)):
            raise ValueError("Published area and root available-water capacity must be positive")
        if np.any(current["reported_completed_yield_factor"] > 1.):
            raise ValueError("Completed Jensen factors must lie in [0,1]")
        if np.any(current["reported_current_stage_et_mm"] > current["reported_current_stage_reference_et_mm"]):
            raise ValueError("Current actual ET cannot exceed its common fully supplied reference")
        return current

    def _check_available(self, current, spent):
        remaining = max(0., self.spec.initial_budget - spent)
        expected = remaining if int(current["day_index"]) < self.spec.horizon else 0.
        actual = float(current["available_water_m3"])
        if abs(actual - expected) > _tolerance(expected):
            raise ValueError("Public available water disagrees with the executed-action ledger")

    def reset(self, observation):
        current = self._copy(observation)
        if int(current["day_index"]) != 0:
            raise ValueError("Reset must start at day zero")
        self._check_available(current, 0.)
        area_scale = float(current["rice_area_ha"].mean())
        if not math.isfinite(area_scale) or area_scale <= 0:
            raise ValueError("Mean public rice area must be finite and positive")
        self._current, self._spent, self._area_scale = current, 0., area_scale
        return self.state()

    def update(self, action, observation):
        if self._current is None:
            raise RuntimeError("Reset before updating the interface")
        if int(self._current["day_index"]) >= self.spec.horizon:
            raise RuntimeError("Episode terminated; reset before updating")
        current = self._copy(observation)
        if int(current["day_index"]) != int(self._current["day_index"]) + 1:
            raise ValueError("Exactly one daily decision is required per update")
        for name in LOCAL_FEATURES[:3]:
            if not np.array_equal(current[name], self._current[name]):
                raise ValueError(f"Fixed public attribute changed within the episode: {name}")
        a = _numeric(action, (self.spec.groups,), "executed action", nonnegative=True)
        remaining = max(0., self.spec.initial_budget - self._spent)
        allocated = math.fsum(a.tolist())
        if (not math.isfinite(allocated) or allocated > remaining + _tolerance(remaining)
                or (remaining == 0. and allocated != 0.)):
            raise ValueError("Executed action exceeds the remaining seasonal supply")
        spent = self._spent + allocated
        self._check_available(current, spent)
        self._current, self._spent = current, spent
        return self.state()

    def observation(self):
        if self._current is None:
            raise RuntimeError("Reset before reading the interface")
        return {key: value.copy() for key, value in self._current.items()}

    def state(self):
        if self._current is None:
            raise RuntimeError("Reset before reading the interface")
        c, s = self._current, self.spec
        t = int(c["day_index"])
        remaining = max(0., s.initial_budget - self._spent)
        upper = remaining if t < s.horizon else 0.
        scales = np.array((self._area_scale, 2., 48., 48., 1., 100.), dtype=np.float64)
        local = np.column_stack([c[key] for key in LOCAL_FEATURES]) / scales
        if s.adapter == 'depth_area_physical':
            # Fixed public theta_fc=.39 and theta_wp=.23; no private live state.
            depth = c['root_available_water_mm'] / (.39-.23)
            physical = np.column_stack((c['reported_field_storage_mm']/depth,
                np.maximum(.40*depth-c['reported_field_storage_mm'],0.)/48.))
            local = np.column_stack((local, physical))
        global_ = np.array((
            t / s.horizon,
            remaining / s.initial_budget if s.initial_budget else 0.,
            float(c["reported_current_stage_reference_et_mm"]) / 100.,
            c["previous_weather"][0] / 10., c["previous_weather"][1] / 5.,
        ), dtype=np.float64)
        return {
            "local": local, "global": global_,
            "caps": np.full(s.groups, upper, dtype=np.float64),
            "budget": np.array(upper, dtype=np.float64),
            "remaining": np.array(remaining, dtype=np.float64),
        }


def append_preference(state, weight):
    """Append a fixed algorithm preference to a detached base state once.

    Both one state and a stacked batch are accepted.  This does not change raw
    observations or compute a reward.  A six-coordinate global input already
    has a preference; rejecting it prevents accidental double appending.
    """
    preference = _real(weight, "coverage weight")
    if preference > 1.:
        raise ValueError("Coverage weight must lie in [0,1]")
    if not isinstance(state, dict) or set(state) != set(STATE_KEYS):
        raise ValueError("Expected the five public state keys")
    raw_local = np.asarray(state["local"])
    if raw_local.ndim not in (2, 3) or raw_local.shape[-1] not in (6,8) or raw_local.shape[-2] < 1:
        raise ValueError("Expected six local coordinates in one state or a batch")
    prefix, n = raw_local.shape[:-2], raw_local.shape[-2]
    shapes = {"local": raw_local.shape, "global": prefix + (5,),
              "caps": prefix + (n,), "budget": prefix, "remaining": prefix}
    output = {key: _numeric(state[key], shapes[key], key,
                           nonnegative=key in ("caps", "budget", "remaining"))
              for key in STATE_KEYS}
    if (np.any(output["global"][..., 0] < 0) or np.any(output["global"][..., 0] > 1)
            or np.any(output["budget"] > output["remaining"] + 1e-10)
            or np.any(output["budget"] > output["caps"].sum(-1) + 1e-10)):
        raise ValueError("Invalid public time or resource geometry")
    tail = np.full(prefix + (1,), preference, dtype=np.float64)
    output["global"] = np.concatenate((output["global"], tail), axis=-1)
    return output


__all__ = ["IrrigationSpec", "IrrigationEncoder", "append_preference", "STATE_KEYS"]
