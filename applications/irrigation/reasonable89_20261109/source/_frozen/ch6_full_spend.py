"""Explicit irrigation full-spend task contract around frozen field physics.

No radio/v5 task identity is fabricated. The latest quota mathematics is used
with the irrigation interval [0,R], or [R,R] on the final decision. The physical
core, compact observations, terminal labels and learning updates are unchanged.
An optional evaluation-only forcing trace is private and never a policy input.
"""
from __future__ import annotations

import copy
from datetime import timedelta
import hashlib
import json
import math
import numpy as np

from irrigation_ch6 import IrrigationSpec
from irrigation_ch6.env import IrrigationEnv
from irrigation_ch6._core.weather_pairs import PairWeatherDay

VERSION = 'hongmiao93-explicit-full-spend-task-v1'


def _digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode()).hexdigest()


def terminal_budget_tolerance(spec):
    return min(float(spec.action_tolerance),
               512 * np.finfo(np.float64).eps * max(abs(float(spec.initial_budget)), np.finfo(float).tiny))


class FullSpendSpec:
    """Public neural execution view; no historical ``chapter`` impersonation."""
    full_spend_required = True
    application_domain = 'hongmiao93_irrigation'

    def __init__(self, physical_spec):
        if getattr(physical_spec, 'domain', None) != 'hongmiao93_irrigation':
            raise ValueError('An actual irrigation specification is required')
        self.physical_spec = physical_spec

    def __getattr__(self, name):
        base = self.__dict__.get('physical_spec')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    @classmethod
    def from_env(cls, env):
        return cls(IrrigationSpec.from_env(env))

    def to_dict(self):
        result = self.physical_spec.to_dict()
        result.update(full_spend_required=True, application_domain=self.application_domain,
                      task_contract=VERSION, quota_interval='[0,R] before final day; [R,R] on final day',
                      terminal_remaining='zero within fixed floating-point roundoff',
                      policy_latent_event='complete Gaussian retained including quota coordinate at final day')
        return result


def irrigation_full_spend_bounds(state, spec):
    """Pure public-state bounds used by the unchanged latest quota equation."""
    if getattr(spec, 'application_domain', None) != 'hongmiao93_irrigation':
        raise ValueError('Explicit irrigation full-spend spec required')
    upper = float(np.asarray(state['budget']))
    remaining = float(np.asarray(state['remaining']))
    phase = float(np.asarray(state['global'])[0])
    period = int(round(phase * spec.horizon))
    if not all(math.isfinite(x) for x in (upper, remaining, phase)) or not 0 <= period <= spec.horizon:
        raise ValueError('Invalid public budget/phase')
    tolerance = terminal_budget_tolerance(spec)
    if upper < 0 or remaining < 0 or upper > remaining + tolerance:
        raise ValueError('Invalid public budget interval')
    if period == spec.horizon:
        if remaining > tolerance:
            raise ArithmeticError('Full-spend irrigation ended with remaining resource')
        return 0., 0.
    if abs(upper-remaining) > tolerance:
        raise ValueError('Irrigation has no additional total delivery cap')
    return (remaining if period == spec.horizon-1 else 0.), upper


class _TraceWeather:
    """Evaluation driver preserving native draw counts and transaction copies.

    The base sample is still drawn AFTER validation, even when its realised pair
    is replaced. Snapshots retain native RNG state; the owning task fingerprint
    separately binds the full forcing trace and prevents cross-trace restore.
    """
    def __init__(self, base, trace):
        self.base, self.trace = base, trace

    def __getattr__(self, name):
        base = self.__dict__.get('base')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __copy__(self):
        # Native step restores the copy before sampling. Keeping a separate
        # base object ensures failed physical transitions do not consume RNG.
        return type(self)(copy.copy(self.base), self.trace)

    def reset(self, seed=None):
        return self.base.reset(seed)

    def snapshot(self):
        return self.base.snapshot()

    def restore(self, state):
        return self.base.restore(state)

    def sample(self, t, previous_class):
        native = self.base.sample(t, previous_class)
        return PairWeatherDay(native.date, self.trace[int(t)], self.base.rain_multiplier)


class FullSpendIrrigationEnv(IrrigationEnv):
    """Validate the new decision contract before invoking unchanged physics."""
    full_spend_required = True
    application_domain = 'hongmiao93_irrigation'

    def __init__(self, *, forcing_trace=None, forcing_label=None, **kwargs):
        super().__init__(**kwargs)
        self._forcing_trace = None
        self._forcing_label = forcing_label
        if forcing_trace is not None:
            a = np.asarray(forcing_trace)
            if a.shape != (self.T, 2) or a.dtype.kind not in 'iuf' or not np.isfinite(a).all() or np.any(a < 0):
                raise ValueError('Forcing must be81 finite nonnegative rain/ET0 pairs')
            a = np.asarray(a, dtype=np.float64).copy()
            driver = self._engine._weather
            if driver.mode != 'persistent_drought':
                raise ValueError('Registered forcing interventions are limited to the dry task')
            for t, pair in enumerate(a):
                pool = driver._donor_pools[t, 0]
                legal = np.asarray([driver._scaled_vector(driver._records[k].vector) for k in pool])
                if not np.any(np.all(legal == pair, axis=1)):
                    raise ValueError(('Forcing pair outside original seasonal drought support', t, pair.tolist()))
            a.setflags(write=False)
            self._forcing_trace = a
            self._engine._weather = _TraceWeather(driver, a)
        self.base_task_fingerprint = self.task_fingerprint
        forcing_hash = (hashlib.sha256(self._forcing_trace.tobytes()).hexdigest()
                        if self._forcing_trace is not None else None)
        self.task_fingerprint = _digest(dict(base_task=self.base_task_fingerprint,
            contract=VERSION, forcing_trace_sha256=forcing_hash))

    def model_config(self):
        result = super().model_config()
        result.update(full_spend_required=True, decision_contract=VERSION,
            daily_total_interval='[0,R]; final day [R,R]', physical_core_unchanged=True,
            exogenous_forcing_override=self._forcing_trace is not None,
            forcing_trace_sha256=(hashlib.sha256(self._forcing_trace.tobytes()).hexdigest()
                                  if self._forcing_trace is not None else None),
            forcing_label=self._forcing_label,
            forcing_visibility='private evaluation input; only previous realised pair enters observation')
        return result

    def step(self, action):
        # No conversion or repair is passed to the field engine.
        raw = np.asarray(action)
        if raw.shape != (self.N,) or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
            raise ValueError('A finite93-dimensional physical action is required')
        q = math.fsum(raw.tolist())
        remaining = float(self._engine._remaining())
        spec = FullSpendSpec.from_env(self)
        tolerance = terminal_budget_tolerance(spec)
        if self._engine._day == self.T-1 and abs(q-remaining) > tolerance:
            raise ValueError('Final irrigation action must allocate the entire remaining budget')
        result = super().step(action)
        if result[2]:
            leftover = float(self._engine._remaining())
            if abs(leftover) > tolerance:
                raise ArithmeticError('Terminal full-spend invariant failed')
        return result

    def set_state(self, state):
        # Native restore checks our trace-specific task fingerprint first.
        engine = state.get('engine_state', {}) if isinstance(state, dict) else {}
        if engine.get('day') == self.T:
            remaining = self.initial_budget_m3-float(engine.get('spent_m3', float('nan')))
            if not math.isfinite(remaining) or abs(remaining) > terminal_budget_tolerance(FullSpendSpec.from_env(self)):
                raise ValueError('Terminal checkpoint violates the full-spend task')
        result = super().set_state(state)
        if self._forcing_trace is not None and not isinstance(self._engine._weather, _TraceWeather):
            self._engine._weather = _TraceWeather(self._engine._weather, self._forcing_trace)
        return result


def make_env(**kwargs):
    return FullSpendIrrigationEnv(**kwargs)


def make_pool(env_config, count):
    allowed = {k: v for k, v in env_config.items() if k in (
        'coverage_threshold','coverage_weight','total_budget_m3','weather_mode',
        'historical_year','rain_multiplier','profile','storage_variation',
        'forcing_trace','forcing_label')}
    allowed['reward_mode'] = 'mean_performance'
    return [make_env(**allowed) for _ in range(int(count))]


def make_models_for_arm(arm, physical_spec, seed):
    if arm != 'full':
        raise ValueError('This registered task trains only the integrated full method')
    from allocation_morl.models import make_models
    spec = physical_spec if isinstance(physical_spec, FullSpendSpec) else FullSpendSpec(physical_spec)
    return make_models(spec, int(seed), 'fixed')
