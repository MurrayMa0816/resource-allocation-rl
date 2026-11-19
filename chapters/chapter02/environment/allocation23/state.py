"""Simplified public observations, neural inputs and exact action bookkeeping.

No environment, simulator parameters, reward model or image data are imported.
The agent retains only a scalar resource ledger; no action-history feature is
supplied to either method. Physical metadata remains available to the decoder.
"""
from __future__ import annotations

from math import isfinite
import numpy as np
from .spec import Spec

OBSERVATION_FIELDS = frozenset((
    'channel_real', 'channel_imag', 'interference_multiplier',
    'confirmed_packets', 'remaining_energy_J', 'round_index',
))


def _vector(value, name):
    a = np.asarray(value)
    if a.ndim != 1 or a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
        raise ValueError(f'Finite real vector required: {name}')
    return a.astype(np.float64, copy=True)


def _scalar(value, name):
    a = np.asarray(value)
    if a.ndim != 0 or a.dtype.kind not in 'iuf' or not np.isfinite(a):
        raise ValueError(f'Finite real scalar required: {name}')
    return float(a)


def _roundoff(total):
    return 256 * np.finfo(np.float64).eps * max(float(total), np.finfo(float).tiny)


def _validate_v5(action, state, spec):
    """Return an execution-safe action; repair only a few summation ULPs.

    There is no fixed-total constraint here: F0 chooses its own fixed total,
    whereas dynamic policies may choose zero or retain central allowance.
    Emergency native execution rejects any positive budget excess, so a
    roundoff excess is corrected before ``step`` and never after execution.
    """
    a, caps = _vector(action, 'action'), _vector(state['caps'], 'caps')
    upper, remaining = _scalar(state['budget'], 'budget'), _scalar(state['remaining'], 'remaining')
    if (a.shape != (spec.groups,) or caps.shape != a.shape or np.any(caps < 0)
            or upper < 0 or remaining < 0 or upper > remaining + _roundoff(remaining)):
        raise ValueError('Action or public feasible geometry differs from task')
    if np.any(a < 0):
        raise ArithmeticError('Negative action is not a rounding correction')
    before = a.copy()
    total = float(a.sum())
    tolerance = min(spec.action_tolerance, _roundoff(upper))
    cap_tolerance = np.minimum(spec.action_tolerance,
                               256 * np.finfo(np.float64).eps * np.maximum(caps, np.finfo(float).tiny))
    if (not isfinite(total) or total > upper + tolerance
            or np.any(a > caps + cap_tolerance) or np.any(a[caps == 0] != 0)):
        raise ArithmeticError('Non-roundoff action constraint violation')
    np.minimum(a, caps, out=a)
    total = float(a.sum())
    if total > upper:
        j = int(np.argmax(a))
        a[j] = max(0., np.nextafter(a[j] - (total - upper), 0.))
        # An array-layout change can move a sum by a final ULP. A second tiny
        # subtraction is bounded, unlike a loop that may stall at small scales.
        if float(a.sum()) > upper:
            a[j] = max(0., a[j] - _roundoff(upper))
    if (np.any(a < 0) or np.any(a > caps) or float(a.sum()) > upper
            or np.max(np.abs(a-before), initial=0.) > tolerance):
        raise ArithmeticError('Could not correct action summation within roundoff')
    return a, float(np.max(np.abs(a-before), initial=0.))


def _validate_radio(action, state, spec):
    """Check actual power actions, correcting summation roundoff only."""
    a, caps = _vector(action, 'action'), _vector(state['caps'], 'caps')
    upper, remaining = _scalar(state['budget'], 'budget'), _scalar(state['remaining'], 'remaining')
    if (a.shape != (spec.groups,) or caps.shape != a.shape or np.any(caps < 0)
            or upper < 0 or remaining < 0 or upper > remaining + _roundoff(remaining)):
        raise ValueError('Action geometry differs from task')
    before, total = a.copy(), float(a.sum())
    tolerance = min(spec.action_tolerance, _roundoff(upper))
    cap_tol = np.minimum(spec.action_tolerance,
                         256 * np.finfo(float).eps * np.maximum(caps, np.finfo(float).tiny))
    if (np.any(a < 0) or not isfinite(total) or total > upper + tolerance
            or np.any(a > caps + cap_tol) or np.any(a[caps == 0] != 0)):
        raise ArithmeticError('Non-roundoff action constraint violation')
    np.minimum(a, caps, out=a)
    if float(a.sum()) > upper:
        j = int(np.argmax(a))
        a[j] = max(0., np.nextafter(a[j] - (float(a.sum()) - upper), 0.))
        if float(a.sum()) > upper:
            a[j] = max(0., a[j] - _roundoff(upper))
    if (np.any(a < 0) or np.any(a > caps) or float(a.sum()) > upper
            or np.max(np.abs(a - before), initial=0.) > tolerance):
        raise ArithmeticError('Could not correct summation within roundoff')
    return a, float(np.max(np.abs(a - before), initial=0.))


def validate_action(action, state, spec):
    """Validate executed native actions, correcting numerical roundoff only."""
    if not isinstance(spec, Spec):
        raise TypeError('Expected the public allocation23 Spec')
    return (_validate_v5(action, state, spec) if spec.study_domain == 'v5'
            else _validate_radio(action, state, spec))


class _NativeRadioEncoder:
    def __init__(self, spec):
        self.spec, self._current, self._remaining_cost = spec, None, None

    def _copy(self, obs):
        if not isinstance(obs, dict) or set(obs) != OBSERVATION_FIELDS:
            raise ValueError('Expected exactly the six current raw observation fields')
        time = np.asarray(obs['round_index'])
        if time.ndim or time.dtype.kind not in 'iu' or not 0 <= int(time) <= self.spec.horizon:
            raise ValueError('Invalid current round')
        re, im = _vector(obs['channel_real'], 'channel_real'), _vector(obs['channel_imag'], 'channel_imag')
        counts = np.asarray(obs['confirmed_packets'])
        if (re.shape != (self.spec.groups,) or im.shape != re.shape or counts.shape != re.shape
                or counts.dtype.kind not in 'iu' or np.any(counts < 0) or np.any(counts > self.spec.source_packets)):
            raise ValueError('Invalid current channel or reception count')
        noise = _scalar(obs['interference_multiplier'], 'interference_multiplier')
        energy = _scalar(obs['remaining_energy_J'], 'remaining_energy_J')
        if noise <= 0 or energy < 0 or energy > self.spec.initial_budget * self.spec.action_cost_scale + 1e-15:
            raise ValueError('Invalid current interference or energy balance')
        return dict(channel_real=re, channel_imag=im, confirmed_packets=counts.astype(np.int32, copy=True),
                    interference_multiplier=noise, remaining_energy_J=energy, round_index=int(time))

    def _check_balance(self, actual, expected):
        tol = max(1e-18, _roundoff(self.spec.initial_budget * self.spec.action_cost_scale))
        if abs(actual - expected) > tol:
            raise ValueError('Public resource balance disagrees with executed actions')

    def reset(self, obs):
        current = self._copy(obs)
        if current['round_index'] != 0 or np.any(current['confirmed_packets']):
            raise ValueError('Reset must start at round zero with empty receivers')
        expected = self.spec.initial_budget * self.spec.action_cost_scale
        self._check_balance(current['remaining_energy_J'], expected)
        self._remaining_cost = expected
        self._current = current
        return self.state()

    def update(self, action, obs):
        if self._current is None:
            raise RuntimeError('Reset before update')
        current = self._copy(obs)
        if current['round_index'] != self._current['round_index'] + 1:
            raise ValueError('Exactly one physical decision per update')
        checked, delta = validate_action(action, self.state(), self.spec)
        if delta != 0.:
            raise ValueError('Validate action before execution, never repair past actions')
        increments = current['confirmed_packets'] - self._current['confirmed_packets']
        if np.any((increments < 0) | (increments > 1)) or np.any((checked == 0) & (increments != 0)):
            raise ValueError('Confirmation events violate one-packet transmission semantics')
        expected = max(0., self._remaining_cost - float(checked.sum()) * self.spec.action_cost_scale)
        self._check_balance(current['remaining_energy_J'], expected)
        self._remaining_cost, self._current = expected, current
        return self.state()

    def state(self):
        if self._current is None:
            raise RuntimeError('Reset before state')
        c, s = self._current, self.spec
        t = c['round_index']
        remaining = c['remaining_energy_J'] / s.action_cost_scale
        # asinh/log/scaling are invertible; no current state information is discarded.
        local = np.column_stack((np.arcsinh(c['channel_real']), np.arcsinh(c['channel_imag']),
                                 c['confirmed_packets'] / s.source_packets))
        global_ = np.asarray((t / s.horizon, np.log(c['interference_multiplier']),
                              remaining / s.initial_budget), dtype=np.float64)
        caps = np.where((c['confirmed_packets'] < s.source_packets) & (t < s.horizon),
                        s.per_object_cap, 0.).astype(np.float64)
        upper = min(remaining, float(caps.sum())) if t < s.horizon else 0.
        return dict(local=local, **{'global': global_}, caps=caps,
                    budget=np.asarray(upper), remaining=np.asarray(remaining))

    def observation(self):
        if self._current is None:
            raise RuntimeError('Reset before observation')
        return self._copy(self._current)


class EmergencyEncoder:
    def __init__(self, spec):
        self.spec = spec
        self._current = None
        self._remaining_cost = float(spec.initial_budget * spec.action_cost_scale)

    def _copy(self, observation):
        expected = {'remaining_demand', 'total_demand_so_far', 'time_index'}
        if self.spec.chapter == 3:
            expected.add('remaining_budget')
        if not isinstance(observation, dict) or set(observation) != expected:
            raise ValueError('Expected exactly the simplified public V5 observation fields')
        t = np.asarray(observation['time_index'])
        if t.ndim or t.dtype.kind not in 'iu' or not 0 <= int(t) <= self.spec.horizon:
            raise ValueError('Valid public time index required')
        result = {'time_index': int(t)}
        for key in ('remaining_demand', 'total_demand_so_far'):
            value = np.asarray(observation[key])
            if (value.shape != (self.spec.groups,) or value.dtype.kind not in 'iuf'
                    or not np.isfinite(value).all() or np.any(value < 0)):
                raise ValueError('Finite nonnegative regional public workload required')
            result[key] = value.astype(np.float64, copy=True)
        if np.any(result['remaining_demand'] > result['total_demand_so_far'] + 1e-7):
            raise ValueError('Remaining demand cannot exceed total demand seen so far')
        if self.spec.chapter == 3:
            value = np.asarray(observation['remaining_budget'])
            if value.ndim or value.dtype.kind not in 'iuf' or not np.isfinite(value) or value < 0:
                raise ValueError('Finite nonnegative public resource balance required')
            result['remaining_budget'] = float(value)
        return result

    def _check_balance(self, current):
        if self.spec.chapter == 3:
            remaining = self._remaining_cost / self.spec.action_cost_scale
            if not np.isclose(current['remaining_budget'], remaining, rtol=1e-12, atol=1e-8):
                raise ValueError('Public resource balance differs from executed-action ledger')

    def reset(self, observation):
        current = self._copy(observation)
        if current['time_index'] != 0 or not np.array_equal(current['remaining_demand'], current['total_demand_so_far']):
            raise ValueError('Reset starts with the initial unfinished demand as total demand seen so far')
        self._remaining_cost = float(self.spec.initial_budget * self.spec.action_cost_scale)
        self._check_balance(current)
        self._current = current
        return self.state()

    def update(self, action, observation):
        if self._current is None:
            raise RuntimeError('Reset before updating state')
        current = self._copy(observation)
        if current['time_index'] != self._current['time_index'] + 1:
            raise ValueError('Exactly one physical round per state update')
        if np.any(current['total_demand_so_far'] < self._current['total_demand_so_far']):
            raise ValueError('Total demand seen so far cannot decrease')
        checked, delta = _validate_v5(action, self.state(), self.spec)
        if delta != 0.:
            raise ValueError('Validate actions before executing them')
        self._remaining_cost = max(0., self._remaining_cost - float(checked.sum()) * self.spec.action_cost_scale)
        self._check_balance(current)
        self._current = current
        return self.state()

    def state(self):
        if self._current is None:
            raise RuntimeError('Reset before state')
        current, spec = self._current, self.spec
        t = current['time_index']
        remaining = self._remaining_cost / spec.action_cost_scale
        local = np.column_stack([np.log1p(current[key] / 100.)
                                 for key in ('remaining_demand', 'total_demand_so_far')])
        global_ = [t / spec.horizon]
        if spec.chapter == 3:
            global_.append(remaining / spec.initial_budget)
        caps = np.full(spec.groups, remaining if t < spec.horizon else 0., dtype=np.float64)
        upper = min(remaining, float(caps.sum())) if t < spec.horizon else 0.
        if spec.chapter == 2:
            upper = min(upper, spec.nominal_total)
        return dict(local=local, **{'global': np.asarray(global_, dtype=np.float64)},
                    caps=caps, budget=np.asarray(upper), remaining=np.asarray(remaining))

    def observation(self):
        if self._current is None:
            raise RuntimeError('Reset before observation')
        return {key: np.asarray(value).copy() for key, value in self._current.items()}


class CommunicationEncoder:
    def __init__(self, spec):
        self.spec = spec
        # The native three-field encoder has no action-history features. Its
        # ledger checks use the public meter, never a private environment field.
        self.raw = _NativeRadioEncoder(spec)

    def reset(self, observation):
        self.raw.reset(observation)
        return self.state()

    def update(self, action, observation):
        self.raw.update(action, observation)
        return self.state()

    def state(self):
        result = self.raw.state()
        if self.spec.chapter == 2:
            result['global'] = result['global'][:2].copy()
        return result

    def observation(self):
        return self.raw.observation()


def make_encoder(spec):
    """Build the shared simplified observation encoder for method A or B."""
    if not isinstance(spec, Spec):
        raise TypeError('Expected the public allocation23 Spec')
    return EmergencyEncoder(spec) if spec.study_domain == 'v5' else CommunicationEncoder(spec)


__all__ = ['make_encoder', 'validate_action', 'EmergencyEncoder', 'CommunicationEncoder']
