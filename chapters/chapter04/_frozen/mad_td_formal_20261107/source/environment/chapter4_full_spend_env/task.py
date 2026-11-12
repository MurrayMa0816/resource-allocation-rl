"""Transparent physical execution guard shared by native and Gym collectors.

Only existing public observations and submitted physical actions enter the
ledger. Rewards, observations, native terminal diagnostics and RNG calls are
unchanged. Ledger rows are audit output, never returned as learner information.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from . import native

_AUDIT_DIR = None
_CONTEXT = {}


def configure_audit(audit_dir=None, context=None):
    global _AUDIT_DIR, _CONTEXT
    _AUDIT_DIR = None if audit_dir is None else Path(audit_dir).resolve()
    _CONTEXT = dict(context or {})
    if _AUDIT_DIR is not None:
        _AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def _write_ledger(record):
    if _AUDIT_DIR is not None:
        path = _AUDIT_DIR / ('resource_episodes_' + str(os.getpid()) + '.jsonl')
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


class FullSpendPhysicalGuard:
    """Delegate native physics, checking its full-spend public contract."""
    def __init__(self, env, spec, *, domain, seed):
        self.native_env, self.spec = env, spec
        self.domain, self.seed = domain, int(seed)
        self.phase, self.episode_index = None, None
        self.encoder = None
        self.rounds = []
        self.complete = False
        self.last_full_spend_audit = None
        self.completed_full_spend_episodes = 0

    def __getattr__(self, name):
        return getattr(self.native_env, name)

    def set_context(self, phase, index):
        self.phase, self.episode_index = str(phase), int(index)

    def reset(self, *args, **kwargs):
        observation, info = self.native_env.reset(*args, **kwargs)
        self.encoder = native.make_encoder(self.spec)
        self.encoder.reset(observation)
        self.rounds, self.complete, self.last_full_spend_audit = [], False, None
        return observation, info

    def step(self, action):
        if self.encoder is None or self.complete:
            raise RuntimeError('Reset is required before physical full-spend execution')
        state = self.encoder.state()
        checked, correction = native.validate_action(action, state, self.spec)
        # The shared decoder/Task.validate must do its ordinary ULP correction
        # BEFORE this guard; execution never introduces a second action repair.
        if correction != 0. or not np.array_equal(np.asarray(action), checked):
            raise ArithmeticError('Validate physical action before submitting it to execution')
        lower, upper = native.full_spend.full_spend_bounds(state, self.spec)
        total = float(np.asarray(action).sum())
        remaining_before = float(state['remaining'])
        observation, reward, terminated, truncated, info = self.native_env.step(action)
        expected_terminal = len(self.rounds) + 1 == self.spec.horizon
        if bool(truncated) or bool(terminated) != expected_terminal:
            raise RuntimeError('Native finite-horizon termination changed')
        if not terminated and (float(reward) != 0. or info != {}):
            raise RuntimeError('Nonterminal reward/information permission changed')
        if not math.isfinite(float(reward)):
            raise FloatingPointError('Nonfinite native reward')
        self.encoder.update(action, observation)
        remaining_after = float(self.encoder.state()['remaining'])
        array = np.asarray(action)
        caps = np.asarray(state['caps'])
        self.rounds.append(dict(round=len(self.rounds), remaining_before=remaining_before,
            lower=lower, upper=upper, quota=total, remaining_after=remaining_after,
            nonzero_objects=int(np.count_nonzero(action)),
            active_objects=int(np.count_nonzero(caps > 0)),
            negative_allocation_violation=max(0.,-float(array.min(initial=0.))),
            per_object_cap_violation=max(0.,float((array-caps).max(initial=0.))),
            completed_object_allocation=float(np.abs(array[caps==0]).max(initial=0.)),
            balance_absolute_error=abs(max(0.,remaining_before-total)-remaining_after)))
        if terminated:
            native.full_spend.assert_terminal_full_spend(remaining_after, self.spec)
            total_used = math.fsum(row['quota'] for row in self.rounds)
            relative_error = abs(total_used / self.spec.initial_budget - 1.)
            if relative_error > 1e-10:
                raise ArithmeticError('Complete episode did not allocate its entire initial budget')
            self.complete = True
            self.completed_full_spend_episodes += 1
            self.last_full_spend_audit = dict(schema='chapter4-full-spend-physical-ledger-v1',
                context=_CONTEXT, domain=self.domain, seed=self.seed, phase=self.phase,
                episode_index=self.episode_index, steps=len(self.rounds),
                initial_budget=self.spec.initial_budget, total_used=total_used,
                remaining=remaining_after, relative_allocation_error=relative_error,
                total_allocation_absolute_error=abs(total_used-self.spec.initial_budget),
                maximum_step_balance_absolute_error=max(r['balance_absolute_error'] for r in self.rounds),
                maximum_negative_allocation_violation=max(r['negative_allocation_violation'] for r in self.rounds),
                maximum_per_object_cap_violation=max(r['per_object_cap_violation'] for r in self.rounds),
                maximum_completed_object_allocation=max(r['completed_object_allocation'] for r in self.rounds),
                physical_resource_unit=self.spec.resource_unit,
                action_cost_scale=self.spec.action_cost_scale,
                initial_physical_resource=self.spec.initial_budget*self.spec.action_cost_scale,
                total_physical_resource=total_used*self.spec.action_cost_scale,
                remaining_physical_resource=remaining_after*self.spec.action_cost_scale,
                terminal_reward=float(reward), nonterminal_rewards_zero=True,
                terminal_full_spend_assertion=True, rounds=self.rounds)
            _write_ledger(self.last_full_spend_audit)
        return observation, reward, terminated, truncated, info


class Task(native.BaseTask):
    """The same native Task with its physical execution entry guarded."""
    def __init__(self, *args, data_root=None, **kwargs):
        super().__init__(*args, data_root=native.ORIGINAL_DATA_ROOT if data_root is None else data_root,
                         **kwargs)
        self.env = FullSpendPhysicalGuard(self.env, self.spec, domain=self.domain, seed=self.seed)

    def reset(self, phase, index):
        self.env.set_context(phase, index)
        return super().reset(phase, index)
