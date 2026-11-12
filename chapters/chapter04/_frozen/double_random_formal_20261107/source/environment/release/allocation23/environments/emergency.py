"""Backlog environment with the current simplified public observation."""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from ._emergency.core import EmergencyAllocationEnv as _PhysicalEnvironment
from ..full_spend import budget_roundoff


class EmergencyEnvironment(_PhysicalEnvironment):
    """Unchanged irreversible inputs and terminal regional completion reward.

    Total demand seen so far is the initial demand plus arrivals already seen.
    It contains no future demand. Completion/arrival events are retained only
    in the experiment audit accessor, never in the returned observation.
    """

    def __init__(self, config, chapter):
        if isinstance(chapter, bool) or chapter not in (2, 3):
            raise ValueError('This release supports chapters 2 and 3')
        if config.budget_mode != ('fixed' if chapter == 2 else 'adaptive'):
            raise ValueError('Environment mode must match chapter')
        self.chapter = chapter
        self._native_observation_for_audit = None
        super().__init__(config)
        self.observation_version = f'emergency-v5-demand-total-seen-chapter{chapter}-v1'
        previous = self.observation_space.spaces
        declared = {key: previous[key] for key in ('remaining_demand', 'time_index')}
        declared['total_demand_so_far'] = spaces.Box(
            0., np.inf, shape=previous['remaining_demand'].shape, dtype=np.float64)
        if chapter == 3:
            declared['remaining_budget'] = previous['remaining_budget']
        self.observation_space = spaces.Dict(declared)
        self.observation_fields = tuple(self.observation_space.spaces)
        self.reward_version = 'terminal-mean-regional-backlog-completion-v5'
        if chapter==3:
            self.action_space._minimum_total=self._minimum_full_spend_input

    def _minimum_full_spend_input(self):
        return (float(self._remaining_budget) if
                getattr(self,'_time_index',0)==self._resolved.horizon-1 else 0.)

    def step(self,action):
        if self.chapter==3 and self._initialized and not self._terminated:
            lower=self._minimum_full_spend_input()
            if float(np.asarray(action,dtype=np.float64).sum())<lower-budget_roundoff(self._resolved.initial_budget):
                raise ValueError('Last period must allocate all remaining emergency allowance')
        result=super().step(action)
        if self.chapter==3 and result[2] and self._remaining_budget>budget_roundoff(self._resolved.initial_budget):
            raise ArithmeticError('Emergency full-allocation task ended with unspent allowance')
        return result

    @property
    def native_observation_for_audit(self):
        if self._native_observation_for_audit is None:
            raise RuntimeError('Reset before reading the experiment audit trace')
        return {key: np.asarray(value).copy() for key, value in self._native_observation_for_audit.items()}

    def _observation(self):
        native = super()._observation()
        self._native_observation_for_audit = native
        result = {key: np.asarray(native[key]).copy() for key in ('remaining_demand', 'time_index')}
        result['total_demand_so_far'] = self._arrived.copy()
        if self.chapter == 3:
            result['remaining_budget'] = np.asarray(native['remaining_budget']).copy()
        return result
