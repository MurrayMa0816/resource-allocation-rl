"""Public-only interface for Chapter 3 online comparators.

The existing physics, train/development/test path isolation, control decoder and
terminal task reward are reused. The cost is actually consumed resource/B;
it is not an attempted violation, a utility oracle, or a shaped task reward.
"""
from __future__ import annotations
from copy import deepcopy
import gymnasium as gym
import numpy as np
from ..baselines.adapter import BaselineEnv


class PublicBudgetEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, domain, *, seed=0, phase='engineering', path_master=None,
                 holdout=False, condition_file=None, train_path_base=None):
        kwargs = dict(domain=domain, chapter=3, model_seed=seed, phase=phase,
                      holdout=holdout, action_family='total_space', method=None,
                      condition_file=condition_file, train_path_base=train_path_base)
        if path_master is not None:
            kwargs['path_master'] = int(path_master)
        self.bridge = BaselineEnv(**kwargs)
        self.specification = self.bridge.spec
        self.horizon = self.bridge.horizon
        self.policy_indices = np.asarray(self.bridge.policy_indices, dtype=np.int64)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
            shape=(len(self.policy_indices),), dtype=np.float32)
        self.action_space = self.bridge.action_space
        self.episode_cost = 0.
        self.episode_steps = 0
        self.next_index = 0
        self._done = True

    def observation(self, full):
        out = np.asarray(full[self.policy_indices], np.float32)
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite public policy observation')
        return out

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('Unregistered task options are forbidden')
        index = self.next_index if seed is None else seed
        full, info = self.bridge.reset(seed=index)
        self.next_index = int(index) + 1
        self.episode_cost = 0.
        self.episode_steps = 0
        self._done = False
        return self.observation(full), dict(info, cost=0., cumulative_cost=0.)

    def step(self, action):
        if self._done:
            raise RuntimeError('Reset before stepping an ended task')
        initial = self.specification.initial_budget
        before = float(self.bridge.decision_state['remaining'])
        full, reward, done, truncated, info = self.bridge.step(action)
        after = float(self.bridge.decision_state['remaining'])
        amount = float(info['physical_amount'])
        cost = amount / initial
        tolerance = 512*np.finfo(float).eps*max(1., initial)
        if abs((before-after)-amount) > tolerance:
            raise ArithmeticError('Actual resource debit differs from physical allocation')
        self.episode_cost += cost
        self.episode_steps += 1
        if self.episode_cost > 1. + 512*np.finfo(float).eps:
            raise ArithmeticError('Physical task exceeded its consumable budget')
        if not done and reward != 0.:
            raise AssertionError('The original task has terminal-only feedback')
        if done and abs(self.episode_cost-1.)>512*np.finfo(float).eps:
            raise ArithmeticError('Every full-spend episode must allocate its entire initial resource')
        self._done = bool(done)
        out = dict(info, cost=cost, cumulative_cost=self.episode_cost,
                   remaining_budget_fraction=after/initial,
                   native_reward=float(reward), episode_step=self.episode_steps)
        # Runtime cost and geometry diagnostics go to the collector, never the network.
        return self.observation(full), reward, done, truncated, out

    def contract(self):
        return dict(schema='chapter3-public-budget-online-v1',
            native_contract=self.bridge.get_contract(),
            neural_observation_fields=['local','global'],
            neural_observation_dim=self.observation_space.shape[0],
            reward='unchanged native terminal reward',
            cost='physical_amount / initial_equivalent_resource_budget',
            physical_initial_cost=self.specification.initial_budget*self.specification.action_cost_scale,
            cost_limit=1., hard_feasibility='native action set; independent of learned cost estimates',
            full_spend_required=True,terminal_resource_fraction=1.,
            total_interval='public robust completion lower bound and unchanged physical upper bound',
            phase=self.bridge.phase, holdout=self.bridge.holdout)

    def replay_last_episode(self):
        return self.bridge.replay_last_episode()

    def close(self):
        return self.bridge.close()
