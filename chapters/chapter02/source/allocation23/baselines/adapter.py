"""Gym bridge to the existing native tasks, with decoder-only public geometry.

Chapter 2 uses the fixed feasible period total. Chapter 3 PASPO/RPO use an
unused-first physical equality simplex; generic continuous learners use one
amount control followed by N spatial controls. Both cover the full native set.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..io import TRAIN_MASTER
from ..methods.constraints import capped_simplex
from ..task import Task
from .actions import (REVISION as ACTION_REVISION, action_family_for,
                      reference_amount, amount_from_control, control_from_amount)

REVISION = 'current-native-baseline-gym-v2-calibrated-actions'


def _same(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    return left == right


class BaselineEnv(gym.Env):
    """Same physical task and state encoding as A/B; a full external action.

    ``train_path_base=None`` selects an evaluation stream; otherwise resets use
    successive predefined training paths. Geometry remains float64 throughout.
    The coordinator's legacy method tag A never constructs or calls an A actor.
    """

    metadata = {'render_modes': []}

    def __init__(self, domain, chapter, *, train_path_base=None, model_seed=0,
                 path_master=TRAIN_MASTER, radio_groups=100, data_root=None,
                 phase='dev', holdout=False, action_family='polytope', method=None,
                 condition_file=None):
        super().__init__()
        if phase not in ('train', 'dev', 'final', 'engineering'):
            raise ValueError('Unknown native task phase')
        if train_path_base is not None:
            if int(train_path_base) != train_path_base or train_path_base < 0:
                raise ValueError('A nonnegative training namespace is required')
            path_master, phase = int(train_path_base), 'train'
        self.domain, self.chapter = domain, int(chapter)
        if action_family not in ('polytope', 'total_space'):
            raise ValueError('Unknown baseline action family')
        self.action_family = (action_family_for(method, chapter) if method is not None
                              else action_family)
        if self.chapter == 2 and self.action_family != 'polytope':
            raise ValueError('Chapter 2 uses only its fixed-total physical polytope')
        self.train_path_base, self.model_seed = train_path_base, int(model_seed)
        self.path_master, self.phase = int(path_master), phase
        self.holdout = bool(holdout)
        self._task_kwargs = dict(domain=domain, chapter=chapter, method='A', seed=model_seed,
                                 path_master=path_master, radio_groups=radio_groups,
                                 data_root=data_root, holdout=holdout, condition_file=condition_file)
        self.task = Task(**self._task_kwargs)
        self.native_env, self.spec = self.task.env, self.task.spec
        self.n_objects, self.horizon = self.spec.groups, self.spec.horizon
        self.action_dim = self.n_objects + int(chapter == 3)
        self.fixed_quota = self.spec.nominal_total
        self.action_space = spaces.Box(-1., 1., shape=(self.action_dim,), dtype=np.float32)
        offset, self._schema = 0, []
        for name, shape, divisor in [
            ('local', (self.n_objects, self.spec.local_dim), 1.),
            ('global', (self.spec.global_dim,), 1.),
            ('caps', (self.action_dim,), self.spec.nominal_total / self.n_objects),
            ('budget', (), self.spec.nominal_total),
        ]:
            size = int(np.prod(shape)) if shape else 1
            self._schema.append(dict(name=name, shape=list(shape), start=offset,
                                     stop=offset + size, divisor=float(divisor)))
            offset += size
        self.policy_indices = list(range(self._schema[1]['stop']))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(offset,), dtype=np.float64)
        self.next_episode = self.environment_steps = 0
        self.episode_records = []
        self._initialized, self._done = False, False
        self._index = None
        self.native_obs = self.decision_state = self._encoder = None
        self._actions, self._rewards, self._observations = [], [], []
        self._raw_return, self._path_seed = 0., None
        self._terminal_info = {}

    def configure_method(self, method):
        """Bind once before a run identity is captured; never switch live tasks."""
        family = action_family_for(method, self.chapter)
        if family != self.action_family:
            if self._initialized:
                raise ValueError('Cannot change baseline action family after reset')
            self.action_family = family
        return self

    def get_contract(self):
        return dict(revision=REVISION, domain=self.domain, chapter=self.chapter,
                    adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    n_objects=self.n_objects, horizon=self.horizon, action_dim=self.action_dim,
                    physical=self.task.physical_dict, spec=self.spec.to_dict(),
                    data=self.task.data_identity, observation_schema=deepcopy(self._schema),
                    policy_indices=self.policy_indices,
                    geometry_indices=list(range(self.policy_indices[-1] + 1,
                                                self.observation_space.shape[0])),
                    policy_fields=['local', 'global'], geometry_fields=['caps', 'budget'],
                    geometry_neural_input=False, observation_dtype='float64',
                    action_family=self.action_family, action_revision=ACTION_REVISION,
                    action_map=('fixed-total bounded simplex' if self.chapter == 2 else
                                'unused first, then physical allocation; exact physical polytope'
                                if self.action_family == 'polytope' else
                                'amount first, then spatial controls; neutral public odds scale'),
                    coordinate_order=('objects' if self.chapter == 2 else
                                      'unused,objects' if self.action_family == 'polytope'
                                      else 'amount,spatial_controls'),
                    amount_scale=('remaining/remaining_rounds; fixed coordinate map'
                                  if self.action_family == 'total_space' else None),
                    reward='unchanged native terminal reward; no reward_scale',
                    reward_version=self.task.reward_version,
                    resource_ledger='native equivalent L' if self.domain == 'v5' else 'energy_J/slot_s',
                    terminal_observation='encoded terminal state with zero feasible geometry')

    def _geometry(self):
        s = self.decision_state
        upper = float(s['budget'])
        target = min(self.spec.nominal_total, upper) if self.chapter == 2 else upper
        caps = np.minimum(np.asarray(s['caps'], np.float64), target)
        if self.chapter == 3:
            caps = np.r_[target, caps]
        return caps, target

    def _flatten(self):
        caps, target = self._geometry()
        values = dict(local=self.decision_state['local'],
                      global_=self.decision_state['global'], caps=caps, budget=np.asarray(target))
        values['global'] = values.pop('global_')
        result = np.concatenate([(np.asarray(values[f['name']]) / f['divisor']).reshape(-1)
                                 for f in self._schema]).astype(np.float64, copy=False)
        if result.shape != self.observation_space.shape or not np.isfinite(result).all():
            raise ValueError('Invalid public baseline observation')
        return result

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('Task overrides are not part of the registered baseline protocol')
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        if self.train_path_base is not None:
            index = self.next_episode
        else:
            if seed is None:
                raise ValueError('Evaluation requires an explicit task index as seed')
            index = int(seed)
            if index < 0 or index != seed:
                raise ValueError('Nonnegative evaluation task index required')
        return self._reset_index(index)

    def _reset_index(self, index):
        self._index = int(index)
        obs, info = self.task.reset(self.phase, self._index)
        if info != {}:
            raise RuntimeError('Native reset information contract changed')
        self._encoder = self.task.encoder()
        self._encoder.reset(obs)
        self.native_obs, self.decision_state = obs, self._encoder.state()
        self._actions, self._rewards, self._observations = [], [], []
        self._raw_return = 0.
        self._terminal_info = {}
        self._path_seed = self.task.plan(self.phase, self._index)['task_seed']
        self.next_episode += 1
        self._initialized, self._done = True, False
        return self._flatten(), dict(path_seed=self._path_seed, task_index=self._index)

    def physical_action(self, control):
        if not self._initialized or self._done:
            raise RuntimeError('Reset an active task before requesting an action')
        control = np.asarray(control, dtype=np.float64)
        if (control.shape != self.action_space.shape or not np.isfinite(control).all()
                or np.any(np.abs(control) > 1. + 1e-7)):
            raise ValueError('Expected a finite bounded full baseline control')
        caps, target = self._geometry()
        control = np.clip(control, -1., 1.)
        if self.chapter == 3 and self.action_family == 'total_space':
            q = amount_from_control(control[0], target,
                                    reference_amount(self.decision_state, self.horizon))
            proposal = q * (control[1:] + 1.) / 2.
            allocation = capped_simplex(proposal, caps[1:], q)
        else:
            proposal = target * (control + 1.) / 2.
            augmented = capped_simplex(proposal, caps, target)
            allocation = augmented if self.chapter == 2 else augmented[1:]
        action, _ = self.task.validate(allocation, self.decision_state)
        if self.chapter == 2 and not np.isclose(action.sum(), target, rtol=1e-12,
                                               atol=self.spec.action_tolerance):
            raise ArithmeticError('Fixed period total was not conserved')
        return action

    def control_for_allocation(self, allocation):
        """A bounded representative for every currently feasible native action.

        This inverse is a permissions/round-trip check, never a business policy.
        """
        if not self._initialized or self._done:
            raise RuntimeError('Reset an active task before encoding an action')
        allocation, _ = self.task.validate(allocation, self.decision_state)
        caps, upper = self._geometry()
        total = float(allocation.sum())
        if self.chapter == 2:
            if not np.isclose(total, upper, rtol=1e-12, atol=self.spec.action_tolerance):
                raise ValueError('Chapter 2 requires its fixed feasible total')
            return 2. * allocation / upper - 1. if upper else np.full(self.action_dim, -1.)
        if self.action_family == 'polytope':
            values = np.r_[upper-total, allocation]
            return np.clip(2. * values / upper - 1., -1., 1.) if upper else np.full(self.action_dim, -1.)
        amount = control_from_amount(total, upper,
                                     reference_amount(self.decision_state, self.horizon))
        spatial = 2. * allocation / total - 1. if total else np.zeros(self.n_objects)
        return np.r_[amount, np.clip(spatial, -1., 1.)]

    def step(self, control):
        action = self.physical_action(control)
        self._observations.append(deepcopy(self.native_obs))
        obs, reward, done, truncated, info = self.native_env.step(action)
        if truncated or done != (len(self._actions) == self.horizon - 1):
            raise RuntimeError('Native complete-episode termination contract changed')
        if not done and (reward != 0. or info != {}):
            raise RuntimeError('Native terminal-only feedback contract changed')
        self._encoder.update(action, obs)
        self.native_obs, self.decision_state = obs, self._encoder.state()
        self._actions.append(action.copy())
        self._rewards.append(float(reward))
        self._raw_return += float(reward)
        self.environment_steps += 1
        self._done = bool(done)
        self._terminal_info = info if done else {}
        out = dict(physical_action=action.copy(), physical_amount=float(action.sum()),
                   path_seed=self._path_seed, raw_reward=float(reward))
        if done:
            record = dict(path_seed=self._path_seed, task_index=self._index,
                          raw_return=self._raw_return, report_score=self._raw_return,
                          score_x100=100 * self._raw_return, steps=len(self._actions),
                          total_used=float(np.asarray(self._actions).sum()),
                          remaining=float(self.decision_state['remaining']))
            self.episode_records.append(record)
            out['episode_result'] = deepcopy(record)
        return self._flatten(), float(reward), bool(done), False, out

    def replay_last_episode(self):
        if not self._done:
            raise ValueError('Native replay requires a completed episode')
        replay = Task(**self._task_kwargs)
        result = dict(actions=np.asarray(self._actions), rewards=np.asarray(self._rewards),
                      observations=self._observations, native_return=self._raw_return)
        try:
            reward, trace_hash = replay.replay(self.phase, self._index, result)
        finally:
            replay.env.close()
        return dict(error=abs(reward - self._raw_return), observed_trace_sha256=trace_hash,
                    actions_sha256=hashlib.sha256(result['actions'].tobytes()).hexdigest())

    def export_state(self):
        return deepcopy(dict(revision=REVISION, contract=self.get_contract(),
            stream=dict(train_path_base=self.train_path_base, model_seed=self.model_seed,
                        path_master=self.path_master, phase=self.phase, holdout=self.holdout),
            initialized=self._initialized, done=self._done, index=self._index,
            next_episode=self.next_episode, environment_steps=self.environment_steps,
            episode_records=self.episode_records, actions=self._actions,
            rewards=self._rewards, native_obs=self.native_obs,
            gym_rng=self.np_random.bit_generator.state, gym_rng_seed=self.np_random_seed,
            action_rng=self.action_space.np_random.bit_generator.state))

    def restore_state(self, snapshot):
        saved = deepcopy(snapshot)
        stream = dict(train_path_base=self.train_path_base, model_seed=self.model_seed,
                      path_master=self.path_master, phase=self.phase, holdout=self.holdout)
        if (saved.get('revision') != REVISION or saved.get('contract') != self.get_contract()
                or saved.get('stream') != stream):
            raise ValueError('Baseline environment continuation identity differs')
        if saved['initialized']:
            self._reset_index(saved['index'])
            for action, reward in zip(saved['actions'], saved['rewards']):
                self._observations.append(deepcopy(self.native_obs))
                obs, actual, done, truncated, info = self.native_env.step(action)
                if truncated or actual != reward:
                    raise ValueError('Native continuation replay differs')
                self._encoder.update(action, obs)
                self.native_obs, self.decision_state = obs, self._encoder.state()
                self._actions.append(np.asarray(action).copy())
                self._rewards.append(float(actual))
                self._raw_return += float(actual)
                self._done = bool(done)
                self._terminal_info = info if done else {}
            if not _same(self.native_obs, saved['native_obs']) or self._done != saved['done']:
                raise ValueError('Saved public state differs after native continuation replay')
        else:
            self._initialized, self._done, self._index = False, False, None
            self.native_obs = self.decision_state = self._encoder = None
            self._actions, self._rewards, self._observations = [], [], []
            self._raw_return, self._path_seed = 0., None
            self._terminal_info = {}
        self.next_episode, self.environment_steps = saved['next_episode'], saved['environment_steps']
        self.episode_records = saved['episode_records']
        self.np_random.bit_generator.state = saved['gym_rng']
        self._np_random_seed = saved['gym_rng_seed']
        self.action_space.np_random.bit_generator.state = saved['action_rng']

    import_state = restore_state

    def close(self):
        self.native_env.close()
