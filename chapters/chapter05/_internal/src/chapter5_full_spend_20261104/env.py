"""One shared whole-budget task/decoder for every Chapter5 learner.

Native observation values, complete latent coordinates, physical rewards and
C/J terminal labels stay unchanged. Public viability bounds change the actual
training/dev/test execution rule uniformly, rather than repairing evaluations.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from . import bridge

_BINDINGS=bridge.bootstrap()
Task,Actor=_BINDINGS['Task'],_BINDINGS['Actor']
TRAIN_MASTER=2026110481101
RELEASE_ROOT=bridge.NATIVE_SOURCE.resolve()
OBSERVATION_VERSION='A-public-local-global-quota-ledger-flat-full-spend-v2'
ACTION_VERSION='A_FULL-adaptive-complete-joint-latent-native-full-spend-v2'
DEFAULT_THRESHOLDS={'radio':.2,'v5':.7}
OBJECTIVE_ORDER=('C','J')
VECTOR_REWARD_VERSION='terminal-coverage-and-native-quality-full-spend-v2'

class NativeEnv(gym.Env):
    """One native task per complete episode, with explicit real-step accounting.

    ``raw`` actions are unbounded joint latent coordinates. ``tanh_latent``
    actions are their tanh coordinates; atanh uses an explicit numerical
    endpoint clip, never a physical-action density or an additional scaling.
    The two added scalar observations are remaining/initial and budget/initial,
    exactly the public inputs of A's quota head (policy.py:100--113).
    """
    metadata = {'render_modes': []}

    def __init__(self, domain: str, seed: int = 0, phase: str = 'train',
                 action_mode: str = 'tanh_latent', *, path_master: int = TRAIN_MASTER,
                 holdout: bool = False, start_episode: int = 0, radio_groups: int = 100,
                 data_root: str | Path | None = None, observation_dtype=np.float32,
                 action_epsilon: float = 1e-6):
        super().__init__()
        if domain not in ('radio', 'v5'):
            raise ValueError('domain must be radio or v5')
        if phase not in ('train', 'dev', 'final', 'engineering'):
            raise ValueError('Unknown task phase')
        if action_mode not in ('raw', 'tanh_latent'):
            raise ValueError('action_mode must be raw or tanh_latent')
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if isinstance(start_episode, bool) or int(start_episode) != start_episode or start_episode < 0:
            raise ValueError('start_episode must be a nonnegative integer')
        dtype = np.dtype(observation_dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError('Only float32/float64 neural observation boundaries are supported')
        if not 0 < action_epsilon < .01:
            raise ValueError('A small positive endpoint epsilon is required')
        self.domain, self.phase, self.action_mode = domain, phase, action_mode
        self.seed_value, self.path_master = int(seed), int(path_master)
        self.holdout, self.radio_groups, self.data_root = bool(holdout), int(radio_groups), data_root
        self.start_episode, self.next_episode = int(start_episode), int(start_episode)
        self.observation_dtype, self.action_epsilon = dtype, float(action_epsilon)
        self.task = self._make_task(self.seed_value)
        # The spec is public experiment metadata; gym.Env.spec remains available
        # to registration wrappers, while native_spec is the physical contract.
        self.native_spec = self.task.spec
        self.horizon, self.groups = self.native_spec.horizon, self.native_spec.groups
        self.decoder = bridge.make_decoder(self.native_spec, 0)
        self.decoder.requires_grad_(False)
        self.action_dim = self.decoder.event_dim
        self.observation_dim = (self.groups * self.native_spec.local_dim
                                + self.native_spec.global_dim + 2)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.observation_dim,), dtype=dtype)
        action_limit = np.inf if action_mode == 'raw' else 1.
        self.action_space = spaces.Box(-action_limit, action_limit, (self.action_dim,), dtype=dtype)
        self.action_space.seed(self.seed_value)
        self.real_steps = 0
        self.completed_episodes = 0
        self.abandoned_episodes = 0
        self.reset_count = 0
        self._active = False
        self._state = None
        self._encoder = None
        self._round = 0
        self._return = 0.
        self._episode_spent = 0.
        self._current_index = None
        self._last_raw = None
        self._last_physical_action = None
        self._last_completed_return = None
        self._last_completed_index = None
        self._last_completed_length = None

    def _make_task(self, seed):
        return bridge.make_task(self.domain, seed, path_master=self.path_master,
                    radio_groups=self.radio_groups, data_root=self.data_root, holdout=self.holdout)

    def _flatten(self, state):
        spec = self.native_spec
        # policy.py quota_mean explicitly consumes BOTH ledger scalars even
        # though remaining is also present in state.global for chapter 3.
        observation = np.concatenate((np.asarray(state['local']).ravel(),
            np.asarray(state['global']).ravel(),
            np.asarray([float(state['remaining']) / spec.initial_budget,
                        float(state['budget']) / spec.initial_budget])))
        observation = observation.astype(self.observation_dtype, copy=False)
        if observation.shape != self.observation_space.shape or not np.isfinite(observation).all():
            raise FloatingPointError('Public neural observation is not finite and fixed-shape')
        return observation

    def raw_action(self, action):
        """Map the learner's recorded coordinate to the complete latent event."""
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (self.action_dim,) or not np.isfinite(a).all():
            raise ValueError('One complete finite joint latent event is required')
        if self.action_mode == 'raw':
            return a.copy()
        if np.any(a < -1.) or np.any(a > 1.):
            raise ValueError('tanh_latent actions must belong to [-1,1]')
        limit = 1. - self.action_epsilon
        return np.arctanh(np.clip(a, -limit, limit))

    def decode_action(self, action):
        """Decode without stepping; private physics and reward are never read."""
        if not self._active or self._state is None:
            raise RuntimeError('reset() is required before decoding or stepping')
        raw = self.raw_action(action)
        with torch.no_grad():
            event = torch.as_tensor(raw, dtype=torch.float64)
            quota = self.decoder.quota(self._state, event[:1])
            physical = self.decoder.decode(self._state, event[1:], quota)[0]
        checked, _roundoff = self.task.validate(physical, self._state)
        return checked

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        unknown = set(options) - {'episode_index'}
        if unknown:
            raise ValueError(f'Unknown reset options: {sorted(unknown)}')
        if self._active and self._round:
            self.abandoned_episodes += 1
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
                raise ValueError('seed must be a nonnegative integer')
            if int(seed) != self.seed_value:
                self.seed_value = int(seed)
                self.task = self._make_task(self.seed_value)
            self.next_episode = self.start_episode
            self.action_space.seed(int(seed))
        index = options.get('episode_index', self.next_episode)
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0:
            raise ValueError('episode_index must be a nonnegative integer')
        index = int(index)
        observation, info = self.task.reset(self.phase, index)
        if info != {}:
            raise RuntimeError('Native reset unexpectedly supplied additional training information')
        encoder = self.task.encoder()
        encoder.reset(observation)
        self._encoder, self._state = encoder, encoder.state()
        self._active, self._round, self._return = True, 0, 0.
        self._episode_spent = 0.
        self._current_index, self.next_episode = index, index + 1
        self.reset_count += 1
        return self._flatten(self._state), {}

    def step(self, action):
        if not self._active:
            raise RuntimeError('A reset is required after true episode termination')
        raw = self.raw_action(action)
        physical = self.decode_action(action)
        observation, reward, terminated, truncated, info = self.task.env.step(physical)
        self.real_steps += 1
        self._round += 1
        expected_terminal = self._round == self.horizon
        if bool(truncated) or bool(terminated) != expected_terminal:
            raise RuntimeError('Native finite-horizon termination contract changed')
        reward = float(reward)
        if not np.isfinite(reward) or (not terminated and (reward != 0. or info != {})):
            raise RuntimeError('Terminal-only native reward/information contract changed')
        self._encoder.update(physical, observation)
        self._state = self._encoder.state()
        self._return += reward
        self._episode_spent += float(physical.sum())
        self._last_raw = raw.copy()
        self._last_physical_action = physical.copy()
        if terminated:
            self._last_full_spend = bridge.assert_full_episode(self._state['remaining'],
                self._episode_spent,self.native_spec)
            self._active = False
            self.completed_episodes += 1
            self._last_completed_return = self._return
            self._last_completed_index = self._current_index
            self._last_completed_length = self._round
        # Native terminal diagnostic info is discarded; no extra labels are
        # exposed to the algorithm, at either terminal or nonterminal steps.
        return self._flatten(self._state), reward, bool(terminated), False, {}

    @property
    def state(self):
        """Public state copy for audits; algorithms normally use flat observations."""
        if self._state is None:
            return None
        return {k: np.asarray(v).copy() for k, v in self._state.items()}

    @property
    def last_physical_action(self):
        return None if self._last_physical_action is None else self._last_physical_action.copy()

    @property
    def audit(self) -> dict[str, Any]:
        return dict(real_steps=self.real_steps, completed_episodes=self.completed_episodes,
            abandoned_episodes=self.abandoned_episodes, resets=self.reset_count,
            last_completed_return=self._last_completed_return,
            last_completed_index=self._last_completed_index,
            last_completed_length=self._last_completed_length,
            current_round=self._round, current_episode_index=self._current_index,
            next_episode_index=self.next_episode)

    @property
    def protocol(self):
        return dict(domain=self.domain, phase=self.phase, chapter=3, method='A_FULL',
            temporal='adaptive', seed=self.seed_value, path_master=self.path_master,
            holdout=self.holdout, horizon=self.horizon, groups=self.groups,
            observation_version=OBSERVATION_VERSION, observation_dim=self.observation_dim,
            observation_fields=['local.flatten', 'global', 'remaining/initial_budget', 'budget/initial_budget'],
            observation_dtype=self.observation_dtype.name, action_version=ACTION_VERSION,
            action_mode=self.action_mode, action_dim=self.action_dim,
            tanh_endpoint_epsilon=self.action_epsilon if self.action_mode == 'tanh_latent' else None,
            terminal_only=True, gamma=1., release_root=str(RELEASE_ROOT),
            task_module_file=str(Path(sys.modules[Task.__module__].__file__).resolve()),
            data_identity=self.task.data_identity, full_spend_required=True,
            budget_constraint='whole episode allocated = initial budget',
            full_spend_bridge_revision=bridge.REVISION,
            quota_interval='public robust [L,U] with exact endpoint branches')

    def close(self):
        if hasattr(self.task.env, 'close'):
            self.task.env.close()



class Env(NativeEnv):
    """Official continuous-MORL-compatible Gymnasium environment.

    ``start_episode=lane, episode_stride=n_envs`` gives disjoint task indices
    to vector lanes.  An explicit reset ``episode_index`` takes precedence.
    Repeating reset(seed=...) does not rewind the episode ledger.

    Bounded actions retain the original adapter's endpoint epsilon1e-6. The
    learner models the complete13/19-dimensional latent event, never a density
    on the many-to-one100/120-dimensional physical allocation.
    """

    reward_dim = 2

    def __init__(self, domain, seed=0, phase='train', action_mode='tanh_latent', *,
                 path_master=TRAIN_MASTER, holdout=False, start_episode=0,
                 episode_stride=1, tau=None, observation_dtype=np.float32,
                 action_epsilon=1e-6, radio_groups=100, data_root=None):
        if isinstance(episode_stride, bool) or int(episode_stride) != episode_stride or episode_stride < 1:
            raise ValueError('Positive integer episode_stride required')
        threshold = DEFAULT_THRESHOLDS.get(domain) if tau is None else float(tau)
        if threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Finite individual threshold in [0,1] required')
        super().__init__(domain, seed, phase, action_mode, path_master=path_master,
                         holdout=holdout, start_episode=start_episode, radio_groups=radio_groups,
                         data_root=data_root, observation_dtype=observation_dtype,
                         action_epsilon=action_epsilon)
        self.tau = float(threshold)
        self.episode_stride = int(episode_stride)
        self._next_plan_index = int(start_episode)
        self.reward_space = spaces.Box(np.array([0., self.native_spec.reward_range[0]], dtype=self.observation_dtype),
                                      np.ones(2, dtype=self.observation_dtype), dtype=self.observation_dtype)
        self._counts = {name: dict(episodes=0, physical_steps=0)
                        for name in ('train', 'development', 'holdout', 'engineering')}
        self._count_phase = {'train':'train', 'dev':'development', 'final':'holdout',
                             'engineering':'engineering'}[phase]
        self._plan = None
        self._result = None
        self._latent_actions, self._raw_actions, self._physical_actions = [], [], []
        self._native_rewards, self._vector_rewards = [], []
        self.terminal_quality_reads = 0

    @property
    def counts(self):
        return copy.deepcopy(self._counts)

    @property
    def steps_physical(self):
        return sum(x['physical_steps'] for x in self._counts.values())

    @property
    def episode_index(self):
        return self._current_index

    @property
    def episode_plan(self):
        if self._plan is None:
            raise RuntimeError('reset() is required before reading the episode plan')
        return copy.deepcopy(self._plan)

    @property
    def plan_sha256(self):
        return hashlib.sha256(json.dumps(self.episode_plan, sort_keys=True).encode()).hexdigest()

    @property
    def terminal_result(self):
        if self._result is None:
            raise RuntimeError('Individual outcomes and traces are available only after true termination')
        return copy.deepcopy(self._result)

    @property
    def last_raw_action(self):
        return None if self._last_raw is None else self._last_raw.copy()

    def reset(self, *, seed=None, options=None):
        options = {} if options is None else dict(options)
        if set(options)-{'episode_index'}:
            raise ValueError('Only episode_index is accepted in reset options')
        index = options.get('episode_index', self._next_plan_index)
        observation, info = super().reset(seed=seed, options={'episode_index':index})
        self._next_plan_index = int(index)+self.episode_stride
        self.next_episode = self._next_plan_index
        self._plan = self.task.plan(self.phase, int(index))
        self._result = None
        self._latent_actions, self._raw_actions, self._physical_actions = [], [], []
        self._native_rewards, self._vector_rewards = [], []
        return observation, info

    def _terminal_quality(self):
        if self._active or self._round != self.horizon:
            raise RuntimeError('Terminal quality must never be read before true done')
        if self.domain == 'radio':
            if self.task.env._round != self.horizon:
                raise AssertionError('Radio terminal horizon mismatch')
            quality = self.task.env._terminal_ssim
        else:
            if not self.task.env._terminated:
                raise AssertionError('Emergency task has not terminated')
            quality = self.task.env._terminal_info()['per_region_completion']
        y = np.asarray(quality, dtype=np.float64).copy()
        if y.shape != (self.groups,) or not np.isfinite(y).all():
            raise AssertionError('Finite individual terminal quality vector required')
        self.terminal_quality_reads += 1
        return y

    def step(self, action):
        observation, native_reward, terminated, truncated, info = super().step(action)
        if info or truncated:
            raise AssertionError('The native clean-info finite-horizon contract changed')
        self._counts[self._count_phase]['physical_steps'] += 1
        self._counts[self._count_phase]['episodes'] += int(terminated)
        self._latent_actions.append(np.asarray(action, dtype=np.float64).copy())
        self._raw_actions.append(self._last_raw.copy())
        self._physical_actions.append(self._last_physical_action.copy())
        self._native_rewards.append(float(native_reward))
        reward = np.zeros(2, dtype=self.observation_dtype)
        if terminated:
            quality = self._terminal_quality()
            k, j = int(np.count_nonzero(quality >= self.tau)), float(quality.mean())
            if abs(j-self._return) > 1e-10 or abs(j-native_reward) > 1e-10:
                raise AssertionError('Native terminal reward does not equal mean quality')
            c = k/self.groups
            reward[:] = (c, j)
            actions = np.asarray(self._physical_actions)
            total, remaining = float(actions.sum()), float(self._state['remaining'])
            if abs(total+remaining-self.native_spec.initial_budget) > max(1e-10, self.native_spec.initial_budget*1e-12):
                raise AssertionError('Episode resource ledger mismatch')
            full_spend = bridge.assert_full_episode(remaining,total,self.native_spec)
            self._result = dict(domain=self.domain, phase=self.phase, index=self.episode_index,
                plan=self.episode_plan, plan_sha256=self.plan_sha256, tau=self.tau,
                N=self.groups, K=k, C=c, J=j, native_return=j, score_x100=100*j,
                quality=quality, latent_actions=np.asarray(self._latent_actions),
                raw_actions=np.asarray(self._raw_actions), actions=actions,
                physical_actions=actions.copy(), native_rewards=np.asarray(self._native_rewards),
                total_used=total, remaining=remaining, period_totals=actions.sum(1),
                physical_total_used=total*self.native_spec.action_cost_scale,
                physical_remaining=remaining*self.native_spec.action_cost_scale,
                physical_resource_unit=self.native_spec.resource_unit,
                full_spend=full_spend,full_spend_required=True)
        elif native_reward != 0:
            raise AssertionError('Only terminal reward is allowed')
        self._vector_rewards.append(reward.copy())
        if terminated:
            self._result['rewards'] = np.asarray(self._vector_rewards)
        return observation, reward, bool(terminated), False, {}

    @property
    def protocol(self):
        result = super().protocol
        result.update(vector_reward_version=VECTOR_REWARD_VERSION, reward_dim=2,
            objective_order=list(OBJECTIVE_ORDER), tau=self.tau,
            coverage='terminal count(quality>=tau)/N', overall='unchanged native terminal mean quality',
            reward_dtype=self.observation_dtype.name,
            intermediate_reward_vector=[0.,0.], extra_training_info=False,
            individual_quality_is_policy_input=False, episode_stride=self.episode_stride,
            explicit_global_episode_index=True)
        return result



def make_env(domain,seed=0,**kwargs):
    return Env(domain,seed,**kwargs)
