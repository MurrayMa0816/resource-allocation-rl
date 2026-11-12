"""Shared Gym API, bound exclusively to the copied full-spend native task.

The public flat observation/action coordinates retain the frozen C4 interface;
only the common physical feasible total mapping changes to the verified C3 one.
"""
from __future__ import annotations
from pathlib import Path
import sys
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from .native import Actor, TRAIN_MASTER, RELEASE_ROOT, _RELEASE_PACKAGE
from .task import Task
OBSERVATION_VERSION = 'A-public-local-global-quota-ledger-flat-v1'
ACTION_VERSION = 'A_FULL-adaptive-complete-joint-latent-C3-full-spend-v1'

class Env(gym.Env):
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
        self.decoder = Actor(self.native_spec, 'A_FULL', 'adaptive', 0).cpu().eval()
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
        self._current_index = None
        self._last_raw = None
        self._last_physical_action = None
        self._last_completed_return = None
        self._last_completed_index = None
        self._last_completed_length = None

    def _make_task(self, seed):
        return Task(self.domain, 3, 'A', seed, path_master=self.path_master,
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
        self._last_raw = raw.copy()
        self._last_physical_action = physical.copy()
        if terminated:
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
            terminal_only=True, gamma=1., full_spend_required=True,
            full_spend_mapping='copied C3 [L,U] public feasibility; exact endpoints',
            inherited_quota_descriptor_is_not_executed_math=True, release_root=str(RELEASE_ROOT),
            task_module_file=str(Path(sys.modules[Task.__module__].__file__).resolve()),
            data_identity=self.task.data_identity)

    def close(self):
        if hasattr(self.task.env, 'close'):
            self.task.env.close()


def make_env(domain: str, seed: int = 0, **kwargs):
    """Return an environment instance (use a lambda when a vector factory is needed)."""
    return Env(domain, seed, **kwargs)
