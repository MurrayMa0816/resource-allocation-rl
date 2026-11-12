"""Pinned SafeDreamer OSRP-Lag, with a public vector/native-resource bridge.

The author RSSM, distributional reward/cost heads, imagined actor and both
critics, and constrained CEM/PID planner are retained. This is online learning
from scratch; no offline dataset, checkpoint, physical model or future RNG is
accepted. The native feasible action decoder makes an exact true-resource
episode constraint redundant. This candidate does not establish an independent
constraint-handling advantage or a learned pathwise safety guarantee.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np


REVISION = 'chapter3-safedreamer-osrp-lag-diagnostic-adaptation-v2'
VENDOR = Path(__file__).parent / 'vendor/safedreamer'


def provenance():
    manifest = json.loads((VENDOR / 'SOURCE_MANIFEST.json').read_text())
    for entry in manifest['files']:
        actual = hashlib.sha256((VENDOR / entry['path']).read_bytes()).hexdigest()
        if actual != entry['vendored_sha256']:
            raise ValueError('SafeDreamer source identity differs: ' + entry['path'])
    return manifest


def _backend(device):
    """Import only on construction; callers supply the isolated JAX runtime."""
    provenance()
    if device not in ('cpu', 'gpu'):
        raise ValueError('Use an explicitly admitted cpu or gpu JAX backend')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    if device == 'cpu':
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    core = VENDOR / 'SafeDreamer'
    existing = sys.modules.get('embodied')
    if existing and Path(existing.__file__).resolve().parent != core / 'embodied':
        raise RuntimeError('Another embodied package is loaded; use an isolated process')
    for path in (str(VENDOR), str(core)):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        embodied = importlib.import_module('embodied')
        upstream = importlib.import_module('SafeDreamer.agent')
        training = importlib.import_module('SafeDreamer.train')
    except ImportError as exc:
        raise ImportError('SafeDreamer needs the registered isolated JAX/TFP runtime '
                          'and shared dependency target; the active Torch environment '
                          'is deliberately unchanged') from exc
    return embodied, upstream, training


@dataclass(frozen=True)
class SafeDreamerConfig:
    horizon: int = 32
    discount: float = 1.0
    profile: str = 'paper_vector'
    batch_size: int = 16
    batch_length: int = 32
    replay_sequences: int = 100_000
    planner_horizon: int = 15
    planner_samples: int = 500
    planner_elites: int = 50
    planner_iterations: int = 6
    imagination_horizon: int = 15
    train_ratio: float = 512.0
    masked_safe_elites: bool = False
    inclusive_budget: bool = False
    planner_cost_mode: str = 'rate'
    finite_horizon_mask: bool = False
    observation_loss_per_dim: bool = False
    terminal_reward_model: bool = False

    @classmethod
    def engineering(cls, horizon=32):
        """Small dimensions for CPU integration only, with every core retained."""
        return cls(horizon=horizon, profile='engineering', batch_size=2,
                   batch_length=4, replay_sequences=128, planner_horizon=2,
                   planner_samples=8, planner_elites=2, planner_iterations=2,
                   imagination_horizon=3, train_ratio=1.0)


class SafeDreamerCandidate:
    def __init__(self, observation_dim, action_dim, seed=0, device='cpu',
                 config=None):
        self.config = config or SafeDreamerConfig()
        c = self.config
        if c.terminal_reward_model and not c.finite_horizon_mask:
            raise ValueError('Conditional terminal reward needs the exact public deadline mask')
        if (c.horizon < 1 or c.batch_length < 2 or c.batch_size < 1
                or not 0 < c.discount <= 1 or c.profile not in ('paper_vector', 'engineering')
                or c.planner_cost_mode not in ('rate', 'remaining_critic', 'native_certificate')
                or not 0 < c.planner_elites <= c.planner_samples
                or c.planner_horizon < 1 or c.planner_iterations < 1):
            raise ValueError('Invalid finite-task SafeDreamer declaration')
        self.observation_dim, self.action_dim = int(observation_dim), int(action_dim)
        self.seed, self.device = int(seed), device
        self.embodied, upstream, training = _backend(device)
        # Apply the official vector architecture to the official OSRP-Lag
        # variant. In particular the learned cost model is never disabled.
        cfg = self.embodied.Config(upstream.Agent.configs['defaults'])
        cfg = cfg.update(upstream.Agent.configs['osrp_lag'])
        cfg = cfg.update(upstream.Agent.configs['osrp_vector'])
        cfg = self.embodied.Config({**dict(cfg), 'native_discount': c.discount,
            'resource_horizon': float(c.horizon),
            'native_masked_safe_elites': c.masked_safe_elites,
            'native_inclusive_budget': c.inclusive_budget,
            'native_planner_cost_mode': c.planner_cost_mode})
        cfg = self.embodied.Config({**dict(cfg),
            'native_finite_horizon_mask': c.finite_horizon_mask,
            'native_observation_loss_per_dim': c.observation_loss_per_dim,
            'native_terminal_reward_model': c.terminal_reward_model})
        cfg = cfg.update({
            'seed': self.seed, 'method': 'osrp_lag', 'task': 'native_allocation3',
            'expl_behavior': 'PIDPlanner', 'use_cost': True, 'use_cost_model': True,
            'cost_limit': 1.0, 'cost_weight': 10.0, 'batch_size': c.batch_size,
            'batch_length': c.batch_length, 'data_loaders': 0,
            'jax.platform': device, 'jax.precision': 'float32', 'jax.prealloc': False,
            'jax.metrics_every': 1, 'encoder.mlp_keys': 'observation',
            'decoder.mlp_keys': 'observation', 'encoder.cnn_keys': '$^',
            'decoder.cnn_keys': '$^', 'imag_horizon': c.imagination_horizon,
            'planner.horizon': c.planner_horizon, 'planner.num_samples': c.planner_samples,
            'planner.num_elites': c.planner_elites,
            'planner.iterations': c.planner_iterations,
            'run.train_ratio': c.train_ratio,
        })
        if c.profile == 'engineering':
            cfg = cfg.update({r'.*\.units': 32, r'.*\.mlp_units': 32,
                              r'.*\.mlp_layers': 2, 'rssm.deter': 32,
                              'rssm.stoch': 4, 'rssm.classes': 4})
        self.effective_config = cfg
        E = self.embodied.Space
        self.obs_space = {
            'observation': E(np.float32, (self.observation_dim,)),
            'reward': E(np.float32), 'cost': E(np.float32),
            'is_first': E(bool), 'is_last': E(bool), 'is_terminal': E(bool),
            # Alias of the final, already-public observation coordinate. It is
            # read by the constraint checker, never a new neural input.
            'native_remaining_budget': E(np.float32),
            'native_steps_left': E(np.float32),
            **{key: E(np.float32) for key in
               ('lagrange_penalty', 'lagrange_p', 'lagrange_i', 'lagrange_d')},
        }
        self.act_space = {'action': E(np.float32, (self.action_dim,), -1., 1.),
                          'reset': E(bool)}
        self.counter = self.embodied.Counter()
        self.agent = upstream.Agent(self.obs_space, self.act_space, self.counter, cfg)
        self.pid = training.PIDLagrangian(cfg)
        self.replay = self.embodied.replay.Uniform(
            c.batch_length, c.replay_sequences, directory=None, seed=self.seed)
        self._stream = self.replay.dataset()
        self._carry = self._train_carry = None
        self._first = True
        self.last_planner_metrics = {}
        self.last_training_metrics = {}
        self._prev_reward = self._prev_cost = self._episode_cost = 0.0
        self._pending = None
        self._evaluation_only = False
        self._evaluation_active = False
        self.accounting = dict(environment_steps=0, train_updates=0,
                               completed_episodes=0, replay_rows=0)

    @property
    def metadata(self):
        return dict(revision=REVISION, algorithm='SafeDreamer(OSRP-Lag)',
                    source_commit=provenance()['commit'], config=asdict(self.config),
                    vendor_manifest_sha256=hashlib.sha256(
                        (VENDOR / 'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
                    adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    effective_config=dict(self.effective_config), device=self.device,
                    observation_dim=self.observation_dim, action_dim=self.action_dim,
                    online_from_scratch=True, offline_data=False, pretrained=False,
                    reward='unchanged native terminal F', cost='executed resource / initial B',
                    cost_limit=1.0, neural_input='public policy indices only; no geometry oracle',
                    resource_constraint_redundant_under_native_feasibility=True,
                    source_adaptations=provenance()['patches'],
                    checkpoint_type='evaluation state; no disk replay or exact training resume')

    def _row(self, observation, *, reward=None, cost=None, first=None, terminal=False):
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self.observation_dim,) or not np.isfinite(obs).all():
            raise ValueError('Expected finite public policy observation')
        global_dim = self.observation_dim // self.action_dim
        if self.observation_dim != self.action_dim * global_dim or global_dim not in (2,3):
            raise ValueError('Native public time coordinate layout differs')
        period = int(round(float(obs[-global_dim]) * self.config.horizon))
        if not 0 <= period <= self.config.horizon:
            raise ValueError('Public finite-horizon time is out of range')
        return dict(observation=obs.copy(),
                    reward=np.float32(self._prev_reward if reward is None else reward),
                    cost=np.float32(self._prev_cost if cost is None else cost),
                    is_first=np.asarray(self._first if first is None else first),
                    is_last=np.asarray(terminal), is_terminal=np.asarray(terminal),
                    native_remaining_budget=np.float32(obs[-1]),
                    native_steps_left=np.float32(self.config.horizon-period),
                    lagrange_penalty=np.float32(self.pid.lagrange_penalty),
                    lagrange_p=np.float32(self.pid.delta_p),
                    lagrange_i=np.float32(self.pid.pid_i),
                    lagrange_d=np.float32(self.pid.pid_d))

    def act(self, observation, deterministic=False):
        if self._pending is not None:
            raise RuntimeError('Observe the last real transition before requesting a new action')
        row = self._row(observation)
        outs, self._carry = self.agent.policy(
            {key: value[None] for key, value in row.items()}, self._carry,
            mode='eval' if deterministic else 'train')
        control = np.asarray(outs['action'][0], dtype=np.float32)
        if control.shape != (self.action_dim,) or not np.isfinite(control).all():
            raise FloatingPointError('Official planner produced invalid control')
        if np.max(np.abs(control)) > 1 + 1e-6:
            raise FloatingPointError('Official planner produced unbounded control')
        control = np.clip(control, -1., 1.)
        self._pending = (row, control.copy())
        # Small scalar diagnostics are kept in RAM for the independent runner.
        self.last_planner_metrics = {key: float(np.asarray(value).reshape(-1)[0])
            for key, value in outs.items() if key in (
                'log_plan_num_safe_traj', 'log_plan_ret', 'log_plan_cost',
                'log_lagrange_penalty')}
        return control

    def reset_episode(self):
        """Reset RSSM/planner carry at a real episode boundary, preserving weights."""
        self._carry, self._pending, self._first = None, None, True
        self._prev_reward = self._prev_cost = self._episode_cost = 0.0

    def observe(self, obs, raw_control, native_reward, normalized_resource_cost,
                next_obs, done, *, training=True):
        if training and (self._evaluation_only or self._evaluation_active):
            raise RuntimeError('Evaluation mode cannot collect training data')
        if self._pending is None:
            raise RuntimeError('A real action is required before observe')
        row, control = self._pending
        if not np.array_equal(row['observation'], np.asarray(obs, np.float32)):
            raise ValueError('Transition starts from another public observation')
        if not np.array_equal(control, np.asarray(raw_control, np.float32)):
            raise ValueError('Store the actual bounded raw control, not a decoded allocation')
        reward, cost = float(native_reward), float(normalized_resource_cost)
        if not np.isfinite(reward) or not np.isfinite(cost) or cost < -1e-12:
            raise ValueError('Use finite native reward and nonnegative true-resource cost')
        if not done and reward != 0.0:
            raise ValueError('The native task only provides terminal reward')
        row['action'] = control
        if training:
            self.replay.add(row)
            self.accounting['replay_rows'] += 1
        self._prev_reward, self._prev_cost = reward, max(0., cost)
        self._episode_cost += max(0., cost)
        self._first = False
        self._pending = None
        if training:
            self.counter.increment()
            self.accounting['environment_steps'] += 1
        if done:
            terminal = self._row(next_obs, first=False, terminal=True)
            terminal['action'] = np.zeros(self.action_dim, np.float32)
            if training:
                self.replay.add(terminal)
                self.accounting['replay_rows'] += 1
                self.pid.pid_update(self._episode_cost, int(self.counter))
                self.accounting['completed_episodes'] += 1
            self._first, self._carry = True, None
            self._prev_reward = self._prev_cost = self._episode_cost = 0.0

    def train_updates(self, count=1):
        if self._evaluation_only or self._evaluation_active:
            raise RuntimeError('Evaluation mode cannot update training state')
        if count < 0 or int(count) != count:
            raise ValueError('Nonnegative integer update count required')
        metrics = []
        if not len(self.replay):
            return metrics
        for _ in range(int(count)):
            sequences = [next(self._stream) for _ in range(self.config.batch_size)]
            batch = {key: np.stack([row[key] for row in sequences]) for key in sequences[0]}
            # Same explicit host-to-device conversion as the author dataset path.
            batch = self.agent._convert_inps(batch, self.agent.train_devices)
            _, self._train_carry, mets = self.agent.train(batch, self._train_carry)
            converted = {key: np.asarray(value) for key, value in mets.items()}
            metrics.append(converted)
            self.last_training_metrics = {key: float(value) for key, value in converted.items()
                if value.ndim == 0 and np.isfinite(value)}
            self.accounting['train_updates'] += 1
        return metrics

    def learn(self, env, episodes=1, updates_per_episode=None):
        if (env.chapter != 3 or env.action_family != 'total_space'
                or env.horizon != self.config.horizon or env.action_dim != self.action_dim
                or len(env.policy_indices) != self.observation_dim):
            raise ValueError('Native chapter-3/action/public-input contract differs')
        if updates_per_episode is None:
            updates_per_episode = max(1, round(self.config.train_ratio * env.horizon /
                                             (self.config.batch_size * self.config.batch_length)))
        budget = float(env.spec.initial_budget)
        out = []
        for _ in range(int(episodes)):
            self.reset_episode()
            obs, _ = env.reset()
            for _ in range(env.horizon):
                public = obs[env.policy_indices].astype(np.float32)
                control = self.act(public)
                nxt, reward, done, truncated, info = env.step(control)
                if truncated:
                    raise ValueError('Native episodes must end at their true horizon')
                self.observe(public, control, reward, info['physical_amount'] / budget,
                             nxt[env.policy_indices].astype(np.float32), done)
                obs = nxt
            mets = self.train_updates(updates_per_episode)
            out.append(dict(episode=copy.deepcopy(env.episode_records[-1]), metrics=mets))
        return out

    def evaluate(self, env, indices=(0,)):
        """Run native dev/holdout episodes without replay, PID or parameter updates."""
        if (env.chapter != 3 or env.action_family != 'total_space' or
                env.horizon != self.config.horizon or len(env.policy_indices) != self.observation_dim):
            raise ValueError('Native evaluation contract differs')
        records = []
        with self.evaluation_scope():
            for index in indices:
                self.reset_episode()
                obs, _ = env.reset(seed=int(index))
                for _ in range(env.horizon):
                    public = obs[env.policy_indices].astype(np.float32)
                    control = self.act(public, deterministic=True)
                    nxt, reward, done, truncated, info = env.step(control)
                    if truncated:
                        raise ValueError('Native evaluation cannot truncate')
                    self.observe(public, control, reward,
                                 info['physical_amount'] / float(env.spec.initial_budget),
                                 nxt[env.policy_indices].astype(np.float32), done, training=False)
                    obs = nxt
                records.append(copy.deepcopy(env.episode_records[-1]))
        return records

    def capture_runtime(self):
        """In-memory transaction state, including all randomness and train carry."""
        replay = self.replay
        return dict(
            model=self.agent.save(), rng=copy.deepcopy(self.agent.rng.bit_generator.state),
            carry=copy.deepcopy(self._carry), train_carry=copy.deepcopy(self._train_carry),
            pending=copy.deepcopy(self._pending), first=self._first,
            previous=(self._prev_reward, self._prev_cost, self._episode_cost),
            accounting=copy.deepcopy(self.accounting), pid=copy.deepcopy(self.pid.__dict__),
            counter=self.counter.save(), updates=self.agent._updates.save(),
            once=self.agent._once, metrics_clock=copy.deepcopy(self.agent._should_metrics.__dict__),
            evaluation_only=self._evaluation_only, evaluation_active=self._evaluation_active,
            replay=dict(table=replay.table.copy(), streams=copy.deepcopy(replay.streams),
                        counters=copy.deepcopy(replay.counters), metrics=copy.deepcopy(replay.metrics),
                        sampler=copy.deepcopy(replay.sampler.__dict__),
                        remover=copy.deepcopy(replay.remover.__dict__), limiter_size=replay.limiter.size),
            diagnostics=dict(planner=copy.deepcopy(self.last_planner_metrics),
                             training=copy.deepcopy(self.last_training_metrics)))

    def restore_runtime(self, state):
        self.agent.load(state['model'])
        self.agent.rng.bit_generator.state = copy.deepcopy(state['rng'])
        self._carry, self._train_carry = copy.deepcopy(state['carry']), copy.deepcopy(state['train_carry'])
        self._pending, self._first = copy.deepcopy(state['pending']), state['first']
        self._prev_reward, self._prev_cost, self._episode_cost = state['previous']
        self.accounting = copy.deepcopy(state['accounting'])
        self.pid.__dict__.clear()
        self.pid.__dict__.update(copy.deepcopy(state['pid']))
        self.counter.load(state['counter'])
        self.agent._updates.load(state['updates'])
        self.agent._once = state['once']
        self.agent._should_metrics.__dict__.update(copy.deepcopy(state['metrics_clock']))
        self._evaluation_only, self._evaluation_active = state['evaluation_only'], state['evaluation_active']
        self.last_planner_metrics = copy.deepcopy(state['diagnostics']['planner'])
        self.last_training_metrics = copy.deepcopy(state['diagnostics']['training'])
        replay, saved = self.replay, state['replay']
        replay.table = saved['table'].copy()
        replay.streams, replay.counters = copy.deepcopy(saved['streams']), copy.deepcopy(saved['counters'])
        replay.metrics = copy.deepcopy(saved['metrics'])
        replay.sampler.__dict__.clear()
        replay.sampler.__dict__.update(copy.deepcopy(saved['sampler']))
        replay.remover.__dict__.clear()
        replay.remover.__dict__.update(copy.deepcopy(saved['remover']))
        replay.limiter.size = saved['limiter_size']
        self._stream = replay.dataset()

    @contextmanager
    def evaluation_scope(self):
        state = self.capture_runtime()
        self._evaluation_active = True
        try:
            yield self
        finally:
            self.restore_runtime(state)

    @staticmethod
    def _optimizer_key(name):
        return any('/' + part + '/' in name for part in ('model_opt', 'actor_opt', 'critic_opt'))

    def state_dict(self, include_replay=False):
        if include_replay:
            raise ValueError('This scope permits evaluation models only, without disk replay')
        complete = self.agent.save()
        model = {key: value for key, value in complete.items() if not self._optimizer_key(key)}
        return dict(revision=REVISION, metadata=self.metadata, model=model,
                    optimizer_state_saved=False,
                    rng=copy.deepcopy(self.agent.rng.bit_generator.state),
                    accounting=copy.deepcopy(self.accounting),
                    pid=copy.deepcopy(self.pid.__dict__))

    def save(self, directory, include_replay=False):
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=False)
        with (target / 'evaluation.pkl').open('xb') as handle:
            pickle.dump(self.state_dict(include_replay), handle, protocol=5)
        return target / 'evaluation.pkl'

    @classmethod
    def load(cls, path, env=None, device=None, restore_training=False):
        if restore_training:
            raise ValueError('Evaluation checkpoint has no replay for exact training continuation')
        path = Path(path)
        if path.is_dir():
            path = path / 'evaluation.pkl'
        with path.open('rb') as handle:
            state = pickle.load(handle)
        if state.get('revision') != REVISION:
            raise ValueError('SafeDreamer checkpoint revision differs')
        m = state['metadata']
        obj = cls(m['observation_dim'], m['action_dim'],
                  m['effective_config']['seed'], device or m['device'],
                  SafeDreamerConfig(**m['config']))
        if any(obj.metadata[key] != m.get(key) for key in
               ('source_commit', 'vendor_manifest_sha256', 'adapter_sha256')):
            raise ValueError('SafeDreamer checkpoint source differs')
        if env is not None and (len(env.policy_indices) != obj.observation_dim or
                                env.action_dim != obj.action_dim or env.horizon != obj.config.horizon):
            raise ValueError('Evaluation environment contract differs')
        fresh = obj.agent.save()
        expected = {key for key in fresh if not cls._optimizer_key(key)}
        if set(state['model']) != expected or state.get('optimizer_state_saved') is not False:
            raise ValueError('Evaluation model has an incompatible parameter/optimizer schema')
        # Rebuild fresh Optax slots for the official variable schema, then bind
        # all stored learned state. These fresh slots are never used to resume.
        obj.agent.load({**fresh, **state['model']})
        obj.agent.rng.bit_generator.state = state['rng']
        obj.accounting = state['accounting']
        obj.pid.__dict__.update(state['pid'])
        obj._evaluation_only = True
        return obj
