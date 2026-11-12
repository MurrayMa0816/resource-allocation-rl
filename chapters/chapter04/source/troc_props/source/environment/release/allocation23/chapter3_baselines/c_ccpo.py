"""ECAI 2023 conservative CCPO, instantiated with PPO (C-PPO).

Paper-derived PyTorch implementation of Wang et al., DOI 10.3233/FAIA230566,
Eqs. 2/7/10/12 and Algorithm 1. This is NOT Yao et al.'s NeurIPS CCPO.
The paper permits PPO/TRPO/SAC backbones; this module fixes PPO, linear SPF,
and records all numerical choices in Config. No author TensorFlow code is copied.
Native task rewards and internally penalized learning rewards are kept separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from copy import deepcopy
import math
import numpy as np
import torch
from torch import nn

PAPER = 'https://doi.org/10.3233/FAIA230566'
REVISION = 'ecai2023-conservative-ccpo-full-spend-normalized-v2'


@dataclass(frozen=True)
class Config:
    hidden_sizes: tuple[int, ...] = (64, 64)
    gamma: float = 1.0
    cost_gamma: float = 1.0
    gae_lambda: float = .95
    cost_limit: float = 1.0
    weighted_cost_normalizer: float = 1.0
    clip_ratio: float = .2
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    policy_iterations: int = 80
    value_iterations: int = 80
    target_kl: float = .01
    penalty_lr: float = .01
    penalty_max: float = 1000.
    unsafe_reward: float = -1000.
    rollout_steps: int = 1024
    device: str = 'cpu'

    def validate(self):
        if self.gamma != 1. or self.cost_gamma != 1. or self.cost_limit != 1.:
            raise ValueError('Finite native task fixes gamma=gamma_c=1 and physical cost/B limit=1')
        if not math.isfinite(self.weighted_cost_normalizer) or self.weighted_cost_normalizer < 1.:
            raise ValueError('The declared weighted-cost normalizer must be finite and at least one')
        if any(x < 1 for x in self.hidden_sizes) or self.rollout_steps < 1:
            raise ValueError('Positive dimensions/rollout length required')
        if not 0 < self.clip_ratio < 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError('Invalid PPO/GAE parameters')
        if min(self.policy_iterations, self.value_iterations) < 1:
            raise ValueError('Positive optimization counts required')


def spf(z):
    """The paper's linear safety promotion function g(z)=2-z."""
    return 2. - z


def penalized_reward(native_reward, cost, safety_state, penalty, unsafe_reward=-1000.):
    """Eq.7: Saute reward minus adaptive SPF-weighted incurred cost."""
    # A fixed binary64 arithmetic guard is not a tunable safety slack.
    r_saute = native_reward if safety_state >= -64*np.finfo(float).eps else unsafe_reward
    return float(r_saute - penalty * spf(safety_state) * cost)


def _network(input_dim, output_dim, hidden):
    layers = []
    previous = input_dim
    for width in hidden:
        layers += [nn.Linear(previous, width), nn.Tanh()]
        previous = width
    layers += [nn.Linear(previous, output_dim)]
    return nn.Sequential(*layers)


class Policy(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden):
        super().__init__()
        self.mean = _network(observation_dim, action_dim, hidden)
        self.log_std = nn.Parameter(torch.full((action_dim,), -.5))

    def parameters_at(self, obs):
        return self.mean(obs), self.log_std.expand(*obs.shape[:-1], -1)

    def log_prob(self, obs, latent):
        mean, log_std = self.parameters_at(obs)
        return (-.5 * (((latent - mean) / log_std.exp()) ** 2 + 2 * log_std + math.log(2*math.pi))
                - 2 * (math.log(2) - latent - torch.nn.functional.softplus(-2*latent))).sum(-1)


class Agent:
    """Online C-PPO. Safety state is already present in the native public state.

    history_cost carries only past physical consumption, equivalently 1-R/B;
    it is checked/used for the algorithm's SPF, never a second business signal.
    All learning uses trajectories collected by this newly initialized policy.
    """
    def __init__(self, observation_dim, action_dim, config=None, seed=0):
        self.config = config or Config()
        self.config.validate()
        self.device = torch.device(self.config.device)
        self.observation_dim, self.action_dim = int(observation_dim), int(action_dim)
        cuda_devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            self.policy = Policy(observation_dim, action_dim, self.config.hidden_sizes).to(self.device)
            self.value = _network(observation_dim, 1, self.config.hidden_sizes).to(self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed) + 380031)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.config.value_lr)
        self.penalty = 0.
        self.safety_state = 1.
        self.episode_weighted_cost = 0.
        self.episode_cost = 0.
        self.steps = self.episodes = self.updates = self.policy_updates = self.value_updates = 0
        self.early_stops = 0
        self._pending = None
        self._buffer = []
        self.last_metrics = {}
        self.evaluation_only = False

    def tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def act(self, observation, history_cost=0., deterministic=False, record=False):
        obs = self.tensor(observation)
        if obs.shape != (self.observation_dim,) or not torch.isfinite(obs).all():
            raise ValueError('Invalid public policy observation')
        mean, log_std = self.policy.parameters_at(obs)
        latent = mean if deterministic else mean + log_std.exp() * torch.randn(
            mean.shape, generator=self.generator, device=self.device)
        action = torch.tanh(latent)
        if record:
            if self.evaluation_only or self._pending is not None:
                raise RuntimeError('Prediction is not a second online learning transition')
            self._pending = (np.asarray(observation, np.float32).copy(),
                latent.cpu().numpy().copy(), float(self.policy.log_prob(obs, latent)),
                float(self.value(obs).squeeze()), action.cpu().numpy().copy())
        return action.cpu().numpy().astype(np.float32, copy=False)

    def observe(self, observation, action, native_reward, cost, next_observation, done):
        if self.evaluation_only or self._pending is None:
            raise RuntimeError('Online observe requires act(record=True)')
        o, latent, logp, val, chosen = self._pending
        if not np.array_equal(np.asarray(observation, np.float32), o) or not np.array_equal(
                np.asarray(action, np.float32), chosen):
            raise ValueError('Rollout must record its actual behavior action and state')
        if not np.isfinite([native_reward, cost]).all() or cost < -1e-12:
            raise ValueError('Native reward and physical resource cost must be finite')
        cost = max(0., float(cost))
        learning_reward = penalized_reward(native_reward, cost, self.safety_state,
                                          self.penalty/self.config.weighted_cost_normalizer,
                                          self.config.unsafe_reward)
        self.episode_weighted_cost += spf(self.safety_state)*cost/self.config.weighted_cost_normalizer
        self.episode_cost += cost
        self.safety_state = (self.safety_state - cost/self.config.cost_limit) / self.config.cost_gamma
        self._buffer.append((o, latent, logp, val, learning_reward, bool(done)))
        self.steps += 1
        self._pending = None
        if done:
            self.penalty = float(np.clip(self.penalty + self.config.penalty_lr *
                (self.episode_weighted_cost - self.config.cost_limit), 0., self.config.penalty_max))
            self.episodes += 1
            self.last_metrics.update(native_reward=float(native_reward),
                episode_resource_cost=self.episode_cost,
                episode_spf_cost=self.episode_weighted_cost, penalty=self.penalty)
            self.safety_state = 1.
            self.episode_weighted_cost = self.episode_cost = 0.
        if len(self._buffer) >= self.config.rollout_steps:
            self.update(next_observation=next_observation, terminal=bool(done))
        return learning_reward

    def update(self, *, next_observation=None, terminal=True):
        if self.evaluation_only:
            raise RuntimeError('Evaluation checkpoint has no optimizer/resume contract')
        if not self._buffer:
            return {}
        if not terminal and next_observation is None:
            raise ValueError('Nonterminal rollout cuts require their next public state')
        cfg = self.config
        with torch.no_grad():
            last = 0. if terminal else float(self.value(self.tensor(next_observation)).squeeze())
        values = np.asarray([row[3] for row in self._buffer] + [last], np.float64)
        advantage = np.zeros(len(self._buffer), np.float64)
        returns = np.zeros_like(advantage)
        carry, target = 0., last
        for i in range(len(self._buffer)-1, -1, -1):
            reward, done = self._buffer[i][4:6]
            mask = 0. if done else 1.
            delta = reward + cfg.gamma*mask*values[i+1] - values[i]
            carry = delta + cfg.gamma*cfg.gae_lambda*mask*carry
            target = reward + cfg.gamma*mask*target
            advantage[i], returns[i] = carry, target
        advantage = (advantage - advantage.mean()) / max(advantage.std(), 1e-8)
        # A single structured host->device transfer per rollout (no per-row upload).
        obs = self.tensor(np.stack([row[0] for row in self._buffer]))
        latent = self.tensor(np.stack([row[1] for row in self._buffer]))
        old_logp = self.tensor(np.asarray([row[2] for row in self._buffer]))
        adv, ret = self.tensor(advantage), self.tensor(returns)
        kl, loss_p = torch.zeros((), device=self.device), torch.zeros((), device=self.device)
        for _ in range(cfg.policy_iterations):
            logp = self.policy.log_prob(obs, latent)
            ratio = (logp-old_logp).exp()
            loss_p = -torch.minimum(ratio*adv, ratio.clamp(1-cfg.clip_ratio, 1+cfg.clip_ratio)*adv).mean()
            kl = (old_logp-logp).mean()
            if float(kl.detach()) > 1.5*cfg.target_kl:
                self.early_stops += 1
                break
            self.policy_optimizer.zero_grad(set_to_none=True)
            loss_p.backward()
            self.policy_optimizer.step()
            self.policy_updates += 1
        for _ in range(cfg.value_iterations):
            loss_v = (self.value(obs).squeeze(-1)-ret).square().mean()
            self.value_optimizer.zero_grad(set_to_none=True)
            loss_v.backward()
            self.value_optimizer.step()
            self.value_updates += 1
        losses = torch.stack((loss_p.detach(), loss_v.detach(), kl.detach())).cpu().tolist()
        self.updates += 1
        self._buffer.clear()
        self.last_metrics.update(policy_loss=losses[0], value_loss=losses[1], kl=losses[2],
            updates=self.updates, policy_updates=self.policy_updates, value_updates=self.value_updates,
            early_stops=self.early_stops)
        if not all(math.isfinite(x) for x in losses):
            raise FloatingPointError('Nonfinite PPO update')
        return dict(self.last_metrics)

    def predict(self, observation, deterministic=True):
        return self.act(observation, deterministic=deterministic)

    def metadata(self):
        return dict(revision=REVISION, paper=PAPER, implementation='paper-derived C-PPO',
            online_from_scratch=True, pretrained_data=False, config=asdict(self.config),
            observation_dim=self.observation_dim, action_dim=self.action_dim,
            reward='native terminal reward; Eq7 learning penalty logged separately',
            safety='SPF cost normalized by its finite-horizon full-spend bound; common native decoder enforces the hard budget',
            counters=dict(steps=self.steps, episodes=self.episodes, updates=self.updates,
                policy_updates=self.policy_updates, value_updates=self.value_updates, early_stops=self.early_stops))

    def state_dict(self, full=False):
        if full and self.evaluation_only:
            raise ValueError("An evaluation-only model cannot be relabelled resumable")
        result = dict(metadata=self.metadata(), policy=deepcopy(self.policy.state_dict()),
                      value=deepcopy(self.value.state_dict()), penalty=self.penalty,
                      evaluation_only=not full, last_metrics=deepcopy(self.last_metrics))
        if full:
            result['learning_state'] = deepcopy(dict(
                policy_optimizer=self.policy_optimizer.state_dict(), value_optimizer=self.value_optimizer.state_dict(),
                generator=self.generator.get_state(), buffer=self._buffer, pending=self._pending,
                safety_state=self.safety_state, episode_weighted_cost=self.episode_weighted_cost,
                episode_cost=self.episode_cost, steps=self.steps, episodes=self.episodes, updates=self.updates,
                policy_updates=self.policy_updates, value_updates=self.value_updates,
                early_stops=self.early_stops, last_metrics=self.last_metrics))
        return result

    def load_state_dict(self, state, full=False):
        meta = state['metadata']
        if full and (state.get('evaluation_only', True) or 'learning_state' not in state):
            raise ValueError('A final evaluation model cannot exactly resume online training')
        if (meta['revision'] != REVISION or meta['observation_dim'] != self.observation_dim or
                meta['action_dim'] != self.action_dim):
            raise ValueError('Checkpoint schema mismatch')
        expected = asdict(self.config)
        actual = dict(meta['config'])
        actual['hidden_sizes'] = tuple(actual['hidden_sizes'])
        if actual != expected:
            raise ValueError('Checkpoint config mismatch (including real execution device)')
        self.policy.load_state_dict(state['policy'])
        self.value.load_state_dict(state['value'])
        self.penalty = float(state['penalty'])
        self.evaluation_only = not full
        for name, value in meta['counters'].items():
            setattr(self, name, int(value))
        self.last_metrics = deepcopy(state.get('last_metrics', {}))
        if full:
            l = state['learning_state']
            self.policy_optimizer.load_state_dict(l['policy_optimizer'])
            self.value_optimizer.load_state_dict(l['value_optimizer'])
            self.generator.set_state(l['generator'].cpu())
            for name in ('buffer','pending'):
                setattr(self, '_'+name, deepcopy(l[name]))
            for name in ('safety_state','episode_weighted_cost','episode_cost','steps','episodes','updates',
                         'policy_updates','value_updates','early_stops','last_metrics'):
                setattr(self, name, deepcopy(l[name]))
        return self

    def save(self, path, full=False):
        torch.save(self.state_dict(full=full), Path(path))
