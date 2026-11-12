"""Compact vector-SAC primitives for the author-code based MORL adapters.

Networks/losses follow Q-Pensieve (ICLR 2023, MIT).  Removed dependencies are
MuJoCo, rltorch, plotting, Visdom and disk replay; their numerical model roles
are retained with ordinary torch modules.  Actions remain tanh-Gaussian.
"""
from __future__ import annotations

import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs, outputs, hidden=256, activation=nn.ReLU):
    return nn.Sequential(nn.Linear(inputs, hidden), activation(), nn.Linear(hidden, hidden),
                         activation(), nn.Linear(hidden, outputs))


def as_tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def finite(value, name):
    # CPU numpy views share the same storage without reading through the much
    # larger torch isfinite/abs/compare/reduction graph for every gradient.
    # This changes only validation, never values, autograd, or RNG.  Retain the
    # native path for devices/layouts/dtypes which do not expose a numpy view.
    if (value.device.type == 'cpu' and value.layout == torch.strided
            and value.dtype in (torch.float32, torch.float64)):
        valid = np.isfinite(value.detach().numpy()).all()
    else:
        valid = torch.isfinite(value).all()
    if not valid:
        raise FloatingPointError(name)


def freeze(module, enabled=True):
    for parameter in module.parameters():
        parameter.requires_grad_(not enabled)


@torch.no_grad()
def polyak(target, source, coefficient):
    for target_parameter, parameter in zip(target.parameters(), source.parameters()):
        target_parameter.lerp_(parameter, coefficient)


class GaussianActor(nn.Module):
    def __init__(self, inputs, actions, hidden=256):
        super().__init__()
        self.net = mlp(inputs, 2*actions, hidden)

    def sample(self, inputs, generator=None, deterministic=False):
        mean, log_std = self.net(inputs).chunk(2, dim=-1)
        log_std = log_std.clamp(-20., 2.)
        raw = mean if deterministic else mean + log_std.exp()*torch.randn(
            mean.shape, device=mean.device, generator=generator)
        action = raw.tanh()
        log_prob = (-.5*((raw-mean)/log_std.exp()).square()-log_std-.5*math.log(2*math.pi)
                    -torch.log(1-action.square()+1e-6)).sum(-1, keepdim=True)
        return action, log_prob

    def actions(self, inputs, generator=None, deterministic=False):
        """Inference needs no log-density; keep the same forward/draw/tanh."""
        mean, log_std = self.net(inputs).chunk(2, dim=-1)
        log_std = log_std.clamp(-20., 2.)
        raw = mean if deterministic else mean + log_std.exp()*torch.randn(
            mean.shape, device=mean.device, generator=generator)
        return raw.tanh()


class VectorCritic(nn.Module):
    def __init__(self, inputs, hidden=256):
        super().__init__()
        self.q1 = mlp(inputs, 2, hidden)
        self.q2 = mlp(inputs, 2, hidden)

    def forward(self, inputs):
        return self.q1(inputs), self.q2(inputs)


def conservative_vector(q1, q2, preferences):
    """Select one entire vector by its scalarized value, not coordinate minima."""
    choose_first = (q1*preferences).sum(-1, keepdim=True) < (q2*preferences).sum(-1, keepdim=True)
    return torch.where(choose_first, q1, q2)


class SACBase:
    on_policy = False
    name = 'vector_sac_base'

    def __init__(self, obs_dim, action_dim, seed=0, device='cpu', config=None):
        self.config = dict(config or {})
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.hidden = int(self.config.get('hidden_dim', 256))
        self.network_initializer = self.config.get('network_initializer', 'legacy_torch_default')
        if self.network_initializer not in ('legacy_torch_default', 'author_xavier_uniform'):
            raise ValueError('Unknown network_initializer')
        self.gamma = float(self.config.get('gamma', 1.))
        self.tau = float(self.config.get('tau', .005))
        self.lr = float(self.config.get('lr', 3e-4))
        self.entropy_tuning = bool(self.config.get('entropy_tuning', True))
        initial_alpha = float(self.config.get('alpha', .2))
        if initial_alpha <= 0:
            raise ValueError('Positive entropy coefficient required')
        self.rng = np.random.default_rng(self.seed+15451)
        self.action_generator = torch.Generator(device=self.device).manual_seed(self.seed+15452)
        self.learn_generator = torch.Generator(device=self.device).manual_seed(self.seed+15453)
        grid = self.config.get('preference_grid', [0., .5, .8, 1.])
        grid = np.asarray(grid, dtype=float)
        self.preference_grid = np.stack([grid, 1-grid], -1) if grid.ndim == 1 else grid
        if self.preference_grid.ndim != 2 or self.preference_grid.shape[1] != 2:
            raise ValueError('Two-objective preferences required')
        if (self.preference_grid < 0).any() or not np.allclose(self.preference_grid.sum(1), 1):
            raise ValueError('Simplex preferences required')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.actor = GaussianActor(self.obs_dim+2, self.action_dim, self.hidden).to(self.device)
            self.critic = VectorCritic(self.obs_dim+self.action_dim+2, self.hidden).to(self.device)
            if self.network_initializer == 'author_xavier_uniform':
                def initialize(layer):
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
                self.actor.apply(initialize)
                self.critic.apply(initialize)
        self.critic_target = copy.deepcopy(self.critic)
        freeze(self.critic_target)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.log_alpha = torch.tensor(math.log(initial_alpha), device=self.device, requires_grad=self.entropy_tuning)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr) if self.entropy_tuning else None
        self.target_entropy = -float(self.action_dim)
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def random_preference(self, exclude=None):
        choices = np.arange(len(self.preference_grid))
        if exclude is not None and len(choices) > 1:
            excluded = np.asarray(exclude)
            grid = self.preference_grid
            if not grid.dtype.hasobject and not excluded.dtype.hasobject:
                # Preserve exact np.isclose acceptance and candidate order.
                # Reusing choices removes the repeated broadcast/reduction;
                # an in-place grid edit invalidates this private derived cache.
                fingerprint = (grid.dtype.str, grid.shape, grid.tobytes())
                if getattr(self, '_preference_cache_grid', None) != fingerprint:
                    self._preference_cache_grid = fingerprint
                    self._preference_exclusions = {}
                key = (excluded.dtype.str, excluded.shape, excluded.tobytes())
                cached = self._preference_exclusions.get(key)
                if cached is None:
                    keep = ~np.isclose(grid, excluded).all(1)
                    cached = choices[keep]
                    self._preference_exclusions[key] = cached
                choices = cached
            else:
                keep = ~np.isclose(grid, excluded).all(1)
                choices = choices[keep]
        return self.preference_grid[int(self.rng.choice(choices))].copy()

    def batch(self, value):
        data = {k: as_tensor(value[k], self.device) for k in
                ('obs', 'actions', 'rewards', 'next_obs', 'dones', 'weights')}
        batch_size = len(data['obs'])
        assert data['obs'].shape == data['next_obs'].shape == (batch_size, self.obs_dim)
        assert data['actions'].shape == (batch_size, self.action_dim)
        assert data['rewards'].shape == data['weights'].shape == (batch_size, 2)
        data['dones'] = data['dones'].reshape(batch_size, 1)
        for key, item in data.items():
            finite(item, key)
        return data

    @torch.no_grad()
    def act(self, obs, prefs, deterministic=False):
        obs, prefs = as_tensor(obs, self.device), as_tensor(prefs, self.device)
        if obs.ndim == 1:
            obs = obs[None]
        if prefs.ndim == 1:
            prefs = prefs[None].expand(len(obs), -1)
        assert obs.shape[1] == self.obs_dim and prefs.shape == (len(obs), 2)
        actions = self.actor.actions(torch.cat([obs, prefs], -1), self.action_generator, deterministic)
        finite(actions, 'inference action')
        return actions.cpu().numpy()

    def critic_values(self, obs, actions, preferences, critic=None):
        return (self.critic if critic is None else critic)(torch.cat([obs, actions, preferences], -1))

    def policy_value(self, obs, preferences, actions):
        q1, q2 = self.critic_values(obs, actions, preferences)
        return torch.minimum((q1*preferences).sum(-1), (q2*preferences).sum(-1))

    def after_update(self):
        return {}

    def update(self, batch):
        batch = self.batch(batch)
        states, next_states, actions = batch['obs'], batch['next_obs'], batch['actions']
        preference = as_tensor(self.random_preference(), self.device)[None].expand(len(states), -1)
        with torch.no_grad():
            next_actions, next_logp = self.actor.sample(torch.cat([next_states, preference], -1), self.learn_generator)
            tq1, tq2 = self.critic_values(next_states, next_actions, preference, self.critic_target)
            target = batch['rewards'] + (1-batch['dones'])*self.gamma*(
                conservative_vector(tq1, tq2, preference)-self.alpha*next_logp)
        q1, q2 = self.critic_values(states, actions, preference)
        q1_loss = ((q1-target).square()*preference).sum(-1).mean()
        q2_loss = ((q2-target).square()*preference).sum(-1).mean()
        critic_loss = q1_loss+q2_loss
        finite(critic_loss, 'critic loss')
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        for parameter in self.critic.parameters():
            if parameter.grad is not None:
                finite(parameter.grad, 'critic gradient')
        self.critic_optimizer.step()
        freeze(self.critic)
        sampled, logp = self.actor.sample(torch.cat([states, preference], -1), self.learn_generator)
        q = self.policy_value(states, preference, sampled)
        actor_loss = (self.alpha*logp.squeeze(-1)-q).mean()
        finite(actor_loss, 'actor loss')
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        for parameter in self.actor.parameters():
            if parameter.grad is not None:
                finite(parameter.grad, 'actor gradient')
        self.actor_optimizer.step()
        freeze(self.critic, False)
        alpha_loss = self.update_alpha(logp)
        polyak(self.critic_target, self.critic, self.tau)
        self.updates += 1
        extra = self.after_update()
        return dict(critic_loss=float(critic_loss.detach()), actor_loss=float(actor_loss.detach()),
                    alpha=float(self.alpha), alpha_loss=alpha_loss, updates=self.updates,
                    mechanism=self.name, **extra)

    def update_alpha(self, logp):
        if not self.entropy_tuning:
            return 0.
        alpha_loss = -(self.log_alpha*(logp.detach()+self.target_entropy)).mean()
        finite(alpha_loss, 'entropy loss')
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward(); self.alpha_optimizer.step()
        return float(alpha_loss.detach())

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    critic_target=self.critic_target.state_dict(), log_alpha=self.log_alpha.detach().cpu(),
                    updates=self.updates, metadata=self.metadata())

    def load_state_dict(self, payload):
        self.actor.load_state_dict(payload['actor']); self.critic.load_state_dict(payload['critic'])
        self.critic_target.load_state_dict(payload.get('critic_target', payload['critic']))
        with torch.no_grad():
            self.log_alpha.copy_(payload['log_alpha'].to(self.device))
        self.updates = int(payload.get('updates', 0))

    def metadata(self):
        return dict(method=self.name, on_policy=False, backbone='vector soft actor critic',
                    gamma=self.gamma, tau=self.tau, learning_rate=self.lr, hidden_dim=self.hidden, network_initializer=self.network_initializer,
                    dtype='float32', entropy_tuning=self.entropy_tuning,
                    initial_alpha=float(self.config.get('alpha', .2)),
                    preference_grid=self.preference_grid.tolist(),
                    action_distribution='tanh-Gaussian in common latent action space',
                    checkpoint='weights and mechanism counters only; optimizer/replay excluded',
                    adaptations=['batched fixed-API observation/latent actions',
                        'finite-horizon gamma is configured explicitly',
                        'ordinary torch MLP replaces rltorch builder',
                        'configured initial alpha is honored; no ignored ent_coef argument',
                        'consolidated SAC update order avoids legacy in-place autograd conflicts'])
