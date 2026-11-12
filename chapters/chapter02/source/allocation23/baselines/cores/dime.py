"""Complete C3 BRO, DIME and simplex PASPO task adapters.

BRO and DIME are PyTorch ports of the pinned author implementations cited in
SOURCES, retaining their algorithmic components, rather than renamed SAC.
PASPO is independently implemented from the paper; no unlicensed author code
is copied. The fixed simplex admits exact interval and initialization formulas.
See docs/EXPANDED_CH3_IMPLEMENTATION.rst for task adaptations and limitations.

Portions derived from BRO and DIME are licensed under the MIT License:
Copyright (c) 2024 naumix
Copyright (c) 2025 DIME authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .recent_replay import ReplayBuffer
SOURCES={"dime_full": {"repository":"https://github.com/ALRhub/DIME", "commit":"7082bb7e8ab0e7f8a036a985333541d08d9d3e7d"}}

def _device(device):
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and device.index not in (None, 0)):
        raise ValueError('expanded C3 baselines permit cpu or cuda:0 only')
    return torch.device('cuda:0') if device.type == 'cuda' else device


def expanded_ch3_parameters(method, *, tiny=False, overrides=None, n_envs=1,horizon=81):
    if method not in SOURCES:
        raise ValueError('unknown expanded C3 method: ' + method)
    if type(n_envs) is not int or n_envs < 1:
        raise ValueError('n_envs must be a positive integer')
    p = dict(gamma=1., learning_rate=3e-4, batch_size=16 if tiny else 256)
    if method in ('bro_full', 'dime_full'):
        p.update(buffer_size=10000 if tiny else 200000, learning_starts=horizon if tiny else 16*horizon)
    if method == 'bro_full':
        p.update(actor_width=32 if tiny else 256, critic_width=32 if tiny else 512,
            n_quantiles=8 if tiny else 100, updates_per_step=10, tau=.005,
            adjustment_lr=3e-5, weight_decay=1e-4, initial_temperature=1.,
            initial_optimism=1., initial_regularizer=.25, kl_target=.05,
            std_multiplier=.75, pessimism=0.,
            reset_steps=[15001, 50001, 250001, 500001, 750001, 1000001, 1500001, 2000001])
    elif method == 'dime_full':
        p.update(actor_width=32 if tiny else 256, actor_layers=3,
            critic_width=32 if tiny else 2048, n_atoms=21 if tiny else 101,
            updates_per_step=2, policy_delay=3, diffusion_steps=4 if tiny else 16,
            prior_std=2.5, initial_dt=.1, initial_friction=1.,
            v_min=-200., v_max=200., critic_entropy_coef=.005,
            bn_momentum=.99, bn_warmup=100000, initial_temperature=1.,
            temperature_lr=1e-3, target_run_cost_per_dimension=4., actor_grad_clip=1.)
    else:
        p.update(learning_rate=1e-3, hidden_width=16 if tiny else 32,
            branch_width=16 if tiny else 32, epochs=1 if tiny else 10,
            batch_size=27 if tiny else 64, rollout_episodes=1 if tiny else 16,
            clip_range=.3, entropy_coef=.01, value_coef=.5, max_grad_norm=2.,
            gae_lambda=.95, weight_decay=.001, adam_beta1=.99, adam_beta2=.99)
    overrides = copy.deepcopy(dict(overrides or {}))
    if set(overrides) - set(p):
        raise ValueError('unknown expanded C3 parameters: ' + ', '.join(sorted(set(overrides) - set(p))))
    p.update(overrides)
    if p['gamma'] != 1.:
        raise ValueError('terminal-yield task requires gamma=1')
    positive_ints = ('batch_size', 'buffer_size', 'actor_width', 'actor_layers', 'critic_width',
        'n_quantiles', 'n_atoms', 'updates_per_step', 'policy_delay', 'diffusion_steps',
        'hidden_width', 'branch_width', 'epochs', 'rollout_episodes')
    for name in positive_ints:
        if name in p and (type(p[name]) is not int or p[name] < 1):
            raise ValueError(name + ' must be a positive integer')
    for name in ('learning_starts', 'bn_warmup'):
        if name in p and (type(p[name]) is not int or p[name] < 0):
            raise ValueError(name + ' must be a nonnegative integer')
    for name, val in p.items():
        if name == 'reset_steps':
            if not isinstance(val, (list, tuple)) or any(type(v) is not int or v < 1 for v in val) or list(val) != sorted(set(val)):
                raise ValueError('reset_steps must be distinct increasing positive integers')
        elif not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(name + ' must be finite')
    for name in ('learning_rate', 'adjustment_lr', 'initial_temperature', 'initial_optimism',
                 'initial_regularizer', 'kl_target', 'std_multiplier', 'prior_std', 'initial_dt',
                 'initial_friction', 'temperature_lr', 'actor_grad_clip', 'max_grad_norm'):
        if name in p and p[name] <= 0:
            raise ValueError(name + ' must be positive')
    for name in ('weight_decay', 'pessimism', 'entropy_coef', 'value_coef', 'critic_entropy_coef'):
        if name in p and p[name] < 0:
            raise ValueError(name + ' must be nonnegative')
    for name in ('tau', 'clip_range', 'bn_momentum', 'adam_beta1', 'adam_beta2'):
        if name in p and not 0 < p[name] < 1:
            raise ValueError(name + ' must be between zero and one')
    if method == 'bro_full' and p['updates_per_step'] not in (2, 10):
        raise ValueError('BRO uses UTD=10 or explicitly labelled BRO Fast UTD=2')
    if method == 'bro_full' and p['updates_per_step'] == 2 and 'reset_steps' not in overrides:
        p['reset_steps'] = p['reset_steps'][:1]
    if method == 'dime_full' and (p['n_atoms'] < 2 or p['v_min'] >= p['v_max']):
        raise ValueError('invalid categorical support')
    if method == 'paspo_full' and not 0 <= p['gae_lambda'] <= 1:
        raise ValueError('invalid GAE lambda')
    if 'buffer_size' in p and p['buffer_size'] < max(p['learning_starts'], p['batch_size'], horizon):
        raise ValueError('replay capacity must cover warmup and batch')
    return p


class _CoreState:
    def state_dict(self):
        result = dict(model=copy.deepcopy(self.model.state_dict()),
            optimizers={k: copy.deepcopy(o.state_dict()) for k, o in self.optimizers.items()},
            training_steps=self.training_steps, actor_updates=self.actor_updates,
            critic_updates=self.critic_updates, last_losses=copy.deepcopy(self.last_losses))
        if hasattr(self, 'replay_buffer'):
            result['replay_buffer'] = self.replay_buffer.state_dict()
        if hasattr(self, 'reset_count'):
            result.update(reset_count=self.reset_count, last_env_step=self.last_env_step)
        return result

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        for name, value in state['optimizers'].items():
            self.optimizers[name].load_state_dict(value)
        for name in ('training_steps', 'actor_updates', 'critic_updates', 'last_losses', 'reset_count', 'last_env_step'):
            if name in state:
                setattr(self, name, copy.deepcopy(state[name]))
        if 'replay_buffer' in state:
            self.replay_buffer.load_state_dict(state['replay_buffer'])

    def _batch(self, batch):
        if batch is None:
            return self.replay_buffer.sample()
        if isinstance(batch, dict):
            batch = (batch['obs'], batch['action'], batch['next_obs'], batch['reward'],
                     1 - torch.as_tensor(batch['terminated'], dtype=torch.float32, device=self.device))
        out = tuple(torch.as_tensor(v, dtype=torch.float32, device=self.device) for v in batch)
        return (*out[:3], out[3].reshape(-1, 1), out[4].reshape(-1, 1))


class BatchRenorm(nn.Module):
    """Author DIME/CrossQ renormalization, including warmup and joint batches."""
    def __init__(self, size, momentum=.99, warmup=100000, eps=.001):
        super().__init__()
        self.weight, self.bias = nn.Parameter(torch.ones(size)), nn.Parameter(torch.zeros(size))
        self.register_buffer('mean', torch.zeros(size)); self.register_buffer('variance', torch.ones(size))
        self.register_buffer('steps', torch.zeros((), dtype=torch.int64))
        self.momentum, self.warmup, self.eps = momentum, warmup, eps

    def forward(self, x, train=False):
        if train:
            mean, var = x.mean(0), x.var(0, unbiased=False)
            # Corrections use the previous running statistics, and are constants
            # for differentiation. Match the author's epsilon placement.
            old_mean, old_var = self.mean.clone(), self.variance.clone()
            r = ((var + self.eps).sqrt() / (old_var + self.eps).sqrt()).detach().clamp(1 / 3, 3)
            d = ((mean - old_mean) / (old_var + self.eps).sqrt()).detach().clamp(-5, 5)
            warmed = (self.steps >= self.warmup).to(x.dtype)
            used_var = warmed * (var / r.square()) + (1 - warmed) * var
            # Constant feature columns are common in the public observation.
            # Stabilize sqrt's derivative at zero even during BRN warmup.
            used_mean = warmed * (mean - d * var.clamp_min(torch.finfo(var.dtype).tiny).sqrt() / r) + (1 - warmed) * mean
            with torch.no_grad():
                self.mean.lerp_(mean.detach(), 1 - self.momentum)
                self.variance.lerp_(var.detach(), 1 - self.momentum); self.steps.add_(1)
        else:
            used_mean, used_var = self.mean, self.variance
        return (x - used_mean) / (used_var + self.eps).sqrt() * self.weight + self.bias


class DIMEValue(nn.Module):
    def __init__(self, obs_dim, action_dim, p):
        super().__init__()
        width = p['critic_width']
        self.layers = nn.ModuleList([nn.Linear(obs_dim + action_dim, width), nn.Linear(width, width)])
        self.norms = nn.ModuleList(BatchRenorm(n, p['bn_momentum'], p['bn_warmup']) for n in (obs_dim + action_dim, width, width))
        self.output = nn.Linear(width, p['n_atoms'])
        # Flax Dense default is fan-in truncated normal (variance corrected).
        for layer in (*self.layers, self.output):
            nn.init.trunc_normal_(layer.weight, std=math.sqrt(1 / layer.in_features) / .8796256610342398,
                a=-2 * math.sqrt(1 / layer.in_features) / .8796256610342398,
                b=2 * math.sqrt(1 / layer.in_features) / .8796256610342398)
            nn.init.zeros_(layer.bias)

    def forward(self, obs, action, train=False):
        x = self.norms[0](torch.cat((obs, action), -1), train)
        for layer, norm in zip(self.layers, self.norms[1:]):
            x = norm(F.relu(layer(x)), train)
        return self.output(x)


class DiffusionActor(nn.Module):
    """DIME DIS sampler: learned controlled drift, dt and per-axis friction."""
    def __init__(self, obs_dim, action_dim, p):
        super().__init__()
        self.action_dim, self.p = action_dim, p
        width = p['actor_width']
        self.phase = nn.Parameter(torch.zeros(1, width))
        self.register_buffer('frequency', torch.linspace(.1, 100, width).reshape(1, width))
        self.time_encoder = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(approximate='tanh'), nn.Linear(width, width))
        layers = []
        for i in range(p['actor_layers']):
            layers.extend([nn.Linear(obs_dim + action_dim + width if i == 0 else width, width), nn.GELU(approximate='tanh')])
        layers.append(nn.Linear(width, action_dim))
        self.drift = nn.Sequential(*layers)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                std = math.sqrt(1 / layer.in_features) / .8796256610342398
                nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std); nn.init.zeros_(layer.bias)
        nn.init.constant_(self.drift[-1].weight, 1e-8)
        self.raw_dt = nn.Parameter(torch.tensor(math.log(math.expm1(p['initial_dt']))))
        self.raw_friction = nn.Parameter(torch.full((action_dim,), math.log(math.expm1(p['initial_friction']))))
        steps = torch.arange(p['diffusion_steps'], dtype=torch.float32)
        t = (p['diffusion_steps'] - steps) / p['diffusion_steps']
        self.register_buffer('schedule', .999 * torch.cos(.5 * math.pi * (1.008 - t) / 1.008).square() + .001)

    def _time_codes(self):
        # Every diffusion step has a distinct code, but its MLP is independent
        # of the state/action chain. Batch this MLP once per sampler call.
        # No-grad calls may reuse it until any producing parameter changes;
        # the cache is derived data and is deliberately absent from checkpoints.
        parameters = (self.phase, *self.time_encoder.parameters())
        key = (str(self.phase.device), self.phase.dtype,
               tuple((id(p), p._version) for p in parameters))
        if not torch.is_grad_enabled() and getattr(self, '_time_cache_key', None) == key:
            return self._time_cache
        steps = torch.arange(self.p['diffusion_steps'], device=self.phase.device,
                             dtype=self.phase.dtype)[:, None]
        angles = steps * self.frequency + self.phase
        codes = self.time_encoder(torch.cat((angles.sin(), angles.cos()), -1))
        if not torch.is_grad_enabled():
            self._time_cache_key, self._time_cache = key, codes
        return codes

    def sample_action(self, obs, deterministic=False, noises=None):
        # Deployment needs the exact diffusion sample, not its training-bound
        # density. Preserve every drift step and every random draw.
        return self(obs, deterministic, noises, compute_cost=False)[0]

    def forward(self, obs, deterministic=False, noises=None, *, compute_cost=True):
        batch = len(obs)
        if noises is None:
            noises = (torch.zeros if deterministic else torch.randn)(
                (self.p['diffusion_steps'] + 1, batch, self.action_dim), device=obs.device, dtype=obs.dtype)
        if tuple(noises.shape) != (self.p['diffusion_steps'] + 1, batch, self.action_dim):
            raise ValueError('diffusion noise shape mismatch')
        x = noises[0] * self.p['prior_std']
        if compute_cost:
            terminal = (-.5 * noises[0].square() - math.log(self.p['prior_std']) - .5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
            log_ratio = torch.zeros((batch, 1), device=obs.device, dtype=obs.dtype)
        dt = F.softplus(self.raw_dt)
        friction = F.softplus(self.raw_friction)
        time_codes = self._time_codes()
        for step in range(self.p['diffusion_steps']):
            time_code = time_codes[step:step + 1].expand(batch, -1)
            control = self.drift(torch.cat((x, obs, time_code), -1)).clamp(-1e4, 1e4)
            eta = dt * self.schedule[step] / friction
            scale = (2 * eta).sqrt()
            fwd_mean = x + eta * (-x / self.p['prior_std'] ** 2 + control)
            new = fwd_mean + scale * noises[step + 1]
            if compute_cost:
                bwd_mean = new - eta * new / self.p['prior_std'] ** 2
                # Exact Gaussian transition log-density ratio with gradients
                # through the whole chain whenever a training cost is needed.
                fwd_log = (-.5 * ((new - fwd_mean) / scale).square() - scale.log() - .5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
                bwd_log = (-.5 * ((x - bwd_mean) / scale).square() - scale.log() - .5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
                log_ratio = log_ratio + bwd_log - fwd_log
            x = new
        if not compute_cost:
            return x.tanh(), None, None
        log_jacobian = (2 * (math.log(2) - x - F.softplus(-2 * x))).sum(-1, keepdim=True)
        running = -(log_ratio + log_jacobian)
        return x.tanh(), running + terminal, running


def categorical_projection(probabilities, rewards, alive, cost, alpha, support):
    """C51 projection, conserving point masses including exact atom hits."""
    step = (support[-1] - support[0]) / (len(support) - 1)
    targets = (rewards + alive * (support[None] - alpha * cost)).clamp(support[0], support[-1])
    b = ((targets - support[0]) / step).clamp(0, len(support) - 1)
    lower, upper = b.floor().long(), b.ceil().long()
    out = torch.zeros_like(probabilities)
    same = lower == upper
    out.scatter_add_(1, lower, probabilities * (upper.to(b.dtype) - b + same.to(b.dtype)))
    out.scatter_add_(1, upper, probabilities * (b - lower.to(b.dtype)))
    return out, targets


class DIMENetworks(nn.Module):
    def __init__(self, obs_dim, action_dim, p):
        super().__init__()
        self.actor = DiffusionActor(obs_dim, action_dim, p)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critics = nn.ModuleList(DIMEValue(obs_dim, action_dim, p) for _ in range(2))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(p['initial_temperature'])))
        self.register_buffer('support', torch.linspace(p['v_min'], p['v_max'], p['n_atoms']))


class DIMECore(_CoreState):
    def __init__(self, obs_dim, action_dim, p, device='cpu', seed=0):
        self.obs_dim, self.action_dim, self.p, self.device = obs_dim, action_dim, p, _device(device)
        torch.manual_seed(seed)
        self.model = DIMENetworks(obs_dim, action_dim, p).to(self.device)
        self.optimizers = dict(critic=torch.optim.Adam(self.model.critics.parameters(), lr=p['learning_rate'], betas=(.5, .999)),
            actor=torch.optim.Adam(self.model.actor.parameters(), lr=p['learning_rate'], betas=(.5, .999)),
            temperature=torch.optim.Adam([self.model.log_temperature], lr=p['temperature_lr']))
        self.replay_buffer = ReplayBuffer(obs_dim, action_dim, self.device, p['buffer_size'], p['batch_size'])
        self.training_steps = self.actor_updates = self.critic_updates = 0
        self.last_losses = {}

    def sample(self, obs, deterministic=False, exploration=False):
        return self.model.actor.sample_action(obs, deterministic)

    def update(self, batch=None):
        obs, action, nxt, rewards, alive = self._batch(batch)
        m, p = self.model, self.p
        alpha = m.log_temperature.exp().detach()
        with torch.no_grad():
            na, next_cost, _ = m.actor_target(nxt)
        joined_obs, joined_actions = torch.cat((obs, nxt)), torch.cat((action, na))
        logits = [q(joined_obs, joined_actions, train=True) for q in m.critics]
        current = [x[:len(obs)] for x in logits]
        with torch.no_grad():
            next_probs = torch.stack([x[len(obs):].softmax(-1) for x in logits]).mean(0)
            target, atoms = categorical_projection(next_probs, rewards, alive, next_cost, alpha, m.support)
            raw_atoms = rewards + alive * (m.support[None] - alpha * next_cost)
            clipped_atoms = (raw_atoms < m.support[0]) | (raw_atoms > m.support[-1])
        qloss = sum(-(target * x.log_softmax(-1)).sum(-1).mean()
            + p['critic_entropy_coef'] * (x.softmax(-1) * x.log_softmax(-1)).sum(-1).mean() for x in current)
        self.optimizers['critic'].zero_grad(set_to_none=True); qloss.backward(); self.optimizers['critic'].step()
        self.training_steps += 1; self.critic_updates += 1
        losses = dict(critic=qloss.detach(), temperature=alpha,
            target_mean=(target * m.support).sum(-1).mean(),
            support_edge_mass=(target[:, 0] + target[:, -1]).mean(),
            target_clipped_atom_fraction=clipped_atoms.float().mean(),
            target_clipped_probability_mass=(next_probs * clipped_atoms).sum(-1).mean())
        if self.training_steps % p['policy_delay'] == 0:
            m.critics.requires_grad_(False)
            try:
                sampled, cost, running = m.actor(obs)
                qs = [(q(obs, sampled, train=False).softmax(-1) * m.support).sum(-1, keepdim=True) for q in m.critics]
                aloss = (-torch.stack(qs).mean(0) + alpha * cost).mean()
                self.optimizers['actor'].zero_grad(set_to_none=True); aloss.backward()
                # Author uses optax.clip: coordinatewise clipping, not norm clipping.
                nn.utils.clip_grad_value_(m.actor.parameters(), p['actor_grad_clip'])
                self.optimizers['actor'].step()
            finally:
                m.critics.requires_grad_(True)
            # Author implementation tunes RUNNING cost, not total path cost.
            tloss = -m.log_temperature.exp() * (running.detach().mean() - p['target_run_cost_per_dimension'] * self.action_dim)
            self.optimizers['temperature'].zero_grad(set_to_none=True); tloss.backward(); self.optimizers['temperature'].step()
            m.actor_target.load_state_dict(m.actor.state_dict())
            self.actor_updates += 1
            losses.update(actor=aloss.detach(), run_cost=running.detach().mean(), path_cost=cost.detach().mean())
        self.last_losses = losses
        return losses

    train = update

