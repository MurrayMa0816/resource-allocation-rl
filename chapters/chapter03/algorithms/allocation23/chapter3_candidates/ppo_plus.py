"""PPO+ (RLJ/RLC 2026) PyTorch task adapter, MIT-author-core derivative.

Reference: kallel-mahdi/PPO-Plus, adfc6c5b7665f24fa5269c2c108046e2aaf7e6a5.
Preserves replay-only twin critic updates, new-rollout PPO actor, soft GAE,
state-dependent Gaussian variance, terminal actor mask, adaptive temperature,
and independent Q targets (no minimum and no target network by default).
The adapter owns no environment and only observes actual public transitions.
Optional shared_objects mean is a disclosed regional task adaptation; flat
default keeps the original port parameter initialization, updates and RNG.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DEFAULTS = dict(hidden_dims=256, use_layer_norm=True, final_fc_init_scale=.01,
                actor_lr=3e-4, critic_lr=3e-4, temp_lr=3e-4,
                clip_grad_norm=.5, clipping_ratio=.25, gae_lambda=.5,
                entropy_coeff=.5, temperature=1., discount=1.,
                min_target=False, policy_steps=2048, num_epochs=10,
                batch_size=256, buffer_size=50000, value_samples=10,
                reward_scale=100., terminal_actor_mask=True,
                mean_architecture='flat')


def _linear(n, m, gain=math.sqrt(2)):
    layer = nn.Linear(n, m)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Backbone(nn.Module):
    def __init__(self, n, width, norm):
        super().__init__()
        self.net = nn.Sequential(_linear(n, width),
                                 nn.LayerNorm(width, eps=1e-6) if norm else nn.Identity(),
                                 nn.Tanh(), _linear(width, width),
                                 nn.LayerNorm(width, eps=1e-6) if norm else nn.Identity(),
                                 nn.Tanh())

    def forward(self, x):
        return self.net(x)


def _shared_mean_class():
    # Local editable dependency; the imported source identity is in provenance.json.
    from .shared_mean import SharedObjectMean
    return SharedObjectMean


class Actor(nn.Module):
    def __init__(self, d, a, c):
        super().__init__()
        self.backbone = Backbone(d, c['hidden_dims'], c['use_layer_norm'])
        self.mean = _linear(c['hidden_dims'], a, c['final_fc_init_scale'])
        self.log_std = _linear(c['hidden_dims'], a, c['final_fc_init_scale'])
        self.mean_architecture = c.get('mean_architecture', 'flat')
        if self.mean_architecture not in ('flat', 'shared_objects'):
            raise ValueError('Unknown declared mean architecture')
        if self.mean_architecture == 'shared_objects':
            # Task adaptation only: public local/global rows, no A response basis.
            # Preserve the original actor variance and subsequent critic RNG draws.
            SharedObjectMean = _shared_mean_class()
            with torch.random.fork_rng(devices=[]):
                shared_mean = SharedObjectMean(d, a)
                for head in (shared_mean.object_head, shared_mean.amount_head):
                    nn.init.zeros_(head[-1].weight)
                    nn.init.zeros_(head[-1].bias)
            # The temporary original mean is discarded, not retained unused.
            self.mean = shared_mean

    def forward(self, x):
        h = self.backbone(x)
        mean = self.mean(x) if self.mean_architecture == 'shared_objects' else self.mean(h)
        return mean, self.log_std(h).clamp(-10, 2)


class Q(nn.Module):
    def __init__(self, d, a, c):
        super().__init__()
        self.backbone = Backbone(d+a, c['hidden_dims'], c['use_layer_norm'])
        self.out = _linear(c['hidden_dims'], 1)

    def forward(self, x, u):
        return self.out(self.backbone(torch.cat((x, u), -1))).squeeze(-1)


def squashed_log_prob(pre_action, mean, log_std):
    z = (pre_action-mean)*torch.exp(-log_std)
    pre = (-.5*z.square()-log_std-.5*math.log(2*math.pi)).sum(-1)
    correction = (2*(math.log(2)-pre_action-F.softplus(-2*pre_action))).sum(-1)
    return pre-correction


def compute_gae(rewards, values, next_values, dones, gamma, lam):
    """Author scan, with truncation=0 because these tasks end truly."""
    delta = rewards+gamma*next_values*(1-dones)-values
    advantage = torch.empty_like(delta)
    carry = torch.zeros((), dtype=delta.dtype, device=delta.device)
    for i in range(len(delta)-1, -1, -1):
        carry = delta[i]+gamma*lam*(1-dones[i])*carry
        advantage[i] = carry
    return advantage


def episode_segments(dones, device):
    """Build fixed gather maps from collected real terminal flags, on CPU."""
    ends = np.flatnonzero(np.asarray(dones)).tolist()
    if not ends or ends[-1] != len(dones)-1:
        ends.append(len(dones)-1)
    starts = [0]+[x+1 for x in ends[:-1]]
    width = max(e-s+1 for s,e in zip(starts,ends))
    indices = np.zeros((len(starts),width),np.int64)
    valid = np.zeros_like(indices,dtype=np.float32)
    inverse = np.empty(len(dones),np.int64)
    for row,(s,e) in enumerate(zip(starts,ends)):
        n = e-s+1
        indices[row,:n] = np.arange(s,e+1)
        valid[row,:n] = 1
        inverse[s:e+1] = row*width+np.arange(n)
    return tuple(torch.as_tensor(x,device=device) for x in (indices,valid,inverse))


def segmented_gae(rewards, values, next_values, dones, gamma, lam, maps):
    """Same backwards recursion, batched by actual episode boundaries.

    This replaces O(rollout_length) GPU launches by O(max_segment_length),
    without crossing true terminal flags or inventing any additional data.
    """
    indices,valid,inverse = maps
    delta = rewards+gamma*next_values*(1-dones)-values
    padded = delta[indices]*valid
    out = torch.empty_like(padded)
    carry = torch.zeros(len(indices),device=delta.device,dtype=delta.dtype)
    for i in range(padded.shape[1]-1,-1,-1):
        carry = padded[:,i]+gamma*lam*carry
        out[:,i] = carry
    return out.flatten()[inverse]


class Candidate:
    def __init__(self, obs_dim, action_dim, seed, device='cuda', config=None):
        self.obs_dim, self.action_dim, self.seed = int(obs_dim), int(action_dim), int(seed)
        self.config = dict(DEFAULTS)
        self.config.update(config or {})
        c = self.config
        self.device = torch.device(device)
        self._inference_only = bool(c.pop('_inference_only', False))
        if c['policy_steps'] < c['batch_size'] or c['buffer_size'] < c['policy_steps']:
            raise ValueError('At least one complete minibatch and sufficient replay required')
        if c['discount'] < 0 or c['discount'] > 1 or c['gae_lambda'] < 0 or c['gae_lambda'] > 1:
            raise ValueError('Invalid discount/lambda')
        # Initialization does not reseed caller/training global random streams.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.actor = Actor(obs_dim, action_dim, c).to(self.device)
            self.critics = nn.ModuleList([Q(obs_dim, action_dim, c) for _ in range(2)]).to(self.device)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(c['temperature']), device=self.device))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=c['actor_lr'], eps=1e-8)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=c['critic_lr'], eps=1e-8)
        self.temp_opt = torch.optim.Adam([self.log_temperature], lr=c['temp_lr'], eps=1e-8)
        self.rng = torch.Generator(device=self.device).manual_seed(seed+177)
        self.replay = None
        if not self._inference_only:
            self.replay = {k: np.empty((c['buffer_size'],)+shape, np.float32) for k,shape in
                           [('s',(obs_dim,)),('a',(action_dim,)),('r',()),
                            ('ns',(obs_dim,)),('done',())]}
        self.pos = self.size = 0
        self.rollout = []
        self.pending = None
        self.environment_steps = self.updates = self.actor_updates = self.temperature_updates = 0
        self.rollout_updates = 0
        self.last = {}

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _sample(self, states):
        mean, logstd = self.actor(states)
        noise = torch.randn(mean.shape, device=self.device, generator=self.rng)
        pre = mean+logstd.exp()*noise
        return pre.tanh(), squashed_log_prob(pre, mean, logstd), pre

    def act(self, observation, deterministic=False):
        s = self._tensor(observation).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                result = self.actor(s)[0].tanh().squeeze(0)
            else:
                if self._inference_only:
                    raise RuntimeError('Evaluation endpoint prohibits training collection')
                a, lp, pre = self._sample(s)
                result = a.squeeze(0)
                # Deterministic DEV act calls never overwrite the training transition.
                packed = torch.cat((pre.squeeze(0), lp.reshape(1))).cpu().numpy()
                self.pending = (np.asarray(observation, np.float32).copy(),
                                packed[:-1].copy(), float(packed[-1]))
            return result.cpu().numpy().astype(np.float32, copy=False)

    def observe(self, obs, action, native_reward, next_obs, done):
        if self._inference_only or self.pending is None:
            raise RuntimeError('observe requires one actual sampled transition')
        pending_s, pre, lp = self.pending
        if not np.array_equal(np.asarray(obs, np.float32), pending_s):
            raise ValueError('Observed state is not the state of pending sampled action')
        if not np.allclose(np.tanh(pre), action, atol=2e-6, rtol=2e-6):
            raise ValueError('Action does not match recorded pre-squash behavior sample')
        r = float(native_reward)*self.config['reward_scale']
        values = dict(s=np.asarray(obs, np.float32), a=np.asarray(action, np.float32),
                      r=r, ns=np.asarray(next_obs, np.float32), done=float(done))
        for k, v in values.items():
            self.replay[k][self.pos] = v
        self.pos = (self.pos+1) % self.config['buffer_size']
        self.size = min(self.size+1, self.config['buffer_size'])
        self.rollout.append((values['s'].copy(), values['a'].copy(), r,
                             values['ns'].copy(), float(done), pre, lp))
        self.pending = None
        self.environment_steps += 1
        if len(self.rollout) >= self.config['policy_steps']:
            self._update()
            return dict(self.last, learning_reward=r)
        return {'learning_reward':r}

    def _q(self, s, a):
        return torch.stack([q(s, a) for q in self.critics])

    @torch.no_grad()
    def _soft_values(self, states):
        # One vectorized upload/evaluation for all 10 independent action samples.
        b, n = len(states), self.config['value_samples']
        tiled = states.unsqueeze(0).expand(n, b, -1).reshape(n*b, -1)
        if self.actor.mean_architecture == 'shared_objects':
            # Row-independent actor outputs are reusable across MC samples.
            # Keep the exact n*b action-noise draw and its generator ordering.
            mean, logstd = self.actor(states)
            mean = mean.unsqueeze(0).expand(n, b, -1).reshape(n*b, -1)
            logstd = logstd.unsqueeze(0).expand(n, b, -1).reshape(n*b, -1)
            noise = torch.randn(mean.shape, device=self.device, generator=self.rng)
            pre = mean+logstd.exp()*noise
            a = pre.tanh()
            lp = squashed_log_prob(pre, mean, logstd)
        else:
            # Preserve old flat math, batch shapes and RNG exactly.
            a, lp, _ = self._sample(tiled)
        v = self._q(tiled, a).mean(0)-self.log_temperature.exp()*lp
        return v.reshape(n, b).mean(0)

    def _update(self):
        c = self.config
        if len(self.rollout) < c['batch_size']:
            return
        columns = list(zip(*self.rollout))
        s, a, rewards, ns, done, pre, old_lp = [self._tensor(np.stack(x)) for x in columns]
        masks = 1-done
        maps = episode_segments(columns[4],self.device)
        replay = {k:self._tensor(v[:self.size]) for k,v in self.replay.items()}
        batch = c['batch_size']
        critic_steps = self.size//batch
        actor_batches = len(s)//batch
        losses = []
        for epoch in range(c['num_epochs']):
            # Exact native schedule: replay-sized critic sweep before actor sweep.
            if critic_steps < 100:
                indices = torch.randint(self.size, (critic_steps, batch), device=self.device, generator=self.rng)
            else:
                indices = torch.randperm(self.size, device=self.device, generator=self.rng)[:critic_steps*batch].reshape(-1,batch)
            for idx in indices:
                rs, ra, rr, rns, rd = [replay[k][idx] for k in ('s','a','r','ns','done')]
                with torch.no_grad():
                    na, nlp, _ = self._sample(rns)
                    nq = self._q(rns, na)
                    target = rr+c['discount']*(1-rd)*(nq-self.log_temperature.exp()*nlp)
                    if c['min_target']:
                        target = target.min(0,keepdim=True).values.expand_as(target)
                critic_loss = (self._q(rs,ra)-target).square().mean()
                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                self.critic_opt.step()
                self.updates += 1
            with torch.no_grad():
                alpha = self.log_temperature.exp()
                if c['gae_lambda'] > 0:
                    # At true episode ends bootstrap mask0 makes reset-state value irrelevant.
                    states = torch.cat((s,ns[-1:]),0)
                    vals = self._soft_values(states)
                    advantages = segmented_gae(rewards-alpha*old_lp, vals[:-1], vals[1:],
                                               done,c['discount'],c['gae_lambda'],maps)
                else:
                    advantages = self._q(s,a).mean(0)-alpha*old_lp-self._soft_values(s)
                # No advantage normalization in author implementation.
            indices = torch.randperm(len(s), device=self.device, generator=self.rng)[:actor_batches*batch].reshape(-1,batch)
            for idx in indices:
                mean, logstd = self.actor(s[idx])
                new_lp = squashed_log_prob(pre[idx],mean,logstd)
                ratio = torch.exp(new_lp-old_lp[idx])
                actor_masks = masks[idx] if c['terminal_actor_mask'] else torch.ones_like(masks[idx])
                gain = actor_masks*advantages[idx]
                actor_loss = -torch.minimum(gain*ratio,gain*ratio.clamp(1-c['clipping_ratio'],1+c['clipping_ratio'])).mean()
                entropy = -(actor_masks*new_lp).sum()/actor_masks.sum().clamp_min(1)
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                grad = nn.utils.clip_grad_norm_(self.actor.parameters(), c['clip_grad_norm'])
                self.actor_opt.step()
                self.actor_updates += 1
                temperature = self.log_temperature.exp()
                temp_loss = temperature*(entropy.detach()+c['entropy_coeff']*self.action_dim)
                # Keep native lower-temperature conditional; no unconditional clamp.
                temp_loss = torch.where((temperature <= .001)&(temp_loss > 0), temp_loss*0, temp_loss)
                self.temp_opt.zero_grad(set_to_none=True)
                temp_loss.backward()
                self.temp_opt.step()
                self.temperature_updates += 1
            losses = [critic_loss.detach(),actor_loss.detach(),entropy.detach(),
                      self.log_temperature.exp().detach(),grad.detach(),advantages.std().detach()]
        values = torch.stack(losses).cpu().tolist()
        if not np.isfinite(values).all():
            raise FloatingPointError('Non-finite PPO+ update diagnostics')
        self.last = dict(zip(('critic_loss','actor_loss','entropy','temperature','actor_gradient_norm','advantage_std'),values))
        self.last.update(critic_updates=self.updates,actor_updates=self.actor_updates,
                         terminal_actor_mask=c['terminal_actor_mask'])
        self.rollout.clear()
        self.rollout_updates += 1

    def finish(self):
        if self.pending is not None:
            raise RuntimeError('Unobserved actual action at finish')
        if self.rollout:
            if not self.rollout[-1][4]:
                raise RuntimeError('Final short rollout must end in a complete physical episode')
            if len(self.rollout) < self.config['batch_size']:
                raise RuntimeError('Final data shorter than a native complete minibatch')
            self._update()

    def accounting(self):
        return dict(environment_steps=self.environment_steps, updates=self.updates,
                    actor_updates=self.actor_updates, temperature_updates=self.temperature_updates,
                    rollout_updates=self.rollout_updates, pending_transitions=len(self.rollout))

    def stats(self):
        return dict(self.last)

    def save(self, directory):
        p = Path(directory); p.mkdir(parents=True,exist_ok=False)
        payload = dict(actor=self.actor.state_dict(),critics=self.critics.state_dict(),
                       log_temperature=self.log_temperature.detach(),accounting=self.accounting())
        torch.save(payload,p/'model.pt')
        (p/'metadata.json').write_text(json.dumps(dict(obs_dim=self.obs_dim,action_dim=self.action_dim,
             seed=self.seed,config=self.config,method='PPO+ RLJ/RLC2026 PyTorch adapter',
             storage='inference endpoint only; no optimizer/replay; not exact training-resumable'),indent=2))
        return str(p/'model.pt')

    @classmethod
    def load(cls,directory,device='cpu'):
        p = Path(directory)
        info = json.loads((p/'metadata.json').read_text())
        config = dict(info['config'],_inference_only=True)
        obj = cls(info['obs_dim'],info['action_dim'],info['seed'],device,config)
        state = torch.load(p/'model.pt',map_location=obj.device,weights_only=True)
        obj.actor.load_state_dict(state['actor']);obj.critics.load_state_dict(state['critics'])
        with torch.no_grad():obj.log_temperature.copy_(state['log_temperature'])
        for k,v in state['accounting'].items():
            if k != 'pending_transitions':setattr(obj,k,int(v))
        return obj
