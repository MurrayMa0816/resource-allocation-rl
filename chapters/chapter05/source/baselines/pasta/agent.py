"""RSS2026 PASTA adapter: official networks/controller, paper-correct GAE/PCGrad.

Native source: AlejandroMllo/PASTA @745d769740b6ce648c7ea8209933733dea8a1eab.
The untouched public source is vendor/pasta_native.py. This adapter explicitly
repairs two indexing defects: actual transition done masks and objective identity
when randomizing PCGrad comparisons. No scalarized actor loss replaces PCGrad.
"""
from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ._native import ContinuousActor, Critic, SmoothnessController

VERSION = 'pasta-official-networks-paper-index-corrected-v1'
COMMIT = '745d769740b6ce648c7ea8209933733dea8a1eab'


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return copy.deepcopy(value)


def vector_gae(rewards, values, dones, gamma=1., gae_lambda=.95):
    """Actual done[t] masks successor value; complete parallel true episodes."""
    if rewards.shape != values.shape or rewards.shape[-1] != 2:
        raise ValueError('Expected two-objective complete trajectory')
    if dones.shape == rewards.shape[:-1] + (1,):
        dones = dones[..., 0]
    if dones.shape != rewards.shape[:-1]:
        raise ValueError('Wrong done shape')
    if not torch.all(dones[-1] == 1) or torch.any(dones[:-1] != 0):
        raise ValueError('Each parallel path must be a complete true-terminal episode')
    advantage = torch.zeros_like(rewards)
    accumulator = torch.zeros_like(rewards[0])
    successor = torch.zeros_like(values[0])
    for t in reversed(range(len(rewards))):
        live = (1. - dones[t]).unsqueeze(-1)
        residual = rewards[t] + float(gamma) * live * successor - values[t]
        accumulator = residual + float(gamma) * float(gae_lambda) * live * accumulator
        advantage[t] = accumulator
        successor = values[t]
    return advantage, advantage + values


def project_conflicting_gradients(objective_grads, rng):
    """Paper Eq15, preserving original objective IDs while shuffling j order.

    The public implementation shuffles only the comparison array, then skips by
    the old position. That can skip the other objective and project onto self.
    Here each projected g_i is compared to every distinct ORIGINAL g_j.
    """
    if len(objective_grads) < 2:
        raise ValueError('Multi-objective PCGrad requires at least two gradients')
    originals = [g.clone() for g in objective_grads]
    projected = [g.clone() for g in objective_grads]
    conflicts = checks = 0
    for i in range(len(originals)):
        order = [j for j in range(len(originals)) if j != i]
        rng.shuffle(order)
        for j in order:
            checks += 1
            dot = torch.dot(projected[i], originals[j])
            if bool(dot < 0):
                conflicts += 1
                projected[i] -= dot / (torch.dot(originals[j], originals[j]) + 1e-8) * originals[j]
    return projected, float(conflicts / max(1, checks))


class Agent:
    manages_observation_normalization = True
    on_policy = True
    action_mode = 'raw'
    name = 'pasta'

    def __init__(self, obs_dim, action_dim, seed=0, device='cpu', config=None):
        self.config = dict(config or {})
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        self.seed, self.device = int(seed), torch.device(device)
        self.horizon = int(self.config.get('horizon', 32 if self.config.get('domain') == 'radio' else 24))
        self.n_env = int(self.config.get('n_env', 8))
        self.rollout_panels = int(self.config.get('rollout_panels', 8))
        self.gamma = float(self.config.get('gamma', 1.))
        self.gae_lambda = float(self.config.get('gae_lambda', .95))
        self.lr = float(self.config.get('lr', 3e-4))
        self.epochs = int(self.config.get('ppo_epochs', 10))
        self.batch_size = int(self.config.get('minibatch_size', 64))
        self.clip = float(self.config.get('clip', .2))
        self.value_coef = float(self.config.get('value_loss_coef', .5))
        self.entropy_coef = float(self.config.get('entropy_coef', .01))
        self.max_grad_norm = float(self.config.get('max_grad_norm', .5))
        self.maintenance_rate = float(self.config.get('maintenance_rate', .15))
        self.utopia_point = float(self.config.get('utopia_point', 1.05))
        self.total_steps = int(self.config.get('learning_steps', 24576))
        if min(self.horizon, self.n_env, self.rollout_panels, self.epochs, self.batch_size) < 1:
            raise ValueError('Positive rollout/update sizes required')
        self.rollout_size = self.horizon * self.n_env * self.rollout_panels
        if self.total_steps % self.rollout_size:
            raise ValueError('Scientific budget must finish the complete registered rollout')
        if min(self.batch_size, self.rollout_size) < 2 or self.rollout_size % self.batch_size == 1:
            raise ValueError('Native unbiased advantage standard deviation requires minibatches of at least two')
        self.preference = np.asarray(self.config.get('training_preference', [.8, .2]), np.float32)
        if self.preference.shape != (2,) or np.any(self.preference < 0) or not np.isclose(self.preference.sum(), 1.):
            raise ValueError('Fixed preference must lie on the two-objective simplex')
        self.use_obs_rms = bool(self.config.get('observation_normalization', False))
        self.obs_mean = np.zeros(self.obs_dim, np.float64)
        self.obs_var = np.ones(self.obs_dim, np.float64)
        self.obs_count = 1e-4
        self.rng = np.random.default_rng(self.seed + 512701)
        self.projection_rng = random.Random(self.seed + 512702)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed + 512703)
        self.eval_generator = torch.Generator(device=self.device).manual_seed(self.seed + 512704)
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed + 512705)
            self.actor = ContinuousActor(self.obs_dim, self.action_dim, 2, hidden_dim=64).to(self.device)
            self.critic = Critic(self.obs_dim, 2, hidden_dim=64).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.sm_controller = SmoothnessController(
            min_mu=float(self.config.get('min_mu', .05)),
            max_mu=float(self.config.get('max_mu', 10.)),
            conflict_threshold=float(self.config.get('conflict_threshold', .4)),
            ema_alpha=float(self.config.get('smoothness_ema', .05)),
            total_steps=self.total_steps, enable_decay=True, enable_conflict=True)
        self.global_min = torch.full((2,), float('inf'), device=self.device)
        self.global_max = torch.full((2,), float('-inf'), device=self.device)
        self.last_conflict = 0.
        self.collected_steps = self.optimized_steps = self.update_groups = self.optimizer_steps = 0
        self.current_preferences = None
        self.collection_inputs = []
        self.pending_panels = []
        self.inference_only = False

    def _normalize(self, observations):
        x = np.asarray(observations, np.float32)
        if not self.use_obs_rms:
            return x.copy()
        return np.clip((x.astype(np.float64) - self.obs_mean) / np.sqrt(self.obs_var + 1e-8),
                       -10., 10.).astype(np.float32)

    def _observe_public_inputs(self, observations):
        if not self.use_obs_rms:
            return
        x = np.asarray(observations, np.float64)
        mean, variance, count = x.mean(0), x.var(0), len(x)
        delta = mean - self.obs_mean
        total = self.obs_count + count
        self.obs_mean += delta * count / total
        self.obs_var = (self.obs_var * self.obs_count + variance * count
                        + delta * delta * self.obs_count * count / total) / total
        self.obs_count = float(total)

    def _inputs(self, obs, prefs, already_normalized=False):
        x, p = np.asarray(obs, np.float32), np.asarray(prefs, np.float32)
        if x.ndim == 1:
            x = x[None]
        if p.ndim == 1:
            p = np.broadcast_to(p, (len(x), 2)).copy()
        if x.shape != (len(x), self.obs_dim) or p.shape != (len(x), 2):
            raise ValueError('Public observation/preference shapes differ')
        if not np.isfinite(x).all() or not np.isfinite(p).all():
            raise ValueError('Nonfinite observation or preference')
        if np.any(p < 0) or not np.allclose(p.sum(-1), 1., rtol=0., atol=1e-6):
            raise ValueError('Invalid preference')
        x = x.copy() if already_normalized else self._normalize(x)
        return torch.as_tensor(x, device=self.device), torch.as_tensor(p, device=self.device)

    def _dist(self, observations, preferences):
        # The author demo uses x=2*y-1 before clipping to its physical Box.
        # Our existing complete raw latent space is unbounded: retain the exact
        # affine map without adding any clipping. Log density includes the
        # constant affine Jacobian, so ratios and entropy gradients are native.
        original = self.actor(observations, preferences)
        return torch.distributions.Normal(2. * original.mean - 1., 2. * original.stddev)

    def train_preferences(self, step, total_steps, rng):
        if self.inference_only:
            raise RuntimeError('Inference endpoint cannot collect training')
        if int(step) != self.collected_steps or int(total_steps) != self.total_steps or step >= total_steps:
            raise ValueError('Physical-step ledger/configuration mismatch')
        if self.current_preferences is not None or self.collection_inputs:
            raise RuntimeError('Unconsumed complete episode panel')
        self.current_preferences = np.broadcast_to(self.preference, (self.n_env, 2)).copy()
        return self.current_preferences.copy()

    @torch.no_grad()
    def sample_info(self, obs, prefs):
        if self.inference_only or self.current_preferences is None:
            raise RuntimeError('Training panel has not started')
        if not np.array_equal(np.asarray(prefs, np.float32), self.current_preferences):
            raise ValueError('Collection preference differs from fixed task preference')
        if len(self.collection_inputs) >= self.horizon:
            raise RuntimeError('Panel is already complete')
        self._observe_public_inputs(obs)
        normalized = self._normalize(obs)
        self.collection_inputs.append(normalized.copy())
        x, p = self._inputs(normalized, prefs, already_normalized=True)
        dist = self._dist(x, p)
        action = dist.mean + dist.stddev * torch.randn(
            dist.mean.shape, dtype=dist.mean.dtype, device=self.device, generator=self.generator)
        return dict(actions=action.cpu().numpy(), logp=dist.log_prob(action).sum(-1, keepdim=True).cpu().numpy(),
                    values=self.critic(x, p).cpu().numpy())

    @torch.no_grad()
    def act(self, obs, prefs, deterministic=False):
        x, p = self._inputs(obs, prefs)
        dist = self._dist(x, p)
        action = dist.mean if deterministic else dist.mean + dist.stddev * torch.randn(
            dist.mean.shape, dtype=dist.mean.dtype, device=self.device, generator=self.eval_generator)
        return action.cpu().numpy()

    def update_rollout(self, trajectory, prefs, learning_steps):
        if self.inference_only:
            raise RuntimeError('Inference endpoint cannot train')
        if self.current_preferences is None or not np.array_equal(
                np.asarray(prefs, np.float32), self.current_preferences):
            raise ValueError('Update/collection preference mismatch')
        shape = (self.horizon, self.n_env)
        if np.asarray(trajectory['obs']).shape != shape + (self.obs_dim,) or len(self.collection_inputs) != self.horizon:
            raise ValueError('Expected one complete parallel episode panel')
        if int(learning_steps) != self.collected_steps + self.horizon * self.n_env:
            raise ValueError('Physical training count mismatch')
        to = lambda v: torch.as_tensor(v, dtype=torch.float32, device=self.device)
        rewards, values, dones = map(to, (trajectory['rewards'], trajectory['values'], trajectory['dones']))
        if rewards.shape != shape + (2,) or not torch.isfinite(rewards).all() or torch.any(rewards[:-1] != 0):
            raise ValueError('Only actual two-objective terminal rewards permitted')
        advantage, returns = vector_gae(rewards, values, dones, self.gamma, self.gae_lambda)
        size = self.horizon * self.n_env
        self.pending_panels.append(dict(
            obs=np.stack(self.collection_inputs).reshape(size, self.obs_dim),
            preferences=np.broadcast_to(prefs, shape + (2,)).reshape(size, 2).copy(),
            actions=np.asarray(trajectory['actions'], np.float32).reshape(size, self.action_dim).copy(),
            logp=np.asarray(trajectory['logp'], np.float32).reshape(size).copy(),
            advantages=advantage.cpu().numpy(), returns=returns.cpu().numpy(),
            values=values.cpu().numpy()))
        self.collected_steps = int(learning_steps)
        self.current_preferences = None
        self.collection_inputs = []
        if len(self.pending_panels) < self.rollout_panels:
            return dict(update_groups=0, optimizer_updated=False, native_update=False,
                        collected_real_training_steps=self.collected_steps,
                        pending_complete_episode_panels=len(self.pending_panels))
        if len(self.pending_panels) != self.rollout_panels:
            raise RuntimeError('Too many pending panels')
        data = {k: to(np.concatenate([p[k].reshape((-1,) + p[k].shape[2:])
                                     if k in ('advantages', 'returns', 'values') else p[k]
                                     for p in self.pending_panels], axis=0))
                for k in self.pending_panels[0]}
        obs_t, pref_t, actions, old_logp = (data[k] for k in ('obs', 'preferences', 'actions', 'logp'))
        with torch.no_grad():
            recalc = self._dist(obs_t, pref_t).log_prob(actions).sum(-1)
            logp_error = float((recalc - old_logp).abs().max())
            if logp_error > 2e-5:
                raise RuntimeError('Collection old-logprob input drift: ' + str(logp_error))
            self.global_min = torch.minimum(self.global_min, data['returns'].amin(0))
            self.global_max = torch.maximum(self.global_max, data['returns'].amax(0))
            ranges = (self.global_max - self.global_min).clamp(min=1e-6)
            normalized_returns = (data['returns'] - self.global_min) / ranges
            mu = self.sm_controller.update(self.optimized_steps, self.last_conflict)
            attention = torch.softmax((self.utopia_point - normalized_returns) * pref_t / mu, dim=1)
            effective = (1. - self.maintenance_rate) * attention + self.maintenance_rate / 2.
        if len(obs_t) != self.rollout_size:
            raise RuntimeError('Registered rollout size differs')
        totals = {k: 0. for k in ('value_loss', 'policy_loss', 'entropy', 'approximate_kl',
                                  'clip_fraction', 'actor_grad_norm', 'critic_grad_norm')}
        conflicts, steps = [], 0
        for _ in range(self.epochs):
            order = self.rng.permutation(len(obs_t))
            for start in range(0, len(obs_t), self.batch_size):
                ids = torch.as_tensor(order[start:start+self.batch_size], device=self.device)
                value = self.critic(obs_t[ids], pref_t[ids])
                value_loss = self.value_coef * ((value - data['returns'][ids]).square()
                                                * effective[ids].detach()).sum(1).mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                if not torch.isfinite(value_loss) or not torch.isfinite(critic_grad):
                    raise FloatingPointError('Nonfinite PASTA critic update')
                self.critic_optimizer.step()
                vector_adv = data['advantages'][ids]
                # Native uses the unbiased minibatch standard deviation.
                vector_adv = (vector_adv - vector_adv.mean(0)) / (vector_adv.std(0, unbiased=True) + 1e-8)
                dist = self._dist(obs_t[ids], pref_t[ids])
                logp = dist.log_prob(actions[ids]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                ratio = (logp - old_logp[ids]).exp()
                losses, gradients = [], []
                actor_parameters = list(self.actor.parameters())
                for objective in range(2):
                    a = vector_adv[:, objective]
                    loss = -torch.minimum(ratio * a, ratio.clamp(1.-self.clip, 1.+self.clip) * a).mean()
                    grads = torch.autograd.grad(loss, actor_parameters, retain_graph=True)
                    gradients.append(torch.cat([g.reshape(-1) for g in grads]))
                    losses.append(loss)
                projected, conflict = project_conflicting_gradients(gradients, self.projection_rng)
                conflicts.append(conflict)
                final_gradient = sum(projected)
                entropy_grads = torch.autograd.grad(-self.entropy_coef * entropy,
                                                   actor_parameters, allow_unused=True)
                self.actor_optimizer.zero_grad(set_to_none=True)
                offset = 0
                for parameter, ent_grad in zip(actor_parameters, entropy_grads):
                    n = parameter.numel()
                    parameter.grad = final_gradient[offset:offset+n].reshape_as(parameter).clone()
                    if ent_grad is not None:
                        parameter.grad.add_(ent_grad)
                    offset += n
                if offset != final_gradient.numel():
                    raise RuntimeError('Incomplete projected-gradient scatter')
                actor_grad = nn.utils.clip_grad_norm_(actor_parameters, self.max_grad_norm)
                if not torch.isfinite(actor_grad) or not torch.isfinite(final_gradient).all():
                    raise FloatingPointError('Nonfinite PASTA actor gradient')
                self.actor_optimizer.step()
                self.optimizer_steps += 1
                steps += 1
                with torch.no_grad():
                    logratio = logp - old_logp[ids]
                    observed = torch.stack([value_loss, sum(losses), entropy,
                        ((ratio-1.)-logratio).mean(),
                        ((ratio-1.).abs() > self.clip).float().mean(),
                        actor_grad, critic_grad]).detach().cpu().tolist()
                    for k, v in zip(totals, observed):
                        totals[k] += float(v)
        self.last_conflict = float(np.mean(conflicts))
        self.optimized_steps += self.rollout_size
        self.update_groups += 1
        self.pending_panels = []
        return dict({k: v / steps for k, v in totals.items()},
                    update_groups=1, optimizer_updated=True, native_update=True,
                    optimizer_steps=self.optimizer_steps, minibatch_steps_this_group=steps,
                    collected_real_training_steps=self.collected_steps,
                    optimized_real_training_steps=self.optimized_steps,
                    old_logp_recompute_max_error=logp_error, mu=float(mu),
                    last_conflict=self.last_conflict, adapter_version=VERSION,
                    mechanism='ASTCH critic attention + per-objective PPO + paper-correct PCGrad sum',
                    raw_negative_J_preserved=True)

    def _observation_state(self):
        return dict(enabled=self.use_obs_rms, mean=self.obs_mean.copy(),
                    var=self.obs_var.copy(), count=self.obs_count)

    def state_dict(self):
        if self.inference_only:
            return self.checkpoint_state_dict()
        return dict(format='pasta-full-ram-v1', actor=_cpu(self.actor.state_dict()),
                    critic=_cpu(self.critic.state_dict()),
                    actor_optimizer=_cpu(self.actor_optimizer.state_dict()),
                    critic_optimizer=_cpu(self.critic_optimizer.state_dict()),
                    global_min=_cpu(self.global_min), global_max=_cpu(self.global_max),
                    smoothness_controller=copy.deepcopy(vars(self.sm_controller)),
                    last_conflict=self.last_conflict, observation=self._observation_state(),
                    current_preferences=_cpu(self.current_preferences),
                    collection_inputs=_cpu(self.collection_inputs), pending_panels=_cpu(self.pending_panels),
                    collected_steps=self.collected_steps, optimized_steps=self.optimized_steps,
                    update_groups=self.update_groups, optimizer_steps=self.optimizer_steps,
                    private_rng=copy.deepcopy(self.rng.bit_generator.state),
                    projection_rng=self.projection_rng.getstate(),
                    generator=self.generator.get_state().cpu().clone(),
                    eval_generator=self.eval_generator.get_state().cpu().clone(),
                    generator_device=self.device.type, inference_only=False)

    def checkpoint_state_dict(self):
        if self.current_preferences is not None or self.collection_inputs or self.pending_panels:
            raise RuntimeError('Fixed endpoint requires complete rollout/update boundary')
        return dict(format='pasta-inference-v1', actor=_cpu(self.actor.state_dict()),
                    observation=self._observation_state(), collected_steps=self.collected_steps,
                    optimized_steps=self.optimized_steps, update_groups=self.update_groups,
                    optimizer_steps=self.optimizer_steps, inference_only=True,
                    training_resume_supported=False, adapter_version=VERSION,
                    policy_architecture=dict(obs_dim=self.obs_dim, action_dim=self.action_dim,
                                             hidden=[64, 64], activation='Tanh',
                                             mean_activation='Sigmoid', native_to_raw_affine=[2.,-1.],
                                             gaussian='unbounded raw joint latent'))

    def load_state_dict(self, state):
        if state['format'] not in ('pasta-full-ram-v1', 'pasta-inference-v1'):
            raise ValueError('Foreign PASTA state')
        if state['format'] == 'pasta-full-ram-v1' and state['generator_device'] != self.device.type:
            raise ValueError('Exact fullRAM restoration requires the same CPU/CUDA generator backend')
        self.actor.load_state_dict(state['actor'])
        self.collected_steps = int(state['collected_steps'])
        self.optimized_steps = int(state['optimized_steps'])
        self.update_groups = int(state['update_groups'])
        self.optimizer_steps = int(state['optimizer_steps'])
        obs = state['observation']
        if bool(obs['enabled']) != self.use_obs_rms:
            raise ValueError('Observation normalization differs from registered configuration')
        self.obs_mean, self.obs_var, self.obs_count = obs['mean'].copy(), obs['var'].copy(), float(obs['count'])
        self.inference_only = state['format'] == 'pasta-inference-v1'
        if self.inference_only:
            self.current_preferences, self.collection_inputs, self.pending_panels = None, [], []
            for name in ('critic', 'actor_optimizer', 'critic_optimizer'):
                if hasattr(self, name):
                    delattr(self, name)
        else:
            self.critic.load_state_dict(state['critic'])
            self.actor_optimizer.load_state_dict(copy.deepcopy(state['actor_optimizer']))
            self.critic_optimizer.load_state_dict(copy.deepcopy(state['critic_optimizer']))
            self.global_min = state['global_min'].to(self.device).clone()
            self.global_max = state['global_max'].to(self.device).clone()
            vars(self.sm_controller).update(copy.deepcopy(state['smoothness_controller']))
            self.last_conflict = float(state['last_conflict'])
            self.current_preferences = _cpu(state['current_preferences'])
            self.collection_inputs = _cpu(state['collection_inputs'])
            self.pending_panels = _cpu(state['pending_panels'])
            self.rng.bit_generator.state = copy.deepcopy(state['private_rng'])
            self.projection_rng.setstate(state['projection_rng'])
            if state['generator_device'] == self.device.type:
                self.generator.set_state(state['generator'].cpu())
                self.eval_generator.set_state(state['eval_generator'].cpu())

    def metadata(self):
        native = Path(__file__).with_name('_native.py')
        return dict(method='PASTA', venue='RSS2026', doi='10.15607/RSS.2026.XXII.134',
                    paper='https://www.roboticsproceedings.org/rss22/p134.html',
                    author_repository='https://github.com/AlejandroMllo/PASTA', commit=COMMIT,
                    native_source=str(native), native_source_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                    adapter_version=VERSION, official_license_declared=False,
                    core='Official ContinuousActor/Critic/SmoothnessController; paper-equation update with two documented indexing fixes',
                    repairs=['GAE successor uses actual done[t], preventing cross-episode reward/value leakage',
                             'PCGrad shuffles distinct original comparison IDs, never mismatches projected-objective IDs'],
                    architecture=dict(actor_hidden=[64,64], critic_shared=[64,64],
                                      critic_per_objective_head=[64], activation='Tanh',
                                      actor_mean_activation='Sigmoid', log_std_initial=-.5),
                    native_actor_aggregation='Sum of unweighted projected per-objective PPO gradients; attention only weights critic',
                    objective_dimension=2, gamma=self.gamma, gae_lambda=self.gae_lambda,
                    lr=self.lr, ppo_epochs=self.epochs, minibatch_size=self.batch_size,
                    clip=self.clip, value_coef=self.value_coef, entropy_coef=self.entropy_coef,
                    max_grad_norm=self.max_grad_norm, maintenance_rate=self.maintenance_rate,
                    utopia_point=self.utopia_point, smoothness_controller=copy.deepcopy(vars(self.sm_controller)),
                    rollout_panels=self.rollout_panels, rollout_steps=self.rollout_size,
                    native_rollout_steps=2048, training_preference=self.preference.tolist(),
                    original_normalization='Running min/max of actual training GAE returns, attention only',
                    observation_normalization=self.use_obs_rms,
                    reward='Unchanged true terminal [K/N, native SSIM or completion], no intermediate signal',
                    action_adapter='Native Gaussian y mapped to rawjointlatent x=2*y-1, with exact affine Jacobian in logp; unbounded current raw space, no clipping',
                    task_adaptations=['Public task observation/action dimensions', 'finite-horizon gamma1',
                                      'fixed task preference[.8,.2]', 'complete parallel episode panels',
                                      'explicit gamma/rollout configuration'],
                    no_teacher=True, no_private_object_labels=True,
                    endpoint='One fixed inference actor and necessary public observation statistics; no critic/Adam/replay',
                    inference_only=self.inference_only)

