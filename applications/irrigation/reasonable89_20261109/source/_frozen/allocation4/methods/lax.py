"""Native terminal-MC and LAX updates on the complete Gaussian latent action.

The control variate used by an actor update was trained on earlier batches.
After that update, a frozen *pre-update* actor is used to train the variate for
the next batch. LAX minimizes the second moment of the complete trajectory
gradient, including all trainable actor parameters and cross-time covariance.
The MSE arm has identical capacity but fits the MC residual instead.

These are unclipped, one-batch policy-gradient estimators, NOT unbiased PPO.
The raw score/pathwise cancellation does not imply unbiased Adam parameters,
nor monotonic improvement after gradient clipping and empirical KL control.
No simulator, physical transition model, private reward, or extra rollout is
available in this module.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Callable

import numpy as np
import torch
from torch import nn


MODES = ('lax', 'cv_mse', 'mc_pg')
STATE_KEYS = ('local', 'global', 'caps', 'budget', 'remaining')
VERSION = 'complete-gaussian-lax-trajectory-second-moment-v1'


def state_features(states):
    """Generic public-state summary; no task-specific responses or rules."""
    x = states['local']
    return torch.cat((x.mean(1), x.std(1, unbiased=False), x.amin(1),
                      x.amax(1), states['global']), -1)


def gaussian_parameters(actor, states):
    """Complete latent Gaussian parameters, both shaped [N, event_dim]."""
    spatial = actor.spatial.mean(states)
    if actor.learned_quota:
        mean = torch.cat((actor.quota_mean(states), spatial), -1)
        sigma = torch.cat((actor.quota_sigma.reshape(1), actor.spatial.sigma))
    else:
        mean, sigma = spatial, actor.spatial.sigma
    sigma = sigma.reshape(1, -1).expand_as(mean)
    if not torch.isfinite(mean).all() or not torch.isfinite(sigma).all() or not bool((sigma > 0).all()):
        raise FloatingPointError('Nonfinite Gaussian or nonpositive scale')
    return mean, sigma


def gaussian_log_prob(mean, sigma, raw):
    return (-.5 * ((raw - mean) / sigma).square() - sigma.log()
            - .5 * math.log(2 * math.pi)).sum(-1, keepdim=True)


def gaussian_kl(old_mean, old_sigma, mean, sigma):
    """KL(old || new), summed over the entire latent event, per state."""
    return (torch.log(sigma / old_sigma)
            + (old_sigma.square() + (old_mean - mean).square())
            / (2 * sigma.square()) - .5).sum(-1)


def lax_surrogate(mean, sigma, raw, advantage, control_fn=None, mask=None):
    """Scalar ascent surrogate at the recorded policy.

    ``raw`` and the state supplied to ``control_fn`` must be detached data.
    The noise is reconstructed at the current (recorded) policy and detached.
    Thus only the action path carries the compensating derivative. Do NOT
    detach the value of control_fn(raw): mixed derivatives with respect to its
    parameters are needed when training the gradient-second-moment objective.
    During an actor update those parameters are frozen instead.
    """
    if raw.shape != mean.shape or sigma.shape != mean.shape:
        raise ValueError('Complete Gaussian shape mismatch')
    if advantage.shape != (len(raw), 1):
        raise ValueError('A scalar MC residual is required per decision')
    if mask is None:
        mask = torch.ones_like(advantage)
    logp = gaussian_log_prob(mean, sigma, raw)
    if control_fn is None:
        return (logp * advantage * mask).sum(-1).mean()
    epsilon = ((raw - mean.detach()) / sigma.detach()).detach()
    rep_action = mean + sigma * epsilon
    sampled = control_fn(raw)
    pathwise = control_fn(rep_action)
    if sampled.shape != advantage.shape or pathwise.shape != advantage.shape:
        raise ValueError('Control variate must output [N,1]')
    return ((logp * (advantage - sampled) + pathwise) * mask).sum(-1).mean()


def kl_step(actor, optimizer, states, loss_fn: Callable[[], torch.Tensor],
            max_kl=.01, *, max_backtracks=12, grad_clip=.5):
    """One Adam/optimizer step with exact visited-state Gaussian KL checking.

    Each attempt restores BOTH parameters and optimizer state before reducing
    the learning rate. loss_fn must rebuild the graph on identical fixed data;
    it must not draw noise, refit a model, or mutate the guidance. Accepted
    Adam state is retained while configured learning rates are restored.
    actor_optimizer_steps counts *all* actual step calls, including rolled-back
    attempts; actor_committed_steps is 0 or 1. Neither is a new data count.
    """
    if not math.isfinite(max_kl) or max_kl <= 0 or max_backtracks < 0:
        raise ValueError('Positive finite KL and nonnegative backtracks required')
    if grad_clip is not None and (not math.isfinite(grad_clip) or grad_clip <= 0):
        raise ValueError('Positive finite gradient clip required')
    state = copy.deepcopy(actor.state_dict())
    opt_state = copy.deepcopy(optimizer.state_dict())
    rates = [float(group['lr']) for group in optimizer.param_groups]
    if not rates or not all(math.isfinite(lr) and lr > 0 for lr in rates):
        raise ValueError('Positive learning rates required')
    with torch.no_grad():
        old_mean, old_sigma = gaussian_parameters(actor, states)
        old_mean, old_sigma = old_mean.detach().clone(), old_sigma.detach().clone()
    accepted = False
    attempts, last_kl, norm_value, loss_value = 0, 0., 0., 0.
    factor = 1.
    try:
        for attempt in range(int(max_backtracks) + 1):
            actor.load_state_dict(state)
            optimizer.load_state_dict(copy.deepcopy(opt_state))
            factor = .5 ** attempt
            for group, lr in zip(optimizer.param_groups, rates):
                group['lr'] = lr * factor
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn()
            if loss.ndim or not torch.isfinite(loss):
                raise FloatingPointError('Finite scalar policy loss required')
            loss.backward()
            params = [p for p in actor.parameters() if p.requires_grad and p.grad is not None]
            if not params:
                raise ValueError('Policy loss did not differentiate the actor')
            if not all(torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError('Nonfinite policy gradient')
            if grad_clip is None:
                norm = torch.sqrt(sum(p.grad.square().sum() for p in params))
            else:
                norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
            norm_value, loss_value = float(norm.detach()), float(loss.detach())
            optimizer.step()
            attempts += 1
            with torch.no_grad():
                try:
                    mean, sigma = gaussian_parameters(actor, states)
                    kl = gaussian_kl(old_mean, old_sigma, mean, sigma).mean()
                    last_kl = float(kl)
                except FloatingPointError:
                    last_kl = math.inf
            if math.isfinite(last_kl) and last_kl <= max_kl + 1e-12:
                accepted = True
                break
    except BaseException:
        actor.load_state_dict(state)
        optimizer.load_state_dict(opt_state)
        optimizer.zero_grad(set_to_none=True)
        raise
    if not accepted:
        actor.load_state_dict(state)
        optimizer.load_state_dict(opt_state)
    for group, lr in zip(optimizer.param_groups, rates):
        group['lr'] = lr
    actual_kl = max(0., last_kl) if accepted else 0.
    return dict(actor_loss=loss_value, actor_grad_norm=norm_value,
                actual_kl=actual_kl, macro_kl=actual_kl,
                actor_optimizer_steps=attempts, actor_committed_steps=int(accepted),
                actor_backtracks=max(0, attempts - 1), actor_step_accepted=accepted,
                actor_lr_factor=factor if accepted else 0.,
                rejected_kl=None if math.isfinite(last_kl) else 'nonfinite',
                clip_fraction=0., early_stop=not accepted)


def fit_mc_critic(critic, optimizer, states, raw, returns, mask, epochs=4):
    """Full-batch terminal-MC regression, identical across matched PG arms."""
    if int(epochs) != epochs or epochs < 0:
        raise ValueError('Nonnegative integer critic epochs required')
    targets = returns.reshape(-1, 1).detach()
    if mask.shape != targets.shape or len(raw) != len(targets):
        raise ValueError('Critic target/mask shape mismatch')
    if not bool((mask > 0).any()):
        return dict(value_loss=0., value_grad_norm=0., critic_optimizer_steps=0,
                    value_optimizer_steps=0)
    losses, norms = [], []
    for _ in range(int(epochs)):
        prediction = critic(states, raw)
        if prediction.shape != targets.shape:
            raise ValueError('Critic must emit one scalar per state')
        loss = ((prediction - targets).square() * mask).sum() / mask.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), .5)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite MC critic update')
        optimizer.step()
        losses.append(float(loss.detach())); norms.append(float(norm.detach()))
    return dict(value_loss=float(np.mean(losses)) if losses else 0.,
                value_grad_norm=float(np.mean(norms)) if norms else 0.,
                critic_optimizer_steps=len(losses), value_optimizer_steps=len(losses))


class ControlVariate(nn.Module):
    def __init__(self, local_dim, global_dim, event_dim, seed, device):
        super().__init__()
        # torch.manual_seed also seeds CUDA generators; restore them as well as
        # the CPU generator so auxiliary initialization cannot perturb rollout
        # or another method's random stream.
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(741000 + int(seed))
            self.net = nn.Sequential(nn.Linear(4 * local_dim + global_dim + event_dim, 64),
                                     nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        self.to(device=device, dtype=torch.float64)

    def forward(self, states, raw):
        feature = state_features(states)
        # Numerical feature scaling only, shared across both auxiliary arms.
        feature = torch.sign(feature) * torch.log1p(feature.abs())
        return self.net(torch.cat((feature, torch.tanh(raw)), -1))


def _tensor_batches(batches, device):
    result = []
    if not batches:
        raise ValueError('At least one complete trajectory required')
    for batch in batches:
        if set(batch['states']) != set(STATE_KEYS):
            raise ValueError('Unexpected public-state keys')
        item = dict(states={k: torch.as_tensor(batch['states'][k], dtype=torch.float64,
                                              device=device).detach() for k in STATE_KEYS})
        for key in ('raw', 'old_logp', 'mask', 'values', 'returns'):
            item[key] = torch.as_tensor(batch[key], dtype=torch.float64, device=device).detach()
        n = len(item['raw'])
        if n == 0 or any(item[k].shape != (n, 1) for k in ('old_logp', 'mask', 'values')):
            raise ValueError('Complete scalar likelihood/value/mask columns required')
        if item['returns'].shape != (n,) or not bool((item['returns'] == item['returns'][0]).all()):
            raise ValueError('Only complete terminal-MC returns are supported')
        if not bool(((item['mask'] == 0) | (item['mask'] == 1)).all()):
            raise ValueError('Binary event masks required')
        if not all(torch.isfinite(item[k]).all() for k in ('raw', 'old_logp', 'mask', 'values', 'returns')):
            raise ValueError('Finite recorded training data required')
        item['advantage'] = item['returns'][:, None] - item['values']
        result.append(item)
    if len({len(x['raw']) for x in result}) != 1:
        raise ValueError('Equal fixed horizons required for matched trajectory weighting')
    joined = dict(states={k: torch.cat([b['states'][k] for b in result]) for k in STATE_KEYS})
    for key in ('raw', 'old_logp', 'mask', 'values', 'returns', 'advantage'):
        joined[key] = torch.cat([b[key] for b in result])
    return result, joined


class Method(nn.Module):
    """Lagged LAX / MSE-CV / native MC, sharing the same policy backend."""
    def __init__(self, spec, event_dim, seed, device, *, mode='lax',
                 auxiliary_epochs=2, auxiliary_lr=3e-4, max_kl=.01, critic_epochs=4):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Mode must be lax, cv_mse, or mc_pg')
        if auxiliary_epochs < 1 or int(auxiliary_epochs) != auxiliary_epochs:
            raise ValueError('Positive integer auxiliary epochs required')
        self.mode, self.device = mode, torch.device(device)
        self.event_dim, self.seed = int(event_dim), int(seed)
        self.auxiliary_epochs, self.critic_epochs = int(auxiliary_epochs), int(critic_epochs)
        self.max_kl = float(max_kl)
        self.control = ControlVariate(spec.local_dim, spec.global_dim, self.event_dim, seed, self.device)
        self.cv_optimizer = torch.optim.Adam(self.control.parameters(), lr=auxiliary_lr)
        self.updates = 0

    def metadata(self):
        """Static scientific settings; external optimizer rates live in the run protocol."""
        auxiliary_group = self.cv_optimizer.param_groups[0]
        return dict(
            version=VERSION, mode=self.mode, event_dim=self.event_dim, seed=self.seed,
            dtype='float64', device=str(self.device),
            estimator='native terminal Monte Carlo' if self.mode == 'mc_pg' else
                      'MC residual score plus action-reparameterization compensation',
            actor_objective='unclipped full-batch Gaussian policy-gradient surrogate',
            actor_committed_steps_per_batch=1,
            actor_gradient_reduction='mean over all recorded decisions, multiplied by event mask',
            actor_gradient_clip_norm=.5, advantage_normalization='none',
            actor_optimizer='provided by runner; its learning rate and other settings are in run protocol',
            kl=dict(direction='old||new', metric='exact complete-event diagonal Gaussian KL',
                    state_reduction='mean over all recorded decision states', limit=self.max_kl,
                    max_backtracks=12, learning_rate_multiplier=.5,
                    rollback='both actor tensors and complete optimizer state',
                    accepted_learning_rate='restore configured rate after accepted reduced-rate step'),
            critic=dict(target='unchanged native terminal return, gamma=1',
                        epochs=self.critic_epochs, batch='all recorded decisions',
                        loss='event-masked squared error divided by active-event count',
                        gradient_clip_norm=.5,
                        optimizer='provided by runner; parameters are in run protocol'),
            control_variate=dict(
                enabled=self.mode != 'mc_pg',
                architecture='public-state summary and raw latent -> 64 tanh -> 32 tanh -> scalar',
                state_input='local mean/std(unbiased=False)/min/max plus global',
                state_scaling='signed log1p absolute value', action_scaling='tanh(raw latent)',
                initialization='zero final linear layer; private forked torch seed 741000+seed',
                used_by_actor='frozen parameters present at batch entry',
                fitting_policy='frozen pre-update actor snapshot',
                fitting_time='after actor estimate/update; fitted model first used on next batch',
                objective=('none' if self.mode == 'mc_pg' else
                    'mean squared norm of complete-trajectory policy gradient' if self.mode == 'lax'
                    else 'event-masked Monte Carlo residual MSE'),
                gradient_scope='all trainable actor parameters; no random projection',
                temporal_scope='complete trajectory before squared norm; includes cross-time terms',
                trajectory_weighting='equal complete-episode weights; steps are not independent labels',
                epochs=0 if self.mode == 'mc_pg' else self.auxiliary_epochs,
                optimizer='Adam', learning_rate=float(auxiliary_group['lr']),
                betas=list(auxiliary_group['betas']), eps=float(auxiliary_group['eps']),
                weight_decay=float(auxiliary_group['weight_decay']),
                amsgrad=bool(auxiliary_group['amsgrad']), gradient_clip_norm=.5),
            privileged_labels=False, additional_environment_interactions=0,
            theory_boundary=('Raw fixed-control-variate score/pathwise cancellation only; '
                'not unbiased PPO, not unbiased clipped/Adam parameter updates, '
                'and no guaranteed variance reduction or policy improvement'))

    def get_extra_state(self):
        return dict(version=VERSION, mode=self.mode, event_dim=self.event_dim, seed=self.seed,
                    auxiliary_epochs=self.auxiliary_epochs, critic_epochs=self.critic_epochs,
                    max_kl=self.max_kl, updates=self.updates,
                    cv_optimizer=copy.deepcopy(self.cv_optimizer.state_dict()))

    def set_extra_state(self, state):
        for key, expected in dict(version=VERSION, mode=self.mode, event_dim=self.event_dim,
                                  seed=self.seed, auxiliary_epochs=self.auxiliary_epochs,
                                  critic_epochs=self.critic_epochs, max_kl=self.max_kl).items():
            if state[key] != expected:
                raise ValueError('LAX checkpoint mismatch: ' + key)
        self.updates = int(state['updates'])
        self.cv_optimizer.load_state_dict(state['cv_optimizer'])

    def _fit_control(self, old_actor, episodes):
        """Train for the NEXT batch; old_actor is never optimized."""
        if self.mode == 'mc_pg':
            return dict(auxiliary_optimizer_steps=0, auxiliary_loss=0.,
                        auxiliary_gradient_trajectories=0, auxiliary_objective='none')
        params = tuple(self.control.parameters())
        actor_params = tuple(p for p in old_actor.parameters() if p.requires_grad)
        losses, norms = [], []
        gradient_trajectories = 0
        for _ in range(self.auxiliary_epochs):
            self.cv_optimizer.zero_grad(set_to_none=True)
            total = 0.
            # Free each trajectory's second-order graph before building the next.
            for episode in episodes:
                states, raw, mask = episode['states'], episode['raw'], episode['mask']
                if self.mode == 'lax':
                    mean, sigma = gaussian_parameters(old_actor, states)
                    surrogate = lax_surrogate(mean, sigma, raw, episode['advantage'],
                        lambda u: self.control(states, u), mask)
                    gradients = torch.autograd.grad(surrogate, actor_params, create_graph=True,
                                                    allow_unused=True)
                    terms = [g.square().sum() for g in gradients if g is not None]
                    loss = torch.stack(terms).sum()
                    gradient_trajectories += 1
                else:
                    prediction = self.control(states, raw)
                    loss = ((prediction - episode['advantage']).square() * mask).sum(-1).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite control-variate objective')
                loss = loss / len(episodes)
                derivative = torch.autograd.grad(loss, params, allow_unused=True)
                for parameter, gradient in zip(params, derivative):
                    if gradient is not None:
                        if parameter.grad is None:
                            parameter.grad = gradient.detach().clone()
                        else:
                            parameter.grad.add_(gradient.detach())
                total += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(params, .5)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite mixed control-variate derivative')
            self.cv_optimizer.step()
            losses.append(total); norms.append(float(norm.detach()))
        return dict(auxiliary_optimizer_steps=len(losses), auxiliary_loss=float(np.mean(losses)),
                    auxiliary_grad_norm=float(np.mean(norms)),
                    auxiliary_gradient_trajectories=gradient_trajectories,
                    auxiliary_objective=('mean_squared_norm_complete_trajectory_actor_gradient'
                                         if self.mode == 'lax' else 'MC_residual_MSE'),
                    auxiliary_gradient_scope='all_trainable_actor_parameters',
                    auxiliary_training_actor='frozen_pre_update_snapshot')

    def update(self, actor, critic, actor_optimizer, critic_optimizer, batches, rng):
        started = time.perf_counter()
        episodes, data = _tensor_batches(batches, self.device)
        if data['raw'].shape[1] != self.event_dim:
            raise ValueError('Wrong complete latent dimension')
        states = data['states']
        if not bool((data['mask'] > 0).any()):
            return dict(mode=self.mode, optimizer_steps=0, actor_optimizer_steps=0,
                        actor_committed_steps=0, critic_optimizer_steps=0, value_optimizer_steps=0,
                        auxiliary_optimizer_steps=0, active_events=0., likelihood_error=0.,
                        actual_kl=0., macro_kl=0., actor_loss=0., value_loss=0.,
                        actor_grad_norm=0., value_grad_norm=0., clip_fraction=0.,
                        early_stop=False, advantage_std=0., update_seconds=time.perf_counter()-started)
        with torch.no_grad():
            mean, sigma = gaussian_parameters(actor, states)
            error = float(((gaussian_log_prob(mean, sigma, data['raw']) - data['old_logp'])
                           * data['mask']).abs().max())
            if error > 1e-7:
                raise AssertionError(('Stored joint Gaussian likelihood mismatch', error))
            cv_before = self.control(states, data['raw']) if self.mode != 'mc_pg' else torch.zeros_like(data['mask'])
            cv_rms = float(torch.sqrt((cv_before.square() * data['mask']).sum() / data['mask'].sum()))
        old_actor = copy.deepcopy(actor) if self.mode != 'mc_pg' else None
        # c used in the actor step is exactly the model present at batch entry.
        for parameter in self.control.parameters():
            parameter.requires_grad_(False)
        try:
            def loss_fn():
                mean, sigma = gaussian_parameters(actor, states)
                control = None if self.mode == 'mc_pg' else lambda u: self.control(states, u)
                return -lax_surrogate(mean, sigma, data['raw'], data['advantage'], control, data['mask'])
            policy_stats = kl_step(actor, actor_optimizer, states, loss_fn, self.max_kl)
        finally:
            for parameter in self.control.parameters():
                parameter.requires_grad_(True)
        auxiliary_stats = self._fit_control(old_actor, episodes)
        value_stats = fit_mc_critic(critic, critic_optimizer, states, data['raw'],
                                   data['returns'], data['mask'], self.critic_epochs)
        self.updates += 1
        active = data['mask'] > 0
        return dict(mode=self.mode, version=VERSION, **policy_stats, **value_stats, **auxiliary_stats,
                    optimizer_steps=policy_stats['actor_optimizer_steps'],
                    likelihood_error=error, active_events=float(data['mask'].sum()),
                    advantage_std=float(data['advantage'][active].std(unbiased=False)),
                    advantage_normalization='none', cv_before_actor_rms=cv_rms,
                    cv_actor_training_batches=self.updates - 1,
                    trajectory_count=len(episodes), update_seconds=time.perf_counter()-started,
                    estimator_scope='native_MC_or_LAX_at_recorded_policy; not_unbiased_PPO_or_Adam')


__all__ = ['Method', 'ControlVariate', 'gaussian_parameters', 'gaussian_log_prob',
           'gaussian_kl', 'lax_surrogate', 'kl_step', 'fit_mc_critic', 'state_features']
