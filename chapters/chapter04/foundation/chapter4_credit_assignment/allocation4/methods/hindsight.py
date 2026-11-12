"""Hindsight estimators from ordinary completed terminal-reward episodes.

COCOA follows Meulemans et al. (2023), App. D.2/G.4: a joint-versus-product
classifier and an independently sampled action integral.  Its learned,
one-batch-lagged, bounded density ratio is an approximation, NOT an unbiased
oracle.  The outcome is the native scalar return, only affinely rescaled.

CCA uses only subsequent *public exogenous* observations as its hindsight
covariate.  The allowlist is specific to the documented task contract: radio
channel/J and emergency total registered demand evolve independently of the
allocation.  Receipt counts, backlog, budgets, reward and actions are never
part of that future covariate.  The actor still sees current state only.

Both auxiliary learners are fitted AFTER the policy update, so the update's
own return labels cannot enter the fitted model used to evaluate that update.
No simulator, conditional reset, private outcome or oracle dependency exists.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


STATE_KEYS = ('local', 'global', 'caps', 'budget', 'remaining')
DEFAULTS = dict(candidates=8, auxiliary_epochs=12, auxiliary_batch_size=256,
                auxiliary_lr=1e-3, auxiliary_grad_clip=1., max_log_ratio=5.,
                max_kl=.01, critic_epochs=4)


def _tensor_batch(batches, device):
    """Keep this public contract independent of runner-specific extra fields."""
    if not batches:
        raise ValueError('At least one complete episode is required')
    convert = lambda value: torch.as_tensor(value, dtype=torch.float64, device=device).detach()
    lengths = [len(b['returns']) for b in batches]
    if any(n < 1 for n in lengths):
        raise ValueError('Empty episode')
    states = {k: convert(np.concatenate([np.asarray(b['states'][k]) for b in batches]))
              for k in STATE_KEYS}
    data = {k: convert(np.concatenate([np.asarray(b[k]) for b in batches]))
            for k in ('raw', 'old_logp', 'mask', 'values', 'returns')}
    n = sum(lengths)
    if any(tuple(data[k].shape) != (n, 1) for k in ('old_logp', 'mask', 'values')):
        raise ValueError('One scalar complete policy event per decision is required')
    if data['returns'].shape != (n,) or data['raw'].ndim != 2 or len(data['raw']) != n:
        raise ValueError('Incorrect return or raw-event shape')
    if not all(torch.isfinite(v).all() for v in (*states.values(), *data.values())):
        raise ValueError('Finite complete trajectories required')
    if not torch.all((data['mask'] == 0.) | (data['mask'] == 1.)):
        raise ValueError('Activity mask must be binary')
    start = 0
    for length in lengths:
        ret = data['returns'][start:start + length]
        if not torch.equal(ret, ret[:1].expand_as(ret)):
            raise ValueError('These methods require undiscounted terminal-only returns')
        start += length
    return dict(states=states, lengths=lengths, **data)


def state_features(states):
    x = states['local']
    return torch.cat((x.mean(1), x.std(1, unbiased=False), x.amin(1), x.amax(1),
                      states['global']), -1)


def _domain(spec):
    domain = getattr(spec, 'study_domain', None)
    if domain is None:
        domain = {'communication_image_fullstate': 'radio',
                  'emergency_allocation_v5': 'v5'}.get(spec.domain, spec.domain)
    if domain not in ('radio', 'v5'):
        raise ValueError('No exogeneity contract for this task')
    return domain


def _moments(x, dim):
    return torch.cat((x.mean(dim), x.std(dim, unbiased=False), x.amin(dim), x.amax(dim)), -1)


def future_exogenous_features(spec, states, lengths):
    """Suffix summary of later observed exogenous fields; never a prediction.

    Each row uses t+1,...,T-1 of its OWN episode, excluding its current row and
    all terminal private information.  Empty suffixes have an all-zero code.
    Radio: arcsinh(real/imag current CSI), log(current J).
    Emergency: log1p(total registered demand / 100), not remaining backlog.
    """
    domain = _domain(spec)
    local = states['local']
    if sum(lengths) != len(local) or any(n < 1 for n in lengths):
        raise ValueError('Episode boundaries do not cover the states')
    if domain == 'radio':
        if local.shape[-1] != 3 or states['global'].shape[-1] < 2:
            raise ValueError('Radio public channel/J contract mismatch')
        observed = torch.cat((_moments(local[..., :2], 1), states['global'][:, 1:2]), -1)
    else:
        if local.shape[-1] != 2:
            raise ValueError('Emergency public total-demand contract mismatch')
        observed = _moments(local[..., 1:2], 1)
    # Four temporal moments plus the last subsequent observation and suffix length.
    result = local.new_zeros((len(local), 5 * observed.shape[-1] + 1))
    offset = 0
    for length in lengths:
        for t in range(length - 1):
            suffix = observed[offset + t + 1:offset + length]
            result[offset + t] = torch.cat((_moments(suffix, 0), suffix[-1],
                                             local.new_tensor([(length - t - 1) / length])))
        offset += length
    return result.detach()


def _new_mlp(n_in, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = nn.Sequential(nn.Linear(n_in, 128), nn.Tanh(), nn.Linear(128, 64),
                              nn.Tanh(), nn.Linear(64, 1)).double()
        nn.init.zeros_(model[-1].weight)
        nn.init.zeros_(model[-1].bias)
    return model


def contribution_from_logits(logits, max_log_ratio=5.):
    """Equal-class joint/product discrimination: D/(1-D)-1 = exp(logit)-1.

    A declared numerical bound induces bias when it binds; callers log it.
    """
    if max_log_ratio <= 0:
        raise ValueError('Positive finite log-ratio bound required')
    bounded = logits.clamp(-max_log_ratio, max_log_ratio)
    return torch.expm1(bounded), (bounded != logits)


def independent_candidates(mean, sigma, count, generator):
    """No observed-action argument: the action integral uses fresh randomness."""
    if count < 1 or mean.shape != sigma.shape or torch.any(sigma <= 0):
        raise ValueError('Valid diagonal Gaussian and positive sample count required')
    eps = torch.randn((len(mean), count, mean.shape[-1]), generator=generator,
                      dtype=mean.dtype, device='cpu').to(mean.device)
    return (mean[:, None, :] + sigma[:, None, :] * eps).detach()


def _normal_logprob(raw, mean, sigma):
    return (-.5 * ((raw - mean) / sigma).square() - torch.log(sigma)
            - .5 * math.log(2. * math.pi)).sum(-1)


def cocoa_loss(mean, sigma, candidates, coefficients, raw, terminal_advantage,
               terminal, mask):
    """Continuous COCOA score estimator; coefficients are detached R*w / K.

    The terminal action receives its native immediate-reward score term.
    Earlier decisions use *independent* candidate actions.  Full-N averaging
    matches the common MC backend; inactive decisions contribute exactly zero.
    """
    lp_candidates = _normal_logprob(candidates, mean[:, None], sigma[:, None])
    lp_actual = _normal_logprob(raw, mean, sigma)
    contributions = ((lp_candidates * coefficients.detach()).mean(-1) * (~terminal)
                     + lp_actual * terminal_advantage.detach() * terminal)
    return -(contributions * mask.reshape(-1)).mean()


def _policy_hash(actor):
    digest = hashlib.sha256()
    for name, value in actor.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(v) for v in value)
    return copy.deepcopy(value)


class _Base:
    def __init__(self, spec, event_dim, seed, device, *, config=None):
        self.spec, self.event_dim, self.seed = spec, int(event_dim), int(seed)
        self.device = torch.device(device)
        self.config = dict(DEFAULTS)
        if config:
            unknown = set(config) - set(DEFAULTS)
            if unknown:
                raise ValueError(f'Unknown method options: {sorted(unknown)}')
            self.config.update(config)
        self.generator = torch.Generator(device='cpu').manual_seed(650301 + self.seed)
        self.updates = self.fit_updates = self.auxiliary_steps = 0
        self.source_policy_hash = None

    def metadata(self):
        return dict(method=self.kind, implementation='native-terminal-hindsight-v1',
            configuration=dict(self.config), complete_latent_event_dim=self.event_dim,
            auxiliary_parameters=sum(p.numel() for p in self.model.parameters()),
            model_timing='previous completed batches only during actor update; fit current batch afterwards',
            auxiliary_warmup='first batch uses matched native MC policy gradient',
            actor_backend='unclipped full-batch score gradient; exact Gaussian KL-constrained Adam step',
            advantage_normalization='none', extra_environment_calls=0,
            future_actor_input=False, native_reward_unchanged=True,
            estimator_caveat=('learned lagged bounded density ratio; not an unbiased oracle'
                              if self.kind == 'cocoa' else
                              'unbiased score-baseline subtraction requires documented conditional action independence; optimizer not unbiased'),
            outcome_encoding=('invertible affine native terminal return; no binning or substitute target'
                              if self.kind == 'cocoa' else 'native terminal-return regression'),
            future_covariates=('none' if self.kind != 'cca' else
                'later publicly observed CSI/J for radio or total registered demand for emergency; suffix excludes current row and terminal private state'))

    def _validate(self, actor, batches):
        data = _tensor_batch(batches, self.device)
        if data['raw'].shape[-1] != self.event_dim:
            raise ValueError('Complete event dimension differs from actor')
        if any(length != self.spec.horizon for length in data['lengths']):
            raise ValueError('Complete task episodes are required')
        with torch.no_grad():
            err = float(((actor.log_prob(data['states'], data['raw']) - data['old_logp'])
                         * data['mask']).abs().max())
        if err > 1e-7:
            raise ValueError(f'Stale or incorrect stored policy likelihood: {err}')
        data['likelihood_error'] = err
        return data

    def _critic(self, critic, optimizer, data):
        from .lax import fit_mc_critic
        return fit_mc_critic(critic, optimizer, data['states'], data['raw'],
                             data['returns'], data['mask'], epochs=self.config['critic_epochs'])

    def _fit_regression(self, features, target):
        n = len(target)
        history = []
        before = self.auxiliary_steps
        self.model.train()
        for _ in range(self.config['auxiliary_epochs']):
            order = torch.randperm(n, generator=self.generator)
            for start in range(0, n, self.config['auxiliary_batch_size']):
                ix = order[start:start + self.config['auxiliary_batch_size']].to(self.device)
                loss = (self.model(features[ix]).reshape(-1) - target[ix]).square().mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['auxiliary_grad_clip'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite auxiliary gradient')
                self.optimizer.step()
                self.auxiliary_steps += 1
                history.append(float(loss.detach()))
        self.model.eval()
        return dict(auxiliary_fit_loss=float(np.mean(history)),
                    auxiliary_optimizer_steps=self.auxiliary_steps - before,
                    auxiliary_train_examples=n * self.config['auxiliary_epochs'])

    def state_dict(self):
        return _cpu_state(dict(version=1, kind=self.kind, config=self.config,
            model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
            generator=self.generator.get_state(), updates=self.updates,
            fit_updates=self.fit_updates, auxiliary_steps=self.auxiliary_steps,
            source_policy_hash=self.source_policy_hash))

    def load_state_dict(self, saved):
        if saved['version'] != 1 or saved['kind'] != self.kind or saved['config'] != self.config:
            raise ValueError('Auxiliary checkpoint contract mismatch')
        self.model.load_state_dict(saved['model'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.generator.set_state(saved['generator'].cpu())
        self.updates, self.fit_updates = int(saved['updates']), int(saved['fit_updates'])
        self.auxiliary_steps = int(saved['auxiliary_steps'])
        self.source_policy_hash = saved['source_policy_hash']
        self.model.eval()

    get_state = state_dict


class CocoaMethod(_Base):
    """Contrastive continuous-action COCOA with explicitly lagged ratio model."""
    kind = 'cocoa'

    def __init__(self, spec, event_dim, seed, device, *, config=None):
        super().__init__(spec, event_dim, seed, device, config=config)
        self.feature_dim = 4 * spec.local_dim + spec.global_dim
        # Policy mean/logstd are context, not a claim that policy lag disappears.
        self.model = _new_mlp(self.feature_dim + 3 * self.event_dim + 1,
                              750301 + self.seed).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['auxiliary_lr'])
        self.source_actor_state = None

    def state_dict(self):
        result = super().state_dict()
        result['source_actor_state'] = _cpu_state(self.source_actor_state)
        return result

    def load_state_dict(self, saved):
        super().load_state_dict(saved)
        self.source_actor_state = _cpu_state(saved.get('source_actor_state'))

    get_state = state_dict

    def _inputs(self, features, raw, mean, sigma, returns):
        low, high = self.spec.reward_range
        outcome = 2. * (returns.reshape(-1, 1) - low) / (high - low) - 1.
        return torch.cat((features, (raw - mean) / sigma, mean, sigma.log(), outcome), -1).detach()

    def _fit_contrastive(self, features, raw, mean, sigma, returns):
        n = len(raw)
        positive = self._inputs(features, raw, mean, sigma, returns)
        history, before = [], self.auxiliary_steps
        self.model.train()
        for _ in range(self.config['auxiliary_epochs']):
            # Product distribution: same (s,R), fresh action from THAT policy.
            negative_raw = independent_candidates(mean, sigma, 1, self.generator)[:, 0]
            negative = self._inputs(features, negative_raw, mean, sigma, returns)
            order = torch.randperm(n, generator=self.generator)
            for start in range(0, n, self.config['auxiliary_batch_size']):
                ix = order[start:start + self.config['auxiliary_batch_size']].to(self.device)
                pos, neg = self.model(positive[ix]), self.model(negative[ix])
                loss = .5 * (F.softplus(-pos).mean() + F.softplus(neg).mean())
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['auxiliary_grad_clip'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite contribution-model gradient')
                self.optimizer.step()
                self.auxiliary_steps += 1
                history.append(float(loss.detach()))
        self.model.eval()
        return dict(auxiliary_fit_loss=float(np.mean(history)),
                    auxiliary_optimizer_steps=self.auxiliary_steps - before,
                    auxiliary_train_examples=2 * n * self.config['auxiliary_epochs'])

    def update(self, actor, critic, actor_optimizer, critic_optimizer, batches, rng):
        from .lax import gaussian_parameters, gaussian_kl, kl_step
        started = time.perf_counter()
        data = self._validate(actor, batches)
        state, raw, mask, returns = data['states'], data['raw'], data['mask'], data['returns']
        features = state_features(state).detach()
        source_hash = _policy_hash(actor)
        n, k = len(raw), self.config['candidates']
        with torch.no_grad():
            old_mean, old_sigma = [x.detach().clone() for x in gaussian_parameters(actor, state)]
            mc_advantage = returns - data['values'].reshape(-1)
        if not mask.any():
            return dict(method=self.kind, optimizer_steps=0, active_events=0., extra_environment_calls=0,
                        auxiliary_optimizer_steps=0, elapsed_seconds=time.perf_counter() - started)
        warmup = self.fit_updates == 0
        source_state = _cpu_state(actor.state_dict())
        policy_lag_kl = 0.
        if self.source_actor_state is not None:
            source_actor = copy.deepcopy(actor)
            source_actor.load_state_dict(self.source_actor_state)
            with torch.no_grad():
                source_mean, source_sigma = gaussian_parameters(source_actor, state)
                policy_lag_kl = float(gaussian_kl(source_mean, source_sigma, old_mean, old_sigma).mean())
            del source_actor
        stats = dict(method=self.kind, auxiliary_warmup_mc=warmup, likelihood_error=data['likelihood_error'],
                     extra_environment_calls=0, active_events=float(mask.sum()),
                     outcome_encoding='fixed-invertible-affine-native-terminal-return',
                     estimator='learned-lagged-bounded-ratio; not unbiased oracle',
                     model_fitted_before_current_batch=not warmup,
                     model_source_policy_sha256=self.source_policy_hash,
                     current_behavior_policy_sha256=source_hash, candidate_action_samples=k,
                     previous_fit_policy_to_current_kl=policy_lag_kl,
                     advantage_normalization='none')
        if warmup:
            def loss_fn():
                return -(actor.log_prob(state, raw).reshape(-1) * mc_advantage * mask.reshape(-1)).mean()
            stats.update(ratio_clip_fraction=0., ratio_rms=0., candidate_policy_queries=0)
        else:
            candidates = independent_candidates(old_mean, old_sigma, k, self.generator)
            with torch.no_grad():
                expanded = lambda x: x[:, None].expand(-1, k, *x.shape[1:]).reshape(n * k, *x.shape[1:])
                inputs = self._inputs(expanded(features), candidates.reshape(n * k, self.event_dim),
                                      expanded(old_mean), expanded(old_sigma), expanded(returns))
                logits = self.model(inputs).reshape(n, k)
                weight, clipped = contribution_from_logits(logits, self.config['max_log_ratio'])
                coefficients = returns[:, None] * weight
                terminal = torch.zeros(n, dtype=torch.bool, device=self.device)
                terminal[torch.as_tensor(np.cumsum(data['lengths']) - 1, device=self.device)] = True
                positive_logits = self.model(self._inputs(features, raw, old_mean, old_sigma, returns))
                calibration = .5 * (F.softplus(-positive_logits).mean() + F.softplus(logits).mean())
            def loss_fn():
                mean, sigma = gaussian_parameters(actor, state)
                return cocoa_loss(mean, sigma, candidates, coefficients, raw,
                                  mc_advantage, terminal, mask)
            stats.update(ratio_clip_fraction=float(clipped.double().mean()),
                ratio_rms=float(weight.square().mean().sqrt()), ratio_mean=float(weight.mean()),
                ratio_absolute_max=float(weight.abs().max()),
                prequential_contrastive_loss=float(calibration),
                candidate_policy_queries=n * k,
                candidate_mean_abs_deviation_from_actual=float((candidates - raw[:, None]).abs().mean()))
        stats.update(kl_step(actor, actor_optimizer, state, loss_fn, max_kl=self.config['max_kl']))
        critic_stats = self._critic(critic, critic_optimizer, data)
        stats.update(critic_stats)
        fit_started = time.perf_counter()
        stats.update(self._fit_contrastive(features, raw, old_mean, old_sigma, returns))
        self.fit_updates += 1
        self.updates += 1
        self.source_policy_hash = source_hash
        self.source_actor_state = source_state
        stats.update(auxiliary_fit_seconds=time.perf_counter() - fit_started,
                     elapsed_seconds=time.perf_counter() - started,
                     ordinary_training_episodes=len(batches), ordinary_training_steps=n,
                     optimizer_steps=stats['actor_optimizer_steps'])
        return stats


class CCAMethod(_Base):
    """Lagged state + action-independent future baseline; state-only control.

    No learned independence adversary is needed for this declared exogenous
    subset.  This does NOT claim that the entire observed future is exogenous.
    """
    def __init__(self, spec, event_dim, seed, device, *, mode='cca', config=None):
        if mode not in ('cca', 'cca_state'):
            raise ValueError('Expected cca or identical-capacity cca_state control')
        self.kind = mode
        super().__init__(spec, event_dim, seed, device, config=config)
        self.state_dim = 4 * spec.local_dim + spec.global_dim
        self.future_dim = 46 if _domain(spec) == 'radio' else 21
        self.model = _new_mlp(self.state_dim + self.future_dim, 850301 + self.seed).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['auxiliary_lr'])

    def features(self, states, lengths):
        future = future_exogenous_features(self.spec, states, lengths)
        if self.kind == 'cca_state':
            future = torch.zeros_like(future)
        return torch.cat((state_features(states), future), -1).detach()

    def update(self, actor, critic, actor_optimizer, critic_optimizer, batches, rng):
        from .lax import kl_step
        started = time.perf_counter()
        data = self._validate(actor, batches)
        state, raw, mask, returns = data['states'], data['raw'], data['mask'], data['returns']
        if not mask.any():
            return dict(method=self.kind, optimizer_steps=0, active_events=0., extra_environment_calls=0,
                        auxiliary_optimizer_steps=0, elapsed_seconds=time.perf_counter() - started)
        features = self.features(state, data['lengths'])
        warmup = self.fit_updates == 0
        with torch.no_grad():
            baseline = (data['values'].reshape(-1) if warmup else self.model(features).reshape(-1))
            advantage = returns - baseline
        def loss_fn():
            return -(actor.log_prob(state, raw).reshape(-1) * advantage * mask.reshape(-1)).mean()
        source_hash = _policy_hash(actor)
        stats = dict(method=self.kind, auxiliary_warmup_mc=warmup, likelihood_error=data['likelihood_error'],
                     extra_environment_calls=0, active_events=float(mask.sum()),
                     future_information='observed-public-exogenous-suffix-only' if self.kind == 'cca' else 'none-state-only-control',
                     model_fitted_before_current_batch=not warmup,
                     model_source_policy_sha256=self.source_policy_hash,
                     current_behavior_policy_sha256=source_hash,
                     prequential_return_mse=float((baseline - returns).square().mean()),
                     baseline_rms=float(baseline.square().mean().sqrt()),
                     advantage_std=float(advantage[mask.reshape(-1).bool()].std(unbiased=False)),
                     candidate_policy_queries=0, advantage_normalization='none')
        stats.update(kl_step(actor, actor_optimizer, state, loss_fn, max_kl=self.config['max_kl']))
        critic_stats = self._critic(critic, critic_optimizer, data)
        stats.update(critic_stats)
        fit_started = time.perf_counter()
        stats.update(self._fit_regression(features, returns))
        self.fit_updates += 1
        self.updates += 1
        self.source_policy_hash = source_hash
        stats.update(auxiliary_fit_seconds=time.perf_counter() - fit_started,
                     elapsed_seconds=time.perf_counter() - started,
                     ordinary_training_episodes=len(batches), ordinary_training_steps=len(raw),
                     optimizer_steps=stats['actor_optimizer_steps'])
        return stats
