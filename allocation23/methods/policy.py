"""Complete-event A/B policies with deterministic feasible action decoders.

Training samples the full latent diagonal Gaussian. Deterministic execution
uses its mean BEFORE the nonlinear decoder; it is neither uniform allocation
nor the conditional mean of the final physical action.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from .constraints import bounded_proportional, capped_simplex, gaussian_logprob
from .networks import DirectSpatial, rms, controls as make_controls
from .method_a import MultiScaleCompact, MatchedRandomCompact
from .method_b import BCompact
from ..full_spend import full_spend_bounds

VARIANTS = ('A_FULL', 'A_RANDOM_MATCH', 'A_RANDOM8', 'A_SOFTMAX', 'A_RMS', 'A_DIRECT',
            'B_FULL', 'B_FIXED_BETA', 'B_NO_RMS', 'B_NEITHER', 'B_DIRECT', 'B_GLOBAL_BETA')
TEMPORALS = ('fixed', 'pace', 'schedule', 'budget', 'adaptive',
             'task_no_budget', 'fixed_nominal')
QUOTA_STD_INITIAL = .8
QUOTA_STD_BOUNDS = (.1, 1.5)


class Actor(nn.Module):
    def __init__(self, spec, variant, temporal, seed, *, quota_reference='pace'):
        super().__init__()
        if variant not in VARIANTS or temporal not in TEMPORALS:
            raise ValueError('Unknown registered A/B spatial or temporal variant')
        for name in ('groups', 'horizon', 'local_dim', 'global_dim'):
            value = getattr(spec, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError('Positive public dimensions required')
        for name in ('nominal_total', 'initial_budget'):
            if not math.isfinite(getattr(spec, name)) or getattr(spec, name) <= 0:
                raise ValueError('Positive public execution scales required')
        if quota_reference not in ('pace', 'nominal'):
            raise ValueError('Quota reference must be pace or nominal')
        self.spec = spec
        self.study_variant = self.spatial_variant = variant
        self.temporal_mode, self.quota_reference = temporal, quota_reference
        self.learned_quota = temporal in ('schedule', 'budget', 'adaptive', 'task_no_budget')
        if variant in ('A_FULL', 'A_SOFTMAX', 'A_RMS'):
            self.spatial = MultiScaleCompact(spec, seed)
        elif variant in ('A_RANDOM_MATCH', 'A_RANDOM8'):
            self.spatial = MatchedRandomCompact(spec, seed,
                8 if variant == 'A_RANDOM8' else 6 * spec.local_dim)
        elif variant in ('A_DIRECT', 'B_DIRECT'):
            self.spatial = DirectSpatial(spec, seed, beta=variant == 'B_DIRECT')
        else:
            self.spatial = BCompact(spec, variant, seed)
        self.event_dim = self.spatial.event_dim + int(self.learned_quota)
        self.num_events = 1
        self.projection = variant.startswith('A_') and variant != 'A_SOFTMAX'
        self.register_buffer('reference', torch.tensor(0., dtype=torch.float64))
        # A zero compatibility buffer retained for strict loading of validated
        # checkpoints. Calendar scheduling is not a public method in this package.
        self.register_buffer('calendar', torch.zeros(spec.horizon, dtype=torch.float64))
        if self.learned_quota:
            self.log_quota_sigma = nn.Parameter(torch.tensor(math.log(QUOTA_STD_INITIAL), dtype=torch.float64))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(83100 + int(seed))
                self.quota_head = nn.Sequential(nn.Linear(35, 32), nn.Tanh(), nn.Linear(32, 1))
                nn.init.zeros_(self.quota_head[-1].weight)
                nn.init.zeros_(self.quota_head[-1].bias)
        self.double()

    @property
    def quota_sigma(self):
        if not self.learned_quota:
            return self.reference.new_tensor(0.)
        return self.log_quota_sigma.clamp(math.log(QUOTA_STD_BOUNDS[0]), math.log(QUOTA_STD_BOUNDS[1])).exp()

    def _prepare(self, state):
        keys = ('local', 'global', 'caps', 'budget', 'remaining')
        data = {key: torch.as_tensor(state[key], dtype=self.reference.dtype,
                device=self.reference.device) for key in keys}
        if data['local'].ndim == 2:
            data = {key: value.unsqueeze(0) for key, value in data.items()}
        count = len(data['local'])
        expected = dict(local=(count, self.spec.groups, self.spec.local_dim),
            caps=(count, self.spec.groups), budget=(count,), remaining=(count,))
        expected['global'] = (count, self.spec.global_dim)
        if any(tuple(data[key].shape) != shape for key, shape in expected.items()):
            raise ValueError('Policy input differs from public spec dimensions')
        if not all(torch.isfinite(value).all() for value in data.values()):
            raise ValueError('Finite policy inputs required')
        if (torch.any(data['caps'] < 0) or torch.any(data['budget'] < 0)
                or torch.any(data['remaining'] < 0)
                or torch.any(data['budget'] > data['remaining'] + 1e-10)
                or torch.any(data['budget'] > data['caps'].sum(-1) + 1e-10)):
            raise ValueError('Inconsistent public feasible geometry')
        return data

    def _round(self, data):
        phase = data['global'][:, 0]
        if torch.any((phase < 0) | (phase > 1)):
            raise ValueError('First global feature must be current round / horizon')
        return torch.round(phase * self.spec.horizon).long()

    def quota_mean(self, state):
        if not self.learned_quota:
            raise ValueError('No trainable quota distribution in fixed/calendar policies')
        data = self._prepare(state)
        phase = data['global'][:, :1]
        if self.temporal_mode == 'task_no_budget':
            # Both current encoders put the chapter-3 budget coordinate last.
            # Recompute this branch's context without that path; the spatial
            # branch, critic and feasible decoder still use the real state.
            masked = dict(data)
            masked['global'] = data['global'].clone()
            masked['global'][:, -1] = 0.
            context = self.spatial.context(masked)
            balance = phase.new_zeros((len(phase), 2))
        elif self.temporal_mode == 'schedule':
            context = phase.new_zeros((len(phase), 32))
            balance = phase.new_zeros((len(phase), 2))
        else:
            context = (phase.new_zeros((len(phase), 32)) if self.temporal_mode == 'budget'
                       else self.spatial.context(data))
            balance = torch.stack((data['remaining'] / self.spec.initial_budget,
                                    data['budget'] / self.spec.initial_budget), -1)
        return self.quota_head(torch.cat((context, phase, balance), -1))

    def _full_spend_interval(self, data):
        """Public viability geometry; no state, look-ahead model or RNG added."""
        arrays = {key: value.detach().cpu().numpy() for key, value in data.items()}
        intervals = np.asarray([full_spend_bounds(
            {key: value[index] for key, value in arrays.items()}, self.spec)
            for index in range(len(data['local']))], dtype=np.float64)
        return (data['budget'].new_tensor(intervals[:, 0]),
                data['budget'].new_tensor(intervals[:, 1]))

    def _full_spend_quota(self, state, eta=None):
        data = self._prepare(state)
        period = self._round(data)
        lower, upper = self._full_spend_interval(data)
        left = (self.spec.horizon - period).clamp_min(1)
        if self.temporal_mode in ('fixed', 'fixed_nominal'):
            if eta is not None:
                raise ValueError('Fixed totals have no sampled quota event')
            reference = upper.new_full(upper.shape, self.spec.nominal_total)
            requested = torch.maximum(lower, torch.minimum(reference, upper))
        else:
            if self.temporal_mode == 'pace':
                if eta is not None:
                    raise ValueError('The pace-only arm has no sampled quota event')
                eta = upper.new_zeros(len(upper))
            if eta is None:
                raise ValueError('Recorded quota event required')
            eta = torch.as_tensor(eta, dtype=upper.dtype, device=upper.device).reshape(-1)
            if eta.shape != upper.shape or not torch.isfinite(eta).all():
                raise ValueError('One finite quota event per state required')
            reference = (data['remaining'] / left if self.quota_reference == 'pace'
                         else upper.new_full(upper.shape, self.spec.nominal_total))
            reference = torch.maximum(lower, torch.minimum(reference, upper))
            span = upper-lower
            fraction = torch.where(span > 0, (reference-lower)/span.clamp_min(
                torch.finfo(span.dtype).tiny), torch.zeros_like(span))
            # Interior odds retain eta's former meaning; the endpoint branches
            # retain the exact neutral reference instead of clipping it to .99.
            interior = (fraction > 0) & (fraction < 1)
            safe = torch.where(interior, fraction, torch.full_like(fraction, .5))
            shifted = torch.sigmoid(torch.log(safe)-torch.log1p(-safe)+eta)
            shifted = torch.where(fraction <= 0, torch.tanh(eta).clamp_min(0), shifted)
            shifted = torch.where(fraction >= 1, 1+torch.tanh(eta).clamp_max(0), shifted)
            requested = lower+span*shifted
            requested = torch.where(eta == 0, reference, requested)
            requested = torch.maximum(lower, torch.minimum(requested, upper))
        return torch.where(period < self.spec.horizon, requested, torch.zeros_like(upper))

    def quota(self, state, eta=None):
        if self.spec.chapter == 3:
            return self._full_spend_quota(state, eta)
        # Chapter 2's original fixed-budget code path is retained verbatim.
        data = self._prepare(state)
        if self.temporal_mode == 'pace':
            if eta is not None:
                raise ValueError('The pace-only arm has no sampled quota event')
            eta = data['budget'].new_zeros(len(data['budget']))
        period = self._round(data)
        upper = data['budget']
        if self.temporal_mode in ('fixed', 'fixed_nominal'):
            if eta is not None:
                raise ValueError('Fixed totals have no sampled quota event')
            requested = upper.new_full(upper.shape, self.spec.nominal_total)
        else:
            if eta is None:
                raise ValueError('Recorded quota event required')
            eta = torch.as_tensor(eta, dtype=upper.dtype, device=upper.device).reshape(-1)
            if eta.shape != upper.shape or not torch.isfinite(eta).all():
                raise ValueError('One finite quota event per state required')
            left = (self.spec.horizon - period).clamp_min(1)
            reference = (data['remaining'] / left if self.quota_reference == 'pace'
                         else upper.new_full(upper.shape, self.spec.nominal_total))
            tiny = torch.finfo(upper.dtype).tiny
            fraction = (reference / upper.clamp_min(tiny)).clamp(.01, .99)
            requested = upper * torch.sigmoid(torch.log(fraction) - torch.log1p(-fraction) + eta)
        return torch.where(period < self.spec.horizon, torch.minimum(requested, upper), torch.zeros_like(upper))

    def spatial_controls(self, data, z):
        controls = self.spatial.controls(data, z)
        if self.study_variant == 'A_RMS':
            scores = rms(controls['scores'])
            return make_controls(scores, controls['beta'])
        return controls

    def decode(self, state, z, quota):
        data = self._prepare(state)
        z = torch.as_tensor(z, dtype=self.reference.dtype, device=self.reference.device)
        if z.ndim == 1:
            z = z[None]
        q = torch.as_tensor(quota, dtype=z.dtype, device=z.device).reshape(-1)
        if (z.shape != (len(data['local']), self.spatial.event_dim)
                or not torch.isfinite(z).all() or q.shape != data['budget'].shape
                or not torch.isfinite(q).all() or torch.any(q < 0) or torch.any(q > data['budget'])):
            raise ValueError('Complete finite spatial event and feasible total required')
        if self.spec.chapter == 3:
            lower, _ = self._full_spend_interval(data)
            if torch.any(q < lower):
                raise ValueError('Total would make complete episode spending infeasible')
        control = self.spatial_controls(data, z)
        caps = data['caps'].detach().cpu().numpy()
        totals = q.detach().cpu().numpy()
        if self.projection:
            scores = control['scores'].detach().cpu().numpy()
            return np.stack([capped_simplex(total * (1. + score) / self.spec.groups, cap, float(total))
                             for score, cap, total in zip(scores, caps, totals)])
        weights = control['mean'].detach().cpu().numpy()
        return np.stack([bounded_proportional(weight, cap, float(total))
                         for weight, cap, total in zip(weights, caps, totals)])

    def log_prob(self, state, raw):
        data = self._prepare(state)
        raw = torch.as_tensor(raw, dtype=self.reference.dtype, device=self.reference.device)
        if raw.ndim == 1:
            raw = raw[None]
        if raw.shape != (len(data['local']), self.event_dim) or not torch.isfinite(raw).all():
            raise ValueError('Complete finite joint Gaussian event required')
        if self.learned_quota:
            eta, z = raw[:, :1], raw[:, 1:]
            lp = gaussian_logprob(eta, self.quota_mean(data), self.quota_sigma)
        else:
            z, lp = raw, 0.
        return lp + gaussian_logprob(z, self.spatial.mean(data), self.spatial.sigma)

    def sample(self, state, generator=None, deterministic=False):
        data = self._prepare(state)
        spatial_mean = self.spatial.mean(data)
        if self.learned_quota:
            mean = torch.cat((self.quota_mean(data), spatial_mean), -1)
            sigma = torch.cat((self.quota_sigma.reshape(1), self.spatial.sigma))
        else:
            mean, sigma = spatial_mean, self.spatial.sigma
        raw = mean.detach().clone() if deterministic else (mean + sigma * torch.randn(
            mean.shape, generator=generator, dtype=mean.dtype, device=mean.device)).detach()
        eta, z = (raw[:, :1], raw[:, 1:]) if self.learned_quota else (None, raw)
        selected = self.quota(data, eta)
        actions = self.decode(data, z, selected)
        control = self.spatial_controls(data, z)
        active = data['budget'] > 0 if self.learned_quota else ((selected > 0) & (selected < data['caps'].sum(-1)))
        return dict(raw=raw, log_prob=self.log_prob(data, raw), mask=active.to(mean.dtype)[:, None],
            actions=actions, quota=selected.detach(), quota_sigma=self.quota_sigma.detach(),
            beta=control['beta'].detach(), share_mean=control['mean'].detach())

    def act(self, state):
        """Return deterministic mean-latent decoded action(s), without exploration.

        An unbatched state returns one (N,) NumPy vector; batched input returns
        (batch,N). Normal execution should still apply the public action guard.
        """
        with torch.no_grad():
            result = self.sample(state, deterministic=True)['actions']
        return result[0] if np.ndim(state['local']) == 2 else result

    def metadata(self):
        variant = self.study_variant
        direct = variant in ('A_DIRECT', 'B_DIRECT')
        multi = variant in ('A_FULL', 'A_SOFTMAX', 'A_RMS')
        random_a = variant in ('A_RANDOM_MATCH', 'A_RANDOM8')
        return dict(method_version='allocation23-complete-event-v1', method=variant[0],
            study_variant=variant, spatial_variant=variant, temporal_mode=self.temporal_mode,
            event_dimension=self.event_dim, num_events=1,
            domain=self.spec.domain, groups=self.spec.groups,
            local_dim=self.spec.local_dim, global_dim=self.spec.global_dim,
            dictionary_kind=('fixed-featurewise-multiscale' if multi else
                'fixed-standardized-nested-random-tanh' if random_a else
                'none-direct-regional-score' if direct else getattr(self.spec, 'dictionary_kind', 'raw')),
            dictionary_seed=self.spec.dictionary_seed, dictionary_seed_used=not (multi or direct),
            score_dimensions=self.spec.groups if direct else self.spatial.k,
            independent_beta=getattr(self.spatial, 'learn_beta', False),
            spatial_decoder='bounded-simplex-projection' if self.projection else 'bounded-proportional-softmax',
            learned_decoder=False, complete_policy_event=True, terminal_feedback_only=True,
            quota_reference=self.quota_reference,
            quota_transform='U*sigmoid(logit(clamp(reference/U,.01,.99))+eta)',
            quota_sigma_initial=QUOTA_STD_INITIAL if self.learned_quota else 0.,
            quota_sigma_bounds=list(QUOTA_STD_BOUNDS),
            logprob='Complete joint latent diagonal Gaussian; one PPO likelihood ratio',
            deterministic='Latent Gaussian mean followed by fixed decoder; not mean physical action or uniform allocation',
            temporal_information={
                'fixed': 'Fixed nominal total; no quota event',
                'pace': 'No quota event; eta=0 in the feasible pacing map',
                'schedule': 'Time-only quota mean; balance remains in the feasible map',
                'budget': 'Time, remaining balance and feasible upper in quota mean',
                'adaptive': 'Regional context, time, remaining balance and feasible upper in quota mean',
                'task_no_budget': 'Task context and time in quota mean; all budget input paths masked only in total head',
                'fixed_nominal': 'Fixed nominal total with chapter-3 public state and spatial branch retained',
            }[self.temporal_mode])
