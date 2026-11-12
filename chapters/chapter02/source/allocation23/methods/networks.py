"""Shared Gaussian coefficient networks, direct-score ablation and critic.

Parameter names, construction seeds and float64 conversions preserve the
validated 960-episode checkpoints. No simulator is imported by this module.
"""
from __future__ import annotations

import math
import torch
from torch import nn

EPS = 1e-6
BETA_INITIAL_RAW = math.log(((1. - .15) / 2.85) / (1. - (1. - .15) / 2.85))


def prepare(state, reference):
    data = {key: torch.as_tensor(state[key], dtype=reference.dtype, device=reference.device)
            for key in ('local', 'global', 'caps', 'budget')}
    if data['local'].ndim == 2:
        data = {key: value.unsqueeze(0) for key, value in data.items()}
    return data


def center(scores):
    return scores - scores.mean(-1, keepdim=True)


def rms(scores):
    return scores / (scores.square().mean(-1, keepdim=True).sqrt() + EPS)


def controls(scores, beta):
    if not torch.isfinite(scores).all():
        raise FloatingPointError('Non-finite spatial score')
    return dict(scores=scores, mean=torch.softmax(scores, dim=-1), beta=beta,
                kappa=scores.new_full((len(scores),), 120.))


class CompactGaussian(nn.Module):
    """A shared object encoder outputs one complete Gaussian coefficient event."""

    def __init__(self, spec, seed, *, k=8, beta=False):
        super().__init__()
        self.local_dim, self.global_dim = int(spec.local_dim), int(spec.global_dim)
        if self.local_dim < 1 or self.global_dim < 1:
            raise ValueError('Positive local/global dimensions required')
        self.domain, self.k = spec.domain, int(k)
        self.has_beta = bool(beta)
        self.event_dim = self.k + int(self.has_beta)
        self.num_events = 1
        self.dictionary_seed = int(spec.dictionary_seed)
        d = self.local_dim
        generator = torch.Generator().manual_seed(self.dictionary_seed)
        projection = torch.randn(d, self.k, generator=generator, dtype=torch.float64) / math.sqrt(d)
        bias = .3 * torch.randn(self.k, generator=generator, dtype=torch.float64)
        self.register_buffer('dictionary_projection', projection)
        self.register_buffer('dictionary_bias', bias)
        scales = [1. / math.sqrt(self.k)] * self.k + [.45] * int(self.has_beta)
        self.register_buffer('sigma', torch.tensor(scales, dtype=torch.float64))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31000 + int(seed))
            self.object_net = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh())
            self.context_net = nn.Sequential(nn.Linear(32 + self.global_dim, 64), nn.Tanh(),
                                             nn.Linear(64, 32), nn.Tanh())
            self.head = nn.Linear(32, self.event_dim)
            if self.has_beta:
                with torch.no_grad():
                    # Preserve the original arithmetic used for the bias.
                    value = (1. - .15) / (3. - .15)
                    self.head.bias[self.k] = math.log(value / (1. - value))
        self.double()

    def _prepare(self, state):
        return prepare(state, self.sigma)

    def _mean(self, state):
        encoded = self.object_net(state['local']).mean(1)
        context = self.context_net(torch.cat((encoded, state['global']), dim=-1))
        return self.head(context)

    def mean(self, state):
        return self._mean(self._prepare(state))

    def forward(self, state):
        return self.mean(state)

    def context(self, state):
        pooled = self.object_net(state['local']).mean(1)
        return self.context_net(torch.cat((pooled, state['global']), dim=-1))

    def dictionary(self, state):
        local = state['local']
        features = torch.tanh(local @ self.dictionary_projection + self.dictionary_bias)
        features = features - features.mean(1, keepdim=True)
        return features / (features.square().mean(1, keepdim=True).sqrt() + EPS)

    def controls(self, state, raw):
        data = self._prepare(state)
        raw = torch.as_tensor(raw, dtype=self.sigma.dtype, device=self.sigma.device)
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        if raw.shape != (data['local'].shape[0], self.event_dim) or not torch.isfinite(raw).all():
            raise ValueError('Complete finite Gaussian event required')
        score = torch.einsum('bnk,bk->bn', self.dictionary(data), raw[:, :self.k])
        score = score - score.mean(1, keepdim=True)
        beta = torch.ones(raw.shape[0], dtype=raw.dtype, device=raw.device)
        kappa = torch.full_like(beta, 120.)
        if self.has_beta:
            beta = .15 + 2.85 * torch.sigmoid(raw[:, self.k])
            score = score / (score.square().mean(1, keepdim=True).sqrt() + EPS)
            score = score * beta[:, None]
        mean = torch.softmax(score, dim=-1)
        if not torch.isfinite(mean).all() or torch.any(mean <= 0.):
            raise FloatingPointError('Conditional share mean not strictly positive')
        return dict(mean=mean, beta=beta, kappa=kappa, scores=score)


class DirectSpatial(nn.Module):
    """Registered direct-score ablation; the main A/B methods use compact events."""

    def __init__(self, spec, seed, beta=False):
        super().__init__()
        self.groups = spec.groups
        self.learn_beta = bool(beta)
        self.score_rms = bool(beta)
        self.event_dim = spec.groups + int(beta)
        self.register_buffer('sigma', torch.tensor([1.] * spec.groups + ([.45] if beta else []),
                                                   dtype=torch.float64))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31000 + int(seed))
            self.object_net = nn.Sequential(nn.Linear(spec.local_dim, 64), nn.Tanh(),
                                            nn.Linear(64, 32), nn.Tanh())
            self.context_net = nn.Sequential(nn.Linear(64 + spec.global_dim, 64), nn.Tanh(),
                                             nn.Linear(64, 32), nn.Tanh())
            self.head = nn.Linear(32, 1)
        self.double()
        if beta:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(73100 + int(seed))
                self.beta_head = nn.Linear(32, 1, dtype=torch.float64)
                nn.init.zeros_(self.beta_head.weight)
                nn.init.constant_(self.beta_head.bias, BETA_INITIAL_RAW)

    def mean(self, data):
        local = self.object_net(data['local'])
        pooled = local.mean(1, keepdim=True).expand_as(local)
        global_ = data['global'][:, None, :].expand(-1, local.shape[1], -1)
        score = self.head(self.context_net(torch.cat((local, pooled, global_), -1))).squeeze(-1)
        score = score - score.mean(-1, keepdim=True)
        if self.learn_beta:
            return torch.cat((score, self.beta_head(self.context(data))), dim=-1)
        return score

    def context(self, data):
        pooled = self.object_net(data['local']).mean(1)
        return self.context_net(torch.cat((pooled, pooled, data['global']), -1))

    def controls(self, state, raw):
        if raw.shape != (len(state['local']), self.event_dim) or not torch.isfinite(raw).all():
            raise ValueError('Complete direct spatial event required')
        score = center(raw[:, :self.groups])
        if self.score_rms:
            score = rms(score)
        beta = (.15 + 2.85 * torch.sigmoid(raw[:, self.groups]) if self.learn_beta
                else raw.new_ones(len(raw)))
        return controls(score * beta[:, None], beta)


class SharedEncoder(nn.Module):
    def __init__(self, local_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(local_dim, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh())

    def forward(self, state):
        encoded = self.net(state['local'])
        context = torch.cat((encoded.mean(1), state['global']), -1)
        return encoded, context


class Value(nn.Module):
    """State-only critic; action/caps are not extra neural features."""
    def __init__(self, spec, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(41000 + seed)
            self.encoder = SharedEncoder(spec.local_dim)
            self.head = nn.Sequential(nn.Linear(32 + spec.global_dim, 32), nn.Tanh(), nn.Linear(32, 1))
        self.double()

    def forward(self, state, raw=None):
        data = prepare(state, next(self.parameters()))
        _, context = self.encoder(data)
        return self.head(context)
