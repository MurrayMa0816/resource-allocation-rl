"""Method B: fixed random features, RMS score normalization and learned beta."""
import math
import torch
from torch import nn

from .networks import CompactGaussian, EPS, BETA_INITIAL_RAW, center, rms, controls


def dictionary_features(local):
    """Derive log channel power from current asinh(real/imag), retaining progress.

This is an algebraic feature transformation, without a channel/success model.
"""
    if local.shape[-1] != 3 or not torch.isfinite(local).all():
        raise ValueError('Finite full current local state with three coordinates required')
    magnitude = local[..., :2].abs()
    log_squared = 2. * (magnitude - math.log(2.)
                       + torch.log(-torch.expm1(-2. * magnitude)))
    terms = torch.cat((torch.zeros_like(log_squared[..., :1]), log_squared), dim=-1)
    log_power = torch.logsumexp(terms, dim=-1)
    return torch.stack((log_power, local[..., 2]), dim=-1)


class BCompact(CompactGaussian):
    def __init__(self, spec, variant, seed):
        self.learn_beta = variant not in ('B_FIXED_BETA', 'B_NEITHER')
        self.score_rms = variant not in ('B_NO_RMS', 'B_NEITHER')
        self.global_beta = variant == 'B_GLOBAL_BETA'
        super().__init__(spec, seed, beta=self.learn_beta)
        self.dictionary_kind = getattr(spec, 'dictionary_kind', 'raw')
        if self.dictionary_kind not in ('raw', 'channel_power'):
            raise ValueError('Unknown fixed dictionary')
        if self.dictionary_kind == 'channel_power':
            if self.local_dim < 3:
                raise ValueError('Power dictionary requires current channel real/imag/count')
            dimension = self.local_dim - 1
            generator = torch.Generator().manual_seed(spec.dictionary_seed)
            self.dictionary_projection = torch.randn(dimension, self.k, generator=generator,
                                                      dtype=torch.float64) / math.sqrt(dimension)
            self.dictionary_bias = .3 * torch.randn(self.k, generator=generator, dtype=torch.float64)
        if self.global_beta:
            previous = self.head
            self.head = nn.Linear(32, self.k, dtype=torch.float64)
            with torch.no_grad():
                self.head.weight.copy_(previous.weight[:self.k])
                self.head.bias.copy_(previous.bias[:self.k])
            self.beta_global_mean = nn.Parameter(torch.tensor(BETA_INITIAL_RAW, dtype=torch.float64))

    def dictionary(self, state):
        if self.dictionary_kind == 'raw':
            return super().dictionary(state)
        local = state['local']
        features = torch.cat((dictionary_features(local[..., :3]), local[..., 3:]), dim=-1)
        features = torch.tanh(features @ self.dictionary_projection + self.dictionary_bias)
        features = features - features.mean(1, keepdim=True)
        return features / (features.square().mean(1, keepdim=True).sqrt() + EPS)

    def _mean(self, state):
        mean = super()._mean(state)
        if self.global_beta:
            mean = torch.cat((mean, self.beta_global_mean.expand(len(mean), 1)), dim=-1)
        return mean

    def controls(self, state, raw):
        data = self._prepare(state)
        raw = torch.as_tensor(raw, dtype=self.sigma.dtype, device=self.sigma.device)
        if raw.ndim == 1:
            raw = raw[None]
        if raw.shape != (len(data['local']), self.event_dim) or not torch.isfinite(raw).all():
            raise ValueError('Complete B spatial event required')
        score = center(torch.einsum('bnk,bk->bn', self.dictionary(data), raw[:, :self.k]))
        if self.score_rms:
            score = rms(score)
        beta = (.15 + 2.85 * torch.sigmoid(raw[:, self.k]) if self.learn_beta
                else raw.new_ones(len(raw)))
        return controls(score * beta[:, None], beta)
