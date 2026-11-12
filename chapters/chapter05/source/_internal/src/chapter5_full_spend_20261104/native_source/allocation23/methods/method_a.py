"""Method A: featurewise multiscale coefficients and simplex projection.

The projection is implemented in policy.py. This module constructs spatial
coefficients and the registered fixed random-dictionary controls for ablation.
"""
import math
import torch

from .networks import CompactGaussian

MULTISCALE_VARIANCE_FLOOR = 1e-4


class MultiScaleCompact(CompactGaussian):
    """Six fixed nonlinear basis functions for every public local coordinate.

    The base class's small random dictionary buffers remain solely for strict
    checkpoint compatibility; this overridden dictionary never reads them.
    """
    def __init__(self, spec, seed):
        super().__init__(spec, seed, k=6 * spec.local_dim)

    def dictionary(self, state):
        values = state['local']
        centered = values - values.mean(1, keepdim=True)
        z = centered / torch.sqrt(centered.square().mean(1, keepdim=True)
                                  + MULTISCALE_VARIANCE_FLOOR)
        features = torch.cat((z, torch.tanh(z), torch.tanh(4 * z),
                              torch.tanh(4 * (z - .5)), torch.tanh(4 * (z + .5)),
                              z.square() / (1 + z.square())), dim=-1)
        features = features - features.mean(1, keepdim=True)
        return features / torch.sqrt(features.square().mean(1, keepdim=True)
                                     + MULTISCALE_VARIANCE_FLOOR)


class MatchedRandomCompact(CompactGaussian):
    """Nested random features on A's standardized input/column convention."""
    def __init__(self, spec, seed, k):
        super().__init__(spec, seed, k=k)
        full_k = 6 * spec.local_dim
        if k not in (8, full_k):
            raise ValueError('Expected eight or 6*local_dim nested dictionary columns')
        generator = torch.Generator().manual_seed(spec.dictionary_seed)
        projection = torch.randn(spec.local_dim, full_k, generator=generator,
                                 dtype=torch.float64) / math.sqrt(spec.local_dim)
        bias = .3 * torch.randn(full_k, generator=generator, dtype=torch.float64)
        self.dictionary_projection = projection[:, :k].clone()
        self.dictionary_bias = bias[:k].clone()

    def dictionary(self, state):
        values = state['local']
        centered = values - values.mean(1, keepdim=True)
        scaled = centered / torch.sqrt(centered.square().mean(1, keepdim=True)
                                       + MULTISCALE_VARIANCE_FLOOR)
        features = torch.tanh(scaled @ self.dictionary_projection + self.dictionary_bias)
        features = features - features.mean(1, keepdim=True)
        return features / torch.sqrt(features.square().mean(1, keepdim=True)
                                     + MULTISCALE_VARIANCE_FLOOR)
