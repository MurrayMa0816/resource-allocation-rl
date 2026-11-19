"""A's feasible decoder with an explicit algorithm preference coordinate.

The frozen physical specification and observation interface are never edited.
Both fixed-preference and conditioned arms use the identical neural capacity.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn

from allocation23.methods import make_actor, make_value
from allocation23.methods.networks import SharedEncoder, prepare
from .initialization import initialize_output_heads


class PreferenceSpec:
    """Read-only neural view of a valid physical Spec, with one extra input."""

    def __init__(self, physical_spec):
        self.physical_spec = physical_spec
        self.global_dim = int(physical_spec.global_dim) + 1

    def __getattr__(self, name):
        # Avoid recursive lookup while copy/deepcopy reconstructs this proxy.
        physical = self.__dict__.get('physical_spec')
        if physical is None:
            raise AttributeError(name)
        return getattr(physical, name)

    def to_dict(self):
        result = copy.deepcopy(self.physical_spec.to_dict())
        result.update(global_dim=self.global_dim,
            physical_global_dim=self.physical_spec.global_dim,
            feature_interface='unchanged physical state plus fixed algorithm preference',
            algorithm_preference_fields=['coverage_weight'],
            preference_is_environment_observation=False)
        return result


class CoverageValue(nn.Module):
    def __init__(self, spec, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(97000 + int(seed))
            self.encoder = SharedEncoder(spec.local_dim)
            self.head = nn.Sequential(nn.Linear(32 + spec.global_dim, 32), nn.Tanh(),
                                      nn.Linear(32, 1))
        self.double()

    def forward(self, states, raw=None):
        data = prepare(states, next(self.parameters()))
        _, context = self.encoder(data)
        return self.head(context).sigmoid()


class DualValue(nn.Module):
    """Independent values in raw target units, ordered (coverage, quality)."""
    def __init__(self, spec, seed):
        super().__init__()
        self.coverage = CoverageValue(spec, seed)
        self.quality = make_value(spec, seed)
        self.double()

    def forward(self, states, raw=None):
        return torch.cat((self.coverage(states), self.quality(states)), dim=-1)


def make_models(physical_spec, seed, method):
    if method not in ('fixed', 'conditioned'):
        raise ValueError('Expected fixed or conditioned multiobjective policy')
    spec = PreferenceSpec(physical_spec)
    actor = make_actor(spec, 'A', 3, int(seed)).to('cpu')
    initialize_output_heads(actor, int(seed))
    critic = DualValue(spec, int(seed)).to('cpu')
    return actor, critic, spec


def add_preference(state, weight):
    """Append an objective configuration to a copy; never change public state."""
    w = float(weight)
    if not np.isfinite(w) or not 0 <= w <= 1:
        raise ValueError('Coverage weight must be finite and in [0,1]')
    result = {k: np.asarray(v).copy() for k, v in state.items()}
    g = result['global']
    tail = np.full(g.shape[:-1] + (1,), w, dtype=np.float64)
    result['global'] = np.concatenate((g, tail), axis=-1)
    return result
