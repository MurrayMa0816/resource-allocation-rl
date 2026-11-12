"""Keep public feasibility fields out of external policy/value neural inputs.

PASPO/RPO retain FP64 caps and total in raw observations for exact geometry.
Their neural state coordinates for those fields are zero. Generic continuous
cores instead receive a reduced vector containing policy_indices only. No
physical model, source image, hidden parameter or action history is imported.
"""
import numpy as np
import torch
from torch import nn
from .common import RunningObservationNorm


class NeuralMask(nn.Module):
    def __init__(self, size, indices):
        super().__init__()
        mask = torch.zeros(size, dtype=torch.float32)
        mask[list(indices)] = 1.
        self.register_buffer('mask', mask)

    def forward(self, value):
        return value * self.mask.to(value)


class GeometryBlindNorm(RunningObservationNorm):
    def __init__(self, size, indices):
        super().__init__(size)
        self.mask = np.zeros(size, dtype=np.float64)
        self.mask[list(indices)] = 1.

    def update(self, values):
        super().update(np.asarray(values) * self.mask)

    def __call__(self, values):
        if isinstance(values, torch.Tensor):
            mask = torch.as_tensor(self.mask, device=values.device, dtype=values.dtype)
        else:
            mask = self.mask
        return super().__call__(values * mask) * mask


class FeatureCore:
    """Raw geometry remains in the environment; replay/neural state is reduced."""
    def __init__(self, core, indices):
        self.delegate = core
        self.policy_indices = np.asarray(indices, dtype=np.int64)

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def act(self, obs, deterministic=False):
        return self.delegate.act(np.asarray(obs)[..., self.policy_indices], deterministic)

    def observe(self, obs, action, nxt, reward, done):
        return self.delegate.observe(np.asarray(obs)[..., self.policy_indices], action,
                                     np.asarray(nxt)[..., self.policy_indices], reward, done)


def public_indices(env):
    contract = env.get_contract()
    size = int(env.observation_space.shape[0])
    schema = contract['observation_schema']
    fields = {field['name']: field for field in schema}
    if set(fields) != {'local', 'global', 'caps', 'budget'}:
        raise ValueError('Only simplified local/global and public caps/budget fields are supported')
    expected = [i for name in ('local', 'global')
                for i in range(fields[name]['start'], fields[name]['stop'])]
    supplied = contract.get('policy_indices', expected)
    if list(supplied) != expected or len(set(expected)) != len(expected):
        raise ValueError('Neural indices must contain exactly local/global in schema order')
    geometry = [i for name in ('caps', 'budget')
                for i in range(fields[name]['start'], fields[name]['stop'])]
    if sorted(expected + geometry) != list(range(size)):
        raise ValueError('Observation schema must partition the complete vector')
    return size, expected, schema


def build_geometry_blind_core(method, env, parameters, seed, device, td7_class, dime_class):
    size, indices, schema = public_indices(env)
    actions = int(env.action_space.shape[0])
    if method in ('td7', 'dime', 'simba_v2'):
        if method == 'td7':
            core = td7_class(len(indices), actions, parameters, device=device)
        elif method == 'dime':
            core = dime_class(len(indices), actions, parameters, device=device, seed=seed)
        else:
            from .chapter2_simbav2 import SimbaV2Core
            core = SimbaV2Core(len(indices), actions, parameters, device=device,
                               observation_schema=schema)
        return FeatureCore(core, indices)
    if method == 'paspo':
        from .chapter2_paspo import PASPOCore
        core = PASPOCore(size, actions, parameters, device=device, observation_schema=schema)
        core.actor.encoder = nn.Sequential(NeuralMask(size, indices).to(core.device), core.actor.encoder)
        core.critic = nn.Sequential(NeuralMask(size, indices).to(core.device), core.critic)
        return core
    if method == 'rpo':
        from .chapter2_rpo import RPOCore
        core = RPOCore(size, actions, parameters, device=device, observation_schema=schema)
        core.obs_norm = GeometryBlindNorm(size, indices)
        return core
    raise ValueError('Unknown external baseline')
