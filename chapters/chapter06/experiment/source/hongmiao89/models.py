"""Identity-safe 89-object models with unchanged native learning mathematics."""
import hashlib
import json
import numpy as np
import torch
from allocation23.methods.policy import Actor
from allocation_morl.models import PreferenceSpec, DualValue
from .constants import DOMAIN


def identity(spec):
    if getattr(spec, 'domain', None) != DOMAIN or spec.groups != 89:
        raise ValueError('Only a true reduced89 specification is accepted')
    payload = json.dumps(spec.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return torch.tensor(list(hashlib.sha256(payload.encode()).digest()), dtype=torch.uint8)


class IdentitySafe:
    def load_state_dict(self, state_dict, strict=True, assign=False):
        expected = self._reduced89_identity.detach().cpu()
        found = state_dict.get('_reduced89_identity')
        if found is None or not torch.equal(found.detach().cpu(), expected):
            raise ValueError('Checkpoint identity mismatch; old93 and other-task models are not loadable')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class Actor89(IdentitySafe, Actor):
    def __init__(self, spec, seed):
        super().__init__(spec, 'A_FULL', 'adaptive', int(seed))
        from .method_bridge import owned_method
        owned_method('initialization').initialize_output_heads(self, int(seed))
        # The same fixed buffer is used by sampling, log_prob, behavior density and PPO.
        self.spatial.sigma.mul_(float(spec.spatial_sigma_scale))
        self._chapter5_initialization_record.update(spatial_sigma_scale=float(spec.spatial_sigma_scale),
            spatial_sigma_after_scaling=self.spatial.sigma.detach().cpu().tolist(),
            quota_sigma_unchanged=True, mean_parameter_draws_unchanged=True)
        self.register_buffer('_reduced89_identity', identity(spec))

    def decode(self, state, z, quota):
        if self.spec.adapter == 'unit':
            return super().decode(state,z,quota)
        from .depth_decoder import weighted_depth_projection
        data = self._prepare(state)
        z = torch.as_tensor(z,dtype=self.reference.dtype,device=self.reference.device)
        if z.ndim == 1: z=z[None]
        q=torch.as_tensor(quota,dtype=z.dtype,device=z.device).reshape(-1)
        if (z.shape != (len(data['local']),self.spatial.event_dim) or
            not torch.isfinite(z).all() or q.shape != data['budget'].shape or
            not torch.isfinite(q).all() or torch.any(q<0) or torch.any(q>data['budget'])):
            raise ValueError('Invalid complete depth event or physical total')
        lower,_=self._full_spend_interval(data)
        if torch.any(q<lower): raise ValueError('Depth total violates final spending contract')
        scores=self.spatial_controls(data,z)['scores'].detach().cpu().numpy()
        caps=data['caps'].detach().cpu().numpy();totals=q.detach().cpu().numpy()
        areas=np.asarray(self.spec.area_ha,dtype=np.float64)
        if not np.allclose(data['local'][:,:,0].detach().cpu().numpy(),areas/areas.mean(),rtol=0,atol=1e-12):
            raise ValueError('Depth geometry differs from observed area/order')
        return np.stack([weighted_depth_projection(score,areas,cap,float(total))
                         for score,cap,total in zip(scores,caps,totals)])

    def _full_spend_interval(self, data):
        from .full_spend import irrigation_full_spend_bounds
        intervals = np.array([irrigation_full_spend_bounds(
            {key:value[index].detach().cpu().numpy() for key,value in data.items()}, self.spec)
            for index in range(len(data['local']))], dtype=np.float64)
        return data['budget'].new_tensor(intervals[:,0]), data['budget'].new_tensor(intervals[:,1])


class DualValue89(IdentitySafe, DualValue):
    def __init__(self, spec, seed):
        super().__init__(spec, int(seed))
        self.register_buffer('_reduced89_identity', identity(spec))


def make_models(physical_spec, seed, method='fixed'):
    if method != 'fixed':
        raise ValueError('Only the explicit existing fixed-preference Full method is prepared')
    identity(physical_spec)
    spec = PreferenceSpec(physical_spec)
    return Actor89(spec, seed), DualValue89(spec, seed), spec


def make_learner(neural_spec, event_dim, seed, *, weight, tau, device='cpu'):
    """Construct the chapter-six-owned Full learner without starting training."""
    identity(neural_spec)
    from .method_bridge import owned_method
    return owned_method('learner').Learner(neural_spec, event_dim, seed, device,
        method='fixed', weight=weight, tau=tau)
