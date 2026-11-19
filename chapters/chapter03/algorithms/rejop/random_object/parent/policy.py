"""Core deletion of object-allocation learning in the frozen Chapter 3 method.

The complete Full constructor runs first.  Its object/context encoders, quota
head and quota sigma therefore retain their same-seed initial tensor values.
Only the learned object output head and its Gaussian action event are removed.
Physical actions use the original bounded projection of a zero-score proposal.
"""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'chapter02/scripts/support/portable_paths.py').is_file()) / 'chapter02/scripts/support'
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root

import hashlib
from pathlib import Path
import sys

sys.dont_write_bytecode = True

CAPSULE = Path(__file__).resolve().parents[2]/'native'
ARM = 'A_NO_OBJECT_LEARNING'
RULE = 'RULE_PACE_EQUAL'
SCHEMA = 'chapter3-core-no-object-learning-v1'
EXPECTED = {
    'train_worker.py': '1ead3d4c49133e80f5744bb4739ce763715ddc97d0c8a025975a8482245c36db',
    'direct_policy.py': '579a53eef26b42765a1096ffba2255494361c710c2d8ae82621b6b7c99649edc',
    'source_registry.json': 'd2b0b359d0dc7fe6d53896beabc4bda07bd4131232d157fca7e30ce90c03e8da',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_identity():
    actual = {name: sha(CAPSULE/name) for name in EXPECTED}
    if actual != EXPECTED:
        raise ValueError('Published base capsule changed')
    return dict(capsule=str(CAPSULE), files_sha256=actual)


frozen_identity()
_SOURCE = CAPSULE.parent
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))
import allocation23
from source_runtime import active_view_name
if active_view_name() != 'chapter03.native':
    raise ValueError('Use a fresh process with the published Chapter 3 package')

import numpy as np
import torch
from torch import nn
from allocation23.methods.policy import Actor
from allocation23.methods.constraints import capped_simplex, gaussian_logprob


def _chapter3(spec):
    if spec.chapter != 3 or spec.study_domain not in ('radio', 'v5'):
        raise ValueError('Core ablation requires an original Chapter 3 task')


def _zero_control(owner, data):
    scores = data['local'].new_zeros((len(data['local']), owner.spec.groups))
    return dict(scores=scores, mean=scores.new_full(scores.shape, 1./owner.spec.groups),
                beta=scores.new_ones(len(scores)))


def _equal_decode(owner, state, z, quota):
    data = owner._prepare(state)
    z = torch.as_tensor(z, dtype=owner.reference.dtype, device=owner.reference.device)
    if z.ndim == 1:
        z = z[None]
    q = torch.as_tensor(quota, dtype=owner.reference.dtype,
                        device=owner.reference.device).reshape(-1)
    if (z.shape != (len(data['local']), 0) or not torch.isfinite(z).all()
            or q.shape != data['budget'].shape or not torch.isfinite(q).all()
            or torch.any(q < 0) or torch.any(q > data['budget'])):
        raise ValueError('No object event and one feasible finite total required')
    lower, _ = owner._full_spend_interval(data)
    if torch.any(q < lower):
        raise ValueError('Total would make complete episode spending infeasible')
    caps = data['caps'].detach().cpu().numpy()
    totals = q.detach().cpu().numpy()
    return np.stack([capped_simplex(np.full(owner.spec.groups, float(total)/owner.spec.groups),
                                   cap, float(total))
                     for cap, total in zip(caps, totals)])


class NoObjectActor(Actor):
    """Full task-conditioned learned total, with exactly one eta event."""

    def __init__(self, spec, seed):
        _chapter3(spec)
        super().__init__(spec, 'A_FULL', 'adaptive', seed, quota_reference='pace')
        # Construct first, then delete: surviving tensors exactly match Full.
        # Keeping a frozen head is insufficient because its input encoder learns.
        del self.spatial.head
        self.spatial.event_dim = 0
        self.event_dim = 1
        self.num_events = 1

    def spatial_controls(self, data, z):
        z = torch.as_tensor(z, dtype=self.reference.dtype, device=self.reference.device)
        if z.shape != (len(data['local']), 0) or not torch.isfinite(z).all():
            raise ValueError('The deleted object branch has zero event dimensions')
        return _zero_control(self, data)

    def decode(self, state, z, quota):
        return _equal_decode(self, state, z, quota)

    def log_prob(self, state, raw):
        data = self._prepare(state)
        raw = torch.as_tensor(raw, dtype=self.reference.dtype, device=self.reference.device)
        if raw.ndim == 1:
            raw = raw[None]
        if raw.shape != (len(data['local']), 1) or not torch.isfinite(raw).all():
            raise ValueError('One finite quota Gaussian event per state required')
        return gaussian_logprob(raw, self.quota_mean(data), self.quota_sigma)

    def sample(self, state, generator=None, deterministic=False):
        data = self._prepare(state)
        mean = self.quota_mean(data)
        raw = mean.detach().clone() if deterministic else (mean + self.quota_sigma * torch.randn(
            mean.shape, generator=generator, dtype=mean.dtype, device=mean.device)).detach()
        selected = self.quota(data, raw)
        control = self.spatial_controls(data, raw[:, :0])
        actions = self.decode(data, raw[:, :0], selected)
        # Literal Full deletion: forced eta events remain recorded when budget>0.
        active = data['budget'] > 0
        return dict(raw=raw, log_prob=self.log_prob(data, raw), mask=active.to(mean.dtype)[:, None],
                    actions=actions, quota=selected.detach(), quota_sigma=self.quota_sigma.detach(),
                    beta=control['beta'].detach(), share_mean=control['mean'].detach())

    def metadata(self):
        result = super().metadata()
        result.update(method_version=SCHEMA, method_arm=ARM,
            spatial_variant='zero-score-original-bounded-projection',
            event_dimension=1, score_dimensions=0, learned_object_event_dimensions=0,
            object_output_head_present=False, total_event_dimensions=1,
            dictionary_kind='unused-fixed-buffers-retained-from-Full-construction',
            dictionary_seed_used=False,
            logprob='Only eta diagonal Gaussian; one native PPO likelihood ratio',
            deterministic='Quota Gaussian mean through original Full residual map, then original projection of q/N',
            quota_transform='published Chapter 3 full-spend residual reference map including endpoint branches',
            actor_active_mask='budget>0; unchanged Full convention includes forced-total null eta records',
            object_execution='zero scores; capped_simplex(full(groups,q/groups),caps,q)',
            intervention=dict(learned_total_retained=True, complete_task_context_retained=True,
                              object_output_head_deleted=True, object_gaussian_events_deleted=True,
                              original_projection_retained=True, native_PPO_retained=True),
            unused_fixed_spatial_buffers=['dictionary_projection', 'dictionary_bias', 'sigma'])
        return result


class PaceEqualRule(nn.Module):
    """Parameterless pacing plus zero scores; no Actor training or random event."""

    def __init__(self, spec):
        super().__init__()
        _chapter3(spec)
        self.spec = spec
        self.temporal_mode = self.quota_reference = 'pace'
        self.learned_quota = False
        self.event_dim = self.num_events = 0
        self.register_buffer('reference', torch.tensor(0., dtype=torch.float64))

    # These are the original public geometry and quota equations, unchanged.
    _prepare = Actor._prepare
    _round = Actor._round
    _full_spend_interval = Actor._full_spend_interval
    _full_spend_quota = Actor._full_spend_quota
    quota = Actor.quota
    act = Actor.act

    @property
    def quota_sigma(self):
        return self.reference.new_tensor(0.)

    def decode(self, state, z, quota):
        return _equal_decode(self, state, z, quota)

    def log_prob(self, state, raw):
        data = self._prepare(state)
        raw = torch.as_tensor(raw, dtype=self.reference.dtype, device=self.reference.device)
        if raw.ndim == 1:
            raw = raw[None]
        if raw.shape != (len(data['local']), 0) or not torch.isfinite(raw).all():
            raise ValueError('The untrained rule has zero random event dimensions')
        return self.reference.new_zeros((len(data['local']), 1))

    def sample(self, state, generator=None, deterministic=True):
        data = self._prepare(state)
        raw = self.reference.new_zeros((len(data['local']), 0))
        selected = self.quota(data)
        control = _zero_control(self, data)
        return dict(raw=raw, log_prob=self.log_prob(data, raw),
                    mask=self.reference.new_zeros((len(data['local']), 1)),
                    actions=self.decode(data, raw, selected), quota=selected.detach(),
                    quota_sigma=self.quota_sigma.detach(), beta=control['beta'], share_mean=control['mean'])

    def metadata(self):
        return dict(method_version=SCHEMA, method_arm=RULE, event_dimension=0, num_events=0,
                    groups=self.spec.groups, domain=self.spec.domain, trainable_parameters=0,
                    total_mapping='published feasible pacing map eta=0',
                    object_execution='zero scores; capped_simplex(full(groups,q/groups),caps,q)',
                    independent_training_replicates=0, trained_models=0,
                    uses_current_public_feasible_geometry=True)


def make_actor(spec, seed=0):
    return NoObjectActor(spec, seed)


def load_policy(training_run, target_task, device='cpu'):
    run = Path(training_run).expanduser().resolve()
    model = run if run.is_file() else run/'model/state.pt'
    payload = torch.load(model, map_location='cpu', weights_only=False)
    if payload.get('schema') != SCHEMA or not payload.get('evaluation_only'):
        raise ValueError('Expected a core NoObject evaluation endpoint')
    identity = payload['identity']
    if identity['method'] != ARM or identity['arm_mapping'] != dict(
            variant='A_FULL', temporal='adaptive', quota_reference='pace'):
        raise ValueError('Wrong core endpoint identity')
    algorithm = identity['initial_algorithm']
    if algorithm.get('object_output_head_present') is not False or algorithm.get('event_dimension') != 1:
        raise ValueError('Endpoint does not implement the registered core deletion')
    spec, trained = target_task.spec, identity['spec']
    for key in ('domain', 'chapter', 'groups', 'local_dim', 'global_dim', 'source_packets', 'feature_version'):
        if trained[key] != getattr(spec, key):
            raise ValueError('Incompatible core policy geometry: '+key)
    actor = NoObjectActor(spec, identity['arguments']['seed'])
    actor.load_state_dict(payload['actor'], strict=True)
    actor.to(device=device, dtype=torch.float64)
    actor.eval()
    return actor
