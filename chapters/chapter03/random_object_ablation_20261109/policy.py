"""Independent uniform random object weights with native learned total control.

This is a new retrained development ablation, not a relabeling of equal-object
or frozen-network intervention results. Object randomization remains active in
evaluation; deterministic=True selects only the learned eta Gaussian mean.
"""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root

import hashlib
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
CORE = Path(str(_resolve_path('/root/gamewar/resource_allocation_rl/chapters/chapter03/core_ablation_20261109')))
ARM = 'A_RANDOM_OBJECT_ALLOCATION'
SCHEMA = 'chapter3-random-object-allocation-v1'
RANDOM_NAMESPACE = 202611094901
PHASES = {'train': 1, 'dev': 2, 'final': 3, 'engineering': 4}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    if name not in sys.modules:
        from frozen_adapter import load_module
        load_module(name,path)
    return sys.modules[name]


if sha(CORE/'policy.py') != '06de2bf7bd06e2deac7890568021f4b372cced50946a8470dd65027f0e81e440':
    raise ValueError('The reviewed native no-object parent changed')
core = module('_chapter03_core_policy', CORE/'policy.py')
import numpy as np
import torch
from allocation23.methods.constraints import capped_simplex


def frozen_identity():
    identity = core.frozen_identity()
    identity['core_parent_policy_sha256'] = sha(CORE/'policy.py')
    return identity


def allocation_identity(task, phase, index):
    """Distinct from all environment, eta, shuffle and global RNG streams."""
    if phase not in PHASES or int(index) != index or index < 0:
        raise ValueError('A native phase and nonnegative integer path index required')
    words = [RANDOM_NAMESPACE, {'radio': 1, 'v5': 2}[task.domain],
             int(task.path_master), PHASES[phase], int(index),
             int(task.seed) if phase in ('train', 'engineering') else 0]
    return dict(namespace=RANDOM_NAMESPACE, seed_sequence_words=words,
        bit_generator='PCG64', common_across_training_seeds=phase in ('dev', 'final'),
        draw_rule='N independent Uniform[0,1) values each period, including zero-quota periods; mask only native caps=0')


class RandomObjectActor(core.NoObjectActor):
    """Eta-only trainable policy; the object-allocation kernel has no parameters."""

    def __init__(self, spec, seed):
        super().__init__(spec, seed)
        self._object_rng = None
        self._object_rng_identity = None
        self._object_draw_trace = None

    def random_actions(self, state, quota, uniforms):
        data = self._prepare(state)
        q = torch.as_tensor(quota, dtype=self.reference.dtype,
                            device=self.reference.device).reshape(-1)
        u = np.asarray(uniforms, dtype=np.float64)
        caps = data['caps'].detach().cpu().numpy()
        if u.shape != caps.shape or not np.isfinite(u).all() or np.any(u < 0.) or np.any(u >= 1.):
            raise ValueError('One finite native-N Uniform[0,1) draw per object and batch row required')
        if q.shape != data['budget'].shape or not torch.isfinite(q).all():
            raise ValueError('One finite total per state required')
        lower, upper = self._full_spend_interval(data)
        if torch.any(q < lower) or torch.any(q > upper):
            raise ValueError('The original native feasible total interval is unchanged')
        actions = []
        for cap, total, draw in zip(caps, q.detach().cpu().numpy(), u):
            active = cap > 0.
            if float(total) == 0.:
                actions.append(np.zeros_like(cap))
                continue
            if not active.any():
                raise ValueError('Positive total cannot be assigned without eligible native caps')
            weights = np.where(active, np.maximum(draw, np.finfo(float).tiny), 0.)
            proposal = float(total)*weights/weights.sum()
            actions.append(capped_simplex(proposal, cap, float(total)))
        return np.stack(actions)

    def decode(self, state, z, quota):
        raise ValueError('Random objects require the episode allocation RNG; use sample() through the episode wrapper')

    def sample(self, state, generator=None, deterministic=False):
        if self._object_rng is None:
            raise ValueError('An independently seeded episode object-RNG context is required')
        data = self._prepare(state)
        mean = self.quota_mean(data)
        raw = (mean.detach().clone() if deterministic else
               (mean + self.quota_sigma*torch.randn(mean.shape, generator=generator,
                dtype=mean.dtype, device=mean.device)).detach())
        q = self.quota(data, raw)
        uniforms = self._object_rng.random((len(data['local']), self.spec.groups))
        if self._object_draw_trace is not None:
            self._object_draw_trace.append(uniforms.copy())
        actions = self.random_actions(data, q, uniforms)
        active = data['budget'] > 0
        return dict(raw=raw, log_prob=self.log_prob(data, raw),
            mask=active.to(mean.dtype)[:, None], actions=actions, quota=q.detach(),
            quota_sigma=self.quota_sigma.detach(), beta=mean.new_ones(len(mean)),
            object_random_uniforms=uniforms.copy(),
            share_mean=mean.new_full((len(mean), self.spec.groups), 1./self.spec.groups))

    def metadata(self):
        result = super().metadata()
        result.update(method_version=SCHEMA, method_arm=ARM,
            spatial_variant='independent-normalized-Uniform-weights-native-capped-simplex',
            event_dimension=1, learned_object_event_dimensions=0,
            object_output_head_present=False, random_object_parameters=0,
            deterministic='Eta Gaussian mean only; object allocation remains random using a fixed per-path independent random stream',
            object_execution='draw N iid Uniform[0,1), zero only caps=0, normalize active weights to q, original capped_simplex',
            share_mean_field='Compatibility neutral 1/N field only; not the sampled random object weights or actual actions',
            random_object_namespace=RANDOM_NAMESPACE,
            random_object_probability_in_PPO=False,
            logprob='Eta Gaussian only; parameter-independent object kernel cancels in conditional-state PPO ratio',
            evaluation_scope='retrained development ablation; no independent new TEST claim',
            prior_equal_object_results_reused_as_random=False)
        return result


def episode_with_random_object(task, phase, index, actor, *args, **kwargs):
    """Only bind the parameter-free kernel; native Task physics remains intact.

    Exact step callbacks may invoke DEV while a train episode is paused. Each
    nested call gets its own generator and restores the outer generator object
    and trace by reference, so DEV never advances the train allocation stream.
    """
    if not isinstance(actor, RandomObjectActor):
        raise ValueError('This wrapper serves only the registered random-object actor')
    names = ('_object_rng', '_object_rng_identity', '_object_draw_trace')
    previous = {name: getattr(actor, name) for name in names}
    identity = allocation_identity(task, phase, index)
    actor._object_rng = np.random.default_rng(np.random.SeedSequence(identity['seed_sequence_words']))
    actor._object_rng_identity = identity
    actor._object_draw_trace = []
    try:
        result = task.episode(phase, index, actor, *args, **kwargs)
        result['random_object_rng_identity'] = identity
        result['random_object_uniforms'] = np.asarray(actor._object_draw_trace)
        return result
    finally:
        for name in names:
            setattr(actor, name, previous[name])


def make_actor(spec, seed=0):
    return RandomObjectActor(spec, seed)
