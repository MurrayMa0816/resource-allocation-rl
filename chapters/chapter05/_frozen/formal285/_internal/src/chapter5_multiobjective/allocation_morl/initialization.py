"""Chapter5-only restoration of the native, seeded random output-layer draws.

The immutable native dependency draws default Linear parameters, then overwrites
the two output layers with zeros. Replay its CPU float32 construction streams to
recover exactly the discarded draws. Only the four output tensors are copied;
the Actor class, hidden layers, critic, decoder and native source remain intact.
"""
from __future__ import annotations

import copy
import hashlib
import math

import torch
from torch import nn

INITIALIZATION_PROFILE = 'chapter5-both-heads-default-random-v1'


def _hash(tensor):
    value = tensor.detach().cpu().contiguous()
    prefix = str((str(value.dtype), tuple(value.shape))).encode()
    return hashlib.sha256(prefix + value.numpy().tobytes()).hexdigest()


def _discarded_default_head(container, output, seed):
    """Replay native Linear draws; reject any changed hidden construction order."""
    layers = [module for module in container.modules() if isinstance(module, nn.Linear)]
    if not layers or layers[-1] is not output:
        raise ValueError('The registered native output layer must be the last Linear')
    result = None
    with torch.random.fork_rng(devices=[]):
        # Seed only the CPU stream. torch.manual_seed would also seed CUDA.
        local = torch.Generator(device='cpu').manual_seed(int(seed))
        torch.set_rng_state(local.get_state())
        for layer in layers:
            fresh = nn.Linear(layer.in_features, layer.out_features,
                              bias=layer.bias is not None, device='cpu', dtype=torch.float32)
            weight = fresh.weight.detach().to(dtype=layer.weight.dtype, device=layer.weight.device)
            bias = (fresh.bias.detach().to(dtype=layer.bias.dtype, device=layer.bias.device)
                    if layer.bias is not None else None)
            if layer is output:
                result = (weight.clone(), None if bias is None else bias.clone())
            elif (not torch.equal(weight, layer.weight.detach()) or
                  (bias is not None and not torch.equal(bias, layer.bias.detach()))):
                raise ValueError('Native hidden initialization changed; refuse an unverified replay')
    return result


def initialize_output_heads(actor, seed):
    """Initialize one fresh Chapter5 Actor, exactly as omitting four zero writes."""
    if hasattr(actor, '_chapter5_initialization_record'):
        raise ValueError('Never reinitialize an existing Chapter5 policy')
    if torch.get_default_dtype() != torch.float32:
        raise ValueError('Native construction requires its original default float32 draw dtype')
    if not actor.learned_quota or actor.study_variant != 'A_FULL':
        raise ValueError('Chapter5 Full requires its compact spatial and learned total branches')
    spatial, quota = actor.spatial.head, actor.quota_head[-1]
    if not isinstance(spatial, nn.Linear) or not isinstance(quota, nn.Linear):
        raise ValueError('Expected the native spatial and total Linear output layers')
    for head in (spatial, quota):
        if head.weight.device.type != 'cpu' or head.weight.dtype != torch.float64:
            raise ValueError('Initialize fresh native CPU float64 parameters before device transfer')
        if head.bias is None or torch.count_nonzero(head.weight).item() or torch.count_nonzero(head.bias).item():
            raise ValueError('Refuse to reset learned or foreign nonzero output parameters')
    spatial_values = _discarded_default_head(actor.spatial, spatial, 31000 + int(seed))
    quota_values = _discarded_default_head(actor.quota_head, quota, 83100 + int(seed))
    with torch.no_grad():
        for head, values in ((spatial, spatial_values), (quota, quota_values)):
            head.weight.copy_(values[0]); head.bias.copy_(values[1])
    heads = {}
    for name, head, stream in [('spatial', spatial, 31000 + int(seed)),
                               ('quota', quota, 83100 + int(seed))]:
        heads[name] = dict(seed=stream, fan_in=head.in_features,
            uniform_bound=1. / math.sqrt(head.in_features),
            weight_sha256=_hash(head.weight), bias_sha256=_hash(head.bias),
            weight_shape=list(head.weight.shape), bias_shape=list(head.bias.shape))
    actor._chapter5_initialization_record = dict(profile=INITIALIZATION_PROFILE,
        fresh=True, old_models_loaded=False, seed=int(seed), heads=heads,
        recipe='native nn.Linear default uniform; omit only the four output zero overwrites',
        draw_dtype='float32', parameter_dtype='float64',
        native_default_draw_replay_exact=True, cpu_rng_preserved=True,
        extra_cuda_rng_calls=0, per_action_random_noise_added=False,
        constructor_provenance_only=True)
    return actor


def initialization_record(actor):
    """Construction provenance; a loaded checkpoint's own protocol is authoritative."""
    value = getattr(actor, '_chapter5_initialization_record', None)
    if value is None or value.get('profile') != INITIALIZATION_PROFILE:
        raise ValueError('Missing current Chapter5 dual-random initialization provenance')
    return copy.deepcopy(value)


def validate_checkpoint_initialization(payload, *, seed=None):
    """Do not relabel an old fixed endpoint as a newly random-initialized run.

    Historical endpoints remain loadable through their own frozen source. Tensor
    shapes alone cannot distinguish the two initialization protocols.
    """
    protocol = payload.get('protocol', {})
    value = protocol.get('initialization', {})
    if (value.get('profile') != INITIALIZATION_PROFILE or value.get('fresh') is not True or
            value.get('old_models_loaded') is not False or
            value.get('native_default_draw_replay_exact') is not True):
        raise ValueError('Checkpoint is not from the registered Chapter5 dual-random initialization')
    actual_seed = protocol.get('seed')
    if actual_seed != value.get('seed') or (seed is not None and actual_seed != int(seed)):
        raise ValueError('Checkpoint initialization seed and training identity disagree')
    for name, offset in [('spatial', 31000), ('quota', 83100)]:
        head = value.get('heads', {}).get(name, {})
        if (head.get('seed') != offset + actual_seed or
                any(not isinstance(head.get(k), str) or len(head[k]) != 64
                    for k in ['weight_sha256', 'bias_sha256'])):
            raise ValueError('Missing or inconsistent output-head construction identity')
    return copy.deepcopy(value)
