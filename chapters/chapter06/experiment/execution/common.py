"""Small chapter-six runtime helpers; all scientific artifacts live on HY."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'source'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def require_hy(path):
    result = Path(path).expanduser().resolve()
    if not result.is_relative_to(Path('/hy-tmp').resolve()):
        raise ValueError('All generated chapter-six artifacts must physically reside under /hy-tmp')
    return result


def jsonable(value):
    import numpy as np
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    destination = require_hy(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    os.replace(temporary, destination)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str((str(array.dtype),array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def state_hash(value):
    import numpy as np
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            visit(item.detach().cpu().contiguous().numpy())
        elif isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype),item.shape)).encode());digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode());visit(item[key])
        elif isinstance(item, (tuple,list)):
            digest.update(str(type(item)).encode())
            for child in item: visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def cpu_copy(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return value


def save_npz(path, **arrays):
    import numpy as np
    destination = require_hy(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp.' + str(os.getpid()))
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, destination)


def save_endpoint(path, value):
    import torch
    destination = require_hy(path)
    if destination.exists():
        raise FileExistsError('Never overwrite a fixed endpoint: ' + str(destination))
    temporary = destination.with_name(destination.name + '.tmp.' + str(os.getpid()))
    torch.save(cpu_copy(value), temporary)
    os.replace(temporary, destination)


def environment_config(config):
    """Canonical environment kwargs, with explicit objective/physical identity."""
    env = dict(config.get('env_config', {}))
    adapter=config.get('adapter',env.get('adapter','unit'))
    if 'adapter' in env and env['adapter']!=adapter: raise ValueError('Conflicting adapter identity')
    env['adapter']=adapter
    scale=float(config.get('spatial_sigma_scale',env.get('spatial_sigma_scale',1.)))
    if 'spatial_sigma_scale' in env and float(env['spatial_sigma_scale'])!=scale:raise ValueError('Conflicting spatial noise scale')
    env['spatial_sigma_scale']=scale
    for short, physical in [('tau','coverage_threshold'),('weight','coverage_weight')]:
        if short in config:
            if physical in env and float(env[physical]) != float(config[short]):
                raise ValueError('Conflicting objective values in the job')
            env[physical] = float(config[short])
    if 'coverage_threshold' not in env or 'coverage_weight' not in env:
        raise ValueError('Explicit coverage threshold and preference required')
    if env.get('reward_mode', 'area_weighted_performance') != 'area_weighted_performance':
        raise ValueError('Dual-value labels require the native scalar terminal reward JA')
    env['reward_mode'] = 'area_weighted_performance'
    return env


def source_hashes(expected=None):
    """Verify a frozen source map, or bind the owned scientific code for engineering."""
    if expected is None:
        files = []
        for directory in (ROOT/'source', ROOT/'method'):
            files.extend(p for p in directory.rglob('*')
                         if p.is_file() and p.suffix in ('.py','.json','.csv'))
        files.extend(ROOT/'execution'/name for name in ('common.py','worker_train.py'))
        return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(files)}
    result = {}
    for name, digest in expected.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or file_hash(path) != digest:
            raise ValueError('Frozen chapter-six source mismatch: ' + name)
        result[name] = digest
    return result


def load_endpoint(path, device='cpu', *, expected_tau=None, expected_weight=None, expected_adapter=None, expected_spatial_sigma_scale=None):
    """Load one genuine89 fixed endpoint into its original training specification."""
    import torch
    from hongmiao89 import make_env, FullSpendSpec, make_models_for_arm
    from hongmiao89.method_bridge import owned_method
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != 'hongmiao89-reasonable-depth-dual-random-fixed-endpoint-v1':
        raise ValueError('Not a chapter-six dual-random89 endpoint')
    protocol = payload.get('protocol', {})
    if protocol.get('case_id') != 'Hongmiao89-reasonable-depth-v1' or protocol.get('region_count') != 89:
        raise ValueError('Foreign physical task identity')
    if protocol.get('objective_order') != ['C', 'JA']:
        raise ValueError('Endpoint does not bind the high-yield C/JA learning objective')
    if not protocol.get('source_files_sha256'):
        raise ValueError('Checkpoint has no bound scientific source closure')
    source_hashes(protocol['source_files_sha256'])
    owned_method('initialization').validate_checkpoint_initialization(payload, seed=protocol.get('seed'))
    if 'spatial_sigma_scale' not in protocol or float(protocol['spatial_sigma_scale'])!=float(protocol.get('initialization',{}).get('spatial_sigma_scale',float('nan'))):
        raise ValueError('Endpoint lacks consistent explicit spatial noise identity')
    if expected_spatial_sigma_scale is not None and float(protocol['spatial_sigma_scale']) != float(expected_spatial_sigma_scale):
        raise ValueError('Endpoint trained under another spatial noise scale')
    if expected_adapter is not None and protocol.get('adapter') != expected_adapter:
        raise ValueError('Endpoint trained under another adapter')
    if expected_tau is not None and float(protocol['tau']) != float(expected_tau):
        raise ValueError('Endpoint trained under another coverage threshold')
    if expected_weight is not None and float(protocol['weight']) != float(expected_weight):
        raise ValueError('Endpoint trained under another preference')
    if payload.get('optimizer_backup') or payload.get('replay_backup') or payload.get('resume_supported'):
        raise ValueError('Only the registered compact evaluation endpoint is accepted')
    env = make_env(**protocol['env_config'])
    try:
        if env.task_fingerprint != protocol['task_fingerprint']:
            raise ValueError('Original environment/source identity differs from endpoint')
        actor, critic, spec = make_models_for_arm('full', FullSpendSpec.from_env(env), protocol['seed'])
        if spec.to_dict() != protocol['neural_spec']:
            raise ValueError('Checkpoint neural interface identity differs')
        actor.load_state_dict(payload['actor']); critic.load_state_dict(payload['critic'])
    finally:
        env.close()
    if model_hash(actor) != payload['actor_sha256'] or model_hash(critic) != payload['critic_sha256']:
        raise ValueError('Endpoint parameter digest mismatch')
    return actor.to(device), critic.to(device), spec, payload
