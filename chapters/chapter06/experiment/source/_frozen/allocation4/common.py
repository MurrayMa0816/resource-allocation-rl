"""Shared public-state tensors; no physical-model access."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

STATE_KEYS = ('local', 'global', 'caps', 'budget', 'remaining')

def tensor_batch(batches, device):
    lengths = [len(b['raw']) for b in batches]
    states = {k: torch.as_tensor(np.concatenate([b['states'][k] for b in batches]),
               dtype=torch.float64, device=device) for k in STATE_KEYS}
    result = {k: torch.as_tensor(np.concatenate([b[k] for b in batches]),
              dtype=torch.float64, device=device)
              for k in ('raw', 'old_logp', 'mask', 'values', 'returns')}
    result.update(states=states, lengths=lengths)
    return result

def state_features(states):
    local = states['local']
    return torch.cat((local.mean(1), local.std(1, unbiased=False),
                      local.amin(1), local.amax(1), states['global']), -1)

def subset(states, indices):
    return {k: v[indices] for k, v in states.items()}

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n')
    temporary.replace(path)

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def model_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def source_hashes():
    roots = [Path(__file__).resolve().parents[1],
             Path('/root/gamewar/chapter23_final')]
    result = {}
    for root in roots:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix in ('.py', '.json', '.toml', '.rst') and '__pycache__' not in str(p):
                result[str(p)] = file_hash(p)
    return result

def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)
