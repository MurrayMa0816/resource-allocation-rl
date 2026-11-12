"""Shared logging/evaluation only. Algorithm updates remain in official code."""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root

import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULT_ROOT = Path(str(_data_root()/'new_runs/chapter04'))


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def array_digest(arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.asarray(array)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def evaluate(action_fn, domain, seed, episodes=32, phase='dev', action_mode='tanh_latent', deterministic=False):
    """Disjoint phase, fixed exogenous paths; no test rewards enter training."""
    from .env_adapter import Env
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch = sys.modules.get('torch')
    torch_rng = torch.get_rng_state() if torch else None
    cuda_rng = torch.cuda.get_rng_state_all() if torch and torch.cuda.is_initialized() else None
    # Common evaluation randomness is fixed separately from the training stream.
    random.seed(4_000_000+seed)
    np.random.seed(4_000_000+seed)
    if torch:
        torch.manual_seed(4_000_000+seed)
    env = Env(domain, seed, phase=phase, action_mode=action_mode, holdout=(phase == 'final'))
    scores = []
    try:
        for index in range(episodes):
            obs, _ = env.reset()
            ended, score = False, 0.0
            while not ended:
                obs, reward, terminated, truncated, _ = env.step(action_fn(obs, deterministic))
                ended = terminated or truncated
                score += reward
            scores.append(score * 100.0)
        return {'mean': float(np.mean(scores)), 'std': float(np.std(scores, ddof=1)) if len(scores)>1 else 0.,
                'episode_scores': scores, 'evaluation_steps': int(env.real_steps)}
    finally:
        env.close()
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        if torch_rng is not None:
            torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


class RunLog:
    def __init__(self, config, method):
        self.output = Path(config['output_dir'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.start = time.perf_counter()
        self.method = method
        self.config = config.copy()
        self.rows = []
        json_write(self.output / 'config.json', config)

    def append(self, **row):
        self.rows.append({'wall_seconds': time.perf_counter()-self.start, **row})
        with (self.output / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(self.rows[-1], default=str) + '\n')

    def finish(self, **receipt):
        out = {'method': self.method, 'domain': self.config['domain'], 'seed': self.config['seed'],
               'smoke_only': bool(self.config.get('smoke', False)),
               'wall_seconds': time.perf_counter()-self.start, **receipt}
        json_write(self.output / 'receipt.json', out)
        return out


def configure_jax(device):
    # Must run before importing JAX. Avoid reserving the whole shared GPU.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
    os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
    os.environ['JAX_PLATFORMS'] = 'cpu' if device == 'cpu' else 'cuda'

