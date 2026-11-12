"""Editable Chapter 3 comparator implementations, with lazy neural imports.

The collector owns environment construction and experiment outputs. These five
agents receive only actual public transitions and learn online from scratch.
"""
from importlib import import_module
import json
from pathlib import Path

METHODS = ('c_ppo', 'pfo', 'gradient_ppo', 'reflective_ppo', 'ppo_plus')
HISTORICAL_METHODS = ('crossq',)
LOADABLE_METHODS = METHODS + HISTORICAL_METHODS
DOMAINS = ('radio', 'v5')


def _method(method, *, historical=False):
    if method not in (LOADABLE_METHODS if historical else METHODS):
        raise ValueError(f'Unknown active Chapter 3 comparator: {method!r}; choose {METHODS}')
    return import_module(f'.{method}', __name__)


def profile(method, domain):
    """Read an independent copy of the screened per-domain task profile."""
    if method not in METHODS or domain not in DOMAINS:
        raise ValueError('Registered comparator and domain required')
    path = Path(__file__).parent / 'profiles' / f'{method}_{domain}.json'
    return json.loads(path.read_text())


def make(method, obs_dim, action_dim, seed=0, device='cpu', config=None):
    """Construct a fresh agent; callers explicitly choose profile and overrides."""
    if (int(obs_dim), int(action_dim)) not in ((303, 101), (242, 121)):
        raise ValueError('The active compact public state and N+1 action dimensions are required')
    return _method(method).Candidate(obs_dim, action_dim, seed, device=device, config=config)


def load(method, directory, device='cpu'):
    """Load a fixed endpoint for deterministic inference, without offline training."""
    directory = Path(directory)
    if method in ('gradient_ppo', 'reflective_ppo'):
        meta = json.loads((directory / 'metadata.json').read_text())
        if meta['algorithm'] != method:
            raise ValueError('Endpoint algorithm does not match the requested comparator')
    elif method == 'crossq':
        meta = json.loads((directory / 'metadata.json').read_text())
        if meta['method'] != method:
            raise ValueError('Endpoint algorithm does not match the requested comparator')
    elif method == 'ppo_plus':
        meta = json.loads((directory / 'metadata.json').read_text())
        if meta.get('method') != 'PPO+ RLJ/RLC2026 PyTorch adapter':
            raise ValueError('Endpoint is not the registered PPO+ adapter')
    return _method(method, historical=True).Candidate.load(directory, device=device)
