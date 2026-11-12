"""Chapter 3 method A and comparator access; shared learning cores stay single-copy."""
from pathlib import Path
import json


def make_actor(spec, seed=0, arm='A_FULL'):
    from allocation23.chapter3_candidates.native import ARMS
    from allocation23.methods import make_actor as factory
    if arm not in ARMS:
        raise ValueError('Unknown Chapter 3 arm: '+str(arm))
    return factory(spec, 'A', 3, seed, variant='A_FULL', **ARMS[arm])


def make_comparator(method, obs_dim, action_dim, seed=0, device='cpu', config=None):
    from allocation23.chapter3_candidates import make
    return make(method, obs_dim, action_dim, seed, device=device, config=config)


def profile(method, domain):
    from allocation23.chapter3_candidates import profile as read_profile
    return read_profile(method, domain)


def train(argv):
    from allocation23.chapter3_candidates.native import main
    return main(['train', *argv])


def evaluate(argv):
    from allocation23.chapter3_candidates.native import main
    return main(['evaluate', *argv])


def baseline(argv):
    from allocation23.chapter3_candidates.cli import main
    return main(argv)


def declaration():
    chapter = Path(__file__).resolve().parent
    return json.loads((chapter/'configs/current.json').read_text())
