"""Published Chapter 3 native policy and comparator access."""
from pathlib import Path
import json


def _published_native():
    import importlib.util
    import sys
    name = '_chapter03_formal_entry'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name('formal_native.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def make_actor(spec, seed=0, arm='A_FULL'):
    return _published_native().make_actor(spec, seed=seed, arm=arm)


def make_comparator(method, obs_dim, action_dim, seed=0, device='cpu', config=None):
    from allocation23.chapter3_candidates import make
    return make(method, obs_dim, action_dim, seed, device=device, config=config)


def profile(method, domain):
    from allocation23.chapter3_candidates import profile as read_profile
    return read_profile(method, domain)


def train(argv):
    return _published_native().main(['train', *argv])


def evaluate(argv):
    return _published_native().main(['evaluate', *argv])


def baseline(argv):
    from allocation23.chapter3_candidates.cli import main
    return main(argv)


def declaration():
    chapter = Path(__file__).resolve().parent
    return json.loads((chapter/'configs/current.json').read_text())
