"""Load the pinned author core, without importing unused simulator packages."""
from __future__ import annotations
import hashlib
import importlib
import importlib.machinery
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
VENDOR = HERE / 'vendor'


def _package(name, path):
    old = sys.modules.get(name)
    if old is not None:
        if Path(old.__file__).resolve().parent != path:
            raise RuntimeError('foreign author package already loaded: ' + name)
        return old
    mod = types.ModuleType(name)
    mod.__file__ = str(path / '__init__.py')
    mod.__path__ = [str(path)]
    mod.__package__ = name
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    mod.__spec__.submodule_search_locations = [str(path)]
    sys.modules[name] = mod
    return mod


def load_native():
    manifest = json.loads((HERE / 'vendor_manifest.json').read_text())
    for rel, digest in manifest['files'].items():
        if hashlib.sha256((VENDOR / rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError('author BRO file changed: ' + rel)
    import flax
    # BRO uses the ordinary dict.pop API, unlike the older SR-SAC core.
    # The supported Flax switch preserves that original API without editing it.
    flax.config.update('flax_return_frozendict', False)
    root = VENDOR / 'jaxrl'
    _package('jaxrl', root)
    _package('jaxrl.bro', root / 'bro')
    _package('jaxrl.networks', root / 'networks')
    learner = importlib.import_module('jaxrl.bro.bro_learner')
    replay = importlib.import_module('jaxrl.replay_buffer')
    return learner, replay.ParallelReplayBuffer, replay.Batch
