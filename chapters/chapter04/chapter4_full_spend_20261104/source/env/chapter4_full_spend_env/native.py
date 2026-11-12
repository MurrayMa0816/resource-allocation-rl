"""Load the copied, source-bound Chapter 3 package under one private name."""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

ENV_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ENV_ROOT / 'release'
COPY_MANIFEST = ENV_ROOT / 'native_copy_manifest.json'
_RELEASE_PACKAGE = 'chapter4_full_spend_env._native'
CORE_FILES = ('full_spend.py', 'methods/policy.py', 'state.py', 'task.py')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify_sources():
    manifest = json.loads(COPY_MANIFEST.read_text())
    for relative, expected in manifest['files'].items():
        if digest(RELEASE_ROOT / relative) != expected:
            raise RuntimeError('Copied C3 source changed: ' + relative)
    original = RELEASE_ROOT
    for relative in CORE_FILES:
        key = 'allocation23/' + relative
        if digest(original / key) != manifest['files'][key]:
            raise RuntimeError('Original full-spend core identity changed: ' + key)
    data_root = _eurosat_root()
    data = {str(data_root / name): digest(data_root / name)
            for name in ('images.npy', 'split_manifest.json')}
    copied_split = data_root / 'split_manifest.json'
    if data[str(data_root / 'split_manifest.json')] != manifest['expected_split_sha256']:
        raise RuntimeError('External image split differs from copied source identity')
    return dict(copy_manifest_sha256=digest(COPY_MANIFEST),
                copied_files=len(manifest['files']), copied_source_sha256=manifest['files'],
                full_spend_core_sha256={name: manifest['files']['allocation23/' + name]
                                       for name in CORE_FILES},
                native_root=str(RELEASE_ROOT), original_native_root=str(original),
                immutable_data_root=str(data_root), immutable_data_sha256=data)


if _RELEASE_PACKAGE not in sys.modules:
    init = RELEASE_ROOT / 'allocation23/__init__.py'
    spec = importlib.util.spec_from_file_location(
        _RELEASE_PACKAGE, init, submodule_search_locations=[str(init.parent)])
    if spec is None or spec.loader is None:
        raise ImportError('Cannot bind copied full-spend source')
    module = importlib.util.module_from_spec(spec)
    sys.modules[_RELEASE_PACKAGE] = module
    spec.loader.exec_module(module)


def load(name):
    return importlib.import_module(_RELEASE_PACKAGE + '.' + name)


task_module = load('task')
BaseTask = task_module.Task
Actor = load('methods.policy').Actor
TRAIN_MASTER = load('io').TRAIN_MASTER
make_encoder = load('state').make_encoder
validate_action = load('state').validate_action
full_spend = load('full_spend')
ORIGINAL_DATA_ROOT = _eurosat_root()
