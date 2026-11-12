"""Chapter 3's retained Full/rule-total protocol, isolated from shared legacy actors.

The verified scientific capsule stays immutable. Training/evaluation run in a
fresh process with that capsule's package. The small in-process actor factory
loads it under a private package name, so chapters 2/4/5 keep their own actors.
"""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root

import ast
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

# Never write Python bytecode into the immutable scientific capsule.
sys.dont_write_bytecode = True

PROJECT = Path(os.environ.get('RESOURCE_ALLOCATION_PROJECT_ROOT', str(_resolve_path('/root/gamewar/resource_allocation_rl')))).resolve()
import portable_paths as _paths
CANONICAL = _paths.chapter_data_root(3)
CAPSULE = _project_root()/'frozen/chapter03/native'
if not CAPSULE.is_dir():CAPSULE=_resolve_path('/root/gamewar/resource_allocation_rl/frozen/chapter03/native')
PRIVATE = '_chapter03_published_native'
EXPECTED_CAPSULE = {'direct_policy.py': '579a53eef26b42765a1096ffba2255494361c710c2d8ae82621b6b7c99649edc',
 'evaluate_worker.py': 'eb8a321b748bd3648304896b0c5573fc8c1a3d6a8bfa6a7f68f654f808815d68',
 'native_cuda.py': '4028cfc4702dae29cf785e1bec76ac39ee3d17ecd7e8b5041f8bcb4fd2259d95',
 'train_worker.py': '1ead3d4c49133e80f5744bb4739ce763715ddc97d0c8a025975a8482245c36db',
 'source_registry.json': 'ad7316357dbc4eff9c27a9a866054e287c206188217167d003d4bc1f7eff9834'}
ARMS = {
    'A_FULL': dict(temporal='adaptive', quota_reference='pace'),
    'A_PACE': dict(temporal='pace', quota_reference='pace'),
}
DISPLAY_NAMES = {
    'A_FULL': 'ReJoP', 'A_PACE': '去总量学习',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def verified_capsule():
    folder = CAPSULE.resolve(strict=True)
    for relative, expected in EXPECTED_CAPSULE.items():
        if sha(folder/relative) != expected:
            raise ValueError('Published Chapter 3 capsule changed: '+relative)
    registry = json.loads((folder/'source_registry.json').read_text())
    source = folder/'source'
    for relative, expected in registry['scientific_source_sha256'].items():
        if sha(source/relative) != expected:
            raise ValueError('Published Chapter 3 source changed: '+relative)
    return folder


def declaration():
    folder = verified_capsule()
    return dict(schema='chapter3-published-four-arm-maintenance-v1',
        scientific_capsule=str(folder), initialization='both_random',
        arms=ARMS, display_names=DISPLAY_NAMES, additional_ablations=2, random_object_entry='chapters/chapter03/random_object_ablation_20261109/train.py',
        source_registry_sha256=sha(folder/'source_registry.json'),
        default_steps=100608, seeds=list(range(5)), dev_every=512,
        dev_paths=8, final_paths=128, final_model='fixed endpoint',
        shared_legacy_core_modified=False)


def _private_package():
    source=verified_capsule()/'source/allocation23'
    if PRIVATE not in sys.modules:
        from frozen_adapter import load_module
        load_module(PRIVATE,source/'__init__.py')
    return PRIVATE


def make_actor(spec, seed=0, arm='A_FULL'):
    if spec.chapter!=3 or arm not in ARMS:
        raise ValueError('Current native arms are A_FULL/A_PACE; random-object uses its isolated explicit worker')
    private=_private_package()
    actor_type=importlib.import_module(private+'.methods.policy').Actor
    return actor_type(spec,'A_FULL',seed=seed,**ARMS[arm])


def _option(args, name):
    for i, value in enumerate(args):
        if value == name and i+1 < len(args):
            return args[i+1]
        if value.startswith(name+'='):
            return value.split('=', 1)[1]
    return None


def command(argv):
    """Build an inspectable command; never start work while declaring a plan."""
    args = list(argv)
    if not args or args[0] not in ('train', 'evaluate'):
        raise ValueError('Expected train or evaluate')
    operation, options = args[0], args[1:]
    folder = verified_capsule()
    if operation == 'train':
        result = [sys.executable, str(folder/'train_worker.py'), *options]
    else:
        result = [sys.executable, str(folder/'evaluate_worker.py'), 'evaluate', *options]
        if not any(x in options for x in ('-h', '--help')):
            if not _option(options, '--registry'):
                result.extend(['--registry', str(folder/'conditions/registry.json')])
            if not _option(options, '--condition'):
                result.extend(['--condition', 'default'])
            if not _option(options, '--direct-module'):
                result.extend(['--direct-module', str(folder/'direct_policy.py')])
    if not any(x in options for x in ('-h', '--help')):
        if not _option(options, '--source-root'):
            result.extend(['--source-root', str(folder/'source')])
        if not _option(options, '--data-root'):
            result.extend(['--data-root', str(_eurosat_root())])
    script=Path(result[1]);options=result[2:]
    bootstrap='import sys; from frozen_adapter import run_file; run_file('+repr(str(script))+',sys.argv[1:])'
    return [sys.executable,'-c',bootstrap,*options]


def main(argv):
    args = list(argv)
    protected_options = ('--output', '--source-root', '--direct-module')
    for value in args:
        key = value.split('=', 1)[0]
        if key.startswith('--') and any(name.startswith(key) and key != name for name in protected_options):
            raise ValueError('Use the complete protected option name: '+key)
    for name in protected_options:
        count = sum(value == name or value.startswith(name+'=') for value in args)
        if count > 1:
            raise ValueError('Repeated protected option: '+name)
    if not any(value in ('-h', '--help') for value in args):
        output = _option(args, '--output')
        if not output:
            raise ValueError('An explicit new HY output directory is required')
        destination = Path(output).expanduser().resolve()
        if not destination.is_relative_to(Path(str(_resolve_path('/hy-tmp')))) or destination.exists():
            raise ValueError('Use a new physical HY output directory')
        protected = [verified_capsule(), (CANONICAL/'experiments/final_organized').resolve(),
                     (CANONICAL/'models').resolve(), (CANONICAL/'data').resolve(),
                     Path(__file__).resolve().parent.parent]
        if any(destination.is_relative_to(folder) for folder in protected):
            raise ValueError('Outputs cannot be written into source or sealed archives')
        source_root = _option(args, '--source-root')
        if source_root and Path(source_root).resolve() != (verified_capsule()/'source').resolve():
            raise ValueError('Published entry requires its verified frozen package')
        direct_module = _option(args, '--direct-module')
        if direct_module and sha(direct_module) != EXPECTED_CAPSULE['direct_policy.py']:
            raise ValueError('Evaluation requires the published Direct implementation')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',ALLOCATION23_DATA_ROOT=str(_eurosat_root()),RESOURCE_ALLOCATION_PROJECT_ROOT=str(_project_root()),RESOURCE_ALLOCATION_DATA_ROOT=str(_data_root()))
    env['PYTHONPATH']=str(_project_root())
    return subprocess.run(command(argv), env=env, check=True).returncode
