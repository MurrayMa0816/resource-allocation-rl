#!/usr/bin/env python3
"""Chapter 2 development entry; explicit commands and fixed-total contract."""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('RESOURCE_ALLOCATION_PROJECT_ROOT',
                              str(_resolve_path('/root/gamewar/resource_allocation_rl')))).resolve()
METHODS = ('paspo', 'rpo', 'td7', 'simba_v2', 'dime')


def runtime_environment():
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env.setdefault(name, '1')
    for name, relative in [('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'xdg'),
                           ('MPLCONFIGDIR', 'matplotlib'), ('TORCH_HOME', 'torch'),
                           ('CUDA_CACHE_PATH', 'cuda'), ('TRITON_CACHE_DIR', 'triton')]:
        path = _data_root()/'runtime/chapter02'/relative
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env['RESOURCE_ALLOCATION_PROJECT_ROOT'] = str(PROJECT)
    env['RESOURCE_ALLOCATION_DATA_ROOT'] = str(_data_root())
    env['ALLOCATION23_DATA_ROOT'] = str(_eurosat_root())
    return env


def _options(arguments, key, *, remove=False):
    values, rest = [], []
    index = 0
    while index < len(arguments):
        item = arguments[index]
        if item == key:
            if index+1 >= len(arguments):
                raise ValueError('Missing value for '+key)
            values.append(arguments[index+1])
            if not remove:
                rest += arguments[index:index+2]
            index += 2
        elif item.startswith(key+'='):
            values.append(item.split('=', 1)[1])
            if not remove:
                rest.append(item)
            index += 1
        else:
            rest.append(item)
            index += 1
    if len(set(values)) > 1:
        raise ValueError('Conflicting values for '+key)
    return (values[0] if values else None), rest


def _chapter2(arguments):
    value, rest = _options(arguments, '--chapter', remove=True)
    if value is not None and value != '2':
        raise ValueError('This workspace only accepts --chapter 2')
    return rest


def _check_output(arguments):
    value, _ = _options(arguments, '--output')
    if value is None:
        raise ValueError('Explicit train/evaluate requires --output')
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(Path(str(_resolve_path('/hy-tmp')))):
        raise ValueError('All chapter-2 result and test artifacts must physically reside under /hy-tmp')
    if path.is_relative_to(ROOT.parent):
        raise ValueError('Results must be outside the maintained code tree')


def _checkpoint_chapter(arguments, *, baseline):
    value, _ = _options(arguments, '--checkpoint')
    if value is None:
        raise ValueError('Evaluation requires --checkpoint')
    checkpoint = Path(value).expanduser().resolve()
    if baseline:
        identity = json.loads((checkpoint/'manifest.json').read_text())['identity']
    else:
        # Evaluation is explicit; CPU loading only inspects this supplied model.
        sys.path.insert(0, str(PROJECT))
        import torch
        identity = torch.load(checkpoint, map_location='cpu', weights_only=False)['identity']
    if identity.get('config', {}).get('chapter') != 2:
        raise ValueError('This workspace only evaluates chapter-2 checkpoints')
    return identity


def _native_backend():
    import importlib.util
    name='_chapter02_native_backend_router'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/'native_backend.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def _exec_native(arguments, *, source_name=None):
    backend=_native_backend();source=backend.source_root(source_name)
    forwarded=list(arguments)
    data_root,_=_options(forwarded,'--data-root')
    if data_root is None:
        forwarded+=['--data-root',backend.registration()['default_data_root']]
    bootstrap=('import sys; sys.path.insert(0, '+repr(str(PROJECT))+'); '
               'from frozen_adapter import install_import_adapter; install_import_adapter(); '
               'sys.path.insert(0, '+repr(str(source))+'); '
               'from allocation23.__main__ import main; main()')
    entry=PROJECT/'python'
    os.execve(str(entry),[str(entry),'-c',bootstrap,*forwarded],runtime_environment())


def _exec(arguments):
    entry = PROJECT/'python'
    source=_native_backend().source_root('legacy_zero_head')
    bootstrap=('import sys; sys.path.insert(0, '+repr(str(PROJECT))+'); '
        'from frozen_adapter import install_import_adapter; install_import_adapter(); '
        'sys.path.insert(0, '+repr(str(source))+'); '
        'from allocation23.baselines.runner import main; main(sys.argv[2:])')
    os.execve(str(entry), [str(entry), '-c', bootstrap, *arguments], runtime_environment())


def inspect(arguments):
    # Metadata only: constructing an environment is never part of inspection.
    data=json.loads((_data_root()/'INDEX.json').read_text())['chapters']['chapter02']
    print(json.dumps(data,ensure_ascii=False,indent=2))
    return data


def check():
    # Publication release source check uses its actual exported inventory.
    if (PROJECT / 'release_manifest.json').is_file():
        result = subprocess.run([sys.executable, '-B', str(PROJECT / 'release_tools.py'),
            '--root', str(PROJECT), 'check'], check=False).returncode
        if result:
            raise SystemExit(result)
        return
    backend=_native_backend()
    sources={name:str(backend.source_root(name)) for name in backend.registration()['sources']}
    print(json.dumps(dict(status='PASS',chapter=2,scientific_sources=sources,
        source_bytes_unchanged=True,training_started=False,model_loads=0),ensure_ascii=False,indent=2))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        args = ['inspect']
    if args[0] in ('-h', '--help'):
        parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''Commands:
  inspect [--domain v5|radio]       Current native task and five baselines
  check                           Static source/compatibility-path check
  train ... [--arm A_FULL]         Native chapter-2 A or a spatial ablation
  evaluate --checkpoint ...       Native chapter-2 checkpoint evaluation
  baseline inspect|train|evaluate  PASPO, RPO-SAC, TD7, SimBaV2, DIME

Only explicit train/evaluate commands execute scientific work.
Use COMMAND --help for the underlying unchanged runner arguments.''')
        parser.print_help()
        return
    command, rest = args[0], _chapter2(args[1:])
    if command in ('inspect','status','plan'):
        inspect(rest)
    elif command == 'check':
        if rest:
            raise ValueError('check takes no arguments')
        check()
    elif command in ('baseline', 'baselines'):
        forwarded = list(rest or ['inspect'])
        subcommand = forwarded[0]
        if subcommand in ('inspect','status','plan'):
            print((ROOT/'configs/baselines.json').read_text());return
        helping = '-h' in forwarded or '--help' in forwarded
        if subcommand in ('train', 'evaluate') and not helping:
            _check_output(forwarded)
        if subcommand == 'evaluate' and not helping:
            _checkpoint_chapter(forwarded, baseline=True)
        # The unchanged evaluator has no chapter flag. Its model was validated
        # above; train and inspect still receive an explicit chapter selector.
        _exec(['baseline', *forwarded] + ([] if subcommand == 'evaluate' else ['--chapter', '2']))
    elif command in ('train', 'evaluate'):
        helping = '-h' in rest or '--help' in rest
        if command == 'train':
            import runpy
            ARMS = runpy.run_path(str(ROOT/'ablations.py'))['ARMS']
            method, _ = _options(rest, '--method')
            if method not in (None, 'A'):
                raise ValueError('Current chapter-2 main method is A; use baseline for external methods')
            arm, rest = _options(rest, '--arm', remove=True)
            if arm is not None:
                if arm not in ARMS:
                    raise ValueError('Unknown chapter-2 arm: '+arm)
                variant, _ = _options(rest, '--variant')
                if variant is not None and variant != ARMS[arm]['variant']:
                    raise ValueError('Conflicting --arm and --variant')
                if variant is None:
                    rest += ['--variant', ARMS[arm]['variant']]
            variant, _ = _options(rest, '--variant')
            if variant is not None and variant not in {v['variant'] for v in ARMS.values()}:
                raise ValueError('Choose a registered chapter-2 spatial variant')
            temporal, _ = _options(rest, '--temporal')
            if temporal not in (None, 'fixed'):
                raise ValueError('Chapter 2 uses a fixed per-period total')
            if method is None:
                rest += ['--method', 'A']
        if not helping:
            _check_output(rest)
            if command == 'evaluate':
                _checkpoint_chapter(rest, baseline=False)
        # Native Chapter2 uses its registered private source; historical model
        # identity selects its actual source without relabeling learned weights.
        source_name=None
        if command=='evaluate' and not helping:
            identity=_checkpoint_chapter(rest,baseline=False)
            source_name=_native_backend().source_for_identity(identity)
        elif command=='train' and '--resume' in rest and not helping:
            output,_=_options(rest,'--output')
            identity=json.loads((Path(output).expanduser().resolve()/'protocol.json').read_text())
            source_name=_native_backend().source_for_identity(identity)
        if command=='train' and '--resume' not in rest:
            variant,_=_options(rest,'--variant')
            temporal,_=_options(rest,'--temporal')
            if variant is None:rest+=['--variant','A_FULL']
            if temporal is None:rest+=['--temporal','fixed']
        _exec_native(([command,'--chapter','2'] if command=='train' else [command])+rest,source_name=source_name)
    else:
        raise ValueError('Unknown chapter-2 command: '+command)



if __name__ == '__main__':
    if not sys.argv[1:] or sys.argv[1] in ('inspect','status','plan'):
        import json as _json
        print(_json.dumps(_json.loads((_data_root()/'INDEX.json').read_text())['chapters']['chapter02'],ensure_ascii=False,indent=2))
    else:
        main()
