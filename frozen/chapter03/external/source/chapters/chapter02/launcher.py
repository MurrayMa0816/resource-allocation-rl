#!/usr/bin/env python3
"""Chapter 2 development entry; explicit commands and fixed-total contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('RESOURCE_ALLOCATION_PROJECT_ROOT',
                              '/root/gamewar/resource_allocation_rl')).resolve()
METHODS = ('paspo', 'rpo', 'td7', 'simba_v2', 'dime')


def runtime_environment():
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env.setdefault(name, '1')
    for name, relative in [('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'xdg'),
                           ('MPLCONFIGDIR', 'matplotlib'), ('TORCH_HOME', 'torch'),
                           ('CUDA_CACHE_PATH', 'cuda'), ('TRITON_CACHE_DIR', 'triton')]:
        path = ROOT/'cache'/relative
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env['RESOURCE_ALLOCATION_PROJECT_ROOT'] = str(PROJECT)
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
    if not path.is_relative_to(Path('/hy-tmp')):
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


def _exec(arguments):
    entry = PROJECT/'python'
    os.execve(str(entry), [str(entry), str(PROJECT/'run.py'), *arguments], runtime_environment())


def inspect(arguments):
    child = subprocess.run([str(PROJECT/'python'), str(PROJECT/'run.py'),
                            'chapter2', 'inspect', *arguments], env=runtime_environment(),
                           check=True, capture_output=True, text=True)
    data = json.loads(child.stdout)
    data.update(workspace=str(ROOT), chapter=2, primary_method='A',
                baselines=list(METHODS), total_policy='fixed per-period total',
                method_entry=str(ROOT/'method.py'),
                baseline_source=str(ROOT/'baselines'),
                configs=str(ROOT/'configs'),
                ablation_registry=str(ROOT/'ablations.py'))
    print(json.dumps(data, ensure_ascii=False, indent=2))


def check():
    manifest = json.loads((ROOT/'SOURCE.json').read_text())
    for row in manifest['files']:
        logical = Path(row['logical_path'])
        canonical = Path(row['canonical_source'])
        if logical.resolve() != canonical.resolve():
            raise ValueError('Compatibility source points to a different implementation: '+str(logical))
        if hashlib.sha256(canonical.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Current source differs from its directory-split registration: '+str(canonical))
    print(json.dumps(dict(status='PASS', chapter=2, source_files=len(manifest['files']),
                         source_bytes_unchanged=True, training_started=False), indent=2))


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
    if command == 'inspect':
        inspect(rest)
    elif command == 'check':
        if rest:
            raise ValueError('check takes no arguments')
        check()
    elif command in ('baseline', 'baselines'):
        forwarded = list(rest or ['inspect'])
        subcommand = forwarded[0]
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
        # The old evaluate parser has no --chapter argument, so validate the
        # registered model here, then call that unchanged parser directly.
        _exec((['chapter2', command] if command == 'train' else [command]) + rest)
    else:
        raise ValueError('Unknown chapter-2 command: '+command)


if __name__ == '__main__':
    main()
