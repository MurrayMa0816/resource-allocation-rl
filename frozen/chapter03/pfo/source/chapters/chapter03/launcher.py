#!/usr/bin/env python3
"""Independent Chapter 3 platform: A, four ablations, five online comparators."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT = Path(os.environ.get('RESOURCE_ALLOCATION_PROJECT_ROOT',
                              '/root/gamewar/resource_allocation_rl')).resolve()
CHAPTER = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(CHAPTER))
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key, leaf in (('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'xdg'),
                  ('MPLCONFIGDIR', 'mpl'), ('TORCH_HOME', 'torch'),
                  ('CUDA_CACHE_PATH', 'cuda'), ('TORCH_EXTENSIONS_DIR', 'extensions'),
                  ('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                  ('JAX_COMPILATION_CACHE_DIR', 'jax'), ('NUMBA_CACHE_DIR', 'numba')):
    cache = CHAPTER / 'cache' / leaf
    cache.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(cache)


def option(argv, name):
    for index, value in enumerate(argv):
        if value == name and index+1 < len(argv):
            return argv[index+1]
        if value.startswith(name+'='):
            return value.split('=', 1)[1]
    return None


def guard_output(argv):
    # One rule applies to native, explicit baseline and compatibility routes.
    # Parser help remains read-only and does not require a destination.
    if any(value in ('-h', '--help') for value in argv):
        return
    effective = argv[1:] if argv and argv[0] in ('baseline', 'baselines') else argv
    if not effective or effective[0] not in ('train', 'evaluate'):
        return
    destination = option(effective, '--output')
    if not destination:
        raise ValueError('Training and evaluation require an explicit new HY output directory')
    output = Path(destination).expanduser().resolve()
    if not output.is_relative_to(Path('/hy-tmp')):
        raise ValueError('Chapter 3 outputs must physically reside under /hy-tmp')
    if output.is_relative_to(CHAPTER.parent):
        raise ValueError('Experiment outputs must be separate from maintained chapter source directories')
    if output.exists():
        raise FileExistsError('Use a new HY output directory; existing runs are never overwritten')


def inspect(argv):
    p = argparse.ArgumentParser(description='Inspect Chapter 3 without starting training')
    p.add_argument('--domain', choices=('radio', 'v5'))
    args = p.parse_args(argv)
    import environment
    import ablations
    from allocation23.chapter3_candidates import METHODS, profile
    result = dict(schema='chapter3-independent-platform-v1', chapter=3,
        chapter_directory=str(CHAPTER), project_root=str(PROJECT),
        source=str(CHAPTER/'src/chapter3_candidates'), main_method='A',
        full_arm='A_FULL', additional_ablations=4,
        ablations=ablations.registry()['arms'], comparators=list(METHODS),
        cases=environment.inspect(args.domain),
        profiles={domain:{method:profile(method,domain) for method in METHODS}
                  for domain in ([args.domain] if args.domain else ('radio','v5'))},
        entrypoints=dict(native='./project chapter3 train|evaluate',
                         baseline='./project chapter3 baseline train|evaluate'),
        storage='complete curves and one fixed endpoint; no disk optimizer/replay/periodic models',
        training_started=False,
        research_status='editable chapter platform; matched formal performance not implied')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _external(argv):
    from allocation23.chapter3_candidates import METHODS
    method = option(argv, '--method')
    if method:
        return method in METHODS
    if argv and argv[0]=='evaluate':
        run = option(argv, '--run')
        if run:
            protocol = json.loads((Path(run).expanduser()/'protocol.json').read_text())
            return protocol.get('schema') == 'chapter3-integrated-online-candidates-v1'
    return False


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    guard_output(args)
    if not args or args[0]=='inspect' and not _external(args):
        return inspect(args[1:] if args else [])
    if args[0] in ('-h','--help'):
        p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""Commands:
  inspect [--domain radio|v5]       A, four ablations and five comparator profiles
  train --domain ... --arm ...     A or its registered temporal ablation
  evaluate --run ...               Completed A endpoint evaluation
  baseline inspect|train|evaluate  The five online comparator implementations
  ablations                       Show the registered four ablations
  config                          Show Chapter 3 current configuration

Explicit --method pfo|c_ppo|gradient_ppo|reflective_ppo|ppo_plus also
routes train/evaluate to comparators for compatibility. All output directories
must be new physical HY directories; no training starts from inspect/help.
""")
        p.print_help();return
    if args[0]=='ablations':
        import ablations
        value=ablations.registry();print(json.dumps(value,ensure_ascii=False,indent=2));return value
    if args[0]=='config':
        value=json.loads((CHAPTER/'configs/current.json').read_text());print(json.dumps(value,ensure_ascii=False,indent=2));return value
    if args[0] in ('baseline','baselines'):
        from allocation23.chapter3_candidates.cli import main as entry
        return entry(args[1:] or ['inspect'])
    if _external(args):
        from allocation23.chapter3_candidates.cli import main as entry
        return entry(args)
    from allocation23.chapter3_candidates.native import main as entry
    return entry(args)


if __name__=='__main__':
    main()
