"""Inspect C6 final materials; scientific programs require an explicit new source capsule."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parent
sys.path.insert(0, str(HERE))
from capsule import inspect_sources, prepare_source_capsule, verify_source_capsule


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('inspect')
    commands.add_parser('check')
    capsule = commands.add_parser('capsule', help='Materialize ordinary source files in a new explicit directory')
    capsule.add_argument('--output', type=Path, required=True)
    for name in ['train', 'evaluate', 'comparison']:
        item = commands.add_parser(name)
        item.add_argument('--config', type=Path, required=True)
        item.add_argument('--source-capsule', type=Path)
        item.add_argument('--execute', action='store_true')
        if name == 'comparison':
            item.add_argument('--method', choices=['pasta', 'cola'], required=True)
    args = parser.parse_args(argv)
    if args.command == 'inspect':
        print(json.dumps({'chapter': 6, 'methods': ['SeDVal', 'PASTA', 'COLA'],
            'region_count': 89, 'horizon_days': 81, 'model_uses': 15,
            'directories': ['environment', 'algorithms', 'configs', 'data', 'models', 'results', 'scripts', 'figures'],
            'source_views': '../configs/source_views.json',
            'shared_comparison_implementation': '../../chapter05/algorithms/baselines',
            'default': 'read existing metadata only', 'scientific_execution': 0}, ensure_ascii=False, indent=2))
        return 0
    if args.command == 'check':
        result = inspect_sources()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(result['status'] != 'PASS')
    if args.command == 'capsule':
        print(json.dumps(prepare_source_capsule(args.output), ensure_ascii=False, indent=2))
        return 0
    config = args.config.expanduser().resolve()
    document = json.loads(config.read_text())
    if not isinstance(document, dict):
        parser.error('Configuration must be a JSON object')
    selection = {'train': ('sedval', 'execution/worker_train.py'),
                 'evaluate': ('sedval', 'evaluation/evaluate.py'),
                 'comparison': ('comparison', 'execution/cola_worker.py' if getattr(args, 'method', '') == 'cola' else 'execution/worker.py')}
    view, logical = selection[args.command]
    if not args.execute:
        print(json.dumps({'status': 'CONFIGURATION_READ', 'config': str(config),
            'source_view': view, 'entry': logical, 'execute': False,
            'note': 'Historical completed job configurations are provenance; a new scientific run needs its own source-bound configuration and new output.'}, ensure_ascii=False, indent=2))
        return 0
    if args.source_capsule is None:
        parser.error('An explicit --source-capsule and new registered configuration are required')
    root = args.source_capsule.expanduser().resolve()
    if not (root / 'capsule.json').is_file():
        parser.error('Materialize and verify a fresh source capsule first')
    verify_source_capsule(root)
    target = root / view / logical
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CHAPTER06_SOURCE_CAPSULE=str(root))
    existing = env.get('PYTHONPATH')
    env['PYTHONPATH'] = str(HERE) + (os.pathsep + existing if existing else '')
    runner = CHAPTER.parent / 'chapter02/scripts/support/run_source.py'
    return subprocess.run([sys.executable, '-B', str(runner),
                           '--source-capsule', str(root), '--script', str(target),
                           '--', '--config', str(config)], env=env, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
