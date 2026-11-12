"""Unified checkout entry; existing A/B training remains unchanged."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def inspection(argv):
    from . import __version__
    from .environments import build_environment
    p = argparse.ArgumentParser(description='Inspect the active research contracts without training')
    p.add_argument('--domain', choices=('v5', 'radio'))
    p.add_argument('--chapter', type=int, choices=(2, 3, 4, 5))
    args = p.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    metadata = root/'PROJECT.json'
    if not metadata.is_file():
        metadata = Path(__file__).resolve().parent/'project.json'
    project = json.loads(metadata.read_text())
    cases = []
    for chapter in ([args.chapter] if args.chapter else [2, 3]):
        if chapter not in (2, 3):
            cases.append(dict(chapter=chapter, status='not_integrated'))
            continue
        for domain in ([args.domain] if args.domain else ['v5', 'radio']):
            env, spec, physical = build_environment(domain, chapter)
            cases.append(dict(chapter=chapter, domain=domain, status='available',
                objects=spec.groups, horizon=spec.horizon, local_dim=spec.local_dim,
                global_dim=spec.global_dim, methods=['A','B'],
                temporal='fixed' if chapter == 2 else 'adaptive',
                initial_budget=spec.initial_budget,
                action_cost_scale=spec.action_cost_scale))
            env.close()
    print(json.dumps(dict(version=__version__, root=str(root), cases=cases,
        chapter_status=project['methods']), ensure_ascii=False, indent=2))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args in (['-h'], ['--help']):
        help_parser = argparse.ArgumentParser(description='Unified resource allocation research entry',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''Commands:\n  inspect                 Current environments and chapter availability\n  train                   Train A/B on a current native task\n  evaluate                Evaluate an A/B checkpoint\n  plot                    Plot registered A/B runs\n  baseline                inspect/train/evaluate shared external algorithms\n  compare                 Preflight or run the unified seven-method protocol\n  chapter2, chapter3      Optional prefix selecting the paper chapter\n\nUse COMMAND --help for its arguments.''')
        help_parser.print_help()
        return
    if args and args[0] in ('chapter2', 'chapter3'):
        chapter = args.pop(0)[-1]
        if not args:
            args = ['inspect']
        if '--chapter' in args:
            raise ValueError('Use either the chapter prefix or --chapter, not both')
        args += ['--chapter', chapter]
    if args and args[0] == 'inspect':
        return inspection(args[1:])
    if args and args[0] in ('baseline', 'baselines'):
        from .baselines.__main__ import main as baseline_main
        return baseline_main(args[1:])
    if args and args[0] in ('compare', 'comparison'):
        from .comparison_suite import main as comparison_main
        return comparison_main(args[1:])
    from .__main__ import native_main
    return native_main(args)
