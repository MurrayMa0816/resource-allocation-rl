"""Current Chapter 3 online comparator command line."""
from __future__ import annotations

import argparse
import json
import os


def parser():
    from . import METHODS
    p = argparse.ArgumentParser(description='Chapter 3 current online comparators; native full-spend task')
    sub = p.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect', help='Inspect current methods and public task contracts')
    inspect.add_argument('--method', choices=METHODS)
    inspect.add_argument('--domain', choices=('radio', 'v5'))
    inspect.add_argument('--data-root', default=os.environ.get('ALLOCATION23_DATA_ROOT'),
                         help='Image data directory; defaults to ALLOCATION23_DATA_ROOT when set')
    train = sub.add_parser('train', help='Train from scratch; keep curves and one fixed endpoint')
    train.add_argument('--method', choices=METHODS, required=True)
    train.add_argument('--domain', choices=('radio', 'v5'), required=True)
    train.add_argument('--seed', type=int, default=0)
    train.add_argument('--steps', type=int, required=True, help='True learning environment steps; whole native episodes')
    train.add_argument('--eval-every', type=int, default=512,
                       help='Evaluate at the first whole-episode boundary reaching each threshold')
    train.add_argument('--development-paths', type=int, default=8)
    train.add_argument('--path-master', type=int, default=202610303101)
    train.add_argument('--config', help='JSON object or JSON file overriding the recorded task-adapted profile')
    train.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    train.add_argument('--cpu-threads', type=int, default=1)
    train.add_argument('--data-root', default=os.environ.get('ALLOCATION23_DATA_ROOT'),
                       help='Image data directory; defaults to ALLOCATION23_DATA_ROOT when set')
    train.add_argument('--storage-profile', choices=('full', 'lean'), default='full',
                       help='full preserves legacy artifacts; lean keeps all numeric curves and path gates, plus endpoint path 0')
    train.add_argument('--output', required=True, help='New physical directory under /hy-tmp')
    train.add_argument('--engineering-gate', action='store_true', help='Mark a short interface check, not performance evidence')
    evaluate = sub.add_parser('evaluate', help='Evaluate a completed fixed endpoint on a separate registered stream')
    evaluate.add_argument('--run', required=True)
    evaluate.add_argument('--output', required=True, help='New physical directory under /hy-tmp')
    evaluate.add_argument('--paths', type=int, default=8)
    evaluate.add_argument('--phase', choices=('dev', 'final', 'engineering'), default='dev')
    evaluate.add_argument('--condition-file',
                          help='Registered complete physical condition; object count, horizon and policy dimensions must match training')
    evaluate.add_argument('--path-master', type=int,
                          help='Evaluation namespace; final requires an explicit namespace distinct from training')
    evaluate.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    evaluate.add_argument('--cpu-threads', type=int, default=1)
    evaluate.add_argument('--data-root', default=os.environ.get('ALLOCATION23_DATA_ROOT'),
                          help='Image data directory; defaults to ALLOCATION23_DATA_ROOT when set')
    evaluate.add_argument('--storage-profile', choices=('full', 'lean'), default='full',
                          help='full preserves legacy artifacts; lean captures only preregistered final path 0')
    # The unified checkout forwards this flag after its chapter prefix.
    for command in (inspect, train, evaluate):
        command.add_argument('--chapter', choices=(3,), type=int, default=3, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    from . import runner
    args = parser().parse_args(argv)
    result = getattr(runner, args.command)(args)
    print(json.dumps(runner.jsonable(result), ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
