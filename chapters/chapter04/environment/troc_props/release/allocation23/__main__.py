"""Public command line: train, evaluate, and plot."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from .io import TRAIN_MASTER, TEST_MASTER


def parser():
    root = argparse.ArgumentParser(description='Chapter 2/3 resource allocation: A and B')
    commands = root.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='Train a new model or resume its own checkpoint')
    train.add_argument('--domain', choices=('v5', 'radio'), required=True)
    train.add_argument('--chapter', type=int, choices=(2, 3), required=True)
    train.add_argument('--method', choices=('A', 'B'), required=True)
    train.add_argument('--seed', type=int, default=0)
    train.add_argument('--episodes', type=int, default=960)
    train.add_argument('--eval-every', type=int, default=240)
    train.add_argument('--development-paths', type=int, default=32)
    train.add_argument('--holdout-paths', type=int, default=128)
    train.add_argument('--path-master', type=int, default=TRAIN_MASTER)
    train.add_argument('--holdout-master', type=int, default=TEST_MASTER)
    train.add_argument('--radio-groups', type=int, default=100)
    train.add_argument('--data-root', help='Prepared EuroSAT data directory; defaults to bundled data')
    train.add_argument('--condition-file', help='Registered physical condition JSON; same identity on train/evaluation/resume')
    train.add_argument('--variant', help='Optional registered spatial ablation; omit for the full method')
    train.add_argument('--temporal', choices=('fixed', 'pace', 'schedule', 'budget', 'adaptive',
                                            'task_no_budget', 'fixed_nominal'),
                       help='Optional temporal ablation; omit for the chapter default')
    train.add_argument('--quota-reference', choices=('pace', 'nominal'), default='pace')
    train.add_argument('--checkpoint-every', type=int, help='Whole-episode checkpoint interval; defaults to evaluation interval')
    train.add_argument('--output', required=True)
    train.add_argument('--resume', action='store_true')
    train.add_argument('--stop-after', type=int, help=argparse.SUPPRESS)
    evaluation = commands.add_parser('evaluate', help='Evaluate a saved release checkpoint')
    evaluation.add_argument('--checkpoint', required=True)
    evaluation.add_argument('--paths', type=int, default=128)
    evaluation.add_argument('--path-master', type=int)
    evaluation.add_argument('--data-root')
    evaluation.add_argument('--output', required=True)
    plot = commands.add_parser('plot', help='Build chapter figures and tables from completed runs')
    plot.add_argument('--runs', required=True)
    plot.add_argument('--output', required=True)
    return root


def native_main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'train':
        from .training import train
        train(args)
    elif args.command == 'evaluate':
        from .training import evaluate_checkpoint
        evaluate_checkpoint(args)
    else:
        from .plotting import plot_runs
        plot_runs(args.runs, args.output)


def main(argv=None):
    from .cli import main as unified_main
    return unified_main(argv)


if __name__ == '__main__':
    main()
