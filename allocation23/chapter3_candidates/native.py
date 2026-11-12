"""Chapter 3 method A and its four temporal ablations, with one final model.

The existing Task episode collector and PPO update are used without changing
their mathematics. Chapter 2 and the historical A/B checkpoint runner are not
modified. This entry is an editable development entry, not a frozen experiment.
"""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root

import argparse
from contextlib import contextmanager
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback


from ..paths import iter_source_files

SCHEMA = 'chapter3-native-a-final-only-v1'
TEMPORALS = ('adaptive', 'pace', 'budget', 'task_no_budget', 'fixed_nominal')
ARMS = {
    'A_FULL': dict(temporal='adaptive', quota_reference='pace'),
    'A_PACE': dict(temporal='pace', quota_reference='pace'),
    'A_BUDGET_ONLY': dict(temporal='budget', quota_reference='pace'),
    'A_NO_BUDGET': dict(temporal='task_no_budget', quota_reference='pace'),
    'A_CONSTANT_REFERENCE': dict(temporal='adaptive', quota_reference='nominal'),
}
TRAIN_MASTER = 202610303101
TEST_MASTER = 202610305199


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def _write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def _hy_path(value, *, new=False):
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(Path(str(_resolve_path('/hy-tmp')))):
        raise ValueError('Chapter 3 outputs and runtime caches must reside under /hy-tmp')
    if new and path.exists():
        raise FileExistsError('Use a new output directory; final evaluation models cannot resume training')
    return path


def _prepare_runtime(output):
    cache = output/'runtime'
    cache.mkdir(parents=True)
    for name, subdir in [('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'xdg'),
                         ('MPLCONFIGDIR', 'matplotlib'), ('TORCH_HOME', 'torch'),
                         ('CUDA_CACHE_PATH', 'cuda'), ('TRITON_CACHE_DIR', 'triton')]:
        target = cache/subdir
        target.mkdir()
        os.environ[name] = str(target)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'


def source_identity():
    """Bind actual maintained sources under stable logical package names."""
    return {logical: _sha(path) for logical, path in iter_source_files()}


def _artifacts(folder):
    """Hash closed scientific outputs, including a full gzip CRC/footer read."""
    result = {}
    for path in sorted(Path(folder).rglob('*')):
        relative = path.relative_to(folder)
        if (not path.is_file() or relative.parts[0] == 'runtime' or
                str(relative) in ('completed.json', 'progress.json', 'failure.json')):
            continue
        if path.suffix == '.gz':
            with gzip.open(path, 'rb') as stream:
                for _ in iter(lambda: stream.read(1024**2), b''):
                    pass
        result[str(relative)] = _sha(path)
    return result


def _verify_artifacts(folder, expected):
    if not isinstance(expected, dict) or not expected:
        raise ValueError('A completed run with its sealed scientific artifacts is required')
    root = Path(folder).resolve()
    for relative, digest in expected.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or _sha(path) != digest:
            raise ValueError('Completed scientific artifact differs: '+str(relative))
    # Scientific additions are also rejected; runtime cache files are excluded.
    if _artifacts(root) != expected:
        raise ValueError('The completed scientific artifact set changed')


class _Rows:
    def __init__(self, path, fields):
        self.stream = gzip.open(path, 'wt', newline='')
        self.writer = csv.DictWriter(self.stream, fieldnames=fields)
        self.writer.writeheader()

    def add(self, **row):
        self.writer.writerow(row)

    def flush(self):
        self.stream.flush()

    def close(self):
        self.stream.close()


@contextmanager
def _preserved_rng():
    import numpy as np
    import torch
    py, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def _verify_inputs(task):
    identity = task.data_identity
    if task.condition_file is not None and (
            _sha(task.condition_file) != identity['condition']['config_sha256']):
        raise RuntimeError('Registered physical condition changed during execution')
    if task.domain == 'radio' and (
            _sha(task.data_root/'split_manifest.json') != identity['split_manifest_sha256']):
        raise RuntimeError('Registered image split changed during execution')


def _evaluate(task, actor, *, paths, phase, step, rows, curve, capture_file=None,
              kind='development'):
    import numpy as np
    from ..io import state_hash
    before = state_hash(actor)
    _verify_inputs(task)
    scores, costs, max_error = [], [], 0.
    action_paths = hashlib.sha256()
    captured = {}
    with _preserved_rng():
        for index in range(paths):
            result = task.episode(phase, index, actor, deterministic=True, capture=True)
            reward, trace = task.replay(phase, index, result)
            error = abs(reward-result['native_return'])
            fraction = result['total_used']/task.spec.initial_budget
            if error > 1e-12 or abs(fraction-1.) > 1e-10:
                raise ArithmeticError('Native deterministic replay or full-spend gate failed')
            max_error = max(max_error, error)
            action_paths.update(int(index).to_bytes(8, 'little'))
            action_paths.update(result['actions'].tobytes())
            scores.append(100*result['native_return'])
            costs.append(fraction)
            rows.add(training_step=step, phase=phase, evaluation_kind=kind, path=index,
                     path_seed=result['plan']['task_seed'], native_return=result['native_return'],
                     score_x100=scores[-1], resource_fraction=fraction,
                     remaining=result['remaining'], replay_error=error,
                     replay_trace_sha256=trace,
                     actions_sha256=hashlib.sha256(result['actions'].tobytes()).hexdigest())
            if capture_file is not None and index == 0:
                captured = dict(actions=result['actions'], rewards=result['rewards'],
                                period_totals=result['period_totals'])
                for name in result['observations'][0]:
                    captured['public_'+name] = np.stack([s[name] for s in result['observations']])
    if before != state_hash(actor):
        raise RuntimeError('Deterministic evaluation changed actor state')
    _verify_inputs(task)
    summary = dict(training_step=step, phase=phase, evaluation_kind=kind, paths=paths,
        score_mean=float(np.mean(scores)), score_path_sd=float(np.std(scores)),
        native_mean=float(np.mean(scores)/100),
        resource_fraction_mean=float(np.mean(costs)), replay_max_error=max_error,
        full_spend_pass=True, action_paths_sha256=action_paths.hexdigest())
    curve.add(**summary)
    rows.flush()
    curve.flush()
    if capture_file is not None:
        np.savez_compressed(capture_file, **captured)
    return summary


def _task(domain, seed, master, *, data_root=None, holdout=False, condition_file=None):
    from ..task import Task
    return Task(domain, 3, 'A', seed, path_master=master, data_root=data_root,
                holdout=holdout, condition_file=condition_file)


def train(args):
    horizon = 32 if args.domain == 'radio' else 24
    episodes = (args.steps//horizon if args.steps is not None else
                960 if args.episodes is None else args.episodes)
    if (episodes <= 0 or episodes % 8 or
            (args.steps is not None and args.steps != episodes*horizon)):
        raise ValueError('Use a positive whole multiple of 8 episodes (8H environment steps)')
    if min(args.eval_every, args.development_paths) <= 0 or args.final_paths < 0:
        raise ValueError('Positive development counts and nonnegative final paths required')
    if args.path_master == args.final_master:
        raise ValueError('Training and independent-test namespaces must differ')
    if args.method != 'A' or args.variant != 'A_FULL':
        raise ValueError('This entry exposes method A and its four temporal ablations only')
    output = _hy_path(args.output, new=True)
    output.mkdir(parents=True)
    _prepare_runtime(output)
    import numpy as np
    import torch
    from ..io import state_hash
    from ..learning import PPO, update
    from ..methods import make_actor
    from ..training import configure_runtime, fresh
    configure_runtime()
    if args.arm is not None:
        chosen = ARMS[args.arm]
        if ((args.temporal is not None and args.temporal != chosen['temporal']) or
                (args.quota_reference is not None and args.quota_reference != chosen['quota_reference'])):
            raise ValueError('Explicit temporal/reference arguments conflict with the registered ablation arm')
        args.temporal, args.quota_reference = chosen['temporal'], chosen['quota_reference']
    else:
        args.temporal = args.temporal or 'adaptive'
        args.quota_reference = args.quota_reference or 'pace'
        args.arm = next((name for name, chosen in ARMS.items()
                         if chosen == dict(temporal=args.temporal, quota_reference=args.quota_reference)),
                        'historical_custom')
    started = time.monotonic()
    sources = source_identity()
    task = test_task = None
    writers = []
    digest_cache = None
    try:
        if args.domain == 'radio':
            from ..chapter3_baselines.execution_cache import install
            digest_cache = install(None if args.data_root is None else Path(args.data_root)/'images.npy')
        task = _task(args.domain, args.seed, args.path_master, data_root=args.data_root,
                     condition_file=args.condition_file)
        if task.spec.horizon != horizon:
            raise ValueError('This entry retains the registered domain horizon; use a separate protocol for horizon changes')
        # fresh() only reads these original construction arguments. PPO math and
        # both private training random streams remain the original implementation.
        args.chapter = 3
        actor, critic, actor_optimizer, value_optimizer, generator, rng = fresh(task, args)
        initial_actor = state_hash(actor)
        configuration = dict(domain=args.domain, chapter=3, method='A', seed=args.seed,
            arm=args.arm, variant=args.variant, temporal=args.temporal, quota_reference=args.quota_reference,
            path_master=args.path_master, final_master=args.final_master,
            data_root=str(task.data_root), condition_file=task.condition_file,
            episodes=episodes, training_steps=episodes*horizon,
            eval_every_steps=args.eval_every, development_paths=args.development_paths,
            final_paths=args.final_paths, device='cpu')
        protocol = dict(schema=SCHEMA, configuration=configuration,
            source_identity=sources, spec=task.spec.to_dict(), physical=task.physical_dict,
            data=task.data_identity, actor_metadata=actor.metadata(), ppo=dict(PPO),
            evaluation='deterministic latent mean; same physical decoder; native replay',
            evaluation_timing='first complete-episode boundary reaching each requested step threshold',
            storage='one fixed endpoint actor; no critic, Adam, replay, periodic models or resume state',
            quota_transform_description='public full-spend [lower,upper] interval with exact neutral-reference endpoint branches; inherited actor metadata prose is historical',
            research_status='editable integration entry; formal comparisons require a separate frozen protocol')
        _write(output/'protocol.json', protocol)
        step_rows = _Rows(output/'training_steps.csv.gz', ['training_step', 'episode',
            'episode_step', 'path_seed', 'native_reward', 'physical_amount'])
        episode_rows = _Rows(output/'training_episodes.csv.gz', ['episode', 'training_step',
            'path_seed', 'native_return', 'score_x100', 'resource_fraction', 'remaining'])
        update_rows = _Rows(output/'updates.csv.gz', ['training_step', 'episode', 'metrics_json'])
        evaluation_rows = _Rows(output/'evaluation_paths.csv.gz', ['training_step', 'phase', 'evaluation_kind',
            'path', 'path_seed', 'native_return', 'score_x100', 'resource_fraction',
            'remaining', 'replay_error', 'replay_trace_sha256', 'actions_sha256'])
        curve_rows = _Rows(output/'evaluation_curve.csv.gz', ['training_step', 'phase', 'evaluation_kind',
            'paths', 'score_mean', 'score_path_sd', 'native_mean',
            'resource_fraction_mean', 'replay_max_error', 'full_spend_pass', 'action_paths_sha256'])
        writers = [step_rows, episode_rows, update_rows, evaluation_rows, curve_rows]
        summaries = []

        def evaluate_at(step, endpoint=False):
            saved_generator = generator.get_state().clone()
            saved_rng = json.dumps(rng.bit_generator.state, sort_keys=True)
            saved_critic = state_hash(critic)
            summary = _evaluate(task, actor, paths=args.development_paths, phase='dev',
                step=step, rows=evaluation_rows, curve=curve_rows,
                capture_file=output/'endpoint_example.npz' if endpoint else None)
            if (not torch.equal(saved_generator, generator.get_state()) or
                    saved_rng != json.dumps(rng.bit_generator.state, sort_keys=True) or
                    saved_critic != state_hash(critic)):
                raise RuntimeError('Evaluation changed training RNG or critic')
            summaries.append(summary)
            return summary

        evaluate_at(0)
        batches, optimizer_steps, next_eval = [], 0, args.eval_every
        for episode in range(1, episodes+1):
            result = task.episode('train', episode-1, actor, critic, generator, training=True)
            step = episode*horizon
            fraction = result['total_used']/task.spec.initial_budget
            if abs(fraction-1.) > 1e-10:
                raise ArithmeticError('Training episode did not allocate the full initial resource')
            for t, reward in enumerate(result['rewards']):
                step_rows.add(training_step=(episode-1)*horizon+t+1, episode=episode,
                    episode_step=t, path_seed=result['plan']['task_seed'],
                    native_reward=reward, physical_amount=result['period_totals'][t])
            episode_rows.add(episode=episode, training_step=step,
                path_seed=result['plan']['task_seed'], native_return=result['native_return'],
                score_x100=100*result['native_return'], resource_fraction=fraction,
                remaining=result['remaining'])
            batches.append(result['batch'])
            if len(batches) == PPO['batch_episodes']:
                stats = update(actor, critic, actor_optimizer, value_optimizer, batches, rng)
                optimizer_steps += int(stats['optimizer_steps'])
                batches.clear()
                update_rows.add(training_step=step, episode=episode,
                                metrics_json=json.dumps(stats, sort_keys=True))
            if step >= next_eval and episode != episodes:
                if source_identity() != sources:
                    raise RuntimeError('Scientific source changed during training')
                evaluate_at(step)
                while next_eval <= step:
                    next_eval += args.eval_every
            for writer in (step_rows, episode_rows, update_rows):
                writer.flush()
            _write(output/'progress.json', dict(status='training', episode=episode,
                episodes=episodes, training_steps=step, optimizer_steps=optimizer_steps,
                elapsed_seconds=time.monotonic()-started, pid=os.getpid()))
        if batches:
            raise AssertionError('Uncommitted final A training batch')
        endpoint = evaluate_at(episodes*horizon, endpoint=True)
        model = output/'model'
        model.mkdir()
        payload = dict(schema=SCHEMA, configuration=configuration, spec=task.spec.to_dict(),
            physical=task.physical_dict, data=task.data_identity, source_identity=sources,
            actor_metadata=actor.metadata(), actor=actor.state_dict())
        torch.save(payload, model/'actor.pt')
        _write(model/'model.json', dict(schema=SCHEMA, method='A',
            configuration=configuration, model_file='actor.pt', sha256=_sha(model/'actor.pt'),
            evaluation_only=True, optimizer_saved=False, replay_saved=False))
        reloaded = make_actor(task.spec, 'A', 3, args.seed, variant=args.variant,
            temporal=args.temporal, quota_reference=args.quota_reference)
        reloaded.load_state_dict(torch.load(model/'actor.pt', map_location='cpu',
                                           weights_only=False)['actor'], strict=True)
        if state_hash(reloaded) != state_hash(actor):
            raise RuntimeError('Endpoint actor save/load changed parameters')
        reload_summary = _evaluate(task, reloaded, paths=args.development_paths,
            phase='dev', step=episodes*horizon, rows=evaluation_rows, curve=curve_rows,
            kind='endpoint_reload')
        if ({k: v for k, v in reload_summary.items() if k != 'evaluation_kind'} !=
                {k: v for k, v in endpoint.items() if k != 'evaluation_kind'}):
            raise RuntimeError('Reloaded actor differs on the fixed development paths')
        final = None
        if args.final_paths:
            test_task = _task(args.domain, args.seed, args.final_master, data_root=args.data_root,
                             holdout=True, condition_file=args.condition_file)
            final = _evaluate(test_task, reloaded, paths=args.final_paths, phase='final',
                step=episodes*horizon, rows=evaluation_rows, curve=curve_rows, kind='independent_test')
        if source_identity() != sources:
            raise RuntimeError('Scientific source changed before completion')
        data_gate = digest_cache.verify_final() if digest_cache is not None else None
        # A successful completion may only become visible after every gzip
        # footer is durable to the filesystem and independently readable.
        for writer in writers:
            writer.close()
        artifacts = _artifacts(output)
        receipt = dict(schema=SCHEMA, status='complete', success=True,
            training_steps=episodes*horizon, episodes=episodes, optimizer_steps=optimizer_steps,
            actor_changed=state_hash(actor) != initial_actor, development=endpoint,
            independent_test=final, endpoint_reload_pass=True, full_spend_pass=True,
            source_identity=sources, source_unchanged=True, input_final_verification=data_gate,
            model=str(model/'actor.pt'), model_sha256=_sha(model/'actor.pt'),
            model_count=1, optimizer_saved=False, replay_saved=False, periodic_models_saved=False,
            artifacts=artifacts, protocol_sha256=artifacts['protocol.json'],
            gzip_complete_before_success=True, seconds=time.monotonic()-started)
        _write(output/'completed.json', receipt)
        _write(output/'progress.json', dict(status='complete', training_steps=episodes*horizon,
                                          episodes=episodes, pid=os.getpid()))
        return receipt
    except BaseException:
        _write(output/'failure.json', dict(status='failed', traceback=traceback.format_exc(),
                                          seconds=time.monotonic()-started))
        raise
    finally:
        for writer in writers:
            writer.close()
        if test_task is not None:
            test_task.env.close()
        if task is not None:
            task.env.close()


def evaluate(args):
    if args.paths <= 0:
        raise ValueError('A positive evaluation path count is required')
    run = Path(args.run).expanduser().resolve()
    completed = json.loads((run/'completed.json').read_text())
    if completed.get('schema') != SCHEMA or completed.get('status') != 'complete':
        raise ValueError('A completed Chapter 3 A endpoint is required')
    _verify_artifacts(run, completed.get('artifacts'))
    protocol = json.loads((run/'protocol.json').read_text())
    if _sha(run/'protocol.json') != completed.get('protocol_sha256'):
        raise ValueError('Completed protocol identity changed')
    meta = json.loads((run/'model'/'model.json').read_text())
    model_file = (run/'model'/meta['model_file']).resolve()
    if (not model_file.is_relative_to(run/'model') or meta['schema'] != SCHEMA or
            not meta.get('evaluation_only') or _sha(model_file) != meta['sha256'] or
            meta['sha256'] != completed.get('model_sha256')):
        raise ValueError('Fixed endpoint model SHA changed')
    output = _hy_path(args.output, new=True)
    output.mkdir(parents=True)
    _prepare_runtime(output)
    import torch
    from ..methods import make_actor
    from ..training import configure_runtime
    configure_runtime()
    payload = torch.load(model_file, map_location='cpu', weights_only=False)
    if payload['schema'] != SCHEMA or payload['source_identity'] != source_identity():
        raise ValueError('Use this implementation or its frozen snapshot to evaluate this endpoint')
    for key in ('configuration', 'spec', 'physical', 'data', 'actor_metadata', 'source_identity'):
        if payload[key] != protocol[key]:
            raise ValueError('Model and protocol differ: '+key)
    if (meta['configuration'] != protocol['configuration'] or
            completed['source_identity'] != protocol['source_identity']):
        raise ValueError('Model manifest or completed source differs from the protocol')
    config = payload['configuration']
    digest_cache = None
    if config['domain'] == 'radio':
        from ..chapter3_baselines.execution_cache import install
        digest_cache = install(Path(args.data_root or config['data_root'])/'images.npy')
    master = args.path_master
    if master is None:
        master = config['path_master'] if args.phase == 'dev' else config['final_master']
    if args.phase == 'final' and master == config['path_master']:
        raise ValueError('Independent test namespace must differ from the training namespace')
    task = _task(config['domain'], config['seed'], master,
        data_root=args.data_root or config['data_root'], holdout=args.phase == 'final',
        condition_file=config['condition_file'])
    rows = curve = None
    try:
        if task.spec.to_dict() != payload['spec'] or task.physical_dict != payload['physical']:
            raise ValueError('Evaluation task physics differ from the endpoint')
        for name in ('images_sha256', 'split_manifest_sha256', 'condition'):
            if task.data_identity.get(name) != payload['data'].get(name):
                raise ValueError('Evaluation input differs from the endpoint: '+name)
        actor = make_actor(task.spec, 'A', 3, config['seed'], variant=config['variant'],
            temporal=config['temporal'], quota_reference=config['quota_reference'])
        actor.load_state_dict(payload['actor'], strict=True)
        rows = _Rows(output/'evaluation_paths.csv.gz', ['training_step', 'phase', 'evaluation_kind', 'path',
            'path_seed', 'native_return', 'score_x100', 'resource_fraction', 'remaining',
            'replay_error', 'replay_trace_sha256', 'actions_sha256'])
        curve = _Rows(output/'evaluation_curve.csv.gz', ['training_step', 'phase', 'evaluation_kind', 'paths',
            'score_mean', 'score_path_sd', 'native_mean', 'resource_fraction_mean',
            'replay_max_error', 'full_spend_pass', 'action_paths_sha256'])
        result = _evaluate(task, actor, paths=args.paths, phase=args.phase,
            step=config['training_steps'], rows=rows, curve=curve,
            capture_file=output/'endpoint_example.npz', kind='independent_test' if args.phase == 'final' else 'development')
        if source_identity() != payload['source_identity']:
            raise RuntimeError('Scientific source changed during endpoint evaluation')
        result.update(schema=SCHEMA, status='complete', model_sha256=meta['sha256'],
            run=str(run), path_master=master, test_split_opened=task.holdout,
            input_final_verification=digest_cache.verify_final() if digest_cache is not None else None)
        rows.close()
        curve.close()
        result['artifacts'] = _artifacts(output)
        result['gzip_complete_before_success'] = True
        _write(output/'completed.json', result)
        return result
    finally:
        if rows is not None:
            rows.close()
        if curve is not None:
            curve.close()
        task.env.close()


def parser():
    parser = argparse.ArgumentParser(description='Chapter 3 A and four temporal ablations; final-only storage')
    commands = parser.add_subparsers(dest='command', required=True)
    t = commands.add_parser('train')
    t.add_argument('--domain', choices=('radio', 'v5'), required=True)
    t.add_argument('--method', choices=('A',), default='A')
    t.add_argument('--variant', choices=('A_FULL',), default='A_FULL')
    t.add_argument('--arm', choices=tuple(ARMS))
    t.add_argument('--temporal', choices=TEMPORALS,
                   help='Explicit historical/custom mode; prefer one of the registered --arm values')
    t.add_argument('--quota-reference', choices=('pace', 'nominal'))
    budget = t.add_mutually_exclusive_group()
    budget.add_argument('--episodes', type=int)
    budget.add_argument('--steps', type=int)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--eval-every', '--eval-every-steps', type=int, default=512,
                   help='Environment-step threshold; evaluation follows the next complete episode')
    t.add_argument('--development-paths', type=int, default=8)
    t.add_argument('--final-paths', '--holdout-paths', type=int, default=0)
    t.add_argument('--path-master', type=int, default=TRAIN_MASTER)
    t.add_argument('--final-master', '--holdout-master', type=int, default=TEST_MASTER)
    t.add_argument('--data-root')
    t.add_argument('--condition-file')
    t.add_argument('--device', choices=('cpu',), default='cpu')
    t.add_argument('--output', required=True)
    e = commands.add_parser('evaluate')
    e.add_argument('--run', required=True)
    e.add_argument('--output', required=True)
    e.add_argument('--paths', type=int, default=128)
    e.add_argument('--phase', choices=('dev', 'final'), default='final')
    e.add_argument('--path-master', type=int)
    e.add_argument('--data-root')
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    result = train(args) if args.command == 'train' else evaluate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
