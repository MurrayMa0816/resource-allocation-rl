"""Native online collection for the current five Chapter 3 comparators.

The physics and full-spend decoder are shared with the main project. This
module owns collection, evaluation isolation and storage, never a learner's
mathematics. Historical scientific run directories are not imported or edited.
"""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root

from contextlib import contextmanager, ExitStack
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

from ..paths import PROJECT_ROOT, iter_source_files

SCHEMA = 'chapter3-integrated-online-candidates-v1'
HY = Path(str(_resolve_path('/hy-tmp')))
PROJECT = PROJECT_ROOT


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(jsonable(value), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def source_identity():
    # Enumerate logical package names through their unique maintained entities;
    # pathlib.rglob does not descend into relocated package-directory symlinks.
    return {logical: sha(path) for logical, path in iter_source_files()}


def new_output(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(HY):
        raise ValueError('Results, test models and caches must physically reside under /hy-tmp')
    path.mkdir(parents=True, exist_ok=False)
    return path


def runtime(args, output=None):
    cache = (output / 'runtime') if output else HY / 'resource_allocation_results/chapter3/project_runtime'
    for key, leaf in [('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'xdg'), ('MPLCONFIGDIR', 'mpl'),
                      ('TORCH_HOME', 'torch'), ('CUDA_CACHE_PATH', 'cuda'),
                      ('TORCH_EXTENSIONS_DIR', 'extensions'), ('TRITON_CACHE_DIR', 'triton'),
                      ('TORCHINDUCTOR_CACHE_DIR', 'inductor')]:
        target = cache / leaf
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    threads = getattr(args, 'cpu_threads', 1)
    if threads < 1:
        raise ValueError('cpu_threads must be positive')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(threads)
    if getattr(args, 'data_root', None):
        os.environ['ALLOCATION23_DATA_ROOT'] = str(Path(args.data_root).expanduser().resolve())
    os.environ['CH3_REPAIR_SPATIAL_DECODER'] = 'proportional'
    import torch
    torch.set_num_threads(threads)
    if getattr(args, 'device', 'cpu') == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; choose an available device explicitly')


def data_identity(domain):
    if domain != 'radio':
        return dict(images_opened=False), None
    from .. import task
    from ..chapter3_baselines.execution_cache import install
    cache = install(task.DATA_ROOT / 'images.npy')
    result = dict(images_opened=True, data_root=str(task.DATA_ROOT.resolve()),
                  images_sha256=cache.expected,
                  split_manifest_sha256=sha(task.DATA_ROOT / 'split_manifest.json'))
    return result, cache


def verify_data(data, cache):
    if cache is None:
        return dict(images_opened=False)
    if sha(Path(data['data_root']) / 'split_manifest.json') != data['split_manifest_sha256']:
        raise RuntimeError('Image split manifest changed during execution')
    return cache.verify_final()


@contextmanager
def preserved_rng():
    import numpy as np
    import torch
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


class Rows:
    def __init__(self, stack, path, fields):
        self.handle = stack.enter_context(gzip.open(path, 'wt', newline='', compresslevel=5))
        self.writer = csv.DictWriter(self.handle, fieldnames=fields)
        self.writer.writeheader()

    def add(self, row):
        self.writer.writerow(row)

    def flush(self):
        self.handle.flush()


def array_sha(value):
    import numpy as np
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def storage_contract(profile):
    if profile not in ('full', 'lean'):
        raise ValueError('Choose an explicit full or lean storage profile')
    lean = profile == 'lean'
    return dict(profile=profile, version=1,
                training_numeric_steps_episodes_update_metrics='complete gzip CSV',
                evaluation_numeric_steps_and_path_summaries='complete gzip CSV',
                native_replay_full_spend_policy_rng_gates='every evaluated path; unchanged',
                per_step_array_sha256_saved=not lean,
                initial_development_arrays_saved=not lean,
                representative_path_indices=[0] if lean else 'all',
                representative_capture_phases=['endpoint_development', 'independent_final'] if lean else
                    ['initial_development', 'endpoint_development', 'standalone_evaluation'],
                representative_arrays=['public_observations', 'latent_actions', 'physical_actions'],
                model='one fixed endpoint with the native save/load fields; no periodic checkpoints')


def action_of(agent, observation, deterministic, action_dim):
    import numpy as np
    action = np.asarray(agent.act(observation, deterministic=deterministic), dtype=np.float32)
    if action.shape != (action_dim,) or not np.isfinite(action).all() or (abs(action) > 1.000001).any():
        raise ValueError('Policy action violates the public continuous-control interface')
    return action


def policy_identity(agent):
    """Read-only fingerprint of networks, moments, pending actions and private RNG.

    Do not serialize a replay buffer or Adam state: evaluation cannot reach them
    through the supported act interface. Accounting separately checks their
    sizes/counters. Tensor copies here are transient, never saved to disk.
    """
    import numpy as np
    import torch
    h = hashlib.sha256()

    def add(name, value):
        h.update(name.encode())
        if isinstance(value, torch.Tensor):
            add(name + '/tensor', value.detach().cpu().contiguous().numpy())
        elif isinstance(value, np.ndarray):
            h.update(str(value.dtype).encode())
            h.update(str(value.shape).encode())
            h.update(value.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                add(name + '/' + str(key), value[key])
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                add(name + '/' + str(i), item)
        else:
            h.update(repr(value).encode())

    roots = [('agent', agent)]
    if hasattr(agent, 'online'):
        roots.append(('agent/online', agent.online))
        roots.append(('agent/online/core', agent.online.agent))
    for root_name, root in roots:
        for name, value in sorted(vars(root).items()):
            key = root_name + '/' + name
            if isinstance(value, torch.nn.Module):
                add(key + '/training', value.training)
                add(key, value.state_dict())
            elif isinstance(value, torch.Tensor):
                add(key, value)
            elif isinstance(value, np.random.Generator):
                add(key, value.bit_generator.state)
            elif isinstance(value, torch.Generator):
                add(key, value.get_state())
            elif 'moments' in name and hasattr(value, 'state'):
                add(key, value.state())
            elif name in ('pending', '_pending', 'cached_next'):
                add(key, value)
    return h.hexdigest()


def evaluate_paths(agent, *, domain, seed, master, phase, paths, step, kind,
                   raw_rows, curve_rows, step_rows, capture_path=None,
                   storage_profile='full', condition_file=None, expected_environment_contract=None):
    import numpy as np
    from ..chapter3_baselines.environment import PublicBudgetEnv
    before = dict(agent.accounting())
    before_policy = policy_identity(agent)
    lean = storage_profile == 'lean'
    storage_contract(storage_profile)
    rows, observations, latent_actions, physical_actions = [], [], [], []
    captured_indices = []
    with preserved_rng():
        env = PublicBudgetEnv(domain, seed=seed, phase=phase, path_master=master,
                              holdout=phase == 'final', condition_file=condition_file)
        try:
            if expected_environment_contract is not None and env.contract() != expected_environment_contract:
                raise RuntimeError('Evaluation environment differs from its preregistered physical condition')
            horizon, action_dim = env.horizon, env.action_space.shape[0]
            for index in range(paths):
                obs, _ = env.reset(seed=index)
                total = cost = 0.
                actual_steps = 0
                remaining = 1.
                capture_this = capture_path is not None and (not lean or index == 0)
                obs_path, latent_path, physical_path = [], [], []
                with ExitStack() as policy_scope:
                    if hasattr(agent, 'evaluation_scope'):
                        policy_scope.enter_context(agent.evaluation_scope(common_seed=master + 8800003 + index))
                    for t in range(horizon):
                        action = action_of(agent, obs, True, action_dim)
                        nxt, reward, done, truncated, info = env.step(action)
                        if truncated or bool(done) != (t == horizon - 1):
                            raise RuntimeError('Evaluation did not finish at the native horizon')
                        total += float(reward)
                        cost += float(info['cost'])
                        actual_steps += 1
                        remaining = float(info['remaining_budget_fraction'])
                        physical = np.asarray(info['physical_action'], dtype=np.float64)
                        physical_path.append(physical.copy())
                        if capture_this:
                            obs_path.append(obs.copy())
                            latent_path.append(action.copy())
                        numeric_row = dict(training_step=step, evaluation_kind=kind, path_index=index,
                            episode_step=t, native_reward=reward, resource_cost=info['cost'], cumulative_cost=cost,
                            **(dict(remaining_budget_fraction=remaining) if lean else
                               dict(observation_sha256=array_sha(obs), latent_action_sha256=array_sha(action),
                                    physical_action_sha256=array_sha(physical), next_observation_sha256=array_sha(nxt))))
                        step_rows.add(numeric_row)
                        obs = nxt
                replay = env.replay_last_episode()
                if float(replay['error']) != 0. or abs(cost - 1.) > 1e-10 or abs(remaining) > 1e-10:
                    raise ArithmeticError('Native evaluation replay or full-spend gate failed')
                row = dict(training_step=step, evaluation_kind=kind, path_index=index,
                           path_seed=env.bridge._path_seed, native_return=total, resource_fraction=cost,
                           steps=actual_steps, replay_error=replay['error'],
                           replay_trace_sha256=replay['observed_trace_sha256'],
                           actions_sha256=array_sha(np.stack(physical_path)))
                if lean:
                    row['remaining_budget_fraction'] = remaining
                raw_rows.add(row)
                rows.append(row)
                if capture_this:
                    obs_path.append(obs.copy())
                    observations.append(np.asarray(obs_path, dtype=np.float32))
                    latent_actions.append(np.asarray(latent_path, dtype=np.float32))
                    physical_actions.append(np.asarray(physical_path, dtype=np.float64))
                    captured_indices.append(index)
        finally:
            env.close()
    if dict(agent.accounting()) != before:
        raise RuntimeError('Evaluation changed the learner training counters or replay accounting')
    if policy_identity(agent) != before_policy:
        raise RuntimeError('Evaluation changed a network, normalization state, pending action or private RNG')
    if capture_path is not None:
        captured_rows = [rows[index] for index in captured_indices]
        np.savez_compressed(capture_path, public_observations=np.asarray(observations),
                            latent_actions=np.asarray(latent_actions), physical_actions=np.asarray(physical_actions),
                            path_seeds=np.asarray([row['path_seed'] for row in captured_rows], dtype=np.uint64),
                            **(dict(path_indices=np.asarray(captured_indices, dtype=np.int64)) if lean else {}))
    scores = np.asarray([row['native_return'] for row in rows])
    summary = dict(training_step=step, evaluation_kind=kind, paths=paths,
                   evaluation_environment_steps=paths * horizon, native_mean=float(scores.mean()),
                   native_std=float(scores.std(ddof=1)) if paths > 1 else 0.,
                   min_return=float(scores.min()), max_return=float(scores.max()))
    curve_rows.add(summary)
    for writer in (raw_rows, curve_rows, step_rows):
        writer.flush()
    return rows, summary


def evaluation_writers(stack, output, prefix='development', storage_profile='full'):
    lean = storage_profile == 'lean'
    storage_contract(storage_profile)
    return dict(raw_rows=Rows(stack, output / f'{prefix}_paths.csv.gz',
                    ['training_step', 'evaluation_kind', 'path_index', 'path_seed', 'native_return',
                     'resource_fraction', 'steps', 'replay_error', 'replay_trace_sha256', 'actions_sha256'] +
                    (['remaining_budget_fraction'] if lean else [])),
                curve_rows=Rows(stack, output / f'{prefix}_curve.csv.gz',
                    ['training_step', 'evaluation_kind', 'paths', 'evaluation_environment_steps',
                     'native_mean', 'native_std', 'min_return', 'max_return']),
                step_rows=Rows(stack, output / f'{prefix}_steps.csv.gz',
                    ['training_step', 'evaluation_kind', 'path_index', 'episode_step', 'native_reward',
                     'resource_cost', 'cumulative_cost'] + (['remaining_budget_fraction'] if lean else
                    ['observation_sha256', 'latent_action_sha256',
                     'physical_action_sha256', 'next_observation_sha256'])))


def inspect(args):
    from . import METHODS, profile
    runtime(args)
    from ..chapter3_baselines.environment import PublicBudgetEnv
    cases = []
    for domain in ([args.domain] if args.domain else ('radio', 'v5')):
        env = PublicBudgetEnv(domain)
        try:
            cases.append(dict(domain=domain, environment=env.contract(),
                              profiles={m: profile(m, domain) for m in ([args.method] if args.method else METHODS)}))
        finally:
            env.close()
    return dict(schema=SCHEMA, factory='allocation23.chapter3_candidates.make',
                main_method='A and its ablations use the original run.py chapter3 train entry',
                methods=list(METHODS), cases=cases,
                research_status='Interface inspection only; scientific performance evidence follows registered frozen experiment receipts')


def validate_pfo_training_budget(method, steps, config):
    """Validate the exact real PFO budget before native task construction."""
    if method != 'pfo':
        return
    rollout = int(config['rollout_steps'])
    minibatches = int(config['minibatches'])
    if (steps <= 0 or rollout <= 0 or minibatches <= 0 or rollout % minibatches
            or rollout // minibatches < 2 or steps % (rollout // minibatches)):
        raise ValueError('PFO学习预算必须按原生小批量完整划分；末段只训练真实样本，不补采样或丢弃。')
    if int(config['total_steps']) != steps:
        raise ValueError('PFO --steps必须与config.total_steps完全一致；请显式提供匹配的total_steps配置。未构造物理环境或开始采样。')


def train(args):
    output = new_output(args.output)
    started = time.time()
    progress = dict(status='initializing', pid=os.getpid(), training_steps=0, training_episodes=0,
                    method=args.method, domain=args.domain, seed=args.seed, target_steps=args.steps)

    def status(phase, **extra):
        progress.update(status=phase, elapsed_seconds=time.time() - started, **extra)
        write(output / 'progress.json', progress)

    status('initializing')
    try:
        storage_profile = getattr(args, 'storage_profile', 'full')
        storage = storage_contract(storage_profile)
        runtime(args, output)
        import numpy as np
        import torch
        from . import make, load, profile
        from ..chapter3_baselines.environment import PublicBudgetEnv
        if min(args.steps, args.eval_every, args.development_paths) <= 0 or args.path_master < 0 or args.seed < 0:
            raise ValueError('Positive counts and nonnegative seed/namespaces required')
        supplied = args.config
        config_file = None
        overrides = {}
        if supplied:
            if supplied.lstrip().startswith('{'):
                overrides = json.loads(supplied)
            else:
                config_file = Path(supplied).expanduser().resolve()
                overrides = json.loads(config_file.read_text())
        if not isinstance(overrides, dict):
            raise ValueError('Configuration must be a JSON object')
        effective = dict(profile(args.method, args.domain), **overrides)
        validate_pfo_training_budget(args.method, args.steps, effective)
        write(output / 'config.json', effective)
        config_sha = sha(output / 'config.json')
        supplied_sha = sha(config_file) if config_file else None
        sources = source_identity()
        data, cache = data_identity(args.domain)
        with ExitStack() as stack:
            env = PublicBudgetEnv(args.domain, seed=args.seed, phase='train', path_master=args.path_master,
                                  train_path_base=args.path_master)
            stack.callback(env.close)
            horizon = env.horizon
            if args.steps < horizon or args.steps % horizon:
                raise ValueError('Learning step budget must contain complete native episodes')
            obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
            agent = make(args.method, obs_dim, action_dim, args.seed, args.device, effective)
            protocol = dict(schema=SCHEMA, arguments=vars(args), source_sha256=sources,
                            effective_config=effective, agent_config=getattr(agent, 'config', effective),
                            config_sha256=config_sha,
                            storage_contract=storage,
                            supplied_config_sha256=supplied_sha, environment=env.contract(), data_identity=data,
                            device_identity=dict(requested=args.device, torch_device_name=(
                                torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'),
                                visible_cuda=os.environ.get('CUDA_VISIBLE_DEVICES')),
                            observation='public current local/global features only',
                            decoder='shared native full-spend total-space decoder; proportional spatial weights',
                            online_from_scratch=True, offline_or_pretrained_or_oracle=False,
                            development=dict(phase='dev', paths=args.development_paths, master=args.path_master,
                                path_indices=list(range(args.development_paths)),
                                interval_requested_steps=args.eval_every,
                                schedule='first complete-episode boundary reaching each threshold; record actual true learning steps',
                                initial=True, endpoint=True, holdout_paths=0),
                            evaluation_randomness='preserve global Python/NumPy/Torch/initialized CUDA state; no observe/update; verify network/moments/pending/private-RNG identity',
                            endpoint='one fixed endpoint; no best-checkpoint selection, periodic models, disk replay or Adam',
                            research_status='engineering interface gate' if args.engineering_gate else 'online development run')
            write(output / 'protocol.json', protocol)
            step_rows = Rows(stack, output / 'training_steps.csv.gz',
                             ['training_step', 'episode', 'episode_step', 'path_seed', 'native_reward', 'learning_reward',
                              'resource_cost', 'cumulative_cost', 'physical_amount'])
            episode_rows = Rows(stack, output / 'training_episodes.csv.gz',
                                ['episode', 'training_step', 'path_seed', 'native_return', 'resource_fraction', 'steps'])
            metrics_rows = Rows(stack, output / 'update_metrics.csv.gz', ['training_step', 'metrics_json'])
            writers = evaluation_writers(stack, output, storage_profile=storage_profile)
            eval_steps = 0
            eval_paths = eval_calls = 0

            def dev(policy, step, kind='development', capture=False):
                nonlocal eval_steps, eval_paths, eval_calls
                status(kind, training_steps=step)
                rows, summary = evaluate_paths(policy, domain=args.domain, seed=args.seed,
                    master=args.path_master, phase='dev', paths=args.development_paths,
                    step=step, kind=kind, **writers,
                    storage_profile=storage_profile,
                    capture_path=output / ('development_arrays_step_%d.npz' % step) if capture else None)
                eval_steps += summary['evaluation_environment_steps']
                eval_paths += summary['paths']
                eval_calls += 1
                status('training', training_steps=step, development_environment_steps=eval_steps,
                       latest_development_mean=summary['native_mean'])
                return rows, summary

            _, initial_summary = dev(agent, 0, capture=storage_profile == 'full')
            true_steps = episode = 0
            next_eval = args.eval_every
            while true_steps < args.steps:
                obs, info = env.reset()
                path_seed, total, cost = info['path_seed'], 0., 0.
                for t in range(horizon):
                    action = action_of(agent, obs, False, action_dim)
                    nxt, reward, done, truncated, info = env.step(action)
                    if truncated or bool(done) != (t == horizon - 1):
                        raise RuntimeError('Training episode did not finish at the native horizon')
                    metrics = agent.observe(obs, action, float(reward), nxt, bool(done)) or {}
                    if not isinstance(metrics, dict):
                        raise TypeError('Candidate.observe must return a scalar metrics dict')
                    numbers = {str(k): float(v) for k, v in metrics.items()
                               if isinstance(v, (int, float, np.integer, np.floating))}
                    learning = float(numbers.get('learning_reward', effective.get('reward_scale', 100.) * reward))
                    if not np.isfinite(list(numbers.values())).all() or not np.isfinite(learning):
                        raise FloatingPointError('Nonfinite learning metrics')
                    true_steps += 1
                    total += float(reward)
                    cost += float(info['cost'])
                    step_rows.add(dict(training_step=true_steps, episode=episode, episode_step=t,
                        path_seed=path_seed, native_reward=reward, learning_reward=learning,
                        resource_cost=info['cost'], cumulative_cost=cost, physical_amount=info['physical_amount']))
                    if numbers:
                        metrics_rows.add(dict(training_step=true_steps, metrics_json=json.dumps(numbers, sort_keys=True)))
                    obs = nxt
                if abs(cost - 1.) > 1e-10:
                    raise ArithmeticError('Training episode did not allocate the entire resource budget')
                episode_rows.add(dict(episode=episode, training_step=true_steps, path_seed=path_seed,
                                      native_return=total, resource_fraction=cost, steps=horizon))
                episode += 1
                for writer in (step_rows, episode_rows, metrics_rows):
                    writer.flush()
                if true_steps >= next_eval and true_steps < args.steps:
                    dev(agent, true_steps)
                    while next_eval <= true_steps:
                        next_eval += args.eval_every
                status('training', training_steps=true_steps, training_episodes=episode,
                       accounting=agent.accounting())
            agent.finish()
            if int(agent.accounting()['environment_steps']) != args.steps:
                raise ArithmeticError('Agent accounting differs from actual learning environment steps')
            endpoint, endpoint_summary = dev(agent, true_steps, capture=True)
            model_path = Path(agent.save(output / 'model')).resolve()
            if not model_path.is_relative_to(output / 'model'):
                raise ValueError('Endpoint model escaped the registered model directory')
            model_files = {str(p.relative_to(output / 'model')): dict(sha256=sha(p), bytes=p.stat().st_size)
                           for p in sorted((output / 'model').rglob('*')) if p.is_file()}
            if len([p for p in model_files if Path(p).suffix in ('.pt', '.pth', '.ckpt')]) != 1:
                raise ValueError('Exactly one fixed endpoint evaluation model is required')
            with preserved_rng():
                loaded = load(args.method, output / 'model', device=args.device)
                reloaded, reload_summary = dev(loaded, true_steps, 'endpoint_reload')
            delta = max(abs(a['native_return'] - b['native_return']) for a, b in zip(endpoint, reloaded))
            same_paths = all(a['path_seed'] == b['path_seed'] for a, b in zip(endpoint, reloaded))
            same_actions = all(a['actions_sha256'] == b['actions_sha256'] for a, b in zip(endpoint, reloaded))
            if delta > 1e-6 or not same_paths or not same_actions:
                raise ArithmeticError('Same-device endpoint model reload changed native paths, actions or returns')
            reload_gate = dict(status='PASS', paths=args.development_paths, max_native_return_error=delta,
                               same_path_seeds=same_paths,
                               physical_actions_bitwise_equal=same_actions,
                               all_native_replay_errors_zero=True, all_resource_fraction_errors_below=1e-10)
            write(output / 'endpoint_reload_gate.json', reload_gate)
            if source_identity() != sources or sha(output / 'config.json') != config_sha:
                raise RuntimeError('Project source or effective configuration changed during the run')
            if config_file and sha(config_file) != supplied_sha:
                raise RuntimeError('Supplied configuration file changed during execution')
            completion = dict(schema=SCHEMA, success=True, method=args.method, domain=args.domain, seed=args.seed,
                storage_contract=storage,
                training_steps=true_steps, training_episodes=episode, development_environment_steps=eval_steps,
                evaluation_validation=dict(evaluation_calls=eval_calls, evaluated_paths=eval_paths,
                    native_replayed_paths=eval_paths, full_spend_checked_paths=eval_paths,
                    retained_path_summary_rows=eval_paths, retained_numeric_step_rows=eval_steps,
                    policy_normalization_pending_private_rng_isolation_checks=eval_calls,
                    global_rng_preservation_scopes=eval_calls,
                    representative_path_indices=[0] if storage_profile == 'lean' else 'all'),
                independent_holdout_paths=0, engineering_gate=args.engineering_gate,
                initial_development=initial_summary, endpoint_development=endpoint_summary,
                reloaded_development=reload_summary, endpoint_reload_gate=reload_gate,
                source_unchanged=True, config_sha256=config_sha, protocol_sha256=sha(output / 'protocol.json'),
                accounting=agent.accounting(), model_files=model_files,
                data_final_verification=verify_data(data, cache),
                optimizer_or_replay_saved=False, periodic_models_saved=False,
                elapsed_seconds=time.time() - started)
        completion['artifacts_sha256'] = {str(p.relative_to(output)): sha(p)
            for pattern in ('*.csv.gz', '*.npz') for p in sorted(output.glob(pattern))}
        write(output / 'completed.json', completion)
        status('completed', training_steps=true_steps, training_episodes=episode, success=True)
        return completion
    except BaseException as error:
        failure = dict(progress, success=False, status='failed', error=repr(error), traceback=traceback.format_exc())
        write(output / 'failed.json', failure)
        status('failed', error=repr(error))
        raise


def evaluate(args):
    output = new_output(args.output)
    try:
        storage_profile = getattr(args, 'storage_profile', 'full')
        storage = storage_contract(storage_profile)
        runtime(args, output)
        from . import load
        run = Path(args.run).expanduser().resolve()
        protocol = json.loads((run / 'protocol.json').read_text())
        completion = json.loads((run / 'completed.json').read_text())
        if protocol.get('schema') != SCHEMA or not completion.get('success'):
            raise ValueError('Use a completed run from this integrated collector')
        if sha(run / 'protocol.json') != completion['protocol_sha256']:
            raise RuntimeError('Registered run protocol changed')
        for name, entry in completion['model_files'].items():
            model_file = (run / 'model' / name).resolve()
            if not model_file.is_relative_to(run / 'model') or sha(model_file) != entry['sha256']:
                raise RuntimeError('Fixed endpoint model identity differs from completion record')
        sources = source_identity()
        if protocol['source_sha256'] != sources:
            raise RuntimeError('This checkout differs from the training source; use its registered source snapshot')
        registered = protocol['arguments']
        master = registered['path_master'] if args.path_master is None else args.path_master
        if args.paths < 1 or master < 0:
            raise ValueError('Positive paths and nonnegative namespace required')
        if args.phase == 'final' and (args.path_master is None or master == registered['path_master']):
            raise ValueError('Independent final evaluation requires an explicit separate namespace')
        if not args.data_root and protocol['data_identity'].get('images_opened'):
            os.environ['ALLOCATION23_DATA_ROOT'] = protocol['data_identity']['data_root']
        data, cache = data_identity(registered['domain'])
        if data != protocol['data_identity']:
            raise RuntimeError('Evaluation data identity differs from the registered run')
        condition_file = (Path(args.condition_file).expanduser().resolve(strict=True)
                          if getattr(args, 'condition_file', None) else None)
        condition_sha = sha(condition_file) if condition_file else None
        from ..chapter3_baselines.environment import PublicBudgetEnv
        condition_env = PublicBudgetEnv(registered['domain'], seed=registered['seed'], phase=args.phase,
            path_master=master, holdout=args.phase == 'final', condition_file=condition_file)
        try:
            actual_environment = condition_env.contract()
        finally:
            condition_env.close()
        trained_environment = protocol['environment']
        for field in ('n_objects', 'horizon', 'action_dim'):
            if actual_environment['native_contract'][field] != trained_environment['native_contract'][field]:
                raise ValueError('Fixed-policy condition evaluation cannot change object count, horizon or action dimension')
        if actual_environment['neural_observation_dim'] != trained_environment['neural_observation_dim']:
            raise ValueError('Fixed-policy condition evaluation cannot change the policy observation dimension')
        with preserved_rng():
            agent = load(registered['method'], run / 'model', device=args.device)
        capture = storage_profile == 'full' or args.phase == 'final'
        evaluation_protocol = dict(schema=SCHEMA, arguments=vars(args), storage_contract=storage,
            source_sha256=sources, training_completion_sha256=sha(run / 'completed.json'),
            training_protocol_sha256=sha(run / 'protocol.json'), method=registered['method'],
            domain=registered['domain'], data_identity=data, environment=actual_environment,
            condition_file=str(condition_file) if condition_file else None,
            condition_file_sha256=condition_sha,
            fixed_policy_geometry_matches_training=True,
            path_registration=dict(phase=args.phase, master=master, paths=args.paths,
                path_indices=list(range(args.paths)),
                captured_path_indices=([0] if storage_profile == 'lean' else list(range(args.paths)))
                    if capture else [],
                independent_from_training_namespace=master != registered['path_master']),
            evaluation_gates='native full replay and full spend for every path; unchanged policy, normalization, pending state and RNG')
        write(output / 'protocol.json', evaluation_protocol)
        with ExitStack() as stack:
            writers = evaluation_writers(stack, output, prefix='evaluation', storage_profile=storage_profile)
            rows, summary = evaluate_paths(agent, domain=registered['domain'], seed=registered['seed'],
                master=master, phase=args.phase, paths=args.paths, step=completion['training_steps'],
                kind=args.phase, **writers, storage_profile=storage_profile,
                condition_file=condition_file, expected_environment_contract=actual_environment,
                capture_path=output / 'evaluation_arrays.npz' if capture else None)
        if sources != source_identity():
            raise RuntimeError('Project source changed during evaluation')
        if condition_file and sha(condition_file) != condition_sha:
            raise RuntimeError('Registered physical condition changed during evaluation')
        result = dict(schema=SCHEMA, success=True, run=str(run), method=registered['method'],
                      storage_contract=storage, protocol_sha256=sha(output / 'protocol.json'),
                      environment=actual_environment, condition_file_sha256=condition_sha,
                      fixed_policy_geometry_matches_training=True,
                      domain=registered['domain'], arguments=vars(args), summary=summary,
                      evaluation_validation=dict(evaluation_calls=1, evaluated_paths=len(rows),
                          native_replayed_paths=len(rows), full_spend_checked_paths=len(rows),
                          retained_path_summary_rows=len(rows),
                          retained_numeric_step_rows=summary['evaluation_environment_steps'],
                          policy_normalization_pending_private_rng_isolation_checks=1,
                          global_rng_preservation_scopes=1,
                          captured_path_indices=evaluation_protocol['path_registration']['captured_path_indices']),
                      true_evaluation_environment_steps=summary['evaluation_environment_steps'],
                      learning_environment_steps=0, all_native_replays_exact=True,
                      max_resource_fraction_error=max(abs(row['resource_fraction'] - 1.) for row in rows),
                      source_unchanged=True, source_sha256=sources,
                      training_completion_sha256=sha(run / 'completed.json'),
                      data_final_verification=verify_data(data, cache))
        result['artifacts_sha256'] = {str(p.relative_to(output)): sha(p)
            for pattern in ('*.csv.gz', '*.npz') for p in sorted(output.glob(pattern))}
        write(output / 'completed.json', result)
        return result
    except BaseException as error:
        write(output / 'failed.json', dict(success=False, error=repr(error), traceback=traceback.format_exc()))
        raise
