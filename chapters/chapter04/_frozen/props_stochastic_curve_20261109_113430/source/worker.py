"""Fixed PROPS retraining with isolated native-Gaussian DEV curves.

The historical native_loop and author update functions are imported read-only.
Only output bookkeeping and evaluation are adapted. No TEST is constructed.
"""
from __future__ import annotations
import argparse
import copy
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
FORMAL = Path('/hy-tmp/resource_allocation_results/chapter04/experiments/double_random_formal_20261107')
PROPS = FORMAL / 'source/props_pgr'
ENVIRONMENT = FORMAL / 'source/environment'
MASTER = 202611043101
ACTION_SEED_BASE = 91090000
HORIZON = 24
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def encoded_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def dump(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Output must remain in this independent HY scope')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(np, torch):
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None)


def restore(state, np, torch):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def rng_equal(a, b, np, torch):
    return (a[0] == b[0] and a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1])
            and a[1][2:] == b[1][2:] and torch.equal(a[2], b[2])
            and ((a[3] is None and b[3] is None) or
                 (a[3] is not None and b[3] is not None and len(a[3]) == len(b[3])
                  and all(torch.equal(x, y) for x, y in zip(a[3], b[3])))))


def rng_sha(state):
    h = hashlib.sha256()
    h.update(repr(state[0]).encode())
    h.update(state[1][0].encode())
    h.update(state[1][1].tobytes())
    h.update(repr(state[1][2:]).encode())
    h.update(state[2].numpy().tobytes())
    if state[3] is not None:
        for value in state[3]:
            h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def state_sha(agent):
    h = hashlib.sha256()
    for name, tensor in sorted(agent.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(array.dtype).encode()); h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def set_seeds(seed, np, torch, *, cpu_only=False):
    random.seed(seed); np.random.seed(seed)
    if cpu_only:
        torch.default_generator.manual_seed(seed)
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed_all(seed)


def sources():
    roots = (FORMAL.resolve(), Path('/root/resource_allocation_run_storage_20261101').resolve())
    paths = set(PROPS.glob('*.py'))
    paths.add(Path(__file__).resolve())
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.suffix == '.py' and any(path.is_relative_to(root) for root in roots):
                paths.add(path)
    return {str(path): sha(path) for path in sorted(paths)}


class Context:
    def __init__(self, args, np, torch):
        self.args, self.np, self.torch = args, np, torch
        self.out = Path(args.output).resolve()
        if not self.out.is_relative_to(ROOT) or self.out.exists():
            raise ValueError('Use a fresh nonexistent output directory under the new experiment root')
        self.out.mkdir(parents=True)
        self.start = time.perf_counter()
        self.phase = 'initializing'; self.node = None; self.repeat = None; self.case = None
        self.counts = {phase: dict(attempted_physical_steps=0, successful_physical_steps=0,
                                  completed_physical_episodes=0, guarded_complete_episodes=0)
                       for phase in ('train', 'mean', 'stochastic')}
        self.training_steps = 0
        self.max_resource_error = 0.
        self.closed = False
        self.environments = []
        self.mirror = None
        self.node_rows = []
        self.reference = self.load_reference()
        self.files = {}
        for name in ('evaluation_paths.csv.gz', 'resource_audit.csv.gz', 'evaluation_panels.csv.gz'):
            self.files[name] = gzip.open(self.out / name, 'wt', newline='', compresslevel=3)
        self.files['physical_episode_ledgers.jsonl.gz'] = gzip.open(self.out / 'physical_episode_ledgers.jsonl.gz', 'wt', compresslevel=3)
        self.path_writer = csv.DictWriter(self.files['evaluation_paths.csv.gz'], fieldnames=[
            'training_step', 'action_mode', 'replicate', 'case_index', 'path_seed', 'plan_sha256',
            'action_seed', 'native_return', 'score_x100', 'steps', 'resource_fraction', 'reference_native_return', 'absolute_reference_error'])
        self.resource_writer = csv.DictWriter(self.files['resource_audit.csv.gz'], fieldnames=[
            'execution_phase', 'training_step', 'replicate', 'case_index', 'phase', 'episode_index', 'steps',
            'initial_budget', 'total_used', 'remaining', 'relative_allocation_error',
            'maximum_step_balance_absolute_error', 'maximum_negative_allocation_violation',
            'maximum_per_object_cap_violation', 'maximum_completed_object_allocation',
            'terminal_reward', 'nonterminal_rewards_zero', 'terminal_full_spend_assertion', 'full_ledger_sha256'])
        self.panel_writer = csv.DictWriter(self.files['evaluation_panels.csv.gz'], fieldnames=[
            'training_step', 'action_mode', 'replicate', 'cases', 'action_seed', 'native_mean', 'native_sample_sd',
            'successful_physical_steps', 'action_rng_before_sha256', 'action_rng_after_sha256'])
        for writer in (self.path_writer, self.resource_writer, self.panel_writer):
            writer.writeheader()
        for key in ('TMPDIR', 'XDG_CACHE_HOME', 'MPLCONFIGDIR', 'TORCH_HOME', 'CUDA_CACHE_PATH',
                    'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR'):
            path = self.out / 'runtime' / key.lower(); path.mkdir(parents=True, exist_ok=True); os.environ[key] = str(path)
        os.environ['WANDB_MODE'] = 'disabled'
        self.progress()

    def load_reference(self):
        path = FORMAL / 'jobs' / f'props__v5__seed{self.args.seed}' / 'evaluation_paths.csv.gz'
        result = {}
        with gzip.open(path, 'rt', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['phase'] == 'dev':
                    result[(int(row['training_step']), int(row['path_index']))] = (
                        int(row['path_seed']), float(row['native_return']))
        if len(result) != 132 * 32:
            raise AssertionError('Original full DEV reference is incomplete')
        self.reference_identity = dict(path=str(path), sha256=sha(path), cases=len(result))
        return result

    def progress(self, **extra):
        value = dict(schema='props-stochastic-curve-worker-progress-v1', pid=os.getpid(), seed=self.args.seed,
                     engineering_only=self.args.engineering_gate, phase=self.phase, training_step=self.node,
                     training_steps=self.counts['train']['successful_physical_steps'], target_steps=self.args.steps,
                     repeat=self.repeat, case=self.case, physical_counts=self.counts,
                     elapsed_seconds=time.perf_counter() - self.start, training_device=self.args.device,
                     stochastic_device='cpu', completed_nodes=len(self.node_rows))
        value.update(extra)
        dump(self.out / 'progress.json', value)

    def install(self):
        sys.path.insert(0, str(ENVIRONMENT))
        from chapter4_full_spend_env.bootstrap import install
        self.binding = install(audit_dir=None, context=dict(scope=str(ROOT), method='props', seed=self.args.seed,
                                                           engineering=self.args.engineering_gate))
        dump(self.out / 'bridge_identity.json', self.binding)
        from chapter4_recent_baselines import env_adapter
        Base = env_adapter.Env
        ctx = self
        class RegisteredEnv(Base):
            def __init__(self, domain, seed=0, phase='train', action_mode='raw', **kwargs):
                if domain != 'v5' or phase not in ('train', 'dev'):
                    raise ValueError('Only registered v5 TRAIN/DEV are permitted, no TEST')
                kwargs['path_master'] = MASTER
                kwargs['holdout'] = False
                super().__init__(domain, seed, phase, action_mode, **kwargs)
                ctx.environments.append(self)
                ctx.observe_physics(self)
            def step(self, action):
                result = super().step(action)
                if result[2]:
                    ctx.audit(self)
                return result
            def close(self):
                super().close()
                if self in ctx.environments:
                    ctx.environments.remove(self)
        env_adapter.Env = RegisteredEnv
        env_adapter.make_env = lambda domain, seed=0, **kwargs: RegisteredEnv(domain, seed, **kwargs)
        self.Env = RegisteredEnv
        return RegisteredEnv

    def observe_physics(self, env):
        native = env.task.env.native_env
        original = native.step
        def observed(action):
            category = 'train' if env.phase == 'train' else self.phase
            if category not in self.counts:
                raise AssertionError('Unregistered physical phase')
            counter = self.counts[category]
            counter['attempted_physical_steps'] += 1
            result = original(action)
            counter['successful_physical_steps'] += 1
            if result[2]:
                counter['completed_physical_episodes'] += 1
            return result
        native.step = observed

    def audit(self, env):
        audit = env.task.env.last_full_spend_audit
        if audit is None or not audit['terminal_full_spend_assertion'] or not audit['nonterminal_rewards_zero']:
            raise AssertionError('Full-spend / terminal-only contract missing')
        if audit['relative_allocation_error'] > 1e-10 or int(audit['steps']) != HORIZON:
            raise AssertionError('Physical resource/horizon contract failed')
        for key in ('maximum_negative_allocation_violation', 'maximum_per_object_cap_violation',
                    'maximum_completed_object_allocation'):
            if audit[key] != 0:
                raise AssertionError('Physical allocation violation: ' + key)
        category = 'train' if env.phase == 'train' else self.phase
        self.counts[category]['guarded_complete_episodes'] += 1
        self.max_resource_error = max(self.max_resource_error, float(audit['relative_allocation_error']))
        tags = dict(execution_phase=category,
                    training_step=self.counts['train']['successful_physical_steps'] if category == 'train' else self.node,
                    replicate=self.repeat, case_index=self.case)
        row = {key: audit[key] for key in self.resource_writer.fieldnames if key in audit}
        row.update(tags, full_ledger_sha256=encoded_sha(audit))
        self.resource_writer.writerow(row)
        self.files['physical_episode_ledgers.jsonl.gz'].write(json.dumps(dict(**tags, ledger=audit), separators=(',', ':'), allow_nan=False) + '\n')

    def panel(self, agent, mode, repeat, seed, step, device):
        np, torch = self.np, self.torch
        self.phase, self.node, self.repeat, self.case = mode, step, repeat, None
        set_seeds(seed, np, torch, cpu_only=mode == 'stochastic')
        env = self.Env('v5', self.args.seed, phase='dev', action_mode='raw')
        values = []
        start_steps = self.counts[mode]['successful_physical_steps']
        # Mean panel preserves original ordering: seed -> Env construction -> action calls.
        # Stochastic panels are seeded AGAIN after Env construction to match the
        # registered endpoint diagnostic, where the Env existed before panel seeding.
        if mode == 'stochastic':
            set_seeds(seed, np, torch, cpu_only=True)
        action_rng_before = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        try:
            for case in range(self.args.dev_cases):
                self.case = case
                obs, _ = env.reset(options={'episode_index': case})
                value, steps, done = 0., 0, False
                while not done:
                    with torch.no_grad():
                        action = agent.get_action(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
                                                  noise=mode == 'stochastic')[0].cpu().numpy()
                    obs, reward, done, truncated, _ = env.step(action)
                    if truncated or (not done and reward != 0.):
                        raise AssertionError('Terminal-only / true-horizon reward contract changed')
                    value += float(reward); steps += 1
                audit = env.task.env.last_full_spend_audit
                plan = env.task.plan('dev', case)
                if steps != HORIZON or abs(value - float(audit['terminal_reward'])) > 1e-12:
                    raise AssertionError('Episode scalar and physical audit mismatch')
                reference_allowed = mode == 'mean' and (not self.args.engineering_gate or step == 0)
                reference = self.reference.get((step, case)) if reference_allowed else None
                if reference is not None and int(plan['task_seed']) != reference[0]:
                    raise AssertionError('Original DEV path identity mismatch')
                row = dict(training_step=step, action_mode=mode, replicate=repeat, case_index=case,
                           path_seed=int(plan['task_seed']), plan_sha256=encoded_sha(plan), action_seed=seed,
                           native_return=value, score_x100=value * 100., steps=steps,
                           resource_fraction=float(audit['total_used']) / env.native_spec.initial_budget,
                           reference_native_return=reference[1] if reference else '',
                           absolute_reference_error=abs(value - reference[1]) if reference else '')
                self.path_writer.writerow(row)
                values.append(value)
            action_rng_after = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
            if (action_rng_before == action_rng_after) != (mode == 'mean'):
                raise AssertionError('Action sampling RNG consumption mismatch')
            result = dict(training_step=step, action_mode=mode, replicate=repeat, cases=len(values), action_seed=seed,
                          native_mean=float(np.mean(values)), native_sample_sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
                          successful_physical_steps=self.counts[mode]['successful_physical_steps'] - start_steps,
                          action_rng_before_sha256=action_rng_before, action_rng_after_sha256=action_rng_after)
            self.panel_writer.writerow(result)
            return values, result
        finally:
            env.close()

    def evaluate(self, _Env, domain, seed, agent, device, episodes, train_steps):
        np, torch = self.np, self.torch
        if domain != 'v5' or seed != self.args.seed or episodes != self.args.dev_cases:
            raise AssertionError('Native evaluator identity mismatch')
        saved = snapshot(np, torch)
        original_mode = agent.training
        original_sha = state_sha(agent)
        panel_start = {k: v['successful_physical_steps'] for k, v in self.counts.items()}
        panels = []
        try:
            agent.eval()
            mean_values, mean_panel = self.panel(agent, 'mean', -1, 4000000, train_steps, device)
            panels.append(mean_panel)
            # Mirrors are made only from the current newly trained target in memory.
            # No model constructor or old checkpoint is used to produce this policy.
            mirror_sha = None
            if self.args.repeats:
                self.mirror = copy.deepcopy(agent).to('cpu') if self.mirror is None else self.mirror
                self.mirror.load_state_dict(agent.state_dict(), strict=True)
                self.mirror.requires_grad_(False); self.mirror.eval()
                for name, tensor in agent.state_dict().items():
                    clone = self.mirror.state_dict()[name]
                    if not torch.equal(tensor.detach().cpu(), clone):
                        raise AssertionError('CPU target mirror is not tensor-exact')
                    if tensor.device.type == 'cpu' and tensor.untyped_storage().data_ptr() == clone.untyped_storage().data_ptr():
                        raise AssertionError('CPU mirror shares target tensor storage')
                mirror_sha = state_sha(self.mirror)
            random_values = []
            for repeat in range(self.args.repeats):
                values, panel = self.panel(self.mirror, 'stochastic', repeat,
                                          ACTION_SEED_BASE + 1000 * self.args.seed + repeat,
                                          train_steps, 'cpu')
                random_values.append(values); panels.append(panel)
                self.progress()
            if self.args.repeats and state_sha(self.mirror) != mirror_sha:
                raise AssertionError('Evaluation altered the CPU target mirror')
            row = dict(train_env_steps=train_steps, mean_native_mean=float(np.mean(mean_values)),
                       stochastic_native_mean=float(np.mean(random_values)) if random_values else None,
                       stochastic_replicate_native_means=[float(np.mean(v)) for v in random_values],
                       stochastic_replicate_sample_sd=float(np.std([np.mean(v) for v in random_values], ddof=1)) if self.args.repeats > 1 else 0.,
                       panels=panels, source_target_state_sha256=original_sha, cpu_mirror_state_sha256=mirror_sha,
                       mean_device=str(device), stochastic_device='cpu', temperature=1.,
                       original_mean_reference_max_absolute_error=max(abs(value - self.reference[(train_steps, case)][1])
                            for case, value in enumerate(mean_values) if (train_steps, case) in self.reference)
                            if (train_steps, 0) in self.reference and (not self.args.engineering_gate or train_steps == 0) else None)
        finally:
            agent.train(original_mode)
            restore(saved, np, torch)
            if not rng_equal(saved, snapshot(np, torch), np, torch):
                raise AssertionError('Full Python / NumPy / CPU / CUDA training RNG was not restored')
            if state_sha(agent) != original_sha:
                raise AssertionError('Evaluation altered the training target')
            self.phase, self.repeat, self.case = 'train', None, None
            for stream in self.files.values():
                stream.flush()
        row.update(training_rng_restored=True, training_rng_sha256=rng_sha(saved),
                   training_target_unmodified=True, cpu_mirror_no_shared_storage=bool(self.args.repeats))
        self.node_rows.append(row)
        with (self.out / 'stochastic_nodes.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.progress()
        # Historical loop receives the mean panel exactly as before. Added
        # stochastic costs remain explicit and are reconciled at completion.
        return dict(train_env_steps=train_steps,
                    evaluation_env_steps=self.counts['mean']['successful_physical_steps'] - panel_start['mean'],
                    episodes=len(mean_values), mean_return=float(np.mean(mean_values)), mean_score=float(np.mean(mean_values)) * 100.,
                    std_return=float(np.std(mean_values, ddof=1)) if len(mean_values) > 1 else 0., returns=mean_values,
                    phase='dev', deterministic_eval=True, training_rng_restored=True, evaluation_master=MASTER)

    def close(self):
        if self.closed:
            return
        for env in list(self.environments):
            env.close()
        for stream in self.files.values():
            stream.close()
        self.closed = True


def verify_args(args):
    if args.seed not in range(5):
        raise ValueError('Exactly the five registered training seeds are allowed')
    if args.steps < 192 or args.steps % 96 or args.eval_every % 96 or args.eval_every < 96:
        raise ValueError('Fixed complete target batches and at least eight episodes are required')
    if args.dev_cases not in range(1, 33) or args.repeats not in (0, 8):
        raise ValueError('Eight fixed stochastic replicates (or engineering mean-only) and at most DEV32 are required')
    if args.repeats == 0 and not args.engineering_gate:
        raise ValueError('Mean-only is exclusively an engineering control')
    if not args.engineering_gate:
        if (args.steps, args.eval_every, args.dev_cases, args.repeats) != (100608, 768, 32, 8):
            raise ValueError('Formal science parameters are fixed before execution')
        expected = 'cuda:0' if args.seed == 2 else 'cpu'
        if args.device != expected:
            raise ValueError('Formal retraining must preserve original seed-specific device')
    plan = ROOT / 'protocol/PLAN.json'
    if not plan.is_file() or sha(plan) != args.plan_sha256:
        raise AssertionError('Frozen plan checksum mismatch or absent plan')
    return dict(path=str(plan), sha256=args.plan_sha256)


def main(args):
    plan_identity = verify_args(args)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Registered CUDA training device is unavailable')
    ctx = Context(args, np, torch)
    before = None
    try:
        ctx.install()
        sys.path.insert(0, str(PROPS))
        runner = load_module(PROPS / 'native_loop.py', '_stochastic_curve_readonly_native')
        before = sources()
        dump(ctx.out / 'source_binding_before.json', before)
        config = dict(method='PROPS', domain='v5', seed=args.seed, output_dir=str(ctx.out), device=args.device,
            torch_threads=1, total_steps=args.steps, episodes=args.steps // HORIZON,
            representation_variant='matched_sf', initialization_scheme='both_random', smoke=False,
            gamma=1., gae_lambda=.95, learning_rate=5e-5, props_learning_rate=1e-4,
            target_episodes=4, behavior_episodes=1, buffer_episodes=8, update_epochs=10,
            props_update_epochs=16, num_minibatches=32, props_num_minibatches=16, clip_coef=.2,
            props_clip_coef=.3, ent_coef=.01, props_lambda=.3, target_kl=.03, anneal_lr=True,
            eval_every_episodes=args.eval_every // HORIZON, eval_episodes=args.dev_cases,
            path_master=MASTER, development_master=MASTER, deterministic_eval=True,
            holdout_paths=0, no_holdout=True, engineering_gate=args.engineering_gate,
            parameter_selection='fixed current half-target-lr configuration; no outcome-based selection')
        dump(ctx.out / 'protocol.json', dict(schema='props-stochastic-curve-worker-v1', arguments=vars(args),
            config=config, plan=plan_identity, reference=ctx.reference_identity,
            learning_implementation='read-only original native_loop.run and author PPO/PROPS updates',
            training_from_scratch=True, old_checkpoint_read_after_new_fixed_endpoint_only=True,
            stochastic_policy='CPU exact mirror of current target, native Normal.sample, temperature1',
            stochastic_action_seeds=[ACTION_SEED_BASE + 1000 * args.seed + r for r in range(args.repeats)],
            RNG_isolation='full Python, complete NumPy, torch CPU and all initialized CUDA states restored in finally',
            physical_audit='successful underlying physics and attempted calls; full compressed per-episode 24-round ledgers',
            independent_TEST=False, periodic_models=False, optimizer_or_replay_saved=False))
        runner._evaluate = ctx.evaluate
        native_append = runner._append_jsonl
        def append(path, rows):
            native_append(path, rows)
            if Path(path).name == 'training.jsonl' and rows:
                ctx.training_steps = int(rows[-1]['train_env_steps'])
                ctx.progress(target_updates=rows[-1].get('target_update_count'),
                             behavior_updates=rows[-1].get('behavior_update_count'))
        runner._append_jsonl = append
        ctx.phase = 'train'
        result = runner.run(config)
        expected_grid = list(range(0, args.steps + 1, args.eval_every))
        if expected_grid[-1] != args.steps:
            expected_grid.append(args.steps)
        if [row['train_env_steps'] for row in ctx.node_rows] != expected_grid:
            raise AssertionError('Full fixed evaluation grid not recorded')
        expected_cost = dict(train=args.steps,
            mean=len(expected_grid) * args.dev_cases * HORIZON,
            stochastic=len(expected_grid) * args.dev_cases * HORIZON * args.repeats)
        for phase, count in ctx.counts.items():
            if count['successful_physical_steps'] != expected_cost[phase] or count['attempted_physical_steps'] != expected_cost[phase]:
                raise AssertionError('Actual physical steps mismatch for ' + phase)
            if count['guarded_complete_episodes'] * HORIZON != expected_cost[phase] or count['completed_physical_episodes'] != count['guarded_complete_episodes']:
                raise AssertionError('Actual complete physical episodes mismatch for ' + phase)
        reference_errors = [row['original_mean_reference_max_absolute_error'] for row in ctx.node_rows
                            if row['original_mean_reference_max_absolute_error'] is not None]
        new_loaded = torch.load(ctx.out / 'checkpoint.pt', map_location='cpu', weights_only=False)
        loads = dict(native_checkpoint_roundtrip=1, new_fixed_endpoint_inspection=1, old_fixed_endpoint_comparison=0)
        comparison = dict(original_mean_compared_nodes=len(reference_errors), original_mean_max_absolute_error=max(reference_errors),
                          complete_grid_and_cases=not args.engineering_gate)
        if not args.engineering_gate:
            old_path = FORMAL / 'jobs' / f'props__v5__seed{args.seed}' / 'checkpoint.pt'
            old_loaded = torch.load(old_path, map_location='cpu', weights_only=False)
            loads['old_fixed_endpoint_comparison'] = 1
            if new_loaded['train_env_steps'] != 100608 or old_loaded['train_env_steps'] != 100608:
                raise AssertionError('Fixed endpoint identity mismatch')
            states_equal = (new_loaded['target'].keys() == old_loaded['target'].keys()
                            and all(torch.equal(value, old_loaded['target'][key]) for key, value in new_loaded['target'].items()))
            comparison.update(old_model_path=str(old_path), old_model_sha256=sha(old_path),
                target_state_all_tensors_exact=states_equal,
                target_state_max_absolute_difference=max(float((value - old_loaded['target'][key]).abs().max())
                    for key, value in new_loaded['target'].items()))
            # A mismatch is reported as a reproducibility failure, never silently
            # repaired with old weights or favorable repeat/seed selection.
            comparison['reproduction_pass'] = states_equal and max(reference_errors) == 0.
        else:
            comparison['reproduction_pass'] = max(reference_errors) == 0.
        dump(ctx.out / 'reproduction_comparison.json', comparison)
        if not comparison['reproduction_pass']:
            raise AssertionError('Original mean/target reproduction failed; complete comparison is preserved')
        after = sources()
        if any(after.get(path) != digest for path, digest in before.items()):
            raise AssertionError('Read-only imported source changed during execution')
        dump(ctx.out / 'source_binding_after.json', after)
        ctx.close()
        for path in ctx.out.glob('*.gz'):
            with gzip.open(path, 'rb') as stream:
                for _ in iter(lambda: stream.read(2**20), b''):
                    pass
        result.update(evaluation_policy='original-device mean plus isolated CPU native-random target mirror',
                      stochastic_evaluation_env_steps=expected_cost['stochastic'],
                      total_evaluation_env_steps=expected_cost['mean'] + expected_cost['stochastic'],
                      actual_physical_counts=ctx.counts, reproduction_comparison=comparison)
        dump(ctx.out / 'result.json', result)
        completion = dict(schema='props-stochastic-curve-completion-v1', status='completed', success=True,
                          engineering_only=args.engineering_gate, seed=args.seed, domain='v5',
                          training_from_scratch=True, training_device=args.device, stochastic_device='cpu',
                          fixed_endpoint_steps=args.steps, evaluation_grid=expected_grid, dev_cases=args.dev_cases,
                          stochastic_replicates=args.repeats, temperature=1., actual_physical_counts=ctx.counts,
                          actual_total_physical_steps=sum(row['successful_physical_steps'] for row in ctx.counts.values()),
                          independent_TEST_steps=0, max_resource_error=ctx.max_resource_error,
                          model=str(ctx.out / 'checkpoint.pt'), model_sha256=sha(ctx.out / 'checkpoint.pt'),
                          model_deserializations=loads, reproduction_comparison=comparison,
                          one_fixed_endpoint=True, optimizer_or_replay_saved=False, periodic_models=False,
                          automatic_retry=False, exact_resume_available=False, plan=plan_identity,
                          wall_seconds=time.perf_counter() - ctx.start)
        dump(ctx.out / 'completion.json', completion)
        dump(ctx.out / 'completed.json', completion)
        ctx.phase = 'complete'; ctx.progress(success=True)
        return completion
    except BaseException:
        dump(ctx.out / 'failed.json', dict(status='failed', traceback=traceback.format_exc(),
             physical_counts=ctx.counts, actual_total_successful_physical_steps=sum(x['successful_physical_steps'] for x in ctx.counts.values()),
             elapsed_seconds=time.perf_counter() - ctx.start, automatic_retry=False, exact_resume_available=False,
             source_binding_before=before))
        ctx.phase = 'failed'; ctx.progress(success=False)
        raise
    finally:
        ctx.close()


def parser():
    result = argparse.ArgumentParser()
    result.add_argument('--seed', type=int, required=True)
    result.add_argument('--device', choices=('cpu', 'cuda:0'), required=True)
    result.add_argument('--output', required=True)
    result.add_argument('--plan-sha256', required=True)
    result.add_argument('--steps', type=int, default=100608)
    result.add_argument('--eval-every', type=int, default=768)
    result.add_argument('--dev-cases', type=int, default=32)
    result.add_argument('--repeats', type=int, default=8)
    result.add_argument('--engineering-gate', action='store_true')
    return result


if __name__ == '__main__':
    main(parser().parse_args())
