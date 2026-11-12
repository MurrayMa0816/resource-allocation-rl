"""Evaluate registered C4 endpoints on DEV32 with mean/native random actions.

This script contains no optimizer, training loop, checkpoint saving, or TEST use.
Run one fresh process for each registered (method, training_seed).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
FORMAL = Path('/hy-tmp/resource_allocation_results/chapter04/experiments/double_random_formal_20261107')
MODEL_INDEX = Path('/hy-tmp/resource_allocation_results/chapter04/models/index.json')
DEV_MASTER = 202611043101
CASES = 32
REPEATS = 8
HORIZON = 24
ACTION_SEED_BASE = 91090000
PLAN_SHA256 = '01b549fc951c29f6e9c6d23361af63af176305b979f9f1f0600916b4419a0eee'
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def encoded_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def dump(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Diagnostic output must stay in its independent HY root')
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
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())


def snapshot_sha(state):
    return hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest()


def restore(state, np, torch):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def equal_rng(a, b, np, torch):
    return a[0] == b[0] and a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1]) and a[1][2:] == b[1][2:] and torch.equal(a[2], b[2])


def state_sha(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def imported_sources():
    roots = (Path('/hy-tmp/resource_allocation_results/chapter04'), Path('/root/resource_allocation_run_storage_20261101'))
    paths = set()
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path:
            path = Path(path).resolve()
            if path.suffix == '.py' and path.exists() and any(path.is_relative_to(r) for r in roots) and not path.is_relative_to(ROOT):
                paths.add(path)
    return {str(p): sha(p) for p in sorted(paths)}


def count_physical_steps(guarded_task, ledger):
    """Observe successful calls at the unchanged underlying physics boundary."""
    env = guarded_task.env.native_env
    original = env.step
    def observed(action):
        result = original(action)
        ledger['actual_physical_steps'] += 1
        if result[2]: ledger['actual_complete_episodes'] += 1
        return result
    env.step = observed


def audit_row(audit, method, seed, mode, repeat, case, plan):
    if not audit['terminal_full_spend_assertion'] or not audit['nonterminal_rewards_zero']:
        raise AssertionError('Full-spend/terminal-only physical contract failed')
    if audit['relative_allocation_error'] > 1e-10 or int(audit['steps']) != HORIZON:
        raise AssertionError('Wrong physical horizon/resource account')
    for key in ('maximum_negative_allocation_violation', 'maximum_per_object_cap_violation', 'maximum_completed_object_allocation'):
        if audit[key] != 0:
            raise AssertionError('Physical allocation violation: ' + key)
    return dict(method=method, training_seed=seed, action_mode=mode, replicate=repeat, case_index=case,
                path_seed=plan['task_seed'], plan_sha256=encoded_sha(plan),
                native_return=float(audit['terminal_reward']), score_x100=float(audit['terminal_reward'])*100,
                steps=int(audit['steps']), total_used=float(audit['total_used']), remaining=float(audit['remaining']),
                initial_budget=float(audit['initial_budget']), relative_allocation_error=float(audit['relative_allocation_error']),
                maximum_step_balance_absolute_error=float(audit['maximum_step_balance_absolute_error']),
                nonterminal_rewards_zero=bool(audit['nonterminal_rewards_zero']),
                terminal_full_spend_assertion=bool(audit['terminal_full_spend_assertion']),
                maximum_negative_allocation_violation=float(audit['maximum_negative_allocation_violation']),
                maximum_per_object_cap_violation=float(audit['maximum_per_object_cap_violation']),
                maximum_completed_object_allocation=float(audit['maximum_completed_object_allocation']),
                physical_ledger_sha256=encoded_sha(audit))


def expected_cases(job):
    path = job/'evaluation_paths.csv.gz'
    with gzip.open(path, 'rt', newline='') as f:
        result = {int(r['path_index']): dict(path_seed=int(r['path_seed']), native_return=float(r['native_return']))
                  for r in csv.DictReader(f) if int(r['training_step']) == 100608 and r['phase'] == 'dev'}
    if sorted(result) != list(range(CASES)):
        raise AssertionError('Existing fixed endpoint DEV32 reference is missing')
    return result, dict(path=str(path), sha256=sha(path))


def build_f1(worker, args, torch):
    native, binding = worker.bind(args)
    record = args.record
    saved = torch.load(record['physical_path'], map_location='cpu', weights_only=False)
    if saved['schema'] != worker.SCHEMA or not saved['evaluation_only'] or saved['method'] != 'f1' or saved['domain'] != 'v5' or saved['seed'] != args.seed or saved['train_env_steps'] != 100608:
        raise AssertionError('Registered F1 fixed endpoint identity mismatch')
    pool = [native['Task']('v5', 3, 'A', args.seed, path_master=DEV_MASTER, holdout=False) for _ in range(CASES)]
    for task in pool: count_physical_steps(task, args.ledger)
    spec = pool[0].spec
    if spec.to_dict() != saved['protocol']['spec']:
        raise AssertionError('F1 physical spec changed')
    actor = native['make_actor'](spec, 'A', 3, args.seed).cpu()
    critic = native['make_value'](spec, args.seed).cpu()
    actor.load_state_dict(saved['actor'], strict=True); critic.load_state_dict(saved['critic'], strict=True)
    if native['model_hash'](actor) != saved['actor_sha256'] or native['model_hash'](critic) != saved['critic_sha256']:
        raise AssertionError('F1 loaded tensor identity mismatch')
    actor.eval(); critic.eval()
    pair = native['CPURolloutPair'](actor, critic); pair.refresh(actor, critic)

    def evaluate(mode, repeat, action_seed):
        generator = torch.Generator(device='cpu').manual_seed(action_seed)
        generator_before = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
        results = native['evaluate_rollout'](pool, pair.actor, pair.critic, generator, 'dev', 0,
                                              count=CASES, training=False, deterministic=mode == 'mean')
        pair.assert_rollout_complete()
        generator_after = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
        if (generator_before == generator_after) != (mode == 'mean'):
            raise AssertionError('F1 action generator consumption does not match action mode')
        rows = []
        for i, result in enumerate(results):
            row = audit_row(pool[i].env.last_full_spend_audit, 'f1', args.seed, mode, repeat, i, result['plan'])
            if abs(row['native_return'] - result['native_return']) > 1e-12:
                raise AssertionError('F1 terminal audit return mismatch')
            rows.append(row)
        return rows, dict(independent_generator_before_sha256=generator_before, independent_generator_after_sha256=generator_after)

    return evaluate, [actor, critic, pair.actor, pair.critic], lambda: [t.env.close() for t in pool], binding


def build_props(worker, args, torch):
    _native, binding = worker.bind(args)
    from chapter4_recent_baselines.env_adapter import Env
    props_root = FORMAL/'source/props_pgr'
    sys.path.insert(0, str(props_root))
    runner = load_module(props_root/'native_loop.py', '_action_diagnostic_props_native')
    saved = torch.load(args.record['physical_path'], map_location='cpu', weights_only=False)
    cfg = saved['config']
    if cfg['domain'] != 'v5' or cfg['seed'] != args.seed or saved['train_env_steps'] != 100608 or cfg['learning_rate'] != 5e-5 or cfg['props_learning_rate'] != 1e-4 or cfg['initialization_scheme'] != 'both_random':
        raise AssertionError('Registered PROPS fixed endpoint identity mismatch')
    env = Env('v5', args.seed, phase='dev', action_mode='raw', path_master=DEV_MASTER, holdout=False)
    count_physical_steps(env.task, args.ledger)
    Agent, *_ = runner._load_official()
    factory = runner.configure_agent_factory(Agent, env.native_spec, args.seed, cfg)
    facade = SimpleNamespace(single_observation_space=env.observation_space, single_action_space=env.action_space)
    agent = factory(facade).cpu()
    agent.load_state_dict(saved['target'], strict=True); agent.eval()

    def evaluate(mode, repeat, action_seed):
        rng_before = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        rows = []
        for i in range(CASES):
            observation, _ = env.reset(options={'episode_index': i})
            done = False; value = 0.; steps = 0
            while not done:
                with torch.no_grad():
                    action = agent.get_action(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0), noise=mode == 'stochastic')[0].cpu().numpy()
                observation, reward, done, truncated, _ = env.step(action)
                if truncated or (not done and reward != 0.):
                    raise AssertionError('PROPS terminal-only finite horizon changed')
                value += float(reward); steps += 1
            row = audit_row(env.task.env.last_full_spend_audit, 'props', args.seed, mode, repeat, i, env.task.plan('dev', i))
            if steps != HORIZON or abs(row['native_return'] - value) > 1e-12:
                raise AssertionError('PROPS terminal audit return mismatch')
            rows.append(row)
        rng_after = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        if (rng_before == rng_after) != (mode == 'mean'):
            raise AssertionError('PROPS action RNG consumption does not match action mode')
        return rows, dict(global_torch_before_sha256=rng_before, global_torch_after_sha256=rng_after)

    return evaluate, [agent], env.close, binding


def main(args):
    started = time.monotonic()
    plan_path = ROOT/'protocol/PLAN.json'
    if sha(plan_path) != PLAN_SHA256:
        raise AssertionError('Pre-execution diagnostic PLAN identity mismatch')
    plan = json.loads(plan_path.read_text())
    if plan['development']['master'] != DEV_MASTER or plan['development']['case_indices'] != list(range(CASES)) or plan['replicate_indices'] != list(range(REPEATS)):
        raise AssertionError('Pre-execution diagnostic settings mismatch')
    if args.method != "props" or args.seed != 2:
        raise ValueError("This preserved second attempt is only for the one numerical-gate failure")
    output = ROOT/'jobs'/f'{args.method}__v5__seed{args.seed}__attempt2'
    if output.exists():
        raise FileExistsError('A diagnostic endpoint attempt is immutable: ' + str(output))
    output.mkdir(parents=True)
    for key in ('TMPDIR', 'XDG_CACHE_HOME', 'MPLCONFIGDIR', 'TORCH_HOME', 'CUDA_CACHE_PATH', 'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR'):
        path = output/'runtime'/key.lower(); path.mkdir(parents=True, exist_ok=True); os.environ[key] = str(path)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    endpoint_index = json.loads(MODEL_INDEX.read_text())
    records = [r for r in endpoint_index['models'] if r['method'] == args.method and r['domain'] == 'v5' and int(r['seed']) == args.seed]
    if len(records) != 1:
        raise AssertionError('Model registry is ambiguous')
    args.record = record = records[0]
    planned = [r for r in plan['training_models'] if r['method'] == args.method and r['seed'] == args.seed]
    if len(planned) != 1 or record['physical_path'] != planned[0]['physical_path'] or record['model_sha256'] != planned[0]['model_sha256']:
        raise AssertionError('Current model index differs from frozen diagnostic plan')
    if sha(record['physical_path']) != record['model_sha256']:
        raise AssertionError('Registered endpoint file SHA changed')
    reference, reference_identity = expected_cases(Path(record['physical_path']).parent)
    args.environment_root = FORMAL/'source/environment'
    args.domain = 'v5'
    worker = load_module(FORMAL/'execution/f1_worker.py', '_action_diagnostic_f1_worker')
    close = lambda: None
    rounds = []; all_rows = []; reference_plans = None
    args.ledger = dict(actual_physical_steps=0, actual_complete_episodes=0)
    try:
        evaluate, models, close, binding = (build_f1 if args.method == 'f1' else build_props)(worker, args, torch)
        before_hashes = [state_sha(m) for m in models]
        source_before = imported_sources()
        dump(output/'input_identity.json', dict(model=record, model_index_sha256=sha(MODEL_INDEX), reference=reference_identity,
             source_sha256=source_before, binding=binding, model_tensor_sha256=before_hashes,
             diagnostic_source_sha256=sha(__file__), PLAN_sha256=PLAN_SHA256))
        for mode, repeat in [('mean', -1)] + [('stochastic', i) for i in range(REPEATS)]:
            action_seed = ACTION_SEED_BASE + 1000*args.seed + max(0, repeat)
            saved_rng = snapshot(np, torch)
            random.seed(action_seed); np.random.seed(action_seed); torch.manual_seed(action_seed)
            try:
                rows, rng_evidence = evaluate(mode, repeat, action_seed)
            finally:
                restore(saved_rng, np, torch)
            if not equal_rng(saved_rng, snapshot(np, torch), np, torch):
                raise AssertionError('Evaluation leaked the global RNG state')
            if before_hashes != [state_sha(m) for m in models]:
                raise AssertionError('Evaluation mutated model tensors')
            plans = [(r['path_seed'], r['plan_sha256']) for r in rows]
            if reference_plans is None: reference_plans = plans
            if plans != reference_plans:
                raise AssertionError('Action sampling changed the exogenous DEV32 panel')
            if mode == 'mean':
                differences = [abs(r['native_return'] - reference[i]['native_return']) for i, r in enumerate(rows)]
                dump(output/'mean_reference_comparison.json', dict(
                    planned_native_return_tolerance=1e-6,
                    maximum_absolute_native_return_difference=max(differences),
                    original_evaluation_device='cuda', diagnostic_device='cpu',
                    cases=[dict(case_index=i, path_seed=r['path_seed'], original_path_seed=reference[i]['path_seed'],
                                new_native_return=r['native_return'], original_native_return=reference[i]['native_return'],
                                absolute_native_return_difference=differences[i]) for i, r in enumerate(rows)]))
                with gzip.open(output/'mean_case_results_before_gate.csv.gz', 'wt', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
                if any(r['path_seed'] != reference[i]['path_seed'] for i, r in enumerate(rows)) or max(differences) > 1e-6:
                    raise AssertionError('Mean execution did not reproduce original DEV32 cases')
                deterministic_max_abs_difference = max(differences)
            values = [r['score_x100'] for r in rows]
            summary = dict(method=args.method, training_seed=args.seed, action_mode=mode, replicate=repeat,
                 action_seed=action_seed, cases=CASES, evaluation_steps=CASES*HORIZON,
                 mean_score_x100=float(np.mean(values)), case_sd_x100=float(np.std(values, ddof=1)),
                 minimum_score_x100=float(min(values)), maximum_score_x100=float(max(values)),
                 global_rng_restored=True, global_rng_snapshot_sha256=snapshot_sha(saved_rng),
                 model_tensors_unchanged=True, RNG_evidence=rng_evidence)
            rounds.append(summary); all_rows.extend(rows)
            with gzip.open(output/'case_results.csv.gz', 'wt', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
            dump(output/'progress.json', dict(status='evaluating', rounds_completed=len(rounds), rounds_expected=1+REPEATS,
                 evaluation_steps=len(all_rows)*HORIZON, training_steps=0, last_round=summary))
            print(json.dumps({k: summary[k] for k in ('method','training_seed','action_mode','replicate','mean_score_x100')}, allow_nan=False), flush=True)
        close(); close = lambda: None
        source_after = imported_sources()
        if any(source_after.get(k) != v for k, v in source_before.items()):
            raise AssertionError('Frozen imported scientific source changed during evaluation')
        if sha(record['physical_path']) != record['model_sha256']:
            raise AssertionError('Registered fixed model file changed')
        scores = [r['mean_score_x100'] for r in rounds if r['action_mode'] == 'stochastic']
        if args.ledger != dict(actual_physical_steps=(1+REPEATS)*CASES*HORIZON, actual_complete_episodes=(1+REPEATS)*CASES):
            raise AssertionError('Actual physical call ledger differs from registered evaluation counts')
        completion = dict(status='complete', method=args.method, training_seed=args.seed, domain='v5',
             evaluation_only=True, training_steps=0, independent_TEST_steps=0, phase='dev', DEV_master=DEV_MASTER,
             DEV_cases=CASES, stochastic_replicates=REPEATS, temperature=1., native_sigma_unchanged=True,
             trajectories=len(all_rows), evaluation_steps=len(all_rows)*HORIZON,
             deterministic_score_x100=rounds[0]['mean_score_x100'], stochastic_score_x100=float(np.mean(scores)),
             stochastic_repeat_sd_x100=float(np.std(scores, ddof=1)),
             stochastic_minus_mean_score_points=float(np.mean(scores)-rounds[0]['mean_score_x100']),
             deterministic_reference_max_abs_return_difference=deterministic_max_abs_difference,
             model_SHA_unchanged=True, model_tensors_unchanged=True, sources_unchanged=True, global_rng_restored=True,
             all_full_spend=True, all_terminal_only=True, all_horizons_24=True,
             max_relative_allocation_error=max(r['relative_allocation_error'] for r in all_rows),
             model_path=record['physical_path'], model_sha256=record['model_sha256'],
             case_results_sha256=sha(output/'case_results.csv.gz'), all_rounds=rounds,
             physical_call_ledger=args.ledger, source_sha256_final=source_after, elapsed_seconds=time.monotonic()-started)
        dump(output/'completed.json', completion)
        dump(output/'progress.json', dict(status='complete', evaluation_steps=len(all_rows)*HORIZON, training_steps=0))
    except BaseException:
        dump(output/'failed.json', dict(status='failed', training_steps=0, completed_rounds=rounds,
             stored_completed_trajectories=len(all_rows), evaluation_steps=args.ledger['actual_physical_steps'],
             physical_call_ledger=args.ledger,
             traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('f1', 'props'), required=True)
    parser.add_argument('--seed', type=int, choices=range(5), required=True)
    main(parser.parse_args())
