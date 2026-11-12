"""Formal five-seed facade for the native dual-random Chapter 5 CUDA learner.

Only registration, source closure, output storage and result schemas are adapted.
The native collector, F1 reuse, Actor/Critic/Adam, objectives and random streams
are called directly. This is a new contract, not the historical 61,440-step run.
"""
from __future__ import annotations

import argparse
import ast
import copy
import inspect
from contextlib import contextmanager
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

CHAPTER = None
SCHEMA = 'chapter5-formal-random-five-seed-own-facade-v1'
VARIANTS = ('full', 'only_j', 'only_c', 'scalar_value')
INITIALIZATION_PROFILE = 'chapter5-both-heads-default-random-v1'
CONTRACT_SCHEMA = 'chapter5-formal-random-5seed-contract-v1'


def formal_evaluation_grid(budget, interval):
    """Keep every exact DEV interval and include the fixed scientific endpoint."""
    require(type(budget) is int and type(interval) is int and budget > 0 and interval > 0,
            'Positive integer evaluation budget and interval required')
    grid = list(range(0, budget + 1, interval))
    if grid[-1] != budget:
        grid.append(budget)
    return grid


def endpoint_clock_type(native_clock):
    """Retain the native exact-tick scheduler, adding one off-grid final tick."""
    class EndpointEvaluationClock(native_clock):
        def __init__(self, interval, budget):
            require(type(interval) is int and type(budget) is int and interval > 0 and budget > 0,
                    'Positive integer clock geometry required')
            self.interval, self.budget = interval, budget
            self.last_step = 0
            self.pending = None
            self.evaluated = [0]

        def observe(self, step, *, complete, assess, collector_fingerprint):
            terminal = step == self.budget and bool(step % self.interval)
            if terminal:
                require(complete, 'Off-grid endpoint must complete its original native collection')
                require(self.pending is None, 'Previous native DEV tick has not been flushed')
                require(step // self.interval == self.last_step // self.interval,
                        'Endpoint crossed an unevaluated regular DEV boundary')
            super().observe(step, complete=complete, assess=assess,
                            collector_fingerprint=collector_fingerprint)
            if terminal:
                self.pending = step

    return EndpointEvaluationClock


def adapt_train_grid(native_train):
    """Change only two DEV-grid expressions in an ephemeral native train function.

    Native files are never modified. AST reversal proves that all other syntax,
    including collection, the native update call and endpoint persistence, is
    identical to the frozen function. New code executes with the native module's
    original globals and the exact same function name.
    """
    original = ast.parse(inspect.getsource(native_train))
    expected = ast.parse('list(range(0, args.learning_steps+1, args.eval_every))', mode='eval').body
    replacement = ast.parse('formal_evaluation_grid(args.learning_steps, args.eval_every)', mode='eval').body
    expected_dump = ast.dump(expected, include_attributes=False)
    replacement_dump = ast.dump(replacement, include_attributes=False)

    class ReplaceGrid(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            if ast.dump(node, include_attributes=False) == expected_dump:
                self.count += 1
                return ast.copy_location(copy.deepcopy(replacement), node)
            return self.generic_visit(node)

    transform = ReplaceGrid()
    adapted = transform.visit(copy.deepcopy(original))
    require(transform.count == 2, 'Native train no longer contains exactly two known DEV-grid expressions')

    class ReverseGrid(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            if ast.dump(node, include_attributes=False) == replacement_dump:
                self.count += 1
                return ast.copy_location(copy.deepcopy(expected), node)
            return self.generic_visit(node)

    reverse = ReverseGrid()
    restored = reverse.visit(copy.deepcopy(adapted))
    require(reverse.count == 2 and ast.dump(restored, include_attributes=False) ==
            ast.dump(original, include_attributes=False), 'DEV control adaptation changed unrelated native syntax')
    ast.fix_missing_locations(adapted)
    namespace = {}
    exec(compile(adapted, native_train.__code__.co_filename + ':formal_DEV_endpoint', 'exec'),
         native_train.__globals__, namespace)
    result = namespace[native_train.__name__]
    result.__module__ = native_train.__module__
    return result, dict(
        schema='chapter5-formal-endpoint-DEV-control-adaptation-v1',
        regular_evaluation_interval=512, transformed_grid_expressions=transform.count,
        all_other_native_ast_identical=True, native_source_file_unchanged=True,
        native_train_source_sha256=hashlib.sha256(inspect.getsource(native_train).encode()).hexdigest(),
        original_ast_sha256=hashlib.sha256(ast.dump(original, include_attributes=False).encode()).hexdigest(),
        adapted_ast_sha256=hashlib.sha256(ast.dump(adapted, include_attributes=False).encode()).hexdigest(),
        learning_mathematics_modified=False, collector_modified=False,
        terminal_assessment_timing='after final native due update')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config_path):
    """Validate the complete new registered scope before Torch or Env imports."""
    global CHAPTER
    cfg = read_json(config_path)
    contract_path = Path(cfg['contract']).resolve(strict=True)
    contract = read_json(contract_path)
    require(contract.get('schema') == CONTRACT_SCHEMA, 'Unknown final execution contract')
    contract_sha = sha(contract_path)
    require(cfg.get('contract_sha256') == contract_sha, 'Contract SHA changed')
    require(contract.get('science_seeds') == [0, 1, 2, 3, 4], 'Formal five-seed registration required')
    require(cfg.get('method') == 'full' and cfg.get('profile') in VARIANTS,
            'One registered ablation variant required')
    require(cfg.get('variant', cfg['profile']) == cfg['profile'], 'Variant/profile identity differs')
    require(cfg['profile'] in contract['supported_variants'], 'Unregistered variant')
    require(cfg['domain'] in contract['domains'], 'Unregistered domain')
    require(type(cfg.get('engineering')) is bool, 'Engineering flag must be explicit')
    namespace = contract['engineering'] if cfg['engineering'] else contract
    require(not cfg['engineering'] or namespace['permitted'], 'Engineering is not permitted')
    for key in ('dev_paths','holdout_paths','train_master','dev_master','test_master',
                'dev_rng_seed','test_rng_seed'):
        require(cfg.get(key) == namespace[key], 'Registered field changed: '+key)
    require(cfg['holdout_paths'] == 0, 'This training stage must keep all holdout closed')
    budget = cfg['learning_steps']
    require(type(budget) is int and budget > 0, 'Invalid budget')
    if cfg['engineering']:
        require(budget in namespace['learning_steps'], 'Unregistered engineering budget')
        require(cfg['seed'] in namespace['seeds'] and cfg['seed'] not in contract['science_seeds'],
                'Engineering must use its separately registered seed')
    else:
        require(budget == contract['learning_steps'] == 100608 and cfg['seed'] in contract['science_seeds'],
                'Unregistered scientific budget or seed')
    h = contract['domains'][cfg['domain']]['horizon']
    require(cfg['eval_every'] == contract['eval_every'] == 512, 'Exact512 grid required')
    require(budget % (8*h) == 0, 'Complete collections and grid required')
    threshold=contract['domains'][cfg['domain']]['task_tau']
    require(cfg['task_tau'] == threshold, 'Task coverage threshold differs')
    require(cfg['device'] == 'cuda:0', 'Registered native CUDA update worker required')
    require(contract['domains'][cfg['domain']]['N'] == (100 if cfg['domain']=='radio' else 120),
            'Native task size differs')
    require(type(cfg['seed']) is int and cfg['seed'] >= 0, 'Invalid seed')
    require(len({cfg['train_master'],cfg['dev_master'],cfg['test_master']}) == 3,
            'Train, development and test namespaces must differ')
    output=Path(cfg['output']).expanduser().resolve()
    root=Path(contract['root']).expanduser().resolve()
    require(root.is_relative_to('/hy-tmp') and output.is_relative_to(root) and output != root,
            'Output must be a new registered HY child directory')
    CHAPTER=Path(cfg['source_root']).resolve(strict=True)
    require(CHAPTER.is_relative_to('/hy-tmp'), 'Frozen chapter sources must be on HY')
    require(CHAPTER == (root / 'execution/source').resolve(strict=True),
            'Source root is not the registered formal execution closure')
    manifest=Path(cfg['source_manifest']).resolve(strict=True)
    require(cfg['source_manifest_sha256'] == sha(manifest), 'Source manifest SHA changed')
    require(manifest == (root / 'execution/source_manifest.json').resolve(strict=True) and
            cfg['source_manifest_sha256'] == contract['source_manifest_sha256'],
            'Source manifest is not the contract-bound formal closure')
    require(isinstance(read_json(manifest).get('files'),dict) and read_json(manifest)['files'],
            'Nonempty source closure required')
    require(cfg['execution_worker_sha256'] == sha(__file__), 'Registered execution worker changed')
    return cfg, contract, contract_sha


def verify_sources(cfg):
    """Bind new closure and separately preserve both original scientific lists."""
    files = read_json(cfg['source_manifest'])['files']
    for name, expected in files.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source key')
        path = (CHAPTER / relative).resolve(strict=True)
        require(path.is_relative_to(CHAPTER), 'Source link escapes the frozen chapter: ' + name)
        require(sha(path) == expected, 'Frozen long source changed: ' + name)
    source = CHAPTER / '_internal/src'
    package = source / 'chapter5_full_spend_20261104'
    for filename, count, base in (
        ('native_source_manifest.json', 175, package / 'native_source'),
        ('development_source_manifest.json', 38, source),
    ):
        manifest_path = package / filename
        require(str(manifest_path.relative_to(CHAPTER)) in files, 'Original manifest not frozen')
        legacy = read_json(manifest_path)['files_sha256']
        require(len(legacy) == count, 'Original scientific source-list size changed')
        if filename == 'development_source_manifest.json':
            require(read_json(manifest_path).get('initialization_profile') == INITIALIZATION_PROFILE,
                    'Frozen closure has a different initialization profile')
        for name, expected in legacy.items():
            path = base / name
            relative = str(path.relative_to(CHAPTER))
            require(files.get(relative) == expected and sha(path) == expected,
                    'Original scientific file changed or absent from closure: ' + relative)
    return dict(sorted(files.items()))


def normalized_row(row, cfg, phase):
    """Keep native numbers and sufficient per-case data for independent checks."""
    import numpy as np
    result = {key: row[key] for key in ('index', 'plan_sha256', 'N', 'K', 'C', 'J',
        'native_return', 'total_used', 'remaining', 'full_spend', 'initial_budget',
        'quality', 'period_totals')}
    result.update(domain=cfg['domain'], tau=float(cfg['task_tau']),
                  phase=phase)
    quality = np.asarray(result['quality'], dtype=np.float64)
    require(quality.shape == (result['N'],) and np.isfinite(quality).all(), 'Malformed native terminal vector')
    require(abs(float(quality.mean()) - result['J']) <= 1e-10, 'Native J differs from terminal mean')
    require(int((quality >= result['tau']).sum()) == result['K'], 'Native K differs from hard threshold')
    require(abs(result['K'] / result['N'] - result['C']) <= 1e-12, 'Native C differs from K/N')
    require(len(result['period_totals']) == {'radio': 32, 'v5': 24}[cfg['domain']], 'Incomplete episode')
    require(result['full_spend']['full_spend_required'] is True, 'Full-spend proof missing')
    tolerance = float(result['full_spend']['tolerance'])
    require(abs(sum(result['period_totals']) - result['total_used']) <= tolerance and
            abs(result['total_used'] - result['initial_budget']) <= tolerance and
            abs(result['remaining']) <= tolerance, 'Full-spend numerical ledger failed')
    for key in ('plan', 'ledger_audit', 'physical_period_totals', 'physical_total_used',
                'physical_remaining', 'physical_resource_unit', 'action_unit'):
        if key in row:
            result[key] = row[key]
    return result


def metric_record(evaluation, cfg, *, kind, step, phase):
    import numpy as np
    require(set(evaluation) == {'w080'}, 'Full evaluation preference changed')
    arm = evaluation['w080']['deterministic']
    rows = [normalized_row(row, cfg, phase) for row in arm['rows']]
    require(rows and [row['index'] for row in rows] == list(range(len(rows))), 'Case indices incomplete')
    means = {key: float(np.mean([row[key] for row in rows])) for key in ('K', 'C', 'J')}
    for key in ('K', 'C', 'J'):
        require(abs(arm['mean_' + key] - means[key]) <= 1e-10, 'Native aggregate differs from its cases')
    return dict(kind=kind, learning_steps=step, rows=rows,
                **{'mean_' + key: value for key, value in means.items()},
                mean_F=.8 * means['C'] + .2 * means['J'])


def append_record(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    handle.flush()


@contextmanager
def native_adapters(native, cfg, contract, contract_sha, development, holdout, sources):
    """Process-local output hooks; all patched functions are restored on exit."""
    old_write = native.common.write_json
    old_sources = native.common.source_manifest
    old_trace = native.save_trace
    old_gpu_proof = native.gpu_proof
    old_assess = native.deterministic_assess
    old_args_valid = native._args_valid
    old_validate_worker = native.common.validate_worker
    out = Path(cfg['output']).resolve()
    emitted_ticks = []
    holdout_emitted = []
    reloaded_models = []
    grid = formal_evaluation_grid(cfg['learning_steps'], cfg['eval_every'])

    def registered_validation(args, family='proposed'):
        require(family == 'proposed', 'Unexpected native registration family')
        expected=dict(variant=cfg['profile'], domain=cfg['domain'], seed=cfg['seed'],
            tau=cfg['task_tau'], learning_steps=cfg['learning_steps'], eval_every=512,
            dev_paths=cfg['dev_paths'], holdout_paths=0, path_master=cfg['train_master'],
            dev_master=cfg['dev_master'], test_master=cfg['test_master'],
            dev_rng_seed=cfg['dev_rng_seed'], test_rng_seed=cfg['test_rng_seed'],
            device='cuda:0',eval_modes='deterministic')
        for name,value in expected.items():
            require(getattr(args,name) == value, 'Native arguments differ from registered contract: '+name)
        require(args.profile == ('engineering' if cfg['engineering'] else 'formal'),
                'Native profile does not reflect registered science/engineering identity')
        require(sha(cfg['contract']) == contract_sha, 'Contract drift before native validation')
        h=contract['domains'][cfg['domain']]['horizon']
        require(args.learning_steps%(8*h)==0,
                'Native collection/grid mismatch')
        return dict(execution=dict(actual_device='cuda:0', registered_training=not cfg['engineering'],
            final_execution_contract_sha256=contract_sha, configuration_sha256=sha(cfg['_config_path']),
            execution_worker_sha256=sha(__file__), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            native_historical_61440_contract_not_reused=True))

    def registered_args_valid(args):
        registered_validation(args)
        return contract['domains'][cfg['domain']]['horizon'], args.learning_steps//contract['domains'][cfg['domain']]['horizon']

    def source_manifest():
        actual = verify_sources(cfg)
        require(actual == sources, 'Long source identity changed')
        return actual

    def gpu_proof(actor, critic, actor_optimizer, critic_optimizer, event_ms):
        import torch
        result = old_gpu_proof(actor, critic, actor_optimizer, critic_optimizer, event_ms)
        tensors = [value for optimizer in (actor_optimizer, critic_optimizer)
                   for state in optimizer.state.values() for value in state.values()
                   if torch.is_tensor(value) and value.device.type == 'cuda']
        require(bool(tensors), 'No real CUDA optimizer tensors found')
        result['cuda_optimizer_tensors'] = len(tensors)
        result['normalizer_required'] = False
        result['normalizer_reason'] = 'Native structured encoder has no fitted running observation state'
        return result

    def assess(pool, actor, critic, count, phase, **kwargs):
        require(phase != 'final', 'All test/holdout evaluation remains closed during registered training')
        prefix = kwargs.get('trace_prefix')
        if phase == 'final':
            require(len(reloaded_models) == 1, 'Independent holdout requires the persisted endpoint reload')
            reload_actor, reload_critic = reloaded_models[0]
            before = native.model_hash(reload_actor), native.model_hash(reload_critic)
            require(before == (native.model_hash(actor), native.model_hash(critic)),
                    'Persisted endpoint differs from fixed training endpoint')
            result = old_assess(pool, reload_actor, reload_critic, count, phase, **kwargs)
            require(before == (native.model_hash(reload_actor), native.model_hash(reload_critic)),
                    'Holdout changed the persisted endpoint model')
            return result
        result = old_assess(pool, actor, critic, count, phase, **kwargs)
        if prefix is not None and Path(prefix).name == 'reload_endpoint':
            require(not reloaded_models, 'Multiple endpoint reload populations')
            reloaded_models.append((actor, critic))
        return result

    def write(path, value):
        path = Path(path)
        require(path.name != 'holdout.json', 'This training contract cannot open or persist holdout')
        if path.name == 'protocol.json':
            initialization = value.get('initialization', {})
            require(initialization.get('profile') == INITIALIZATION_PROFILE and initialization.get('fresh') is True
                    and initialization.get('old_models_loaded') is False
                    and initialization.get('seed') == cfg['seed'], 'Dual-random initialization identity failed')
            require(set(initialization.get('heads', {})) == {'spatial', 'quota'}, 'Both random heads required')
            require(value.get('evaluation_ticks') == grid, 'Native protocol lacks terminal DEV tick')
            value.update(traces=dict(policy='No action NPZ files',
                retained='Per-case native quality, period totals, plan and full-spend ledger in JSON',
                training_batches_actions_and_rng_unchanged=True),
                final_contract_sha256=contract_sha, registered_training=not cfg['engineering'],
                formal_execution_facade=SCHEMA, initialization_profile=INITIALIZATION_PROFILE)
        old_write(path, value)
        if path.name == 'progress.json':
            write_json(out / 'progress.json', dict(value, method='full', profile=cfg['profile'],
                engineering=cfg['engineering'], target=cfg['learning_steps']))
        elif path.name.startswith('evaluation_') and path.suffix == '.json':
            tick = int(value['learning_step'])
            require(len(emitted_ticks) < len(grid) and tick == grid[len(emitted_ticks)], 'DEV grid changed')
            record = metric_record(value['preferences'], cfg, kind='curve', step=tick, phase='dev')
            require(len(record['rows']) == cfg['dev_paths'], 'DEV population changed')
            audit = value['parameter_audit']
            require(audit['unchanged'] is True and audit['before'] == audit['after'],
                    'Native DEV failed its complete training/RNG audit')
            record.update(updates=value['policy_update'], timing=value['timing'],
                          training_state_unchanged=True, rng_and_learning_state_unchanged=True)
            append_record(development, record)
            emitted_ticks.append(tick)
        elif path.name == 'holdout.json':
            require(not holdout_emitted and value['model_selection'] is False and
                    value['holdout_selection'] is False, 'Unexpected holdout selection')
            record = metric_record(value['endpoint']['preferences'], cfg,
                kind='fixed_endpoint_holdout', step=cfg['learning_steps'], phase='final')
            require(len(record['rows']) == cfg['holdout_paths'], 'Independent holdout population changed')
            audit = value['endpoint']['parameter_audit']
            require(audit['unchanged'] is True and audit['before'] == audit['after'] and
                    len(reloaded_models) == 1, 'Native holdout failed training/RNG or persisted-reload audit')
            record.update(model_selection=False, holdout_selection=False,
                          training_state_unchanged=True, rng_and_learning_state_unchanged=True,
                          evaluated_from_persisted_endpoint=True, endpoint_reload_exact=True,
                          heldout_data=True, case_master=cfg['test_master'],
                          evaluation_rng_seed=cfg['test_rng_seed'])
            append_record(holdout, record)
            holdout_emitted.append(True)

    old_clock = native.DenseEvaluationClock
    old_train = native.train
    old_grid = native.__dict__.get('formal_evaluation_grid')
    native.formal_evaluation_grid = formal_evaluation_grid
    native.DenseEvaluationClock = endpoint_clock_type(old_clock)
    native.train, grid_adaptation = adapt_train_grid(old_train)
    write_json(out / 'evaluation_clock_adaptation.json', grid_adaptation)
    native._args_valid = registered_args_valid
    native.common.validate_worker = registered_validation
    native.common.write_json = write
    native.common.source_manifest = source_manifest
    native.save_trace = lambda path, results: None
    native.gpu_proof = gpu_proof
    native.deterministic_assess = assess
    try:
        yield emitted_ticks, holdout_emitted
    finally:
        native.train = old_train
        native.DenseEvaluationClock = old_clock
        if old_grid is None:
            native.__dict__.pop('formal_evaluation_grid', None)
        else:
            native.formal_evaluation_grid = old_grid
        native._args_valid = old_args_valid
        native.common.validate_worker = old_validate_worker
        native.common.write_json = old_write
        native.common.source_manifest = old_sources
        native.save_trace = old_trace
        native.gpu_proof = old_gpu_proof
        native.deterministic_assess = old_assess


def train(config_path):
    cfg, contract, contract_sha = validate_config(config_path)
    sources = verify_sources(cfg)
    initial_source_sha = sha(cfg['source_manifest'])
    initial_config_sha = sha(config_path)
    cfg = dict(cfg, _config_path=str(Path(config_path).resolve()))
    initial_worker_sha = sha(__file__)
    out = Path(cfg['output']).expanduser().resolve()
    require(not out.exists(), 'Never overwrite/recover a long Full job')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(visible.startswith('GPU-') and ',' not in visible and os.environ.get('CH5_GPU_UUID') == visible,
            'Scheduler must expose exactly one registered physical GPU UUID')
    os.environ['CH5_FULL_SPEND_DATA_ROOT'] = str(cfg['data_root'])
    os.environ['CH5_WORKSPACE'] = str(CHAPTER / '_internal/src')
    sys.path.insert(0, str(CHAPTER / '_internal/src'))
    sys.path.insert(0, str(CHAPTER))
    native = importlib.import_module('chapter5_full_spend_20261104.worker')
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One real native CUDA device required')
    out.mkdir(parents=True)
    started = time.monotonic()
    native_config = out / 'own_native_config.json'
    tau = cfg['task_tau']
    write_json(native_config, dict(variant=cfg['profile'], tau=tau))
    args = argparse.Namespace(variant=cfg['profile'], domain=cfg['domain'], seed=cfg['seed'], tau=tau,
        learning_steps=cfg['learning_steps'], eval_every=cfg['eval_every'], dev_paths=cfg['dev_paths'],
        holdout_paths=cfg['holdout_paths'], eval_batch=8, output=str(out / 'native'),
        profile='engineering' if cfg['engineering'] else 'formal',
        config_json=str(native_config), run_protocol=cfg['contract'], job_id=cfg.get('id'),
        device='cuda:0', path_master=cfg['train_master'], dev_master=cfg['dev_master'],
        test_master=cfg['test_master'], dev_rng_seed=cfg['dev_rng_seed'],
        test_rng_seed=cfg['test_rng_seed'], eval_modes='deterministic', diagnostics_every=512)
    protocol = dict(schema=SCHEMA, config={key:value for key,value in cfg.items() if not key.startswith('_')}, config_sha256=initial_config_sha,
        contract_sha256=contract_sha, source_manifest_sha256=initial_source_sha,
        registered_training=not cfg['engineering'], full_spend_required=True,
        gpu_uuid=os.environ.get('CH5_GPU_UUID', visible), cuda_visible_devices=visible,
        native_profile=args.profile, original_formal_scope_reused=False,
        evaluation_grid=formal_evaluation_grid(cfg['learning_steps'], cfg['eval_every']),
        holdout_paths=cfg['holdout_paths'], holdout_selection=False, model_selection=False,
        observation_normalization='native structured state encoder; no fitted external observation map',
        model_normalizer_state='native encoder has no learned running observation normalizer',
        storage=dict(endpoint='native/checkpoint_endpoint.pt', optimizer_backup=False,
            replay_backup=False, resume_supported=False, action_arrays_on_disk=False,
            per_case_quality_plan_period_totals_and_full_spend_retained=True),
        native_scientific_core='unchanged variant-specific learner/collector/Actor/Critic/Adam',
        variant=cfg['profile'], execution_worker_sha256=initial_worker_sha,
        registration_overrides=['_args_valid','common.validate_worker'],
        evaluation_control_overrides=['DenseEvaluationClock','two train grid expressions'],
        initialization_profile=INITIALIZATION_PROFILE, holdout_not_opened=True)
    write_json(out / 'protocol.json', protocol)
    try:
        with gzip.open(out / 'development.jsonl.gz', 'wt') as dev:
            with native_adapters(native, cfg, contract, contract_sha, dev, None, sources) as (ticks, final):
                completed = native.train(args)
            require(ticks == protocol['evaluation_grid'] and final == [], 'Missing DEV grid or holdout opened')
        require(sha(cfg['source_manifest']) == initial_source_sha and
                sha(config_path) == initial_config_sha and sha(cfg['contract']) == contract_sha and
                sha(__file__) == initial_worker_sha,
                'Registered source/config/contract mutated during Full training')
        require(verify_sources(cfg) == sources, 'Source changed after endpoint evaluation')
        require(completed['source_unchanged'] and completed['endpoint_reload_verified'] and
                completed['saved_model_count'] == 1 and completed['actor_changed'] and
                completed['critic_changed'], 'Native endpoint or genuine update verification failed')
        h = contract['domains'][cfg['domain']]['horizon']
        counts = dict(completed['counts'], internal=dict(episodes=0, physical_steps=0))
        require(counts['train']['physical_steps'] == cfg['learning_steps'] and
                counts['development']['physical_steps'] == len(ticks) * cfg['dev_paths'] * h and
                counts['holdout']['physical_steps'] == cfg['holdout_paths'] * h and
                counts['reload_verification']['physical_steps'] == cfg['dev_paths'] * h,
                'Native physical interaction accounting differs from contract')
        endpoint = out / 'native/checkpoint_endpoint.pt'
        require(sha(endpoint) == completed['endpoint_sha256'], 'Endpoint checksum changed')
        gpu = dict(completed['gpu_proof'], device='cuda:0',
            parameter_devices=completed['gpu_proof']['model_parameter_devices'],
            gpu_uuid=visible, cuda_update_event_ms=completed['gpu_proof']['gpu_update_event_milliseconds'],
            optimizer_moments_on_cuda=True)
        result = dict(success=True, schema=SCHEMA, method='full', profile=cfg['profile'], variant=cfg['profile'],
            domain=cfg['domain'], seed=cfg['seed'], engineering=cfg['engineering'],
            registered_training=not cfg['engineering'], contract_sha256=contract_sha,
            learning_steps=cfg['learning_steps'], updates=None,
            counts=counts, source_manifest_sha256=initial_source_sha, config_sha256=initial_config_sha,
            source_unchanged=True, endpoint_sha256=completed['endpoint_sha256'],
            endpoint_filename='native/checkpoint_endpoint.pt', endpoint_reload_exact=True,
            holdout_evaluated_from_persisted_endpoint=False, holdout_not_opened=True,
            normalizer_required=False, observation_normalization=False,
            normalizer_state_in_endpoint='not applicable: stateless native structured encoder',
            neural_parameters_changed=True, optimizer_backup=False, replay_backup=False,
            holdout_paths=cfg['holdout_paths'], gpu_proof=gpu, seconds=time.monotonic() - started,
            model_selection=False, holdout_selection=False, extra_exchange_updates=0,
            saved_model_count=1, native_completed='native/completed.json',
            native_protocol='native/protocol.json', evaluation_grid=ticks,
            dtype='float64', training_restore_supported=False,
            execution_worker_sha256=initial_worker_sha, native_registered_formal_training=completed['formal_training'],
            initialization_profile=INITIALIZATION_PROFILE,
            initialization_heads=read_json(out/'native/protocol.json')['initialization']['heads'],
            initial_actor_sha256=read_json(out/'native/protocol.json')['initialization']['actor_sha256'],
            initial_critic_sha256=read_json(out/'native/protocol.json')['initialization']['critic_sha256'])
        # Count the native committed update records, independently of audit hashes.
        with (out / 'native/updates.jsonl').open() as handle:
            result['updates'] = sum(1 for line in handle if line.strip())
        require(result['updates'] > 0, 'No native CUDA update recorded')
        write_json(out / 'completed.json', result)
        return result
    except BaseException:
        write_json(out / 'failure.json', dict(success=False, schema=SCHEMA,
            method='full', domain=cfg['domain'], seed=cfg['seed'],
            traceback=traceback.format_exc(), no_automatic_score_dependent_retry=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        cfg, contract, _ = validate_config(args.config)
        verify_sources(cfg)
        print(json.dumps(dict(status='PASS_PURE_FILE_VALIDATION',variant=cfg['profile'],environment_steps=0)))
    else:
        train(args.config)


if __name__ == '__main__':
    main()
