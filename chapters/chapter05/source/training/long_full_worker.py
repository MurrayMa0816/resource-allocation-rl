"""Registered long-budget facade for the unchanged native Full CUDA learner.

Only registration, source closure, output storage and result schemas are adapted.
The native collector, F1 reuse, Actor/Critic/Adam, objectives and random streams
are called directly. This is a new contract, not the historical 61,440-step run.
"""
from __future__ import annotations

import argparse
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

CHAPTER = Path(__file__).resolve().parents[1]
SCHEMA = 'chapter5-long-full-facade-v1'


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
    """Pure-file validation; does not import Torch or construct an environment."""
    cfg = read_json(config_path)
    contract_path = Path(cfg['contract']).resolve(strict=True)
    contract = read_json(contract_path)
    require(contract.get('schema') == 'chapter5-long-comparison-contract-v1', 'Unknown long contract')
    contract_sha = sha(contract_path)
    require(cfg.get('contract_sha256') == contract_sha, 'Config must bind the exact long contract SHA')
    require(cfg.get('method') == 'full' and cfg.get('profile') == 'full', 'Full identity required')
    require(cfg['domain'] in contract['domains'], 'Unknown registered domain')
    require(type(cfg.get('engineering')) is bool, 'Explicit engineering boolean required')
    namespace = contract['engineering'] if cfg['engineering'] else contract
    require(not cfg['engineering'] or namespace['permitted'], 'Engineering gate not permitted')
    for key in ('dev_paths', 'holdout_paths', 'train_master', 'dev_master', 'test_master',
                'dev_rng_seed', 'test_rng_seed'):
        require(cfg.get(key) == namespace[key], 'Registered Full field changed: ' + key)
    budget = cfg['learning_steps']
    if cfg['engineering']:
        require(budget in namespace['learning_steps'], 'Unregistered engineering budget')
    else:
        require(budget == contract['learning_steps'] and cfg['seed'] in contract['science_seeds'],
                'Unregistered scientific budget or seed')
        require(contract['science_profiles']['full'] == cfg['profile'], 'Full profile changed')
    h = contract['domains'][cfg['domain']]['horizon']
    require(cfg['eval_every'] == contract['eval_every'] == 512, 'Exact 512 DEV required')
    require(budget > 0 and budget % 512 == 0 and budget % (8 * h) == 0,
            'Complete native Full collections and DEV grid required')
    require(cfg['device'] == 'cuda:0', 'Full facade preserves the native CUDA-only worker')
    output = Path(cfg['output']).expanduser().resolve()
    root = Path(contract['root']).expanduser().resolve()
    require(output.is_relative_to(root) and root.is_relative_to('/hy-tmp') and output != root,
            'New outputs must be a job directory inside the registered HY root')
    manifest_path = Path(cfg['source_manifest']).resolve(strict=True)
    manifest = read_json(manifest_path)
    files = manifest['files']
    require(isinstance(files, dict) and bool(files), 'Nonempty frozen source mapping required')
    own_name = str(Path(__file__).resolve().relative_to(CHAPTER))
    require(own_name in files, 'The Full facade must itself be in the frozen source mapping')
    if 'source_manifest_sha256' in cfg:
        require(cfg['source_manifest_sha256'] == sha(manifest_path), 'Source manifest SHA changed')
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
        ('development_source_manifest.json', 37, source),
    ):
        manifest_path = package / filename
        require(str(manifest_path.relative_to(CHAPTER)) in files, 'Original manifest not frozen')
        legacy = read_json(manifest_path)['files_sha256']
        require(len(legacy) == count, 'Original scientific source-list size changed')
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
    result.update(domain=cfg['domain'], tau=float(cfg.get('tau', {'radio': .2, 'v5': .7}[cfg['domain']])),
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
def native_adapters(native, cfg, contract_sha, development, holdout, sources):
    """Process-local output hooks; all patched functions are restored on exit."""
    old_write = native.common.write_json
    old_sources = native.common.source_manifest
    old_trace = native.save_trace
    old_gpu_proof = native.gpu_proof
    old_assess = native.deterministic_assess
    out = Path(cfg['output']).resolve()
    emitted_ticks = []
    holdout_emitted = []
    reloaded_models = []

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
        if path.name == 'protocol.json':
            value = dict(value, traces=dict(policy='No action NPZ files',
                retained='Per-case native quality, period totals, plan and full-spend ledger in JSON',
                training_batches_actions_and_rng_unchanged=True),
                long_contract_sha256=contract_sha, registered_training=not cfg['engineering'])
        old_write(path, value)
        if path.name == 'progress.json':
            write_json(out / 'progress.json', dict(value, method='full', profile='full',
                engineering=cfg['engineering'], target=cfg['learning_steps']))
        elif path.name.startswith('evaluation_') and path.suffix == '.json':
            tick = int(value['learning_step'])
            require(tick not in emitted_ticks and tick == len(emitted_ticks) * 512, 'DEV grid changed')
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

    native.common.write_json = write
    native.common.source_manifest = source_manifest
    native.save_trace = lambda path, results: None
    native.gpu_proof = gpu_proof
    native.deterministic_assess = assess
    try:
        yield emitted_ticks, holdout_emitted
    finally:
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
    out = Path(cfg['output']).expanduser().resolve()
    require(not out.exists(), 'Never overwrite/recover a long Full job')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(bool(visible) and ',' not in visible, 'Scheduler must expose exactly one explicit GPU')
    os.environ['CH5_FULL_SPEND_DATA_ROOT'] = str(cfg['data_root'])
    os.environ['CH5_WORKSPACE'] = str(CHAPTER / '_internal/src')
    sys.path.insert(0, str(CHAPTER / '_internal/src'))
    sys.path.insert(0, str(CHAPTER))
    native = importlib.import_module('chapter5_full_spend_20261104.worker')
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One real native CUDA device required')
    out.mkdir(parents=True)
    started = time.monotonic()
    native_config = out / 'full_native_config.json'
    tau = contract['domains'][cfg['domain']]['tau']
    require(cfg.get('tau', tau) == tau, 'Registered task threshold changed')
    write_json(native_config, dict(variant='full', tau=tau))
    args = argparse.Namespace(variant='full', domain=cfg['domain'], seed=cfg['seed'], tau=tau,
        learning_steps=cfg['learning_steps'], eval_every=cfg['eval_every'], dev_paths=cfg['dev_paths'],
        holdout_paths=cfg['holdout_paths'], eval_batch=8, output=str(out / 'native'),
        profile='long_registered_engineering_gate' if cfg['engineering'] else 'long_registered',
        config_json=str(native_config), run_protocol=cfg['contract'], job_id=cfg.get('id'),
        device='cuda:0', path_master=cfg['train_master'], dev_master=cfg['dev_master'],
        test_master=cfg['test_master'], dev_rng_seed=cfg['dev_rng_seed'],
        test_rng_seed=cfg['test_rng_seed'], eval_modes='deterministic', diagnostics_every=512)
    protocol = dict(schema=SCHEMA, config=cfg, config_sha256=initial_config_sha,
        contract_sha256=contract_sha, source_manifest_sha256=initial_source_sha,
        registered_training=not cfg['engineering'], full_spend_required=True,
        gpu_uuid=os.environ.get('CH5_GPU_UUID', visible), cuda_visible_devices=visible,
        native_profile=args.profile, original_formal_scope_reused=False,
        evaluation_grid=list(range(0, cfg['learning_steps'] + 1, 512)),
        holdout_paths=cfg['holdout_paths'], holdout_selection=False, model_selection=False,
        observation_normalization='native structured state encoder; no fitted external observation map',
        model_normalizer_state='native encoder has no learned running observation normalizer',
        storage=dict(endpoint='native/checkpoint_endpoint.pt', optimizer_backup=False,
            replay_backup=False, resume_supported=False, action_arrays_on_disk=False,
            per_case_quality_plan_period_totals_and_full_spend_retained=True),
        native_scientific_core='unchanged Full learner/collector/Actor/Critic/Adam')
    write_json(out / 'protocol.json', protocol)
    try:
        with gzip.open(out / 'development.jsonl.gz', 'wt') as dev, \
                gzip.open(out / 'holdout.jsonl.gz', 'wt') as hold:
            with native_adapters(native, cfg, contract_sha, dev, hold, sources) as (ticks, final):
                completed = native.train(args)
            require(ticks == protocol['evaluation_grid'] and final == [True], 'Missing exact evaluation population')
        require(sha(cfg['source_manifest']) == initial_source_sha and
                sha(config_path) == initial_config_sha and sha(cfg['contract']) == contract_sha,
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
        result = dict(success=True, schema=SCHEMA, method='full', profile='full',
            domain=cfg['domain'], seed=cfg['seed'], engineering=cfg['engineering'],
            registered_training=not cfg['engineering'], contract_sha256=contract_sha,
            learning_steps=cfg['learning_steps'], updates=None,
            counts=counts, source_manifest_sha256=initial_source_sha, config_sha256=initial_config_sha,
            source_unchanged=True, endpoint_sha256=completed['endpoint_sha256'],
            endpoint_filename='native/checkpoint_endpoint.pt', endpoint_reload_exact=True,
            holdout_evaluated_from_persisted_endpoint=True,
            normalizer_required=False, observation_normalization=False,
            normalizer_state_in_endpoint='not applicable: stateless native structured encoder',
            neural_parameters_changed=True, optimizer_backup=False, replay_backup=False,
            holdout_paths=cfg['holdout_paths'], gpu_proof=gpu, seconds=time.monotonic() - started,
            model_selection=False, holdout_selection=False, extra_exchange_updates=0,
            saved_model_count=1, native_completed='native/completed.json',
            native_protocol='native/protocol.json', evaluation_grid=ticks,
            dtype='float64', training_restore_supported=False)
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
    args = parser.parse_args()
    train(args.config)


if __name__ == '__main__':
    main()
