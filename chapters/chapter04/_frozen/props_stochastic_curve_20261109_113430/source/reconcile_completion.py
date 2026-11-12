"""Accept complete fixed science after the narrowly identified path-key audit failure.

Original worker, failure records, native results and controller exit status are
read-only. No model is deserialized and no environment or inference is run.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = Path('/root/resource_allocation_run_storage_20261101')
ARCHIVE = Path('/hy-tmp/resource_allocation_results/project_archives_20261109/runtime_sources/legacy_execution_store')
EXPECTED_ERROR = "AssertionError: Read-only imported source changed during execution"
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('_curve_completion_aggregate', ROOT / 'source/aggregate.py')
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)
sha, read, dump = aggregate.sha, aggregate.read, aggregate.dump


def canonical_sources(job):
    before_path = job / 'source_binding_before.json'
    before = read(before_path)
    assert len(before) == 39
    if not OLD_ROOT.is_symlink() or OLD_ROOT.resolve() != ARCHIVE:
        raise AssertionError('Only the identified old-root to archive symlink migration may be reconciled')
    records = []
    changed = []
    keys = set()
    for logical, digest in before.items():
        path = Path(logical)
        canonical = path.resolve()
        relocated = canonical != path
        if relocated:
            if not path.is_relative_to(OLD_ROOT) or canonical != ARCHIVE / path.relative_to(OLD_ROOT):
                raise AssertionError('Unexpected canonical path migration: ' + logical)
            changed.append(logical)
        if str(canonical) in keys:
            raise AssertionError('Canonical manifest collision')
        keys.add(str(canonical))
        if sha(canonical) != digest:
            raise AssertionError('Before-bound source content changed: ' + logical)
        records.append(dict(logical_path=logical, canonical_path=str(canonical), sha256=digest,
                            path_key_relocated=relocated, content_unchanged=True))
    if len(changed) != 1 or changed[0] != str(OLD_ROOT / 'chapter4_recent_baselines_formal_20261102_source/chapter4_recent_baselines/__init__.py'):
        raise AssertionError('Failure must be exactly the identified dynamic package path-key change')
    return dict(schema='props-source-path-identity-reconciliation-v1', status='PASS',
        cause='sources() used dynamically resolved absolute paths as keys; the legacy root moved to a symlink during execution, so one unchanged package source obtained a different dictionary key',
        before_manifest_sha256=sha(before_path), files=records,
        allowed_mapping=dict(logical_root=str(OLD_ROOT), canonical_root=str(ARCHIVE)),
        relocated_key_count=len(changed), unchanged_source_count=len(records),
        old_root_symlink_mtime_epoch=OLD_ROOT.lstat().st_mtime,
        added_environment_steps=0, model_deserializations=0)


def validate_vendor(job):
    receipt = read(job / 'source_receipt.json')
    expected_commit = 'a0c8708529047f85dcf98f90d5ea264dd43aeba2'
    assert receipt['expected_sha'] == receipt['actual_sha'] == expected_commit
    assert receipt['source_modified'] is False and receipt['functions'] == ['update_ppo', 'update_props']
    path = Path(receipt['source_file'])
    assert sha(path) == receipt['sha256']
    if path.resolve() != ARCHIVE / path.relative_to(OLD_ROOT):
        raise AssertionError('Vendor must use the same narrowly allowed migration')
    repository = path.parents[1]
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    pristine = subprocess.check_output(['git', 'show', expected_commit + ':PROPS/ppo_props_continuous.py'], cwd=repository)
    assert actual == expected_commit and path.read_bytes() == pristine
    return dict(status='PASS', commit=actual, source_sha256=receipt['sha256'],
                source_receipt_sha256=sha(job / 'source_receipt.json'), source_unmodified=True)


def crc_checks(job):
    checks = []
    for path in sorted(job.glob('*.gz')):
        size = 0
        with gzip.open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(2**20), b''):
                size += len(block)
        checks.append(dict(path=str(path), sha256=sha(path), uncompressed_bytes=size, full_gzip_CRC_pass=True))
    assert {Path(record['path']).name for record in checks} == {
        'evaluation_paths.csv.gz', 'evaluation_panels.csv.gz', 'resource_audit.csv.gz', 'physical_episode_ledgers.jsonl.gz'}
    return checks


def proposed_completion(seed, job, failed, reconciliation, source_check):
    assert failed['traceback'].strip().endswith(EXPECTED_ERROR)
    assert 'line 509, in main' in failed['traceback']
    assert failed['source_binding_before'] == read(job / 'source_binding_before.json')
    assert not (job / 'completion.json').exists() and not (job / 'completed.json').exists()
    result = read(job / 'result.json')
    assert result['status'] == 'completed' and result['method'] == 'PROPS' and result['seed'] == seed and result['domain'] == 'v5'
    assert result['train_env_steps'] == 100608 and result['episodes'] == 4192
    assert result['target_updates'] == 1048 and result['behavior_updates'] == 4192
    mechanism = read(job / 'mechanism_checks.json')
    assert mechanism['checkpoint_roundtrip_exact'] and mechanism['nonterminal_reward_count'] == 0
    assert mechanism['completed_episodes'] == mechanism['terminal_feedback_count'] == 4192
    assert mechanism['target_updates'] == 1048 and mechanism['behavior_updates'] == 4192
    counts = failed['physical_counts']
    for phase, wanted in dict(train=100608, mean=101376, stochastic=811008).items():
        counter = counts[phase]
        assert counter['attempted_physical_steps'] == counter['successful_physical_steps'] == wanted
        assert counter['completed_physical_episodes'] == counter['guarded_complete_episodes'] == wanted // 24
    assert failed['actual_total_successful_physical_steps'] == 1012992
    repro = read(job / 'reproduction_comparison.json')
    assert repro['reproduction_pass'] and repro['target_state_all_tensors_exact']
    assert repro['target_state_max_absolute_difference'] == repro['original_mean_max_absolute_error'] == 0.
    assert repro['complete_grid_and_cases'] and repro['original_mean_compared_nodes'] == 132
    return dict(schema='props-independent-scientific-completion-acceptance-v1',
        acceptance_status='ACCEPTED_AFTER_SOURCE_IDENTITY_RECONCILIATION', scientific_status='COMPLETE',
        original_worker_process_status='failed_in_postrun_dynamic_source_registry_guard',
        original_worker_failure_retained=True, original_failed_path=str(job / 'failed.json'),
        original_failed_sha256=sha(job / 'failed.json'), original_native_result_sha256=sha(job / 'result.json'),
        seed=seed, domain='v5', engineering_only=False, training_from_scratch=True,
        training_device='cuda:0' if seed == 2 else 'cpu', stochastic_device='cpu',
        fixed_endpoint_steps=100608, evaluation_grid=aggregate.NODES, dev_cases=32,
        stochastic_replicates=8, temperature=1., actual_physical_counts=counts,
        actual_total_physical_steps=1012992, independent_TEST_steps=0,
        model=str(job / 'checkpoint.pt'), model_sha256=sha(job / 'checkpoint.pt'),
        model_deserializations=dict(native_checkpoint_roundtrip=1, new_fixed_endpoint_inspection=1, old_fixed_endpoint_comparison=1),
        model_deserialization_evidence='Frozen worker reaches saved exact reproduction comparison before the line509 failure; its three explicit loads therefore already occurred. Acceptor loads zero models.',
        reproduction_comparison=repro, one_fixed_endpoint=True, periodic_models=False, optimizer_or_replay_saved=False,
        plan=dict(path=str(ROOT / 'protocol/PLAN.json'), sha256=aggregate.PLAN_SHA),
        postrun_registry_failure=1, physical_execution_failures=0, scientific_retries=0,
        added_environment_steps=0, added_model_deserializations=0, automatic_retry=False,
        wall_seconds=failed['elapsed_seconds'],
        source_identity_reconciliation_sha256=None, protected_original_source_check=source_check,
        acceptance_source_sha256=sha(__file__), aggregate_verification_source_sha256=sha(ROOT / 'source/aggregate.py'))


def main(args):
    started = time.perf_counter()
    source_check = aggregate.source_protection()
    assert source_check['protected_original_files'] == 192
    status = read(ROOT / 'runtime/status.json')
    workers = {record['seed']: record for record in status['workers']}
    common_plans = {}
    accepted, pending, receipts = [], [], []
    for seed in aggregate.SEEDS:
        job = ROOT / 'jobs' / f'props__v5__seed{seed}'
        target = aggregate.acceptance_directory(seed)
        receipt_path = target / 'accepted_completion.json'
        if receipt_path.is_file():
            receipt = read(receipt_path)
            assert receipt['acceptance_status'] == 'ACCEPTED_AFTER_SOURCE_IDENTITY_RECONCILIATION'
            assert receipt['original_failed_sha256'] == sha(job / 'failed.json')
            assert receipt['source_identity_reconciliation_sha256'] == sha(target / 'source_identity_reconciliation.json')
            assert receipt['scientific_verification_sha256'] == sha(target / 'scientific_verification.json')
            assert receipt['original_native_result_sha256'] == sha(job / 'result.json')
            assert receipt['model_sha256'] == sha(job / 'checkpoint.pt')
            verification = read(target / 'scientific_verification.json')
            for name, digest in verification['files'].items():
                assert sha(job / name) == digest
            for check in verification['full_gzip_CRC_checks']:
                assert sha(check['path']) == check['sha256']
            aggregate.verify_completed_sources(job)
            for key, identity in verification['DEV32_plan_identity'].items():
                case, value = int(key), tuple(identity)
                assert case not in common_plans or common_plans[case] == value
                common_plans[case] = value
            accepted.append(seed); receipts.append(dict(seed=seed, path=str(receipt_path), sha256=sha(receipt_path)))
            continue
        if workers[seed]['returncode'] is None:
            pending.append(dict(seed=seed, reason='original worker process is still active; no files accepted or modified'))
            continue
        if (job / 'completion.json').is_file():
            # A genuine normal completion needs no registry repair, but enters
            # the same final scientific acceptance index unchanged.
            aggregate.recompute_seed(seed, common_plans)
            accepted.append(seed)
            receipts.append(dict(seed=seed, path=str(job / 'completion.json'), sha256=sha(job / 'completion.json')))
            continue
        if not (job / 'failed.json').is_file():
            pending.append(dict(seed=seed, reason='closed process lacks failure/completion evidence'))
            continue
        failed = read(job / 'failed.json')
        reconciliation = canonical_sources(job)
        vendor = validate_vendor(job)
        checks = crc_checks(job)
        proposed = proposed_completion(seed, job, failed, reconciliation, source_check)
        _, verification = aggregate.recompute_seed(seed, common_plans, completion_override=proposed,
                                                  reconciliation_override=reconciliation)
        verification.update(DEV32_plan_identity={str(key): list(value) for key, value in sorted(common_plans.items())},
                            full_gzip_CRC_checks=checks, vendor=vendor)
        if target.exists():
            raise FileExistsError('Never overwrite a partial acceptance directory: ' + str(target))
        target.mkdir(parents=True)
        dump(target / 'source_identity_reconciliation.json', reconciliation)
        dump(target / 'scientific_verification.json', verification)
        proposed['source_identity_reconciliation_sha256'] = sha(target / 'source_identity_reconciliation.json')
        proposed['scientific_verification_sha256'] = sha(target / 'scientific_verification.json')
        accepted_result = read(job / 'result.json')
        accepted_result.update(status='scientific_complete_after_independent_path_identity_reconciliation',
            original_worker_exit_status_preserved='failed', original_native_result_unchanged=True,
            actual_physical_counts=proposed['actual_physical_counts'], postrun_registry_failure=1,
            physical_execution_failures=0, scientific_retries=0, added_environment_steps=0, added_model_deserializations=0,
            stochastic_evaluation_env_steps=811008, total_evaluation_env_steps=912384,
            evaluation_policy='original-device mean plus isolated CPU native Gaussian random target, eight fixed streams',
            independent_completion_receipt=str(receipt_path))
        dump(target / 'accepted_result.json', accepted_result)
        dump(receipt_path, proposed)
        accepted.append(seed); receipts.append(dict(seed=seed, path=str(receipt_path), sha256=sha(receipt_path)))
        print(json.dumps(dict(seed=seed, scientific_status='ACCEPTED', original_failure_preserved=True,
                              added_environment_steps=0, added_model_deserializations=0)), flush=True)
    output_status = dict(schema='props-fixed-science-independent-acceptance-index-v1',
        status='ALL_FIVE_SCIENTIFIC_RUNS_ACCEPTED' if accepted == aggregate.SEEDS and status['running'] == 0 else 'PENDING_ACTIVE_OR_INCOMPLETE_ORIGINAL_WORKERS',
        accepted_seeds=accepted, pending=pending, receipts=receipts, original_controller_status=status,
        original_controller_status_unchanged=True, original_failed_records_retained=True,
        added_environment_steps=0, added_model_deserializations=0, scientific_retries=0,
        postrun_source_registry_failures=sum(read(record['path']).get('postrun_registry_failure', 0) for record in receipts),
        protected_original_source_check=source_check, acceptor_source_sha256=sha(__file__),
        aggregate_source_sha256=sha(ROOT / 'source/aggregate.py'), elapsed_seconds=time.perf_counter() - started)
    dump(ROOT / 'audit/postprocess_acceptance/status.json', output_status)
    if output_status['status'] == 'ALL_FIVE_SCIENTIFIC_RUNS_ACCEPTED':
        final = ROOT / 'runtime/scientific_acceptance.json'
        if final.exists():
            existing = read(final)
            assert existing['receipts'] == receipts and existing['original_controller_status'] == status
        else:
            dump(final, output_status)
    print(json.dumps(dict(status=output_status['status'], accepted_seeds=accepted, pending=pending,
                          added_environment_steps=0, added_model_deserializations=0)), flush=True)


if __name__ == '__main__':
    main(argparse.ArgumentParser(description=__doc__).parse_args())
