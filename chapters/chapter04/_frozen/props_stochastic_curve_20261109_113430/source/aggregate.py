"""Read-only recomputation of the five completed fixed PROPS stochastic curves.

No environment, torch, optimizer, inference, or model deserialization is used.
All outputs are derived from persisted complete per-case records. The full
physical ledgers are independently checked by the separate audit workflow.
"""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORMAL = Path('/hy-tmp/resource_allocation_results/chapter04/experiments/double_random_formal_20261107')
ORIGINAL_FIGURE = Path('/hy-tmp/resource_allocation_results/chapter04/figures/readability_refinement_20261108_234201/assets/figure4-4')
PREVIOUS_DIAGNOSTIC = Path('/hy-tmp/resource_allocation_results/chapter04/experiments/action_sampling_diagnostic_20261109/results/summary.json')
PLAN_SHA = '5f5965769674e6788005aba34cc3e493d633417cdb2227f57e3f132dbcc26afd'
WORKER_SHA = '7a01c742960c2624c540e0673acf029c94c800dbb982bd4521a0d06267aec899'
NODES = list(range(0, 100609, 768))
SEEDS = list(range(5))
REPEATS = 8
CASES = 32
HORIZON = 24
sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Only independent new-root output is allowed')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows:
        raise ValueError('Empty result table')
    fields = fields or list(rows[0])
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def source_protection():
    seal = read(ROOT / 'protocol/execution_seal.json')
    assert seal['worker_sha256'] == WORKER_SHA and seal['plan_sha256'] == PLAN_SHA
    assert sha(ROOT / 'source/worker.py') == WORKER_SHA
    assert sha(ROOT / 'protocol/PLAN.json') == PLAN_SHA
    launch = read(ROOT / 'runtime/formal_launch.json')
    assert sha(ROOT / 'source/controller.py') == launch['controller_sha256']
    original = read(ROOT / 'protocol/protected_original_inputs.json')
    roles = collections.Counter()
    for record in original['files']:
        if sha(record['path']) != record['sha256']:
            raise AssertionError('Protected original changed: ' + record['path'])
        roles[record.get('role', 'unspecified')] += 1
    return dict(status='PASS', worker_sha256=WORKER_SHA, plan_sha256=PLAN_SHA,
                controller_sha256=launch['controller_sha256'],
                protected_original_files=len(original['files']), protected_original_roles=dict(roles),
                protection_manifest_sha256=sha(ROOT / 'protocol/protected_original_inputs.json'))


def training_exact(seed, job):
    old = FORMAL / 'jobs' / f'props__v5__seed{seed}' / 'training.jsonl'
    current = job / 'training.jsonl'
    count = 0
    with old.open() as old_stream, current.open() as new_stream:
        for count, pair in enumerate(itertools.zip_longest(old_stream, new_stream), start=1):
            if pair[0] is None or pair[1] is None or json.loads(pair[0]) != json.loads(pair[1]):
                raise AssertionError(f'Complete training record differs: seed {seed}, record {count}')
    if count != 8384:
        raise AssertionError('Incomplete native training episode/update log')
    return dict(records=count, parsed_objects_exact=True, current_sha256=sha(current), old_sha256=sha(old),
                file_bytes_exact=sha(current) == sha(old))


def old_reference(seed):
    path = FORMAL / 'jobs' / f'props__v5__seed{seed}' / 'evaluation_paths.csv.gz'
    values = {}
    with gzip.open(path, 'rt', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['phase'] == 'dev':
                key = int(row['training_step']), int(row['path_index'])
                if key in values:
                    raise AssertionError('Duplicate original DEV reference')
                values[key] = int(row['path_seed']), float(row['native_return'])
    assert len(values) == len(NODES) * CASES
    return values, dict(path=str(path), sha256=sha(path))


def acceptance_directory(seed):
    return ROOT / 'audit/postprocess_acceptance' / f'seed{seed}'


def completion_record(seed, job):
    accepted = acceptance_directory(seed) / 'accepted_completion.json'
    if accepted.is_file():
        record = read(accepted)
        if record['acceptance_status'] != 'ACCEPTED_AFTER_SOURCE_IDENTITY_RECONCILIATION' or record['scientific_status'] != 'COMPLETE':
            raise AssertionError('Independent scientific acceptance is not complete')
        assert record['seed'] == seed
        return record
    return read(job / 'completion.json')


def verify_completed_sources(job, *, reconciliation_override=None):
    before = read(job / 'source_binding_before.json')
    if not (job / 'source_binding_after.json').is_file():
        reconciliation = reconciliation_override or read(acceptance_directory(int(job.name.rsplit('seed', 1)[1])) / 'source_identity_reconciliation.json')
        assert reconciliation['status'] == 'PASS'
        assert reconciliation['before_manifest_sha256'] == sha(job / 'source_binding_before.json')
        for record in reconciliation['files']:
            assert before[record['logical_path']] == record['sha256']
            assert str(Path(record['logical_path']).resolve()) == record['canonical_path']
            assert sha(record['canonical_path']) == record['sha256']
        assert len(reconciliation['files']) == len(before) == 39
        return dict(before_count=len(before), after_count=len(reconciliation['files']), all_unchanged=True,
                    before_manifest_sha256=sha(job / 'source_binding_before.json'),
                    original_dynamic_after_manifest_absent=True,
                    accepted_by='independent canonical logical-to-physical identity reconciliation',
                    postrun_registry_failure=1)
    after = read(job / 'source_binding_after.json')
    assert all(after.get(path) == digest for path, digest in before.items())
    for path, digest in after.items():
        if sha(path) != digest:
            raise AssertionError('Imported source changed after run: ' + path)
    return dict(before_count=len(before), after_count=len(after), all_unchanged=True,
                before_manifest_sha256=sha(job / 'source_binding_before.json'),
                after_manifest_sha256=sha(job / 'source_binding_after.json'))


def recompute_seed(seed, common_plans, *, completion_override=None, reconciliation_override=None):
    job = ROOT / 'jobs' / f'props__v5__seed{seed}'
    completed = completion_override or completion_record(seed, job)
    independently_accepted = completed.get('acceptance_status') == 'ACCEPTED_AFTER_SOURCE_IDENTITY_RECONCILIATION'
    if independently_accepted:
        complete = completed['scientific_status'] == 'COMPLETE'
    else:
        complete = completed['success'] and completed['status'] == 'completed'
    if not complete or completed['engineering_only']:
        raise AssertionError('Formal worker did not naturally complete')
    assert completed['seed'] == seed and completed['fixed_endpoint_steps'] == 100608
    assert completed['evaluation_grid'] == NODES and completed['dev_cases'] == CASES
    assert completed['stochastic_replicates'] == REPEATS and completed['temperature'] == 1.
    assert completed['training_from_scratch'] and completed['independent_TEST_steps'] == 0
    assert completed['one_fixed_endpoint'] and not completed['periodic_models'] and not completed['optimizer_or_replay_saved']
    assert completed['training_device'] == ('cuda:0' if seed == 2 else 'cpu')
    assert completed['stochastic_device'] == 'cpu'
    assert completed['plan']['sha256'] == PLAN_SHA
    assert sha(job / 'checkpoint.pt') == completed['model_sha256']
    reproduction = read(job / 'reproduction_comparison.json')
    assert reproduction['reproduction_pass'] and reproduction['target_state_all_tensors_exact']
    assert reproduction['original_mean_compared_nodes'] == len(NODES)
    assert reproduction['original_mean_max_absolute_error'] == reproduction['target_state_max_absolute_difference'] == 0.
    assert completed['reproduction_comparison'] == reproduction
    reference, reference_identity = old_reference(seed)
    values = np.empty((len(NODES), REPEATS + 1, CASES), dtype=np.float64)
    seen = np.zeros(values.shape, dtype=bool)
    count = 0
    with gzip.open(job / 'evaluation_paths.csv.gz', 'rt', newline='') as stream:
        for row in csv.DictReader(stream):
            step, case = int(row['training_step']), int(row['case_index'])
            if step not in NODES or case not in range(CASES):
                raise AssertionError('Unexpected DEV node/case')
            index = NODES.index(step)
            mode, repeat = row['action_mode'], int(row['replicate'])
            if mode == 'mean':
                if repeat != -1 or int(row['action_seed']) != 4000000:
                    raise AssertionError('Unexpected deterministic evaluation')
                panel = 0
            elif mode == 'stochastic':
                if repeat not in range(REPEATS) or int(row['action_seed']) != 91090000 + 1000 * seed + repeat:
                    raise AssertionError('Unexpected stochastic replicate/stream')
                panel = repeat + 1
            else:
                raise AssertionError('Unknown evaluation mode')
            if seen[index, panel, case]:
                raise AssertionError('Duplicate per-case evaluation')
            value = float(row['native_return'])
            if not math.isfinite(value) or int(row['steps']) != HORIZON or abs(float(row['resource_fraction']) - 1.) > 1e-10:
                raise AssertionError('Case scalar/physical account invalid')
            if float(row['score_x100']) != value * 100.:
                raise AssertionError('Native score conversion differs')
            path_identity = int(row['path_seed']), row['plan_sha256']
            if case in common_plans and common_plans[case] != path_identity:
                raise AssertionError('DEV32 plan identity changes across nodes/replicates/seeds')
            common_plans[case] = path_identity
            if mode == 'mean':
                if reference[(step, case)] != (path_identity[0], value):
                    raise AssertionError(f'Original mean case differs: seed {seed}, step {step}, case {case}')
                if float(row['reference_native_return']) != value or float(row['absolute_reference_error']) != 0.:
                    raise AssertionError('Stored exact-reference comparison differs')
            elif row['reference_native_return'] or row['absolute_reference_error']:
                raise AssertionError('Stochastic case incorrectly tagged as old mean reference')
            values[index, panel, case] = value
            seen[index, panel, case] = True
            count += 1
    assert bool(seen.all()) and count == len(NODES) * (REPEATS + 1) * CASES
    panel_means = values.mean(axis=2)
    panel_sds = values.std(axis=2, ddof=1)
    panel_seen = set()
    with gzip.open(job / 'evaluation_panels.csv.gz', 'rt', newline='') as stream:
        for row in csv.DictReader(stream):
            index = NODES.index(int(row['training_step']))
            mode, repeat, action_seed = row['action_mode'], int(row['replicate']), int(row['action_seed'])
            if mode == 'mean':
                if repeat != -1 or action_seed != 4000000:
                    raise AssertionError('Panel has invalid mean execution identity')
                panel = 0
            elif mode == 'stochastic':
                if repeat not in range(REPEATS) or action_seed != 91090000 + 1000 * seed + repeat:
                    raise AssertionError('Panel has invalid stochastic execution identity')
                panel = repeat + 1
            else:
                raise AssertionError('Panel has unknown action mode')
            if (index, panel) in panel_seen:
                raise AssertionError('Duplicate panel summary')
            assert int(row['cases']) == CASES and int(row['successful_physical_steps']) == CASES * HORIZON
            if float(row['native_mean']) != panel_means[index, panel] or float(row['native_sample_sd']) != panel_sds[index, panel]:
                raise AssertionError('Stored panel summary does not match independently recomputed cases')
            assert (row['action_rng_before_sha256'] == row['action_rng_after_sha256']) == (panel == 0)
            panel_seen.add((index, panel))
    assert len(panel_seen) == len(NODES) * (REPEATS + 1)
    nodes = [json.loads(line) for line in (job / 'stochastic_nodes.jsonl').read_text().splitlines()]
    assert [row['train_env_steps'] for row in nodes] == NODES
    maximum_node_mean_error = 0.
    for index, row in enumerate(nodes):
        assert row['training_rng_restored'] and row['training_target_unmodified'] and row['cpu_mirror_no_shared_storage']
        assert row['source_target_state_sha256'] == row['cpu_mirror_state_sha256']
        assert row['original_mean_reference_max_absolute_error'] == 0.
        assert row['mean_native_mean'] == panel_means[index, 0]
        assert row['stochastic_replicate_native_means'] == panel_means[index, 1:].tolist()
        error = abs(float(row['stochastic_native_mean']) - float(np.mean(panel_means[index, 1:])))
        if error > 2e-15:
            raise AssertionError('Worker stochastic node mean differs from hierarchical recomputation')
        maximum_node_mean_error = max(maximum_node_mean_error, error)
    # Complete native training records are compared without dropping any field.
    train_check = training_exact(seed, job)
    source_check = verify_completed_sources(job, reconciliation_override=reconciliation_override)
    completion_path = acceptance_directory(seed) / 'accepted_completion.json' if independently_accepted else job / 'completion.json'
    metadata = dict(seed=seed, case_records=count, panels=len(panel_seen), mean_original_cases_exact=len(NODES) * CASES,
                    training_check=train_check, source_check=source_check, reproduction=reproduction,
                    model_sha256=completed['model_sha256'], model_deserialization_in_aggregation=0,
                    maximum_hierarchical_vs_worker_flat_mean_roundoff=maximum_node_mean_error,
                    wall_seconds=completed['wall_seconds'], actual_physical_counts=completed['actual_physical_counts'],
                    model_deserializations_during_worker=completed['model_deserializations'],
                    postrun_registry_failure=completed.get('postrun_registry_failure', 0),
                    independent_acceptance=independently_accepted,
                    reference=reference_identity,
                    completion_evidence_path=str(completion_path),
                    files={name: sha(job / name) for name in ('evaluation_paths.csv.gz',
                        'evaluation_panels.csv.gz', 'stochastic_nodes.jsonl', 'training.jsonl', 'reproduction_comparison.json')})
    return panel_means, metadata


def aggregate_counts(formal_records):
    phase_keys = ('train', 'mean', 'stochastic')
    count_keys = ('attempted_physical_steps', 'successful_physical_steps',
                  'completed_physical_episodes', 'guarded_complete_episodes')
    def total(records):
        return {phase: {key: sum(record['actual_physical_counts'][phase][key] for record in records)
                        for key in count_keys} for phase in phase_keys}
    formal = total(formal_records)
    expected = dict(train=503040, mean=506880, stochastic=4055040)
    for phase, wanted in expected.items():
        assert formal[phase]['attempted_physical_steps'] == formal[phase]['successful_physical_steps'] == wanted
        assert formal[phase]['completed_physical_episodes'] == formal[phase]['guarded_complete_episodes'] == wanted // HORIZON
    engineering_files = sorted((ROOT / 'engineering').glob('*/completion.json'))
    assert len(engineering_files) == 4
    engineering_records = [read(path) for path in engineering_files]
    assert all(record['success'] and record['engineering_only'] for record in engineering_records)
    engineering = total(engineering_records)
    for phase, wanted in dict(train=1536, mean=1152, stochastic=4608).items():
        assert engineering[phase]['attempted_physical_steps'] == engineering[phase]['successful_physical_steps'] == wanted
        assert engineering[phase]['completed_physical_episodes'] == engineering[phase]['guarded_complete_episodes'] == wanted // HORIZON
    failures = []
    source_registry_failures = []
    for base in (ROOT / 'jobs', ROOT / 'engineering'):
        for path in sorted(base.rglob('failed.json')):
            failed = read(path)
            if base == ROOT / 'jobs':
                seed = int(path.parent.name.rsplit('seed', 1)[1])
                acceptance_path = acceptance_directory(seed) / 'accepted_completion.json'
                if acceptance_path.is_file():
                    acceptance = read(acceptance_path)
                    assert acceptance['original_failed_sha256'] == sha(path)
                    assert acceptance['postrun_registry_failure'] == 1
                    source_registry_failures.append(dict(path=str(path), sha256=sha(path),
                        acceptance_path=str(acceptance_path), acceptance_sha256=sha(acceptance_path),
                        already_counted_in_formal_scientific_cost=True))
                    continue
            if (path.parent / 'completion.json').exists():
                raise AssertionError('Attempt has both failure and completion; costs require explicit audit')
            if 'physical_counts' not in failed:
                raise AssertionError('Failure missing exact partial physical counts')
            failures.append(dict(path=str(path), sha256=sha(path), actual_physical_counts=failed['physical_counts']))
    partial = total(failures)
    all_counts = {phase: {key: formal[phase][key] + engineering[phase][key] + partial[phase][key]
                         for key in count_keys} for phase in phase_keys}
    successful = sum(row['successful_physical_steps'] for row in all_counts.values())
    assert successful == 5072256 + sum(row['successful_physical_steps'] for row in partial.values())
    deserializations = collections.Counter()
    for record in formal_records:
        deserializations.update(record['model_deserializations_during_worker'])
    for record in engineering_records:
        deserializations.update(record['model_deserializations'])
    engineering_gate_path = ROOT / 'audit/engineering_pair_gate.json'
    engineering_gate = read(engineering_gate_path)
    assert engineering_gate['status'] == 'PASS'
    extra_audit_loads = engineering_gate['additional_model_deserializations_for_pair_audit']
    assert extra_audit_loads == 4
    worker_loads = sum(deserializations.values())
    assert worker_loads == 23 and deserializations['old_fixed_endpoint_comparison'] == 5
    return dict(schema='props-stochastic-curve-computational-ledger-v1',
                formal=formal, engineering=engineering, preserved_failed_partial=partial, combined=all_counts,
                expected_formal_successful_physical_steps=5064960, actual_formal_successful_physical_steps=sum(row['successful_physical_steps'] for row in formal.values()),
                expected_engineering_successful_physical_steps=7296, actual_engineering_successful_physical_steps=sum(row['successful_physical_steps'] for row in engineering.values()),
                actual_successful_physical_steps=successful,
                actual_attempted_physical_steps=sum(row['attempted_physical_steps'] for row in all_counts.values()),
                actual_complete_physical_episodes=sum(row['completed_physical_episodes'] for row in all_counts.values()),
                independent_TEST_steps=0, failures=failures, failed_attempt_count=len(failures),
                source_registry_postrun_failures=len(source_registry_failures),
                source_registry_postrun_failure_history=source_registry_failures,
                physical_execution_failures=len(failures), scientific_retries=0,
                automatic_retry=False, model_deserializations_during_workers=dict(deserializations),
                model_deserializations_workers_total=worker_loads,
                model_deserializations_formal_workers=15, model_deserializations_engineering_workers=8,
                additional_engineering_pair_audit_model_deserializations=extra_audit_loads,
                model_deserializations_total=worker_loads + extra_audit_loads,
                old_model_deserializations=deserializations['old_fixed_endpoint_comparison'],
                old_models_loaded_only_after_new_fixed_training_endpoint=True,
                engineering_pair_gate_sha256=sha(engineering_gate_path),
                aggregation_model_deserializations=0, aggregation_environment_steps=0,
                engineering_sources={str(path): sha(path) for path in engineering_files},
                full_compressed_episode_ledger_check='performed by independent audit, not claimed by this scalar aggregation')


def point_rows(random_values):
    result = []
    for index, step in enumerate(NODES):
        start = max(0, index - 4)
        smooth = random_values[:, start:index + 1].mean(axis=1)
        raw = random_values[:, index]
        mean, sd = float(smooth.mean()) * 100., float(smooth.std(ddof=1)) * 100.
        result.append(dict(domain='v5', method='props', training_step=step, mean_x100=mean, sample_sd_x100=sd,
            seed_count=5, raw_mean_x100=float(raw.mean()) * 100., raw_sample_sd_x100=float(raw.std(ddof=1)) * 100.,
            smoothed_mean_x100=mean, smoothed_sample_sd_x100=sd, smoothing='causal_trailing_per_seed',
            window_evaluation_points=5, effective_window_points=index - start + 1,
            window_start_training_step=NODES[start], window_end_training_step=step, window_full_span_interactions=3072))
    return result


def compatibility_tables(out, random_values, points):
    with (ORIGINAL_FIGURE / 'data.csv').open(newline='') as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
    changed = 0
    for row in rows:
        if row['domain'] == 'v5' and row['method'] == 'props':
            value = float(random_values[int(row['seed']), NODES.index(int(row['training_step']))])
            row.update(mean_return=value, score_x100=value * 100., development_cases=32,
                       scientific_origin='fresh_fixed100608_native_stochastic8_DEV32', execution_required_this_cohort=True)
            changed += 1
    assert changed == len(SEEDS) * len(NODES)
    write_csv(out / 'figure44_data.csv', rows, fields)
    with (ORIGINAL_FIGURE / 'plotted_points.csv').open(newline='') as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; original = list(reader)
    replacement = {int(row['training_step']): row for row in points}
    changed = 0; merged = []
    for row in original:
        if row['domain'] == 'v5' and row['method'] == 'props':
            row = replacement[int(row['training_step'])]; changed += 1
        merged.append(row)
    assert changed == len(NODES)
    write_csv(out / 'figure44_plotted_points.csv', merged, fields)
    return dict(original_data_sha256=sha(ORIGINAL_FIGURE / 'data.csv'),
                original_plotted_points_sha256=sha(ORIGINAL_FIGURE / 'plotted_points.csv'),
                original_row_count=len(rows), replaced_raw_rows=len(SEEDS) * len(NODES),
                original_plotted_row_count=len(original), replaced_plotted_rows=len(NODES),
                unchanged_non_target_row_values=True,
                scope='Only v5 PROPS rows replaced; all radio and all other v5 methods preserve original cell values/order',
                execution='v5 PROPS rows are random native Gaussian; all other rows inherit original deterministic evaluation')


def main(args):
    started = time.perf_counter()
    # No output is created until all five naturally completed runs and their
    # controller are present. A currently running cohort cannot be aggregated.
    status = read(ROOT / 'runtime/status.json')
    acceptance_path = ROOT / 'runtime/scientific_acceptance.json'
    if status['running'] != 0 or not acceptance_path.is_file():
        raise RuntimeError('All five fixed scientific runs and independent completion acceptance must finish first')
    acceptance_index = read(acceptance_path)
    if acceptance_index['status'] != 'ALL_FIVE_SCIENTIFIC_RUNS_ACCEPTED' or acceptance_index['accepted_seeds'] != SEEDS:
        raise RuntimeError('Independent acceptance index is incomplete')
    assert acceptance_index['original_controller_status'] == status
    for record in acceptance_index['receipts']:
        assert sha(record['path']) == record['sha256']
    for seed in SEEDS:
        completion_record(seed, ROOT / 'jobs' / f'props__v5__seed{seed}')
    out = ROOT / 'results'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Aggregation results already exist; inspect instead of overwriting')
    source_check = source_protection()
    plan = read(ROOT / 'protocol/PLAN.json')
    assert plan['evaluation']['nodes'] == NODES and plan['training']['seeds'] == SEEDS
    assert read(ROOT / 'audit/engineering_pair_gate.json')['status'] == 'PASS'
    common_plans = {}
    arrays, metadata = [], []
    for seed in SEEDS:
        values, record = recompute_seed(seed, common_plans)
        arrays.append(values); metadata.append(record)
    array = np.asarray(arrays, dtype=np.float64)  # seed, node, mean+8random
    assert array.shape == (5, 132, 9) and len(common_plans) == CASES
    mean_values = array[:, :, 0]
    # Repeated action draws are averaged inside each training seed first.
    random_values = array[:, :, 1:].mean(axis=2)
    points = point_rows(random_values)
    by_seed, repeats = [], []
    for seed in SEEDS:
        for index, step in enumerate(NODES):
            start = max(0, index - 4)
            by_seed.append(dict(domain='v5', method='props', seed=seed, training_step=step,
                mean_action_native_return=float(mean_values[seed, index]), mean_action_score_x100=float(mean_values[seed, index]) * 100.,
                stochastic_native_return=float(random_values[seed, index]), stochastic_score_x100=float(random_values[seed, index]) * 100.,
                stochastic_replicate_sample_sd_x100=float(array[seed, index, 1:].std(ddof=1)) * 100.,
                smoothed_stochastic_score_x100=float(random_values[seed, start:index + 1].mean()) * 100.,
                unique_DEV_paths=32, stochastic_replicates=8, effective_window_points=index - start + 1,
                original_mean_case_max_absolute_error=0.))
            for repeat in range(REPEATS):
                repeats.append(dict(seed=seed, training_step=step, replicate=repeat,
                    action_seed=91090000 + 1000 * seed + repeat, dev_paths=32,
                    native_mean=float(array[seed, index, repeat + 1]), score_x100=float(array[seed, index, repeat + 1]) * 100.))
    ledger = aggregate_counts(metadata)
    previous = read(PREVIOUS_DIAGNOSTIC)
    old_scores = previous['methods']['props']['stochastic']['per_training_seed_scores_x100']
    endpoints = []
    for seed in SEEDS:
        score = float(random_values[seed, -1]) * 100.
        endpoints.append(dict(seed=seed, training_device='cuda:0' if seed == 2 else 'cpu',
            mean_action_score_x100=float(mean_values[seed, -1]) * 100., stochastic_score_x100=score,
            stochastic_replicate_sample_sd_x100=float(array[seed, -1, 1:].std(ddof=1)) * 100.,
            stochastic_repeat_scores_x100=(array[seed, -1, 1:] * 100.).tolist(),
            trailing5_stochastic_score_x100=float(random_values[seed, -5:].mean()) * 100.,
            previous_CPU_stochastic_endpoint_score_x100=old_scores[seed],
            actual_minus_previous_score_points=score - old_scores[seed],
            target_and_complete_training_and_mean_cases_exact=True))
    endpoint_summary = dict(raw_stochastic_mean_score_x100=float(random_values[:, -1].mean()) * 100.,
        raw_stochastic_training_seed_sample_sd_x100=float(random_values[:, -1].std(ddof=1)) * 100.,
        trailing5_stochastic_mean_score_x100=points[-1]['mean_x100'],
        trailing5_stochastic_training_seed_sample_sd_x100=points[-1]['sample_sd_x100'],
        raw_mean_action_score_x100=float(mean_values[:, -1].mean()) * 100.,
        raw_mean_action_training_seed_sample_sd_x100=float(mean_values[:, -1].std(ddof=1)) * 100.,
        previous_CPU_stochastic_endpoint_mean_score_x100=previous['methods']['props']['stochastic']['mean_score_x100'],
        actual_minus_previous_score_points=float(random_values[:, -1].mean()) * 100. - previous['methods']['props']['stochastic']['mean_score_x100'],
        no_target_score_forced=True, fixed_endpoint_selected_before_execution=True)
    verification = dict(case_records=sum(row['case_records'] for row in metadata),
        evaluation_panels=sum(row['panels'] for row in metadata),
        original_mean_cases_exact=sum(row['mean_original_cases_exact'] for row in metadata),
        complete_native_training_records_exact=sum(row['training_check']['records'] for row in metadata),
        fixed_target_tensor_exact_seed_count=5, same_DEV32_plan_identity_all_nodes_replicates_seeds=True,
        all_660_nodes_training_RNG_and_target_unmodified=True,
        aggregation_model_loads=0, aggregation_environment_steps=0,
        maximum_hierarchical_vs_worker_flat_mean_roundoff=max(row['maximum_hierarchical_vs_worker_flat_mean_roundoff'] for row in metadata),
        source_protection=source_check)
    assert verification['case_records'] == 190080 and verification['evaluation_panels'] == 5940
    assert verification['original_mean_cases_exact'] == 21120 and verification['complete_native_training_records_exact'] == 41920
    out.mkdir(exist_ok=True)
    write_csv(out / 'curve_by_seed.csv', by_seed)
    write_csv(out / 'replicate_curve.csv', repeats)
    write_csv(out / 'plotted_random_curve.csv', points)
    write_csv(out / 'raw_random_curve.csv', [dict(training_step=row['training_step'], mean_x100=row['raw_mean_x100'],
        sample_sd_x100=row['raw_sample_sd_x100'], independent_training_seeds=5, action_replicates_within_seed=8,
        unique_DEV_paths=32) for row in points])
    compatibility = compatibility_tables(out, random_values, points)
    dump(out / 'computational_ledger.json', ledger)
    dump(out / 'verification.json', verification)
    dump(out / 'per_seed_endpoints.json', endpoints)
    summary = dict(schema='chapter04-props-full-stochastic-curve-results-v1', status='COMPLETE_ALL_FIXED_RESULTS',
        endpoint=endpoint_summary, per_seed_endpoints=endpoints, verification=verification,
        computational_ledger=ledger, compatibility_tables=compatibility,
        original_controller_status=status, independent_scientific_acceptance=acceptance_index,
        statistics='32 common DEV paths averaged within each action repeat, 8 repeats averaged within training seed, equal mean and sample SD across five training seeds. Display trailing5 is applied separately within seed before across-seed mean/SD.',
        execution_scope='Only PROPS v5 new curve is native random target execution at temperature1; all other original figure curves inherit deterministic execution. Cross-mode gaps do not establish an intrinsic algorithm advantage.',
        old_learning_and_independent_TEST_tables_unchanged=True,
        original_test_reused_for_new_selection=False, independent_TEST_steps=0,
        previous_endpoint_diagnostic=dict(path=str(PREVIOUS_DIAGNOSTIC), sha256=sha(PREVIOUS_DIAGNOSTIC),
            role='descriptive endpoint comparison only; no endpoint target, tuning or result selection'),
        fixed_run_details=metadata, aggregation_source_sha256=sha(__file__),
        aggregation_wall_seconds=time.perf_counter() - started)
    dump(out / 'summary.json', summary)
    report = [
        'PROPS 完整随机评价学习曲线重测结果',
        '=' * 44, '',
        f'本轮五个训练种子均从零运行至固定100608步，保留132个评价节点。原生高斯策略不缩放标准差，每节点评价32个共同DEV情景和8个固定动作随机流。',
        '每个训练种子先对8轮评价取平均，再计算五种子的均值和样本标准差。8轮动作随机评价不作为额外独立训练种子。', '',
        f'固定终点的随机执行原始结果为 {endpoint_summary["raw_stochastic_mean_score_x100"]:.8f}% ± {endpoint_summary["raw_stochastic_training_seed_sample_sd_x100"]:.8f}%。',
        f'图中相同五点尾窗的终点为 {endpoint_summary["trailing5_stochastic_mean_score_x100"]:.8f}% ± {endpoint_summary["trailing5_stochastic_training_seed_sample_sd_x100"]:.8f}%。原始终点与尾窗显示分别保留。',
        f'原均值动作评价的完整132×32×5条case与原训练记录及五个最终target张量精确复现。原均值动作原始终点为 {endpoint_summary["raw_mean_action_score_x100"]:.8f}%。', '',
        '五个固定训练种子的结果：', '']
    for row in endpoints:
        report.append(f'种子{row["seed"]}：均值动作 {row["mean_action_score_x100"]:.8f}%，随机动作 {row["stochastic_score_x100"]:.8f}%，五点尾窗 {row["trailing5_stochastic_score_x100"]:.8f}%。')
    report += ['', f'此前CPU随机终点诊断为 {endpoint_summary["previous_CPU_stochastic_endpoint_mean_score_x100"]:.8f}%，本轮与其差值 {endpoint_summary["actual_minus_previous_score_points"]:+.12f} 个百分点。此处仅比较实际结果，没有为达到76.65%调参或选择种子。', '',
        f'逐case复算190080条评价记录和5940个面板，原均值动作21120条case及41920条完整训练日志对象精确一致。模型比较读取已完成worker的比较证据，聚合阶段未加载模型或运行环境。',
        f'正式物理步 {ledger["actual_formal_successful_physical_steps"]}，工程物理步 {ledger["actual_engineering_successful_physical_steps"]}，另计失败部分后总成功物理步 {ledger["actual_successful_physical_steps"]}。新独立TEST为0步。',
        f'保留 {ledger["source_registry_postrun_failures"]} 次训练完成后的来源路径键审计失败历史。旧存储目录在运行中迁移为符号链接，独立接受器按逻辑路径与新物理路径的SHA一致性修复审计身份，原failed记录和controller失败状态均保留，科学训练重跑为0次。',
        '完整24期压缩物理台账由独立审计检查，此聚合报告只声明case、日志、节点、来源与计数核验。', '',
        '供图4.4(b)使用的兼容CSV只替换v5 PROPS的660条原始行和132条显示行，其余方法及通信图数据保持。该PROPS曲线采用随机执行，其他曲线沿用原均值执行，跨执行方式的图上差距不单独证明算法收益。原学习与独立TEST表不改写。', '']
    (out / '完整随机曲线重测结果.rst').write_text('\n'.join(report))
    outputs = {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()}
    dump(out / 'aggregation_completion.json', dict(status='PASS', summary=str(out / 'summary.json'),
        output_sha256=outputs, aggregation_source_sha256=sha(__file__), model_deserializations=0,
        environment_steps=0, no_original_or_worker_or_figure_mutations=True))
    print(json.dumps(dict(status='PASS', endpoint=endpoint_summary, verification=verification,
                         total_physical_steps=ledger['actual_successful_physical_steps']), ensure_ascii=False))


if __name__ == '__main__':
    main(argparse.ArgumentParser(description=__doc__).parse_args())
