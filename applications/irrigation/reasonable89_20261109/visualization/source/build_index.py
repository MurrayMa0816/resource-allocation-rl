#!/usr/bin/env python3
"""Bind the NEW accepted result registry, without guessing phases from filenames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_adapter import ROOT, CASE_ID, Evidence, budget_class, dump, sha, normalized_design, initial_actor_identities


def bind(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('An earlier experiment cannot enter the new publication index')
    return dict(path=str(path), sha256=sha(path))


def read_binding(item):
    path = Path(item['path']).resolve()
    if not path.is_relative_to(ROOT) or sha(path) != item['sha256']:
        raise ValueError('Result registry does not match its frozen SHA')
    return path


def build(accepted_path, audit_path, destination):
    accepted_path, audit_path = Path(accepted_path).resolve(), Path(audit_path).resolve()
    accepted = json.loads(accepted_path.read_text()); audit = json.loads(audit_path.read_text())
    if accepted.get('status') != 'ACCEPTED_FOR_PUBLICATION' or audit.get('status') != 'PASS':
        raise ValueError('Formal results must first pass independent scientific acceptance')
    if accepted['case_id'] != CASE_ID or audit.get('initialization_identities_verified') is not True:
        raise ValueError('The new task and exactly reconstructed initial policies must be verified')
    if read_binding(accepted['independent_numeric_audit']) != audit_path:
        raise ValueError('Accepted manifest must SHA-bind the same independent final numeric audit')
    run = Path(accepted['run_root']).resolve()
    if not run.is_relative_to(ROOT):
        raise ValueError('Use only the new frozen run')
    registry_path = read_binding(accepted['plot_registry']); registry = json.loads(registry_path.read_text())
    design_path = read_binding(registry['experiment_design']); design = normalized_design(json.loads(design_path.read_text()))
    levels = design['formal']['budgets_m3']; records = []
    for entry in registry['evaluation_records']:
        folder = Path(entry['directory']).resolve()
        if not folder.is_relative_to(run):
            raise ValueError('A DEV, prior or foreign result was offered as formal evidence')
        for name, expected in entry['files_sha256'].items():
            if sha(folder / name) != expected:
                raise ValueError('Result changed after explicit registry was frozen: ' + str(folder / name))
        metadata = json.loads((folder / 'metadata.json').read_text())
        completion = json.loads((folder / 'complete.json').read_text())
        if (completion.get('formal') is not True or metadata.get('job', {}).get('formal') is not True
                or completion.get('status') != 'COMPLETE'):
            raise ValueError('formalFalse data are never used for formal publication figures')
        if metadata['job']['source_release'] != accepted['source_release']:
            raise ValueError('Adapter/source release is not the accepted one')
        identity_field = entry.get('actor_identity_metadata_field', 'executed_actor_sha256')
        identity = metadata[identity_field]
        item = dict(id=metadata['job']['job_id'], role=entry['role'],
            objective_order=['C90', 'JA'], training_seed=int(metadata['trained_seed']),
            trained_budget_m3=budget_class(metadata['model_training_budget_m3'], levels),
            budget_m3=budget_class(metadata['evaluation_budget_m3'], levels),
            raw_model_training_budget_m3=float(metadata['model_training_budget_m3']),
            raw_evaluation_budget_m3=float(metadata['evaluation_budget_m3']),
            actor_identity_sha256=identity, actor_identity_metadata_field=identity_field,
            files=dict(results=bind(folder / 'results.npz'), metadata=bind(folder / 'metadata.json'),
                       completion=bind(folder / 'complete.json'),
                       model_protocol=bind(read_binding(entry['model_protocol']))))
        if item['role'] == 'initial':
            receipt_path = read_binding(entry['initialization_identity_receipt'])
            proof = json.loads(receipt_path.read_text())
            if proof['source_release'] != accepted['source_release']:
                raise ValueError('Missing independent initial-actor reconstruction proof')
            expected = initial_actor_identities(proof)[str(item['training_seed'])]
            if identity != expected:
                raise ValueError('Initial path must bind the actually evaluated initial actor hash, not the final checkpoint container')
            item.update(initialization_verified=True, initial_actor_sha256=expected)
            item['files']['initialization_identity'] = bind(receipt_path)
        if item['role'] in ('feedback', 'representative_feedback', 'descriptive_feedback'):
            item['et0_multiplier'] = float(metadata['job']['et0_multiplier'])
            item['execution_mode'] = {'feedback': 'new_observation_feedback',
                                      'replay': 'original_actions_replay'}[metadata['job']['controller_mode']]
            if item['role'] in ('representative_feedback', 'descriptive_feedback'):
                if (metadata.get('kind') != 'descriptive_feedback' or
                        entry.get('excluded_from_feedback_192_pair_summary') is not True or
                        int(metadata.get('paths', 0)) != 1):
                    raise ValueError('Only explicitly registered native single-path descriptive extras are excluded from192 pairs')
                item['excluded_from_feedback_192_pair_summary'] = True
        if item['role'] == 'historical_final':
            item['year'] = int(metadata['job']['env_config']['historical_year'])
        records.append(item)
    # These file bindings are supplied explicitly by the evaluator; no model
    # checkpoint is loaded and no candidate DEV is adopted through a filename.
    learning = []
    for entry in registry['main_development_records']:
        files = {key: bind(read_binding(value)) for key, value in entry['files'].items()}
        learning.append(dict(training_seed=int(entry['training_seed']), formal_training=True, files=files,
                             completion_actor_identity_field=entry.get('completion_actor_identity_field', 'endpoint_actor_sha256')))
    index = dict(schema='reasonable89-publication-index-v1', accepted_scientific_run=bind(accepted_path),
        independent_numeric_audit=bind(audit_path), frozen_plot_registry=bind(registry_path),
        experiment_design=bind(design_path), fixed_inputs=registry['fixed_inputs'],
        records=records, main_development=learning, feedback_design=registry['feedback_design'],
        shared_execution_operator_source=registry['shared_execution_operator_source'])
    for key in ('selected_application', 'TEST_namespace_amendment', 'sealed_TEST_panel', 'evaluation_capsule'):
        index[key] = registry[key]
    for key in ('sealed_TEST_panel_metadata', 'TEST_panel_seed_array'):
        if key in registry:
            index[key] = registry[key]
    for key in ('representative_selection', 'dispatch_completion', 'TEST_release', 'training_parameter_identity_audit'):
        if key in registry:
            index[key] = registry[key]
    if 'execution_points' in registry:
        index['execution_points'] = registry['execution_points']
    # A temporary index is validated before the stable input index is published.
    destination = Path(destination); temporary = destination.with_name(destination.name + '.validation.json')
    dump(temporary, index)
    Evidence(temporary)
    dump(destination, index)
    temporary.unlink()
    print(json.dumps(dict(status='NEW_FORMAL_INDEX_READY_NOT_RENDERED', records=len(records),
                         index=str(destination)), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accepted-run', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'figures/new_plot_index.json')
    args = parser.parse_args(); build(args.accepted_run, args.audit, args.output)


if __name__ == '__main__':
    main()
