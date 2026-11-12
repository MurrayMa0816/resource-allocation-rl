#!/usr/bin/env python3
"""Replace only the measured stochastic PROPS artists in original panel (b).

The adjacent original data and original plotted points remain historical inputs.
The new random inputs are admitted only after all five fixed runs and the complete
aggregation have finished. This module never imports a policy or environment.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PUBLICATION = ROOT / 'publication'
PARENT = Path('/hy-tmp/resource_allocation_results/chapter04/figures/readability_refinement_20261108_234201/assets/figure4-4')
NODES = list(range(0, 100609, 768))
SEEDS = list(range(5))
ANNOTATION = None
LOCAL_INPUTS = {
    'curve_by_seed.csv': 'random_curve_by_seed.csv',
    'plotted_random_curve.csv': 'random_plotted_points.csv',
    'summary.json': 'random_aggregation_summary.json',
    'aggregation_completion.json': 'random_aggregation_completion.json',
}
AUDIT_INPUTS = {
    'runtime/scientific_acceptance.json': 'audit_scientific_acceptance.json',
    'audit/independent_formal_numeric_audit_posthoc.json': 'audit_independent_full_numeric.json',
    'audit/aggregation_outputs_independent_review.json': 'audit_aggregation_crosscheck.json',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rows(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    fields = list(dict.fromkeys(key for record in records for key in record))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def check_parent_binding():
    binding = read(HERE / 'parent_binding.json')
    copies = {
        'data.csv': 'data.csv', 'plotted_points.csv': 'plotted_points.csv',
        'figure_config.json': 'figure_config.json', 'presentation_style.py': 'presentation_style.py',
        'render_geometry.json': 'original_render_geometry.json',
        'presentation_geometry.json': 'original_presentation_geometry.json',
    }
    for original, local in copies.items():
        if sha(HERE / local) != binding['parent_files'][original]:
            raise ValueError('Historical parent copy changed: ' + local)
    source = (HERE / 'parent_plot.py').read_text(encoding='utf-8')
    adaptation = binding['source_adaptation']['only_replacement']
    if source.count(adaptation['new']) != 1:
        raise ValueError('Parent cache root must have one narrow adaptation')
    original = source.replace(adaptation['new'], adaptation['old'])
    if hashlib.sha256(original.encode()).hexdigest() != binding['parent_files']['plot.py']:
        raise ValueError('Parent plotting code changed outside private cache/output root')
    return binding


def admit_complete_random_inputs():
    """Stage real complete CSV once; later reproduction uses adjacent saved CSV."""
    binding_path = HERE / 'random_input_binding.json'
    if binding_path.is_file():
        binding = read(binding_path)
        if binding['status'] != 'PASS_COMPLETE_FIVE_FIXED_RUNS_POSTHOC_ACCEPTED':
            raise ValueError('Random input binding is not complete')
        for filename, expected in binding['local_sha256'].items():
            if sha(HERE / filename) != expected:
                raise ValueError('Saved random input changed: ' + filename)
        return binding
    if any((HERE / leaf).exists() for leaf in list(LOCAL_INPUTS.values()) + list(AUDIT_INPUTS.values())):
        raise ValueError('Unbound random files already exist; inspect rather than overwrite')
    status = read(ROOT / 'runtime/status.json')
    # Original workers completed every scientific operation, then failed their
    # dynamic source-path-key guard after a concurrent legacy store migration.
    # Preserve that failure; independently accepted receipts are a separate route
    # with full closed data/source reconciliation, never a rewritten worker pass.
    acceptance_path = ROOT / 'runtime/scientific_acceptance.json'
    acceptance = read(acceptance_path)
    if (acceptance['status'] != 'ALL_FIVE_SCIENTIFIC_RUNS_ACCEPTED'
            or acceptance['accepted_seeds'] != SEEDS or acceptance['pending'] != []
            or acceptance['original_failed_records_retained'] is not True
            or acceptance['original_controller_status_unchanged'] is not True
            or acceptance['scientific_retries'] != 0
            or acceptance['postrun_source_registry_failures'] != 5
            or acceptance['protected_original_source_check']['status'] != 'PASS'
            or acceptance['protected_original_source_check']['protected_original_files'] != 192
            or acceptance['original_controller_status'] != status or status['running'] != 0):
        raise RuntimeError('All five scientific runs require explicit transparent posthoc acceptance')
    independent_path = ROOT / 'audit/independent_formal_numeric_audit_posthoc.json'
    independent = read(independent_path)
    if (independent['status'] != 'PASS_POSTHOC_FULL_DATA_ACCEPTED_WITH_WORKER_FAILURES_PRESERVED'
            or independent['job_count'] != 5 or independent['case_records'] != 190080
            or independent['panel_records'] != 5940 or independent['node_records'] != 660
            or independent['physical_episode_count'] != 211040
            or independent['actual_total_physical_steps'] != 5064960
            or independent['original_protected_hashes_unchanged'] != 192
            or independent['source_and_training_unchanged'] is not True
            or independent['original_worker_failure_count'] != 5
            or independent['original_failed_receipts_and_controller_status_preserved'] is not True):
        raise RuntimeError('Independent complete physical/numeric posthoc audit must pass')
    cross_path = ROOT / 'audit/aggregation_outputs_independent_review.json'
    cross = read(cross_path)
    if (cross['status'] != 'PASS'
            or cross['independent_numeric_audit_sha256'] != sha(independent_path)
            or cross['aggregate_summary_sha256'] != sha(ROOT / 'results/summary.json')
            or cross['checked_counts'] != {'seed_nodes': 660, 'replicate_nodes': 5280,
                                           'raw_curve_nodes': 132, 'plotted_curve_nodes': 132}):
        raise RuntimeError('Independent aggregation-output crosscheck must pass')
    plan_sha = sha(ROOT / 'protocol/PLAN.json')
    completion_paths = []
    receipts = acceptance['receipts']
    if len(receipts) != 5 or [item['seed'] for item in receipts] != SEEDS:
        raise ValueError('Exactly five fixed accepted receipts are required')
    for seed in SEEDS:
        path = ROOT / 'audit/postprocess_acceptance' / f'seed{seed}' / 'accepted_completion.json'
        if receipts[seed]['path'] != str(path) or receipts[seed]['sha256'] != sha(path):
            raise ValueError('Acceptance receipt identity changed')
        complete = read(path)
        if not (complete['acceptance_status'] == 'ACCEPTED_AFTER_SOURCE_IDENTITY_RECONCILIATION'
                and complete['scientific_status'] == 'COMPLETE'
                and complete['original_worker_failure_retained'] is True
                and complete['engineering_only'] is False and complete['seed'] == seed
                and complete['fixed_endpoint_steps'] == 100608 and complete['evaluation_grid'] == NODES
                and complete['dev_cases'] == 32 and complete['stochastic_replicates'] == 8
                and complete['temperature'] == 1 and complete['plan']['sha256'] == plan_sha
                and complete['reproduction_comparison']['reproduction_pass'] is True
                and complete['reproduction_comparison']['target_state_all_tensors_exact'] is True
                and complete['actual_total_physical_steps'] == 1012992
                and complete['physical_execution_failures'] == 0 and complete['scientific_retries'] == 0):
            raise ValueError('Invalid full fixed completion at seed ' + str(seed))
        reconciliation = path.parent / 'source_identity_reconciliation.json'
        verification_path = path.parent / 'scientific_verification.json'
        if (sha(reconciliation) != complete['source_identity_reconciliation_sha256']
                or read(reconciliation)['status'] != 'PASS'
                or sha(verification_path) != complete['scientific_verification_sha256']
                or sha(complete['original_failed_path']) != complete['original_failed_sha256']):
            raise ValueError('Source reconciliation/verification/failure evidence changed')
        completion_paths.append({'path': str(path), 'sha256': sha(path),
                                 'original_worker_status': complete['original_worker_process_status'],
                                 'original_failed_path': complete['original_failed_path'],
                                 'original_failed_sha256': complete['original_failed_sha256'],
                                 'source_identity_reconciliation_sha256': sha(reconciliation),
                                 'scientific_verification_sha256': sha(verification_path)})
    out = ROOT / 'results'
    summary = read(out / 'summary.json')
    aggregate = read(out / 'aggregation_completion.json')
    if summary['status'] != 'COMPLETE_ALL_FIXED_RESULTS' or aggregate['status'] != 'PASS':
        raise ValueError('Complete real aggregation is required')
    verification = summary['verification']
    if not (verification['case_records'] == 190080 and verification['evaluation_panels'] == 5940
            and verification['original_mean_cases_exact'] == 21120
            and verification['fixed_target_tensor_exact_seed_count'] == 5):
        raise ValueError('Full case/reproduction aggregation gate failed')
    for source in LOCAL_INPUTS:
        if source != 'aggregation_completion.json' and sha(out / source) != aggregate['output_sha256'][source]:
            raise ValueError('Aggregation output changed: ' + source)
    for source, destination in LOCAL_INPUTS.items():
        shutil.copyfile(out / source, HERE / destination)
    for source, destination in AUDIT_INPUTS.items():
        shutil.copyfile(ROOT / source, HERE / destination)
    binding = {
        'status': 'PASS_COMPLETE_FIVE_FIXED_RUNS_POSTHOC_ACCEPTED',
        'plan_sha256': plan_sha,
        'fixed_completions': completion_paths,
        'admission_route': 'explicit posthoc complete scientific acceptance with five original terminal source-path-key failures preserved',
        'original_worker_process_status': 'failed',
        'original_failed_receipts_and_controller_status_preserved': True,
        'scientific_retries': 0,
        'independent_full_audit_status': independent['status'],
        'independent_full_audit_sha256': sha(independent_path),
        'independent_aggregation_crosscheck_status': cross['status'],
        'independent_aggregation_crosscheck_sha256': sha(cross_path),
        'aggregation_completion': str(out / 'aggregation_completion.json'),
        'aggregation_completion_sha256': sha(out / 'aggregation_completion.json'),
        'local_sha256': {destination: sha(HERE / destination)
                         for destination in list(LOCAL_INPUTS.values()) + list(AUDIT_INPUTS.values())},
        'model_loads_during_staging': 0, 'environment_steps_during_staging': 0,
        'source_data': '32 DEV cases x 8 action repeats averaged inside each of 5 training seeds at each of 132 real nodes',
    }
    dump(binding_path, binding)
    return binding


def load_parent():
    # The sole root adaptation in this saved parent routes its import-time cache
    # creation to this new publication, never to the historical parent cache.
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location('figure44_saved_parent', HERE / 'parent_plot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parent(module):
    config = read(HERE / 'figure_config.json')
    module.setup()
    fig, points = module.learning(rows(HERE / 'data.csv'), config)
    fig.set_dpi(200)
    module.finalize_presentation(fig, config)
    # Force original autoscale before any replacement. New y-data must never
    # change the axes limits shared with the historical comparison curves.
    fig.canvas.draw()
    return fig, points, config


def artist_styles(fig):
    def line_style(line):
        return {key: getattr(line, 'get_' + key)() for key in (
            'color', 'linestyle', 'linewidth', 'alpha', 'marker', 'markersize',
            'markevery', 'markerfacecolor', 'markeredgecolor', 'markeredgewidth',
            'zorder', 'label', 'visible')}
    def collection_style(collection):
        return {
            'facecolor': collection.get_facecolor().tolist(),
            'edgecolor': collection.get_edgecolor().tolist(),
            'linewidth': collection.get_linewidth().tolist(),
            'linestyle': [[float(offset), None if sequence is None else list(sequence)]
                          for offset, sequence in collection.get_linestyle()],
            'alpha': collection.get_alpha(), 'zorder': collection.get_zorder(),
            'visible': collection.get_visible(),
        }
    return [{'lines': [line_style(line) for line in axis.lines],
             'collections': [collection_style(collection) for collection in axis.collections]}
            for axis in fig.axes]


def random_values(module):
    import numpy as np
    records = rows(HERE / 'random_curve_by_seed.csv')
    plotted = rows(HERE / 'random_plotted_points.csv')
    if len(records) != 660 or len(plotted) != 132:
        raise ValueError('Exactly five seeds and 132 complete nodes required')
    by_key = {(int(row['seed']), int(row['training_step'])): row for row in records}
    if len(by_key) != 660 or sorted(by_key) != [(seed, step) for seed in SEEDS for step in NODES]:
        raise ValueError('Random curve grid/seed keys differ')
    for row in records:
        if not (row['domain'] == 'v5' and row['method'] == 'props'
                and int(row['unique_DEV_paths']) == 32 and int(row['stochastic_replicates']) == 8):
            raise ValueError('Random curve conditions differ')
    # Work in original native units so multiplication and per-seed smoothing
    # match the unchanged parent plotting order exactly.
    values = np.asarray([[float(by_key[seed, step]['stochastic_native_return']) for step in NODES] for seed in SEEDS])
    if not (np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()):
        raise ValueError('Nonfinite or invalid native demand-completion values')
    smooth = module.causal_trailing_per_seed(values, 5)
    expected = {
        'raw_mean_x100': values.mean(0) * 100,
        'raw_sample_sd_x100': values.std(0, ddof=1) * 100,
        'mean_x100': smooth.mean(0) * 100,
        'sample_sd_x100': smooth.std(0, ddof=1) * 100,
    }
    for index, point in enumerate(plotted):
        if point['domain'] != 'v5' or point['method'] != 'props' or int(point['training_step']) != NODES[index]:
            raise ValueError('Plotted random curve order differs')
        if int(point['seed_count']) != 5 or int(point['window_evaluation_points']) != 5:
            raise ValueError('Independent seed count or display smoothing differs')
        for field, vector in expected.items():
            if not np.isclose(float(point[field]), vector[index], rtol=0, atol=1e-12):
                raise ValueError('Displayed random statistic does not reproduce: ' + field)
    return np.asarray(NODES, dtype=float), values, plotted, expected


def replace_only_props(fig, x, expected):
    import numpy as np
    axis = fig.axes[1]
    if len(axis.lines) != 13 or len(axis.collections) != 6:
        raise ValueError('Original six-method panel artist count changed')
    raw, line, band = axis.lines[2], axis.lines[3], axis.collections[1]
    if line.get_label() != 'PROPS' or raw.get_color() != '#0072B2' or line.get_color() != '#0072B2':
        raise ValueError('Expected original PROPS artist ordering changed')
    if not np.array_equal(line.get_xdata(), x / 1000):
        raise ValueError('Original PROPS x-grid changed')
    limits = axis.get_xlim(), axis.get_ylim()
    raw.set_ydata(expected['raw_mean_x100'])
    line.set_ydata(expected['mean_x100'])
    if not hasattr(band, 'set_data'):
        raise RuntimeError('Matplotlib >=3.10 FillBetweenPolyCollection.set_data is required')
    band.set_data(x / 1000, expected['mean_x100'] - expected['sample_sd_x100'],
                  expected['mean_x100'] + expected['sample_sd_x100'])
    axis.set_xlim(limits[0]); axis.set_ylim(limits[1])


def compare_geometry(baseline, changed, before_styles, after_styles):
    """Prove every original artist except the three target objects is identical."""
    if before_styles != after_styles:
        raise ValueError('Original curve artist styles changed')
    for key in ('canvas_shape', 'figure_inches', 'dpi', 'axes_positions', 'legends', 'spacing'):
        if baseline[key] != changed[key]:
            raise ValueError('Original figure layout changed: ' + key)
    for axis_index, (old, new) in enumerate(zip(baseline['geometry'], changed['geometry'])):
        for key in ('xlim', 'ylim', 'xscale', 'yscale', 'xticks', 'yticks', 'patches'):
            if old[key] != new[key]:
                raise ValueError('Original axis geometry changed: ' + key)
        for index, (a, b) in enumerate(zip(old['lines'], new['lines'])):
            if (axis_index, index) not in ((1, 2), (1, 3)) and a != b:
                raise ValueError('An untouched curve line changed')
        for index, (a, b) in enumerate(zip(old['collections'], new['collections'])):
            if (axis_index, index) != (1, 1) and a != b:
                raise ValueError('An untouched SD band changed')
    prior_text = {record['key']: record for record in baseline['texts']}
    current_text = {record['key']: record for record in changed['texts']}
    if any(current_text.get(key) != value for key, value in prior_text.items()):
        raise ValueError('An original text object moved or changed')
    added = [record for key, record in current_text.items() if key not in prior_text]
    if added:
        raise ValueError('No local execution annotation is added; protocol remains in the experiment settings')
    return {'status': 'PASS', 'all_panel_a_artists_exact': True,
            'all_non_PROPS_panel_b_artists_exact': True, 'all_original_artist_styles_exact': True,
            'all_original_axes_ranges_ticks_positions_text_exact': True, 'shared_legend_exact': True,
            'changed_line_indices': {'axis1': [2, 3]}, 'changed_band_indices': {'axis1': [1]},
            'added_annotation': None}


def save_fig(fig, destination, parent_only=False):
    for suffix in ('pdf', 'png', 'svg'):
        metadata = ({'Creator': 'C4 independent curve presentation', 'CreationDate': None, 'ModDate': None}
                    if suffix == 'pdf' else ({'Date': None} if suffix == 'svg' else None))
        fig.savefig(destination / ('figure.' + suffix), dpi=200, metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-parent-only', action='store_true',
                        help='Only reproduce original deterministic parent as a clearly labeled presentation control')
    args = parser.parse_args()
    parent_binding = check_parent_binding()
    if args.check_parent_only:
        output = args.output_dir or PUBLICATION / 'preflight/parent_reproduction'
    else:
        output = args.output_dir or HERE
        random_binding = admit_complete_random_inputs()
    output = output.resolve()
    if not output.is_relative_to(PUBLICATION.resolve()):
        raise ValueError('All outputs and cache must remain below the new publication root')
    if args.check_parent_only and not output.is_relative_to((PUBLICATION / 'preflight').resolve()):
        raise ValueError('Parent control outputs must remain under publication/preflight')
    output.mkdir(parents=True, exist_ok=True)
    module = load_parent()
    fig, original_computed_points, config = build_parent(module)
    before_styles = artist_styles(fig)
    temporary_geometry = output / 'parent_reproduction_geometry.json'
    module.export_geometry(fig, temporary_geometry)
    baseline = read(temporary_geometry)
    original_geometry = read(HERE / 'original_render_geometry.json')
    for key in ('canvas_shape', 'figure_inches', 'dpi', 'axes_positions', 'geometry', 'texts', 'legends', 'spacing'):
        if baseline[key] != original_geometry[key]:
            raise ValueError('Saved parent reproduction differs from accepted source geometry: ' + key)
    if args.check_parent_only:
        save_fig(fig, output, parent_only=True)
        control = {'status': 'PASS_ORIGINAL_PARENT_REPRODUCTION_ONLY',
                   'new_stochastic_data_used': False, 'new_stochastic_result_claimed': False,
                   'accepted_parent_geometry_exact': True,
                   'original_png_sha256': parent_binding['parent_files']['figure.png'],
                   'reproduced_png_sha256': sha(output / 'figure.png'),
                   'exact_original_png_bytes': sha(output / 'figure.png') == parent_binding['parent_files']['figure.png'],
                   'models_loaded': 0, 'environment_steps': 0}
        dump(output / 'CONTROL_ONLY.json', control)
        print(json.dumps(control, ensure_ascii=False))
        module.plt.close(fig)
        return
    x, values, random_points, expected = random_values(module)
    replace_only_props(fig, x, expected)
    module.export_geometry(fig, output / 'render_geometry.json')
    changed = read(output / 'render_geometry.json')
    proof = compare_geometry(baseline, changed, before_styles, artist_styles(fig))
    dump(output / 'narrow_replacement_proof.json', proof)
    # Retain original rows/cells for every non-target display point, including
    # their original order. Never overwrite historical plotted_points.csv.
    old_points = rows(HERE / 'plotted_points.csv')
    new_by_step = {int(record['training_step']): record for record in random_points}
    display = []
    for record in old_points:
        target = record['domain'] == 'v5' and record['method'] == 'props'
        replacement = dict(new_by_step[int(record['training_step'])]) if target else dict(record)
        replacement['action_execution'] = 'native_stochastic' if target else 'original_mean'
        replacement['action_repeats_within_training_seed'] = 8 if target else 1
        display.append(replacement)
    write_csv(output / 'current_display_points.csv', display)
    save_fig(fig, output)
    source = {
        'schema': 'c4-figure44b-real-props-stochastic-narrow-replacement-v1',
        'status': 'COMPLETE_REAL_RANDOM_CURVE_RENDERED',
        'parent_binding': parent_binding, 'random_input_binding': random_binding,
        'only_replaced': 'panel(b) PROPS raw line, trailing5 mean line, five-training-seed sample SD band',
        'display_conditions': 'PROPS in panel(b): native random actions; all other original curves: original mean actions',
        'local_label': None,
        'presentation_amendment': 'Standalone annotation removed; complete execution protocol retained in paper settings; all scientific curve data unchanged',
        'scientific_tables_Nq_AUC_TEST': 'retain original mean execution protocol',
        'cross_execution_gap': 'does not on its own establish an intrinsic algorithm benefit',
        'statistics': '32 cases per action repeat; equal mean of 8 repeats within seed; 5 independent training seeds; trailing5 separately inside seed before across-seed mean/sample SD',
        'actual_raw_endpoint_score_x100': float(expected['raw_mean_x100'][-1]),
        'actual_display_endpoint_score_x100': float(expected['mean_x100'][-1]),
        'endpoint_score_was_forced': False,
        'narrow_replacement_proof': proof,
        'historical_data_sha256': {name: sha(HERE / name) for name in ('data.csv', 'plotted_points.csv', 'figure_config.json')},
        'code_sha256': {name: sha(HERE / name) for name in ('plot.py', 'parent_plot.py', 'presentation_style.py')},
        'outputs_sha256': {name: sha(output / name) for name in ('figure.pdf', 'figure.svg', 'figure.png', 'current_display_points.csv', 'render_geometry.json', 'narrow_replacement_proof.json')},
        'render_model_loads': 0, 'render_environment_steps': 0,
    }
    dump(output / 'source.json', source)
    module.plt.close(fig)
    print(json.dumps({'status': 'PASS', 'output': str(output),
                      'raw_endpoint': source['actual_raw_endpoint_score_x100'],
                      'display_endpoint': source['actual_display_endpoint_score_x100']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
