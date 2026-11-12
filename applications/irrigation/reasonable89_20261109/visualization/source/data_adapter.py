"""Read only newly frozen and independently audited reasonable89 evidence.

There is no fallback to an earlier experiment, no environment/model import, and
no path that relabels old numeric results as evidence for the new adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('REASONABLE89_RESULTS_ROOT',
    '/hy-tmp/resource_allocation_results/chapter06/experiments/reasonable_allocation89_20261109')).resolve()
if not ROOT.is_relative_to(Path('/hy-tmp').resolve()):
    raise ValueError('All generated artifacts must remain under HY')
CASE_ID = 'Hongmiao89-reasonable-depth-v1'
SEEDS = (6400, 6401, 6402)
TEST = None  # Resolved only from the prospective amendment + actual sealed panel.
MAP_DAYS = (0, 46, 62, 63, 72)
ROLES = {'final', 'initial', 'fixed_budget', 'historical_final', 'feedback', 'representative_feedback'}
M64 = 'fixed64_antithetic_mean_feasible_actions'


def selected_scalar(application, keys):
    values = [float(application[key]) for key in keys if key in application]
    if not values or any(value != values[0] for value in values):
        raise ValueError('Missing or inconsistent selected-application scalar: ' + '/'.join(keys))
    return values[0]


def execution_identity(description):
    """Compare actual shared-module/point identity without importing the model."""
    if not isinstance(description, dict):
        raise ValueError('M64 requires its full immutable contract, not a display label')
    operator = description.get('operator', description.get('deployment_execution', description.get('execution_operator')))
    if operator != M64:
        raise ValueError('Old latent-mean execution cannot be relabelled as M64')
    source = description.get('operator_source_sha256', description.get('source_sha256'))
    points = description.get('points_array_sha256')
    if any(not isinstance(value, str) or len(value) != 64 for value in (source, points)):
        raise ValueError('Both actual shared-module sourceSHA and point-arraySHA are required')
    count = int(description.get('point_count', 0))
    dim = int(description.get('event_dim', description.get('point_event_dimension', 0)))
    seed = int(description.get('seed', description.get('point_generator_seed', -1)))
    if count != 64 or dim < 1 or seed < 0 or description.get('training') is True:
        raise ValueError('Invalid deterministic64-point application execution contract')
    return dict(operator=operator, source_sha256=source, points_array_sha256=points,
                point_count=count, event_dim=dim, seed=seed)


def execution_from(owner):
    candidates = []
    for key in ('policy_execution_contract', 'execution_contract', 'development_execution',
                'deployment_execution', 'execution_operator', 'operator', 'execution_description', 'baseline_execution_contract'):
        value = owner.get(key)
        if isinstance(value, dict) and 'points_array_sha256' in value:
            candidates.append(execution_identity(value))
    if not candidates and 'points_array_sha256' in owner:
        candidates.append(execution_identity(owner))
    if not candidates or any(value != candidates[0] for value in candidates):
        raise ValueError('Missing or mixed application execution/point identities')
    return candidates[0]


def validate_application_identity(model, metadata, adapter_id, weight, sigma_scale, execution):
    """Pure metadata gate shared by all deployment phases, including initial."""
    if any(value != adapter_id for value in (model['adapter'], model['env_config']['adapter'], metadata['adapter'])):
        raise ValueError('Selected adapter, model environment and evaluated actor differ')
    if any(float(value) != weight for value in (
            model['weight'], model['env_config']['coverage_weight'],
            metadata['trained_weight'], metadata['preference_weight'])):
        raise ValueError('Selected, trained, environmental and evaluated weights differ')
    if any(float(value) != sigma_scale for value in (
            model['spatial_sigma_scale'], model['env_config']['spatial_sigma_scale'], metadata['spatial_sigma_scale'])):
        raise ValueError('Initial/final spatial sigma scales do not match')
    if execution_identity(model['development_execution']) != execution or execution_from(metadata) != execution:
        raise ValueError('Old latent-mean or mismatched M64 source/points cannot enter the deployment evidence')


def normalized_design(raw):
    """Read final frozen design without treating its superseded TEST list as final."""
    if 'formal_profile' not in raw:
        return raw
    result = dict(raw)
    result['learned_objective_order'] = raw['objective']['order']
    result['formal'] = dict(seeds=raw['seeds'], budgets_m3=raw['budgets_m3'],
        episodes_per_model=raw['formal_profile']['episodes'],
        evaluate_every=raw['formal_profile']['eval_every'],
        development_weather_seeds=raw['development_weather_seeds'])
    return result


def validate_namespace_panel(amendment, panel, capsule, amendment_sha, panel_sha,
                             base_design_sha, selected_application_sha):
    """Pure prospective-namespace check; no forcing generation or outcome access."""
    if amendment.get('status') != 'LOCKED_BEFORE_ANY_NEW_TEST_FORCING_OR_OUTCOME':
        raise ValueError('Require the explicitly prospective unopened-TEST amendment')
    if (amendment['base_final_design_sha256'] != base_design_sha or
            amendment['selected_application_sha256'] != selected_application_sha):
        raise ValueError('Namespace amendment is not bound to the immutable model design/application')
    registered_panel = panel.get('panels', {}).get('test128', panel)
    seeds = np.asarray(registered_panel['weather_seeds'], dtype=np.int64)
    if seeds.shape != (128,) or len(set(seeds.tolist())) != 128 or np.any(seeds < 0):
        raise ValueError('The actual sealed TEST panel must contain128 unique integer seeds')
    if not np.array_equal(seeds, np.asarray(amendment['test_weather_seeds'], dtype=np.int64)):
        raise ValueError('Do not use the superseded original final-design TEST namespace')
    if (amendment['feedback32_weather_seeds'] != seeds[:32].tolist() or
            int(amendment['historical_reset_seed']) != int(seeds[0])):
        raise ValueError('Feedback/history namespaces do not follow the registered amendment')
    if 'panels' in panel:
        expected = {'test128': (list(range(128)), seeds.tolist()),
                    'feedback32': (list(range(32)), seeds[:32].tolist()),
                    'history2022': ([0], [int(seeds[0])]),
                    'history2024': ([0], [int(seeds[0])])}
        for name, (weather_ids, weather_seeds) in expected.items():
            if panel['panels'][name] != dict(weather_ids=weather_ids, weather_seeds=weather_seeds):
                raise ValueError('Actual sealed ' + name + ' panel differs from the prospective amendment')
        if panel.get('case_id') != CASE_ID or panel.get('source_release') != amendment['source_release']:
            raise ValueError('Actual sealed panel belongs to a different task or source')
    for owner, name in ((panel, 'sealed panel'), (capsule, 'evaluation capsule')):
        actual = owner.get('TEST_namespace_amendment_sha256', owner.get('test_namespace_amendment_sha256'))
        if actual != amendment_sha:
            raise ValueError(name + ' is not bound to the new namespace amendment SHA')
    actual_panel = capsule.get('test_panel_sha256', capsule.get('sealed_test_panel_sha256', capsule.get('weather_panel_sha256')))
    if actual_panel != panel_sha:
        raise ValueError('Evaluation capsule does not bind the actual sealed panel SHA')
    if capsule.get('scientific_source_release', amendment['source_release']) != amendment['source_release']:
        raise ValueError('Evaluation capsule belongs to a different scientific source')
    return seeds


def provenance_value(owner, names):
    """Read explicit native provenance fields without inferring from a filename."""
    candidates = []
    for value in (owner, owner.get('job', {}), owner.get('provenance', {}),
                  owner.get('namespace_contract', {})):
        for name in names:
            if name in value:
                candidates.append(value[name])
    if not candidates or any(value != candidates[0] for value in candidates):
        raise ValueError('Missing or inconsistent provenance: ' + '/'.join(names))
    return candidates[0]


def initial_actor_identities(proof):
    """Read real independent initial-parameter audit, without manufacturing one."""
    if proof.get('status') == 'PASS' and 'initial_actor_sha256_by_seed' in proof:
        return proof['initial_actor_sha256_by_seed']
    if (proof.get('status') != 'PASS_ALL9_IMMUTABLE_FIXED3200_PARAMETER_IDENTITY_NO_TEST' or
            proof.get('native_model_and_initial_parameter_hashes_checked') is not True):
        raise ValueError('An actual independent initial-parameter identity proof is required')
    rows = [row for row in proof['models'] if budget_class(row['budget_m3'], (3500000, 4000000, 4500000)) == 4000000]
    if sorted(int(row['seed']) for row in rows) != list(SEEDS):
        raise ValueError('The independent identity audit lacks the three actual main initial actors')
    return {str(row['seed']): row['initial_actor_sha256'] for row in rows}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def rows_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Refuse an unexplained empty plotting CSV')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def budget_class(value, levels):
    matches = [float(level) for level in levels
               if math.isclose(float(value), float(level), rel_tol=0, abs_tol=1e-6)]
    if len(matches) != 1:
        raise ValueError('No unique registered budget category within1e-6m3')
    return matches[0]


def metrics_from_y(y, area):
    y, area = np.asarray(y, dtype=np.float64), np.asarray(area, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 89 or area.shape != (89,) or not np.isfinite(y).all():
        raise ValueError('Need genuine paths×89 terminal relative yields')
    mask = y >= .9
    return dict(C90=mask.mean(axis=1), JA=(y * area[None, :]).sum(axis=1) / area.sum(),
                J=y.mean(axis=1), CA90=(mask * area[None, :]).sum(axis=1) / area.sum())


def nice_ceiling(value):
    if value <= 0 or not math.isfinite(value):
        raise ValueError('Need a finite positive scale')
    power = 10 ** math.floor(math.log10(value))
    return next(step * power for step in (1., 2., 2.5, 5., 10.) if step * power >= value)


def common_volume_edges(case_a):
    """Six explicit fixed water-volume classes, shared over the complete81 days.

    Fine low-volume boundaries preserve hundred-m3 differences. The highest
    bound covers the real complete-case maximum; no per-frame rescaling.
    Equal-width legend swatches are discrete classes, not a linear value axis.
    """
    maximum = float(np.max(case_a))
    upper = max(6000., nice_ceiling(maximum / 6.) * 6.)
    return np.asarray([0., 125., 500., 1000., 2000., 4000., upper])


@dataclass
class Record:
    info: dict
    arrays: dict
    metrics: dict

    @property
    def seed(self):
        return int(self.info['training_seed'])

    @property
    def role(self):
        return self.info['role']

    def row_for(self, weather_seed):
        positions = np.flatnonzero(self.arrays['weather_seed'] == int(weather_seed))
        if len(positions) != 1:
            raise ValueError('Missing or duplicate paired weather identity')
        return int(positions[0])

    def weather_row(self, weather_seed):
        return self.row_for(weather_seed)


class Evidence:
    def __init__(self, index_path):
        self.index_path = Path(index_path).resolve(); self.index = json.loads(self.index_path.read_text())
        if self.index.get('schema') != 'reasonable89-publication-index-v1':
            raise ValueError('A new reasonable89 evidence index is required')
        self.bindings = {str(self.index_path): sha(self.index_path)}
        accepted_path = self.bound(self.index['accepted_scientific_run'])
        self.accepted = json.loads(accepted_path.read_text())
        if self.accepted.get('status') != 'ACCEPTED_FOR_PUBLICATION':
            raise ValueError('No explicit scientific-run acceptance for publication')
        self.run_root = Path(self.accepted['run_root']).resolve()
        if not self.run_root.is_relative_to(ROOT):
            raise ValueError('Old or foreign scientific evidence cannot be relabelled')
        self.release = self.accepted['source_release']
        application_path = self.bound(self.index['selected_application'])
        self.application = json.loads(application_path.read_text())
        self.application_sha = sha(application_path)
        accepted_application = self.accepted.get('selected_application')
        if isinstance(accepted_application, dict) and 'path' in accepted_application:
            if self.bound(accepted_application) != application_path:
                raise ValueError('Accepted run and registry select different application files')
        elif accepted_application != self.application:
            raise ValueError('Accepted run must bind the same complete selected application')
        self.weight = selected_scalar(self.application, ('training_weight', 'coverage_weight', 'weight'))
        self.sigma_scale = selected_scalar(self.application, ('spatial_sigma_scale',))
        self.adapter_id = self.application.get('adapter_id', self.application.get('adapter'))
        self.execution = execution_from(self.application)
        if not 0 <= self.weight <= 1 or self.application['source_release'] != self.release:
            raise ValueError('Use the actually selected application weight and frozen source, not the original.8 default')
        operator_source = self.bound(self.index['shared_execution_operator_source'])
        if sha(operator_source) != self.execution['source_sha256']:
            raise ValueError('Application operator differs from the actual new frozen shared module')
        if 'execution_points' in self.index:
            binding = self.index['execution_points']; point_path = self.bound(binding)
            loaded = np.load(point_path, allow_pickle=False)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                try:
                    points = loaded[binding['array_name']].copy()
                finally:
                    loaded.close()
            else:
                points = loaded
            if (points.shape != (64, self.execution['event_dim']) or points.dtype != np.float64 or
                    hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest() != self.execution['points_array_sha256'] or
                    not np.array_equal(points[32:], -points[:32])):
                raise ValueError('Recorded deterministic antithetic points differ from the adopted M64 contract')
        audit = json.loads(self.bound(self.index['independent_numeric_audit']).read_text())
        if self.bound(self.accepted['independent_numeric_audit']) != self.bound(self.index['independent_numeric_audit']):
            raise ValueError('The accepted run does not bind the same independent final numeric audit')
        if (audit.get('status') != 'PASS' or audit.get('formal_results_only') is not True or
                Path(audit['run_root']).resolve() != self.run_root or audit['source_release'] != self.release):
            raise ValueError('Independent final-data audit does not bind this new frozen run')
        if audit.get('initialization_identities_verified') is not True:
            raise ValueError('Independent audit must verify the exactly reconstructed initial actor identities')
        design_path = self.bound(self.index['experiment_design'])
        self.protocol = normalized_design(json.loads(design_path.read_text()))
        if self.protocol['case_id'] != CASE_ID or self.protocol['learned_objective_order'] != ['C90', 'JA']:
            raise ValueError('New task or learned objective differs')
        amendment_path = self.bound(self.index['TEST_namespace_amendment'])
        amendment = json.loads(amendment_path.read_text())
        self.namespace_amendment_sha = sha(amendment_path)
        panel_path = self.bound(self.index['sealed_TEST_panel'])
        if panel_path.suffix == '.json':
            panel = json.loads(panel_path.read_text())
        else:
            # Only reached after explicit scientific acceptance.  Read the seed
            # identity array, never regenerate or transform sealed forcing.
            panel = json.loads(self.bound(self.index['sealed_TEST_panel_metadata']).read_text())
            with np.load(panel_path, allow_pickle=False) as store:
                key = self.index.get('TEST_panel_seed_array', 'weather_seed')
                panel['weather_seeds'] = store[key].astype(np.int64).tolist()
        capsule = json.loads(self.bound(self.index['evaluation_capsule']).read_text())
        self.test_panel_sha = sha(panel_path)
        self.test_weather = validate_namespace_panel(amendment, panel, capsule,
            self.namespace_amendment_sha, self.test_panel_sha, sha(design_path), self.application_sha)
        if amendment['source_release'] != self.release:
            raise ValueError('Prospective TEST amendment belongs to another scientific release')
        self.historical_reset_seed = int(amendment['historical_reset_seed'])
        self.levels = self.protocol['formal']['budgets_m3']
        if tuple(self.protocol['formal']['seeds']) != SEEDS:
            raise ValueError('Do not reuse earlier training seeds as this formal study')
        feedback = self.index['feedback_design']
        if (feedback['changed_days'] != [62, 63, 64] or feedback['et0_multipliers'] != [.85, 1.15]
                or feedback['statistical_weather_seeds'] != self.test_weather[:32].tolist()):
            raise ValueError('Feedback display must match the newly frozen explicit intervention protocol')
        self.fixed = json.loads(self.bound(self.index['fixed_inputs']).read_text())
        self.ids = [str(value) for value in self.fixed['region_ids']]
        self.area = np.asarray(self.fixed['area_ha'], dtype=np.float64)
        if len(self.ids) != 89 or self.area.shape != (89,) or (self.area <= 0).any():
            raise ValueError('The unchanged89 original service areas are required')
        self.top5 = np.asarray(sorted(range(89), key=lambda k: (-self.area[k], int(self.ids[k])))[:5])
        self.records = [self.load_record(item) for item in self.index['records']]
        for budget in self.levels:
            self.select('final', budget)
            if budget != 4000000:
                self.select('fixed_budget', budget)
        self.main = self.select('final', 4000000)
        self.initial = self.select('initial', 4000000)
        if any(record.role == 'initial' and record.info['budget_m3'] != 4000000 for record in self.records):
            raise ValueError('No initialization evidence was registered for the two secondary budgets')
        for final, initial in zip(self.main, self.initial):
            self.paired(final, initial)
            if initial.info.get('initialization_verified') is not True:
                raise ValueError('Initialization must be exactly reconstructed and hash verified')
            if not initial.info.get('initial_actor_sha256'):
                raise ValueError('An initial actor identity receipt is required')
        main_identity = {record.seed: record.info['actor_identity_sha256'] for record in self.main}
        for record in self.records:
            if record.role in ('fixed_budget', 'historical_final', 'feedback', 'representative_feedback'):
                if (record.info['trained_budget_m3'] != 4000000 or
                        record.info['actor_identity_sha256'] != main_identity[record.seed]):
                    raise ValueError('Fixed-policy condition is not the same accepted main endpoint')
        self.representative = self.choose_case()
        if 'representative_selection' in self.index:
            stored = json.loads(self.bound(self.index['representative_selection']).read_text())
            chosen = self.representative
            if ((int(stored['trained_seed']), int(stored['weather_id']), int(stored['weather_seed'])) !=
                    (chosen['training_seed'], chosen['weather_id'], chosen['weather_seed']) or
                    float(stored['JA']) != chosen['native_JA'] or
                    float(stored['pooled_median_JA']) != chosen['native_population_median']):
                raise ValueError('The displayed case differs from the predeclared storedJA median rule')
        extras = [record for record in self.records if record.role == 'representative_feedback']
        outside32 = self.representative['weather_seed'] not in self.test_weather[:32]
        if len(extras) != (4 if outside32 else 0):
            raise ValueError('Exactly the registered extra representative paths are required, without duplicate statistics')
        for record in extras:
            if (record.seed != self.representative['training_seed'] or
                    not np.array_equal(record.arrays['weather_seed'], [self.representative['weather_seed']])):
                raise ValueError('Extra feedback must describe the identical representative case')

    def bound(self, item):
        path = Path(item['path']).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != item['sha256']:
            raise ValueError('Input must be a SHA-bound file inside the new experiment only')
        self.bindings[str(path)] = item['sha256']; return path

    def load_record(self, item):
        item = dict(item)
        if item.get('role') == 'descriptive_feedback':
            item['native_role'] = 'descriptive_feedback'
            item['role'] = 'representative_feedback'
        if item['role'] not in ROLES or item.get('objective_order') != ['C90', 'JA']:
            raise ValueError('Foreign result role or objective order')
        if int(item['training_seed']) not in SEEDS:
            raise ValueError('A prior experiment or DEV candidate cannot enter the formal figure set')
        paths = {key: self.bound(value) for key, value in item['files'].items()}
        if not {'results', 'metadata', 'completion', 'model_protocol'} <= set(paths):
            raise ValueError('Need originalNPZ, metadata/completion and the actual trained-model protocol')
        if any(not paths[key].is_relative_to(self.run_root) for key in ('results', 'metadata', 'completion', 'model_protocol')):
            raise ValueError('Data are outside the explicitly accepted new frozen run')
        complete = json.loads(paths['completion'].read_text()); meta = json.loads(paths['metadata'].read_text())
        if (complete.get('status') != 'COMPLETE' or complete.get('formal') is not True or
                meta.get('job', {}).get('formal') is not True):
            raise ValueError('Engineering/DEV/pilot results cannot enter formal numeric figures')
        if meta['job']['source_release'] != self.release or meta['objective_order'] != ['C90', 'JA']:
            raise ValueError('Another adapter or source release was mixed into the new study')
        if provenance_value(meta, ('TEST_namespace_amendment_sha256', 'test_namespace_amendment_sha256')) != self.namespace_amendment_sha:
            raise ValueError('Each formal job must bind the prospective TEST amendment')
        if provenance_value(meta, ('test_panel_sha256', 'sealed_test_panel_sha256', 'weather_panel_sha256')) != self.test_panel_sha:
            raise ValueError('Each formal job must bind the actual amended TEST panel')
        if meta['scoring_tau'] != .9 or meta.get('learning_updates') != 0:
            raise ValueError('Evaluation threshold or frozen-policy execution differs')
        model = json.loads(paths['model_protocol'].read_text())
        if model['source_release'] != self.release:
            raise ValueError('A previous stochastic/native-mean model protocol cannot be relabelled')
        validate_application_identity(model, meta, self.adapter_id, self.weight, self.sigma_scale, self.execution)
        if int(meta['trained_seed']) != int(item['training_seed']) or meta.get('region_count') != 89:
            raise ValueError('A metadata label cannot substitute a new training or region identity')
        for native, registered in [('evaluation_budget_m3', 'budget_m3'), ('model_training_budget_m3', 'trained_budget_m3')]:
            if not math.isclose(float(meta[native]), float(item[registered]), rel_tol=0, abs_tol=1e-6):
                raise ValueError('Actual budget differs from the declared plotting category')
        identity_field = item.get('actor_identity_metadata_field', 'executed_actor_sha256')
        if meta.get(identity_field) != item['actor_identity_sha256']:
            raise ValueError('Recorded actor identity differs from the bound native metadata')
        if item['role'] == 'initial':
            proof = meta.get('initialization_reconstruction', {})
            if (meta.get('executed_policy_episode') != 0 or
                    proof.get('exact_actual_training_initialization') is not True or
                    proof.get('checkpoint_weights_not_used_for_actions') is not True):
                raise ValueError('Initial paths must actually execute the reconstructed original actor')
        for name, key in [('results.npz', 'results'), ('metadata.json', 'metadata')]:
            if complete['files_sha256'][name] != sha(paths[key]):
                raise ValueError('Completion receipt no longer matches its scientific arrays')
        with np.load(paths['results'], allow_pickle=False) as store:
            arrays = {name: store[name].copy() for name in store.files}
        if list(arrays['region_ids'].astype(str)) != self.ids:
            raise ValueError('Region identity/order changed')
        metrics = metrics_from_y(arrays['Y'], self.area)
        for canonical, name in [('C90', 'C90'), ('C90', 'C'), ('JA', 'JA'), ('J', 'J'), ('CA90', 'CA90')]:
            if name in arrays:
                np.testing.assert_allclose(arrays[name], metrics[canonical], rtol=0, atol=2e-12)
        n = len(arrays['Y'])
        if item['role'] == 'historical_final':
            if n != 1 or not np.array_equal(arrays['weather_seed'], [self.historical_reset_seed]):
                raise ValueError('Historical reset must use the amended first TEST seed')
        if arrays['a_m3'].shape != (n, 81, 89) or arrays['B_mm'].shape != (n, 82, 89):
            raise ValueError('All new paths must keep actual allocations and full recorded storage')
        if (not np.isfinite(arrays['a_m3']).all() or (arrays['a_m3'] < 0).any() or
                not np.isfinite(arrays['B_mm']).all()):
            raise ValueError('Actual allocations must be finite/nonnegative; storage must be finite')
        np.testing.assert_allclose(arrays['a_m3'].sum(axis=2), arrays['q_m3'], rtol=0, atol=1e-7)
        np.testing.assert_allclose(arrays['balance_m3'][:, :-1] - arrays['q_m3'],
                                   arrays['balance_m3'][:, 1:], rtol=0, atol=1e-7)
        budget = budget_class(item['budget_m3'], self.levels)
        np.testing.assert_allclose(arrays['q_m3'].sum(axis=1), budget, rtol=0, atol=1e-6)
        metrics['used_water_m3'] = arrays['q_m3'].sum(axis=1)
        metrics['I_mm_by_region'] = arrays['a_m3'].sum(axis=1) / (10 * self.area[None, :])
        metrics['zero_days_by_region'] = (arrays['a_m3'] == 0).sum(axis=1)
        info = dict(item); info['budget_m3'] = budget
        info['trained_budget_m3'] = budget_class(item['trained_budget_m3'], self.levels)
        return Record(info, arrays, metrics)

    def select(self, role, budget=None):
        records = sorted([record for record in self.records if record.role == role
                          and (budget is None or record.info['budget_m3'] == budget)], key=lambda x: x.seed)
        if [record.seed for record in records] != list(SEEDS):
            raise ValueError('Need exactly all three accepted independent policy identities')
        if role in ('final', 'initial', 'fixed_budget'):
            for record in records:
                if not np.array_equal(record.arrays['weather_seed'], self.test_weather):
                    raise ValueError('Formal128 must use the new sealed TEST panel')
                self.paired(records[0], record)
        return records

    @staticmethod
    def paired(first, second):
        for key in ('weather_seed', 'forcing_sha256'):
            if not np.array_equal(first.arrays[key], second.arrays[key]):
                raise ValueError('Initial/final or controller paths do not share the same complete forcing')

    @staticmethod
    def assert_paired(first, second):
        Evidence.paired(first, second)

    def choose_case(self):
        pooled = np.concatenate([record.arrays['JA'] for record in self.main]); median = float(np.median(pooled))
        options = [(abs(float(value) - median), record.seed, int(record.arrays['weather_id'][row]), row, record)
                   for record in self.main for row, value in enumerate(record.arrays['JA'])]
        _, seed, weather_id, row, record = min(options, key=lambda value: value[:3])
        return dict(record=record, row=row, training_seed=seed, weather_id=weather_id,
                    weather_seed=int(record.arrays['weather_seed'][row]), native_JA=float(record.arrays['JA'][row]),
                    native_population_median=median, selection='new main384 nativeJA median; tie seed then weatherID')

    def case(self):
        chosen = self.representative; record, row = chosen['record'], chosen['row']
        return dict(a_m3=record.arrays['a_m3'][row], q_m3=record.arrays['q_m3'][row],
            balance_m3=record.arrays['balance_m3'][row], B_mm=record.arrays['B_mm'][row],
            Y=record.arrays['Y'][row], dates=record.arrays['date'], ids=self.ids,
            trained_seed=chosen['training_seed'],
            **{key: value for key, value in chosen.items() if key not in ('record', 'row')})

    def development_curves(self):
        curves = []
        for item in sorted(self.index['main_development'], key=lambda value: value['training_seed']):
            if int(item['training_seed']) not in SEEDS or item.get('formal_training') is not True:
                raise ValueError('Only the three fresh formal main-policy learning records are allowed')
            paths = {key: self.bound(value) for key, value in item['files'].items()}
            completion = json.loads(paths['completion'].read_text()); protocol = json.loads(paths['protocol'].read_text())
            if completion.get('formal_training') is not True or protocol['episodes'] != 3200:
                raise ValueError('DEV learning plot cannot use a selected candidate or short pilot')
            if protocol['source_release'] != self.release:
                raise ValueError('DEV record is from another frozen adapter')
            if float(protocol['weight']) != self.weight or float(protocol['env_config']['coverage_weight']) != self.weight:
                raise ValueError('Formal learning curve uses a different selected preference')
            if (float(protocol['spatial_sigma_scale']) != self.sigma_scale or
                    float(protocol['env_config']['spatial_sigma_scale']) != self.sigma_scale or
                    execution_identity(protocol['development_execution']) != self.execution):
                raise ValueError('Old latent-mean/sigma-mismatched DEV cannot enter the M64 learning curve')
            if int(protocol['seed']) != int(item['training_seed']):
                raise ValueError('A candidate seed cannot be relabelled as a formal learning curve')
            if not math.isclose(float(protocol['env_config']['total_budget_m3']), 4000000, rel_tol=0, abs_tol=1e-6):
                raise ValueError('The learning graph is restricted to the main4M task')
            final = next(record for record in self.main if record.seed == int(item['training_seed']))
            identity_field = item.get('completion_actor_identity_field', 'endpoint_actor_sha256')
            if completion.get(identity_field) != final.info['actor_identity_sha256']:
                raise ValueError('DEV learning records and the formal TEST endpoint are not the same trained actor')
            with np.load(paths['development'], allow_pickle=False) as store:
                z = {name: store[name].copy() for name in store.files}
            names = list(z['metric_names'].astype(str)); aliases = {'C90': ['C90', 'C'], 'JA': ['JA', 'area_weighted_performance']}
            episodes = np.unique(z['episode']); expected = np.arange(0, 3201, 128)
            if not np.array_equal(episodes, expected):
                raise ValueError('Keep the true0 point and every fixed DEV position, without smoothing or best-step selection')
            data = {}
            for key, choices in aliases.items():
                found = next((name for name in choices if name in names), None)
                if found is None:
                    raise ValueError('Missing native DEV metric: ' + key)
                column = names.index(found); values = []
                for episode in episodes:
                    mask = z['episode'] == episode
                    if not np.array_equal(np.sort(z['weather_seed'][mask]), np.asarray(self.protocol['formal']['development_weather_seeds'])):
                        raise ValueError('DEV point does not contain the full fixed32 weather panel')
                    values.append(float(z['metrics'][mask, column].mean()))
                data[key] = np.asarray(values)
            curves.append(dict(seed=int(item['training_seed']), episodes=episodes, **data))
        if [item['seed'] for item in curves] != list(SEEDS):
            raise ValueError('Missing a formal main-policy DEV curve')
        return curves


def aggregate(records, key):
    values = np.asarray([record.metrics[key].mean(axis=0) for record in records])
    return dict(mean=values.mean(axis=0), sd=values.std(axis=0, ddof=1), by_policy=values)


def public_deficit(storage, fixed):
    fields = fixed['field_parameters']
    threshold = np.asarray([field['soil_depth_mm'] * field['theta_crit'] for field in fields])
    return np.maximum(threshold - np.asarray(storage), 0.)


def select_cross_day_examples(case, area, fixed, days=(62, 63)):
    """Objective signed actual-volume changes, never invented stop/start classes.

    Choose the largest decrease and largest increase separately, exact-sign
    tests and original-ID tie breaks.  A missing sign is reported explicitly;
    it produces no substitute example.  If all changes are zero, no example is
    manufactured.  All89 rows remain in the companion complete table.
    """
    first, second = days
    delta = np.asarray(case['a_m3'][second] - case['a_m3'][first], dtype=np.float64)
    if delta.shape != (89,) or not np.isfinite(delta).all():
        raise ValueError('Need all89 genuine finite allocation changes')
    deficit = public_deficit(case['B_mm'], fixed)
    choices, status = [], {}
    for role, sign, label in [('maximum_decrease', delta < 0, '当日减配最多区域'),
                             ('maximum_increase', delta > 0, '当日增配最多区域')]:
        positions = np.flatnonzero(sign)
        if not len(positions):
            status[role] = dict(present=False, count=0, reason='No actual negative change' if role == 'maximum_decrease' else 'No actual positive change')
            continue
        chosen = min(positions, key=lambda i: ((float(delta[i]) if role == 'maximum_decrease' else -float(delta[i])), int(case['ids'][i])))
        status[role] = dict(present=True, count=int(len(positions)))
        choices.append(dict(selection_role=role, label=label, region_id=str(case['ids'][chosen]),
            area_mu=float(np.asarray(area)[chosen] * 15),
            day_before=first, date_before=str(case['dates'][first]),
            day_after=second, date_after=str(case['dates'][second]),
            allocation_before_m3=float(case['a_m3'][first, chosen]),
            allocation_after_m3=float(case['a_m3'][second, chosen]),
            actual_change_m3=float(delta[chosen]),
            storage_before_mm=float(case['B_mm'][first, chosen]),
            deficit_before_mm=float(deficit[first, chosen]),
            storage_after_mm=float(case['B_mm'][second, chosen]),
            deficit_after_mm=float(deficit[second, chosen]),
            next_storage_mm=float(case['B_mm'][second + 1, chosen]),
            next_deficit_mm=float(deficit[second + 1, chosen])))
    return dict(days=list(days), rule='minimum negative delta and maximum positive delta; exact m3; tie smaller originalID',
                class_status=status, selected=choices, all_changes_zero=bool(np.all(delta == 0)),
                exact_zero_changes=int(np.sum(delta == 0)),
                missing_class_policy='disclose absence and keep only existing sign; no substitute, stop label or invented example')
