#!/usr/bin/env python3
"""Independent reasonable89 evaluation; no rule benchmark and no learning.

Formal weather is opened only after the complete9 fixed-endpoint acceptance and
an explicit source-bound TEST release. All dynamics are the frozen model. Every
path keeps actual actions,82 state boundaries and full regional water accounts.
"""
from __future__ import annotations
import argparse
import dataclasses
from datetime import timedelta
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
STUDY = Path('/hy-tmp/resource_allocation_results/chapter06/experiments/reasonable_allocation89_20261109')
RUN = STUDY / 'formal'
CASE_ID = 'Hongmiao89-reasonable-depth-v1'
DOMAIN = 'hongmiao89_reasonable_depth_irrigation'
ENDPOINT_FORMAT = 'hongmiao89-reasonable-depth-dual-random-fixed-endpoint-v1'
CONTRACT = Path(__file__).resolve().with_name('evaluation_contract.json')
KINDS = ('main','trained_budget','initial','fixed_policy_budget','history','feedback','descriptive_feedback')
METRICS = ('C','C90','JA','J','joint_score','CA90','area_coverage_rate',
           'area_weighted_performance','water_used_m3','remaining_budget_m3','coverage_count')
M64_OPERATOR='fixed64_antithetic_mean_feasible_actions'
M64_POINTS_SHA='9b59078bcb00785b7febcfe2b78d35fd1bd227c0ec7a85af45ef25b1b6a6a500'
TEST_NAMESPACE_AMENDMENT_SHA='a8213768eeb0de85f7e774030670dcd9614b7a7c260b5cdb5bcc5fb4766fb079'
APP = np = torch = None


def require_hy(path):
    result = Path(path).expanduser().resolve()
    if not result.is_relative_to(Path('/hy-tmp').resolve()):
        raise ValueError('Generated or execution-owned data must physically reside onHY')
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def write_json(path,value):
    path=require_hy(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temporary.write_bytes(canonical(value)+b'\n');os.replace(temporary,path)


def verify_execution_capsule(job):
    expected=job.get('evaluation_files_sha256',{})
    required={str(Path(__file__).resolve()),str(CONTRACT),
              str(Path(__file__).resolve().with_name('prepare_evaluation.py')),
              str(Path(__file__).resolve().with_name('run_evaluation.py'))}
    if not required.issubset(expected):
        raise ValueError('Evaluation capsule must bind evaluator and its field contract')
    for name,digest in expected.items():
        if sha(require_hy(name))!=digest:raise ValueError('Evaluation auxiliary/source changed:'+name)
    if job.get('evaluation_source_sha256')!=sha(__file__):
        raise ValueError('Evaluator differs from registered job')
    return expected



def test_namespace_amendment(job):
    """Resolve only future TEST/reset seeds; retain the actual trained base design."""
    path=require_hy(job['test_namespace_amendment_path'])
    if (path!=STUDY/'protocol/TEST_namespace_amendment.json' or
        sha(path)!=TEST_NAMESPACE_AMENDMENT_SHA or job['test_namespace_amendment_sha256']!=TEST_NAMESPACE_AMENDMENT_SHA):
        raise ValueError('Only the sealed prospective TEST namespace amendment is accepted')
    amendment=json.loads(path.read_text())
    if (amendment.get('schema')!='reasonable89-prospective-unopened-TEST-namespace-amendment-v1' or
        amendment.get('status')!='LOCKED_BEFORE_ANY_NEW_TEST_FORCING_OR_OUTCOME' or
        amendment.get('source_release')!=job['source_release'] or
        amendment.get('base_final_design_sha256')!=job['experiment_design_sha256'] or
        require_hy(amendment['base_final_design_path'])!=require_hy(job['experiment_design_path']) or
        amendment.get('selected_application_sha256')!=job['selected_application_sha256'] or
        amendment.get('training_registration_sha256')!=job['formal_registration_sha256']):
        raise ValueError('Namespace amendment cannot replace source, training identity or base final design')
    plan=amendment['unchanged_evaluation_plan']
    expected=dict(base_jobs=36,base_paths=2694,max_descriptive_feedback_paths=4,representative_dates=[0,46,62,63,72],feedback_pairs_per_direction=96,all_initial_final_budget_feedback_operators=M64_OPERATOR)
    if any(plan.get(key)!=value for key,value in expected.items()):
        raise ValueError('Namespace amendment cannot change scientific conditions, representative dates or statistical roles')
    seeds=amendment['test_weather_seeds'];feedback=amendment['feedback32_weather_seeds']
    if (len(seeds)!=128 or len(set(seeds))!=128 or
        any(type(seed) is not int or not 0<=seed<2**32 for seed in seeds) or
        feedback!=seeds[:32] or amendment['historical_reset_seed']!=seeds[0] or
        amendment['weather_ids']!=list(range(128)) or amendment['feedback_weather_ids']!=list(range(32)) or
        amendment['historical_years']!=[2022,2024] or set(seeds)&set(amendment['superseded_test_weather_seeds'])):
        raise ValueError('Malformed, reused or misaligned amended TEST/reset namespace')
    audit=amendment['unused_candidate_audit']
    if (audit.get('status')!='PASS_CANDIDATE_NAMESPACE_NO_KNOWN_LOCAL_USAGE' or
        audit.get('scan_exceptions_resolved') is not True or sha(require_hy(audit['path']))!=audit['sha256']):
        raise ValueError('The unused-namespace audit is missing or changed')
    if amendment.get('TEST_release_authorized') is not False:
        raise ValueError('Namespace amendment is not an execution authorization')
    return amendment


def expected_test_panel(job):
    """Pure metadata construction; generates no forcing and opens no sealed panel."""
    amendment=test_namespace_amendment(job);seeds=amendment['test_weather_seeds'];feedback=amendment['feedback32_weather_seeds']
    return dict(schema='reasonable89-sealed-evaluation-weather-v1',case_id=CASE_ID,
        status='SEALED_NAMESPACE_AMENDED_AWAIT_ALL9_AND_SEPARATE_ROOT_RELEASE',
        base_final_design_sha256=job['experiment_design_sha256'],source_release=job['source_release'],
        test_namespace_amendment_path=job['test_namespace_amendment_path'],test_namespace_amendment_sha256=job['test_namespace_amendment_sha256'],
        superseded_panel_sha256=amendment['superseded_panel_sha256'],
        panels=dict(test128=dict(weather_ids=list(range(128)),weather_seeds=seeds),
            feedback32=dict(weather_ids=list(range(32)),weather_seeds=feedback),
            history2022=dict(weather_ids=[0],weather_seeds=[amendment['historical_reset_seed']]),
            history2024=dict(weather_ids=[0],weather_seeds=[amendment['historical_reset_seed']])),
        historical_seed_interpretation=amendment['historical_seed_meaning'],model_runs_at_sealing=0,environment_steps_at_sealing=0,TEST_forcing_generated_at_sealing=0)


def verify_test_barrier(job):
    """No TEST panel opening, environment construction or model inference here."""
    if not job.get('formal',True):return None
    test_namespace_amendment(job)
    root=require_hy(job['run_root'])
    path=require_hy(job.get('training_barrier',root/'protocol/all9_endpoints_verified.json'))
    if path!=root/'protocol/all9_endpoints_verified.json':
        raise ValueError('A historical or foreign training barrier is forbidden')
    barrier=json.loads(path.read_text())
    models=barrier.get('models',{})
    if barrier.get('status')!='PASS' or not isinstance(models,dict) or len(models)!=9:
        raise ValueError('All9 independently verified formal endpoints are required')
    expected={(budget,seed) for budget in (3500000,4000000,4500000) for seed in (6400,6401,6402)}
    actual={(int(item['budget_m3']),int(item['seed'])) for item in models.values()}
    if actual!=expected or barrier.get('fixed_endpoint_episode')!=3200:
        raise ValueError('The complete predeclared3-budget/3-seed matrix is required')
    for item in models.values():
        if item.get('episode')!=3200 or item.get('adapter')!=job['adapter']:
            raise ValueError('Engineering/DEV or foreign-adapter endpoint in formal barrier')
        model=require_hy(item['model_path'])
        if not model.is_relative_to(root) or sha(model)!=item['model_sha256']:
            raise ValueError('Missing or damaged retained formal endpoint')
    for key in ('source_release','adapter','training_weight','spatial_sigma_scale','execution_operator','execution_operator_module_sha256','execution_points_array_sha256','experiment_design_sha256'):
        if barrier.get(key)!=job.get(key):raise ValueError('Training/evaluation scientific identity differs:'+key)
    if require_hy(barrier['frozen_code_root'])!=require_hy(job['frozen_code_root']):
        raise ValueError('Formal evaluator must use the accepted formal frozen source')
    if 'development' in require_hy(job['frozen_code_root']).relative_to(root).parts:
        raise ValueError('DEV source location cannot be relabelled as formal execution')
    release_path=root/'protocol/test_release.json';release=json.loads(release_path.read_text())
    if (release.get('status')!='AUTHORIZED_AFTER_DEV_REVIEW' or
        release.get('training_barrier_sha256')!=sha(path) or
        release.get('evaluation_source_sha256')!=sha(__file__) or
        release.get('selected_adapter')!=job['adapter'] or
        release.get('training_weight')!=job['training_weight'] or
        release.get('spatial_sigma_scale')!=job['spatial_sigma_scale'] or
        any(release.get(key)!=job[key] for key in ('execution_operator','execution_operator_module_sha256','execution_points_array_sha256')) or
        release.get('evaluation_files_sha256')!=job['evaluation_files_sha256'] or
        release.get('source_release')!=job['source_release'] or
        release.get('test_namespace_amendment_sha256')!=job['test_namespace_amendment_sha256']):
        raise ValueError('Explicit source/adapter-bound TEST release is missing')
    if job.get('kind')=='descriptive_feedback' and release.get('max_descriptive_feedback_paths')!=4:
        raise ValueError('Root TEST release does not authorize the fixed descriptive branches')
    verify_execution_capsule(job)
    for key in ('weather_panel_file','weather_panel_sha256'):
        if release.get(key)!=job.get(key):raise ValueError('Job uses another sealed weather panel')
    selection_paths=(STUDY/'protocol/selected_application.json',root/'protocol/selected_application.json',
                     require_hy(job['frozen_code_root'])/'protocol/selected_application.json')
    for role,expected_path in (('selected_application',selection_paths[2]),
                               ('formal_registration',root/'protocol/formal_registration.json')):
        identity_path=require_hy(job[role+'_path'])
        if identity_path!=require_hy(expected_path) or sha(identity_path)!=job[role+'_sha256']:
            raise ValueError('Unregistered selected application/formal registration:'+role)
        identity=json.loads(identity_path.read_text())
        if identity.get('adapter')!=job['adapter'] or identity.get('training_weight')!=job['training_weight'] or identity.get('spatial_sigma_scale')!=job['spatial_sigma_scale'] or identity.get('execution_operator')!=job['execution_operator']:
            raise ValueError('Adapter/preference differs from frozen selection/registration:'+role)
    if any(sha(path)!=job['selected_application_sha256'] for path in selection_paths):
        raise ValueError('Study/formal/frozen selected-application bytes must agree')
    if job['execution_operator']!=M64_OPERATOR or job['execution_points_array_sha256']!=M64_POINTS_SHA or job['execution_point_count']!=64 or job['execution_event_dim']!=49:
        raise ValueError('Only the adopted shared fixed64 full49-dimensional execution is accepted')
    adoption=require_hy(job['application_adoption_decision_path'])
    if sha(adoption)!=job['application_adoption_decision_sha256']:
        raise ValueError('Application adoption decision changed')
    adopted=json.loads(adoption.read_text())['adopted']
    if any(job[key]!=adopted[key] for key in ('adapter','training_weight','spatial_sigma_scale')) or adopted['deployment_execution']!=M64_OPERATOR:
        raise ValueError('Formal identity differs from adopted application configuration')
    if float(job['training_weight']) not in (.2,.5,.8):
        raise ValueError('Only the DEV-reviewed preference candidates may be registered')
    if float(job['spatial_sigma_scale']) not in (.25,.5,1.) or (job['adapter']!='depth_area_physical' and float(job['spatial_sigma_scale'])!=1.):
        raise ValueError('Only the registered adapter/noise candidate combinations are accepted')
    design=require_hy(job['experiment_design_path'])
    if sha(design)!=job['experiment_design_sha256']:
        raise ValueError('Frozen experiment design changed')
    amendment=require_hy(job['preference_dev_amendment_path'])
    if sha(amendment)!=job['preference_dev_amendment_sha256']:
        raise ValueError('Frozen DEV preference amendment changed')
    if sha(require_hy(job['noise_dev_amendment_path']))!=job['noise_dev_amendment_sha256']:
        raise ValueError('Frozen DEV exploration-noise amendment changed')
    if sha(require_hy(job['noise_midpoint_dev_amendment_path']))!=job['noise_midpoint_dev_amendment_sha256']:
        raise ValueError('Frozen DEV midpoint-noise amendment changed')
    if sha(require_hy(job['parent_experiment_design_path']))!=job['parent_experiment_design_sha256']:
        raise ValueError('Original parent experiment design changed')
    if sha(require_hy(job['representative_feedback_contract_path']))!=job['representative_feedback_contract_sha256']:
        raise ValueError('PreTEST representative/feedback contract changed')
    return dict(path=str(path),sha256=sha(path),release_path=str(release_path),
        release_sha256=sha(release_path),models=models)


def bootstrap(job):
    global APP,np,torch,IrrigationSpec,IrrigationEncoder,append_preference,FullSpendSpec
    global FullSpendIrrigationEnv,make_env,make_models_for_arm,common,make_execution_actor
    APP=require_hy(job['frozen_code_root'])
    root=require_hy(job['run_root'])
    if not APP.is_relative_to(root):raise ValueError('Science source must belong to this independent runroot')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[key]='1'
    if os.environ.get('CUDA_VISIBLE_DEVICES','') not in ('','-1'):
        raise ValueError('Frozen-policy evaluation is CPU-only; CUDA must be hidden')
    for key,name in [('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp'),('MPLCONFIGDIR','matplotlib'),('XDG_CACHE_HOME','xdg')]:
        p=require_hy(root/'runtime_cache/evaluation'/name);p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    sys.path.insert(0,str(APP));sys.path.insert(0,str(APP/'source'))
    import numpy as _np
    import torch as _torch
    from execution import common as _common
    import hongmiao89 as h
    np,torch,common=_np,_torch,_common
    if Path(common.__file__).resolve()!=APP/'execution/common.py' or Path(h.__file__).resolve()!=APP/'source/hongmiao89/__init__.py':
        raise ValueError('Foreign/shared module import refused')
    IrrigationSpec,IrrigationEncoder,append_preference=h.IrrigationSpec,h.IrrigationEncoder,h.append_preference
    FullSpendSpec,FullSpendIrrigationEnv,make_env,make_models_for_arm=h.FullSpendSpec,h.FullSpendIrrigationEnv,h.make_env,h.make_models_for_arm
    if h.CASE_ID!=CASE_ID or h.DOMAIN!=DOMAIN:raise ValueError('Foreign case/domain in frozen package')
    from hongmiao89 import execution_operator as operator_module
    if Path(operator_module.__file__).resolve()!=APP/'source/hongmiao89/execution_operator.py' or sha(operator_module.__file__)!=job['execution_operator_module_sha256']:
        raise ValueError('Foreign or changed shared execution operator')
    if operator_module.M64_OPERATOR!=M64_OPERATOR or job['execution_operator']!=M64_OPERATOR:
        raise ValueError('Formal execution cannot fall back to native latentmean')
    make_execution_actor=operator_module.make_execution_actor
    if job.get('formal',True):
        manifest_path=require_hy(job['scientific_manifest_path'])
        if not manifest_path.is_relative_to(root) or 'development' in manifest_path.relative_to(root).parts:
            raise ValueError('Only this formal source manifest is accepted')
        if sha(manifest_path)!=job['scientific_manifest_sha256']:
            raise ValueError('Formal source manifest digest changed')
        manifest=json.loads(manifest_path.read_text())
        expected=manifest.get('source_files_sha256',manifest.get('files_sha256'))
        if not expected or expected!=job.get('source_files_sha256'):
            raise ValueError('Formal source map differs from its immutable manifest')
        release=manifest.get('source_release',manifest.get('release'))
        if release!=job['source_release']:
            raise ValueError('Formal source release differs from its manifest')
        if common.source_hashes()!=expected:
            raise ValueError('Evaluation must bind the entire imported scientific closure')
    if job.get('source_files_sha256'):common.source_hashes(job['source_files_sha256'])
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass


def environment(config,controller='full',forcing_trace=None,*,base_trace=None,transform=None):
    if controller!='full':raise ValueError('No external or diagnostic rule jobs in this study')
    options=dict(config);options['reward_mode']='area_weighted_performance'
    if forcing_trace is not None:
        options['forcing_trace']=forcing_trace
        options['forcing_label']='registered_original_donor_trace' if transform is None else 'explicit_et0_demand_shock'
        if transform is not None:options.update(forcing_base_trace=base_trace,forcing_transform=transform)
    return FullSpendIrrigationEnv(**options)


def model_input(encoder, weight, training_budget):
    state = append_preference(encoder.state(), weight)
    # Ledger and physical bounds use the evaluation budget. The learned input
    # divisor stays the original training budget, including +/-20% conditions.
    state['global'][1] = float(state['remaining']) / float(training_budget)
    return state


def prefix_update(digest, label, value):
    digest.update(label.encode() + b'\0')
    if isinstance(value, dict):
        for key in sorted(value):
            prefix_update(digest, key, value[key])
    else:
        a = np.ascontiguousarray(value)
        digest.update(str((str(a.dtype), a.shape)).encode() + b'\0' + a.tobytes())


def shared_fixed_inputs(env, run_root):
    root = require_hy(run_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'fixed_inputs89.json'
    fields = []
    for p in env.scenario.field_params:
        values = dataclasses.asdict(p)
        fields.append({k: float(v) for k, v in values.items()})
    data = dict(schema='reasonable89-fixed-physical-inputs-v1',
        case_id=CASE_ID, region_ids=list(env.region_ids),
        area_ha=[float(p.area_ha) for p in env.scenario.field_params],
        field_parameters=fields, initial_storage_mm=env.scenario.initial_storage_mm.tolist(),
        yield_lambdas=np.asarray(env.public_observation()['yield_lambdas'], dtype=np.float64).tolist(),
        allocation_month_days=[(env.scenario.start_date + timedelta(days=i)).strftime('%m-%d') for i in range(81)],
        horizon=81, stage_boundaries_zero_based=[0, 9, 37, 46, 62, 72, 81],
        calendar=json.loads((APP / 'source/_frozen/irrigation_ch6/_core/data/hongmiao93/case_model.json').read_text())['calendar'],
        geometry_maintained_source='/root/gamewar/resource_allocation_rl/applications/irrigation/reasonable89_20261109/visualization/data/layout.json',
        geometry_identity='schematic display; no hydraulic or measured-GIS adjacency')
    body = canonical(data) + b'\n'
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError('Fixed physics/input identity differs between evaluation jobs')
    else:
        temporary = root / ('fixed_inputs89.' + str(os.getpid()) + '.tmp')
        temporary.write_bytes(body)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != body:
                raise ValueError('Concurrent fixed-input identity mismatch')
        finally:
            temporary.unlink()
    return dict(path=str(path), sha256=sha(path))


def run_batch(env_config, seeds, *, actor=None, controller='full', weight=.8,
              training_budget=None, forcings=None, replay_actions=None,
              save_states=True, prefix_nodes=(46, 62), base_forcings=None, transform=None, step_callback=None, inference_callback=None):
    """Complete 81-day paths; all observations are available before each action."""
    envs = [environment(env_config, controller,
                        None if forcings is None else forcings[i],
                        base_trace=None if base_forcings is None else base_forcings[i],
                        transform=transform) for i in range(len(seeds))]
    encoders, observations, rows = [], [], []
    try:
        for env, seed in zip(envs, seeds):
            obs, info = env.reset(seed=int(seed))
            if info:
                raise AssertionError('No private reset feedback is allowed')
            enc = IrrigationEncoder(IrrigationSpec.from_env(env))
            enc.reset(obs)
            encoders.append(enc); observations.append(obs)
            digest = hashlib.sha256()
            prefix_update(digest, 'observation0', obs)
            rows.append(dict(actions=[], q=[], balance=[env.initial_budget_m3],
                B=[np.asarray(obs['reported_field_storage_mm']).copy()],
                weather=[], initial_previous_weather=np.asarray(obs['previous_weather']).copy(),
                prefix=digest, prefix_sha={}, task_fingerprint=env.task_fingerprint,
                water_balance=[], et_mm=[], percolation_mm=[], overflow_mm=[],
                completed_factor=[np.asarray(obs['reported_completed_yield_factor']).copy()],
                current_stage_ETa=[np.asarray(obs['reported_current_stage_et_mm']).copy()],
                current_stage_reference=[float(obs['reported_current_stage_reference_et_mm'])]))
        for day in range(81):
            if actor is not None and replay_actions is None:
                actions = []
                # Fixed one-path forward shape makes any representative extra branch
                # bitwise prefix comparable without rerunning unrelated weather rows.
                with torch.no_grad():
                    for enc in encoders:
                        state = model_input(enc, weight, training_budget)
                        combined = {k: np.stack([v]) for k,v in state.items()}
                        failed=True
                        try:
                            sampled_action=actor.sample(combined, deterministic=True)['actions'][0]
                            actions.append(sampled_action);failed=False
                        finally:
                            if inference_callback is not None:inference_callback(failed)
            elif replay_actions is not None:
                actions = replay_actions[:, day, :]
            else:
                raise ValueError('Only the registered Full actor or its original-action replay is allowed')
            for index, (env, enc, row) in enumerate(zip(envs, encoders, rows)):
                if day in prefix_nodes:
                    row['prefix_sha'][day] = row['prefix'].hexdigest()
                action = np.asarray(actions[index], dtype=np.float64).copy()
                if action.shape != (89,) or not np.isfinite(action).all() or np.any(action < 0):
                    raise ValueError('Invalid actual89 action')
                obs, reward, done, truncated, info = env.step(action)
                if step_callback is not None:step_callback(bool(done))
                if truncated or done != (day == 80):
                    raise AssertionError('Only the81st transition may end a path')
                if not done and (reward != 0 or info):
                    raise AssertionError('Intermediate private feedback exposed')
                enc.update(action, obs)
                row['actions'].append(action)
                row['q'].append(math.fsum(action.tolist()))
                row['balance'].append(float(enc.state()['remaining']))
                row['B'].append(np.asarray(obs['reported_field_storage_mm']).copy())
                row['weather'].append(np.asarray(obs['previous_weather']).copy())
                row['completed_factor'].append(np.asarray(obs['reported_completed_yield_factor']).copy())
                row['current_stage_ETa'].append(np.asarray(obs['reported_current_stage_et_mm']).copy())
                row['current_stage_reference'].append(float(obs['reported_current_stage_reference_et_mm']))
                observed = env.water_balance_observer()
                names = ('irrigation_m3','rain_m3','et_m3','drainage_m3','overflow_m3','water_mass_residual_m3')
                row['water_balance'].append([observed['daily_totals_m3'][k] for k in names])
                if save_states:
                    for k in ('et_mm','percolation_mm','overflow_mm'): row[k].append(observed[k].copy())
                prefix_update(row['prefix'], 'action' + str(day), action)
                prefix_update(row['prefix'], 'weather' + str(day), obs['previous_weather'])
                prefix_update(row['prefix'], 'observation' + str(day + 1), obs)
                observations[index] = obs
        output = []
        for env, seed, row in zip(envs, seeds, rows):
            metrics = env.terminal_metrics()
            public = env.public_observation()
            reference = np.asarray(public['reported_stage_reference_et_mm'], dtype=np.float64)
            if reference.shape != (89, 4) or not np.all(reference == reference[0]):
                raise AssertionError('Expected shared4-stage reference ET across all89 regions')
            arrays = dict(a_m3=np.asarray(row['actions'], dtype=np.float64),
                q_m3=np.asarray(row['q'], dtype=np.float64),
                balance_m3=np.asarray(row['balance'], dtype=np.float64),
                Y=np.asarray(metrics['quality'], dtype=np.float64),
                initial_previous_weather=np.asarray(row['initial_previous_weather'], dtype=np.float64),
                weather_trace=np.asarray(row['weather'], dtype=np.float64),
                stage_ETa_mm=np.asarray(public['reported_stage_et_mm'], dtype=np.float64).T.copy(),
                stage_reference_ETc_mm=reference[0].copy(),
                terminal_storage_mm=np.asarray(public['reported_field_storage_mm'],dtype=np.float64),
                water_balance_m3=np.asarray(row['water_balance'],dtype=np.float64),
                completed_yield_factor=np.asarray(row['completed_factor'],dtype=np.float64),
                current_stage_ETa_mm=np.asarray(row['current_stage_ETa'],dtype=np.float64),
                current_stage_reference_ETc_mm=np.asarray(row['current_stage_reference'],dtype=np.float64))
            if save_states:
                arrays['B_mm'] = np.asarray(row['B'], dtype=np.float64)
                for k in ('et_mm','percolation_mm','overflow_mm'):arrays[k]=np.asarray(row[k],dtype=np.float64)
            weather_hash = hashlib.sha256(arrays['initial_previous_weather'].tobytes() + arrays['weather_trace'].tobytes()).hexdigest()
            scalars = dict(C=float(metrics['C']), C90=float(metrics['C']), JA=float(metrics['JA']), J=float(metrics['J']),
                joint_score=float(metrics['joint_score']), CA90=float(metrics['area_coverage_rate']),
                area_coverage_rate=float(metrics['area_coverage_rate']),
                area_weighted_performance=float(metrics['area_weighted_performance']),
                water_used_m3=float(metrics['allocated_water_m3']),
                remaining_budget_m3=float(metrics['remaining_budget_m3']),
                coverage_count=int(metrics['coverage_count']))
            output.append(dict(arrays=arrays, metrics=scalars, weather_seed=int(seed),
                forcing_sha256=weather_hash, prefix_sha=row['prefix_sha'],
                task_fingerprint=row['task_fingerprint']))
        for env in envs[1:]:
            env.close()
        return output, envs[0]
    except BaseException:
        for env in envs:
            env.close()
        raise



def validate_formal_endpoint(payload,job):
    p=payload['protocol']
    if payload.get('format')!=ENDPOINT_FORMAT or p.get('case_id')!=CASE_ID:
        raise ValueError('Foreign endpoint format/case')
    if p.get('adapter')!=job['adapter'] or p.get('objective_order')!=['C','JA']:
        raise ValueError('Wrong adapter or critic target order')
    if float(p['tau'])!=.9 or float(p['weight'])!=float(job['training_weight']):
        raise ValueError('This task uses C90 and its actually selected training preference')
    if float(p.get('spatial_sigma_scale',1.))!=float(job.get('spatial_sigma_scale',1.)):
        raise ValueError('Endpoint exploration scale differs from selected application')
    if job.get('formal',True):
        if p.get('execution_operator')!=job['execution_operator'] or p.get('execution_description',{}).get('points_array_sha256')!=job['execution_points_array_sha256']:
            raise ValueError('Endpoint lacks explicit adopted execution identity')
        if p.get('training_execution')!={'operator':'native_stochastic_complete_joint_Gaussian','M64_in_training_density':False}:
            raise ValueError('Do not train density labels on averaged actions')
        if 'spatial_sigma_scale' not in p:raise ValueError('Formal exploration identity must be explicit')
        scale=float(job['spatial_sigma_scale'])
        if any(float(p[scope].get('spatial_sigma_scale',-1))!=scale for scope in ('env_config','spec','initialization')):
            raise ValueError('Formal env/spec/initialization exploration identity differs')
        if (p.get('formal_training') is not True or p.get('episodes')!=3200 or
            payload.get('episode')!=3200 or p.get('eval_every')!=128 or p.get('dev_paths')!=32 or
            p.get('schema')!='hongmiao89-reasonable-depth-independent-full-training-v1'):
            raise ValueError('Only genuine3200/128/32 formal endpoints may enter TEST')
        if (p.get('source_files_sha256')!=job['source_files_sha256'] or
            p.get('source_release')!=job['source_release'] or
            p.get('experiment_design_sha256')!=job['experiment_design_sha256']):
            raise ValueError('Endpoint and evaluator belong to different scientific releases')
        if job.get('expected_training_phase') is not None and p.get('phase')!=job['expected_training_phase']:
            raise ValueError('Wrong formal training phase')
    elif int(payload['episode'])!=int(job['expected_endpoint_episode']):
        raise ValueError('Engineering endpoint differs from explicit fixture budget')


def checkpoint_for_job(job,barrier):
    if barrier is None:return require_hy(job['checkpoint'])
    ident=job['training_identity']
    matches=[item for item in barrier['models'].values() if
        int(item['budget_m3'])==int(ident['budget_m3']) and int(item['seed'])==int(ident['seed'])]
    if len(matches)!=1:raise ValueError('No unique registered budget/seed endpoint')
    checkpoint=require_hy(matches[0]['model_path'])
    if job.get('checkpoint') is not None and require_hy(job['checkpoint'])!=checkpoint:
        raise ValueError('Job requested a different endpoint than its fixed matrix identity')
    return checkpoint


def load_policy(job,barrier):
    checkpoint=checkpoint_for_job(job,barrier)
    load_options=dict(device='cpu',expected_tau=.9,expected_weight=float(job['training_weight']),expected_adapter=job['adapter'])
    if 'expected_spatial_sigma_scale' in inspect.signature(common.load_endpoint).parameters:
        load_options['expected_spatial_sigma_scale']=float(job.get('spatial_sigma_scale',1.))
    elif job.get('formal',True) or float(job.get('spatial_sigma_scale',1.))!=1.:
        raise ValueError('Formal or scaled-noise models require the explicit exploration-aware loader')
    actor,critic,neural_spec,payload=common.load_endpoint(checkpoint,**load_options)
    validate_formal_endpoint(payload,job)
    p=payload['protocol'];kind=job['kind']
    if int(p['seed'])!=int(job['training_identity']['seed']) or float(p['env_config']['total_budget_m3'])!=float(job['training_identity']['budget_m3']):
        raise ValueError('Training identity does not match endpoint')
    policy_episode=int(payload['episode']);initialization_proof=None
    if kind=='initial':
        if float(p['env_config']['total_budget_m3'])!=4_000_000.:
            raise ValueError('Initial-policy comparison is predeclared only at4M')
        env=make_env(**p['env_config'])
        try:
            actor,critic,rebuilt=make_models_for_arm('full',FullSpendSpec.from_env(env),int(p['seed']))
            if rebuilt.to_dict()!=p['neural_spec']:
                raise ValueError('Initial reconstruction changed its adapter/neural interface')
            if common.model_hash(actor)!=p['initial_actor_sha256'] or common.model_hash(critic)!=p['initial_critic_sha256']:
                raise ValueError('Reconstructed initial parameters differ from actual training start')
            initialization_proof=dict(actor_sha256=common.model_hash(actor),critic_sha256=common.model_hash(critic),
                exact_actual_training_initialization=True,checkpoint_weights_not_used_for_actions=True)
        finally:env.close()
        policy_episode=0
    actor.eval();critic.eval();native_sha=common.model_hash(actor)
    inference_actor,description=make_execution_actor(actor,M64_OPERATOR)
    expected=dict(operator=job['execution_operator'],source_sha256=job['execution_operator_module_sha256'],
        points_array_sha256=job['execution_points_array_sha256'],point_count=job['execution_point_count'],event_dim=job['execution_event_dim'])
    if any(description.get(key)!=value for key,value in expected.items()):
        raise ValueError('Actual shared operator/points differ from frozen execution identity')
    recorded=p.get('execution_description') if job.get('formal',True) else p.get('execution_description',p.get('development_execution'))
    if recorded!=description:raise ValueError('Training DEV and deployment execution contracts differ')
    if common.model_hash(actor)!=native_sha:raise ValueError('Execution wrapping changed native actor')
    return inference_actor,neural_spec,payload,dict(reference_checkpoint=str(checkpoint),
        reference_checkpoint_sha256=sha(checkpoint),actor_sha256=native_sha,executed_actor_sha256=native_sha,
        executed_policy_episode=policy_episode,reference_endpoint_episode=int(payload['episode']),
        initialization_reconstruction=initialization_proof,execution_operator=M64_OPERATOR,execution_description=description)


def job_environment(job,protocol):
    kind=job['kind'];config=dict(protocol['env_config'])
    overrides=dict(job.get('env_config',{}))
    allowed={'total_budget_m3','weather_mode','historical_year','rain_multiplier'}
    if set(overrides)-allowed:raise ValueError('Unregistered physical/input override')
    config.update(overrides)
    training_budget=float(protocol['env_config']['total_budget_m3'])
    evaluation_budget=float(config['total_budget_m3'])
    if config.get('adapter')!=job['adapter']:raise ValueError('Environment adapter identity changed')
    if float(config.get('spatial_sigma_scale',1.))!=float(job.get('spatial_sigma_scale',1.)):
        raise ValueError('Environment exploration scale identity changed')
    if float(config['coverage_threshold'])!=.9 or float(config['coverage_weight'])!=float(job['training_weight']):
        raise ValueError('Do not relabel the learned objective')
    if kind in ('main','initial','history','feedback','descriptive_feedback') and (training_budget!=4_000_000. or evaluation_budget!=4_000_000.):
        raise ValueError('Main/initial/history/feedback must use the4M policy and supply')
    if kind=='trained_budget' and (training_budget not in (3_500_000.,4_500_000.) or evaluation_budget!=training_budget):
        raise ValueError('Budget-specific training must retain its actual training supply')
    if kind=='fixed_policy_budget' and (training_budget!=4_000_000. or evaluation_budget not in (3_500_000.,4_500_000.)):
        raise ValueError('Only the frozen4M policy may cross the two declared budgets')
    if kind=='history':
        if config.get('weather_mode')!='historical' or int(config.get('historical_year',0)) not in (2022,2024) or float(config.get('rain_multiplier',1.))!=1.:
            raise ValueError('Historical case must use the original declared daily rain record')
    elif config.get('weather_mode','persistent_drought')!='persistent_drought':
        raise ValueError('Dry task unexpectedly changed weather mode')
    return config,training_budget


def registered_weather(job,barrier):
    """This is the only function that opens sealed TEST values, after acceptance."""
    if barrier is None:
        if job.get('purpose')!='engineering_nonTEST' or job.get('weather_panel_file'):
            raise ValueError('Engineering must use explicit nonTEST seeds, never the sealed panel')
        seeds=[int(x) for x in job['weather_seeds']];ids=[int(x) for x in job.get('weather_ids',range(len(seeds)))]
    else:
        panel=require_hy(job['weather_panel_file'])
        if sha(panel)!=job['weather_panel_sha256']:raise ValueError('Sealed weather file digest changed')
        data=json.loads(panel.read_text())
        if data!=expected_test_panel(job):raise ValueError('Actual panel differs from the prospective namespace amendment')
        if data.get('schema')!='reasonable89-sealed-evaluation-weather-v1' or data.get('case_id')!=CASE_ID:
            raise ValueError('Foreign sealed weather panel')
        panels=data['panels'];test=panels['test128'];feedback=panels['feedback32']
        if (len(test['weather_seeds'])!=128 or test['weather_ids']!=list(range(128)) or
            feedback['weather_seeds']!=test['weather_seeds'][:32] or feedback['weather_ids']!=list(range(32))):
            raise ValueError('128 common test paths and their first32 feedback subset are required')
        role=job['weather_panel_role'];kind=job['kind']
        expected='feedback32' if kind=='feedback' else ('history'+str(int(job['env_config']['historical_year'])) if kind=='history' else 'test128')
        if role!=expected:raise ValueError('Weather role does not match scientific condition')
        selected=panels[role];seeds=[int(x) for x in selected['weather_seeds']];ids=[int(x) for x in selected['weather_ids']]
        if kind=='descriptive_feedback':
            path=require_hy(job['representative_case_path'])
            if path!=require_hy(job['run_root'])/'evaluation/representative_case.json' or sha(path)!=job['representative_case_sha256']:
                raise ValueError('Unregistered representative case record')
            case=json.loads(path.read_text())
            if case!=representative_from_main(job):raise ValueError('Representative must be the canonical pooled384 median case')
            index=int(case['weather_id'])
            if (index<32 or job.get('subset_indices')!=[index] or int(job['training_identity']['seed'])!=case['trained_seed'] or
                seeds[index]!=case['weather_seed']):raise ValueError('Descriptive branch is not the one registered representative outside32')
            seeds=[seeds[index]];ids=[ids[index]]
        expected_count=32 if kind=='feedback' else 1 if kind in ('history','descriptive_feedback') else 128
        if len(seeds)!=expected_count:raise ValueError('Wrong registered evaluation path count')
    if not seeds or len(seeds)!=len(ids) or len(set(seeds))!=len(seeds) or len(set(ids))!=len(ids):
        raise ValueError('Unique matched weather seeds and identifiers required')
    return seeds,ids



def representative_from_main(job):
    """Canonical stored-JA median from384 complete paths; only saved data, no rollout."""
    root=require_hy(job['run_root']);cases=[];sources=[];weather_reference=None;forcing_reference=None
    for seed in (6400,6401,6402):
        path=require_hy(root/'evaluation'/('main__seed'+str(seed))/'results.npz')
        done=json.loads(path.with_name('complete.json').read_text());meta=json.loads(path.with_name('metadata.json').read_text())
        if (done.get('status')!='COMPLETE' or done.get('formal') is not True or done.get('paths')!=128 or
            meta.get('execution_operator')!=M64_OPERATOR or meta.get('execution_description',{}).get('points_array_sha256')!=job['execution_points_array_sha256'] or
            meta.get('kind')!='main' or meta.get('trained_seed')!=seed or meta.get('executed_policy_episode')!=3200 or
            meta.get('adapter')!=job['adapter'] or meta.get('trained_weight')!=job['training_weight'] or
            meta.get('spatial_sigma_scale')!=job['spatial_sigma_scale'] or
            meta.get('job',{}).get('test_namespace_amendment_sha256')!=job['test_namespace_amendment_sha256'] or
            meta.get('scientific_source_release')!=job['source_release'] or meta.get('evaluation_files_sha256')!=job['evaluation_files_sha256'] or
            meta.get('evaluation_execution_sha256')!=sha(__file__) or meta.get('executed_actor_sha256')!=meta.get('actor_sha256')):
            raise ValueError('Representative requires all3 intact formal main128 records')
        if set(done.get('files_sha256',{}))!={'results.npz','metadata.json','execution_work.json'} or any(sha(path.parent/name)!=digest for name,digest in done['files_sha256'].items()):
            raise ValueError('Representative donor arrays/metadata changed')
        fixed=require_hy(meta['fixed_inputs']['path'])
        if sha(fixed)!=meta['fixed_inputs']['sha256']:raise ValueError('Representative true-area inputs changed')
        areas=np.asarray(json.loads(fixed.read_text())['area_ha'],dtype=np.float64)
        with np.load(path,allow_pickle=False) as data:
            JA=np.asarray(data['JA']);Y=np.asarray(data['Y']);ids=data['weather_id'];seeds=data['weather_seed'];forcing=data['forcing_sha256']
            if JA.shape!=(128,) or Y.shape!=(128,89) or not np.isfinite(JA).all() or ids.tolist()!=list(range(128)):
                raise ValueError('Incomplete main128 representative population')
            np.testing.assert_allclose(JA,(Y*areas).sum(1)/areas.sum(),rtol=0,atol=1e-12)
            if weather_reference is None:weather_reference=seeds.copy();forcing_reference=forcing.copy()
            else:
                np.testing.assert_array_equal(seeds,weather_reference);np.testing.assert_array_equal(forcing,forcing_reference)
            cases.extend(dict(JA=float(JA[i]),trained_seed=seed,weather_id=int(ids[i]),weather_seed=int(seeds[i])) for i in range(128))
        sources.append(dict(seed=seed,path=str(path),results_sha256=sha(path),metadata_sha256=sha(path.with_name('metadata.json')),executed_actor_sha256=meta['executed_actor_sha256']))
    median=float(np.median(np.asarray([row['JA'] for row in cases],dtype=np.float64)))
    chosen=min(cases,key=lambda row:(abs(row['JA']-median),row['trained_seed'],row['weather_id']))
    return dict(schema='reasonable89-canonical-main384-representative-v1',status='REGISTERED_FROM_COMPLETE_MAIN384',
        case_id=CASE_ID,**chosen,pooled_median_JA=median,selection='Min(abs(storedJA-pooledMedian),seed,weatherID); storedJA independently verified to1e-12',
        selection_population=384,source_main_results=sources,descriptive_feedback_required=chosen['weather_id']>=32,
        adapter=job['adapter'],training_weight=job['training_weight'],spatial_sigma_scale=job['spatial_sigma_scale'],source_release=job['source_release'],
        evaluation_files_sha256=job['evaluation_files_sha256'],representative_feedback_contract_sha256=job['representative_feedback_contract_sha256'])


def feedback_baseline(job,seeds,policy):
    path=require_hy(job['baseline_npz']);meta=json.loads(path.with_name('metadata.json').read_text());done=json.loads(path.with_name('complete.json').read_text())
    if (meta.get('case_id')!=CASE_ID or meta.get('kind')!='main' or
        (meta.get('executed_policy_episode')!=3200 and job.get('formal',True)) or
        meta.get('execution_operator')!=M64_OPERATOR or meta.get('execution_description')!=policy['execution_description'] or meta.get('actor_sha256')!=policy['actor_sha256'] or meta.get('adapter')!=job['adapter'] or
        meta.get('reference_checkpoint_sha256')!=policy['reference_checkpoint_sha256'] or
        meta.get('policy_batching')!='one_path_per_forward' or
        done.get('status')!='COMPLETE' or
        any(sha(path.parent/name)!=digest for name,digest in done.get('files_sha256',{}).items()) or
        not {'results.npz','metadata.json','execution_work.json'}.issubset(done.get('files_sha256',{}))):
        raise ValueError('Feedback needs the intact same-model main path')
    if job.get('formal',True):
        if done.get('formal') is not True or meta.get('formal') is not True:
            raise ValueError('Formal feedback cannot use an engineering baseline')
        if (meta.get('scientific_source_release')!=job['source_release'] or
            meta.get('scientific_source_files_sha256')!=job['source_files_sha256'] or
            meta.get('evaluation_execution_sha256')!=sha(__file__) or
            meta.get('evaluation_contract_sha256')!=sha(CONTRACT) or
            meta.get('evaluation_files_sha256')!=job['evaluation_files_sha256'] or
            meta.get('job',{}).get('experiment_design_sha256')!=job['experiment_design_sha256'] or
            meta.get('job',{}).get('weather_panel_sha256')!=job['weather_panel_sha256'] or
            meta.get('job',{}).get('test_namespace_amendment_sha256')!=job['test_namespace_amendment_sha256'] or
            meta.get('run_root')!=str(require_hy(job['run_root']))):
            raise ValueError('Formal baseline source/panel/design/capsule identity differs')
    with np.load(path,allow_pickle=False) as data:
        indices=[]
        for seed in seeds:
            match=np.flatnonzero(data['weather_seed']==seed)
            if len(match)!=1:raise ValueError('No unique same-weather baseline')
            indices.append(int(match[0]))
        baseline={key:np.asarray(data[key])[indices].copy() for key in
            ('a_m3','weather_trace','initial_previous_weather','prefix_sha','weather_seed')}
    if job.get('formal',True):
        expected_indices=job['subset_indices'] if job['kind']=='descriptive_feedback' else list(range(32))
        if indices!=expected_indices:raise ValueError('Feedback baseline indices differ from registered statistical/descriptive role')
    if 'baseline_indices' in job and indices!=job['baseline_indices']:
        raise ValueError('Registered baseline row indices differ')
    multiplier=float(job['et0_multiplier'])
    if multiplier not in ((.85,1.15) if job.get('formal',True) else (.85,1.,1.15)) or job.get('changed_days',[62,63,64])!=[62,63,64]:
        raise ValueError('Only the declared three-day demand change is allowed')
    if job.get('controller_mode') not in ('feedback','replay'):
        raise ValueError('Need frozen-policy feedback or the original action sequence')
    transform=None if multiplier==1. else dict(schema='explicit-et0-demand-shock-v1',days=[62,63,64],et0_multiplier=multiplier)
    modified=baseline['weather_trace'].copy();modified[:,[62,63,64],1]*=multiplier
    return baseline,modified,transform,dict(path=str(path),sha256=sha(path),metadata_sha256=sha(path.with_name('metadata.json')),indices=indices)


def validate_arrays(arrays,fixed,budget,weight):
    """Recompute physical/resource/terminal accounts from recorded float64 data."""
    n=len(arrays['weather_seed'])
    shapes=dict(a_m3=(n,81,89),q_m3=(n,81),balance_m3=(n,82),B_mm=(n,82,89),
        Y=(n,89),et_mm=(n,81,89),percolation_mm=(n,81,89),overflow_mm=(n,81,89),
        completed_yield_factor=(n,82,89),current_stage_ETa_mm=(n,82,89),
        current_stage_reference_ETc_mm=(n,82),stage_ETa_mm=(n,4,89),
        stage_reference_ETc_mm=(n,4),weather_trace=(n,81,2),initial_previous_weather=(n,2),
        water_balance_m3=(n,81,6),terminal_storage_mm=(n,89))
    for key,shape in shapes.items():
        value=arrays[key]
        if value.shape!=shape or value.dtype!=np.float64 or not np.isfinite(value).all():
            raise ValueError('Incomplete/nonfloat64/invalid recorded array:'+key)
    for key in ('a_m3','B_mm','Y','et_mm','percolation_mm','overflow_mm','stage_ETa_mm'):
        if np.any(arrays[key]<0):raise ValueError('Negative physical/terminal value:'+key)
    if arrays['region_ids'].tolist()!=fixed['region_ids'] or len(fixed['region_ids'])!=89:
        raise ValueError('Full original89 identities/order must be retained')
    q=np.asarray([[math.fsum(row.tolist()) for row in trajectory] for trajectory in arrays['a_m3']])
    np.testing.assert_allclose(q,arrays['q_m3'],rtol=0,atol=1e-9)
    np.testing.assert_allclose(arrays['balance_m3'][:,0],budget,rtol=0,atol=1e-9)
    np.testing.assert_allclose(arrays['balance_m3'][:,:-1]-q,arrays['balance_m3'][:,1:],rtol=0,atol=1e-7)
    np.testing.assert_allclose(arrays['balance_m3'][:,-1],0,rtol=0,atol=1e-7)
    if np.any(arrays['balance_m3']< -1e-7):raise ValueError('Negative balance')
    area=np.asarray(fixed['area_ha'],dtype=np.float64);Y=arrays['Y'];covered=Y>=.9
    for name,expected in dict(C=covered.mean(1),C90=covered.mean(1),J=Y.mean(1),
        JA=(Y*area).sum(1)/area.sum(),CA90=(covered*area).sum(1)/area.sum(),
        area_weighted_performance=(Y*area).sum(1)/area.sum(),area_coverage_rate=(covered*area).sum(1)/area.sum()).items():
        np.testing.assert_allclose(arrays[name],expected,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(arrays['coverage_count'],covered.sum(1))
    np.testing.assert_allclose(arrays['joint_score'],weight*arrays['C']+(1-weight)*arrays['JA'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(arrays['water_used_m3'],q.sum(1),rtol=0,atol=1e-7)
    np.testing.assert_allclose(arrays['remaining_budget_m3'],arrays['balance_m3'][:,-1],rtol=0,atol=1e-7)
    if np.any(Y>1):raise ValueError('Relative yield exceeds model reference')
    reference=arrays['stage_reference_ETc_mm'][:,:,None]
    ratio=np.divide(arrays['stage_ETa_mm'],reference,out=np.ones_like(arrays['stage_ETa_mm']),where=reference>0)
    if np.any(ratio< -1e-12) or np.any(ratio>1+1e-8):raise ValueError('Stage actualET exceeds its reference')
    exponents=np.asarray(fixed['yield_lambdas'],dtype=np.float64).T[None,:,:]
    projected=np.prod(np.clip(ratio,0,1)**exponents,axis=1)
    np.testing.assert_allclose(Y,projected,rtol=0,atol=1e-10)
    np.testing.assert_allclose(arrays['completed_yield_factor'][:,-1],Y,rtol=0,atol=1e-10)
    predicted=(arrays['B_mm'][:,:-1]+arrays['a_m3']/(10*area)+arrays['weather_trace'][:,:,0,None]
        -arrays['et_mm']-arrays['percolation_mm']-arrays['overflow_mm'])
    np.testing.assert_allclose(arrays['B_mm'][:,1:],predicted,rtol=1e-11,atol=1e-8)
    np.testing.assert_allclose(arrays['terminal_storage_mm'],arrays['B_mm'][:,-1],rtol=0,atol=1e-10)
    totals=np.stack([q,arrays['weather_trace'][:,:,0]*area.sum()*10,
        (arrays['et_mm']*area).sum(2)*10,(arrays['percolation_mm']*area).sum(2)*10,
        (arrays['overflow_mm']*area).sum(2)*10],axis=2)
    np.testing.assert_allclose(arrays['water_balance_m3'][:,:,:5],totals,rtol=1e-11,atol=1e-6)
    np.testing.assert_allclose(arrays['water_balance_m3'][:,:,5],0,rtol=0,atol=1e-6)
    for i in range(n):
        forcing=hashlib.sha256(arrays['initial_previous_weather'][i].tobytes()+arrays['weather_trace'][i].tobytes()).hexdigest()
        if forcing!=str(arrays['forcing_sha256'][i]):raise ValueError('Actual realized forcing hash mismatch')


def _run_job(job,work):
    start=time.monotonic();formal=bool(job.get('formal',True));kind=job['kind']
    if kind not in KINDS or job.get('controller','full')!='full':raise ValueError('No old rule/foreign study jobs')
    barrier=verify_test_barrier(job)
    bootstrap(job)
    actor,neural,payload,policy=load_policy(job,barrier);protocol=payload['protocol']
    config,training_budget=job_environment(job,protocol)
    seeds,ids=registered_weather(job,barrier)
    out=require_hy(job['output_dir']);root=require_hy(job['run_root'])
    if not out.is_relative_to(root):raise ValueError('Output belongs to another scientific runroot')
    if not formal and not out.is_relative_to(root/'engineering'):
        raise ValueError('Engineering paths must remain outside the formal evaluation pool')
    digest=hashlib.sha256(canonical(job)).hexdigest();completed=out/'complete.json'
    if completed.exists():
        done=json.loads(completed.read_text())
        if done['job_sha256']!=digest or any(sha(out/name)!=value for name,value in done['files_sha256'].items()):
            raise ValueError('Existing result differs; retain it and use another attempt directory')
        return done
    if (out/'results.npz').exists() or (out/'failure.json').exists():
        raise FileExistsError('Failed/incomplete attempt cannot be overwritten')
    baseline=modified=transform=baseline_proof=None
    if kind in ('feedback','descriptive_feedback'):baseline,modified,transform,baseline_proof=feedback_baseline(job,seeds,policy)
    out.mkdir(parents=True,exist_ok=True)
    def count_step(done):
        work['environment_steps']+=1;work['paths']+=int(done)
    def count_inference(failed=False):
        work['model_inferences']=actor.parameter_evaluations
        work['parameter_evaluations']=actor.parameter_evaluations
        work['raw_feasible_action_vectors']=actor.feasible_vectors
        work['policy_parameter_batch_calls']=actor.batch_calls
        work['incomplete_operator_calls']+=int(failed)
    rows_all=[];fixed=None;size=int(job.get('batch_size',8))
    if not 1<=size<=16:raise ValueError('Environment batch size must lie in1..16')
    for first in range(0,len(seeds),size):
        stop=min(first+size,len(seeds))
        rows,env=run_batch(config,seeds[first:stop],actor=actor,weight=float(protocol['weight']),
            training_budget=training_budget,forcings=None if baseline is None else modified[first:stop],
            base_forcings=None if baseline is None else baseline['weather_trace'][first:stop],transform=transform,
            replay_actions=baseline['a_m3'][first:stop] if baseline is not None and job['controller_mode']=='replay' else None,
            save_states=True,step_callback=count_step,inference_callback=count_inference)
        if fixed is None:
            fixed=shared_fixed_inputs(env,root);dates=[(env.scenario.start_date+timedelta(days=i)).isoformat() for i in range(81)];region_ids=list(env.region_ids)
        for row in rows:
            if baseline is not None:
                index=len(rows_all)
                np.testing.assert_array_equal(row['arrays']['initial_previous_weather'],baseline['initial_previous_weather'][index])
                np.testing.assert_array_equal(row['arrays']['a_m3'][:63],baseline['a_m3'][index,:63])
                if row['prefix_sha'][62]!=str(baseline['prefix_sha'][index,1]):raise AssertionError('Pre-shock prefix changed')
                np.testing.assert_array_equal(row['arrays']['weather_trace'],modified[index])
            rows_all.append(row)
        env.close()
        write_json(out/'progress.json',dict(status='running',completed_paths=len(rows_all),total_paths=len(seeds),actual_work=work,learning_updates=0))
    arrays={key:np.stack([row['arrays'][key] for row in rows_all]) for key in rows_all[0]['arrays']}
    for metric in METRICS:arrays[metric]=np.asarray([r['metrics'][metric] for r in rows_all],dtype=np.int64 if metric=='coverage_count' else np.float64)
    arrays.update(weather_seed=np.asarray(seeds,dtype=np.int64),weather_id=np.asarray(ids,dtype=np.int64),
        forcing_sha256=np.asarray([row['forcing_sha256'] for row in rows_all]),task_fingerprint=np.asarray([row['task_fingerprint'] for row in rows_all]),
        prefix_sha=np.asarray([[row['prefix_sha'][node] for node in (46,62)] for row in rows_all]),region_ids=np.asarray(region_ids),date=np.asarray(dates),
        calendar_stage_index=np.searchsorted(np.asarray([9,37,46,62,72,81]),np.arange(82),side='right').astype(np.int64),
        water_balance_names=np.asarray(('irrigation_m3','rain_m3','et_m3','drainage_m3','overflow_m3','water_mass_residual_m3')))
    expected_forwards=0 if baseline is not None and job['controller_mode']=='replay' else len(seeds)*81
    if actor.parameter_evaluations!=expected_forwards or actor.feasible_vectors!=64*expected_forwards or actor.batch_calls!=expected_forwards:
        raise AssertionError('Shared M64 actual counters differ from registered statewise execution')
    if work!=dict(environment_steps=len(seeds)*81,paths=len(seeds),model_inferences=expected_forwards,parameter_evaluations=expected_forwards,raw_feasible_action_vectors=64*expected_forwards,policy_parameter_batch_calls=expected_forwards,incomplete_operator_calls=0):
        raise AssertionError('Actual callback work differs from complete registered paths')
    if common.model_hash(actor.actor)!=policy['executed_actor_sha256']:raise ValueError('Shared execution mutated native actor parameters')
    validate_arrays(arrays,json.loads(Path(fixed['path']).read_text()),float(config['total_budget_m3']),float(protocol['weight']))
    temporary=out/('results.'+str(os.getpid())+'.npz');np.savez_compressed(temporary,**arrays);os.replace(temporary,out/'results.npz')
    write_json(out/'execution_work.json',dict(status='COMPLETE',**work,learning_updates=0,model_updates=0,formal=formal))
    metadata=dict(schema='reasonable89-fixed3200-frozen-evaluation-v1',case_id=CASE_ID,domain=DOMAIN,adapter=job['adapter'],
        job=job,job_sha256=digest,controller='full',kind=kind,region_count=89,region_ids=region_ids,
        trained_seed=int(protocol['seed']),trained_tau=float(protocol['tau']),trained_weight=float(protocol['weight']),
        spatial_sigma_scale=float(protocol.get('spatial_sigma_scale',1.)),
        model_training_budget_m3=training_budget,evaluation_budget_m3=float(config['total_budget_m3']),
        scoring_tau=float(protocol['tau']),preference_weight=float(protocol['weight']),objective_order=['C90','JA'],J_role='unweighted auxiliary only',
        **policy,model_sha256=policy['reference_checkpoint_sha256'],fixed_inputs=fixed,
        normalization='input divisor fixed to actual training budget; physical ledger/bounds use evaluation supply',
        policy_batching='one_path_per_forward',training_spec=protocol['spec'],neural_spec=protocol['neural_spec'],
        initialization_provenance=protocol['initialization'],scientific_source_release=protocol.get('source_release'),
        scientific_source_files_sha256=protocol['source_files_sha256'],evaluation_files_sha256=job.get('evaluation_files_sha256'),
        evaluation_execution_sha256=sha(__file__),evaluation_contract_sha256=sha(CONTRACT),
        barrier=None if barrier is None else {key:value for key,value in barrier.items() if key!='models'},
        run_root=str(root),formal=formal,paths=len(seeds),environment_steps=work['environment_steps'],learning_updates=0,model_updates=0,
        deterministic=True,execution_meaning='fixed64 native feasible physical actions averaged before one physical transition; not latentmean or exact stochastic expectedreturn',inference_device='cpu',complete_region_fluxes_saved=True,all82_state_boundaries_saved=True,
        no_cross_forcing_state_restore=True,actual_work={**work,'parameter_evaluations':actor.parameter_evaluations,'raw_feasible_action_vectors':actor.feasible_vectors,'policy_parameter_batch_calls':actor.batch_calls},seconds=time.monotonic()-start,
        derived_public_diagnostics=dict(total_storage_depth_ratio='B_mm/Z_mm; includes ponding, not root-only theta',
            root_theta='min(B_mm/Z_mm,theta_sat)',no_stress_deficit_mm='max(theta_crit*Z_mm-B_mm,0)',irrigation_depth_mm='a_m3/(10*area_ha)'),
        runtime_versions=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__),
        test_namespace_amendment_path=job.get('test_namespace_amendment_path'),test_namespace_amendment_sha256=job.get('test_namespace_amendment_sha256'),
        scientific_manifest_path=job.get('scientific_manifest_path'),scientific_manifest_sha256=job.get('scientific_manifest_sha256'),
        arrays={key:dict(shape=list(value.shape),dtype=str(value.dtype)) for key,value in arrays.items()})
    if baseline is not None:
        metadata.update(baseline=baseline_proof,forcing_transform=transform,first_possible_feedback_day_zero_based=63,
            exact_action_prefix_through_day62=True,replay_role='own original-action sequence under changed forcing; no optimal-open-loop or external-rule claim')
    write_json(out/'metadata.json',metadata)
    done=dict(status='COMPLETE',schema='reasonable89-evaluation-completion-v1',formal=formal,job_sha256=digest,
        paths=work['paths'],environment_steps=work['environment_steps'],model_inferences=work['model_inferences'],parameter_evaluations=actor.parameter_evaluations,raw_feasible_action_vectors=actor.feasible_vectors,policy_parameter_batch_calls=actor.batch_calls,learning_updates=0,
        files_sha256={name:sha(out/name) for name in ('results.npz','metadata.json','execution_work.json')})
    write_json(completed,done);write_json(out/'progress.json',dict(status='complete',completed_paths=len(seeds),total_paths=len(seeds)))
    return done


def run_job(job):
    work=dict(environment_steps=0,paths=0,model_inferences=0,parameter_evaluations=0,raw_feasible_action_vectors=0,policy_parameter_batch_calls=0,incomplete_operator_calls=0)
    try:return _run_job(job,work)
    except BaseException:
        out=require_hy(job['output_dir'])
        if not out.exists() and not work['environment_steps']:
            root=require_hy(job['run_root'])/'audit/evaluation_gate_rejections';root.mkdir(parents=True,exist_ok=True)
            target=root/('rejection_'+str(os.getpid())+'_'+str(time.time_ns())+'.json')
        else:out.mkdir(parents=True,exist_ok=True);target=out/'failure.json'
        write_json(target,dict(status='FAILED_OR_REJECTED',actual_work=work,operator_work_exact=False,work_exact=False,counter_semantics='failed-attempt callback counters are conservative lower bounds; unreturned native or physical calls may contain additional work',learning_updates=0,
            formal=bool(job.get('formal',True)),job_sha256=hashlib.sha256(canonical(job)).hexdigest(),
            evaluation_execution_sha256=sha(__file__),traceback=traceback.format_exc()))
        raise


def config_templates():
    """Unbound36-job matrix; no sealed weather values are read or generated."""
    jobs=[]
    def add(name,kind,budget,seed,**extra):
        job=dict(job_id=name,kind=kind,controller='full',formal=True,run_root=str(RUN),
            training_identity=dict(budget_m3=budget,seed=seed),adapter='__SELECTED_AFTER_DEV__',training_weight='__SELECTED_TRAINING_WEIGHT__',spatial_sigma_scale='__SELECTED_SPATIAL_SIGMA_SCALE__',
            execution_operator=M64_OPERATOR,execution_operator_module_sha256='__FORMAL_OPERATOR_MODULE_SHA256__',execution_points_array_sha256=M64_POINTS_SHA,execution_point_count=64,execution_event_dim=49,
            selected_application_path=str(RUN/'frozen_code/protocol/selected_application.json'),selected_application_sha256='__SELECTED_APPLICATION_SHA256__',
            formal_registration_path=str(RUN/'protocol/formal_registration.json'),formal_registration_sha256='__FORMAL_REGISTRATION_SHA256__',
            frozen_code_root='__FORMAL_FROZEN_ROOT__',source_files_sha256={},source_release='__FORMAL_SOURCE_RELEASE__',
            scientific_manifest_path=str(RUN/'protocol/scientific_source_manifest.json'),scientific_manifest_sha256='__FORMAL_MANIFEST_SHA256__',
            experiment_design_path=str(STUDY/'protocol/experiment_design.json'),experiment_design_sha256='__FROZEN_DESIGN_SHA256__',
            evaluation_source_sha256='__EVALUATOR_SHA256__',evaluation_files_sha256={},
            weather_panel_file=str(RUN/'protocol/test_weather_panel.json'),weather_panel_sha256='__SEALED_PANEL_SHA256__',
            test_namespace_amendment_path=str(STUDY/'protocol/TEST_namespace_amendment.json'),test_namespace_amendment_sha256=TEST_NAMESPACE_AMENDMENT_SHA,
            weather_panel_role='test128',env_config={},batch_size=8,
            output_dir=str(RUN/'evaluation'/name),checkpoint=None)
        job.update(extra);jobs.append(job)
    for budget in (3500000,4000000,4500000):
        for seed in (6400,6401,6402):add(('main' if budget==4000000 else 'trained_budget_'+str(budget))+'__seed'+str(seed),'main' if budget==4000000 else 'trained_budget',budget,seed)
    for seed in (6400,6401,6402):
        add('initial__seed'+str(seed),'initial',4000000,seed)
        for budget in (3500000,4500000):add('fixed_budget_'+str(budget)+'__seed'+str(seed),'fixed_policy_budget',4000000,seed,env_config=dict(total_budget_m3=budget))
        for multiplier,label in ((.85,'low'),(1.15,'high')):
            for mode in ('feedback','replay'):
                add('feedback_'+label+'_'+mode+'__seed'+str(seed),'feedback',4000000,seed,
                    weather_panel_role='feedback32',et0_multiplier=multiplier,changed_days=[62,63,64],controller_mode=mode,
                    baseline_npz=str(RUN/'evaluation'/('main__seed'+str(seed))/'results.npz'),baseline_indices=list(range(32)))
        for year in (2022,2024):add('history_'+str(year)+'__seed'+str(seed),'history',4000000,seed,
            weather_panel_role='history'+str(year),env_config=dict(weather_mode='historical',historical_year=year,rain_multiplier=1.))
    if len(jobs)!=36:raise AssertionError('Wrong registered matrix')
    return jobs


def prepare_templates():
    folder=RUN/'configs/evaluation_templates';folder.mkdir(parents=True,exist_ok=True)
    jobs=config_templates();paths=[]
    for job in jobs:
        path=folder/(job['job_id']+'.json');body=canonical(job)+b'\n'
        if path.exists() and path.read_bytes()!=body:raise FileExistsError('Retain earlier configuration release:'+str(path))
        path.write_bytes(body);paths.append(str(path))
    write_json(folder/'matrix.json',dict(status='UNBOUND_PREPARED_NO_TEST_OPENED',jobs=paths,
        matrix_counts=dict(trained=9,initial=3,fixed_budget=6,feedback=12,history=6),complete_paths=2694,
        expected_environment_steps=2694*81,read_TEST_values=False,environment_steps_executed=0))
    return dict(status='TEMPLATES_ONLY_NOT_RELEASED',jobs=len(jobs),paths=2694,folder=str(folder))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path);parser.add_argument('--prepare-templates',action='store_true');args=parser.parse_args()
    if args.prepare_templates:result=prepare_templates()
    elif args.config:result=run_job(json.loads(require_hy(args.config).read_text()))
    else:parser.error('Provide --config or --prepare-templates')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
