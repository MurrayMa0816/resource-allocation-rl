#!/usr/bin/env python3
"""Bind the predeclared evaluation matrix without reading sealed TEST values."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('reasonable89_evaluation',HERE/'evaluate.py')
ev=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(ev)
ROOT=ev.RUN;STUDY=ev.STUDY
DESIGN_SHA='dee62d67416b0c5af5d111da68e581659f1fe2c655cbffc1646f4747ceb35e69'
AMENDMENT_SHA='21814bcd3633103382daccbc36f4f8a42f0cff9326be61fc1ddd4af36339b185'
NOISE_SHA='41a2a8283762b82e8c69f9083a9b3c3dd1ad39af94ce5072d761cb5042db6192'
MIDPOINT_SHA='761da1418ff05e5b90952946441c4db724f18a5193ff345e4b405a45a0f5924e'
REPRESENTATIVE_SHA='cacd702470f079a484d697e8bb0eaf78aa350539931c2ef382107fcc4c5d5b59'
ADOPTION_SHA='0f05b4fae56f451f9fa53f3dc238d0d38b249295d887b97da9eb30f6d3dcc7a5'
CAPSULE_NAMES=('evaluate.py','evaluation_contract.json','prepare_evaluation.py','run_evaluation.py')


def immutable_json(path,value):
    path=ev.require_hy(path);path.parent.mkdir(parents=True,exist_ok=True)
    body=ev.canonical(value)+b'\n'
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid())+'.'+str(ev.time.time_ns()))
    try:
        with temporary.open('xb') as stream:
            stream.write(body);stream.flush();os.fsync(stream.fileno())
        try:os.link(temporary,path)
        except FileExistsError:
            if path.read_bytes()!=body:raise FileExistsError('Existing immutable evaluation registration differs:'+str(path))
    finally:
        if temporary.exists():temporary.unlink()
    return path


def read_identity():
    """Only source/protocol metadata and panel byte hash; no panel JSON parse."""
    design=STUDY/'protocol/experiment_design.json';amend=STUDY/'protocol/preference_dev_amendment.json';noise=STUDY/'protocol/noise_dev_amendment.json'
    if ev.sha(design)!=DESIGN_SHA or ev.sha(amend)!=AMENDMENT_SHA or ev.sha(noise)!=NOISE_SHA or ev.sha(STUDY/'protocol/noise_midpoint_dev_amendment.json')!=MIDPOINT_SHA or ev.sha(STUDY/'protocol/representative_feedback_contract.json')!=REPRESENTATIVE_SHA:
        raise ValueError('Original design or DEV-only preference amendment changed')
    manifest_path=ROOT/'protocol/scientific_source_manifest.json';manifest=json.loads(manifest_path.read_text())
    frozen=ev.require_hy(manifest['frozen_code_root'])
    if frozen!=ROOT/'frozen_code':raise ValueError('Expected this formal frozen_code root')
    expected=manifest.get('source_files_sha256',manifest.get('files_sha256'));release=manifest.get('source_release',manifest.get('release'))
    if not isinstance(expected,dict) or len(expected)!=140 or not isinstance(release,str):
        raise ValueError('A complete formal140 scientific source release is required')
    found=[]
    for scope in ('source','method'):
        found.extend(p for p in (frozen/scope).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.csv'))
    found.extend(frozen/'execution'/name for name in ('common.py','worker_train.py'))
    actual={str(p.relative_to(frozen)):ev.sha(p) for p in sorted(found)}
    if actual!=expected:raise ValueError('Formal scientific closure differs from its manifest')
    selections=[STUDY/'protocol/selected_application.json',ROOT/'protocol/selected_application.json',frozen/'protocol/selected_application.json']
    hashes=[ev.sha(p) for p in selections]
    if len(set(hashes))!=1:raise ValueError('Selected-application copies are not identical bytes')
    chosen=json.loads(selections[-1].read_text());adapter=chosen['adapter'];weight=float(chosen['training_weight']);scale=float(chosen['spatial_sigma_scale'])
    if adapter not in ('unit','depth_area','depth_area_physical') or weight not in (.2,.5,.8) or scale not in (.25,.5,1.) or (adapter!='depth_area_physical' and scale!=1.):
        raise ValueError('Foreign DEV application/preference selection')
    adoption_path=STUDY/'protocol/application_adoption_decision.json'
    if ev.sha(adoption_path)!=ADOPTION_SHA:raise ValueError('Configuration adoption decision changed')
    adopted=json.loads(adoption_path.read_text())['adopted']
    if any(chosen[key]!=adopted[key] for key in ('adapter','training_weight','spatial_sigma_scale')):
        raise ValueError('Selected application differs from adopted physical8/preference.2/sigma1 configuration')
    operator=chosen['execution_operator'];description=chosen['execution_description']
    module_sha=expected.get('source/hongmiao89/execution_operator.py')
    if (operator!=ev.M64_OPERATOR or adopted['deployment_execution']!=operator or
        description.get('operator')!=operator or description.get('source_sha256')!=module_sha or
        description.get('points_array_sha256')!=ev.M64_POINTS_SHA or description.get('point_count')!=64 or
        description.get('event_dim')!=49 or description.get('seed')!=2026110964):
        raise ValueError('Only the shared fixed64 adopted execution descriptor may enter TEST')
    registration=ev.require_hy(ROOT/'protocol/formal_registration.json')
    if registration!=ev.require_hy(ROOT/'protocol/training_registration.json'):
        raise ValueError('Registration compatibility alias must identify canonical training registration')
    reg=json.loads(registration.read_text())
    if reg.get('adapter')!=adapter or reg.get('training_weight')!=weight or reg.get('spatial_sigma_scale')!=scale or reg.get('execution_operator')!=operator or reg.get('execution_description')!=description:
        raise ValueError('Selected and formally registered training identity differ')
    for key,value in [('source_release',release),('preference_dev_amendment_sha256',AMENDMENT_SHA),('noise_dev_amendment_sha256',NOISE_SHA),('noise_midpoint_dev_amendment_sha256',MIDPOINT_SHA)]:
        if key in reg and reg[key]!=value:raise ValueError('Formal registration mismatch:'+key)
    # The actual formal worker design may be the recorded final merged design.
    # Original design plus all DEV amendments remain individually SHA-bound.
    actual_design=ev.require_hy(reg.get('experiment_design_path',design));actual_design_sha=ev.sha(actual_design)
    if reg.get('experiment_design_sha256',actual_design_sha)!=actual_design_sha:
        raise ValueError('Formal worker design path/SHA differs from registration')
    selected_final=chosen.get('final_design')
    if selected_final is not None:
        if isinstance(selected_final,dict):
            final_path=selected_final.get('path',selected_final.get('experiment_design_path'))
            final_sha=selected_final.get('sha256',selected_final.get('experiment_design_sha256'))
            if final_path is None or final_sha is None or ev.require_hy(final_path)!=actual_design or final_sha!=actual_design_sha:
                raise ValueError('selection.final_design differs from formal worker design')
        elif isinstance(selected_final,str):
            if selected_final!=actual_design_sha and ev.require_hy(selected_final)!=actual_design:
                raise ValueError('selection.final_design differs from formal worker design')
        else:raise ValueError('Unrecognized selected final_design contract')
    panel=ROOT/'protocol/test_weather_panel.json'
    # A byte digest does not deserialize or use the sealed weather values.
    panel_sha=ev.sha(panel)
    capsule={str(HERE/name):ev.sha(HERE/name) for name in CAPSULE_NAMES}
    identity=dict(adapter=adapter,training_weight=weight,spatial_sigma_scale=scale,execution_operator=operator,execution_description=description,
        execution_operator_module_sha256=module_sha,execution_points_array_sha256=ev.M64_POINTS_SHA,execution_point_count=64,execution_event_dim=49,
        application_adoption_decision_path=str(adoption_path),application_adoption_decision_sha256=ADOPTION_SHA,frozen_code_root=str(frozen),source_files_sha256=expected,
        source_release=release,scientific_manifest_path=str(manifest_path),scientific_manifest_sha256=ev.sha(manifest_path),
        selected_application_path=str(selections[-1]),selected_application_sha256=hashes[0],
        formal_registration_path=str(ROOT/'protocol/formal_registration.json'),formal_registration_sha256=ev.sha(registration),
        experiment_design_path=str(actual_design),experiment_design_sha256=actual_design_sha,parent_experiment_design_path=str(design),parent_experiment_design_sha256=DESIGN_SHA,
        preference_dev_amendment_path=str(amend),preference_dev_amendment_sha256=AMENDMENT_SHA,noise_dev_amendment_path=str(noise),noise_dev_amendment_sha256=NOISE_SHA,noise_midpoint_dev_amendment_path=str(STUDY/'protocol/noise_midpoint_dev_amendment.json'),noise_midpoint_dev_amendment_sha256=MIDPOINT_SHA,
        representative_feedback_contract_path=str(STUDY/'protocol/representative_feedback_contract.json'),representative_feedback_contract_sha256=REPRESENTATIVE_SHA,
        weather_panel_file=str(panel),weather_panel_sha256=panel_sha,evaluation_source_sha256=ev.sha(HERE/'evaluate.py'),evaluation_files_sha256=capsule,
        test_namespace_amendment_path=str(STUDY/'protocol/TEST_namespace_amendment.json'),test_namespace_amendment_sha256=ev.TEST_NAMESPACE_AMENDMENT_SHA)
    # Hash a pure expected metadata payload rather than deserializing sealed TEST values.
    expected_panel_sha=hashlib.sha256(ev.canonical(ev.expected_test_panel(identity))+b'\n').hexdigest()
    if panel_sha!=expected_panel_sha:raise ValueError('Sealed panel metadata do not match the namespace-only amendment')
    return identity


def bound_jobs(identity):
    """Pure deterministic binding, with explicit first-attempt output directories."""
    jobs=[]
    for template in ev.config_templates():
        job={**template,**identity};job['output_dir']=str(ROOT/'evaluation'/job['job_id']/'attempt0')
        job['training_barrier']=str(ROOT/'protocol/all9_endpoints_verified.json')
        if any('__SELECTED' in str(v) or '__FORMAL' in str(v) or '__SEALED' in str(v) or '__FROZEN' in str(v) or '__EVALUATOR' in str(v) for v in job.values()):
            raise ValueError('Unbound field in formal evaluation job')
        jobs.append(job)
    if len(jobs)!=36:raise AssertionError('Matrix changed')
    return jobs


def prepare(write=False):
    identity=read_identity();jobs=bound_jobs(identity);folder=ROOT/'configs/evaluation'
    paths=[folder/(j['job_id']+'.json') for j in jobs]
    matrix=dict(schema='reasonable89-bound-evaluation-matrix-v1',status='BOUND_AWAITING_ROOT_TEST_RELEASE',
        run_root=str(ROOT),case_id=ev.CASE_ID,adapter=identity['adapter'],training_weight=identity['training_weight'],spatial_sigma_scale=identity['spatial_sigma_scale'],execution_operator=identity['execution_operator'],execution_description=identity['execution_description'],
        complete_paths=2694,environment_steps=2694*81,learning_updates=0,TEST_values_read=False,
        evaluation_files_sha256=identity['evaluation_files_sha256'],source_files_sha256=identity['source_files_sha256'],source_release=identity['source_release'],
        test_namespace_amendment_sha256=identity['test_namespace_amendment_sha256'],experiment_design_sha256=identity['experiment_design_sha256'],parent_experiment_design_sha256=DESIGN_SHA,preference_dev_amendment_sha256=AMENDMENT_SHA,noise_dev_amendment_sha256=NOISE_SHA,noise_midpoint_dev_amendment_sha256=MIDPOINT_SHA,
        jobs=[dict(job_id=j['job_id'],path=str(p),sha256=hashlib.sha256(ev.canonical(j)+b'\n').hexdigest(),
            kind=j['kind'],paths=32 if j['kind']=='feedback' else 1 if j['kind']=='history' else 128) for p,j in zip(paths,jobs)])
    if write:
        for path,job in zip(paths,jobs):immutable_json(path,job)
        immutable_json(folder/'matrix.json',matrix)
        immutable_json(ROOT/'protocol/evaluation_execution_capsule.json',dict(schema='reasonable89-evaluation-execution-capsule-v1',
            evaluation_files_sha256=identity['evaluation_files_sha256'],scientific_source_release=identity['source_release'],
            scientific_source_files_sha256=identity['source_files_sha256'],test_namespace_amendment_path=identity['test_namespace_amendment_path'],test_namespace_amendment_sha256=identity['test_namespace_amendment_sha256'],weather_panel_sha256=identity['weather_panel_sha256'],evaluation_matrix_path=str(folder/'matrix.json'),evaluation_matrix_sha256=ev.sha(folder/'matrix.json'),TEST_values_read=False))
    return dict(status='BOUND_NOT_EXECUTED' if write else 'IDENTITY_READY_NOT_WRITTEN',matrix=str(folder/'matrix.json'),jobs=len(jobs),paths=2694,TEST_values_read=False,environment_steps=0,evaluation_files_sha256=identity['evaluation_files_sha256'])



def prepare_descriptive_feedback(base_jobs):
    """After TEST release and completed main384, bind zero or four descriptive jobs."""
    import numpy as np
    ev.np=np
    entries=[entry for entry in base_jobs.values() if entry['job']['kind']=='main']
    if len(entries)!=3:raise ValueError('Expected exactly3 main-policy jobs')
    job=sorted(entries,key=lambda row:row['job']['training_identity']['seed'])[0]['job']
    ev.verify_test_barrier(job)
    case=ev.representative_from_main(job);case_path=ROOT/'evaluation/representative_case.json';immutable_json(case_path,case)
    extra=[]
    if case['descriptive_feedback_required']:
        for entry in base_jobs.values():
            source=entry['job']
            if source['kind']!='feedback' or source['training_identity']['seed']!=case['trained_seed']:continue
            bound=dict(source);name='descriptive_'+source['job_id']+'__weather'+str(case['weather_id'])
            bound.update(job_id=name,kind='descriptive_feedback',weather_panel_role='test128',subset_indices=[case['weather_id']],baseline_indices=[case['weather_id']],
                representative_case_path=str(case_path),representative_case_sha256=ev.sha(case_path),output_dir=str(ROOT/'evaluation'/name/'attempt0'))
            path=ROOT/'configs/descriptive_feedback'/(name+'.json');immutable_json(path,bound)
            extra.append(dict(job_id=name,path=str(path),sha256=ev.sha(path),kind='descriptive_feedback',paths=1))
        if len(extra)!=4:raise AssertionError('Exactly4 descriptive branches are required outside feedback32')
    matrix=dict(schema='reasonable89-conditional-descriptive-evaluation-v1',status='REGISTERED_FROM_COMPLETE_MAIN384',
        base_evaluation_matrix_sha256=ev.sha(ROOT/'configs/evaluation/matrix.json'),representative_case_path=str(case_path),representative_case_sha256=ev.sha(case_path),
        representative_feedback_contract_sha256=REPRESENTATIVE_SHA,evaluation_files_sha256=job['evaluation_files_sha256'],
        jobs=sorted(extra,key=lambda row:row['job_id']),complete_paths=len(extra),included_in_feedback_statistics=False)
    return immutable_json(ROOT/'configs/descriptive_feedback/matrix.json',matrix)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    try:print(json.dumps(prepare(args.prepare),ensure_ascii=False))
    except (OSError,ValueError,KeyError) as exc:
        print(json.dumps(dict(status='NOT_READY_NO_TEST_OPENED',reason=str(exc),environment_steps=0,model_inferences=0,TEST_values_read=False),ensure_ascii=False));raise SystemExit(2)
if __name__=='__main__':main()
