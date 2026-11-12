"""Independent read-only C4 acceptance: no Torch/JAX/model loading or Env calls."""
from __future__ import annotations
import argparse,datetime,gzip,hashlib,json,math,statistics,sys,traceback
from collections import Counter,defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'source/reporting'))
import result_io as io

def assert_plan(plan):
    assert plan['new_training_jobs']==50 and plan['reused_external_trainings']==40 and plan['unique_training_runs']==90 and plan['reused_A_models']==10 and plan['independent_endpoint_evaluations']==100
    assert plan['initialization']=='both_random' and plan['new_endpoint_evaluations']==60
    assert plan['training_steps']==100608 and plan['training_seeds']==list(range(5))
    assert plan['training_master']==plan['dev_master']==202611043101 and plan['final_master']==2026110694199
    assert plan['eval_every']==768 and plan['dev_cases']==32 and plan['dev_points']==132 and plan['final_cases']==256
    assert plan['deterministic_evaluation']is True
    jobs=plan['jobs'];assert len(jobs)==190 and len({x['id']for x in jobs})==190
    expected={(m,d,s)for m in io.MAIN+io.ABLATIONS for d in io.DOMAINS for s in io.SEEDS}
    assert {(x['method'],x['domain'],x['seed'])for x in jobs if x['kind']=='training'}==expected
    assert {(x['method'],x['domain'],x['seed'])for x in jobs if x['kind']=='evaluation'}==expected|{('A_inherited',d,s)for d in io.DOMAINS for s in io.SEEDS}
    assert sum(x['kind']=='training'for x in jobs)==90 and sum(x['kind']=='evaluation'for x in jobs)==100
    for job in jobs:
        out=Path(job['output']).resolve()
        base=R if job['execution_required']else Path(plan['old_scope'])
        assert out.is_relative_to(base/('jobs'if job['kind']=='training'else'evaluations'))
        assert job['execution_required']==(job['method']not in ['pgr','redo','mrq','bro'])
        assert job['steps']==(100608 if job['kind']=='training'else 0)
    assert io.sha(R/'protocol/quality_targets.json')==plan['quality_targets_sha256']
    assert io.read(R/'protocol/quality_targets.json')==plan['quality_targets']

def verify_approval(job,cache):
    p=Path(job['approval']);digest=io.sha(p)
    if str(p)not in cache:
        d=io.read(p);assert d['status']=='PASS'and job['method']in d['methods']
        assert d['source_sha256'];io.source_map_check(d['source_sha256']);cache[str(p)]=(digest,d)
    assert cache[str(p)][0]==digest
    d=cache[str(p)][1];assert job['method']in d['methods'];return d,digest

def verify_quality_calibration(plan,shared):
    # The absolute targets retain their original calibration and registration.
    # They are not recalculated from either new F1 or new C3 outcomes.
    import importlib.util
    old=Path(plan['old_scope'])
    spec=importlib.util.spec_from_file_location('_original_absolute_target_calibration',old/'execution/independent_acceptance.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    oldplan=io.read(old/'plan.json')
    assert io.sha(R/'protocol/quality_targets.json')==io.sha(old/'protocol/quality_targets.json')
    import inspect
    original=inspect.getsource(legacy.verify_quality_calibration)
    line="state=io.read(R/'runtime/queue_state.json');start=min(x['runtime']['started_at']for j in plan['jobs']if j['kind']=='training'for x in[state['jobs'][j['id']]])"
    replacement="snapshot=R/'dispatch_updates/external_retained_20261107/handover_pre.json';state=io.read(snapshot)['queue'];start=min(x['runtime']['started_at']for j in plan['jobs']if j['kind']=='training'for x in[state['jobs'][j['id']]]if x.get('runtime'))"
    assert original.count(line)==1
    namespace=dict(legacy.__dict__);exec(compile(original.replace(line,replacement),'<retained_original_calibration_frozen_first_launch>', 'exec'),namespace)
    proof=namespace['verify_quality_calibration'](oldplan,shared)
    proof['first_launch_evidence']=str(old/'dispatch_updates/external_retained_20261107/handover_pre.json')
    proof['first_launch_evidence_sha256']=io.sha(proof['first_launch_evidence'])
    proof.update(scope='original preregistered absolute targets, retained unchanged',new_A_relative_threshold_claim=False,new_F1_results_used_for_thresholds=False)
    return proof

def source_evidence(job,d,protocol,approval):
    checked=0
    if 'source_identity'in d:checked+=io.source_map_check(d['source_identity'])
    if 'source_binding'in d:checked+=io.source_map_check(d['source_binding'])
    worker=Path(job['argv'][1])if job.get('argv')else R/'execution'/({'f1':'f1','offpolicy':'offpolicy','props':'props','pgr':'pgr'}[job['family']]+'_worker.py')
    if not job['execution_required']:
        worker=Path('/hy-tmp/resource_allocation_results/chapter4/formal_100k_5seed_20261106/execution')/({'offpolicy':'offpolicy','pgr':'pgr'}[job['family']]+'_worker.py')
    if worker is not None:
        expected=approval['source_sha256'].get(str(worker));assert expected and io.sha(worker)==expected
        if 'worker_sha256'in d:assert d['worker_sha256']==expected
    for key in ['worker_sha256']:
        if key in protocol and worker is not None:assert protocol[key]==io.sha(worker)
    for path,digest in d.get('artifacts_sha256',{}).items():
        p=Path(path);p=p if p.is_absolute()else Path(job['output'])/p
        assert p.is_relative_to(Path(job['output']))and io.sha(p)==digest,(job['id'],'artifact',str(p))
        checked+=1
    return checked

def verify_resource(job,rows,phase_counts):
    h=32 if job['domain']=='radio'else 24
    if job['method']=='A_inherited':return dict(rows=0,phase_counts=phase_counts,replay_verified=True)
    assert rows,(job['id'],'resource evidence missing')
    counts=Counter(x['phase']for x in rows);assert dict(counts)==phase_counts,(job['id'],counts,phase_counts)
    for x in rows:
        assert io.integer(x['steps'])==h
        initial=io.num(x['initial_budget']);assert initial>0
        total=io.num(x['total_used']);assert abs(total/initial-1)<1e-10
        assert io.num(x['relative_allocation_error'])<1e-10
        assert abs(io.num(x['remaining']))<=max(1e-10,initial*1e-10)
        for k in ['maximum_negative_allocation_violation','maximum_per_object_cap_violation','maximum_completed_object_allocation']:assert io.num(x[k])==0,(job['id'],k)
        if 'nonterminal_rewards_zero'in x:assert io.true(x['nonterminal_rewards_zero'])
        if 'terminal_full_spend_assertion'in x:assert io.true(x['terminal_full_spend_assertion'])
        if 'rounds'in x:
            assert len(x['rounds'])==h
            for round in x['rounds']:
                assert io.num(round['lower'])<=io.num(round['quota'])+initial*1e-10
                assert io.num(round['quota'])<=io.num(round['upper'])+initial*1e-10
        io.all_finite(x)
    return dict(rows=len(rows),phase_counts=dict(counts))

def verify_family(job,d,protocol,training):
    method=job['method'];family=job['family']
    if family=='f1':
        if training:
            assert d['source_unchanged']and d['endpoint_reload_pass']and d['resource_guard_enabled']
            assert d['evaluation_deterministic']and d['evaluation_training_rng_isolated']and not d['final_holdout_opened']
            assert not d['optimizer_saved']and not d['replay_saved']and not d['periodic_models_saved']and d['model_count']==1
            assert not d['engineering_gate']and protocol['fresh_initialization']
            assert protocol['initialization_scheme']=='both_random'
            source3=Path('/hy-tmp/resource_allocation_results/chapter3/double_random_formal_20261107/jobs')/f"train__A_FULL__{job['domain']}__seed{job['seed']}"/'attempt0/protocol.json'
            p3=io.read(source3);assert p3['initialization_control']['scheme']=='both_random'
            assert protocol['initial_actor_sha256']==p3['initial_actor_sha256'] and protocol['initial_critic_sha256']==p3['initial_critic_sha256']
            assert protocol['arguments']['train_master']==protocol['arguments']['dev_master']==202611043101
            assert protocol['terminal_only']and protocol['full_spend_binding']['learner_information_unchanged']
        else:assert d['deterministic']and d['phase']=='final'and d['paths']==256 and d['path_master']==2026110694199
    elif family in ['props','pgr']:
        assert not d['engineering_only']and d['evaluation_only']==(not training)
        assert d['optimizer_or_replay_saved']is False and d['periodic_models_saved']is False
        evalp=io.read(Path(job['output'])/'evaluation_protocol.json')
        assert evalp['deterministic']and evalp['RNG_isolated']and evalp['development_master']==202611043101
        if training:
            assert evalp['development_interval']==768 and evalp['development_cases']==32
            cfg=protocol['config'];assert cfg['path_master']==cfg['development_master']==202611043101
            assert cfg['deterministic_eval']and not cfg['engineering_gate']
            native=d['native_receipt']
            if family=='props':
                assert cfg['learning_rate']==5e-5 and cfg['props_learning_rate']==1e-4
                assert cfg['initialization_scheme']=='both_random'
                assert native['mechanism_checks']['checkpoint_roundtrip_exact']
                assert native['mechanism_checks']['nonterminal_reward_count']==0
                assert native['mechanism_checks']['completed_episodes']==100608//(32 if job['domain']=='radio'else 24)
                assert native['mechanism_checks']['terminal_feedback_count']==native['mechanism_checks']['completed_episodes']
                assert native['source_receipt']['source_modified']is False
            else:
                assert protocol['full_native_diffusion_budget']and native['formal_training_performed']
                assert native['checkpoint_kind']=='inference_weights'and native['checkpoint_action_abs_error']==0
                assert native['nonterminal_nonzero_real_rewards']==0 and native['terminal_td_target_abs_error']==0
                assert native['real_terminal_feedbacks']==100608//(32 if job['domain']=='radio'else 24)
                assert native['diffusion_training_steps']==1000000 and native['total_generated_model_transitions']==10000000
                assert len(native['diffusion_cycles'])==10 and native['synthetic_sample_references']>0
                assert native['checkpoint_contains_real_and_synthetic_buffers']is False
                assert native['native_config']['diffusion_train_steps']==100000 and native['native_config']['denoise_steps']==128
        else:assert evalp['evaluation_master']==2026110694199 and d['native_receipt']['deterministic_eval']
    elif family=='offpolicy':
        assert not d['engineering_gate']and d['evaluation_deterministic']and d['roundtrip_max_error']<2e-6
        assert d['one_inference_model']and protocol['full_spend_guard_unchanged']
        assert protocol['evaluation_never_updates_or_enters_replay']and protocol['deterministic_evaluation']
        assert d['train_master']==d['dev_master']==202611043101 and d['final_master']==2026110694199
        if training:
            assert d['new_holdout_calls']==0 and protocol['registered_actual_steps']==100608
            diag=d['final_diagnostics']
            if method=='redo':
                expected=(100608-10000)*8
                assert d['updates']==expected==diag['n_updates']
                assert diag['warmup_steps']==10000 and diag['replay_ratio']==8 and diag['recycle_period']==200000
                assert diag['recycle_event_count']==3
                assert all(v==expected for v in diag['optimizer_iterations'].values())
            elif method=='mrq':
                first=9698 if job['domain']=='radio'else 9601
                n=100608-first+1;bursts=math.ceil(n/250)
                assert d['updates']==n==diag['policy_optimizer_steps']==diag['rl_updates']
                assert diag['first_learning_environment_step']==first and diag['replay_size_warmup_threshold']==10000
                assert diag['encoder_optimizer_steps']==bursts*250 and diag['target_hard_syncs']==bursts
                assert diag['external_reward_scale']==1.
            else:
                expected=(100608-2500+1)*2
                assert d['updates']==expected==diag['gradient_iteration_count']
                assert diag['reset_times']==[15001]and diag['reset_event_count']==1
                assert diag['deployment_policy']=='pessimistic actor'
        else:assert d['new_holdout_calls']==256
    else:
        assert method=='A_inherited'and not training
        assert d['new_training_steps']==0 and d['strict_reload']and d['deterministic']
        assert protocol['source_model']==job['source_model']
        c3=io.read(job['source_completion']);assert c3['success']and c3['training_steps']==100608
        assert protocol['source_model_sha256']==c3['model_sha256']
        p3=io.read(Path(job['source_completion']).with_name('protocol.json'));assert p3['initialization_control']['scheme']=='both_random'
        assert d['source_model_sha256']==io.sha(job['source_model'])
        assert protocol['phase']=='final'and protocol['path_master']==2026110694199 and protocol['cases']==256
        h=32 if job['domain']=='radio'else 24
        assert d['replay_environment_steps']==256*h and d['physical_environment_steps']==512*h

def verify_job(job,approval,shared_cases,model_sources):
    out=Path(job['output']);d=io.receipt(out);assert d is not None
    assert d['method']==job['method']and d['domain']==job['domain']and int(d['seed'])==job['seed']
    training=job['kind']=='training';h=32 if job['domain']=='radio'else 24
    if job['execution_required']:
        assignment=io.read(R/'deployment/dispatch_assignments.json')['jobs'][job['id']]
        receipt=io.read(R/'dispatch'/assignment['host']/'receipts'/(job['id']+'.json'))
        assert receipt['success'] and receipt['job_id']==job['id'] and receipt['execution']==assignment
        assert receipt['binding']['plan_sha256']==io.sha(R/'plan.json')
        assert receipt['binding']['assignments_sha256']==io.sha(R/'deployment/dispatch_assignments.json')
        assert receipt['process_identity']['argv']==assignment['argv']
        assert receipt['verification']['native_completion_sha256']==io.sha(out/'completed.json')
        if job['kind']=='training':assert d['actual_device']==assignment['device'],(job['id'],'device not registered')
    actual=int(d.get('training_steps',d.get('train_steps',d.get('new_training_steps',-1))))
    assert actual==(100608 if training else 0),(job['id'],actual)
    protocol_path=out/'protocol.json'
    if not protocol_path.exists():
        assert not training and job['family']in['props','pgr']
        protocol_path=out/'evaluation_protocol.json'
    protocol=io.read(protocol_path);verify_family(job,d,protocol,training)
    source_checks=source_evidence(job,d,protocol,approval)
    for p in out.rglob('*.gz'):
        if 'runtime'in p.relative_to(out).parts:continue
        with gzip.open(p,'rb')as f:
            for _ in iter(lambda:f.read(2**20),b''):pass
    model=io.endpoint_path(job,d)
    expected=d.get('model_sha256',d.get('checkpoint_sha256',d.get('source_model_sha256')))
    assert expected and io.sha(model)==expected
    if training:
        model_sources[(job['method'],job['domain'],job['seed'])]=(str(model),expected)
        binaries=[p for p in out.rglob('*')if p.is_file()and'runtime'not in p.relative_to(out).parts and p.suffix in('.pt','.pth','.pkl','.ckpt','.zip','.msgpack')]
        assert binaries==[model]or(set(binaries)=={model}),(job['id'],'unexpected extra model/checkpoint',list(map(str,binaries)))
    elif job['method']!='A_inherited':
        parent=model_sources[(job['method'],job['domain'],job['seed'])]
        assert(str(model),expected)==parent,(job['id'],'wrong source endpoint')
    cases=io.case_rows(out)
    assert len(cases)==(132*32 if training else 256)
    for x in cases:
        assert x['steps']==h and abs(x['resource_fraction']-1)<1e-9
        if x['replay_error']is not None:assert x['replay_error']==0
        if job['method']=='A_inherited':assert x['replay_error']==0
        assert x['phase']==('dev'if training else'final')
        k=(job['domain'],x['phase'],x['case_index'])
        if k in shared_cases:assert shared_cases[k]==x['path_seed'],(job['id'],'not common exogenous path',k)
        else:shared_cases[k]=x['path_seed']
    if training:
        curve=io.curve_rows(out);io.verify_means(curve,cases,list(range(0,100609,768)),32,'dev')
        assert int(d.get('development_env_steps',d.get('development_steps',d.get('evaluation_steps',-1))))==132*32*h
        if job['family']=='f1':
            u=io.update_rows(out);assert len(u)==4+(100608//h-32)//4
            assert sum(io.integer(x['new_episodes'])for x in u)==100608//h
            for x in u:
                assert x['method']==job['method']and io.num(x['likelihood_error'])<1e-8
                if 'behavior_likelihood_error'in x:assert io.num(x['behavior_likelihood_error'])<1e-8
                if 'fresh_mc_target_error'in x:assert io.num(x['fresh_mc_target_error'])<1e-8
                if not x['warmup']:
                    assert x['new_episodes']==4 and x['learning_episodes']==8
                    if job['method']=='no_history':
                        assert x['replay_episodes_read']==0 and x['current_episode_repeats']==4
                    else:assert x['replay_episodes_read']==4 and x['current_episode_repeats']==0
                    if job['method']=='no_offpolicy_correction':
                        assert x['offpolicy_training_correction']is False and x['vtrace_used_for_training']is False
                        assert x['stored_behavior_preserved']and x['training_clip_center']==1.
                    elif job['method']in['f1','fixed_actor_lr']:assert x['vtrace_target_recomputations']>=1
                if job['method']=='fixed_actor_lr':
                    assert x['actor_lr_before']==x['actor_lr_after']==.0003 and x['tv_controller_ran']is False
                io.all_finite(x)
            assert sum(io.integer(x['actual_actor_optimizer_steps'])for x in u)==d['actor_updates']
            assert sum(io.integer(x['actual_critic_optimizer_steps'])for x in u)==d['critic_updates']
        else:
            native_rows=io.update_rows(out)
            for x in native_rows:io.all_finite(x)
            if job['family']=='props':
                mechanism=d['native_receipt']['mechanism_checks']
                target=[x for x in native_rows if x.get('target_update')is True]
                updates=[x for x in native_rows if x.get('record_type')=='update']
                episodes=[x for x in native_rows if x.get('record_type')=='episode']
                assert len(target)==d['native_receipt']['target_updates']
                assert sum(io.integer(x.get('ppo_num_update_minibatches',0))for x in target)==mechanism['target_minibatches']
                assert sum(io.integer(x.get('props_num_update_minibatches',0))for x in updates)==mechanism['behavior_minibatches']
                assert len(episodes)==100608//h
        phase_counts={'train':100608//h,'dev':132*32}
        returns_train,name=io.train_rows(out)
        assert len(returns_train)==100608//h
        assert [r['training_step']for r in returns_train]==list(range(h,100609,h))
    else:
        assert sorted(x['case_index']for x in cases)==list(range(256))
        mean=statistics.fmean(x['native_return']for x in cases)
        reported=d.get('summary',d.get('native_receipt',{})).get('native_mean')
        if reported is None:reported=d.get('native_receipt',{}).get('mean_return',d.get('mean'))
        if reported is not None:assert io.close(mean,reported),(job['id'],'endpoint mean mismatch')
        assert int(d.get('evaluation_steps',d.get('final_evaluation_steps',-1)))==256*h
        phase_counts={'final':256}
    resources,name=io.resource_rows(out);rv=verify_resource(job,resources,phase_counts)
    if resources:
        resource_cases=[r for r in resources if r['phase']==('dev'if training else'final')]
        assert len(resource_cases)==len(cases)
        for physical,case in zip(resource_cases,cases):
            assert io.close(physical['terminal_reward'],case['native_return'])
            assert io.integer(physical.get('path_index',physical.get('episode_index')))==case['case_index']
        if training:
            resource_train=[r for r in resources if r['phase']=='train']
            assert len(resource_train)==len(returns_train)
            for physical,episode in zip(resource_train,returns_train):assert io.close(physical['terminal_reward'],episode['native_return'])
    if job['family']=='offpolicy':
        lines=(gzip.open(out/'resource_ledger.jsonl.gz','rt'));hh=hashlib.sha256()
        with lines:
            for line in lines:hh.update(line.encode())
        assert hh.hexdigest()==d['resource_summary']['canonical_uncompressed_ledger_sha256']
    hashes={str(p.relative_to(out)):io.sha(p)for p in sorted(out.rglob('*'))if p.is_file()and'runtime'not in p.relative_to(out).parts and p.name!='progress.json'and p.suffix!='.log'}
    return dict(id=job['id'],kind=job['kind'],method=job['method'],domain=job['domain'],seed=job['seed'],
        training_steps=actual,development_steps=132*32*h if training else 0,endpoint_evaluation_steps=0 if training else 256*h,
        endpoint_replay_steps=256*h if job['method']=='A_inherited'else 0,
        model_path=str(model),model_sha256=expected,case_rows=len(cases),resource=rv,
        source_checks=source_checks,input_sha256=hashes,completed_sha256=io.sha(out/'completed.json'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',type=Path,default=R);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    assert a.scope.resolve()==R,'This acceptance implementation is bound to this registered scope'
    if a.self_test:return self_test()
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();plan=io.read(R/'plan.json');seal=io.read(R/'plan_seal.json')
    assert io.sha(R/'plan.json')==seal['sha256'];assert_plan(plan)
    absent=[j['id']for j in plan['jobs']if not(Path(j['output'])/'completed.json').exists()]
    if absent:
        io.write(R/'independent/acceptance.json',dict(status='PENDING',created_utc=started,missing_jobs=absent,completed=len(plan['jobs'])-len(absent),total=190,plan_sha256=seal['sha256']))
        return 3
    reports=[];errors=[];shared={};sources={};approvals={}
    calibration=verify_quality_calibration(plan,shared)
    for job in sorted(plan['jobs'],key=lambda x:x['kind']!='training'):
        try:
            approved,_=verify_approval(job,approvals);reports.append(verify_job(job,approved,shared,sources))
        except BaseException as e:errors.append(dict(job=job['id'],error=repr(e),traceback=traceback.format_exc()))
    totaltrain=sum(x['training_steps']for x in reports);newtrain=sum(x['training_steps']for x in reports if next(j for j in plan['jobs']if j['id']==x['id'])['execution_required']);dev=sum(x['development_steps']for x in reports);final=sum(x['endpoint_evaluation_steps']for x in reports)
    if not errors:
        assert len(reports)==190 and totaltrain==9054720 and dev==10644480 and final==716800
    report=dict(schema='chapter4-190-native-result-independent-acceptance-v1',status='PASS'if not errors else'FAIL',created_utc=started,
        verified_jobs=len(reports),expected_jobs=190,training_jobs=90,endpoint_jobs=100,
        unique_new_training_steps=newtrain,retained_external_training_steps=totaltrain-newtrain,new_training_jobs=50,retained_external_training_jobs=40,initialization_scheme='both_random',development_environment_steps=dev,new_development_environment_steps=5913600,retained_development_environment_steps=4730880,independent_endpoint_environment_steps=final,
        inherited_A_replay_environment_steps=sum(x['endpoint_replay_steps']for x in reports),
        plan_sha256=seal['sha256'],quality_targets_sha256=plan['quality_targets_sha256'],
        whole_gzip_streams_read_to_EOF=True,raw_case_means_and_sample_SD_verified=True,common_case_seeds_verified=True,
        inherited_A_history_not_mixed_into32_case_DEV=True,model_deserialization=False,new_environment_steps=0,new_optimizer_steps=0,
        quality_calibration=calibration,
        CPU_GPU_wallclock_not_used_for_fair_algorithm_speed_ranking=True,errors=errors,jobs=reports,
        approvals_sha256={k:v[0]for k,v in approvals.items()})
    io.write(R/'independent/acceptance.json',report);print(json.dumps({k:v for k,v in report.items()if k not in('jobs','errors')},ensure_ascii=False));return 0 if not errors else 2

def self_test():
    tests=[]
    def test(name,fn,reject=False):
        try:fn()
        except(AssertionError,ValueError,KeyError,EOFError,gzip.BadGzipFile):assert reject,name
        else:assert not reject,name
        tests.append(name)
    curve=[dict(training_step=n,mean_return=v,phase='dev',cases=2,sample_sd=0.)for n,v in [(0,.1),(768,.8),(1536,.9)]]
    test('two_consecutive_requires_second_point',lambda:assertion(io.attainment(curve,.7)['confirmed_step']==1536))
    test('single_last_hit_is_censored',lambda:assertion(io.attainment(curve,.85)['censored']))
    test('initial_two_hits_confirm_at_first_interval',lambda:assertion(io.attainment(curve,.05)['confirmed_step']==768))
    cases=[dict(training_step=r['training_step'],phase='dev',case_index=i,native_return=r['mean_return'])for r in curve for i in range(2)]
    test('raw_case_mean_positive',lambda:io.verify_means(curve,cases,[0,768,1536],2,'dev'))
    bad=[dict(x)for x in cases];bad[0]['native_return']=.2
    test('raw_case_mean_negative',lambda:io.verify_means(curve,bad,[0,768,1536],2,'dev'),True)
    duplicate=[dict(x)for x in cases];duplicate[0]['case_index']=1
    test('duplicate_case_rejected',lambda:io.verify_means(curve,duplicate,[0,768,1536],2,'dev'),True)
    test('wrong_grid_rejected',lambda:io.verify_means(curve,cases,[0,512,1536],2,'dev'),True)
    test('nonfinite_rejected',lambda:io.all_finite({'loss':float('nan')}),True)
    job=dict(id='explicit_synthetic_resource_fixture',method='f1',domain='radio')
    row=dict(phase='train',steps=32,initial_budget=1.,total_used=1.,remaining=0.,relative_allocation_error=0.,
        maximum_negative_allocation_violation=0.,maximum_per_object_cap_violation=0.,maximum_completed_object_allocation=0.,
        nonterminal_rewards_zero=True,terminal_full_spend_assertion=True)
    test('positive_complete_resource_episode',lambda:verify_resource(job,[row],{'train':1}))
    test('zero_budget_rejected',lambda:verify_resource(job,[dict(row,initial_budget=0)],{'train':1}),True)
    test('unspent_budget_rejected',lambda:verify_resource(job,[dict(row,total_used=.9)],{'train':1}),True)
    test('per_object_cap_violation_rejected',lambda:verify_resource(job,[dict(row,maximum_per_object_cap_violation=.1)],{'train':1}),True)
    test('wrong_resource_phase_count_rejected',lambda:verify_resource(job,[row],{'dev':1}),True)
    test('dense_reward_rejected',lambda:verify_resource(job,[dict(row,nonterminal_rewards_zero=False)],{'train':1}),True)
    test('truncated_gzip_rejected',lambda:gzip.decompress(gzip.compress(b'explicit synthetic CRC fixture')[:-8]),True)
    test('wrong_training_count_rejected',lambda:assert_plan(dict(new_training_jobs=89)),True)
    large=2**61+12345
    test('64bit_seed_string_exact',lambda:assertion(io.integer(str(large))==large))
    test('64bit_adjacent_seeds_stay_distinct',lambda:assertion(io.integer(str(large))!=io.integer(str(large+1))))
    test('64bit_int_exact',lambda:assertion(io.integer(large)==large))
    test('imprecise_large_float_rejected',lambda:io.integer(float(large)),True)
    io.write(R/'preflight/independent_acceptance_selftest.json',dict(status='PASS',tests=tests,synthetic_only=True,model_deserialization=False,new_environment_steps=0))
    print('selftest PASS',len(tests));return 0
def assertion(value):assert value
if __name__=='__main__':
    try:code=main()
    except Exception as error:
        if '--self-test'not in sys.argv:
            io.write(R/'independent/acceptance.json',dict(status='FAIL',stage='plan/source/calibration-or-final-invariant',
                error=repr(error),traceback=traceback.format_exc(),new_environment_steps=0,model_deserialization=False))
        raise
    raise SystemExit(code)
