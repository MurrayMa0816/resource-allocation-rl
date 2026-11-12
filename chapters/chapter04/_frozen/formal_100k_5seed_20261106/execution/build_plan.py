"""Register one common-budget C4 experiment; does not launch processes."""
from pathlib import Path
import csv,gzip,hashlib,json,statistics,datetime
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main():
    assert not (R/'plan.json').exists(),'Existing scientific plan is immutable'
    refs=json.loads((R/'protocol/previous_A_reference.json').read_text());reference=[]
    for x in refs:
        p=R/'preflight'/f"A_reference32__{x['domain']}__seed{x['seed']}"/'completed.json'
        d=json.loads(p.read_text());assert d['success'] and d['cases']==32
        assert abs(d['first8_mean']-float(x['last_dev']['native_mean']))<1e-12
        assert sha(x['model'])==d['source_model_sha256']
        reference.append(d)
    means={d:statistics.mean(x['mean']for x in reference if x['domain']==d)for d in ['radio','v5']}
    q={'schema':'chapter4-preregistered-quality-targets-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'primary_fraction':.95,'secondary_fractions':[.9,1.0],'A_common32_endpoint_means':means,
       'thresholds':{d:{str(f):means[d]*f for f in [.9,.95,1.0]}for d in means},
       'source':'existing A fixed endpoints, newly evaluated common32 DEV before new formal training; no test used',
       'rule':'second of two consecutive unsmoothed DEV points >=q; no attainment is right-censored at100608',
       'first8_old_A_reproduction_all10':True,'reference_new_training_steps':0,
       'reference_dev_steps':sum(x['evaluation_steps']for x in reference),
       'reference_replay_steps':sum(x['replay_environment_steps']for x in reference)}
    save(R/'protocol/quality_targets.json',q)
    methods=['f1','props','pgr','redo','mrq','bro','no_history','no_offpolicy_correction','fixed_actor_lr']
    families={m:('f1'if m in ['f1','no_history','no_offpolicy_correction','fixed_actor_lr']else'props'if m=='props'else'pgr'if m=='pgr'else'offpolicy')for m in methods}
    workers={'f1':'f1_worker.py','props':'props_worker.py','pgr':'pgr_worker.py','offpolicy':'offpolicy_worker.py'}
    devices={m:('cpu'if families[m]in ['f1','props']else'cuda')for m in methods}
    gpu_mib={'pgr':6144,'redo':1280,'mrq':1536,'bro':2048}
    runtime=json.loads((R/'protocol/runtime_environment.json').read_text())
    torch=runtime['python'];jax='/hy-tmp/resource_allocation_results/chapter4/recent_baselines_ready_20261102/runtime/jax/bin/python'
    jobs=[]
    for m in methods:
      for d in ['radio','v5']:
       for seed in range(5):
        family=families[m];id=f'{m}__{d}__seed{seed}';output=R/'jobs'/id
        argv=[jax if m=='bro'else torch,str(R/'execution'/workers[family]),'--method',m,'--domain',d,'--seed',str(seed),
              '--steps','100608','--output',str(output),'--device',devices[m],
              '--train-master','202611043101','--dev-master','202611043101','--eval-every','768','--dev-cases','32',
              '--environment-root',str(R/'source/environment')]
        job=dict(id=id,kind='training',method=m,family=family,domain=d,seed=seed,steps=100608,
                 device=devices[m],argv=argv,output=str(output),depends=[],priority=(0 if m=='pgr'else 1 if m in ['f1','props','redo','mrq','bro']else 2),
                 gpu_reserve_mib=gpu_mib.get(m,0),ram_reserve_mib=16384 if m=='pgr'else 8192 if m in ['redo','mrq','bro']else 1024,
                 output_reserve_mib=192 if m=='pgr'else 96,approval=str(R/'preflight'/f'approval_{family}.json'))
        jobs.append(job)
    assert len(jobs)==90
    for j in list(jobs):
        k=dict(j);k.update(id='final__'+j['id'],kind='evaluation',steps=0,priority=3,depends=[j['id']],
                        output=str(R/'evaluations'/j['id']),output_reserve_mib=8,
                        gpu_reserve_mib=1024 if j['device']!='cpu'else 0,ram_reserve_mib=2048)
        k['argv']=None;k['training_job_id']=j['id'];jobs.append(k)
    for x in refs:
        id=f"A_inherited__{x['domain']}__seed{x['seed']}";output=R/'evaluations'/id
        jobs.append(dict(id='final__'+id,kind='evaluation',method='A_inherited',family='a_reference',domain=x['domain'],seed=x['seed'],
                         steps=0,device='cpu',priority=3,depends=[],output=str(output),source_model=x['model'],source_model_sha256=sha(x['model']),
                         argv=[torch,str(R/'execution/a_reference_evaluate.py'),'--model',x['model'],'--domain',x['domain'],'--seed',str(x['seed']),
                               '--master','2026110694199','--phase','final','--cases','256','--output',str(output),
                               '--environment-root',str(R/'source/environment')],gpu_reserve_mib=0,ram_reserve_mib=1024,output_reserve_mib=8,
                         approval=str(R/'preflight/approval_a_reference.json')))
    assert len(jobs)==190 and len({x['id']for x in jobs})==190
    plan={'schema':'chapter4-formal-100k-five-seed-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope':str(R),'training_steps':100608,'training_seeds':list(range(5)),'training_master':202611043101,
          'dev_master':202611043101,'eval_every':768,'dev_cases':32,'dev_points':132,
          'final_master':2026110694199,'final_cases':256,'deterministic_evaluation':True,
          'new_training_jobs':90,'reused_A_models':10,'independent_endpoint_evaluations':100,
          'main_methods':methods[:6],'ablations':methods[6:],'jobs':jobs,
          'quality_targets':q,'quality_targets_sha256':sha(R/'protocol/quality_targets.json'),
          'storage':'complete compressed training/development/internal diagnostics; one fixed endpoint; no disk replay/periodic checkpoints',
          'new_training_environment_steps':9054720,'development_physical_steps':10644480,'endpoint_physical_steps_without_replay':716800,
          'engineering_and_A_calibration_costs_separate':True,
          'PROPS_configuration':{'target_lr':5e-5,'behavior_lr':1e-4,'user_selected':True,'not_validation_optimal':True,'prior_default_sensitivity_retained':True},
          'statistics':'independent training seed is statistical unit; mean/sample SD; no forced monotone curves or guaranteed F1 win',
          'budget_windows':[6144,12288,24576,49152,100608],'budget_prefixes_not_independently_tuned_short_runs':True,
          'postprocess_authorized':'after full independent acceptance generate figures/tables and update thesis C4 experiments using current writing guide and whole-thesis build.sh',
          'does_not_restore_retired_queues':True,'resource_policy':{'cpu_quota':42,'target_cpu_cores':40,'max_processes':90,
              'gpu_uuid':'GPU-a3130b13-8288-caaa-0f96-f2916c92d8e3','gpu_safety_mib':1536,'ram_floor_mib':16384,'disk_floor_mib':2048},
          'previous_design_preserved':str(R.parent/'experiment_design_20261106/experiment_design.json')}
    save(R/'plan.json',plan);save(R/'plan_seal.json',{'sha256':sha(R/'plan.json')})
    save(R/'preflight/approval_a_reference.json',{'status':'PASS','methods':['A_inherited'],'source_sha256':{str(R/'execution/a_reference_evaluate.py'):sha(R/'execution/a_reference_evaluate.py')},
        'evidence':'ten native endpoints strict reload and first8 DEV exactly reproduce prior scalar results; full32 DEV/physical replay complete','new_training_steps':0})
    with(R/'protocol/training_matrix.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=['id','method','domain','seed','steps','device','family']);w.writeheader();w.writerows({k:j[k]for k in w.fieldnames}for j in jobs if j['kind']=='training')
    print(json.dumps({'plan_sha256':sha(R/'plan.json'),'training':90,'evaluation':100,'quality_targets':q['thresholds']},ensure_ascii=False))

if __name__=='__main__':main()
