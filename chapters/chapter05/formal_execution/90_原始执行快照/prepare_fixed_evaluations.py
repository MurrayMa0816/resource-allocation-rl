"""Prepare exactly210 immutable formal evaluations only after the90-model seal."""
from pathlib import Path
import argparse, json, sys
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'execution'))
from formal_science_audit import require, read, sha, write, plan_only, REGISTRY_SCHEMA
from evaluate_formal_fixed import CONFIG_SCHEMA

def capsule():
 files=['execution/baseline_worker.py','execution/formal_science_audit.py','execution/prepare_fixed_evaluations.py','execution/evaluate_formal_fixed.py','execution/evaluator_native.py','execution/budget_conditions.py','execution/baseline_worker.py','execution/own_worker.py','execution/formal_evaluation_audit.py','execution/dispatch.py','execution/run_fixed_evaluations.py']
 files+=sorted(str(p.relative_to(R))for p in(R/'evaluation_conditions').glob('*.json'))
 return dict(schema='chapter5-formal-postprocess-preparation-capsule-v1',scope=str(R),status='PREPARED_NO_TEST_OPENED',files={p:sha(R/p)for p in files},native_evaluation_mathematics_changed=False,formal_test_activation_requires='actual90training audit PASS and all fixed endpoint/model registry seals',test_resets_during_preparation=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--validate-preparation-only',action='store_true');p.add_argument('--device',choices=['cpu','cuda:0'],default='cpu');p.add_argument('--gpu-uuid');p.add_argument('--host',choices=['local','remote'],default='remote');a=p.parse_args()
 c,training_plan=plan_only();require(c['expected_evaluations']==210 and c['evaluation']['holdout_paths']==256,'Registered210 endpoint panels required')
 for domain in ['radio','v5']:
  for scale in[.8,1.,1.2]:require((R/'evaluation_conditions'/f'{domain}__scale{scale:.1f}.json').is_file(),'Missing fixed budget condition')
 pack=capsule()
 if a.validate_preparation_only:print(json.dumps(dict(status='PASS_CODE_PREPARATION_ONLY',test_resets=0,training_complete_claim=False,postprocess_files=len(pack['files']))));return 0
 rp=R/'model_registry.json';require(rp.exists(),'Formal test stays closed: no sealed90-model registry')
 registry=read(rp);audit=read(R/'training_audit.json')
 require(registry['schema']==REGISTRY_SCHEMA and registry['status']=='SEALED_ALL_FIXED_MODELS'and len(registry['models'])==90 and registry['all_training_complete']is True and registry['source_validation_passed']is True,'Formal90-model seal incomplete')
 require(sha(R/'training_audit.json')==registry['training_audit_sha256']and audit['status']=='PASS'and audit['contract_sha256']==sha(R/'contract.json'),'Actual completed-training independent audit required')
 if a.device=='cpu':require(a.gpu_uuid is None,'CPU evaluation must not claim a GPU')
 else:require(a.gpu_uuid in c['gpu_uuids'],'Registered physical inference GPU UUID required')
 jobs=[]
 for m in registry['models']:
  scales=[.8,1.,1.2]if m['method']!='full'or m['variant']=='full'else[1.]
  for scale in scales:
   jid=m['model_id']+f'__scale{scale:.1f}';condition=R/'evaluation_conditions'/f"{m['domain']}__scale{scale:.1f}.json";cp=R/'evaluation_configs'/(jid+'.json')
   cfg=dict(schema=CONFIG_SCHEMA,id=jid,model_id=m['model_id'],engineering=False,contract=str(R/'contract.json'),contract_sha256=sha(R/'contract.json'),model_registry=str(rp),model_registry_sha256=sha(rp),paths=256,batch_size=32,scale=scale,device=a.device,gpu_uuid=a.gpu_uuid,host=a.host,data_root=c['data_root'],test_master=c['evaluation']['test_master'],test_rng_seed=c['evaluation']['test_rng_seed'],condition_file=str(condition),condition_sha256=sha(condition),output=str(R/'evaluation_jobs'/jid),execution_worker_sha256=sha(R/'execution/evaluate_formal_fixed.py'),training_worker_sha256=sha(R/'execution'/('own_worker.py'if m['method']=='full'else 'baseline_worker.py')),native_evaluator_sha256=sha(R/'execution/evaluator_native.py'),budget_adapter_sha256=sha(R/'execution/budget_conditions.py'))
   if cp.exists():require(read(cp)==cfg,'Refuse changing a registered fixed evaluation config')
   else:write(cp,cfg)
   jobs.append(dict(id=jid,kind='fixed_evaluation',worker='evaluate_formal_fixed.py',config=str(cp),config_sha256=sha(cp),output=cfg['output'],model_id=m['model_id'],method=m['method'],variant=m['variant'],domain=m['domain'],seed=m['seed'],host=a.host,device=a.device,gpu_uuid=a.gpu_uuid,scale=scale,paths=256,endpoint_sha256=m['endpoint_sha256']))
 require(len(jobs)==210 and len({j['id']for j in jobs})==210,'Fixed panel plan must be210 unique jobs')
 plan=dict(schema='chapter5-formal-five-seed-210-fixed-evaluation-plan-v1',root=str(R),contract_sha256=sha(R/'contract.json'),registry_sha256=sha(rp),training_audit_sha256=sha(R/'training_audit.json'),jobs=jobs,expected_jobs=210,paths_per_job=256,total_episodes=53760,physical_steps=1505280,new_learning_steps=0,new_optimizer_steps=0,all_models_fixed_before_panel_open=True,no_automatic_retry=True)
 pp=R/'fixed_evaluation_plan.json'
 if pp.exists():require(read(pp)==plan,'Refuse changing an existing210-evaluation plan')
 else:write(pp,plan)
 pack['files']['fixed_evaluation_plan.json']=sha(pp)
 for rel in ['contract.json','plan.json','model_registry.json','training_audit.json']:pack['files'][rel]=sha(R/rel)
 for j in jobs:pack['files'][str(Path(j['config']).relative_to(R))]=j['config_sha256']
 write(R/'deployment/postprocess_capsule.json',pack)
 print(json.dumps(dict(status='PREPARED_210_NO_TEST_RESET',jobs=210,plan_sha256=sha(pp),test_resets=0,command='Run evaluate_formal_fixed.py --config <registered config>; scheduling and mirrors require separately bound deployment.')));return 0
if __name__=='__main__':sys.exit(main())
