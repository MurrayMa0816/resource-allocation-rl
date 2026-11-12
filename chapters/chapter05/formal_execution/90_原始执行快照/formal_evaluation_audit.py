"""Audit210 already-completed fixed panels; export a read-only plotting input bundle."""
from pathlib import Path
import datetime, json, math, sys
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'execution'))
from formal_science_audit import require,read,sha,write,rows,REGISTRY_SCHEMA

def check_case(x,cfg,domain):
 n=100 if domain=='radio'else 120;h=32 if domain=='radio'else 24;tau=.2 if domain=='radio'else .7;s=float(cfg['scale']);budget=(.48 if domain=='radio'else 20160.)*s
 q=x['quality'];require(len(q)==n and all(math.isfinite(float(v))for v in q),'Malformed per-object final task outcomes')
 k=sum(float(v)>=tau for v in q);j=sum(float(v)for v in q)/n;require(x['N']==n and x['K']==k and abs(x['C']-k/n)<1e-12 and abs(x['J']-j)<1e-10,'Fixed task metric mismatch')
 require(len(x['period_totals'])==h and all(math.isfinite(float(v))and v>=0 for v in x['period_totals']),'Incomplete final resource periods')
 proof=x['full_spend'];tol=float(proof['tolerance']);require(proof['full_spend_required']is True and 0<tol<=max(1e-8,budget*1e-10),'Invalid full-spend proof tolerance')
 for actual,target in [(x['initial_budget'],budget),(x['total_used'],budget),(x['remaining'],0),(sum(x['period_totals']),budget)]:require(abs(float(actual)-target)<=tol,'Frozen budget transfer failed the whole resource ledger')
 require(x['index']>=0 and len(x['plan_sha256'])==64,'Missing common exogenous case identity')
 return k,k/n,j,.8*(k/n)+.2*j

def main():
 c=read(R/'contract.json');registry=read(R/'model_registry.json');p=read(R/'fixed_evaluation_plan.json')
 require(registry['schema']==REGISTRY_SCHEMA and registry['status']=='SEALED_ALL_FIXED_MODELS'and len(registry['models'])==90,'Actual90-model registry required')
 require(p['contract_sha256']==sha(R/'contract.json')and p['registry_sha256']==sha(R/'model_registry.json')and p['training_audit_sha256']==sha(R/'training_audit.json'),'Actual fixed-model source seals differ')
 require(len(p['jobs'])==210 and len({j['id']for j in p['jobs']})==210,'Complete210 fixed evaluation plan required')
 missing=[j['id']for j in p['jobs']if not(Path(j['output'])/'completed.json').exists()]
 if missing:write(R/'fixed_evaluation_audit_pending.json',dict(status='NOT_READY',complete=210-len(missing),expected=210,missing=missing));return 2
 records=[];panels={};total_physical=0
 for job in p['jobs']:
  cfg=read(job['config']);out=Path(job['output']);d=read(out/'completed.json');domain=job['domain']
  require(sha(job['config'])==job['config_sha256']and d['config_sha256']==job['config_sha256'],'Evaluation config SHA changed')
  require(d['success']is True and d['engineering']is False and d['paths']==256 and d['test_split_opened']is True and d['formal_test_cases_opened']==256,'Missing actual complete common test panel')
  for k in ['new_training_steps','optimizer_steps','PSA_calls']:require(d[k]==0,'Frozen evaluation performed updates')
  for k in ['model_unchanged','rng_unchanged','source_unchanged','no_new_model_saved','normalization_state_unchanged']:require(d[k]is True,'Fixed model/RNG/normalizer changed')
  for k in ['model_selection','holdout_selection']:require(d[k]is False,'Fixed endpoint evaluation performed selection')
  require(d['endpoint_sha256']==job['endpoint_sha256']and d['model_registry_sha256']==sha(R/'model_registry.json')and d['contract_sha256']==sha(R/'contract.json')and d['source_manifest_sha256']==c['source_manifest_sha256'],'Evaluation endpoint/source identity differs')
  require(d['test_master']==c['evaluation']['test_master']and d['test_rng_seed']==c['evaluation']['test_rng_seed'],'Final test panel/RNG differs')
  require(d['formal_evaluator_sha256']==sha(R/'execution/evaluate_formal_fixed.py')and d['evaluator_sha256']==sha(R/'execution/evaluator_native.py')and d['budget_adapter_sha256']==sha(R/'execution/budget_conditions.py'),'Frozen inference source changed')
  require(d['evaluation_sha256']==sha(out/'evaluation.jsonl.gz')and d['condition_sha256']==cfg['condition_sha256'],'Final raw array/condition SHA differs')
  require(d['physical_steps']==256*c['domains'][domain]['horizon'],'Incomplete actual fixed physical panel');total_physical+=d['physical_steps']
  vals=[];seen=[]
  for row in rows(out/'evaluation.jsonl.gz'):
   v=check_case(row,cfg,domain);vals.append(v);seen.append(row['index']);key=(domain,float(cfg['scale']),row['index'])
   if key in panels:require(panels[key]==row['plan_sha256'],'Methods/seeds did not evaluate common exogenous cases')
   else:panels[key]=row['plan_sha256']
  require(seen==list(range(256)),'Repeated/missing common test case indexes')
  means={k:sum(v[i]for v in vals)/256 for i,k in enumerate(['K','C','J','F_0_8'])}
  for k,v in means.items():require(abs(d['mean_'+k]-v)<1e-10,'Final aggregate differs from its real case rows')
  records.append(dict(id=job['id'],model_id=job['model_id'],method=job['method'],variant=job['variant'],domain=domain,seed=job['seed'],scale=float(job['scale']),paths=256,means=means,evaluation_path=str(out/'evaluation.jsonl.gz'),evaluation_sha256=sha(out/'evaluation.jsonl.gz'),completed_path=str(out/'completed.json'),completed_sha256=sha(out/'completed.json'),physical_steps=d['physical_steps']))
 require(total_physical==1505280,'Fixed evaluation physical accounting differs')
 bundle=dict(schema='chapter5-formal-report-input-bundle-v1',root=str(R),contract=str(R/'contract.json'),contract_sha256=sha(R/'contract.json'),model_registry=str(R/'model_registry.json'),model_registry_sha256=sha(R/'model_registry.json'),training_audit=str(R/'training_audit.json'),training_audit_sha256=sha(R/'training_audit.json'),training_steps_per_model=c['learning_steps'],training_seeds=list(range(5)),development_every=512,development_paths=32,evaluation_grid=c['evaluation_grid'],curve_points=198,models=registry['models'],evaluations=records,raw_data_only=True,model_or_seed_selection=False,threshold_plot_semantics='Recount saved fixed-policy per-object task outcomes at alternative reporting thresholds; not retraining or policy adaptation to changed threshold.')
 bp=R/'report_bundle.json'
 if bp.exists():require(read(bp)==bundle,'Refuse changing a previously sealed report bundle')
 else:write(bp,bundle)
 report=dict(schema='chapter5-formal-210-independent-evaluation-audit-v1',status='PASS',training_models=90,evaluation_jobs=210,test_paths_per_job=256,formal_test_episodes=53760,formal_test_physical_steps=total_physical,new_learning_steps=0,optimizer_steps=0,all_raw_task_and_full_spend_ledgers_verified=True,all_domain_scale_panels_common_across_methods_and_seeds=True,contract_sha256=sha(R/'contract.json'),model_registry_sha256=sha(R/'model_registry.json'),report_bundle_sha256=sha(bp),auditor_sha256=sha(__file__))
 ap=R/'final_evaluation_audit.json'
 if ap.exists():require(read(ap)==report,'Refuse changing a previous distinct final evaluation seal')
 else:write(ap,report)
 print(json.dumps(dict(status='PASS_90_MODELS_210_REAL_FIXED_PANELS',bundle=str(bp),audit=str(ap),new_learning_steps=0)));return 0
if __name__=='__main__':sys.exit(main())
