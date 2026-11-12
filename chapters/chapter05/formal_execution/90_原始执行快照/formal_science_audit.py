"""No-Env independent audit and seal for the ninety actual formal endpoints."""
from pathlib import Path
import argparse, collections, datetime, gzip, hashlib, json, math, sys
R=Path(__file__).resolve().parents[1]
REGISTRY_SCHEMA='chapter5-formal-five-seed-model-registry-v1'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def physical_file(p):
 p=Path(p)
 if p.exists():return p
 q=p.with_name(p.name+'.gz');require(q.exists(),'Missing raw or losslessly compressed file: '+str(p));return q
def raw_open(p):
 p=physical_file(p);return gzip.open(p,'rb')if p.suffix=='.gz'else p.open('rb')
def raw_sha(p):
 h=hashlib.sha256()
 with raw_open(p)as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with raw_open(p)as f:
  for line in f:
   if line.strip():yield json.loads(line)
def grid(n):
 a=list(range(0,n+1,512));return a if a[-1]==n else a+[n]
def plan_only():
 c=read(R/'contract.json');p=read(R/'plan.json');require(c['schema']=='chapter5-formal-random-5seed-contract-v1','Wrong formal contract')
 require(p['contract_sha256']==sha(R/'contract.json'),'Plan contract SHA differs')
 require(c['science_seeds']==list(range(5))and c['learning_steps']==100608 and c['holdout_paths']==0 and c['dev_paths']==32 and c['evaluation_grid']==grid(100608),'Wrong formal geometry')
 js=p['jobs'];require(len(js)==90 and len({j['id']for j in js})==90,'Need ninety unique jobs')
 actual=collections.Counter()
 for j in js:
  x=read(j['config']);require(sha(j['config'])==j['config_sha256'],'Config drift: '+j['id']);require(x['contract_sha256']==sha(R/'contract.json'),'Foreign config contract')
  require(x['seed']in c['science_seeds']and x['learning_steps']==100608 and x['engineering']is False and x['holdout_paths']==0,'Foreign formal geometry')
  variant=x['profile']if x['method']=='full'else x['method'];actual[(variant,x['domain'],x['seed'])]+=1
  require(Path(x['output']).resolve()==(R/'jobs'/j['id']).resolve(),'Wrong output identity')
  require(x['source_manifest_sha256']==c['source_manifest_sha256']==sha(R/'execution/source_manifest.json'),'Frozen closure drift')
  require(x['execution_worker_sha256']==sha(R/'execution'/j['worker']),'Execution worker drift')
 expected={(m,d,s):1 for m in ['full','only_j','only_c','scalar_value','capql','qpensieve','cola','morld','pasta']for d in ['radio','v5']for s in range(5)}
 require(dict(actual)==expected,'Registered method/domain/seed grid is incomplete')
 manifest=read(R/'execution/source_manifest.json')
 for name,h in manifest['files'].items():
  fp=(R/'execution/source'/name).resolve();require(fp.is_relative_to(R/'execution/source')and sha(fp)==h,'Frozen scientific source drift: '+name)
 return c,p

def case(row,cfg):
 n=100 if cfg['domain']=='radio'else 120;h=32 if cfg['domain']=='radio'else 24;tau=.2 if cfg['domain']=='radio'else .7
 q=row['quality'];require(len(q)==n and all(math.isfinite(float(x))for x in q),'Malformed object task values')
 k=sum(float(x)>=tau for x in q);j=sum(float(x)for x in q)/n
 require(row['N']==n and row['K']==k and abs(row['C']-k/n)<1e-12 and abs(row['J']-j)<1e-10,'Task metrics do not equal saved physical outcomes')
 require(len(row['period_totals'])==h and all(math.isfinite(float(x))and float(x)>=0 for x in row['period_totals']),'Invalid complete episode resource ledger')
 b=.48 if cfg['domain']=='radio'else 20160.;proof=row['full_spend'];tol=float(proof['tolerance'])
 require(proof['full_spend_required']is True and 0<tol<=max(1e-8,b*1e-10),'Full spend missing or inflated tolerance')
 require(all(abs(float(a)-float(t))<=tol for a,t in [(row['initial_budget'],b),(row['total_used'],b),(row['remaining'],0),(sum(row['period_totals']),b)]),'Episode fails exact whole budget')
 require(isinstance(row['plan_sha256'],str)and len(row['plan_sha256'])==64,'Exogenous case hash missing')
 return k,j


def compact_training_case(row,cfg):
 """Native lean training rows retain aggregates/proofs; DEV retains full vectors."""
 n=100 if cfg['domain']=='radio' else 120;b=.48 if cfg['domain']=='radio' else 20160.
 require(row['domain']==cfg['domain']and row['phase']=='train'and row['N']==n and type(row['K'])is int and 0<=row['K']<=n and abs(row['C']-row['K']/n)<1e-12,'Malformed compact native training task row')
 require(math.isfinite(float(row['J']))and abs(row['native_return']-row['J'])<1e-10,'Compact native return differs from task indicator')
 proof=row['full_spend'];tol=float(proof['tolerance']);require(proof['full_spend_required']is True and 0<tol<=max(1e-8,b*1e-10),'Compact native full-spend proof invalid')
 for actual,target in [(row['initial_budget'],b),(row['total_used'],b),(row['remaining'],0),(proof['initial_budget'],b),(proof['executed_total'],row['total_used']),(proof['terminal_remaining'],0)]:require(abs(float(actual)-target)<=tol,'Compact native whole-budget ledger mismatch')
 require(len(row['plan_sha256'])==64,'Compact native case plan identity missing')

def audit_job(job,c,*,engineering_fixture=False):
 cfg=read(job['config']);out=Path(job['output']);d=read(out/'completed.json');h=c['domains'][cfg['domain']]['horizon'];counts=d['counts'];own=cfg['method']=='full'
 budget=cfg['learning_steps'];dev_paths=cfg['dev_paths'];is_formal=not engineering_fixture
 if engineering_fixture:
  require(cfg['engineering']is True and cfg['seed']in c['engineering']['seeds']and cfg['seed']not in c['science_seeds']and budget in c['engineering']['learning_steps'],'Reader fixture must be a real disjoint registered engineering run')
 else:require(cfg['engineering']is False and budget==100608 and cfg['seed']in c['science_seeds']and dev_paths==32,'Formal identity/geometry cannot be weakened by fixture readers')
 require(d['success']is True and d['engineering']is engineering_fixture and d['learning_steps']==budget and d['source_unchanged']is True,'Unsuccessful/nonformal endpoint')
 require(d['contract_sha256']==sha(R/'contract.json')and d['source_manifest_sha256']==c['source_manifest_sha256']and d['config_sha256']==sha(job['config'])and d['execution_worker_sha256']==sha(R/'execution'/job['worker']),'Completion provenance differs')
 for k in ['method','profile','domain','seed']:require(d[k]==cfg[k],'Completion identity changed: '+k)
 require(d['holdout_not_opened']is True and d['holdout_paths']==0 and counts['holdout']['physical_steps']==counts['holdout']['episodes']==0,'Training opened formal holdout')
 require(d['endpoint_reload_exact']is True and d['neural_parameters_changed']is True and d['optimizer_backup']is False and d['replay_backup']is False and d['updates']>0,'No genuine fixed inference endpoint')
 require(counts['train']['physical_steps']+counts.get('internal',{}).get('physical_steps',0)==budget,'Charged train+PSA length differs')
 require(all(v['physical_steps']==v['episodes']*h for v in counts.values()),'Physical accounting does not comprise complete episodes')
 endpoint=out/d['endpoint_filename'];require(sha(endpoint)==d['endpoint_sha256'],'Endpoint SHA changed')
 require(len(list(out.rglob('*.pt')))+len(list(out.rglob('*.pt.gz')))==1,'Expected exactly one retained endpoint binary')
 require(not(out/'failure.json').exists(),'Failure record cannot be relabelled completed')
 if cfg['device']=='cuda:0':
  g=d['gpu_proof'];expected_gpu=read(out/'protocol.json')['gpu_uuid'] if engineering_fixture else job['gpu_uuid']
  require(g and g['gpu_uuid']==expected_gpu and g['gpu_uuid']in c['gpu_uuids']and g['cuda_optimizer_tensors']>0 and g['cuda_update_event_ms']>0,'Missing real registered CUDA update proof')
 else:require(not d.get('gpu_proof')and d.get('cpu_proof'),'Explicit CPU proof missing')
 seen=[];dev_episodes=0;calibration=0
 for e in rows(out/'development.jsonl.gz'):
  vals=[case(x,cfg)for x in e['rows']];require(len(vals)==dev_paths,'DEV case count changed');dev_episodes+=len(vals)
  require(abs(e['mean_K']-sum(x[0]for x in vals)/dev_paths)<1e-10 and abs(e['mean_J']-sum(x[1]for x in vals)/dev_paths)<1e-10,'DEV aggregate changed')
  require(e['rng_and_learning_state_unchanged']is True,'DEV changed training/RNG state')
  if e['kind']=='curve':seen.append(e['learning_steps'])
  else:
   require(not own and e['kind']=='frozen_map_before_first_update'and e['learning_steps']==2048,'Unexpected off-grid DEV event');calibration+=1
 require(seen==grid(budget)and calibration<=1,'Complete actual 198-point curve required')
 require(counts['development']==dict(episodes=dev_episodes,physical_steps=dev_episodes*h),'DEV physical ledger differs')
 if own:
  require(d['registered_training']is is_formal and d['native_registered_formal_training']is is_formal and d['initialization_profile']==c['initialization_profile'],'New random native formal identity missing')
  heads=d['initialization_heads'];require(set(heads)=={'spatial','quota'},'Both random heads required')
  for name,offset in [('spatial',31000),('quota',83100)]:require(heads[name]['seed']==offset+cfg['seed']and all(len(heads[name][k])==64 for k in ['weight_sha256','bias_sha256']),'Random initialization head identity failed')
  native=read(out/'native/completed.json');proto=read(out/'native/protocol.json');reload=read(out/'native/endpoint_reload_verification.json')
  require(native['formal_training']is is_formal and native['evaluation_ticks']==seen and native['train_physical_steps']==budget and native['endpoint_sha256']==d['endpoint_sha256']and reload['status']=='pass','Native final source/reload/ticks mismatch')
  require(proto['initialization']['profile']==c['initialization_profile']and proto['initialization']['seed']==cfg['seed'],'Native random initializer receipt missing')
  ledger=out/'native/training_ledger.jsonl';require(raw_sha(ledger)==native['training_ledger_sha256'],'Original-byte training ledger SHA differs, including gzip fallback')
  train_rows=0
  for x in rows(ledger):case(x,cfg);train_rows+=1
  updates=sum(1 for _ in rows(out/'native/updates.jsonl'))
 else:
  require(d['formal_training_run']is is_formal and d['continuation_from_old_model']is False and d['loss_and_optimization_equations_modified']is False and d['final_rollout_batch_geometry_modified']is (cfg['method']=='pasta'),'Baseline formal identity changed')
  require(read(out/'holdout_not_opened.json')['status']==('NOT_OPENED_FORMAL_BASELINE_TRAINING'if is_formal else'NOT_OPENED_ENGINEERING_PREFLIGHT'),'Baseline holdout seal missing')
  train_rows=0
  for x in rows(out/'training.jsonl.gz'):compact_training_case(x,cfg);train_rows+=1
  updates=0;last_update=None
  for event in rows(out/'updates.jsonl.gz'):updates+=1;last_update=event
  if cfg['method']=='pasta':
   proof=d['terminal_rollout_adapter'];require(proof['exact_inverse_ast_equal'] is True and proof['final_batch_geometry_modified'] is True and proof['all_real_final_samples_optimized_without_padding'] is True,'Registered PASTA terminal adapter proof missing')
   stats=last_update.get('stats',last_update);require(stats['collected_real_training_steps']==stats['optimized_real_training_steps']==budget and stats['optimizer_updated'] is True,'PASTA real terminal samples were not fully optimized before sealing')
   full_rollout=h*cfg['n_env']*cfg['rollout_panels'];actual_final_batch=budget%full_rollout or full_rollout
   expected_minibatches=actual_final_batch//64*cfg['ppo_epochs']
   require(stats['minibatch_steps_this_group']==expected_minibatches,'PASTA final actual minibatch count differs')
  internal_steps=counts['internal']['physical_steps']
  internal_rows=0
  for packet in rows(out/'internal.jsonl.gz'):
   require(cfg['method']=='morld' and len(packet['rows'])==15 and packet['stats']['psa_applied'] is True and packet['stats']['value_source']=='charged_internal','Unexpected or uncharged PSA packet')
   for x in packet['rows']:case(x,cfg);internal_rows+=1
  require(internal_rows==counts['internal']['episodes'],'PSA raw episodes missing')
  if cfg['method']!='morld':require(internal_steps==0,'Only MORL/D uses charged PSA interactions')
 require(train_rows==counts['train']['episodes']and updates>0,'Complete train rows/native updates missing')
 if own:require(updates==d['updates'],'Native update record count differs')
 return dict(model_id=job['id'],method=cfg['method'],variant=cfg['profile'],profile=cfg['profile'],domain=cfg['domain'],seed=cfg['seed'],host=job['host'],device=cfg['device'],endpoint_kind='native_full'if own else'baseline_inference',endpoint_path=str(endpoint),endpoint_sha256=sha(endpoint),source_root=str(R/'execution/source'),source_manifest=str(R/'execution/source_manifest.json'),source_manifest_sha256=c['source_manifest_sha256'],config_path=job['config'],config_sha256=sha(job['config']),training_completed_path=str(out/'completed.json'),training_completed_sha256=sha(out/'completed.json'),development_path=str(out/'development.jsonl.gz'),development_sha256=sha(out/'development.jsonl.gz'),learning_steps=budget,initialization=c['initialization_profile']if own else'fresh_seeded_native_baseline_policy',formal_training=is_formal,curve_points=len(seen),charged_internal_steps=counts['internal']['physical_steps'],native_update_records=updates)

def main():
 a=argparse.ArgumentParser();a.add_argument('--validate-plan-only',action='store_true');args=a.parse_args();c,p=plan_only()
 if args.validate_plan_only:print(json.dumps(dict(status='PASS_PLAN_ONLY',jobs=90,environment_steps=0,holdout_calls=0)));return 0
 missing=[j['id']for j in p['jobs']if not(Path(j['output'])/'completed.json').exists()]
 if missing:
  write(R/'training_audit_pending.json',dict(status='NOT_READY_NO_TEST_OPENED',missing=missing,complete=90-len(missing),expected=90));return 2
 records=[audit_job(j,c)for j in p['jobs']]
 report=dict(schema='chapter5-formal-five-seed-independent-training-audit-v1',status='PASS',contract_sha256=sha(R/'contract.json'),plan_sha256=sha(R/'plan.json'),source_manifest_sha256=c['source_manifest_sha256'],training_jobs=90,models=90,learning_steps_per_model=100608,training_seeds=list(range(5)),development_points=198,training_holdout_steps=0,total_charged_training_steps=9054720,all_fixed_endpoints_fresh=True,no_test_opened_by_audit=True,auditor_sha256=sha(__file__),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 target=R/'model_registry.json'
 if target.exists():
  old=read(target);require(old['status']=='SEALED_ALL_FIXED_MODELS' and old['models']==records and old['contract_sha256']==report['contract_sha256'] and old['plan_sha256']==report['plan_sha256'] and sha(R/'training_audit.json')==old['training_audit_sha256'],'Refuse modifying an existing distinct seal')
  print(json.dumps(dict(status='PASS_ALREADY_SEALED_90',registry_sha256=sha(target),model_count=90,holdout_steps=0)));return 0
 write(R/'training_audit.json',report)
 registry=dict(schema=REGISTRY_SCHEMA,status='SEALED_ALL_FIXED_MODELS',all_training_complete=True,source_validation_passed=True,contract_sha256=sha(R/'contract.json'),plan_sha256=sha(R/'plan.json'),source_manifest_sha256=c['source_manifest_sha256'],training_audit_sha256=sha(R/'training_audit.json'),models=records,no_score_based_selection=True,reused_models=0,new_holdout_opened=False)
 target=R/'model_registry.json'
 if target.exists():require(read(target)==registry,'Refuse replacing an existing distinct sealed registry')
 else:write(target,registry)
 print(json.dumps(dict(status='PASS_SEALED_90',registry_sha256=sha(target),model_count=90,holdout_steps=0)));return 0
if __name__=='__main__':sys.exit(main())
