"""Formal no-update evaluator; closed until all ninety endpoints are independently sealed."""
from pathlib import Path
import argparse, importlib.util, json, sys
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'execution'))
from formal_science_audit import REGISTRY_SCHEMA, require, read, sha
CONFIG_SCHEMA='chapter5-formal-five-seed-fixed-evaluation-config-v1'
SCHEMA='chapter5-formal-five-seed-fixed-evaluation-v1'

def validate(config_path):
 p=Path(config_path).resolve(strict=True);cfg=read(p)
 require(cfg.get('schema')==CONFIG_SCHEMA and cfg.get('engineering')is False,'This evaluator accepts registered formal fixed models only')
 cp=Path(cfg['contract']).resolve(strict=True);c=read(cp)
 require(cp==R/'contract.json'and c['schema']=='chapter5-formal-random-5seed-contract-v1'and sha(cp)==cfg['contract_sha256'],'Formal evaluation contract differs')
 rp=Path(cfg['model_registry']).resolve(strict=True);registry=read(rp)
 require(rp==R/'model_registry.json'and registry['schema']==REGISTRY_SCHEMA and sha(rp)==cfg['model_registry_sha256'],'Registry seal differs')
 require(registry['status']=='SEALED_ALL_FIXED_MODELS'and registry['all_training_complete']is True and registry['source_validation_passed']is True and len(registry['models'])==90==c['expected_models'],'All ninety actual formal models must be sealed before any test reset')
 audit=read(R/'training_audit.json');require(sha(R/'training_audit.json')==registry['training_audit_sha256']and audit['status']=='PASS'and audit['contract_sha256']==sha(cp)and audit['models']==90 and audit['training_holdout_steps']==0,'Independent actual-training gate is missing')
 require(c['evaluation']['opens_only_after_all90_training_and_registry_sealed']is True,'Wrong test activation policy')
 require(cfg['paths']==c['evaluation']['holdout_paths']==256 and cfg['batch_size']==c['evaluation']['batch_size']==32,'Formal common panel geometry changed')
 found=[m for m in registry['models']if m['model_id']==cfg['model_id']];require(len(found)==1,'Unknown fixed model')
 m=found[0];domain=m['domain'];require(domain in c['domains']and m['method']in c['main_methods']and m['seed']in c['science_seeds'],'Foreign formal model')
 require(float(cfg['scale'])in c['evaluation']['budget_scales'],'Unregistered resource condition')
 if m['method']=='full':require(m['variant']in c['supported_variants'],'Unknown objective variant')
 if float(cfg['scale'])!=1.:require(m['method']!='full'or m['variant']=='full','Budget transfer belongs to six main methods only')
 require(cfg['device']in c['supported_devices'],'Unregistered inference device')
 if cfg['device']=='cuda:0':require(cfg['gpu_uuid']in c['gpu_uuids'],'Foreign inference GPU')
 else:require(cfg.get('gpu_uuid')is None,'CPU inference cannot claim a GPU')
 source=Path(m['source_root']).resolve(strict=True);manifest=Path(m['source_manifest']).resolve(strict=True)
 require(source==R/'execution/source'and manifest==R/'execution/source_manifest.json'and sha(manifest)==m['source_manifest_sha256']==c['source_manifest_sha256'],'Frozen model source identity differs')
 op=Path(m['config_path']).resolve(strict=True);original=read(op)
 require(sha(op)==m['config_sha256']and original['engineering']is False and original['learning_steps']==c['learning_steps']and original['holdout_paths']==0,'Original formal endpoint geometry differs')
 for k in ['method','domain','seed']:require(original[k]==m[k],'Original endpoint identity differs: '+k)
 donep=Path(m['training_completed_path']).resolve(strict=True);done=read(donep)
 require(sha(donep)==m['training_completed_sha256']and done['success']is True and done['learning_steps']==c['learning_steps']and done['source_unchanged']is True and done['holdout_not_opened']is True,'Actual formal training receipt differs')
 for k in ['method','domain','seed','config_sha256','source_manifest_sha256','endpoint_sha256']:require(done[k]==m[k],'Completion registry identity differs: '+k)
 if m['method']=='full':require(done['initialization_profile']==c['initialization_profile']and done['native_registered_formal_training']is True,'Old or nonrandom Full cannot enter new formal pool')
 else:require(original['profile']==c['method_profiles'][m['method']],'Selected native baseline recipe differs')
 master=c['evaluation']['test_master'];rng=c['evaluation']['test_rng_seed']
 require(master==c['test_master']and master not in(original['train_master'],original['dev_master'])and cfg['test_master']==master and cfg['test_rng_seed']==rng,'Test split/RNG differs or overlaps training/DEV')
 endpoint=Path(m['endpoint_path']).resolve(strict=True);require(sha(endpoint)==m['endpoint_sha256'],'Fixed endpoint SHA differs')
 condition=Path(cfg['condition_file']).resolve(strict=True);require(condition.is_relative_to(R/'evaluation_conditions')and sha(condition)==cfg['condition_sha256'],'Explicit immutable resource condition required')
 require(cfg['training_worker_sha256']==original['execution_worker_sha256']==sha(R/'execution'/('own_worker.py'if m['method']=='full'else 'baseline_worker.py')),'Registered training facade changed')
 require(cfg['execution_worker_sha256']==sha(__file__)and cfg['native_evaluator_sha256']==sha(R/'execution/evaluator_native.py')and cfg['budget_adapter_sha256']==sha(R/'execution/budget_conditions.py'),'Evaluation source identity differs')
 output=Path(cfg['output']).resolve();require(output==R/'evaluation_jobs'/cfg['id']and not output.exists(),'Output is foreign, exists, or would overwrite a prior attempt')
 return dict(cfg=cfg,config_path=p,config_sha256=sha(p),contract=c,contract_path=cp,registry_path=rp,model=m,source=source,manifest=manifest,original=original,original_path=op,endpoint=endpoint,output=output,test_master=int(master),test_rng=int(rng))

def main():
 a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--validate-only',action='store_true');args=a.parse_args()
 c=validate(args.config)
 if args.validate_only:print(json.dumps(dict(status='PASS_PURE_FILE_ACTIVATION_GATE',test_resets=0)));return 0
 spec=importlib.util.spec_from_file_location('chapter5_formal_frozen_evaluator_native',R/'execution/evaluator_native.py');backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
 pasta_constructor_adaptation=None
 if c['model']['method']=='pasta':
  # The fixed100608 endpoint has a registered final partial complete-episode
  # rollout. Inference constructs the same architecture but never trains.
  # Admit that registered budget in __init__ only; no learner function changes.
  backend.bind_source(c)
  import baseline_worker as training_facade
  module=backend.importlib.import_module('baselines.pasta.agent')
  original=(c['source']/training_facade.PASTA_NAME).read_text()
  adapted=training_facade.adapted_pasta_methods(original)['__init__']
  namespace=dict(module.__dict__);exec(compile(adapted,str(c['source']/training_facade.PASTA_NAME),'exec'),namespace)
  module.Agent.__init__=namespace['__init__']
  pasta_constructor_adaptation=dict(training_facade_sha256=sha(R/'execution/baseline_worker.py'),native_agent_sha256=training_facade.NATIVE_PASTA_SHA,only_constructor_budget_admission_modified=True,no_training_method_installed=True,no_optimizer_update=True)
 native_write=backend.write
 def write_registered(path,payload):
  if Path(path).name in ['protocol.json','completed.json']:
   payload=dict(payload,formal_evaluator_version=SCHEMA,formal_evaluator_sha256=sha(__file__),fixed_registry_sha256=sha(R/'model_registry.json'),formal_learning_steps=c['contract']['learning_steps'],formal_training_seed_count=5,formal_training_audit_sha256=sha(R/'training_audit.json'),pasta_constructor_adaptation=pasta_constructor_adaptation)
  return native_write(path,payload)
 backend.validate=validate;backend.SCHEMA=SCHEMA;backend.write=write_registered
 backend.run(args.config);return 0
if __name__=='__main__':sys.exit(main())
