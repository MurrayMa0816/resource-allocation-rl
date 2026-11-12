"""Registered, no-update evaluation of Chapter5 fixed endpoints.

All source/model identities are bound before any evaluation reset. Formal test
cases open only after the complete 36-model registry is sealed. Each process
loads exactly one model's original frozen source; no old configuration, model,
normalization state, optimizer or scientific source is rewritten.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib
import json
import os
import random
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parent))
from budget_conditions import (SCHEMA as BUDGET_SCHEMA, assert_spec, make_env,
                               make_task, verify_condition)

SCHEMA='chapter5-final-fixed-model-evaluation-v1'
CONFIG_SCHEMA='chapter5-final-fixed-evaluation-config-v1'
REGISTRY_SCHEMA='chapter5-final-model-registry-v1'


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as handle:
        for block in iter(lambda:handle.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def validate(config_path):
    """Pure-file activation gate; Torch, models and physical tasks not imported."""
    config_path=Path(config_path).resolve(strict=True);cfg=read(config_path)
    require(cfg.get('schema')==CONFIG_SCHEMA,'Unknown fixed-evaluation config')
    contract_path=Path(cfg['contract']).resolve(strict=True);contract=read(contract_path)
    require(contract['schema']=='chapter5-final-execution-contract-v1'and sha(contract_path)==cfg['contract_sha256'],
            'Independent evaluation contract differs')
    registry_path=Path(cfg['model_registry']).resolve(strict=True);registry=read(registry_path)
    require(registry['schema']==REGISTRY_SCHEMA and sha(registry_path)==cfg['model_registry_sha256'],'Model registry seal changed')
    require(type(cfg.get('engineering'))is bool,'Explicit engineering identity required')
    models=registry['models'];require(isinstance(models,list)and models,'Nonempty model registry required')
    require(len({m['model_id']for m in models})==len(models),'Repeated model identities')
    if cfg['engineering']:
        require(contract['engineering']['permitted']is True and registry['status']=='ENGINEERING_FIXTURES_ONLY',
                'Engineering evaluation requires its own fixture registry')
        require(type(cfg['paths'])is int and 1<=cfg['paths']<=32,'Engineering cases must be a small bounded fixture')
        test_master=contract['evaluation']['engineering_test_master'];test_rng=contract['evaluation']['engineering_rng_seed']
    else:
        require(registry['status']=='SEALED_ALL_FIXED_MODELS'and registry['all_training_complete']is True
                and registry['source_validation_passed']is True and len(models)==contract['expected_models']==36,
                'Formal test remains closed before all 36 fixed models are sealed')
        require(cfg['paths']==contract['evaluation']['holdout_paths']==256 and
                cfg['batch_size']==contract['evaluation']['batch_size']==32,'Formal common panel geometry changed')
        test_master=contract['evaluation']['test_master'];test_rng=contract['evaluation']['test_rng_seed']
    matches=[m for m in models if m['model_id']==cfg['model_id']];require(len(matches)==1,'Unknown fixed model')
    model=matches[0];domain=model['domain'];require(domain in contract['domains'],'Unknown native task')
    if model['method']=='full':
        require(model['variant']in contract['supported_variants'],'Unknown registered Full/ablation identity')
    require(float(cfg['scale'])in contract['evaluation']['budget_scales'],'Unregistered resource condition')
    if float(cfg['scale'])!=1.:
        require(model['variant']=='full' or model['method']!='full','Budget transfer applies to main comparison models only')
    require(model['method']in contract['main_methods']and model['seed']in contract['science_seeds'],'Foreign method or seed')
    require(cfg['device']in contract['supported_devices'],'Unregistered evaluation device')
    if cfg['device']=='cuda:0':
        require(cfg['gpu_uuid']in contract['gpu_uuids'],'A registered physical GPU UUID is required')
    else:require(cfg.get('gpu_uuid')is None,'CPU evaluation must not claim a GPU')
    source=Path(model['source_root']).resolve(strict=True);manifest=Path(model['source_manifest']).resolve(strict=True)
    require(source.is_relative_to('/hy-tmp')and sha(manifest)==model['source_manifest_sha256'],'Frozen model source identity changed')
    original_path=Path(model['config_path']).resolve(strict=True);original=read(original_path)
    require(sha(original_path)==model['config_sha256'],'Original model configuration changed')
    require(original['domain']==domain and original['seed']==model['seed']and original['method']==model['method'],
            'Model/config task identity differs')
    if not cfg['engineering']:
        require(original.get('engineering')is False and original['learning_steps']==contract['learning_steps']==98304,
                'Formal comparison must use a complete registered scientific endpoint')
        done_path=Path(model['training_completed_path']).resolve(strict=True)
        require(sha(done_path)==model['training_completed_sha256'],'Original training completion SHA changed')
        done=read(done_path)
        require(done['success']is True and done['learning_steps']==98304 and done['source_unchanged']is True,
                'Formal model lacks a successful complete scientific training receipt')
        for key,value in [('method',model['method']),('domain',domain),('seed',model['seed']),
                          ('config_sha256',model['config_sha256']),('endpoint_sha256',model['endpoint_sha256']),
                          ('source_manifest_sha256',model['source_manifest_sha256'])]:
            require(done[key]==value,'Training completion/model-registry identity differs: '+key)
        require(isinstance(cfg.get('condition_file'),str)and isinstance(cfg.get('condition_sha256'),str)
                and len(cfg['condition_sha256'])==64,'Formal default and transfer conditions need explicit immutable identities')
    if model['method']!='full':
        require(original['profile']==contract['method_profiles'][model['method']],
                'The model does not use the selected method configuration')
    require(original['train_master']!=test_master and original['dev_master']!=test_master,'Evaluation master reuses training or DEV')
    endpoint=Path(model['endpoint_path']).resolve(strict=True)
    require(sha(endpoint)==model['endpoint_sha256'],'Fixed endpoint SHA changed')
    output=Path(cfg['output']).resolve();root=Path(contract['root']).resolve()
    require(root.is_relative_to('/hy-tmp')and output.is_relative_to(root)and output!=root,
            'Evaluation output must remain in the registered HY scope')
    require(not output.exists(),'Never overwrite or automatically retry an evaluation job')
    return dict(cfg=cfg,config_path=config_path,config_sha256=sha(config_path),contract=contract,
        contract_path=contract_path,registry_path=registry_path,model=model,source=source,manifest=manifest,
        original=original,original_path=original_path,endpoint=endpoint,output=output,
        test_master=int(test_master),test_rng=int(test_rng))


def verify_source_files(c):
    manifest=read(c['manifest']);files=manifest.get('files')
    require(isinstance(files,dict)and files,'Source manifest needs a complete relative-file closure')
    for name,digest in files.items():
        relative=Path(name);require(not relative.is_absolute()and'..'not in relative.parts,'Unsafe frozen source key')
        path=(c['source']/relative).resolve(strict=True)
        require(path.is_relative_to(c['source'])and sha(path)==digest,'Frozen scientific source changed: '+name)


def bind_source(c):
    verify_source_files(c)
    sys.path.insert(0,str(c['source']/'_internal/src'));sys.path.insert(0,str(c['source']))
    os.environ['CH5_WORKSPACE']=str(c['source']/'_internal/src')
    data_root=c['cfg'].get('data_root',c['original'].get('data_root',c['contract']['data_root']))
    os.environ['CH5_FULL_SPEND_DATA_ROOT']=str(data_root)
    env_module=importlib.import_module('chapter5_full_spend_20261104.env')
    bridge=env_module.bridge
    require(Path(env_module.__file__).resolve().is_relative_to(c['source']),'A foreign environment package was imported')
    condition=verify_condition(bridge,c['model']['domain'],c['cfg']['scale'],c['cfg'].get('condition_file'),c['cfg'].get('condition_sha256'))
    return env_module,bridge,data_root,condition


def module_provenance(source):
    result={}
    for name,module in list(sys.modules.items()):
        if not any(name==key or name.startswith(key+'.')for key in
                   ('allocation23','allocation_morl','allocation4','baselines','chapter5_full_spend_20261104')):
            continue
        filename=getattr(module,'__file__',None)
        require(filename is not None and Path(filename).resolve().is_relative_to(source),
                'A foreign scientific module was imported: '+name)
        result[name]=str(Path(filename).resolve())
    return result


def freeze_learning(agent):
    """Block optimizer/learner entry points without touching inference tensors."""
    import numpy as np
    import torch
    counts=dict(optimizer_steps=0,learning_calls=0,PSA_calls=0)
    optimizers=[];seen=set()
    def blocked_optimizer(*args,**kwargs):counts['optimizer_steps']+=1;raise RuntimeError('Optimizer.step forbidden in independent evaluation')
    def blocked_learning(*args,**kwargs):counts['learning_calls']+=1;raise RuntimeError('Learning forbidden in independent evaluation')
    def blocked_PSA(*args,**kwargs):counts['PSA_calls']+=1;raise RuntimeError('PSA/selection forbidden in independent evaluation')
    def visit(value):
        if id(value)in seen:return
        seen.add(id(value))
        if isinstance(value,torch.optim.Optimizer):optimizers.append(value);value.step=blocked_optimizer
        elif isinstance(value,(dict,list,tuple)):
            for child in(value.values()if isinstance(value,dict)else value):visit(child)
        elif hasattr(value,'__dict__')and not isinstance(value,(torch.Tensor,torch.nn.Module,np.ndarray)):
            for child in list(vars(value).values()):visit(child)
            for key in ('update','update_policy','update_rollout','after_exchange'):
                if callable(getattr(value,key,None)):setattr(value,key,blocked_learning)
            for key in ('accept_internal_evaluation','maybe_adapt','select_policy'):
                if callable(getattr(value,key,None)):setattr(value,key,blocked_PSA)
    visit(agent)
    return counts,optimizers


def verify_row(row,index,c):
    import numpy as np
    domain=c['model']['domain'];native=c['contract']['domains'][domain];n=native['N'];tau=native['task_tau']
    quality=np.asarray(row['quality'],np.float64);periods=np.asarray(row['period_totals'],np.float64)
    require(quality.shape==(n,)and np.isfinite(quality).all(),'Malformed native terminal object outcomes')
    require((quality>=(-1. if domain=='radio'else 0.)).all()and(quality<=1.).all(),'Native terminal outcomes outside declared range')
    k=int((quality>=tau).sum());j=float(quality.mean());coverage=k/n
    require(row['K']==k and abs(row['C']-coverage)<1e-12 and abs(row['J']-j)<1e-12 and
            abs(row['native_return']-j)<1e-10,'K/C/J do not match original terminal outcomes')
    require(periods.shape==(native['horizon'],)and np.isfinite(periods).all()and(periods>=0).all(),'Incomplete executed-period ledger')
    expected=(.48 if domain=='radio'else 20160.)*float(c['cfg']['scale']);proof=row['full_spend']
    tolerance=float(proof['tolerance'])
    require(proof['full_spend_required']is True and 0<tolerance<=max(1e-8,expected*1e-10),'Full-spend proof missing or inflated')
    for actual,target in [(row['initial_budget'],expected),(row['total_used'],expected),(row['remaining'],0.),
             (proof['initial_budget'],expected),(proof['executed_total'],row['total_used']),
             (proof['terminal_remaining'],row['remaining']),(float(periods.sum()),row['total_used'])]:
        require(abs(float(actual)-float(target))<=tolerance,'Actual whole-budget physical ledger differs')
    require(len(row['plan_sha256'])==64,'Exogenous case identity missing')
    row.update(domain=domain,index=index,N=n,tau=tau,scale=float(c['cfg']['scale']),
        phase='engineering'if c['cfg']['engineering']else'final',model_id=c['model']['model_id'],
        method=c['model']['method'],variant=c['model']['variant'],seed=c['model']['seed'],preference=.8,
        condition_sha256=c['cfg'].get('condition_sha256'),F_0_8=.8*coverage+.2*j)
    return row


def run(config_path):
    c=validate(config_path);cfg=c['cfg'];model=c['model'];original=c['original'];out=c['output']
    evaluator_sha=sha(__file__);adapter_sha=sha(Path(__file__).with_name('budget_conditions.py'))
    if cfg['device']=='cuda:0':
        require(os.environ.get('CUDA_VISIBLE_DEVICES')==cfg['gpu_uuid'],'Scheduler must expose exactly the registered physical GPU')
    else:require(os.environ.get('CUDA_VISIBLE_DEVICES','')=='','CPU evaluation must not expose CUDA devices')
    env_module,bridge,data_root,condition=bind_source(c)
    import numpy as np
    import torch
    from baselines._support.common import digest_state,cpu_state
    from baselines._support.runtime import rng_state,restore_rng,parameter_devices
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    if cfg['device']=='cuda:0':require(torch.cuda.is_available()and torch.cuda.device_count()==1,'Registered native CUDA inference unavailable')
    random.seed(model['seed']+91000);np.random.seed(model['seed']+92000);torch.manual_seed(model['seed']+93000)
    domain=model['domain'];obs_dim=305 if domain=='radio'else 244;action_dim=19 if domain=='radio'else 13
    initial_spec=None;critic=None;native_rollout=None
    if model['endpoint_kind']in('native_full','native'):
        task=make_task(bridge,domain,model['seed'],path_master=c['test_master'],holdout=not cfg['engineering'],
                       data_root=data_root,condition_file=condition)
        assert_spec(task.spec,domain,cfg['scale']);initial_spec=task.spec
        actor,critic,_=bridge.make_models(initial_spec,model['seed'],'fixed')
        payload=torch.load(c['endpoint'],map_location='cpu',weights_only=False)
        require(payload['format']=='chapter5-full-spend-fixed-endpoint-v1'and payload['optimizer_backup']is False
                and payload['replay_backup']is False,'Wrong native fixed endpoint format')
        require(payload['protocol']['domain']==domain and payload['protocol']['seed']==model['seed']
                and payload['protocol']['variant']==model['variant'],'Native endpoint policy identity differs')
        require(payload['learning_step']==original['learning_steps']and
                payload['protocol']['learning_steps']==original['learning_steps'],
                'Native endpoint is not the registered fixed training length')
        actor.load_state_dict(payload['actor'],strict=True);critic.load_state_dict(payload['critic'],strict=True)
        actor.to(cfg['device']);critic.to(cfg['device'])
        from allocation_morl.common import model_hash
        require(model_hash(actor)==payload['actor_sha256']and model_hash(critic)==payload['critic_sha256'],
                'Native fixed endpoint did not strictly reload')
        native_rollout=importlib.import_module('chapter5_full_spend_20261104.rollout')
        policy=dict(actor=actor,critic=critic);model_state=lambda:dict(actor=cpu_state(actor.state_dict()),critic=cpu_state(critic.state_dict()))
        if hasattr(task.env,'close'):task.env.close()
    else:
        module=importlib.import_module('baselines.'+model['method'])
        runtime_cfg={**original,'device':cfg['device']}
        policy=module.Agent(obs_dim,action_dim,model['seed'],cfg['device'],runtime_cfg)
        if not getattr(policy,'on_policy',False)and not getattr(policy,'manages_observation_normalization',False):
            normalizer=importlib.import_module('baselines._support.normalization')
            policy=normalizer.wrap_off_policy(policy,obs_dim,runtime_cfg)
        with gzip.open(c['endpoint'],'rb')as handle:payload=torch.load(handle,map_location='cpu',weights_only=False)
        require(payload['config']==original and payload['inference_only']is True and payload['training_restore_supported']is False,
                'Actual saved endpoint differs from original fixed training configuration')
        require(digest_state(payload['state'])==payload['model_sha256'],'Persisted baseline state digest differs')
        loader=getattr(policy,'load_checkpoint_state_dict',policy.load_state_dict);loader(payload['state'])
        exporter=getattr(policy,'checkpoint_state_dict',None)or getattr(policy,'endpoint',None)or policy.state_dict
        require(digest_state(cpu_state(exporter()))==payload['model_sha256'],'Baseline fixed endpoint failed exact inference reload')
        model_state=lambda:cpu_state(policy.state_dict())
    counts,optimizers=freeze_learning(policy)
    before_model=digest_state(model_state());before_optimizer=digest_state([o.state_dict()for o in optimizers])
    actual_modules=module_provenance(c['source'])
    saved=rng_state(policy)
    def rng_digest(state):return digest_state((state[0],state[1],state[2],state[3],[(kind,value)for owner,kind,value in state[4]]))
    before_rng=rng_digest(saved);envs=[];rows=[];physical=0;started=time.monotonic();cuda_event_ms=0.
    out.mkdir(parents=True)
    write(out/'protocol.json',dict(schema=SCHEMA,config=cfg,config_sha256=c['config_sha256'],model_record=model,
        original_config_sha256=model['config_sha256'],source_manifest_sha256=model['source_manifest_sha256'],
        endpoint_sha256=model['endpoint_sha256'],evaluator_sha256=evaluator_sha,budget_adapter_sha256=adapter_sha,
        fixed_model=True,new_training_steps=0,optimizer_steps=0,model_selection=False,
        engineering=cfg['engineering'],test_master=c['test_master'],test_rng_seed=c['test_rng'],
        condition_sha256=cfg.get('condition_sha256'),condition_scale=float(cfg['scale']),
        observation_normalizer_frozen=True,public_Task_Spec_and_decoder_budget_consistent=True,
        loaded_scientific_modules=actual_modules,test_split_opened=not cfg['engineering']))
    try:
        try:
            random.seed(c['test_rng']);np.random.seed(c['test_rng']);torch.manual_seed(c['test_rng'])
            batch_size=min(int(cfg['batch_size']),int(cfg['paths']))
            if initial_spec is not None:
                envs=[make_task(bridge,domain,model['seed'],path_master=c['test_master'],holdout=not cfg['engineering'],
                    data_root=data_root,condition_file=condition)for _ in range(batch_size)]
                require(all(task.spec==initial_spec for task in envs)and actor.spec.initial_budget==initial_spec.initial_budget,
                        'Native Actor.spec and physical Task budgets differ')
            else:
                envs=[make_env(env_module,domain,model['seed'],phase='final',path_master=c['test_master'],
                    holdout=not cfg['engineering'],action_mode=original['action_mode'],data_root=data_root,condition_file=condition)
                    for _ in range(batch_size)]
                require(all(e.observation_dim==obs_dim and e.action_dim==action_dim for e in envs),'Public inference geometry changed')
                for e in envs:assert_spec(e.native_spec,domain,cfg['scale'])
            begin=end=None
            if cfg['device']=='cuda:0':
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
            with gzip.open(out/'evaluation.jsonl.gz','wt',compresslevel=1)as handle:
                for first in range(0,cfg['paths'],batch_size):
                    batch=envs[:min(batch_size,cfg['paths']-first)]
                    if initial_spec is not None:
                        generator=torch.Generator(device='cpu').manual_seed(c['test_rng'])
                        result=native_rollout.rollout(batch,actor,critic,generator,'final',first,
                            tau=c['contract']['domains'][domain]['task_tau'],preferences=.8,deterministic=True,training=False)
                        values=[native_rollout.result_row(row,first+j)for j,row in enumerate(result)]
                        physical+=len(batch)*initial_spec.horizon
                    else:
                        observations=np.stack([e.reset(options={'episode_index':first+j})[0]for j,e in enumerate(batch)])
                        preferences=np.tile([.8,.2],(len(batch),1)).astype(np.float32)
                        for tick in range(c['contract']['domains'][domain]['horizon']):
                            actions=policy.act(observations,preferences,deterministic=True)
                            transition=[e.step(action)for e,action in zip(batch,actions)]
                            require(all(row[2]==(tick==c['contract']['domains'][domain]['horizon']-1)and not row[3]and not row[4]
                                for row in transition),'Fixed model evaluation truncated or exposed private intermediate info')
                            observations=np.stack([row[0]for row in transition]);physical+=len(batch)
                        values=[]
                        for e in batch:
                            result=e.terminal_result
                            row={key:result[key]for key in ('index','plan_sha256','quality','K','C','J','native_return','period_totals',
                                                        'total_used','remaining','full_spend')}
                            row['initial_budget']=e.native_spec.initial_budget
                            for key in ('quality','period_totals'):row[key]=np.asarray(row[key]).tolist()
                            values.append(row)
                    for j,row in enumerate(values):
                        row=verify_row(row,first+j,c);rows.append(row)
                        handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    handle.flush()
                    write(out/'progress.json',dict(paths=len(rows),target_paths=cfg['paths'],physical_steps=physical,
                        new_training_steps=0,model_id=model['model_id'],scale=cfg['scale'],seconds=time.monotonic()-started))
            if end is not None:end.record();torch.cuda.synchronize();cuda_event_ms=begin.elapsed_time(end)
        finally:
            for environment in envs:
                if hasattr(environment,'close'):environment.close()
                elif hasattr(environment,'env')and hasattr(environment.env,'close'):environment.env.close()
            restore_rng(saved)
        after_model=digest_state(model_state());after_rng=rng_digest(rng_state(policy))
        require(before_model==after_model and before_rng==after_rng,'Fixed evaluation changed model or process/private RNG')
        require(before_optimizer==digest_state([o.state_dict()for o in optimizers])and not any(counts.values()),
                'Fixed evaluation performed a learner, optimizer or PSA update')
        require(len(rows)==cfg['paths']and physical==cfg['paths']*c['contract']['domains'][domain]['horizon'],
                'Incomplete fixed-policy panel')
        require(sha(c['endpoint'])==model['endpoint_sha256']and sha(c['original_path'])==model['config_sha256']
                and sha(c['manifest'])==model['source_manifest_sha256'],'Bound model/config/source changed during evaluation')
        require(sha(c['registry_path'])==cfg['model_registry_sha256']and sha(c['contract_path'])==cfg['contract_sha256']
                and sha(c['config_path'])==c['config_sha256']and sha(__file__)==evaluator_sha,
                'Evaluation contract/registry/config/code changed')
        require(sha(Path(__file__).with_name('budget_conditions.py'))==adapter_sha,'Shared budget adapter changed')
        verify_source_files(c)
        if cfg.get('condition_file')is not None:
            require(sha(cfg['condition_file'])==cfg['condition_sha256'],'Budget condition changed during evaluation')
        means={key:float(np.mean([row[key]for row in rows]))for key in('K','C','J','F_0_8')}
        done=dict(success=True,schema=SCHEMA,model_id=model['model_id'],method=model['method'],variant=model['variant'],
            domain=domain,seed=model['seed'],scale=float(cfg['scale']),paths=len(rows),physical_steps=physical,
            new_training_steps=0,optimizer_steps=0,PSA_calls=0,model_unchanged=True,rng_unchanged=True,source_unchanged=True,
            endpoint_sha256=model['endpoint_sha256'],config_sha256=c['config_sha256'],
            original_config_sha256=model['config_sha256'],model_registry_sha256=cfg['model_registry_sha256'],
            contract_sha256=cfg['contract_sha256'],evaluator_sha256=evaluator_sha,budget_adapter_sha256=adapter_sha,
            source_manifest_sha256=model['source_manifest_sha256'],evaluation_sha256=sha(out/'evaluation.jsonl.gz'),
            condition_sha256=cfg.get('condition_sha256'),device=cfg['device'],gpu_uuid=cfg.get('gpu_uuid'),
            parameter_devices=parameter_devices(policy),cuda_inference_event_ms=cuda_event_ms,
            test_master=c['test_master'],test_rng_seed=c['test_rng'],engineering=cfg['engineering'],
            test_split_opened=not cfg['engineering'],formal_test_cases_opened=0 if cfg['engineering']else len(rows),
            model_before_sha256=before_model,model_after_sha256=after_model,rng_before_sha256=before_rng,rng_after_sha256=after_rng,
            optimizer_state_before_sha256=before_optimizer,optimizer_state_after_sha256=before_optimizer,
            no_new_model_saved=True,normalization_state_unchanged=True,model_selection=False,holdout_selection=False,
            mean_K=means['K'],mean_C=means['C'],mean_J=means['J'],mean_F_0_8=means['F_0_8'],seconds=time.monotonic()-started)
        write(out/'completed.json',done);return done
    except BaseException:
        write(out/'failure.json',dict(success=False,schema=SCHEMA,model_id=model['model_id'],physical_steps=physical,
            new_training_steps=0,traceback=traceback.format_exc(),no_automatic_retry=True));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args()
    print(json.dumps(run(args.config),ensure_ascii=False))
