"""Source-bound, terminal-only full-episode benchmark for replacement candidates.

The algorithm receives public observations/actions/rewards only. Physical ledger
data are validated after training and never supplied to a learner.
"""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root
import argparse, copy, hashlib, importlib, json, os, random, sys, time, traceback
from pathlib import Path

TRANSITION_METHODS = frozenset(['fac', 'srsac', 'redo'])
AUDIT_ENVS = {}
AUDIT_AGENT = None

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def plain(v):
    if isinstance(v,dict): return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [plain(x) for x in v]
    if hasattr(v,'detach'): return plain(v.detach().cpu().numpy())
    if hasattr(v,'tolist'): return v.tolist()
    if isinstance(v,Path): return str(v)
    return v

def write(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(plain(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(q,p)

def append(p,v):
    with p.open('a') as f:f.write(json.dumps(plain(v),ensure_ascii=False,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();cfg=json.loads(a.config.read_text())
    out=Path(cfg['output_dir']).resolve();scope=Path(cfg['scope']).resolve()
    assert out.is_relative_to(scope/'jobs') and str(scope).startswith(str(_resolve_path('/hy-tmp/')))
    assert not out.exists(),'Never overwrite an experiment output'
    assert cfg['method'] in ['rtpg','gpi','fac','srsac','redo'] and cfg['domain'] in ['radio','v5']
    assert cfg['eval_paths'] in [2,32] and cfg['dev_master']==202610303101
    manifest=json.loads(Path(cfg['source_manifest']).read_text())
    for p,h in manifest['files'].items(): assert digest(p)==h, p
    out.mkdir(parents=True);write(out/'config.json',cfg)
    try: run(cfg,out,manifest,a.config)
    except BaseException as e:
        costs={name:int(env.real_steps) for name,env in AUDIT_ENVS.items()}
        updates=int(getattr(AUDIT_AGENT,'n_updates',getattr(AUDIT_AGENT,'updates',0))) if AUDIT_AGENT is not None else 0
        write(out/'failure.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time(),actual_environment_steps=costs,actual_updates=updates))
        raise

def run(cfg,out,manifest,config_path):
    global AUDIT_AGENT
    import numpy as np
    import torch
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    assert cfg['device']=='cuda' and torch.cuda.is_available()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==cfg['physical_gpu_uuid']
    torch.cuda.set_device(0)
    seed=int(cfg['seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    envroot=Path(cfg['environment_root']).resolve();sys.path.insert(0,str(envroot))
    from chapter4_full_spend_env import bootstrap
    identity=bootstrap.install(audit_dir=out/'resource_audit',context=dict(method=cfg['method'],scope=Path(cfg['scope']).name,role=cfg['role']))
    from chapter4_recent_baselines.env_adapter import Env
    common=dict(domain=cfg['domain'],seed=seed,action_mode=cfg['action_mode'],observation_dtype=np.float64)
    train=Env(**common,phase='train');dev=Env(**common,phase='dev',path_master=cfg['dev_master'])
    AUDIT_ENVS.update(train=train,dev=dev)
    h=train.horizon;expected=(305,19,32) if cfg['domain']=='radio' else (244,13,24)
    assert (train.observation_dim,train.action_dim,h)==expected
    assert cfg['train_steps']%h==0 and cfg['eval_interval']%h==0 and cfg['train_steps']%cfg['eval_interval']==0
    n_ep=cfg['train_steps']//h
    if cfg['method'] not in TRANSITION_METHODS:assert n_ep%cfg['batch_episodes']==0
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    variant=cfg.get('execution_variant','native')
    assert variant=='native' or (cfg['method']=='redo' and variant=='steady_graph') or (cfg['method']=='srsac' and variant=='gpu_replay_cache')
    module='graph_agent' if cfg['method']=='redo' and variant=='steady_graph' else 'agent'
    Agent=importlib.import_module('replacement_candidates.'+cfg['method']+'.'+module).Agent
    algo=dict(cfg['algorithm']);algo.update(seed=seed,total_steps=cfg['train_steps'])
    agent=Agent(algo,train.observation_dim,train.action_dim,device='cuda:0')
    AUDIT_AGENT=agent
    if cfg['method']=='srsac':
        assert hasattr(agent,'native_state_fingerprint') and hasattr(agent,'get_rng_state') and hasattr(agent,'set_rng_state')
        import jax
        assert jax.default_backend()=='gpu' and all(x.platform=='gpu' for x in jax.devices())
    scale=np.asarray(cfg.get('action_scale',[1.]*train.action_dim),dtype=np.float64)
    assert scale.shape==(train.action_dim,) and np.isfinite(scale).all() and (scale>0).all()
    if np.any(scale!=1):assert cfg['action_mode']=='raw' and cfg['method'] in ['rtpg','gpi']
    execution_contract=dict(mapping='environment_raw = action_scale * algorithm_action',action_scale=scale.tolist(),
        stored_actions_and_means='algorithm coordinates',stored_log_prob='algorithm density; physical density subtracts constant logdet',
        physical_logdensity_constant=-float(np.log(scale).sum()),no_dimension_dropped=True)
    write(out/'execution_coordinates.json',execution_contract)
    write(out/'protocol.json',dict(environment=train.protocol,source_manifest=digest(cfg['source_manifest']),source_identity=identity,physical_gpu_uuid=cfg['physical_gpu_uuid'],logical_device=str(torch.cuda.current_device()),native_torch=torch.__version__,learner_backend='native_jax' if cfg['method']=='srsac' else 'torch',algorithm_gamma=algo.get('gamma'),dtype_boundary='float64 physical public observations; learner dtype declared by each algorithm',new_holdout_calls=0))
    def parameters_hash():
        result={}
        for key in ['actor','critic','value','model']:
            value=getattr(agent,key,None)
            if key=='model' and hasattr(value,'policy'):value=value.policy
            if not isinstance(value,torch.nn.Module):continue
            z=hashlib.sha256()
            for n,v in value.state_dict().items():z.update(n.encode());z.update(v.detach().cpu().contiguous().numpy().tobytes())
            result[key]=z.hexdigest()
        if hasattr(agent,'native_state_fingerprint'):
            result['native_state']=plain(agent.native_state_fingerprint())
        return result
    initial=parameters_hash()
    if hasattr(agent,'initial_fingerprints'):initial.update(agent.initial_fingerprints())
    write(out/'initial_policy.json',initial)
    evaluations=[];update_rows=0;real_steps=0;pending=[];roundtrip_obs=[]
    start=time.time()
    def rng_state():
        extra={}
        for k,v in vars(agent).items():
            if isinstance(v,np.random.Generator):extra[k]=('generator',copy.deepcopy(v.bit_generator.state))
            elif isinstance(v,np.random.RandomState):extra[k]=('randomstate',v.get_state())
        native=copy.deepcopy(agent.get_rng_state()) if hasattr(agent,'get_rng_state') else None
        return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all(),extra,native)
    def restore(s):
        random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2]);torch.cuda.set_rng_state_all(s[3])
        for k,(kind,v) in s[4].items():
            if kind=='generator':getattr(agent,k).bit_generator.state=v
            else:getattr(agent,k).set_state(v)
        if hasattr(agent,'set_rng_state'):agent.set_rng_state(s[5])
    def evaluation():
        saved=rng_state();before=parameters_hash();scores=[];policy_action=getattr(agent,'policy_action',agent.act)
        try:
            for index in range(cfg['eval_paths']):
                obs,_=dev.reset(options={'episode_index':index});total=0.
                for t in range(h):
                    action=np.asarray(policy_action(obs,deterministic=False));obs,reward,done,truncated,info=dev.step(scale*action)
                    assert done==(t==h-1) and not truncated and info=={} and (reward==0 or done)
                    total+=reward
                scores.append(total)
        finally:restore(saved)
        assert before==parameters_hash(),'Development evaluation changed policy state'
        row=dict(train_steps=real_steps,episode=real_steps//h,returns=scores,mean_score=100*float(np.mean(scores)),std_return=float(np.std(scores)),dev_paths=cfg['eval_paths'])
        evaluations.append(row);write(out/'evaluations.json',evaluations);append(out/'evaluations.jsonl',row)
        write(out/'progress.json',dict(status='running',train_steps=real_steps,target_steps=cfg['train_steps'],episodes=real_steps//h,updates=update_rows,last_dev_score=row['mean_score'],wall_seconds=time.time()-start))
    evaluation()
    for ep in range(n_ep):
        obs,_=train.reset(options={'episode_index':ep});trace={k:[] for k in ['obs','next_obs','actions','rewards','dones','log_probs','behavior_means']};ret=0.
        version=getattr(agent,'updates',0)
        for t in range(h):
            action=np.asarray(agent.act(obs,deterministic=False))
            assert action.shape==(train.action_dim,) and np.isfinite(action).all()
            if cfg['method'] not in TRANSITION_METHODS:
                inf=(agent.distribution_info(obs) if hasattr(agent,'distribution_info') else agent.behavior_info(obs,action))
                mean=inf.get('mean',inf.get('means'))
            lp=float(agent.log_prob(obs,action)) if cfg['method'] not in TRANSITION_METHODS else 0.
            nxt,reward,done,truncated,info=train.step(scale*action);real_steps+=1;ret+=reward
            assert not truncated and done==(t==h-1) and info=={} and (done or reward==0)
            if cfg['method'] in TRANSITION_METHODS:
                metrics=agent.observe(obs,action,reward,nxt,done)
                if real_steps%128==0:
                    diagnostics=agent.diagnostics()
                    assert diagnostics.get('train_steps',real_steps)==real_steps
                    append(out/'updates.jsonl',{**diagnostics,'train_steps':real_steps})
            else:
                for k,v in [('obs',obs),('next_obs',nxt),('actions',action),('rewards',reward),('dones',done),('log_probs',lp),('behavior_means',mean)]:trace[k].append(v)
            if len(roundtrip_obs)<8:roundtrip_obs.append(np.array(obs,copy=True))
            obs=nxt
        if cfg['method'] not in TRANSITION_METHODS:
            trace={k:np.asarray(v) for k,v in trace.items()};trace['policy_version']=version;pending.append(trace)
            if len(pending)==cfg['batch_episodes']:
                metrics=agent.update(pending);pending=[];update_rows+=1;append(out/'updates.jsonl',dict(train_steps=real_steps,**(metrics or {})))
        else:update_rows=agent.n_updates
        append(out/'train_episodes.jsonl',dict(episode=ep+1,train_steps=real_steps,return_raw=ret,score_x100=100*ret))
        if real_steps%cfg['eval_interval']==0:evaluation()
        elif (ep+1)%8==0:write(out/'progress.json',dict(status='running',train_steps=real_steps,target_steps=cfg['train_steps'],episodes=ep+1,updates=update_rows,wall_seconds=time.time()-start))
    assert not pending and real_steps==cfg['train_steps']
    assert len(evaluations)==1+cfg['train_steps']//cfg['eval_interval'] and train.completed_episodes==n_ep
    model=out/'endpoint.pt';agent.save(model)
    if np.any(scale!=1):
        payload=torch.load(model,map_location='cpu',weights_only=False)
        payload['execution_coordinates']=execution_contract
        torch.save(payload,model)
    loaded=Agent.load(model,device='cuda:0')
    policy_action=getattr(agent,'policy_action',agent.act);loaded_action=getattr(loaded,'policy_action',loaded.act)
    errors=[float(np.max(np.abs(scale*policy_action(o,deterministic=True)-scale*loaded_action(o,deterministic=True)))) for o in roundtrip_obs]
    assert max(errors)<2e-6
    ledgers=[json.loads(line) for p in (out/'resource_audit').glob('resource_episodes_*.jsonl') for line in p.read_text().splitlines()]
    counts={phase:sum(x['phase']==phase for x in ledgers) for phase in ['train','dev']}
    assert counts==dict(train=n_ep,dev=cfg['eval_paths']*len(evaluations))
    assert all(x['terminal_full_spend_assertion'] and x['nonterminal_rewards_zero'] for x in ledgers)
    assert max(x['relative_allocation_error'] for x in ledgers)<1e-10
    assert all(x['maximum_negative_allocation_violation']==x['maximum_per_object_cap_violation']==x['maximum_completed_object_allocation']==0 for x in ledgers)
    for p,v in manifest['files'].items(): assert digest(p)==v,p
    final_policy=parameters_hash()
    if hasattr(agent,'initial_fingerprints'):final_policy.update(agent.initial_fingerprints())
    write(out/'completion.json',dict(status='PASS',method=cfg['method'],role=cfg['role'],domain=cfg['domain'],seed=seed,train_steps=real_steps,development_steps=dev.real_steps,episodes=n_ep,updates=update_rows,development_points=len(evaluations),resource_episode_counts=counts,max_resource_error=max(x['relative_allocation_error'] for x in ledgers),source_manifest_sha256=digest(cfg['source_manifest']),config_sha256=digest(config_path),model_path=str(model),model_sha256=digest(model),roundtrip_max_error=max(errors),physical_gpu_uuid=cfg['physical_gpu_uuid'],device='cuda:0',wall_seconds=time.time()-start,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),cuda_memory_counter_scope='Torch allocator only; excludes native JAX' if cfg['method']=='srsac' else 'Torch allocator',final_policy_fingerprints=final_policy,one_inference_model=True,new_holdout_calls=0,final_diagnostics=agent.diagnostics() if hasattr(agent,'diagnostics') else {}))
    write(out/'progress.json',dict(status='complete',train_steps=real_steps,target_steps=cfg['train_steps'],episodes=n_ep,updates=update_rows,wall_seconds=time.time()-start))
    print(json.dumps(dict(status='PASS',method=cfg['method'],domain=cfg['domain'],seed=seed,real_steps=real_steps,wall_seconds=time.time()-start)),flush=True)

if __name__=='__main__':main()
