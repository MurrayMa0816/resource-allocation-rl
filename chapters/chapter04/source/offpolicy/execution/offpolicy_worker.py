"""Selected native ReDo, MR.Q and BRO; common paid-step/dev/final contract.

No old queue is imported or resumed. Complete inference endpoints are the only
models saved. All learning replay and optimizers remain RAM-only. The physical
full-spend guard still executes; its complete ledger is streamed directly to
gzip instead of producing a large uncompressed secondary copy.
"""
from __future__ import annotations
import argparse, copy, csv, datetime, gzip, hashlib, importlib, json, os
from pathlib import Path
import random, resource, sys, time, traceback

SCOPE=Path(__file__).resolve().parents[1]
SOURCE=SCOPE/'source/offpolicy'
DEFAULT_ENV=SCOPE/'source/environment'
DEFAULTS={
 'redo':dict(gamma=.99,reward_scale=1.,warmup_steps=10000,replay_ratio=8,
    batch_size=256,recycle_period=200000,dormant_threshold=0.,initial_log_alpha=0.,
    reset_mode='neurons',reset_actor=True,reset_critic=True,
    reset_target_models=False,scale_recycled_weights=False),
 'mrq':dict(gamma=.99),
 'bro':dict(actor_lr=.0003,critic_lr=.0003,temp_lr=.0003,adj_lr=.00003,
    gamma=.99,tau=.005,target_entropy=None,init_temperature=1.,init_optimism=1.,
    init_regularizer=.25,pessimism=0.,n_quantiles=100,kl_target=.05,
    std_multiplier=.75,start_training=2500,updates_per_step=2,batch_size=128,
    replay_capacity=1000000,reward_scale=1.)}


def plain(x):
    if isinstance(x,dict):return {str(k):plain(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v)for v in x]
    if isinstance(x,Path):return str(x)
    if hasattr(x,'detach'):return plain(x.detach().cpu().numpy())
    if hasattr(x,'tolist'):return x.tolist()
    return x


def canonical(x):return json.dumps(plain(x),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb')as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(plain(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


class GzipRows:
    def __init__(self,path):self.stream=gzip.open(path,'wt',compresslevel=1,encoding='utf-8');self.count=0;self.digest=hashlib.sha256()
    def add(self,row):
        line=canonical(row)+'\n';self.stream.write(line);self.digest.update(line.encode());self.count+=1
    def flush(self):self.stream.flush()
    def close(self):self.stream.close()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=DEFAULTS,required=True)
    p.add_argument('--domain',choices=['radio','v5'],required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--steps',type=int,default=100608)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--train-master',type=int,default=202611043101)
    p.add_argument('--dev-master',type=int,default=202611043101)
    p.add_argument('--eval-every',type=int,default=768)
    p.add_argument('--dev-cases',type=int,default=32)
    p.add_argument('--environment-root',type=Path,default=DEFAULT_ENV)
    p.add_argument('--source-manifest',type=Path)
    p.add_argument('--engineering-gate',action='store_true')
    p.add_argument('--evaluate-model',type=Path)
    p.add_argument('--eval-phase',choices=['dev','final'],default='final')
    p.add_argument('--eval-master','--final-master',type=int,default=2026110694199)
    p.add_argument('--eval-cases','--final-cases',type=int,default=256)
    return p


def run(a):
    out=a.output.resolve()
    assert out.is_relative_to(SCOPE) and not out.exists(), 'New output under this scope required'
    assert a.seed>=0 and a.dev_cases>0 and a.eval_cases>0
    h=32 if a.domain=='radio'else 24
    if not a.evaluate_model:
        assert a.steps>0 and a.steps%h==0 and a.eval_every>0 and a.eval_every%h==0
        if not a.engineering_gate:
            assert a.steps==100608 and a.seed in range(5) and a.eval_every==768 and a.dev_cases==32
            assert a.train_master==a.dev_master==202611043101
    if a.evaluate_model and a.eval_phase=='final':assert a.eval_master!=a.train_master
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    for key in ['TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','JAX_COMPILATION_CACHE_DIR']:
        p=out/'runtime'/key.lower();p.mkdir(parents=True,exist_ok=False);os.environ[key]=str(p)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    os.environ['JAX_ENABLE_COMPILATION_CACHE']='false'
    os.environ['WANDB_MODE']='disabled'
    if a.device.startswith('cpu'):os.environ['JAX_PLATFORMS']='cpu'
    worker_sha=sha(__file__)
    manifest_path=SOURCE/'agent_source_manifest.json';manifest=json.loads(manifest_path.read_text())
    for path,digest in manifest['files'].items():assert sha(path)==digest,path
    if a.source_manifest:
        parent_manifest=json.loads(a.source_manifest.read_text())
        for path,digest in parent_manifest.get('files',{}).items():assert sha(path)==digest,path
    import numpy as np
    import torch
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch_gpu=a.device.startswith('cuda')and a.method!='bro'
    if a.device.startswith('cuda'):assert torch.cuda.is_available()
    if torch_gpu:torch.cuda.set_device(0)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch_gpu:torch.cuda.manual_seed_all(a.seed)
    sys.path.insert(0,str(a.environment_root.resolve()))
    from chapter4_full_spend_env import bootstrap
    identity=bootstrap.install(audit_dir=None,context=dict(scope=SCOPE.name,method=a.method,role='engineering'if a.engineering_gate else'formal'))
    from chapter4_recent_baselines.env_adapter import Env
    sys.path.insert(0,str(SOURCE))
    module='graph_agent'if a.method=='redo'and torch_gpu else'agent'
    Agent=importlib.import_module('offpolicy_agents.'+a.method+'.'+module).Agent
    envkwargs=dict(domain=a.domain,seed=a.seed,action_mode='tanh_latent',observation_dtype=np.float64)
    train=None if a.evaluate_model else Env(**envkwargs,phase='train',path_master=a.train_master)
    dev=None if a.evaluate_model else Env(**envkwargs,phase='dev',path_master=a.dev_master)
    evaluator=Env(**envkwargs,phase=a.eval_phase,path_master=a.eval_master,holdout=a.eval_phase=='final')if a.evaluate_model else None
    env=train if train is not None else evaluator
    assert (env.observation_dim,env.action_dim,env.horizon)==((305,19,32)if a.domain=='radio'else(244,13,24))
    # A short MR.Q engineering prefix must still instantiate the registered
    # long-budget native profile (author adapter correctly rejects <=10k total
    # budgets). Physical gate cost remains a.steps and is never overstated.
    native_budget=100608 if a.engineering_gate and a.method=='mrq'and a.steps<=10000 else a.steps
    algo={**DEFAULTS[a.method], 'seed':a.seed,'total_steps':native_budget}
    agent=Agent.load(a.evaluate_model,device=a.device)if a.evaluate_model else Agent(algo,env.observation_dim,env.action_dim,device=a.device)
    if a.method=='bro':
        assert agent.device.platform==('cpu'if a.device.startswith('cpu')else'gpu')
    write(out/'config.json',{**vars(a),'algorithm':algo,'eval_deterministic':True})
    write(out/'protocol.json',dict(schema='chapter4-formal-selected-offpolicy-v1',environment=env.protocol,
        train_master=a.train_master,dev_master=a.dev_master,final_master=a.eval_master,
        source_identity=identity,agent_source_manifest_sha256=sha(manifest_path),worker_sha256=worker_sha,
        parent_source_manifest_sha256=sha(a.source_manifest)if a.source_manifest else None,
        device=a.device,physical_gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        learner_backend='jax'if a.method=='bro'else'torch',deterministic_evaluation=True,
        registered_actual_steps=a.steps,native_agent_total_step_cap=native_budget,
        evaluation_never_updates_or_enters_replay=True,full_spend_guard_unchanged=True,
        complete_physical_ledgers='streamed gzip plus canonical SHA; physical guard audit_dir=None prevents duplicate plain file',
        observation_dtype='float64 public; native neural dtype unchanged',action_mode='tanh_latent',
        new_holdout_calls=bool(a.evaluate_model and a.eval_phase=='final'),engineering_gate=a.engineering_gate,
        one_fixed_inference_endpoint=True,periodic_models=False,disk_replay=False,disk_optimizers=False))
    episodes=GzipRows(out/'train_episodes.jsonl.gz');updates=GzipRows(out/'updates.jsonl.gz')
    ledger_rows=GzipRows(out/'resource_ledger.jsonl.gz');case_rows=GzipRows(out/'evaluation_paths.jsonl.gz')
    curves=[];roundtrip=[];steps=0;start=time.perf_counter();update_seconds=0.;train_action_seconds=0.;dev_seconds=0.
    max_error=0.;resource_counts={'train':0,'dev':0,'final':0};full_spend_count=0
    def rng_state():
        extra={}
        for k,v in vars(agent).items():
            if isinstance(v,np.random.Generator):extra[k]=('generator',copy.deepcopy(v.bit_generator.state))
            elif isinstance(v,np.random.RandomState):extra[k]=('randomstate',v.get_state())
        native=copy.deepcopy(agent.get_rng_state())if hasattr(agent,'get_rng_state')else None
        return random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()if torch_gpu else[],extra,native
    def restore(s):
        random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2])
        if s[3]:torch.cuda.set_rng_state_all(s[3])
        for k,(kind,v)in s[4].items():
            if kind=='generator':getattr(agent,k).bit_generator.state=v
            else:getattr(agent,k).set_state(v)
        if hasattr(agent,'set_rng_state'):agent.set_rng_state(s[5])
    def fingerprint():
        return canonical(agent.native_state_fingerprint())
    def resource_summary(env,phase,index):
        nonlocal max_error,full_spend_count
        record=env.task.env.last_full_spend_audit
        assert record['terminal_full_spend_assertion']and record['nonterminal_rewards_zero']
        assert record['initial_budget']>0 and record['steps']==h
        err=record['relative_allocation_error'];assert err<1e-10
        assert all(record[k]==0 for k in ['maximum_negative_allocation_violation','maximum_per_object_cap_violation','maximum_completed_object_allocation'])
        max_error=max(max_error,err);resource_counts[phase]+=1;full_spend_count+=1
        ledger_rows.add(record)
        return dict(resource_fraction=record['total_used']/record['initial_budget'],remaining=record['remaining'],
            resource_error=err,resource_ledger_sha256=hashlib.sha256(canonical(record).encode()).hexdigest(),
            initial_budget=record['initial_budget'],total_used=record['total_used'],terminal_full_spend=True)
    def progress(status='running'):
        write(out/'progress.json',dict(status=status,pid=os.getpid(),method=a.method,domain=a.domain,seed=a.seed,
            train_steps=steps,training_steps=steps,target_steps=0 if a.evaluate_model else a.steps,
            episodes=steps//h,updates=int(agent.n_updates),development_points=len(curves),
            development_steps=dev.real_steps if dev is not None else 0,
            last_dev_score=curves[-1]['mean_score']if curves else None,
            wall_seconds=time.perf_counter()-start))
    def evaluation(e,count,phase):
        nonlocal dev_seconds
        tick=time.perf_counter();saved=rng_state();before=fingerprint();returns=[];case_plans=[]
        try:
            policy=getattr(agent,'policy_action',agent.act)
            for index in range(count):
                plan=e.task.plan(phase,index);case_id=hashlib.sha256(canonical(plan).encode()).hexdigest()
                obs,_=e.reset(options={'episode_index':index});ret=0.
                for t in range(h):
                    action=np.asarray(policy(obs,deterministic=True));obs,r,done,truncated,info=e.step(action)
                    assert done==(t==h-1)and not truncated and info=={}and(r==0 or done);ret+=r
                physical=resource_summary(e,phase,index)
                case_rows.add(dict(training_step=steps,train_steps=steps,phase=phase,path_index=index,
                    path_seed=plan.get('task_seed'),case_identity_sha256=case_id,native_return=ret,
                    score_x100=100*ret,steps=h,**physical))
                case_plans.append(dict(path_index=index,plan=plan,case_identity_sha256=case_id));returns.append(ret)
        finally:restore(saved)
        assert before==fingerprint(),'Evaluation mutated native learning state or sampling RNG'
        row=dict(train_steps=steps,training_step=steps,episode=steps//h,phase=phase,returns=returns,
            native_mean=float(np.mean(returns)),native_std=float(np.std(returns,ddof=1))if len(returns)>1 else 0.,
            mean_score=100*float(np.mean(returns)),dev_paths=count,development_cases=count,
            evaluation_environment_steps=e.real_steps,deterministic=True)
        curves.append(row)
        with(out/'evaluations.jsonl').open('a')as f:f.write(canonical(row)+'\n')
        write(out/'evaluations.json',curves)
        if len(curves)==1:write(out/'evaluation_case_manifest.json',dict(phase=phase,path_master=e.path_master,cases=case_plans))
        ledger_rows.flush();case_rows.flush();dev_seconds+=time.perf_counter()-tick;progress()
    initial=fingerprint();write(out/'initial_policy.json',dict(native_state=initial))
    try:
        if a.evaluate_model:
            evaluation(evaluator,a.eval_cases,a.eval_phase)
            model=a.evaluate_model.resolve();roundtrip_error=0.
        else:
            evaluation(dev,a.dev_cases,'dev')
            for ep in range(a.steps//h):
                obs,_=train.reset(options={'episode_index':ep});ret=0.
                for t in range(h):
                    tick=time.perf_counter();action=np.asarray(agent.act(obs,deterministic=False));train_action_seconds+=time.perf_counter()-tick
                    assert action.shape==(train.action_dim,)and np.isfinite(action).all()
                    nxt,reward,done,truncated,info=train.step(action);steps+=1;ret+=reward
                    assert done==(t==h-1)and not truncated and info=={}and(reward==0 or done)
                    tick=time.perf_counter();agent.observe(obs,action,reward,nxt,done);update_seconds+=time.perf_counter()-tick
                    if len(roundtrip)<8:roundtrip.append(np.array(obs,copy=True))
                    if steps%128==0:updates.add({**agent.diagnostics(),'train_steps':steps,'wall_seconds':time.perf_counter()-start})
                    obs=nxt
                episodes.add(dict(episode=ep+1,train_steps=steps,training_step=steps,native_return=ret,
                    return_raw=ret,score_x100=100*ret,**resource_summary(train,'train',ep)))
                if steps%a.eval_every==0:evaluation(dev,a.dev_cases,'dev')
                elif(ep+1)%8==0:progress()
            if curves[-1]['train_steps']!=steps:evaluation(dev,a.dev_cases,'dev')
            assert train.real_steps==steps==a.steps and train.completed_episodes==a.steps//h
            assert int(agent.n_updates)>0,'Native update gate must pass; do not report a warmup-only run'
            model=out/('endpoint.msgpack'if a.method=='bro'else'endpoint.pt');agent.save(model)
            saved=rng_state();native_action=getattr(agent,'policy_action',agent.act)
            expected=[np.asarray(native_action(o,deterministic=True)).copy()for o in roundtrip]
            loaded=Agent.load(model,device=a.device);loaded_action=getattr(loaded,'policy_action',loaded.act)
            errors=[float(np.max(np.abs(x-np.asarray(loaded_action(o,deterministic=True)))))for x,o in zip(expected,roundtrip)]
            restore(saved);roundtrip_error=max(errors);assert roundtrip_error<2e-6
        final_diagnostics=agent.diagnostics();final_fingerprint=fingerprint()
        expected_points=1+(a.steps//a.eval_every)+(int(a.steps%a.eval_every>0))if not a.evaluate_model else 1
        assert len(curves)==expected_points
        assert resource_counts['train']==steps//h
        assert resource_counts['dev']==(a.dev_cases*len(curves)if not a.evaluate_model else(a.eval_cases if a.eval_phase=='dev'else 0))
        if a.evaluate_model:assert resource_counts[a.eval_phase]==a.eval_cases
        for path,digest in manifest['files'].items():assert sha(path)==digest,path
        assert sha(__file__)==worker_sha,'Worker source changed during this run'
        resource_report=dict(counts=resource_counts,complete_full_spend_episodes=full_spend_count,
            maximum_relative_allocation_error=max_error,ledger_rows=ledger_rows.count,
            canonical_uncompressed_ledger_sha256=ledger_rows.digest.hexdigest(),
            ledger_gzip='resource_ledger.jsonl.gz',learner_receives_ledger=False)
        write(out/'resource_summary.json',resource_report)
        completion=dict(status='PASS',method=a.method,domain=a.domain,seed=a.seed,
            train_steps=steps,training_steps=steps,episodes=steps//h,updates=int(agent.n_updates),
            development_steps=dev.real_steps if dev is not None else 0,
            final_evaluation_steps=evaluator.real_steps if evaluator is not None else 0,
            development_points=len(curves)if not a.evaluate_model else 0,
            evaluation_points=len(curves),evaluation_deterministic=True,model_path=str(model),model_sha256=sha(model),
            one_inference_model=True,roundtrip_max_error=roundtrip_error,final_diagnostics=final_diagnostics,
            initial_policy_sha256=hashlib.sha256(initial.encode()).hexdigest(),
            final_policy_sha256=hashlib.sha256(final_fingerprint.encode()).hexdigest(),
            agent_source_manifest_sha256=sha(manifest_path),worker_sha256=worker_sha,
            train_master=a.train_master,dev_master=a.dev_master,final_master=a.eval_master,
            device=a.device,physical_gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
            learner_backend='jax'if a.method=='bro'else'torch',engineering_gate=a.engineering_gate,
            wall_seconds=time.perf_counter()-start,observe_update_seconds=update_seconds,
            train_action_seconds=train_action_seconds,development_wall_seconds=dev_seconds,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            torch_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()if torch_gpu else 0,
            torch_cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved()if torch_gpu else 0,
            memory_counter_excludes_jax=a.method=='bro',new_holdout_calls=a.eval_cases if a.evaluate_model and a.eval_phase=='final'else 0,
            resource_summary=resource_report,config_sha256=sha(out/'config.json'))
        write(out/'completion.json',completion);write(out/'completed.json',completion);progress('complete')
        print(canonical(completion),flush=True)
    except BaseException as e:
        write(out/'failure.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),
            train_steps=steps,development_steps=dev.real_steps if dev is not None else 0,
            final_evaluation_steps=evaluator.real_steps if evaluator is not None else 0,
            updates=int(agent.n_updates),wall_seconds=time.perf_counter()-start))
        raise
    finally:
        episodes.close();updates.close();ledger_rows.close();case_rows.close()


if __name__=='__main__':run(parser().parse_args())
