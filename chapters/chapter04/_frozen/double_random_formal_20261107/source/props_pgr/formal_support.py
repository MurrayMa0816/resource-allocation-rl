"""Registered C4 I/O/environment/evaluation only; no reward shaping or learner math."""
from __future__ import annotations
import argparse,copy,csv,gzip,hashlib,importlib.util,json,os,random,sys,time
from pathlib import Path

SCOPE=Path('/hy-tmp/resource_allocation_results/chapter4/double_random_formal_20261107')
HERE=SCOPE/'source/props_pgr'
ACCEL=Path('/root/resource_allocation_run_storage_20261101/ch4_acceleration_v3_20261103')
DEFAULT_ENV=SCOPE/'source/environment'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb')as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
    p=Path(path).resolve()
    if not p.is_relative_to(SCOPE):raise ValueError('new formal-scope HY output only')
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.writing');q.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def parser(method):
    p=argparse.ArgumentParser()
    p.add_argument('--domain',choices=['radio','v5'],required=True);p.add_argument('--method',choices=[method],default=method)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=100608)
    p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','cuda','cuda:0'],default='cpu'if method=='props'else'cuda')
    p.add_argument('--train-master',type=int,default=202611043101);p.add_argument('--dev-master',type=int,default=202611043101)
    p.add_argument('--eval-every',type=int,default=768);p.add_argument('--dev-cases',type=int,default=32)
    p.add_argument('--environment-root',default=str(DEFAULT_ENV));p.add_argument('--engineering-gate',action='store_true')
    p.add_argument('--evaluate-model');p.add_argument('--eval-master',type=int,default=2026110694199)
    p.add_argument('--eval-cases',type=int,default=256);p.add_argument('--eval-phase',choices=['dev','final'],default='final')
    return p
def configure(args):
    out=Path(args.output).resolve()
    if not out.is_relative_to(SCOPE)or(out.exists()and any(out.iterdir())):raise ValueError('Use a fresh new-scope output directory')
    out.mkdir(parents=True,exist_ok=True)
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
    for k in ['TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR']:
        p=out/'runtime'/k.lower();p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
    os.environ['WANDB_MODE']='disabled';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    return out

class Context:
    def __init__(self,args):
        self.args=args;self.out=configure(args);self.start=time.perf_counter();self.training_steps=0
        self.evaluation_steps=0;self.resource_episodes=0;self.resource_counts={};self.max_resource_error=0.
        self.resource_file=gzip.open(self.out/'resource_audit.csv.gz','wt',newline='')
        self.resource_columns=['phase','episode_index','steps','initial_budget','total_used','remaining','relative_allocation_error','maximum_step_balance_absolute_error','maximum_negative_allocation_violation','maximum_per_object_cap_violation','maximum_completed_object_allocation','terminal_reward','nonterminal_rewards_zero','terminal_full_spend_assertion','rounds_sha256']
        self.resource_writer=csv.DictWriter(self.resource_file,fieldnames=self.resource_columns);self.resource_writer.writeheader()
        self.eval_file=gzip.open(self.out/'evaluation_paths.csv.gz','wt',newline='')
        self.eval_writer=csv.DictWriter(self.eval_file,fieldnames=['training_step','phase','path_index','path_seed','native_return','resource_fraction','steps']);self.eval_writer.writeheader()
        self.curve_file=gzip.open(self.out/'evaluation_curve.csv.gz','wt',newline='')
        self.curve_writer=csv.DictWriter(self.curve_file,fieldnames=['training_step','phase','paths','evaluation_environment_steps','native_mean','native_std','min_return','max_return']);self.curve_writer.writeheader()
        self.progress('initializing')
    def progress(self,phase='training',extra=None):
        row=dict(schema='chapter4-formal-100k-worker-v1',pid=os.getpid(),method=self.args.method,domain=self.args.domain,seed=self.args.seed,phase=phase,training_steps=self.training_steps,target_training_steps=self.args.steps,evaluation_steps=self.evaluation_steps,elapsed_seconds=time.perf_counter()-self.start,actual_device=self.args.device,resource_episodes=self.resource_episodes,max_resource_error=self.max_resource_error)
        row.update(extra or {});dump(self.out/'progress.json',row)
    def audit(self,env):
        a=env.task.env.last_full_spend_audit
        if a is None:raise AssertionError('Native full-spend terminal audit is missing')
        if not a['terminal_full_spend_assertion']or not a['nonterminal_rewards_zero']or a['relative_allocation_error']>1e-10:raise AssertionError('physical/terminal contract failed')
        if any(a[k]!=0 for k in ['maximum_negative_allocation_violation','maximum_per_object_cap_violation','maximum_completed_object_allocation']):raise AssertionError('physical allocation violation')
        row={k:a[k]for k in self.resource_columns if k!='rounds_sha256'}
        row['rounds_sha256']=hashlib.sha256(json.dumps(a['rounds'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        self.resource_writer.writerow(row);self.resource_file.flush();self.resource_episodes+=1
        self.resource_counts[a['phase']]=self.resource_counts.get(a['phase'],0)+1
        self.max_resource_error=max(self.max_resource_error,a['relative_allocation_error'])
    def install(self):
        sys.path.insert(0,str(Path(self.args.environment_root).resolve()))
        from chapter4_full_spend_env.bootstrap import install
        self.binding=install(audit_dir=None,context=dict(scope=str(SCOPE),method=self.args.method,seed=self.args.seed,engineering=self.args.engineering_gate))
        dump(self.out/'bridge_identity.json',self.binding)
        from chapter4_recent_baselines import env_adapter
        Base=env_adapter.Env;ctx=self
        class RegisteredEnv(Base):
            def __init__(self,domain,seed=0,phase='train',action_mode='tanh_latent',**kwargs):
                if phase in ['train','engineering']:master=ctx.args.train_master
                elif phase=='dev':master=ctx.args.dev_master
                else:master=ctx.args.eval_master
                kwargs['path_master']=master
                if phase=='final':kwargs['holdout']=True
                super().__init__(domain,seed,phase,action_mode,**kwargs)
            def step(self,action):
                value=super().step(action)
                if value[2]:ctx.audit(self)
                return value
        env_adapter.Env=RegisteredEnv;env_adapter.make_env=lambda domain,seed=0,**kw:RegisteredEnv(domain,seed,**kw)
        self.Env=RegisteredEnv
        dump(self.out/'evaluation_protocol.json',dict(training_master=self.args.train_master,development_master=self.args.dev_master,evaluation_master=self.args.eval_master,development_cases=self.args.dev_cases,development_interval=self.args.eval_every,deterministic=True,action_sampling_disabled=True,RNG_isolated=True,physical_contract='fullspend terminal reward; exact inherited bridge',resource_storage='complete per-episode physical assertions+scalar audit+SHA of round ledger; no uncompressed repeated rounds'))
        return self.Env
    def evaluate(self,action_fn,steps,*,action_mode,phase='dev',cases=None,policy=None):
        import numpy as np,torch
        cases=self.args.dev_cases if cases is None else int(cases)
        if cases<1:raise ValueError('positive evaluation case count required')
        saved=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
        mode=policy.training if policy is not None else None
        env=None;values=[];self.progress('development'if phase=='dev'else'holdout')
        try:
            random.seed(4000000);np.random.seed(4000000);torch.manual_seed(4000000)
            if torch.cuda.is_initialized():torch.cuda.manual_seed_all(4000000)
            env=self.Env(self.args.domain,self.args.seed,phase=phase,action_mode=action_mode,holdout=phase=='final')
            if policy is not None:policy.eval()
            for i in range(cases):
                obs,_=env.reset(options={'episode_index':i});done=False;value=0.
                while not done:
                    action=action_fn(obs)
                    obs,reward,terminated,truncated,_=env.step(action)
                    if truncated:raise AssertionError('true finite horizon only')
                    done=terminated;value+=float(reward)
                values.append(value)
                self.eval_writer.writerow(dict(training_step=steps,phase=phase,path_index=i,path_seed=env.task.plan(phase,i)['task_seed'],native_return=value,resource_fraction=env.task.env.last_full_spend_audit['total_used']/env.native_spec.initial_budget,steps=env.horizon))
            count=env.real_steps
        finally:
            if env is not None:env.close()
            if policy is not None:policy.train(mode)
            random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2])
            if saved[3]is not None:torch.cuda.set_rng_state_all(saved[3])
        assert random.getstate()==saved[0]and np.array_equal(np.random.get_state()[1],saved[1][1])and torch.equal(torch.get_rng_state(),saved[2])
        if saved[3]is not None:assert all(torch.equal(a,b)for a,b in zip(torch.cuda.get_rng_state_all(),saved[3]))
        self.evaluation_steps+=count;self.eval_file.flush()
        result=dict(training_step=steps,phase=phase,paths=cases,evaluation_environment_steps=count,native_mean=float(np.mean(values)),native_std=float(np.std(values,ddof=1))if cases>1 else 0.,min_return=float(min(values)),max_return=float(max(values)))
        self.curve_writer.writerow(result);self.curve_file.flush();self.progress('training')
        return dict(train_env_steps=steps,evaluation_env_steps=count,episodes=cases,mean_return=result['native_mean'],mean_score=result['native_mean']*100,std_return=result['native_std'],returns=values,phase=phase,deterministic_eval=True,training_rng_restored=True,evaluation_master=self.args.dev_master if phase=='dev'else self.args.eval_master)
    def close(self):
        for f in [self.resource_file,self.eval_file,self.curve_file]:f.close()
    def completion(self,receipt,model,bindings,*,evaluation_only=False):
        self.close()
        for p in self.out.glob('*.gz'):
            with gzip.open(p,'rb')as f:
                for _ in iter(lambda:f.read(2**20),b''):pass
        p=Path(model)if model else None
        out=dict(schema='chapter4-formal-100k-worker-completion-v1',status='completed',success=True,method=self.args.method,domain=self.args.domain,seed=self.args.seed,engineering_only=self.args.engineering_gate,evaluation_only=evaluation_only,training_steps=0 if evaluation_only else self.training_steps,evaluation_steps=self.evaluation_steps,fullspend_episode_counts=self.resource_counts,max_resource_error=self.max_resource_error,model=str(p)if p else None,model_sha256=sha(p)if p else None,source_binding=bindings,actual_device=self.args.device,wall_seconds=time.perf_counter()-self.start,native_receipt=receipt,optimizer_or_replay_saved=False,one_fixed_final_model=not evaluation_only,exact_resume_available=False,periodic_models_saved=False,model_deserialization_for_evaluation_only=evaluation_only)
        dump(self.out/'completion.json',out);dump(self.out/'completed.json',out);self.progress('complete',{'success':True});return out

def bindings():
    paths=list(HERE.glob('*.py'))+[HERE/'copy_provenance.json']+[SCOPE/'execution'/n for n in ('props_worker.py','pgr_worker.py')]
    return {str(p):sha(p)for p in sorted(paths)}
