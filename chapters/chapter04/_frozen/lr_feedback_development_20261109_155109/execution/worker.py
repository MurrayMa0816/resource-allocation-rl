"""Development-only four-arm LR comparison with deterministic fixed-path DEV.

Only final inference weights are saved. Source F1/rollout mathematics are
imported unchanged. All full-spend physical guards remain active; bulky ledger
duplicates are replaced by compressed per-episode summaries plus stream SHA.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True

SCOPE=Path(__file__).resolve().parents[1]
SOURCE_DIR=Path(__file__).resolve().parent
ORIGINAL_SCOPE=Path('/hy-tmp/resource_allocation_results/chapter04/experiments/double_random_formal_20261107')
SCHEMA='chapter04-lr-feedback-development-evaluation-only-v1'
METHODS=('original_feedback','fixed_3e4','fixed_1e4','relaxed_feedback')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n');t.replace(p)


class CSV:
    def __init__(self,path,fields):
        self.stream=gzip.open(path,'xt',newline='')
        self.writer=csv.DictWriter(self.stream,fieldnames=fields);self.writer.writeheader()
        self.count=0
    def add(self,row):self.writer.writerow(row);self.count+=1
    def flush(self):self.stream.flush()
    def close(self):self.stream.close()


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--domain',choices=('radio','v5'),required=True)
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--steps',type=int,default=49152)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--train-master',type=int,default=202611043101)
    p.add_argument('--dev-master',type=int,default=202611043101)
    p.add_argument('--eval-every',type=int,default=768)
    p.add_argument('--dev-cases',type=int,default=32)
    p.add_argument('--environment-root','--env-root',type=Path,default=ORIGINAL_SCOPE/'source/environment')
    p.add_argument('--engineering-gate',action='store_true')
    p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--case-id',required=True)
    p.add_argument('--expected-registry-sha',required=True)
    return p


def validate_registration(args):
    path=args.registry.resolve()
    if digest(path)!=args.expected_registry_sha:
        raise ValueError('Registered job file SHA differs from launch seal')
    registry=json.loads(path.read_text())
    rows=registry['jobs']
    if len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Repeated registered job identity')
    matches=[r for r in rows if r['id']==args.case_id]
    if len(matches)!=1:raise ValueError('Exactly one registered job must match')
    row=matches[0]
    for name in ('method','domain','seed','steps','device','train_master','dev_master','eval_every','dev_cases','engineering_gate'):
        if row[name]!=getattr(args,name):
            raise ValueError('Registered job mismatch: '+name)
    if Path(row['output']).resolve()!=args.output.resolve():
        raise ValueError('Output differs from registered exact case')
    scope=path.parent.parent.resolve()
    expected='engineering' if args.engineering_gate else 'development'
    if registry['phase']!=expected:
        raise ValueError('Engineering/development phase mismatch')
    if not args.output.resolve().is_relative_to(scope) or not scope.is_relative_to(Path('/hy-tmp')):
        raise ValueError('Independent registered physical HY output required')
    if args.environment_root.resolve()!=(ORIGINAL_SCOPE/'source/environment').resolve():
        raise ValueError('The original double-random physical task is fixed')
    if args.eval_every!=768 or args.dev_cases!=32:
        raise ValueError('DEV protocol is fixed to 768 steps and 32 cases')
    if not args.engineering_gate and args.steps!=49152:
        raise ValueError('Development endpoint is fixed to 49152 steps')
    if args.engineering_gate and args.steps!=1536:
        raise ValueError('Engineering endpoint is fixed to 1536 steps')
    return row


def bind(args):
    sys.path.insert(0,str(args.environment_root.resolve()))
    from chapter4_full_spend_env.bootstrap import install
    binding=install(audit_dir=None,context=dict(scope=SCOPE.name,method=args.method,domain=args.domain,seed=args.seed))
    from chapter4_recent_baselines.runners import reference_runner
    from chapter4_recent_baselines.env_adapter import Task
    native=reference_runner._native()
    from allocation4.rollout import rollout as evaluate_rollout
    from allocation4.common import model_hash,cpu_copy
    from allocation23.training import fresh
    sys.path.insert(0,str(SOURCE_DIR))
    from candidates import DevelopmentLR,SETTINGS
    native.update(Task=Task,evaluate_rollout=evaluate_rollout,model_hash=model_hash,cpu_copy=cpu_copy,
                  DevelopmentLR=DevelopmentLR,LR_SETTINGS=SETTINGS,
                  source_lineage=reference_runner._source_lineage)
    return native,binding


def source_identity(native,args):
    identity=native['source_lineage']()
    identity['development/worker.py']=digest(__file__)
    identity['development/candidates.py']=digest(SOURCE_DIR/'candidates.py')
    identity['parent/f1_worker.py']=digest(ORIGINAL_SCOPE/'execution/f1_worker.py')
    identity['registered_jobs_sha256']=digest(args.registry)
    identity['registered_case_id']=args.case_id
    identity['environment_root']=str(args.environment_root.resolve())
    return identity


def cache_immutable_data_digests():
    """Cache only repeated dataset identity hashes, never data or RNG/math.

    Every hit checks the original inode/size/mtime/ctime tuple.  Each file is
    fully SHA-read on its first use and fully reread by verify() before success.
    This avoids 40 identical image scans when independent Task pools are built.
    """
    from chapter4_full_spend_env import native as env_native
    module=env_native.task_module
    original=module.digest
    records={};counts=dict(hits=0,misses=0,final_full_rereads=0)
    def signature(p):
        s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    def cached(path):
        p=Path(path).resolve()
        if p.name not in ('images.npy','split_manifest.json'):return original(p)
        key=str(p);identity=signature(p)
        if key in records:
            old=records[key]
            if identity!=old['signature']:raise AssertionError('Immutable dataset changed during run')
            counts['hits']+=1;return old['sha256']
        value=original(p)
        if signature(p)!=identity:raise AssertionError('Dataset changed while hashing')
        records[key]=dict(signature=identity,sha256=value);counts['misses']+=1
        return value
    def verify():
        for p,row in records.items():
            if signature(Path(p))!=row['signature']or original(p)!=row['sha256']:
                raise AssertionError('Dataset full SHA changed after training/evaluation')
            counts['final_full_rereads']+=1
        return dict(counts,records=records,scope='identity hashing only; no file contents/algorithm/RNG changed')
    module.digest=cached
    return verify


def rng_snapshot(torch,np):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state_all()if torch.cuda.is_initialized()else None)


def restore_rng(state,torch,np):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']is not None:torch.cuda.set_rng_state_all(state['cuda'])


def summarize_guard(task,phase,index,training_step,writer,stream):
    audit=task.env.last_full_spend_audit
    if audit is None or not audit['terminal_full_spend_assertion'] or not audit['nonterminal_rewards_zero']:
        raise AssertionError('Real complete guarded physical episode required')
    if audit['relative_allocation_error']>1e-10:raise AssertionError('Full budget not used')
    encoded=json.dumps(audit,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    stream.update(encoded);stream.update(b'\n')
    row=dict(training_step=training_step,phase=phase,path_index=index,
        steps=audit['steps'],initial_budget=audit['initial_budget'],total_used=audit['total_used'],
        remaining=audit['remaining'],relative_allocation_error=audit['relative_allocation_error'],
        maximum_step_balance_absolute_error=audit['maximum_step_balance_absolute_error'],
        maximum_negative_allocation_violation=audit['maximum_negative_allocation_violation'],
        maximum_per_object_cap_violation=audit['maximum_per_object_cap_violation'],
        maximum_completed_object_allocation=audit['maximum_completed_object_allocation'],
        terminal_reward=audit['terminal_reward'],audit_sha256=hashlib.sha256(encoded).hexdigest())
    writer.add(row)
    return row


RESOURCE_FIELDS=('training_step','phase','path_index','steps','initial_budget','total_used','remaining',
    'relative_allocation_error','maximum_step_balance_absolute_error','maximum_negative_allocation_violation',
    'maximum_per_object_cap_violation','maximum_completed_object_allocation','terminal_reward','audit_sha256')


def evaluate(native,actor,critic,pair,pool,phase,count,training_step,raw,curve,resource,stream,seed):
    import numpy as np
    import torch
    before=rng_snapshot(torch,np)
    before_hash=(native['model_hash'](actor),native['model_hash'](critic))
    mode=actor.training
    try:
        actor.eval();pair.refresh(actor,critic)
        rows=[]
        for first in range(0,count,len(pool)):
            take=min(len(pool),count-first)
            results=native['evaluate_rollout'](pool,pair.actor,pair.critic,None,phase,first,count=take,deterministic=True)
            for j,result in enumerate(results):
                index=first+j
                if not np.all(result['rewards'][:-1]==0.):raise AssertionError('Dense reward leakage')
                ledger=summarize_guard(pool[j],phase,index,training_step,resource,stream)
                row=dict(training_step=training_step,phase=phase,path_index=index,
                    path_seed=result['plan']['task_seed'],native_return=result['native_return'],score_x100=result['score_x100'],
                    total_used=result['total_used'],remaining=result['remaining'],
                    plan_sha256=result['plan_sha256'],steps=pool[j].spec.horizon,
                    relative_allocation_error=ledger['relative_allocation_error'])
                raw.add(row);rows.append(row)
        scores=np.asarray([x['native_return']for x in rows],dtype=float)
        result=dict(training_step=training_step,phase=phase,paths=count,evaluation_environment_steps=count*pool[0].spec.horizon,
                    native_mean=float(scores.mean()),native_std=float(scores.std(ddof=1))if len(scores)>1 else 0.,
                    score_x100_mean=float(scores.mean()*100),minimum=float(scores.min()),maximum=float(scores.max()))
        curve.add(result);raw.flush();curve.flush();resource.flush()
        if before_hash!=(native['model_hash'](actor),native['model_hash'](critic)):
            raise AssertionError('Evaluation changed the policy or critic')
        return result
    finally:
        actor.train(mode);restore_rng(before,torch,np)


def writers(out,training=True):
    result=dict(raw=CSV(out/'evaluation_paths.csv.gz',('training_step','phase','path_index','path_seed','native_return','score_x100',
                'total_used','remaining','plan_sha256','steps','relative_allocation_error')),
                curve=CSV(out/'evaluation_curve.csv.gz',('training_step','phase','paths','evaluation_environment_steps',
                'native_mean','native_std','score_x100_mean','minimum','maximum')),
                resource=CSV(out/'resource_episodes.csv.gz',RESOURCE_FIELDS))
    if training:
        result['episodes']=CSV(out/'training_episodes.csv.gz',('episode','training_step','native_return','score_x100','total_used','remaining','plan_sha256','period_totals'))
        result['updates']=gzip.open(out/'updates.jsonl.gz','xt')
    return result


def close_all(all_writers):
    for w in all_writers.values():w.close()


def artifact_hashes(out):
    result={}
    for p in sorted(out.rglob('*')):
        if not p.is_file()or p.name in ('progress.json','completed.json','failed.json')or p.suffix=='.log'or'runtime'in p.relative_to(out).parts:
            continue
        if p.suffix=='.gz':
            with gzip.open(p,'rb')as f:
                for _ in iter(lambda:f.read(2**20),b''):pass
        result[str(p.relative_to(out))]=digest(p)
    return result


def prepare_output(args):
    out=args.output.resolve()
    if not out.is_relative_to(Path('/hy-tmp')):raise ValueError('All output/cache must be physically HY')
    out.mkdir(parents=True,exist_ok=True)
    if any((out/n).exists()for n in ('protocol.json','completed.json','failed.json')):
        raise FileExistsError('Attempts are immutable; choose a new registered attempt')
    for key in ('TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):
        p=out/'runtime'/key.lower();p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    return out


def train(args):
    validate_registration(args)
    import numpy as np
    import torch
    if args.domain is None:raise ValueError('Training requires --domain')
    h=32 if args.domain=='radio'else 24
    if args.steps<32*h or args.steps%h or(args.steps//h-32)%4:
        raise ValueError('Use complete warm32 plus complete four-episode update blocks')
    if args.eval_every<=0 or args.eval_every%768 or args.dev_cases<1:
        raise ValueError('DEV interval must align both registered complete update grids (multiple768)')
    if args.dev_master!=args.train_master:raise ValueError('Formal shared C3 DEV master must match registered train master')
    out=prepare_output(args);torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    if args.device=='cuda'and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')
    native,binding=bind(args)
    verify_data=cache_immutable_data_digests()
    random.seed(830000+args.seed);torch.manual_seed(840000+args.seed)
    rng=np.random.default_rng(820000+args.seed)
    sampling_device='cuda:0' if args.device=='cuda' else 'cpu'
    generator=torch.Generator(device=sampling_device).manual_seed(810000+args.seed)
    start=time.monotonic();pool=[];dev_pool=[];ww={};stream=hashlib.sha256()
    learning_steps=eval_steps=outer_updates=complete_episodes=0
    collection_seconds=update_seconds=evaluation_seconds=0.
    try:
        pool=[native['Task'](args.domain,3,'A',args.seed,path_master=args.train_master)for _ in range(8)]
        dev_pool=[native['Task'](args.domain,3,'A',args.seed,path_master=args.dev_master)for _ in range(min(32,args.dev_cases))]
        spec=pool[0].spec
        actor=native['make_actor'](spec,'A',3,args.seed).to(args.device)
        critic=native['make_value'](spec,args.seed).to(args.device)
        initial_actor=native['model_hash'](actor);initial_critic=native['model_hash'](critic)
        initialization_heads=(actor.spatial.head,actor.quota_head[-1])
        if not all(torch.count_nonzero(h.weight) and torch.count_nonzero(h.bias) for h in initialization_heads):
            raise AssertionError('Both spatial and quota output heads must use fresh random initialization')
        initial_lr=native['LR_SETTINGS'][args.method]['initial_lr']
        ao=torch.optim.Adam(actor.parameters(),lr=initial_lr);vo=torch.optim.Adam(critic.parameters(),lr=1e-3)
        method=native['DevelopmentLR'](spec,actor.event_dim,args.seed,args.device,method=args.method)
        pair=native['CPURolloutPair'](actor,critic)
        source=source_identity(native,args)
        protocol=dict(schema=SCHEMA,development_only=True,registered_case_id=args.case_id,arguments={k:str(v)if isinstance(v,Path)else v for k,v in vars(args).items()},
            method=args.method,domain=args.domain,seed=args.seed,spec=spec.to_dict(),
            native_ppo=native['PPO'],native_ppo_is_parent_defaults=True,
            actual_optimizer_initial_rates=dict(actor=initial_lr,critic=1e-3),
            implementation=method.metadata(),actor=actor.metadata(),
            source_identity=source,full_spend_binding=binding,physical=pool[0].physical_dict,data=pool[0].data_identity,
            initial_actor_sha256=initial_actor,initial_critic_sha256=initial_critic,
            actual_device=args.device,optimizer_device=args.device,inference_device='cpu',sampling_device=sampling_device,dtype='float64',
            initial_optimizer_entries=dict(actor=len(ao.state),critic=len(vo.state)),
            fresh_initialization=True,initialization_scheme='both_random',initialization_reference='C3 double_random_formal_20261107',
            training_steps=args.steps,development_cases=args.dev_cases,
            evaluation_rule='deterministic native latent mean; same physical decoder',
            evaluation_timing='after complete frozen-policy update block at exact768-step grid, plus fixed final endpoint',
            evaluation_rng='global Python/NumPy/Torch/CUDA snapshots restored; no training generator or shuffle RNG consumed',
            storage='one fixed endpoint actor+critic inference payload, no disk optimizer/replay/periodic model',
            resource_audit='physical guard active; gzip per-complete-episode checks and full-ledger stream SHA; no verbose JSONL duplicate',
            final_holdout_opened=False,terminal_only=True,gamma=1.,
            data_identity_cache='only immutable image/split SHA; stat guarded first+last full content hashing',
            numerical_scope='same native float64 equations; batched inference and CUDA arithmetic not claimed bitwise identical to sequential CPU')
        write(out/'protocol.json',protocol);ww=writers(out)

        def progress(phase):
            write(out/'progress.json',dict(schema=SCHEMA,pid=os.getpid(),phase=phase,method=args.method,domain=args.domain,seed=args.seed,
                training_steps=learning_steps,target_training_steps=args.steps,complete_episodes=complete_episodes,
                outer_updates=outer_updates,evaluation_steps=eval_steps,actual_device=args.device,
                collection_seconds=collection_seconds,update_seconds=update_seconds,evaluation_seconds=evaluation_seconds,
                elapsed_seconds=time.monotonic()-start))

        def dev():
            nonlocal eval_steps,evaluation_seconds
            progress('development');before=time.monotonic()
            g=generator.get_state().clone();r=copy.deepcopy(rng.bit_generator.state);s=copy.deepcopy(method.shuffle_rng.bit_generator.state)
            result=evaluate(native,actor,critic,pair,dev_pool,'dev',args.dev_cases,learning_steps,ww['raw'],ww['curve'],ww['resource'],stream,args.seed)
            if not torch.equal(g,generator.get_state())or r!=rng.bit_generator.state or s!=method.shuffle_rng.bit_generator.state:
                raise AssertionError('DEV consumed training/shuffle RNG')
            evaluation_seconds+=time.monotonic()-before;eval_steps+=result['evaluation_environment_steps']
            progress('training');print(json.dumps(dict(method=args.method,domain=args.domain,seed=args.seed,steps=learning_steps,dev=result['score_x100_mean']),allow_nan=False),flush=True)

        dev()
        while learning_steps<args.steps:
            new_count=8 if complete_episodes<32 else 4
            if learning_steps+new_count*h>args.steps:raise AssertionError('Incomplete last update block')
            pair.refresh(actor,critic);progress('collecting');before=time.monotonic()
            results=native['rollout'](pool[:new_count],pair.actor,pair.critic,generator,'train',complete_episodes,behavior_version=outer_updates)
            pair.assert_rollout_complete();collection_elapsed=time.monotonic()-before;collection_seconds+=collection_elapsed
            batches=[x['batch']for x in results]
            old_actor=native['optimizer_steps'](ao);old_critic=native['optimizer_steps'](vo)
            before=time.monotonic();stats=method.update(actor,critic,ao,vo,batches,rng)
            elapsed=time.monotonic()-before;update_seconds+=elapsed
            learning_steps+=new_count*h;outer_updates+=1
            for j,result in enumerate(results):
                index=complete_episodes+j
                summarize_guard(pool[j],'train',index,(index+1)*h,ww['resource'],stream)
                ww['episodes'].add(dict(episode=index,training_step=(index+1)*h,native_return=result['native_return'],score_x100=result['score_x100'],
                    total_used=result['total_used'],remaining=result['remaining'],plan_sha256=result['plan_sha256'],period_totals=json.dumps(result['period_totals'].tolist(),separators=(',',':'))))
            complete_episodes+=new_count
            stats.update(training_step=learning_steps,episode=complete_episodes,behavior_version=outer_updates-1,
                actual_actor_optimizer_steps=native['optimizer_steps'](ao)-old_actor,actual_critic_optimizer_steps=native['optimizer_steps'](vo)-old_critic,
                collection_seconds=collection_elapsed,update_seconds=elapsed,method=args.method)
            ww['updates'].write(json.dumps(stats,allow_nan=False)+'\n');ww['updates'].flush();ww['episodes'].flush();ww['resource'].flush()
            if learning_steps%args.eval_every==0 or learning_steps==args.steps:dev()
            else:progress('training')
        if source_identity(native,args)!=source:raise AssertionError('Worker or scientific source changed')
        endpoint=dict(schema=SCHEMA,evaluation_only=True,development_only=True,method=args.method,domain=args.domain,seed=args.seed,
            train_env_steps=learning_steps,complete_episodes=complete_episodes,protocol=protocol,
            actor=native['cpu_copy'](actor.state_dict()),critic=native['cpu_copy'](critic.state_dict()),
            actor_sha256=native['model_hash'](actor),critic_sha256=native['model_hash'](critic),optimizer_or_replay_saved=False)
        temporary=out/'checkpoint_final.tmp';torch.save(endpoint,temporary);temporary.replace(out/'checkpoint_final.pt')
        check=torch.load(out/'checkpoint_final.pt',map_location='cpu',weights_only=False)
        copy_actor=native['make_actor'](spec,'A',3,args.seed).cpu();copy_critic=native['make_value'](spec,args.seed).cpu()
        copy_actor.load_state_dict(check['actor'],strict=True);copy_critic.load_state_dict(check['critic'],strict=True)
        if native['model_hash'](copy_actor)!=endpoint['actor_sha256']or native['model_hash'](copy_critic)!=endpoint['critic_sha256']:
            raise AssertionError('Actual fixed endpoint reload changed tensor bytes')
        close_all(ww);ww={}
        data_digest_verification=verify_data()
        receipt=dict(schema=SCHEMA,status='completed',success=True,development_only=True,registered_case_id=args.case_id,method=args.method,domain=args.domain,seed=args.seed,
            training_steps=learning_steps,train_env_steps=learning_steps,episodes=complete_episodes,development_env_steps=eval_steps,
            development_points=1+args.steps//args.eval_every+int(args.steps%args.eval_every!=0),development_cases=args.dev_cases,
            total_physical_environment_steps=learning_steps+eval_steps,outer_updates=outer_updates,
            actor_updates=native['optimizer_steps'](ao),critic_updates=native['optimizer_steps'](vo),actual_device=args.device,
            collection_seconds=collection_seconds,update_seconds=update_seconds,evaluation_seconds=evaluation_seconds,elapsed_seconds=time.monotonic()-start,
            initial_actor_sha256=initial_actor,initial_critic_sha256=initial_critic,actor_changed=endpoint['actor_sha256']!=initial_actor,
            endpoint_reload_pass=True,source_unchanged=True,final_holdout_opened=False,resource_guard_enabled=True,
            resource_audit_stream_sha256=stream.hexdigest(),checkpoint='checkpoint_final.pt',model_count=1,
            model_sha256=digest(out/'checkpoint_final.pt'),optimizer_saved=False,replay_saved=False,periodic_models_saved=False,
            evaluation_deterministic=True,evaluation_training_rng_isolated=True,source_identity=source,
            data_digest_verification=data_digest_verification,
            artifacts_sha256=artifact_hashes(out),engineering_gate=args.engineering_gate)
        write(out/'completed.json',receipt);progress('complete');return receipt
    except BaseException:
        write(out/'failed.json',dict(schema=SCHEMA,method=args.method,domain=args.domain,seed=args.seed,
            training_steps=learning_steps,traceback=traceback.format_exc(),exact_resume_available=False))
        raise
    finally:
        close_all(ww)
        for task in pool+dev_pool:task.env.close()


if __name__=='__main__':
    args=parser().parse_args()
    result=train(args)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
