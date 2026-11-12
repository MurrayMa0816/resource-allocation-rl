"""Independent Chapter 3 online baselines; one fixed terminal evaluation model."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import subprocess
import time
import numpy as np

from ..io import TRAIN_MASTER, TEST_MASTER, source_hashes, task_seed, write_json, digest
from .environment import PublicBudgetEnv
from .runtime import METHODS, make, OnlineAgent

SCHEMA='chapter3-full-budget-online-v2'


def select_device(method, requested):
    """Prefer the measured CUDA path when GPU0 has its declared safe margin."""
    if requested == 'cpu':
        return 'cpu', dict(requested=requested, selected='cpu', reason='explicit CPU')
    import torch
    if not torch.cuda.is_available():
        if requested != 'auto':
            raise RuntimeError('CUDA was explicitly requested but is unavailable')
        return 'cpu', dict(requested=requested, selected='cpu', reason='CUDA unavailable')
    free_mib=int(subprocess.check_output(['nvidia-smi','--id=0',
        '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())
    required={'cal':2304,'ccpo':3072,'c_ccpo':2048,'safe_sac':2048,'safedreamer':5120}[method]
    if free_mib < required:
        if requested != 'auto':
            raise RuntimeError('GPU0 lacks the measured model peak and 1.5GiB safety margin')
        return 'cpu',dict(requested=requested,selected='cpu',free_mib=free_mib,
                         required_free_mib=required,reason='GPU0 resource admission')
    return 'cuda',dict(requested=requested,selected='cuda',physical_gpu=0,
                      free_mib=free_mib,required_free_mib=required,
                      reason='CUDA path validated on both full native object counts')


def verify_release():
    root=Path(__file__).resolve().parents[2]
    release=json.loads((root/'CHAPTER3_RELEASE.json').read_text())
    registry=json.loads((root/'CHAPTER3_BASELINES.json').read_text())
    if registry['status']!='implementation_ready_formal_performance_not_run':
        raise ValueError('The active five-method release has not passed final integration')
    for name, expected in release['files_sha256'].items():
        if digest(root/name)!=expected:
            raise ValueError('Release source or entrypoint changed: '+name)
    return release


def source_identity():
    root=Path(__file__).resolve().parents[2]
    names=('CHAPTER3_BASELINES.json','CHAPTER2_BASELINES.json')
    return dict(package=source_hashes(),registries={n:digest(root/n) for n in names if (root/n).exists()})


def runtime_identity():
    versions={}
    for name in ('numpy','torch','gymnasium','scipy','jax','jaxlib','optax','tensorflow-probability'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    return dict(executable=sys.executable,python=sys.version,libraries=versions)


def model_files(directory):
    return {str(p.relative_to(directory)):digest(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def jsonable(value):
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(x) for x in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


class Rows:
    def __init__(self,path,fields):
        self.handle=gzip.open(path,'wt',newline='')
        self.writer=csv.DictWriter(self.handle,fieldnames=fields)
        self.writer.writeheader()
    def add(self,row):self.writer.writerow(row)
    def flush(self):self.handle.flush()
    def close(self):self.handle.close()


def evaluate_agent(policy,*,phase='dev',paths=32,path_master=TRAIN_MASTER,
                   writer=None,training_step=0,example_file=None,condition_file=None):
    if phase not in ('dev','final') or paths<1:raise ValueError('Independent evaluation paths required')
    if phase=='final' and path_master==TRAIN_MASTER:raise ValueError('Holdout master must differ from training')
    env=PublicBudgetEnv(policy.domain,seed=policy.seed,phase=phase,path_master=path_master,
                        holdout=phase=='final',condition_file=condition_file)
    rows=[];examples=[]
    try:
        with policy.evaluation_scope():
            for index in range(paths):
                if policy.method=='safedreamer':
                    # Fixed planner randomness, independent of the training RNG cursor.
                    policy.agent.agent.rng=np.random.default_rng(task_seed(phase,0,index,path_master))
                policy.start_episode();obs,_=env.reset(seed=index)
                total=history=0.;record=[]
                for t in range(env.horizon):
                    action=policy.act(obs,history,True)
                    nxt,reward,done,trunc,info=env.step(action)
                    policy.evaluation_observe(obs,action,reward,info['cost'],nxt,done)
                    if trunc:raise RuntimeError('Evaluation cannot truncate the task')
                    if example_file is not None and index<4:record.append(info['physical_action'].copy())
                    total+=reward;history+=info['cost'];obs=nxt
                replay=env.replay_last_episode()
                if replay['error']!=0.:raise ArithmeticError('Original action replay differs')
                if abs(history-1.)>1e-9:
                    raise ArithmeticError('Full-spend task ended with unallocated resource')
                row=dict(training_step=training_step,phase=phase,path_index=index,
                         path_seed=env.bridge._path_seed,native_return=total,
                         resource_fraction=history,steps=env.horizon,replay_error=replay['error'])
                rows.append(row)
                if writer:writer.add(row)
                if record:examples.append(np.asarray(record))
        if example_file is not None and examples:
            np.savez_compressed(example_file,allocations=np.asarray(examples))
    finally:env.close()
    scores=np.asarray([r['native_return'] for r in rows])
    return dict(training_step=training_step,phase=phase,paths=paths,
                evaluation_environment_steps=paths*env.horizon,
                native_mean=float(scores.mean()),native_std=float(scores.std(ddof=1)) if paths>1 else 0.,
                min_return=float(scores.min()),max_return=float(scores.max()))


def inspect(args):
    root=Path(__file__).resolve().parents[2]
    registry=root/'CHAPTER3_BASELINES.json'
    result=json.loads(registry.read_text()) if registry.exists() else dict(methods=list(METHODS),status='integration_in_progress')
    if getattr(args,'domain',None):
        e=PublicBudgetEnv(args.domain)
        try:
            c=e.contract()
            result['environment']=(c if getattr(args,'details',False) else dict(
                domain=args.domain,objects=e.bridge.n_objects,horizon=e.horizon,
                neural_observation_dim=e.observation_space.shape[0],control_dim=e.action_space.shape[0],
                policy_fields=c['neural_observation_fields'],cost=c['cost'],reward=c['reward']))
        finally:e.close()
    return result


def train(args):
    import torch
    torch.set_num_threads(1)
    if args.final_paths is None:
        args.final_paths = 512 if args.profile=='formal' else 0
    if args.profile=='engineering' and args.final_paths:
        raise ValueError('Engineering checks use development paths, not the final test split')
    if args.eval_every<0:
        raise ValueError('Evaluation interval must be nonnegative (zero disables the development curve)')
    if args.steps<1 or args.development_paths<1 or args.final_paths<0:
        raise ValueError('Positive training/development budgets and nonnegative final paths required')
    horizon=24 if args.domain=='v5' else 32
    if args.steps%horizon:raise ValueError('Use a whole number of native complete episodes')
    if args.method=='ccpo' and args.steps%(4*horizon):
        raise ValueError('CCPO keeps four behavior conditions per complete epoch; steps must be a multiple of 4H')
    if args.final_master==args.path_master:raise ValueError('Training and final masters must differ')
    overrides=json.loads(Path(args.config).read_text()) if args.config else {}
    if args.profile=='formal':
        verify_release()
    output=Path(args.output).resolve()
    if not output.is_relative_to(Path('/hy-tmp')):
        raise ValueError('New experiment outputs must live on /hy-tmp')
    if output.exists():raise FileExistsError('Use a new independent run directory')
    sources=source_identity()
    args.device,device_choice=select_device(args.method,args.device)
    policy=make(args.method,args.domain,seed=args.seed,device=args.device,
                profile=args.profile,overrides=overrides)
    if args.profile=='formal' and args.method=='ccpo' and args.steps<=4*horizon*policy.agent.config.pretraining_epochs:
        raise ValueError('A formal CCPO run must include both behavior pretraining and versatile fine-tuning')
    env=PublicBudgetEnv(args.domain,seed=args.seed,phase='train',path_master=args.path_master,
                        train_path_base=args.path_master,condition_file=args.condition_file)
    output.mkdir(parents=True)
    protocol=dict(schema=SCHEMA,arguments=vars(args),device_choice=device_choice,environment=env.contract(),
        initial_algorithm=policy.metadata,source_identity=sources,runtime=runtime_identity(),
        storage='all step rewards and evaluation data; one fixed terminal evaluation model; no disk replay/Adam',
        evaluation='exact learning-step triggers, isolated environment/RNG; native terminal objective',
        selection='fixed endpoint; no development or test model selection',
        result_status='engineering only' if args.profile=='engineering' else 'formal online run',
        dataset='no pretrained model, demonstration data, offline dataset, or physics/utility oracle')
    write_json(output/'protocol.json',jsonable(protocol))
    trace=Rows(output/'training_steps.csv.gz', ['training_step','episode','episode_step','condition',
        'native_reward','learning_reward','resource_cost','cumulative_cost','physical_amount'])
    episodes=Rows(output/'training_episodes.csv.gz',['episode','training_step','condition','native_return','resource_fraction'])
    evalraw=Rows(output/'evaluation_paths.csv.gz',['training_step','phase','path_index','path_seed',
        'native_return','resource_fraction','steps','replay_error'])
    evalsummary=Rows(output/'evaluation_curve.csv.gz',['training_step','phase','paths',
        'evaluation_environment_steps','native_mean','native_std','min_return','max_return'])
    total_eval_steps=0;started=time.perf_counter();success=False
    try:
        if args.eval_every:
            row=evaluate_agent(policy,paths=args.development_paths,path_master=args.path_master,
                writer=evalraw,training_step=0,condition_file=args.condition_file)
            evalsummary.add(row);total_eval_steps+=row['evaluation_environment_steps']
        episode=0
        while policy.environment_steps<args.steps:
            policy.start_episode();obs,_=env.reset();history=return_=0.
            condition=policy.condition
            for t in range(horizon):
                action=policy.act(obs,history,False)
                nxt,reward,done,trunc,info=env.step(action)
                update=policy.observe(obs,action,reward,info['cost'],nxt,done,
                    history_cost=history,episode_step=t)
                history+=info['cost'];return_+=reward;obs=nxt
                learning_reward=update.get('learning_reward',update.get('algorithm_training_reward',reward))
                trace.add(dict(training_step=policy.environment_steps,episode=episode,episode_step=t,
                    condition=condition,native_reward=reward,learning_reward=learning_reward,
                    resource_cost=info['cost'],cumulative_cost=history,physical_amount=info['physical_amount']))
                if args.eval_every and policy.environment_steps%args.eval_every==0 and policy.environment_steps<args.steps:
                    row=evaluate_agent(policy,paths=args.development_paths,path_master=args.path_master,
                        writer=evalraw,training_step=policy.environment_steps,condition_file=args.condition_file)
                    evalsummary.add(row);total_eval_steps+=row['evaluation_environment_steps'];evalsummary.flush()
                if trunc or done!=(t==horizon-1):raise RuntimeError('Original horizon changed')
            episodes.add(dict(episode=episode,training_step=policy.environment_steps,condition=condition,
                              native_return=return_,resource_fraction=history))
            if abs(history-1.)>1e-9:
                raise ArithmeticError('Training episode did not allocate its full initial resource')
            trace.flush();episodes.flush();episode+=1
        policy.finish()
        if args.eval_every:
            row=evaluate_agent(policy,paths=args.development_paths,path_master=args.path_master,
                writer=evalraw,training_step=policy.environment_steps,condition_file=args.condition_file)
            evalsummary.add(row);total_eval_steps+=row['evaluation_environment_steps']
        if source_identity()!=sources:raise ValueError('Source changed during training; result cannot be accepted')
        payload=policy.save(output/'model')
        final=None
        if args.final_paths:
            final=evaluate_agent(policy,phase='final',paths=args.final_paths,path_master=args.final_master,
                writer=evalraw,training_step=policy.environment_steps,example_file=output/'allocation_examples.npz',
                condition_file=args.condition_file)
            evalsummary.add(final);total_eval_steps+=final['evaluation_environment_steps']
        if source_identity()!=sources:raise ValueError('Source changed during final evaluation')
        result=dict(schema=SCHEMA,success=True,method=args.method,domain=args.domain,
            profile=args.profile,training_steps=policy.environment_steps,evaluation_steps=total_eval_steps,
            elapsed_seconds=time.perf_counter()-started,final_evaluation=final,
            model=str(payload.relative_to(output)),model_sha256=digest(payload),
            model_files_sha256=model_files(output/'model'),
            exact_training_resume=False,optimizer_or_replay_saved=False,
            full_spend_required=True,resource_tolerance_fraction=1e-9,
            accounting=policy.accounting,source_identity=sources)
        write_json(output/'completed.json',jsonable(result));success=True
        return result
    finally:
        for writer in (trace,episodes,evalraw,evalsummary):writer.close()
        env.close()
        if not success:
            write_json(output/'failed.json',dict(schema=SCHEMA,training_steps=policy.environment_steps,
                elapsed_seconds=time.perf_counter()-started,exact_resume_available=False,
                action='retain logs; any new attempt uses a new scope and the same registered seed'))


def evaluate(args):
    root=Path(args.run).resolve();protocol=json.loads((root/'protocol.json').read_text())
    done=json.loads((root/'completed.json').read_text())
    if done['schema']!=SCHEMA or done['source_identity']!=source_identity():
        raise ValueError('Model and current scientific source release differ')
    payload=root/done['model']
    if digest(payload)!=done['model_sha256']:raise ValueError('Terminal model changed')
    if model_files(root/'model')!=done['model_files_sha256']:
        raise ValueError('Terminal model or its deployment metadata changed')
    if args.phase == 'final' and args.path_master == protocol['arguments']['path_master']:
        raise ValueError('Final evaluation master must differ from this run actual training master')
    policy=OnlineAgent.load(root/'model')
    return evaluate_agent(policy,phase=args.phase,paths=args.paths,path_master=args.path_master,
                          training_step=done['training_steps'],
                          condition_file=protocol['arguments'].get('condition_file'))


def parser():
    p=argparse.ArgumentParser(description='Five independent Chapter 3 online-from-scratch comparators')
    sub=p.add_subparsers(dest='command',required=True)
    v=sub.add_parser('inspect');v.add_argument('--domain',choices=('v5','radio'));v.add_argument('--details',action='store_true')
    t=sub.add_parser('train')
    t.add_argument('--method',choices=METHODS,required=True);t.add_argument('--domain',choices=('v5','radio'),required=True)
    t.add_argument('--steps',type=int,required=True);t.add_argument('--seed',type=int,default=0)
    t.add_argument('--device',choices=('auto','cpu','cuda','cuda:0'),default='auto')
    t.add_argument('--profile',choices=('formal','engineering'),default='formal')
    t.add_argument('--eval-every',type=int,default=512);t.add_argument('--development-paths',type=int,default=32)
    t.add_argument('--final-paths',type=int);t.add_argument('--path-master',type=int,default=TRAIN_MASTER)
    t.add_argument('--final-master',type=int,default=TEST_MASTER);t.add_argument('--condition-file');t.add_argument('--config')
    t.add_argument('--output',required=True)
    e=sub.add_parser('evaluate');e.add_argument('--run',required=True);e.add_argument('--paths',type=int,default=512)
    e.add_argument('--path-master',type=int,default=TEST_MASTER)
    e.add_argument('--phase',choices=('dev','final'),default='final')
    return p


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    # Compatibility chapter selector is accepted only when it denotes chapter 3.
    if '--chapter' in args:
        i=args.index('--chapter')
        if i+1>=len(args) or args[i+1]!='3':raise ValueError('This entry belongs to Chapter 3 only')
        args=args[:i]+args[i+2:]
    a=parser().parse_args(args)
    result={'inspect':inspect,'train':train,'evaluate':evaluate}[a.command](a)
    print(json.dumps(jsonable(result),ensure_ascii=False,allow_nan=False))
    return result
