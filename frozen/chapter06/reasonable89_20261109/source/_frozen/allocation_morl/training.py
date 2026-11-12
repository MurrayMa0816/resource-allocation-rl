"""Fresh vector-terminal learners with explicit preference and interaction costs."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch
from allocation4.common import STATE_KEYS
from allocation4.methods.lax import gaussian_parameters,gaussian_kl
from allocation23.learning import PPO
from .common import (WEIGHTS,METHODS,write_json,cpu_copy,model_hash,file_hash,source_hashes,
    optimizer_steps,state_hash,atomic_torch_save,save_trace)
from .rollout import make_pool,rollout,assess,result_row

TRAIN_MASTER=202611029101
TEST_MASTER=202611029299


def jsonable(value):
    if torch.is_tensor(value):return value.detach().cpu().item() if value.numel()==1 else value.detach().cpu().tolist()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value


@torch.no_grad()
def actual_kl(before,actor,batches):
    states={k:np.concatenate([b['states'][k] for b in batches]) for k in STATE_KEYS}
    om,osigma=gaussian_parameters(before,states);nm,nsigma=gaussian_parameters(actor,states)
    return float(gaussian_kl(om,osigma,nm,nsigma).mean())


def train(args):
    from .learner import Learner
    from .models import make_models
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    if args.device!='cpu':raise ValueError('Registered CPU-only profile')
    if args.episodes<=0 or args.episodes%8 or args.eval_every<=0 or args.eval_every%8:
        raise ValueError('Positive complete eight-episode budgets and evaluation intervals required')
    if args.dev_paths<=0 or args.holdout_paths<0 or args.eval_batch<=0:raise ValueError('Invalid evaluation sizes')
    if not np.isfinite(args.tau) or not 0<=args.tau<=1:raise ValueError('Threshold in [0,1] required')
    if not np.isfinite(args.weight) or not 0<=args.weight<=1:raise ValueError('Weight in [0,1] required')
    folder=Path(args.output)
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True)
    start=time.monotonic();pool=[];devpool=[];testpool=[]
    counts={name:dict(episodes=0,physical_steps=0) for name in ('train','development','holdout')}
    def counter(name):
        def increment(done):
            counts[name]['physical_steps']+=1;counts[name]['episodes']+=int(done)
        return increment
    try:
        random.seed(830000+args.seed);torch.manual_seed(840000+args.seed)
        rng=np.random.default_rng(820000+args.seed)
        generator=torch.Generator(device='cpu').manual_seed(810000+args.seed)
        pool=make_pool(args.domain,args.seed,8,args.path_master)
        devpool=make_pool(args.domain,args.seed,min(args.eval_batch,args.dev_paths),args.path_master)
        spec=pool[0].spec
        actor,critic,neural_spec=make_models(spec,args.seed,args.method)
        actor=actor.to('cpu');critic=critic.to('cpu')
        ao=torch.optim.Adam(actor.parameters(),lr=PPO['actor_lr']);vo=torch.optim.Adam(critic.parameters(),lr=PPO['value_lr'])
        learner=Learner(neural_spec,actor.event_dim,args.seed,'cpu',method=args.method,weight=args.weight,tau=args.tau)
        preferences=[args.weight] if args.method=='fixed' else list(WEIGHTS)
        hashes=source_hashes();initial_actor=model_hash(actor);initial_critic=model_hash(critic)
        protocol=dict(schema='chapter5-multiobjective-terminal-v1',study_chapter=5,task_chapter=3,
            domain=args.domain,method=args.method,seed=args.seed,weight=args.weight,
            evaluated_preferences=preferences,training_preferences=preferences,episodes=args.episodes,
            evaluation_interval=args.eval_every,dev_paths=args.dev_paths,holdout_paths=args.holdout_paths,
            tau=args.tau,path_master=args.path_master,test_master=args.test_master,
            dev_rng_seed=args.dev_rng_seed,test_rng_seed=args.test_rng_seed,
            device='cpu',dtype='float64',threads=1,
            objectives=dict(vector_order=['C','J'],coverage='K/N; terminal quality >= tau',
                overall='original mean terminal quality',policy='maximize lambda*C+(1-lambda)*J directly from first training episode',
                F1_return_floor=False,F1_proposal_supplement=False,strict_lexicographic_priority=False),
            preference=dict(kind='fixed per complete episode',neural_global_final_coordinate=True,
                environmental_state_unchanged=True,history_original_weight_retained=True,cross_preference_relabel=False,
                collection='balanced four preferences per fresh4; rotated by behavior version' if args.method=='conditioned' else 'single fixed preference'),
            spec=spec.to_dict(),neural_spec=neural_spec.to_dict(),physical=pool[0].physical_dict,data=pool[0].data_identity,
            native_ppo=PPO,actor=actor.metadata(),implementation=learner.metadata(),
            feedback='unchanged native terminal reward; vector terminal label C,J computed only after true done',
            architecture='identical preference-augmented A architecture in fixed and conditioned arms; adapted F1 workflow, not bitwise original F1',
            initialization=dict(fresh=True,actor_sha256=initial_actor,critic_sha256=initial_critic,old_models_loaded=False),
            checkpoint=dict(policy='one fixed endpoint model',episode=args.episodes,model_selection=False,
                optimizer_backup=False,replay_backup=False,resume_supported=False),
            source_sha256=hashes,engineering_capture_batches=args.engineering_capture_batches,
            accounting='all physical train/development/holdout simulator steps charged, all preferences included',
            holdout_selection=False,software=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__))
        write_json(folder/'protocol.json',protocol)
        collection_seconds=update_seconds=evaluation_seconds=0.
        def progress(episode,phase):
            write_json(folder/'progress.json',dict(episode=episode,episodes=args.episodes,phase=phase,
                seconds=time.monotonic()-start,pid=os.getpid(),method=args.method,domain=args.domain,
                seed=args.seed,weight=args.weight,counts=counts))
        def parameter_audit():
            return dict(actor=model_hash(actor),critic=model_hash(critic),actor_optimizer=state_hash(ao.state_dict()),
                critic_optimizer=state_hash(vo.state_dict()),learner=state_hash(learner.state_dict()))
        def development(episode):
            nonlocal evaluation_seconds
            progress(episode,'development');before=parameter_audit();then=time.monotonic()
            evaluations=assess(devpool,actor,critic,args.dev_paths,'cpu','dev',tau=args.tau,preferences=preferences,
                eval_seed=args.dev_rng_seed,trace_prefix=folder/'traces'/f'development_{episode:04d}',step_callback=counter('development'))
            evaluation_seconds+=time.monotonic()-then;after=parameter_audit()
            if before!=after:raise AssertionError('Development changed learner/optimizer')
            value=dict(preferences=evaluations,checkpoint_episode=episode,
                parameter_audit=dict(unchanged=True,before=before,after=after),counts=copy.deepcopy(counts),
                model_selection=False,cost=dict(collection_seconds=collection_seconds,update_seconds=update_seconds,
                    evaluation_seconds=evaluation_seconds,elapsed_seconds=time.monotonic()-start))
            write_json(folder/f'evaluation_{episode:04d}.json',value)
            print(json.dumps(dict(episode=episode,domain=args.domain,method=args.method,seed=args.seed,
                preferences={k:dict(K=v['stochastic']['mean_K'],J=v['stochastic']['score']) for k,v in evaluations.items()},
                seconds=time.monotonic()-start)),flush=True)
        development(0)
        with (folder/'training.jsonl').open('w') as ft,(folder/'updates.jsonl').open('w') as fu:
            first=0;behavior_version=0
            while first<args.episodes:
                limit=min(8,args.episodes-first,args.eval_every-first%args.eval_every)
                batch_size=learner.collection_size(actor,max_new=limit)
                if batch_size not in (4,8) or batch_size>limit:raise AssertionError(('Invalid collection size',first,batch_size,limit))
                weights=(np.array([WEIGHTS[(behavior_version+j)%len(WEIGHTS)] for j in range(batch_size)])
                    if args.method=='conditioned' else np.full(batch_size,args.weight))
                progress(first,'training');then=time.monotonic()
                results=rollout(pool[:batch_size],actor,critic,generator,'train',first,tau=args.tau,
                    preferences=weights,behavior_version=behavior_version,step_callback=counter('train'))
                collection=time.monotonic()-then;collection_seconds+=collection;batches=[x['batch'] for x in results]
                if args.engineering_capture_batches and first in (0,32):
                    atomic_torch_save(folder/f'engineering_batches_{first:04d}.pt',cpu_copy(batches))
                    atomic_torch_save(folder/f'engineering_before_{first:04d}.pt',dict(actor=cpu_copy(actor.state_dict()),
                        critic=cpu_copy(critic.state_dict()),learner=cpu_copy(learner.state_dict()),
                        actor_optimizer=cpu_copy(ao.state_dict()),critic_optimizer=cpu_copy(vo.state_dict()),
                        numpy_rng=copy.deepcopy(rng.bit_generator.state),protocol=protocol))
                before=copy.deepcopy(actor);old_steps=(optimizer_steps(ao),optimizer_steps(vo))
                then=time.monotonic();stats=learner.update(actor,critic,ao,vo,batches,rng)
                elapsed=time.monotonic()-then;update_seconds+=elapsed
                stats.update(actor_committed_optimizer_steps=optimizer_steps(ao)-old_steps[0],
                    critic_committed_optimizer_steps=optimizer_steps(vo)-old_steps[1],episode=first+batch_size,
                    new_episodes=batch_size,behavior_version=behavior_version,collection_seconds=collection,
                    update_seconds=elapsed,measured_outer_actor_kl=actual_kl(before,actor,batches),counts=copy.deepcopy(counts))
                trace=save_trace(folder/'traces'/f'training_{first:04d}.npz',results)
                for j,result in enumerate(results):
                    row=result_row(result,first+j,trace=trace,trace_index=j);row['episode']=first+j+1
                    ft.write(json.dumps(jsonable(row),allow_nan=False)+'\n')
                fu.write(json.dumps(jsonable(stats),allow_nan=False)+'\n');ft.flush();fu.flush()
                first+=batch_size;behavior_version+=1
                if first%args.eval_every==0 or first==args.episodes:development(first)
        if counts['train']['episodes']!=args.episodes or counts['train']['physical_steps']!=args.episodes*spec.horizon:
            raise AssertionError('Training interaction accounting mismatch')
        endpoint_actor=model_hash(actor);endpoint_critic=model_hash(critic);endpoint=folder/'checkpoint_endpoint.pt'
        atomic_torch_save(endpoint,dict(format='compact-morl-fixed-endpoint-v1',episode=args.episodes,
            actor=cpu_copy(actor.state_dict()),critic=cpu_copy(critic.state_dict()),actor_sha256=endpoint_actor,
            critic_sha256=endpoint_critic,protocol=protocol,optimizer_backup=False,replay_backup=False,resume_supported=False))
        if args.holdout_paths:
            progress(args.episodes,'independent_evaluation')
            testpool=make_pool(args.domain,args.seed,min(args.eval_batch,args.holdout_paths),args.test_master,holdout=True)
            before=parameter_audit();then=time.monotonic()
            evaluations=assess(testpool,actor,critic,args.holdout_paths,'cpu','final',tau=args.tau,preferences=preferences,
                eval_seed=args.test_rng_seed,trace_prefix=folder/'traces'/'holdout_endpoint',step_callback=counter('holdout'))
            evaluation_seconds+=time.monotonic()-then;after=parameter_audit()
            if before!=after:raise AssertionError('Independent test changed learner/optimizer')
            write_json(folder/'holdout.json',dict(endpoint=dict(checkpoint_episode=args.episodes,
                checkpoint_actor_sha256=endpoint_actor,parameter_audit=dict(unchanged=True,before=before,after=after),
                preferences=evaluations)))
        if source_hashes()!=hashes:raise AssertionError('Runtime source changed during run')
        if model_hash(actor)!=endpoint_actor or model_hash(critic)!=endpoint_critic:raise AssertionError('Fixed endpoint changed')
        progress(args.episodes,'complete')
        write_json(folder/'completed.json',dict(status='complete',episodes=args.episodes,source_unchanged=True,
            counts=counts,total_physical_steps=sum(x['physical_steps'] for x in counts.values()),
            total_episodes=sum(x['episodes'] for x in counts.values()),train_episodes=counts['train']['episodes'],
            train_physical_steps=counts['train']['physical_steps'],dev_episodes=counts['development']['episodes'],
            dev_physical_steps=counts['development']['physical_steps'],holdout_episodes=counts['holdout']['episodes'],
            holdout_physical_steps=counts['holdout']['physical_steps'],seconds=time.monotonic()-start,
            training_collection_seconds=collection_seconds,training_update_seconds=update_seconds,evaluation_seconds=evaluation_seconds,
            actor_changed=endpoint_actor!=initial_actor,critic_changed=endpoint_critic!=initial_critic,
            endpoint_actor_sha256=endpoint_actor,endpoint_critic_sha256=endpoint_critic,endpoint_sha256=file_hash(endpoint),
            endpoint_retained=True,model_selection=False,holdout_selection=False,fresh_training=True,device='cpu',
            evaluated_preferences=preferences,saved_model_count=1,resume_supported=False,optimizer_backup=False,replay_backup=False))
    except BaseException:
        write_json(folder/'failure.json',dict(traceback=traceback.format_exc(),seconds=time.monotonic()-start,counts=counts));raise
    finally:
        for task in pool+devpool+testpool:task.env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain',choices=['radio','v5'],required=True);p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--weight',type=float,default=.5);p.add_argument('--seed',type=int,required=True);p.add_argument('--tau',type=float,required=True)
    p.add_argument('--episodes',type=int,default=480);p.add_argument('--eval-every',type=int,default=96)
    p.add_argument('--dev-paths',type=int,default=64);p.add_argument('--holdout-paths',type=int,default=256)
    p.add_argument('--eval-batch',type=int,default=16);p.add_argument('--device',choices=['cpu'],default='cpu')
    p.add_argument('--output',required=True);p.add_argument('--path-master',type=int,default=TRAIN_MASTER)
    p.add_argument('--test-master',type=int,default=TEST_MASTER);p.add_argument('--dev-rng-seed',type=int,default=996101)
    p.add_argument('--test-rng-seed',type=int,default=996199);p.add_argument('--engineering-capture-batches',action='store_true')
    train(p.parse_args())

if __name__=='__main__':main()
