"""Fresh native-environment training with fixed terminal-only feedback."""
import argparse,copy,json,os,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from .rollout import make_pool,rollout,assess
from allocation4.common import tensor_batch
from allocation23.methods import make_actor,make_value
from allocation23.learning import PPO
from .common import write_json,cpu_copy,model_hash,source_hashes,optimizer_steps
from .methods import METHODS,make_method
from allocation4.methods.lax import gaussian_parameters,gaussian_kl

TRAIN_MASTER=202611019101
TEST_MASTER=202611019199


@torch.no_grad()
def actual_kl(old,new,states):
    om,os=gaussian_parameters(old,states);nm,ns=gaussian_parameters(new,states)
    return float(gaussian_kl(om,os,nm,ns).mean())


def train(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    if args.episodes%8 or args.eval_every%8:raise ValueError('Complete8episode update budget required')
    folder=Path(args.output)
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True)
    start=time.monotonic();pool=devpool=testpool=[]
    try:
        random.seed(830000+args.seed);torch.manual_seed(840000+args.seed)
        torch.cuda.manual_seed_all(840000+args.seed)
        rng=np.random.default_rng(820000+args.seed)
        generator=torch.Generator(device=args.device).manual_seed(810000+args.seed)
        pool=make_pool(args.domain,args.seed,8,args.path_master)
        devpool=make_pool(args.domain,args.seed,min(16,args.dev_paths),args.path_master)
        spec=pool[0].spec
        actor=make_actor(spec,'A',3,args.seed).to(args.device)
        critic=make_value(spec,args.seed).to(args.device)
        ao=torch.optim.Adam(actor.parameters(),lr=PPO['actor_lr'])
        vo=torch.optim.Adam(critic.parameters(),lr=PPO['value_lr'])
        method=make_method(args.method,spec,actor.event_dim,args.seed,args.device)
        hashes=source_hashes();initial=model_hash(actor)
        protocol=dict(schema='chapter4-calibrated-snapshot-v5',study_chapter=4,task_chapter=3,
            domain=args.domain,method=args.method,seed=args.seed,episodes=args.episodes,
            evaluation_interval=args.eval_every,dev_paths=args.dev_paths,holdout_paths=args.holdout_paths,
            path_master=args.path_master,test_master=args.test_master,device=args.device,dtype='float64',
            spec=spec.to_dict(),physical=pool[0].physical_dict,data=pool[0].data_identity,
            native_ppo=PPO,actor=actor.metadata(),implementation=method.metadata(),
            feedback='native terminalR only, allintermediate0; nooracle/modelkernel labels',
            initialization=dict(fresh=True,actor_sha256=initial,critic_sha256=model_hash(critic),
                actor_optimizer_entries=0,critic_optimizer_entries=0),
            source_sha256=hashes,resume_supported=False,
            evaluation='registered pilot learning gate then conditional three-seed snapshot comparison; actual threshold confirmation cost and censoring, no confirmatory p-values',
            software=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__))
        write_json(folder/'protocol.json',protocol)
        collection_seconds=update_seconds=planning_seconds=0.;dev_episodes=holdout_episodes=0
        def progress(episode,phase):
            write_json(folder/'progress.json',dict(episode=episode,episodes=args.episodes,phase=phase,
                seconds=time.monotonic()-start,pid=os.getpid(),method=args.method,domain=args.domain))
        def checkpoint(episode):
            payload=dict(episode=episode,actor=cpu_copy(actor.state_dict()),critic=cpu_copy(critic.state_dict()),
                actor_optimizer=cpu_copy(ao.state_dict()),critic_optimizer=cpu_copy(vo.state_dict()),
                method=cpu_copy(method.state_dict()),generator=generator.get_state().cpu(),
                numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),protocol=protocol)
            temporary=folder/'checkpoint.tmp';torch.save(payload,temporary);temporary.replace(folder/'checkpoint_latest.pt')
            if episode in (0,32,48,96,480,960):torch.save(payload,folder/f'checkpoint_{episode:04d}.pt')
        def development(episode):
            nonlocal dev_episodes
            progress(episode,'development')
            value=assess(devpool,actor,critic,args.dev_paths,args.device,'dev')
            dev_episodes+=args.dev_paths
            value['cost']=dict(collection_seconds=collection_seconds,update_seconds=update_seconds,
                planning_seconds=planning_seconds,
                elapsed_seconds=time.monotonic()-start)
            write_json(folder/f'evaluation_{episode:04d}.json',value)
            print(json.dumps(dict(episode=episode,domain=args.domain,method=args.method,seed=args.seed,
                score=value['stochastic']['score'],seconds=time.monotonic()-start)),flush=True)
        development(0);checkpoint(0)
        with (folder/'training.jsonl').open('w') as ft,(folder/'updates.jsonl').open('w') as fu:
            first=0;behavior_version=0
            while first<args.episodes:
                plan_start=time.monotonic()
                limit=min(8,args.episodes-first,args.eval_every-first%args.eval_every)
                if callable(getattr(method,'collection_size',None)):
                    batch_size=method.collection_size(actor,max_new=limit)
                else:
                    batch_size=4 if args.method=='f1' and first>=32 else 8
                if batch_size not in (4,8) or batch_size>limit:
                    raise AssertionError(('Invalid or checkpoint-crossing collection',first,batch_size,limit))
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize()
                planning=time.monotonic()-plan_start;planning_seconds+=planning
                stop=first+batch_size
                progress(first,'training');then=time.monotonic()
                results=rollout(pool[:batch_size],actor,critic,generator,'train',first,behavior_version=behavior_version)
                collection=time.monotonic()-then;collection_seconds+=collection
                batches=[x['batch'] for x in results]
                before=copy.deepcopy(actor)
                old_steps=(optimizer_steps(ao),optimizer_steps(vo),optimizer_steps(getattr(method,'q_optimizer',None)))
                then=time.monotonic();stats=method.update(actor,critic,ao,vo,batches,rng)
                torch.cuda.synchronize() if args.device.startswith('cuda') else None
                elapsed=time.monotonic()-then;update_seconds+=elapsed
                d=tensor_batch(batches,args.device)
                new_steps=(optimizer_steps(ao),optimizer_steps(vo),optimizer_steps(getattr(method,'q_optimizer',None)))
                committed_actor_steps=new_steps[0]-old_steps[0]
                attempted_actor_steps=int(stats.get('actor_optimizer_steps',stats.get('optimizer_steps',committed_actor_steps)))
                if attempted_actor_steps<committed_actor_steps:raise AssertionError('Optimizer attempt count below committed steps')
                stats.update(actor_optimizer_steps=attempted_actor_steps,
                    actor_committed_optimizer_steps=committed_actor_steps,
                    actor_parameter_updates=committed_actor_steps+int(stats.get('manual_actor_committed_steps',0)),
                    native_critic_optimizer_steps=new_steps[1]-old_steps[1],
                    q_optimizer_steps=new_steps[2]-old_steps[2],
                    episode=stop,new_episodes=batch_size,behavior_version=behavior_version,
                    planning_seconds=planning,collection_seconds=collection,update_seconds=elapsed,
                             actual_joint_kl=actual_kl(before,actor,d['states']))
                for j,x in enumerate(results):
                    ft.write(json.dumps(dict(episode=first+j+1,score_x100=x['score_x100'],
                        period_totals=x['period_totals'].tolist(),total_used=x['total_used'],remaining=x['remaining'],
                        plan_sha256=x['plan_sha256']),allow_nan=False)+'\n')
                fu.write(json.dumps(stats,allow_nan=False)+'\n');ft.flush();fu.flush()
                if stop==32:checkpoint(32)
                if stop%args.eval_every==0 or stop==args.episodes:
                    development(stop);checkpoint(stop)
                if stop==480 and args.episodes>480 and args.holdout_paths:
                    progress(480,'fixed480_independent_evaluation')
                    testpool=make_pool(args.domain,args.seed,min(16,args.holdout_paths),args.test_master,holdout=True)
                    value=assess(testpool,actor,critic,args.holdout_paths,args.device,'final',mean_control=True)
                    write_json(folder/'holdout_0480.json',value)
                    holdout_episodes+=2*args.holdout_paths
                first=stop;behavior_version+=1
        if args.holdout_paths:
            progress(args.episodes,'independent_evaluation')
            if not testpool:testpool=make_pool(args.domain,args.seed,min(16,args.holdout_paths),args.test_master,holdout=True)
            evaluation=assess(testpool,actor,critic,args.holdout_paths,args.device,'final',mean_control=True)
            write_json(folder/'holdout.json',evaluation)
            holdout_episodes+=2*args.holdout_paths
        if source_hashes()!=hashes:raise AssertionError('Frozen computational source changed')
        progress(args.episodes,'complete')
        write_json(folder/'completed.json',dict(status='complete',episodes=args.episodes,source_unchanged=True,
            train_episodes=args.episodes,train_physical_steps=args.episodes*spec.horizon,
            dev_episodes=dev_episodes,dev_physical_steps=dev_episodes*spec.horizon,
            holdout_episodes=holdout_episodes,holdout_physical_steps=holdout_episodes*spec.horizon,
            seconds=time.monotonic()-start,training_collection_seconds=collection_seconds,
            training_update_seconds=update_seconds,training_planning_seconds=planning_seconds,
            actor_changed=model_hash(actor)!=initial,
            fresh_training=True,device=args.device,max_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0,
            max_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if args.device.startswith('cuda') else 0))
    except BaseException:
        write_json(folder/'failure.json',dict(traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
    finally:
        for task in pool+devpool+testpool:task.env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--domain',choices=['radio','v5'],required=True)
    p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--episodes',type=int,default=480);p.add_argument('--eval-every',type=int,default=48)
    p.add_argument('--dev-paths',type=int,default=32);p.add_argument('--holdout-paths',type=int,default=128)
    p.add_argument('--device',default='cuda:0');p.add_argument('--output',required=True)
    p.add_argument('--path-master',type=int,default=TRAIN_MASTER);p.add_argument('--test-master',type=int,default=TEST_MASTER)
    train(p.parse_args())

if __name__=='__main__':main()
