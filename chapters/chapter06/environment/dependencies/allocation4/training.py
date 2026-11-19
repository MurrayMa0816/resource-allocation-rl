"""Fresh GPU training on the pinned simplified chapter-3 task."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch
from allocation23.methods import make_actor, make_value
from allocation23.learning import PPO
from .common import write_json, source_hashes, model_hash, cpu_copy
from .registry import METHODS, make_method
from .rollout import make_pool, rollout, evaluate

TRAIN_MASTER = 202611014101
TEST_MASTER = 202611014199

def configure_runtime():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)

def train(args):
    configure_runtime()
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        raise RuntimeError('GPU requested but unavailable')
    if args.episodes % 8 or args.eval_every % 8:
        raise ValueError('Episode budget and evaluation interval must be multiples of eight')
    folder = Path(args.output).resolve()
    if folder.exists():
        raise FileExistsError('Every run must use a fresh output directory')
    folder.mkdir(parents=True)
    started = time.monotonic()
    try:
        random.seed(830000+args.seed)
        torch.manual_seed(840000+args.seed)
        torch.cuda.manual_seed_all(840000+args.seed)
        rng = np.random.default_rng(820000+args.seed)
        generator = torch.Generator(device=args.device).manual_seed(810000+args.seed)
        pool = make_pool(args.domain, args.seed, 8, args.path_master)
        devpool = make_pool(args.domain, args.seed, min(16,args.dev_paths), args.path_master)
        spec = pool[0].spec
        actor = make_actor(spec, 'A', 3, args.seed).to(args.device)
        critic = make_value(spec, args.seed).to(args.device)
        pg_backend = args.method in ('mc_pg','cv_mse','lax','cocoa','cca_state','cca')
        actor_lr = 3e-3 if pg_backend else PPO['actor_lr']
        ao = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        vo = torch.optim.Adam(critic.parameters(), lr=PPO['value_lr'])
        method = make_method(args.method, spec, actor.event_dim, args.seed, args.device)
        initial = model_hash(actor)
        hashes = source_hashes()
        protocol = dict(schema='chapter4-credit-framework-v1', study_chapter=4, task_chapter=3,
            method=args.method, domain=args.domain, seed=args.seed, episodes=args.episodes,
            evaluation_interval=args.eval_every, dev_paths=args.dev_paths, holdout_paths=args.holdout_paths,
            path_master=args.path_master, test_master=args.test_master,
            feedback='native terminal only; intermediate rewards zero',
            source_sha256=hashes, baseline_package=str(Path(__import__('allocation23').__file__).resolve()),
            spec=spec.to_dict(), physical=pool[0].physical_dict, data=pool[0].data_identity,
            actor=actor.metadata(), original_ppo=PPO, device=args.device,
            actor_optimizer=dict(name='Adam',lr=actor_lr,
                reason=('single full-batch PG step with exact KL cap .01' if pg_backend
                        else 'released PPO multi-minibatch setting')),
            software=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__),
            initialization=dict(fresh=True, actor_sha256=initial, critic_sha256=model_hash(critic),
                                actor_optimizer_entries=0, critic_optimizer_entries=0),
            resume_supported=False, checkpoint_purpose='full state snapshots; no runner resume interface in this screen',
            implementation_notes=(method.metadata() if hasattr(method,'metadata') else {}),
            evaluation='fixed endpoints; stochastic primary; mean-latent secondary; same task paths',
            batching='eight independent complete episodes; frozen policy within collection')
        write_json(folder/'protocol.json',protocol)

        def progress(episode, phase):
            write_json(folder/'progress.json',dict(episode=episode, episodes=args.episodes, phase=phase,
                        seconds=time.monotonic()-started,pid=os.getpid(),method=args.method,domain=args.domain))

        dev_episode_count = 0
        collection_total = update_total = 0.

        def assessment(episode):
            nonlocal dev_episode_count
            progress(episode,'development')
            ev = evaluate(devpool,actor,critic,args.dev_paths,'dev',args.device)
            dev_episode_count += 2*args.dev_paths
            ev['cost'] = dict(training_collection_seconds=collection_total,
                              training_update_seconds=update_total,
                              training_seconds=collection_total+update_total,
                              elapsed_seconds=time.monotonic()-started)
            write_json(folder/f'evaluation_{episode:04d}.json',ev)
            print(json.dumps(dict(episode=episode, domain=args.domain, method=args.method, seed=args.seed,
                                  score=ev['stochastic']['score'],seconds=time.monotonic()-started)),flush=True)

        def checkpoint(episode):
            payload = dict(episode=episode, actor=cpu_copy(actor.state_dict()),critic=cpu_copy(critic.state_dict()),
                actor_optimizer=cpu_copy(ao.state_dict()),critic_optimizer=cpu_copy(vo.state_dict()),
                method=cpu_copy(method.state_dict()) if hasattr(method,'state_dict') else {},
                generator=generator.get_state().cpu(),numpy_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                protocol=protocol)
            tmp=folder/'checkpoint.tmp'
            torch.save(payload,tmp)
            tmp.replace(folder/'checkpoint_latest.pt')

        assessment(0)
        checkpoint(0)
        with (folder/'training.jsonl').open('w') as trainlog, (folder/'updates.jsonl').open('w') as updates:
            for first in range(0,args.episodes,8):
                progress(first,'training')
                before = time.monotonic()
                results = rollout(pool,actor,critic,generator,'train',first,training=True)
                collect_s = time.monotonic()-before
                collection_total += collect_s
                before = time.monotonic()
                stats = method.update(actor,critic,ao,vo,[r['batch'] for r in results],rng)
                torch.cuda.synchronize() if args.device.startswith('cuda') else None
                update_s = time.monotonic()-before
                update_total += update_s
                for j,r in enumerate(results):
                    row=dict(episode=first+j+1, score_x100=r['score_x100'],total_used=r['total_used'],
                             remaining=r['remaining'],period_totals=r['period_totals'].tolist(),
                             plan_sha256=r['plan_sha256'])
                    trainlog.write(json.dumps(row,allow_nan=False)+'\n')
                row=dict(stats)
                row.update(episode=first+8,collection_seconds=collect_s,update_seconds=update_s)
                updates.write(json.dumps(row,allow_nan=False)+'\n')
                trainlog.flush();updates.flush()
                if (first+8)%args.eval_every==0 or first+8==args.episodes:
                    assessment(first+8)
                    checkpoint(first+8)
        if args.holdout_paths:
            progress(args.episodes,'independent_evaluation')
            testpool=make_pool(args.domain,args.seed,min(16,args.holdout_paths),args.test_master,holdout=True)
            ev=evaluate(testpool,actor,critic,args.holdout_paths,'final',args.device)
            write_json(folder/'holdout.json',ev)
            for task in testpool:task.env.close()
        if source_hashes()!=hashes:
            raise AssertionError('Runtime source changed during the run')
        for task in pool+devpool:task.env.close()
        progress(args.episodes,'complete')
        write_json(folder/'completed.json',dict(status='complete',episodes=args.episodes,
            train_episodes=args.episodes,train_physical_steps=args.episodes*spec.horizon,
            dev_episodes=dev_episode_count,
            holdout_episodes=2*args.holdout_paths,seconds=time.monotonic()-started,
            training_collection_seconds=collection_total,training_update_seconds=update_total,
            actor_changed=model_hash(actor)!=initial,actor_sha256=model_hash(actor),
            source_unchanged=True, fresh_training=True))
    except BaseException:
        write_json(folder/'failure.json',dict(traceback=traceback.format_exc(),seconds=time.monotonic()-started))
        raise

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--domain',choices=('radio','v5'),required=True)
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--episodes',type=int,default=480)
    p.add_argument('--eval-every',type=int,default=96)
    p.add_argument('--dev-paths',type=int,default=16)
    p.add_argument('--holdout-paths',type=int,default=64)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--path-master',type=int,default=TRAIN_MASTER)
    p.add_argument('--test-master',type=int,default=TEST_MASTER)
    p.add_argument('--output',required=True)
    train(p.parse_args())
