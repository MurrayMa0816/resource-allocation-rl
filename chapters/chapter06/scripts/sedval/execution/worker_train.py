#!/usr/bin/env python3
"""Fresh chapter-six89 Full training, with compact data and one fixed endpoint."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT/'source'):
    if str(directory) not in sys.path: sys.path.insert(0, str(directory))
from execution.common import (require_hy, jsonable, write_json, file_hash, model_hash,
    state_hash, source_hashes, cpu_copy, save_npz, save_endpoint, environment_config)



def validate_formal_profile(config, env_config):
    """Reject every nonregistered formal profile before constructing an environment."""
    if not config.get('formal', False):
        return
    operator = 'fixed64_antithetic_mean_feasible_actions'
    keep_model = bool(config.get('save_model', config.get('save_endpoint', True)))
    if (int(config.get('episodes', 3200)) != 3200 or
            int(config.get('eval_every', 128)) != 128 or
            int(config.get('dev_paths', 32)) != 32 or not keep_model):
        raise ValueError('Formal89 requires fixed3200/128/32 and its unique endpoint')
    if (float(config['tau']) != .90 or float(config['weight']) != .2 or
            env_config.get('adapter') != 'depth_area_physical' or
            float(env_config.get('spatial_sigma_scale', 1.)) != 1. or
            config.get('development_execution') != operator or
            config.get('phase') != 'formal_fixed3200'):
        raise ValueError('Formal89 requires adopted C90/JA lambda.2, physical8, sigma1 and M64 execution')
    budget = float(env_config.get('total_budget_m3', 4_000_000.))
    if min(abs(budget-b) for b in (3_500_000., 4_000_000., 4_500_000.)) > 1e-6:
        raise ValueError('Formal89 budget is outside the three registered settings')
    seed = int(config['seed'])
    if seed not in (6400, 6401, 6402):
        raise ValueError('Formal89 requires the three fresh registered seeds')
    first = 2611101000 + 100000*(seed-6400)
    if (config.get('train_weather_seeds') != list(range(first, first+3200)) or
            config.get('dev_weather_seeds') != list(range(2611095000, 2611095032))):
        raise ValueError('Formal89 training or development forcing namespace differs')
    if not config.get('source_files_sha256'):
        raise ValueError('Formal89 requires its frozen scientific source map')
    design_path = require_hy(config['experiment_design_path'])
    if file_hash(design_path) != config.get('experiment_design_sha256'):
        raise ValueError('Formal89 final design changed')
    declared = json.loads(design_path.read_text())
    profile = declared.get('formal_profile', {})
    expected = dict(adapter='depth_area_physical', training_weight=.2,
        spatial_sigma_scale=1., execution_operator=operator,
        coverage_threshold=.9, episodes=3200, eval_every=128, dev_paths=32)
    if profile != expected:
        raise ValueError('Formal89 is not bound to the adopted final application design')


def train(config, *, output_override=None, device_override=None):
    folder = require_hy(output_override or config['output_dir'])
    if folder.exists():
        raise FileExistsError('A fresh independent job directory is required: ' + str(folder))
    folder.mkdir(parents=True)
    start = time.monotonic()
    pool = []; devpool = []
    counts = {name:dict(episodes=0,physical_steps=0) for name in ('train','development')}
    episode = 0
    try:
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
            os.environ[name] = '1'
        cache = require_hy(config.get('cache_root', '/hy-tmp/resource_allocation_results/chapter06/experiments/reasonable_allocation89_20261109/runtime_cache'))
        for name, relative in [('NUMBA_CACHE_DIR','numba'),('MPLCONFIGDIR','matplotlib'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
            value = cache/relative;value.mkdir(parents=True,exist_ok=True);os.environ[name]=str(value)
        import numpy as np
        import torch
        from hongmiao89 import (make_env, make_pool, FullSpendSpec, make_models_for_arm,
                               make_learner, collect, METRIC_NAMES)
        from hongmiao89.method_bridge import owned_method
        from allocation23.learning import PPO
        torch.set_num_threads(1)
        try: torch.set_num_interop_threads(1)
        except RuntimeError: pass
        seed=int(config['seed']); tau=float(config['tau']); weight=float(config['weight'])
        episodes=int(config.get('episodes',3200)); every=int(config.get('eval_every',128))
        dev_paths=int(config.get('dev_paths',32)); device=str(device_override or config.get('device','cpu'))
        formal=bool(config.get('formal',False))
        keep_model=bool(config.get('save_model',config.get('save_endpoint',formal)))
        if episodes <= 0 or episodes % 8 or every <= 0 or every % 8 or dev_paths <= 0:
            raise ValueError('Positive complete8-episode training/evaluation budgets required')
        if not 0 <= tau <= 1 or not 0 <= weight <= 1:
            raise ValueError('Threshold and preference must lie in [0,1]')
        if formal and (episodes != 3200 or every != 128 or dev_paths != 32 or not keep_model):
            raise ValueError('Formal89 jobs use the registered3200/128/32 fixed endpoint profile')
        expected=config.get('source_files_sha256')
        if formal and not expected:
            raise ValueError('Formal execution requires a frozen scientific source map')
        if config.get('experiment_design_path'):
            if file_hash(require_hy(config['experiment_design_path'])) != config.get('experiment_design_sha256'):
                raise ValueError('Predeclared experiment design changed')
        sources=source_hashes(expected)
        if torch.device(device).type == 'cpu':
            if os.environ.get('CUDA_VISIBLE_DEVICES','') not in ('','-1'):
                raise ValueError('CPU workers must have CUDA hidden')
        elif torch.device(device).type == 'cuda':
            visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
            if not visible.startswith('GPU-') or ',' in visible or torch.cuda.device_count()!=1:
                raise ValueError('One explicitly UUID-bound CUDA device required')
        else: raise ValueError('Only CPU or one-visible CUDA supported')
        random.seed(830000+seed)
        cpu_rng=torch.Generator(device='cpu').manual_seed(840000+seed)
        torch.set_rng_state(cpu_rng.get_state())
        generator=torch.Generator(device='cpu').manual_seed(810000+seed)
        minibatch_rng=np.random.default_rng(820000+seed)
        training_rng=np.random.default_rng(np.random.SeedSequence([int(config.get('train_weather_master',202611076101)),seed]))
        train_weather=np.asarray(config.get('train_weather_seeds',training_rng.integers(0,2**31-1,size=episodes)),dtype=np.int64)
        development_rng=np.random.default_rng(int(config.get('dev_weather_master',202611076201)))
        dev_weather=np.asarray(config.get('dev_weather_seeds',development_rng.integers(0,2**31-1,size=dev_paths)),dtype=np.int64)
        if train_weather.shape!=(episodes,) or dev_weather.shape!=(dev_paths,):
            raise ValueError('Registered training/development weather seed shapes differ')
        env_config=environment_config(config)
        validate_formal_profile(config, env_config)
        if float(env_config['coverage_threshold'])!=tau or float(env_config['coverage_weight'])!=weight:
            raise ValueError('Environment objective must equal learner objective')
        if env_config.get('weather_mode','persistent_drought')!='persistent_drought':
            raise ValueError('Formal training uses the registered dry-season donor task')
        pool=make_pool(env_config,8);devpool=make_pool(env_config,min(8,dev_paths))
        actor,critic,neural_spec=make_models_for_arm('full',FullSpendSpec.from_env(pool[0]),seed)
        initial_actor=model_hash(actor);initial_critic=model_hash(critic)
        initialization=owned_method('initialization').initialization_record(actor)
        sampler_actor=actor;sampler_critic=critic
        if torch.device(device).type=='cuda':
            sampler_actor=copy.deepcopy(actor);sampler_critic=copy.deepcopy(critic)
        from hongmiao89.execution_operator import make_execution_actor
        dev_actor,dev_execution=make_execution_actor(sampler_actor,config.get('development_execution','latent_mean'))
        actor=actor.to(device);critic=critic.to(device)
        ao=torch.optim.Adam(actor.parameters(),lr=PPO['actor_lr'])
        vo=torch.optim.Adam(critic.parameters(),lr=PPO['value_lr'])
        learner=make_learner(neural_spec,actor.event_dim,seed,weight=weight,tau=tau,device=device)
        protocol=dict(schema='hongmiao89-reasonable-depth-independent-full-training-v1',case_id='Hongmiao89-reasonable-depth-v1',
            region_count=89,seed=seed,adapter=env_config['adapter'],spatial_sigma_scale=env_config['spatial_sigma_scale'],phase=config.get('phase','engineering'),
            experiment_design_sha256=config.get('experiment_design_sha256'),tau=tau,weight=weight,objective_order=["C","JA"],auxiliary_metrics=["J"],method='Full',job_id=config.get('job_id'),
            env_config=env_config,task_fingerprint=pool[0].task_fingerprint,
            physical_fingerprint=pool[0].physical_fingerprint,env_model_config=pool[0].model_config(),
            neural_spec=neural_spec.to_dict(),spec=FullSpendSpec.from_env(pool[0]).to_dict(),
            initialization=initialization,initial_actor_sha256=initial_actor,initial_critic_sha256=initial_critic,
            episodes=episodes,eval_every=every,dev_paths=dev_paths,device=device,dtype='float64',threads=1,
            training_execution=dict(operator='native_stochastic_complete_joint_Gaussian',M64_in_training_density=False),
            execution_operator=dev_execution['operator'],execution_description=dev_execution,
            development_execution=dev_execution,
            cpu_sampling=True,gpu_updates_native=torch.device(device).type=='cuda',
            train_weather_seeds=train_weather.tolist(),dev_weather_seeds=dev_weather.tolist(),
            source_files_sha256=sources,source_release=config.get('source_release'),
            native_ppo=dict(PPO),implementation=learner.metadata(),formal_training=formal,
            normalization=dict(kind='original_six_linear_plus_declared_public_derived_storage',learned_statistics=False,
                training_budget_m3=pool[0].initial_budget_m3,local_divisors=['reset_mean_area_ha',2.,48.,48.,1.,100.] + (['TAW/(.39-.23)',48.] if neural_spec.local_dim==8 else []),
                global_divisors=[81,'training_budget_m3',100.,10.,5.]),
            endpoint_policy='one predeclared endpoint within phase; no bestDEV/TEST checkpoint selection; no optimizer/replay snapshot',
            optimizer_backup=False,replay_backup=False,resume_supported=False)
        write_json(folder/'protocol.json',protocol);write_json(folder/'config.json',config)
        def counter(name):
            def callback(index,done):
                counts[name]['physical_steps']+=1;counts[name]['episodes']+=int(done)
            return callback
        def progress(phase):
            value=dict(status='running' if phase!='complete' else 'complete',phase=phase,episode=episode,
                episodes=episodes,learning_steps=counts['train']['physical_steps'],
                counts=copy.deepcopy(counts),pid=os.getpid(),seed=seed,tau=tau,weight=weight,device=device,
                seconds=time.monotonic()-start,formal_training=formal)
            write_json(folder/'progress.json',value)
        def audit():
            return dict(actor=model_hash(actor),critic=model_hash(critic),
                actor_optimizer=state_hash(ao.state_dict()),critic_optimizer=state_hash(vo.state_dict()),
                learner=state_hash(learner.state_dict()),sampling_rng=state_hash(generator.get_state()),
                minibatch_rng=state_hash(minibatch_rng.bit_generator.state))
        def update_health(stats):
            def check(value):
                if isinstance(value,(float,np.floating)) and not np.isfinite(value):
                    raise FloatingPointError('Nonfinite recorded learner statistic')
                if isinstance(value,dict):
                    for child in value.values():check(child)
                if isinstance(value,(list,tuple)):
                    for child in value:check(child)
            check(stats)
            measured={}
            for name,model in [('actor',actor),('critic',critic)]:
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise FloatingPointError('Nonfinite updated ' + name + ' parameter')
                gradients=[p.grad for p in model.parameters() if p.grad is not None]
                if not all(bool(torch.isfinite(g).all()) for g in gradients):
                    raise FloatingPointError('Nonfinite updated ' + name + ' gradient')
                norm=(torch.stack([g.detach().square().sum() for g in gradients]).sum().sqrt()
                      if gradients and stats['optimizer_steps'] else None)
                measured[name+'_gradient_norm_last_clipped']=(float(norm) if norm is not None else None)
            measured['nonfinite_guard']='passed_actual_parameter_gradient_and_stats_checks'
            return measured
        train_rows=[];train_Y=[];train_q=[];train_weather_trace=[];train_previous=[];train_forcing_sha=[];train_region_water=[];train_region_zero=[]
        dev_rows=[];dev_Y=[];dev_q=[];dev_trace=[];dev_previous=[];dev_forcing_sha=[];dev_region_water=[];dev_region_zero=[]
        def persist_arrays():
            def save(name,rows,Y,q,weather,previous,forcing,region_water,region_zero):
                if not rows:return
                save_npz(folder/name,episode=np.asarray([v['episode'] for v in rows],dtype=np.int64),
                    weather_seed=np.asarray([v['weather_seed'] for v in rows],dtype=np.int64),
                    metric_names=np.asarray(METRIC_NAMES,dtype='U40'),
                    metrics=np.asarray([[v[k] for k in METRIC_NAMES] for v in rows],dtype=np.float64),
                    quality=np.stack(Y),period_totals=np.stack(q),weather_trace=np.stack(weather),
                    initial_previous_weather=np.stack(previous),forcing_sha256=np.asarray(forcing,dtype='U64'),
                    regional_water_m3=np.asarray(region_water,dtype=np.float64),regional_zero_days=np.asarray(region_zero,dtype=np.int16))
            save('training.npz',train_rows,train_Y,train_q,train_weather_trace,train_previous,train_forcing_sha,train_region_water,train_region_zero)
            save('development.npz',dev_rows,dev_Y,dev_q,dev_trace,dev_previous,dev_forcing_sha,dev_region_water,dev_region_zero)
        def develop():
            progress('development');before=audit();all_rows=[]
            for first in range(0,dev_paths,len(devpool)):
                seeds=dev_weather[first:first+len(devpool)].tolist()
                results=collect(devpool,'full',dev_actor,None,seeds,weight=weight,
                    training=False,step_callback=counter('development'))
                for index,result in enumerate(results):
                    row={key:float(result[key]) for key in METRIC_NAMES}
                    row.update(episode=episode,weather_seed=int(result['weather_seed']),weather_index=first+index)
                    dev_rows.append(row);all_rows.append(row)
                    dev_Y.append(result['quality']);dev_q.append(result['period_totals'])
                    dev_region_water.append(result['actions'].sum(0));dev_region_zero.append((result['actions']==0).sum(0))
                    dev_trace.append(result['weather_trace']);dev_previous.append(result['initial_previous_weather'])
                    dev_forcing_sha.append(result['forcing_sha256'])
            if audit()!=before:raise AssertionError('Development changed model/optimizer/learner or RNG')
            write_json(folder/f'development_{episode:04d}.json',dict(episode=episode,paths=dev_paths,
                means={key:float(np.mean([row[key] for row in all_rows])) for key in METRIC_NAMES},
                learner_unchanged=True,model_selection=False,counts=counts,
                development_execution=dev_execution,
                execution_work={name:getattr(dev_actor,name) for name in ('parameter_evaluations','feasible_vectors','batch_calls','max_mass_error') if hasattr(dev_actor,name)}))
            persist_arrays()
        develop()
        collection_seconds=update_seconds=0.;optimizer_steps=0;behavior_version=0
        with gzip.open(folder/'training.jsonl.gz','wt',encoding='utf-8') as ft, gzip.open(folder/'updates.jsonl.gz','wt',encoding='utf-8') as fu:
            while episode < episodes:
                limit=min(8,episodes-episode,every-episode%every)
                size=learner.collection_size(actor,max_new=limit)
                if size not in (4,8) or size>limit:raise AssertionError('Complete native batch required')
                progress('training_collection');then=time.monotonic()
                results=collect(pool,'full',sampler_actor,sampler_critic,train_weather[episode:episode+size],
                    weight=weight,generator=generator,training=True,behavior_version=behavior_version,
                    step_callback=counter('train'))
                collection_elapsed=time.monotonic()-then;collection_seconds+=collection_elapsed
                batches=[r['batch'] for r in results]
                progress('training_update');then=time.monotonic()
                stats=learner.update(actor,critic,ao,vo,batches,minibatch_rng)
                stats.update(update_health(stats))
                elapsed=time.monotonic()-then;update_seconds+=elapsed
                optimizer_steps+=int(stats['optimizer_steps'])
                if sampler_actor is not actor:
                    sampler_actor.load_state_dict(cpu_copy(actor.state_dict()))
                    sampler_critic.load_state_dict(cpu_copy(critic.state_dict()))
                for index,result in enumerate(results):
                    row={key:float(result[key]) for key in METRIC_NAMES}
                    row.update(episode=episode+index+1,weather_seed=int(result['weather_seed']),
                        forcing_sha256=result['forcing_sha256'],learning_steps=(episode+index+1)*81,
                        K=int(result['K']),task_fingerprint=result['task_fingerprint'])
                    train_rows.append(row);train_Y.append(result['quality']);train_q.append(result['period_totals'])
                    train_region_water.append(result['actions'].sum(0));train_region_zero.append((result['actions']==0).sum(0))
                    train_weather_trace.append(result['weather_trace']);train_previous.append(result['initial_previous_weather'])
                    train_forcing_sha.append(result['forcing_sha256'])
                    ft.write(json.dumps(jsonable(row),allow_nan=False)+'\n')
                episode+=size;behavior_version+=1
                stats.update(episode=episode,learning_steps=counts['train']['physical_steps'],
                    actor_optimizer_steps=int(stats['optimizer_steps']),critic_optimizer_steps=int(stats['optimizer_steps']),
                    collection_seconds=collection_elapsed,update_seconds=elapsed,device=device,
                    sampling_device='cpu',behavior_version=behavior_version-1)
                fu.write(json.dumps(jsonable(stats),allow_nan=False)+'\n');ft.flush();fu.flush()
                if episode % every == 0 or episode==episodes:
                    develop()
                    print(json.dumps(dict(episode=episode,seed=seed,tau=tau,weight=weight,device=device,
                        seconds=time.monotonic()-start)),flush=True)
        if counts['train']!=dict(episodes=episodes,physical_steps=episodes*81):
            raise AssertionError('True training interaction counts differ')
        expected_dev=(episodes//every+1 if episodes%every==0 else episodes//every+2)*dev_paths
        if counts['development']!=dict(episodes=expected_dev,physical_steps=expected_dev*81):
            raise AssertionError('True development interaction counts differ')
        if learner.episodes!=episodes:raise AssertionError('Learner count differs from true collection')
        final_actor=model_hash(actor);final_critic=model_hash(critic);endpoint=folder/'model.pt'
        if keep_model:
            save_endpoint(endpoint,dict(format='hongmiao89-reasonable-depth-dual-random-fixed-endpoint-v1',episode=episodes,
                actor=actor.state_dict(),critic=critic.state_dict(),actor_sha256=final_actor,critic_sha256=final_critic,
                protocol=protocol,optimizer_backup=False,replay_backup=False,resume_supported=False))
        if source_hashes(sources)!=sources:raise AssertionError('Frozen source changed during job')
        persist_arrays();progress('complete')
        write_json(folder/'completed.json',dict(status='complete',formal_training=formal,seed=seed,tau=tau,weight=weight,
            episodes=episodes,train_episodes=episodes,train_physical_steps=episodes*81,
            dev_episodes=expected_dev,dev_physical_steps=expected_dev*81,counts=counts,
            total_episodes=episodes+expected_dev,total_physical_steps=(episodes+expected_dev)*81,
            source_unchanged=True,saved_model_count=int(keep_model),endpoint_retained=keep_model,
            endpoint_sha256=file_hash(endpoint) if keep_model else None,
            endpoint_actor_sha256=final_actor,endpoint_critic_sha256=final_critic,
            actor_changed=final_actor!=initial_actor,critic_changed=final_critic!=initial_critic,
            actor_optimizer_steps=optimizer_steps,critic_optimizer_steps=optimizer_steps,
            collection_seconds=collection_seconds,update_seconds=update_seconds,
            seconds=time.monotonic()-start,device=device,cpu_sampling=True,
            gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if torch.device(device).type=='cuda' else 0,
            optimizer_backup=False,replay_backup=False,resume_supported=False,
            model_selection=False,holdout_selection=False))
        return folder
    except BaseException:
        write_json(folder/'failure.json',dict(status='failed',traceback=traceback.format_exc(),episode=episode,
            counts=counts,seconds=time.monotonic()-start))
        raise
    finally:
        for env in pool+devpool:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output');parser.add_argument('--device')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    train(config,output_override=args.output,device_override=args.device)


if __name__=='__main__':main()
