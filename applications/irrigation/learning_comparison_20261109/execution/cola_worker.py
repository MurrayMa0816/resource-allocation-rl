"""Additional COLA comparison: native8-step TD and real1-step OADM labels."""
from pathlib import Path
import argparse,copy,gzip,hashlib,importlib,json,os,random,sys,time,traceback
APP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP))
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[name]='1'
from paths import STUDY,bootstrap
STUDY=STUDY/"additional_cola_20261109"
sys.dont_write_bytecode=True
for name,child in [("NUMBA_CACHE_DIR","numba_shared"),("MPLCONFIGDIR","mpl"),("TMPDIR","tmp"),("XDG_CACHE_HOME","xdg")]:
    folder=STUDY/"runtime_cache"/child;folder.mkdir(parents=True,exist_ok=True);os.environ[name]=str(folder)
bootstrap()
import numpy as np
import torch
from execution.bridge import PublicPool,PREFERENCE
from baselines._support.replay import NStepReplay
from baselines._support.update_clock import physical_update_target
from baselines._support.runtime import rng_state,restore_rng,parameter_devices,optimizer_cuda_count
from baselines._support.common import digest_state,cpu_state,finite_stats

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):
    p=Path(p);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def safe(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return{k:safe(v)for k,v in x.items()}
    if isinstance(x,(tuple,list)):return[safe(v)for v in x]
    if torch.is_tensor(x):return x.detach().cpu().tolist()
    return x
def line(f,j):f.write(json.dumps(safe(j),separators=(',',':'),allow_nan=False)+'\n');f.flush()
def check_sources(cfg):
    for p,h in cfg['source_files_sha256'].items():
        if sha(p)!=h:raise RuntimeError('Scientific source changed:'+p)
def endpoint(agent):
    f=getattr(agent,'checkpoint_state_dict',None)or getattr(agent,'endpoint',None)
    return cpu_state(f()if f else agent.state_dict())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config_path=Path(args.config).resolve();cfg=json.loads(config_path.read_text());cfg_sha=sha(config_path)
    check_sources(cfg)
    if cfg['method']!='cola' or cfg['algorithm_config'].get('n_step')!=8 or cfg['algorithm_config'].get('oadm_one_step_target') is not True:raise ValueError('Registered native COLA8stepTD/1stepOADM only')
    out=Path(cfg['output']).resolve()
    if not out.is_relative_to(STUDY)or out.exists():raise FileExistsError('Fresh output required')
    out.mkdir(parents=True);started=time.monotonic();learning=0;episode=0;updates=0
    counts={k:dict(physical_steps=0,episodes=0)for k in ['train','development']};pool=None;dev=None
    try:
        for env,child in [('NUMBA_CACHE_DIR','numba'),('MPLCONFIGDIR','mpl'),('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg')]:
            p=STUDY/'runtime_cache'/cfg['job_id']/child;p.mkdir(parents=True,exist_ok=True);os.environ[env]=str(p)
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        device=cfg['device'];seed=int(cfg['seed'])
        if device=='cuda:0':
            if os.environ.get('CUDA_VISIBLE_DEVICES')!=cfg['gpu_uuid']or torch.cuda.device_count()!=1:raise RuntimeError('One UUID-bound CUDA GPU required')
            torch.cuda.set_device(0)
        random.seed(seed+91000);np.random.seed(seed+92000);torch.manual_seed(seed+93000)
        rng=np.random.default_rng(seed+94000)
        module=importlib.import_module('methods.'+cfg['method']+'_adapter');native_config=cfg['algorithm_config']
        pool=PublicPool(8);dev=PublicPool(min(8,cfg['dev_paths']))
        agent=module.make_agent(pool.obs_dim,pool.action_dim,seed,device,native_config)
        initial=digest_state(endpoint(agent));initial_training=digest_state(agent.state_dict())
        total=cfg['learning_steps'];every=cfg['eval_every'];horizon=81
        if total%648 or every%648:raise ValueError('Complete8lane81day panels required')
        replay=None if agent.on_policy else NStepReplay(total,pool.obs_dim,pool.action_dim,n_step=8,n_env=8)
        protocol=dict(schema='chapter6-COLA-additional-fixed-endpoint-v1',config=cfg,config_sha256=cfg_sha,
            method=cfg['method'],seed=seed,initial_model_sha256=initial,initial_training_state_sha256=initial_training,
            obs_dim=pool.obs_dim,action_dim=pool.action_dim,task_fingerprint=pool.envs[0].task_fingerprint,
            implementation=agent.metadata(),device=device,parameters_on_devices=parameter_devices(agent),
            deployment='fixed64 original Gaussian→native action coordinates→C6 feasible water→physical arithmeticmean',
            point_array_sha256=hashlib.sha256(pool.points.tobytes()).hexdigest(),
            training_preferences='native per-episode positive continuous simplex; preference_floor=.02',
            deployment_preference=[.2,.8],objective_order=['C','JA'],coverage_threshold=.9,
            optimizer_backup=False,replay_backup=False,resume_supported=False,selection='fixed endpoint only;DEV not selection',TD_n_step=8,OADM_target='actual same-transition one-step successor',replay_rewards='sum of actual rewards over up to8same-episode transitions;gamma1',candidate_history='COLA chosen after observedCAPQL DEV collapse; originalCAPQL retained unchanged')
        write(out/'protocol.json',safe(protocol))
        dev_records=[];dev_rows=[];update_seconds=0.;last_progress=0.
        def progress(phase,force=False):
            nonlocal last_progress
            now=time.monotonic()
            if not force and now-last_progress<3:return
            v=dict(status='running',phase=phase,pid=os.getpid(),method=cfg['method'],seed=seed,
                learning_steps=learning,target=total,episodes=episode,updates=updates,counts=copy.deepcopy(counts),
                seconds=now-started,training_update_seconds=update_seconds,device=device)
            if device.startswith('cuda'):v.update(cuda_allocated=torch.cuda.memory_allocated(),cuda_reserved=torch.cuda.memory_reserved())
            write(out/'progress.json',v);last_progress=now
        def evaluate(f):
            snapshot=rng_state(dict(agent=agent,worker_rng=rng));before=digest_state(agent.state_dict());rows=[]
            try:
                for start in range(0,cfg['dev_paths'],len(dev.envs)):
                    seeds=cfg['dev_weather_seeds'][start:start+len(dev.envs)]
                    assert len(seeds)==len(dev.envs)
                    obs=dev.reset(seeds)
                    for day in range(81):
                        points=module.fixed64_actions(agent,obs,dev.points,PREFERENCE)
                        obs,r,d=dev.step(dev.decode(points,agent.action_mode),physical=True)
                        counts['development']['physical_steps']+=len(seeds);counts['development']['episodes']+=int(d.sum())
                    rows.extend(dev.rows());progress('development')
            finally:restore_rng(snapshot)
            if digest_state(agent.state_dict())!=before:raise RuntimeError('DEV mutated learning state')
            dev_rows.extend(dict(row,learning_steps=learning,completed_episodes=episode)for row in rows)
            rec=dict(learning_steps=learning,completed_episodes=episode,mean_C=float(np.mean([r['C']for r in rows])),mean_JA=float(np.mean([r['JA']for r in rows])),counts=copy.deepcopy(counts),rows=rows,rng_restored=True,learning_state_unchanged=True)
            line(f,rec);dev_records.append({k:v for k,v in rec.items()if k!='rows'});persist()
        def persist():
            if not dev_rows:return
            dest=out/'development.npz';tmp=dest.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,learning_steps=np.array([r['learning_steps']for r in dev_rows]),
                weather_seed=np.array([r['weather_seed']for r in dev_rows]),C=np.array([r['C']for r in dev_rows]),JA=np.array([r['JA']for r in dev_rows]),
                quality=np.stack([r['quality']for r in dev_rows]),period_totals=np.stack([r['period_totals']for r in dev_rows]),
                weather_trace=np.stack([r['weather_trace']for r in dev_rows]),initial_previous_weather=np.stack([r['initial_previous_weather']for r in dev_rows]),
                region_ids=np.array(pool.envs[0].region_ids),area_ha=np.asarray(pool.envs[0].scenario.areas_ha))
            tmp.replace(dest)
        with gzip.open(out/'development.jsonl.gz','wt',compresslevel=1)as fe,gzip.open(out/'training.jsonl.gz','wt',compresslevel=1)as ft,gzip.open(out/'updates.jsonl.gz','wt',compresslevel=1)as fu:
            evaluate(fe)
            while learning<total:
                obs=pool.reset(cfg['train_weather_seeds'][episode:episode+8])
                if hasattr(agent,'train_preferences'):prefs=agent.train_preferences(learning,total,rng)
                elif hasattr(agent,'sample_weights'):prefs=agent.sample_weights(8,rng)
                else:prefs=np.tile(PREFERENCE,(8,1))
                traj={k:[]for k in ['obs','actions','logp','values','rewards','dones']}
                for day in range(81):
                    if getattr(agent,'resample_preference_each_step',False):prefs=agent.sample_weights(8,rng)
                    if agent.on_policy:sample=agent.sample_info(obs,prefs);actions=np.asarray(sample['actions'])
                    else:actions=rng.uniform(-1,1,(8,pool.action_dim)).astype(np.float32)if learning<native_config['start_steps']else agent.act(obs,prefs,False)
                    if hasattr(agent,'observe_training'):agent.observe_training(obs)
                    nxt,rewards,dones=pool.step(actions,agent.action_mode);learning+=8;counts['train']['physical_steps']+=8;counts['train']['episodes']+=int(dones.sum())
                    if bool(dones.all())!=(day==80)or(day<80 and np.any(rewards)):raise RuntimeError('Nonterminal feedback')
                    if agent.on_policy:
                        for k,v in dict(obs=obs,actions=actions,logp=sample['logp'],values=sample['values'],rewards=rewards,dones=dones).items():traj[k].append(np.asarray(v).copy())
                    else:
                        replay.add(obs=obs,actions=actions,weights=np.asarray(prefs,np.float32),rewards=rewards,next_obs=nxt,dones=dones)
                        if replay.size>=native_config['batch_size']and learning>=native_config['learning_starts']:
                            target=physical_update_target(learning,native_config['learning_starts'],native_config['batch_size'],native_config['updates_per_step'])
                            while updates<target:
                                t=time.monotonic();stats=finite_stats(agent.update(replay.sample(native_config['batch_size'],rng)));update_seconds+=time.monotonic()-t;updates+=1
                                line(fu,dict(learning_steps=learning,updates=updates,stats=stats))
                    obs=nxt;progress('training')
                episode+=8
                if replay is not None:replay.assert_episode_boundary()
                for row in pool.rows():line(ft,dict(learning_steps=learning,completed_episodes=episode,**row))
                if agent.on_policy:
                    t=time.monotonic();stats=finite_stats(agent.update_rollout({k:np.stack(v)for k,v in traj.items()},prefs,learning));update_seconds+=time.monotonic()-t
                    updates+=int(stats.get('update_groups',stats.get('optimizer_updated',True)));line(fu,dict(learning_steps=learning,updates=updates,stats=stats))
                if learning%every==0:evaluate(fe)
                progress('training')
        if learning!=total or counts['train']['episodes']!=cfg['episodes']or len(dev_records)!=cfg['episodes']//(every//81)+1:raise AssertionError('Incomplete fixed budget')
        replay.assert_episode_boundary()
        if replay.original_transitions!=total or replay.emitted_transitions!=total:raise AssertionError('Replay transition accounting changed')
        check_sources(cfg);assert sha(config_path)==cfg_sha
        payload=endpoint(agent);final=digest_state(payload)
        if final==initial or updates<1:raise RuntimeError('No actual learning update')
        snapshot=rng_state(dict(agent=agent,worker_rng=rng));path=out/'endpoint.pt'
        torch.save(dict(state=payload,config=cfg,protocol=protocol,model_sha256=final,inference_only=True),path)
        persisted=torch.load(path,map_location='cpu',weights_only=False)
        if digest_state(persisted['state'])!=final:raise RuntimeError('Endpoint bytes differ')
        clone=module.make_agent(pool.obs_dim,pool.action_dim,seed,device,native_config)
        loader=getattr(clone,'load_checkpoint_state_dict',None)or clone.load_state_dict;loader(persisted['state'])
        np.testing.assert_array_equal(module.fixed64_actions(agent,obs,pool.points,PREFERENCE),module.fixed64_actions(clone,obs,pool.points,PREFERENCE))
        restore_rng(snapshot);check_sources(cfg)
        proof=dict(parameter_devices=parameter_devices(agent),cuda_optimizer_tensors=optimizer_cuda_count(agent),peak_cuda_reserved=torch.cuda.max_memory_reserved()if device.startswith('cuda')else 0)
        if device.startswith('cuda')and proof['cuda_optimizer_tensors']<1:raise RuntimeError('No actual CUDA optimizer state')
        receipt=dict(status='COMPLETE',method=cfg['method'],seed=seed,learning_steps=learning,episodes=episode,updates=updates,
            counts=counts,actual_pool_counters=dict(train=pool.counters,development=dev.counters),seconds=time.monotonic()-started,update_seconds=update_seconds,initial_model_sha256=initial,model_sha256=final,endpoint_sha256=sha(path),
            endpoint_reload_exact=True,source_unchanged=True,replay_metadata=replay.metadata(),replay_pending_transitions=0,maximum_projection_mass_error_m3=max(pool.maximum_mass_error,dev.maximum_mass_error),
            device_proof=proof,new_TEST_paths=0,recovery=dict(command=None),optimizer_backup=False,replay_backup=False)
        write(out/'completed.json',receipt);write(out/'progress.json',receipt)
    except BaseException:
        for name,p in [('train',pool),('development',dev)]:
            if p is not None:counts[name]=copy.deepcopy(p.counters)
        write(out/'failed.json',dict(status='FAILED',method=cfg['method'],seed=cfg['seed'],counts=counts,learning_steps=learning,episodes=episode,error=traceback.format_exc(),seconds=time.monotonic()-started,recovery=dict(command=None)))
        raise
    finally:
        for p in [pool,dev]:
            if p is not None:p.close()
if __name__=='__main__':main()
