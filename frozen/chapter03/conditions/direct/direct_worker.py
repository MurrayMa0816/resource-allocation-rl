"""Direct learned total coordinate using the frozen formal Chapter 3 native A/PPO.

The only runtime Task adapter inserts a logging/evaluation callback after a real
physical step. Native complete-episode Monte Carlo batches and PPO update are
reused without editing the frozen package. No optimizer/replay is persisted.
"""
from __future__ import annotations
import argparse,ast,copy,csv,gzip,hashlib,inspect,json,os,random,sys,textwrap,time,types
from pathlib import Path

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True

ARMS={'A_DIRECT_TOTAL':dict(variant='A_FULL',temporal='adaptive',quota_reference='pace')}

SCHEMA='chapter3-A-direct-native-supplement-v1'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True)
    p.add_argument('--domain',choices=('v5','radio'),required=True)
    p.add_argument('--arm',choices=ARMS,default='A_DIRECT_TOTAL')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--steps',type=int,required=True)
    p.add_argument('--eval-every',type=int,default=512)
    p.add_argument('--development-paths',type=int,default=8)
    p.add_argument('--final-paths',type=int,default=128)
    p.add_argument('--path-master',type=int,default=202611043101)
    p.add_argument('--final-master',type=int,default=202611058199)
    p.add_argument('--data-root',default=os.environ.get('ALLOCATION23_DATA_ROOT','/hy-tmp/resource_allocation_results/chapter3/full_spend_revision_20261104/execution/shared_data/eurosat_rgb'))
    p.add_argument('--output',required=True)
    p.add_argument('--engineering-gate',action='store_true',help='Marks a short gate; never changes architecture or update equations')
    return p


class Rows:
    def __init__(self,path,fields):
        self.f=gzip.open(path,'wt',newline='');self.w=csv.DictWriter(self.f,fieldnames=fields);self.w.writeheader()
    def add(self,row):self.w.writerow(row)
    def flush(self):self.f.flush()
    def close(self):self.f.close()


def install_step_callback(task,callback):
    """Proof-carrying instrumentation: stripping one injected node gives native AST."""
    original=textwrap.dedent(inspect.getsource(type(task).episode))
    before=ast.parse(original);after=copy.deepcopy(before)
    fn=after.body[0]
    loops=[n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='t']
    if len(loops)!=1:raise AssertionError('Native Task episode structure changed')
    hook=ast.parse('if training:\n    self._learning_step_callback(t, action, float(reward))').body[0]
    loops[0].body.append(hook)
    stripped=copy.deepcopy(after)
    next(n for n in stripped.body[0].body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='t').body.pop()
    if ast.dump(before,include_attributes=False)!=ast.dump(stripped,include_attributes=False):
        raise AssertionError('Instrumentation modified native learning/physics statements')
    ast.fix_missing_locations(after)
    namespace={}
    exec(compile(after,'<native Task.episode + registered side-effect callback>','exec'),type(task).episode.__globals__,namespace)
    task.episode=types.MethodType(namespace['episode'],task)
    task._learning_step_callback=callback
    return dict(native_task_episode_sha256=hashlib.sha256(original.encode()).hexdigest(),
                native_AST_identical_after_removing_single_callback=True,
                callback='logs actual physical step; eval at exact step, before next action')


def train(args):
    source=Path(args.source_root).resolve()
    sys.path.insert(0,str(source))
    import numpy as np
    import torch
    from allocation23.training import configure_runtime,fresh
    from allocation23.learning import PPO,update as native_update
    from allocation23.methods import make_value
    from direct_policy import DirectTotalActor
    from native_cuda import make_device_update
    update,device_adapter=make_device_update(native_update)
    from allocation23.task import Task
    from allocation23.io import source_hashes,digest,state_hash,write_json
    configure_runtime()
    horizon=24 if args.domain=='v5' else 32
    if args.steps<=0 or args.steps%(horizon*PPO['batch_episodes']):
        raise ValueError('Use a positive whole native 8-episode PPO batch budget')
    if args.eval_every<0 or args.development_paths<=0 or args.final_paths<0:
        raise ValueError('Invalid evaluation budget')
    if args.engineering_gate and args.final_paths:
        raise ValueError('Engineering gate does not open the final test split')
    if args.path_master==args.final_master:raise ValueError('Train/final namespace must differ')
    output=Path(args.output).resolve()
    if not output.is_relative_to(Path('/hy-tmp')) or output.exists():
        raise ValueError('Use a new, independent HY output directory')
    mapping=ARMS[args.arm]
    device=torch.device(args.device)
    if device.type not in ('cpu','cuda'):
        raise ValueError('Only native CPU or CUDA runtime supported')
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; no silent CPU fallback')
    output.mkdir(parents=True)
    for name in ('TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','CUDA_CACHE_PATH',
                 'TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):
        cache=output/'runtime'/name.lower();cache.mkdir(parents=True)
        os.environ[name]=str(cache)
    cfg=argparse.Namespace(domain=args.domain,chapter=3,method='A',seed=args.seed,
        path_master=args.path_master,holdout_master=args.final_master,radio_groups=100,
        condition_file=None,data_root=args.data_root,**mapping)
    task=Task(args.domain,3,'A',args.seed,path_master=args.path_master,data_root=args.data_root)
    dev=Task(args.domain,3,'A',args.seed,path_master=args.path_master,data_root=args.data_root)
    random.seed(830000+args.seed)
    torch.manual_seed(840000+args.seed)
    actor=DirectTotalActor(task.spec,args.seed).to(device=device,dtype=torch.float64)
    critic=make_value(task.spec,args.seed).to(device=device,dtype=torch.float64)
    state=(actor,critic,torch.optim.Adam(actor.parameters(),lr=PPO['actor_lr']),
        torch.optim.Adam(critic.parameters(),lr=PPO['value_lr']),
        torch.Generator(device=device).manual_seed(810000+args.seed),np.random.default_rng(820000+args.seed))
    actor,critic,ao,vo,generator,rng=state
    initial_actor=state_hash(actor)
    initial_critic=state_hash(critic)
    sources=source_hashes()
    worker_sha=digest(Path(__file__))
    trace=Rows(output/'training_steps.csv.gz',['training_step','episode','episode_step','condition',
        'native_reward','learning_reward','resource_cost','cumulative_cost','physical_amount'])
    episodes=Rows(output/'training_episodes.csv.gz',['episode','training_step','condition','native_return','resource_fraction'])
    updates=Rows(output/'updates.csv.gz',['training_step','episode','likelihood_error','optimizer_steps',
        'actor_loss','value_loss','macro_kl','clip_fraction','actor_grad_norm','value_grad_norm',
        'early_stop','active_events','advantage_std'])
    raw=Rows(output/'evaluation_paths.csv.gz',['training_step','phase','path_index','path_seed',
        'native_return','resource_fraction','steps','replay_error'])
    curve=Rows(output/'evaluation_curve.csv.gz',['training_step','phase','paths',
        'evaluation_environment_steps','native_mean','native_std','min_return','max_return'])
    start=time.perf_counter();learning_steps=optimizer_steps=eval_steps=0
    replay_steps=actor_update_calls=0
    maximum_resource_error=0.
    full_spend_episodes=0
    episode_index=0;history_cost=0.;pending_eval=False;batches=[];succeeded=False

    def progress(phase):
        write_json(output/'progress.json',dict(schema=SCHEMA,pid=os.getpid(),phase=phase,
            method=args.arm,domain=args.domain,seed=args.seed,training_steps=learning_steps,
            target_training_steps=args.steps,complete_episodes=episode_index,
            ppo_optimizer_steps=optimizer_steps,evaluation_steps=eval_steps,
            native_replay_steps=replay_steps,actor_update_calls=actor_update_calls,
            actual_device=str(device),full_spend_episodes_verified=full_spend_episodes,
            maximum_resource_fraction_error=maximum_resource_error,
            elapsed_seconds=time.perf_counter()-start))

    def evaluate_current(phase,paths,evaluation_task,save_examples=False):
        nonlocal eval_steps,replay_steps,maximum_resource_error,full_spend_episodes
        cuda_before=[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []
        before=(state_hash(actor),state_hash(critic),generator.get_state().clone(),
                copy.deepcopy(rng.bit_generator.state),torch.get_rng_state().clone(),random.getstate())
        scores=[];allocations=[]
        progress('development' if phase=='dev' else 'holdout')
        for i in range(paths):
            result=evaluation_task.episode(phase,i,actor,deterministic=True,capture=True)
            replayed,_=evaluation_task.replay(phase,i,result)
            error=abs(replayed-result['native_return'])
            if error!=0.:raise AssertionError('Original physical action replay differs')
            resource_error=abs(result['total_used']/evaluation_task.spec.initial_budget-1.)
            maximum_resource_error=max(maximum_resource_error,resource_error)
            full_spend_episodes+=1
            if resource_error>1e-9:
                raise AssertionError('Evaluation did not allocate the full episode resource')
            value=float(result['native_return']);scores.append(value)
            raw.add(dict(training_step=learning_steps,phase=phase,path_index=i,
                path_seed=result['plan']['task_seed'],native_return=value,
                resource_fraction=result['total_used']/evaluation_task.spec.initial_budget,
                steps=evaluation_task.spec.horizon,replay_error=error))
            if (save_examples or learning_steps==0) and i==0:allocations.append(result['actions'].copy())
        cuda_after=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
        if len(cuda_before)!=len(cuda_after) or any(not torch.equal(a,b) for a,b in zip(cuda_before,cuda_after)):
            raise AssertionError('Independent evaluation changed global CUDA RNG')
        if before[0]!=state_hash(actor) or before[1]!=state_hash(critic) or not torch.equal(before[2],generator.get_state()) or before[3]!=rng.bit_generator.state or not torch.equal(before[4],torch.get_rng_state()) or before[5]!=random.getstate():
            raise AssertionError('Independent evaluation changed training model or RNG')
        scores=np.asarray(scores)
        result=dict(training_step=learning_steps,phase=phase,paths=paths,
            evaluation_environment_steps=paths*evaluation_task.spec.horizon,
            native_mean=float(scores.mean()),native_std=float(scores.std(ddof=1)) if paths>1 else 0.,
            min_return=float(scores.min()),max_return=float(scores.max()))
        curve.add(result);curve.flush();raw.flush()
        eval_steps+=result['evaluation_environment_steps']
        replay_steps+=result['evaluation_environment_steps']
        if allocations:np.savez_compressed(output/('initial_allocation_examples.npz' if learning_steps==0 else 'allocation_examples.npz'),allocations=np.asarray(allocations))
        progress('training')
        return result

    def callback(t,action,reward):
        nonlocal learning_steps,history_cost,pending_eval
        amount=float(np.asarray(action).sum());cost=amount/task.spec.initial_budget
        history_cost+=cost;learning_steps+=1
        if t!=horizon-1 and reward!=0.:raise AssertionError('Native reward must be terminal-only')
        trace.add(dict(training_step=learning_steps,episode=episode_index,episode_step=t,
            condition=1.,native_reward=reward,learning_reward=reward,resource_cost=cost,
            cumulative_cost=history_cost,physical_amount=amount))
        if args.eval_every and learning_steps%args.eval_every==0 and learning_steps<args.steps:
            if t==horizon-1:pending_eval=True
            else:evaluate_current('dev',args.development_paths,dev)

    transform=install_step_callback(task,callback)
    protocol=dict(schema=SCHEMA,arguments=vars(args),method=args.arm,domain=args.domain,
        profile='engineering_gate' if args.engineering_gate else 'formal_100k_five_seed',
        native_implementation='allocation23.training.fresh / learning.update / Task.episode',
        source_identity=dict(package=sources,worker_sha256=worker_sha,direct_capsule={p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}),runtime_adapter=transform,
        arm_mapping=mapping,ppo=dict(PPO),initial_algorithm=actor.metadata(),
        direct_mapping=actor.direct_initialization,device_adapter=device_adapter,
        comparison_scope='same A spatial backbone, state, native PPO and feasibility; Direct replaces only state-dependent pacing shift by fixed initialized total coordinate',
        physical=task.physical_dict,spec=task.spec.to_dict(),data=task.data_identity,
        device_choice=dict(selected=str(device),reason='explicit CUDA native float64 PPO; same equations with tensor-device keywords only',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None),
        reward='unchanged native terminal F',online_from_scratch=True,offline_data=False,
        storage='one fixed terminal actor+critic evaluation payload, all compressed scalar CSV, one path0 allocation; no disk replay/Adam/periodic checkpoint',
        actual_device=str(device),fresh_initialization=True,
        initial_actor_sha256=initial_actor,initial_critic_sha256=initial_critic,
        initial_optimizer_entries=dict(actor=len(ao.state),critic=len(vo.state)),
        evaluation_interval_actual_steps=args.eval_every,
        evaluation_timing='exact step threshold; native terminal-step PPO update precedes DEV, inside-episode DEV preserves native model and all training RNG',
        final_model_selection='fixed final100608environment-step endpoint; independent final is never used for model selection',
        evaluation='same native task, deterministic mean-latent decoder, independent dev/test and exact native replay',
        selection='fixed endpoint; no development/test best-model selection',
        software=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,torch=torch.__version__))
    write_json(output/'protocol.json',protocol)
    try:
        if args.eval_every:evaluate_current('dev',args.development_paths,dev)
        while learning_steps<args.steps:
            history_cost=0.
            result=task.episode('train',episode_index,actor,critic,generator,training=True)
            if learning_steps!=(episode_index+1)*horizon:raise AssertionError('Real learning-step counter mismatch')
            resource_error=abs(result['total_used']/task.spec.initial_budget-1.)
            maximum_resource_error=max(maximum_resource_error,resource_error)
            full_spend_episodes+=1
            if resource_error>1e-9:
                raise AssertionError('Training did not allocate the full episode resource')
            episodes.add(dict(episode=episode_index,training_step=learning_steps,condition=1.,
                native_return=result['native_return'],resource_fraction=result['total_used']/task.spec.initial_budget))
            batches.append(result['batch'])
            episode_index+=1
            if len(batches)==PPO['batch_episodes']:
                stats=update(actor,critic,ao,vo,batches,rng);batches=[]
                actor_update_calls+=1
                optimizer_steps+=stats['optimizer_steps']
                if not all(np.isfinite(x) for k,x in stats.items() if isinstance(x,(int,float))):
                    raise FloatingPointError('Nonfinite native PPO diagnostic')
                updates.add(dict(training_step=learning_steps,episode=episode_index,**stats));updates.flush()
            if pending_eval:
                evaluate_current('dev',args.development_paths,dev);pending_eval=False
            if episode_index%8==0:trace.flush();episodes.flush();progress('training')
            if episode_index%128==0 and source_hashes()!=sources:
                raise AssertionError('Native scientific source changed during training')
        if batches:raise AssertionError('Final budget must commit a complete native PPO batch')
        if args.eval_every:evaluate_current('dev',args.development_paths,dev)
        if source_hashes()!=sources or digest(Path(__file__))!=worker_sha or {p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}!=protocol['source_identity']['direct_capsule']:
            raise AssertionError('Native source or execution worker changed during training')
        model_dir=output/'model';model_dir.mkdir()
        payload=model_dir/'state.pt'
        torch.save(dict(schema=SCHEMA,evaluation_only=True,actor=actor.state_dict(),critic=critic.state_dict(),
            identity=protocol,environment_steps=learning_steps,optimizer_steps=optimizer_steps,
            optimizer_or_replay_saved=False),payload)
        write_json(model_dir/'online_manifest.json',dict(method=args.arm,domain=args.domain,seed=args.seed,
            profile=protocol['profile'],arm_mapping=mapping,training_steps=learning_steps,
            evaluation_only=True,optimizer_or_replay_saved=False))
        # Verify the actual single evaluation payload can be strictly reloaded.
        check=torch.load(payload,map_location='cpu',weights_only=False)
        saved_actor,saved_critic=state_hash(actor),state_hash(critic)
        actor.load_state_dict(check['actor'],strict=True);critic.load_state_dict(check['critic'],strict=True)
        if state_hash(actor)!=saved_actor or state_hash(critic)!=saved_critic:
            raise AssertionError('Endpoint tensor bytes changed after strict reloading')
        final=None
        if args.final_paths:
            final_task=Task(args.domain,3,'A',args.seed,path_master=args.final_master,
                            data_root=args.data_root,holdout=True)
            final=evaluate_current('final',args.final_paths,final_task,True)
            final_task.env.close()
        if source_hashes()!=sources:raise AssertionError('Source changed during final evaluation')
        # Completion is published only after all gzip footers are closed and
        # read to EOF, preventing consumers from accepting a partial stream.
        for writer in (trace,episodes,updates,raw,curve):writer.close()
        artifacts={}
        for file in sorted(output.rglob('*')):
            relative=file.relative_to(output)
            if not file.is_file() or relative.parts[0]=='runtime' or relative.as_posix() in ('progress.json','completed.json','failed.json'):
                continue
            if file.suffix=='.gz':
                with gzip.open(file,'rb') as stream:
                    for _ in iter(lambda:stream.read(1024**2),b''):pass
            artifacts[relative.as_posix()]=digest(file)
        result=dict(schema=SCHEMA,success=True,method=args.arm,domain=args.domain,
            profile=protocol['profile'],seed=args.seed,training_steps=learning_steps,
            evaluation_steps=eval_steps,native_replay_environment_steps=replay_steps,
            total_physical_environment_steps=learning_steps+eval_steps+replay_steps,
            elapsed_seconds=time.perf_counter()-start,actual_device=str(device),
            endpoint_reload_pass=True,initial_actor_sha256=initial_actor,initial_critic_sha256=initial_critic,
            model_count=1,optimizer_saved=False,replay_saved=False,periodic_models_saved=False,
            artifacts_sha256=artifacts,gzip_complete_before_success=True,
            maximum_resource_fraction_error=maximum_resource_error,
            full_spend_episodes_verified=full_spend_episodes,
            final_evaluation=final,model='model/state.pt',model_sha256=digest(payload),
            model_files_sha256={str(p.relative_to(model_dir)):digest(p) for p in model_dir.rglob('*') if p.is_file()},
            exact_training_resume=False,optimizer_or_replay_saved=False,full_spend_required=True,resource_tolerance_fraction=1e-9,
            source_identity=protocol['source_identity'],source_unchanged=True,
            actor_changed=state_hash(actor)!=initial_actor,
            accounting=dict(environment_steps=learning_steps,complete_episodes=episode_index,
                            ppo_optimizer_steps=optimizer_steps,ppo_batch_episodes=PPO['batch_episodes'],
                            actor_update_calls=actor_update_calls,evaluation_environment_steps=eval_steps,
                            native_replay_environment_steps=replay_steps),
            arm_mapping=mapping,native_replay_error=0.)
        write_json(output/'completed.json',result);succeeded=True;progress('complete')
        print(json.dumps(result,allow_nan=False),flush=True)
        return result
    finally:
        for writer in (trace,episodes,updates,raw,curve):writer.close()
        task.env.close();dev.env.close()
        if not succeeded:
            import traceback
            write_json(output/'failed.json',dict(schema=SCHEMA,training_steps=learning_steps,
                traceback=traceback.format_exc(),exact_resume_available=False,
                action='new independent attempt with same registered seed; never silently concatenate'))


def main():
    if len(sys.argv)>1 and sys.argv[1]=='evaluate':
        from evaluate_direct import main as evaluate
        evaluate(sys.argv[2:]);return
    if len(sys.argv)>1 and sys.argv[1]=='train':del sys.argv[1]
    train(parser().parse_args())

if __name__=='__main__':main()
