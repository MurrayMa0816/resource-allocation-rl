"""Registered frozen-policy Chapter 3 conditions and rich public trajectory audit.

No learner update, checkpoint mutation, reward change or policy-visible future data.
Run each unit in a fresh process with the original frozen package on sys.path.
"""
from __future__ import annotations
import argparse, contextlib, copy, csv, gzip, hashlib, importlib.util, json, math, os, sys, time, types
from pathlib import Path

SCHEMA='chapter3-frozen-conditions-rich-v1'
MASTER=202611058199
NATIVE_SCHEMA='chapter3-double-random-native-formal-v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()

def write(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def physical_of(protocol):
    return protocol.get('physical') or protocol['environment']['native_contract']['physical']

def register(args):
    root=Path(args.output).resolve()
    if not root.is_relative_to(Path('/hy-tmp')):raise ValueError('Registry must reside physically on HY')
    root.mkdir(parents=True,exist_ok=False)
    records={}
    for domain,filename in [('radio',args.radio_protocol),('v5',args.v5_protocol)]:
        protocol=json.loads(Path(filename).read_text());base=physical_of(protocol)
        records[domain]={}
        levels=[('default',None,None),('budget08','budget',.8),('budget12','budget',1.2)]
        levels += ([('persistence02','persistence',.2),('persistence095','persistence',.95)] if domain=='radio'
                   else [('arrivals_front','arrival_permutation','front'),('arrivals_middle','arrival_permutation','middle')])
        for key,factor,level in levels:
            physical=copy.deepcopy(base);permutation=None
            if factor=='budget':
                if domain=='radio':
                    b=base.get('episode_budget_override_J')
                    if b is None:
                        q=base.get('period_quota_override_J')
                        if q is None:q=base['groups']*base['quota_power_per_terminal_W']*base['packet_slot_s']
                        b=base['rounds']*q
                    physical['episode_budget_override_J']=float(b)*level
                else:
                    b=base['initial_budget']
                    if b is None:b=base['horizon']*base['fixed_period_budget']
                    physical['initial_budget']=float(b)*level
            elif factor=='persistence':physical['global_stay_probability']=level
            elif factor=='arrival_permutation':
                if base['horizon']!=24:raise ValueError('Registered temporal permutation is specifically T=24')
                permutation=list(reversed(range(23))) if level=='front' else list(range(4))+list(range(8,12))+list(range(16,23))+list(range(12,16))+list(range(4,8))
                if sorted(permutation)!=list(range(23)):raise AssertionError('Not a complete permutation')
            payload=dict(schema='allocation23-completion-physical-condition-v1',
                metadata=dict(domain=domain,chapter=3,condition_id=key,family=factor or 'default',level=str(level or 'default')),
                physical_config=physical,invariants=dict(same_objects=True,same_horizon=True,same_source_physics=True,
                    same_reward=True,full_spend_required=True,policy_visible_future=False))
            path=root/f'{domain}__{key}.json';write(path,payload)
            records[domain][key]=dict(physical_condition_file=str(path),physical_condition_sha256=sha(path),
                base_physical=base,physical=physical,changed_fields=[k for k in physical if physical[k]!=base[k]],
                arrival_permutation=permutation,factor=factor or 'default',level=level)
    write(root/'registry.json',dict(schema=SCHEMA,domains=records,source_protocols={
        d:dict(path=str(Path(p).resolve()),sha256=sha(p)) for d,p in [('radio',args.radio_protocol),('v5',args.v5_protocol)]},
        final_master=MASTER,paths=128,rule='Frozen weights and learned moments; current task public remaining/B0 and common decoder geometry; no condition label or future event input'))
    return dict(registry=str(root/'registry.json'),sha256=sha(root/'registry.json'))

def install_arrival_permutation(task, permutation):
    """Transform only the private exogenous TaskStream, keeping actor inputs unchanged.

    At reset, realize the original action-independent stream exactly once, then
    deliver those same event vectors in a registered permutation. There are no
    precomputed events in returned observations or neural inputs. Native physics
    calls this stream only for the current event after applying each action.
    """
    import numpy as np
    if task.domain!='v5':raise ValueError('Only V5 has task arrival permutations')
    env=task.env;original=env.reset
    order=np.asarray(permutation,np.int64)
    if sorted(order.tolist())!=list(range(task.spec.horizon-1)):raise ValueError('Invalid arrival permutation')
    class Stream:
        def __init__(self,initial,events):self.initial,self.events,self.next=initial,events,1
        def initial_demand(self):return self.initial.copy()
        def arrivals(self,next_step):
            if next_step!=self.next:raise ValueError('Sequential arrivals required')
            self.next+=1
            return (np.zeros_like(self.initial) if next_step==task.spec.horizon else self.events[next_step-1].copy())
    def reset(self,*,seed=None,options=None):
        observation,info=original(seed=seed,options=options)
        initial=self._stream.initial_demand()
        events=np.stack([self._stream.arrivals(t) for t in range(1,task.spec.horizon)])
        transformed=events[order].copy()
        # Exact multiset equality, independent of floating-point summation order.
        if not np.array_equal(np.sort(events,axis=0),np.sort(transformed,axis=0)):
            raise ArithmeticError('Permutation changed exogenous event values')
        total=np.asarray([math.fsum([float(initial[i]),*map(float,events[:,i])]) for i in range(len(initial))])
        self._conditions_stream_audit=dict(original_events_sha256=hashlib.sha256(events.tobytes()).hexdigest(),
            permuted_events_sha256=hashlib.sha256(transformed.tobytes()).hexdigest(),
            event_multiset_sha256=hashlib.sha256(np.sort(events,axis=0).tobytes()).hexdigest(),
            initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),expected_total=total,
            exact_event_multiset_preserved=True)
        self._stream=Stream(initial,transformed)
        return observation,info
    env.reset=types.MethodType(reset,env)

def initialize(args):
    output=Path(args.output).resolve()
    if not output.is_relative_to(Path('/hy-tmp')):raise ValueError('Outputs must physically reside on HY')
    output.mkdir(parents=True,exist_ok=False)
    for k,n in [('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg'),('TORCH_HOME','torch'),('CUDA_CACHE_PATH','cuda')]:
        p=output/'runtime'/n;p.mkdir(parents=True);os.environ[k]=str(p)
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    os.environ['CH3_REPAIR_SPATIAL_DECODER']='proportional'
    os.environ['ALLOCATION23_DATA_ROOT']=str(Path(args.data_root).resolve())
    sys.path.insert(0,str(Path(args.source_root).resolve()))
    import torch
    torch.set_num_threads(1)
    return output

def verify_inputs(run,source_identity):
    protocol=json.loads((run/'protocol.json').read_text());done=json.loads((run/'completed.json').read_text())
    if not done.get('success') or done.get('source_unchanged') is not True:raise ValueError('Completed successful source-verified training endpoint required')
    if protocol['schema']==NATIVE_SCHEMA:
        if protocol['source_identity']['package']!=source_identity:raise ValueError('Native frozen package identity differs')
        if done.get('source_identity')!=protocol['source_identity']:raise ValueError('Native completion source identity differs')
        model=run/done.get('model','model/state.pt')
        if sha(model)!=done['model_sha256']:raise ValueError('Native endpoint digest differs')
        import torch
        payload=torch.load(model,map_location='cpu',weights_only=False)
        if payload['schema']!=protocol['schema'] or payload['identity']!=protocol:raise ValueError('Native payload/protocol differs')
        if payload['environment_steps']!=done['training_steps']:raise ValueError('Training counters differ')
        return protocol,done,payload,True
    raise ValueError('Only this scope double-random native endpoints are supported')
    if protocol['schema']!='chapter3-integrated-online-candidates-v1':raise ValueError('Unsupported training endpoint schema')
    if protocol['source_sha256']!=source_identity:raise ValueError('External frozen source identity differs')
    if sha(run/'protocol.json')!=done['protocol_sha256']:raise ValueError('External training protocol digest differs')
    for name,entry in done['model_files'].items():
        p=(run/'model'/name).resolve()
        if not p.is_relative_to((run/'model').resolve()) or sha(p)!=entry['sha256']:raise ValueError('External fixed endpoint digest differs')
    return protocol,done,None,False

class Rows:
    def __init__(self,path,fields):
        self.f=gzip.open(path,'wt',newline='');self.w=csv.DictWriter(self.f,fields);self.w.writeheader()
    def add(self,row):self.w.writerow(row)
    def close(self):self.f.close()

def evaluate(args):
    if not args.engineering_gate:
        if not args.endpoint_seal:raise ValueError('All40 frozen endpoint seal required before opening final panel')
        seal=json.loads(Path(args.endpoint_seal).read_text())
        if seal.get('status')!='PASS' or len(seal.get('endpoints',[]))!=40:raise ValueError('Full40 endpoint seal is not complete')
        keys=set()
        for entry in seal['endpoints']:
            runpath=Path(entry['run']).resolve();d=json.loads((runpath/'completed.json').read_text());p=json.loads((runpath/'protocol.json').read_text())
            key=(p['method'],p['domain'],p['arguments']['seed']);keys.add(key)
            if d.get('success') is not True or d['training_steps']!=100608 or p['arguments']['final_paths']!=0 or p['arguments'].get('engineering_gate') or sha(runpath/d['model'])!=entry['model_sha256'] or entry['model_sha256']!=d['model_sha256']:raise ValueError('Unsealed endpoint changed')
        if keys!={(a,b,c) for a in ('A_FULL','A_PACE','A_BUDGET_ONLY','A_DIRECT_TOTAL') for b in ('radio','v5') for c in range(5)}:raise ValueError('Endpoint seal omits registered arms/domain/seeds')
    output=initialize(args)
    import numpy as np
    import torch
    from allocation23.io import source_hashes,state_hash
    from allocation23.task import Task
    from allocation23.full_spend import full_spend_bounds,assert_terminal_full_spend,terminal_budget_tolerance
    from allocation23.chapter3_candidates.runner import policy_identity,preserved_rng,action_of
    sources=source_hashes();run=Path(args.run).resolve()
    evaluation_phase='engineering' if args.engineering_gate else 'final'
    protocol,done,payload,is_native=verify_inputs(run,sources)
    training_done=done
    if not args.engineering_gate and (done['training_steps']!=100608 or
            protocol['arguments'].get('engineering_gate',False)):
        raise ValueError('Formal frozen evaluation requires a formal 100608-step endpoint')
    domain=protocol.get('domain',protocol['arguments'].get('domain'))
    method=protocol.get('method',protocol['arguments'].get('method'))
    seed=protocol['arguments']['seed']
    registry_path=Path(args.registry).resolve();registry=json.loads(registry_path.read_text())
    if registry['schema']!=SCHEMA or args.path_master!=registry['final_master']:raise ValueError('Unregistered evaluation namespace')
    rec=registry['domains'][domain][args.condition]
    if physical_of(protocol)!=rec['base_physical']:raise ValueError('Training base physics differ from registry')
    condition_file=Path(rec['physical_condition_file'])
    if sha(condition_file)!=rec['physical_condition_sha256']:raise ValueError('Registered condition changed')
    # Default and temporal replay leave physical-condition metadata unchanged.
    target_condition=None if rec['factor'] in ('default','arrival_permutation') else str(condition_file)
    if args.paths<1:raise ValueError('Positive path count required')
    if not args.engineering_gate and args.paths!=registry['paths']:raise ValueError('Formal unit requires all 128 preregistered paths')
    if args.path_master==protocol['arguments']['path_master']:raise ValueError('Training and final namespaces must differ')
    data_root=Path(args.data_root).resolve()
    training_data=protocol.get('data') or protocol.get('data_identity')
    if domain=='radio':
        for name,file in [('images_sha256','images.npy'),('split_manifest_sha256','split_manifest.json')]:
            if sha(data_root/file)!=training_data[name]:raise ValueError('Radio input data identity differs')
        from allocation23.chapter3_baselines.execution_cache import install
        digest_cache=install(data_root/'images.npy')
    else:digest_cache=None
    external_env=None
    if is_native:
        task=Task(domain,3,'A',seed,path_master=args.path_master,data_root=data_root,holdout=not args.engineering_gate,condition_file=target_condition)
        if protocol['spec']['groups']!=task.spec.groups or protocol['spec']['horizon']!=task.spec.horizon:raise ValueError('Geometry dimension changed')
        if method=='A_DIRECT_TOTAL':
            if not args.direct_module:raise ValueError('Registered direct module required')
            mpath=Path(args.direct_module).resolve();spec=importlib.util.spec_from_file_location('registered_ch3_direct_policy',mpath)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            actor=module.load_policy(run,task,device='cpu')
        else:
            from allocation23.methods import make_actor
            actor=make_actor(task.spec,'A',3,seed,**protocol['arm_mapping'])
            actor.load_state_dict(payload['actor'],strict=True)
        actor.eval();before_identity=state_hash(actor);before_accounting=None
    else:
        from allocation23.chapter3_candidates import load
        from allocation23.chapter3_baselines.environment import PublicBudgetEnv
        external_env=PublicBudgetEnv(domain,seed=seed,phase='final',path_master=args.path_master,holdout=not args.engineering_gate,condition_file=target_condition)
        task=external_env.bridge.task
        actual=external_env.contract();trained=protocol['environment']
        for field in ('n_objects','horizon','action_dim'):
            if actual['native_contract'][field]!=trained['native_contract'][field]:raise ValueError('External geometry dimension changed')
        if actual['neural_observation_dim']!=trained['neural_observation_dim']:raise ValueError('Policy input dimension changed')
        with preserved_rng():actor=load(method,run/'model',device=args.device)
        before_identity=policy_identity(actor);before_accounting=dict(actor.accounting())
    if task.physical_dict!=rec['physical']:raise ValueError('Constructed target physics differ from registry')
    if rec['arrival_permutation'] is not None:install_arrival_permutation(task,rec['arrival_permutation'])
    evaluation_protocol=dict(schema=SCHEMA,run=str(run),method=method,domain=domain,seed=seed,condition=args.condition,
        engineering_gate=args.engineering_gate,evaluation_phase=evaluation_phase,test_split_opened=not args.engineering_gate,paths=args.paths,path_master=args.path_master,
        training_protocol_sha256=sha(run/'protocol.json'),training_completion_sha256=sha(run/'completed.json'),
        training_source_identity=sources,training_model_files=done.get('model_files_sha256',done.get('model_files')),
        registry_sha256=sha(registry_path),condition_registration=rec,target_spec=task.spec.to_dict(),target_physical=task.physical_dict,
        weights_and_learned_moments_frozen=True,learning_environment_steps=0,
        current_task_normalization='public phase=t/T; remaining/B0 use target task values; learned running moments frozen',
        policy_input_contract='native actor receives exactly local/global/caps/budget/remaining public encoding; external network receives only flattened local/global',
        private_stream_not_in_policy_input=True,
        no_future_policy_input=True,execution_common_to_all_methods=True,
        worker_sha256=sha(__file__),direct_module_sha256=sha(args.direct_module) if args.direct_module and is_native and method=='A_DIRECT_TOTAL' else None,
        endpoint_seal_sha256=sha(args.endpoint_seal) if args.endpoint_seal else None,
        captured_paths=list(range(args.paths)),storage='compressed scalar CSV and public trajectory NPZ; no models/optimizers/replay checkpoint')
    write(output/'protocol.json',evaluation_protocol)
    step_fields=['path_index','path_seed','step','native_reward','q','q_fraction_initial','R_before','R_after','L','U','q_reference_pace',
        'interval_forced','lower_bound_hit','upper_bound_hit','feasible','action_roundoff','positive_allocations','sum_current_backlog',
        'sum_demand_seen','interference_multiplier','confirmed_packets_before','new_confirmed_packets','useful_resource','ineffective_resource','new_arrivals']
    ep_fields=['path_index','path_seed','native_return','score_x100','steps','total_used','resource_fraction','remaining','replay_error','actions_sha256','trace_sha256',
        'last_quarter_resource_fraction','forced_steps','lower_hits_nonforced','upper_hits_nonforced','mean_confirmed_packet_fraction',
        'total_demand','total_completed','total_ineffective','ineffective_fraction','exact_event_multiset_preserved','stream_total_max_error',
        'original_events_sha256','permuted_events_sha256','event_multiset_sha256','initial_sha256']
    sw=Rows(output/'evaluation_steps.csv.gz',step_fields);ew=Rows(output/'evaluation_paths.csv.gz',ep_fields)
    summaries=[];captures={};terminal_arrays={};start=time.monotonic()
    try:
        with preserved_rng(),torch.no_grad():
            for index in range(args.paths):
                if is_native:
                    obs,_=task.reset(evaluation_phase,index);encoder=task.encoder();encoder.reset(obs);flat=None
                else:
                    flat,_=external_env.reset(seed=index);obs=external_env.bridge.native_obs;encoder=external_env.bridge._encoder
                observations=[];actions=[];rewards=[];last_info={};forced=lohit=uhit=0
                plan=task.plan(evaluation_phase,index);path_seed=plan['task_seed']
                with contextlib.ExitStack() as scope:
                    if hasattr(actor,'evaluation_scope'):scope.enter_context(actor.evaluation_scope(common_seed=args.path_master+8800003+index))
                    for t in range(task.spec.horizon):
                        observations.append({k:np.asarray(v).copy() for k,v in obs.items()})
                        state=encoder.state()
                        if set(state)!={'local','global','caps','budget','remaining'}:raise ArithmeticError('Unregistered policy/decoder input field')
                        L,U=full_spend_bounds(state,task.spec);R=float(state['remaining'])
                        qref=min(U,max(L,R/(task.spec.horizon-t)));tol=terminal_budget_tolerance(task.spec)
                        isforced=abs(U-L)<=tol;forced+=int(isforced)
                        if is_native:
                            sampled=actor.sample(state,deterministic=True)
                            action,delta=task.validate(sampled['actions'][0],state)
                            nxt,reward,done,truncated,last_info=task.env.step(action);encoder.update(action,nxt)
                        else:
                            control=action_of(actor,flat,True,external_env.action_space.shape[0])
                            flat,reward,done,truncated,info=external_env.step(control)
                            action=np.asarray(info['physical_action']);delta=0.;nxt=external_env.bridge.native_obs
                            last_info=external_env.bridge._terminal_info
                        if truncated or done!=(t==task.spec.horizon-1):raise ArithmeticError('Native termination changed')
                        q=float(action.sum());after=float(encoder.state()['remaining'])
                        if q<L-tol or q>U+tol:raise ArithmeticError('Shared total interval violation')
                        lower=abs(q-L)<=tol;upper=abs(q-U)<=tol
                        lohit+=int(lower and not isforced);uhit+=int(upper and not isforced)
                        row={k:'' for k in step_fields};row.update(path_index=index,path_seed=path_seed,step=t,native_reward=float(reward),q=q,
                            q_fraction_initial=q/task.spec.initial_budget,R_before=R,R_after=after,L=L,U=U,q_reference_pace=qref,
                            interval_forced=int(isforced),lower_bound_hit=int(lower),upper_bound_hit=int(upper),feasible=1,action_roundoff=float(delta),
                            positive_allocations=int(np.count_nonzero(action)))
                        if domain=='v5':
                            native=task.env.native_observation_for_audit
                            useful=float((native['last_completed']/task.env._resolved.conversion).sum())
                            row.update(sum_current_backlog=float(obs['remaining_demand'].sum()),sum_demand_seen=float(obs['total_demand_so_far'].sum()),
                                useful_resource=useful,ineffective_resource=float(np.maximum(action-native['last_completed']/task.env._resolved.conversion,0.).sum()),
                                new_arrivals=float(native['last_arrivals'].sum()))
                        else:row.update(interference_multiplier=float(obs['interference_multiplier']),
                            confirmed_packets_before=int(obs['confirmed_packets'].sum()),new_confirmed_packets=int(nxt['confirmed_packets'].sum()-obs['confirmed_packets'].sum()))
                        sw.add(row);actions.append(action.copy());rewards.append(float(reward));obs=nxt
                state=encoder.state();assert_terminal_full_spend(state['remaining'],task.spec)
                actions=np.asarray(actions);native_return=float(sum(rewards))
                result=dict(actions=actions,rewards=np.asarray(rewards),observations=observations,native_return=native_return)
                replay_return,trace=task.replay(evaluation_phase,index,result)
                if replay_return!=native_return:raise ArithmeticError('Native exact replay failed')
                row={k:'' for k in ep_fields};row.update(path_index=index,path_seed=path_seed,native_return=native_return,score_x100=100*native_return,
                    steps=task.spec.horizon,total_used=float(actions.sum()),resource_fraction=float(actions.sum()/task.spec.initial_budget),
                    remaining=float(state['remaining']),replay_error=0.,actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),trace_sha256=trace,
                    last_quarter_resource_fraction=float(actions[-math.ceil(task.spec.horizon/4):].sum()/task.spec.initial_budget),
                    forced_steps=forced,lower_hits_nonforced=lohit,upper_hits_nonforced=uhit)
                if abs(row['resource_fraction']-1)>1e-10:raise ArithmeticError('Full-spend fraction failed')
                if domain=='v5':
                    row.update(total_demand=float(last_info['total_arrived'].sum()),total_completed=float(last_info['total_completed'].sum()),
                        total_ineffective=float(last_info['total_ineffective']),ineffective_fraction=float(last_info['total_ineffective']/task.spec.initial_budget))
                    for k in ('per_region_completion','total_arrived','total_completed','remaining_demand','allocated_resource','ineffective_resource'):
                        terminal_arrays.setdefault(k,[]).append(np.asarray(last_info[k]).copy())
                    if rec['arrival_permutation'] is not None:
                        a=task.env._conditions_stream_audit;err=float(np.max(np.abs(last_info['total_arrived']-a['expected_total'])))
                        if err>1e-8:raise ArithmeticError('Arrival permutation changed per-scenario total demand')
                        row.update(exact_event_multiset_preserved=1,stream_total_max_error=err,
                            **{k:a[k] for k in ('original_events_sha256','permuted_events_sha256','event_multiset_sha256','initial_sha256')})
                else:
                    row.update(mean_confirmed_packet_fraction=float(obs['confirmed_packets'].mean()/task.spec.source_packets))
                    terminal_arrays.setdefault('confirmed_packets',[]).append(obs['confirmed_packets'].copy())
                    terminal_arrays.setdefault('terminal_ssim',[]).append(task.env._terminal_ssim.copy())
                ew.add(row);summaries.append(row)
                observations.append({k:np.asarray(v).copy() for k,v in obs.items()})
                for key in observations[0]:captures.setdefault('public__'+key,[]).append(np.stack([o[key] for o in observations]))
                captures.setdefault('physical_actions',[]).append(actions)
                write(output/'progress.json',dict(schema=SCHEMA,status='evaluating',completed_paths=index+1,total_paths=args.paths,
                    learning_environment_steps=0,evaluation_environment_steps=(index+1)*task.spec.horizon,native_replay_environment_steps=(index+1)*task.spec.horizon))
        if is_native:
            if state_hash(actor)!=before_identity:raise ArithmeticError('Evaluation mutated native actor')
        elif policy_identity(actor)!=before_identity or dict(actor.accounting())!=before_accounting:raise ArithmeticError('Evaluation mutated policy, moments, RNG or learner counters')
        if source_hashes()!=sources or sha(registry_path)!=evaluation_protocol['registry_sha256'] or sha(condition_file)!=rec['physical_condition_sha256']:
            raise ArithmeticError('Registered scientific inputs changed during evaluation')
        if sha(__file__)!=evaluation_protocol['worker_sha256']:
            raise ArithmeticError('Condition evaluator changed during execution')
        if args.endpoint_seal and sha(args.endpoint_seal)!=evaluation_protocol['endpoint_seal_sha256']:raise ArithmeticError('Endpoint seal changed during evaluation')
        if sha(run/'protocol.json')!=evaluation_protocol['training_protocol_sha256'] or sha(run/'completed.json')!=evaluation_protocol['training_completion_sha256'] or sha(run/training_done['model'])!=training_done['model_sha256']:raise ArithmeticError('Frozen endpoint identity changed during evaluation')
        if evaluation_protocol['direct_module_sha256'] is not None and sha(args.direct_module)!=evaluation_protocol['direct_module_sha256']:
            raise ArithmeticError('Registered Direct adapter changed during execution')
        sw.close();ew.close()
        np.savez_compressed(output/'trajectories.npz',**{k:np.asarray(v) for k,v in captures.items()},
            path_indices=np.arange(args.paths,dtype=np.int64),path_seeds=np.asarray([r['path_seed'] for r in summaries],np.uint64))
        np.savez_compressed(output/'terminal_diagnostics.npz',**{k:np.asarray(v) for k,v in terminal_arrays.items()})
        for p in output.glob('*.gz'):
            with gzip.open(p,'rb') as f:
                for _ in iter(lambda:f.read(1024**2),b''):pass
        scores=np.asarray([r['native_return'] for r in summaries])
        completion=dict(schema=SCHEMA,success=True,method=method,domain=domain,seed=seed,condition=args.condition,
            engineering_gate=args.engineering_gate,paths=args.paths,native_mean=float(scores.mean()),native_path_sd=float(scores.std(ddof=1)) if args.paths>1 else 0.,
            learning_environment_steps=0,evaluation_environment_steps=args.paths*task.spec.horizon,native_replay_environment_steps=args.paths*task.spec.horizon,
            elapsed_seconds=time.monotonic()-start,all_native_replays_exact=True,all_full_spend=True,
            policy_and_normalization_unchanged=True,source_unchanged=True,protocol_sha256=sha(output/'protocol.json'),
            input_final_verification=digest_cache.verify_final() if digest_cache else None,
            artifacts_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.glob('*')) if p.is_file() and p.name not in ('progress.json','completed.json','failed.json')})
        write(output/'completed.json',completion)
        return completion
    finally:
        sw.close();ew.close()
        if external_env:external_env.close()
        else:task.env.close()

def parser():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    r=s.add_parser('register');r.add_argument('--radio-protocol',required=True);r.add_argument('--v5-protocol',required=True);r.add_argument('--output',required=True)
    e=s.add_parser('evaluate')
    for name in ('source-root','registry','condition','data-root','output'):e.add_argument('--'+name,required=True)
    e.add_argument('--run','--model-run',dest='run',required=True)
    e.add_argument('--direct-module');e.add_argument('--paths','--cases',dest='paths',type=int,default=128);e.add_argument('--path-master','--master',dest='path_master',type=int,default=MASTER)
    e.add_argument('--device',choices=('cpu','cuda'),default='cpu');e.add_argument('--engineering-gate',action='store_true');e.add_argument('--endpoint-seal')
    return p

def main():
    args=parser().parse_args()
    try:result=register(args) if args.command=='register' else evaluate(args)
    except BaseException as exc:
        import traceback
        if getattr(args,'output',None) and Path(args.output).is_dir():write(Path(args.output)/'failed.json',dict(schema=SCHEMA,success=False,error=repr(exc),traceback=traceback.format_exc()))
        raise
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':main()
