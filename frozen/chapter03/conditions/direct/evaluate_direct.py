"""Frozen Direct endpoint evaluation; no training or checkpoint rewriting."""
from __future__ import annotations
import argparse,csv,gzip,json,os,sys,time
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True)
    p.add_argument('--training-run',required=True)
    p.add_argument('--condition-file')
    p.add_argument('--device',default='cpu')
    p.add_argument('--paths',type=int,default=128)
    p.add_argument('--path-master',type=int,default=202611058199)
    p.add_argument('--phase',choices=('dev','final','engineering'),default='final')
    p.add_argument('--data-root',default=os.environ.get('ALLOCATION23_DATA_ROOT','/hy-tmp/resource_allocation_results/chapter3/full_spend_revision_20261104/execution/shared_data/eurosat_rgb'))
    p.add_argument('--output',required=True)
    args=p.parse_args(argv)
    output=Path(args.output).resolve()
    if not output.is_relative_to(Path('/hy-tmp')) or output.exists() or args.paths<=0:
        raise ValueError('Use new HY output and positive path count')
    output.mkdir(parents=True)
    sys.dont_write_bytecode=True
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    for key in ('TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):
        folder=output/'runtime'/key.lower();folder.mkdir(parents=True);os.environ[key]=str(folder)
    sys.path.insert(0,str(Path(args.source_root).resolve()))
    import numpy as np
    import torch
    from allocation23.io import digest,write_json,state_hash,source_hashes
    from allocation23.full_spend import full_spend_bounds
    from allocation23.task import Task
    from allocation23.training import configure_runtime
    from direct_policy import load_policy,_model_file,SCHEMA
    configure_runtime()
    payload=torch.load(_model_file(args.training_run),map_location='cpu',weights_only=False)
    identity=payload['identity'];seed=identity['arguments']['seed'];domain=identity['domain']
    if source_hashes()!=identity['source_identity']['package']:
        raise ValueError('Direct evaluation native source differs from training snapshot')
    task=Task(domain,3,'A',seed,path_master=args.path_master,data_root=args.data_root,
        holdout=args.phase=='final',condition_file=args.condition_file)
    actor=load_policy(args.training_run,task,args.device)
    initial=state_hash(actor);start=time.perf_counter();scores=[];errors=[];fractions=[]
    fields=['path_index','path_seed','period','remaining_before','lower','upper','period_total',
        'reference_pace','direct_coordinate','native_reward','binding_lower','binding_upper','zero_width']
    step_file=gzip.open(output/'evaluation_steps.csv.gz','wt',newline='')
    path_file=gzip.open(output/'evaluation_paths.csv.gz','wt',newline='')
    steps=csv.DictWriter(step_file,fieldnames=fields);steps.writeheader()
    paths=csv.DictWriter(path_file,fieldnames=['path_index','path_seed','native_return','resource_fraction','replay_error']);paths.writeheader()
    try:
        for i in range(args.paths):
            result=task.episode(args.phase,i,actor,deterministic=True,capture=True)
            replayed,_=task.replay(args.phase,i,result)
            err=abs(replayed-result['native_return']);frac=result['total_used']/task.spec.initial_budget
            if err!=0. or abs(frac-1.)>1e-9:
                raise AssertionError('Frozen Direct replay or full-spend verification failed')
            scores.append(float(result['native_return']));fractions.append(frac);errors.append(err)
            paths.writerow(dict(path_index=i,path_seed=result['plan']['task_seed'],native_return=scores[-1],resource_fraction=frac,replay_error=err))
            encoder=task.encoder();encoder.reset(result['observations'][0]);public=[]
            for t,action in enumerate(result['actions']):
                state=encoder.state();lower,upper=full_spend_bounds(state,task.spec)
                remaining=float(state['remaining']);total=float(action.sum());tol=max(task.spec.action_tolerance,abs(upper)*1e-12)
                with torch.no_grad():coordinate=float(actor.quota_mean(state).reshape(-1)[0])
                steps.writerow(dict(path_index=i,path_seed=result['plan']['task_seed'],period=t,
                    remaining_before=remaining,lower=lower,upper=upper,period_total=total,
                    reference_pace=min(upper,max(lower,remaining/(task.spec.horizon-t))),
                    direct_coordinate=coordinate,native_reward=float(result['rewards'][t]),
                    binding_lower=abs(total-lower)<=tol,binding_upper=abs(total-upper)<=tol,
                    zero_width=upper-lower<=tol))
                public.append({key:np.asarray(value).copy() for key,value in state.items()})
                if t+1<len(result['actions']):encoder.update(action,result['observations'][t+1])
            if i==0:
                arrays=dict(allocations=np.asarray([result['actions']]),rewards=result['rewards'])
                arrays.update({'state_'+key:np.stack([s[key] for s in public]) for key in public[0]})
                np.savez_compressed(output/'allocation_examples.npz',**arrays)
            write_json(output/'progress.json',dict(completed_paths=i+1,target_paths=args.paths,phase='evaluation',pid=os.getpid()))
        if state_hash(actor)!=initial:raise AssertionError('Evaluation changed actor')
    finally:
        step_file.close();path_file.close();task.env.close()
    summary=dict(schema=SCHEMA,success=True,evaluation_only=True,training_run=str(Path(args.training_run).resolve()),
        model_sha256=digest(_model_file(args.training_run)),method='A_DIRECT_TOTAL',domain=domain,seed=seed,
        condition_file=args.condition_file,physical=task.physical_dict,spec=task.spec.to_dict(),
        paths=args.paths,path_master=args.path_master,phase=args.phase,actual_device=str(next(actor.parameters()).device),
        native_mean=float(np.mean(scores)),native_std=float(np.std(scores,ddof=1)) if len(scores)>1 else 0.,
        resource_fraction_mean=float(np.mean(fractions)),replay_max_error=max(errors),
        environment_steps=args.paths*task.spec.horizon,replay_environment_steps=args.paths*task.spec.horizon,
        elapsed_seconds=time.perf_counter()-start,actor_unchanged=True)
    write_json(output/'completed.json',summary)
    print(json.dumps(summary,allow_nan=False),flush=True)

if __name__=='__main__':main()
