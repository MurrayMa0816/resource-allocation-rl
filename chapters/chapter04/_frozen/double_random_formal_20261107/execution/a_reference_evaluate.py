"""Evaluate inherited C3 A endpoints on the new common C4 panels, no training."""
import argparse,csv,gzip,hashlib,json,os,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--domain',choices=['radio','v5'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--master',type=int,required=True)
    p.add_argument('--phase',choices=['dev','final'],required=True);p.add_argument('--cases',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--environment-root',required=True)
    a=p.parse_args()
    out=Path(a.output);assert not out.exists();out.mkdir(parents=True)
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    sys.path.insert(0,a.environment_root)
    from chapter4_full_spend_env.bootstrap import install
    binding=install(audit_dir=None,context={'scope':'double_random_formal_20261107','role':'inherited_A_evaluation'})
    from chapter4_full_spend_env.task import Task
    from chapter4_full_spend_env.native import Actor
    task=Task(a.domain,3,'A',a.seed,path_master=a.master,holdout=a.phase=='final')
    model=Path(a.model);digest=hashlib.sha256(model.read_bytes()).hexdigest()
    payload=torch.load(model,map_location='cpu',weights_only=False)
    assert payload.get('evaluation_only') and payload['environment_steps']==100608
    identity=payload['identity'];assert identity['method']=='A_FULL' and identity['domain']==a.domain
    completion=json.loads(model.parent.parent.joinpath('completed.json').read_text())
    assert completion['success'] and completion['training_steps']==100608 and completion['model_sha256']==digest
    original_protocol=json.loads(model.parent.parent.joinpath('protocol.json').read_text())
    assert original_protocol['initialization_control']['scheme']=='both_random', 'Use new double-random C3 A only'
    assert int(identity['arguments']['seed'])==a.seed
    actor=Actor(task.spec,'A_FULL','adaptive',a.seed,quota_reference='pace').double().cpu().eval()
    actor.load_state_dict(payload['actor'],strict=True);actor.requires_grad_(False)
    protocol={'method':'A_inherited','domain':a.domain,'seed':a.seed,'training_steps':100608,
              'source_model':str(model),'source_model_sha256':digest,'phase':a.phase,'path_master':a.master,
              'cases':a.cases,'deterministic':True,'new_training_steps':0,'environment_binding':binding,
              'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    rows=[];start=time.perf_counter();maxerr=0.;physical=0
    fields=['case','task_seed','native_return','resource_fraction','steps','replay_error','episode_sha256']
    with gzip.open(out/'evaluation_paths.csv.gz','wt',newline='')as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i in range(a.cases):
            result=task.episode(a.phase,i,actor,deterministic=True,capture=True)
            score,trace=task.replay(a.phase,i,result)
            assert score==result['native_return']
            frac=result['total_used']/task.spec.initial_budget;err=abs(frac-1.);maxerr=max(maxerr,err)
            assert err<=1e-9
            row={'case':i,'task_seed':result['plan']['task_seed'],'native_return':score,
                 'resource_fraction':frac,'steps':task.spec.horizon,'replay_error':0.,'episode_sha256':trace}
            w.writerow(row);rows.append(row);physical+=2*task.spec.horizon
            if i==0:np.savez_compressed(out/'allocation_example.npz',actions=result['actions'],rewards=result['rewards'])
            (out/'progress.json').write_text(json.dumps({'phase':a.phase,'cases_complete':i+1,'target_cases':a.cases,
                'physical_environment_steps':physical,'new_training_steps':0,'elapsed_seconds':time.perf_counter()-start})+'\n')
    scores=np.asarray([x['native_return']for x in rows],dtype=np.float64)
    summary={'success':True,'method':'A_inherited','domain':a.domain,'seed':a.seed,'phase':a.phase,
             'cases':a.cases,'mean':float(scores.mean()),'std':float(scores.std(ddof=1)),
             'first8_mean':float(scores[:8].mean()),'new_training_steps':0,'evaluation_steps':a.cases*task.spec.horizon,
             'replay_environment_steps':a.cases*task.spec.horizon,'physical_environment_steps':physical,
             'maximum_resource_fraction_error':maxerr,'strict_reload':True,'deterministic':True,
             'elapsed_seconds':time.perf_counter()-start,'source_model':str(model),'source_model_sha256':digest,
             'worker_sha256':protocol['worker_sha256']}
    assert hashlib.sha256(model.read_bytes()).hexdigest()==digest
    summary['artifacts_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in out.iterdir()
                                if p.is_file()and p.name not in ['progress.json','completed.json']}
    (out/'completed.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    task.env.close();print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
