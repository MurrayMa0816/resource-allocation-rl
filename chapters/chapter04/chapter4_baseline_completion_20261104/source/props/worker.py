"""Complete the exact previously tested PROPS matched_sf adaptation in V5."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREVIOUS=ROOT.parent/'chapter4_full_spend_20261104'
sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def bindings():
    return {str(p):sha(p) for p in [Path(__file__),*[PREVIOUS/'source/communication'/n for n in ['matched_agent.py','runtime_hooks.py','native_loop.py']]]}
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    c=json.loads(a.config.read_text());out=Path(c['output_dir']).resolve();prefix=c.get('engineering_gate',False)
    if not out.is_relative_to(ROOT/'jobs') and not out.is_relative_to(ROOT/'engineering/props'):
        raise ValueError('New scope only')
    if c['domain']!='v5' or c['seed'] not in (730,731) or c['representation_variant']!='matched_sf':raise ValueError('Only registered V5 completion, not another new variant')
    if c['device']!='cpu' or c['torch_threads']!=1 or c['gamma']!=1 or c['update_epochs']!=10 or c['props_update_epochs']!=16:raise ValueError('Exact previously tested native contract required')
    if c['episodes']!=(8 if prefix else 960) or c.get('holdout_paths')!=0:raise ValueError('Registered complete episode budget and no holdout required')
    before=bindings();config_sha=sha(a.config)
    if not prefix:
        gate=ROOT/'engineering/props/v5_prefix8/completion.json'
        g=json.loads(gate.read_text());assert g['status']=='PASS' and g['source_sha256']==before
    if out.exists():raise FileExistsError('Never overwrite or silently resume an attempt')
    out.mkdir(parents=True);start=time.perf_counter()
    try:
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TMPDIR']=str(ROOT/'tmp')
        sys.path.insert(0,str(PREVIOUS/'source/env'))
        from chapter4_full_spend_env.bootstrap import install
        identity=install(audit_dir=out/'resource_audit',context=dict(method='PROPS',variant='structured_sf',new_scope=str(ROOT),engineering_gate=prefix))
        sys.path.insert(0,str(PREVIOUS/'source/communication'))
        import native_loop
        import torch,numpy as np
        result=native_loop.run(c)
        assert result['status']==('smoke_passed' if prefix else 'completed')
        assert result['train_env_steps']==c['episodes']*24
        assert result['mechanism_checks']['checkpoint_roundtrip_exact']
        assert result['mechanism_checks']['nonterminal_reward_count']==0
        assert result['target_updates']==c['episodes']//4 and result['behavior_updates']==c['episodes']
        assert not result['source_receipt']['source_modified'] and not torch.cuda.is_initialized()
        points=[]
        for row in json.loads((out/'evaluations.json').read_text()):
            scores=np.asarray(row['returns'],dtype=float)*100
            points.append(dict(episode=row['train_env_steps']//24,train_steps=row['train_env_steps'],
                scores_x100=scores.tolist(),mean_x100=float(scores.mean()),sd_x100=float(scores.std(ddof=1))))
        if not prefix:assert len(points)==21 and [r['episode'] for r in points]==list(range(0,961,48)) and all(len(r['scores_x100'])==32 for r in points)
        write(out/'evaluation_points.json',points)
        ledgers=[json.loads(s) for f in (out/'resource_audit').glob('resource_episodes_*.jsonl') for s in f.read_text().splitlines()]
        assert sum(x['phase']=='train' for x in ledgers)==c['episodes']
        assert all(x['terminal_full_spend_assertion'] and x['nonterminal_rewards_zero'] and x['relative_allocation_error']<=1e-10 for x in ledgers)
        assert all(x['maximum_negative_allocation_violation']==x['maximum_per_object_cap_violation']==x['maximum_completed_object_allocation']==0 for x in ledgers)
        assert bindings()==before and sha(a.config)==config_sha
        m=out/'checkpoint.pt';assert m.is_file() and len(list(out.glob('*.pt')))==1
        receipt=dict(status='PASS',method='PROPS',domain='v5',seed=c['seed'],variant='structured_sf',
            episodes=c['episodes'],engineering_gate=prefix,training_steps=result['train_env_steps'],
            development_steps=sum(x['evaluation_env_steps'] for x in json.loads((out/'evaluations.json').read_text())),
            holdout_calls=0,source_sha256=before,config_sha256=config_sha,config_path=str(a.config),
            model_path=str(m),model_sha256=sha(m),native_receipt_path=str(out/'result.json'),
            native_update_gate_path=str(ROOT/'engineering/props/v5_prefix8/completion.json'),
            bridge_identity_path=str(out/'resource_audit/bridge_identity.json'),
            actual_native_target_updates=result['target_updates'],actual_native_behavior_updates=result['behavior_updates'],
            complete_full_spend_paths=len(ledgers),max_relative_allocation_error=max(x['relative_allocation_error'] for x in ledgers),
            actor_critic_initialization='Exact previously tested matched_sf: shared-object split actor and zero-output flat value; zero mean,initialsigma quota.8/spatial1sqrtK,all logstdlearnable',
            wall_seconds=time.perf_counter()-start,one_final_model=True,replay_saved=False)
        write(out/'completion.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)
    except BaseException:
        write(out/'failure.json',dict(traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start));raise
if __name__=='__main__':main()
