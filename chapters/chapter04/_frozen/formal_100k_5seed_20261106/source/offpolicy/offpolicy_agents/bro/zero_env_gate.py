"""No Env calls: native BRO CPU/CUDA state, counter and endpoint gate.

The reset counter fixture does not pretend to execute 25002 updates or
15001 physical steps.  Actual synthetic computation is accounted separately.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.device.startswith('cpu'):
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['JAX_PLATFORMS']='cpu'
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false')
    os.environ.setdefault('JAX_ENABLE_COMPILATION_CACHE','false')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from replacement_candidates.bro.agent import Agent,MODEL_NAMES
    from replacement_candidates.bro.native_loader import load_native
    load_native()
    from jaxrl.bro import critic as critic_module
    rows=[];start=time.perf_counter()

    def state(native):
        return {'rng':native.rng,'step':native.step,**{k:getattr(native,k) for k in MODEL_NAMES}}

    def exact(a,b):
        # Compare all learned params, Adam/AdamW slots and native PRNG, not logits alone.
        la,_=jax.tree_util.tree_flatten_with_path(a);lb,_=jax.tree_util.tree_flatten_with_path(b)
        if len(la)!=len(lb):raise AssertionError('native state leaf counts differ')
        for i,((pa,x),(pb,y)) in enumerate(zip(la,lb)):
            if repr(pa)!=repr(pb):raise AssertionError('native state paths differ')
            if not np.array_equal(np.asarray(x),np.asarray(y)):
                raise AssertionError('native state mismatch leaf '+str(i))
        return len(la)

    def native_clone(agent):
        c=agent.cfg
        with jax.default_device(agent.device):
            return agent._bro.BRO(c.seed,np.zeros((1,agent.observation_dim),np.float32),
                np.zeros((1,agent.action_dim),np.float32),num_seeds=1,updates_per_step=2)

    for domain,obsdim,actdim in [('radio',305,19),('v5',244,13)]:
        t=time.perf_counter();seed=8930;np.random.seed(seed)
        agent=Agent({'seed':seed,'total_steps':49152},obsdim,actdim,device=args.device)
        reference=native_clone(agent)
        leaf_count=exact(state(agent._native),state(reference))
        rng=np.random.default_rng(seed)
        x=rng.normal(0,.2,obsdim).astype(np.float32)
        initial_rng=agent.get_rng_state();true_action=agent.policy_action(x,deterministic=True)
        assert true_action.shape==(actdim,) and np.isfinite(true_action).all()
        agent.set_rng_state(initial_rng)
        before=agent.native_state_fingerprint();rs=agent.get_rng_state()
        agent.policy_action(x,deterministic=True);agent.policy_action(x,deterministic=False)
        agent.set_rng_state(rs);assert agent.native_state_fingerprint()==before
        for i in range(2499):
            obs=rng.normal(0,.2,obsdim).astype(np.float32)
            action=rng.uniform(-1,1,actdim).astype(np.float32)
            nxt=rng.normal(0,.2,obsdim).astype(np.float32)
            done=(i%32==31);reward=float(rng.random()) if done else 0.
            agent.observe(obs,action,reward,nxt,done)
        assert agent.train_steps==2499 and agent.n_updates==0
        # Full native comparison on the same replay draw at the actual first-update index.
        data=(x,rng.uniform(-1,1,actdim).astype(np.float32),.7,x*.5,True)
        np_state=np.random.get_state();agent.observe(*data);np_after=np.random.get_state()
        np.random.set_state(np_state);batch=agent.replay.sample_parallel_multibatch(128,2);np.random.set_state(np_after)
        with jax.default_device(agent.device):reference.update(batch,2,2500)
        exact(state(agent._native),state(reference))
        assert agent.n_updates==2 and int(agent._native.step)==3
        # Independent NumPy loss and unchanged update under all-terminal next-state perturbation.
        def unseed(model):return jax.tree_util.tree_map(lambda a:a[0],model)
        actor,critic,target,temp=(unseed(getattr(agent._native,k)) for k in ('actor','critic','target_critic','temp'))
        b=agent._Batch(np.asarray(batch.observations)[0,0],np.asarray(batch.actions)[0,0],
            np.asarray(batch.rewards)[0,0],np.zeros(128,np.float32),np.ones(128,np.float32),np.asarray(batch.next_observations)[0,0])
        key=jax.random.PRNGKey(seed+1)
        with jax.default_device(agent.device):
            ca,info=critic_module.update_quantile(key,actor,critic,target,temp,b,.99,0.,agent._native.quantile_taus)
            cb,_=critic_module.update_quantile(key,actor,critic,target,temp,b._replace(next_observations=b.next_observations+1.),.99,0.,agent._native.quantile_taus)
        exact(ca,cb)
        q1,q2=critic(b.observations,b.actions);taus=(np.arange(100,dtype=np.float64)+.5)[None,:,None]/100
        oracle=0.
        for q in (q1,q2):
            err=np.asarray(b.rewards)[:,None,None]-np.asarray(q)[:,:,None]
            huber=np.where(np.abs(err)<=1,.5*err**2,np.abs(err)-.5)
            oracle+=float((np.abs(taus-(err<0))*huber).sum(1).mean())
        np.testing.assert_allclose(float(np.asarray(info['critic_loss'])),oracle,rtol=2e-5,atol=2e-5)
        # Explicit synthetic clock fixture for the ORIGINAL reset-before-update branch.
        agent.train_steps=15000;agent.gradient_iteration_count=25002;agent.updates_since_reset=25002
        agent._native.step=25003;reference.step=25003
        np_state=np.random.get_state();agent.observe(*data);np_after=np.random.get_state()
        np.random.set_state(np_state);batch=agent.replay.sample_parallel_multibatch(128,2);np.random.set_state(np_after)
        with jax.default_device(agent.device):reference.update(batch,2,15001)
        exact(state(agent._native),state(reference))
        assert agent.n_updates==25004 and agent.updates_since_reset==2 and int(agent._native.step)==3
        assert len(agent.reset_events)==1
        ev=agent.reset_events[0]
        assert ev['env_step']==15001 and ev['lifetime_updates_before']==25002 and ev['complete_learning_state_equals_initial']
        assert ev['replay_size']==2501 and ev['replay_sha256_before']==ev['replay_sha256_after']
        with tempfile.TemporaryDirectory(dir=args.output,prefix='inference_fixture_') as td:
            path=Path(td)/'endpoint.pt';agent.save(path);model_sha=hashlib.sha256(path.read_bytes()).hexdigest()
            loaded=Agent.load(path,device=args.device)
            err=max(float(np.max(np.abs(agent.policy_action(o,True)-loaded.policy_action(o,True)))) for o in rng.normal(0,.2,(8,obsdim)).astype(np.float32))
            assert err<2e-6 and loaded.replay is None
            try:loaded.observe(*data)
            except RuntimeError:pass
            else:raise AssertionError('inference model allowed training')
        diag=agent.diagnostics();assert diag['native_jax_backend']==('cpu' if args.device.startswith('cpu') else 'gpu')
        assert all(np.isfinite(v) for k,v in diag.items() if isinstance(v,float))
        rows.append({'domain':domain,'status':'PASS','state_leaves_compared':leaf_count,
            'native_initial_and_two_update_blocks_exact':True,'terminal_next_state_invariance_exact':True,
            'independent_quantile_loss':oracle,'native_quantile_loss':float(np.asarray(info['critic_loss'])),
            'rng_restore_exact':True,'single_inference_endpoint_roundtrip_max_abs_error':err,
            'temporary_endpoint_sha256':model_sha,'temporary_endpoint_removed':True,
            'actual_synthetic_transition_insertions':2501,'actual_full_SAC_iterations':8,
            'actual_standalone_critic_updates':2,'physical_environment_steps':0,
            'synthetic_clock_fixture':{'train_steps_before':15000,'lifetime_updates_before':25002,
                'not_executed_updates':25000,'purpose':'exercise reset branch, never a physical/scientific event'},
            'reset_event':ev,'diagnostics':diag,'wall_seconds':time.perf_counter()-t})
        print(json.dumps({'domain':domain,'status':'PASS','wall_seconds':rows[-1]['wall_seconds']}),flush=True)
    result={'status':'PASS','device':str(jax.devices()),'domains':rows,'new_environment_steps':0,
        'actual_synthetic_transition_insertions':5002,'actual_full_SAC_iterations':16,
        'actual_standalone_critic_updates':4,'synthetic_fixtures_not_actual_training':True,
        'source_manifest_sha256':hashlib.sha256((Path(__file__).parent/'vendor_manifest.json').read_bytes()).hexdigest(),
        'wall_seconds':time.perf_counter()-start}
    (args.output/'zero_env_gate.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','output':str(args.output/'zero_env_gate.json'),'wall_seconds':result['wall_seconds']}))


if __name__=='__main__':main()
