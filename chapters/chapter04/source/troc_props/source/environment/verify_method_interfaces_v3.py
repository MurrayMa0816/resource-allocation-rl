#!/usr/bin/env python3
"""Untrained, full-width native actor/Env interface check; one method/process.

TD-MPC2 uses the official full-width CPU WorldModel prior, not its CUDA planner.
PGR uses its official 256x256 SAC policy, not a diffusion training/generation loop.
No optimizer step, replay population, checkpoint load/save or holdout is used.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
SCOPE=ROOT.parents[1]
TMP=SCOPE/'tmp/env'
TMP.mkdir(parents=True,exist_ok=True)
os.environ.update(TMPDIR=str(TMP),TMP=str(TMP),TEMP=str(TMP),PYTHONDONTWRITEBYTECODE='1',
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    XDG_CACHE_HOME=str(TMP/'cache'),JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',
    XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1',
    WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES='')
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(1)
from chapter4_full_spend_env.bootstrap import install,FROZEN_C4


def actor_for(method,env):
    dim=env.action_dim;obsdim=env.observation_dim
    if method in ('a','f1'):
        from chapter4_recent_baselines.runners.reference_runner import _native
        refs=_native();actor=refs['make_actor'](env.native_spec,'A',3,730)
        critic=refs['make_value'](env.native_spec,730)
        backend=refs['NativePPO'](env.native_spec,dim,730,'cpu',extra=False) if method=='a' else refs['FrequentPPO'](env.native_spec,dim,730,'cpu',replay=True)
        generator=torch.Generator().manual_seed(39101)
        def act(obs):
            with torch.no_grad():
                result=actor.sample(env.state,generator=generator)
                mu,sigma=refs['gaussian_parameters'](actor,env.state)
                lp=torch.distributions.Normal(mu,sigma).log_prob(result['raw']).sum(-1,keepdim=True)
                assert torch.max(torch.abs(lp-result['log_prob']))<1e-10
                assert torch.isfinite(critic(env.state)).all()
            return result['raw'][0].numpy()
        return act,{'model':'copied C3 A_FULL actor/value; original native A/F1 learner constructed',
                    'learning_backend':backend.metadata(),'full_width':True,'likelihood':'complete raw joint Gaussian'},actor
    if method=='props':
        from chapter4_recent_baselines.runners.props_runner import _load_official
        Agent,_,_=_load_official()
        model=Agent(SimpleNamespace(single_observation_space=env.observation_space,single_action_space=env.action_space))
        def act(obs):
            with torch.no_grad():
                result=model.get_action_and_value(torch.tensor(obs,dtype=torch.float32)[None])
                assert len(result)==6
                for value in result:assert torch.isfinite(value).all()
                lp=torch.distributions.Normal(result[1],result[2]).log_prob(result[0]).sum(1)
                assert torch.equal(lp,result[3])
            return result[0][0].numpy()
        return act,{'model':'author PROPS Agent 64x64, output tanh, learnable diagonal std',
                    'full_width':True,'likelihood':'original six-value API Gaussian'},model
    if method=='addq':
        from chapter4_recent_baselines.runners.addq_runner import _load_official
        ADQRSAC=_load_official()
        model=ADQRSAC('MlpPolicy',env,device='cpu',seed=730,verbose=0,
            buffer_size=2*env.horizon+1,learning_starts=32*env.horizon,batch_size=512,gamma=1.,
            policy_kwargs={'n_quantiles':25,'n_critics':2})
        def act(obs):
            with torch.no_grad():
                obs_tensor=torch.tensor(obs,dtype=torch.float32)[None]
                action,lp=model.actor.action_log_prob(obs_tensor)
                assert torch.isfinite(lp).all() and torch.isfinite(action).all()
                quantiles=model.critic(obs_tensor,action)
                assert torch.isfinite(quantiles).all()
            return action[0].numpy()
        return act,{'model':'author ADDQ full default actor/25-quantile double critic',
                    'full_width':True,'empty_replay_capacity_only':2*env.horizon+1,
                    'likelihood':'author squashed actor action_log_prob'},model.policy
    if method=='fog':
        import jax
        jax.config.update('jax_compilation_cache_dir',str(TMP/'fog_cache'))
        from chapter4_recent_baselines.runners.fog_runner import _native
        Learner,_=_native()
        model=Learner(seed=730,observations=np.zeros((1,obsdim),np.float32),
            actions=np.zeros((1,dim),np.float32),critic_hidden_dims=512,actor_hidden_dims=128,
            num_seeds=1,init_discount=1.,final_discount=1.,max_steps=960*env.horizon*10,bc_weight=1e-3)
        def act(obs):
            action=np.asarray(model.sample_actions(obs[None].astype(np.float32)))[0]
            assert np.isfinite(action).all()
            return action
        return act,{'model':'author OBACLearner actor128/critic512, native JAX sampler',
                    'full_width':True,'learning_updates':0,'device':str(jax.devices()[0])},model
    if method=='tdmpc2':
        from chapter4_recent_baselines.runners.tdmpc2_runner import _config,VENDOR
        sys.path.insert(0,str(VENDOR/'tdmpc2'))
        from common.world_model import WorldModel
        cfg=_config({'episodes':960,'seed':730,'gamma':1.,'compile':False},obsdim,dim,env.horizon)
        model=WorldModel(cfg).cpu().eval()
        def act(obs):
            with torch.no_grad():
                z=model.encode(torch.tensor(obs,dtype=torch.float32)[None],None)
                action,info=model.pi(z,None)
                nxt=model.next(z,action,None)
                assert torch.isfinite(z).all() and torch.isfinite(action).all() and torch.isfinite(nxt).all()
                assert all(torch.isfinite(v).all() for v in info.values())
            return action[0].numpy()
        return act,{'model':'official full-width WorldModel enc256/mlp512/latent512/5Q prior forward',
                    'full_width':True,'CUDA_planner_exercised':False,'learning_updates':0,
                    'scope':'CPU encode/pi/next interface only, not an MPPI/replay/update closed loop'},model
    if method=='pgr':
        vendor=FROZEN_C4/'chapter4_recent_baselines/vendor/pgr/synther/REDQ'
        sys.path.insert(0,str(vendor))
        from redq.algos.core import TanhGaussianPolicy
        model=TanhGaussianPolicy(obsdim,dim,(256,256),action_limit=1.).cpu()
        def act(obs):
            with torch.no_grad():
                result=model(torch.tensor(obs,dtype=torch.float32)[None],deterministic=False,return_log_prob=True)
                assert torch.isfinite(result[0]).all() and torch.isfinite(result[3]).all()
            return result[0][0].numpy()
        return act,{'model':'official PGR REDQ TanhGaussianPolicy 256x256',
                    'full_width':True,'diffusion_training_generation_exercised':False,
                    'scope':'original actor density and Env rollout only, no REDQ/diffusion update'},model
    raise ValueError(method)


def run(method):
    out=ROOT/'acceptance/native_interfaces_v3'/method
    out.mkdir(parents=True,exist_ok=True)
    identity=install(audit_dir=out/'resource_audit',context={'purpose':'native-interface-no-training','method':method})
    from chapter4_recent_baselines.env_adapter import Env
    from chapter4_full_spend_env.native import digest
    rows=[];source_before={}
    for domain in ('radio','v5'):
        mode='raw' if method in ('a','f1','props') else 'tanh_latent'
        env=Env(domain,730,phase='engineering',action_mode=mode)
        obs,info=env.reset(options={'episode_index':17})
        assert info=={} and env.unwrapped is env and env.spec is None
        assert env.observation_space.contains(obs)
        assert env.action_space.shape==(19 if domain=='radio' else 13,)
        action_fn,scope,model=actor_for(method,env)
        module=sys.modules[model.__class__.__module__]
        source=getattr(module,'__file__',None)
        if source:source_before[source]=digest(source)
        widths=[]
        if isinstance(model,torch.nn.Module):
            widths=[(m.in_features,m.out_features) for m in model.modules() if isinstance(m,torch.nn.Linear)]
            state_before={k:v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in model.state_dict().items()}
        else:
            import jax
            state_before=[np.asarray(v).copy() for v in jax.tree_util.tree_leaves(
                (model.actor.params,model.critic.params))]
        returned=[]
        for t in range(env.horizon):
            action=np.asarray(action_fn(obs),dtype=np.float64)
            assert action.shape==env.action_space.shape and np.isfinite(action).all()
            raw=env.raw_action(action)
            if mode=='raw':assert np.array_equal(raw,action)
            else:
                assert np.all(np.abs(action)<=1.)
                assert np.array_equal(raw,np.arctanh(np.clip(action,-1+env.action_epsilon,1-env.action_epsilon)))
            obs,reward,done,truncated,info=env.step(action)
            assert env.observation_space.contains(obs) and info=={} and not truncated
            assert done==(t==env.horizon-1)
            if not done:assert reward==0.
            returned.append(reward)
        ledger=env.task.env.last_full_spend_audit
        assert ledger['terminal_full_spend_assertion'] and ledger['relative_allocation_error']<=1e-10
        if isinstance(model,torch.nn.Module):
            assert all(torch.equal(v,model.state_dict()[k]) if torch.is_tensor(v) else v==model.state_dict()[k] for k,v in state_before.items())
        else:
            state_after=jax.tree_util.tree_leaves((model.actor.params,model.critic.params))
            assert len(state_before)==len(state_after) and all(np.array_equal(a,np.asarray(b))
                for a,b in zip(state_before,state_after))
        rows.append({'method':method,'domain':domain,'status':'PASS','source':source,
            'observation_shape':list(env.observation_space.shape),'action_shape':list(env.action_space.shape),
            'action_mode':mode,'real_environment_steps':env.real_steps,'scope':scope,
            'linear_layer_shapes':widths,'original_native_actor_parameters_unchanged':True,
            'total_used':ledger['total_used'],'remaining':ledger['remaining'],
            'relative_allocation_error':ledger['relative_allocation_error'],
            'nonterminal_rewards_zero':all(x==0 for x in returned[:-1]),'true_terminal':True})
        env.close()
    assert all(digest(path)==sha for path,sha in source_before.items())
    assert not torch.cuda.is_initialized()
    result={'status':'PASS','method':method,'cases':rows,'identity':identity,
        'actual_environment_steps':sum(r['real_environment_steps'] for r in rows),
        'training_updates':0,'holdout_paths':0,'model_objects_loaded_from_checkpoint':0,
        'cuda_initialized':False,'actor_module_source_sha256':source_before,
        'gate_script_sha256':digest(__file__),
        'scope':'Full-width untrained native inference/API gate; not training efficacy or full TD/PGR learning-loop acceptance'}
    (out/'gate.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
    print(json.dumps({'method':method,'status':'PASS','actual_environment_steps':result['actual_environment_steps']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--method',required=True,
        choices=['a','f1','props','fog','addq','tdmpc2','pgr']);args=parser.parse_args()
    try:run(args.method)
    except BaseException:
        out=ROOT/'acceptance/native_interfaces_v3'/args.method;out.mkdir(parents=True,exist_ok=True)
        (out/'failure.json').write_text(json.dumps({'status':'FAIL','traceback':traceback.format_exc()},indent=2)+'\n')
        raise
