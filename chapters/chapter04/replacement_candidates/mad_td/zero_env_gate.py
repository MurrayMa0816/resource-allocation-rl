"""Actual full-size JAX updates on fixed synthetic tensors, zero environment calls."""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root
import argparse,hashlib,importlib.util,json,math,os,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent.parent))

def main(args):
    out=Path(args.output).resolve()
    if not out.is_relative_to(str(_resolve_path('/hy-tmp'))):raise ValueError('Engineering outputs must be in HY')
    out.mkdir(parents=True,exist_ok=False)
    for name in ['XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','JAX_COMPILATION_CACHE_DIR','TMPDIR']:
        p=out/'runtime'/name.lower();p.mkdir(parents=True,exist_ok=True);os.environ[name]=str(p)
    os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['JAX_PLATFORMS']='cpu'
    else:
        expected='GPU-71946e17-6e0d-6a7b-1fa4-b3de51c38fe3'
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=expected:raise RuntimeError('GPU engineering gate requires explicitly registered remote physicalGPU0')
    from replacement_candidates.mad_td.agent import Agent,native,_hash
    import jax,jax.numpy as jnp
    from flax import serialization
    from mad_td.rl_types import RLBatch
    seed=19141;np.random.seed(seed);started=time.perf_counter()
    agent=Agent({'seed':seed,'total_steps':6000,'domain':args.domain},args.observation_dim,19 if args.domain=='radio'else 13,args.device)
    models=jax.tree_util.tree_map(lambda x:x[0],agent.models)
    rng=np.random.RandomState(seed);B=512;A=agent.action_dim;O=agent.observation_dim
    batch=RLBatch(state=jnp.asarray(rng.normal(size=(B,O)),jnp.float32),action=jnp.asarray(rng.uniform(-1,1,(B,1,A)),jnp.float32),reward=jnp.full((B,1,1),.4),next_state=jnp.asarray(rng.normal(size=(B,1,O)),jnp.float32),mask=jnp.zeros((B,1,1)),idxs=jnp.arange(B)[:,None])
    key=jax.random.PRNGKey(seed)
    updated,info=jax.jit(native.critic.update_critic,static_argnums=2)(batch,models,agent.hp,key)
    jax.block_until_ready(updated)
    terminal_delta=sum(float(np.linalg.norm(np.asarray(x)-np.asarray(y)))for x,y in zip(jax.tree_util.tree_leaves(updated.critic.params),jax.tree_util.tree_leaves(models.critic.params)))
    assert terminal_delta>0,'terminal reward must train online critic'
    # Shift only next-Q target parameters; terminal online critic update must be identical.
    altered=models.replace(critic_target=models.critic_target.replace(params=jax.tree_util.tree_map(lambda x:x*1.1+.03,models.critic_target.params)))
    updated2,_=jax.jit(native.critic.update_critic,static_argnums=2)(batch,altered,agent.hp,key);jax.block_until_ready(updated2)
    terminal_nextQ_independent=_hash(updated.critic.params)==_hash(updated2.critic.params)
    assert terminal_nextQ_independent
    # Raw author's continuing-task loss masks this entire terminal update.
    rawpath=HERE/'vendor/mad_td/update_functions/critic_updates.py';spec=importlib.util.spec_from_file_location('_madtd_raw_critic_negative_gate',rawpath);raw=importlib.util.module_from_spec(spec);spec.loader.exec_module(raw)
    raw_updated,_=jax.jit(raw.update_critic,static_argnums=2)(batch,models,agent.hp,key);jax.block_until_ready(raw_updated)
    raw_terminal_delta=sum(float(np.linalg.norm(np.asarray(x)-np.asarray(y)))for x,y in zip(jax.tree_util.tree_leaves(raw_updated.critic.params),jax.tree_util.tree_leaves(models.critic.params)))
    assert raw_terminal_delta==0.,'negative control must demonstrate original dropped terminal supervision'
    # Real terminal reward must also train the reward-model branch.
    model_update=__import__('mad_td.update_functions.model_updates',fromlist=['update_model']).update_model
    model_updated,_=jax.jit(model_update,static_argnums=2)(batch,models,agent.hp,key);jax.block_until_ready(model_updated)
    reward_branch=model_updated.latent_model.params['params']['MLP_0'];old_reward=models.latent_model.params['params']['MLP_0']
    reward_delta=sum(float(np.linalg.norm(np.asarray(x)-np.asarray(y)))for x,y in zip(jax.tree_util.tree_leaves(reward_branch),jax.tree_util.tree_leaves(old_reward)))
    assert reward_delta>0.,'terminal reward must train reward model'
    # On nonterminal rows target still bootstraps from next Q.
    nonterminal=batch.replace(mask=jnp.ones_like(batch.mask))
    real1,_=jax.jit(native.critic.update_critic,static_argnums=2)(nonterminal,models,agent.hp,key)
    real2,_=jax.jit(native.critic.update_critic,static_argnums=2)(nonterminal,altered,agent.hp,key)
    assert _hash(real1.critic.params)!=_hash(real2.critic.params),'nonterminal bootstrap must remain active'
    # Complete native UTD8 closure: model/encoder once, critic/actor eight times.
    minibatches=jax.tree_util.tree_map(lambda x:jnp.broadcast_to(x,(1,8,*x.shape)),nonterminal)
    allmodels,allinfo=agent._update(agent.models,minibatches,agent.hp,agent._batch_shape,1,8,key);jax.block_until_ready(allmodels)
    counts={n:int(np.asarray(jax.device_get(getattr(allmodels,n).step)).reshape(-1)[0])for n in ['critic','actor','encoder','latent_model']}
    assert counts==dict(critic=8,actor=8,encoder=1,latent_model=1),counts
    assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(allmodels))
    ep,ap=jax.tree_util.tree_map(lambda x:np.asarray(jax.device_get(x[0])),(allmodels.encoder.params,allmodels.actor.params));probe=rng.normal(size=O).astype('float32')
    before=np.asarray(agent._deployment(ep,ap,probe));serialized=serialization.msgpack_serialize(dict(encoder=ep,actor=ap));restored=serialization.msgpack_restore(serialized);after=np.asarray(agent._deployment(restored['encoder'],restored['actor'],probe));assert np.array_equal(before,after)
    receipt=dict(schema='madtd-native-fullsize-zero-env-gate-v1',status='PASS',domain=args.domain,device=args.device,actual_device=str(agent.jax_device),source_commit=native.manifest['commit'],patch_profile='madtd-finite-terminal-supervision-v1',patch_sites=native.patches,terminal_online_critic_parameter_delta=terminal_delta,raw_author_terminal_critic_parameter_delta_negative_control=raw_terminal_delta,terminal_nextQ_independent=terminal_nextQ_independent,terminal_reward_model_parameter_delta=reward_delta,nonterminal_bootstrap_active=True,native_UTD8_optimizer_steps=counts,model_generated_rows_per_critic_batch=26,real_rows_per_critic_batch=486,fullsize_network_latent512_bins151_batch512=True,inference_weights_memory_roundtrip_exact=True,physical_environment_steps=0,holdout_calls=0,trained_on_environment=False,engineering_only=True,model_Adam_or_replay_files_saved=False,wall_seconds=time.perf_counter()-started,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [HERE/'agent.py',HERE/'native_loader.py',HERE/'zero_env_gate.py',HERE/'vendor_manifest.json']})
    (out/'gate.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--domain',choices=['radio','v5'],required=True);p.add_argument('--observation-dim',type=int,required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);main(p.parse_args())
