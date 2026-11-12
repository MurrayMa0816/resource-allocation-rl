"""Focused real native critic update: nonterminal zero-reward bootstrap remains active."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def main(args):
    out=Path(args.output).resolve()
    if not out.is_relative_to('/hy-tmp'):raise ValueError('HY engineering output required')
    out.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['JAX_PLATFORMS']='cpu';sys.dont_write_bytecode=True
    for name in ['XDG_CACHE_HOME','JAX_COMPILATION_CACHE_DIR','TMPDIR']:
        p=out/'runtime'/name.lower();p.mkdir(parents=True,exist_ok=True);os.environ[name]=str(p)
    sys.path.insert(0,str(HERE.parent.parent))
    from replacement_candidates.mad_td.agent import Agent,native,_hash
    import jax,jax.numpy as jnp
    from mad_td.rl_types import RLBatch
    start=time.perf_counter();seed=19142;np.random.seed(seed)
    agent=Agent(dict(seed=seed,total_steps=6000,domain=args.domain),args.observation_dim,19 if args.domain=='radio'else 13,'cpu')
    m=jax.tree_util.tree_map(lambda x:x[0],agent.models);rng=np.random.RandomState(seed);B=512
    b=RLBatch(state=jnp.asarray(rng.normal(size=(B,args.observation_dim)),jnp.float32),action=jnp.asarray(rng.uniform(-1,1,(B,1,agent.action_dim)),jnp.float32),reward=jnp.zeros((B,1,1)),next_state=jnp.asarray(rng.normal(size=(B,1,args.observation_dim)),jnp.float32),mask=jnp.ones((B,1,1)),idxs=jnp.arange(B)[:,None])
    changed=m.replace(critic_target=m.critic_target.replace(params=jax.tree_util.tree_map(lambda x:x*1.1+.03,m.critic_target.params)))
    fn=jax.jit(native.critic.update_critic,static_argnums=2);key=jax.random.PRNGKey(seed)
    a,_=fn(b,m,agent.hp,key);c,_=fn(b,changed,agent.hp,key);jax.block_until_ready(c)
    active=_hash(a.critic.params)!=_hash(c.critic.params)
    assert active and np.count_nonzero(np.asarray(b.reward))==0 and np.all(np.asarray(b.mask)==1)
    record=dict(schema='madtd-zero-reward-nonterminal-native-bootstrap-gate-v1',status='PASS',domain=args.domain,reward_zero=True,nonterminal_mask_one=True,nextQ_change_changes_actual_online_critic_update=active,engineering_optimizer_calls=dict(critic=2,actor=0,encoder=0,model=0),physical_environment_steps=0,holdout_calls=0,engineering_only=True,model_or_Adam_or_replay_files_saved=False,wall_seconds=time.perf_counter()-start,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [HERE/'agent.py',HERE/'native_loader.py',HERE/'zero_reward_bootstrap_gate.py']})
    (out/'gate.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--domain',choices=['radio','v5'],required=True);p.add_argument('--observation-dim',required=True,type=int);p.add_argument('--output',required=True);main(p.parse_args())
