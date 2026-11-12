"""Full native BRO: regularized residual critic plus optimistic exploration."""
from __future__ import annotations

import pickle
import subprocess
import sys
import time

import numpy as np

from ..common import ROOT, RunLog, array_digest, configure_jax, evaluate


def run(config):
    configure_jax(config.get('device','cuda'))
    sys.path.insert(0, str(ROOT / 'vendor/bro'))
    import jax
    import jaxlib
    import jax.numpy as jnp
    from flax import serialization
    from jaxrl.bro.bro_learner import BRO
    from jaxrl.replay_buffer import ParallelReplayBuffer
    from ..env_adapter import Env
    np.random.seed(config['seed'])
    env = Env(config['domain'], config['seed'], action_mode='tanh_latent')
    h = env.horizon
    smoke = config.get('smoke',False)
    total = int(config.get('total_steps',int(config.get('episodes',2 if smoke else 960))*h))
    warm = int(config.get('learning_starts',h if smoke else 32*h))
    utd = int(config.get('updates_per_step',10))
    batch = int(config.get('batch_size',16 if smoke else 128))
    if total <= warm or total % h:
        raise ValueError('BRO needs whole episodes and total_steps > learning_starts')
    if utd < 1 or batch < 1:
        raise ValueError('BRO updates_per_step and batch_size must be positive')
    agent = BRO(config['seed'], np.zeros((1,env.observation_space.shape[0]),np.float32),
                np.zeros((1,env.action_space.shape[0]),np.float32),num_seeds=1,
                discount=float(config.get('gamma',1.)), updates_per_step=utd,
                distributional=True, n_quantiles=100)
    replay = ParallelReplayBuffer(env.observation_space,env.action_space.shape[-1],
                                  int(config.get('buffer_size',100_000)),num_seeds=1)
    log = RunLog(config,'bro')
    components=('actor','actor_o','critic','target_critic','temp','optimism','regularizer')
    before = {key:array_digest(jax.tree_util.tree_leaves(getattr(agent,key).params))
              for key in components}
    obs,_ = env.reset()
    updates,returns = 0,[]
    nonterminal_reward_count=0
    terminal_feedback_count=0
    optimistic_collection_steps=0
    eval_env_steps=0
    last_info={}
    action_fn = lambda observation, deterministic=False: agent.sample_actions(
        observation[None], temperature=0. if deterministic else 1.)[0]
    def evaluate_policy():
        rng=agent.rng
        agent.rng=jax.random.split(jax.random.PRNGKey(config['seed']+4_000_000),1)
        try:
            return evaluate(action_fn,config['domain'],config['seed'],config.get('eval_episodes',32))
        finally:
            agent.rng=rng
    if not smoke:
        initial_evaluation=evaluate_policy()
        eval_env_steps+=initial_evaluation['evaluation_steps']
        log.append(real_steps=0,critic_updates=0,actor_updates=0,evaluation=initial_evaluation)
    for step in range(1,total+1):
        action = env.action_space.sample() if step <= warm else agent.sample_actions_o(obs[None])[0]
        optimistic_collection_steps += int(step > warm)
        nxt,reward,terminated,truncated,_ = env.step(action)
        done=bool(terminated or truncated)
        if not done and float(reward)!=0.:
            nonterminal_reward_count+=1
            raise AssertionError('BRO environment produced a nonterminal reward')
        replay.insert(obs[None],action[None],np.array([reward]),np.array([1.-float(terminated)]),
                      np.array([float(done)]),nxt[None])
        obs = nxt
        if done:
            terminal_feedback_count+=1
            returns.append(reward*100.)
            log.append(real_steps=step,episode=len(returns),terminal_score=float(reward)*100.)
            obs,_=env.reset()
        if step>warm:
            info=agent.update(replay.sample_parallel_multibatch(batch,utd),utd,step)
            # Synchronize the real device computation for honest wall-clock accounting.
            jax.block_until_ready(info)
            updates+=utd
            last_info={key:float(np.asarray(value).mean()) for key,value in info.items()}
        if step%(int(config.get('eval_every_episodes',48))*h)==0 and not smoke:
            # Evaluation must not alter the exploration RNG.
            result=evaluate_policy()
            eval_env_steps+=result['evaluation_steps']
            log.append(real_steps=step,critic_updates=updates,actor_updates=updates,evaluation=result)
    after = {key:array_digest(jax.tree_util.tree_leaves(getattr(agent,key).params)) for key in before}
    assert all(before[k]!=after[k] for k in before), 'BRO core components were not all trained'
    fields=('actor','actor_o','critic','target_critic','temp','optimism','regularizer','rng','step')
    state={key:serialization.to_state_dict(getattr(agent,key)) for key in fields}
    state_digest=array_digest(jax.tree_util.tree_leaves(state))
    with (log.output/'checkpoint.pkl').open('wb') as f: pickle.dump(state,f)
    rng=agent.rng
    expected=action_fn(obs,True)
    with (log.output/'checkpoint.pkl').open('rb') as f: saved=pickle.load(f)
    for key in fields:
        setattr(agent,key,serialization.from_state_dict(getattr(agent,key),saved[key]))
    restored_state={key:serialization.to_state_dict(getattr(agent,key)) for key in fields}
    restored_digest=array_digest(jax.tree_util.tree_leaves(restored_state))
    assert state_digest==restored_digest, 'BRO full checkpoint state was not restored exactly'
    actual=action_fn(obs,True)
    assert np.allclose(expected,actual,atol=1e-6)
    agent.rng=rng
    finite=all(np.isfinite(np.asarray(a)).all() for k in before for a in jax.tree_util.tree_leaves(getattr(agent,k).params))
    assert finite
    assert all(np.isfinite(value) for value in last_info.values()), 'BRO native update metrics were non-finite'
    source_root=ROOT/'vendor/bro'
    source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source_root,text=True).strip()
    source_modified=subprocess.run(['git','diff','--quiet','HEAD','--','jaxrl'],cwd=source_root).returncode!=0
    assert not source_modified, 'BRO vendored algorithm source was modified'
    receipt=log.finish(real_steps=total,completed_episodes=terminal_feedback_count,critic_updates=updates,actor_updates=updates,
        component_changes={k:before[k]!=after[k] for k in before},all_parameters_finite=finite,
        checkpoint_roundtrip=True,checkpoint_state_exact=True,checkpoint_state_digest=state_digest,
        nonterminal_reward_count=nonterminal_reward_count,terminal_feedback_count=terminal_feedback_count,
        optimistic_collection_steps=optimistic_collection_steps,temperature_updates=updates,
        optimism_updates=updates,regularizer_updates=updates,evaluation_steps=eval_env_steps,
        batch_size=batch,learning_starts=warm,replay_rows=replay.size,latest_native_metrics=last_info,
        source_commit=source_sha,source_modified=source_modified,
        evaluation_policy='stochastic_temperature1_fixed_dev_key_restored_training_rng',
        runtime={'python':sys.executable,'jax':jax.__version__,'jaxlib':jaxlib.__version__,'numpy':np.__version__},
        jax_devices=[str(d) for d in jax.devices()],
        distinctive_mechanism={'native_full_bro':True,'optimistic_exploration':True,
                              'distributional':True,'n_quantiles':100,'utd':utd,'scheduled_resets':agent.reset_list},
        source_module=sys.modules[BRO.__module__].__file__)
    env.close()
    return receipt
