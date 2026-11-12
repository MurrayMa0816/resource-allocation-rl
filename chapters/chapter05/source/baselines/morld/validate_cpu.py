"""Zero-Env CPU checks for the single user-selected MORL/D implementation."""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_morld_selected_validation',HERE/'agent.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def run():
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    configurations=json.loads((HERE/'configs.json').read_text())
    checked=[]
    for domain,obs_dim,action_dim in [('radio',305,19),('v5',244,13)]:
        assert list(configurations[domain])==[module.SELECTED_PROFILE]
        cfg=configurations[domain][module.SELECTED_PROFILE]
        before=torch.get_rng_state().clone()
        agent=module.Agent(obs_dim,action_dim,9840,'cpu',cfg)
        assert torch.equal(before,torch.get_rng_state())
        for member in agent.population:
            assert all(g['lr']==.0001 for g in member.wrapped.q_optimizer.param_groups)
            assert all(g['lr']==.0003 for g in member.wrapped.actor_optimizer.param_groups)
        rng=np.random.default_rng(7312)
        def sample():
            return dict(obs=rng.normal(size=(256,obs_dim)).astype(np.float32),
                actions=np.tanh(rng.normal(size=(256,action_dim))).astype(np.float32),
                next_obs=rng.normal(size=(256,obs_dim)).astype(np.float32),
                rewards=rng.uniform(0,1,(256,2)).astype(np.float32),
                dones=np.ones((256,1),np.float32))
        agent.select_policy(0);agent.set_physical_step(3072)
        stats=agent.update(sample())
        assert all(np.isfinite(x)for x in stats.values()if isinstance(x,(int,float)))
        hook=agent.after_exchange(sample)
        assert hook['extra_native_critic_updates_this_exchange']==20
        assert hook['extra_native_actor_updates_this_exchange']==40
        state=agent.checkpoint_state_dict()
        assert state['format']==module.INFERENCE_VERSION
        assert state['metadata']['selected_profile']=='critic_lr1e4'
        assert state['metadata']['critic_lr']==.0001
        assert state['metadata']['other_update_every_training_episodes']==1
        assert all(state[k]is False for k in ('optimizer_backup','replay_backup','critic_backup','archive_actor_backup'))
        clone=module.Agent(obs_dim,action_dim,9840,'cpu',cfg)
        clone.load_state_dict(state)
        observations=rng.normal(size=(3,obs_dim)).astype(np.float32)
        before=torch.get_rng_state().clone()
        for member in range(3):
            np.testing.assert_array_equal(agent.act_policy(member,observations,True),
                                          clone.act_policy(member,observations,True))
        assert torch.equal(before,torch.get_rng_state())
        rejected=[]
        for name,patch in [('old_profile',dict(profile='repair')),('old_critic',dict(critic_lr=.001)),
                           ('old_q_alias',dict(q_lr=.001)),('cadence16',dict(morld_other_update_every_training_episodes=16)),
                           ('zero_passes',dict(morld_update_passes=0)),('old_constructor',dict(legacy_constructor_cuda_reseed=True))]:
            try:module.selected_config({**cfg,**patch})
            except ValueError:rejected.append(name)
            else:raise AssertionError('Retired configuration remained active: '+name)
        for name,change in [('old_format',dict(format='morld-three-current-actors-inference-v1')),
                            ('wrong_identity',dict(metadata={**state['metadata'],'critic_lr':.001}))]:
            wrong=copy.deepcopy(state);wrong.update(change)
            try:clone.load_state_dict(wrong)
            except ValueError:rejected.append(name)
            else:raise AssertionError('Unselected model identity accepted: '+name)
        checked.append(dict(domain=domain,selected_profile=module.SELECTED_PROFILE,
            native_Q_updates=21,native_actor_updates=42,rejected=rejected,
            actual_optimizer_rates=True,all_three_actor_reload_exact=True))
    assert not torch.cuda.is_initialized()
    return dict(status='PASS_SELECTED_IMPLEMENTATION_CPU_GATE',domains=checked,
        environment_constructions=0,physical_steps=0,new_test_cases=0,new_models_saved=0,
        CUDA_initialized=False,formal_training=False,
        limits='Synthetic native-update/API checks only; not new wireless learning or independent-test evidence.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.expanduser().resolve()
    if not out.is_relative_to('/hy-tmp')or out.exists():
        parser.error('Use a new JSON output path physically under /hy-tmp')
    started=time.monotonic();result=run();result['seconds']=time.monotonic()-started
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
