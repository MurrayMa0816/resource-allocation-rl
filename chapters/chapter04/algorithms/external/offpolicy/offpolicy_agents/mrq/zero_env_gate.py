"""CPU math/schedule preparation only; no Env/Gym/Atari or CUDA execution."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import torch
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from mrq.agent import Agent,_hash,native
    from mrq.native_loader import verify_vendor
    args.output.parent.mkdir(parents=True,exist_ok=True)
    checks={};clocks=[];timings=[]
    def check(k,v):
        checks[k]=bool(v);assert v,k
    def rng():return copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone()
    def restore(x):np.random.set_state(x[0]);torch.set_rng_state(x[1])
    check('author_blobs',verify_vendor()=='073adcd6b30f84694c241e00bd684db50242a215')
    check('no_Gym_Atari_imported',not any(k in sys.modules for k in ['gym','gymnasium','ale_py','env_preprocessing']))
    check('explicit_cuda0_keeps_native_CPU_obs_branch',torch.device('cuda:0')!=torch.device('cuda'))
    # Independent terminal reward and mask formulas.
    reward=torch.tensor([[[0.],[.7],[9.]],[[0.],[0.],[.9]],[[.2],[.3],[.4]]])
    mask=torch.tensor([[[1.],[0.],[1.]],[[1.],[1.],[0.]],[[1.],[1.],[1.]]])
    actual,discount=native.multi_step_reward(reward,mask,.99)
    expected=torch.tensor([[.99*.7],[.99**2*.9],[.2+.99*.3+.99**2*.4]])
    check('three_step_true_terminal_mask',torch.allclose(actual,expected,atol=1e-7,rtol=0) and torch.allclose(discount,torch.tensor([[0.],[0.],[.99**3]]),atol=1e-7,rtol=0))
    pred=torch.tensor([[2.,3.],[1000.,1000.]]);target=torch.ones(2,2);m=torch.tensor([[1.],[0.]])
    check('masked_dynamics_after_terminal',(native.masked_mse(pred,target,m)==torch.tensor(1.25)).item())
    two=native.TwoHot(torch.device('cpu'),-10,10,65);targets=torch.tensor([[0.],[.3],[.7],[1.]])
    hot=two.transform(targets)
    check('reward_two_hot_conserves_mass',torch.allclose(hot.sum(1),torch.ones(4),atol=1e-7,rtol=0))
    check('reward_two_hot_expected_raw_reward',torch.allclose((hot*two.bins).sum(1,keepdim=True),targets,atol=1e-7,rtol=0))
    for domain,obs,act,H,first in [('radio',305,19,32,9698),('v5',244,13,24,9601)]:
        torch.manual_seed(981);np.random.seed(981);adapter=Agent({'seed':981},obs,act,'cpu')
        torch.manual_seed(981);np.random.seed(981);reference=native.Agent((obs,),act,1.,False,False,torch.device('cpu'))
        for name in ['encoder','encoder_target','policy','policy_target','value','value_target']:
            check(domain+'_initial_'+name,_hash(getattr(adapter.core,name).state_dict())==_hash(getattr(reference,name).state_dict()))
        B=256
        state=torch.randn(B,5,obs);action=torch.rand(B,5,act)*2-1;next_state=torch.randn(B,5,obs)
        reward=torch.zeros(B,5,1);not_done=torch.ones(B,5,1)
        reward[:4,1]=.7;not_done[:4,1:]=0
        reward[4:8,4]=.5;not_done[4:8,4]=0
        before=rng();t=time.perf_counter()
        adapter.core.train_encoder(state,action,next_state,reward,not_done,True)
        timings.append(dict(domain=domain,one_native_encoder_step_seconds=time.perf_counter()-t))
        after=rng();restore(before);reference.train_encoder(state,action,next_state,reward,not_done,True)
        check(domain+'_encoder_RNG_exact',_hash(rng())==_hash(after))
        for name in ['encoder','encoder_optimizer']:
            check(domain+'_encoder_math_'+name,_hash(getattr(adapter.core,name).state_dict())==_hash(getattr(reference,name).state_dict()))
        # Actual Q_horizon=3 scalar target (raw reward and gamma*true-done masks).
        r,term=native.multi_step_reward(reward[:,:3],not_done[:,:3],.99)
        before=rng();t=time.perf_counter()
        q1=adapter.core.train_rl(state[:,0],action[:,0],next_state[:,2],r,term,.2,.3)
        timings.append(dict(domain=domain,one_native_value_policy_step_seconds=time.perf_counter()-t))
        after=rng();restore(before);q2=reference.train_rl(state[:,0],action[:,0],next_state[:,2],r,term,.2,.3)
        check(domain+'_RL_RNG_exact',_hash(rng())==_hash(after))
        check(domain+'_Q_and_scaled_target_exact',all(torch.equal(x,y) for x,y in zip(q1,q2)))
        for name in ['encoder','encoder_target','policy','policy_target','value','value_target','encoder_optimizer','policy_optimizer','value_optimizer']:
            check(domain+'_all_learning_state_'+name,_hash(getattr(adapter.core,name).state_dict())==_hash(getattr(reference,name).state_dict()))
        for name in ['encoder','policy','value']:
            check(domain+'_gradient_'+name,_hash([p.grad for p in getattr(adapter.core,name).parameters()])==_hash([p.grad for p in getattr(reference,name).parameters()]))
        before=adapter.native_state_fingerprint();o=np.zeros(obs)
        deployed=adapter.policy_action(o,True)
        check(domain+'_deterministic_deploy',np.array_equal(deployed,adapter.policy_action(o,True)))
        check(domain+'_DEV_no_mutation',before==adapter.native_state_fingerprint())
        with tempfile.TemporaryDirectory(dir=args.output.parent,prefix='mrq_inference_fixture_') as temp:
            path=Path(temp)/'endpoint.pt';adapter.save(path);loaded=Agent.load(path,'cpu')
            check(domain+'_actor_and_state_encoder_roundtrip',np.array_equal(deployed,loaded.policy_action(o,True)))
            check(domain+'_inference_has_no_replay_value_or_optimizer',not hasattr(loaded,'core') and not hasattr(loaded,'replay') and not hasattr(loaded,'critic'))
            check(domain+'_training_deterministic_flag_bypasses_warmup',np.array_equal(deployed,adapter.act(o,True)))
        del adapter,reference
        # Clock-only native replay fixture: smaller backing capacity and batch for
        # schedule traversal, no encoder/value/policy optimizer calls. These are
        # NOT claimed as 500 completed neural updates or physical environment steps.
        class Clock(native.Agent):
            def __init__(self):
                super().__init__((obs,),act,1.,False,False,torch.device('cpu'),hp={'buffer_size':11000,'batch_size':8})
                self.encoder_calls=0;self.rl_calls=0
            def train_encoder(self,*unused,**kw):self.encoder_calls+=1
            def train_rl(self,state,*unused,**kw):
                self.rl_calls+=1;return torch.zeros(state.shape[0],2),torch.zeros(state.shape[0],1)
        clock=Clock();first_seen=None;first_size=None;terminal_count=0
        for step in range(1,first+251):
            done=step%H==0;terminal_count+=int(done)
            o=np.full(obs,float(step%H)/H,dtype=np.float32);nxt=np.full(obs,float((step+1)%H)/H,dtype=np.float32)
            clock.replay_buffer.add(o,np.zeros(act),nxt,.5 if done else 0.,done,False)
            clock.train()
            if clock.training_steps and first_seen is None:first_seen=step;first_size=clock.replay_buffer.size
        check(domain+'_first_update_physical_clock',first_seen==first and first_size==10001)
        check(domain+'_native_target_burst_schedule',clock.training_steps==251 and clock.encoder_calls==500 and clock.rl_calls==251)
        check(domain+'_terminal_slots_not_transitions',clock.replay_buffer.ind==step+terminal_count and clock.replay_buffer.size==step+(step-1)//H)
        check(domain+'_true_done_head_enabled',clock.replay_buffer.env_terminates)
        check(domain+'_LAP_sampling_mask_nonempty',float((clock.replay_buffer.priority*clock.replay_buffer.mask).sum())>0)
        clocks.append(dict(domain=domain,horizon=H,first_native_RL_physical_step=first_seen,
            replay_size_at_first_update=first_size,synthetic_transition_calls=step,
            native_schedule_RL_calls_without_optimizer=clock.rl_calls,
            encoder_schedule_calls_without_optimizer=clock.encoder_calls,
            real_optimizer_updates_in_schedule_fixture=0))
        del clock
    result=dict(status='PASS_CPU_PREPARATION_ONLY',physical_environment_steps=0,environments_constructed=0,
        CUDA_used=False,torch=torch.__version__,actual_synthetic_optimizer_steps=dict(encoder=4,value=4,policy=4),
        direct_math_updates_do_not_claim_full_250_step_encoder_bursts=True,
        schedule_fixtures=clocks,checks=checks,timings=timings,persistent_models=0,
        full_native_CUDA_and_real_environment_gates_still_required=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','physical_environment_steps','CUDA_used','actual_synthetic_optimizer_steps','persistent_models']}))


if __name__=='__main__':main()
