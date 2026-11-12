# SPDX-License-Identifier: Apache-2.0
"""CPU-only exact-batch/forced-clock tests; no physical Env or GPU probe."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from agent import Agent, tensor_hash
from networks import DenseTFAdam


def manual_adam(opt):
    opt.iterations+=1
    rate=opt.lr*(1-opt.beta2**opt.iterations)**.5/(1-opt.beta1**opt.iterations)
    with torch.no_grad():
        for p in opt.parameters:
            state=opt.state[p]
            state['m'].copy_(opt.beta1*state['m']+(1-opt.beta1)*p.grad)
            state['v'].copy_(opt.beta2*state['v']+(1-opt.beta2)*p.grad.square())
            p.add_(-rate*state['m']/(state['v'].sqrt()+opt.epsilon))


def reference_sample(actor,obs,noise):
    params=actor(obs);mean,ls=params.chunk(2,-1)
    raw=mean+ls.exp()*noise
    distribution=torch.distributions.Normal(mean,ls.exp())
    transform=torch.distributions.TanhTransform(cache_size=1)
    return raw.tanh(),(distribution.log_prob(raw)-transform.log_abs_det_jacobian(raw,raw.tanh())).sum(-1)


def reference_update(a,batch,noises):
    o,u,r,n,d=a._parts(batch)
    with torch.no_grad():
        na,nlp=reference_sample(a.actor,n,noises[0]);q1,q2=a.critic_target(n,na)
        y=r*a.reward_scale+(1-d)*a.gamma*(torch.min(q1,q2)-a.log_alpha.exp()*nlp)
    a.critic_optimizer.zero_grad();q1,q2=a.critic(o,u)
    cl=.5*((q1-y).square().mean()+(q2-y).square().mean());cl.backward();manual_adam(a.critic_optimizer)
    for p in a.critic.parameters():p.requires_grad_(False)
    a.actor_optimizer.zero_grad();pa,lp=reference_sample(a.actor,o,noises[1]);q1,q2=a.critic(o,pa)
    al=(a.log_alpha.detach().exp()*lp-torch.min(q1,q2)).mean();al.backward();manual_adam(a.actor_optimizer)
    for p in a.critic.parameters():p.requires_grad_(True)
    with torch.no_grad():_,lp=reference_sample(a.actor,o,noises[2])
    a.alpha_optimizer.zero_grad();el=-(a.log_alpha*(lp+a.target_entropy)).mean();el.backward();manual_adam(a.alpha_optimizer)
    a.n_updates+=1
    with torch.no_grad():
        for p,t in zip(a.critic.parameters(),a.critic_target.parameters()):t.copy_((1-a.tau)*t+a.tau*p)
    return torch.stack([cl.detach(),al.detach(),el.detach()])


def max_difference(a,b):
    differences=[]
    for name in ['actor','critic','critic_target']:
        differences += [float((x-y).abs().max()) for x,y in zip(getattr(a,name).parameters(),getattr(b,name).parameters())]
    differences += [float((a.log_alpha-b.log_alpha).abs())]
    return max(differences)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',required=True);args=parser.parse_args()
    torch.set_num_threads(1);assert not torch.cuda.is_initialized()
    receipt=Path(args.receipt);receipt.parent.mkdir(parents=True,exist_ok=True)
    tests={};cfg={'seed':9920,'total_steps':49152,'replay_capacity':256,'batch_size':16}
    # Distinguish epsilon-hat from the ordinary Torch Adam denominator.
    p=torch.nn.Parameter(torch.tensor([.1,-.7,.3],dtype=torch.float64));q=torch.nn.Parameter(p.detach().clone())
    opt=DenseTFAdam([p]);ref=DenseTFAdam([q])
    for k in range(32):
        g=torch.tensor([.01*(-1)**k,.001*(k+1),.03],dtype=torch.float64);p.grad=g;q.grad=g.clone()
        opt.step();manual_adam(ref)
    assert torch.allclose(p,q,atol=5e-17,rtol=0)
    tests['TF_Adam_epsilon_hat_32step_float64_reference']=float((p-q).abs().max())
    a=Agent(cfg,6,3);b=copy.deepcopy(a)
    batch=torch.randn(16,17);batch[:,6:9]=batch[:,6:9].tanh();batch[:,-1]=0;batch[-1,-1]=1
    noises=[torch.randn(16,3) for _ in range(3)]
    a._update_batch(batch,noises=noises);expected=reference_update(b,batch,noises)
    error=max_difference(a,b);loss_error=float((a._last_losses-expected).abs().max())
    assert error<2e-6 and loss_error<2e-5,(error,loss_error)
    tests['fixed_batch_full_SAC_critic_actor_alpha_target_reference']={'parameter_max_error':error,'loss_max_error':loss_error}
    assert all(opt.iterations==1 for opt in [a.actor_optimizer,a.critic_optimizer,a.alpha_optimizer])

    # Actual source masks/transpose, overlap order, partial slots, bias behavior,
    # target/replay invariance and local seeded initializer RNG behavior.
    c=Agent(cfg,6,3)
    for opt in [c.actor_optimizer,c.critic_optimizer]:
        for _ in range(3):
            for p in opt.parameters:p.grad=torch.full_like(p,.01)
            opt.step()
    x=np.ones(6,dtype=np.float32);u=np.zeros(3,dtype=np.float32)
    for _ in range(8):c.replay.add(x,u,0.,x,False)
    acts=[]
    for dead in ([2],[3,9]):
        value=torch.ones(256,256);value[:,dead]=0;acts.append(value)
    before=copy.deepcopy(c);rng=torch.random.get_rng_state().clone()
    c._recycle([acts,acts],acts,200000)
    assert torch.equal(rng,torch.random.get_rng_state())
    for name in ['actor','critic.q1','critic.q2']:
        network=c.get_submodule(name) if isinstance(c,torch.nn.Module) else (
            c.actor if name=='actor' else getattr(c.critic,name.split('.')[1]))
        old=before.actor if name=='actor' else getattr(before.critic,name.split('.')[1])
        opt=c.actor_optimizer if name=='actor' else c.critic_optimizer
        oldopt=before.actor_optimizer if name=='actor' else before.critic_optimizer
        for i,layer in enumerate(network.layers):
            oldlayer=old.layers[i]
            mask=torch.zeros_like(layer.weight,dtype=torch.bool)
            if i<2:mask[[2] if i==0 else [3,9],:]=True
            if i>0:mask[:,[2] if i==1 else [3,9]]=True
            assert torch.equal(layer.weight[~mask],oldlayer.weight[~mask])
            if i>0:assert torch.count_nonzero(layer.weight[:,[2] if i==1 else [3,9]])==0
            for slot in ['m','v']:
                assert torch.count_nonzero(opt.state[layer.weight][slot][mask])==0
                assert torch.equal(opt.state[layer.weight][slot][~mask],oldopt.state[oldlayer.weight][slot][~mask])
                assert torch.equal(opt.state[layer.bias][slot],oldopt.state[oldlayer.bias][slot])
        assert opt.iterations==oldopt.iterations==3
    event=c.events[-1]
    assert event['total_dead_neurons']==9 and event['target_before_sha256']==event['target_after_direct_recycle_sha256']
    assert event['replay_before_sha256']==event['replay_after_sha256']
    tests['allhidden_actor_twincritic_mask_axes_overlap_slots_bias_target_replay_RNG']='PASS'

    # Threshold/normalization independently checked on nonuniform positive acts.
    for act in acts:
        score=act.abs().mean(0)/(act.abs().mean(0).mean()+1e-9)
        assert torch.equal(score<=0,act.abs().sum(0)==0)
    tests['normalized_activation_score_and_tau0']='PASS'
    try:Agent({**cfg,'reset_mode':None},6,3)
    except ValueError:pass
    else:raise AssertionError('stale gin reset_mode=None must not disable registered ReDo')
    w=Agent({**cfg,'warmup_steps':3,'replay_ratio':2},6,3)
    initial=w.native_state_fingerprint();w.policy_action(x,False)
    assert initial==w.native_state_fingerprint()
    for k in range(8):
        w.observe(x,u,.7 if k%2 else 0.,x,k%2==1)
        assert w.n_updates==max(0,k+1-3)*2
    assert not w.events
    tests['charged_warmup_RR_threshold_true_pi0_DEV_state_guard']='PASS'

    # A labelled synthetic counter fixture, not200000 claimed real updates.
    f=Agent(cfg,6,3)
    for _ in range(8):f.replay.add(x,u,0.,x,False)
    f.steps=35001;f.n_updates=200000
    for opt in [f.actor_optimizer,f.critic_optimizer,f.alpha_optimizer]:opt.iterations=200000
    with torch.no_grad():
        for network in [f.actor,f.critic.q1,f.critic.q2]:
            network.layers[0].weight[2].zero_();network.layers[0].bias[2]=-100
            network.layers[1].weight[7].zero_();network.layers[1].bias[7]=-100
    f._update_batch(batch,noises=noises)
    assert f.n_updates==200001 and len(f.events)==1
    e=f.events[0];assert e['environment_step']==35001 and e['update_counter_before']==200000 and e['total_dead_neurons']>=6
    assert all(opt.iterations==200001 for opt in [f.actor_optimizer,f.critic_optimizer,f.alpha_optimizer])
    tests['forced_author_preincrement_clock_event200001_env35001']=e
    with tempfile.TemporaryDirectory(dir=receipt.parent,prefix='redo_roundtrip_') as temporary:
        path=Path(temporary)/'actor.pt';w.save(path);loaded=Agent.load(path)
        assert np.array_equal(w.policy_action(x,True),loaded.policy_action(x,True))
        rng=torch.random.get_rng_state();pa=w.policy_action(x,False);torch.random.set_rng_state(rng)
        assert np.array_equal(pa,loaded.policy_action(x,False))
        try:loaded.observe(x,u,0.,x,False)
        except RuntimeError:pass
        else:raise AssertionError('inference endpoint must not continue a random critic')
    tests['one_actor_endpoint_roundtrip_no_optimizer_replay']='PASS'

    # CPU-only fixed-batch timing at the actual larger observation dimension.
    perf=Agent({'seed':9920,'replay_capacity':64},305,19)
    fixed=torch.randn(256,631);fixed[:,305:324]=fixed[:,305:324].tanh();fixed[:,-1]=0;fixed[-1,-1]=1
    t=time.perf_counter()
    for _ in range(32):perf._update_batch(fixed)
    elapsed=time.perf_counter()-t
    tests['fullwidth_CPU_fixed_batch_timing']={'iterations':32,'wall_seconds':elapsed,'seconds_per_iteration':elapsed/32,
        'singlethread_313216_iteration_core_seconds_estimate':elapsed/32*313216,
        'not_CUDA_or_full_training_estimate':True}
    assert not torch.cuda.is_initialized()
    output={'status':'PASS','environment_calls':0,'GPU_used':False,'synthetic_full_SAC_iterations':45,
        'synthetic_Adam_only_steps':70,'tests':tests,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'limitations':['CPU synthetic gates and forced clocks do not prove actual task learning or CUDA behavior.'],
        'registered_science_expected':{'steps':49152,'warmup':10000,'RR':8,'iterations':313216,'events':1,
            'first_event_environment_step':35001,'first_event_update_number':200001}}
    receipt.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'status':'PASS','tests':len(tests),'CPU_timing':tests['fullwidth_CPU_fixed_batch_timing'],'receipt':str(receipt)},indent=2))


if __name__=='__main__':main()
