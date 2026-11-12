# SPDX-License-Identifier: Apache-2.0
"""Root-invoked full-state CUDA graph equivalence gate, zero Env calls."""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root
import argparse
import copy
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from agent import Agent as ReferenceAgent
from graph_agent import Agent as GraphAgent


def rng():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())


def setrng(x):
    random.setstate(x[0]);np.random.set_state(x[1]);torch.set_rng_state(x[2]);torch.cuda.set_rng_state(x[3])


def same_rng(a,b):
    return (a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1])
            and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2]) and torch.equal(a[3],b[3]))


def compare(a,b):
    checked=0
    for name in ['actor','critic','critic_target']:
        left,right=getattr(a,name),getattr(b,name)
        for key,value in left.state_dict().items():
            if not torch.equal(value,right.state_dict()[key]):raise AssertionError(('parameter',name,key,float((value-right.state_dict()[key]).abs().max())))
            checked+=value.numel()
        for x,y in zip(left.parameters(),right.parameters()):
            if (x.grad is None)!=(y.grad is None):raise AssertionError(('gradient_presence',name))
            if x.grad is not None and not torch.equal(x.grad,y.grad):raise AssertionError(('gradient',name))
    if not torch.equal(a.log_alpha,b.log_alpha):raise AssertionError('logalpha')
    if not torch.equal(a.log_alpha.grad,b.log_alpha.grad):raise AssertionError('alpha gradient')
    for ao,bo in zip([a.actor_optimizer,a.critic_optimizer,a.alpha_optimizer],
                     [b.actor_optimizer,b.critic_optimizer,b.alpha_optimizer]):
        if ao.iterations!=bo.iterations:raise AssertionError('Adam counter')
        for x,y in zip(ao.parameters,bo.parameters):
            for slot in ['m','v']:
                if not torch.equal(ao.state[x][slot],bo.state[y][slot]):raise AssertionError(('Adam',slot))
                checked+=ao.state[x][slot].numel()
    if a.n_updates!=b.n_updates or a.steps!=b.steps or a.events!=b.events:raise AssertionError('agent/event counters/masks')
    if not torch.equal(a._last_losses,b._last_losses):raise AssertionError('last losses')
    if a.rng.bit_generator.state!=b.rng.bit_generator.state:raise AssertionError('replay Generator RNG')
    if a.replay.size!=b.replay.size or a.replay.fingerprint()!=b.replay.fingerprint():raise AssertionError('replay')
    return checked


def counter(a,n):
    a.n_updates=n
    for opt in [a.actor_optimizer,a.critic_optimizer,a.alpha_optimizer]:opt.iterations=n


def main():
    p=argparse.ArgumentParser();p.add_argument('--physical-gpu',required=True);p.add_argument('--receipt',required=True)
    p.add_argument('--timing-blocks',type=int,default=16);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.physical_gpu:raise RuntimeError('root must fix physical GPU before invoking')
    output=Path(args.receipt).resolve()
    if not str(output).startswith(str(_resolve_path('/hy-tmp/'))) or output.exists():raise ValueError('new HY engineering receipt required')
    if not 1<=args.timing_blocks<=64:raise ValueError('bounded one-candidate acceleration gate')
    torch.set_num_threads(1);torch.cuda.set_device(0)
    results=[]
    for domain,d,u in [('radio',305,19),('v5',244,13)]:
        cfg={'seed':9960,'replay_capacity':128,'replay_ratio':8,'batch_size':256}
        ref=ReferenceAgent(cfg,d,u,'cuda:0');fast=GraphAgent(cfg,d,u,'cuda:0')
        cpu_rng=torch.Generator().manual_seed(9961)
        cpu=torch.randn(8,256,2*d+u+2,generator=cpu_rng)
        cpu[:,:,d:d+u]=cpu[:,:,d:d+u].tanh();cpu[:,:,d+u]=0;cpu[:,-8:,d+u]=.7
        cpu[:,:,-1]=0;cpu[:,-8:,-1]=1;batches=cpu.cuda()
        obs=np.zeros(d,dtype=np.float32);act=np.zeros(u,dtype=np.float32)
        for agent in [ref,fast]:
            for _ in range(8):agent.replay.add(obs,act,0.,obs,False)
        # Warm Adam with actual native updates, then use an explicitly labelled
        # synthetic counter fixture; never claim40000 physical optimizer calls.
        state=rng()
        for batch in batches[:2]:ref._update_batch(batch)
        end=rng();setrng(state)
        for batch in batches[:2]:fast._update_batch(batch)
        assert same_rng(end,rng());compare(ref,fast)
        counter(ref,39992);counter(fast,39992);ref.steps=fast.steps=14999
        stages=[]
        def block(label):
            state=rng()
            for batch in batches:ref._update_batch(batch)
            expected=rng();setrng(state);fast.update_block(batches)
            if not same_rng(expected,rng()):raise AssertionError(('full global RNG',label))
            checked=compare(ref,fast)
            stages.append({'stage':label,'updates':ref.n_updates,'path':fast._last_execution,
                           'parameters_Adam_scalars_checked':checked,'bitwise_equal':True,'global_RNG_equal':True})
        block('early39992_to40000_eager')
        for k in range(4):block('steady_multiblock_'+str(k))
        counter(ref,199992);counter(fast,199992);ref.steps=fast.steps=35000
        # Force recognizable dead neurons, identically in the two fixtures.
        with torch.no_grad():
            for agent in [ref,fast]:
                for network in [agent.actor,agent.critic.q1,agent.critic.q2]:
                    network.layers[0].weight[2].zero_();network.layers[0].bias[2]=-100
                    network.layers[1].weight[7].zero_();network.layers[1].bias[7]=-100
        block('steady199992_to200000')
        ref.steps=fast.steps=35001;block('200001_recycle_eager_block')
        assert fast._last_execution=='eager_early_or_event' and len(fast.events)==1
        assert fast.events[0]['update_number']==200001 and fast.events[0]['total_dead_neurons']>=6
        block('post_recycle_graph_0');block('post_recycle_graph_1')
        state=rng();torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(args.timing_blocks):
            for batch in batches:ref._update_batch(batch)
        torch.cuda.synchronize();normal_seconds=time.perf_counter()-start;end=rng();setrng(state)
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(args.timing_blocks):fast.update_block(batches)
        torch.cuda.synchronize();graph_seconds=time.perf_counter()-start
        if not same_rng(end,rng()):raise AssertionError('timing full RNG')
        checked=compare(ref,fast)
        results.append({'domain':domain,'stages':stages,'timing_blocks':args.timing_blocks,
            'timed_native_iterations':args.timing_blocks*8,'eager_seconds':normal_seconds,
            'graph_seconds':graph_seconds,'core_speedup':normal_seconds/graph_seconds,
            'bitwise_full_parameters_Adam_gradients_losses_equal':True,'global_RNG_equal':True,
            'post_event_masks_targets_replay_equal':True,'scalars_checked_after_timing':checked,
            'execution':fast.diagnostics()['execution'],'event':fast.events[0],
            'CUDA_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'CUDA_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        del ref,fast,batches;gc.collect();torch.cuda.empty_cache()
    result={'status':'PASS','candidate':'only_steady_native_eight_step_CUDAgraph',
        'physical_gpu_uuid':args.physical_gpu,'environment_steps':0,'new_holdout_calls':0,
        'synthetic_counter_fixtures_not_real40000_or200000_updates':True,
        'native_math_rewritten':False,'native_e3f_reference_unchanged':True,'rows':results,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'limitations':['Core timing includes no physical sampling/DEV or host replay upload; full training speedup is not established.']}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','speedups':[r['core_speedup'] for r in results],'receipt':str(output)}))


if __name__=='__main__':main()
