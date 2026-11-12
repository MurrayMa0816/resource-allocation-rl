# SPDX-License-Identifier: Apache-2.0
"""Explicitly invoked CUDA-only engineering gate. Never constructs an Env."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import os
import numpy as np
import torch
from agent import Agent


def maximum_error(cpu,gpu):
    errors=[]
    for name in ['actor','critic','critic_target']:
        errors += [float((a.detach()-b.detach().cpu()).abs().max())
                   for a,b in zip(getattr(cpu,name).parameters(),getattr(gpu,name).parameters())]
    errors += [float((cpu.log_alpha.detach()-gpu.log_alpha.detach().cpu()).abs())]
    return max(errors)


def main():
    p=argparse.ArgumentParser();p.add_argument('--physical-gpu',required=True);p.add_argument('--receipt',required=True)
    p.add_argument('--iterations',type=int,default=32);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.physical_gpu:
        raise RuntimeError('physical GPU must be fixed by root admission before invocation')
    output=Path(a.receipt).resolve()
    if not str(output).startswith('/hy-tmp/') or output.exists():
        raise RuntimeError('use a new physical HY receipt path')
    if not 1<=a.iterations<=256:raise ValueError('bounded pure CUDA timing only')
    torch.set_num_threads(1);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    rows=[]
    for domain,d,u in [('radio',305,19),('v5',244,13)]:
        cfg={'seed':9930,'replay_capacity':512,'batch_size':256}
        cpu=Agent(cfg,d,u,'cpu');gpu=Agent(cfg,d,u,'cuda:0')
        generator=torch.Generator().manual_seed(9940)
        batch=torch.randn(256,2*d+u+2,generator=generator)
        batch[:,d:d+u]=batch[:,d:d+u].tanh()
        batch[:,d+u]=0;batch[-8:,d+u]=.6
        batch[:,-1]=0;batch[-8:,-1]=1
        noises=[torch.randn(256,u,generator=generator) for _ in range(3)]
        cpu._update_batch(batch,noises=noises)
        device_batch=batch.cuda();device_noises=[n.cuda() for n in noises]
        gpu._update_batch(device_batch,noises=device_noises)
        torch.cuda.synchronize()
        error=maximum_error(cpu,gpu)
        loss_error=float((cpu._last_losses-gpu._last_losses.cpu()).abs().max())
        if error>=5e-5 or loss_error>=5e-4:raise AssertionError((domain,error,loss_error))
        # Labelled forced-clock fixture: this executes exactly one real update,
        # not200000 previously trained iterations or any physical collection.
        obs=np.zeros(d,dtype=np.float32);action=np.zeros(u,dtype=np.float32)
        for _ in range(8):gpu.replay.add(obs,action,0.,obs,False)
        gpu.steps=35001;gpu.n_updates=200000
        for opt in [gpu.actor_optimizer,gpu.critic_optimizer,gpu.alpha_optimizer]:opt.iterations=200000
        with torch.no_grad():
            for network in [gpu.actor,gpu.critic.q1,gpu.critic.q2]:
                network.layers[0].weight[2].zero_();network.layers[0].bias[2]=-100
                network.layers[1].weight[7].zero_();network.layers[1].bias[7]=-100
        gpu._update_batch(device_batch,noises=device_noises)
        event=gpu.events[-1]
        assert len(gpu.events)==1 and event['update_number']==200001 and event['total_dead_neurons']>=6
        assert event['target_before_sha256']==event['target_after_direct_recycle_sha256']
        assert event['replay_before_sha256']==event['replay_after_sha256']
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(a.iterations):gpu._update_batch(device_batch)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        rows.append({'domain':domain,'observation_dim':d,'action_dim':u,
            'CPU_vs_CUDA_one_update_parameter_max_error':error,'CPU_vs_CUDA_loss_max_error':loss_error,
            'cross_device_bitwise_claim':False,'forced_clock_event':event,
            'timed_native_iterations':a.iterations,'timed_wall_seconds':elapsed,
            'seconds_per_native_iteration':elapsed/a.iterations,
            'isolated_core_only_313216_iteration_seconds_estimate':elapsed/a.iterations*313216,
            'CUDA_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'CUDA_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        del cpu,gpu,device_batch,device_noises
        gc.collect();torch.cuda.empty_cache()
    output.parent.mkdir(parents=True,exist_ok=True)
    result={'status':'PASS','role':'pure_engineering_not_training','physical_gpu_uuid':a.physical_gpu,
        'logical_device':'cuda:0','actual_environment_steps':0,'new_holdout_calls':0,
        'synthetic_CPU_native_iterations':2,'synthetic_CUDA_native_iterations':2*(a.iterations+2),
        'rows':rows,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'performance_limit':'Fixed batch core only, no collection/DEV/host upload/shared contention/full training claims.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','receipt':str(output),'timing':[{'domain':x['domain'],'seconds_per_iteration':x['seconds_per_native_iteration']} for x in rows]}))


if __name__=='__main__':main()
