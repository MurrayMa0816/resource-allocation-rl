"""Batched native transitions; preference is a constant algorithm setting."""
import copy
import hashlib
import json
import numpy as np
import torch
from allocation4.common import STATE_KEYS
from allocation4.methods.lax import gaussian_parameters,gaussian_log_prob
from allocation23.task import Task
from allocation23.environments import build_environment
from .common import coverage_metrics,summary,save_trace,weight_key


def make_pool(domain,seed,size,master,*,holdout=False):
    if size<=0:raise ValueError('Positive pool size required')
    first=Task(domain,3,'A',seed,path_master=master,holdout=holdout);output=[first]
    for _ in range(size-1):
        task=copy.copy(first)
        task.env,task.spec,task.physical_dict=build_environment(domain,3,radio_groups=first.spec.groups)
        task.image_sets=dict(first.image_sets);task.data_identity=copy.deepcopy(first.data_identity);output.append(task)
    return output


def conditioned_state(state,weight):
    if not np.isfinite(weight) or not 0<=weight<=1:raise ValueError('Preference must be in [0,1]')
    result={k:np.asarray(state[k]).copy() for k in STATE_KEYS}
    if result['global'].ndim!=1:raise ValueError('One unbatched native state required')
    result['global']=np.concatenate((result['global'],np.array([weight],dtype=np.float64)))
    return result


def terminal_quality(task):
    if task.domain=='radio':
        if task.env._round!=task.spec.horizon:raise AssertionError('Early terminal radio quality')
        quality=task.env._terminal_ssim
    else:
        if not task.env._terminated:raise AssertionError('Early terminal emergency quality')
        quality=task.env._terminal_info()['per_region_completion']
    y=np.asarray(quality,dtype=np.float64).copy()
    if y.shape!=(task.spec.groups,) or not np.isfinite(y).all():raise AssertionError('Invalid native terminal vector')
    return y


def rollout(pool,actor,critic,generator,phase,first,*,tau,preferences,behavior_version=0,
            deterministic=False,training=True,step_callback=None):
    weights=np.broadcast_to(np.asarray(preferences,dtype=float),(len(pool),)).copy()
    if not len(pool) or not np.isfinite(weights).all() or (weights<0).any() or (weights>1).any():
        raise ValueError('Nonempty pool and finite preferences in [0,1] required')
    encoders=[];plans=[]
    for j,task in enumerate(pool):
        obs,info=task.reset(phase,first+j)
        if info:raise AssertionError('Reset info must be empty')
        enc=task.encoder();enc.reset(obs);encoders.append(enc);plans.append(task.plan(phase,first+j))
    saved=[dict(states={k:[] for k in STATE_KEYS},next_states={k:[] for k in STATE_KEYS},
        raw=[],old_logp=[],mask=[],values=[],rewards=[],done=[],behavior_mean=[],behavior_sigma=[]) for _ in pool]
    actions=[[] for _ in pool];horizon=pool[0].spec.horizon
    for t in range(horizon):
        native=[enc.state() for enc in encoders]
        states=[conditioned_state(st,w) for st,w in zip(native,weights)]
        combined={k:np.stack([st[k] for st in states]) for k in STATE_KEYS}
        with torch.no_grad():
            result=actor.sample(combined,generator=generator,deterministic=deterministic)
            values=critic(combined) if training else torch.zeros((len(pool),2),dtype=torch.float64)
            if values.shape!=(len(pool),2):raise AssertionError('Two value heads ordered C,J required')
            mu,sigma=gaussian_parameters(actor,combined)
            check=gaussian_log_prob(mu,sigma,result['raw'])
            if float((check-result['log_prob']).abs().max())>1e-8:raise AssertionError('Complete Gaussian mismatch')
        a={k:result[k].detach().cpu().numpy().copy() for k in ('raw','log_prob','mask')}
        vv=values.detach().cpu().numpy();mm=mu.detach().cpu().numpy();ss=sigma.detach().cpu().numpy()
        if not all(np.isfinite(x).all() for x in (*a.values(),vv,mm,ss)):raise FloatingPointError('Nonfinite event')
        for j,(task,enc,state,ns) in enumerate(zip(pool,encoders,states,native)):
            # This is the original physical state, without the preference coordinate.
            action,_=task.validate(result['actions'][j],ns)
            obs,r,done,truncated,info=task.env.step(action)
            if step_callback is not None:step_callback(bool(done))
            if truncated or done!=(t==horizon-1):raise AssertionError('Native horizon mismatch')
            if not done and (r!=0. or info):raise AssertionError('Intermediate signal leakage')
            enc.update(action,obs);next_state=conditioned_state(enc.state(),weights[j]);b=saved[j]
            if training:
                for k in STATE_KEYS:
                    b['states'][k].append(state[k].copy());b['next_states'][k].append(next_state[k].copy())
                for k,x in [('raw',a['raw'][j]),('old_logp',a['log_prob'][j]),('mask',a['mask'][j]),
                    ('values',vv[j]),('behavior_mean',mm[j]),('behavior_sigma',ss[j])]:b[k].append(x.copy())
            b['rewards'].append(float(r));b['done'].append(bool(done));actions[j].append(action.copy())
    output=[]
    for j,(task,b) in enumerate(zip(pool,saved)):
        rewards=np.asarray(b.pop('rewards'));done=np.asarray(b.pop('done'),dtype=bool)
        native=float(rewards.sum());aa=np.asarray(actions[j]);quality=terminal_quality(task)
        if abs(native-float(quality.mean()))>1e-10:raise AssertionError('Native reward/quality mismatch')
        if np.any(rewards[:-1]!=0) or not done[-1] or done[:-1].any():raise AssertionError('Terminal-only reward required')
        metrics=coverage_metrics(quality,tau);remaining=float(encoders[j].state()['remaining'])
        row=dict(native_return=native,score_x100=100*native,quality=quality,actions=aa,rewards=rewards,
            total_used=float(aa.sum()),remaining=remaining,period_totals=aa.sum(1),preference=float(weights[j]),
            physical_total_used=float(aa.sum())*task.spec.action_cost_scale,
            physical_remaining=remaining*task.spec.action_cost_scale,
            physical_period_totals=aa.sum(1)*task.spec.action_cost_scale,
            physical_resource_unit=task.spec.resource_unit,action_unit='W' if task.domain=='radio' else 'equivalent diesel L',
            plan=plans[j],plan_sha256=hashlib.sha256(json.dumps(plans[j],sort_keys=True).encode()).hexdigest(),**metrics)
        if training:
            batch={k:{a:np.stack(v) for a,v in b[k].items()} for k in ('states','next_states')}
            batch.update({k:np.asarray(v) for k,v in b.items() if k not in ('states','next_states')})
            batch.update(rewards=rewards,done=done,returns=np.full(horizon,native),behavior_version=int(behavior_version),
                K=metrics['K'],C=metrics['C'],coverage_count=metrics['K'],coverage=metrics['C'],
                terminal_quality=quality.copy(),preference=float(weights[j]))
            row['batch']=batch
        output.append(row)
    return output


def result_row(result,index,*,trace=None,trace_index=None):
    row={k:result[k] for k in ('score_x100','native_return','total_used','remaining','K','N','Q','C',
        'quality_mean','quality_min','quality_p10','plan_sha256','plan','preference')}
    row.update(index=index,quality=result['quality'].tolist(),period_totals=result['period_totals'].tolist(),
        physical_total_used=result['physical_total_used'],physical_remaining=result['physical_remaining'],
        physical_period_totals=result['physical_period_totals'].tolist(),physical_resource_unit=result['physical_resource_unit'],
        action_unit=result['action_unit'])
    if trace is not None:row.update(trace_file=trace['path'],trace_sha256=trace['sha256'],trace_index=trace_index)
    return row


def assess(pool,actor,critic,count,device,phase,*,tau,preferences,trace_prefix=None,
           eval_seed=996101,step_callback=None):
    output={}
    for weight in preferences:
        # Reset sampler and native case index for every preference, common random numbers.
        generator=torch.Generator(device=device).manual_seed(eval_seed);rows=[]
        for first in range(0,count,len(pool)):
            results=rollout(pool[:min(len(pool),count-first)],actor,critic,generator,phase,first,
                tau=tau,preferences=float(weight),training=False,step_callback=step_callback)
            trace=save_trace(f'{trace_prefix}_{weight_key(weight)}_{first:04d}.npz',results) if trace_prefix else None
            rows.extend(result_row(x,first+j,trace=trace,trace_index=j) for j,x in enumerate(results))
        output[weight_key(weight)]=dict(weight=float(weight),stochastic=dict(**summary(rows),rows=rows))
    return output
