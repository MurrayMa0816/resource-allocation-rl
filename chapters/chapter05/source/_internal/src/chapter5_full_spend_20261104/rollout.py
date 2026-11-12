"""Shared whole-budget complete rollouts with exact public execution ledgers.

This execution-scope copy retains native sampling and update batches. It adds
complete decoded/raw Gaussian traces and an assessment hook without splitting
trajectories at 512-step boundaries. Frozen science sources are unchanged.
"""
import copy
import hashlib
import json
import numpy as np
import torch
from . import bridge
bridge.import_legacy('chapter5_cross_server_20261103')
from allocation4.common import STATE_KEYS
from allocation4.methods.lax import gaussian_parameters,gaussian_log_prob
from allocation_morl.common import coverage_metrics, state_hash
from allocation_morl.rollout import conditioned_state, terminal_quality
from .bridge import make_pool

_FULL_SPEND = bridge.bootstrap(verify_source=False)
full_spend_bounds = _FULL_SPEND["full_spend_bounds"]
terminal_budget_tolerance = _FULL_SPEND["terminal_budget_tolerance"]


def rollout(pool,actor,critic,generator,phase,first,*,tau,preferences,behavior_version=0,
            deterministic=False,training=True,step_callback=None,after_vector_step=None):
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
    actions=[[] for _ in pool];decoded=[[] for _ in pool];latent=[[] for _ in pool]
    ledgers=[{key:[] for key in ('quota_lower','quota_upper','remaining_before','remaining_after','caps')} for _ in pool]
    audits=[dict(status='pass',max_ledger_error=0.,max_quota_violation=0.,max_cap_excess=0.) for _ in pool]
    horizon=pool[0].spec.horizon
    def collector_fingerprint():
        # Called only at assessment points; includes every pending learning tensor
        # and each physical encoder, while the separate dev pool is advanced.
        return state_hash(dict(saved=saved,actions=actions,decoded=decoded,latent=latent,
            states=[enc.state() for enc in encoders],plans=plans,preferences=weights,
            ledgers=ledgers,audits=audits))
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
            if task.spec.chapter!=3:raise AssertionError('Collector received an old non-full-spend task')
            lower,upper=full_spend_bounds(ns,task.spec)
            remaining_before=float(ns['remaining'])
            caps=np.asarray(ns['caps'],dtype=np.float64).copy()
            action,_=task.validate(result['actions'][j],ns)
            actual_total=float(np.asarray(action).sum())
            tolerance=terminal_budget_tolerance(task.spec)
            quota_error=max(0.,lower-actual_total,actual_total-upper)
            cap_error=max(0.,float(np.max(np.asarray(action)-caps)))
            if (quota_error>tolerance or cap_error>task.spec.action_tolerance or
                    np.any(np.asarray(action)<0) or np.any(np.asarray(action)[caps==0]!=0)):
                raise AssertionError('Shared public [L,U] or native physical cap violated')
            ledgers[j]['quota_lower'].append(lower);ledgers[j]['quota_upper'].append(upper)
            ledgers[j]['remaining_before'].append(remaining_before);ledgers[j]['caps'].append(caps)
            obs,r,done,truncated,info=task.env.step(action)
            if step_callback is not None:step_callback(bool(done))
            if truncated or done!=(t==horizon-1):raise AssertionError('Native horizon mismatch')
            if not done and (r!=0. or info):raise AssertionError('Intermediate signal leakage')
            enc.update(action,obs)
            remaining_after=float(enc.state()['remaining'])
            ledger_error=abs(remaining_before-actual_total-remaining_after)
            if remaining_after < -tolerance or ledger_error>tolerance:
                raise AssertionError('Actual physical action and public resource ledger disagree')
            ledgers[j]['remaining_after'].append(remaining_after)
            audits[j]['max_ledger_error']=max(audits[j]['max_ledger_error'],ledger_error)
            audits[j]['max_quota_violation']=max(audits[j]['max_quota_violation'],quota_error)
            audits[j]['max_cap_excess']=max(audits[j]['max_cap_excess'],cap_error)
            next_state=conditioned_state(enc.state(),weights[j]);b=saved[j]
            if training:
                for k in STATE_KEYS:
                    b['states'][k].append(state[k].copy());b['next_states'][k].append(next_state[k].copy())
                for k,x in [('raw',a['raw'][j]),('old_logp',a['log_prob'][j]),('mask',a['mask'][j]),
                    ('values',vv[j]),('behavior_mean',mm[j]),('behavior_sigma',ss[j])]:b[k].append(x.copy())
            b['rewards'].append(float(r));b['done'].append(bool(done));actions[j].append(action.copy())
            decoded[j].append(np.asarray(result['actions'][j]).copy());latent[j].append(a['raw'][j].copy())
        if after_vector_step is not None:
            after_vector_step(complete=t==horizon-1,collector_fingerprint=collector_fingerprint)
    output=[]
    for j,(task,b) in enumerate(zip(pool,saved)):
        rewards=np.asarray(b.pop('rewards'));done=np.asarray(b.pop('done'),dtype=bool)
        native=float(rewards.sum());aa=np.asarray(actions[j]);quality=terminal_quality(task)
        if abs(native-float(quality.mean()))>1e-10:raise AssertionError('Native reward/quality mismatch')
        if np.any(rewards[:-1]!=0) or not done[-1] or done[:-1].any():raise AssertionError('Terminal-only reward required')
        metrics=coverage_metrics(quality,tau);remaining=float(encoders[j].state()['remaining'])
        full_spend=bridge.assert_full_episode(remaining,float(aa.sum()),task.spec)
        ledger_arrays={key:np.asarray(value) for key,value in ledgers[j].items()}
        if any(len(value)!=horizon for value in ledger_arrays.values()):raise AssertionError('Incomplete physical ledger')
        row=dict(native_return=native,score_x100=100*native,quality=quality,actions=aa,rewards=rewards,
            initial_budget=task.spec.initial_budget,horizon=horizon,full_spend=full_spend,
            ledger_audit=audits[j],**ledger_arrays,
            raw_actions=np.asarray(decoded[j]),latent_actions=np.asarray(latent[j]),
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
        J=result['native_return'],
        initial_budget=result['initial_budget'],horizon=result['horizon'],
        full_spend={key:(value.item() if isinstance(value,np.generic) else value)
                    for key,value in result['full_spend'].items()},
        ledger_audit=result['ledger_audit'],full_spend_required=True,
        physical_total_used=result['physical_total_used'],physical_remaining=result['physical_remaining'],
        physical_period_totals=result['physical_period_totals'].tolist(),physical_resource_unit=result['physical_resource_unit'],
        action_unit=result['action_unit'])
    if trace is not None:row.update(trace_file=trace['path'],trace_sha256=trace['sha256'],trace_index=trace_index)
    return row

