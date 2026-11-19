"""Algorithm-side trajectory recording; no environment mechanisms changed."""
import hashlib,json
import numpy as np
import torch
from allocation4.common import STATE_KEYS
from allocation4.rollout import make_pool,rollout as native_rollout
from allocation4.methods.lax import gaussian_parameters,gaussian_log_prob

def rollout(pool,actor,critic,generator,phase,first,*,behavior_version=0):
    encoders=[];plans=[]
    for j,task in enumerate(pool):
        obs,info=task.reset(phase,first+j)
        if info:raise AssertionError('Reset info must be empty')
        enc=task.encoder();enc.reset(obs);encoders.append(enc);plans.append(task.plan(phase,first+j))
    saved=[dict(states={k:[] for k in STATE_KEYS},next_states={k:[] for k in STATE_KEYS},
                raw=[],old_logp=[],mask=[],values=[],rewards=[],done=[],
                behavior_mean=[],behavior_sigma=[]) for _ in pool]
    actions=[[] for _ in pool]
    h=pool[0].spec.horizon
    for t in range(h):
        states=[enc.state() for enc in encoders]
        combined={k:np.stack([s[k] for s in states]) for k in STATE_KEYS}
        with torch.no_grad():
            result=actor.sample(combined,generator=generator)
            value=critic(combined);mu,sigma=gaussian_parameters(actor,combined)
            check=gaussian_log_prob(mu,sigma,result['raw'])
            if float((check-result['log_prob']).abs().max())>1e-8:
                raise AssertionError('Complete Gaussian density mismatch')
        a={k:result[k].detach().cpu().numpy().copy() for k in ('raw','log_prob','mask')}
        vv=value.detach().cpu().numpy();mm=mu.detach().cpu().numpy();ss=sigma.detach().cpu().numpy()
        for j,(task,enc,state) in enumerate(zip(pool,encoders,states)):
            action,_=task.validate(result['actions'][j],state)
            obs,r,done,truncated,info=task.env.step(action)
            if truncated or done!=(t==h-1):raise AssertionError('Unexpected native termination')
            if not done and (r!=0. or info):raise AssertionError('Nonterminal signal leakage')
            enc.update(action,obs);next_state=enc.state();b=saved[j]
            for k in STATE_KEYS:
                b['states'][k].append(np.asarray(state[k]).copy())
                b['next_states'][k].append(np.asarray(next_state[k]).copy())
            for k,x in [('raw',a['raw'][j]),('old_logp',a['log_prob'][j]),('mask',a['mask'][j]),
                        ('values',vv[j]),('behavior_mean',mm[j]),('behavior_sigma',ss[j])]:b[k].append(x.copy())
            b['rewards'].append(float(r));b['done'].append(bool(done));actions[j].append(action.copy())
    output=[]
    for j,b in enumerate(saved):
        reward=np.asarray(b.pop('rewards'));done=np.asarray(b.pop('done'),dtype=bool)
        native=float(reward.sum());aa=np.asarray(actions[j])
        batch={k:{a:np.stack(v) for a,v in b[k].items()} for k in ('states','next_states')}
        batch.update({k:np.asarray(v) for k,v in b.items() if k not in ('states','next_states')})
        batch.update(rewards=reward,done=done,returns=np.full(h,native),behavior_version=int(behavior_version))
        if np.any(reward[:-1]!=0) or not done[-1] or done[:-1].any():raise AssertionError('Terminal reward contract')
        output.append(dict(batch=batch,native_return=native,score_x100=100*native,
            total_used=float(aa.sum()),remaining=float(encoders[j].state()['remaining']),
            period_totals=aa.sum(1),plan_sha256=hashlib.sha256(json.dumps(plans[j],sort_keys=True).encode()).hexdigest()))
    return output

def assess(pool,actor,critic,count,device,phase,*,mean_control=False):
    output={}
    modes=[('stochastic',False)]+([('mean_control',True)] if mean_control else [])
    for name,deterministic in modes:
        generator=torch.Generator(device=device).manual_seed(995103 if phase=='dev' else 995199)
        rows=[]
        for first in range(0,count,len(pool)):
            results=native_rollout(pool,actor,critic,generator,phase,first,
                count=min(len(pool),count-first),deterministic=deterministic)
            rows.extend(dict(index=first+j,score_x100=x['score_x100'],total_used=x['total_used'],
                remaining=x['remaining'],plan_sha256=x['plan_sha256']) for j,x in enumerate(results))
        x=np.asarray([r['score_x100'] for r in rows])
        output[name]=dict(score=float(x.mean()),sd=float(x.std(ddof=1)) if len(x)>1 else 0.,rows=rows)
    return output
