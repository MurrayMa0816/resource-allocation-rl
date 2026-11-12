"""PPO update over public states and complete recorded Gaussian events.

The training runner supplies native terminal Monte Carlo returns with gamma=1.
Neither the update nor any agent model imports a physical environment.
"""
import numpy as np
import torch
PPO=dict(batch_episodes=8,epochs=4,minibatch=64,actor_lr=3e-4,value_lr=1e-3,clip=.2,grad_clip=.5,macro_kl_stop=.12,gamma=1.,entropy_bonus=0.)

def update(actor,critic,ao,vo,batches,rng):
    states={k:torch.as_tensor(np.concatenate([b['states'][k] for b in batches]),dtype=torch.float64) for k in batches[0]['states']}
    data={k:torch.as_tensor(np.concatenate([b[k] for b in batches]),dtype=torch.float64)
          for k in ('raw','old_logp','mask','values','returns')}
    raw,old,mask=data['raw'],data['old_logp'],data['mask']
    adv=data['returns'][:,None]-data['values'];active=mask>0
    if not torch.any(active):
        # An exhausted budget or fully capped terminal batch carries no active
        # policy event. Leave models, optimizers and the shuffle RNG untouched.
        return dict(likelihood_error=0.,optimizer_steps=0,actor_loss=0.,value_loss=0.,
                    macro_kl=0.,clip_fraction=0.,actor_grad_norm=0.,value_grad_norm=0.,
                    early_stop=False,active_events=0.,advantage_std=0.)
    advantage_mean=adv[active].mean();advantage_std=adv[active].std(unbiased=False)
    adv=(adv-advantage_mean)/(advantage_std+1e-8)
    with torch.no_grad():recompute=float(((actor.log_prob(states,raw)-old)*mask).abs().max())
    if recompute>1e-7:raise AssertionError(('stored likelihood mismatch',recompute))
    stats=[];stop=False
    for epoch in range(PPO['epochs']):
        indices=rng.permutation(len(raw))
        for offset in range(0,len(raw),PPO['minibatch']):
            ix=torch.as_tensor(indices[offset:offset+PPO['minibatch']],dtype=torch.long)
            s={k:v[ix] for k,v in states.items()};m=mask[ix];oldp=old[ix]
            lp=actor.log_prob(s,raw[ix]);logratio=lp-oldp
            if not torch.isfinite(lp).all():raise FloatingPointError('nonfinite recomputed logp')
            ratio=logratio.clamp(-40,40).exp()
            obj=torch.minimum(ratio*adv[ix],ratio.clamp(1-PPO['clip'],1+PPO['clip'])*adv[ix])
            # Sum conditional policy-gradient contributions per physical decision.
            loss=-(obj*m).sum(-1).mean()
            ao.zero_grad();loss.backward();agn=torch.nn.utils.clip_grad_norm_(actor.parameters(),PPO['grad_clip'])
            if not torch.isfinite(agn):raise FloatingPointError('actor gradient nonfinite')
            ao.step()
            pred=critic(s,raw[ix]);vl=(((pred-data['returns'][ix,None])**2)*m).sum()/m.sum().clamp_min(1)
            vo.zero_grad();vl.backward();vgn=torch.nn.utils.clip_grad_norm_(critic.parameters(),PPO['grad_clip'])
            if not torch.isfinite(vgn):raise FloatingPointError('value gradient nonfinite')
            vo.step()
            with torch.no_grad():
                kl=((ratio-1-logratio)*m).sum(-1).mean()
                clip=(((ratio-1).abs()>.2)*m).sum()/m.sum().clamp_min(1)
            stats.append([float(loss),float(vl),float(kl),float(clip),float(agn),float(vgn)])
            if float(kl)>PPO['macro_kl_stop']:stop=True;break
        if stop:break
    a=np.asarray(stats)
    return dict(likelihood_error=recompute,optimizer_steps=len(stats),actor_loss=float(a[:,0].mean()),
        value_loss=float(a[:,1].mean()),macro_kl=float(a[:,2].mean()),clip_fraction=float(a[:,3].mean()),
        actor_grad_norm=float(a[:,4].mean()),value_grad_norm=float(a[:,5].mean()),early_stop=stop,
        active_events=float(mask.sum()),advantage_std=float(advantage_std))
