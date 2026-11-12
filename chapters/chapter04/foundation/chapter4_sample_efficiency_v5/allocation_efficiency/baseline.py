"""Original MC-PPO and a current-batch-only additional-compute control."""
import numpy as np
import torch
from allocation4.common import tensor_batch,subset
from allocation4.methods.baseline import MCPPO
from allocation23.learning import PPO

class NativePPO:
    def __init__(self,spec,event_dim,seed,device,*,extra=False):
        self.spec,self.event_dim,self.seed,self.device=spec,event_dim,seed,str(device)
        self.extra,self.episodes=bool(extra),0
        self.native=MCPPO(spec,event_dim,seed,device)

    def metadata(self):
        return dict(version='native-PPO-plus-current-epochs-v1',extra=self.extra,
                    warmup=32,epochs_after_warmup=8 if self.extra else 4,
                    old_trajectory_replay=False,native_ppo=PPO)

    def update(self,actor,critic,ao,vo,batches,rng):
        if not self.extra or self.episodes<32:
            result=self.native.update(actor,critic,ao,vo,batches,rng)
        else:
            d=tensor_batch(batches,self.device)
            s,raw,old,mask=d['states'],d['raw'],d['old_logp'],d['mask']
            active=mask>0
            if not active.any():
                result=dict(actor_loss=0.,value_loss=0.,macro_kl=0.,optimizer_steps=0)
            else:
                adv=d['returns'][:,None]-d['values'];sd=adv[active].std(unbiased=False)
                adv=(adv-adv[active].mean())/(sd+1e-8)
                error=float(((actor.log_prob(s,raw)-old)*mask).abs().max().detach())
                if error>1e-7:raise AssertionError(('likelihood mismatch',error))
                stats=[];stop=False
                for epoch in range(8):
                    indices=rng.permutation(len(raw))
                    for offset in range(0,len(raw),PPO['minibatch']):
                        ix=torch.as_tensor(indices[offset:offset+PPO['minibatch']],device=self.device)
                        st,m=subset(s,ix),mask[ix];lp=actor.log_prob(st,raw[ix])
                        logratio=lp-old[ix];ratio=logratio.clamp(-40,40).exp()
                        obj=torch.minimum(ratio*adv[ix],ratio.clamp(1-PPO['clip'],1+PPO['clip'])*adv[ix])
                        loss=-(obj*m).sum(-1).mean()
                        ao.zero_grad();loss.backward()
                        norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),PPO['grad_clip'])
                        if not torch.isfinite(norm):raise FloatingPointError('actor gradient')
                        ao.step();pred=critic(st,raw[ix])
                        vl=(((pred-d['returns'][ix,None])**2)*m).sum()/m.sum().clamp_min(1)
                        vo.zero_grad();vl.backward()
                        norm=torch.nn.utils.clip_grad_norm_(critic.parameters(),PPO['grad_clip'])
                        if not torch.isfinite(norm):raise FloatingPointError('critic gradient')
                        vo.step()
                        kl=((ratio-1-logratio)*m).sum(-1).mean().detach()
                        stats.append([float(loss),float(vl),float(kl)])
                        if float(kl)>PPO['macro_kl_stop']:stop=True;break
                    if stop:break
                a=np.asarray(stats)
                result=dict(actor_loss=float(a[:,0].mean()),value_loss=float(a[:,1].mean()),
                            macro_kl=float(a[:,2].mean()),optimizer_steps=len(a),
                            advantage_std=float(sd),likelihood_error=error)
        self.episodes+=len(batches)
        result.update(episodes_seen=self.episodes,candidate_policy_events=0,
                      replay_episodes_read=0,buffer_episodes=0)
        return result

    def state_dict(self):return dict(extra=self.extra,episodes=self.episodes)
    def load_state_dict(self,state):
        if bool(state['extra'])!=self.extra:raise ValueError('baseline mode mismatch')
        self.episodes=int(state['episodes'])
