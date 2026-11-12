"""Two explicitly named ablations of the immutable F1 update.

F1/no_history call the original FrequentPPO unchanged.  FixedActorLR retains
the original SGD path and discards only the proposed *next* actor LR after the
update.  NoOffpolicyCorrection uses original recorded behavior probabilities
only as diagnostics: its train reference is the frozen pre-update actor, its
clip center is one, and its value/advantage labels are complete MC returns.
"""
from __future__ import annotations

import numpy as np
import torch

from allocation4.common import STATE_KEYS, subset, tensor_batch
from allocation4.methods.lax import gaussian_kl, gaussian_parameters, gaussian_log_prob
from allocation_efficiency.frequent import FrequentPPO, EPSILON, adjusted_lr


class FixedActorLR(FrequentPPO):
    def metadata(self):
        result=super().metadata()
        result.update(formal_variant='fixed_actor_lr',tv_controller='disabled; TV is diagnostic only',
                      actor_lr_fixed=3e-4, value_lr_fixed=1e-3)
        return result

    def update(self,actor,critic,ao,vo,batches,rng):
        if any(float(g['lr'])!=3e-4 for g in ao.param_groups):
            raise ValueError('Fixed-LR ablation must enter each SGD update at 3e-4')
        stats=super().update(actor,critic,ao,vo,batches,rng)
        # The native proposal occurs strictly after all committed SGD steps.
        # Replacing that unused next-update proposal cannot alter this update.
        proposed=stats['actor_lr_after']
        self.controller.update(calls=0,actor_lr=3e-4,last_action='fixed_no_TV_control')
        for group in ao.param_groups:group['lr']=3e-4
        stats.update(actor_lr_after=3e-4,tv_lr_factor=1.,tv_controller_ran=False,
                     tv_controller_calls=0,tv_controller_action='fixed_no_TV_control',
                     diagnostic_native_lr_proposal=proposed,tv_used_for_lr=False)
        return stats


class NoOffpolicyCorrection(FrequentPPO):
    def metadata(self):
        result=super().metadata()
        result.update(formal_variant='no_offpolicy_correction',
                      targets='complete native Monte Carlo return; advantage R - frozen current V',
                      surrogate='differentiable pi_theta / frozen pi_anchor, standard center1 clip',
                      behavior_probability='original complete joint behavior density retained, diagnostic only',
                      offpolicy_training_correction=False,vtrace_used_for_training=False,
                      tv='same behavior-relative post-SGD TV used by full F1',
                      theoretical_scope='naive stale-state/action MC PPO control, intentionally no off-policy correction')
        return result

    def update(self,actor,critic,ao,vo,batches,rng):
        if self.episodes<32:
            stats=super().update(actor,critic,ao,vo,batches,rng)
            stats.update(offpolicy_training_correction=False,vtrace_used_for_training=False)
            return stats
        self._validate_count(batches,4,self.updates)
        self._validate_count(self.history,4,self.updates-1)
        lr_before=self._optimizer_rates(ao,vo)
        combined=list(batches)+list(self.history)
        data=tensor_batch(combined,self.device)
        states,raw,behavior_logp,mask=data['states'],data['raw'],data['old_logp'],data['mask']
        active=mask>0
        behavior_mean=torch.as_tensor(np.concatenate([b['behavior_mean'] for b in combined]),dtype=raw.dtype,device=raw.device)
        behavior_sigma=torch.as_tensor(np.concatenate([b['behavior_sigma'] for b in combined]),dtype=raw.dtype,device=raw.device)
        with torch.no_grad():
            reconstructed=gaussian_log_prob(behavior_mean,behavior_sigma,raw)
            behavior_error=float((reconstructed-behavior_logp).abs().max())
            if behavior_error>1e-7:raise AssertionError('Original stored behavior density changed')
            anchor_mean,anchor_sigma=gaussian_parameters(actor,states)
            anchor_mean,anchor_sigma=anchor_mean.clone(),anchor_sigma.clone()
            anchor_logp=actor.log_prob(states,raw).detach().clone()
            anchor_values=critic(states,raw).detach()
            targets=data['returns'].reshape(-1,1).detach().clone()
            advantage=(targets-anchor_values).detach()
            scale=advantage[active].std(unbiased=False)if active.any()else raw.new_tensor(0.)
            if active.any():advantage=((advantage-advantage[active].mean())/(scale+1e-8)).detach()
            behavioral_anchor_logratio=torch.where(active,anchor_logp-behavior_logp,torch.zeros_like(behavior_logp))
            behavioral_anchor_ratio=behavioral_anchor_logratio.clamp(-40,40).exp()
        records=[];early_stop=False;sgd_guards=0;actor_norms=[];value_norms=[]
        if active.any():
            for _ in range(4):
                indices=self.shuffle_rng.permutation(len(raw))
                for first in range(0,len(raw),64):
                    ix=torch.as_tensor(indices[first:first+64],device=raw.device)
                    st,m=subset(states,ix),mask[ix]
                    lp=actor.log_prob(st,raw[ix])
                    if not torch.isfinite(lp).all():raise FloatingPointError('Nonfinite current density')
                    # This reference is distinct from and never overwrites mu.
                    delta=torch.where(m>0,lp-anchor_logp[ix],torch.zeros_like(lp))
                    sgd_guards+=int(((delta.abs()>40)*(m>0)).sum())
                    ratio=delta.clamp(-40,40).exp()
                    clipped=ratio.clamp(1-EPSILON,1+EPSILON)
                    loss=-(torch.minimum(ratio*advantage[ix],clipped*advantage[ix])*m).sum(-1).mean()
                    ao.zero_grad();loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),.5)
                    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite naive actor gradient')
                    actor_norms.append(float(norm));ao.step()
                    prediction=critic(st,raw[ix])
                    value_loss=(((prediction-targets[ix])**2)*m).sum()/m.sum().clamp_min(1)
                    vo.zero_grad();value_loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(critic.parameters(),.5)
                    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite naive critic gradient')
                    value_norms.append(float(norm));vo.step()
                    with torch.no_grad():
                        kl=((ratio-1-delta)*m).sum(-1).mean()
                        fraction=(((ratio-1).abs()>EPSILON)*m).sum()/m.sum().clamp_min(1)
                    records.append((float(loss),float(value_loss),float(kl),float(fraction)))
                    if float(kl)>.12:early_stop=True;break
                if early_stop:break
        with torch.no_grad():
            final_lp=actor.log_prob(states,raw)
            final_mean,final_sigma=gaussian_parameters(actor,states)
            final_behavior_logratio=torch.where(active,final_lp-behavior_logp,torch.zeros_like(behavior_logp))
            final_behavior_ratio=final_behavior_logratio.clamp(-40,40).exp()
            # Keep the original TV control; only training credit/probability
            # correction is ablated, not the stability controller.
            tv=float(.5*((final_behavior_ratio-behavioral_anchor_ratio).abs()*mask).sum()/mask.sum().clamp_min(1))
            akl=gaussian_kl(anchor_mean,anchor_sigma,final_mean,final_sigma)[:,None]
            bkl=gaussian_kl(behavior_mean,behavior_sigma,anchor_mean,anchor_sigma)[:,None]
            fresh_count=4*self.spec.horizon
            hm=mask[fresh_count:]
            old_kl=float(((bkl[fresh_count:]*hm).sum()/hm.sum().clamp_min(1)).clamp_min(0))
            history_active=int(hm.sum())
            hl=behavioral_anchor_logratio[fresh_count:][hm>0]
            hess=None
            if len(hl):
                hn=(hl-torch.logsumexp(hl,0)).exp();hess=float(1/hn.square().sum())/len(hl)
            logs=behavioral_anchor_logratio[active]
            ess=0.
            if len(logs):
                normalized=(logs-torch.logsumexp(logs,0)).exp();ess=float(1/normalized.square().sum())
            trajectory_logs=[];offset=0
            for length in data['lengths']:
                sl=slice(offset,offset+length);trajectory_logs.append((behavioral_anchor_logratio[sl]*mask[sl]).sum());offset+=length
            tl=torch.stack(trajectory_logs);tn=(tl-torch.logsumexp(tl,0)).exp()
            ht=tl[4:];htn=(ht-torch.logsumexp(ht,0)).exp()
            fresh_error=float(((anchor_logp[:fresh_count]-behavior_logp[:fresh_count])*mask[:fresh_count]).abs().max())
        lr_after,action=adjusted_lr(lr_before,tv)
        self.controller.update(calls=self.controller['calls']+1,actor_lr=lr_after,last_tv=tv,last_lr_before=lr_before,last_action=action)
        for g in ao.param_groups:g['lr']=lr_after
        self._finish(batches)
        a=np.asarray(records)if records else np.zeros((1,4))
        return dict(actor_loss=float(a[:,0].mean()),value_loss=float(a[:,1].mean()),macro_kl=float(a[:,2].mean()),
            clip_fraction=float(a[:,3].mean()),optimizer_steps=len(records),advantage_std=float(scale),early_stop=early_stop,
            actor_grad_norm=float(np.mean(actor_norms))if actor_norms else 0.,value_grad_norm=float(np.mean(value_norms))if value_norms else 0.,
            warmup=False,replay_enabled=True,new_episodes=4,learning_episodes=8,
            likelihood_error=fresh_error,behavior_likelihood_error=behavior_error,
            behavior_to_anchor_kl=float(((bkl*mask).sum()/mask.sum().clamp_min(1)).clamp_min(0)),old_policy_kl=old_kl,
            actual_joint_kl=float(akl[:fresh_count].mean().clamp_min(0)),anchor_kl=float(((akl*mask).sum()/mask.sum().clamp_min(1)).clamp_min(0)),
            ess=ess,ess_fraction=ess/max(int(active.sum()),1),history_only_ess_fraction=hess,
            trajectory_ess=float(1/tn.square().sum()),trajectory_ess_fraction=float(1/tn.square().sum())/8,
            history_only_trajectory_ess_fraction=float(1/htn.square().sum())/4,
            history_only_active_events=history_active,
            max_importance_weight=float(logs.clamp(-40,40).exp().max())if len(logs)else 0.,
            rho_capped_fraction=float((logs>1e-12).double().mean())if len(logs)else 0.,
            zeroevent_count=int((mask==0).sum()),active_events=int(active.sum()),sgd_ratio_guard_count=sgd_guards,
            replay_episodes_read=4,replay_decisions_read=4*self.spec.horizon,current_episode_repeats=0,
            buffer_episodes=4,mean_policy_age=.5,mean_age_plus_one=1.5,generalized_clip=EPSILON,
            candidate_policy_events=0,vtrace_target_recomputations=0,vtrace_used_for_training=False,
            offpolicy_training_correction=False,training_clip_center=1.,stored_behavior_preserved=True,
            post_final_tv=tv,tv=tv,tv_target=EPSILON/2,tv_lower_target=EPSILON/4,
            actor_lr_before=lr_before,actor_lr_after=lr_after,tv_lr_factor=lr_after/lr_before,
            tv_controller_action=action,tv_controller_calls=self.controller['calls'],tv_controller_ran=True,
            episodes_seen=self.episodes,updates=self.updates)
