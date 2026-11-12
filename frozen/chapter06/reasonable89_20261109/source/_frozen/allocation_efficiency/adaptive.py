"""One preregistered old-policy TV gate; finite-horizon heuristic adaptation.

A0 uses the same historical-policy gate as A1 but never uses old trajectories
in a gradient batch. Thus A0 is not a history-information-free comparator.
No gate examines a new episode's reward, no random rollout is made by the gate,
and no hard-TV, unbiased-policy-gradient, or original-theory claim is made.
"""
from __future__ import annotations
import copy
import hashlib
import math

import numpy as np
import torch

from allocation23.learning import PPO
from allocation4.common import STATE_KEYS,model_hash,subset
from allocation4.methods.lax import gaussian_parameters,gaussian_log_prob,gaussian_kl
from .frequent import FrequentPPO,EPSILON,TV_ALPHA,TV_BETA,ACTOR_LR_BOUNDS,SHUFFLE_SEED
from .geppo import generalized_clip,_same_json

VERSION='old-TV-gated-4or8-new-adaptation-v1'
GATE_THRESHOLD=EPSILON/2


def adaptive_lr(lr,tv,epsilon):
    if not all(isinstance(x,(float,int)) and math.isfinite(x) for x in (lr,tv,epsilon)) or tv<0 or epsilon<=0:
        raise ValueError('Finite registered TV/LR values required')
    if not ACTOR_LR_BOUNDS[0]<=lr<=ACTOR_LR_BOUNDS[1]:raise ValueError('Registered actor-LR bounds')
    target=epsilon/2
    action='decrease' if tv>target else 'increase' if tv<TV_BETA*target else 'hold'
    proposed=lr/(1+TV_ALPHA) if action=='decrease' else lr*(1+TV_ALPHA) if action=='increase' else lr
    return float(np.clip(proposed,*ACTOR_LR_BOUNDS)),action


def _ess(logweights):
    if not len(logweights):return None,None
    normalized=(logweights-torch.logsumexp(logweights,0)).exp()
    ess=float(1/normalized.square().sum())
    return ess,ess/len(logweights)


def _history_policy_digest(history):
    """Only gate-eligible fields; rewards/returns are deliberately not read."""
    h=hashlib.sha256()
    for batch in history:
        for key in STATE_KEYS:
            a=np.asarray(batch['states'][key]);h.update(key.encode());h.update(a.tobytes())
        for key in ('raw','mask','behavior_mean','behavior_sigma'):
            a=np.asarray(batch[key]);h.update(key.encode());h.update(a.tobytes())
        h.update(str(batch['behavior_version']).encode())
    return h.hexdigest()


class AdaptivePPO(FrequentPPO):
    def __init__(self,spec,event_dim,seed,device,replay=True):
        super().__init__(spec,event_dim,seed,device,replay=replay)
        self.controller['last_epsilon']=None
        self.batch_sizes=[]
        self.pending_plan=None
        self.collection_plans=0
        self.gate_evaluations=0

    def metadata(self):
        m=super().metadata()
        m.update(version=VERSION,replay=self.replay,
            fresh_episodes_after_warmup='4 if old TV<=fixed threshold, otherwise8; boundary may cap to4',
            learning_population='accepted: current4+previous4' if self.replay else 'accepted: current4 duplicated; history gate still used',
            fallback='current8 only, or boundary current4 duplicated; zero old gradient weight',
            fixed_backend='A0/A1 same gate, clipping and controller; only old-gradient participation differs',
            gate_threshold=GATE_THRESHOLD,gate_formula='.5 mean_active abs(pi_current/mu_previous4 -1)',
            gate_inputs='previous4 public states/raw/mask/behavior mean+sigma only; no returns/rewards',
            gate_zero_active='reject old; no likelihood forward, driftNone',
            gate_plan='cached before collection; update checks actor hash, version, history identity and actual count',
            generalized_clip='accept gate: .2/1.5 in BOTH arms; reject gate: .2',
            epochs='warmup4; postwarmactual_new4or8',
            epochs_after_warmup='actual newly collected episode count4or8',
            tv_target='branch-dependent: current clip/2 (.0666667 reuse or .1 fallback)',
            controller_target='current chosen clip /2; gate threshold remains fixed .0666667',
            a0_scope='history affects gate but NEVER gradient targets; not an information-free control',
            theory='single preregistered heuristic finite-gamma1 adaptation; no hard-TV/original GePPO theorem/unbiasedness claim')
        return m

    @torch.no_grad()
    def _old_gate(self,actor):
        if len(self.history)!=4 or any(b['behavior_version']!=self.updates-1 for b in self.history):
            raise ValueError('The gate requires immediately preceding four episodes')
        # Do not call the complete episode validator here: it reads rewards.
        states={k:torch.as_tensor(np.concatenate([b['states'][k] for b in self.history]),dtype=torch.float64,device=self.device) for k in STATE_KEYS}
        raw,mask,mu,sigma=[torch.as_tensor(np.concatenate([b[k] for b in self.history]),dtype=torch.float64,device=self.device)
                           for k in ('raw','mask','behavior_mean','behavior_sigma')]
        active=mask>0;count=int(active.sum())
        if not count:
            return dict(gate_drift=None,gate_active_events=0,gate_ratio_guard_count=0,
                gate_policy_evaluations=0,gate_history_ess_fraction=None,
                gate_history_trajectory_ess_fraction=None,gate_old_policy_kl=None,
                gate_accepted=False,gate_reason='no_active_history')
        cm,cs=gaussian_parameters(actor,states)
        logs=gaussian_log_prob(cm,cs,raw)-gaussian_log_prob(mu,sigma,raw)
        if not torch.isfinite(logs).all():raise FloatingPointError('Finite full-joint gate likelihoods required')
        effective=torch.where(active,logs,torch.zeros_like(logs))
        ratio=effective.clamp(-40,40).exp()
        drift=float(.5*((ratio-1).abs()*mask).sum()/mask.sum())
        _,ef=_ess(logs[active])
        trajectory=torch.stack([(effective[j*self.spec.horizon:(j+1)*self.spec.horizon]).sum() for j in range(4)])
        _,tf=_ess(trajectory)
        kl=gaussian_kl(mu,sigma,cm,cs)[:,None]
        old_kl=float(((kl*mask).sum()/mask.sum()).clamp_min(0))
        accept=drift<=GATE_THRESHOLD
        return dict(gate_drift=drift,gate_active_events=count,
            gate_ratio_guard_count=int(((logs.abs()>40)*active).sum()),gate_policy_evaluations=len(raw),
            gate_history_ess_fraction=ef,gate_history_trajectory_ess_fraction=tf,
            gate_old_policy_kl=old_kl,gate_accepted=accept,
            gate_reason='within_threshold' if accept else 'drift_reject')

    def collection_size(self,actor,max_new):
        if isinstance(max_new,bool) or max_new not in (4,8):raise ValueError('Runner must supply an evaluation-safe4or8 limit')
        actor_hash=model_hash(actor)
        if self.pending_plan is not None:
            p=self.pending_plan
            if (p['actor_sha256']!=actor_hash or p['version']!=self.updates or
                p['episodes_before']!=self.episodes or p['max_new']!=max_new or
                (not p['warmup'] and p['history_policy_sha256']!=_history_policy_digest(self.history))):
                raise ValueError('A cached collection plan cannot be changed before its update')
            return p['actual_new']
        warm=self.episodes<32
        if warm:
            if max_new!=8:raise ValueError('The first32 episodes require native eight-episode collections')
            gate=dict(gate_drift=None,gate_active_events=0,gate_ratio_guard_count=0,
                gate_policy_evaluations=0,gate_history_ess_fraction=None,
                gate_history_trajectory_ess_fraction=None,gate_old_policy_kl=None,
                gate_accepted=False,gate_reason='native_warmup')
            desired=8
        else:
            gate=self._old_gate(actor);self.gate_evaluations+=1
            desired=4 if gate['gate_accepted'] else 8
        actual=min(desired,max_new)
        self.pending_plan=dict(gate,warmup=warm,version=self.updates,episodes_before=self.episodes,
            actor_sha256=actor_hash,history_policy_sha256=None if warm else _history_policy_digest(self.history),
            gate_threshold=GATE_THRESHOLD,desired_new=desired,actual_new=actual,max_new=int(max_new),
            boundary_cap=actual<desired,gate_read_count=0 if warm else 4,
            gate_read_decisions=0 if warm else 4*self.spec.horizon)
        self.collection_plans+=1
        return actual

    def _finish(self,batches):
        self.batch_sizes.append(len(batches))
        super()._finish(batches)
        self.pending_plan=None

    def update(self,actor,critic,ao,vo,batches,rng):
        if self.pending_plan is None:raise ValueError('Call collection_size before collecting a new batch')
        plan=copy.deepcopy(self.pending_plan)
        if plan['version']!=self.updates or plan['episodes_before']!=self.episodes or plan['actor_sha256']!=model_hash(actor):
            raise ValueError('Actor/version changed after the cached pre-collection gate')
        if not plan['warmup'] and plan['history_policy_sha256']!=_history_policy_digest(self.history):
            raise ValueError('Gated historical policy data changed before update')
        self._validate_count(batches,plan['actual_new'],self.updates)
        lr_before=self._optimizer_rates(ao,vo)
        gate_log={k:v for k,v in plan.items() if k not in ('actor_sha256','history_policy_sha256','version','episodes_before','max_new')}
        if plan['warmup']:
            stats=self.native.update(actor,critic,ao,vo,batches,rng)
            self._finish(batches)
            stats.update(gate_log,new_episodes=8,learning_episodes=8,replay_enabled=False,
                old_in_gradient=False,oldweight=0.,clip_reference_oldweight=0.,
                generalized_clip=.2,epochs=4,replay_episodes_read=0,replay_decisions_read=0,
                current_episode_repeats=0,buffer_episodes=4,mean_policy_age=0.,candidate_policy_events=0,
                actor_lr_before=lr_before,actor_lr_after=lr_before,tv_controller_calls=0,tv_controller_ran=False,
                history_only_ess_fraction=None,history_only_trajectory_ess_fraction=None,
                history_only_active_events=0,history_only_trajectory_active_count=0,
                episodes_seen=self.episodes,updates=self.updates)
            return stats
        actual=plan['actual_new'];reuse=plan['gate_accepted'];old_in_gradient=reuse and self.replay
        oldweight=.5 if old_in_gradient else 0.
        reference_weight=.5 if reuse else 0.
        epsilon=.2/(1+reference_weight)
        combined=list(batches)+(list(self.history) if old_in_gradient else list(batches)) if actual==4 else list(batches)
        if len(combined)!=8:raise AssertionError('Every learning batch must contain exactly eight episodes')
        data,behavior_error=self._prepare(actor,critic,combined)
        states,raw,old,mask=data['states'],data['raw'],data['old_logp'],data['mask'];active=mask>0
        fresh_count=actual*self.spec.horizon
        error=float(((data['anchor_logp'][:fresh_count]-old[:fresh_count])*mask[:fresh_count]).abs().max())
        if error>1e-7:raise AssertionError(('Fresh behavior likelihood',error))
        mc=torch.as_tensor(np.concatenate([b['returns'] for b in batches]),dtype=raw.dtype,device=raw.device)[:,None]
        mc_error=float((data['targets'][:fresh_count]-mc).abs().max())
        if not old_in_gradient:
            expected=torch.cat((mc,mc)) if actual==4 else mc
            mc_error=max(mc_error,float((data['targets']-expected).abs().max()))
        if mc_error>1e-10:raise AssertionError(('Fresh on-policy MC identity',mc_error))
        advantage=data['advantage'];scale=advantage[active].std(unbiased=False) if active.any() else raw.new_tensor(0.)
        if active.any():advantage=((advantage-advantage[active].mean())/(scale+1e-8)).detach()
        anchor_logratio=torch.where(active,data['anchor_logp']-old,torch.zeros_like(old))
        center=anchor_logratio.clamp(-40,40).exp().detach();records=[];stop=False;guards=0
        if active.any():
            for _ in range(actual):
                permutation=self.shuffle_rng.permutation(len(raw))
                for first in range(0,len(raw),64):
                    ix=torch.as_tensor(permutation[first:first+64],device=raw.device);st,m=subset(states,ix),mask[ix]
                    lp=actor.log_prob(st,raw[ix])
                    if not torch.isfinite(lp).all():raise FloatingPointError('Finite current Gaussian density required')
                    logs=torch.where(m>0,lp-old[ix],torch.zeros_like(lp));guards+=int(((logs.abs()>40)*(m>0)).sum())
                    ratio=logs.clamp(-40,40).exp();clipped=generalized_clip(ratio,center[ix],epsilon)
                    loss=-(torch.minimum(ratio*advantage[ix],clipped*advantage[ix])*m).sum(-1).mean()
                    ao.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),.5)
                    if not torch.isfinite(norm):raise FloatingPointError('Finite adaptive actor gradient required')
                    ao.step();pred=critic(st,raw[ix]);vl=(((pred-data['targets'][ix])**2)*m).sum()/m.sum().clamp_min(1)
                    vo.zero_grad();vl.backward();norm=torch.nn.utils.clip_grad_norm_(critic.parameters(),.5)
                    if not torch.isfinite(norm):raise FloatingPointError('Finite adaptive value gradient required')
                    vo.step()
                    with torch.no_grad():
                        delta=torch.where(m>0,lp-data['anchor_logp'][ix],torch.zeros_like(lp))
                        kl=((delta.clamp(-40,40).exp()-1-delta)*m).sum(-1).mean()
                        cf=(((ratio-center[ix]).abs()>epsilon)*m).sum()/m.sum().clamp_min(1)
                    records.append((float(loss),float(vl),float(kl),float(cf)))
                    if float(kl)>.12:stop=True;break
                if stop:break
        with torch.no_grad():
            final_lp=actor.log_prob(states,raw);final_logs=torch.where(active,final_lp-old,torch.zeros_like(old))
            if not torch.isfinite(final_logs).all():raise FloatingPointError('Finite final TV likelihood required')
            tv=float(.5*((final_logs.clamp(-40,40).exp()-center).abs()*mask).sum()/mask.sum().clamp_min(1))
            fm,fs=gaussian_parameters(actor,states)
            akl=gaussian_kl(data['anchor_mean'],data['anchor_sigma'],fm,fs)[:,None]
            bkl=gaussian_kl(data['behavior_mean'],data['behavior_sigma'],data['anchor_mean'],data['anchor_sigma'])[:,None]
            anchor_kl=float(((akl*mask).sum()/mask.sum().clamp_min(1)).clamp_min(0));fresh_kl=float(akl[:fresh_count].mean().clamp_min(0))
            behavior_kl=float(((bkl*mask).sum()/mask.sum().clamp_min(1)).clamp_min(0))
            mixed_ess,mixed_fraction=_ess(data['log_rhos'][active.reshape(-1)])
            trajectory_logs=torch.stack([(data['log_rhos'][j*self.spec.horizon:(j+1)*self.spec.horizon]*mask[j*self.spec.horizon:(j+1)*self.spec.horizon].reshape(-1)).sum() for j in range(8)])
            trajectory_ess,trajectory_fraction=_ess(trajectory_logs)
            history_fraction=history_trajectory_fraction=None;history_active=history_active_episodes=0;old_kl=0.
            if old_in_gradient:
                hm=active[fresh_count:].reshape(-1);history_active=int(hm.sum())
                _,history_fraction=_ess(data['log_rhos'][fresh_count:][hm]);_,history_trajectory_fraction=_ess(trajectory_logs[4:])
                old_mask=mask[fresh_count:];old_kl=float(((bkl[fresh_count:]*old_mask).sum()/old_mask.sum().clamp_min(1)).clamp_min(0))
                history_active_episodes=sum(bool(mask[(4+j)*self.spec.horizon:(5+j)*self.spec.horizon].any()) for j in range(4))
        lr_after,action=adaptive_lr(lr_before,tv,epsilon)
        self.controller.update(calls=self.controller['calls']+1,actor_lr=lr_after,last_tv=tv,
            last_lr_before=lr_before,last_action=action,last_epsilon=epsilon)
        for group in ao.param_groups:group['lr']=lr_after
        self._finish(batches);a=np.asarray(records) if records else np.zeros((1,4))
        result=dict(actor_loss=float(a[:,0].mean()),value_loss=float(a[:,1].mean()),macro_kl=float(a[:,2].mean()),
            clip_fraction=float(a[:,3].mean()),optimizer_steps=len(records),advantage_std=float(scale),early_stop=stop,
            warmup=False,replay_enabled=old_in_gradient,new_episodes=actual,learning_episodes=8,
            old_in_gradient=old_in_gradient,oldweight=oldweight,clip_reference_oldweight=reference_weight,
            generalized_clip=epsilon,epochs=actual,likelihood_error=error,behavior_likelihood_error=behavior_error,
            fresh_mc_target_error=mc_error,actual_joint_kl=fresh_kl,anchor_kl=anchor_kl,
            mixture_anchor_kl_active=anchor_kl,old_policy_kl=old_kl,behavior_to_anchor_kl=behavior_kl,
            ess=0. if mixed_ess is None else mixed_ess,ess_fraction=0. if mixed_fraction is None else mixed_fraction,
            trajectory_ess=trajectory_ess,trajectory_ess_fraction=trajectory_fraction,
            history_only_ess_fraction=history_fraction,history_only_trajectory_ess_fraction=history_trajectory_fraction,
            history_only_active_events=history_active,history_only_trajectory_active_count=history_active_episodes,
            zeroevent_count=int((mask==0).sum()),active_events=int(active.sum()),
            anchor_ratio_guard_count=int(((anchor_logratio.abs()>40)*active).sum()),
            final_ratio_guard_count=int(((final_logs.abs()>40)*active).sum()),sgd_ratio_guard_count=guards,
            replay_episodes_read=4 if old_in_gradient else 0,replay_decisions_read=4*self.spec.horizon if old_in_gradient else 0,
            current_episode_repeats=4 if actual==4 and not old_in_gradient else 0,buffer_episodes=4,
            mean_policy_age=.5 if old_in_gradient else 0.,candidate_policy_events=0,vtrace_target_recomputations=1,
            post_final_tv=tv,tv=tv,tv_target=epsilon/2,tv_lower_target=TV_BETA*epsilon/2,
            actor_lr_before=lr_before,actor_lr_after=lr_after,tv_lr_factor=lr_after/lr_before,
            tv_controller_action=action,tv_controller_calls=self.controller['calls'],tv_controller_ran=True,
            episodes_seen=self.episodes,updates=self.updates)
        result.update(gate_log);return result

    def state_dict(self):
        return dict(metadata=self.metadata(),episodes=self.episodes,updates=self.updates,
            history=copy.deepcopy(self.history),shuffle_rng=copy.deepcopy(self.shuffle_rng.bit_generator.state),
            controller=copy.deepcopy(self.controller),batch_sizes=list(self.batch_sizes),
            pending_plan=copy.deepcopy(self.pending_plan),collection_plans=self.collection_plans,
            gate_evaluations=self.gate_evaluations)

    def load_state_dict(self,state):
        keys={'metadata','episodes','updates','history','shuffle_rng','controller','batch_sizes','pending_plan','collection_plans','gate_evaluations'}
        if set(state)!=keys or not _same_json(state['metadata'],self.metadata()):raise ValueError('Adaptive checkpoint configuration mismatch')
        sizes=list(state['batch_sizes']);updates=state['updates'];episodes=state['episodes']
        if (isinstance(updates,bool) or not isinstance(updates,int) or updates<0 or len(sizes)!=updates or
            sizes[:min(updates,4)]!=[8]*min(updates,4) or any(x not in (4,8) for x in sizes[4:]) or
            isinstance(episodes,bool) or not isinstance(episodes,int) or episodes!=sum(sizes)):
            raise ValueError('Dynamic batch-size chronology mismatch')
        history=copy.deepcopy(state['history'])
        if len(history)!=(4 if updates else 0):raise ValueError('Last four episodes required')
        if history:self._validate_count(history,4,updates-1)
        controller=copy.deepcopy(state['controller'])
        if set(controller)!={'calls','actor_lr','last_tv','last_lr_before','last_action','last_epsilon'} or controller['calls']!=max(updates-4,0):raise ValueError('Controller chronology mismatch')
        if updates<=4:
            expected=dict(calls=0,actor_lr=3e-4,last_tv=None,last_lr_before=None,last_action='not_started',last_epsilon=None)
            if controller!=expected:raise ValueError('Warmup must retain native LR and untouched controller')
        else:
            if controller['last_epsilon'] not in (EPSILON,.2):raise ValueError('Registered adaptive epsilon required')
            wanted,action=adaptive_lr(controller['last_lr_before'],controller['last_tv'],controller['last_epsilon'])
            if controller['last_action']!=action or not math.isclose(controller['actor_lr'],wanted,abs_tol=1e-15,rel_tol=0):raise ValueError('Saved controller transition mismatch')
        pending=copy.deepcopy(state['pending_plan'])
        if state['collection_plans']!=updates+int(pending is not None) or state['gate_evaluations']!=max(updates-4,0)+int(pending is not None and not pending['warmup']):raise ValueError('Cached gate chronology mismatch')
        if pending is not None:
            if (pending['version']!=updates or pending['episodes_before']!=episodes or
                pending['actual_new']!=min(pending['desired_new'],pending['max_new']) or
                pending['boundary_cap']!=(pending['actual_new']<pending['desired_new']) or
                pending['gate_threshold']!=GATE_THRESHOLD or
                (not pending['warmup'] and pending['history_policy_sha256']!=_history_policy_digest(history))):
                raise ValueError('Cached pre-collection plan mismatch')
        probe=np.random.default_rng();probe.bit_generator.state=copy.deepcopy(state['shuffle_rng'])
        if updates<=4 and not _same_json(probe.bit_generator.state,np.random.default_rng(SHUFFLE_SEED+self.seed).bit_generator.state):raise ValueError('Private shuffle must not advance in warmup')
        self.episodes,self.updates,self.history=episodes,updates,history
        self.controller,self.shuffle_rng=controller,probe
        self.batch_sizes,self.pending_plan=sizes,pending
        self.collection_plans,self.gate_evaluations=state['collection_plans'],state['gate_evaluations']
