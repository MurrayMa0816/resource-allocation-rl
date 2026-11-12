"""Matched frozen-state-baseline snapshot PG with KL-sized natural steps.

P2 and S2 share sampling, critic fitting, the frozen cycle baseline, and the
natural-gradient solver. S2 alone uses the finite reference gradient plus the
exact trajectory-weighted gradient difference. This is a finite-horizon
adaptation, not a conditional-unbiasedness or monotonic-improvement theorem.
Natural updates write parameters directly; Adam is used only by native warmup
and its state is deliberately not advanced afterward. All such steps are
reported separately from optimizer calls.
"""
from __future__ import annotations
import copy
import math
import numpy as np
import torch

from allocation4.common import cpu_copy, tensor_batch, subset
from allocation4.methods.lax import gaussian_parameters, gaussian_kl, fit_mc_critic
from .svrpg import (SnapshotPG, trajectory_log_weights, variance_trace, _nullable,
                    REFERENCE_EPISODES, INNER_BATCH_EPISODES, INNER_UPDATES)

VERSION = 'finite-terminal-frozen-baseline-natural-snapshot-reference16-inner4x8-v1'
NATURAL_KL = .01
FISHER_DAMPING = .01
CG_ITERATIONS = 10
CG_RELATIVE_TOLERANCE = 1e-10
BACKTRACKS = 8
CRITIC_EPOCHS = 4


def _flat(values, parameters):
    return torch.cat([torch.zeros_like(p).reshape(-1) if v is None else v.reshape(-1)
                      for v, p in zip(values, parameters)])


def baseline_episode_gradients(actor, baseline, batches, device):
    """h=Sum_t (native terminal R - fixed b(s_t))*grad log pi(z_t|s_t)."""
    result=[]
    for batch in batches:
        data=tensor_batch([batch], device)
        with torch.no_grad():
            b=baseline(data['states']).detach()
        if b.shape != data['values'].shape or not torch.isfinite(b).all():
            raise ValueError('Frozen baseline must give one finite state-only value')
        residual=data['returns'][:,None]-b
        objective=(actor.log_prob(data['states'],data['raw'])*data['mask']*residual).sum()
        params=tuple(actor.parameters())
        gradient=_flat(torch.autograd.grad(objective,params,allow_unused=True),params).detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('Finite baseline trajectory gradients required')
        result.append(gradient)
    if not result:
        raise ValueError('A nonempty complete trajectory population required')
    return torch.stack(result)


def conjugate_gradient(product, b, iterations=CG_ITERATIONS, relative_tolerance=CG_RELATIVE_TOLERANCE):
    """Small fixed-budget damped-Fisher solve; returns work and residuals."""
    x=torch.zeros_like(b);r=b.detach().clone();p=r.clone()
    initial=float(r.square().sum());rr=initial;used=0;reason='iteration_limit'
    if not math.isfinite(initial):
        raise FloatingPointError('Finite CG right-hand side required')
    if initial == 0:
        return x,dict(cg_iterations=0,cg_residual_squared=0.,cg_reason='zero_gradient')
    for _ in range(iterations):
        ap=product(p);used+=1
        pap=float(torch.dot(p,ap))
        if not math.isfinite(pap) or pap <= 0:
            reason='nonpositive_curvature';break
        alpha=rr/pap
        x=x+alpha*p;r=r-alpha*ap
        next_rr=float(r.square().sum())
        if not math.isfinite(next_rr):
            reason='nonfinite_residual';break
        if next_rr <= relative_tolerance*initial:
            rr=next_rr;reason='relative_tolerance';break
        p=r+(next_rr/rr)*p;rr=next_rr
    return x.detach(),dict(cg_iterations=used,cg_residual_squared=rr,cg_reason=reason)


def natural_step(actor, gradient, states, *, kl_limit=NATURAL_KL,
                 damping=FISHER_DAMPING, cg_iterations=CG_ITERATIONS):
    """One direct natural ascent proposal, guarded by analytic visited-state KL.

    The direction solves (F+damping I)d=g. Scale uses the same damped curvature,
    then only the exact empirical joint-Gaussian KL and finiteness determine
    acceptance. There is no fresh-MC improvement gate favouring the P2 estimator.
    No Adam call or new data occurs during CG/backtracking.
    """
    parameters=tuple(actor.parameters());before=copy.deepcopy(actor.state_dict())
    gradient=gradient.detach()
    if gradient.numel()!=sum(p.numel() for p in parameters) or not torch.isfinite(gradient).all():
        raise ValueError('A complete finite flattened actor gradient required')
    norm=float(gradient.norm())
    empty=dict(actor_loss=_nullable(-gradient.square().sum()),value_loss=0.,macro_kl=0.,
        optimizer_steps=0,actor_optimizer_steps=0,actor_committed_optimizer_steps=0,
        manual_actor_proposals=0,manual_actor_committed_steps=0,manual_actor_backtracks=0,
        fisher_vector_products=0,cg_iterations=0,cg_residual_squared=0.,cg_reason='not_started',
        pg_gradient_norm=_nullable(norm),pg_update_accepted=False,pg_backtracks=0,
        pg_actor_step_scale=0.,pg_last_proposed_kl=None,natural_step_norm=0.,
        natural_gradient_dot_direction=0.,natural_curvature=0.,natural_fallback=False,
        natural_attempted_kl_limit=kl_limit)
    if not math.isfinite(norm) or norm==0:
        empty['cg_reason']='zero_or_nonfinite_gradient';return empty
    if len(states['local'])==0:
        return empty
    mean,sigma=gaussian_parameters(actor,states)
    old_mean,old_sigma=mean.detach().clone(),sigma.detach().clone()
    divergence=gaussian_kl(old_mean,old_sigma,mean,sigma).mean()
    first=_flat(torch.autograd.grad(divergence,parameters,create_graph=True,allow_unused=True),parameters)
    products=0
    def product(v):
        nonlocal products
        products+=1
        directional=(first*v).sum()
        second=torch.autograd.grad(directional,parameters,retain_graph=True,allow_unused=True)
        answer=_flat(second,parameters).detach()+damping*v
        if not torch.isfinite(answer).all():
            raise FloatingPointError('Finite damped Fisher product required')
        return answer
    direction,stats=conjugate_gradient(product,gradient,iterations=cg_iterations)
    fallback=False
    slope=float(torch.dot(gradient,direction))
    if not torch.isfinite(direction).all() or not math.isfinite(slope) or slope<=0:
        direction=gradient.clone();fallback=True;slope=float(torch.dot(gradient,direction))
    curvature=float(torch.dot(direction,product(direction)))
    empty.update(stats,fisher_vector_products=products,natural_gradient_dot_direction=_nullable(slope),
                 natural_curvature=_nullable(curvature),natural_fallback=fallback)
    if not math.isfinite(curvature) or curvature<=0:
        return empty
    scale=math.sqrt(2*kl_limit/curvature)
    step=direction*scale
    initial_parameters=torch.cat([p.detach().reshape(-1) for p in parameters])
    accepted=False;attempts=0;proposed=None;factor=0.
    try:
        for backtrack in range(BACKTRACKS+1):
            actor.load_state_dict(before)
            factor=.5**backtrack
            proposed_parameters=initial_parameters+factor*step
            offset=0
            with torch.no_grad():
                for parameter in parameters:
                    count=parameter.numel()
                    parameter.copy_(proposed_parameters[offset:offset+count].reshape_as(parameter))
                    offset+=count
            attempts+=1
            try:
                with torch.no_grad():
                    candidate_mean,candidate_sigma=gaussian_parameters(actor,states)
                    proposed=float(gaussian_kl(old_mean,old_sigma,candidate_mean,candidate_sigma).mean())
            except FloatingPointError:
                proposed=math.inf
            if math.isfinite(proposed) and proposed<=kl_limit+1e-12:
                accepted=True;break
    except BaseException:
        actor.load_state_dict(before)
        raise
    if not accepted:
        actor.load_state_dict(before)
    actor.zero_grad(set_to_none=True)
    empty.update(macro_kl=max(0.,proposed) if accepted else 0.,
        manual_actor_proposals=attempts,manual_actor_committed_steps=int(accepted),
        manual_actor_backtracks=max(0,attempts-1),pg_update_accepted=accepted,
        pg_backtracks=max(0,attempts-1),pg_actor_step_scale=factor if accepted else 0.,
        pg_last_proposed_kl=_nullable(proposed) if proposed is not None else None,
        natural_step_norm=_nullable((factor*step).norm()) if accepted else 0.)
    return empty


class CalibratedSnapshotPG(SnapshotPG):
    def __init__(self,spec,event_dim,seed,device,variance_reduced=True):
        super().__init__(spec,event_dim,seed,device,variance_reduced)
        self.baseline=None
        self.pending_baseline=None
        self.baseline_capture_episode=None
        self.total_manual_proposals=0
        self.total_manual_committed=0
        self.total_fisher_products=0
        self.total_critic_updates=0

    def metadata(self):
        return dict(version=VERSION,spec=self.spec.to_dict(),event_dim=self.event_dim,
            seed=self.seed,variance_reduced=self.variance_reduced,
            warmup_episodes=32,warmup='exact native MC-PPO4epochs',
            reference_episodes=REFERENCE_EPISODES,inner_new_episodes=INNER_BATCH_EPISODES,
            inner_updates=INNER_UPDATES,collection='reference16 followed by eight current4 batches',
            baseline='state-only critic copy frozen before reference collection, unchanged for full cycle',
            score_gradient='sum active (terminalR-frozen baseline(state))*joint Gaussian score',
            advantage_normalization=False,reference='finite sampled reference gradient; conditional error retained',
            correction='exact exp(sum active logpi_reference-logpi_current), no clipping/selfnormalization',
            critic='after each postwarm collection,4 fullbatch MC terminal-return epochs; this never changes frozen cycle baseline',
            actor='one manual natural-gradient proposal per reference completion/inner batch; postwarm Adam state unchanged',
            fisher='Hessian of active visited-state mean KL(anchor||candidate) plus .01 I',
            fisher_damping=FISHER_DAMPING,cg_iterations=CG_ITERATIONS,
            cg_relative_tolerance=CG_RELATIVE_TOLERANCE,kl_limit=NATURAL_KL,
            scaling='sqrt(2 KLbudget / dT(F+damping I)d)',max_backtracks=BACKTRACKS,
            acceptance='finite exact visited-state joint KL only; no current-batch return ascent gate',
            environment_calls=0,candidate_policy_events=0,private_feedback=False,
            theoretical_scope='finite reference/critic error and empirical Fisher/KL; no conditional unbiasedness, complexity or monotonic improvement guarantee')

    def _freeze_baseline(self,critic):
        self.baseline=copy.deepcopy(critic).eval()
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        self.pending_baseline=None
        self.baseline_capture_episode=self.episodes

    def _ensure_baseline(self,critic):
        if self.baseline is None:
            if self.pending_baseline is None:
                # The collector does not modify critic weights. This fallback
                # uses only weights fitted before this collection, never its R.
                self._freeze_baseline(critic)
            else:
                self.baseline=copy.deepcopy(critic).eval()
                self.baseline.load_state_dict(self.pending_baseline)
                for parameter in self.baseline.parameters():
                    parameter.requires_grad_(False)
                self.pending_baseline=None

    def _proposal(self,actor,optimizer,gradient,states):
        active=states['budget']>0
        chosen=subset(states,active)
        return natural_step(actor,gradient,chosen)

    def update(self,actor,critic,ao,vo,batches,rng):
        data,likelihood_error=self._validate_current(actor,batches)
        if self.episodes<32:
            result=super().update(actor,critic,ao,vo,batches,rng)
            result.update(manual_actor_proposals=0,manual_actor_committed_steps=0,
                          manual_actor_backtracks=0,fisher_vector_products=0,
                          baseline_capture_episode=None)
            if self.episodes==32:
                self._freeze_baseline(critic)
                result['baseline_capture_episode']=32
            return result
        self._ensure_reference(actor)
        self._ensure_baseline(critic)
        if self.phase=='inner' and len(batches)!=INNER_BATCH_EPISODES:
            raise ValueError('Inner updates require exactly four current episodes')
        phase=self.phase
        common=dict(warmup=False,svr_phase=phase,variance_reduced=self.variance_reduced,
            likelihood_error=likelihood_error,replay_episodes_read=0,buffer_episodes=0,
            candidate_policy_events=0,reference_episodes_read=0,trajectory_gradient_evaluations=0,
            actor_loss=0.,value_loss=0.,macro_kl=0.,optimizer_steps=0,actor_optimizer_steps=0,
            actor_committed_optimizer_steps=0,manual_actor_proposals=0,
            manual_actor_committed_steps=0,manual_actor_backtracks=0,fisher_vector_products=0,
            pg_gradient_norm=0.,pg_update_accepted=False,pg_backtracks=0,pg_actor_step_scale=0.,
            pg_last_proposed_kl=None,pg_numeric_rejection=False,
            correction_diagnostic_numeric_failure=False,raw_gradient_variance_trace=None,
            difference_variance_trace=None,difference_to_raw_variance_ratio=None,
            correction_norm=None,trajectory_weight_ess_fraction=None,trajectory_weight_max=None,
            trajectory_logweight_max=None,trajectory_logweight_min=None,
            reference_gradient_norm=None,reference_sample_variance_trace=self.reference_variance,
            reference_mean_score=self.reference_mean_score,
            baseline_used_capture_episode=self.baseline_capture_episode)
        reset_baseline=False
        if phase=='reference':
            if len(self.reference_batches)+len(batches)>REFERENCE_EPISODES:
                raise ValueError('Reference population exceeds registered size')
            with torch.no_grad():
                error=float(((self.reference_actor.log_prob(data['states'],data['raw'])-
                              data['old_logp'])*data['mask']).abs().max())
            if error>1e-7:
                raise AssertionError(('Reference actor changed during collection',error))
            self.reference_batches.extend(copy.deepcopy(batches))
            if len(self.reference_batches)==REFERENCE_EPISODES:
                rows=baseline_episode_gradients(self.reference_actor,self.baseline,self.reference_batches,self.device)
                self.reference_gradient=rows.mean(0).detach()
                self.reference_variance=variance_trace(rows)
                self.reference_mean_score=float(np.mean([b['returns'][-1] for b in self.reference_batches]))
                merged=tensor_batch(self.reference_batches,self.device)
                common.update(self._proposal(actor,ao,self.reference_gradient,merged['states']))
                common.update(trajectory_gradient_evaluations=REFERENCE_EPISODES,
                    reference_episodes_read=REFERENCE_EPISODES,
                    raw_gradient_variance_trace=self.reference_variance,
                    reference_gradient_norm=_nullable(self.reference_gradient.norm()),
                    reference_sample_variance_trace=self.reference_variance,
                    reference_mean_score=self.reference_mean_score)
                self.reference_batches=[]
                self.phase,self.inner_updates='inner',0
                self.cycles+=1
        else:
            current=baseline_episode_gradients(actor,self.baseline,batches,self.device)
            previous=baseline_episode_gradients(self.reference_actor,self.baseline,batches,self.device)
            logs=trajectory_log_weights(self.reference_actor,actor,batches,self.device)
            weights=logs.exp()
            raw_variance=variance_trace(current)
            common.update(trajectory_gradient_evaluations=2*len(batches),
                trajectory_logweight_max=_nullable(logs.max()),trajectory_logweight_min=_nullable(logs.min()),
                raw_gradient_variance_trace=raw_variance,
                reference_gradient_norm=_nullable(self.reference_gradient.norm()))
            numerical=bool(torch.isfinite(weights).all() and (weights>0).all())
            if numerical:
                difference=current-weights[:,None]*previous
                variance=variance_trace(difference)
                correction=difference.mean(0)
                gradient=self.reference_gradient+correction if self.variance_reduced else current.mean(0)
                common.update(correction_norm=_nullable(correction.norm()),difference_variance_trace=variance,
                    difference_to_raw_variance_ratio=(_nullable(variance/raw_variance) if variance is not None and raw_variance and raw_variance>0 else None),
                    trajectory_weight_ess_fraction=float(torch.exp(2*torch.logsumexp(logs,0)-math.log(len(weights))-torch.logsumexp(2*logs,0))),
                    trajectory_weight_max=_nullable(weights.max()))
                diagnostic=(variance is not None and raw_variance is not None and common['correction_norm'] is not None)
                common['correction_diagnostic_numeric_failure']=not diagnostic
                if not torch.isfinite(gradient).all() or not torch.isfinite(gradient.norm()) or (self.variance_reduced and not diagnostic):
                    common['pg_numeric_rejection']=True
                else:
                    common.update(self._proposal(actor,ao,gradient,data['states']))
            else:
                common['correction_diagnostic_numeric_failure']=True
                if self.variance_reduced:
                    common['pg_numeric_rejection']=True
                else:
                    common.update(self._proposal(actor,ao,current.mean(0),data['states']))
            self.inner_updates+=1
            if self.inner_updates==INNER_UPDATES:
                self.phase,self.inner_updates='reference',0
                self.reference_actor=None;self.reference_gradient=None
                self.reference_variance=self.reference_mean_score=None
                reset_baseline=True
        # Fit only after the actor estimator has consumed the precollection,
        # frozen state baseline. Same four fullbatch critic steps in P2/S2.
        common.update(fit_mc_critic(critic,vo,data['states'],data['raw'],data['returns'],data['mask'],epochs=CRITIC_EPOCHS))
        self.episodes+=len(batches);self.updates+=1
        self.total_manual_proposals+=common['manual_actor_proposals']
        self.total_manual_committed+=common['manual_actor_committed_steps']
        self.total_fisher_products+=common['fisher_vector_products']
        self.total_critic_updates+=common.get('critic_optimizer_steps',0)
        if reset_baseline:
            self._freeze_baseline(critic)
        common.update(episodes_seen=self.episodes,updates=self.updates,svr_cycles=self.cycles,
            next_svr_phase=self.phase,partial_reference_episodes=len(self.reference_batches),
            inner_updates_since_reference=self.inner_updates,
            baseline_capture_episode=self.baseline_capture_episode,
            total_manual_actor_proposals=self.total_manual_proposals,
            total_manual_actor_committed_steps=self.total_manual_committed,
            total_fisher_vector_products=self.total_fisher_products,
            total_postwarm_critic_updates=self.total_critic_updates)
        return common

    def state_dict(self):
        result=super().state_dict()
        baseline=self.pending_baseline if self.baseline is None else self.baseline.state_dict()
        result.update(calibrated_version=VERSION,calibrated_metadata=self.metadata(),
            baseline=cpu_copy(baseline),baseline_capture_episode=self.baseline_capture_episode,
            total_manual_proposals=self.total_manual_proposals,total_manual_committed=self.total_manual_committed,
            total_fisher_products=self.total_fisher_products,total_critic_updates=self.total_critic_updates)
        return result

    def load_state_dict(self,state):
        if state.get('calibrated_version')!=VERSION or state['calibrated_metadata']!=self.metadata():
            raise ValueError('Calibrated snapshot configuration mismatch')
        super().load_state_dict(state)
        self.baseline=None;self.pending_baseline=copy.deepcopy(state['baseline'])
        self.baseline_capture_episode=state['baseline_capture_episode']
        self.total_manual_proposals=int(state['total_manual_proposals'])
        self.total_manual_committed=int(state['total_manual_committed'])
        self.total_fisher_products=int(state['total_fisher_products'])
        self.total_critic_updates=int(state['total_critic_updates'])
