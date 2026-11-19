"""Registered Chapter 5 mechanism controls, without changing native sources.

The full arm delegates its complete update to allocation_morl.  The three
controls retain the same two independent value towers and actor initialization.
Only the objective or critic supervision changes.  The nonfull update below is
an explicit copy of the native update with the critic objective and active
parameter list changed; collection, reuse, clipping and TV control stay native.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from allocation23.learning import PPO
from allocation4.common import subset
from allocation4.methods.lax import gaussian_parameters, gaussian_kl
from allocation_efficiency.frequent import EPSILON, adjusted_lr
from allocation_efficiency.geppo import generalized_clip
from allocation_morl.learner import Learner as NativeLearner
from allocation_morl.values import terminal_labels, scalarize_advantages

VARIANTS = ('full', 'only_j', 'only_c', 'scalar_value')
VARIANT_VERSION = 'chapter5-explicit-mechanism-controls-20261103-v1'


@torch.no_grad()
def scalar_terminal_trace(values, labels, log_rhos, mask, lengths):
    """The same capped complete-trajectory recursion for a single scalar value.

    Fixed preference makes this linear recursion commute with scalarization.
    That identity is audited by the scalar control; it is not a new advantage
    estimator or a claim of algebraic superiority for separate value heads.
    """
    if values.shape != (sum(lengths), 1) or labels.shape != (len(lengths), 1):
        raise ValueError('One scalar per state and complete terminal path required')
    if any(int(h) != h or h <= 0 for h in lengths):
        raise ValueError('Complete positive trajectory lengths required')
    logs, masks = log_rhos.reshape(-1), mask.reshape(-1)
    if logs.shape != (len(values),) or masks.shape != logs.shape:
        raise ValueError('One scalar event likelihood and mask per decision required')
    if not all(torch.isfinite(x).all() for x in (values, labels, logs, masks)):
        raise FloatingPointError('Finite scalar trace inputs required')
    rho = torch.where(masks > 0, logs.clamp_max(0).exp(), torch.ones_like(logs))
    targets, advantages = torch.empty_like(values), torch.empty_like(values)
    offset = 0
    for label, length in zip(labels, lengths):
        following = label
        for t in range(offset + int(length) - 1, offset - 1, -1):
            advantages[t] = following - values[t]
            targets[t] = (1 - rho[t]) * values[t] + rho[t] * following
            following = targets[t]
        offset += int(length)
    return targets.detach(), advantages.detach(), rho.detach()


def _normalize_scalar(raw, mask):
    active = mask > 0
    scale = raw[active].std(unbiased=False) if bool(active.any()) else raw.new_tensor(0.)
    mean = raw[active].mean() if bool(active.any()) else raw.new_tensor(0.)
    normalized = ((raw - mean) / (scale + 1e-8)).detach() if bool(active.any()) else raw.detach()
    return normalized, raw.detach(), float(scale), float(mean)


class Learner(NativeLearner):
    """Native interface plus one explicitly registered mechanism-control name."""

    def __init__(self, spec, event_dim, seed, device='cpu', method='fixed', weight=.8,
                 tau=.2, variant='full'):
        if variant not in VARIANTS:
            raise ValueError(('Unknown Chapter 5 variant', variant))
        if variant != 'full' and (method != 'fixed' or float(weight) != .8):
            raise ValueError('Registered ablations retain the fixed .8 actor input')
        super().__init__(spec, event_dim, seed, device, method, weight, tau)
        self.variant = variant

    @property
    def objective_coefficients(self):
        if self.variant == 'only_j':
            return (0., 1.)
        if self.variant == 'only_c':
            return (1., 0.)
        return (self.weight, 1. - self.weight)

    @property
    def critic_supervision(self):
        return {'full': 'independent C and J terminal targets',
                'only_j': 'J only; C tower frozen and excluded from gradient clipping',
                'only_c': 'C only; J tower frozen and excluded from gradient clipping',
                'scalar_value': 'one scalar F=.8*C+.2*J target; no separate C/J losses'}[self.variant]

    @property
    def actor_advantage_description(self):
        return {'full': 'mix raw-unit C/J then normalize once',
                'only_j': 'raw-unit A_J only then normalize once',
                'only_c': 'raw-unit A_C only then normalize once',
                'scalar_value': 'scalar F trace advantage then normalize once'}[self.variant]

    def metadata(self):
        result = super().metadata()
        # Full preserves the native metadata/state payload exactly.  Its public
        # variant is recorded by the independent run protocol rather than by
        # changing the native learner payload.
        if self.variant == 'full':
            return result
        result.update(variant_version=VARIANT_VERSION, variant=self.variant,
            actor_input_weight=self.weight, objective_coefficients=list(self.objective_coefficients),
            terminal_objective={'only_j': 'J only', 'only_c': 'C only',
                'scalar_value': '.8*C+.2*J'}[self.variant],
            critic_supervision=self.critic_supervision,
            value_heads='same initialized sigmoid-C and unbounded-J tower capacity as full',
            actor_advantage=self.actor_advantage_description,
            inactive_critic={'only_j': 'coverage', 'only_c': 'quality', 'scalar_value': None}[self.variant],
            separately_supervised_components=[] if self.variant == 'scalar_value' else
                (['J'] if self.variant == 'only_j' else ['C']),
            scalar_trace_identity='fixed lambda linear recursion; no distinct algebraic estimator')
        return result

    def _configure_critic(self, critic):
        if not hasattr(critic, 'coverage') or not hasattr(critic, 'quality'):
            raise ValueError('Registered controls require the unchanged independent two towers')
        inactive = {'only_j': 'coverage', 'only_c': 'quality'}.get(self.variant)
        if inactive is not None:
            for parameter in getattr(critic, inactive).parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
        parameters = tuple(p for p in critic.parameters() if p.requires_grad)
        if not parameters:
            raise ValueError('At least one active critic tower required')
        return parameters

    @torch.no_grad()
    def _prepare(self, actor, critic, batches):
        data = super()._prepare(actor, critic, batches)
        if self.variant == 'full':
            return data
        if self.variant in ('only_j', 'only_c'):
            weights = torch.full_like(data['weights'], self.objective_coefficients[0])
            normalized, raw, scale, mean = scalarize_advantages(
                data['dual_advantages'], weights, data['mask'])
        else:
            values = critic(data['states']).detach()
            scalar_values = self.weight * values[:, :1] + (1. - self.weight) * values[:, 1:]
            scalar_labels = self.weight * data['labels'][:, :1] + (1. - self.weight) * data['labels'][:, 1:]
            targets, advantages, _ = scalar_terminal_trace(scalar_values, scalar_labels,
                data['log_rhos'], data['mask'], data['lengths'])
            mixed_targets = self.weight * data['targets'][:, :1] + (1. - self.weight) * data['targets'][:, 1:]
            mixed_advantages = self.weight * data['dual_advantages'][:, :1] + (1. - self.weight) * data['dual_advantages'][:, 1:]
            linear_error = max(float((targets-mixed_targets).abs().max()),
                               float((advantages-mixed_advantages).abs().max()))
            if linear_error > 1e-10:
                raise AssertionError('Fixed-preference scalar and dual trace identity failed')
            data.update(scalar_targets=targets, scalar_advantages=advantages,
                        scalar_trace_linear_error=linear_error)
            normalized, raw, scale, mean = _normalize_scalar(advantages, data['mask'])
        data.update(advantage=normalized, raw_advantage=raw,
                    advantage_scale=scale, advantage_mean=mean)
        return data

    def _value_objective(self, prediction, data, ix, mask):
        """Return the optimized loss plus labelled diagnostic losses.

        In the scalar arm the two component diagnostics are detached: neither
        is added to the backward graph.  The optimized loss is a single MSE on
        the same-capacity aggregate value.
        """
        denominator = mask.sum().clamp_min(1)
        if self.variant == 'only_j':
            loss = ((prediction[:, 1:] - data['targets'][ix, 1:]).square()*mask).sum()/denominator
            return loss, torch.stack((loss.detach().new_tensor(0.), loss.detach()))
        if self.variant == 'only_c':
            loss = ((prediction[:, :1] - data['targets'][ix, :1]).square()*mask).sum()/denominator
            return loss, torch.stack((loss.detach(), loss.detach().new_tensor(0.)))
        if self.variant != 'scalar_value':
            raise ValueError('Full arm must use the native value update')
        scalar_prediction = self.weight * prediction[:, :1] + (1. - self.weight) * prediction[:, 1:]
        loss = ((scalar_prediction-data['scalar_targets'][ix]).square()*mask).sum()/denominator
        with torch.no_grad():
            diagnostics = ((prediction-data['targets'][ix]).square()*mask).sum(0)/denominator
        return loss, diagnostics

    def update(self, actor, critic, ao, vo, batches, rng):
        if self.variant == 'full':
            return super().update(actor, critic, ao, vo, batches, rng)
        return self._update_variant(actor, critic, ao, vo, batches, rng)

    def _update_variant(self, actor, critic, ao, vo, batches, rng):
        start = time.perf_counter()
        active_critic_parameters = self._configure_critic(critic)
        warm = self.episodes < 32
        fresh_size = 8 if warm else 4
        self._validate(batches, fresh_size, self.updates)
        if self.history:
            self._validate(self.history, 4, self.updates - 1)
        if not warm and len(self.history) != 4:
            raise ValueError('Four previous complete episodes required after first32')
        lr_before = float(self.controller['actor_lr'])
        if (not ao.param_groups or any(abs(float(g['lr'])-lr_before) > 1e-15 for g in ao.param_groups)
                or not vo.param_groups or any(abs(float(g['lr'])-1e-3) > 1e-15 for g in vo.param_groups)):
            raise ValueError('Registered actor/controller and value learning rates required')
        combined = list(batches) if warm else list(batches)+list(self.history)
        data = self._prepare(actor, critic, combined)
        raw, old, mask, states = data['raw'], data['old_logp'], data['mask'], data['states']
        active = mask > 0
        fresh_count = fresh_size * self.spec.horizon
        fresh_error = float(((data['anchor_logp'][:fresh_count]-old[:fresh_count])*mask[:fresh_count]).abs().max())
        if fresh_error > 1e-7:
            raise AssertionError('Fresh original-preference paths must match current policy')
        fresh_labels = terminal_labels(batches, self.spec.groups, self.tau, self.device).repeat_interleave(self.spec.horizon,0)
        mc_error = float((data['targets'][:fresh_count]-fresh_labels).abs().max())
        if mc_error > 1e-10:
            raise AssertionError('Both on-policy trace heads must equal true terminal MC labels')
        epsilon = float(PPO['clip']) if warm else EPSILON
        records, early_stop = [], False
        if bool(active.any()):
            for _ in range(4):
                permutation = (rng if warm else self.shuffle_rng).permutation(len(raw))
                for first in range(0,len(raw),64):
                    ix = torch.as_tensor(permutation[first:first+64],device=self.device)
                    st,m = subset(states,ix),mask[ix]
                    lp = actor.log_prob(st,raw[ix])
                    if not torch.isfinite(lp).all():
                        raise FloatingPointError('Finite complete current Gaussian density required')
                    logratio = torch.where(m>0,lp-old[ix],torch.zeros_like(lp))
                    ratio = logratio.clamp(-40,40).exp()
                    clipped = generalized_clip(ratio,data['center'][ix],epsilon)
                    loss = -(torch.minimum(ratio*data['advantage'][ix],clipped*data['advantage'][ix])*m).sum(-1).mean()
                    ao.zero_grad(set_to_none=True);loss.backward()
                    agn = torch.nn.utils.clip_grad_norm_(actor.parameters(),.5)
                    if not torch.isfinite(agn):
                        raise FloatingPointError('Nonfinite multiobjective actor gradient')
                    ao.step()
                    prediction = critic(st)
                    value_loss, losses = self._value_objective(prediction, data, ix, m)
                    vo.zero_grad(set_to_none=True);value_loss.backward()
                    vgn = torch.nn.utils.clip_grad_norm_(active_critic_parameters,.5)
                    if not torch.isfinite(vgn):
                        raise FloatingPointError('Nonfinite dual-value gradient')
                    vo.step()
                    with torch.no_grad():
                        # Match native F1's lagged minibatch stop statistic.
                        delta = torch.where(m>0,lp-data['anchor_logp'][ix],torch.zeros_like(lp))
                        kl = ((delta.clamp(-40,40).exp()-1-delta)*m).sum(-1).mean()
                        clip_fraction = (((ratio-data['center'][ix]).abs()>epsilon)*m).sum()/m.sum().clamp_min(1)
                    records.append((float(loss.detach()),float(losses[0].detach()),float(losses[1].detach()),float(kl),float(clip_fraction),float(value_loss.detach())))
                    if float(kl) > .12:
                        early_stop=True;break
                if early_stop:break
        with torch.no_grad():
            final_lp = actor.log_prob(states,raw)
            final_ratio = torch.where(active,final_lp-old,torch.zeros_like(old)).clamp(-40,40).exp()
            tv = float(.5*((final_ratio-data['center']).abs()*mask).sum()/mask.sum().clamp_min(1))
            fm,fs = gaussian_parameters(actor,states)
            kl = gaussian_kl(data['anchor_mean'],data['anchor_sigma'],fm,fs)[:,None]
            actual_kl = float((kl*mask).sum()/mask.sum().clamp_min(1))
            weights = data['log_rhos'][active.reshape(-1)]
            if len(weights):
                normalized = (weights-torch.logsumexp(weights,0)).exp()
                ess_fraction = float(1/normalized.square().sum())/len(weights)
            else:ess_fraction=0.
            hlogs = data['log_rhos'][fresh_count:][active[fresh_count:].reshape(-1)]
            if len(hlogs):
                hn = (hlogs-torch.logsumexp(hlogs,0)).exp()
                history_ess = float(1/hn.square().sum())/len(hlogs)
            else:history_ess=None
        lr_after,action = (lr_before,'warmup_hold') if warm else adjusted_lr(lr_before,tv)
        if not warm:
            self.controller.update(calls=self.controller['calls']+1,actor_lr=lr_after,last_tv=tv,
                last_lr_before=lr_before,last_action=action)
            for group in ao.param_groups:group['lr']=lr_after
        self.history = copy.deepcopy(list(batches)[-4:])
        self.episodes += len(batches);self.updates += 1
        a = np.asarray(records) if records else np.zeros((1,6))
        return dict(method=self.method,variant=self.variant,actor_input_weight=self.weight,
            objective_coefficients=list(self.objective_coefficients),
            critic_supervision=self.critic_supervision,
            active_critic_parameters=sum(p.numel() for p in active_critic_parameters),
            scalar_trace_linear_error=data.get('scalar_trace_linear_error'),
            warmup=warm,actor_loss=float(a[:,0].mean()),
            coverage_value_loss=float(a[:,1].mean()),quality_value_loss=float(a[:,2].mean()),
            value_loss=float(a[:,5].mean()),macro_kl=float(a[:,3].mean()),
            clip_fraction=float(a[:,4].mean()),optimizer_steps=len(records),early_stop=early_stop,
            likelihood_error=fresh_error,behavior_likelihood_error=data['behavior_error'],
            fresh_mc_target_error=mc_error,fresh_C_mc_target_error=float((data['targets'][:fresh_count,:1]-fresh_labels[:,:1]).abs().max()),
            fresh_J_mc_target_error=float((data['targets'][:fresh_count,1:]-fresh_labels[:,1:]).abs().max()),
            advantage_std=data['advantage_scale'],advantage_mean=data['advantage_mean'],
            advantage_scalarization=self.actor_advantage_description,
            actual_joint_kl=actual_kl,tv=tv,post_final_tv=tv,actor_lr_before=lr_before,actor_lr_after=lr_after,
            tv_controller_action=action,tv_controller_calls=self.controller['calls'],tv_controller_ran=not warm,
            ess_fraction=ess_fraction,history_only_ess_fraction=history_ess,
            active_events=int(active.sum()),zeroevent_count=int((mask==0).sum()),
            new_episodes=len(batches),learning_episodes=len(combined),
            replay_episodes_read=0 if warm else 4,replay_decisions_read=0 if warm else 4*self.spec.horizon,
            fresh_C=float(fresh_labels[:,0].mean()),fresh_J=float(fresh_labels[:,1].mean()),
            fresh_weights=[float(np.asarray(b['states']['global'])[0,-1]) for b in batches],
            replay_weights=[] if warm else [float(np.asarray(b['states']['global'])[0,-1]) for b in combined[len(batches):]],
            preference_relabelled=False,relative_F1_constraint=False,quality_floor=None,
            episodes_seen=self.episodes,updates=self.updates,private_feedback=False,
            update_seconds=time.perf_counter()-start)

