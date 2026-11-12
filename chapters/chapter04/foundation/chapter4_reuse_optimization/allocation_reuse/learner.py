"""Frozen exclusive-prefix state weighting of the released full-event F1.

Only the registered state measure is changed. Sampling, finite V-trace labels,
full-event action ratios, generalized clip centres and SGD chronology stay F1.
This is an exploratory biased/variance tradeoff, not a correction of an F1 bug.
"""
from __future__ import annotations

import numpy as np
import torch

from allocation4.common import subset
from allocation4.methods.lax import gaussian_kl, gaussian_parameters
from allocation_efficiency.frequent import (FrequentPPO, adjusted_lr, EPSILON, TV_BETA)
from allocation_efficiency.geppo import generalized_clip

VERSION = 'frozen-exclusive-prefix-frequent-state-measure-v1'
MODES = ('p1', 'p2')
LOG_BOUND = 40.


@torch.no_grad()
def prefix_weights(log_rhos, mask, lengths, fresh_episodes=4, mode='p1'):
    """Return detached (8H,1) state weights and JSON-safe mechanism diagnostics.

    At step t, the prefix contains active-event log ratios at steps 0..t-1.
    Fresh four paths always have weight one. P1 uses raw IS prefixes (only the
    explicit numeric exp guard). P2 caps old prefixes then normalizes each
    time column across *all four* old paths, including masked events.
    """
    if mode not in MODES:
        raise ValueError('Registered prefix mode must be p1 or p2')
    if fresh_episodes != 4 or len(lengths) != 8:
        raise ValueError('Registered four fresh plus four old complete paths required')
    if any(isinstance(n, bool) or int(n) != n or n <= 0 for n in lengths) or len(set(lengths)) != 1:
        raise ValueError('Equal positive native episode lengths required')
    logs, masks = log_rhos.detach().reshape(-1), mask.detach().reshape(-1)
    if logs.shape != masks.shape or len(logs) != sum(lengths):
        raise ValueError('Step aligned scalar ratios/mask required')
    if not torch.is_floating_point(logs) or not torch.isfinite(logs).all():
        raise FloatingPointError('Finite floating full-event log ratios required')
    if not bool(((masks == 0) | (masks == 1)).all()):
        raise ValueError('Binary action-event mask required')
    horizon = int(lengths[0])
    effective = torch.where(masks > 0, logs, torch.zeros_like(logs)).reshape(8, horizon)
    fresh_error = float(effective[:4].abs().max())
    if fresh_error > 1e-7:
        raise AssertionError(('Fresh paths must come from frozen current anchor', fresh_error))
    history = effective[4:]
    # Slice the cumulative sum rather than subtracting current log ratio; this
    # preserves exclusivity even when the current event is numerically large.
    exclusive = torch.cat([history.new_zeros((4, 1)), history.cumsum(1)[:, :-1]], 1)
    if not torch.isfinite(exclusive).all():
        raise FloatingPointError('Nonfinite cumulative historical prefix')
    raw = exclusive.clamp(-LOG_BOUND, LOG_BOUND).exp()
    adopted = raw
    capped = (raw < .5) | (raw > 2.)
    if mode == 'p2':
        adopted = raw.clamp(.5, 2.)
        adopted = adopted / adopted.mean(0, keepdim=True)
    weights = torch.cat([torch.ones_like(adopted), adopted], 0).reshape(-1, 1).detach()
    if not torch.isfinite(weights).all() or not bool((weights > 0).all()):
        raise FloatingPointError('Positive finite adopted state weights required')
    def ess_fraction(x):
        return x.sum(0).square() / (4. * x.square().sum(0))
    def vector(x):
        return [float(v) for v in x.cpu().reshape(-1)]
    raw_ess, used_ess = ess_fraction(raw), ess_fraction(adopted)
    stats = dict(prefix_mode=mode, prefix_version=VERSION,
        prefix_numeric_guard_count=int((exclusive.abs() > LOG_BOUND).sum()),
        prefix_fresh_logratio_error=fresh_error,
        prefix_fresh_weight_error=float((weights[:4*horizon]-1).abs().max()),
        prefix_old_raw_min=float(raw.min()), prefix_old_raw_max=float(raw.max()),
        prefix_old_raw_mean=float(raw.mean()),
        prefix_old_weight_min=float(adopted.min()), prefix_old_weight_max=float(adopted.max()),
        prefix_old_weight_mean=float(adopted.mean()),
        prefix_old_ess_fraction=float(used_ess.mean()),
        prefix_old_raw_ess_fraction=float(raw_ess.mean()),
        prefix_cap_fraction=float(capped.double().mean()) if mode == 'p2' else 0.,
        prefix_raw_outside_cap_fraction=float(capped.double().mean()),
        prefix_ess_fraction_by_time=vector(used_ess),
        prefix_raw_ess_fraction_by_time=vector(raw_ess),
        prefix_weight_mean_by_time=vector(adopted.mean(0)),
        prefix_weight_max_by_time=vector(adopted.max(0).values),
        prefix_raw_log_min_by_time=vector(exclusive.min(0).values),
        prefix_raw_log_max_by_time=vector(exclusive.max(0).values),
        prefix_old_max_mass_fraction_by_time=vector(adopted.max(0).values/adopted.sum(0)),
        prefix_weights_frozen=not weights.requires_grad)
    return weights, stats


def weighted_advantage(advantage, mask, weight):
    """Freeze weighted centering/scaling; keep the exact released all-one path."""
    active = mask > 0
    if not active.any():
        return advantage.detach(), advantage.new_tensor(0.)
    if bool((weight == 1).all()):
        scale = advantage[active].std(unbiased=False)
        return ((advantage-advantage[active].mean())/(scale+1e-8)).detach(), scale
    wm = weight * mask
    mass = wm.sum()
    mean = (wm*advantage).sum()/mass
    scale = ((wm*(advantage-mean).square()).sum()/mass).sqrt()
    return ((advantage-mean)/(scale+1e-8)).detach(), scale


class PrefixPPO(FrequentPPO):
    """F1 with one of two preregistered frozen state-measure variants."""
    def __init__(self, spec, event_dim, seed, device, mode='p1'):
        if mode not in MODES:
            raise ValueError('Registered prefix mode must be p1 or p2')
        self.mode = mode
        super().__init__(spec, event_dim, seed, device, replay=True)

    def metadata(self):
        d = super().metadata()
        d.update(version=VERSION, prefix_mode=self.mode,
            prefix_scope='exclusive full prefix t-1 of frozen anchor/actual behavior full eta+z ratios',
            prefix_old_rule='raw prefix, no self normalization' if self.mode=='p1' else 'cap raw prefix to [.5,2], then mean1 over all old4 at each t',
            prefix_numeric_log_bound=LOG_BOUND, prefix_fresh_rule='exact one',
            state_measure_hooks='actor numerator, weighted advantage normalization, critic MSE, macroKL, TV, analyticKL',
            actor_reduction='original minibatch length; not sum of weights',
            value_tv_reduction='sum(state_weight*mask), clamped below at one as released F1',
            prefix_freeze='detach before all four epochs, no prefix derivative',
            fixed_backend='released F1 sampler, Vtrace labels, full-event ratios, generalized clipping, SGD and update cadence',
            theoretical_scope='finite-horizon exploratory state-measure surrogate; clipped/self-normalized P2 is biased; no unbiased PG or performance guarantee')
        return d

    def _weights(self, data):
        return prefix_weights(data['log_rhos'], data['mask'], data['lengths'], mode=self.mode)

    def update(self, actor, critic, ao, vo, batches, rng):
        warm = self.episodes < 32
        new_size = 8 if warm else 4
        self._validate_count(batches, new_size, self.updates)
        if self.history:
            self._validate_count(self.history, 4, self.updates-1)
        lr_before = self._optimizer_rates(ao, vo)
        if warm:
            stats = self.native.update(actor, critic, ao, vo, batches, rng)
            self._finish(batches)
            stats.update(warmup=True, replay_enabled=False, new_episodes=8, learning_episodes=8,
                replay_episodes_read=0, replay_decisions_read=0, current_episode_repeats=0,
                buffer_episodes=4, mean_policy_age=0., candidate_policy_events=0,
                actor_lr_before=lr_before, actor_lr_after=lr_before,
                tv_controller_calls=0, tv_controller_ran=False,
                history_only_ess_fraction=None,history_only_trajectory_ess_fraction=None,
                history_only_active_events=0,history_only_trajectory_active_count=0,
                episodes_seen=self.episodes, updates=self.updates)
            return stats
        if len(self.history) != 4:
            raise ValueError('First frequent update requires the last four warmup episodes')
        combined = self._learning_batches(batches)
        data, behavior_error = self._prepare(actor, critic, combined)
        states, raw, old, mask = data['states'], data['raw'], data['old_logp'], data['mask']
        active = mask > 0
        fresh_count = 4 * self.spec.horizon
        fresh_error = float(((data['anchor_logp'][:fresh_count]-old[:fresh_count])*mask[:fresh_count]).abs().max())
        if fresh_error > 1e-7:
            raise AssertionError(('Fresh collection is not from the pre-update anchor', fresh_error))
        # At mu=pi and lambda=gamma=1, each complete path has target R. Check it
        # on the current half (both halves in F0), without introducing labels.
        current_mc = torch.as_tensor(np.concatenate([b['returns'] for b in batches]),
                                     dtype=raw.dtype, device=raw.device)[:, None]
        target_error = float((data['targets'][:fresh_count]-current_mc).abs().max())
        if not self.replay:
            target_error = max(target_error, float((data['targets'][fresh_count:]-current_mc).abs().max()))
        if target_error > 1e-10:
            raise AssertionError(('On-policy finite-horizon V-trace failed MC identity', target_error))
        weight, prefix_stats = self._weights(data)
        weighted_mask = weight * mask
        advantage, scale = weighted_advantage(data['advantage'], mask, weight)
        anchor_logratio = torch.where(active, data['anchor_logp']-old, torch.zeros_like(old))
        center = anchor_logratio.clamp(-40, 40).exp().detach()
        records, early_stop, sgd_guards = [], False, 0
        if active.any():
            for _ in range(4):
                permutation = self.shuffle_rng.permutation(len(raw))
                for first in range(0, len(raw), 64):
                    ix = torch.as_tensor(permutation[first:first+64], device=raw.device)
                    st, m = subset(states, ix), mask[ix]
                    wm = weighted_mask[ix]
                    lp = actor.log_prob(st, raw[ix])
                    if not torch.isfinite(lp).all():
                        raise FloatingPointError('Finite full current Gaussian likelihood required')
                    logratio = torch.where(m > 0, lp-old[ix], torch.zeros_like(lp))
                    sgd_guards += int(((logratio.abs()>40)*(m>0)).sum())
                    ratio = logratio.clamp(-40, 40).exp()
                    clipped = generalized_clip(ratio, center[ix], EPSILON)
                    loss = -(torch.minimum(ratio*advantage[ix], clipped*advantage[ix])*wm).sum(-1).mean()
                    ao.zero_grad(); loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Nonfinite frequent actor gradient')
                    actor_norm = float(norm)
                    ao.step()
                    pred = critic(st, raw[ix])
                    vl = (((pred-data['targets'][ix])**2)*wm).sum()/wm.sum().clamp_min(1)
                    vo.zero_grad(); vl.backward()
                    norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), .5)
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Nonfinite frequent value gradient')
                    value_norm = float(norm)
                    vo.step()
                    with torch.no_grad():
                        delta = torch.where(m>0,lp-data['anchor_logp'][ix],torch.zeros_like(lp))
                        macro_kl = ((delta.clamp(-40,40).exp()-1-delta)*wm).sum(-1).mean()
                        clip_fraction = (((ratio-center[ix]).abs()>EPSILON)*m).sum()/m.sum().clamp_min(1)
                    records.append((float(loss),float(vl),float(macro_kl),float(clip_fraction),actor_norm,value_norm))
                    if float(macro_kl) > .12:
                        early_stop = True
                        break
                if early_stop: break
        with torch.no_grad():
            final_lp = actor.log_prob(states, raw)
            final_logratio = torch.where(active,final_lp-old,torch.zeros_like(old))
            if not torch.isfinite(final_logratio).all():
                raise FloatingPointError('Finite final TV likelihoods required')
            final_ratio = final_logratio.clamp(-40,40).exp()
            tv = float(.5*((final_ratio-center).abs()*weighted_mask).sum()/weighted_mask.sum().clamp_min(1))
            final_mean, final_sigma = gaussian_parameters(actor, states)
            akl = gaussian_kl(data['anchor_mean'],data['anchor_sigma'],final_mean,final_sigma)[:,None]
            bkl = gaussian_kl(data['behavior_mean'],data['behavior_sigma'],data['anchor_mean'],data['anchor_sigma'])[:,None]
            anchor_kl = float(((akl*weighted_mask).sum()/weighted_mask.sum().clamp_min(1)).clamp_min(0))
            fresh_kl = float(akl[:fresh_count].mean().clamp_min(0))
            old_mask = weighted_mask[fresh_count:]
            old_kl = float(((bkl[fresh_count:]*old_mask).sum()/old_mask.sum().clamp_min(1)).clamp_min(0)) if self.replay else 0.
            behavior_kl = float(((bkl*weighted_mask).sum()/weighted_mask.sum().clamp_min(1)).clamp_min(0))
            weights_log = data['log_rhos'][active.reshape(-1)]
            if len(weights_log):
                normalized=(weights_log-torch.logsumexp(weights_log,0)).exp()
                ess=float(1/normalized.square().sum())
                max_weight=float(weights_log.clamp(-40,40).exp().max())
                capped_fraction=float((weights_log>1e-12).double().mean())
            else: ess,max_weight,capped_fraction=0.,0.,0.
            trajectory_logs=[];offset=0
            for length in data['lengths']:
                sl=slice(offset,offset+length)
                trajectory_logs.append((data['log_rhos'][sl]*mask[sl].reshape(-1)).sum());offset+=length
            trajectory_logs=torch.stack(trajectory_logs)
            tn=(trajectory_logs-torch.logsumexp(trajectory_logs,0)).exp()
            trajectory_ess=float(1/tn.square().sum())
            history_ess_fraction=history_trajectory_fraction=None
            history_active=history_active_episodes=0
            if self.replay:
                history_mask=active[fresh_count:].reshape(-1)
                history_active=int(history_mask.sum())
                history_logs=data['log_rhos'][fresh_count:][history_mask]
                if len(history_logs):
                    hn=(history_logs-torch.logsumexp(history_logs,0)).exp()
                    history_ess_fraction=float(1/hn.square().sum())/history_active
                hlogs=trajectory_logs[4:]
                htn=(hlogs-torch.logsumexp(hlogs,0)).exp()
                history_trajectory_fraction=float(1/htn.square().sum())/4
                history_active_episodes=sum(bool(mask[(4+j)*self.spec.horizon:(5+j)*self.spec.horizon].any()) for j in range(4))
        lr_after, action = adjusted_lr(lr_before,tv)
        self.controller.update(calls=self.controller['calls']+1,actor_lr=lr_after,
                               last_tv=tv,last_lr_before=lr_before,last_action=action)
        for group in ao.param_groups: group['lr']=lr_after
        self._finish(batches)
        a=np.asarray(records) if records else np.zeros((1,6))
        return dict(prefix_stats,
            actor_grad_norm=float(a[:,4].mean()),value_grad_norm=float(a[:,5].mean()),
            actor_grad_clipped_fraction=float((a[:,4]>.5).mean()),
            value_grad_clipped_fraction=float((a[:,5]>.5).mean()),
            actor_loss=float(a[:,0].mean()),value_loss=float(a[:,1].mean()),
            macro_kl=float(a[:,2].mean()),clip_fraction=float(a[:,3].mean()),
            optimizer_steps=len(records),advantage_std=float(scale),early_stop=early_stop,
            warmup=False,replay_enabled=self.replay,new_episodes=4,learning_episodes=8,
            likelihood_error=fresh_error,behavior_likelihood_error=behavior_error,
            fresh_mc_target_error=target_error,actual_joint_kl=fresh_kl,
            anchor_kl=anchor_kl,mixture_anchor_kl_active=anchor_kl,
            old_policy_kl=old_kl,behavior_to_anchor_kl=behavior_kl,
            ess=ess,ess_fraction=ess/max(int(active.sum()),1),
            trajectory_ess=trajectory_ess,trajectory_ess_fraction=trajectory_ess/8,
            history_only_ess_fraction=history_ess_fraction,
            history_only_trajectory_ess_fraction=history_trajectory_fraction,
            history_only_active_events=history_active,
            history_only_trajectory_active_count=history_active_episodes,
            max_importance_weight=max_weight,rho_capped_fraction=capped_fraction,
            zeroevent_count=int((mask==0).sum()),active_events=int(active.sum()),
            anchor_ratio_guard_count=int(((anchor_logratio.abs()>40)*active).sum()),
            final_ratio_guard_count=int(((final_logratio.abs()>40)*active).sum()),
            sgd_ratio_guard_count=sgd_guards,
            replay_episodes_read=4 if self.replay else 0,
            replay_decisions_read=4*self.spec.horizon if self.replay else 0,
            current_episode_repeats=0 if self.replay else 4,
            buffer_episodes=4,mean_policy_age=.5 if self.replay else 0.,
            mean_age_plus_one=1.5 if self.replay else 1.,generalized_clip=EPSILON,
            candidate_policy_events=0,vtrace_target_recomputations=1,
            post_final_tv=tv,tv=tv,tv_target=EPSILON/2,tv_lower_target=TV_BETA*EPSILON/2,
            actor_lr_before=lr_before,actor_lr_after=lr_after,tv_lr_factor=lr_after/lr_before,
            tv_controller_action=action,tv_controller_calls=self.controller['calls'],tv_controller_ran=True,
            episodes_seen=self.episodes,updates=self.updates)

