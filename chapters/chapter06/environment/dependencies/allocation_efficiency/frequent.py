"""Matched frequent-update controls on ordinary terminal-reward trajectories.

F0 and F1 differ only in the source of the second half of the learning batch:
current4 repeated versus immediately previous4. Both use the same generalized
clip, frozen finite-horizon V-trace and post-update TV learning-rate controller.
No infinite-horizon GePPO theorem, unbiased current-policy gradient, or exact
reproduction of the original paper is claimed.
"""
from __future__ import annotations

import copy
import math
import numpy as np
import torch

from allocation23.learning import PPO
from allocation4.common import STATE_KEYS, subset
from allocation4.methods.lax import gaussian_kl, gaussian_parameters
from .geppo import GePPO, generalized_clip, _spec_dict, _same_json


VERSION = 'frequent-four-new-eight-learn-matched-TV-v1'
EPSILON = .2 / 1.5
TV_ALPHA = .03
TV_BETA = .5
ACTOR_LR_BOUNDS = (1e-5, 1e-3)
SHUFFLE_SEED = 580117


def adjusted_lr(lr, tv):
    """One outer-update TV controller, not one controller per SGD minibatch."""
    if not math.isfinite(lr) or not ACTOR_LR_BOUNDS[0] <= lr <= ACTOR_LR_BOUNDS[1]:
        raise ValueError('Actor learning rate outside registered bounds')
    if not math.isfinite(tv) or tv < 0:
        raise ValueError('Nonnegative finite post-final-update TV required')
    target = EPSILON / 2
    if tv > target:
        proposal, action = lr / (1 + TV_ALPHA), 'decrease'
    elif tv < TV_BETA * target:
        proposal, action = lr * (1 + TV_ALPHA), 'increase'
    else:
        proposal, action = lr, 'hold'
    return float(np.clip(proposal, *ACTOR_LR_BOUNDS)), action


def _integer(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)) and value >= 0


class FrequentPPO(GePPO):
    """The shared allocation4 method API; replay=False is the F0 control."""
    def __init__(self, spec, event_dim, seed, device, replay=True):
        if not isinstance(replay, (bool, np.bool_)):
            raise ValueError('Explicit F0/F1 replay flag required')
        super().__init__(spec, event_dim, seed, device)
        self.replay = bool(replay)
        self.shuffle_rng = np.random.default_rng(SHUFFLE_SEED + self.seed)
        self.controller = dict(calls=0, actor_lr=float(PPO['actor_lr']), last_tv=None,
                               last_lr_before=None, last_action='not_started')

    def metadata(self):
        return dict(version=VERSION, spec=_spec_dict(self.spec), event_dim=self.event_dim,
            seed=self.seed, replay=self.replay, warmup_episodes=32, warmup_batch_episodes=8,
            fresh_episodes_after_warmup=4, learning_episodes=8, persistent_buffer_episodes=4,
            learning_population='current4 plus immediately previous4' if self.replay else 'current4 repeated once',
            fixed_backend='same clip, Vtrace, SGD, private shuffle seed and TV controller in F0/F1',
            gamma=1., trace_lambda=1., rho_cap=1., c_cap=1., generalized_clip=EPSILON,
            clip_note='fixed M2-reference epsilon in BOTH arms; F0 is a matched backend control',
            epochs=4, minibatch=64, grad_clip=.5, macro_kl_stop=.12,
            actor_initial_lr=3e-4, value_lr=1e-3,
            tv='0.5*sum(mask*abs(pi_final/mu-pi_anchor/mu))/sum(mask); pre-update anchor frozen',
            tv_target=EPSILON/2, tv_alpha=TV_ALPHA, tv_beta=TV_BETA,
            tv_actor_lr_bounds=list(ACTOR_LR_BOUNDS), tv_controller='once AFTER final SGD per postwarmup outer update',
            zero_active_controller='TV0, clock advances once, registered low-TV LR increase still applies',
            mask='inactive event rho1/actor0, never task termination; propagate native terminalR',
            likelihood='complete joint quota+spatial latent Gaussian, never projected-action density',
            ratio_numeric_log_bound=40.,
            ess_scope='main ESS is the entire eight-episode learning mixture; history_only metrics isolate previous4 in F1 and are not applicable in F0; duplicates never imply new independent data',
            shuffle='caller RNG only for exact native warmup; identical private per-seed shuffle RNG afterward',
            theoretical_scope='finite-horizon gamma1 adaptation with clipped-rho implicit-policy bias; no original theorem or unbiased PG claim',
            auxiliary_models=False, candidate_policy_events=0, private_feedback=False)

    def _validate_count(self, batches, expected, version):
        if len(batches) != expected or expected not in (4, 8):
            raise ValueError(('Registered fresh/retained episode count', expected, len(batches)))
        # Reuse the strict shared ordinary-transition validator. Its fixed-eight
        # validation interface is harmlessly padded for a four-episode group;
        # this does not create learning data or advance any random generator.
        supplied = list(batches) if expected == 8 else list(batches) + list(batches)
        observed = super()._validate(supplied)
        if observed != version:
            raise ValueError(('Behavior version must follow actual outer updates', observed, version))

    def _learning_batches(self, batches):
        return list(batches) + (list(self.history) if self.replay else list(batches))

    def _finish(self, batches):
        self.history = copy.deepcopy(list(batches)[-4:])
        self.episodes += len(batches)
        self.updates += 1

    def _optimizer_rates(self, ao, vo):
        rates = [float(g['lr']) for g in ao.param_groups]
        expected = self.controller['actor_lr']
        if not rates or any(not math.isclose(x, expected, rel_tol=0, abs_tol=1e-15) for x in rates):
            raise ValueError('Actor optimizer LR and checkpointed controller disagree')
        if not vo.param_groups or any(not math.isclose(float(g['lr']), 1e-3, rel_tol=0, abs_tol=1e-15) for g in vo.param_groups):
            raise ValueError('Value optimizer must retain the fixed native learning rate')
        return expected

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
        advantage = data['advantage']
        scale = advantage[active].std(unbiased=False) if active.any() else raw.new_tensor(0.)
        if active.any():
            advantage = ((advantage-advantage[active].mean())/(scale+1e-8)).detach()
        anchor_logratio = torch.where(active, data['anchor_logp']-old, torch.zeros_like(old))
        center = anchor_logratio.clamp(-40, 40).exp().detach()
        records, early_stop, sgd_guards = [], False, 0
        if active.any():
            for _ in range(4):
                permutation = self.shuffle_rng.permutation(len(raw))
                for first in range(0, len(raw), 64):
                    ix = torch.as_tensor(permutation[first:first+64], device=raw.device)
                    st, m = subset(states, ix), mask[ix]
                    lp = actor.log_prob(st, raw[ix])
                    if not torch.isfinite(lp).all():
                        raise FloatingPointError('Finite full current Gaussian likelihood required')
                    logratio = torch.where(m > 0, lp-old[ix], torch.zeros_like(lp))
                    sgd_guards += int(((logratio.abs()>40)*(m>0)).sum())
                    ratio = logratio.clamp(-40, 40).exp()
                    clipped = generalized_clip(ratio, center[ix], EPSILON)
                    loss = -(torch.minimum(ratio*advantage[ix], clipped*advantage[ix])*m).sum(-1).mean()
                    ao.zero_grad(); loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Nonfinite frequent actor gradient')
                    ao.step()
                    pred = critic(st, raw[ix])
                    vl = (((pred-data['targets'][ix])**2)*m).sum()/m.sum().clamp_min(1)
                    vo.zero_grad(); vl.backward()
                    norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), .5)
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Nonfinite frequent value gradient')
                    vo.step()
                    with torch.no_grad():
                        delta = torch.where(m>0,lp-data['anchor_logp'][ix],torch.zeros_like(lp))
                        macro_kl = ((delta.clamp(-40,40).exp()-1-delta)*m).sum(-1).mean()
                        clip_fraction = (((ratio-center[ix]).abs()>EPSILON)*m).sum()/m.sum().clamp_min(1)
                    records.append((float(loss),float(vl),float(macro_kl),float(clip_fraction)))
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
            tv = float(.5*((final_ratio-center).abs()*mask).sum()/mask.sum().clamp_min(1))
            final_mean, final_sigma = gaussian_parameters(actor, states)
            akl = gaussian_kl(data['anchor_mean'],data['anchor_sigma'],final_mean,final_sigma)[:,None]
            bkl = gaussian_kl(data['behavior_mean'],data['behavior_sigma'],data['anchor_mean'],data['anchor_sigma'])[:,None]
            anchor_kl = float(((akl*mask).sum()/mask.sum().clamp_min(1)).clamp_min(0))
            fresh_kl = float(akl[:fresh_count].mean().clamp_min(0))
            old_mask = mask[fresh_count:]
            old_kl = float(((bkl[fresh_count:]*old_mask).sum()/old_mask.sum().clamp_min(1)).clamp_min(0)) if self.replay else 0.
            behavior_kl = float(((bkl*mask).sum()/mask.sum().clamp_min(1)).clamp_min(0))
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
        a=np.asarray(records) if records else np.zeros((1,4))
        return dict(actor_loss=float(a[:,0].mean()),value_loss=float(a[:,1].mean()),
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

    def state_dict(self):
        return dict(metadata=self.metadata(),episodes=self.episodes,updates=self.updates,
                    history=copy.deepcopy(self.history),
                    shuffle_rng=copy.deepcopy(self.shuffle_rng.bit_generator.state),
                    controller=copy.deepcopy(self.controller))

    def load_state_dict(self,state):
        if set(state)!={'metadata','episodes','updates','history','shuffle_rng','controller'} or not _same_json(state['metadata'],self.metadata()):
            raise ValueError('Frequent method/spec/seed/backend checkpoint mismatch')
        episodes,updates=state['episodes'],state['updates']
        if not _integer(episodes) or not _integer(updates) or episodes!=8*min(updates,4)+4*max(updates-4,0):
            raise ValueError('Inconsistent frequent 8-to4 chronology')
        history=copy.deepcopy(state['history'])
        if len(history)!=(4 if updates else 0):
            raise ValueError('Exactly the last four complete episodes must be checkpointed')
        if history:self._validate_count(history,4,updates-1)
        controller=copy.deepcopy(state['controller'])
        if set(controller)!={'calls','actor_lr','last_tv','last_lr_before','last_action'} or not _integer(controller['calls']) or controller['calls']!=max(updates-4,0):
            raise ValueError('Controller must advance once per postwarm outer update')
        if updates<=4:
            expected=dict(calls=0,actor_lr=float(PPO['actor_lr']),last_tv=None,last_lr_before=None,last_action='not_started')
            if controller!=expected:raise ValueError('Warmup must not advance the TV controller')
        else:
            wanted,action=adjusted_lr(controller['last_lr_before'],controller['last_tv'])
            if controller['last_action']!=action or not math.isclose(controller['actor_lr'],wanted,rel_tol=0,abs_tol=1e-15):
                raise ValueError('Inconsistent saved TV/LR controller transition')
        probe=np.random.default_rng();probe.bit_generator.state=copy.deepcopy(state['shuffle_rng'])
        if updates<=4 and not _same_json(probe.bit_generator.state,np.random.default_rng(SHUFFLE_SEED+self.seed).bit_generator.state):
            raise ValueError('Warmup must not advance private shuffle RNG')
        self.episodes,self.updates,self.history=int(episodes),int(updates),history
        self.controller,self.shuffle_rng=controller,probe
