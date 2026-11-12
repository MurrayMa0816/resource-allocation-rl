"""Direct dual-objective adaptation of the complete-trajectory F1 workflow.

There is no native-F1 proposal, reward floor, projected coverage supplement,
or relative learning-rate guard. This is a finite-horizon clipped-ratio
adaptation, not an unbiased-gradient or true Pareto-front guarantee.
"""
from __future__ import annotations

import copy
import math
import time
import numpy as np
import torch

from allocation23.learning import PPO
from allocation4.common import STATE_KEYS, subset
from allocation4.methods.lax import gaussian_parameters, gaussian_log_prob, gaussian_kl
from allocation_efficiency.frequent import EPSILON, adjusted_lr, SHUFFLE_SEED
from allocation_efficiency.geppo import generalized_clip
from .values import terminal_labels, dual_terminal_trace, scalarize_advantages


VERSION = 'high-yield89-area-objective-Full-original-preference-replay-v1'
METHODS = ('fixed', 'conditioned')


class Learner:
    def __init__(self, spec, event_dim, seed, device='cpu', method='fixed', weight=.5, tau=.2):
        if method not in METHODS or not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError('Registered method and preference in [0,1] required')
        if not np.isfinite(tau) or not 0 <= tau <= 1:
            raise ValueError('Finite threshold in [0,1] required')
        execution_device = torch.device(device)
        if execution_device.type not in ('cpu', 'cuda'):
            raise ValueError('Explicit CPU or CUDA execution required')
        if execution_device.type == 'cuda' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise ValueError('CUDA update requires exactly one explicitly visible physical GPU')
        self.spec, self.event_dim, self.seed = spec, int(event_dim), int(seed)
        self.device, self.method, self.weight, self.tau = str(device), method, float(weight), float(tau)
        self.episodes = self.updates = 0
        self.history = []
        self.shuffle_rng = np.random.default_rng(SHUFFLE_SEED + self.seed)
        self.controller = dict(calls=0, actor_lr=float(PPO['actor_lr']), last_tv=None,
                               last_lr_before=None, last_action='not_started')

    def collection_size(self, actor=None, max_new=None):
        count = 8 if self.episodes < 32 else 4
        return count if max_new is None or max_new >= count else 0

    def metadata(self):
        return dict(version=VERSION, method=self.method, seed=self.seed, spec=self.spec.to_dict(),
            event_dim=self.event_dim, fixed_weight=self.weight if self.method == 'fixed' else None,
            tau=self.tau, terminal_objective='lambda*C+(1-lambda)*JA from first training episode',
            components=['C=K/N at fixed tau=.90', 'JA=area-weighted native terminal relative yield'], auxiliary_J='unweighted mean, logged only',
            quality_floor=None, relative_F1_constraint=False,
            value_heads='independent sigmoid-C and unbounded-JA state-only models',
            preference='last global input; algorithm setting fixed for each physical episode',
            replay='after32 episodes, current4 plus immediately previous4 with original preferences',
            preference_relabelling=False, gamma=1., trace_lambda=1., rho_cap=1., c_cap=1.,
            actor_advantage='raw-unit lambda*A_C+(1-lambda)*A_JA then one joint scalar normalization',
            warmup='first32 episodes still optimize requested objective; current8, clip.2',
            postwarm='current4+previous4, generalized clip .2/1.5; frozen Vtrace targets',
            epochs=4, minibatch=64, grad_clip=.5, macro_kl_stop=.12,
            TV_controller='one final measured TV transition per postwarm outer update',
            no_private_feedback=True, physical_model_used=False,
            theoretical_scope='finite-horizon clipped-rho adaptation; no unbiasedness, monotonicity or true-front theorem')

    def _validate(self, batches, expected, version):
        if len(batches) != expected:
            raise ValueError(('Complete episode batch count', len(batches), expected))
        for b in batches:
            n = self.spec.horizon
            required = {'states', 'next_states', 'raw', 'old_logp', 'mask', 'values', 'returns',
                        'rewards', 'done', 'behavior_mean', 'behavior_sigma', 'behavior_version',
                        'K', 'C', 'JA', 'J', 'terminal_quality', 'terminal_area_ha'}
            if required - set(b):
                raise ValueError(('Missing ordinary terminal episode fields', sorted(required-set(b))))
            shapes = dict(raw=(n, self.event_dim), old_logp=(n, 1), mask=(n, 1),
                values=(n, 2), returns=(n,), rewards=(n,), done=(n,),
                behavior_mean=(n, self.event_dim), behavior_sigma=(n, self.event_dim))
            for key, shape in shapes.items():
                if np.shape(b[key]) != shape or not np.isfinite(b[key]).all():
                    raise ValueError(('Finite step-aligned episode field', key, np.shape(b[key]), shape))
            if (not np.isin(b['mask'], [0, 1]).all() or not np.all(np.asarray(b['behavior_sigma']) > 0)
                    or not np.all(np.asarray(b['done'])[:-1] == 0) or b['done'][-1] != 1
                    or not np.all(np.asarray(b['rewards'])[:-1] == 0)):
                raise ValueError('Positive scales, binary masks and true terminal-only feedback required')
            if (isinstance(b['behavior_version'], (bool, np.bool_))
                    or not isinstance(b['behavior_version'], (int, np.integer))
                    or b['behavior_version'] != version):
                raise ValueError('Actual behavior-policy version must match complete collection')
            for container in ('states', 'next_states'):
                if set(b[container]) != set(STATE_KEYS):
                    raise ValueError('Shared neural state fields required')
                state_shapes = dict(local=(n, self.spec.groups, self.spec.local_dim),
                    caps=(n, self.spec.groups), budget=(n,), remaining=(n,))
                state_shapes['global'] = (n, self.spec.global_dim)
                for key, shape in state_shapes.items():
                    x = np.asarray(b[container][key])
                    if x.shape != shape or not np.isfinite(x).all():
                        raise ValueError(('Finite public state plus preference required', key))
            for key in STATE_KEYS:
                if not np.allclose(b['next_states'][key][:-1], b['states'][key][1:], rtol=0, atol=1e-12):
                    raise ValueError(('Physical state continuity', key))
            weights = np.asarray(b['states']['global'])[:, -1]
            next_weights = np.asarray(b['next_states']['global'])[:, -1]
            w = float(weights[0])
            if (not 0 <= w <= 1 or not np.all(weights == w) or not np.all(next_weights == w)):
                raise ValueError('Preference must stay exactly fixed for each complete episode')
            if self.method == 'fixed' and w != self.weight:
                raise ValueError('Fixed-preference learner received another preference')
            if 'preference' in b and float(b['preference']) != w:
                raise ValueError('Preference metadata differs from actual neural input')
        terminal_labels(batches, self.spec.groups, self.tau, self.device)

    @torch.no_grad()
    def _prepare(self, actor, critic, batches):
        states = {k: torch.as_tensor(np.concatenate([b['states'][k] for b in batches]),
                    dtype=torch.float64, device=self.device) for k in STATE_KEYS}
        flat = {k: torch.as_tensor(np.concatenate([b[k] for b in batches]), dtype=torch.float64,
                    device=self.device) for k in ('raw', 'old_logp', 'mask', 'behavior_mean', 'behavior_sigma')}
        lengths = [len(b['raw']) for b in batches]
        behavior_error = float((gaussian_log_prob(flat['behavior_mean'], flat['behavior_sigma'], flat['raw'])
                               - flat['old_logp']).abs().max())
        if behavior_error > 1e-7:
            raise AssertionError('Stored complete behavior Gaussian density mismatch')
        am, ass = gaussian_parameters(actor, states)
        anchor_logp = actor.log_prob(states, flat['raw']).detach().clone()
        logs = (anchor_logp - flat['old_logp']).reshape(-1)
        values = critic(states).detach()
        labels = terminal_labels(batches, self.spec.groups, self.tau, self.device)
        targets, advantages, rhos = dual_terminal_trace(values, labels, logs, flat['mask'], lengths)
        weights = states['global'][:, -1:]
        advantage, raw_advantage, scale, mean = scalarize_advantages(advantages, weights, flat['mask'])
        active = flat['mask'] > 0
        center = torch.where(active, anchor_logp-flat['old_logp'], torch.zeros_like(anchor_logp)).clamp(-40,40).exp()
        return dict(**flat, states=states, lengths=lengths, labels=labels,
            targets=targets, dual_advantages=advantages, raw_advantage=raw_advantage, advantage=advantage,
            advantage_scale=scale, advantage_mean=mean, weights=weights,
            anchor_logp=anchor_logp, anchor_mean=am.detach().clone(), anchor_sigma=ass.detach().clone(),
            log_rhos=logs, capped_rhos=rhos, center=center.detach(), behavior_error=behavior_error)

    def update(self, actor, critic, ao, vo, batches, rng):
        start = time.perf_counter()
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
                    losses = ((prediction-data['targets'][ix]).square()*m).sum(0)/m.sum().clamp_min(1)
                    value_loss = losses.sum()
                    vo.zero_grad(set_to_none=True);value_loss.backward()
                    vgn = torch.nn.utils.clip_grad_norm_(critic.parameters(),.5)
                    if not torch.isfinite(vgn):
                        raise FloatingPointError('Nonfinite dual-value gradient')
                    vo.step()
                    with torch.no_grad():
                        # Match native F1's lagged minibatch stop statistic.
                        delta = torch.where(m>0,lp-data['anchor_logp'][ix],torch.zeros_like(lp))
                        kl = ((delta.clamp(-40,40).exp()-1-delta)*m).sum(-1).mean()
                        clip_fraction = (((ratio-data['center'][ix]).abs()>epsilon)*m).sum()/m.sum().clamp_min(1)
                    records.append((float(loss.detach()),float(losses[0].detach()),float(losses[1].detach()),float(kl),float(clip_fraction)))
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
        a = np.asarray(records) if records else np.zeros((1,5))
        return dict(method=self.method,warmup=warm,actor_loss=float(a[:,0].mean()),
            coverage_value_loss=float(a[:,1].mean()),area_yield_value_loss=float(a[:,2].mean()),
            value_loss=float(a[:,1:3].sum(1).mean()),macro_kl=float(a[:,3].mean()),
            clip_fraction=float(a[:,4].mean()),optimizer_steps=len(records),early_stop=early_stop,
            likelihood_error=fresh_error,behavior_likelihood_error=data['behavior_error'],
            fresh_mc_target_error=mc_error,fresh_C_mc_target_error=float((data['targets'][:fresh_count,:1]-fresh_labels[:,:1]).abs().max()),
            fresh_JA_mc_target_error=float((data['targets'][:fresh_count,1:]-fresh_labels[:,1:]).abs().max()),
            advantage_std=data['advantage_scale'],advantage_mean=data['advantage_mean'],
            advantage_scalarization='mix raw-unit C/JA then normalize once',
            actual_joint_kl=actual_kl,tv=tv,post_final_tv=tv,actor_lr_before=lr_before,actor_lr_after=lr_after,
            tv_controller_action=action,tv_controller_calls=self.controller['calls'],tv_controller_ran=not warm,
            ess_fraction=ess_fraction,history_only_ess_fraction=history_ess,
            active_events=int(active.sum()),zeroevent_count=int((mask==0).sum()),
            new_episodes=len(batches),learning_episodes=len(combined),
            replay_episodes_read=0 if warm else 4,replay_decisions_read=0 if warm else 4*self.spec.horizon,
            fresh_C=float(fresh_labels[:,0].mean()),fresh_JA=float(fresh_labels[:,1].mean()),
            fresh_auxiliary_J=float(np.mean([b["J"] for b in batches])),
            fresh_weights=[float(np.asarray(b['states']['global'])[0,-1]) for b in batches],
            replay_weights=[] if warm else [float(np.asarray(b['states']['global'])[0,-1]) for b in combined[len(batches):]],
            preference_relabelled=False,relative_F1_constraint=False,quality_floor=None,
            episodes_seen=self.episodes,updates=self.updates,private_feedback=False,
            update_seconds=time.perf_counter()-start)

    def state_dict(self):
        return dict(metadata=self.metadata(),episodes=self.episodes,updates=self.updates,
            history=copy.deepcopy(self.history),controller=copy.deepcopy(self.controller),
            shuffle_rng=copy.deepcopy(self.shuffle_rng.bit_generator.state))
