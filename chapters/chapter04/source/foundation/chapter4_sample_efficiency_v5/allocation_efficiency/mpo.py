"""Finite-horizon MPO-style policy iteration on the released A policy.

This is an explicitly labelled adaptation, not a reproduction or a theorem of
the original MPO algorithm.  Native complete terminal-reward trajectories fit
a Retrace action-value function.  Sampled-current-action E-step weights obey an
empirical average KL budget; a weighted-likelihood M-step is checked against a
visited-state joint-Gaussian KL budget.  The decoder, native reward and public
observation contract are untouched.  No candidate is executed in an environment.

M0 and M1 share every compute quota.  M0 draws from the current collection;
M1 draws from the most recent 32 complete episodes, including the current eight.
Their targets are recomputed under the current policy rather than labelling old
actions with an uncorrected old Monte Carlo return.
"""
from __future__ import annotations

import copy
from collections import deque
import math

import numpy as np
import torch

from allocation4.common import cpu_copy, subset
from allocation4.methods.lax import (fit_mc_critic, gaussian_log_prob,
                                     gaussian_parameters, kl_step)
from .retrace import RetraceAC, StateActionQ, retrace_targets


VERSION = 'finite-terminal-MPO-K16-retrace-recent32-v1'
WARMUP = 32
CAPACITY = 32
Q_UPDATES = 8
ACTOR_UPDATES = 4
CANDIDATES = 16
E_KL = .1
M_KL = .01
ACTOR_LR = 3e-4
Q_LR = 1e-3


@torch.no_grad()
def e_step_weights(qvalues, epsilon=E_KL):
    """One global temperature for mean KL(q_i || uniform(K)) <= epsilon.

    Each row is a fixed bank of current-policy candidate evaluations (antithetic
    candidates have the right marginals but are not independent).  The
    empirical E-step optimizes their weighted value, not entropy-augmented task
    reward.  Shifting every value in a row leaves the answer unchanged.
    """
    if (qvalues.ndim != 2 or qvalues.shape[1] < 2 or len(qvalues) == 0
            or not torch.isfinite(qvalues).all()
            or not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError('Finite nonempty state/candidate values and positive KL required')
    count = qvalues.shape[1]
    centered = qvalues - qvalues.mean(1, keepdim=True)
    spread = float(centered.abs().max())
    if spread <= 1e-14:
        return torch.full_like(qvalues, 1. / count), 1., 0.

    def distribution(temperature):
        logweights = torch.log_softmax(centered / temperature, dim=1)
        weights = logweights.exp()
        divergence = (weights * (logweights + math.log(count))).sum(1).mean()
        return weights, float(divergence)

    low, high = max(spread * 1e-12, 1e-15), max(spread, 1e-8)
    for _ in range(30):
        _, divergence = distribution(high)
        if divergence <= epsilon:
            break
        high *= 2.
    else:
        raise FloatingPointError('Could not bracket empirical MPO E-step temperature')
    _, low_divergence = distribution(low)
    if low_divergence <= epsilon:
        weights, divergence = distribution(low)
        return weights.detach(), float(low), divergence
    for _ in range(32):
        middle = (low + high) / 2
        _, divergence = distribution(middle)
        if divergence > epsilon:
            low = middle
        else:
            high = middle
    weights, divergence = distribution(high)
    if divergence > epsilon + 1e-9:
        raise AssertionError('MPO candidate E-step exceeded its empirical KL budget')
    return weights.detach(), float(high), divergence


class MPO(RetraceAC):
    """Same recorded-data API as the shared learner; no environment handle."""

    def __init__(self, spec, event_dim, seed, device, replay=True):
        # The inherited Q constructor is old code.  Protect CUDA as well as CPU
        # global generators from its seed, leaving only explicit private RNGs.
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            super().__init__(spec, event_dim, seed, device, replay=replay)
        self.buffer = deque(maxlen=CAPACITY if self.replay else 8)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=Q_LR)
        self.sampled_old_episodes = 0
        self.replay_episode_reads = 0
        self.e_steps = 0

    def metadata(self):
        return dict(version=VERSION, framework='finite-horizon MPO-style adaptation',
            replay=self.replay, warmup_episodes=WARMUP,
            warmup='exact native MC-PPO; no Q or candidate updates before episode32',
            fresh_episodes=8, learning_episodes=8,
            pool_episodes=CAPACITY if self.replay else 8,
            sampling='uniform eight complete episodes without replacement for every Q and actor step',
            q_updates_per_collection=Q_UPDATES, actor_updates_per_collection=ACTOR_UPDATES,
            q_lr=Q_LR, actor_lr=ACTOR_LR, native_critic_fullbatch_epochs=4,
            q='public full state and complete joint raw Gaussian event; 64tanh64tanh1',
            targets='finite-horizon native terminal-only Retrace gamma=lambda=1, c=min(1,pi/mu), target_tau=.005',
            candidates=CANDIDATES, candidate_sampling='private antithetic Gaussian, current policy, detached',
            e_step='global-temperature softmax Q with mean candidate KL(q||uniform)<=.1',
            m_step='weighted Gaussian maximum likelihood, no Q-action gradient',
            m_kl=M_KL, m_constraint='visited active-state KL(anchor||proposal) empirical, max8 Adam tries',
            likelihood='complete joint quota+spatial raw Gaussian, never projected-action density',
            mask='inactive actor0, raw canonical zero in Q, trace ratio1, never episode truncation',
            auxiliary_reward=False, environment_candidate_calls=0,
            theoretical_scope='approximate Q/integrals/replay occupancy; no unbiased gradient, original MPO theorem or monotonic task improvement guarantee')

    def _select(self):
        population = list(self.buffer)
        indices = self.replay_rng.choice(len(population), size=8, replace=False)
        selected = [population[int(i)] for i in indices]
        old = sum(b['behavior_version'] < self.updates for b in selected)
        self.replay_episode_reads += len(selected)
        self.sampled_old_episodes += old
        return selected, old

    @torch.no_grad()
    def _targets(self, actor, data):
        # Retrace only needs min(1, ratio).  Compute this directly in log space;
        # no uncontrolled exponentiation or artificial lower clipping is needed.
        mean, sigma = gaussian_parameters(actor, data['states'])
        behavior = gaussian_log_prob(data['behavior_mean'], data['behavior_sigma'], data['raw'])
        logratio = gaussian_log_prob(mean, sigma, data['raw']) - behavior
        logratio = torch.where(data['mask'] > 0, logratio, torch.zeros_like(logratio))
        if not torch.isfinite(logratio).all():
            raise FloatingPointError('Finite complete joint Gaussian ratio required')
        ratios = logratio.clamp(max=0).exp()
        old_q = self.target_q(data['states'], data['raw'], data['mask'])
        next_values = torch.zeros_like(old_q)
        alive = ~data['done']
        if bool(alive.any()):
            next_values[alive] = self._value(actor, self.target_q, subset(data['next_states'], alive))
        target = retrace_targets(data['rewards'], data['done'], old_q,
                                 next_values, ratios, data['lengths'])
        return target, ratios, logratio

    def _q_step(self, actor, data):
        targets, trace, logratio = self._targets(actor, data)
        prediction = self.q(data['states'], data['raw'], data['mask'])
        loss = (prediction - targets).square().mean()
        self.q_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.q.parameters(), .5)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite MPO action-value fit')
        self.q_optimizer.step()
        self.q_steps += 1
        with torch.no_grad():
            for destination, source in zip(self.target_q.parameters(), self.q.parameters()):
                destination.lerp_(source, .005)
        return dict(q_loss=float(loss.detach()), q_grad_norm=float(norm),
            q_target_mean=float(targets.mean()), q_target_std=float(targets.std(unbiased=False)),
            trace_weight_mean=float(trace.mean()), trace_weight_min=float(trace.min()),
            trace_weight_ess=float(trace.sum().square() / trace.square().sum().clamp_min(1e-30)),
            trace_clip_fraction=float((logratio > 0).double().mean()),
            ratio_log_max_abs=float(logratio.abs().max()))

    def _mpo_actor_step(self, actor, optimizer, data):
        active = data['mask'].reshape(-1) > 0
        if not bool(active.any()):
            return dict(actor_loss=0., actor_grad_norm=0., macro_kl=0., actual_kl=0.,
                actor_optimizer_steps=0, actor_committed_steps=0, actor_backtracks=0,
                actor_step_accepted=False, actor_lr_factor=0., early_stop=False,
                e_temperature=0., e_kl=0., q_candidate_std=0., q_weighted_advantage=0.,
                e_weight_ess=CANDIDATES, e_max_weight=1. / CANDIDATES)
        states = subset(data['states'], active)
        with torch.no_grad():
            mean, sigma = gaussian_parameters(actor, states)
            candidates = self._draw(mean, sigma)
            expanded = {k: v.repeat_interleave(CANDIDATES, dim=0) for k, v in states.items()}
            qvalues = self.q(expanded, candidates.flatten(0, 1),
                             torch.ones((len(mean) * CANDIDATES, 1), dtype=mean.dtype, device=mean.device))
            qvalues = qvalues.reshape(len(mean), CANDIDATES)
            weights, temperature, divergence = e_step_weights(qvalues)

        def loss_fn():
            current_mean, current_sigma = gaussian_parameters(actor, states)
            logp = gaussian_log_prob(current_mean[:, None, :], current_sigma[:, None, :], candidates)
            return -(weights * logp[..., 0]).sum(1).mean()

        for group in optimizer.param_groups:
            group['lr'] = ACTOR_LR
        stats = kl_step(actor, optimizer, states, loss_fn, max_kl=M_KL,
                        max_backtracks=7, grad_clip=.5)
        self.actor_attempts += stats['actor_optimizer_steps']
        self.actor_committed += stats['actor_committed_steps']
        self.e_steps += 1
        stats.update(e_temperature=temperature, e_kl=divergence,
            q_candidate_std=float(qvalues.std(1, unbiased=False).mean()),
            q_weighted_advantage=float(((weights - 1. / CANDIDATES) * qvalues).sum(1).mean()),
            e_weight_ess=float((1. / weights.square().sum(1)).mean()),
            e_max_weight=float(weights.max(1).values.mean()))
        return stats

    def update(self, actor, critic, ao, vo, batches, rng):
        if len(batches) != 8:
            raise ValueError('MPO requires eight fresh complete episodes per collection')
        current = [self._episode(batch) for batch in batches]
        if any(b['behavior_version'] != self.updates for b in current):
            raise ValueError('New episodes must have current collection version')
        fresh = self._pack(current)
        with torch.no_grad():
            observed = actor.log_prob(fresh['states'], fresh['raw'])
            supplied = gaussian_log_prob(fresh['behavior_mean'], fresh['behavior_sigma'], fresh['raw'])
            likelihood_error = float(((observed - fresh['old_logp']) * fresh['mask']).abs().max())
            behavior_error = float((supplied - fresh['old_logp']).abs().max())
        if likelihood_error > 1e-7 or behavior_error > 1e-7:
            raise AssertionError(('MPO fresh/recorded behavior mismatch', likelihood_error, behavior_error))
        if not self.replay:
            self.buffer.clear()
        self.buffer.extend(current)
        start_candidates = self.candidate_policy_events
        start_reads, start_old = self.replay_episode_reads, self.sampled_old_episodes
        if self.episodes < WARMUP:
            stats = self.native.update(actor, critic, ao, vo, current, rng)
            attempts = int(stats.get('optimizer_steps', 0))
            self.actor_attempts += attempts
            self.actor_committed += attempts
            stats.update(warmup=True, actor_optimizer_steps=attempts,
                actor_committed_steps=attempts, q_optimizer_steps=0,
                e_steps=0, e_kl=0., value_optimizer_steps=attempts)
        else:
            qstats, astats, ages = [], [], []
            for _ in range(Q_UPDATES):
                selected, _ = self._select()
                ages.extend(self.updates - b['behavior_version'] for b in selected)
                qstats.append(self._q_step(actor, self._pack(selected)))
            for _ in range(ACTOR_UPDATES):
                selected, _ = self._select()
                ages.extend(self.updates - b['behavior_version'] for b in selected)
                astats.append(self._mpo_actor_step(actor, ao, self._pack(selected)))
            stats = {key: float(np.mean([item[key] for item in qstats])) for key in qstats[0]}
            for key in ('actor_loss', 'actor_grad_norm', 'macro_kl', 'actual_kl',
                        'actor_lr_factor', 'e_temperature', 'e_kl', 'q_candidate_std',
                        'q_weighted_advantage', 'e_weight_ess', 'e_max_weight'):
                stats[key] = float(np.mean([item[key] for item in astats]))
            stats.update(fit_mc_critic(critic, vo, fresh['states'], fresh['raw'],
                                      fresh['returns'], fresh['mask'], epochs=4))
            stats.update(warmup=False, q_optimizer_steps=Q_UPDATES,
                actor_optimizer_steps=sum(item['actor_optimizer_steps'] for item in astats),
                actor_committed_steps=sum(item['actor_committed_steps'] for item in astats),
                optimizer_steps=sum(item['actor_optimizer_steps'] for item in astats),
                actor_backtracks=sum(item['actor_backtracks'] for item in astats),
                early_stop=any(item['early_stop'] for item in astats), e_steps=ACTOR_UPDATES,
                replay_data_age=float(np.mean(ages)), replay_max_data_age=max(ages, default=0))
        self.episodes += len(current)
        self.updates += 1
        reads, old = self.replay_episode_reads - start_reads, self.sampled_old_episodes - start_old
        stats.update(likelihood_error=likelihood_error, behavior_likelihood_error=behavior_error,
            episodes_seen=self.episodes, updates=self.updates,
            replay_episodes_read=reads, historical_episode_reads=old,
            current_episode_reads=reads - old, buffer_episodes=len(self.buffer),
            candidate_policy_events=self.candidate_policy_events - start_candidates,
            candidate_environment_calls=0, total_q_optimizer_steps=self.q_steps,
            total_actor_adam_attempts=self.actor_attempts, total_actor_committed_steps=self.actor_committed,
            total_e_steps=self.e_steps)
        return stats

    def state_dict(self):
        return dict(version=VERSION, spec=self.spec.to_dict(), event_dim=self.event_dim,
            seed=self.seed, replay=self.replay, metadata=self.metadata(), episodes=self.episodes,
            updates=self.updates, q=cpu_copy(self.q.state_dict()), target_q=cpu_copy(self.target_q.state_dict()),
            q_optimizer=cpu_copy(self.q_optimizer.state_dict()), buffer=cpu_copy(list(self.buffer)),
            torch_generator=self.generator.get_state().cpu().clone(),
            replay_rng=copy.deepcopy(self.replay_rng.bit_generator.state), q_steps=self.q_steps,
            actor_attempts=self.actor_attempts, actor_committed=self.actor_committed,
            candidate_policy_events=self.candidate_policy_events, e_steps=self.e_steps,
            replay_episode_reads=self.replay_episode_reads, sampled_old_episodes=self.sampled_old_episodes)

    def load_state_dict(self, state):
        if (state['version'] != VERSION or state['spec'] != self.spec.to_dict()
                or state['event_dim'] != self.event_dim or state['seed'] != self.seed
                or state['replay'] != self.replay or state['metadata'] != self.metadata()):
            raise ValueError('MPO method/configuration checkpoint mismatch')
        self.q.load_state_dict(state['q'])
        self.target_q.load_state_dict(state['target_q'])
        self.q_optimizer.load_state_dict(state['q_optimizer'])
        self.buffer.clear()
        self.buffer.extend(self._episode(batch) for batch in state['buffer'])
        self.generator.set_state(state['torch_generator'].cpu())
        self.replay_rng.bit_generator.state = copy.deepcopy(state['replay_rng'])
        for name in ('episodes', 'updates', 'q_steps', 'actor_attempts', 'actor_committed',
                     'candidate_policy_events', 'e_steps', 'replay_episode_reads', 'sampled_old_episodes'):
            setattr(self, name, int(state[name]))
