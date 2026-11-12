"""Terminal-reward stochastic Retrace actor--critic with ordinary replay.

This is an ACER-inspired adaptation, not the complete continuous-ACER paper
implementation.  It retains the released A Gaussian and its fixed decoder,
uses no entropy objective, and evaluates the *current* policy with Retrace.
In particular an old trajectory's terminal R is never regressed as current
Q(s,z).  No simulator, decoder derivative, or private transition law is used.

T0 and T1 differ only in the population of complete episodes sampled after
warm-up: the current collection versus a FIFO of the most recent 128.  Both
run four full-8-episode Q/actor updates and the same trust-region backtracking.
Finite-horizon gamma=1, approximate Q, numerical Gaussian integrals, and replay
state occupancy mean this is not an unbiased-policy-gradient guarantee.
"""
from __future__ import annotations

import copy
from collections import deque
import math

import numpy as np
import torch
from torch import nn

from allocation4.common import STATE_KEYS, cpu_copy, subset, tensor_batch
from allocation4.methods.baseline import MCPPO
from allocation4.methods.lax import gaussian_parameters, gaussian_log_prob, kl_step


VERSION = 'complete-gaussian-terminal-retrace-ac-v1'
WARMUP_EPISODES = 32
REPLAY_CAPACITY = 128
LEARN_EPISODES = 8
LEARN_UPDATES = 4
VALUE_CANDIDATES = 16
ACTOR_TRUNCATION = 5.
TARGET_TAU = .005
ACTOR_LR = 3e-3
Q_LR = 1e-3
MAX_KL = .01
KL_ATTEMPTS = 8


def acer_actor_objective(logp, advantage, truncated_ratio, candidate_logp,
                         candidate_advantage, correction):
    """Per-state score surrogate with fixed truncation/bias-correction data.

    All supplied ratios, actions and advantages must be detached.  Only the
    log probabilities carry actor gradients; in particular alternate actions
    must not be reparameterized inside this score-function objective.
    """
    return (truncated_ratio.detach() * logp * advantage.detach()
        + (correction.detach() * candidate_logp * candidate_advantage.detach()).mean(1))


def retrace_targets(rewards, done, q, next_values, ratios, lengths):
    """Detached complete-episode Retrace(lambda=1, gamma=1) targets.

    The ratio at t+1 multiplies the correction from t+1, not the reward or
    correction at t.  No-action states have ratio one, supplied by the caller;
    their zero actor mask must not interrupt an otherwise valid return trace.
    ``next_values`` is E_current_pi Q_target(s[t+1], z), zero at termination.
    """
    q = q.reshape(-1, 1)
    rewards = rewards.to(q).reshape_as(q)
    next_values = next_values.to(q).reshape_as(q)
    ratios = ratios.to(q).reshape_as(q)
    done = done.to(device=q.device, dtype=torch.bool).reshape_as(q)
    if (sum(lengths) != len(q) or any(int(n) != n or n < 1 for n in lengths)
            or not all(torch.isfinite(x).all() for x in (q, rewards, next_values, ratios))
            or bool((ratios < 0).any())):
        raise ValueError('Finite aligned complete trajectories and positive ratios required')
    result = torch.empty_like(q)
    offset = 0
    with torch.no_grad():
        for length in lengths:
            end = offset + int(length)
            if not bool(done[end-1]) or bool(done[offset:end-1].any()):
                raise ValueError('Each supplied episode must end exactly once')
            for t in range(end-1, offset-1, -1):
                if bool(done[t]):
                    result[t] = rewards[t]
                else:
                    correction = ratios[t+1].clamp(max=1.) * (result[t+1]-q[t+1])
                    result[t] = rewards[t] + next_values[t] + correction
            offset = end
    return result.detach()


def _repeat_states(states, count):
    return {key: value.repeat_interleave(count, dim=0) for key, value in states.items()}


class StateActionQ(nn.Module):
    """64/64 MLP over all public regional coordinates, geometry and raw z.

    Regional order is the environment's stable object order.  Only execution
    scales from the public Spec normalize geometry; no response formula or
    demand/reception prediction enters these features.  A zero policy mask
    canonicalizes the entire latent event to zero before the network sees it.
    """
    def __init__(self, spec, event_dim, seed):
        super().__init__()
        self.spec, self.event_dim = spec, int(event_dim)
        width = spec.groups * (spec.local_dim + 1) + spec.global_dim + 2 + self.event_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(294731 + int(seed))
            self.net = nn.Sequential(nn.Linear(width, 64), nn.Tanh(),
                                     nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        self.double()

    def forward(self, states, raw, mask=None):
        if raw.ndim != 2 or raw.shape != (len(states['local']), self.event_dim):
            raise ValueError('One complete raw event required per Q state')
        if mask is None:
            mask = (states['budget'] > 0).to(raw.dtype)[:, None]
        mask = torch.as_tensor(mask, device=raw.device, dtype=raw.dtype).reshape(-1, 1)
        if mask.shape != (len(raw), 1) or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError('Binary complete-event mask required')
        features = torch.cat((states['local'].flatten(1), states['global'],
            states['caps'] / float(self.spec.nominal_total),
            states['budget'][:, None] / float(self.spec.initial_budget),
            states['remaining'][:, None] / float(self.spec.initial_budget), raw * mask), -1)
        if not bool(torch.isfinite(features).all()):
            raise FloatingPointError('Finite public state and event required')
        return self.net(features)


class RetraceAC:
    def __init__(self, spec, event_dim, seed, device, replay=True):
        self.spec, self.event_dim, self.seed = spec, int(event_dim), int(seed)
        self.device, self.replay = str(device), bool(replay)
        self.q = StateActionQ(spec, event_dim, seed).to(self.device)
        self.target_q = copy.deepcopy(self.q).eval()
        for parameter in self.target_q.parameters():
            parameter.requires_grad_(False)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=Q_LR)
        self.generator = torch.Generator(device=self.device).manual_seed(918731 + self.seed)
        self.replay_rng = np.random.default_rng(631921 + self.seed)
        self.buffer = deque(maxlen=REPLAY_CAPACITY if self.replay else LEARN_EPISODES)
        self.episodes, self.updates = 0, 0
        self.q_steps = self.actor_attempts = self.actor_committed = 0
        self.candidate_policy_events = 0
        self.native = MCPPO(spec, event_dim, seed, device)

    def metadata(self):
        return dict(version=VERSION, method='Retrace-AC adaptation', replay=self.replay,
            history='last128 complete episodes' if self.replay else 'current collection only',
            warmup_episodes=WARMUP_EPISODES, warmup='unchanged nativeMC-PPO4epochs plus4private-Q steps',
            post_warmup='four8episode full-batch Q+actor updates per collection',
            actor_lr=ACTOR_LR, q_lr=Q_LR, q_architecture='flatten public local/global/normalized geometry/raw ->64tanh64tanh1',
            policy='unchanged complete A Gaussian: fixed spatial sigma and learned quota sigma',
            actor_objective='truncatedIS score advantage plus current-action bias-correction integral; no entropy',
            actor_truncation=ACTOR_TRUNCATION, advantage_normalization=False,
            actor_kl_limit=MAX_KL, max_adam_attempts=KL_ATTEMPTS, grad_clip=.5,
            retrace_lambda=1., gamma=1., trace_cap=1., target_tau=TARGET_TAU,
            value_candidates=VALUE_CANDIDATES, integral='antithetic Gaussian; separate baseline and correction banks',
            raw_logratio_numeric_bound=40.,
            mask0='raw canonical zero, rho1, actor0, trace continues',
            original_critic='native MC during warmup; frozen compatibility values afterwards',
            replay_target='recomputed under current actor/targetQ; terminal reward only at terminal transition',
            limitations='bootstrapping, finite integrals, function approximation, replay-state occupancy; not full continuous ACER reproduction',
            environment_queries=0, private_model_access=False)

    def _episode(self, batch):
        """Validate/copy only ordinary recorded fields; no incidental info."""
        required = ('raw', 'old_logp', 'mask', 'values', 'returns', 'rewards', 'done',
                    'behavior_mean', 'behavior_sigma', 'behavior_version')
        if any(key not in batch for key in required) or 'next_states' not in batch:
            raise ValueError('Complete transitions and recorded behavior Gaussian required')
        n = len(batch['raw'])
        if n < 1 or n > self.spec.horizon:
            raise ValueError('A nonempty complete episode within horizon required')
        result = {key: cpu_copy(batch[key]) for key in required}
        for which in ('states', 'next_states'):
            if any(key not in batch[which] for key in STATE_KEYS):
                raise ValueError('Public state contract required')
            result[which] = {key: np.asarray(batch[which][key], dtype=np.float64).copy()
                             for key in STATE_KEYS}
            expected = dict(local=(n, self.spec.groups, self.spec.local_dim),
                **{'global':(n, self.spec.global_dim)}, caps=(n, self.spec.groups),
                budget=(n,), remaining=(n,))
            if any(result[which][key].shape != expected[key] for key in STATE_KEYS):
                raise ValueError('Recorded state shape differs from public Spec')
            if not all(np.isfinite(value).all() for value in result[which].values()):
                raise ValueError('Finite recorded public states required')
        for key in ('raw', 'behavior_mean', 'behavior_sigma'):
            result[key] = np.asarray(result[key], dtype=np.float64).copy()
            if result[key].shape != (n, self.event_dim) or not np.isfinite(result[key]).all():
                raise ValueError('Finite full Gaussian event dimensions required')
        if np.any(result['behavior_sigma'] <= 0):
            raise ValueError('Strictly positive recorded Gaussian sigma required')
        for key in ('old_logp', 'mask', 'values'):
            result[key] = np.asarray(result[key], dtype=np.float64).copy()
            if result[key].shape != (n, 1) or not np.isfinite(result[key]).all():
                raise ValueError('One finite complete-event scalar per transition required')
        if not np.isin(result['mask'], (0., 1.)).all():
            raise ValueError('Binary complete-event mask required')
        for key in ('returns', 'rewards'):
            result[key] = np.asarray(result[key], dtype=np.float64).reshape(-1).copy()
            if result[key].shape != (n,) or not np.isfinite(result[key]).all():
                raise ValueError('Finite transition rewards/compatibility returns required')
        done = np.asarray(result['done'])
        if done.shape != (n,) or not np.isin(done, (0, 1)).all():
            raise ValueError('One binary termination flag per transition required')
        result['done'] = done.astype(bool, copy=True)
        if not result['done'][-1] or result['done'][:-1].any():
            raise ValueError('Only complete, singly terminated episodes may enter replay')
        if np.any(result['rewards'][:-1] != 0.):
            raise ValueError('This registered task supplies only terminal rewards')
        if not np.allclose(result['returns'], result['rewards'][-1], rtol=0, atol=1e-12):
            raise ValueError('Native MC compatibility return must equal terminal reward')
        version = result['behavior_version']
        if isinstance(version, bool) or int(version) != version or version < 0:
            raise ValueError('Nonnegative behavior version required')
        result['behavior_version'] = int(version)
        return result

    def _pack(self, batches):
        data = tensor_batch(batches, self.device)
        data['next_states'] = {key: torch.as_tensor(np.concatenate([b['next_states'][key] for b in batches]),
            dtype=torch.float64, device=self.device) for key in STATE_KEYS}
        for key in ('rewards', 'behavior_mean', 'behavior_sigma'):
            data[key] = torch.as_tensor(np.concatenate([b[key] for b in batches]), dtype=torch.float64, device=self.device)
        data['done'] = torch.as_tensor(np.concatenate([b['done'] for b in batches]), dtype=torch.bool, device=self.device)
        return data

    def _draw(self, mean, sigma):
        half = torch.randn((len(mean), VALUE_CANDIDATES//2, self.event_dim),
            dtype=mean.dtype, device=mean.device, generator=self.generator)
        noise = torch.cat((half, -half), dim=1)
        self.candidate_policy_events += len(mean) * VALUE_CANDIDATES
        return (mean[:, None, :] + sigma[:, None, :] * noise).detach()

    @torch.no_grad()
    def _value(self, actor, q, states, mask=None):
        mean, sigma = gaussian_parameters(actor, states)
        actions = self._draw(mean, sigma)
        if mask is None:
            mask = (states['budget'] > 0).to(mean.dtype)[:, None]
        prediction = q(_repeat_states(states, VALUE_CANDIDATES), actions.flatten(0, 1),
                       mask.repeat_interleave(VALUE_CANDIDATES, dim=0))
        return prediction.reshape(len(mean), VALUE_CANDIDATES, 1).mean(1)

    @torch.no_grad()
    def _targets(self, actor, data):
        mean, sigma = gaussian_parameters(actor, data['states'])
        logratio = gaussian_log_prob(mean, sigma, data['raw']) - gaussian_log_prob(
            data['behavior_mean'], data['behavior_sigma'], data['raw'])
        logratio = torch.where(data['mask'] > 0, logratio, torch.zeros_like(logratio))
        ratios = logratio.clamp(-40., 40.).exp()
        old_q = self.target_q(data['states'], data['raw'], data['mask'])
        next_v = torch.zeros_like(old_q)
        alive = ~data['done']
        if bool(alive.any()):
            next_v[alive] = self._value(actor, self.target_q, subset(data['next_states'], alive))
        target = retrace_targets(data['rewards'], data['done'], old_q, next_v, ratios, data['lengths'])
        return target, ratios, logratio

    def _q_step(self, actor, data):
        target, ratios, logratio = self._targets(actor, data)
        prediction = self.q(data['states'], data['raw'], data['mask'])
        loss = (prediction-target).square().mean()
        self.q_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.q.parameters(), .5)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite Retrace Q fit')
        self.q_optimizer.step()
        self.q_steps += 1
        with torch.no_grad():
            for destination, source in zip(self.target_q.parameters(), self.q.parameters()):
                destination.lerp_(source, TARGET_TAU)
        return target, dict(q_loss=float(loss.detach()), q_grad_norm=float(norm),
            q_target_mean=float(target.mean()), q_target_std=float(target.std(unbiased=False)),
            replay_ess=float(ratios.sum().square()/ratios.square().sum().clamp_min(1e-30)),
            replay_ratio_mean=float(ratios.mean()), replay_ratio_max=float(ratios.max()),
            replay_trace_clip_fraction=float((ratios>1).to(torch.float64).mean()),
            numeric_ratio_clip_fraction=float((logratio.abs()>40).to(torch.float64).mean()))

    def _actor_step(self, actor, optimizer, data, target):
        active = data['mask'].reshape(-1) > 0
        if not bool(active.any()):
            return dict(actor_loss=0., actor_grad_norm=0., macro_kl=0., actual_kl=0.,
                actor_optimizer_steps=0, actor_committed_steps=0, actor_backtracks=0,
                actor_step_accepted=False, actor_lr_factor=0., early_stop=False,
                actor_correction_rms=0., raw_advantage_std=0.)
        states, raw, mask = data['states'], data['raw'], data['mask']
        # Everything except log pi is frozen throughout retries.  The baseline
        # bank is independent of the correction bank: using the same sample's
        # Q in its own baseline introduces a finite-bank score bias.
        with torch.no_grad():
            mean, sigma = gaussian_parameters(actor, states)
            baseline = self._value(actor, self.q, states, mask)
            candidates = self._draw(mean, sigma)
            expanded = _repeat_states(states, VALUE_CANDIDATES)
            q_actions = self.q(expanded, candidates.flatten(0, 1),
                mask.repeat_interleave(VALUE_CANDIDATES, dim=0)).reshape(len(raw), VALUE_CANDIDATES, 1)
            candidate_adv = q_actions-baseline[:, None, :]
            main_adv = target.detach()-baseline
            behavior_lp = gaussian_log_prob(data['behavior_mean'], data['behavior_sigma'], raw)
            logratio = gaussian_log_prob(mean, sigma, raw)-behavior_lp
            rho_bar = torch.where(mask > 0, logratio.clamp(-40, 40).exp(), torch.ones_like(mask)).clamp(max=ACTOR_TRUNCATION)
            candidate_mu_lp = gaussian_log_prob(data['behavior_mean'][:, None, :],
                data['behavior_sigma'][:, None, :], candidates)
            candidate_ratio_log = gaussian_log_prob(mean[:, None, :], sigma[:, None, :], candidates)-candidate_mu_lp
            correction = (1.-torch.exp((math.log(ACTOR_TRUNCATION)-candidate_ratio_log).clamp(max=0.))).clamp_min(0.)
            correction *= mask[:, None, :]

        def loss_fn():
            current_mean, current_sigma = gaussian_parameters(actor, states)
            lp = gaussian_log_prob(current_mean, current_sigma, raw)
            alternate_lp = gaussian_log_prob(current_mean[:, None, :], current_sigma[:, None, :], candidates)
            objective = acer_actor_objective(lp, main_adv, rho_bar, alternate_lp,
                                            candidate_adv, correction)
            return -(objective * mask).mean()

        for group in optimizer.param_groups:
            group['lr'] = ACTOR_LR
        stats = kl_step(actor, optimizer, subset(states, active), loss_fn,
            max_kl=MAX_KL, max_backtracks=KL_ATTEMPTS-1, grad_clip=.5)
        self.actor_attempts += stats['actor_optimizer_steps']
        self.actor_committed += stats['actor_committed_steps']
        stats.update(actor_correction_rms=float((correction*candidate_adv).square().mean().sqrt()),
            raw_advantage_std=float(main_adv[active].std(unbiased=False)))
        return stats

    def update(self, actor, critic, ao, vo, batches, rng):
        if not batches or len(batches) > LEARN_EPISODES:
            raise ValueError('One to eight newly collected complete episodes required')
        current = [self._episode(batch) for batch in batches]
        fresh = self._pack(current)
        with torch.no_grad():
            mean, sigma = gaussian_parameters(actor, fresh['states'])
            observed = gaussian_log_prob(mean, sigma, fresh['raw'])
            supplied = gaussian_log_prob(fresh['behavior_mean'], fresh['behavior_sigma'], fresh['raw'])
            error = float(((observed-fresh['old_logp'])*fresh['mask']).abs().max())
            behavior_error = float(((supplied-fresh['old_logp'])*fresh['mask']).abs().max())
        if error > 1e-7 or behavior_error > 1e-7:
            raise AssertionError(('Fresh behavior likelihood mismatch', error, behavior_error))
        if not self.replay:
            self.buffer.clear()
        self.buffer.extend(current)
        candidates_before = self.candidate_policy_events
        read_episodes = 0
        if self.episodes < WARMUP_EPISODES:
            stats = self.native.update(actor, critic, ao, vo, current, rng)
            qstats = []
            for _ in range(LEARN_UPDATES):
                _, item = self._q_step(actor, fresh)
                qstats.append(item)
                read_episodes += len(current)
            stats.update({key:float(np.mean([item[key] for item in qstats])) for key in qstats[0]})
            stats.update(warmup=True, q_optimizer_steps=LEARN_UPDATES,
                actor_committed_steps=int(stats.get('optimizer_steps', 0)),
                actor_optimizer_steps=int(stats.get('optimizer_steps', 0)),
                value_optimizer_steps=int(stats.get('optimizer_steps', 0)), replay_data_age=0.)
            self.actor_attempts += int(stats.get('optimizer_steps', 0))
            self.actor_committed += int(stats.get('optimizer_steps', 0))
        else:
            qstats, astats, ages = [], [], []
            pool = list(self.buffer)
            for _ in range(LEARN_UPDATES):
                indices = self.replay_rng.choice(len(pool), size=min(LEARN_EPISODES, len(pool)), replace=False)
                selected = [pool[int(index)] for index in indices]
                data = self._pack(selected)
                target, qs = self._q_step(actor, data)
                act = self._actor_step(actor, ao, data, target)
                qstats.append(qs)
                astats.append(act)
                ages.extend(max(0, self.updates-b['behavior_version']) for b in selected)
                read_episodes += len(selected)
            stats = {key:float(np.mean([item[key] for item in qstats])) for key in qstats[0]}
            for key in ('actor_loss', 'actor_grad_norm', 'macro_kl', 'actual_kl',
                        'actor_correction_rms', 'raw_advantage_std', 'actor_lr_factor'):
                stats[key] = float(np.mean([item.get(key, 0.) for item in astats]))
            stats.update(warmup=False, value_loss=0., value_optimizer_steps=0,
                q_optimizer_steps=LEARN_UPDATES,
                actor_optimizer_steps=sum(item['actor_optimizer_steps'] for item in astats),
                actor_committed_steps=sum(item['actor_committed_steps'] for item in astats),
                actor_backtracks=sum(item['actor_backtracks'] for item in astats),
                optimizer_steps=sum(item['actor_optimizer_steps'] for item in astats),
                early_stop=any(item.get('early_stop', False) for item in astats),
                replay_data_age=float(np.mean(ages)), replay_max_data_age=max(ages, default=0))
        self.episodes += len(current)
        self.updates += 1
        stats.update(likelihood_error=error, behavior_likelihood_error=behavior_error,
            episodes_seen=self.episodes, updates=self.updates,
            candidate_policy_events=self.candidate_policy_events-candidates_before,
            replay_episodes_read=read_episodes, buffer_episodes=len(self.buffer),
            total_q_optimizer_steps=self.q_steps, total_actor_adam_attempts=self.actor_attempts,
            total_actor_committed_steps=self.actor_committed)
        return stats

    def state_dict(self):
        return dict(version=VERSION, spec=self.spec.to_dict(), event_dim=self.event_dim,
            seed=self.seed, replay=self.replay, metadata=self.metadata(), episodes=self.episodes,
            updates=self.updates, q=cpu_copy(self.q.state_dict()), target_q=cpu_copy(self.target_q.state_dict()),
            q_optimizer=cpu_copy(self.q_optimizer.state_dict()), buffer=cpu_copy(list(self.buffer)),
            torch_generator=self.generator.get_state().cpu().clone(),
            replay_rng=copy.deepcopy(self.replay_rng.bit_generator.state), q_steps=self.q_steps,
            actor_attempts=self.actor_attempts, actor_committed=self.actor_committed,
            candidate_policy_events=self.candidate_policy_events)

    def load_state_dict(self, state):
        if (state['version'] != VERSION or state['spec'] != self.spec.to_dict()
                or state['event_dim'] != self.event_dim or state['seed'] != self.seed
                or state['replay'] != self.replay or state['metadata'] != self.metadata()):
            raise ValueError('Retrace method/configuration checkpoint mismatch')
        self.q.load_state_dict(state['q'])
        self.target_q.load_state_dict(state['target_q'])
        self.q_optimizer.load_state_dict(state['q_optimizer'])
        self.buffer.clear()
        self.buffer.extend(self._episode(batch) for batch in state['buffer'])
        self.generator.set_state(state['torch_generator'].cpu())
        self.replay_rng.bit_generator.state = copy.deepcopy(state['replay_rng'])
        for name in ('episodes', 'updates', 'q_steps', 'actor_attempts', 'actor_committed', 'candidate_policy_events'):
            setattr(self, name, int(state[name]))
