"""Finite-horizon GePPO-style recent replay on complete Gaussian events.

This is an explicitly labelled adaptation: fixed native learning rates, M=2
uniform recent-policy mixture, and frozen canonical V-trace targets. It does
not claim the discounted infinite-horizon theorem at gamma=1, an unbiased
current-policy gradient, or reproduction of the paper's adaptive learning rate.
Only ordinary public transitions and the native terminal reward are used.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import torch

from allocation23.learning import PPO
from allocation4.common import STATE_KEYS, subset, tensor_batch
from allocation4.methods.baseline import MCPPO
from allocation4.methods.lax import gaussian_kl, gaussian_log_prob, gaussian_parameters


VERSION = 'finite-horizon-geppo-style-M2-vtrace-fixed-native-lr-v1'


def generalized_clip(ratio, center, epsilon):
    """Clip pi_theta/mu around pi_k/mu, NOT around one for old policies."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Positive finite generalized clip required')
    return torch.minimum(torch.maximum(ratio, center - epsilon), center + epsilon)


@torch.no_grad()
def vtrace_targets(rewards, done, values, next_values, log_rhos, mask, lengths):
    """Canonical finite-episode V-trace, gamma=lambda=rho_cap=c_cap=1.

    Inputs are flat one-dimensional tensors, with explicit episode boundaries.
    An inactive policy event has ratio one; it is not an environment terminal.
    The PG label omits the local importance factor, because the GePPO surrogate
    already contains pi_theta/mu. Rho clipping changes the evaluated policy;
    c clipping shortens traces and is not a proof of unbiased current Q.
    """
    arrays = [rewards, done, values, next_values, log_rhos, mask]
    if any(x.ndim != 1 or x.shape != values.shape for x in arrays):
        raise ValueError('Flat, equally sized V-trace arrays required')
    if any(not torch.isfinite(x).all() for x in arrays):
        raise FloatingPointError('Finite ordinary V-trace inputs required')
    if sum(lengths) != len(values) or any(int(n) != n or n <= 0 for n in lengths):
        raise ValueError('Explicit nonempty episode lengths required')
    if not bool(((done == 0) | (done == 1)).all()):
        raise ValueError('Binary true terminal flags required')
    rho = torch.exp(torch.minimum(log_rhos, torch.zeros_like(log_rhos)))
    rho = torch.where(mask > 0, rho, torch.ones_like(rho))
    targets, advantage = torch.empty_like(values), torch.empty_like(values)
    first = 0
    for count in lengths:
        stop = first + int(count)
        # A collection cut can bootstrap; a true task end below sets discount=0.
        following = next_values[stop - 1]
        for t in range(stop - 1, first - 1, -1):
            discount = 1. - done[t]
            delta = rho[t] * (rewards[t] + discount * next_values[t] - values[t])
            targets[t] = values[t] + delta + discount * rho[t] * (following - next_values[t])
            advantage[t] = rewards[t] + discount * following - values[t]
            following = targets[t]
        first = stop
    return targets.detach(), advantage.detach(), rho.detach()


def _spec_dict(spec):
    if not callable(getattr(spec, 'to_dict', None)):
        raise TypeError('The shared public Spec.to_dict interface is required')
    return copy.deepcopy(spec.to_dict())


def _same_json(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


class GePPO:
    def __init__(self, spec, event_dim, seed, device):
        self.spec, self.event_dim, self.seed, self.device = spec, int(event_dim), int(seed), str(device)
        if self.event_dim <= 0:
            raise ValueError('Complete positive joint event dimension required')
        self.episodes, self.updates = 0, 0
        self.history = []
        self.shuffle_rng = np.random.default_rng(740117 + self.seed)
        self.native = MCPPO(spec, event_dim, seed, device)

    def metadata(self):
        return dict(version=VERSION, spec=_spec_dict(self.spec), event_dim=self.event_dim,
                    seed=self.seed, warmup_episodes=32, new_batch_episodes=8,
                    policy_mixture='uniform current8 plus immediately previous8 complete episodes',
                    persistent_buffer_episodes=8, policy_count=2, gamma=1., trace_lambda=1.,
                    rho_cap=1., c_cap=1., epochs=4, minibatch=PPO['minibatch'],
                    clip=PPO['clip'], mixed_clip=PPO['clip']/1.5,
                    clipping_center='frozen current pi_k / stored behavior mu, joint latent event',
                    targets='frozen pre-update V-trace; actor label r + vtrace_next - currentV',
                    learning_rate='fixed native; NO original GePPO adaptive learning-rate controller',
                    shuffle='caller RNG during exact native warmup; private checkpointed RNG afterward',
                    terminal_contract='native intermediate0, terminalR once; action mask is not done',
                    theoretical_scope='finite-horizon gamma1 adaptation; clipped rho evaluates an implicit truncated policy; no unbiasedness or original-theorem claim',
                    candidate_policy_events=0, auxiliary_models=False, private_feedback=False)

    def _validate(self, batches):
        if len(batches) != 8:
            raise ValueError('Exactly eight new complete episodes required')
        versions = []
        for batch in batches:
            missing = set(('states', 'raw', 'old_logp', 'mask', 'values', 'returns',
                           'next_states', 'rewards', 'done', 'behavior_mean',
                           'behavior_sigma', 'behavior_version')) - set(batch)
            if missing:
                raise ValueError(('Missing ordinary episode fields', sorted(missing)))
            n = len(batch['raw'])
            if n != self.spec.horizon or np.shape(batch['raw']) != (n, self.event_dim):
                raise ValueError('Complete native-horizon, joint-event episodes required')
            for key in ('old_logp', 'mask', 'values'):
                if np.shape(batch[key]) != (n, 1):
                    raise ValueError(('Scalar macro-event shape', key))
            for key in ('returns', 'rewards', 'done'):
                if np.shape(batch[key]) != (n,):
                    raise ValueError(('Flat episode shape', key))
            for key in ('behavior_mean', 'behavior_sigma'):
                if np.shape(batch[key]) != (n, self.event_dim):
                    raise ValueError(('Full behavior Gaussian required', key))
            for key in ('raw', 'old_logp', 'mask', 'values', 'returns', 'rewards',
                        'done', 'behavior_mean', 'behavior_sigma'):
                if not np.isfinite(batch[key]).all():
                    raise FloatingPointError(('Nonfinite saved ordinary field', key))
            if not np.all(np.asarray(batch['behavior_sigma']) > 0):
                raise ValueError('Positive behavior Gaussian scales required')
            expected_logp = (-.5 * ((np.asarray(batch['raw']) - batch['behavior_mean']) /
                                    batch['behavior_sigma']) ** 2 - np.log(batch['behavior_sigma'])
                             - .5 * np.log(2 * np.pi)).sum(-1, keepdims=True)
            if not np.allclose(expected_logp, batch['old_logp'], rtol=0, atol=1e-7):
                raise ValueError('Saved log probability must include the full behavior Gaussian event')
            if not np.isin(batch['mask'], [0, 1]).all():
                raise ValueError('Binary action-event mask required')
            done, rewards = np.asarray(batch['done']), np.asarray(batch['rewards'])
            if not (np.all(done[:-1] == 0) and done[-1] == 1):
                raise ValueError('Only the final native task step is terminal')
            if not np.all(rewards[:-1] == 0):
                raise ValueError('The environment must supply only native terminal feedback')
            if not np.allclose(batch['returns'], rewards[-1], rtol=0, atol=1e-12):
                raise ValueError('Stored native MC return must equal terminal reward')
            for container in ('states', 'next_states'):
                if set(batch[container]) != set(STATE_KEYS):
                    raise ValueError('Exactly the shared public encoded state fields required')
                shapes = dict(local=(n, self.spec.groups, self.spec.local_dim),
                              global_=(n, self.spec.global_dim), caps=(n, self.spec.groups),
                              budget=(n,), remaining=(n,))
                for key in STATE_KEYS:
                    value = np.asarray(batch[container][key])
                    shape = shapes['global_' if key == 'global' else key]
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError(('Finite step-aligned public state required', key))
            for key in STATE_KEYS:
                if not np.allclose(batch['next_states'][key][:-1], batch['states'][key][1:], rtol=0, atol=1e-12):
                    raise ValueError(('Next states must stay inside each physical episode', key))
            version = batch['behavior_version']
            if isinstance(version, bool) or not isinstance(version, (int, np.integer)) or version < 0:
                raise ValueError('Nonnegative integer behavior-policy version required')
            versions.append(int(version))
        if len(set(versions)) != 1:
            raise ValueError('A new batch must come from one frozen behavior policy')
        return versions[0]

    @torch.no_grad()
    def _prepare(self, actor, critic, batches):
        data = tensor_batch(batches, self.device)
        states, raw = data['states'], data['raw']
        next_states = {k: torch.as_tensor(np.concatenate([b['next_states'][k] for b in batches]),
                                        dtype=torch.float64, device=self.device) for k in STATE_KEYS}
        flat = {k: torch.as_tensor(np.concatenate([b[k] for b in batches]), dtype=torch.float64,
                                  device=self.device) for k in ('rewards', 'done', 'behavior_mean', 'behavior_sigma')}
        reconstructed = gaussian_log_prob(flat['behavior_mean'], flat['behavior_sigma'], raw)
        behavior_error = float((reconstructed - data['old_logp']).abs().max())
        if behavior_error > 1e-7:
            raise AssertionError(('Incomplete or inconsistent behavior likelihood', behavior_error))
        anchor_mean, anchor_sigma = gaussian_parameters(actor, states)
        anchor_mean, anchor_sigma = anchor_mean.detach().clone(), anchor_sigma.detach().clone()
        anchor_logp = actor.log_prob(states, raw).detach().clone()
        values = critic(states, raw).reshape(-1).detach()
        next_values = critic(next_states, raw).reshape(-1).detach()
        log_rhos = (anchor_logp - data['old_logp']).reshape(-1)
        target, advantage, rho = vtrace_targets(flat['rewards'], flat['done'], values, next_values,
                                              log_rhos, data['mask'].reshape(-1), data['lengths'])
        data.update(anchor_logp=anchor_logp, anchor_mean=anchor_mean, anchor_sigma=anchor_sigma,
                    targets=target[:, None], advantage=advantage[:, None],
                    log_rhos=log_rhos, capped_rhos=rho,
                    behavior_mean=flat['behavior_mean'], behavior_sigma=flat['behavior_sigma'])
        return data, behavior_error

    def _finish(self, batches):
        self.history = copy.deepcopy(batches)
        self.episodes += len(batches)
        self.updates += 1

    def update(self, actor, critic, ao, vo, batches, rng):
        current_version = self._validate(batches)
        if self.history:
            old_version = self._validate(self.history)
            if old_version != current_version - 1:
                raise ValueError('Replay must be the immediately preceding behavior-policy batch')
        if self.episodes < 32:
            # Do not recompute/replace native labels or advance private RNG here.
            stats = self.native.update(actor, critic, ao, vo, batches, rng)
            self._finish(batches)
            stats.update(replay_enabled=False, replay_episodes_read=0, buffer_episodes=8,
                         candidate_policy_events=0, episodes_seen=self.episodes, updates=self.updates,
                         mean_policy_age=0., mean_age_plus_one=1., generalized_clip=PPO['clip'])
            return stats
        if not self.history:
            raise ValueError('Post-warmup GePPO requires its checkpointed previous batch')
        # Equal episode counts and fixed domain horizon give a uniform M=2 mixture.
        combined = list(batches) + list(self.history)
        data, behavior_error = self._prepare(actor, critic, combined)
        states, raw, old, mask = data['states'], data['raw'], data['old_logp'], data['mask']
        active = mask > 0
        new_count = len(batches) * self.spec.horizon
        current_error = float(((data['anchor_logp'][:new_count] - old[:new_count]) * mask[:new_count]).abs().max())
        if current_error > 1e-7:
            raise AssertionError(('New batch is not from the current pre-update policy', current_error))
        epsilon = PPO['clip'] / 1.5
        advantage = data['advantage']
        scale = advantage[active].std(unbiased=False) if active.any() else advantage.new_tensor(0.)
        if active.any():
            advantage = ((advantage - advantage[active].mean()) / (scale + 1e-8)).detach()
        center = torch.exp((data['anchor_logp'] - old).clamp(-40., 40.)).detach()
        records, stop = [], False
        if active.any():
            for _ in range(PPO['epochs']):
                indices = self.shuffle_rng.permutation(len(raw))
                for offset in range(0, len(raw), PPO['minibatch']):
                    ix = torch.as_tensor(indices[offset:offset + PPO['minibatch']], device=self.device)
                    st, m = subset(states, ix), mask[ix]
                    lp = actor.log_prob(st, raw[ix])
                    if not torch.isfinite(lp).all():
                        raise FloatingPointError('Nonfinite current joint latent likelihood')
                    logratio = lp - old[ix]
                    ratio = logratio.clamp(-40., 40.).exp()
                    clipped = generalized_clip(ratio, center[ix], epsilon)
                    loss = -(torch.minimum(ratio * advantage[ix], clipped * advantage[ix]) * m).sum(-1).mean()
                    ao.zero_grad()
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), PPO['grad_clip'])
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Nonfinite GePPO actor gradient')
                    ao.step()
                    prediction = critic(st, raw[ix])
                    value_loss = (((prediction - data['targets'][ix]) ** 2) * m).sum() / m.sum().clamp_min(1)
                    vo.zero_grad()
                    value_loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), PPO['grad_clip'])
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Nonfinite GePPO value gradient')
                    vo.step()
                    with torch.no_grad():
                        # pi_k-relative update statistic, rather than confusing old
                        # policy mismatch with a new SGD update away from pi_k.
                        delta = lp - data['anchor_logp'][ix]
                        update_ratio = delta.clamp(-40., 40.).exp()
                        kl = ((update_ratio - 1. - delta) * m).sum(-1).mean()
                        fraction = (((ratio - center[ix]).abs() > epsilon) * m).sum() / m.sum().clamp_min(1)
                    records.append((float(loss), float(value_loss), float(kl), float(fraction)))
                    if float(kl) > PPO['macro_kl_stop']:
                        stop = True
                        break
                if stop:
                    break
        with torch.no_grad():
            mean, sigma = gaussian_parameters(actor, states)
            actual_kl = float(gaussian_kl(data['anchor_mean'], data['anchor_sigma'], mean, sigma).mean().clamp_min(0))
            log_weights = data['log_rhos'][active.reshape(-1)]
            if len(log_weights):
                normalized = (log_weights - torch.logsumexp(log_weights, 0)).exp()
                ess = float(1. / normalized.square().sum())
                max_weight = float(log_weights.clamp(-40, 40).exp().max())
                cap_fraction = float((log_weights > 1e-12).double().mean())
            else:
                ess, max_weight, cap_fraction = 0., 0., 0.
            trajectory_logweights, first = [], 0
            for count in data['lengths']:
                sl = slice(first, first + count)
                trajectory_logweights.append((data['log_rhos'][sl] * mask[sl].reshape(-1)).sum())
                first += count
            tw = torch.stack(trajectory_logweights)
            tn = (tw - torch.logsumexp(tw, 0)).exp()
            trajectory_ess = float(1. / tn.square().sum())
            behavior_kl = float(gaussian_kl(data['behavior_mean'], data['behavior_sigma'],
                                           data['anchor_mean'], data['anchor_sigma']).mean().clamp_min(0))
        a = np.asarray(records) if records else np.zeros((1, 4))
        self._finish(batches)
        return dict(actor_loss=float(a[:, 0].mean()), value_loss=float(a[:, 1].mean()),
                    macro_kl=float(a[:, 2].mean()), clip_fraction=float(a[:, 3].mean()),
                    optimizer_steps=len(records), advantage_std=float(scale), early_stop=stop,
                    likelihood_error=current_error, behavior_likelihood_error=behavior_error,
                    actual_joint_kl=actual_kl, replay_actual_joint_kl=actual_kl,
                    behavior_to_anchor_kl=behavior_kl, ess=ess,
                    ess_fraction=ess / max(int(active.sum()), 1), trajectory_ess=trajectory_ess,
                    trajectory_ess_fraction=trajectory_ess / len(combined),
                    max_importance_weight=max_weight, rho_capped_fraction=cap_fraction,
                    logratio_numerical_guard_count=int((data['log_rhos'].abs() > 40).sum()),
                    vtrace_target_recomputations=1, replay_enabled=True,
                    replay_episodes_read=8, replay_decisions_read=8 * self.spec.horizon,
                    buffer_episodes=8, mixture_episodes=16, mean_policy_age=.5,
                    mean_age_plus_one=1.5, generalized_clip=epsilon,
                    candidate_policy_events=0, episodes_seen=self.episodes, updates=self.updates)

    def state_dict(self):
        return dict(metadata=self.metadata(), episodes=self.episodes, updates=self.updates,
                    history=copy.deepcopy(self.history), shuffle_rng=copy.deepcopy(self.shuffle_rng.bit_generator.state))

    def load_state_dict(self, state):
        expected = {'metadata', 'episodes', 'updates', 'history', 'shuffle_rng'}
        if set(state) != expected or not _same_json(state['metadata'], self.metadata()):
            raise ValueError('GePPO method/spec/seed checkpoint mismatch')
        episodes, updates = state['episodes'], state['updates']
        if (isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 0
                or isinstance(updates, bool) or not isinstance(updates, int) or updates < 0
                or episodes != 8 * updates):
            raise ValueError('Inconsistent GePPO checkpoint chronology')
        history = copy.deepcopy(state['history'])
        if bool(history) != bool(updates):
            raise ValueError('Previous complete batch is required after any update')
        if history:
            self._validate(history)
        probe = np.random.default_rng()
        probe.bit_generator.state = copy.deepcopy(state['shuffle_rng'])
        self.episodes, self.updates, self.history = episodes, updates, history
        self.shuffle_rng = probe
