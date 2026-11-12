"""Finite-episode snapshot policy gradients on native terminal returns.

S1 reuses a *reference gradient*, not a replay PPO training population.  P0 is
the matched current-trajectory estimator with exactly the same collection and
update schedule.  Complete raw Gaussian event densities are used.  No reward
shaping, simulator labels, per-batch advantage normalization, likelihood clipping,
or self-normalized importance sampling is applied to the estimator.

A finite sampled reference gradient is not the exact reference gradient.  Thus
the S1 estimate is NOT conditionally unbiased given that sampled reference.
The registered Adam/gradient clipping/KL guard further concern an optimizer,
not a claim that every resulting update is an unbiased policy gradient step.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from allocation4.common import cpu_copy, subset, tensor_batch
from allocation4.methods.baseline import MCPPO
from allocation4.methods.lax import gaussian_kl, gaussian_parameters
from .geppo import GePPO, _same_json, _spec_dict


VERSION = 'finite-terminal-snapshot-PG-reference16-inner4x8-v1'
REFERENCE_EPISODES = 16
INNER_BATCH_EPISODES = 4
INNER_UPDATES = 8
KL_LIMIT = .02
MAX_BACKTRACKS = 8


def flatten_gradients(model, objective):
    """Detached ascent gradient; unused registered parameters contribute zeros."""
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(objective, parameters, allow_unused=True)
    return torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1)
                      for p, g in zip(parameters, gradients)])


def episode_gradients(actor, batches, device):
    """Rows h_theta(tau)=R(tau) sum_active grad log pi_theta(raw|state)."""
    rows = []
    for batch in batches:
        data = tensor_batch([batch], device)
        objective = (actor.log_prob(data['states'], data['raw']) * data['mask']
                     * data['returns'][:, None]).sum()
        gradient = flatten_gradients(actor, objective)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('Finite trajectory-score gradient required')
        rows.append(gradient)
    if not rows:
        raise ValueError('A nonempty complete-episode population is required')
    return torch.stack(rows)


@torch.no_grad()
def trajectory_log_weights(reference, current, batches, device):
    """Exact log p_phi(tau)/p_theta(tau); inactive draws are marginalized out."""
    values = []
    for batch in batches:
        data = tensor_batch([batch], device)
        delta = reference.log_prob(data['states'], data['raw']) - current.log_prob(data['states'], data['raw'])
        values.append((delta * data['mask']).sum())
    return torch.stack(values)


def variance_trace(rows):
    """Across-trajectory sample covariance trace, not gradient error to truth."""
    if len(rows) < 2:
        return None
    result = float((rows - rows.mean(0)).square().sum() / (len(rows)-1))
    return result if math.isfinite(result) else None


def _nullable(value):
    value = float(value)
    return value if math.isfinite(value) else None


class SnapshotPG:
    def __init__(self, spec, event_dim, seed, device, variance_reduced=True):
        if not isinstance(variance_reduced, (bool, np.bool_)):
            raise ValueError('An explicit P0/S1 estimator flag is required')
        self.spec, self.event_dim = spec, int(event_dim)
        self.seed, self.device = int(seed), str(device)
        self.variance_reduced = bool(variance_reduced)
        self.native = MCPPO(spec, event_dim, seed, device)
        self.validator = GePPO(spec, event_dim, seed, device)
        self.episodes = self.updates = self.cycles = 0
        self.phase = 'reference'
        self.inner_updates = 0
        self.reference_batches = []
        self.reference_gradient = None
        self.reference_actor = None
        self.reference_variance = None
        self.reference_mean_score = None
        self.pending_reference_state = None

    def metadata(self):
        return dict(version=VERSION, spec=_spec_dict(self.spec), event_dim=self.event_dim,
            seed=self.seed, variance_reduced=self.variance_reduced,
            warmup_episodes=32, warmup_batch_episodes=8, reference_episodes=REFERENCE_EPISODES,
            inner_new_episodes=INNER_BATCH_EPISODES, inner_updates=INNER_UPDATES,
            collection='reference16 (8+8), then eight current4 batches; all new episodes charged',
            score_gradient='native terminalR times sum of active complete-joint Gaussian log-score gradients',
            baseline='zero; no return centering or advantage normalization',
            reference='finite sampled mean trajectory gradient, fixed until refresh',
            correction='reference/current full-trajectory density ratio, exact exp(sum active log ratios)',
            clipping='no likelihood clipping or self normalization; nonfinite correction rejects actor update',
            actor_optimizer='passed native Adam, one proposal per complete reference or inner batch',
            actor_initial_lr=3e-4, gradient_clip=.5, kl_limit=KL_LIMIT,
            guard='shared P0/S1 analytic joint Gaussian KL backtracking on current states; 8 halvings',
            critic='native critic unchanged after warm32; no Q model or learned auxiliary feedback',
            match='P0/S1 same collection and proposal count; not compute-matched to multi-epoch PPO',
            actor_loss_log='negative squared pre-clipping estimator norm, a diagnostic proxy rather than a PPO objective',
            mask='inactive Gaussian events marginalized with ratio1; mask is not task termination',
            gamma=1., auxiliary_models=False, private_feedback=False, candidate_policy_events=0,
            theoretical_scope='finite-reference error remains; no conditional unbiasedness, complexity or monotonic-improvement guarantee')

    def _ensure_reference(self, actor):
        if self.pending_reference_state is not None:
            self.reference_actor = copy.deepcopy(actor)
            self.reference_actor.load_state_dict(self.pending_reference_state)
            self.pending_reference_state = None
        if self.episodes >= 32 and self.phase == 'reference' and not self.reference_batches and self.reference_actor is None:
            self.reference_actor = copy.deepcopy(actor)

    def collection_size(self, actor, max_new=8):
        if max_new not in (4, 8):
            raise ValueError('The runner must cap collection to four/eight complete episodes')
        self._ensure_reference(actor)
        if self.episodes < 32:
            if max_new < 8:
                raise ValueError('Native common warmup requires eight complete episodes')
            return 8
        if self.phase == 'inner':
            return INNER_BATCH_EPISODES
        remaining = REFERENCE_EPISODES - len(self.reference_batches)
        if remaining not in (4, 8, 12, 16):
            raise ValueError('Invalid partial reference population')
        return min(8, max_new, remaining)

    def _validate_current(self, actor, batches):
        if len(batches) not in (4, 8):
            raise ValueError('Four/eight fresh complete episodes required')
        supplied = list(batches) if len(batches) == 8 else list(batches)*2
        self.validator._validate(supplied)
        data = tensor_batch(batches, self.device)
        with torch.no_grad():
            error = float(((actor.log_prob(data['states'], data['raw']) - data['old_logp']) * data['mask']).abs().max())
        if error > 1e-7:
            raise AssertionError(('The new batch is not from the current policy', error))
        return data, error

    def _proposal(self, actor, optimizer, gradient, states):
        """One Adam ascent proposal, retrying only its step length, not data."""
        if not torch.isfinite(gradient).all() or not torch.isfinite(gradient.norm()):
            raise FloatingPointError('Finite candidate gradient required')
        model_state = copy.deepcopy(actor.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        base_rates = [float(g['lr']) for g in optimizer.param_groups]
        with torch.no_grad():
            old_mean, old_sigma = gaussian_parameters(actor, states)
            old_mean, old_sigma = old_mean.clone(), old_sigma.clone()
        accepted, attempts, actual_kl, proposed_kl = False, 0, 0., None
        norm_before = float(gradient.norm())
        for retry in range(MAX_BACKTRACKS + 1):
            actor.load_state_dict(model_state)
            optimizer.load_state_dict(copy.deepcopy(optimizer_state))
            for group, rate in zip(optimizer.param_groups, base_rates):
                group['lr'] = rate * (.5 ** retry)
            optimizer.zero_grad(set_to_none=True)
            offset = 0
            for parameter in actor.parameters():
                count = parameter.numel()
                parameter.grad = -gradient[offset:offset+count].reshape_as(parameter).clone()
                offset += count
            torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
            optimizer.step()
            attempts += 1
            try:
                with torch.no_grad():
                    new_mean, new_sigma = gaussian_parameters(actor, states)
                    proposed_kl = float(gaussian_kl(old_mean, old_sigma, new_mean, new_sigma).mean())
            except FloatingPointError:
                # A failed proposal must not leave corrupt parameters or Adam
                # moments behind.  The next retry starts from the saved pair.
                actor.load_state_dict(model_state)
                optimizer.load_state_dict(copy.deepcopy(optimizer_state))
                proposed_kl = math.inf
            if math.isfinite(proposed_kl) and proposed_kl <= KL_LIMIT:
                accepted, actual_kl = True, proposed_kl
                break
        if not accepted:
            actor.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
        for group, rate in zip(optimizer.param_groups, base_rates):
            group['lr'] = rate
        return dict(actor_loss=_nullable(-gradient.square().sum()), value_loss=0.,
            macro_kl=actual_kl, optimizer_steps=attempts,
            actor_optimizer_steps=attempts, actor_committed_optimizer_steps=int(accepted),
            pg_gradient_norm=norm_before, pg_update_accepted=accepted,
            pg_backtracks=attempts-1, pg_last_proposed_kl=_nullable(proposed_kl),
            pg_actor_step_scale=(.5**(attempts-1)) if accepted else 0.)

    def update(self, actor, critic, ao, vo, batches, rng):
        data, likelihood_error = self._validate_current(actor, batches)
        if self.episodes < 32:
            if len(batches) != 8:
                raise ValueError('Native warmup population must be eight')
            result = self.native.update(actor, critic, ao, vo, batches, rng)
            self.episodes += len(batches)
            self.updates += 1
            result.update(warmup=True, svr_phase='warmup',
                variance_reduced=self.variance_reduced, reference_episodes_read=0,
                replay_episodes_read=0, trajectory_gradient_evaluations=0,
                candidate_policy_events=0, episodes_seen=self.episodes, updates=self.updates)
            return result
        self._ensure_reference(actor)
        if self.phase == 'inner' and len(batches) != INNER_BATCH_EPISODES:
            raise ValueError('Inner updates require exactly four current episodes')
        phase = self.phase
        common = dict(warmup=False, svr_phase=phase, variance_reduced=self.variance_reduced,
            likelihood_error=likelihood_error, replay_episodes_read=0, buffer_episodes=0,
            candidate_policy_events=0, reference_episodes_read=0,
            trajectory_gradient_evaluations=0, pg_gradient_norm=0., pg_update_accepted=False,
            actor_loss=0., value_loss=0., macro_kl=0., optimizer_steps=0, actor_optimizer_steps=0,
            actor_committed_optimizer_steps=0, pg_backtracks=0, pg_actor_step_scale=0.,
            pg_last_proposed_kl=None, pg_numeric_rejection=False,
            correction_diagnostic_numeric_failure=False,
            raw_gradient_variance_trace=None, difference_variance_trace=None,
            difference_to_raw_variance_ratio=None, correction_norm=None,
            trajectory_weight_ess_fraction=None, trajectory_weight_max=None,
            trajectory_logweight_max=None, trajectory_logweight_min=None,
            reference_gradient_norm=None, reference_sample_variance_trace=self.reference_variance,
            reference_mean_score=self.reference_mean_score)
        if phase == 'reference':
            if len(self.reference_batches) + len(batches) > REFERENCE_EPISODES:
                raise ValueError('Reference population exceeds registered size')
            if self.reference_actor is None:
                raise AssertionError('Reference actor must be fixed before collection')
            with torch.no_grad():
                error = float(((self.reference_actor.log_prob(data['states'], data['raw'])-data['old_logp'])*data['mask']).abs().max())
            if error > 1e-7:
                raise AssertionError(('Reference collection changed its policy', error))
            self.reference_batches.extend(copy.deepcopy(batches))
            if len(self.reference_batches) == REFERENCE_EPISODES:
                gradients = episode_gradients(self.reference_actor, self.reference_batches, self.device)
                self.reference_gradient = gradients.mean(0).detach()
                self.reference_variance = variance_trace(gradients)
                self.reference_mean_score = float(np.mean([b['returns'][-1] for b in self.reference_batches]))
                merged = tensor_batch(self.reference_batches, self.device)
                result = self._proposal(actor, ao, self.reference_gradient, merged['states'])
                common.update(result, trajectory_gradient_evaluations=REFERENCE_EPISODES,
                    reference_episodes_read=REFERENCE_EPISODES,
                    raw_gradient_variance_trace=self.reference_variance,
                    reference_gradient_norm=_nullable(self.reference_gradient.norm()),
                    reference_sample_variance_trace=self.reference_variance,
                    reference_mean_score=self.reference_mean_score)
                self.reference_batches = []
                self.phase, self.inner_updates = 'inner', 0
                self.cycles += 1
        else:
            current = episode_gradients(actor, batches, self.device)
            previous = episode_gradients(self.reference_actor, batches, self.device)
            logs = trajectory_log_weights(self.reference_actor, actor, batches, self.device)
            weights = torch.exp(logs)
            numeric_ok = bool(torch.isfinite(weights).all() and (weights > 0).all())
            raw_variance = variance_trace(current)
            common.update(trajectory_gradient_evaluations=2*len(batches),
                trajectory_logweight_max=_nullable(logs.max()),
                trajectory_logweight_min=_nullable(logs.min()),
                raw_gradient_variance_trace=raw_variance,
                reference_gradient_norm=_nullable(self.reference_gradient.norm()))
            if numeric_ok:
                difference = current - weights[:, None]*previous
                correction = difference.mean(0)
                gradient = self.reference_gradient + correction if self.variance_reduced else current.mean(0)
                difference_variance = variance_trace(difference)
                common.update(correction_norm=_nullable(correction.norm()),
                    difference_variance_trace=difference_variance,
                    difference_to_raw_variance_ratio=(_nullable(difference_variance/raw_variance) if difference_variance is not None and raw_variance and raw_variance > 0 else None),
                    trajectory_weight_ess_fraction=float(torch.exp(2*torch.logsumexp(logs,0)-math.log(len(weights))-torch.logsumexp(2*logs,0))),
                    trajectory_weight_max=_nullable(weights.max()))
                diagnostic_ok = (difference_variance is not None and raw_variance is not None
                                 and common['correction_norm'] is not None)
                common['correction_diagnostic_numeric_failure'] = not diagnostic_ok
                if (not torch.isfinite(gradient).all() or not torch.isfinite(gradient.norm())
                        or (self.variance_reduced and not diagnostic_ok)):
                    common['pg_numeric_rejection'] = True
                else:
                    common.update(self._proposal(actor, ao, gradient, data['states']))
            else:
                common['correction_diagnostic_numeric_failure'] = True
                if self.variance_reduced:
                    common['pg_numeric_rejection'] = True
                else:
                    # P0 is ordinary current-policy PG.  A failed *unused*
                    # history diagnostic must not suppress its valid update.
                    gradient = current.mean(0)
                    if torch.isfinite(gradient).all() and torch.isfinite(gradient.norm()):
                        common.update(self._proposal(actor, ao, gradient, data['states']))
                    else:
                        common['pg_numeric_rejection'] = True
            self.inner_updates += 1
            if self.inner_updates == INNER_UPDATES:
                self.phase, self.inner_updates = 'reference', 0
                self.reference_actor = None
                self.reference_gradient = None
                self.reference_variance = self.reference_mean_score = None
        self.episodes += len(batches)
        self.updates += 1
        common.update(episodes_seen=self.episodes, updates=self.updates,
            svr_cycles=self.cycles, next_svr_phase=self.phase,
            partial_reference_episodes=len(self.reference_batches),
            inner_updates_since_reference=self.inner_updates)
        return common

    def state_dict(self):
        reference_state = self.pending_reference_state
        if self.reference_actor is not None:
            reference_state = self.reference_actor.state_dict()
        return dict(version=VERSION, spec=_spec_dict(self.spec), event_dim=self.event_dim,
            seed=self.seed, variance_reduced=self.variance_reduced, episodes=self.episodes,
            updates=self.updates, cycles=self.cycles, phase=self.phase,
            inner_updates=self.inner_updates, reference_batches=cpu_copy(self.reference_batches),
            reference_gradient=cpu_copy(self.reference_gradient),
            reference_actor=cpu_copy(reference_state), reference_variance=self.reference_variance,
            reference_mean_score=self.reference_mean_score)

    def load_state_dict(self, state):
        if (state['version'] != VERSION or not _same_json(state['spec'], _spec_dict(self.spec))
                or int(state['event_dim']) != self.event_dim or int(state['seed']) != self.seed
                or bool(state['variance_reduced']) != self.variance_reduced):
            raise ValueError('Snapshot estimator checkpoint is incompatible')
        self.episodes, self.updates, self.cycles = map(int, (state['episodes'], state['updates'], state['cycles']))
        self.phase, self.inner_updates = state['phase'], int(state['inner_updates'])
        if self.phase not in ('reference', 'inner'):
            raise ValueError('Invalid estimator phase')
        self.reference_batches = copy.deepcopy(state['reference_batches'])
        gradient = state['reference_gradient']
        self.reference_gradient = None if gradient is None else torch.as_tensor(gradient, dtype=torch.float64, device=self.device).clone()
        self.reference_actor = None
        self.pending_reference_state = copy.deepcopy(state['reference_actor'])
        self.reference_variance = state['reference_variance']
        self.reference_mean_score = state['reference_mean_score']
