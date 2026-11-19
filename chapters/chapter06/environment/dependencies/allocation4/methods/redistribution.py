"""Causal return redistribution and a matched native-reward GAE comparator.

This is a RUDDER-inspired adaptation, not a reproduction of its Atari system.
It retains the chapter-three allocation policy and learns a prefix return
predictor solely from completed, previously collected native trajectories.
The two policy signals use gamma=1, lambda=.95 GAE, with a fixed .5 mixture
after 32 replay episodes.  Exact return conservation does NOT imply unbiased
GAE or improved credit.  With full MC and a matching value translation the
redistribution would reduce to the original advantage (tested separately).

No environment, future observation, physical reward formula or oracle is read.
An extra value head receives the public state and the algorithm's known
pre-action prefix prediction. This removes omission of that explicit history
term without claiming it is a sufficient statistic of the entire GRU history.
"""
from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch
from torch import nn

from allocation23.learning import PPO


CONFIG = dict(gamma=1., gae_lambda=.95, mixture=.5, warmup_episodes=32,
              replay_capacity=128, predictor_steps=4, predictor_batch=16,
              predictor_lr=1e-3, prefix_loss_weight=.1, hidden=64,
              shaped_value_version='independent-public-state-plus-prefix-v1')


def terminal_rewards(returns: torch.Tensor) -> torch.Tensor:
    """Recover native rewards from the complete-episode MC labels, B x H."""
    if returns.ndim != 2 or returns.shape[1] == 0:
        raise ValueError('Complete B x H episode returns are required')
    if not torch.isfinite(returns).all() or not torch.all(returns == returns[:, -1:]):
        raise ValueError('Each episode must repeat its finite native terminal return')
    result = torch.zeros_like(returns)
    result[:, -1] = returns[:, -1]
    return result


def redistribute(prefix: torch.Tensor, native_return: torch.Tensor):
    """Causal prediction differences, with the actual residual at the last step.

    prefix[:, t] may use (s_0,a_0,...,s_t,a_t), never later transitions.
    The final adjustment is issued after that final action.  Negative rewards
    and corrections are retained.  Shape is B x H; returns have shape B.
    """
    if prefix.ndim != 2 or native_return.shape != prefix.shape[:1]:
        raise ValueError('Expected prefix B x H and native return B')
    if not torch.isfinite(prefix).all() or not torch.isfinite(native_return).all():
        raise ValueError('Finite predictions and native labels required')
    before = torch.cat((torch.zeros_like(prefix[:, :1]), prefix[:, :-1]), dim=1)
    rewards = prefix - before
    correction = native_return - prefix[:, -1]
    rewards = rewards.clone()
    rewards[:, -1] += correction
    return rewards, before, correction


def gae_targets(rewards: torch.Tensor, values: torch.Tensor, lam=.95):
    """Finite-horizon gamma-one GAE; there is no terminal bootstrap."""
    if rewards.ndim != 2 or rewards.shape != values.shape:
        raise ValueError('Rewards and values must have identical B x H shape')
    if not 0 <= lam <= 1:
        raise ValueError('lambda must be in [0,1]')
    if not torch.isfinite(rewards).all() or not torch.isfinite(values).all():
        raise ValueError('Finite GAE inputs required')
    next_values = torch.cat((values[:, 1:], torch.zeros_like(values[:, :1])), 1)
    delta = rewards + next_values - values
    advantage = torch.empty_like(delta)
    carry = torch.zeros_like(delta[:, 0])
    for t in range(delta.shape[1] - 1, -1, -1):
        carry = delta[:, t] + lam * carry
        advantage[:, t] = carry
    return advantage, advantage + values


class PrefixReturn(nn.Module):
    """Unidirectional sequence model; outputs a return estimate at each prefix."""
    def __init__(self, local_dim, global_dim, event_dim, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(52103 + int(seed))
            self.objects = nn.Sequential(nn.Linear(local_dim, 32), nn.Tanh(),
                                         nn.Linear(32, 32), nn.Tanh())
            self.recurrent = nn.GRU(32 + global_dim + event_dim + 1,
                                    CONFIG['hidden'], batch_first=True)
            self.head = nn.Linear(CONFIG['hidden'], 1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        self.double()

    def forward(self, states, raw, mask):
        # Local/global are the same encoded observations as those of the actor.
        local = self.objects(states['local']).mean(-2)
        features = torch.cat((local, states['global'], torch.tanh(raw), mask), -1)
        sequence, _ = self.recurrent(features)
        return self.head(sequence).squeeze(-1)


class PrefixValue(nn.Module):
    """Independent state value plus a learned scalar-prefix correction.

    Neither the native value weights nor a coefficient of -1 is shared or
    enforced. The correction starts at zero, so its initial predictions match
    the copied native critic exactly. The prefix is an algorithm-side value
    available before the current action, not new environment information.
    """
    def __init__(self, native_critic, seed):
        super().__init__()
        self.base = copy.deepcopy(native_critic)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(61903 + int(seed))
            self.prefix_correction = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
            nn.init.zeros_(self.prefix_correction[-1].weight)
            nn.init.zeros_(self.prefix_correction[-1].bias)
        self.double()

    def forward(self, state, raw=None, *, prefix_before):
        reference = next(self.base.parameters())
        prefix = torch.as_tensor(prefix_before, dtype=reference.dtype, device=reference.device)
        value = self.base(state, raw)
        if prefix.shape != value.shape or not torch.isfinite(prefix).all():
            raise ValueError('One finite pre-action prefix prediction per value required')
        return value + self.prefix_correction(prefix)


class RedistributionMethod:
    """Owns reward predictor/replay and (for RUDDER) a second value estimator.

    The supplied actor and native critic remain owned by the common trainer.
    ``state_dict`` includes replay, optimizer states and the private fit RNG,
    so checkpoint/resume does not silently start a different algorithm.
    """
    def __init__(self, spec, event_dim, seed, device, mode='rudder'):
        if mode not in ('rudder', 'gae'):
            raise ValueError('Implemented modes are rudder and gae; DAE is separate')
        self.mode, self.device = mode, torch.device(device)
        self.seed = int(seed)
        self.horizon, self.event_dim = int(spec.horizon), int(event_dim)
        self.config = dict(CONFIG)
        self.replay = deque(maxlen=self.config['replay_capacity'])
        self.fit_rng = np.random.default_rng(92117 + int(seed))
        self.updates = self.seen_episodes = 0
        self.predictor = (PrefixReturn(spec.local_dim, spec.global_dim, event_dim, seed)
                          .to(self.device) if mode == 'rudder' else None)
        self.predictor_optimizer = (torch.optim.Adam(self.predictor.parameters(),
                                    lr=self.config['predictor_lr']) if self.predictor else None)
        self.shaped_critic = self.shaped_optimizer = None
        self._pending_shaped = None
        self.last_signals = None

    def _ensure_shaped(self, critic):
        if self.mode != 'rudder' or self.shaped_critic is not None:
            return
        self.shaped_critic = PrefixValue(critic, self.seed).to(self.device)
        self.shaped_optimizer = torch.optim.Adam(self.shaped_critic.parameters(), lr=PPO['value_lr'])
        if self._pending_shaped is not None:
            self.shaped_critic.load_state_dict(self._pending_shaped['model'])
            self.shaped_optimizer.load_state_dict(self._pending_shaped['optimizer'])
            self._pending_shaped = None

    def _stack(self, batches):
        if not batches:
            raise ValueError('At least one complete episode is required')
        for batch in batches:
            if np.shape(batch['returns']) != (self.horizon,):
                raise ValueError('A batch item must contain one complete native episode')
            if np.shape(batch['raw']) != (self.horizon, self.event_dim):
                raise ValueError('A complete latent action event is required')
        tensor = lambda arrays: torch.as_tensor(np.stack(arrays), dtype=torch.float64,
                                                device=self.device)
        states = {key: tensor([b['states'][key] for b in batches])
                  for key in ('local', 'global', 'caps', 'budget', 'remaining')}
        data = {key: tensor([b[key] for b in batches])
                for key in ('raw', 'old_logp', 'mask', 'values', 'returns')}
        for key in ('old_logp', 'mask', 'values'):
            if data[key].shape != (len(batches), self.horizon, 1):
                raise ValueError(f'Expected one complete policy event for {key}')
        if not torch.all((data['mask'] == 0) | (data['mask'] == 1)):
            raise ValueError('Policy mask must be binary')
        return states, data

    def _fit_completed(self, batches):
        for batch in batches:
            # Do not retain audit-only fields or mutate caller-owned arrays.
            self.replay.append(dict(states={k: np.array(batch['states'][k], copy=True)
                                             for k in ('local', 'global')},
                                    raw=np.array(batch['raw'], copy=True),
                                    mask=np.array(batch['mask'], copy=True),
                                    returns=np.array(batch['returns'], copy=True)))
        self.seen_episodes += len(batches)
        losses, finals, prefixes = [], [], []
        self.predictor.train()
        for _ in range(self.config['predictor_steps']):
            count = min(self.config['predictor_batch'], len(self.replay))
            selected = self.fit_rng.choice(len(self.replay), size=count, replace=False)
            records = [self.replay[int(i)] for i in selected]
            to = lambda key: torch.as_tensor(np.stack([b[key] for b in records]),
                                             dtype=torch.float64, device=self.device)
            states = {key: torch.as_tensor(np.stack([b['states'][key] for b in records]),
                                          dtype=torch.float64, device=self.device)
                      for key in ('local', 'global')}
            labels = to('returns')[:, -1]
            prediction = self.predictor(states, to('raw'), to('mask'))
            final_loss = (prediction[:, -1] - labels).square().mean()
            prefix_loss = (prediction - labels[:, None]).square().mean()
            loss = final_loss + self.config['prefix_loss_weight'] * prefix_loss
            self.predictor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite return predictor gradient')
            self.predictor_optimizer.step()
            losses.append(float(loss.detach()))
            finals.append(float(final_loss.detach()))
            prefixes.append(float(prefix_loss.detach()))
        self.predictor.eval()
        return dict(predictor_loss=float(np.mean(losses)),
                    predictor_train_final_mse=float(np.mean(finals)),
                    predictor_train_prefix_mse=float(np.mean(prefixes)),
                    predictor_optimizer_steps=len(losses), replay_episodes=len(self.replay))

    def update(self, actor, critic, actor_optimizer, critic_optimizer, batches, rng):
        states_seq, data_seq = self._stack(batches)
        batch_count = len(batches)
        states = {key: x.flatten(0, 1) for key, x in states_seq.items()}
        data = {key: x.flatten(0, 1) for key, x in data_seq.items()}
        native_rewards = terminal_rewards(data_seq['returns'])
        mask, raw, old = data['mask'], data['raw'], data['old_logp']
        active = mask > 0
        if not torch.any(active):
            return dict(likelihood_error=0., optimizer_steps=0, actor_loss=0., value_loss=0.,
                        macro_kl=0., clip_fraction=0., actor_grad_norm=0., value_grad_norm=0.,
                        early_stop=False, active_events=0., advantage_std=0., mixture=0.,
                        predictor_optimizer_steps=0, shaped_optimizer_steps=0)
        self._ensure_shaped(critic)
        with torch.no_grad():
            likelihood_error = float(((actor.log_prob(states, raw) - old) * mask).abs().max())
            if likelihood_error > 1e-7:
                raise AssertionError(('stored likelihood mismatch', likelihood_error))
            native_adv, native_targets = gae_targets(native_rewards, data_seq['values'].squeeze(-1),
                                                     self.config['gae_lambda'])
            shaped_rewards, prefix_before = native_rewards.clone(), torch.zeros_like(native_rewards)
            correction = data_seq['returns'][:, -1].clone()
            shaped_adv, shaped_targets = native_adv, native_targets
            mixture = 0.
            prediction_mse = 0.
            if self.mode == 'rudder':
                self.predictor.eval()
                prediction = self.predictor(states_seq, data_seq['raw'], data_seq['mask'])
                shaped_rewards, prefix_before, correction = redistribute(
                    prediction, data_seq['returns'][:, -1])
                prediction_mse = float((prediction[:, -1] - data_seq['returns'][:, -1]).square().mean())
                shaped_values = self.shaped_critic(states, prefix_before=prefix_before.reshape(-1, 1)).reshape(batch_count, self.horizon)
                shaped_adv, shaped_targets = gae_targets(shaped_rewards, shaped_values,
                                                         self.config['gae_lambda'])
                if self.seen_episodes >= self.config['warmup_episodes']:
                    mixture = self.config['mixture']
            advantage = ((1. - mixture) * native_adv + mixture * shaped_adv).reshape(-1, 1)
            advantage_mean = advantage[active].mean()
            advantage_std = advantage[active].std(unbiased=False)
            normalized = (advantage - advantage_mean) / (advantage_std + 1e-8)
            native_targets = native_targets.reshape(-1, 1)
            shaped_targets = shaped_targets.reshape(-1, 1)
        # Small CPU arrays provide a no-environment audit of the actual targets.
        self.last_signals = {key: value.detach().cpu().numpy().copy() for key, value in dict(
            native_rewards=native_rewards, redistributed_rewards=shaped_rewards,
            prefix_before=prefix_before, terminal_correction=correction,
            native_advantage=native_adv, redistributed_advantage=shaped_adv,
            native_targets=native_targets, redistributed_targets=shaped_targets,
            advantage=advantage, normalized_advantage=normalized).items()}
        self.last_signals['mixture'] = mixture
        stats, stop = [], False
        for _ in range(PPO['epochs']):
            order = rng.permutation(len(raw))
            for start in range(0, len(raw), PPO['minibatch']):
                ix = torch.as_tensor(order[start:start + PPO['minibatch']], dtype=torch.long,
                                     device=self.device)
                s, m = {key: value[ix] for key, value in states.items()}, mask[ix]
                lp = actor.log_prob(s, raw[ix])
                if not torch.isfinite(lp).all():
                    raise FloatingPointError('Nonfinite recomputed likelihood')
                logratio = lp - old[ix]
                ratio = logratio.clamp(-40, 40).exp()
                objective = torch.minimum(ratio * normalized[ix],
                    ratio.clamp(1. - PPO['clip'], 1. + PPO['clip']) * normalized[ix])
                actor_loss = -(objective * m).sum(-1).mean()
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_norm = nn.utils.clip_grad_norm_(actor.parameters(), PPO['grad_clip'])
                if not torch.isfinite(actor_norm):
                    raise FloatingPointError('Nonfinite actor gradient')
                actor_optimizer.step()
                value_loss = ((critic(s, raw[ix]) - native_targets[ix]).square() * m).sum() / m.sum().clamp_min(1)
                critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                value_norm = nn.utils.clip_grad_norm_(critic.parameters(), PPO['grad_clip'])
                if not torch.isfinite(value_norm):
                    raise FloatingPointError('Nonfinite native value gradient')
                critic_optimizer.step()
                shaped_loss = raw.new_zeros(())
                if self.mode == 'rudder':
                    shaped_prediction = self.shaped_critic(s, raw[ix],
                        prefix_before=prefix_before.reshape(-1, 1)[ix])
                    shaped_loss = ((shaped_prediction - shaped_targets[ix]).square() * m).sum() / m.sum().clamp_min(1)
                    self.shaped_optimizer.zero_grad(set_to_none=True)
                    shaped_loss.backward()
                    shaped_norm = nn.utils.clip_grad_norm_(self.shaped_critic.parameters(), PPO['grad_clip'])
                    if not torch.isfinite(shaped_norm):
                        raise FloatingPointError('Nonfinite redistributed value gradient')
                    self.shaped_optimizer.step()
                with torch.no_grad():
                    kl = ((ratio - 1. - logratio) * m).sum(-1).mean()
                    clip = (((ratio - 1.).abs() > PPO['clip']) * m).sum() / m.sum().clamp_min(1)
                stats.append([float(x.detach()) for x in (actor_loss, value_loss, kl, clip,
                                                          actor_norm, value_norm, shaped_loss)])
                if float(kl) > PPO['macro_kl_stop']:
                    stop = True
                    break
            if stop:
                break
        table = np.asarray(stats)
        report = dict(likelihood_error=likelihood_error, optimizer_steps=len(stats),
            actor_loss=float(table[:, 0].mean()), value_loss=float(table[:, 1].mean()),
            macro_kl=float(table[:, 2].mean()), clip_fraction=float(table[:, 3].mean()),
            actor_grad_norm=float(table[:, 4].mean()), value_grad_norm=float(table[:, 5].mean()),
            shaped_value_loss=float(table[:, 6].mean()), early_stop=stop,
            active_events=float(mask.sum()), advantage_std=float(advantage_std), mixture=mixture,
            return_conservation_max_error=float((shaped_rewards.sum(1) - data_seq['returns'][:, -1]).abs().max()),
            terminal_correction_rms=float(correction.square().mean().sqrt()),
            predictor_preupdate_final_mse=prediction_mse,
            redistributed_negative_fraction=float((shaped_rewards < 0).double().mean()),
            native_advantage_rms=float(native_adv.square().mean().sqrt()),
            redistributed_advantage_rms=float(shaped_adv.square().mean().sqrt()),
            shaped_optimizer_steps=len(stats) if self.mode == 'rudder' else 0,
            predictor_optimizer_steps=0, replay_episodes_before=len(self.replay))
        # The present labels affect only future policy updates through predictor
        # training. Current signals above were already frozen and consumed.
        if self.mode == 'rudder':
            report.update(self._fit_completed(batches))
        else:
            self.seen_episodes += len(batches)
        self.updates += 1
        return report

    def state_dict(self):
        return dict(mode=self.mode, config=dict(self.config), updates=self.updates,
                    seen_episodes=self.seen_episodes, fit_rng=copy.deepcopy(self.fit_rng.bit_generator.state),
                    replay=copy.deepcopy(list(self.replay)),
                    predictor=None if self.predictor is None else self.predictor.state_dict(),
                    predictor_optimizer=None if self.predictor_optimizer is None else self.predictor_optimizer.state_dict(),
                    shaped=(self._pending_shaped if self.shaped_critic is None else
                            dict(model=self.shaped_critic.state_dict(), optimizer=self.shaped_optimizer.state_dict())))

    def metadata(self):
        return dict(method='native_gae' if self.mode == 'gae' else 'rudder_inspired_dual_value_gae',
                    original_reproduction=False, config=dict(self.config),
                    predictor_inputs=['public_local', 'public_global', 'recorded_latent_action', 'policy_mask'],
                    supervision='Native terminal return only; no intermediate true reward or oracle',
                    signal_data_boundary='Predictor is fitted after current PPO update; only earlier episodes generate its signal',
                    redistribution='Causal prefix prediction differences plus actual terminal residual',
                    critic_inputs='Native head: public state; independent shaped head: public state plus pre-action algorithm prefix prediction',
                    prefix_value='Independent copied state critic plus zero-output-initialized 1-16-1 tanh prefix correction; no fixed value translation',
                    prefix_sufficiency_caveat='Known prefix subtraction is represented; scalar prefix need not summarize the entire recurrent history',
                    critic_mask='Same active-policy mask as allocation23 native PPO',
                    objective_caveat='Pathwise return conservation; approximate GAE is not an unbiasedness guarantee',
                    adaptation='Fixed .5 dual-advantage mix after 32 episodes; no paper quality gate or expert exploration',
                    additional_compute='Predictor and extra value optimizer steps reported separately')

    def load_state_dict(self, state):
        if state['mode'] != self.mode or state['config'] != self.config:
            raise ValueError('Cannot resume with a different redistribution protocol')
        self.updates, self.seen_episodes = state['updates'], state['seen_episodes']
        self.fit_rng.bit_generator.state = copy.deepcopy(state['fit_rng'])
        self.replay.clear()
        self.replay.extend(copy.deepcopy(state['replay']))
        if self.predictor is not None:
            self.predictor.load_state_dict(state['predictor'])
            self.predictor_optimizer.load_state_dict(state['predictor_optimizer'])
        self._pending_shaped = copy.deepcopy(state['shaped'])
        if self.shaped_critic is not None and self._pending_shaped is not None:
            self.shaped_critic.load_state_dict(self._pending_shaped['model'])
            self.shaped_optimizer.load_state_dict(self._pending_shaped['optimizer'])
            self._pending_shaped = None
