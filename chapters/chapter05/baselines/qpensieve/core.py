"""Q-Pensieve highest historical task adapter plus optional initializer repair."""
import math
import numpy as np
import torch
from .memory import Agent as NativeQP
from .primitives import as_tensor,conservative_vector,finite,freeze,polyak

VERSION='ch5-candidate-review-capql-qp-v2'

def _positive(config, key, default):
    value = float(config.get(key, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{key} must be finite and positive')
    return value


def _rate(optimizer, value):
    if optimizer is not None:
        for group in optimizer.param_groups:
            group['lr'] = value


def _scaled_batch(batch, scale):
    if scale == 1.:
        return batch
    result = dict(batch)
    result['rewards'] = np.asarray(batch['rewards'], dtype=np.float32) * scale
    return result


class QPAgent(NativeQP):
    """Q-Pensieve with continuous preferences and labeled optional controls.

    Historical max, twin vector Q and Q-snapshot memory are always retained.
    Full-vector supervision and temperature-comparable memory are explicit
    research adaptations rather than claims of unmodified author defaults.
    """

    def __init__(self, obs_dim, action_dim, seed=0, device='cpu', config=None):
        config = dict(config or {})
        self.reward_scale = _positive(config, 'reward_scale', 1.)
        self.preference_sampling = config.get('preference_sampling', 'author_positive_uniform')
        if self.preference_sampling not in ('author_positive_uniform', 'legacy_four_point'):
            raise ValueError('Unknown QP preference_sampling')
        self.critic_objective_loss = config.get('critic_objective_loss', 'native_preference_weighted')
        if self.critic_objective_loss not in ('native_preference_weighted', 'full_vector_mean'):
            raise ValueError('Unknown QP critic_objective_loss')
        self.actor_lr = _positive(config, 'actor_lr', config.get('lr', 3e-4))
        self.critic_lr = _positive(config, 'critic_lr', config.get('lr', 3e-4))
        self.temperature_lr = _positive(config, 'temperature_lr', config.get('lr', 3e-4))
        self.entropy_per_dimension = float(config.get('target_entropy_per_dimension', -1.))
        if not math.isfinite(self.entropy_per_dimension):
            raise ValueError('Nonfinite target entropy')
        self.history_alpha_ratio_limit = config.get('qp_history_alpha_ratio_limit')
        if self.history_alpha_ratio_limit is not None:
            self.history_alpha_ratio_limit = _positive(config, 'qp_history_alpha_ratio_limit', 2.)
            if self.history_alpha_ratio_limit < 1.:
                raise ValueError('Historical alpha ratio limit must be >=1')
        self.gradient_clip_norm = config.get('gradient_clip_norm')
        if self.gradient_clip_norm is not None:
            self.gradient_clip_norm = _positive(config, 'gradient_clip_norm', 10.)
        self.snapshot_temperatures = []
        self.history_eligible_count = 0
        super().__init__(obs_dim, action_dim, seed, device, config)
        if self.snapshot_capacity < 1:
            raise ValueError('Complete adapted Q-Pensieve retains at least one historical Q slot')
        self.target_entropy = self.entropy_per_dimension*self.action_dim
        _rate(self.actor_optimizer, self.actor_lr)
        _rate(self.critic_optimizer, self.critic_lr)
        _rate(self.alpha_optimizer, self.temperature_lr)

    @staticmethod
    def _positive_weights(rng, count):
        # Matches author's independent Uniform(0,1) coordinates / L1 sampler.
        # The tiny guard excludes an exact floating-point zero without adding
        # a practical preference cutoff or collapsing the support to a grid.
        values = np.maximum(rng.random((int(count), 2)), np.finfo(np.float32).tiny)
        return (values/values.sum(1, keepdims=True)).astype(np.float32)

    def sample_weights(self, count, rng):
        # The earlier collector already used positive continuous weights;
        # random_preference below also restores that support for Q updates.
        return self._positive_weights(rng, count)

    def random_preference(self, exclude=None):
        if self.preference_sampling == 'legacy_four_point':
            return super().random_preference(exclude)
        for _ in range(100):
            result = self._positive_weights(self.rng, 1)[0]
            if exclude is None or not np.isclose(result, np.asarray(exclude)).all():
                return result
        raise RuntimeError('Could not draw a distinct continuous preference')

    def policy_value(self, obs, preferences, actions):
        original = self.snapshots
        if self.history_alpha_ratio_limit is None:
            eligible = original
        else:
            current_alpha = float(self.alpha)
            eligible = [critic for critic, old_alpha in zip(original, self.snapshot_temperatures)
                        if max(current_alpha/old_alpha, old_alpha/current_alpha) <= self.history_alpha_ratio_limit]
        self.history_eligible_count = len(eligible)
        self.snapshots = eligible
        try:
            return super().policy_value(obs, preferences, actions)
        finally:
            self.snapshots = original

    def after_update(self):
        stats = super().after_update()
        if stats['q_snapshot_created']:
            slot = (self.snapshot_position-1) % self.snapshot_capacity
            temperature = float(self.alpha)
            if len(self.snapshot_temperatures) < len(self.snapshots):
                self.snapshot_temperatures.append(temperature)
            else:
                self.snapshot_temperatures[slot] = temperature
        stats.update(q_history_eligible_count=self.history_eligible_count,
                     q_history_stored_count=len(self.snapshots),
                     q_history_alpha_ratio_limit=self.history_alpha_ratio_limit)
        return stats

    def _clip(self, module):
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.gradient_clip_norm, error_if_nonfinite=True)

    def update(self, batch):
        # This explicit copy of the pinned SACBase order only changes the
        # registered reward units, supervision weights or gradient cap.
        # The author's Q-Pensieve historical/preference max remains delegated
        # to the original, independently exercised policy_value implementation.
        batch = self.batch(_scaled_batch(batch, self.reward_scale))
        states, next_states, actions = batch['obs'], batch['next_obs'], batch['actions']
        preference = as_tensor(self.random_preference(), self.device)[None].expand(len(states), -1)
        with torch.no_grad():
            next_actions, next_logp = self.actor.sample(torch.cat([next_states, preference], -1), self.learn_generator)
            tq1, tq2 = self.critic_values(next_states, next_actions, preference, self.critic_target)
            target = batch['rewards'] + (1-batch['dones'])*self.gamma*(
                conservative_vector(tq1, tq2, preference)-self.alpha*next_logp)
        q1, q2 = self.critic_values(states, actions, preference)
        loss_weights = preference if self.critic_objective_loss == 'native_preference_weighted' else torch.full_like(preference, .5)
        q1_loss = ((q1-target).square()*loss_weights).sum(-1).mean()
        q2_loss = ((q2-target).square()*loss_weights).sum(-1).mean()
        critic_loss = q1_loss+q2_loss
        finite(critic_loss, 'critic loss')
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        for parameter in self.critic.parameters():
            if parameter.grad is not None:
                finite(parameter.grad, 'critic gradient')
        self._clip(self.critic)
        self.critic_optimizer.step()
        freeze(self.critic)
        try:
            sampled, logp = self.actor.sample(torch.cat([states, preference], -1), self.learn_generator)
            q = self.policy_value(states, preference, sampled)
            actor_loss = (self.alpha*logp.squeeze(-1)-q).mean()
            finite(actor_loss, 'actor loss')
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            for parameter in self.actor.parameters():
                if parameter.grad is not None:
                    finite(parameter.grad, 'actor gradient')
            self._clip(self.actor)
            self.actor_optimizer.step()
        finally:
            freeze(self.critic, False)
        alpha_loss = self.update_alpha(logp)
        polyak(self.critic_target, self.critic, self.tau)
        self.updates += 1
        extra = self.after_update()
        return dict(critic_loss=float(critic_loss.detach()), actor_loss=float(actor_loss.detach()),
                    alpha=float(self.alpha), alpha_loss=alpha_loss, updates=self.updates,
                    mechanism=self.name, reward_scale=self.reward_scale,
                    effective_alpha_in_task_units=float(self.alpha)/self.reward_scale,
                    critic_C_mse=float(((q1-target).square()[:, 0]+(q2-target).square()[:, 0]).mean().detach()),
                    critic_J_mse=float(((q1-target).square()[:, 1]+(q2-target).square()[:, 1]).mean().detach()),
                    sampled_preference_C=float(preference[0, 0]),
                    critic_objective_loss=self.critic_objective_loss,
                    five_round_adapter=VERSION, **extra)

    def metadata(self):
        result = super().metadata()
        result['legacy_preference_grid'] = result.get('preference_grid')
        if self.preference_sampling == 'author_positive_uniform':
            result['preference_grid'] = None
            result['preference_support'] = 'positive continuous simplex; independent uniform coordinates normalized by L1'
        result['task_adaptation'] = dict(
            version=VERSION, label='Q-Pensieve: disclosed continuous-preference task adaptation',
            preference_sampling=self.preference_sampling,
            critic_objective_loss=self.critic_objective_loss,
            reward_scale=self.reward_scale, environment_reward_changed=False,
            actor_lr=self.actor_lr, critic_lr=self.critic_lr, temperature_lr=self.temperature_lr,
            target_entropy_per_dimension=self.entropy_per_dimension,
            history_alpha_ratio_limit=self.history_alpha_ratio_limit,
            history_alpha_filter='retain stored Q, compare only snapshots with multiplicatively comparable alpha; no claimed exact Q recalibration',
            gradient_clip_norm=self.gradient_clip_norm,
            native_core='max over historical/current twin vector Q and alternative preference conditions',
            extra_training_feedback=False, raw_C_J_reported=True,
            numerical_equivalence_claimed=False,
        )
        return result

    def state_dict(self):
        result = super().state_dict()
        result['five_round_qp'] = dict(snapshot_temperatures=list(self.snapshot_temperatures),
                                      history_eligible_count=self.history_eligible_count)
        return result

    def load_state_dict(self, payload):
        super().load_state_dict(payload)
        extra = payload.get('five_round_qp')
        if extra is None:
            if self.snapshots and self.history_alpha_ratio_limit is not None:
                raise ValueError('Historical alpha metadata required for temperature-comparable Q memory')
            self.snapshot_temperatures = [float(self.alpha)]*len(self.snapshots)
            self.history_eligible_count = len(self.snapshots)
        else:
            self.snapshot_temperatures = [float(x) for x in extra['snapshot_temperatures']]
            if len(self.snapshot_temperatures) != len(self.snapshots) or any(not math.isfinite(x) or x <= 0 for x in self.snapshot_temperatures):
                raise ValueError('Invalid historical snapshot temperature ledger')
            self.history_eligible_count = int(extra['history_eligible_count'])

