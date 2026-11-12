"""CAPQL highest historical task adapter; pinned author equations are unchanged."""
import math
import numpy as np
import torch
from .native import Agent as NativeCAPQL

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


class CAPQLAgent(NativeCAPQL):
    """Author CAPQL update with explicit reward units and optimizer rates."""

    def __init__(self, obs_dim, action_dim, seed=0, device='cpu', config=None):
        config = dict(config or {})
        self.reward_scale = _positive(config, 'reward_scale', 1.)
        self.actor_lr = _positive(config, 'actor_lr', config.get('lr', 3e-4))
        self.critic_lr = _positive(config, 'critic_lr', config.get('lr', 3e-4))
        super().__init__(obs_dim, action_dim, seed, device, config)
        _rate(self.native.policy_optim, self.actor_lr)
        _rate(self.native.critic_optim, self.critic_lr)

    def update(self, batch):
        stats = super().update(_scaled_batch(batch, self.reward_scale))
        stats.update(reward_scale=self.reward_scale, effective_alpha_in_task_units=float(self.native.alpha)/self.reward_scale,
                     actor_lr=self.actor_lr, critic_lr=self.critic_lr,
                     five_round_adapter=VERSION)
        return stats

    def metadata(self):
        result = super().metadata()
        result['task_adaptation'] = dict(
            version=VERSION, label='CAPQL: disclosed task-scale adaptation',
            reward_scale=self.reward_scale, environment_reward_changed=False,
            fixed_alpha=float(self.native.alpha), effective_alpha_in_task_units=float(self.native.alpha)/self.reward_scale,
            actor_lr=self.actor_lr, critic_lr=self.critic_lr,
            native_update='pinned author vector twin-Q, concave entropy augmentation and policy update',
            raw_C_J_reported=True, extra_training_feedback=False,
            numerical_equivalence_claimed=False,
        )
        return result


