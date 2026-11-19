"""COLA task adapters: historical recipe versus one-step OADM label repair.

Both profiles retain the selected old COLA recipe, OADM/COR/SAC mathematics,
initialization, reward targets and preference law. The fixed profile changes
only the temporal identity of OADM's target when TD replay uses n_step>1.
Observation normalization belongs to the common worker, not this class.
"""
from __future__ import annotations
from contextlib import contextmanager
import math
import numpy as np
import torch
from torch import nn
from .native import Agent as NativeCOLA

VERSION = 'cola-one-step-oadm-target-paired-audit-v1'

def _number(config, key, default, *, allow_zero=False):
    value = float(config.get(key, default))
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f'{key} must be finite and {"nonnegative" if allow_zero else "positive"}')
    return value


def _scale_final_linear(module, coefficient):
    layers = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    if not layers:
        raise TypeError('An explicit final Linear layer is required')
    with torch.no_grad():
        layers[-1].weight.mul_(coefficient)
        if layers[-1].bias is not None:
            layers[-1].bias.mul_(coefficient)


@contextmanager
def _temporary_step(optimizer, replacement):
    owned = 'step' in vars(optimizer)
    prior = vars(optimizer).get('step')
    optimizer.step = replacement
    try:
        yield
    finally:
        if owned:
            optimizer.step = prior
        else:
            del optimizer.step


class Agent(NativeCOLA):
    """Retain OADM/COR with explicit continuous preferences and temperature."""
    manages_observation_normalization = False
    action_mode = 'tanh_latent'

    def __init__(self, obs_dim, action_dim, seed=0, device='cpu', config=None):
        config = dict(config or {})
        self.oadm_one_step_target = config.get('oadm_one_step_target', False)
        if not isinstance(self.oadm_one_step_target, bool):
            raise ValueError('oadm_one_step_target must be an explicit boolean')
        self.n_step = config.get('n_step', 1)
        if isinstance(self.n_step, bool) or int(self.n_step) != self.n_step or self.n_step < 1:
            raise ValueError('n_step must be a positive integer')
        self.n_step = int(self.n_step)
        self.preference_floor = _number(config, 'preference_floor', .02)
        if self.preference_floor >= .5:
            raise ValueError('preference_floor must be strictly below .5')
        self.reward_scale = _number(config, 'reward_scale', 1.)
        self.target_entropy_scale = _number(config, 'target_entropy_scale', 1.)
        self.alpha_min = _number(config, 'alpha_min', 1e-6)
        self.alpha_max = _number(config, 'alpha_max', 1.)
        if self.alpha_max < self.alpha_min:
            raise ValueError('alpha_max must not be below alpha_min')
        self.actor_gradient_clip = _number(config, 'actor_gradient_clip', 0., allow_zero=True)
        self.actor_output_init_scale = _number(config, 'actor_output_init_scale', 1.)
        super().__init__(obs_dim, action_dim, seed, device, config)
        self.cola_batched_finite_checks=bool(config.get('cola_batched_finite_checks',False))
        if self.cola_batched_finite_checks:
            from .cola_exact_runtime import make_batched_finite_update
            self._batched_finite_update=make_batched_finite_update()
        self.target_entropy = -self.action_dim*self.target_entropy_scale
        for group in self.actor_optimizer.param_groups:
            group['lr'] = _number(config, 'actor_lr', self.lr)
        _scale_final_linear(self.actor, self.actor_output_init_scale)
        # Fixed alpha must be exactly the configured coefficient, not a hidden
        # clipped different value. Bounds concern adaptive temperature only.
        if self.entropy_tuning and not self.alpha_min <= float(self.alpha) <= self.alpha_max:
            raise ValueError('Initial adaptive alpha must lie within registered bounds')

    def _positive_preferences(self, count, rng):
        values = rng.random((int(count), 2))
        values /= values.sum(-1, keepdims=True)
        return self.preference_floor + (1-2*self.preference_floor)*values

    def sample_weights(self, count, rng):
        return self._positive_preferences(count, rng).astype(np.float32)

    def random_preference(self, exclude=None):
        for _ in range(64):
            result = self._positive_preferences(1, self.rng)[0]
            if exclude is None or not np.isclose(result, np.asarray(exclude)).all():
                return result
        raise RuntimeError('Could not draw independent continuous positive preference')

    def batch(self, value):
        data = super().batch(value)
        data['rewards'] = data['rewards']*self.reward_scale
        if self.oadm_one_step_target:
            if self.n_step > 1 and 'oadm_next_obs' not in value:
                raise ValueError('Fixed n-step COLA requires actual one-step oadm_next_obs from replay')
            observed = torch.as_tensor(value.get('oadm_next_obs', value['next_obs']),
                                       dtype=torch.float32, device=self.device)
            if observed.shape != data['next_obs'].shape or not torch.isfinite(observed).all():
                raise ValueError('Invalid public one-step OADM observation')
            if self.n_step == 1:
                if not torch.equal(observed, data['next_obs']):
                    raise ValueError('One-step replay and OADM successor must agree exactly')
                # Retain the original forward/dtype/RNG path for n_step=1.
                data['oadm_next_obs'] = data['next_obs']
            else:
                data['oadm_next_obs'] = observed
        return data

    def train_preferences(self, step, total_steps, rng):
        # Same once-per-episode continuous preference draw as the old worker.
        return self.sample_weights(int(self.config.get('n_env', 1)), rng)

    def update(self, batch):
        original_step = self.actor_optimizer.step
        clipping = {}

        def guarded(*args, **kwargs):
            if self.actor_gradient_clip:
                norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_gradient_clip,
                                                      error_if_nonfinite=True)
                clipping['actor_gradient_norm_before_clip'] = float(norm)
            return original_step(*args, **kwargs)

        with _temporary_step(self.actor_optimizer, guarded):
            if self.cola_batched_finite_checks:
                result=self._batched_finite_update(self,batch)
            else:
                result = super().update(batch)
        if self.entropy_tuning:
            with torch.no_grad():
                self.log_alpha.clamp_(math.log(self.alpha_min), math.log(self.alpha_max))
        result.update(clipping)
        result.update(alpha=float(self.alpha), reward_scale=self.reward_scale,
                      target_entropy=self.target_entropy, preference_floor=self.preference_floor)
        return result

    def metadata(self):
        result = super().metadata()
        result['candidate_status'] = 'development_recheck_not_formally_adopted'
        result['oadm_one_step_target'] = self.oadm_one_step_target
        result['n_step'] = self.n_step
        result['oadm_target_contract'] = ('one-step observed successor independent of n-step Bellman target'
            if self.oadm_one_step_target else 'legacy uses the n-step replay successor for OADM')
        result['author_configuration'] = 'Disclosed task-selected radio R4 / v5 R5 recipe; actual values recorded below and in configs.json, not unchanged paper defaults'
        # SACBase's historical grid remains an unused compatibility attribute;
        # reporting it as the actual training law would be misleading.
        result['preference_grid'] = None
        result['five_round_adaptation'] = dict(
            version=VERSION, continuous_positive_preferences=True,
            preference_floor=self.preference_floor,
            preference_law='normalized independent Uniform(0,1) coordinates, then affine interior floor; no discrete endpoints',
            reward_scale=self.reward_scale,
            reward_scale_semantics='common positive multiplier of both training terminal rewards only; evaluation metrics remain unscaled',
            target_entropy_scale=self.target_entropy_scale,
            adaptive_temperature_bounds=[self.alpha_min, self.alpha_max],
            actor_output_init_scale=self.actor_output_init_scale,
            actor_gradient_clip=self.actor_gradient_clip,
            batched_finite_gradient_checks=self.cola_batched_finite_checks,
            core_preserved=['OADM latent temporal consistency', 'COR old-Q preservation with final-layer conflict gate'],
            native_numerical_equivalence_claimed=False,
        )
        return result

