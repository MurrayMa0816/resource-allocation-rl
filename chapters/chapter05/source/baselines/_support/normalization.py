"""Train-warmup-only observation calibration for off-policy task adapters.

The environment and replay keep raw public observations. Native policies and
critics receive one saved, frozen map after calibration. Development/test
inputs never fit that map. This module does not import the separate mutable
validation package and never launches training.
"""
from __future__ import annotations

import math
import numpy as np
import torch

VERSION = 'ch5-candidate-frozen-normalizer-v2-one-step-aux'
STATE_KEY = 'five_round_observation_normalizer'


class ObservationMoments:
    """Float64 parallel Welford moments, without artificial prior samples."""

    def __init__(self, dimension, *, clip=10., variance_epsilon=1e-8):
        if isinstance(dimension, bool) or int(dimension) != dimension or dimension < 1:
            raise ValueError('Positive integer observation dimension required')
        self.mean = np.zeros(int(dimension), np.float64)
        self.m2 = np.zeros(int(dimension), np.float64)
        self.count = 0
        self.clip = float(clip)
        self.variance_epsilon = float(variance_epsilon)
        if not math.isfinite(self.clip) or self.clip <= 0:
            raise ValueError('Positive finite observation clipping range required')
        if not math.isfinite(self.variance_epsilon) or self.variance_epsilon <= 0:
            raise ValueError('Positive finite variance epsilon required')

    def _array(self, observations):
        x = np.asarray(observations)
        if x.ndim < 1 or x.shape[-1] != len(self.mean) or not np.issubdtype(x.dtype, np.floating):
            raise ValueError('Floating observations with the registered final dimension required')
        if not np.isfinite(x).all():
            raise ValueError('Finite public observations required')
        return x

    def update(self, observations):
        x = self._array(observations).astype(np.float64).reshape(-1, len(self.mean))
        if not len(x):
            raise ValueError('Nonempty calibration observations required')
        batch_mean = x.mean(0)
        batch_m2 = ((x-batch_mean)**2).sum(0)
        total = self.count+len(x)
        delta = batch_mean-self.mean
        self.m2 += batch_m2+delta**2*self.count*len(x)/total
        self.mean += delta*len(x)/total
        self.count = int(total)

    def transform(self, observations):
        x = self._array(observations)
        if self.count < 1:
            raise RuntimeError('Observation normalization needs fitted training moments')
        denominator = np.sqrt(np.maximum(self.m2/self.count, 0.)+self.variance_epsilon)
        return np.clip((x-self.mean)/denominator, -self.clip, self.clip).astype(x.dtype, copy=False)

    def state_dict(self):
        return dict(mean=self.mean.copy(), m2=self.m2.copy(), count=self.count,
                    clip=self.clip, variance_epsilon=self.variance_epsilon)

    def load_state_dict(self, state):
        mean, m2 = np.asarray(state['mean'], np.float64), np.asarray(state['m2'], np.float64)
        count = state['count']
        if (mean.shape != self.mean.shape or m2.shape != self.m2.shape
                or not np.isfinite(mean).all() or not np.isfinite(m2).all() or (m2 < 0).any()
                or isinstance(count, bool) or int(count) != count or count < 0):
            raise ValueError('Invalid observation moment state')
        checked = ObservationMoments(len(mean), clip=state['clip'], variance_epsilon=state['variance_epsilon'])
        self.mean, self.m2, self.count = mean.copy(), m2.copy(), int(count)
        self.clip, self.variance_epsilon = checked.clip, checked.variance_epsilon


class NormalizedOffPolicy:
    """Calibrate exact first N random training inputs, then freeze permanently.

    Call observe_training(raw_obs) once per collected physical tick, before its
    native update. The existing worker does this after choosing its warmup
    action. No extra environment action, reward or random draw is introduced.
    """

    def __init__(self, native, obs_dim, config):
        if bool(getattr(native, 'on_policy', False)):
            raise ValueError('This normalizer is for off-policy learners only')
        self.native = native
        self.normalizer = ObservationMoments(obs_dim,
            clip=config.get('observation_clip', 10.),
            variance_epsilon=config.get('observation_variance_epsilon', 1e-8))
        steps = config.get('observation_calibration_steps', 2048)
        if isinstance(steps, bool) or int(steps) != steps or steps < 1:
            raise ValueError('Positive integer calibration budget required')
        self.calibration_steps = int(steps)
        self.calibration_observations = 0
        self.normalizer_frozen = False
        if int(config.get('learning_starts', 0)) < self.calibration_steps:
            raise ValueError('Normalizer must freeze before first learner update')
        if int(config.get('start_steps', 0)) < self.calibration_steps:
            raise ValueError('Normalizer calibration requires random training warmup')
        self._alias_private_rngs()

    def __getattr__(self, name):
        return getattr(self.native, name)

    @property
    def diagnostic_target(self):
        return getattr(self.native, 'diagnostic_target', self.native)

    def _alias_private_rngs(self):
        # The shared preserve_rng/seed_evaluation helper inspects vars(agent),
        # rather than forwarded attributes. Expose the identical RNG objects,
        # including nested native owners, so evaluation restore remains exact.
        for name in list(vars(self)):
            if name.startswith('_normalization_rng_'):
                delattr(self, name)
        seen_owners, seen_rngs = set(), set()
        def visit(owner, prefix):
            if owner is None or id(owner) in seen_owners:
                return
            seen_owners.add(id(owner))
            for name, value in vars(owner).items():
                if isinstance(value, (torch.Generator, np.random.Generator)) and id(value) not in seen_rngs:
                    alias = '_normalization_rng_'+prefix+name
                    setattr(self, alias, value)
                    seen_rngs.add(id(value))
            for name in ('native', 'ppo'):
                visit(getattr(owner, name, None), prefix+name+'_')
        visit(self.native, '')

    def observe_training(self, observations):
        if self.normalizer_frozen:
            return
        x = self.normalizer._array(observations).reshape(-1, len(self.normalizer.mean))
        take = min(len(x), self.calibration_steps-self.calibration_observations)
        if take < 1:
            raise ValueError('Nonempty uncalibrated training batch required')
        self.normalizer.update(x[:take])
        self.calibration_observations += take
        self.normalizer_frozen = self.calibration_observations == self.calibration_steps

    def _observations(self, observations):
        # Untrained initial development points remain fixed throughout random
        # calibration. At the calibration boundary a disclosed new input map
        # takes effect; that jump must not be presented as optimizer learning.
        return self.normalizer.transform(observations) if self.normalizer_frozen else observations

    def act(self, observations, prefs, deterministic=False):
        return self.native.act(self._observations(observations), prefs, deterministic)

    def update(self, batch):
        if not self.normalizer_frozen:
            raise RuntimeError('Native update attempted before observation calibration froze')
        data = dict(batch)
        data['obs'] = self._observations(batch['obs'])
        data['next_obs'] = self._observations(batch['next_obs'])
        if 'oadm_next_obs' in batch:
            data['oadm_next_obs'] = self._observations(batch['oadm_next_obs'])
        stats = self.native.update(data)
        stats.update(observation_calibration_count=self.calibration_observations,
                     observation_normalizer_frozen=True,
                     observation_normalizer_version=VERSION)
        return stats

    @property
    def after_exchange(self):
        """Expose a population hook only if the wrapped algorithm has one."""
        hook = getattr(self.native, 'after_exchange')
        def exchange(sample_batch, *, replay_ready=True):
            def transformed():
                batch = dict(sample_batch())
                for key in ('obs', 'next_obs', 'oadm_next_obs'):
                    if key in batch:
                        batch[key] = self._observations(batch[key])
                return batch
            return hook(transformed, replay_ready=replay_ready)
        return exchange

    def metadata(self):
        result = dict(self.native.metadata())
        result['frozen_observation_adaptation'] = dict(
            version=VERSION, observation_normalization='training-random-warmup-only-then-frozen',
            calibration_steps=self.calibration_steps, clip=self.normalizer.clip,
            variance_epsilon=self.normalizer.variance_epsilon,
            moments_dtype='float64', neural_input_dtype='preserve original boundary dtype',
            raw_environment_and_replay_preserved=True, objectives_unchanged=True,
            development_updates_statistics=False, holdout_updates_statistics=False,
            prefreeze_inference='raw observations; untrained initial policy',
            calibration_boundary_note='frozen input map starts at calibration boundary; not an optimizer-only change',
            native_algorithm_unchanged_by_this_wrapper=True,
        )
        return result

    def state_dict(self):
        result = dict(self.native.state_dict())
        if STATE_KEY in result:
            raise ValueError('Normalizer state key collision')
        result[STATE_KEY] = dict(version=VERSION, moments=self.normalizer.state_dict(),
            calibration_steps=self.calibration_steps,
            calibration_observations=self.calibration_observations,
            frozen=self.normalizer_frozen)
        return result

    def load_state_dict(self, state):
        self._load_state(state, checkpoint=False)

    def load_checkpoint_state_dict(self, state):
        self._load_state(state, checkpoint=True)

    def checkpoint_state_dict(self):
        export = getattr(self.native, "checkpoint_state_dict", self.native.state_dict)
        result = dict(export())
        result[STATE_KEY] = dict(version=VERSION, moments=self.normalizer.state_dict(),
            calibration_steps=self.calibration_steps, calibration_observations=self.calibration_observations,
            frozen=self.normalizer_frozen)
        return result

    def _load_state(self, state, checkpoint=False):
        item = state[STATE_KEY]
        if item['version'] != VERSION or item['calibration_steps'] != self.calibration_steps:
            raise ValueError('Normalizer version or calibration contract changed')
        count, frozen = item['calibration_observations'], item['frozen']
        if (isinstance(count, bool) or int(count) != count or not 0 <= count <= self.calibration_steps
                or not isinstance(frozen, bool) or frozen != (count == self.calibration_steps)
                or item['moments']['count'] != count):
            raise ValueError('Invalid calibration state')
        moments = ObservationMoments(len(self.normalizer.mean))
        moments.load_state_dict(item['moments'])
        if moments.clip != self.normalizer.clip or moments.variance_epsilon != self.normalizer.variance_epsilon:
            raise ValueError('Normalizer clipping or epsilon contract changed')
        native_state = {key:value for key,value in state.items() if key != STATE_KEY}
        loader = (getattr(self.native, "load_checkpoint_state_dict", self.native.load_state_dict)
                  if checkpoint else self.native.load_state_dict)
        loader(native_state)
        self.normalizer = moments
        self.calibration_observations, self.normalizer_frozen = int(count), frozen
        self._alias_private_rngs()


def wrap_off_policy(native, obs_dim, config):
    """Optional integration point; caller decides whether R3 enables it."""
    if not bool(config.get('observation_normalization', False)):
        return native
    return NormalizedOffPolicy(native, obs_dim, config)
