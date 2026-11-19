"""Task streams and rollouts; physical diagnostics never become policy inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .environments import build_environment
from .state import make_encoder, validate_action
from .image_data import DATA_ROOT, ImageEpisodes
from .io import STATE_KEYS, TRAIN_MASTER, digest, task_seed
from .full_spend import assert_terminal_full_spend


class Task:
    def __init__(self, domain, chapter, method='A', seed=0, *, path_master=TRAIN_MASTER,
                 radio_groups=100, data_root=None, holdout=False, condition_file=None):
        if domain not in ('v5', 'radio') or chapter not in (2, 3) or method not in ('A', 'B'):
            raise ValueError('Choose a registered environment, chapter 2/3, and method A/B')
        if isinstance(seed, bool) or int(seed) != seed or seed < 0:
            raise ValueError('A nonnegative integer training seed is required')
        self.domain, self.chapter, self.method = domain, chapter, method
        self.seed, self.path_master, self.holdout = int(seed), int(path_master), bool(holdout)
        self.env, self.spec, self.physical_dict = build_environment(
            domain, chapter, radio_groups=radio_groups, condition_file=condition_file)
        self.condition_identity = self.env.condition_identity
        self.condition_file = self.env.condition_file
        self.reward_version = self.env.reward_version
        self.image_sets = {}
        self.data_root = Path(data_root).expanduser().resolve() if data_root is not None else DATA_ROOT.resolve()
        self.data_identity = dict(task_seed_master=self.path_master, test_split_opened=holdout)
        if self.condition_identity is not None:
            self.data_identity['condition'] = dict(self.condition_identity)
        if domain == 'radio':
            for phase, code in [('train', 1), ('dev', 2), ('final', 3), ('engineering', 4)]:
                master = int(np.random.SeedSequence([
                    path_master, code, seed if phase in ('train', 'engineering') else 0
                ]).generate_state(1)[0])
                split = ('train' if phase in ('train', 'engineering') else
                         'test' if holdout and phase == 'final' else 'validation')
                self.image_sets[phase] = ImageEpisodes(split, groups=self.spec.groups,
                                                      root=self.data_root, master_seed=master)
            self.data_identity.update(
                split_manifest_sha256=digest(self.data_root/'split_manifest.json'),
                images_sha256=digest(self.data_root/'images.npy'),
                image_sources={phase: stream.master_seed for phase, stream in self.image_sets.items()},
                image_shape=list(next(iter(self.image_sets.values())).images.shape),
                groups=self.spec.groups)

    def plan(self, phase, index):
        if self.domain == 'v5':
            return dict(phase=phase, index=index,
                        task_seed=task_seed(phase, self.seed, index, self.path_master))
        ids, channel_seed = self.image_sets[phase].plan(index)
        return dict(phase=phase, index=index, task_seed=channel_seed, image_ids=ids.tolist())

    def reset(self, phase, index):
        if self.domain == 'v5':
            return self.env.reset(seed=self.plan(phase, index)['task_seed'])
        return self.image_sets[phase].reset(self.env, index)

    def encoder(self):
        return make_encoder(self.spec)

    def validate(self, action, state):
        return validate_action(action, state, self.spec)

    def episode(self, phase, index, actor, critic=None, generator=None, *, training=False,
                deterministic=None, capture=False):
        if deterministic is None:
            deterministic = not training
        if training and (critic is None or generator is None):
            raise ValueError('Training requires a critic and an explicit random generator')
        plan = self.plan(phase, index)
        path_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode())
        observation, info = self.reset(phase, index)
        assert info == {}
        if self.domain == 'v5':
            path_hash.update(observation['remaining_demand'].tobytes())
        encoder = self.encoder()
        encoder.reset(observation)
        states = []
        storage = {key: [] for key in ('raw', 'old_logp', 'mask', 'values')}
        actions, rewards, betas, hhis, roundoff, observations = [], [], [], [], [], []
        for t in range(self.spec.horizon):
            if capture:
                observations.append({key: np.asarray(value).copy() for key, value in observation.items()})
            state = encoder.state()
            with torch.no_grad():
                sampled = actor.sample(state, generator=generator, deterministic=deterministic)
                value = critic(state).detach() if training else torch.zeros_like(sampled['log_prob'])
            for key in ('raw', 'log_prob', 'mask'):
                if not torch.isfinite(sampled[key]).all():
                    raise FloatingPointError(key)
            assert sampled['log_prob'].shape == sampled['mask'].shape == value.shape == (1, 1)
            action, delta = self.validate(sampled['actions'][0], state)
            if self.chapter == 2:
                expected = min(self.spec.nominal_total, float(state['budget']))
                assert abs(action.sum()-expected) <= max(self.spec.action_tolerance, expected*1e-12)
            if training:
                states.append({key: np.asarray(state[key]).copy() for key in STATE_KEYS})
                for key, source in [('raw', 'raw'), ('old_logp', 'log_prob'), ('mask', 'mask')]:
                    storage[key].append(sampled[source].detach()[0].cpu().numpy().copy())
                storage['values'].append(value[0].cpu().numpy().copy())
            observation, reward, done, truncated, info = self.env.step(action)
            assert not truncated and done == (t == self.spec.horizon-1)
            if not done:
                assert reward == 0. and info == {}
            if self.domain == 'v5':
                # Audit identity only; neither model nor PPO sees these events.
                path_hash.update(self.env.native_observation_for_audit['last_arrivals'].tobytes())
            encoder.update(action, observation)
            actions.append(action)
            rewards.append(float(reward))
            roundoff.append(delta)
            betas.append(float(sampled.get('beta', torch.ones(1)).reshape(-1)[0]))
            total = float(action.sum())
            hhis.append(float(np.square(action/total).sum()) if total else 0.)
        actions = np.asarray(actions)
        assert_terminal_full_spend(encoder.state()['remaining'],self.spec)
        result = dict(native_return=float(sum(rewards)), rewards=np.asarray(rewards), actions=actions,
                      period_totals=actions.sum(1), total_used=float(actions.sum()),
                      remaining=float(encoder.state()['remaining']), mean_beta=float(np.mean(betas)),
                      mean_hhi=float(np.mean(hhis)), max_roundoff=float(max(roundoff)), terminal_info=info,
                      path_sha256=path_hash.hexdigest(), plan=plan)
        if capture:
            result['observations'] = observations
        if training:
            result['batch'] = dict(states={key: np.stack([state[key] for state in states]) for key in STATE_KEYS},
                                  **{key: np.asarray(values) for key, values in storage.items()},
                                  returns=np.full(self.spec.horizon, result['native_return']))
        return result

    def replay(self, phase, index, result):
        """Reexecute actual actions and match every captured public observation."""
        observation, _ = self.reset(phase, index)
        trace = hashlib.sha256()
        total = 0.
        for t, action in enumerate(result['actions']):
            expected = result['observations'][t]
            assert set(observation) == set(expected)
            for key in sorted(observation):
                value = np.asarray(observation[key])
                assert np.array_equal(value, expected[key]), (t, key)
                trace.update(key.encode())
                trace.update(value.tobytes())
            observation, reward, done, truncated, _ = self.env.step(action)
            assert not truncated and done == (t == self.spec.horizon-1)
            assert reward == result['rewards'][t]
            trace.update(action.tobytes())
            trace.update(np.asarray(reward, dtype=np.float64).tobytes())
            total += reward
        return float(total), trace.hexdigest()
