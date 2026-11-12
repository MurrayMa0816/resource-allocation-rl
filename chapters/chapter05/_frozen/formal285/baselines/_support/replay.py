"""Original vector n-step replay retaining the actual one-step OADM label.

This is the disclosed gamma=1 task adaptation, without intermediate soft
entropy or off-policy importance correction. It does not change the reward.
"""
from collections import deque
import numpy as np


class NStepReplay:
    def __init__(self, capacity, obs_dim, action_dim, n_step=1, n_env=1, **unused):
        self.capacity, self.n_step, self.n_env = map(int, (capacity, n_step, n_env))
        if min(self.capacity, self.n_step, self.n_env, obs_dim, action_dim) < 1:
            raise ValueError('Positive replay geometry required')
        self.size = self.cursor = self.original_transitions = self.emitted_transitions = 0
        self.data = {k: np.empty((self.capacity, n), np.float32) for k, n in
            [('obs', obs_dim), ('actions', action_dim), ('weights', 2), ('rewards', 2),
             ('next_obs', obs_dim), ('dones', 1), ('oadm_next_obs', obs_dim)]}
        self.pending = [deque() for _ in range(self.n_env)]

    def _emit(self, lane):
        first = lane[0]
        total = np.zeros(2, np.float64)
        for t, row in enumerate(lane):
            if t >= self.n_step:
                break
            total += row['rewards']
            last = row
            if row['dones'][0]:
                break
        saved = dict(first, rewards=total.astype(np.float32), next_obs=last['next_obs'],
                     dones=last['dones'], oadm_next_obs=first['next_obs'])
        for name, array in self.data.items():
            array[self.cursor] = saved[name]
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.emitted_transitions += 1

    def add(self, **batch):
        expected = set(self.data) - {'oadm_next_obs'}
        if set(batch) != expected:
            raise ValueError('The original six transition fields are required')
        checked = {}
        for name, value in batch.items():
            a = np.asarray(value, np.float32)
            shape = (self.n_env, self.data[name].shape[1])
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError('Replay field shape/finite mismatch: ' + name)
            checked[name] = a
        if not np.isin(checked['dones'], [0., 1.]).all():
            raise ValueError('Real terminal masks must be0or1')
        for i, lane in enumerate(self.pending):
            row = {k: a[i].copy() for k, a in checked.items()}
            lane.append(row)
            if len(lane) >= self.n_step:
                self._emit(lane)
                lane.popleft()
            if row['dones'][0]:
                while lane:
                    self._emit(lane)
                    lane.popleft()
        self.original_transitions += self.n_env
        if self.original_transitions != self.emitted_transitions + sum(map(len, self.pending)):
            raise AssertionError('Transition loss or duplicate emission')

    def sample(self, count, rng, preferences=None):
        if self.size < 1:
            raise ValueError('Empty replay')
        indices = rng.integers(self.size, size=int(count))
        batch = {k: a[indices] for k, a in self.data.items()}
        if preferences is not None:
            batch['weights'] = np.asarray(preferences, np.float32)
        return batch

    def assert_episode_boundary(self):
        if any(self.pending) or self.original_transitions != self.emitted_transitions:
            raise AssertionError('Replay contains an incomplete episode tail')

    def metadata(self):
        return dict(n_step=self.n_step, gamma=1., native_reward_vectors_unchanged=True,
            original_transitions=self.original_transitions, emitted_transitions=self.emitted_transitions,
            single_step_auxiliary_label_retained=True, cross_episode_mixing=False,
            intermediate_soft_entropy_accumulated=False, importance_correction=False)
