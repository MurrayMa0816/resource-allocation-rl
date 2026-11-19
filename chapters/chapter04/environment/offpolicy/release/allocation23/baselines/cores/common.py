"""Environment-free shared normalization, TD7 parameter schema and replay batching."""
import copy
import math
import numpy as np
import torch
SOURCES={"td7_full": {"repository": "https://github.com/sfujim/TD7"}}

def recent_algorithm_parameters(method, *, tiny=False, overrides=None, n_envs=1,horizon=81):
    if method not in SOURCES:
        raise ValueError('unknown recent continuous baseline: ' + method)
    if type(n_envs) is not int or n_envs < 1:
        raise ValueError('positive integer n_envs required')
    p = dict(gamma=1., batch_size=32 if tiny else 256, buffer_size=10000 if tiny else 200000,
             learning_starts=horizon if tiny else 16*horizon, updates_per_step=1)
    if method == 'td7_full':
        p.update(hidden_dim=32 if tiny else 256, embedding_dim=32 if tiny else 256,
                 learning_rate=3e-4, target_update_rate=250, policy_freq=2,
                 exploration_noise=.1, target_policy_noise=.2, noise_clip=.5,
                 priority_alpha=.4, min_priority=1., steps_before_checkpointing=750000,
                 max_eps_when_checkpointing=20, reset_weight=.9)
    else:
        p.update(actor_hidden_dim=32 if tiny else 128, critic_hidden_dim=32 if tiny else 512,
                 actor_blocks=1, critic_blocks=2, learning_rate=1e-4, weight_decay=.01,
                 tau=.005, initial_temperature=.01, target_entropy_coef=-.5)
    overrides = copy.deepcopy(dict(overrides or {}))
    unknown = set(overrides) - set(p)
    if unknown:
        raise ValueError('unsupported recent parameters: ' + ', '.join(sorted(unknown)))
    p.update(overrides)
    if p['gamma'] != 1.:
        raise ValueError('terminal-yield task requires gamma=1')
    if p['updates_per_step'] != 1:
        raise ValueError('registered main comparison requires UTD=1')
    integers = ['batch_size', 'buffer_size'] + (['hidden_dim', 'embedding_dim', 'target_update_rate',
        'policy_freq', 'max_eps_when_checkpointing'] if method == 'td7_full' else
        ['actor_hidden_dim', 'critic_hidden_dim', 'actor_blocks', 'critic_blocks'])
    for name in integers:
        if type(p[name]) is not int or p[name] < 1:
            raise ValueError(name + ' must be a positive integer')
    for name in ['learning_starts'] + (['steps_before_checkpointing'] if method == 'td7_full' else []):
        if type(p[name]) is not int or p[name] < 0:
            raise ValueError(name + ' must be a nonnegative integer')
    if p['buffer_size'] < max(p['batch_size'], p['learning_starts'], horizon):
        raise ValueError('replay capacity must cover warmup and batch')
    for name, value in p.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(name + ' must be finite')
    if p['learning_rate'] <= 0:
        raise ValueError('learning_rate must be positive')
    if method == 'td7_full':
        if any(p[k] < 0 for k in ('exploration_noise', 'target_policy_noise', 'noise_clip', 'priority_alpha')):
            raise ValueError('invalid TD7 noise or priority exponent')
        if p['min_priority'] <= 0 or not 0 < p['reset_weight'] <= 1:
            raise ValueError('invalid TD7 priority/reset settings')
    elif not (0 < p['tau'] <= 1 and p['weight_decay'] >= 0 and p['initial_temperature'] > 0 and p['target_entropy_coef'] < 0):
        raise ValueError('invalid SimBa optimization settings')
    return p


class RunningObservationNorm:
    """Parallel running moments; updated once per training action observation."""
    def __init__(self, size):
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = 1e-4
        self._cache = {}

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        n = len(values)
        delta, total = values.mean(0) - self.mean, self.count + n
        self.var = (self.var * self.count + values.var(0) * n + delta ** 2 * self.count * n / total) / total
        self.mean += delta * n / total
        self.count = total
        self._cache.clear()

    def __call__(self, values):
        if isinstance(values, torch.Tensor):
            key = (str(values.device), values.dtype)
            if key not in self._cache:
                self._cache[key] = (
                    torch.as_tensor(self.mean.copy(), device=values.device, dtype=values.dtype),
                    torch.as_tensor(np.sqrt(self.var + 1e-8), device=values.device, dtype=values.dtype))
            mean, std = self._cache[key]
            return (values - mean) / std
        return ((values - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)


def _uniform_replay_batches(replay, count, *, prefetch=None):
    """Batch immutable uniform-replay uploads, preserving draw/update order.

    No environment transition enters the buffer during this update block.
    Each draw still has the original replacement distribution and sample order.
    LAP is excluded because an update changes the next draw's distribution.
    CPU keeps the original path; the explicit option supports equivalence tests.
    """
    if replay.prioritized:
        raise ValueError('priority-dependent replay cannot be prefetched')
    if prefetch is None:
        prefetch = replay.device.type == 'cuda'
    if not prefetch:
        yield from (None for _ in range(count))
        return
    if replay.size == 0:
        raise ValueError('cannot sample an empty replay')
    for first in range(0, count, 8):
        size = min(8, count - first)
        indices = np.random.randint(replay.size, size=(size, replay.batch_size))
        # One contiguous host-to-device copy replaces five copies per update.
        # Each split retains contiguous rows for the network matrix products.
        lengths = [size * replay.batch_size * a.shape[1] for a in replay.arrays]
        packed = np.concatenate([a[indices].reshape(-1) for a in replay.arrays])
        uploaded = torch.as_tensor(packed, device=replay.device)
        tensors = [x.reshape(size, replay.batch_size, a.shape[1])
                   for x, a in zip(uploaded.split(lengths), replay.arrays)]
        for i in range(size):
            replay.ind = indices[i]
            yield tuple(x[i] for x in tensors)

