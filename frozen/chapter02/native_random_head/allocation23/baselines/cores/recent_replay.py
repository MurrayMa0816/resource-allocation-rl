"""Raw-observation replay for the recent continuous baselines.

TD7 LAP sampling follows sfujim/TD7 buffer.py (MIT, Scott Fujimoto 2023;
license retained in recent_td7_core.py). Storage uses float32 instead of the
author's float64 NumPy arrays; tensors in both implementations are float32.
"""
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device, max_size, batch_size,
                 prioritized=False):
        self.device = torch.device(device)
        self.max_size, self.batch_size = int(max_size), int(batch_size)
        self.ptr = self.size = 0
        self.prioritized = prioritized
        self.arrays = [np.empty((self.max_size, dim), dtype=np.float32)
                       for dim in (state_dim, action_dim, state_dim, 1, 1)]
        self.priority = torch.zeros(self.max_size, device=self.device) if prioritized else None
        self.max_priority = 1.

    def add(self, state, action, next_state, reward, done):
        for array, value in zip(self.arrays, (state, action, next_state, reward, 1. - float(done))):
            array[self.ptr] = value
        if self.prioritized:
            self.priority[self.ptr] = self.max_priority
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        if self.size == 0:
            raise ValueError('cannot sample an empty replay')
        if self.prioritized:
            cumulative = self.priority[:self.size].cumsum(0)
            draws = torch.rand(self.batch_size, device=self.device) * cumulative[-1]
            self.ind = torch.searchsorted(cumulative, draws).cpu().numpy()
        else:
            self.ind = np.random.randint(self.size, size=self.batch_size)
        return tuple(torch.as_tensor(a[self.ind], device=self.device) for a in self.arrays)

    def update_priority(self, priority):
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(self.max_priority, float(priority.max()))

    def reset_max_priority(self):
        self.max_priority = float(self.priority[:self.size].max())

    def state_dict(self):
        return dict(ptr=self.ptr, size=self.size, arrays=[a[:self.size].copy() for a in self.arrays],
                    priority=None if self.priority is None else self.priority[:self.size].cpu(),
                    max_priority=self.max_priority)

    def load_state_dict(self, state):
        self.ptr, self.size = state['ptr'], state['size']
        if self.size > self.max_size:
            raise ValueError('saved replay exceeds configured capacity')
        for dest, src in zip(self.arrays, state['arrays']):
            dest[:self.size] = src
        if self.prioritized:
            self.priority[:self.size] = state['priority'].to(self.device)
        self.max_priority = state['max_priority']
