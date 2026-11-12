"""Exogenous tasks from independent RNG streams, never conditioned on actions."""
from __future__ import annotations

import numpy as np

from .config import ResolvedConfig


class TaskStream:
    """Sample on demand; future task arrays are not exposed by the environment.

    Region identity and distribution columns are fixed. Reusing a scenario seed
    yields the same tasks independently of actions, feedback or budget mode.
    """

    def __init__(self, config: ResolvedConfig, seed: int):
        self._config = config
        sequences = np.random.SeedSequence(seed).spawn(4)
        self._initial_rng, self._regime_rng, self._event_rng, self._size_rng = [
            np.random.Generator(np.random.PCG64(s)) for s in sequences
        ]
        self._initial = None
        self._next_step = 1

    def initial_demand(self) -> np.ndarray:
        if self._initial is None:
            c = self._config
            self._initial = self._initial_rng.uniform(c.initial_demand_low, c.initial_demand_high)
        return self._initial.copy()

    def arrivals(self, next_step: int) -> np.ndarray:
        c = self._config
        if isinstance(next_step, (bool, np.bool_)) or not isinstance(next_step, (int, np.integer)):
            raise ValueError("arrival step must be an integer")
        if next_step == c.horizon:
            if next_step != self._next_step:
                raise ValueError("arrival calls must be sequential")
            return np.zeros(c.num_regions, dtype=np.float64)
        if next_step != self._next_step or not 1 <= next_step < c.horizon:
            raise ValueError("arrival calls must be sequential, from 1 through horizon")
        row = next_step - 1
        mode = self._regime_rng.choice(len(c.arrival_shared_multipliers), p=c.arrival_shared_probabilities)
        probability = c.arrival_probability[row] * c.arrival_shared_multipliers[mode]
        # Always draw both arrays, including deterministic or zero-probability
        # cases, to keep task streams independent of action-dependent branching.
        occur = self._event_rng.random(c.num_regions) < probability
        sizes = self._size_rng.uniform(c.arrival_low[row], c.arrival_high[row])
        self._next_step += 1
        return np.where(occur, sizes, 0.0)
