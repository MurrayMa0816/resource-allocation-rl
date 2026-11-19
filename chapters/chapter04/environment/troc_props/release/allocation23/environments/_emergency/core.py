"""Raw-interaction environment for irreversible, multi-region resource inputs.

An action commits resource *allowance*.  It is not a measurement of diesel
physically burned.  The whole submitted amount is charged even when demand or
processing capacity limits useful service.  There is no local resource stock,
refund, action projection, demand masking, or automatic redistribution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .arrivals import TaskStream
__version__ = "5.0.0"
from .config import EnvConfig, load_default_config


ENVIRONMENT_VERSION = __version__
OBSERVATION_SCHEMA = "raw-backlog-events-v5"
ACTION_SCHEMA = "irreversible-input-v5"


class AllocationBox(spaces.Box):
    """Box-shaped action space with a joint resource-budget constraint.

    ``sample`` draws a legal random action using only the public total budget;
    it never reads regional demand, conversion rates, or processing capacities.
    Its RNG is independent of the task stream.  Sampling is a Gym space
    contract, not a repair or projection of submitted policy actions.

    The Box bounds are a static envelope. ``contains`` additionally checks the
    fixed sum, or the current remaining allowance in adaptive mode.
    """

    def __init__(
        self,
        num_regions: int,
        maximum_total: float,
        mode: str,
        current_total: Callable[[], float],
        tolerance: float,
    ) -> None:
        envelope_total = maximum_total + (tolerance if mode == "fixed" else 0.0)
        if not np.isfinite(envelope_total):
            raise ValueError("budget plus action tolerance exceeds float64 space bounds")
        super().__init__(
            low=np.zeros(num_regions, dtype=np.float64),
            high=np.full(num_regions, envelope_total, dtype=np.float64),
            dtype=np.float64,
        )
        self._mode = mode
        self._current_total = current_total
        self._tolerance = tolerance

    def sample(self, mask=None, probability=None) -> np.ndarray:
        if mask is not None or probability is not None:
            raise ValueError("AllocationBox does not accept masks or probabilities")
        limit = float(self._current_total())
        if limit <= 0:
            return np.zeros(self.shape, dtype=np.float64)
        minimum=getattr(self,'_minimum_total',None)
        lower=float(minimum()) if minimum is not None else 0.
        total = limit if self._mode == "fixed" else float(lower+self.np_random.random()*(limit-lower))
        weights = self.np_random.dirichlet(np.ones(self.shape[0]))
        action = np.asarray(total * weights, dtype=np.float64)
        # Correct only our own random draw's floating-point sum. This is never
        # called on an action supplied to the environment's step method.
        largest = int(np.argmax(action))
        action[largest] += total - float(action.sum())
        if self._mode == "adaptive" and float(action.sum()) > limit:
            action *= np.nextafter(limit, 0.0) / float(action.sum())
        if not self.contains(action):
            raise RuntimeError("action-space sampling could not satisfy the budget")
        return action

    def contains(self, x: Any) -> bool:
        try:
            action = np.asarray(x, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return False
        if action.shape != self.shape or not np.isfinite(action).all() or (action < 0).any():
            return False
        with np.errstate(over="ignore"):
            total = float(action.sum())
        if not np.isfinite(total):
            return False
        limit = float(self._current_total())
        if self._mode == "fixed":
            return abs(total - limit) <= self._tolerance
        minimum=getattr(self,'_minimum_total',None)
        lower=float(minimum()) if minimum is not None else 0.
        return total <= limit and total >= lower-self._tolerance


class EmergencyAllocationEnv(gym.Env):
    """Finite-horizon task service with a terminal regional-completion reward.

    The complete internal reward state contains elapsed time, remaining demand,
    cumulative arrived demand, and (in adaptive mode) remaining central budget.
    Observations contain only current raw state and just-completed events.
    Agents may maintain histories; the environment never constructs them.

    With no explicit config, construction loads ``configs/environment.json``.
    Explicit configurations remain supported for controlled experiments.

    Negative, nonfinite, wrongly-shaped, or over-budget actions are rejected
    before state or RNG mutation.  Fixed-sum validation permits the configured
    numerical tolerance but still accounts for the original submitted vector.
    Adaptive overspending is rejected strictly; it is never silently clipped.
    """

    metadata = {"render_modes": []}
    render_mode = None

    def __init__(self, config: EnvConfig | Mapping[str, Any] | None = None, *, render_mode: str | None = None):
        super().__init__()
        if render_mode is not None:
            raise ValueError("this numerical environment has no rendering modes")
        if config is None:
            config = load_default_config()
        self._config = config if isinstance(config, EnvConfig) else EnvConfig.from_dict(dict(config))
        self._resolved = self._config.resolve()
        c = self._resolved
        self._initialized = False
        self._terminated = False
        self._remaining_budget = float(c.initial_budget)
        self._stream: TaskStream | None = None

        # Space metadata is public too. Use nonnegative unbounded envelopes
        # rather than revealing private H or arrival-distribution bounds here.
        # Numerical finiteness is still enforced by config and step accounting.
        observation_spaces: dict[str, gym.Space] = {
            "time_index": spaces.Box(
                low=np.array(0, dtype=np.int64),
                high=np.array(c.horizon, dtype=np.int64),
                dtype=np.int64,
            ),
            "remaining_demand": spaces.Box(
                low=np.zeros(c.num_regions), high=np.full(c.num_regions, np.inf), dtype=np.float64
            ),
            "last_completed": spaces.Box(
                low=np.zeros(c.num_regions), high=np.full(c.num_regions, np.inf), dtype=np.float64
            ),
            "last_arrivals": spaces.Box(
                low=np.zeros(c.num_regions), high=np.full(c.num_regions, np.inf), dtype=np.float64
            ),
        }
        if c.budget_mode == "adaptive":
            observation_spaces["remaining_budget"] = spaces.Box(
                low=np.array(0.0), high=np.array(c.initial_budget), dtype=np.float64
            )
        self.observation_space = spaces.Dict(observation_spaces)
        maximum_action_total = c.fixed_period_budget if c.budget_mode == "fixed" else c.initial_budget
        self.action_space = AllocationBox(
            c.num_regions, maximum_action_total, c.budget_mode,
            self._action_budget, c.action_tolerance,
        )

    def _action_budget(self) -> float:
        if self._resolved.budget_mode == "fixed":
            return float(self._resolved.fixed_period_budget)
        return float(self._remaining_budget)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options:
            raise ValueError("reset options are not supported; configure tasks at construction")
        super().reset(seed=seed)
        c = self._resolved
        # Gym's RNG supplies a fresh task-stream seed on each unseeded reset.
        # The task stream owns separate initial/arrival RNGs, unaffected by
        # allocation-space sampling, actions, or budget mode.
        stream_seed = int(self.np_random.integers(0, np.iinfo(np.int64).max))
        self._stream = TaskStream(c, stream_seed)
        self._remaining = self._stream.initial_demand()
        self._arrived = self._remaining.copy()
        self._completed = np.zeros(c.num_regions, dtype=np.float64)
        self._allocated = np.zeros(c.num_regions, dtype=np.float64)
        self._ineffective = np.zeros(c.num_regions, dtype=np.float64)
        self._last_completed = np.zeros(c.num_regions, dtype=np.float64)
        self._last_arrivals = np.zeros(c.num_regions, dtype=np.float64)
        self._remaining_budget = float(c.initial_budget)
        self._time_index = 0
        self._terminated = False
        self._initialized = True
        self._assert_invariants()
        return self._observation(), {}

    def _require_active(self) -> None:
        if not self._initialized:
            raise RuntimeError("call reset before stepping the environment")
        if self._terminated:
            raise RuntimeError("episode has terminated; reset before another step")

    def _validated_action(self, action: Any) -> np.ndarray:
        try:
            allocation = np.asarray(action, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("action must be a finite numeric vector") from error
        c = self._resolved
        if allocation.shape != (c.num_regions,):
            raise ValueError(f"action must have shape ({c.num_regions},)")
        if not np.isfinite(allocation).all():
            raise ValueError("action must contain finite values")
        if (allocation < 0).any():
            raise ValueError("negative resource inputs are not allowed")
        with np.errstate(over="ignore"):
            total = float(allocation.sum())
        if not np.isfinite(total):
            raise ValueError("total input exceeds finite float64 range")
        if c.budget_mode == "fixed":
            if abs(total - c.fixed_period_budget) > c.action_tolerance:
                raise ValueError("fixed mode requires the configured total input each period")
        elif total > self._remaining_budget:
            raise ValueError("input exceeds the remaining central budget")
        return allocation.copy()

    def step(self, action: Any):
        self._require_active()
        allocation = self._validated_action(action)
        c = self._resolved
        # All action-dependent calculations are checked before drawing an event.
        # A bad action never changes either the state or the task random stream.
        try:
            with np.errstate(over="raise", invalid="raise"):
                completed = np.minimum(
                    self._remaining,
                    np.minimum(c.conversion * allocation, c.period_capacity),
                )
                effective_input = completed / c.conversion
                ineffective = allocation - effective_input
                accounting_tolerance = c.accounting_atol + 1e-10 * np.maximum(allocation, effective_input)
                if (ineffective < -accounting_tolerance).any():
                    raise ArithmeticError("service exceeded the submitted resource input")
                # A product followed by division may leave a negative roundoff
                # residue; this changes neither the action nor the service.
                ineffective = np.maximum(ineffective, 0.0)
                allocated_total = self._allocated + allocation
                completed_total = self._completed + completed
                ineffective_total = self._ineffective + ineffective
                demand_after_service = self._remaining - completed
        except (FloatingPointError, ArithmeticError) as error:
            raise ValueError("action leads to unrepresentable service accounting") from error
        if not all(np.isfinite(v).all() for v in (
            allocated_total, completed_total, ineffective_total, demand_after_service
        )):
            raise ValueError("action leads to nonfinite service accounting")
        next_step = self._time_index + 1
        assert self._stream is not None
        arrivals = self._stream.arrivals(next_step)
        self._remaining = demand_after_service + arrivals
        self._arrived += arrivals
        self._completed = completed_total
        self._allocated = allocated_total
        self._ineffective = ineffective_total
        self._last_completed = completed.copy()
        self._last_arrivals = arrivals.copy()
        if c.budget_mode == "adaptive":
            self._remaining_budget -= float(allocation.sum())
        self._time_index = next_step
        self._terminated = next_step == c.horizon
        self._assert_invariants()
        if self._terminated:
            info = self._terminal_info()
            reward = info["region_mean_completion"]
        else:
            info, reward = {}, 0.0
        return self._observation(), float(reward), bool(self._terminated), False, info

    def _observation(self) -> dict[str, np.ndarray]:
        observation = {
            "time_index": np.array(self._time_index, dtype=np.int64),
            "remaining_demand": self._remaining.copy(),
            "last_completed": self._last_completed.copy(),
            "last_arrivals": self._last_arrivals.copy(),
        }
        if self._resolved.budget_mode == "adaptive":
            observation["remaining_budget"] = np.array(self._remaining_budget, dtype=np.float64)
        return observation

    def _terminal_info(self) -> dict[str, Any]:
        ratios = np.divide(
            self._completed, self._arrived,
            out=np.ones(self._resolved.num_regions, dtype=np.float64),
            where=self._arrived > 0,
        )
        if (not np.isfinite(ratios).all() or (ratios < -1e-12).any()
                or (ratios > 1 + 1e-12).any()):
            raise AssertionError("terminal regional completion outside [0,1]")
        # Separate valid metric bounds from floating-point ledger arithmetic.
        # Independently accumulated service/task sums can differ by one ULP
        # after a region has fully completed its tasks. Never change either
        # ledger (or an action) to repair this bounded reporting metric.
        ratios = np.clip(ratios, 0.0, 1.0)
        total_demand = float(self._arrived.sum())
        aggregate = float(self._completed.sum() / total_demand) if total_demand > 0 else 1.0
        if not np.isfinite(aggregate) or not -1e-12 <= aggregate <= 1 + 1e-12:
            raise AssertionError("terminal aggregate completion outside [0,1]")
        aggregate = float(np.clip(aggregate, 0.0, 1.0))
        result = {
            "per_region_completion": ratios.copy(),
            "total_arrived": self._arrived.copy(),
            "total_completed": self._completed.copy(),
            "remaining_demand": self._remaining.copy(),
            "allocated_resource": self._allocated.copy(),
            "ineffective_resource": self._ineffective.copy(),
            "total_allocated": float(self._allocated.sum()),
            "total_ineffective": float(self._ineffective.sum()),
            "zero_demand_regions": int(np.count_nonzero(self._arrived == 0)),
            "region_mean_completion": float(ratios.mean()),
            "aggregate_completion": aggregate,
            "environment_version": ENVIRONMENT_VERSION,
            "observation_schema": OBSERVATION_SCHEMA,
            "action_schema": ACTION_SCHEMA,
        }
        if self._resolved.budget_mode == "adaptive":
            result["remaining_budget"] = float(self._remaining_budget)
        return result

    def _assert_invariants(self) -> None:
        c = self._resolved
        values = (self._remaining, self._arrived, self._completed, self._allocated, self._ineffective)
        if not all(np.isfinite(v).all() for v in values):
            raise AssertionError("nonfinite physical or allowance ledger")
        if any((v < 0).any() for v in values):
            raise AssertionError("negative physical or allowance ledger")
        if not np.allclose(
            self._remaining + self._completed, self._arrived,
            atol=c.accounting_atol, rtol=1e-10,
        ):
            raise AssertionError("task conservation failed")
        if not np.allclose(
            self._completed / c.conversion + self._ineffective, self._allocated,
            atol=c.accounting_atol, rtol=1e-10,
        ):
            raise AssertionError("effective plus ineffective input does not equal committed allowance")
        if c.budget_mode == "adaptive":
            if not np.isfinite(self._remaining_budget) or self._remaining_budget < 0:
                raise AssertionError("invalid central remaining budget")
            if not np.isclose(
                float(self._allocated.sum()) + self._remaining_budget, c.initial_budget,
                atol=c.accounting_atol, rtol=1e-10,
            ):
                raise AssertionError("central resource allowance conservation failed")

