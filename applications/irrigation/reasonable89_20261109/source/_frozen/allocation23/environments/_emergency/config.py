"""Configuration of the world, without algorithm features or allocation rules."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "emergency_environment.json"


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        a = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative values")
    valid_shapes = {(), shape}
    if len(shape) == 2:
        valid_shapes.add((shape[1],))
    if a.shape not in valid_shapes:
        raise ValueError(f"{name} shape must be scalar, regional vector, or exactly {shape}")
    a = np.broadcast_to(a, shape).copy()
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class ResolvedConfig:
    num_regions: int
    horizon: int
    budget_mode: str
    fixed_period_budget: float
    initial_budget: float
    conversion: np.ndarray
    period_capacity: np.ndarray
    initial_demand_low: np.ndarray
    initial_demand_high: np.ndarray
    arrival_probability: np.ndarray
    arrival_low: np.ndarray
    arrival_high: np.ndarray
    arrival_shared_multipliers: np.ndarray
    arrival_shared_probabilities: np.ndarray
    period_hours: float
    action_tolerance: float
    accounting_atol: float


@dataclass(frozen=True)
class EnvConfig:
    """All amounts are float64; arrival rows correspond to decisions 1..T-1.

    Regional arrays accept a scalar or N values. Arrival arrays accept a scalar,
    N values, or a (T-1,N) matrix. Period-dependent scalars must be explicitly
    repeated across regions; a vector is always interpreted as regional.

    The optional shared multiplier is freshly sampled at each transition. It
    induces same-period correlation only, not a hidden persistent regime.
    This dataclass is a generic configuration builder; its scalar defaults are
    examples, not the active research experiment. Use ``load_default_config``
    for the versioned experiment, or construct ``EmergencyAllocationEnv()``.
    Keeping the builder independent of a fixed N-vector allows small explicit
    fixtures and alternate problem sizes without silently resizing regions.
    """

    num_regions: int = 120
    horizon: int = 24
    budget_mode: str = "fixed"
    fixed_period_budget: float = 840.0
    initial_budget: float | None = None
    conversion: Any = 80.0 / 21.7
    period_capacity: Any = 240.0
    initial_demand_low: Any = 400.0
    initial_demand_high: Any = 1400.0
    arrival_probability: Any = 0.25
    arrival_low: Any = 200.0
    arrival_high: Any = 650.0
    arrival_shared_multipliers: Any = (1.0,)
    arrival_shared_probabilities: Any = (1.0,)
    period_hours: float = 3.0
    action_tolerance: float = 1e-8
    accounting_atol: float = 1e-7

    def resolve(self) -> ResolvedConfig:
        for name in ("num_regions", "horizon"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        n, t = int(self.num_regions), int(self.horizon)
        if self.budget_mode not in {"fixed", "adaptive"}:
            raise ValueError("budget_mode must be 'fixed' or 'adaptive'")
        q = _number(self.fixed_period_budget, "fixed_period_budget")
        period_hours = _number(self.period_hours, "period_hours", positive=True)
        atol = _number(self.action_tolerance, "action_tolerance")
        accounting_atol = _number(self.accounting_atol, "accounting_atol")
        total_fixed = q * t
        if not np.isfinite(total_fixed):
            raise ValueError("horizon times fixed_period_budget overflows")
        b = total_fixed if self.initial_budget is None else _number(self.initial_budget, "initial_budget")
        if self.budget_mode == "fixed" and abs(b - total_fixed) > atol:
            raise ValueError("fixed mode requires initial_budget == horizon * fixed_period_budget")
        if self.budget_mode == "fixed":
            b = total_fixed
        conversion = _array(self.conversion, (n,), "conversion")
        capacity = _array(self.period_capacity, (n,), "period_capacity")
        if (conversion <= 0).any():
            raise ValueError("conversion must be strictly positive")
        init_low = _array(self.initial_demand_low, (n,), "initial_demand_low")
        init_high = _array(self.initial_demand_high, (n,), "initial_demand_high")
        p_raw = np.asarray(self.arrival_probability, dtype=np.float64)
        if not np.isfinite(p_raw).all() or ((p_raw < 0) | (p_raw > 1)).any():
            raise ValueError("arrival_probability must be in [0,1]")
        p = _array(self.arrival_probability, (t - 1, n), "arrival_probability")
        low = _array(self.arrival_low, (t - 1, n), "arrival_low")
        high = _array(self.arrival_high, (t - 1, n), "arrival_high")
        if (init_low > init_high).any() or (low > high).any():
            raise ValueError("demand/arrival low bounds must not exceed high bounds")
        mult_raw = np.asarray(self.arrival_shared_multipliers, dtype=np.float64)
        prob_raw = np.asarray(self.arrival_shared_probabilities, dtype=np.float64)
        if mult_raw.ndim != 1 or not mult_raw.size or prob_raw.shape != mult_raw.shape:
            raise ValueError("shared multipliers and probabilities must be equally sized nonempty vectors")
        mult = _array(mult_raw, mult_raw.shape, "arrival_shared_multipliers")
        probabilities = _array(prob_raw, prob_raw.shape, "arrival_shared_probabilities")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-12, rtol=0):
            raise ValueError("shared regime probabilities must sum to 1")
        if (p_raw * float(mult.max()) > 1).any():
            raise ValueError("arrival probability times shared multiplier must not exceed 1")
        probabilities = probabilities.copy() / probabilities.sum()
        probabilities.setflags(write=False)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            maximum_demand = init_high + high.sum(axis=0)
            checks = [maximum_demand, maximum_demand / conversion, capacity / conversion,
                      conversion * max(q, b), maximum_demand.sum(), t * capacity,
                      n * max(q, b)]
        if not all(np.isfinite(x).all() for x in checks):
            raise ValueError("configuration exceeds finite float64 accounting capacity")
        return ResolvedConfig(n, t, self.budget_mode, q, b, conversion, capacity,
                              init_low, init_high, p, low, high, mult, probabilities,
                              period_hours, atol, accounting_atol)

    def to_dict(self) -> dict:
        def convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            return value
        return {k: convert(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, value: dict) -> "EnvConfig":
        if not isinstance(value, dict):
            raise ValueError("environment configuration must be a mapping")
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown environment configuration fields: {sorted(unknown)}")
        result = cls(**value)
        result.resolve()
        return result


def load_config(path: str | Path) -> EnvConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "environment" in data:
        unknown = set(data) - {"environment", "metadata"}
        if unknown:
            raise ValueError(f"unknown configuration wrapper fields: {sorted(unknown)}")
        data = data["environment"]
    return EnvConfig.from_dict(data)


def load_default_config() -> EnvConfig:
    """Load the sole active configuration, failing if missing or malformed.

    Loading on each call deliberately avoids a stale cached research default.
    Existing environment instances retain their resolved, copied parameters.
    """
    return load_config(DEFAULT_CONFIG_PATH)


def expected_total_tasks(cfg: EnvConfig) -> float:
    """Design-time expected scale only; never an observation or policy feature."""
    c = cfg.resolve()
    expected_multiplier = float(c.arrival_shared_multipliers @ c.arrival_shared_probabilities)
    initial = c.initial_demand_low / 2 + c.initial_demand_high / 2
    arrivals = c.arrival_probability * expected_multiplier * (c.arrival_low / 2 + c.arrival_high / 2)
    return float(initial.sum() + arrivals.sum())
