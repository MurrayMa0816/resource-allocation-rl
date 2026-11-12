"""Per-region expected positive terminal shortfall, separate from reward.

Thresholds are explicit research configuration, not established management
standards. Each sample is one completed allocation season under the same
policy and weather evaluation protocol. This module cannot check how samples
were collected and makes no statistical guarantee of population feasibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _fractions(value, label):
    """Copy real fractions without silently coercing booleans or strings."""
    if value is None:
        raise ValueError(f"{label} must be explicitly provided")
    try:
        raw = np.asarray(value)
        objects = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain real fractions in [0, 1]") from exc
    if (raw.dtype.kind not in "iuf"
            or any(isinstance(item, (bool, np.bool_)) for item in objects.flat)):
        raise ValueError(f"{label} must contain real fractions, not booleans, in [0, 1]")
    values = np.array(raw, dtype=np.float64, copy=True)
    if np.any(~np.isfinite(values)) or np.any(values < 0.) or np.any(values > 1.):
        raise ValueError(f"{label} must contain finite fractions in [0, 1]")
    # Equivalent scalar/vector controls should have canonical JSON values.
    values[values == 0.] = 0.
    return values


def _configuration_vector(value, label, n_regions):
    values = _fractions(value, label)
    if values.shape == ():
        values = np.full(n_regions, float(values), dtype=np.float64)
    elif values.shape != (n_regions,):
        raise ValueError(f"{label} must be a scalar or vector with shape ({n_regions},)")
    return tuple(float(item) for item in values)


@dataclass(frozen=True, slots=True, init=False)
class RegionalGuarantee:
    """Immutable regional targets and limits for E[max(target - Y, 0)].

Both arguments are required. Scalars expand to all regions; vectors follow
the environment's fixed region order. Fractions use [0, 1], so a tolerance
of 0.003 means 0.3 percentage points, not 0.3 percent of the target.
    """

    n_regions: int
    _target: tuple[float, ...]
    _tolerance: tuple[float, ...]

    def __init__(self, guarantee_target, shortfall_tolerance, n_regions=93):
        if (isinstance(n_regions, (bool, np.bool_))
                or not isinstance(n_regions, Integral) or n_regions <= 0):
            raise ValueError("n_regions must be a positive integer")
        n_regions = int(n_regions)
        target = _configuration_vector(guarantee_target, "guarantee_target", n_regions)
        tolerance = _configuration_vector(shortfall_tolerance, "shortfall_tolerance", n_regions)
        object.__setattr__(self, "n_regions", n_regions)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_tolerance", tolerance)

    @property
    def guarantee_target(self):
        """Detached target fractions in fixed regional order."""
        return np.array(self._target, dtype=np.float64)

    @property
    def shortfall_tolerance(self):
        """Detached limits on regional expected positive shortfall."""
        return np.array(self._tolerance, dtype=np.float64)

    def costs(self, relative_yield):
        """Return one season's positive shortfall for each region."""
        values = _fractions(relative_yield, "relative_yield")
        if values.shape != (self.n_regions,):
            raise ValueError(f"relative_yield must have shape ({self.n_regions},)")
        return np.maximum(self.guarantee_target - values, 0.)

    def specification(self):
        """Return detached JSON configuration and its research meaning."""
        return {
            "type": "expected_positive_terminal_shortfall",
            "unit": "fraction",
            "n_regions": self.n_regions,
            "guarantee_target": list(self._target),
            "shortfall_tolerance": list(self._tolerance),
            "cost_formula": "c_i = max(guarantee_target_i - terminal_relative_yield_i, 0)",
            "constraint": "E[c_i] <= shortfall_tolerance_i for every region",
            "evaluation_timing": "once per completed allocation season at terminal evaluation",
            "aggregation": "expectation of each region's positive shortfall across weather outcomes; no area weighting or division by horizon",
            "threshold_status": "explicit_research_configuration_not_official_standard",
            "per_episode_yield_guarantee": False,
        }

    def summarize(self, relative_yield_samples):
        """Summarize completed-season outcomes without asserting feasibility.

The caller must supply aligned regions and completed samples from the same
policy, targets and weather evaluation protocol, including policy randomness
when applicable. Independent samples are needed for the reported usual
sample standard error. Repeating a fixed historical weather trajectory does
not create independent weather samples merely by changing a reset seed.
        """
        values = _fractions(relative_yield_samples, "relative_yield_samples")
        if values.ndim != 2 or values.shape[1] != self.n_regions or values.shape[0] == 0:
            raise ValueError(
                f"relative_yield_samples must have shape (M, {self.n_regions}) with M >= 1")
        costs = np.maximum(self.guarantee_target[None, :] - values, 0.)
        n_samples = values.shape[0]
        mean = np.mean(costs, axis=0)
        standard_error = (np.std(costs, axis=0, ddof=1) / np.sqrt(n_samples)
                          if n_samples > 1 else None)
        limits = self.shortfall_tolerance
        excess = np.maximum(mean - limits, 0.)
        return {
            "n_samples": n_samples,
            "mean_shortfall": mean.tolist(),
            "standard_error": None if standard_error is None else standard_error.tolist(),
            "empirical_satisfied": (mean <= limits).tolist(),
            "empirical_excess": excess.tolist(),
            "n_exceeding": int(np.count_nonzero(mean > limits)),
            "max_excess": float(np.max(excess)),
            "statistical_guarantee": False,
            "sample_protocol_responsibility": "caller supplies completed seasons with aligned regions from the same policy and weather protocol; usual sample standard errors require independent samples",
            "interpretation": "empirical regional expected-shortfall estimates, not per-season guarantees or certified population feasibility",
        }
