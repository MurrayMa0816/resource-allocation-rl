"""One real terminal outcome vector, two learning objectives and area diagnostics."""
from __future__ import annotations

import math
import numpy as np

from ._util import real


def evaluate_outcomes(regional_performance, rice_area_ha, *, coverage_threshold, coverage_weight):
    """No clipping, smoothing, surrogate labels, or expected-shortfall constraints."""
    tau = real(coverage_threshold, "coverage_threshold", maximum=1.)
    weight = real(coverage_weight, "coverage_weight", maximum=1.)
    q_raw, a_raw = np.asarray(regional_performance), np.asarray(rice_area_ha)
    if (q_raw.ndim != 1 or not q_raw.size or q_raw.dtype.kind not in "iuf" or
            a_raw.shape != q_raw.shape or a_raw.dtype.kind not in "iuf"):
        raise ValueError("outcomes and areas must be real one-dimensional arrays of equal length")
    quality, area = q_raw.astype(np.float64, copy=True), a_raw.astype(np.float64, copy=True)
    if (not np.isfinite(quality).all() or (quality < 0).any() or (quality > 1).any() or
            not np.isfinite(area).all() or (area <= 0).any()):
        raise ValueError("outcomes must lie in [0,1] and areas must be finite positive")
    count = int(np.count_nonzero(quality >= tau))
    coverage = count / len(quality)
    mean = float(quality.mean())
    area_total = math.fsum(area.tolist())
    area_score = math.fsum((quality * (area / area_total)).tolist())
    return {"regional_performance": quality.copy(), "covered_mask": (quality >= tau),
            "covered_count": count, "N": len(quality), "coverage_rate": coverage,
            "mean_performance": mean, "area_weighted_performance": area_score,
            "area_coverage_rate": math.fsum((area[quality >= tau] / area_total).tolist()),
            "performance_min": float(quality.min()), "performance_p10": float(np.quantile(quality, .1)),
            "joint_score": weight * coverage + (1. - weight) * mean,
            "coverage_threshold": tau, "coverage_weight": weight,
            "C": coverage, "J": mean, "quality": quality.copy(), "coverage_count": count}

