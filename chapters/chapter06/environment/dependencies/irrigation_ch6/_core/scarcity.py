"""A sufficient all-action shortage certificate for a declared dry scenario.

The certified quantity is the instantaneous volume needed to bring every root
zone to its no-stress threshold, not daily full ET, a delivery limit, or ROI.
Nothing here changes water, actions, rewards, or the physical transition.

Let S be total stored m3 and W total wilting-point storage. For any region,
Ta >= c * (B - Bwp)_+, where c=min(1,Kcb*ET0_min/(n*Rmax)) and
Rmax=max(Bmax-Bwp). This also covers the kernel's available-water clamp.
After adding a total q<=Q_upper/n, overflow only reduces storage. Because
x-c*(x-W)_+ is increasing for 0<=c<=1, ignoring evaporation and drainage gives
S_next <= S+q-c*(S+q-W)_+. Iterate this bound without resetting storage.
The actual sum of regional no-stress deficits is >= C-S, with C total
critical storage. C-S_upper>Q_upper therefore certifies shortage for every
legal constant-total allocation, including concentration in one region.

The caller must establish the zero-rain and ET0 support assumptions. This is
not an empirical drought guarantee, an optimality claim, or a C4--6 guarantee.
"""
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import math
from numbers import Integral, Real

import numpy as np

from .field import FieldParams


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _vector(value, name, length=None):
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iuf" or not array.size:
        raise ValueError(f"{name} must be a nonempty real vector")
    if any(isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat):
        raise TypeError(f"{name} must not contain booleans")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} has the wrong length")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _outward(value, upper):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("certificate arithmetic exceeds finite float range")
    exact = Decimal.from_float(result)
    if (upper and exact < value) or (not upper and exact > value):
        result = math.nextafter(result, math.inf if upper else -math.inf)
    return result


def scarcity_certificate(*, field_params, initial_storage_mm, crop_kcb,
                         daily_quota_m3, min_et0_mm_day, no_rain,
                         chapter=3, substeps_per_day=96):
    """Return a conservative, JSON-serializable sufficient certificate.

    Valid unsupported conditions return ``status='not_certified'``; malformed
    numeric inputs raise. ``not_certified`` means this proof does not establish
    the claim, not that a non-shortage trajectory necessarily exists.
    Crop coefficients contain one shared Kcb per decision day. The bound
    includes the existing action-sum tolerance and a conservative FP64 guard;
    its scalar arithmetic uses 60-digit directed decimal rounding.
    """
    parameters = tuple(field_params)
    if not parameters or any(not isinstance(p, FieldParams) for p in parameters):
        raise TypeError("field_params must be a nonempty sequence of FieldParams")
    storage = _vector(initial_storage_mm, "initial_storage_mm", len(parameters))
    coefficients = _vector(crop_kcb, "crop_kcb")
    quota = _real(daily_quota_m3, "daily_quota_m3")
    et_min = _real(min_et0_mm_day, "min_et0_mm_day")
    if type(no_rain) is not bool:
        raise TypeError("no_rain must be a boolean describing verified weather support")
    if isinstance(chapter, (bool, np.bool_)) or not isinstance(chapter, Integral) or chapter not in (3, 4, 5, 6):
        raise ValueError("chapter must be 3, 4, 5, or 6")
    if isinstance(substeps_per_day, (bool, np.bool_)) or not isinstance(substeps_per_day, Integral) or substeps_per_day < 1:
        raise ValueError("substeps_per_day must be a positive integer")
    maximum = np.array([p.max_store_mm for p in parameters])
    if np.any(storage > maximum):
        raise ValueError("initial storage exceeds field capacity")
    result = dict(schema="current-deficit-scarcity-certificate-v1", certified=False,
                  status="not_certified", reasons=[], chapter=int(chapter),
                  days=len(coefficients), substeps_per_day=int(substeps_per_day),
                  daily_quota_m3=quota, minimum_et0_mm_day=et_min, no_rain=no_rain,
                  quantity="sum_i 10*area_ha_i*max(Z_i*theta_crit_i-B_i,0), m3 at day start",
                  scope="all legal constant-total C3 actions under verified zero-rain and ET0 support; not daily full-ET need, terminal value, or C4--6",
                  daily_storage_upper_m3=[], daily_deficit_lower_m3=[],
                  daily_margin_m3=[], minimum_margin_m3=None)
    if chapter != 3:
        result["reasons"].append("carryover chapters do not have a fixed daily actual total")
    if not no_rain:
        result["reasons"].append("zero-rain support has not been established")
    if quota <= 0:
        result["reasons"].append("the requested strict positive-supply condition is absent")
    if et_min <= 0 or np.any(coefficients <= 0):
        result["reasons"].append("positive transpiration forcing on every day is required")
    if result["reasons"]:
        return result

    # FieldParams also accepts NumPy real scalars. Convert their represented
    # value to a Python float before Decimal; no precision beyond that input
    # value is invented, and existing Python-float parameters are unchanged.
    def D(value):
        return Decimal.from_float(float(value))
    with localcontext() as ctx:
        ctx.prec = 60
        ctx.rounding = ROUND_CEILING
        factors = [D(10.0 * p.area_ha) for p in parameters]
        initial = sum((a * D(float(b)) for a, b in zip(factors, storage)), Decimal(0))
        wilting = sum((a * D(p.root_wilting_mm) for a, p in zip(factors, parameters)), Decimal(0))
        capacity = sum((a * D(p.max_store_mm) for a, p in zip(factors, parameters)), Decimal(0))
        rmax = max(D(p.max_store_mm) - D(p.root_wilting_mm) for p in parameters)
        quota_tolerance = D(1e-10) + D(1e-12) * D(quota)
        quota_upper = D(quota) + quota_tolerance
        q = quota_upper / Decimal(int(substeps_per_day))
        # FP64 aggregation/transition allowance, charged at every substep.
        guard = Decimal(128) * D(np.finfo(np.float64).eps) * max(capacity, quota_upper, Decimal(1))
        with localcontext() as low:
            low.rounding = ROUND_FLOOR
            critical = sum((a * D(p.soil_depth_mm * p.theta_crit)
                            for a, p in zip(factors, parameters)), Decimal(0))
        current = initial + guard
        for kcb in coefficients:
            upper = current
            with localcontext() as low:
                low.rounding = ROUND_FLOOR
                deficit = critical - upper
                margin = deficit - quota_upper
                coefficient = min(Decimal(1), D(float(kcb)) * D(et_min) /
                                  (Decimal(int(substeps_per_day)) * rmax))
                coefficient *= Decimal(1) - Decimal(64) * D(np.finfo(np.float64).eps)
            result["daily_storage_upper_m3"].append(_outward(upper, True))
            result["daily_deficit_lower_m3"].append(_outward(deficit, False))
            result["daily_margin_m3"].append(_outward(margin, False))
            for _ in range(int(substeps_per_day)):
                before = current + q
                with localcontext() as low:
                    low.rounding = ROUND_FLOOR
                    removed = coefficient * max(before - wilting, Decimal(0))
                current = before - removed + guard
        result.update(critical_volume_m3=_outward(critical, False),
                      wilting_volume_m3=_outward(wilting, True),
                      maximum_available_store_depth_mm=_outward(rmax, True),
                      initial_storage_volume_m3=_outward(initial, True),
                      quota_upper_m3=_outward(quota_upper, True),
                      per_substep_rounding_guard_m3=_outward(guard, True),
                      final_storage_upper_m3=_outward(current, True))
    minimum = min(result["daily_margin_m3"])
    result["minimum_margin_m3"] = minimum
    result["certified"] = minimum > 0.
    result["status"] = "certified" if result["certified"] else "not_certified"
    if not result["certified"]:
        result["reasons"].append("the conservative current-deficit lower bound does not exceed daily supply on every day")
    return result
