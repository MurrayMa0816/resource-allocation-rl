"""Compact policy-facing reduction of a Hongmiao public observation.

This module only transforms the supplied public report. It has no environment,
weather driver, physical state, or random generator to inspect. Values remain
in their original physical units; no normalization is applied.
The derived conditional weather mean is not a policy input. The observed
previous-day rain/ET0 pair is retained.
Report age is metadata in the explicit public view, not a policy input.
"""
from collections.abc import Mapping
from datetime import date
from functools import lru_cache

import numpy as np
from gymnasium import spaces

from irrigation_ch6._core.direct_execution import FIELD_PARAMETER_NAMES


N_REGIONS = 89
HORIZON_DAYS = 81
POLICY_SCHEMA = "hongmiao89-high-yield-compact-observation-v1"


def make_compact_observation_space(chapter):
    """Return the common 539-value policy input for every chapter."""
    nonnegative = lambda shape: spaces.Box(0., np.inf, shape=shape, dtype=np.float64)
    fields = {
        "day_index": spaces.Box(0, HORIZON_DAYS, shape=(), dtype=np.int64),
        "available_water_m3": nonnegative(()),
        "rice_area_ha": nonnegative((N_REGIONS,)),
        "root_available_water_mm": nonnegative((N_REGIONS,)),
        "percolation_sat_mm_day": nonnegative((N_REGIONS,)),
        "reported_field_storage_mm": nonnegative((N_REGIONS,)),
        "reported_completed_yield_factor": spaces.Box(
            0., 1., shape=(N_REGIONS,), dtype=np.float64),
        "reported_current_stage_et_mm": nonnegative((N_REGIONS,)),
        "reported_current_stage_reference_et_mm": nonnegative(()),
        "previous_weather": nonnegative((2,)),
    }
    return spaces.Dict(fields)


def _physical_array(observation, key, shape):
    raw = np.asarray(observation[key])
    if raw.shape != shape or raw.dtype.kind not in "iuf":
        raise ValueError(f"{key} must be a real array with shape {shape}")
    values = np.array(raw, dtype=np.float64, copy=True)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{key} must contain finite nonnegative values")
    return values


def _integer_array(observation, key, shape):
    raw = np.asarray(observation[key])
    if (raw.shape != shape or raw.dtype.kind not in "iu"
            or np.any(raw > HORIZON_DAYS) or np.any(raw < 0)):
        raise ValueError(f"{key} must be integers from 0 to 81 with shape {shape}")
    return np.array(raw, dtype=np.int64, copy=True)


def _response_intervals(start_date, calendar):
    """Merge coefficient subperiods with the same Jensen stage index.

    Intervals use daily boundaries, with an exclusive end. In particular,
    July 6 changes crop coefficients, not the accumulated yield-response stage.
    Only public fixed calendar data are used; no current environment is read.
    """
    start = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
    rows = []
    for row in calendar:
        stage = row["jensen_index"]
        if stage is None:
            continue
        if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage < 4:
            raise ValueError("calendar must have four ordered Jensen stages")
        rows.append((stage, row["start_mm_dd"], row["end_mm_dd_inclusive"]))
    # Key by values, not by a mutable calendar's identity. Modified supplied
    # calendars are revalidated; cached results cannot alias caller objects.
    return _cached_response_intervals(start, tuple(rows))


@lru_cache(maxsize=64)
def _cached_response_intervals(start, rows):
    periods = [[] for _ in range(4)]
    for stage, first_md, last_md in rows:
        first = date.fromisoformat(f'{start.year}-{first_md}')
        last = date.fromisoformat(f'{start.year}-{last_md}')
        periods[stage].append(((first - start).days, (last - start).days + 1))
    intervals = []
    for entries in periods:
        entries.sort()
        if not entries or any(not 0 <= lo < hi <= HORIZON_DAYS for lo, hi in entries):
            raise ValueError("each response stage must fall within the allocation horizon")
        if any(left[1] != right[0] for left, right in zip(entries, entries[1:])):
            raise ValueError("coefficient subperiods within a response stage must be contiguous")
        intervals.append((entries[0][0], entries[-1][1]))
    if any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("response stages must be contiguous and ordered")
    return tuple(intervals)


def _compress_et(actual, reference, report_day, *, start_date, calendar, yield_lambdas):
    """Keep the Jensen sufficient statistics at the received report boundary.

    Completed stages contribute their fixed product. The active stage keeps
    its numerator AND denominator so future increments can be accumulated.
    Before the response period and after its end, the active sums are zero.
    The complete ledger is retained elsewhere for audit, not reconstructed
    from this reduction. This is not a reward or a future-yield prediction.
    """
    weights = np.asarray(yield_lambdas, dtype=np.float64)
    if weights.shape != (4,) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("yield_lambdas must be four finite nonnegative values")
    # The source ledger already bounds roundoff excess by its reference.
    # A public report must preserve that invariant; do not repair its values.
    if np.any(actual > reference):
        raise ValueError("actual ET must not exceed fully supplied reference ET")
    intervals = _response_intervals(start_date, calendar)
    closed = [k for k, (_, end) in enumerate(intervals) if end <= report_day]
    if np.any(reference[closed] <= 0):
        raise ValueError("completed stages require positive reference ET")
    ratios = np.ones_like(actual)
    np.divide(actual, reference, out=ratios, where=reference > 0)
    completed = np.prod(ratios[:, closed] ** weights[closed], axis=1)
    active = [k for k, (begin, end) in enumerate(intervals) if begin <= report_day < end]
    current_actual = np.zeros(N_REGIONS, dtype=np.float64)
    current_reference = np.array(0., dtype=np.float64)
    if active:
        current_actual = actual[:, active[0]].copy()
        current_reference = np.array(reference[active[0]], dtype=np.float64)
    return completed, current_actual, current_reference


def compact_current_report(report, *, day_index, available_water_m3,
                           rice_area_ha, field_parameters, previous_weather,
                           start_date, calendar, yield_lambdas):
    """Build policy input from an environment's already validated report.

    This internal path reads the same saved public report as the display, not
    live field storage or a future weather forecast. Static parameters and
    timestamps have passed the environment boundary. External report callers
    continue to use the fully validating ``compact_policy_observation``.
    Every returned array is detached from the report and the environment.
    """
    actual = report["stage_et_mm"]
    reference = report["stage_reference_et_mm"]
    if not np.array_equal(reference, np.broadcast_to(reference[0], reference.shape)):
        raise ValueError("compact observations require equal reference-ET rows in all 89 regions")
    completed, current_actual, current_reference = _compress_et(
        actual, reference[0], day_index, start_date=start_date,
        calendar=calendar, yield_lambdas=yield_lambdas)
    parameters = field_parameters
    return {
        "day_index": np.array(day_index, dtype=np.int64),
        "available_water_m3": np.array(available_water_m3, dtype=np.float64),
        "rice_area_ha": rice_area_ha.copy(),
        "root_available_water_mm": (
            parameters[:, FIELD_PARAMETER_NAMES.index("soil_depth_mm")] *
            (parameters[:, FIELD_PARAMETER_NAMES.index("theta_fc")] -
             parameters[:, FIELD_PARAMETER_NAMES.index("theta_wp")])),
        "percolation_sat_mm_day": parameters[:, FIELD_PARAMETER_NAMES.index("percolation_sat_mm_day")].copy(),
        "reported_field_storage_mm": report["field_storage_mm"].copy(),
        "reported_completed_yield_factor": completed,
        "reported_current_stage_et_mm": current_actual,
        "reported_current_stage_reference_et_mm": current_reference,
        "previous_weather": previous_weather.copy(),
    }


def compact_policy_observation(public_observation, *, chapter, start_date,
                               calendar, yield_lambdas):
    """Convert received public reports to detached, unnormalized policy input.

    Compression requires exactly equal reference-ET rows and one shared report
    age. Inconsistent rows are rejected instead of silently choosing a region.
    All four chapters require current reports. Report age is validated
    in the public report but omitted from the policy input in every chapter.
    Both the closed-stage product and active-stage sums use the report date,
    which must equal the current day in every formal chapter.
    """
    if chapter not in (3, 4, 5, 6):
        raise ValueError("chapter must be 3, 4, 5 or 6")
    if not isinstance(public_observation, Mapping):
        raise TypeError("public_observation must be a mapping")
    day = _integer_array(public_observation, "day_index", ())
    ages = _integer_array(public_observation, "report_age_days", (N_REGIONS,))
    if not np.all(ages == ages[0]):
        raise ValueError("compact observations require uniform report_age_days")
    report_days = _integer_array(public_observation, "report_day_index", (N_REGIONS,))
    if not np.all(report_days == day - ages):
        raise ValueError("report timestamps disagree with day_index and report_age_days")
    if ages[0] != 0:
        raise ValueError("all chapters require current reports")
    reference = _physical_array(
        public_observation, "reported_stage_reference_et_mm", (N_REGIONS, 4))
    if not np.array_equal(reference, np.broadcast_to(reference[0], reference.shape)):
        raise ValueError("compact observations require equal reference-ET rows in all 89 regions")
    actual = _physical_array(public_observation, "reported_stage_et_mm", (N_REGIONS, 4))
    completed, current_actual, current_reference = _compress_et(
        actual, reference[0], int(report_days[0]), start_date=start_date,
        calendar=calendar, yield_lambdas=yield_lambdas)
    parameters = _physical_array(
        public_observation, "field_params", (N_REGIONS, len(FIELD_PARAMETER_NAMES)))
    result = {
        "day_index": day,
        "available_water_m3": _physical_array(public_observation, "available_water_m3", ()),
        "rice_area_ha": _physical_array(public_observation, "rice_area_ha", (N_REGIONS,)),
        "root_available_water_mm": (
            parameters[:, FIELD_PARAMETER_NAMES.index("soil_depth_mm")] *
            (parameters[:, FIELD_PARAMETER_NAMES.index("theta_fc")] -
             parameters[:, FIELD_PARAMETER_NAMES.index("theta_wp")])),
        "percolation_sat_mm_day": parameters[:, FIELD_PARAMETER_NAMES.index("percolation_sat_mm_day")].copy(),
        "reported_field_storage_mm": _physical_array(public_observation, "reported_field_storage_mm", (N_REGIONS,)),
        "reported_completed_yield_factor": completed,
        "reported_current_stage_et_mm": current_actual,
        "reported_current_stage_reference_et_mm": current_reference,
        "previous_weather": _physical_array(public_observation, "previous_weather", (2,)),
    }
    return result
