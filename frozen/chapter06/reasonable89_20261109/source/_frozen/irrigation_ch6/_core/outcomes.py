"""Stage ET accounting and once-only window-conditioned relative yield proxies.

This module uses the four-stage Jensen response transferred from Chen et al.
(2024), River, DOI 10.1002/rvr2.96. It is not a locally calibrated Hongmiao yield
model. Omission of ripening and the absence of a direct heat-yield response
are explicit simplifications requiring local validation. The Hongmiao caller
requires all 37/16/10/9 days of the four stages before settling any outcome.

The caller supplies *actual total ET*, Ta + Ea, and same-weather fully supplied
reference total ET. Neither ET0, irrigation volume, nor transpiration alone is
a replacement for either quantity. Stage accumulators cover only simulated
days. Completely unrepresented stages have ratio one. Partially represented
stages use window-only ET ratios as an explicit truncated-stage response
approximation. Omitting an earlier part of a stage is NOT mathematically
equivalent to assuming that part was fully supplied: that alternative requires
adding its reference ET to both stage totals. No such unknown prehistory is
invented here. A stage with zero accumulated reference ET has ratio one.

Maturity settlement and terminal projection are distinct explicit operations.
``settle`` requires calendar maturity; ``evaluate_terminal`` instead reports
a model projection at the end of the allocation window without claiming crop
maturity. With gamma=1, rewards sum to the
area-weighted, window-conditioned relative yield proxy once every represented
cohort has matured. This proxy does not reconstruct full-season actual yield.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


STAGES = ("tillering", "booting", "heading", "milky")
DEFAULT_LAMBDAS = (0.4782, 0.5236, 0.5923, 0.1504)
REFERENCE_PROFILES = {
    "zhanghe_river2024": DEFAULT_LAMBDAS,
}
REFERENCE_DOI = "10.1002/rvr2.96"
_ATOL = 1e-10
_RTOL = 1e-8


def yield_lambdas_for_profile(name):
    """Return published coefficients for an explicitly named external profile.

    These coefficients are transferred from another part of Zhanghe irrigation
    district; they are not measurements or calibration of all Hongmiao units.
    """
    if not isinstance(name, str) or name not in REFERENCE_PROFILES:
        raise ValueError(f"unknown yield profile; choose one of {tuple(REFERENCE_PROFILES)}")
    return tuple(REFERENCE_PROFILES[name])


def _numeric_array(name, value, shape=None, *, positive=False):
    raw = np.asarray(value)
    bool_elements = any(isinstance(item, (bool, np.bool_))
                        for item in np.asarray(value, dtype=object).flat)
    if raw.dtype.kind not in "iuf" or bool_elements:
        raise ValueError(f"{name} must contain real numeric values, not booleans")
    result = raw.astype(np.float64, copy=True)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    if np.any(result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def _check_et(actual, reference):
    if np.any(actual - reference > _ATOL + _RTOL * reference):
        raise ValueError("actual total ET exceeds fully supplied reference ET")
    # Only arithmetic excess within the declared tolerance is normalized.
    return np.minimum(actual, reference)


class OutcomeLedger:
    """Accumulate arrays indexed by region, cohort and four response stages.

    ``areas_ha`` has shape (N, C) with positive cohort areas. ``lambdas`` can
    have shape (4,) or (N, C, 4). All cohorts retain their initial area weights
    throughout the episode, including cohorts that have not yet matured.
    Invalid inputs leave all existing accumulators and settlement flags intact.
    Properties return copies so callers cannot mutate the ledger accidentally.
    """

    def __init__(self, areas_ha, lambdas=DEFAULT_LAMBDAS):
        areas = _numeric_array("areas_ha", areas_ha, positive=True)
        if areas.ndim != 2 or not areas.size or any(size == 0 for size in areas.shape):
            raise ValueError("areas_ha must be a nonempty (N, C) matrix")
        try:
            total_area = math.fsum(areas.flat)
        except OverflowError as exc:
            raise ValueError("total represented area must be finite") from exc
        if not math.isfinite(total_area):
            raise ValueError("total represented area must be finite")
        coefficients = _numeric_array("lambdas", lambdas)
        if coefficients.shape == (4,):
            coefficients = np.broadcast_to(coefficients, areas.shape + (4,)).copy()
        if coefficients.shape != areas.shape + (4,):
            raise ValueError("lambdas must have shape (4,) or (N, C, 4)")
        self._areas = areas
        self._lambdas = coefficients
        self._weights = areas / total_area
        self._actual = np.zeros(areas.shape + (4,), dtype=np.float64)
        self._reference = np.zeros_like(self._actual)
        self._counts = np.zeros_like(self._actual, dtype=np.int64)
        self._settled = np.zeros(areas.shape, dtype=bool)
        self._settled_values = np.zeros(areas.shape, dtype=np.float64)

    @property
    def shape(self):
        return self._areas.shape

    @property
    def areas_ha(self):
        return self._areas.copy()

    @property
    def lambdas(self):
        return self._lambdas.copy()

    @property
    def weights(self):
        return self._weights.copy()

    @property
    def actual_et_mm(self):
        """Cumulative actual Ta + Ea, shape (N, C, 4)."""
        return self._actual.copy()

    @property
    def reference_et_mm(self):
        """Cumulative same-weather fully supplied total ET, shape (N, C, 4)."""
        return self._reference.copy()

    @property
    def stage_observations(self):
        return self._counts.copy()

    @property
    def settled_mask(self):
        """Whether an outcome was emitted, not a biological maturity flag."""
        return self._settled.copy()

    @property
    def settled_relative_yield(self):
        """Settled values; zero means unreported where settled_mask is false."""
        return self._settled_values.copy()

    @property
    def settled_return(self):
        """Sum of original rewards emitted so far, not a terminal forecast."""
        return math.fsum((self._weights * self._settled_values).flat)

    @property
    def all_settled(self):
        return bool(np.all(self._settled))

    def accumulate(self, stage_index, actual_et_mm, reference_et_mm):
        """Add one day's cohort ET; indices -1 omit ripening/mature stages.

        The caller must add only once for each physical day. This pure ledger
        has no calendar and cannot detect a duplicated day. A settled cohort
        must have stage index -1; modifying its completed history is rejected.
        All supplied ET values, including ignored stages, must be finite and
        nonnegative; ET/reference consistency applies to included stages.
        """
        stage = np.asarray(stage_index)
        if stage.shape != self.shape or stage.dtype.kind not in "iu":
            raise ValueError("stage_index must be an integer (N, C) matrix")
        if np.any(stage < -1) or np.any(stage > 3):
            raise ValueError("stage_index entries must be -1 or 0 through 3")
        actual = _numeric_array("actual_et_mm", actual_et_mm, self.shape)
        reference = _numeric_array("reference_et_mm", reference_et_mm, self.shape)
        active = stage >= 0
        if np.any(active & self._settled):
            raise ValueError("cannot accumulate a previously settled cohort")
        selected_actual = _check_et(actual[active], reference[active])
        rows, cohorts = np.nonzero(active)
        index = (rows, cohorts, stage[active])
        updated_actual = self._actual.copy()
        updated_reference = self._reference.copy()
        updated_counts = self._counts.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            updated_actual[index] += selected_actual
            updated_reference[index] += reference[active]
        if not np.all(np.isfinite(updated_actual)) or not np.all(np.isfinite(updated_reference)):
            raise ValueError("cumulative stage ET must remain finite")
        if np.any(updated_counts[index] == np.iinfo(np.int64).max):
            raise ValueError("stage observation count overflow")
        updated_counts[index] += 1
        self._actual = _check_et(updated_actual, updated_reference)
        self._reference = updated_reference
        self._counts = updated_counts

    @staticmethod
    def _ratios(actual, reference):
        ratios = np.ones_like(actual)
        np.divide(actual, reference, out=ratios, where=reference > 0)
        return ratios

    def preview_relative_yield(self):
        """Window-conditioned response, NOT a forecast or matured outcome.

        Completely unrepresented stages have ratio one; partially represented
        stages use the window-only ET ratio, without fabricated prehistory.
        These values may change until final evaluation and are not measured crop yield.
        """
        return np.prod(self._ratios(self._actual, self._reference) ** self._lambdas, axis=2)

    def settle(self, mature_mask):
        """Return (reward, records) for newly mature cohorts, exactly once.

        The boolean (N, C) mask must represent actual completion according to
        the caller's declared phenology model. Passing the episode terminal
        flag as if it meant crop maturity violates this API's contract.
        """
        mature = np.asarray(mature_mask)
        if mature.shape != self.shape or mature.dtype.kind != "b":
            raise ValueError("mature_mask must be a boolean (N, C) matrix")
        return self._emit_outcomes(mature, "maturity")

    def evaluate_terminal(self, evaluation_mask):
        """Emit a terminal model projection, without asserting crop maturity.

        Callers must explicitly declare this evaluation mode and validate the
        required stage history. This operation never alters phenology dates.
        Like maturity settlement, each cohort can emit its outcome only once.
        """
        selected = np.asarray(evaluation_mask)
        if selected.shape != self.shape or selected.dtype.kind != "b":
            raise ValueError("evaluation_mask must be a boolean (N, C) matrix")
        return self._emit_outcomes(selected, "terminal_projection")

    def _emit_outcomes(self, selected, evaluation_mode):
        newly = selected & ~self._settled
        if not np.any(newly):
            return 0.0, []
        values = self.preview_relative_yield()
        ratios = self._ratios(self._actual, self._reference)
        records = []
        contributions = []
        for row, cohort in zip(*np.nonzero(newly)):
            contribution = float(self._weights[row, cohort] * values[row, cohort])
            contributions.append(contribution)
            records.append({
                "region_index": int(row), "cohort_index": int(cohort),
                "area_ha": float(self._areas[row, cohort]),
                "area_weight": float(self._weights[row, cohort]),
                "relative_yield": float(values[row, cohort]),
                "reward_contribution": contribution,
                "stage_actual_et_mm": self._actual[row, cohort].tolist(),
                "stage_reference_et_mm": self._reference[row, cohort].tolist(),
                "stage_et_ratios": ratios[row, cohort].tolist(),
                "stage_observations": self._counts[row, cohort].tolist(),
                "unobserved_stages_assumed_no_deficit": [
                    name for k, name in enumerate(STAGES) if self._counts[row, cohort, k] == 0
                ],
                "outcome_kind": "research_model_relative_yield_not_observed_harvest",
                "response_scope": "window_conditioned_relative_yield_proxy",
                "partial_stage_rule": "window_only_ET_ratio_truncated_stage_approximation",
                "maturity_source": "caller_declared_phenology_not_inferred_from_ET",
                "evaluation_mode": evaluation_mode,
                "evaluated_at_crop_maturity": evaluation_mode == "maturity",
            })
        reward = math.fsum(contributions)
        self._settled_values[newly] = values[newly]
        self._settled |= newly
        return reward, records

    def snapshot(self):
        """Return an independent JSON-serializable checkpoint."""
        return {
            "schema_version": 1, "stages": list(STAGES),
            "areas_ha": self._areas.tolist(), "lambdas": self._lambdas.tolist(),
            "actual_et_mm": self._actual.tolist(),
            "reference_et_mm": self._reference.tolist(),
            "stage_observations": self._counts.tolist(),
            "settled_mask": self._settled.tolist(),
            "settled_relative_yield": self._settled_values.tolist(),
        }

    def restore(self, snapshot):
        """Atomically restore a checkpoint with identical areas and coefficients."""
        if not isinstance(snapshot, Mapping):
            raise ValueError("snapshot must be a mapping")
        required = set(self.snapshot())
        if set(snapshot) != required:
            raise ValueError("snapshot keys do not match the outcome schema")
        if type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != 1:
            raise ValueError("unsupported outcome snapshot version")
        if snapshot["stages"] != list(STAGES):
            raise ValueError("snapshot stage order differs")
        areas = _numeric_array("snapshot areas", snapshot["areas_ha"], self.shape, positive=True)
        coefficients = _numeric_array("snapshot lambdas", snapshot["lambdas"], self._lambdas.shape)
        if not np.array_equal(areas, self._areas) or not np.array_equal(coefficients, self._lambdas):
            raise ValueError("snapshot areas or response coefficients differ")
        actual = _numeric_array("snapshot actual ET", snapshot["actual_et_mm"], self._actual.shape)
        reference = _numeric_array("snapshot reference ET", snapshot["reference_et_mm"], self._reference.shape)
        actual = _check_et(actual, reference)
        counts = np.asarray(snapshot["stage_observations"])
        if counts.shape != self._counts.shape or counts.dtype.kind not in "iu" or np.any(counts < 0):
            raise ValueError("snapshot stage observations must be nonnegative integers")
        if np.any(counts > np.iinfo(np.int64).max):
            raise ValueError("snapshot stage observations overflow")
        counts = counts.astype(np.int64, copy=True)
        if np.any((counts == 0) & ((actual > 0) | (reference > 0))):
            raise ValueError("unobserved stages cannot contain accumulated ET")
        settled = np.asarray(snapshot["settled_mask"])
        if settled.shape != self.shape or settled.dtype.kind != "b":
            raise ValueError("snapshot settled mask must be a boolean (N, C) matrix")
        values = _numeric_array("snapshot settled values", snapshot["settled_relative_yield"], self.shape)
        expected = np.prod(self._ratios(actual, reference) ** coefficients, axis=2)
        if np.any(values[~settled] != 0) or not np.allclose(values[settled], expected[settled], rtol=_RTOL, atol=_ATOL):
            raise ValueError("snapshot settled values disagree with their stage history")
        if np.any(values > 1):
            raise ValueError("settled relative yield cannot exceed one in this response model")
        self._actual, self._reference, self._counts = actual, reference, counts
        self._settled, self._settled_values = settled.copy(), values
