"""Validated direct-field inputs and the packaged Hongmiao93 scenario loader."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, timedelta
import math
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np

from .field import FieldParams
from .weather import WeatherDay, _readonly_vector


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class DirectScenario:
    """One representative crop per region and actual field-inlet quotas (m3).

    All supplied containers are copied. Vectors cannot be written in-place.
    Weather tables may cover a longer horizon than this scenario; every used
    day and conditional forecast must be present. No network is constructed.
    """

    region_ids: tuple[str, ...]
    field_params: tuple[FieldParams, ...]
    initial_storage_mm: np.ndarray
    heading_dates: tuple[date, ...]
    quota_plan: np.ndarray
    start_date: date
    weather_records: Mapping[str, WeatherDay]
    donor_pools: Mapping[str, tuple[str, ...]]
    forecast_table: Mapping[tuple[int, int], tuple[float, float, float]]
    weather_contract: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    maturity_offset_days: int = 35
    field_substeps_per_day: int = 96
    evaluation_mode: str = "maturity"

    def __post_init__(self) -> None:
        if type(self.start_date) is not date:
            raise ValueError("start_date must be a calendar date without a time component")
        ids = tuple(self.region_ids)
        if not ids or any(not isinstance(x, str) or not x.strip() for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("region_ids must be unique nonempty strings")
        params = tuple(self.field_params)
        if len(params) != len(ids) or any(not isinstance(p, FieldParams) for p in params):
            raise ValueError("one FieldParams is required for each region")
        # Frozen FieldParams validates scalars itself; enforce its public type
        # contract again rather than trusting a hand-mutated instance.
        for p in params:
            for name, value in vars(p).items():
                _real(value, f"field {name}")
            p.__post_init__()
        initial = _readonly_vector(self.initial_storage_mm, len(ids), "initial storage")
        if np.any(initial < 0) or any(x > p.max_store_mm for x, p in zip(initial, params)):
            raise ValueError("initial storage must lie within field capacity bounds")
        raw = np.asarray(self.quota_plan)
        if raw.ndim != 1 or raw.size == 0:
            raise ValueError("quota plan must be a nonempty one-dimensional vector")
        quota = _readonly_vector(self.quota_plan, raw.size, "quota plan")
        if np.any(quota < 0) or not math.isfinite(math.fsum(quota)):
            raise ValueError("quota plan must have a finite nonnegative total")
        # Fixed vegetative/booting/flowering/grain-fill stages end at day25
        # after heading; this kernel requires a positive ripening interval.
        offset = _integer(self.maturity_offset_days, "maturity_offset_days", minimum=26)
        substeps = _integer(self.field_substeps_per_day, "field_substeps_per_day")
        headings = tuple(self.heading_dates)
        if len(headings) != len(ids) or any(type(x) is not date for x in headings):
            raise ValueError("one calendar heading date is required for each region")
        final_boundary = self.start_date + timedelta(days=len(quota))
        if self.evaluation_mode not in ("maturity", "terminal_projection"):
            raise ValueError("evaluation_mode must be maturity or terminal_projection")
        for h in headings:
            maturity = h + timedelta(days=offset)
            if not self.start_date < maturity:
                raise ValueError("every crop must mature after start")
            if self.evaluation_mode == "maturity" and maturity > final_boundary:
                raise ValueError("every crop must mature after start and within the evaluation horizon")
        records = dict(self.weather_records)
        if not records or any(not isinstance(w, WeatherDay) or k != w.date.isoformat()
                              for k, w in records.items()):
            raise ValueError("weather records must map occurrence dates to WeatherDay")
        # Reconstruct WeatherDay to copy its vector and restore readonly flags.
        records = {key: WeatherDay(w.date, w.vector, w.eto_mm_day) for key, w in records.items()}
        for t in range(-1, len(quota)):
            if (self.start_date + timedelta(days=t)).isoformat() not in records:
                raise ValueError("historical weather is missing an initial or evaluation day")
        contract = deepcopy(dict(self.weather_contract))
        years = contract.get("train_donor_years", ())
        if not years or any(isinstance(y, bool) or not isinstance(y, int) for y in years):
            raise ValueError("weather contract must declare integer training years")
        initial_dates = contract.get("initial_donor_dates", ())
        if not initial_dates or any(d not in records or records[d].date.year not in years for d in initial_dates):
            raise ValueError("initial donors must exist and belong to declared training years")
        grid = contract.get("common_driver", {})
        latitude = _real(grid.get("latitude"), "weather grid latitude")
        elevation = _real(grid.get("grid_elevation_m"), "weather grid elevation")
        if not -90 < latitude < 90 or .75 + 2e-5 * elevation <= 0:
            raise ValueError("weather grid is outside the daily ET0 geometry domain")
        pools = {key: tuple(values) for key, values in self.donor_pools.items()}
        forecasts = {key: tuple(values) for key, values in self.forecast_table.items()}
        for t in range(len(quota)):
            for previous_class in range(4):
                candidates = pools.get(f"{t}:{previous_class}", ())
                if not candidates or any(d not in records or records[d].date.year not in years for d in candidates):
                    raise ValueError("daily donor pools must contain only declared training records")
                f = _readonly_vector(forecasts.get((t, previous_class), ()), 3, "forecast")
                if np.any(f[:2] < 0) or not 0 <= f[2] <= 1:
                    raise ValueError("forecast rain/ET0 must be nonnegative; hot probability in [0,1]")
        object.__setattr__(self, "region_ids", ids)
        object.__setattr__(self, "field_params", params)
        object.__setattr__(self, "initial_storage_mm", initial)
        object.__setattr__(self, "heading_dates", headings)
        object.__setattr__(self, "quota_plan", quota)
        object.__setattr__(self, "maturity_offset_days", offset)
        object.__setattr__(self, "field_substeps_per_day", substeps)
        object.__setattr__(self, "weather_records", records)
        object.__setattr__(self, "donor_pools", pools)
        object.__setattr__(self, "forecast_table", forecasts)
        object.__setattr__(self, "weather_contract", contract)
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    @property
    def N(self) -> int:
        return len(self.region_ids)

    @property
    def T(self) -> int:
        return len(self.quota_plan)

    @property
    def C(self) -> int:
        return 1

    @property
    def areas_ha(self) -> np.ndarray:
        return _readonly_vector([p.area_ha for p in self.field_params], self.N, "areas")

    @property
    def end_date(self) -> date:
        return self.start_date + timedelta(days=self.T - 1)

    @property
    def maturity_dates(self) -> tuple[date, ...]:
        return tuple(h + timedelta(days=self.maturity_offset_days) for h in self.heading_dates)



def load_default_direct_scenario() -> DirectScenario:
    """Load the published Hongmiao93 areas and packaged reference parameters.

    Construct the public environment through ``make_env`` so the transferred
    crop calendar and terminal simulated-yield evaluation are applied as well.
    """
    from .hongmiao import load_hongmiao_scenario
    return load_hongmiao_scenario()[0]


load_default_scenario = load_default_direct_scenario
