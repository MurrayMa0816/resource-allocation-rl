"""Chapter-six task around the unchanged Hongmiao89-reasonable-depth-v1 carryover case and unchanged physical algorithms.

This is an environment, not a policy, optimizer, or training launcher.
Historical source chapter 4 denotes carryover physics, not this thesis's chapter.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
from datetime import timedelta
import math

import gymnasium as gym
import numpy as np

from .engine import Hongmiao89Engine
from .constants import DEFAULT_BUDGET_M3, DOMAIN, CASE_ID
from ._util import digest, jsonable, package_files, real
from irrigation_ch6.metrics import evaluate_outcomes as _legacy_outcomes


def evaluate_outcomes(regional_performance, rice_area_ha, *, coverage_threshold, coverage_weight):
    """C counts original service units; JA averages yield using fixed public areas."""
    result = _legacy_outcomes(regional_performance, rice_area_ha,
        coverage_threshold=coverage_threshold, coverage_weight=coverage_weight)
    result["JA"] = result["area_weighted_performance"]
    result["joint_score"] = coverage_weight * result["C"] + (1. - coverage_weight) * result["JA"]
    result["learning_objective_order"] = ["C", "JA"]
    return result

TASK_SCHEMA = "hongmiao89-high-yield-chapter6-coverage-performance-v1"
CHECKPOINT_SCHEMA = "hongmiao89-high-yield-chapter6-environment-state-v1"


def _state_digest(payload):
    """Checksum excludes its own field; not a substitute for semantic validation."""
    return digest({k: v for k, v in payload.items() if k != "integrity_sha256"})


class IrrigationEnv(gym.Env):
    """539 physical observation values, 89 actual m3 actions and 81 daily steps.

    Both threshold and preference are explicit research choices. ``joint``
    returns lambda*C+(1-lambda)*JA; ``area_weighted_performance`` returns JA for dual-value
    collectors, which must use the independently provided [C,JA] exactly once.
    Reset and intermediate info are empty. Full audit/display is explicit.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 2}
    application_chapter = 6
    application_domain = DOMAIN

    def __init__(self, *, coverage_threshold, coverage_weight,
                 total_budget_m3=DEFAULT_BUDGET_M3, weather_mode="persistent_drought",
                 historical_year=2022, rain_multiplier=None,
                 profile="heterogeneous", storage_variation=.25,
                 reward_mode="joint", render_mode=None, adapter="unit", spatial_sigma_scale=1.):
        super().__init__()
        if adapter not in ('unit','depth_area','depth_area_physical'):
            raise ValueError('Unknown irrigation adapter')
        self.adapter = adapter
        scale=real(spatial_sigma_scale,'spatial_sigma_scale',maximum=1.)
        if scale not in (1.,.5,.25): raise ValueError('Only declared spatial noise scales1/.5/.25 are allowed')
        self.spatial_sigma_scale=scale
        tau = real(coverage_threshold, "coverage_threshold", maximum=1.)
        weight = real(coverage_weight, "coverage_weight", maximum=1.)
        budget = real(total_budget_m3, "total_budget_m3")
        if reward_mode not in ("joint", "area_weighted_performance"):
            raise ValueError("reward_mode must be 'joint' or 'area_weighted_performance'")
        if render_mode not in (None, "ansi"):
            raise ValueError("unsupported render_mode")
        if weather_mode not in ("persistent_drought", "historical", "markov"):
            raise ValueError("unsupported weather_mode")
        rain = (0. if weather_mode == "persistent_drought" else 1.) if rain_multiplier is None else rain_multiplier
        # The archived physical core and inputs are private and self-contained.
        self._engine = Hongmiao89Engine(profile=profile, historical_year=historical_year,
            daily_quota_m3=budget / 81., storage_variation=storage_variation,
            weather_mode=weather_mode, rain_multiplier=rain, render_mode=None)
        self._coverage_threshold, self._coverage_weight = tau, weight
        self._reward_mode = reward_mode
        self._requested_budget = budget
        self.N, self.T = self._engine.N, self._engine.T
        self.initial_budget_m3 = float(self._engine._total_budget)
        self.region_ids = tuple(self._engine.scenario.region_ids)
        self.action_space = self._engine.action_space
        self.observation_space = self._engine.observation_space
        self.render_mode = render_mode
        self._episode_seed = None
        self._implementation_sha256 = digest(package_files())
        self.physical_fingerprint = self._engine._fingerprint
        self.task_fingerprint = digest({"schema": TASK_SCHEMA, "adapter": self.adapter, "spatial_sigma_scale": self.spatial_sigma_scale,
            "physical_fingerprint": self.physical_fingerprint,
            "implementation_sha256": self._implementation_sha256,
            "requested_budget_m3": budget, "objective": self.objective_spec()})

    @property
    def scenario(self):
        """Frozen physical scenario; do not use future records as policy inputs."""
        return self._engine.scenario

    @property
    def reward_mode(self):
        return self._reward_mode

    def objective_spec(self):
        return {"schema": TASK_SCHEMA, "application_chapter": 6,
            "coverage_threshold": self._coverage_threshold,
            "coverage_weight": self._coverage_weight, "reward_mode": self._reward_mode,
            "objectives": ["coverage_rate", "area_weighted_performance"],
            "objective_order": ["C", "JA"], "threshold_comparison": ">=",
            "coverage_unit": "service_unit_count_not_area",
            "performance_aggregation": "fixed_public_rice_area_weighted_relative_yield",
            "auxiliary_J_aggregation": "equal_service_unit_arithmetic_mean",
            "scalar_reward_formula": ("lambda*C+(1-lambda)*JA" if self.reward_mode == "joint" else "JA"),
            "terminal_feedback": "once_after_81_daily_actions", "gamma": 1.,
            "threshold_identity": "research_setting_not_agronomic_standard",
            "preference_identity": "fixed_research_preference_not_claimed_agronomic_optimum",
            "individual_hard_guarantees": False,
            "dual_value_usage": "use [C,JA]; J is auxiliary only; do not mix joint scalar reward twice"}

    def model_config(self):
        core = self._engine.model_config()
        # Correct historical task semantics while retaining physical provenance.
        core["chapter"] = 6
        core["application_chapter"] = 6
        core["adapter"] = self.adapter
        core["spatial_sigma_scale"] = self.spatial_sigma_scale
        core["physical_core_legacy_task_code"] = 4
        core["environment_class"] = type(self).__name__
        core["research_problem"] = "integrated_irrigation_coverage_and_performance"
        core["regional_guarantee"] = None
        core["information_assumptions"]["chapter6_problem"] = "joint C90 and area-weighted relative yield; no per-unit hard guarantee"
        core["reward_identity"] = self.objective_spec()["scalar_reward_formula"]
        core["objective_spec"] = self.objective_spec()
        core["requested_total_budget_m3"] = self._requested_budget
        core["initial_budget_m3"] = self.initial_budget_m3
        core["task_fingerprint"] = self.task_fingerprint
        core["physical_fingerprint"] = self.physical_fingerprint
        core["implementation_sha256"] = self._implementation_sha256
        core["physical_data_identity"] = "published_areas_transferred_parameters_frozen_research_settings"
        core["policy_input_schema"] = "hongmiao89-high-yield-compact-observation-v1"
        core["policy_info"] = "empty until terminal; audit/display APIs are separate"
        core["weather_interpretation"] = (
            "controlled_no_rain_drought_not_a_historical_drought_record"
            if self._engine._weather_mode == "persistent_drought" else
            "original_archived_weather" if self._engine._weather_mode == "historical" and self._engine.rain_multiplier == 1. else
            "explicitly_transformed_archived_weather" if self._engine._weather_mode == "historical" else
            "seasonal_training_donor_weather")
        return core

    def reset(self, *, seed=None, options=None):
        observation, _ = self._engine.reset(seed=seed, options=options)
        # Core validates before either owner changes its state.
        super().reset(seed=seed)
        # The physical checkpoint owns the public Gym RNG as well.
        self._np_random = self._engine.np_random
        self._episode_seed = int(seed) if seed is not None else None
        return observation, {}

    def step(self, action):
        observation, physical_reward, done, truncated, _ = self._engine.step(action)
        if not done:
            return observation, 0., False, truncated, {}
        metrics = self.terminal_metrics()
        reward = metrics["joint_score"] if self.reward_mode == "joint" else metrics["area_weighted_performance"]
        info = {"terminal_objectives": np.array([metrics["coverage_rate"], metrics["area_weighted_performance"]]),
                "metrics": metrics, "physical_reward": physical_reward,
                "reward_mode": self.reward_mode, "task_fingerprint": self.task_fingerprint}
        return observation, float(reward), done, truncated, info

    def terminal_metrics(self):
        if not self._engine._initialized or self._engine._day != self.T:
            raise RuntimeError("terminal metrics require a completed 81-step episode")
        public = self._engine.public_observation()
        if not np.all(public["settled_mask"]):
            raise ArithmeticError("all terminal outcomes must be evaluated")
        result = evaluate_outcomes(public["settled_relative_yield"], self.scenario.areas_ha,
            coverage_threshold=self._coverage_threshold, coverage_weight=self._coverage_weight)
        result.update(region_ids=list(self.region_ids),
            allocated_water_m3=float(self._engine._spent),
            remaining_budget_m3=float(self._engine._remaining()),
            initial_budget_m3=self.initial_budget_m3)
        return result

    def terminal_objectives(self):
        m = self.terminal_metrics()
        return np.array([m["coverage_rate"], m["area_weighted_performance"]], dtype=np.float64)

    def episode_summary(self):
        result = self._engine.episode_summary()
        result["core_return"] = result["return"]
        result["return"] = 0.
        if result["complete"]:
            m = self.terminal_metrics()
            result["metrics"] = m
            result["return"] = m["joint_score"] if self.reward_mode == "joint" else m["area_weighted_performance"]
        result.update(task_fingerprint=self.task_fingerprint,
            physical_fingerprint=self.physical_fingerprint, application_chapter=6,
            objective_spec=self.objective_spec())
        return result

    def action_constraints(self):
        return self._engine.action_constraints()

    def feasible_allocation(self, weights=None, *, fraction=1., selected_mask=None):
        """Explicit caller convenience; step never repairs submitted actions."""
        return self._engine.feasible_allocation(weights, fraction=fraction, selected_mask=selected_mask)

    def public_observation(self):
        """Detached public display, distinct from raw policy observation."""
        return self._engine.public_observation()

    def water_balance_observer(self):
        """Detached evaluator-only physical account; never used in neural inputs."""
        diagnostics = self._engine.private_diagnostics()
        return {"daily_totals_m3": dict(diagnostics['daily_physics']),
            "et_mm": np.asarray(diagnostics['daily_et_mm'], dtype=np.float64).copy(),
            "percolation_mm": np.asarray(diagnostics['daily_drainage_mm'], dtype=np.float64).copy(),
            "overflow_mm": np.asarray(diagnostics['daily_overflow_mm'], dtype=np.float64).copy()}

    def policy_observation(self, public_observation):
        return self._engine.policy_observation(public_observation)

    def render_context(self):
        result = self._engine.render_context()
        result["physical_reward"] = result["reward"]
        result["physical_cumulative_return"] = result["cumulative_return"]
        scalar = self.episode_summary()["return"]
        result.update(reward=scalar if result["episode_complete"] else 0.,
            task_reward=scalar if result["episode_complete"] else 0., cumulative_return=scalar,
            application_chapter=6, task_fingerprint=self.task_fingerprint,
            objective_spec=self.objective_spec())
        if result["episode_complete"]:
            result["terminal_metrics"] = jsonable(self.terminal_metrics())
        return result

    def render(self):
        if self.render_mode == "ansi":
            s = self.episode_summary()
            return f"Hongmiao89-high-yield chapter6 day={s['days_executed']}/81 reward_mode={self.reward_mode} return={s['return']:.6f} remaining_m3={s['remaining_budget_m3']:.6f}"
        if self.render_mode in ("svg", "rgb_array"):
            from irrigation_ch6._core.direct_visualization import render_observation
            return render_observation(self.public_observation(), self.region_ids,
                chapter=6, last_action=self.render_context()["executed_action_m3"],
                mode=self.render_mode, context=self.render_context())
        return None

    def get_state(self):
        payload = {"schema": CHECKPOINT_SCHEMA, "task_fingerprint": self.task_fingerprint,
            "objective_spec": self.objective_spec(), "episode_seed": self._episode_seed,
            "engine_state": self._engine.get_state()}
        payload["integrity_sha256"] = _state_digest(payload)
        return jsonable(payload)

    def _validate_checkpoint_weather(self, state):
        """Close a known old-core gap before atomic restore; no RNG draws."""
        day = state.get("day")
        if isinstance(day, (bool, np.bool_)) or not isinstance(day, (int, np.integer)) or not 0 <= day <= self.T:
            raise ValueError("invalid checkpoint day")
        previous = np.asarray(state.get("previous_weather"))
        if previous.shape != (2,) or previous.dtype.kind not in "iuf" or not np.isfinite(previous).all() or (previous < 0).any():
            raise ValueError("invalid checkpoint previous weather")
        driver = self._engine._weather
        if day > 0:
            diagnostics = state.get("diagnostics", {})
            if not isinstance(diagnostics, Mapping):
                raise ValueError("checkpoint physical diagnostics must be a mapping")
            for key, value in (("daily_rain_mm", previous[0]), ("daily_eto_mm", previous[1])):
                arr = np.asarray(diagnostics.get(key))
                if arr.shape != (self.N,) or not np.array_equal(arr, np.full(self.N, value)):
                    raise ValueError("checkpoint previous weather disagrees with executed physical forcing")
        if driver.mode == "historical":
            allowed = [driver._scaled_vector(driver._historical_record(int(day)-1).vector)]
        else:
            keys = (driver._initial_dates if day == 0 else
                    set(driver._donor_pools[int(day)-1, 0]) | set(driver._donor_pools[int(day)-1, 1]))
            allowed = [driver._scaled_vector(driver._records[key].vector) for key in keys]
        if not any(np.array_equal(previous, x) for x in allowed):
            raise ValueError("checkpoint previous weather is outside the documented forcing support")

    def set_state(self, checkpoint):
        if not isinstance(checkpoint, Mapping):
            raise ValueError("checkpoint must be a mapping")
        state = copy.deepcopy(checkpoint)
        expected = {"schema", "task_fingerprint", "objective_spec", "episode_seed", "engine_state", "integrity_sha256"}
        if set(state) != expected:
            raise ValueError("checkpoint fields do not match the application contract")
        if (state["schema"] != CHECKPOINT_SCHEMA or state["task_fingerprint"] != self.task_fingerprint or
                state["objective_spec"] != self.objective_spec()):
            raise ValueError("checkpoint belongs to a different application task")
        if state["integrity_sha256"] != _state_digest(state):
            raise ValueError("checkpoint integrity checksum mismatch")
        seed = state["episode_seed"]
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("invalid checkpoint episode seed")
        if not isinstance(state["engine_state"], Mapping):
            raise ValueError("checkpoint lacks a physical state")
        self._validate_checkpoint_weather(state["engine_state"])
        # Validate against a candidate, including the final physical transition,
        # so any rejected cross-field inconsistency leaves the live RNG intact.
        candidate = copy.copy(self._engine)
        candidate.action_space = copy.copy(self._engine.action_space)
        candidate.action_space._env = candidate
        candidate.set_state(state["engine_state"])
        if candidate._day:
            before = candidate._history[-2]
            day = candidate.scenario.start_date + timedelta(days=candidate._day - 1)
            replay = candidate._advance_physics_day(
                field_params=candidate.scenario.field_params,
                storage_mm=before["field_storage_mm"],
                heading_dates=candidate.scenario.heading_dates, day=day,
                maturity_offset_days=candidate.scenario.maturity_offset_days,
                action_m3=candidate._actions[-1],
                rain_mm=np.full(self.N, candidate._previous_weather[0]),
                eto_mm_day=np.full(self.N, candidate._previous_weather[1]),
                substeps_per_day=candidate.scenario.field_substeps_per_day,
                crop_profiles=candidate._crop_profiles(day))
            for key, expected_value in replay.items():
                actual = np.asarray(candidate._diagnostics.get(key))
                if actual.shape != np.asarray(expected_value).shape or not np.allclose(
                        actual, expected_value, atol=1e-12, rtol=1e-13):
                    raise ValueError(f"checkpoint physical transition disagrees with executed weather: {key}")
            after = candidate._history[-1]
            for key, daily in (("stage_et_mm", "daily_et_mm"),
                               ("stage_reference_et_mm", "daily_reference_et_mm")):
                expected = before[key].copy()
                for i, stage in enumerate(replay["outcome_stage_index"]):
                    if stage >= 0:
                        amount = replay[daily][i]
                        if daily == "daily_et_mm":
                            amount = min(amount, replay["daily_reference_et_mm"][i])
                        expected[i, stage] += amount
                if not np.allclose(after[key], expected, atol=1e-11, rtol=1e-13):
                    raise ValueError("checkpoint last stage increment disagrees with physical transition")
        # Retain the caller-visible action-space object, with restored RNG.
        old_space = self.action_space
        old_space._env = candidate
        old_space._np_random = candidate.action_space._np_random
        candidate.action_space = old_space
        self._engine = candidate
        self._episode_seed = seed
        self._np_random = self._engine.np_random

    def close(self):
        self._engine.close()


def make_env(**kwargs):
    """Create the final application; threshold and preference must be explicit."""
    return IrrigationEnv(**kwargs)

