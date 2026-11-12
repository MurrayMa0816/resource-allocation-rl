"""Detached records for the direct-to-field environment.

The default record contains public observations only. Evaluator truth is an
explicit opt-in and is stored separately; never feed those records to actors.
"""
from __future__ import annotations

from datetime import timedelta
import math

import numpy as np


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def run_configuration(env, *, environment_seed=None, action_seed=None, requested_steps=None):
    """Record the effective controls, without reading random future weather."""
    from . import __version__

    scenario = env.scenario
    controls = {
        "package_version": __version__, "environment_family": "direct_to_field",
        "environment_seed": environment_seed, "action_seed": action_seed,
        "requested_steps": requested_steps, "chapter": env.chapter,
        "horizon_days": env.T, "regions": env.N,
        "decision_horizon_steps": getattr(env, "decision_horizon_steps", env.T),
        "start_date": scenario.start_date.isoformat(),
        "scenario_fingerprint": getattr(env, "_fingerprint", None),
        "field_substeps_per_day": scenario.field_substeps_per_day,
        "maturity_offset_days": getattr(scenario, "maturity_offset_days", None),
        "evaluation_mode": getattr(scenario, "evaluation_mode", "maturity"),
        "resource_accounting": "actual_field_irrigation_m3",
        "custom_input_requirement": "archive custom scenario inputs separately",
    }
    for key, attribute in (("total_budget_m3", "_total_budget"),
                           ("report_delay_days", "_report_delay"),
                           ("yield_lambdas", "_yield_lambdas"),
                           ("weather_mode", "_weather_mode"),
                           ("historical_year", "_historical_year"),
                           ("daily_limit_m3", "_daily_limit_config")):
        if hasattr(env, attribute):
            controls[key] = getattr(env, attribute)
    if hasattr(scenario, "quota_plan"):
        controls["daily_quota_plan_m3"] = scenario.quota_plan
    # Frozen case parameters are independent of reset/weather/action seeds.
    metadata = getattr(scenario, "metadata", {})
    for key in ("case_id", "parameter_profile", "parameter_generation_seed",
                "initial_total_water_m3", "area_weighted_initial_surface_depth_mm",
                "area_weighted_percolation_sat_mm_day", "daily_quota_m3",
                "seasonal_budget_m3", "budget_provenance"):
        if key in metadata:
            controls[key] = metadata[key]
    if hasattr(env, "public_observation_space"):
        from gymnasium.spaces import flatdim
        from .compact_observation import POLICY_SCHEMA
        controls["policy_observation_schema"] = POLICY_SCHEMA
        controls["policy_observation_dimension"] = flatdim(env.observation_space)
        controls["public_display_observation_dimension"] = flatdim(env.public_observation_space)
        controls["shared_model_configuration"] = env.model_config()
    if hasattr(env, "constraint_spec"):
        controls["regional_guarantee"] = env.constraint_spec()
    return _jsonable(controls)


def rollout_metadata(env, reset_info, *, controller_label="unspecified",
                     environment_seed=None, action_seed=None, requested_steps=None,
                     include_truth=False):
    """Metadata makes schematic geometry and privileged access explicit."""
    from .hongmiao import Hongmiao93Env
    hongmiao = isinstance(env, Hongmiao93Env)
    return _jsonable({
        "schema": "direct_irrigation_replay_v5", "chapter": env.chapter,
        "regions": env.N, "region_ids": env.scenario.region_ids,
        "rice_area_ha": env.scenario.areas_ha,
        "geometry_identity": ("hongmiao93_published_chainage_area_schematic" if hongmiao
                              else "generated_area_proportional_schematic_not_geographic"),
        "area_identity": ("published_midseason_rice_area_not_measured_polygon" if hongmiao
                          else "configured_rice_area_not_verified_service_boundary"),
        "controller": controller_label,
        "run_configuration": run_configuration(
            env, environment_seed=environment_seed, action_seed=action_seed,
            requested_steps=requested_steps),
        "reset_info": reset_info,
        "access": "privileged_evaluation_not_policy_input" if include_truth else "public_observation_only",
        "frame_timing": "state at returned daily policy boundary; allocation belongs to decision_date; each step advances one day",
    })


def capture_frame(env, observation, *, reward=None, info=None, allocation=None,
                  include_truth=False):
    """Copy a reset/step boundary without mutating the environment or RNG.

``include_truth=False`` never calls private_diagnostics, including chapter 6.
The allocation is the action that produced this frame, not a proposed next
action. Omitted allocation/reward/date use the environment's public step
context; explicit conflicting values are rejected. A reset frame has zeros.
Lightweight recorder fixtures without step context retain zero defaults.
"""
    info = {} if info is None else info
    policy_observation = None
    if hasattr(env, "public_observation"):
        # The policy interface omits derived display fields. Obtain those only
        # from timestamped public reports, never from evaluator diagnostics.
        public = env.public_observation()
        compact = env.policy_observation(public)
        keys = set(observation)
        if keys == set(compact):
            expected = compact
        elif keys == set(public):
            expected = public
        else:
            raise ValueError("observation fields match neither policy nor public display interface")
        for key, value in expected.items():
            if not np.array_equal(observation[key], value):
                raise ValueError(f"observation {key} disagrees with the current public report")
        policy_observation = compact
        observation = public
    day = int(observation["day_index"])
    context = env.render_context() if hasattr(env, "render_context") else None
    if context is not None:
        if int(context["day"]) != day:
            raise ValueError("observation and current render context belong to different day boundaries")
        # Check public metadata only, never current evaluator water or ET.
        for observed, expected in (("report_day_index", "current_report_day_index"),
                                   ("report_age_days", "report_age_days")):
            if not np.array_equal(observation[observed], context[expected]):
                raise ValueError(f"observation {observed} disagrees with public step context")
        if not math.isclose(float(observation["available_water_m3"]), float(context["next_available_m3"]),
                            rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError("observation available water disagrees with public step context")
        if bool(observation["budget_is_carryable"]) != bool(context["carryable"]):
            raise ValueError("observation budget mode disagrees with public step context")
        if int(np.count_nonzero(observation["settled_mask"])) != int(context["settled_count"]):
            raise ValueError("observation settlement count disagrees with public step context")
        if context["carryable"] and not math.isclose(
                float(observation["remaining_budget_m3"]), float(context["remaining_after_m3"]),
                rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError("observation remaining budget disagrees with public step context")
    if allocation is None:
        allocation = context["executed_action_m3"] if context is not None else np.zeros(env.N)
    allocation = np.asarray(allocation, dtype=float)
    if allocation.shape != (env.N,) or not np.all(np.isfinite(allocation)) or np.any(allocation < 0):
        raise ValueError("allocation must be a finite nonnegative vector of shape (N,)")
    if context is not None and not np.array_equal(allocation, context["executed_action_m3"]):
        raise ValueError("allocation disagrees with the executed public action")
    if reward is None:
        reward = context["reward"] if context is not None else 0.
    reward = float(reward)
    if not math.isfinite(reward):
        raise ValueError("reward must be finite")
    if context is not None:
        if not math.isclose(reward, float(context["reward"]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("reward disagrees with the public outcome evaluation")
        for key, expected in (("date", context["executed_date"]),
                              ("day_index_executed", context.get("executed_day_index", day - 1) if context["has_step"] else None)):
            if key in info and info[key] != expected:
                raise ValueError(f"info {key} disagrees with public step context")
        if "executed_action_m3" in info and not np.array_equal(info["executed_action_m3"], allocation):
            raise ValueError("info executed action disagrees with public step context")
        if "allocated_water_m3" in info and not math.isclose(
                float(info["allocated_water_m3"]), float(context["allocated_m3"]), rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError("info allocated water disagrees with public step context")
    decision_date = context["executed_date"] if context is not None else info.get("date")
    frame = {
        "day": day,
        "date": (env.scenario.start_date + timedelta(days=day)).isoformat(),
        "decision_date": decision_date,
        "reward": reward, "allocation_m3": allocation.copy(),
        "observation": observation,
        "access": "privileged_evaluation_not_policy_input" if include_truth else "public_observation_only",
    }
    if hasattr(env, "episode_summary"):
        frame["return"] = float(env.episode_summary()["return"])
    if context is not None:
        frame["step_context"] = context
    if policy_observation is not None:
        frame["policy_observation"] = policy_observation
    if hasattr(env, "constraint_feedback"):
        feedback = env.constraint_feedback()
        for key, value in feedback.items():
            if key in info and not np.array_equal(info[key], value):
                raise ValueError(f"info {key} disagrees with public constraint feedback")
        frame["constraint_feedback"] = feedback
    if include_truth:
        diagnostics = env.private_diagnostics()
        keys = ("field_storage_mm", "stage_et_mm", "stage_reference_et_mm", "daily_et_mm",
                "daily_drainage_mm", "daily_overflow_mm", "daily_irrigation_mm", "daily_rain_mm",
                "daily_reference_et_mm", "water_mass_residual_m3")
        frame["truth"] = {key: diagnostics[key] for key in keys if key in diagnostics}
        frame["truth"]["flow_scope"] = "executed_day" if day > 0 else "initial_boundary"
        before_storage = None
        if day > 0:
            history = getattr(env, "_history", ())
            start_day = int(context.get("executed_day_index", day - 1)) if context else day - 1
            if len(history) > start_day and "field_storage_mm" in history[start_day]:
                before_storage = np.array(history[start_day]["field_storage_mm"], copy=True)
            elif all(key in diagnostics for key in ("field_storage_mm", "daily_et_mm",
                    "daily_drainage_mm", "daily_overflow_mm", "daily_irrigation_mm", "daily_rain_mm")):
                before_storage = (np.asarray(diagnostics["field_storage_mm"])
                    + np.asarray(diagnostics["daily_et_mm"]) + np.asarray(diagnostics["daily_drainage_mm"])
                    + np.asarray(diagnostics["daily_overflow_mm"]) - np.asarray(diagnostics["daily_irrigation_mm"])
                    - np.asarray(diagnostics["daily_rain_mm"]))
        frame["truth"]["before_field_storage_mm"] = before_storage
    # Conversion copies every array/list. Detached replays cannot mutate state.
    return _jsonable(frame)
