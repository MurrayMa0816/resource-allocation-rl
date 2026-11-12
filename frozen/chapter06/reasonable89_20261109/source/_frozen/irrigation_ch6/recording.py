"""Complete, detached Chapter 6 episodes and offline publication checks.

Controllers receive copies of the 563-value policy observation only.  Detailed
daily fluxes are evaluator records, not an additional policy information
channel.  The original physical reward remains available for the frozen-core
replay; task rewards and terminal C/J objectives are recorded separately.
"""
from __future__ import annotations

import copy
import csv
from collections.abc import Mapping
from datetime import date, timedelta
import hashlib
import html
import io
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re

import numpy as np

from ._core.direct_recording import capture_frame, rollout_metadata
from ._core.direct_visualization import write_rollout_html


EPISODE_SCHEMA = "irrigation-ch6-complete-episode-v1"
MANIFEST_SCHEMA = "irrigation-ch6-episode-files-v1"
N_REGIONS, HORIZON = 93, 81
RESPONSE_ENDS = (46, 62, 72, 81)
YIELD_COEFFICIENTS = (0.4782, 0.5236, 0.5923, 0.1504)
OUTPUT_FILES = frozenset({"episode.json", "daily.csv", "regional_daily.csv",
                          "terminal.csv", "replay.html"})


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _canonical(value):
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _real(value, name, *, lower=None, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (lower is not None and result < lower) or (
            upper is not None and result > upper):
        raise ValueError(f"{name} is outside its finite allowed range")
    return result


def _integer(value, name, *, lower=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < lower:
        raise ValueError(f"{name} must be an integer >= {lower}")
    return int(value)


def _array(value, shape, name, *, nonnegative=True):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != shape:
        raise ValueError(f"{name} must be a real array with shape {shape}")
    out = raw.astype(np.float64, copy=False)
    if not np.isfinite(out).all() or (nonnegative and np.any(out < 0)):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return out


def _close(actual, expected, name, *, atol=1e-9, rtol=2e-12):
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(f"{name} disagrees with the recorded physical/task process")


def _identity(metadata):
    return {k: metadata[k] for k in (
        "task_fingerprint", "physical_fingerprint", "model_config_sha256",
        "objective_spec", "region_ids", "rice_area_ha", "initial_budget_m3",
        "horizon_days", "policy_observation_schema", "controller_provenance",
        "controller_provenance_status")}


def _controller_provenance(value):
    """Check a caller declaration's format, not model bytes or training claims."""
    if value is None:
        return {"status": "not_supplied"}, "not_supplied"
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ValueError("controller_provenance must be a JSON mapping with string keys")
    result = _plain(dict(value))
    try:
        _canonical(result)
    except (TypeError, ValueError) as exc:
        raise ValueError("controller_provenance must contain finite JSON values") from exc
    for key, declared in result.items():
        if key.endswith("_sha256") and (
                not isinstance(declared, str) or re.fullmatch(r"[0-9a-fA-F]{64}", declared) is None):
            raise ValueError(f"controller_provenance {key} must contain exactly 64 hex characters")
    if "execution_mode" in result and (
            not isinstance(result["execution_mode"], str) or not result["execution_mode"].strip()):
        raise ValueError("controller_provenance execution_mode must be a nonempty string")
    return result, "caller_declared_not_independently_verified"


def _seal_record(record):
    """Seal newly captured data; hashes detect changes, not author identity."""
    previous = None
    for frame in record["frames"]:
        frame.pop("frame_sha256", None)
        frame["previous_frame_sha256"] = previous
        frame["frame_sha256"] = _sha(frame)
        previous = frame["frame_sha256"]
    record.pop("integrity", None)
    record["integrity"] = {"algorithm": "sha256-canonical-json",
                           "record_sha256": _sha(record),
                           "last_frame_sha256": previous}
    return record


def run_episode(env, controller, *, environment_seed, controller_label,
                controller_seed=None, controller_provenance=None):
    """Execute exactly one full episode without implementing any training.

    ``controller_seed`` documents a seed managed by the caller/controller; this
    function never changes global NumPy/Python random state.  The controller
    must be a callable ``controller(copied_policy_observation) -> m3[93]``.
    The environment is left at its terminal state and is owned by the caller.
    ``controller_provenance`` optionally records caller-declared policy/model
    SHA256 and execution mode. No model is loaded or independently verified;
    the environment model-config hash is never a policy-model hash.
    """
    environment_seed = _integer(environment_seed, "environment_seed")
    if controller_seed is not None:
        controller_seed = _integer(controller_seed, "controller_seed")
    if not callable(controller):
        raise TypeError("controller must be callable")
    if not isinstance(controller_label, str) or not controller_label.strip():
        raise ValueError("controller_label must be a nonempty description")
    provenance, provenance_status = _controller_provenance(controller_provenance)
    if (env.N, env.T) != (N_REGIONS, HORIZON):
        raise ValueError("complete Chapter 6 recording requires 93 regions and 81 decisions")
    obs, reset_info = env.reset(seed=environment_seed)
    engine = env._engine
    model_config = _plain(env.model_config())
    objective = _plain(env.objective_spec())
    metadata = rollout_metadata(
        engine, reset_info, controller_label=controller_label,
        environment_seed=environment_seed, action_seed=controller_seed,
        requested_steps=HORIZON, include_truth=True)
    metadata.update(
        application_chapter=6, physical_engine_chapter=engine.chapter,
        task_fingerprint=env.task_fingerprint,
        physical_fingerprint=env.physical_fingerprint,
        model_config=model_config, model_config_sha256=_sha(model_config),
        objective_spec=objective, initial_budget_m3=float(env.initial_budget_m3),
        horizon_days=HORIZON, controller_label=controller_label,
        controller_seed=controller_seed,
        controller_provenance=provenance,
        controller_provenance_status=provenance_status,
        controller_seed_management="caller_owned; recorder does not seed global RNG",
        training_performed_by_recorder=False, policy_quality_claim=False,
        physical_reward_semantics="area_weighted_model_relative_performance",
        task_reward_semantics=objective["reward_mode"],
        diagnostic_access="evaluator_only; never supplied to controller",
        policy_observation_schema=metadata["run_configuration"]["policy_observation_schema"],
    )
    metadata["identity_sha256"] = _sha(_identity(metadata))
    frames = []

    def append(observation, task_reward, terminated, info):
        frame = capture_frame(engine, observation, include_truth=True)
        # The archived recorder predates explicit Ta/Ea/ET0 output.  Keep the
        # core unchanged and add those evaluator-only diagnostics here.
        diagnostics = engine.private_diagnostics()
        for key in ("daily_transpiration_mm", "daily_evaporation_mm", "daily_eto_mm"):
            value = (diagnostics[key] if int(observation["day_index"]) > 0
                     else np.zeros(N_REGIONS))
            frame["truth"][key] = _plain(np.array(value, copy=True))
        frame.update(
            task_reward=float(task_reward), physical_reward=float(frame["reward"]),
            terminal_objectives=_plain(info.get("terminal_objectives")) if terminated else None,
            metrics=_plain(env.terminal_metrics()) if terminated else None,
            task_fingerprint=metadata["task_fingerprint"],
            model_config_sha256=metadata["model_config_sha256"],
            identity_sha256=metadata["identity_sha256"],
        )
        frames.append(frame)

    append(obs, 0., False, {})
    for day in range(HORIZON):
        # A controller cannot mutate the environment by writing to its input.
        action = controller(copy.deepcopy(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        if truncated or bool(terminated) != (day == HORIZON - 1):
            raise ValueError("environment did not complete exactly 81 physical daily decisions")
        append(obs, reward, terminated, info)
    record = _plain({"schema": EPISODE_SCHEMA, "metadata": metadata,
                     "frames": frames, "summary": env.episode_summary()})
    _seal_record(record)
    validate_episode(record)
    return record


def _check_metrics(metrics, *, yields, areas, threshold, weight, allocated, remaining):
    if not isinstance(metrics, dict):
        raise ValueError("terminal metrics must be a mapping")
    covered = int(np.count_nonzero(yields >= threshold))
    expected = {
        "coverage_rate": covered / N_REGIONS,
        "mean_performance": float(np.mean(yields)),
        "area_weighted_performance": float(np.dot(areas / areas.sum(), yields)),
        "covered_count": covered,
        "allocated_water_m3": allocated,
        "remaining_budget_m3": remaining,
        "coverage_threshold": threshold,
        "coverage_weight": weight,
        "joint_score": weight * covered / N_REGIONS + (1. - weight) * float(np.mean(yields)),
        "area_coverage_rate": float(np.sum(areas[yields >= threshold]) / np.sum(areas)),
        "performance_min": float(np.min(yields)),
        "performance_p10": float(np.quantile(yields, .1)),
    }
    for key, value in expected.items():
        if key not in metrics:
            raise ValueError(f"terminal metrics are missing {key}")
        actual = _real(metrics[key], f"terminal metric {key}")
        _close(actual, value, f"terminal metric {key}", atol=1e-7 if "m3" in key else 1e-12)
    _close(_array(metrics.get("regional_performance"), (N_REGIONS,), "regional_performance"),
           yields, "terminal regional_performance", atol=1e-12)
    mask = np.asarray(metrics.get("covered_mask"))
    if mask.shape != (N_REGIONS,) or mask.dtype.kind != "b" or not np.array_equal(mask, yields >= threshold):
        raise ValueError("terminal covered_mask disagrees with emitted outcomes")
    if metrics.get("N") != N_REGIONS:
        raise ValueError("terminal metric N disagrees with region identity")
    for alias, key in (("C", "coverage_rate"), ("J", "mean_performance"),
                       ("coverage_count", "covered_count")):
        if alias in metrics:
            _close(_real(metrics[alias], alias), expected[key], f"terminal metric alias {alias}", atol=1e-12)
    if "quality" in metrics:
        _close(_array(metrics["quality"], (N_REGIONS,), "quality alias"), yields,
               "terminal quality alias", atol=1e-12)
    return expected


def validate_episode(record):
    """Independently check identities, 82 frames, budgets, fluxes and C/J.

    Jensen outcomes are recalculated from the recorded daily fluxes and the
    fixed four-stage calendar.  Integrity hashes add change detection; they
    are not signatures or independent evidence of a trained policy's quality.
    This function checks bundled static parameter data, but does not read
    future weather realizations or create an environment.
    """
    if not isinstance(record, dict) or record.get("schema") != EPISODE_SCHEMA:
        raise ValueError("unsupported complete-episode schema")
    # Also reject NaN/infinity anywhere in an otherwise unused JSON field.
    _canonical(record)
    metadata = record.get("metadata", {})
    frames = record.get("frames", [])
    if metadata.get("horizon_days") != HORIZON or len(frames) != HORIZON + 1:
        raise ValueError("a complete episode requires exactly 82 frames")
    ids = metadata.get("region_ids", [])
    if len(ids) != N_REGIONS or len(set(ids)) != N_REGIONS or any(
            not isinstance(v, str) or not v for v in ids):
        raise ValueError("episode region identities must contain 93 unique strings")
    areas = _array(metadata.get("rice_area_ha"), (N_REGIONS,), "rice_area_ha")
    if np.any(areas <= 0):
        raise ValueError("rice areas must be positive")
    _close(areas.sum(), 1327.2, "published total area")
    budget = _real(metadata.get("initial_budget_m3"), "initial_budget_m3", lower=0)
    objective = metadata.get("objective_spec", {})
    threshold = _real(objective.get("coverage_threshold"), "coverage_threshold", lower=0, upper=1)
    weight = _real(objective.get("coverage_weight"), "coverage_weight", lower=0, upper=1)
    mode = objective.get("reward_mode")
    if mode not in ("joint", "mean_performance"):
        raise ValueError("unsupported task reward mode")
    declared_status = metadata.get("controller_provenance_status")
    declared = metadata.get("controller_provenance")
    if declared_status == "not_supplied":
        if declared != {"status": "not_supplied"}:
            raise ValueError("unsupplied controller provenance must be explicit")
    elif declared_status == "caller_declared_not_independently_verified":
        _controller_provenance(declared)
    else:
        raise ValueError("controller provenance must distinguish declaration from verification")
    for key in ("task_fingerprint", "physical_fingerprint"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"episode is missing {key}")
    if metadata.get("model_config_sha256") != _sha(metadata.get("model_config")):
        raise ValueError("model configuration SHA256 mismatch")
    model_config = metadata["model_config"]
    if (model_config.get("task_fingerprint") != metadata["task_fingerprint"] or
            model_config.get("physical_fingerprint") != metadata["physical_fingerprint"] or
            model_config.get("objective_spec") != objective):
        raise ValueError("model configuration disagrees with task/physical identity")
    expected_fingerprint = _sha({"schema": objective["schema"],
        "physical_fingerprint": metadata["physical_fingerprint"],
        "implementation_sha256": model_config["implementation_sha256"],
        "requested_budget_m3": model_config["requested_total_budget_m3"],
        "objective": objective})
    if expected_fingerprint != metadata["task_fingerprint"]:
        raise ValueError("task fingerprint does not bind the recorded implementation/objective")
    identity_sha = _sha(_identity(metadata))
    if metadata.get("identity_sha256") != identity_sha:
        raise ValueError("task identity SHA256 mismatch")
    core_config = metadata.get("run_configuration", {})
    if core_config.get("scenario_fingerprint") != metadata["physical_fingerprint"]:
        raise ValueError("physical fingerprint disagrees with core configuration")
    if (model_config.get("region_ids") != ids or model_config.get("n_regions") != N_REGIONS):
        raise ValueError("model configuration disagrees with region identities")
    _close(_real(model_config.get("initial_budget_m3"), "model initial budget", lower=0),
           budget, "model/record initial budget", atol=max(1e-7, budget * 2e-12))
    _close(_real(core_config.get("total_budget_m3"), "core total budget", lower=0),
           budget, "core/record initial budget", atol=max(1e-7, budget * 2e-12))
    # Bind recorded areas, initial water and the full public parameter matrix
    # to the checksummed input pack, rather than merely to the first frame.
    from ._core.direct_execution import FIELD_PARAMETER_NAMES
    from ._core.hongmiao import load_hongmiao_scenario
    reference_scene, reference_model = load_hongmiao_scenario(
        profile=model_config["parameter_profile"],
        historical_year=model_config["historical_year"],
        daily_quota_m3=model_config["daily_quota_m3"],
        storage_variation=model_config["storage_variation"])
    if list(reference_scene.region_ids) != ids:
        raise ValueError("recorded region order disagrees with frozen data")
    _close(areas, reference_scene.areas_ha, "frozen published regional areas", atol=0, rtol=0)
    if (model_config.get("field_parameter_names") != list(FIELD_PARAMETER_NAMES) or
            _canonical(model_config.get("calendar")) != _canonical(reference_model["calendar"])):
        raise ValueError("model parameter columns or crop calendar disagree with frozen data")
    reference_parameters = np.array([[getattr(p, key) for key in FIELD_PARAMETER_NAMES]
                                     for p in reference_scene.field_params])
    reference_capacity = np.array([p.root_saturation_mm + p.h_max_mm
                                   for p in reference_scene.field_params])
    reference_policy_properties = {
        "percolation_sat_mm_day": np.array([p.percolation_sat_mm_day for p in reference_scene.field_params]),
        "root_available_water_mm": np.array([p.root_available_water_mm for p in reference_scene.field_params]),
    }
    start = date.fromisoformat(core_config["start_date"])
    if (start.month, start.day) != (5, 30):
        raise ValueError("Chapter 6 calendar must start on May 30")
    coefficients = _array(core_config.get("yield_lambdas"), (N_REGIONS, 4), "yield_lambdas")
    _close(coefficients, np.broadcast_to(YIELD_COEFFICIENTS, coefficients.shape),
           "frozen Jensen coefficients", atol=0, rtol=0)
    stage_actual = np.zeros((N_REGIONS, 4))
    stage_reference = np.zeros_like(stage_actual)
    allocated = 0.
    daily_volumes = []
    previous = None
    prior_storage = None
    max_residual = 0.
    metrics_result = None
    budget_tolerance = max(1e-7, budget * 2e-12)
    initial_policy = frames[0].get("policy_observation", {})
    required_policy = {"day_index", "available_water_m3", "rice_area_ha",
                       "percolation_sat_mm_day", "root_available_water_mm",
                       "reported_field_storage_mm", "reported_completed_yield_factor",
                       "reported_current_stage_et_mm", "reported_current_stage_reference_et_mm",
                       "previous_weather"}

    for index, frame in enumerate(frames):
        if frame.get("day") != index:
            raise ValueError("frame day indices must be consecutive from 0 through 81")
        expected_date = (start + timedelta(days=index)).isoformat()
        expected_decision = None if index == 0 else (start + timedelta(days=index - 1)).isoformat()
        if frame.get("date") != expected_date or frame.get("decision_date") != expected_decision:
            raise ValueError("frame decision/state dates are inconsistent")
        for key, expected in (("task_fingerprint", metadata["task_fingerprint"]),
                              ("model_config_sha256", metadata["model_config_sha256"]),
                              ("identity_sha256", identity_sha)):
            if frame.get(key) != expected:
                raise ValueError(f"frame {index} {key} disagrees with episode identity")
        if frame.get("previous_frame_sha256") != previous:
            raise ValueError("frame SHA256 chain is broken")
        unsigned = {k: v for k, v in frame.items() if k != "frame_sha256"}
        if frame.get("frame_sha256") != _sha(unsigned):
            raise ValueError(f"frame {index} SHA256 mismatch")
        previous = frame["frame_sha256"]
        action = _array(frame.get("allocation_m3"), (N_REGIONS,), "allocation_m3")
        context = frame.get("step_context", {})
        if context.get("state_date") != expected_date or context.get("executed_date") != expected_decision:
            raise ValueError("step context dates disagree with the frame")
        _close(_array(context.get("executed_action_m3"), (N_REGIONS,), "executed action"),
               action, "recorded executed action", atol=0, rtol=0)
        if context.get("day") != index or bool(context.get("episode_complete")) != (index == HORIZON):
            raise ValueError("step context boundary or completion flag is inconsistent")
        if context.get("carryable") is not True:
            raise ValueError("Chapter 6 requires a carryable seasonal budget")
        if index == 0 and np.any(action != 0):
            raise ValueError("the initial frame cannot contain an executed action")
        before_budget = budget - allocated
        daily_volume = math.fsum(action)
        if index:
            daily_volumes.append(daily_volume)
            allocated = math.fsum(daily_volumes)
        remaining = budget - allocated
        if remaining < -budget_tolerance:
            raise ValueError("episode actions exceed the seasonal budget")
        _close(float(context["available_before_m3"]), before_budget,
               "available budget before action", atol=budget_tolerance)
        _close(float(context["allocated_m3"]), daily_volume,
               "allocated daily volume", atol=budget_tolerance)
        _close(float(context["remaining_after_m3"]), remaining,
               "remaining seasonal budget", atol=budget_tolerance)
        observation = frame.get("observation", {})
        policy = frame.get("policy_observation", {})
        if set(policy) != required_policy:
            raise ValueError("policy recording must retain the exact compact observation fields")
        if int(policy["day_index"]) != index or int(observation.get("day_index", -1)) != index:
            raise ValueError("policy/public observation dates disagree")
        _close(_array(observation.get("rice_area_ha"), (N_REGIONS,), "public rice area"),
               areas, "public region area identity", atol=0, rtol=0)
        _close(_array(observation.get("field_params"), reference_parameters.shape, "public field parameters"),
               reference_parameters, "frozen public physical parameters", atol=0, rtol=0)
        _close(_array(observation.get("field_storage_capacity_mm"), (N_REGIONS,), "public capacity"),
               reference_capacity, "frozen regional storage capacity", atol=1e-10)
        _close(_array(policy["rice_area_ha"], (N_REGIONS,), "policy rice area"),
               areas, "policy region area identity", atol=0, rtol=0)
        available = max(remaining, 0.) if index < HORIZON else 0.
        _close(float(policy["available_water_m3"]), available,
               "policy available budget", atol=budget_tolerance)
        _close(float(observation["remaining_budget_m3"]), remaining,
               "public remaining budget", atol=budget_tolerance)
        for key in ("percolation_sat_mm_day", "root_available_water_mm"):
            _close(_array(policy[key], (N_REGIONS,), key),
                   _array(initial_policy[key], (N_REGIONS,), key),
                   f"fixed policy property {key}", atol=0, rtol=0)
            _close(_array(policy[key], (N_REGIONS,), key), reference_policy_properties[key],
                   f"frozen policy property {key}", atol=1e-10)
        policy_weather = _array(policy["previous_weather"], (2,), "policy previous weather")
        _close(policy_weather, _array(observation.get("previous_weather"), (2,), "public previous weather"),
               "policy/public previous weather", atol=0, rtol=0)
        truth = frame.get("truth", {})
        if frame.get("access") != "privileged_evaluation_not_policy_input":
            raise ValueError("flux diagnostics must be labelled evaluator-only")
        storage = _array(truth.get("field_storage_mm"), (N_REGIONS,), "field_storage_mm")
        _close(_array(policy["reported_field_storage_mm"], (N_REGIONS,), "policy storage"),
               storage, "accurate current policy storage", atol=1e-10)
        _close(_array(observation.get("reported_field_storage_mm"), (N_REGIONS,), "public storage"),
               storage, "accurate public storage", atol=1e-10)
        transpiration = _array(truth.get("daily_transpiration_mm"), (N_REGIONS,), "daily transpiration")
        evaporation = _array(truth.get("daily_evaporation_mm"), (N_REGIONS,), "daily evaporation")
        eto = _array(truth.get("daily_eto_mm"), (N_REGIONS,), "daily ET0")
        if index:
            if truth.get("flow_scope") != "executed_day":
                raise ValueError("daily fluxes must belong to the executed day")
            before = _array(truth.get("before_field_storage_mm"), (N_REGIONS,), "before storage")
            _close(before, prior_storage, "adjacent frame water state", atol=1e-10)
            irrigation = _array(truth.get("daily_irrigation_mm"), (N_REGIONS,), "daily irrigation")
            rain = _array(truth.get("daily_rain_mm"), (N_REGIONS,), "daily rain")
            et = _array(truth.get("daily_et_mm"), (N_REGIONS,), "daily ET")
            drainage = _array(truth.get("daily_drainage_mm"), (N_REGIONS,), "daily drainage")
            overflow = _array(truth.get("daily_overflow_mm"), (N_REGIONS,), "daily overflow")
            reference = _array(truth.get("daily_reference_et_mm"), (N_REGIONS,), "daily reference ET")
            _close(et, transpiration + evaporation, "daily ET equals Ta plus Ea", atol=1e-10)
            _close(eto, np.full(N_REGIONS, eto[0]), "shared daily ET0", atol=0, rtol=0)
            _close(float(policy_weather[1]), eto[0], "observed previous-day ET0", atol=0, rtol=0)
            month_day = expected_decision[5:]
            periods = [p for p in model_config["calendar"]
                       if p["start_mm_dd"] <= month_day <= p["end_mm_dd_inclusive"]]
            if len(periods) != 1:
                raise ValueError("each recorded decision day requires one crop coefficient period")
            coefficient = float(periods[0]["kcb"]) + float(periods[0]["ke_wet"])
            _close(reference, coefficient * eto, "daily reference ET equals Kc times ET0", atol=1e-9)
            _close(irrigation, action / (10. * areas), "m3 to mm action conversion", atol=1e-10)
            residual = storage - (before + irrigation + rain - et - drainage - overflow)
            max_residual = max(max_residual, float(np.max(np.abs(residual))))
            _close(residual, 0., "regional daily water balance", atol=1e-8, rtol=0)
            _close(rain, np.full(N_REGIONS, rain[0]), "shared daily rain", atol=0, rtol=0)
            _close(float(policy["previous_weather"][0]), rain[0],
                   "observed previous-day rain", atol=1e-10)
            # Offsets 0--8 are establishment; the response stages have exactly
            # 37, 16, 10 and 9 physical days, including both tillering Kc parts.
            day = index - 1
            stage = None if day < 9 else (0 if day < 46 else 1 if day < 62 else 2 if day < 72 else 3)
            if stage is not None:
                stage_actual[:, stage] += et
                stage_reference[:, stage] += reference
        elif truth.get("before_field_storage_mm") is not None:
            raise ValueError("initial frame must not invent a preceding field state")
        else:
            _close(storage, reference_scene.initial_storage_mm, "frozen initial field storage", atol=1e-10)
            for flux in (transpiration, evaporation, eto):
                _close(flux, 0., "initial frame has no executed-day forcing", atol=0, rtol=0)
        _close(_array(truth.get("stage_et_mm"), (N_REGIONS, 4), "stage actual ET"),
               stage_actual, "stage ET reconstructed from daily fluxes", atol=2e-8)
        _close(_array(truth.get("stage_reference_et_mm"), (N_REGIONS, 4), "stage reference ET"),
               stage_reference, "stage reference reconstructed from daily fluxes", atol=2e-8)
        _close(_array(observation.get("reported_stage_et_mm"), (N_REGIONS, 4), "public stage ET"),
               stage_actual, "public stage ET ledger", atol=2e-8)
        _close(_array(observation.get("reported_stage_reference_et_mm"), (N_REGIONS, 4), "public reference ET"),
               stage_reference, "public reference ET ledger", atol=2e-8)
        ratios = np.ones_like(stage_actual)
        np.divide(stage_actual, stage_reference, out=ratios, where=stage_reference > 0)
        closed = [k for k, end in enumerate(RESPONSE_ENDS) if index >= end]
        factor = np.prod(ratios[:, closed] ** coefficients[:, closed], axis=1)
        _close(_array(policy["reported_completed_yield_factor"], (N_REGIONS,), "completed yield factor"),
               factor, "compressed completed-stage factor", atol=1e-10)
        active = None if index < 9 or index >= HORIZON else (0 if index < 46 else 1 if index < 62 else 2 if index < 72 else 3)
        current_et = np.zeros(N_REGIONS) if active is None else stage_actual[:, active]
        current_ref = 0. if active is None else float(stage_reference[0, active])
        _close(_array(policy["reported_current_stage_et_mm"], (N_REGIONS,), "current stage ET"),
               current_et, "compressed current-stage ET", atol=2e-8)
        _close(float(policy["reported_current_stage_reference_et_mm"]), current_ref,
               "compressed current-stage reference", atol=2e-8)
        physical_reward = _real(frame.get("physical_reward"), "physical_reward")
        task_reward = _real(frame.get("task_reward"), "task_reward")
        _close(_real(frame.get("reward"), "legacy physical reward"), physical_reward,
               "physical reward alias", atol=0, rtol=0)
        _close(float(context["reward"]), physical_reward, "physical reward context", atol=1e-12)
        if index < HORIZON:
            _close(physical_reward, 0., "sparse physical reward", atol=0, rtol=0)
            _close(task_reward, 0., "sparse task reward", atol=0, rtol=0)
            if frame.get("terminal_objectives") is not None or frame.get("metrics") is not None:
                raise ValueError("terminal objectives cannot appear before the true terminal boundary")
        else:
            if np.any(stage_reference <= 0):
                raise ValueError("all four completed response stages require positive reference ET")
            if np.any(stage_actual - stage_reference > 2e-8):
                raise ValueError("actual stage ET exceeds the fully supplied reference")
            reconstructed_yields = np.prod(np.minimum(ratios, 1.) ** coefficients, axis=1)
            yields = _array(observation.get("settled_relative_yield"), (N_REGIONS,), "terminal yields")
            _close(yields, reconstructed_yields, "Jensen terminal performance", atol=1e-10)
            # Apply the >= threshold to the exact emitted Y, not an independently
            # summed approximation that could straddle an equality by one ULP.
            metrics_result = _check_metrics(frame.get("metrics"), yields=yields, areas=areas,
                                            threshold=threshold, weight=weight,
                                            allocated=allocated, remaining=remaining)
            expected_objectives = [metrics_result["coverage_rate"], metrics_result["mean_performance"]]
            _close(_array(frame.get("terminal_objectives"), (2,), "terminal objectives"),
                   expected_objectives, "terminal C/J objectives", atol=1e-12)
            expected_reward = (weight * expected_objectives[0] + (1. - weight) * expected_objectives[1]
                               if mode == "joint" else expected_objectives[1])
            _close(task_reward, expected_reward, "terminal task reward", atol=1e-12)
            _close(physical_reward, metrics_result["area_weighted_performance"],
                   "terminal physical reward", atol=1e-12)
        prior_storage = storage

    integrity = record.get("integrity", {})
    unsigned_record = {k: v for k, v in record.items() if k != "integrity"}
    if integrity.get("record_sha256") != _sha(unsigned_record) or integrity.get("last_frame_sha256") != previous:
        raise ValueError("complete episode SHA256 mismatch")
    summary = record.get("summary", {})
    if summary.get("complete") is not True:
        raise ValueError("episode summary must mark the complete terminal task")
    if (summary.get("task_fingerprint") != metadata["task_fingerprint"] or
            summary.get("physical_fingerprint") != metadata["physical_fingerprint"] or
            summary.get("objective_spec") != objective or
            _canonical(summary.get("metrics")) != _canonical(frames[-1]["metrics"])):
        raise ValueError("episode summary disagrees with terminal task identity or metrics")
    _close(float(summary["allocated_water_m3"]), allocated, "summary allocated volume", atol=budget_tolerance)
    _close(float(summary["remaining_budget_m3"]), budget - allocated, "summary remaining volume", atol=budget_tolerance)
    _close(float(summary["return"]), frames[-1]["task_reward"], "task episode return", atol=1e-12)
    _close(float(summary["core_return"]), frames[-1]["physical_reward"], "physical episode return", atol=1e-12)
    return {"valid": True, "schema": EPISODE_SCHEMA, "frames": len(frames),
            "decisions": HORIZON, "regions": N_REGIONS,
            "record_sha256": integrity["record_sha256"],
            "task_fingerprint": metadata["task_fingerprint"],
            "max_regional_water_balance_residual_mm": max_residual,
            "metrics": metrics_result}


def _csv_bytes(fieldnames, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _tables(record):
    meta, frames = record["metadata"], record["frames"]
    ids, areas = meta["region_ids"], np.asarray(meta["rice_area_ha"])
    daily, regional = [], []
    for frame in frames[1:]:
        truth, context = frame["truth"], frame["step_context"]
        daily.append(dict(day=frame["day"], decision_date=frame["decision_date"], state_date=frame["date"],
                          allocated_water_m3=context["allocated_m3"], remaining_budget_m3=context["remaining_after_m3"],
                          irrigated_regions=int(np.count_nonzero(frame["allocation_m3"])),
                          rain_mm=truth["daily_rain_mm"][0],
                          observed_eto_mm_day=frame["policy_observation"]["previous_weather"][1],
                          physical_reward=frame["physical_reward"], task_reward=frame["task_reward"]))
        for i, rid in enumerate(ids):
            row = dict(day=frame["day"], decision_date=frame["decision_date"], state_date=frame["date"],
                       region_id=rid, rice_area_ha=areas[i], allocation_m3=frame["allocation_m3"][i])
            for key in ("before_field_storage_mm", "field_storage_mm", "daily_irrigation_mm", "daily_rain_mm",
                        "daily_transpiration_mm", "daily_evaporation_mm", "daily_eto_mm",
                        "daily_et_mm", "daily_reference_et_mm", "daily_drainage_mm", "daily_overflow_mm"):
                row[key] = truth[key][i]
            regional.append(row)
    final = frames[-1]
    ys = final["metrics"]["regional_performance"]
    threshold = meta["objective_spec"]["coverage_threshold"]
    terminal = [dict(region_id=rid, rice_area_ha=areas[i], model_relative_performance=ys[i],
                     coverage_threshold=threshold, covered=int(ys[i] >= threshold),
                     final_field_storage_mm=final["truth"]["field_storage_mm"][i])
                for i, rid in enumerate(ids)]
    return {"daily.csv": _csv_bytes(list(daily[0]), daily),
            "regional_daily.csv": _csv_bytes(list(regional[0]), regional),
            "terminal.csv": _csv_bytes(list(terminal[0]), terminal)}


_TASK_SCRIPT = r'''<script>
(()=>{const data=JSON.parse(document.getElementById('replay-data').textContent);
const box=document.getElementById('ch6-task-status');
function update(){const slider=document.getElementById('slider');const f=data.frames[Number(slider.value)||0];
let text='本步任务回报 '+Number(f.task_reward).toPrecision(7)+'。';
if(f.terminal_objectives){text+=' 终点覆盖率 C='+Number(f.terminal_objectives[0]).toFixed(6)+'，等单元平均成效 J='+Number(f.terminal_objectives[1]).toFixed(6)+'。';}
else{text+=' 终点评价尚未发生。';}box.textContent=text;}
const status=document.getElementById('status');new MutationObserver(update).observe(status,{childList:true,subtree:true,characterData:true});
document.getElementById('slider').addEventListener('input',update);update();})();
</script>'''


def write_episode(record, output_dir):
    """Publish a validated episode to a NEW directory; never overwrite output."""
    receipt = validate_episode(record)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"episode output already exists: {output}")
    tables = _tables(record)
    payload = _canonical(record) + b"\n"
    output.mkdir(parents=True, exist_ok=False)
    contents = {"episode.json": payload, **tables}
    for name, data in contents.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    # Keep the frozen renderer's reward physical; a separate task panel names
    # C/J and its reward explicitly and does not invent old hard guarantees.
    replay = write_rollout_html(record["frames"], output / "replay.html", metadata=record["metadata"])
    objective = record["metadata"]["objective_spec"]
    title = ('<section class="panel"><h2>第六章灌溉配置推演</h2>'
             f'<p>控制器：{html.escape(record["metadata"]["controller_label"])}。'
             f'研究门槛 τ={objective["coverage_threshold"]}，覆盖权重={objective["coverage_weight"]}。</p>'
             '<p id="ch6-task-status"></p>'
             '<p class="muted">下方原始物理回报表示面积加权模型成效；本任务的覆盖率 C、'
             '等单元平均成效 J 与任务回报在此单独报告。覆盖联合目标不构成逐区硬保障。'
             '本文件记录配置过程，不据此声称训练方法优越性。</p></section>')
    document = replay.read_text(encoding="utf-8").replace("<main>", "<main>" + title, 1)
    document = document.replace("</html>", _TASK_SCRIPT + "</html>")
    replay.write_text(document, encoding="utf-8")
    files = {name: {"sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
                    "bytes": (output / name).stat().st_size}
             for name in sorted(OUTPUT_FILES)}
    manifest = {"schema": MANIFEST_SCHEMA, "complete": True,
                "record_sha256": receipt["record_sha256"],
                "task_fingerprint": receipt["task_fingerprint"], "files": files,
                "validation": receipt}
    with (output / "manifest.json").open("xb") as stream:
        stream.write(_canonical(manifest) + b"\n")
    return output


def verify_episode_directory(path):
    """Read files independently and verify the published SHA256 manifest."""
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("complete") is not True:
        raise ValueError("incomplete or unsupported episode publication manifest")
    if set(manifest.get("files", {})) != OUTPUT_FILES:
        raise ValueError("episode publication file inventory is incomplete")
    for name, spec in manifest["files"].items():
        raw = (root / name).read_bytes()
        if len(raw) != spec.get("bytes") or hashlib.sha256(raw).hexdigest() != spec.get("sha256"):
            raise ValueError(f"published episode file SHA256 mismatch: {name}")
    record = json.loads((root / "episode.json").read_text(encoding="utf-8"))
    receipt = validate_episode(record)
    for name, expected in _tables(record).items():
        if (root / name).read_bytes() != expected:
            raise ValueError(f"published episode table disagrees with trajectory: {name}")
    if manifest.get("record_sha256") != receipt["record_sha256"] or manifest.get("task_fingerprint") != receipt["task_fingerprint"]:
        raise ValueError("publication manifest disagrees with episode identity")
    return receipt


def read_episode(path):
    """Load and validate an episode JSON or a complete publication directory."""
    path = Path(path)
    if path.is_dir():
        verify_episode_directory(path)
        path = path / "episode.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    validate_episode(record)
    return record
