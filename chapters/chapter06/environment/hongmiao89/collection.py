"""Native complete-episode collection with an explicit reduced89 interface."""
import math
import hashlib
import numpy as np
import torch
from .interface import STATE_KEYS, append_preference

METRIC_NAMES = ('C', 'JA', 'J', 'joint_score', 'area_weighted_performance', 'area_coverage_rate',
                'water_used_m3', 'remaining_budget_m3', 'performance_min', 'performance_p10')


def encoded_state(encoder, arm, weight, encoding_budget_m3=None):
    if arm != 'full':
        raise ValueError('Reduced89 supports only the explicit Full arm')
    state = encoder.state()
    if encoding_budget_m3 is not None:
        scale = float(encoding_budget_m3)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Positive frozen training-budget normalization required')
        state['global'][1] = float(state['remaining']) / scale
    return append_preference(state, weight)


def collect(pool, arm, actor, critic, weather_seeds, *, weight, generator=None,
            training=False, behavior_version=0, record_states=False,
            encoding_budget_m3=None, step_callback=None, record_fluxes=False):
    """One vectorized batch of complete81 physical trajectories, no cutting."""
    if arm != 'full':
        raise ValueError('Reduced89 collection supports only the explicit Full arm')
    from .interface import IrrigationSpec, IrrigationEncoder
    from allocation4.methods.lax import gaussian_parameters, gaussian_log_prob
    seeds = [int(s) for s in weather_seeds]
    if not seeds or len(seeds) > len(pool):
        raise ValueError("Nonempty seed list within the independent environment pool required")
    if training and (generator is None or critic is None):
        raise ValueError("Training requires an explicit CPU generator and critic")
    if next(actor.parameters()).device.type != 'cpu':
        raise ValueError('Native collection uses the synchronized CPU policy mirror')
    envs = pool[:len(seeds)]
    encoders = []
    observations = []
    saved = []
    for env, seed in zip(envs, seeds):
        obs, info = env.reset(seed=seed)
        if info:
            raise AssertionError("Reset policy info must be empty")
        enc = IrrigationEncoder(IrrigationSpec.from_env(env))
        enc.reset(obs)
        encoders.append(enc)
        observations.append(obs)
        saved.append(dict(states={k: [] for k in STATE_KEYS}, next_states={k: [] for k in STATE_KEYS},
            raw=[], old_logp=[], mask=[], values=[], behavior_mean=[], behavior_sigma=[],
            actions=[], rewards=[], done=[], weather_trace=[], hhi=[],
            initial_previous_weather=np.asarray(obs['previous_weather']).copy(),
            fixed_public_area_ha=np.asarray(obs["rice_area_ha"], dtype=np.float64).copy(),
            B_mm=([np.asarray(obs['reported_field_storage_mm']).copy()] if record_states else []),
            balance_m3=[float(enc.state()['remaining'])],
            water_balance=[], et_mm=[], percolation_mm=[], overflow_mm=[]))
    horizon = envs[0].T
    for day in range(horizon):
        states = [encoded_state(enc, arm, weight, encoding_budget_m3) for enc in encoders]
        combined = {key: np.stack([st[key] for st in states]) for key in STATE_KEYS}
        with torch.no_grad():
            sampled = actor.sample(combined, generator=generator, deterministic=not training)
            if training:
                values = critic(combined)
                mu, sigma = gaussian_parameters(actor, combined)
                err = float((gaussian_log_prob(mu, sigma, sampled["raw"]) - sampled["log_prob"]).abs().max())
                if err > 1e-7:
                    raise AssertionError("Complete Gaussian density mismatch")
                value_shape = (len(seeds), 2 if arm == "full" else 1)
                if tuple(values.shape) != value_shape:
                    raise AssertionError("Value output shape differs from arm")
                arrays = {name: sampled[name].detach().cpu().numpy().copy() for name in ("raw", "log_prob", "mask")}
                vv, mm, ss = [x.detach().cpu().numpy().copy() for x in (values, mu, sigma)]
                if not all(np.isfinite(x).all() for x in (*arrays.values(), vv, mm, ss)):
                    raise FloatingPointError("Nonfinite collection event")
        for j, (env, enc, state) in enumerate(zip(envs, encoders, states)):
            action = np.asarray(sampled["actions"][j], dtype=np.float64).copy()
            if action.shape != (env.N,) or not np.isfinite(action).all() or np.any(action < 0):
                raise ValueError("Finite nonnegative actual action required")
            if arm == "chapter2":
                expected = float(state["budget"])
                if abs(math.fsum(action.tolist()) - expected) > 1e-10 + 1e-12 * expected:
                    raise AssertionError("Chapter2 must execute its fixed actual daily total")
            obs, reward, done, truncated, info = env.step(action)
            if step_callback is not None:
                step_callback(j, bool(done))
            if truncated or done != (day == horizon - 1):
                raise AssertionError("Only the81st native action terminates the task")
            if not done and (reward != 0. or info):
                raise AssertionError("Intermediate policy feedback must remain empty")
            enc.update(action, obs)
            b = saved[j]
            if training:
                following = encoded_state(enc, arm, weight, encoding_budget_m3)
                for key in STATE_KEYS:
                    b["states"][key].append(state[key].copy())
                    b["next_states"][key].append(following[key].copy())
                for key, value in (("raw", arrays["raw"][j]), ("old_logp", arrays["log_prob"][j]),
                                   ("mask", arrays["mask"][j]), ("values", vv[j]),
                                   ("behavior_mean", mm[j]), ("behavior_sigma", ss[j])):
                    b[key].append(value.copy())
            b["actions"].append(action)
            b["rewards"].append(float(reward))
            b["done"].append(bool(done))
            b["weather_trace"].append(np.asarray(obs["previous_weather"]).copy())
            b['balance_m3'].append(float(enc.state()['remaining']))
            if record_states:
                b['B_mm'].append(np.asarray(obs['reported_field_storage_mm']).copy())
            if record_states or record_fluxes:
                observed = env.water_balance_observer()
                names = ('irrigation_m3','rain_m3','et_m3','drainage_m3','overflow_m3','water_mass_residual_m3')
                b['water_balance'].append([observed['daily_totals_m3'][k] for k in names])
                if record_states:
                    for k in ('et_mm','percolation_mm','overflow_mm'):
                        b[k].append(observed[k].copy())
            q = math.fsum(action.tolist())
            b["hhi"].append(float(np.square(action / q).sum()) if q else 0.)
            observations[j] = obs
    results = []
    for env, enc, seed, b in zip(envs, encoders, seeds, saved):
        m = env.terminal_metrics()
        rewards, done = np.asarray(b["rewards"]), np.asarray(b["done"], dtype=bool)
        quality = np.asarray(m["quality"], dtype=np.float64)
        if np.any(rewards[:-1] != 0.) or abs(rewards[-1] - float(m["area_weighted_performance"])) > 1e-12:
            raise AssertionError("Native scalar reward must be terminal area-weighted JA")
        actions = np.asarray(b["actions"], dtype=np.float64)
        weather = np.asarray(b["weather_trace"], dtype=np.float64)
        metrics = dict(C=float(m["C"]), JA=float(m["JA"]), J=float(m["J"]), joint_score=float(m["joint_score"]),
            area_weighted_performance=float(m["area_weighted_performance"]),
            area_coverage_rate=float(m["area_coverage_rate"]),
            water_used_m3=float(m["allocated_water_m3"]),
            remaining_budget_m3=float(m["remaining_budget_m3"]),
            performance_min=float(m["performance_min"]), performance_p10=float(m["performance_p10"]))
        if float(enc.state()["remaining"]) != metrics["remaining_budget_m3"]:
            raise AssertionError("Public executed-action ledger differs from environment")
        previous = np.asarray(b['initial_previous_weather'], dtype=np.float64)
        forcing_hash = hashlib.sha256(previous.tobytes() + weather.tobytes()).hexdigest()
        public = env.public_observation()
        stage_actual = np.asarray(public['reported_stage_et_mm'], dtype=np.float64)
        stage_reference = np.asarray(public['reported_stage_reference_et_mm'], dtype=np.float64)
        if stage_actual.shape != (89,4) or stage_reference.shape != (89,4) or not np.all(stage_reference == stage_reference[0]):
            raise AssertionError('Stage observer must preserve source [region,stage] axes')
        row = dict(**metrics, quality=quality.copy(), actions=actions,
            period_totals=actions.sum(1), rewards=rewards, weather_seed=seed,
            weather_trace=weather, initial_previous_weather=previous,
            weather_sha256=forcing_hash, forcing_sha256=forcing_hash,
            weather_only_sha256=hashlib.sha256(weather.tobytes()).hexdigest(),
            balance_m3=np.asarray(b['balance_m3'], dtype=np.float64),
            stage_ETa_mm=stage_actual.T.copy(), stage_reference_ETc_mm=stage_reference[0].copy(),
            mean_hhi=float(np.mean(b["hhi"])), K=int(m["coverage_count"]), N=env.N,
            task_fingerprint=env.task_fingerprint)
        row['terminal_storage_mm'] = np.asarray(public['reported_field_storage_mm'], dtype=np.float64).copy()
        if record_states or record_fluxes:
            row['water_balance_names'] = np.asarray(('irrigation_m3','rain_m3','et_m3','drainage_m3','overflow_m3','water_mass_residual_m3'))
            row['water_balance_m3'] = np.asarray(b['water_balance'], dtype=np.float64)
        if record_states:
            row['B_mm'] = np.asarray(b['B_mm'], dtype=np.float64)
            for k in ('et_mm','percolation_mm','overflow_mm'):
                row[k] = np.asarray(b[k], dtype=np.float64)
        if training:
            batch = {key: {name: np.stack(value) for name, value in b[key].items()}
                     for key in ("states", "next_states")}
            batch.update({key: np.asarray(b[key]) for key in ("raw", "old_logp", "mask", "values", "behavior_mean", "behavior_sigma")})
            batch.update(returns=np.full(horizon, metrics["JA"]), rewards=rewards, done=done,
                behavior_version=int(behavior_version), terminal_quality=quality.copy(),
                terminal_area_ha=b["fixed_public_area_ha"].copy(),
                JA=metrics["JA"], J=metrics["J"],
                K=int(m["coverage_count"]), C=metrics["C"], preference=float(weight))
            row["batch"] = batch
        results.append(row)
    return results
