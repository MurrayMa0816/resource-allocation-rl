"""Author PROPS update functions with the shared finite-horizon resource environment.

The vendored PPOROS checkout is read-only.  The runner replaces collection, environment
creation and experiment bookkeeping; the PPO/PROPS objectives are imported verbatim.
"""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root

import copy
import csv
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from runtime_hooks import configure_agent_factory, readonly_signal_audit, readonly_gradient_audit

SOURCE_SHA = "a0c8708529047f85dcf98f90d5ea264dd43aeba2"
ROOT = Path(str(_project_root()/'chapters/chapter04/foundation'/'chapter4_recent_baselines'))


def _load_official():
    source = ROOT / "vendor" / "props" / "PROPS"
    # Gymnasium 1.x moved this utility; author updates do not use the legacy
    # normalization wrappers, but importing their Agent still imports the name.
    normalization_name = "gymnasium.wrappers.normalize"
    if normalization_name not in sys.modules:
        try:
            __import__(normalization_name)
        except ModuleNotFoundError:
            from gymnasium.wrappers.utils import RunningMeanStd
            compatibility_module = ModuleType(normalization_name)
            compatibility_module.RunningMeanStd = RunningMeanStd
            sys.modules[normalization_name] = compatibility_module
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, source / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    utilities = load("chapter4_author_props_utils", "utils.py")
    previous_utils = sys.modules.get("utils")
    sys.modules["utils"] = utilities
    try:
        implementation = load("chapter4_author_props_updates", "ppo_props_continuous.py")
    finally:
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils
    return utilities.Agent, implementation.update_ppo, implementation.update_props


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer, np.floating)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, Path):
        return str(x)
    return x


def _dump(path, value):
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)


def _vector(agent, prefix=None):
    # These small diagnostics are faster as CPU copies under shared GPU load;
    # a GPU concatenate launches an extra kernel and regressed in measured tests.
    return torch.cat([p.detach().flatten().cpu() for name, p in agent.named_parameters()
                      if prefix is None or name.startswith(prefix)])


def _check_finite(target, behavior):
    # Read two booleans once; do not synchronize for every individual parameter.
    checks=torch.stack([torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()
                        for model in (target,behavior)]).cpu().tolist()
    if not checks[0]:
        raise FloatingPointError("Non-finite PROPS target parameters")
    if not checks[1]:
        raise FloatingPointError("Non-finite PROPS behavior parameters")


def _append_jsonl(path, rows):
    with path.open('a',encoding='utf-8') as handle:
        handle.writelines(json.dumps(_jsonable(row),ensure_ascii=False)+'\n' for row in rows)


def _delta(before, agent, prefix=None):
    return float(torch.linalg.vector_norm(_vector(agent, prefix) - before))


def _evaluate(Env, domain, seed, agent, device, episodes, train_steps):
    from chapter4_recent_baselines.common import evaluate
    def action_fn(observation, deterministic=False):
        with torch.no_grad():
            action = agent.get_action(torch.as_tensor(observation, dtype=torch.float32,
                                      device=device).unsqueeze(0), noise=not deterministic)
        return action[0].cpu().numpy()
    # The common evaluator fixes dev paths/action randomness and restores Python,
    # NumPy, Torch and CUDA RNG states. It is shared with the other four baselines.
    result = evaluate(action_fn, domain, seed, episodes=episodes, phase="dev",
                      action_mode="raw", deterministic=False)
    return {"train_env_steps": train_steps, "evaluation_env_steps": result["evaluation_steps"],
            "episodes": episodes, "mean_return": result["mean"] / 100.0,
            "std_return": result["std"] / 100.0,
            "returns": [score / 100.0 for score in result["episode_scores"]]}


def run(config: dict) -> dict:
    """Run author PROPS, return a serializable receipt, and write all files below output_dir.

    Required keys: domain ('radio'/'v5'), output_dir.  A smoke run should request at
    least eight complete episodes so that actual old-data reuse can be checked.
    Extra config keys are preserved in config.json. No formal run is launched here.
    """
    from chapter4_recent_baselines.env_adapter import Env

    started = time.perf_counter()
    domain = config.get("domain", "radio")
    seed = int(config.get("seed", 0))
    requested_device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for PROPS but is unavailable")
    device = torch.device(requested_device)
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    output = Path(config.get("output_dir", config.get("result_dir", ""))).resolve()
    if not str(config.get("output_dir", config.get("result_dir", ""))):
        raise ValueError("PROPS requires an explicit output_dir under /hy-tmp")
    if not output.is_relative_to(Path(str(_resolve_path('/hy-tmp')))):
        raise ValueError("Experiment outputs must be under /hy-tmp")
    output.mkdir(parents=True, exist_ok=True)
    env = Env(domain, seed, phase="train", action_mode="raw")
    horizon = int(env.horizon)
    target_steps = int(config.get("target_episodes", 4)) * horizon
    behavior_steps = int(config.get("behavior_episodes", 1)) * horizon
    capacity = int(config.get("buffer_episodes", 8)) * horizon
    if target_steps % behavior_steps or capacity < target_steps or capacity % behavior_steps:
        raise ValueError("PROPS requires target/buffer multiples of behavior batch and buffer >= target")
    total_steps = int(config.get("total_steps", config.get("total_timesteps",
                       int(config.get("episodes", 8 if config.get("smoke", False) else 960)) * horizon)))
    if total_steps < target_steps or total_steps % target_steps:
        raise ValueError("Use at least one whole target batch and total_steps divisible by 4H")
    smoke = bool(config.get("smoke", False))
    args = SimpleNamespace(
        track=False, num_envs=1, num_steps=target_steps, buffer_size=capacity,
        props_num_steps=behavior_steps,
        learning_rate=float(config.get("learning_rate", 1e-4)),
        props_learning_rate=float(config.get("props_learning_rate", 1e-4)),
        gamma=float(config.get("gamma", 1.0)), gae_lambda=float(config.get("gae_lambda", 0.95)),
        update_epochs=int(config.get("update_epochs", 1 if smoke else 10)),
        props_update_epochs=int(config.get("props_update_epochs", 1 if smoke else 16)),
        minibatch_size=max(2, capacity // int(config.get("num_minibatches", 32))),
        props_minibatch_size=max(2, (capacity - behavior_steps) // int(config.get("props_num_minibatches", 16))),
        norm_adv=True, clip_coef=float(config.get("clip_coef", 0.2)),
        clip_vloss=True, ent_coef=float(config.get("ent_coef", 0.01)), vf_coef=0.5,
        max_grad_norm=0.5, target_kl=float(config.get("target_kl", 0.03)),
        props_clip_coef=float(config.get("props_clip_coef", 0.3)), props_target_kl=0.05,
        props_lambda=float(config.get("props_lambda", 0.3)), props_adv=False,
        props_max_grad_norm=0.5, ros=False,
    )
    facade = SimpleNamespace(single_observation_space=env.observation_space,
                             single_action_space=env.action_space)
    Agent, update_ppo, update_props = _load_official()
    Agent = configure_agent_factory(Agent, env.native_spec, seed, config)
    target = Agent(facade).to(device)
    behavior = copy.deepcopy(target).to(device)
    optimizer = torch.optim.Adam(target.parameters(), lr=args.learning_rate, eps=1e-5)
    behavior_optimizer = torch.optim.Adam(behavior.parameters(), lr=args.props_learning_rate, eps=1e-5)
    initial_actor = _vector(target, "actor")
    initial_critic = _vector(target, "critic")
    rows = deque(maxlen=capacity)
    training = []
    evaluations = []
    training_flushed = 0
    evaluations_written = 0
    (output/'training.jsonl').write_text('',encoding='utf-8')
    target_updates = 0
    behavior_updates = 0
    target_minibatches = 0
    behavior_minibatches = 0
    sample_uses = 0
    nonterminal_reward_count = 0
    terminal_reward_count = 0
    episode = 0
    global_step = 0
    episode_return = 0.0
    obs, _ = env.reset()
    previous_done = 0.0
    eval_every_episodes = int(config.get("eval_every_episodes",
                             int(config.get("eval_every_updates", 12)) * target_steps // horizon))
    if eval_every_episodes < 1 or eval_every_episodes % (target_steps // horizon):
        raise ValueError("PROPS evaluation period must align with full target batches")
    eval_episodes = int(config.get("eval_episodes", 2 if smoke else 32))
    _dump(output / "config.json", {**config, "effective_args": vars(args), "horizon": horizon,
                                    "target_steps": target_steps, "behavior_steps": behavior_steps,
                                    "buffer_capacity": capacity})
    if not smoke:
        evaluations.append(_evaluate(Env, domain, seed, target, device, eval_episodes, 0))

    while global_step < total_steps:
        for _ in range(behavior_steps):
            obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                # Same native actor, distribution and RNG draw. The collector
                # discarded the critic result; values are recomputed in batch.
                action, _, _, behavior_logprob, _ = behavior.get_action_and_info(obs_tensor)
            action_cpu=action[0].cpu().numpy()
            next_obs, reward, terminal, truncated, _ = env.step(action_cpu)
            done = bool(terminal or truncated)
            reward = float(reward)
            if not done and reward != 0.0:
                nonterminal_reward_count += 1
                raise AssertionError("Shared environment leaked a nonterminal reward")
            rows.append({"obs": np.asarray(obs, dtype=np.float32).copy(),
                         "action": action_cpu.copy(),
                         "reward": reward, "done_before": previous_done,
                         "episode": episode, "behavior_logprob": float(behavior_logprob.item())})
            global_step += 1
            episode_return += reward
            obs = next_obs
            previous_done = float(done)
            if done:
                terminal_reward_count += 1
                episode += 1
                training.append({"record_type": "episode", "train_env_steps": global_step,
                                 "episode": episode, "episode_return": episode_return})
                episode_return = 0.0
                obs, _ = env.reset()
        # The buffer is chronologically ordered, matching author's reordered ring buffer.
        batch = list(rows)
        observations = torch.as_tensor(np.stack([r["obs"] for r in batch]), device=device).unsqueeze(1)
        actions = torch.as_tensor(np.stack([r["action"] for r in batch]), device=device).unsqueeze(1)
        rewards = torch.tensor([r["reward"] for r in batch], device=device).unsqueeze(1)
        dones = torch.tensor([r["done_before"] for r in batch], device=device).unsqueeze(1)
        with torch.no_grad():
            _, means, stds, logprobs, _, values = target.get_action_and_value(
                observations.flatten(0, 1), actions.flatten(0, 1))
            logprobs = logprobs.reshape(-1, 1)
            values = values.reshape(-1, 1)
            next_value = target.get_value(torch.as_tensor(obs, device=device,
                                                          dtype=torch.float32).unsqueeze(0)).reshape(1, 1)
            advantages = torch.zeros_like(rewards)
            gae = torch.zeros(1, device=device)
            for index in reversed(range(len(batch))):
                next_alive = 1.0 - (previous_done if index == len(batch) - 1 else dones[index + 1])
                bootstrap = next_value if index == len(batch) - 1 else values[index + 1]
                delta = rewards[index] + args.gamma * bootstrap * next_alive - values[index]
                gae = delta + args.gamma * args.gae_lambda * next_alive * gae
                advantages[index] = gae
            returns = advantages + values
        update_record = {"record_type": "update", "train_env_steps": global_step,
                         "buffer_rows": len(batch), "target_update": False}
        update_record.update(readonly_signal_audit(means, stds, advantages, returns, values, rewards, dones, previous_done, config))
        if global_step % target_steps == 0:
            before_actor = _vector(target, "actor")
            before_critic = _vector(target, "critic")
            if config.get("anneal_lr", True):
                optimizer.param_groups[0]["lr"] = args.learning_rate * (1.0 - target_updates / (total_steps // target_steps))
            stats = update_ppo(target, optimizer, facade, observations, logprobs, actions,
                               advantages, returns, values, args, global_step, None)
            target_updates += 1
            target_minibatches += int(stats.get("ppo_num_update_minibatches", 0))
            sample_uses += len(batch)
            old_rows = sum(r["episode"] < episode - target_steps // horizon for r in batch)
            update_record.update(stats)
            update_record.update(target_update=True, target_update_index=target_updates,
                                 old_rows_in_target_batch=old_rows, new_rows_in_target_batch=len(batch) - old_rows,
                                 target_actor_delta=_delta(before_actor, target, "actor"),
                                 target_critic_delta=_delta(before_critic, target, "critic"))
        if update_record['target_update']:
            update_record.update(readonly_gradient_audit(target))
        # Preserve author main order: reference logprobs/means are computed before
        # target update; then behavior is reset to the (possibly just updated) target.
        behavior.load_state_dict(target.state_dict())
        behavior_before = _vector(behavior)
        props_stats = update_props(behavior, facade, behavior_optimizer, observations,
                                   logprobs, actions, advantages, global_step, args, None, means, stds)
        behavior_updates += 1
        behavior_minibatches += int(props_stats.get("props_num_update_minibatches", 0))
        update_record.update(props_stats)
        update_record["behavior_parameter_delta_from_target"] = _delta(behavior_before, behavior)
        update_record["target_update_count"] = target_updates
        update_record["behavior_update_count"] = behavior_updates
        _check_finite(target,behavior)
        training.append(update_record)
        if global_step % target_steps == 0 and (episode % eval_every_episodes == 0 or global_step == total_steps):
            evaluations.append(_evaluate(Env, domain, seed, target, device, eval_episodes, global_step))
        # Preserve every row in a live append-only log. The same full CSV is
        # materialized once, instead of rewriting O(episodes^2) rows.
        _append_jsonl(output/'training.jsonl',training[training_flushed:])
        training_flushed=len(training)
        if len(evaluations)>evaluations_written:
            _append_jsonl(output/'evaluations.jsonl',evaluations[evaluations_written:])
            _write_csv(output / "evaluations.csv", [{k: v for k, v in r.items() if k != "returns"} for r in evaluations])
            evaluations_written=len(evaluations)

    _write_csv(output/'training.csv',training)

    checkpoint = {"algorithm": "PROPS", "target": target.state_dict(), "behavior": behavior.state_dict(),
                  "optimizer": optimizer.state_dict(), "behavior_optimizer": behavior_optimizer.state_dict(),
                  "train_env_steps": global_step, "episodes": episode, "config": config,
                  "effective_args": vars(args), "source_sha": SOURCE_SHA}
    torch.save(checkpoint, output / "checkpoint.pt")
    loaded = torch.load(output / "checkpoint.pt", map_location=device, weights_only=False)
    checkpoint_matches = all(torch.equal(value, loaded["target"][key])
                             for key, value in target.state_dict().items())
    from chapters.chapter04.source_provenance import verify_props_source
    source_file = ROOT / "vendor" / "props" / "PROPS" / "ppo_props_continuous.py"
    accepted_source = verify_props_source(ROOT / "vendor" / "props", SOURCE_SHA)
    actual_sha = accepted_source["actual_sha"]
    source_modified = accepted_source["source_modified"]
    source_receipt = {"repository": "https://github.com/NicholasCorrado/PPOROS", "expected_sha": SOURCE_SHA,
                      "actual_sha": actual_sha, "source_file": str(source_file),
                      "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
                      "functions": ["update_ppo", "update_props"], "source_modified": source_modified,
                      "compatibility": "Gymnasium 1.x RunningMeanStd legacy import alias only",
                      "license": "not_declared", "framework": "PyTorch"}
    if actual_sha != SOURCE_SHA or source_modified:
        raise AssertionError("Unexpected PROPS vendor commit")
    checks = {"nonterminal_reward_count": nonterminal_reward_count,
              "completed_episodes": episode, "terminal_feedback_count": terminal_reward_count,
              "target_updates": target_updates, "behavior_updates": behavior_updates,
              "target_minibatches": target_minibatches, "behavior_minibatches": behavior_minibatches,
              "target_actor_parameter_delta": _delta(initial_actor, target, "actor"),
              "target_critic_parameter_delta": _delta(initial_critic, target, "critic"),
              "maximum_behavior_delta": max(r.get("behavior_parameter_delta_from_target", 0) for r in training),
              "maximum_old_rows_in_target_update": max(r.get("old_rows_in_target_batch", 0) for r in training),
              "checkpoint_roundtrip_exact": checkpoint_matches,
              "sample_references_in_target_batches": sample_uses,
              "behavior_reference_order": "author_main_pre_target_update_reference",
              "gamma_note": "finite_horizon gamma defaults 1.0; author MuJoCo default 0.99; configurable"}
    result = {"method": "PROPS", "domain": domain, "seed": seed, "smoke": smoke,
              "status": "smoke_passed" if smoke else "completed", "device": str(device),
              "train_env_steps": global_step, "evaluation_env_steps": sum(r["evaluation_env_steps"] for r in evaluations),
              "episodes": episode, "target_updates": target_updates, "behavior_updates": behavior_updates,
              "target_steps": target_steps, "behavior_steps": behavior_steps, "buffer_capacity": capacity,
              "evaluation_policy": "stochastic_temperature1_common_dev_paths_rng_restored",
              "eval_every_episodes": eval_every_episodes, "eval_episodes": eval_episodes,
              "final_evaluation": evaluations[-1] if evaluations else None,
              "elapsed_seconds": time.perf_counter() - started, "output_dir": str(output),
              "mechanism_checks": checks, "source_receipt": source_receipt}
    env.close()
    _dump(output / "source_receipt.json", source_receipt)
    _dump(output / "mechanism_checks.json", checks)
    _dump(output / "evaluations.json", evaluations)
    _dump(output / "result.json", result)
    return result
