"""Native Prioritized Generative Replay, with only task/experiment adaptation.

No d4rl dataset or MuJoCo task is loaded.  The author's curiosity, conditional
distribution, diffusion training/generation and real/synthetic SAC update are
all executed.  Small smoke settings prove integration, not effectiveness.
"""
from __future__ import annotations

import ast
import copy
import csv
import functools
import hashlib
import importlib
import json
import math
import random
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/resource_allocation_run_storage_20261101/chapter4_recent_baselines_formal_20261102_source/chapter4_recent_baselines')
VENDOR = ROOT / "vendor" / "pgr"
PGR_SHA = "b52b41b681f67608f8a5678211ce6cd5d933f9cc"
REDQ_SHA = "7b5d1bff39291a57325a2836bd397a55728960bb"


def _dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def install_safe_trainer_optimizations(trainer):
    """Remove logging synchronization and repeated same-device EMA traversal.

    The native caller only formats the returned loss every 1000 steps.  Returning
    its detached scalar tensor postpones that CPU read; gradients, optimizer,
    clipping, EMA mathematics, schedule and all RNG calls retain their order.
    """
    if hasattr(trainer, "_pgr_native_train_on_batch"):
        return
    trainer._pgr_native_train_on_batch = trainer.train_on_batch
    trainer.ema.to(trainer.accelerator.device)

    def train_on_batch(self, data, **kwargs):
        accelerator = self.accelerator
        data = data.to(accelerator.device)
        if "cond" in kwargs:
            kwargs["cond"] = kwargs["cond"].to(accelerator.device)
        with accelerator.autocast():
            loss = self.model(data, **kwargs)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(self.model.parameters(), 1.)
        self.opt.step()
        self.opt.zero_grad()
        self.step += 1
        self.ema.update()
        if self.step != 0 and self.step % self.save_and_sample_every == 0:
            self.save(self.step)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return loss.detach()

    trainer.train_on_batch = types.MethodType(train_on_batch, trainer)


def install_safe_sampler_optimization(diffusion):
    """Read the native GPU sigma schedule once instead of 3 reads per step.

    The three vectors are computed by the unchanged native float32 code.  Their
    Python float values equal the previous individual .item() reads exactly.
    Only scalar host transfers are batched; no RNG operation is added or moved
    relative to another RNG operation, and all sampling equations stay native.
    """
    if hasattr(diffusion, "_pgr_native_sample"):
        return
    source = VENDOR / "synther" / "diffusion" / "elucidated_diffusion.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ElucidatedDiffusion")
    function = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "sample"))

    class ReadScheduleOnce(ast.NodeTransformer):
        changes = 0

        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "sigmas_and_gammas":
                value = ast.parse("list(zip(sigmas[:-1].tolist(), sigmas[1:].tolist(), gammas[:-1].tolist()))", mode="eval").body
                node.value = value
                self.changes += 1
                return node
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
                    and [getattr(v, "id", None) for v in node.targets[0].elts] == ["sigma", "sigma_next", "gamma"]
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "map"):
                self.changes += 1
                return ast.copy_location(ast.Pass(), node)
            return self.generic_visit(node)

    transformer = ReadScheduleOnce()
    function = transformer.visit(function)
    if transformer.changes != 2:
        raise RuntimeError("Upstream sampler changed: refusing an unverified schedule transfer patch")
    transformed = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(sys.modules[diffusion.__class__.__module__].__dict__)
    exec(compile(transformed, str(source), "exec"), namespace)
    diffusion._pgr_native_sample = diffusion.sample
    diffusion.sample = types.MethodType(namespace["sample"], diffusion)


def _load_author():
    for path in (VENDOR, VENDOR / "synther" / "REDQ"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import gin
    gin.clear_config()
    # The utility module imports d4rl solely for task registration and ipdb
    # solely for an unused debugging alias. Drop exactly those import nodes in
    # memory; every algorithm/function body retains the original source lines.
    name = "synther.diffusion.utils"
    source = VENDOR / "synther" / "diffusion" / "utils.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    dropped = []
    kept = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.name == "d4rl" for a in node.names):
            dropped.append("import d4rl"); continue
        if isinstance(node, ast.ImportFrom) and node.module == "ipdb":
            dropped.append("from ipdb import set_trace as st"); continue
        kept.append(node)
    tree.body = kept
    module = types.ModuleType(name)
    module.__file__ = str(source)
    module.__package__ = "synther.diffusion"
    sys.modules[name] = module
    exec(compile(tree, str(source), "exec"), module.__dict__)
    from synther.online.redq_rlpd_agent import REDQRLPDCondAgent
    from synther.diffusion.elucidated_diffusion import REDQCondTrainer
    from synther.diffusion.diffusion_generator import CondDiffusionGenerator
    from synther.diffusion.denoiser_network_cond import ResidualMLPDenoiser
    gin.bind_parameter("split_diffusion_samples.modelled_terminals", True)
    gin.bind_parameter("split_diffusion_samples.terminal_threshold", 0.5)
    return dict(Agent=REDQRLPDCondAgent, Trainer=REDQCondTrainer,
                Generator=CondDiffusionGenerator, Denoiser=ResidualMLPDenoiser,
                construct=module.construct_diffusion_model, dropped_imports=dropped)


def _cfg(config, env):
    smoke = bool(config.get("smoke", False))
    limits = []
    if config.get("total_steps") is not None:
        limits.append(int(config["total_steps"]))
    if config.get("episodes") is not None:
        limits.append(int(config["episodes"]) * env.horizon)
    steps = min(limits) if limits else (2 if smoke else 960) * env.horizon
    if steps < 2 * env.horizon:
        raise ValueError("PGR preparation needs >=2H real steps to exercise complete terminal trajectories")
    values = dict(
        total_steps=steps, gamma=float(config.get("gamma", 1.0)),
        learning_starts=int(config.get("learning_starts", env.horizon if smoke else 32 * env.horizon)),
        hidden_sizes=[256, 256], replay_size=max(steps + 1, 256 if smoke else 1_000_000),
        batch_size=int(config.get("batch_size", 8 if smoke else 256)),
        lr=3e-4, polyak=.995, alpha=.2, auto_alpha=True, target_entropy="auto",
        utd_ratio=int(config.get("updates_per_step", 1 if smoke else 20)), num_Q=2,
        num_min=2, policy_update_delay=1 if smoke else 20,
        cond_hidden_size=256, diffusion_buffer_size=256 if smoke else 1_000_000,
        diffusion_sample_ratio=.5, retrain_diffusion_every=steps if smoke else 10_000,
        diffusion_start=0, diffusion_train_steps=2 if smoke else 100_000,
        diffusion_batch_size=8 if smoke else 256,
        num_samples=32 if smoke else 1_000_000,
        # Chunking changes peak memory only: retain the author's total 1M
        # samples and full 128-step sampler in non-smoke runs.
        generation_batch_size=32 if smoke else 10_000,
        denoise_steps=4 if smoke else 128,
        denoiser_width=128 if smoke else 1024, denoiser_layers=2 if smoke else 6,
        denoiser_time_dim=32 if smoke else 256,
        cfg_dropout=.25, cond_top_frac=.25, cfg_scale=2.,
        model_terminals=True, normalizer="minmax",
        eval_episodes=int(config.get("eval_episodes", 1 if smoke else 32)),
        eval_every_episodes=int(config.get("eval_every_episodes", 48)),
        deterministic_eval=bool(config.get("deterministic_eval", False)),
        safe_runtime_optimizations=bool(config.get("safe_runtime_optimizations", True)),
    )
    values.update(config.get("pgr", {}))
    if not values["model_terminals"]:
        raise ValueError("Finite allocation tasks require model_terminals=True")
    if values["replay_size"] <= steps:
        raise ValueError("Native PGR uses replay.ptr for diffusion input: keep capacity > requested real steps")
    if values["num_samples"] % values["generation_batch_size"]:
        raise ValueError("Native conditional generator requires samples divisible by generation batch size")
    if values["diffusion_buffer_size"] < values["num_samples"]:
        raise ValueError("Synthetic buffer must hold a full generated cycle")
    if values["batch_size"] < 2 or values["diffusion_batch_size"] < 2:
        raise ValueError("Curiosity BatchNorm and diffusion require batch size >=2")
    if not 0 < values["cond_top_frac"] <= 1 or not 0 < values["diffusion_sample_ratio"] < 1:
        raise ValueError("Invalid priority fraction or real/synthetic mixture")
    if values["denoise_steps"] < 2:
        raise ValueError("Native denoising schedule requires at least two steps")
    return values


def _agent(native, cfg, env, device):
    return native["Agent"](
        cfg["cond_hidden_size"], cfg["diffusion_buffer_size"], cfg["diffusion_sample_ratio"],
        "resource_allocation", env.observation_space.shape[0], env.action_space.shape[0], 1., device,
        hidden_sizes=tuple(cfg["hidden_sizes"]), replay_size=cfg["replay_size"],
        batch_size=cfg["batch_size"], lr=cfg["lr"], gamma=cfg["gamma"], polyak=cfg["polyak"],
        alpha=cfg["alpha"], auto_alpha=True, target_entropy="auto",
        start_steps=cfg["learning_starts"], delay_update_steps=cfg["learning_starts"],
        utd_ratio=cfg["utd_ratio"], num_Q=cfg["num_Q"], num_min=cfg["num_min"],
        q_target_mode="min", policy_update_delay=cfg["policy_update_delay"],
    )


def _vector(module):
    return torch.cat([parameter.detach().flatten().cpu() for parameter in module.parameters()])


class _Log:
    def __init__(self):
        self.last = {}

    def store(self, **values):
        self.last = {key: float(np.asarray(value).mean()) for key, value in values.items()}


class _Counts:
    def __init__(self, agent):
        self.phase = "training"
        self.critic_minibatches = 0
        self.policy_steps = 0
        self.curiosity_steps = 0
        self.q_optimizer_steps = 0
        self.real_sample_references = 0
        self.synthetic_sample_references = 0
        self.mixed_minibatches = 0
        self.diffusion_gradient_steps = 0
        self.denoiser_example_evaluations = {}
        self.hooks = []
        native_sample = agent.sample_data

        def sample(batch_size):
            synthetic_count = int(batch_size * agent.diffusion_sample_ratio)
            mixed = agent.diffusion_buffer.size >= synthetic_count
            self.critic_minibatches += 1
            self.real_sample_references += batch_size - synthetic_count if mixed else batch_size
            self.synthetic_sample_references += synthetic_count if mixed else 0
            self.mixed_minibatches += int(mixed)
            return native_sample(batch_size)
        agent.sample_data = sample

        def count(which):
            def hook(optimizer, args, kwargs):
                setattr(self, which, getattr(self, which) + 1)
            return hook
        for optimizer in agent.q_optimizer_list:
            self.hooks.append(optimizer.register_step_post_hook(count("q_optimizer_steps")))
        self.hooks.append(agent.policy_optimizer.register_step_post_hook(count("policy_steps")))
        self.hooks.append(agent.cond_optimizer.register_step_post_hook(count("curiosity_steps")))

    def diffusion_hook(self, module, args):
        number = int(args[0].shape[0])
        self.denoiser_example_evaluations[self.phase] = self.denoiser_example_evaluations.get(self.phase, 0) + number

    def close(self):
        for hook in self.hooks:
            hook.remove()


def _rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()


def _set_rng(saved):
    random.setstate(saved[0]); np.random.set_state(saved[1]); torch.set_rng_state(saved[2])
    torch.cuda.set_rng_state_all(saved[3])


def _evaluate(agent, Env, config, cfg, train_steps):
    saved = _rng()
    mode = agent.policy_net.training
    values = []
    env = None
    try:
        eval_seed = 4_000_000 + config["seed"]
        random.seed(eval_seed); np.random.seed(eval_seed); torch.manual_seed(eval_seed)
        torch.cuda.manual_seed_all(eval_seed)
        env = Env(config["domain"], config["seed"], phase="dev", action_mode="tanh_latent")
        agent.policy_net.eval()
        for index in range(cfg["eval_episodes"]):
            obs, _ = env.reset(options={"episode_index": index})
            done, value = False, 0.
            while not done:
                with torch.no_grad():
                    action = agent.policy_net(torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0),
                                              deterministic=cfg["deterministic_eval"], return_log_prob=False)[0]
                obs, reward, done, truncated, _ = env.step(action[0].cpu().numpy())
                assert not truncated
                value += reward
            values.append(value)
        steps = env.real_steps
    finally:
        if env is not None:
            env.close()
        _set_rng(saved); agent.policy_net.train(mode)
    assert random.getstate() == saved[0]
    assert torch.equal(torch.get_rng_state(), saved[2])
    assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), saved[3]))
    assert np.array_equal(np.random.get_state()[1], saved[1][1])
    return dict(train_env_steps=train_steps, evaluation_env_steps=steps,
                mean_return=float(np.mean(values)), mean_score=float(np.mean(values)) * 100.,
                std_return=float(np.std(values)), returns=values, phase="dev",
                deterministic_eval=cfg["deterministic_eval"], evaluation_rng_seed=eval_seed,
                training_rng_restored=True)


def _buffer_state(buffer):
    size = buffer.size
    return {"size": size, "ptr": buffer.ptr, "max_size": buffer.max_size,
            **{key: getattr(buffer, key)[:size].copy() for key in
               ("obs1_buf", "obs2_buf", "acts_buf", "rews_buf", "done_buf")}}


def _restore_buffer(buffer, state):
    if buffer.max_size != state["max_size"]:
        raise ValueError("Cannot restore a differently sized native replay")
    for key in ("obs1_buf", "obs2_buf", "acts_buf", "rews_buf", "done_buf"):
        getattr(buffer, key)[:state["size"]] = state[key]
    buffer.size, buffer.ptr = state["size"], state["ptr"]


def _save_agent(agent, full_training_state=True):
    state = dict(policy=agent.policy_net.state_dict(), q=[q.state_dict() for q in agent.q_net_list],
                target_q=[q.state_dict() for q in agent.q_target_net_list],
                curiosity=agent.cond_net.state_dict(),
                log_alpha=agent.log_alpha.detach().clone(), alpha=agent.alpha)
    if full_training_state:
        state.update(policy_optimizer=agent.policy_optimizer.state_dict(),
                q_optimizers=[o.state_dict() for o in agent.q_optimizer_list],
                curiosity_optimizer=agent.cond_optimizer.state_dict(),
                alpha_optimizer=agent.alpha_optim.state_dict(),
                real_buffer=_buffer_state(agent.replay_buffer), synthetic_buffer=_buffer_state(agent.diffusion_buffer))
    return state


def _restore_agent(agent, state):
    agent.policy_net.load_state_dict(state["policy"])
    agent.cond_net.load_state_dict(state["curiosity"])
    for q, saved in zip(agent.q_net_list, state["q"]): q.load_state_dict(saved)
    for q, saved in zip(agent.q_target_net_list, state["target_q"]): q.load_state_dict(saved)
    with torch.no_grad(): agent.log_alpha.copy_(state["log_alpha"])
    agent.alpha = state["alpha"]
    if "policy_optimizer" in state:
        agent.policy_optimizer.load_state_dict(state["policy_optimizer"])
        for optimizer, saved in zip(agent.q_optimizer_list, state["q_optimizers"]): optimizer.load_state_dict(saved)
        agent.cond_optimizer.load_state_dict(state["curiosity_optimizer"])
        agent.alpha_optim.load_state_dict(state["alpha_optimizer"])
        _restore_buffer(agent.replay_buffer, state["real_buffer"])
        _restore_buffer(agent.diffusion_buffer, state["synthetic_buffer"])


def run(config):
    from chapter4_recent_baselines.env_adapter import Env
    started = time.perf_counter()
    config = dict(config)
    config.setdefault("seed", 0); config.setdefault("domain", "radio")
    if config.get("device", "cuda") not in ("cuda", "cuda:0") or not torch.cuda.is_available():
        raise RuntimeError("PGR preparation is configured for the available CUDA GPU")
    output = Path(config["output_dir"]).resolve()
    if not output.is_relative_to(Path("/hy-tmp")):
        raise ValueError("All experiment artifacts must stay under /hy-tmp")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    native = _load_author()
    env = Env(config["domain"], config["seed"], phase="train", action_mode="tanh_latent")
    cfg = _cfg(config, env)
    full_training_state = bool(config.get("save_training_state", config.get("smoke", False)))
    _dump(output / "config.json", config); _dump(output / "native_config.json", cfg)
    agent = _agent(native, cfg, env, torch.device("cuda:0"))
    logger, counts = _Log(), _Counts(agent)
    initial = {"policy": _vector(agent.policy_net), "critic": _vector(agent.q_net_list[0]),
               "curiosity": _vector(agent.cond_net)}
    evaluations = [_evaluate(agent, Env, config, cfg, 0)]
    # Keep each completed evaluation durable during this long native run.
    # This is logging only; update formulas, samples and RNG are unchanged.
    with (output / 'evaluations.jsonl').open('x') as handle:
        handle.write(json.dumps(evaluations[-1], allow_nan=False) + '\n')
    obs, _ = env.reset()
    episodes, real_steps = 0, 0
    episode_return = 0.0
    nonterminal_nonzero, real_terminals = 0, 0
    diffusion_cycles = []
    latest_trainer, latest_cond = None, None
    observation_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    transition_dim = 2 * observation_dim + action_dim + 2
    train_log = (output / "training.jsonl").open("w")
    for step in range(1, cfg["total_steps"] + 1):
        action = agent.get_exploration_action(obs, env)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        assert not truncated
        if not terminated and reward != 0: nonterminal_nonzero += 1
        real_terminals += int(terminated)
        # Preserve the true task terminal.  Do NOT copy the author's DMC-only
        # `d=False if ep_len==max_ep_len` continuing-task treatment.
        agent.store_data(obs, action, reward, next_obs, terminated)
        agent.train(logger)
        real_steps += 1
        episode_return += float(reward)
        obs = next_obs
        if terminated:
            episodes += 1; obs, _ = env.reset()
        if step % cfg["retrain_diffusion_every"] == 0 and step >= cfg["diffusion_start"]:
            cycle_dir = output / f"diffusion_cycle_{len(diffusion_cycles)+1:03d}"
            cycle_dir.mkdir(exist_ok=True)
            denoising = functools.partial(native["Denoiser"], dim_t=cfg["denoiser_time_dim"],
                mlp_width=cfg["denoiser_width"], num_layers=cfg["denoiser_layers"],
                learned_sinusoidal_cond=False, random_fourier_features=True,
                learned_sinusoidal_dim=16, layer_norm=False)
            diffusion = native["construct"](torch.zeros(128, transition_dim), normalizer_type=cfg["normalizer"],
                denoising_network=denoising, activation="mish", disable_terminal_norm=True,
                skip_dims=[observation_dim + action_dim], cond_dim=1, cfg_dropout=cfg["cfg_dropout"])
            latest_trainer = native["Trainer"](diffusion,
                train_batch_size=cfg["diffusion_batch_size"], train_lr=3e-4, lr_scheduler="cosine",
                train_num_steps=cfg["diffusion_train_steps"], save_and_sample_every=100_000_000,
                results_folder=str(cycle_dir), model_terminals=True)
            if not full_training_state:
                # Native save_final is I/O only, not training or sampling.  Its
                # unconditional optimizer/EMA dump would accumulate hundreds of
                # MB per cycle. Preserve all algorithm computation and save only
                # the cycle metadata under the default inference-only protocol.
                latest_trainer.save_final = lambda cond_distri, epoch, milestone: _dump(
                    cycle_dir / "diffusion_training_completed.json",
                    dict(epoch=epoch, steps=milestone, full_training_state_saved=False))
            old_parameter = next(latest_trainer.model.net.parameters()).detach().clone()
            train_hook = latest_trainer.model.net.register_forward_pre_hook(counts.diffusion_hook)
            ema_hook = latest_trainer.ema.ema_model.net.register_forward_pre_hook(counts.diffusion_hook)
            latest_trainer.update_normalizer(agent.replay_buffer, device=agent.device)
            if cfg["safe_runtime_optimizations"]:
                install_safe_trainer_optimizations(latest_trainer)
            counts.phase = "diffusion_training"
            latest_cond = latest_trainer.train_from_redq_buffer(agent.replay_buffer, agent.cond_net,
                top_frac=cfg["cond_top_frac"], curr_epoch=len(diffusion_cycles) + 1)
            counts.diffusion_gradient_steps += latest_trainer.step
            priority_indices = latest_cond.top_frac_indices
            if int(cfg["cond_top_frac"] * agent.replay_buffer.size) < 1:
                raise AssertionError("Empty top fraction would select unused replay capacity")
            if len(priority_indices) == 0 or priority_indices.max() >= agent.replay_buffer.size:
                raise AssertionError("Native priority selection reached uncollected data")
            diffusion_change = float((next(latest_trainer.model.net.parameters()).detach() - old_parameter).norm().cpu())
            if diffusion_change <= 0: raise AssertionError("Native diffusion did not learn")
            agent.reset_diffusion_buffer()
            generator = native["Generator"](env, latest_trainer.ema.ema_model, latest_cond,
                num_sample_steps=cfg["denoise_steps"], sample_batch_size=cfg["generation_batch_size"])
            if cfg["safe_runtime_optimizations"]:
                install_safe_sampler_optimization(generator.diffusion)
            counts.phase = "conditional_generation"
            generated = generator.sample(num_samples=cfg["num_samples"], cfg_scale=cfg["cfg_scale"])
            if not all(np.isfinite(value).all() for value in generated):
                raise FloatingPointError("Non-finite native generated transitions")
            if not set(np.unique(generated[-1])).issubset({0., 1.}):
                raise AssertionError("Generated terminals were not thresholded")
            for values in zip(*generated): agent.diffusion_buffer.store(*values)
            old_mixed = counts.mixed_minibatches
            # Smoke's last cycle is at budget end: explicitly execute one native
            # mixed update so readiness cannot be satisfied by REDQ alone.
            extra_updates = 1 if config.get("smoke", False) else 0
            for _ in range(extra_updates): agent.train(logger)
            cycle = dict(train_env_steps=real_steps, real_buffer_size=agent.replay_buffer.size,
                priority_top_count=len(priority_indices), priority_max_index=int(priority_indices.max()),
                condition_priority_mean=float(latest_cond.irews_buf[priority_indices].mean()),
                diffusion_gradient_steps=latest_trainer.step, diffusion_parameter_delta=diffusion_change,
                generated_model_transitions=len(generated[0]), synthetic_buffer_size=agent.diffusion_buffer.size,
                generated_terminal_count=int(generated[-1].sum()),
                native_mixed_minibatches_after_generation=counts.mixed_minibatches - old_mixed,
                generated_reward_min=float(generated[2].min()), generated_reward_max=float(generated[2].max()))
            nonterminal_generated = generated[-1] == 0
            nonterminal_generated_count = int(nonterminal_generated.sum())
            nonzero_generated_count = int((generated[2][nonterminal_generated] != 0).sum())
            cycle.update(generated_nonterminal_count=nonterminal_generated_count,
                generated_nonterminal_nonzero_reward_count=nonzero_generated_count,
                generated_nonterminal_nonzero_reward_rate=(nonzero_generated_count / nonterminal_generated_count
                                                           if nonterminal_generated_count else None),
                generated_rewards_are_model_predictions=True)
            diffusion_cycles.append(cycle)
            _dump(cycle_dir / "cycle_receipt.json", cycle)
            train_hook.remove(); ema_hook.remove()
        if terminated or step == cfg["total_steps"]:
            row = dict(train_env_steps=real_steps, episodes=episodes, critic_minibatches=counts.critic_minibatches,
                actor_updates=counts.policy_steps, curiosity_updates=counts.curiosity_steps,
                real_sample_references=counts.real_sample_references,
                synthetic_sample_references=counts.synthetic_sample_references,
                mixed_minibatches=counts.mixed_minibatches,
                episode_return=episode_return if terminated else None, **logger.last)
            train_log.write(json.dumps(row, allow_nan=False) + "\n"); train_log.flush()
            formal_progress(row)
            if terminated: episode_return = 0.0
            if (terminated and episodes % cfg["eval_every_episodes"] == 0) or step == cfg["total_steps"]:
                evaluations.append(_evaluate(agent, Env, config, cfg, real_steps))
                with (output / 'evaluations.jsonl').open('a') as handle:
                    handle.write(json.dumps(evaluations[-1], allow_nan=False) + '\n')
    train_log.close()
    assert real_steps == cfg["total_steps"] and nonterminal_nonzero == 0
    changed = {"policy": float((_vector(agent.policy_net) - initial["policy"]).norm()),
               "critic": float((_vector(agent.q_net_list[0]) - initial["critic"]).norm()),
               "curiosity": float((_vector(agent.cond_net) - initial["curiosity"]).norm())}
    if config.get("smoke", False):
        assert all(value > 0 for value in changed.values())
        assert diffusion_cycles and counts.mixed_minibatches > 0 and counts.synthetic_sample_references > 0
    assert all(torch.isfinite(p).all() for module in [agent.policy_net, agent.cond_net, *agent.q_net_list] for p in module.parameters())
    terminal_index = np.flatnonzero(agent.replay_buffer.done_buf[:agent.replay_buffer.size])[-1]
    rewards = torch.as_tensor(agent.replay_buffer.rews_buf[terminal_index:terminal_index+1, None], device=agent.device)
    next_states = torch.as_tensor(agent.replay_buffer.obs2_buf[terminal_index:terminal_index+1], device=agent.device)
    target, _ = agent.get_redq_q_target_no_grad(next_states, rewards, torch.ones_like(rewards))
    terminal_target_error = float((target - rewards).abs().max().cpu())
    assert terminal_target_error == 0.
    live_replay_bytes = sum(getattr(buffer, key)[:buffer.size].nbytes
        for buffer in (agent.replay_buffer, agent.diffusion_buffer)
        for key in ("obs1_buf", "obs2_buf", "acts_buf", "rews_buf", "done_buf"))
    print(f"PGR checkpoint: full_training_state={full_training_state}; live replay estimate={live_replay_bytes/1e9:.3f} GB")
    checkpoint = dict(agent=_save_agent(agent, full_training_state), config=config, native_config=cfg,
                      checkpoint_kind="training_state" if full_training_state else "inference_weights",
                      counters=dict(real_steps=real_steps, episodes=episodes),
                      rng=_rng(), diffusion=None)
    if latest_trainer is not None and full_training_state:
        checkpoint["diffusion"] = dict(model=latest_trainer.accelerator.get_state_dict(latest_trainer.model),
            optimizer=latest_trainer.opt.state_dict(), ema=latest_trainer.ema.state_dict(), step=latest_trainer.step,
            priority=latest_cond.irews_buf[:agent.replay_buffer.size], top_indices=latest_cond.top_frac_indices)
    checkpoint_path = output / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    restored = _agent(native, cfg, env, agent.device)
    _restore_agent(restored, loaded["agent"])
    probe = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        expected = agent.policy_net(probe, deterministic=True)[0]
        actual = restored.policy_net(probe, deterministic=True)[0]
    restore_error = float((expected - actual).abs().max().cpu())
    assert restore_error == 0.
    restored_logger = _Log()
    if full_training_state:
        restored.train(restored_logger)
        assert all(math.isfinite(value) for value in restored_logger.last.values())
    diffusion_restored = latest_trainer is None
    if latest_trainer is not None and full_training_state:
        with torch.no_grad():
            next(latest_trainer.model.net.parameters()).add_(1.)
            next(latest_trainer.ema.ema_model.net.parameters()).add_(1.)
        latest_trainer.model.load_state_dict(loaded["diffusion"]["model"])
        latest_trainer.opt.load_state_dict(loaded["diffusion"]["optimizer"])
        latest_trainer.ema.load_state_dict(loaded["diffusion"]["ema"])
        diffusion_restored = all(torch.equal(value, loaded["diffusion"]["model"][key])
                                 for key, value in latest_trainer.model.state_dict().items())
        assert diffusion_restored
    actual_sha = subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip()
    sub_sha = subprocess.check_output(["git", "-C", str(VENDOR / "synther" / "REDQ"), "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(VENDOR), "diff", "--name-only", "HEAD"], text=True).strip()
    assert actual_sha == PGR_SHA and sub_sha == REDQ_SHA and not diff
    _dump(output / "evaluations.json", evaluations)
    with (output / "evaluations.csv").open("w", newline="") as handle:
        columns = [key for key in evaluations[0] if key != "returns"]
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in evaluations)
    receipt = dict(method="PGR", domain=config["domain"], seed=config["seed"],
        status="smoke_passed" if config.get("smoke", False) else "run_completed", smoke=bool(config.get("smoke", False)),
        formal_training_performed=False if config.get("smoke", False) else True,
        official_commit=actual_sha, redq_submodule_commit=sub_sha, vendor_modified=False,
        compatibility=dict(in_memory_removed_unused_imports=native["dropped_imports"],
                           d4rl_dataset_access=False, mujoco_environment_access=False),
        real_steps=real_steps, train_env_steps=real_steps, completed_episodes=episodes,
        evaluation_env_steps=sum(row["evaluation_env_steps"] for row in evaluations),
        obs_dim=observation_dim, action_dim=action_dim, diffusion_transition_dim=transition_dim,
        nonterminal_nonzero_real_rewards=nonterminal_nonzero, real_terminal_feedbacks=real_terminals,
        terminal_td_target_abs_error=terminal_target_error, parameter_l2_changes=changed,
        critic_minibatches=counts.critic_minibatches, q_optimizer_steps=counts.q_optimizer_steps,
        actor_updates=counts.policy_steps, curiosity_updates=counts.curiosity_steps,
        real_sample_references=counts.real_sample_references,
        synthetic_sample_references=counts.synthetic_sample_references, mixed_minibatches=counts.mixed_minibatches,
        diffusion_training_steps=counts.diffusion_gradient_steps,
        denoiser_example_evaluations=counts.denoiser_example_evaluations,
        total_generated_model_transitions=sum(row["generated_model_transitions"] for row in diffusion_cycles),
        model_samples_count_as_real_interactions=False, diffusion_cycles=diffusion_cycles,
        checkpoint=str(checkpoint_path), checkpoint_action_abs_error=restore_error,
        restored_native_mixed_update_finite=(True if full_training_state else None),
        diffusion_checkpoint_exact=(diffusion_restored if full_training_state else None),
        checkpoint_contains_real_and_synthetic_buffers=full_training_state,
        checkpoint_kind="training_state" if full_training_state else "inference_weights",
        estimated_live_replay_checkpoint_bytes=live_replay_bytes,
        distinct_mechanism="ICM priority -> top-fraction conditions -> conditional diffusion -> generated replay -> mixed native RL update",
        curiosity_is_environment_reward=False,
        synthetic_rewards_are_model_predictions_not_environment_feedback=True,
        normalizer_note="Official minmax factory ignores skip_dims; terminal values inverse-transform to observed range then threshold at0.5",
        gamma=agent.gamma, target_entropy=agent.target_entropy, device=str(agent.device),
        native_config=cfg, wall_seconds=time.perf_counter() - started,
        safe_runtime_optimizations=cfg["safe_runtime_optimizations"],
        safe_runtime_optimization_scope="defer logging loss CPU read; move EMA to its device once; read native scalar schedule once; equations/dtype/RNG order unchanged",
        smoke_settings_are_performance_evidence=False)
    counts.close(); env.close(); _dump(output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("radio", "v5"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = vars(parser.parse_args())
    print(json.dumps(run(arguments), indent=2, ensure_ascii=False))
