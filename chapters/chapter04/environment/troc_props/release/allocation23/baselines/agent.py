"""Recent-paper agents for the existing chapter-2 experiment contract.

Algorithms retain their own update and deployment semantics.  Native dynamics,
public features and terminal rewards are supplied by Chapter2BaselineEnv.
Checkpoints contain replay, optimizers, pending on-policy data and random state.
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch

import hashlib

def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
from .cores.chapter2_runtime import capture_rng as _rng, restore_rng as _restore_rng, apply_execution, resolve_device
from .cores.chapter2_runtime import save_replay_archive
from .cores.common import recent_algorithm_parameters
from .cores.recent_td7_core import Agent as TD7Core, Hyperparameters
from .cores.dime import DIMECore, expanded_ch3_parameters
from .actions import (REVISION as ACTION_REVISION, reference_amount,
                      neutral_polytope_samples,
                      neutral_context, neutral_mean_tensor)

REVISION = "chapter23-external-cores-calibrated-actions-v2"
METHODS = ("paspo", "rpo", "td7", "simba_v2", "dime")


def algorithm_parameters(method, env, *, profile="formal", training_episodes=960):
    if method not in METHODS or profile not in ("formal", "smoke"):
        raise ValueError("unknown recent method or profile")
    if type(training_episodes) is not int or training_episodes < 1:
        raise ValueError("positive training episode budget required")
    tiny, h = profile == "smoke", int(env.horizon)
    if method == "paspo":
        from .cores.chapter2_paspo import paspo_parameters
        p = paspo_parameters(tiny=tiny, horizon=h)
    elif method == "rpo":
        from .cores.chapter2_rpo import rpo_parameters
        p = rpo_parameters(tiny=tiny, horizon=h)
    elif method == "simba_v2":
        from .cores.chapter2_simbav2 import simbav2_parameters
        p = simbav2_parameters(tiny=tiny, horizon=h, action_dim=env.action_space.shape[0])
    elif method == "td7":
        p = recent_algorithm_parameters("td7_full", tiny=tiny, horizon=h)
    else:
        p = expanded_ch3_parameters("dime_full", tiny=tiny, horizon=h)
        # Reward is a dimensionless terminal fraction, unlike the original
        # locomotion sum. Temperature is a declared task hyperparameter; retain
        # diffusion, categorical critic, BRN, tuning and UTD update equations.
        p["initial_temperature"] = .01
        # Fixed task-scale exploration calibration, shared across both chapters
        # and environments. The same prior_std enters initial draws, OU drift
        # and DIS density, so no unrecorded post-sampling noise scaling is used.
        p['prior_std'] = .5
    if method != "paspo":
        p["learning_starts"] = (2 if tiny else 8) * h
        p["buffer_size"] = max((training_episodes + 1)*h,
                               p["learning_starts"]+h, p["batch_size"])
    if method == "simba_v2":
        p["total_updates"] = max(1, (training_episodes*h-p["learning_starts"])
                                 * p["updates_per_step"])
    return p


class _TD7Adapter(TD7Core):
    def __init__(self, obs_dim, action_dim, p, device="cpu"):
        hp = Hyperparameters(batch_size=p["batch_size"], buffer_size=p["buffer_size"],
            discount=1., target_update_rate=p["target_update_rate"],
            exploration_noise=p["exploration_noise"], target_policy_noise=p["target_policy_noise"],
            noise_clip=p["noise_clip"], policy_freq=p["policy_freq"], alpha=p["priority_alpha"],
            min_priority=p["min_priority"], zs_dim=p["embedding_dim"], enc_hdim=p["hidden_dim"],
            critic_hdim=p["hidden_dim"], actor_hdim=p["hidden_dim"], encoder_lr=p["learning_rate"],
            critic_lr=p["learning_rate"], actor_lr=p["learning_rate"],
            steps_before_checkpointing=p["steps_before_checkpointing"],
            max_eps_when_checkpointing=p["max_eps_when_checkpointing"], reset_weight=p["reset_weight"])
        super().__init__(obs_dim, action_dim, 1., hp=hp, device=device)
        self.last_losses = {}

    def act(self, obs, deterministic=False):
        return self.select_action(np.asarray(obs), use_checkpoint=deterministic,
                                  use_exploration=not deterministic)

    def observe(self, obs, action, nxt, reward, done):
        self.replay_buffer.add(obs, action, nxt, reward, done)

    def state_dict(self):
        objects, values = {}, {}
        for name, value in vars(self).items():
            if isinstance(value, (torch.nn.Module, torch.optim.Optimizer)):
                objects[name] = copy.deepcopy(value.state_dict())
            elif isinstance(value, (float, int, bool, str, dict)):
                values[name] = copy.deepcopy(value)
        return dict(objects=objects, values=values)

    def load_state_dict(self, state):
        for name, value in state["objects"].items():
            getattr(self, name).load_state_dict(value)
        for name, value in state["values"].items():
            setattr(self, name, copy.deepcopy(value))


class _DIMEAdapter(DIMECore):
    @property
    def actor(self):
        return self.model.actor

    def act(self, obs, deterministic=False):
        if deterministic:
            raise ValueError("DIME deploys diffusion samples; zero-noise is not its policy mean")
        with torch.no_grad():
            return self.sample(torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
                               .reshape(1, -1), deterministic=False)[0].cpu().numpy()

    def observe(self, obs, action, nxt, reward, done):
        self.replay_buffer.add(obs, action, nxt, reward, done)

    def state_dict(self):
        # Keep raw replay in a separate compressed array artifact, never clone
        # an entire high-dimensional buffer into a model checkpoint.
        return dict(model=copy.deepcopy(self.model.state_dict()),
            optimizers={k: copy.deepcopy(v.state_dict()) for k,v in self.optimizers.items()},
            training_steps=self.training_steps, actor_updates=self.actor_updates,
            critic_updates=self.critic_updates, last_losses=copy.deepcopy(self.last_losses))


def _build_core(method, env, p, seed, device="cpu"):
    from .cores.neural_input import build_geometry_blind_core
    core = build_geometry_blind_core(method, env, p, seed, device,
                                    _TD7Adapter, _DIMEAdapter)
    if method == 'rpo':
        core.policy_warmup = getattr(env, 'chapter', 2) == 3
    if getattr(env,'chapter',2)==3 and method in ('paspo','rpo'):
        if method=='paspo':
            core.actor.neutral_context=neutral_context(env)
        else:
            core.neutral_context=neutral_context(env)
            core.neutral_context['constraint_unit']=core.cap_field['divisor']
    return core


class RecentChapter2Agent:
    def __init__(self, method, env, seed, *, profile="formal", training_episodes=960,
                 initialize=True, device="cpu", cpu_threads=1):
        if type(seed) is not int or seed < 0:
            raise ValueError("nonnegative integer seed required")
        if hasattr(env, 'configure_method'):
            env.configure_method(method)
        self.method, self.env, self.seed = method, env, seed
        self.profile, self.training_episodes = profile, training_episodes
        self.execution = apply_execution(device, cpu_threads)
        self.device = self.execution["device"]
        self.parameters = algorithm_parameters(method, env, profile=profile,
                                                training_episodes=training_episodes)
        self.contract = _json(env.get_contract())
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.core = _build_core(method, env, self.parameters, seed, self.device)
        self._steps, self._episode_return, self._episode_steps = 0, 0., 0
        self._last_obs, self._at_boundary = None, True
        self.model = None
        chapter3 = getattr(env, 'chapter', 2) == 3
        self.initialization = dict(action_revision=ACTION_REVISION,
            calibration=('unused-first neutral geometry prior with fixed dynamic public offset' if chapter3 and method in ('paspo','rpo')
                         else 'generic amount coordinate uses fixed public neutral scale'
                         if chapter3 else 'original chapter2 initialization'),
            reward_or_future_information=False,
            warmup=('calibrated actor samples' if chapter3 and method == 'rpo'
                    else 'original algorithm warmup'))
        if initialize and (method == "paspo" or chapter3 and method == 'rpo'):
            if env.train_path_base is None:
                raise ValueError("Baseline geometry priors must initialize on a training public state")
            self._last_obs = env.reset(seed=seed)[0]
            if method == 'paspo':
                samples = None
                if chapter3:
                    tensor = torch.as_tensor(self._last_obs, device=self.core.device).reshape(1,-1)
                    caps, target = self.core.actor.geometry(tensor)
                    q = reference_amount(env.decision_state, env.horizon)
                    r = q / float(target[0])
                    samples = neutral_polytope_samples(caps[0].cpu().numpy(), r,
                                                      self.parameters)
                    self.initialization.update(reference_amount=q, beta_concentration=96.,
                                               spatial_gamma_shape=8., first_training_state=True)
                self.core.initialize(self._last_obs, calibration_samples=samples)
            else:
                self._calibrate_rpo_initial_geometry()

    def _calibrate_rpo_initial_geometry(self):
        """Set only initial output biases from the first public training geometry."""
        core, env = self.core, self.env
        obs = torch.as_tensor(self._last_obs, dtype=torch.float64,
                              device=core.device).reshape(1,-1)
        q = reference_amount(env.decision_state, env.horizon)
        with torch.no_grad():
            caps,target=core.constraints(obs)
            mean=neutral_mean_tensor(obs,caps,target,core.neutral_context)
            core.actor.neutral_anchor_mean.copy_(mean[0])
            hidden = core.actor.encoder(core.obs_norm(obs.float()))
            core.actor.mean.bias.add_((mean-core.actor.mean(hidden))[0])
            core.actor.log_std.bias.sub_(core.actor.log_std(hidden)[0])
        self.initialization.update(reference_amount=q, first_training_state=True,
                                   dynamic_public_geometry_offset=True,
                                   initial_log_std=self.parameters['log_std_bias'])

    @property
    def num_timesteps(self):
        return self._steps

    @property
    def deployment_deterministic(self):
        return self.method != "dime"

    @property
    def metadata(self):
        return dict(revision=REVISION, method=self.method, seed=self.seed,
            profile=self.profile, training_episodes=self.training_episodes,
            parameters=copy.deepcopy(self.parameters), environment=self.contract,
            initialization=copy.deepcopy(self.initialization),
            neural_observation="local/global only; caps/budget reserved for feasible geometry",
            execution=copy.deepcopy(self.execution),
            deployment=("sampled diffusion policy" if self.method == "dime" else
                        "TD7 training-selected checkpoint" if self.method == "td7" else
                        self.parameters["deterministic_rule"] if self.method == "paspo" else
                        "deterministic actor"))

    @property
    def accounting(self):
        c = self.core
        losses = {k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)
                  for k,v in c.last_losses.items() if np.asarray(v.detach().cpu() if isinstance(v,torch.Tensor) else v).size == 1}
        if not all(np.isfinite(v) for v in losses.values()):
            raise RuntimeError("nonfinite recent-agent diagnostics")
        result = dict(environment_steps=self._steps, complete_episodes=self._steps//self.env.horizon,
            training_updates=int(c.training_steps), losses=losses, simulated_auxiliary_steps=0,
            profile=self.profile)
        if self.method == "td7":
            result.update(critic_updates=c.training_steps, encoder_updates=c.training_steps,
                actor_updates=c.training_steps//c.hp.policy_freq,
                pending_td7_updates=c.timesteps_since_update)
        else:
            for name in ("actor_updates", "critic_updates", "temperature_updates", "optimizer_steps"):
                if hasattr(c, name): result[name] = int(getattr(c, name))
            if self.method == "rpo":
                result.update(critic_updates=c.training_steps, dual_updates=c.actor_updates,
                              temperature_updates=c.actor_updates,
                              optimizer_steps=c.training_steps+3*c.actor_updates)
            elif self.method == "dime":
                result.update(temperature_updates=c.actor_updates,
                              optimizer_steps=c.critic_updates+2*c.actor_updates)
            elif self.method == "paspo":
                result["joint_policy_value_optimizer_steps"] = c.training_steps
        return result

    def predict(self, observation, state=None, episode_start=None, deterministic=None):
        deterministic = self.deployment_deterministic if deterministic is None else bool(deterministic)
        # Geometry must reach PASPO/RPO before any neural-network FP32 cast.
        obs = np.asarray(observation, dtype=np.float64 if self.method in ("paspo", "rpo") else np.float32)
        single = obs.ndim == 1
        obs = obs.reshape(-1, self.env.observation_space.shape[0])
        values = np.stack([self.core.act(row, deterministic=deterministic) for row in obs])
        if not np.isfinite(values).all() or np.any(np.abs(values)>1.):
            raise RuntimeError("invalid recent-policy controls")
        return (values[0] if single else values), state

    def learn(self, total_timesteps, reset_num_timesteps=False, episode_callback=None):
        h, p = self.env.horizon, self.parameters
        if not self._at_boundary or (reset_num_timesteps and self._steps):
            raise ValueError("resume requires a full-state training load")
        if type(total_timesteps) is not int or total_timesteps < 1 or total_timesteps % h:
            raise ValueError("learn requires complete episodes")
        if self.method == "paspo" and total_timesteps % p["n_steps"]:
            raise ValueError("PASPO requires complete on-policy rollouts")
        if self._steps+total_timesteps > self.training_episodes*h:
            raise ValueError("requested training exceeds registered budget")
        self._at_boundary = False
        if self._last_obs is None:
            self._last_obs = self.env.reset(seed=self.seed)[0]
        for _ in range(total_timesteps):
            obs = np.asarray(self._last_obs, dtype=np.float64 if self.method in ("paspo", "rpo") else np.float32)
            warm = self._steps < p.get("learning_starts", 0)
            if warm and self.method not in ("rpo", "paspo"):
                action = np.random.uniform(-1., 1., self.env.action_space.shape).astype(np.float32)
            else:
                action = self.core.act(obs, deterministic=False)
            nxt, reward, done, truncated, _ = self.env.step(action)
            if truncated:
                raise RuntimeError("native task must terminate, not truncate")
            self.core.observe(obs, action, nxt, reward, done)
            self._steps += 1
            self._episode_steps += 1
            self._episode_return += reward
            if self.method == "paspo":
                if self._steps % p["n_steps"] == 0: self.core.train()
            elif self.method != "td7" and self._steps > p["learning_starts"]:
                for _update in range(p.get("updates_per_step", 1)):
                    self.core.train()
            if done:
                if self.method == "td7" and not warm:
                    self.core.maybe_train_and_checkpoint(self._episode_steps, self._episode_return)
                if episode_callback is not None:
                    # Reporting consumes no policy samples or additional transitions.
                    episode_callback(self._steps, copy.deepcopy(self.env.episode_records[-1]))
                self._episode_steps, self._episode_return = 0, 0.
                self._last_obs = self.env.reset()[0]
            else:
                self._last_obs = nxt
        self._at_boundary = True
        self.accounting  # Reject nonfinite diagnostics before a checkpoint.
        return self

    def health_diagnostics(self):
        """Read-only last-backward diagnostics; no sampling or extra updates.

        Gradients are the currently retained buffers, not a running average or
        an assertion about the full training trajectory. A missing gradient is
        explicitly recorded (e.g. before warmup ends).
        """
        result = {}
        for name in ("actor", "critic"):
            module = getattr(self.core, name, None)
            if module is None and hasattr(self.core, "model"):
                module = getattr(self.core.model, name, None)
                if module is None and name == "critic":
                    module = getattr(self.core.model, "critics", None)
            if not isinstance(module, torch.nn.Module):
                continue
            parameters = list(module.parameters())
            gradients = [p.grad.detach() for p in parameters if p.grad is not None]
            with torch.no_grad():
                weights = [p.detach() for p in parameters]
                values = dict(parameter_count=sum(p.numel() for p in weights),
                    gradient_parameter_count=sum(g.numel() for g in gradients),
                    has_gradient=bool(gradients),
                    parameter_absmax=float(torch.stack([p.abs().max() for p in weights]).max().cpu()))
                if gradients:
                    values.update(gradient_l2=float(torch.stack([g.double().square().sum() for g in gradients]).sum().sqrt().cpu()),
                                  gradient_absmax=float(torch.stack([g.abs().max() for g in gradients]).max().cpu()))
            if not all(np.isfinite(v) for v in values.values()):
                raise RuntimeError("nonfinite " + name + " health diagnostics")
            result[name] = values
        return result


def make_agent(method, env, seed, *, profile="formal", training_episodes=960,
               device="cpu", cpu_threads=1):
    return RecentChapter2Agent(method, env, seed, profile=profile, training_episodes=training_episodes,
                               device=device, cpu_threads=cpu_threads)


def agent_parameter_vector(agent):
    return torch.cat([p.detach().cpu().reshape(-1) for p in agent.core.actor.parameters()]).numpy().copy()


def _sources():
    # The environment adapter and native task are scientific inputs too.
    from ..io import source_hashes
    return source_hashes()


def _write_replay(core, folder):
    r = getattr(core, "replay_buffer", None)
    if r is None: return None
    arrays = {f"array_{i}": a[:r.size] for i,a in enumerate(r.arrays)}
    if r.prioritized: arrays["priority"] = r.priority[:r.size].cpu().numpy()
    save_replay_archive(folder/"replay.npz", arrays)
    return dict(size=r.size, ptr=r.ptr, capacity=r.max_size, prioritized=r.prioritized,
                batch_size=r.batch_size, max_priority=r.max_priority)


def _read_replay(core, folder, meta):
    r = getattr(core, "replay_buffer", None)
    if (meta is None) != (r is None): raise ValueError("replay presence differs")
    if r is None: return
    if (meta["capacity"] != r.max_size or meta["prioritized"] != r.prioritized
            or meta["batch_size"] != r.batch_size or not 0<=meta["size"]<=r.max_size
            or not 0<=meta["ptr"]<r.max_size):
        raise ValueError("invalid replay metadata")
    with np.load(folder/"replay.npz", allow_pickle=False) as stored:
        expected_keys = {f"array_{i}" for i in range(len(r.arrays))}
        if r.prioritized:
            expected_keys.add("priority")
        if set(stored.files) != expected_keys:
            raise ValueError("replay array schema differs (including exact geometry)")
        # Reject a legacy rounded replay instead of silently promoting it to
        # the new precise schema. Validate all arrays before writing any.
        for i, dest in enumerate(r.arrays):
            source = stored[f"array_{i}"]
            if source.dtype != dest.dtype or source.shape != dest[:meta["size"]].shape:
                raise ValueError("replay array shape or dtype differs")
        for i,dest in enumerate(r.arrays): dest[:meta["size"]] = stored[f"array_{i}"]
        if r.prioritized: r.priority[:meta["size"]] = torch.as_tensor(stored["priority"], device=r.device)
    r.size, r.ptr, r.max_priority = meta["size"], meta["ptr"], meta["max_priority"]


def _clear_snapshot_optimizer_moments(value):
    """Trim standard optimizer states only inside an isolated core snapshot."""
    if isinstance(value, dict):
        if isinstance(value.get('state'), dict) and isinstance(value.get('param_groups'), list):
            value['state'] = {}
            return 1
        return sum(_clear_snapshot_optimizer_moments(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_clear_snapshot_optimizer_moments(child) for child in value)
    return 0


def dump_agent(agent, path, env, identity, *, include_replay=True):
    if env is not agent.env or _json(env.get_contract()) != agent.contract:
        raise ValueError("checkpoint environment mismatch")
    if (not agent._at_boundary or agent.num_timesteps % env.horizon
            or env.environment_steps != agent.num_timesteps
            or (agent.method == "paspo" and agent.num_timesteps % agent.parameters["n_steps"])):
        raise ValueError("checkpoint requires complete episode/rollout boundary")
    path = Path(path)
    if path.exists(): raise FileExistsError("checkpoint cannot be overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".recent-agent-", dir=path.parent))
    try:
        resume_available = bool(include_replay or getattr(agent.core, 'replay_buffer', None) is None)
        core_snapshot = agent.core.state_dict()
        moments_retained = resume_available
        trimmed_optimizers = 0
        if not moments_retained:
            # Full checkpoints and PASPO retain every training state. A final
            # off-policy evaluation checkpoint has already omitted replay and
            # cannot resume; its Adam moments are irrelevant to prediction.
            core_snapshot = copy.deepcopy(core_snapshot)
            trimmed_optimizers = _clear_snapshot_optimizer_moments(core_snapshot)
        state = dict(schema=REVISION, method=agent.method, seed=agent.seed, profile=agent.profile,
            training_episodes=agent.training_episodes, parameters=agent.parameters, execution=agent.execution,
            core=core_snapshot, replay=_write_replay(agent.core,temp) if include_replay else None,
            training_resume_available=resume_available, include_replay=bool(include_replay),
            optimizer_moments_retained=moments_retained,
            optimizer_state_dicts_trimmed=trimmed_optimizers,
            environment=env.export_state(), rng=_rng(), steps=agent._steps,
            last_obs=agent._last_obs, episode_return=agent._episode_return,
            episode_steps=agent._episode_steps, initialization=agent.initialization)
        torch.save(state, temp/"state.pt")
        manifest = dict(schema=REVISION, identity=_json(identity), environment=agent.contract,
            training_resume_available=resume_available, include_replay=bool(include_replay),
            optimizer_moments_retained=moments_retained,
            optimizer_state_dicts_trimmed=trimmed_optimizers,
            execution=agent.execution,
            method=agent.method, profile=agent.profile, source_hashes=_sources(),
            software={p: importlib.metadata.version(p) for p in ("torch","numpy","gymnasium")},
            accounting=agent.accounting, files={p.name:_hash(p) for p in temp.iterdir() if p.is_file()})
        (temp/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
        os.rename(temp, path)
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True)
        raise
    return manifest


def load_agent(path, env, expected_identity, restore_rng=False, device=None):
    path = Path(path)
    manifest = json.loads((path/"manifest.json").read_text())
    if (manifest.get("schema") != REVISION or manifest.get("identity") != _json(expected_identity)
            or manifest.get("environment") != _json(env.get_contract())):
        raise ValueError("checkpoint schema, identity or environment differs")
    if manifest["source_hashes"] != _sources(): raise ValueError("checkpoint implementation differs")
    if manifest["software"] != {p:importlib.metadata.version(p) for p in manifest["software"]}:
        raise ValueError("checkpoint runtime differs")
    if restore_rng and not manifest.get('training_resume_available', True):
        raise ValueError('This evaluation checkpoint omits replay; exact training continuation is unavailable')
    for name,digest in manifest["files"].items():
        if Path(name).name != name or _hash(path/name)!=digest: raise ValueError("checkpoint checksum mismatch")
    before, env_before, success = _rng(), env.export_state(), False
    try:
        saved = torch.load(path/"state.pt", map_location="cpu", weights_only=False)
        if saved.get('training_resume_available') != manifest.get('training_resume_available'):
            raise ValueError('Checkpoint continuation availability differs between state and manifest')
        if saved.get('optimizer_moments_retained') != manifest.get('optimizer_moments_retained'):
            raise ValueError('Checkpoint optimizer retention differs between state and manifest')
        saved_device = saved["execution"]["device"]
        requested_device = saved_device if device is None else resolve_device(device)
        if restore_rng and requested_device != saved_device:
            raise ValueError("exact training continuation requires the saved execution device")
        agent = RecentChapter2Agent(saved["method"],env,saved["seed"],profile=saved["profile"],
                                    training_episodes=saved["training_episodes"],initialize=False,
                                    device=requested_device,cpu_threads=saved["execution"]["cpu_threads"])
        if agent.parameters != saved["parameters"]: raise ValueError("algorithm settings differ")
        agent.core.load_state_dict(saved["core"])
        agent.initialization = copy.deepcopy(saved['initialization'])
        # Prediction never samples replay. Still verify every artifact checksum
        # above; only a full training continuation decompresses/copies the buffer.
        if restore_rng:
            _read_replay(agent.core,path,saved["replay"])
        agent._steps, agent._last_obs = saved["steps"],saved["last_obs"]
        agent._episode_steps, agent._episode_return = saved["episode_steps"],saved["episode_return"]
        if restore_rng:
            env.import_state(saved["environment"]); _restore_rng(saved["rng"])
        else: agent._at_boundary = False
        success = True
        return agent
    finally:
        if not restore_rng or not success:
            env.import_state(env_before); _restore_rng(before)
