"""Isolated, runnable CAL candidate; no active chapter registry is changed.

ICLR 2024, Off-Policy Primal-Dual Safe Reinforcement Learning.
The pinned MIT author core supplies every learning update. This wrapper binds
device, current public neural inputs, true episode termination, resource cost,
isolated RNG, complete-episode collection and durable save/load.

This is online expected-CMDP learning, not a pathwise budget guarantee. With the
native feasible decoder and cost=executed resource/initial budget, the additional
cost constraint is redundant for an exact critic. It cannot establish a separate
advantage of CAL's constraint mechanism. Native terminal rewards are unchanged.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
from types import SimpleNamespace, MethodType
import uuid

import numpy as np
import torch
from gymnasium.spaces import Box


REVISION = "chapter3-cal-candidate-v1"
VENDOR = Path(__file__).parent / "vendor/cal"
MODULES = ("critic", "critic_target", "safety_critics", "safety_critic_targets", "policy")
OPTIMIZERS = ("actor_optimizer", "critic_optimizer", "safety_critic_optimizer",
              "log_alpha_optimizer", "log_lam_optimizer", "alpha_optim")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scientific_sources():
    base = Path(__file__).resolve().parents[1]
    paths = [base / name for name in ("task.py", "spec.py", "state.py", "io.py", "full_spend.py",
                                      "image_data.py", "methods/constraints.py")]
    paths += [p for p in (base / "environments").rglob("*")
              if p.is_file() and p.suffix in (".py", ".json")]
    return {str(p.relative_to(base)): _sha(p) for p in sorted(paths)}


def _software():
    return {name: importlib.metadata.version(name) for name in ("torch", "numpy", "gymnasium", "scipy")}


def provenance():
    """Verify original author files before compiling any candidate core."""
    manifest = json.loads((VENDOR / "SOURCE_MANIFEST.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if _sha(VENDOR / "upstream" / name) != expected:
            raise ValueError("CAL pinned upstream source differs: " + name)
    return copy.deepcopy(manifest)


def _upstream_class(device, *, stable_tanh_logprob=False):
    provenance()
    source = VENDOR / "upstream"
    ns = {"__name__": "_allocation23_chapter3_cal_author_core"}
    for name in ("agent/__init__.py", "agent/utils.py", "agent/model.py"):
        exec(compile((source / name).read_text(), str(source / name), "exec"), ns)
    if stable_tanh_logprob:
        ns["GaussianPolicy"].sample = _stable_gaussian_sample
    path = source / "agent/cal.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module in
        ("agent", "agent.utils", "agent.model"))]
    substitutions = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "self"
                and node.targets[0].attr == "device"
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value) == "torch.device('cuda')"):
            node.value.args = [ast.Constant(str(device))]
            substitutions += 1
    if substitutions != 1:
        raise ValueError("Pinned CAL device assignment no longer matches")
    # Retain the exact native scalar losses without changing their graphs or math.
    class CaptureLosses(ast.NodeTransformer):
        labels={"critic_loss":"reward_critic","safety_critic_loss":"cost_critic",
                "actor_loss":"actor","alpha_loss":"temperature","lam_loss":"multiplier"}
        count=0
        def visit_Assign(self,node):
            if (len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                    and node.targets[0].id in self.labels):
                name=node.targets[0].id
                extra=ast.parse("self._diagnostic_losses["+repr(self.labels[name])+"]= "+name+".detach()").body[0]
                self.count+=1
                return [node,extra]
            return node
    capture=CaptureLosses();tree=capture.visit(tree)
    if capture.count!=5:raise ValueError("Pinned CAL loss diagnostics no longer match")
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    exec(compile((source / "agent/replay_memory.py").read_text(),
                 str(source / "agent/replay_memory.py"), "exec"), ns)
    return ns["CALAgent"], ns["ReplayMemory"]



def _stable_gaussian_sample(policy, state):
    """Same squashed Gaussian, evaluated with its exact stable Jacobian.

    No epsilon/inverse-action reconstruction; raw samples remain available.
    Pinned author source remains unmodified and provenance-verified.
    """
    mean, log_std = policy.forward(state)
    normal = torch.distributions.Normal(mean, log_std.exp())
    raw = normal.rsample()
    action = raw.tanh() * policy.action_scale + policy.action_bias
    log_jac = 2. * (np.log(2.) - raw - torch.nn.functional.softplus(-2. * raw))
    log_prob = (normal.log_prob(raw) - log_jac - torch.log(policy.action_scale)).sum(1, keepdim=True)
    mean_action = mean.tanh() * policy.action_scale + policy.action_bias
    return action, log_prob, mean_action


def _shared_policy_forward(policy,state):
    x=torch.relu(policy.linear1(state))
    x=torch.relu(policy.linear2(x))
    log_std=policy.log_std_linear(x).clamp(min=-20.,max=policy.log_sig_max)
    return policy.shared_mean(state),log_std

def _shared_policy_deterministic_mean(policy,state):
    return policy.shared_mean(state).tanh()*policy.action_scale+policy.action_bias


@dataclass(frozen=True)
class CALConfig:
    """Declared current-task adaptations; official defaults are retained otherwise.

    Native episodic objective requires reward/cost gamma=1 and genuine done
    masks. Setting safetygym=False avoids the author's continuing-task branch
    that overwrites all cost terminal masks with one. No cost/reward shaping.
    Collection schedule is explicitly this wrapper's online protocol, not an
    assertion of reproducing the author's 1000-step evaluation/reset loop.
    """
    gamma: float = 1.0
    safety_gamma: float = 1.0
    hidden_size: int = 256
    qc_ens_size: int = 4
    M: int = 4
    intrgt_max: bool = False
    k: float = 0.5
    c: float = 10.0
    lr: float = 3e-4
    qc_lr: float = 3e-4
    tau: float = 0.005
    critic_target_update_frequency: int = 2
    cost_lim: float = 1.0
    replay_size: int = 1_000_000
    init_exploration_steps: int = 5000
    min_pool_size: int = 1000
    policy_train_batch_size: int = 12
    num_train_repeat: int = 10
    # Independent adaptation candidates; defaults retain the original experiment.
    reward_scale: float = 1.0
    initial_alpha: float = 1.0
    target_entropy_scale: float = 1.0
    stable_tanh_logprob: bool = False
    actor_architecture: str = "flat"

    def __post_init__(self):
        if self.actor_architecture not in ("flat","shared_objects"):
            raise ValueError("Unknown explicit actor mean architecture")
        if (type(self.stable_tanh_logprob) is not bool
                or min(self.reward_scale,self.initial_alpha,self.target_entropy_scale)<=0):
            raise ValueError("Positive reward/entropy scales and explicit stable-density flag required")
        for name in ("hidden_size", "qc_ens_size", "M", "replay_size",
                     "policy_train_batch_size", "num_train_repeat",
                     "critic_target_update_frequency"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive integer required: " + name)
        for name in ("init_exploration_steps", "min_pool_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("Nonnegative integer required: " + name)
        if self.qc_ens_size < 2 or self.M > self.qc_ens_size:
            raise ValueError("Retain a nontrivial cost ensemble and M<=ensemble")
        if (not np.isfinite([x for x in asdict(self).values() if isinstance(x,(float,int,bool))]).all()
                or not 0 <= self.gamma <= 1 or not 0 <= self.safety_gamma <= 1
                or not 0 < self.tau <= 1 or min(self.lr, self.qc_lr, self.c) <= 0
                or self.k < 0 or self.cost_lim != 1.0):
            raise ValueError("Invalid CAL parameters or changed native normalized budget")

    def namespace(self):
        return SimpleNamespace(**asdict(self), safetygym=False, epoch_length=1,
                               env_name="allocation23-chapter3-candidate")


def resource_cost(physical_action, spec):
    """Executed native resource / initial resource; both units cancel exactly.

    In radio, numerator and denominator would both multiply by slot seconds.
    The cost is separate from, and never added to, the original terminal reward.
    """
    action = np.asarray(physical_action, dtype=np.float64)
    if action.shape != (spec.groups,) or not np.isfinite(action).all() or np.any(action < 0):
        raise ValueError("Finite nonnegative native allocation required")
    return float(action.sum() / spec.initial_budget)


class CALCandidate:
    def __init__(self, observation_dim, action_dim, seed=0, *, device="cpu", config=None):
        if any(type(v) is not int or v < 1 for v in (observation_dim, action_dim)):
            raise ValueError("Positive policy observation/action dimensions required")
        if type(seed) is not int or seed < 0:
            raise ValueError("Nonnegative registered seed required")
        self.observation_dim, self.action_dim, self.seed = observation_dim, action_dim, seed
        self.config = config or CALConfig()
        self.device = str(torch.device(device))
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("Only explicit CPU or CUDA execution is supported")
        self._python_rng = random.Random(seed).getstate()
        self._numpy_rng = np.random.RandomState(seed).get_state()
        self._torch_rng = torch.Generator(device="cpu").manual_seed(seed).get_state()
        self._cuda_rng = (torch.Generator(device=self.device).manual_seed(seed).get_state()
                          if self.device.startswith("cuda") else None)
        cls, replay_cls = _upstream_class(self.device, stable_tanh_logprob=self.config.stable_tanh_logprob)
        with self._rng_scope():
            self.core = cls(observation_dim, Box(-1., 1., (action_dim,), dtype=np.float32),
                            self.config.namespace())
            if self.config.actor_architecture == "shared_objects":
                from .generic_shared_actor import SharedObjectMean
                # The author core has already consumed its original initialization.
                # Isolate the extra mean initializer so all critics/std/noise match.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(seed)+47013)
                    self.core.policy.shared_mean=SharedObjectMean(observation_dim,action_dim).to(self.device)
                self.core.policy.forward=MethodType(_shared_policy_forward,self.core.policy)
                self.core.policy.get_a_mean=MethodType(_shared_policy_deterministic_mean,self.core.policy)
                self.core.actor_optimizer=torch.optim.Adam(self.core.policy.parameters(),lr=self.config.lr)
            self.replay = replay_cls(self.config.replay_size)
            self.core._diagnostic_losses = {}
            with torch.no_grad():
                self.core.log_alpha.fill_(np.log(self.config.initial_alpha))
            self.core.target_entropy *= self.config.target_entropy_scale
        self.last_update_diagnostics = None
        self.environment_steps = self.complete_episodes = self.target_syncs = 0
        self.supervised_cost_head_examples = 0
        self.contract = None
        self._training_env = None
        self._pending_environment = None
        self._resume_training_available = True

    @contextmanager
    def _rng_scope(self):
        """Keep the original global-RNG author updates inside one private stream.

        Each candidate must run in its own process; this scope is not thread safe.
        No CUDA API is called on the CPU path.
        """
        before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
        cuda_before = (torch.cuda.get_rng_state(self.device)
                       if self._cuda_rng is not None else None)
        random.setstate(self._python_rng)
        np.random.set_state(self._numpy_rng)
        torch.random.set_rng_state(self._torch_rng)
        if self._cuda_rng is not None:
            torch.cuda.set_rng_state(self._cuda_rng, self.device)
        try:
            yield
        finally:
            self._python_rng, self._numpy_rng = random.getstate(), np.random.get_state()
            self._torch_rng = torch.random.get_rng_state()
            if self._cuda_rng is not None:
                self._cuda_rng = torch.cuda.get_rng_state(self.device)
                torch.cuda.set_rng_state(cuda_before, self.device)
            random.setstate(before[0])
            np.random.set_state(before[1])
            torch.random.set_rng_state(before[2])

    def _observation(self, value):
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (self.observation_dim,) or not np.isfinite(value).all():
            raise ValueError("Finite reduced public neural observation required")
        return value

    def act(self, observation, deterministic=False):
        observation = self._observation(observation)
        with torch.no_grad():
            if deterministic:
                # Exactly the author's returned deterministic mean; skip the
                # unused stochastic draw in select_action(eval=True).
                result = self.core.policy.get_a_mean(torch.as_tensor(
                    observation, device=self.device).unsqueeze(0))[0].cpu().numpy()
            else:
                with self._rng_scope():
                    result = self.core.select_action(observation, eval=False)
        if not np.isfinite(result).all() or np.any(np.abs(result) > 1):
            raise ArithmeticError("CAL produced invalid bounded controls")
        return result.copy()

    def observe(self, observation, action, reward, cost, next_observation, done):
        if not self._resume_training_available:
            raise ValueError("Evaluation-only CAL state cannot continue training")
        obs, nxt = self._observation(observation), self._observation(next_observation)
        action = np.asarray(action, dtype=np.float32)
        if (action.shape != (self.action_dim,) or not np.isfinite(action).all()
                or np.any(np.abs(action) > 1) or not np.isfinite([reward, cost]).all()
                or cost < 0 or not isinstance(done, (bool, np.bool_))):
            raise ValueError("Invalid native online transition")
        self.replay.push(obs.copy(), action.copy(), np.asarray([reward, cost], np.float32),
                         nxt.copy(), bool(done))
        self.environment_steps += 1
        self.complete_episodes += int(done)

    def update_batch(self, states, actions, rewards, costs, next_states, dones, *, update_index=0):
        """One complete, unchanged native critic/ensemble/actor/alpha/dual update."""
        if not self._resume_training_available:
            raise ValueError("Evaluation-only CAL state cannot continue training")
        s, n, a = (np.asarray(x, dtype=np.float32) for x in (states, next_states, actions))
        r, c = (np.asarray(x, dtype=np.float32).reshape(-1) for x in (rewards, costs))
        d = np.asarray(dones)
        batch = len(s)
        if (batch < 1 or s.shape != (batch, self.observation_dim) or n.shape != s.shape
                or a.shape != (batch, self.action_dim) or r.shape != (batch,)
                or c.shape != r.shape or d.shape != r.shape or d.dtype.kind != "b"
                or any(not np.isfinite(x).all() for x in (s, n, a, r, c))
                or np.any(np.abs(a) > 1) or np.any(c < 0)
                or type(update_index) is not int or update_index < 0):
            raise ValueError("Invalid native CAL minibatch")
        with self._rng_scope():
            self.core.update_parameters((s, a, np.column_stack((r*self.config.reward_scale, c)), n,
                                         (~d).astype(np.int64)), update_index)
        self.target_syncs += int(update_index % self.config.critic_target_update_frequency == 0)
        self.supervised_cost_head_examples += batch * self.config.qc_ens_size
        self.health_diagnostics()
        if self.core.update_counter % 128 == 0:
            with torch.no_grad():
                st=torch.as_tensor(s,device=self.device)
                at=torch.as_tensor(a,device=self.device)
                mean,ls=self.core.policy(st)
                q1,q2=self.core.critic(st,at)
                qc=self.core.safety_critics(st,at)
                self.last_update_diagnostics=dict(
                    exact_native_losses={k:float(v) for k,v in self.core._diagnostic_losses.items()},
                    updates=self.core.update_counter,alpha=float(self.core.alpha),
                    multiplier=float(self.core.lam),rectification=float(self.core.rect),
                    target_entropy=float(self.core.target_entropy),
                    raw_mean_abs=float(mean.abs().mean()),log_std_mean=float(ls.mean()),
                    mean_action_saturation_fraction=float((mean.tanh().abs()>.999).float().mean()),
                    reward_q_mean=float(torch.minimum(q1,q2).mean()),cost_q_mean=float(qc.mean()),
                    native_batch_nonzero_fraction=float(np.mean(r!=0.)),
                    native_batch_reward_mean=float(np.mean(r)),reward_scale=self.config.reward_scale)
        return self.accounting

    def train_updates(self, count=None):
        if not self._resume_training_available:
            raise ValueError("Evaluation-only CAL state cannot continue training")
        count = self.config.num_train_repeat if count is None else count
        if type(count) is not int or count < 1 or len(self.replay) < 1:
            raise ValueError("Positive repeated updates and nonempty replay required")
        for index in range(count):
            with self._rng_scope():
                s, a, rc, n, d = self.replay.sample(self.config.policy_train_batch_size)
            self.update_batch(s, a, rc[:, 0], rc[:, 1], n, d.astype(bool), update_index=index)
        return self.accounting

    @property
    def ready_to_update(self):
        return (self.environment_steps > self.config.init_exploration_steps
                and len(self.replay) > self.config.min_pool_size)

    def collect_transition(self, observation, action, reward, cost, next_observation, done):
        """One collector step; official CAL warmup samples its Gaussian policy.

        init_exploration_steps is a collection-without-updates gate, not a
        uniform-action switch. See upstream main.exploration_before_start and
        MuJoCoEnvSampler.sample. The caller obtains actions through act().
        """
        self.observe(observation, action, reward, cost, next_observation, done)
        if self.ready_to_update:
            self.train_updates()
        return self.accounting

    def learn(self, env, episodes):
        """Online complete episodes on an isolated native training/engineering env.

        No independent test split is opened. Evaluation belongs to another env.
        The caller registers scientific budgets before any formal experiment.
        """
        if not self._resume_training_available:
            raise ValueError("Evaluation-only CAL state cannot continue training")
        if (type(episodes) is not int or episodes < 1 or env.chapter != 3
                or env.holdout or env.phase not in ("train", "engineering")
                or env.action_family != "total_space"):
            raise ValueError("Isolated chapter3 native training/engineering env required")
        contract = json.loads(json.dumps(env.get_contract()))
        if (len(env.policy_indices) != self.observation_dim or env.action_dim != self.action_dim
                or self.contract is not None and contract != self.contract):
            raise ValueError("Native CAL environment binding differs")
        if self._pending_environment is not None:
            env.import_state(self._pending_environment)
            self._pending_environment = None
        self.contract, self._training_env = contract, env
        rows = []
        for _ in range(episodes):
            if env.train_path_base is None:
                obs, _ = env.reset(seed=self.complete_episodes)
            else:
                obs, _ = env.reset()
            reward_sum = cost_sum = 0.
            for t in range(env.horizon):
                policy_obs = obs[env.policy_indices]
                control = self.act(policy_obs)
                nxt, reward, done, truncated, info = env.step(control)
                if truncated or done != (t == env.horizon - 1) or (not done and reward != 0.):
                    raise RuntimeError("Native terminal-only episode contract differs")
                cost = resource_cost(info["physical_action"], env.spec)
                self.collect_transition(policy_obs, control, reward, cost, nxt[env.policy_indices], done)
                reward_sum += reward
                cost_sum += cost
                obs = nxt
            if cost_sum > 1. + 1e-12:
                raise ArithmeticError("Native decoder violated original total budget")
            replay = env.replay_last_episode()
            if replay["error"] != 0.:
                raise ArithmeticError("CAL native action replay differs")
            rows.append(dict(episode=self.complete_episodes, native_return=reward_sum,
                             cost=cost_sum, native_replay_error=replay["error"],
                             accounting=self.accounting))
        return rows

    @property
    def accounting(self):
        n = self.core.update_counter
        return dict(environment_steps=self.environment_steps, complete_episodes=self.complete_episodes,
                    actor_updates=n, reward_critic_updates=n, cost_ensemble_optimizer_steps=n,
                    cost_head_count=self.config.qc_ens_size,
                    cost_head_supervised_examples=self.supervised_cost_head_examples,
                    alpha_updates=n, multiplier_updates=n, target_syncs=self.target_syncs,
                    replay_transitions=len(self.replay), simulated_auxiliary_steps=0)

    @property
    def metadata(self):
        return dict(revision=REVISION, method="CAL", candidate_only=True, seed=self.seed,
                    parameters=asdict(self.config), source=provenance(), device=self.device,
                    native_reward="unchanged terminal reward", cost="executed resource / initial budget",
                    internal_reward_scale=self.config.reward_scale,
                    initial_alpha=self.config.initial_alpha,
                    target_entropy_scale=self.config.target_entropy_scale,
                    stable_tanh_logprob=self.config.stable_tanh_logprob,
                    actor_architecture=self.config.actor_architecture,
                    shared_mean_hidden=(64,32) if self.config.actor_architecture=="shared_objects" else None,
                    neural_input="current local/global only; no caps or budget geometry fields",
                    expected_cmdp_not_pathwise_guarantee=True,
                    exact_cost_constraint_redundant_under_native_feasible_decoder=True,
                    objective_adaptation="true episodic reward/cost gamma1 and terminal masks",
                    collection_schedule="online episodes; declared warmup/min-pool/repeat blocks",
                    warmup="author Gaussian policy collection; init_exploration_steps gates updates only",
                    endpoint_persistence="one evaluation model; no disk replay or optimizer state",
                    deterministic_evaluation="author policy mean without unused random sample",
                    active_registry_changed=False)

    def health_diagnostics(self):
        if self.device.startswith("cuda"):
            return self._cuda_health_diagnostics()
        result = {}
        for name in ("policy", "critic", "safety_critics"):
            params = list(getattr(self.core, name).parameters())
            grads = [p.grad.detach() for p in params if p.grad is not None]
            if any(not torch.isfinite(p).all() for p in params + grads):
                raise FloatingPointError("Nonfinite CAL parameter/gradient: " + name)
            result[name] = dict(parameters=sum(p.numel() for p in params), has_gradient=bool(grads))
        if not torch.isfinite(self.core.log_alpha).all() or not torch.isfinite(self.core.log_lam).all():
            raise FloatingPointError("Nonfinite CAL temperature/multiplier")
        return result

    def _cuda_health_diagnostics(self):
        """Same checks and outputs with one device-to-host boolean transfer."""
        result, flags, names = {}, [], []
        for name in ("policy", "critic", "safety_critics"):
            params = list(getattr(self.core, name).parameters())
            grads = [p.grad.detach() for p in params if p.grad is not None]
            flat = torch.cat([p.detach().reshape(-1) for p in params + grads])
            flags.append(torch.isfinite(flat).all()); names.append(name)
            result[name] = dict(parameters=sum(p.numel() for p in params), has_gradient=bool(grads))
        flags.extend((torch.isfinite(self.core.log_alpha).all(), torch.isfinite(self.core.log_lam).all()))
        names.extend(("temperature", "multiplier"))
        bad = (~torch.stack(flags)).cpu().numpy()
        if bad.any():
            raise FloatingPointError("Nonfinite CAL parameter/gradient: " + names[int(np.flatnonzero(bad)[0])])
        return result

    def state_dict(self, *, include_replay=True):
        if include_replay and not self._resume_training_available:
            raise ValueError("Evaluation-only CAL state cannot be relabelled resumable")
        return copy.deepcopy(dict(revision=REVISION, config=asdict(self.config), seed=self.seed,
            observation_dim=self.observation_dim, action_dim=self.action_dim, device=self.device,
            modules={name: getattr(self.core, name).state_dict() for name in MODULES},
            optimizers=({name: getattr(self.core, name).state_dict() for name in OPTIMIZERS}
                        if include_replay else None),
            log_alpha=self.core.log_alpha.detach(), log_lam=self.core.log_lam.detach(),
            core_values={name: copy.deepcopy(getattr(self.core, name)) for name in
                         ("update_counter", "kappa", "cost_lr_scale", "target_cost")},
            rect=getattr(self.core, "rect", None),
            replay=(dict(buffer=self.replay.buffer, position=self.replay.position)
                    if include_replay else None),
            python_rng=self._python_rng, numpy_rng=self._numpy_rng, torch_rng=self._torch_rng,
            cuda_rng=self._cuda_rng, contract=self.contract,
            environment=(self._training_env.export_state() if self._training_env is not None
                         else self._pending_environment),
            environment_steps=self.environment_steps, complete_episodes=self.complete_episodes,
            target_syncs=self.target_syncs, supervised_cost_head_examples=self.supervised_cost_head_examples,
            training_resume_available=bool(include_replay)))

    def load_state_dict(self, saved, *, env=None, restore_training=True):
        if (saved.get("revision") != REVISION or saved["config"] != asdict(self.config)
                or saved["seed"] != self.seed or saved["observation_dim"] != self.observation_dim
                or saved["action_dim"] != self.action_dim):
            raise ValueError("CAL state identity differs")
        if restore_training and (saved["device"] != self.device or saved["replay"] is None):
            raise ValueError("Exact continuation requires same device and complete replay")
        if env is not None and json.loads(json.dumps(env.get_contract())) != saved["contract"]:
            raise ValueError("CAL restore environment differs")
        for name in MODULES:
            getattr(self.core, name).load_state_dict(saved["modules"][name])
        with torch.no_grad():
            self.core.log_alpha.copy_(saved["log_alpha"])
            self.core.log_lam.copy_(saved["log_lam"])
        if saved["optimizers"] is not None:
            for name in OPTIMIZERS:
                getattr(self.core, name).load_state_dict(saved["optimizers"][name])
        for name, value in saved["core_values"].items():
            setattr(self.core, name, copy.deepcopy(value))
        if saved["rect"] is not None:
            self.core.rect = saved["rect"].to(self.device)
        if saved["replay"] is not None:
            self.replay.buffer = copy.deepcopy(saved["replay"]["buffer"])
            self.replay.position = saved["replay"]["position"]
        self._resume_training_available = bool(saved["training_resume_available"])
        self._python_rng, self._numpy_rng = saved["python_rng"], saved["numpy_rng"]
        self._torch_rng, self._cuda_rng = saved["torch_rng"], saved["cuda_rng"]
        for name in ("environment_steps", "complete_episodes", "target_syncs",
                     "supervised_cost_head_examples", "contract"):
            setattr(self, name, copy.deepcopy(saved[name]))
        if env is not None:
            env.import_state(saved["environment"])
            self._training_env = env
        else:
            self._pending_environment = copy.deepcopy(saved["environment"])
        return self

    def save(self, path, *, include_replay=False):
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError("CAL candidate checkpoints cannot overwrite existing artifacts")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("." + path.name + ".tmp-" + uuid.uuid4().hex)
        tmp.mkdir()
        try:
            torch.save(self.state_dict(include_replay=include_replay), tmp / "state.pt")
            manifest = dict(revision=REVISION, metadata=self.metadata,
                            wrapper_source_sha256=_sha(__file__),
                            vendor_manifest_sha256=_sha(VENDOR / "SOURCE_MANIFEST.json"),
                            scientific_source_sha256=_scientific_sources(), software=_software(),
                            state_sha256=_sha(tmp / "state.pt"),
                            optimizer_state_retained=bool(include_replay),
                            training_resume_available=bool(include_replay))
            (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            os.rename(tmp, path)
        except BaseException:
            shutil.rmtree(tmp)
            raise
        return manifest

    @classmethod
    def load(cls, path, *, env=None, device=None, restore_training=True):
        path = Path(path).resolve()
        manifest = json.loads((path / "manifest.json").read_text())
        if (manifest["revision"] != REVISION or manifest["wrapper_source_sha256"] != _sha(__file__)
                or manifest["vendor_manifest_sha256"] != _sha(VENDOR / "SOURCE_MANIFEST.json")
                or manifest["scientific_source_sha256"] != _scientific_sources()
                or manifest["software"] != _software()
                or manifest["state_sha256"] != _sha(path / "state.pt")):
            raise ValueError("CAL checkpoint source or payload differs")
        saved = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        if bool(saved["training_resume_available"]) != bool(manifest["training_resume_available"]):
            raise ValueError("CAL continuation capability differs")
        agent = cls(saved["observation_dim"], saved["action_dim"], saved["seed"],
                    device=saved["device"] if device is None else device,
                    config=CALConfig(**saved["config"]))
        return agent.load_state_dict(saved, env=env, restore_training=restore_training)


__all__ = ["CALCandidate", "CALConfig", "resource_cost", "provenance"]
