"""Paper-derived NeurIPS2023 CCPO candidate, isolated from active registries.

VVE factorized reward/cost critics, online behavior conditions and shared replay,
conditioned variational E-steps and KL-constrained weighted-likelihood M-steps.
Sources include the author's arXiv-v2 TeX implementation/proof appendices.
No author source code or exact-author numerical reproduction is claimed.
"""
from __future__ import annotations

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
import uuid

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence


REVISION = "chapter3-ccpo-fixed-full-budget-v2"
VENDOR = Path(__file__).parent / "vendor/ccpo"


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
    value = json.loads((VENDOR / "SOURCE_MANIFEST.json").read_text())
    for name, digest in value["files_sha256"].items():
        if _sha(VENDOR / name) != digest:
            raise ValueError("CCPO paper source differs: " + name)
    return value


def dual_value_gradient(eta_lambda, rewards, costs, threshold, kl_bound):
    """Appendix B convex dual and analytic gradient, with MC policy particles.

    eta*kappa + lambda*epsilon + eta*E_s log E_pi exp((Qr-lambda*Qc)/eta).
    Particles are sampled from the old policy, hence their base masses are 1/K.
    """
    eta, lam = np.asarray(eta_lambda, dtype=np.float64)
    qr, qc = np.asarray(rewards, np.float64), np.asarray(costs, np.float64)
    if eta <= 0 or lam < 0 or qr.ndim != 2 or qr.shape != qc.shape or qr.shape[1] < 2:
        raise ValueError("Positive eta, nonnegative lambda and BxK particles required")
    score = (qr - lam * qc) / eta
    log_normalizer = logsumexp(score, axis=1, keepdims=True)
    log_partition = log_normalizer[:, 0] - np.log(qr.shape[1])
    weights = np.exp(score - log_normalizer)
    value = lam * threshold + eta * kl_bound + eta * log_partition.mean()
    gradient = np.asarray([
        kl_bound + log_partition.mean() - (weights * score).sum(1).mean(),
        threshold - (weights * qc).sum(1).mean()])
    return float(value), gradient


def solve_conditioned_dual(rewards, costs, threshold, kl_bound, *, initial=(1., 0.),
                           max_iterations=500, tolerance=1e-6):
    qr, qc = np.asarray(rewards, np.float64), np.asarray(costs, np.float64)
    if (qr.ndim != 2 or qr.shape != qc.shape or qr.shape[1] < 2
            or not np.isfinite(qr).all() or not np.isfinite(qc).all()
            or threshold <= 0 or kl_bound <= 0):
        raise ValueError("Invalid CCPO dual problem")
    eta_floor = 1e-6
    evaluations = 0
    minimum_cost = float(qc.min(1).mean())
    if minimum_cost > threshold + tolerance:
        return dict(weights=np.full_like(qr, 1. / qr.shape[1]), eta=1., lam=0.,
                    expected_cost=float(qc.mean()), particle_kl=0., threshold=float(threshold),
                    objective=None, optimizer_success=False, feasible=False, success=False,
                    iterations=0, message="No feasible distribution among sampled particles",
                    function_evaluations=0, eta_floor=eta_floor, minimum_particle_cost=minimum_cost)

    def objective(x):
        nonlocal evaluations
        evaluations += 1
        return dual_value_gradient(x, qr, qc, threshold, kl_bound)

    def inspect(result):
        eta, lam = result.x
        logits = (qr - lam * qc) / eta
        log_weights = logits - logsumexp(logits, axis=1, keepdims=True)
        weights = np.exp(log_weights)
        cost = float((weights * qc).sum(1).mean())
        kl = float((weights * (log_weights + np.log(qr.shape[1]))).sum(1).mean())
        feasible = bool(cost <= threshold + tolerance and kl <= kl_bound + tolerance)
        return dict(weights=weights, eta=float(eta), lam=float(lam),
                    expected_cost=cost, particle_kl=kl, threshold=float(threshold),
                    objective=float(result.fun), optimizer_success=bool(result.success),
                    feasible=feasible, success=bool(result.success and feasible),
                    iterations=int(result.nit), message=str(result.message))

    start = np.maximum(np.asarray(initial, np.float64), [eta_floor, 0.])
    result = minimize(objective, start, jac=True, method="L-BFGS-B",
                      bounds=((eta_floor, None), (0., None)),
                      options=dict(maxiter=max_iterations, ftol=1e-12, gtol=1e-9, maxls=50))
    answer = inspect(result)
    if not answer["success"]:
        # Same two-dimensional convex objective, alternative numerical solver.
        retry = minimize(objective, result.x, jac=True, method="SLSQP",
                         bounds=((eta_floor, None), (0., None)),
                         options=dict(maxiter=max_iterations, ftol=1e-11))
        other = inspect(retry)
        if other["success"] or other["feasible"] and not answer["feasible"]:
            answer = other
    answer["function_evaluations"] = evaluations
    answer["eta_floor"] = eta_floor
    answer["minimum_particle_cost"] = float(qc.min(1).mean())
    return answer


@dataclass(frozen=True)
class CCPOConfig:
    # Appendix C specifies feature=32, psi=[256,256], z=[32,32], Con actor.
    hidden_size: int = 256
    feature_dim: int = 32
    condition_hidden_size: int = 32
    feature_bound: float = 1.0
    gamma: float = 1.0
    target_retention: float = .995
    behavior_conditions: tuple = (1.0,)
    versatile_conditions: tuple = (1.0,)
    fixed_full_budget: bool = True
    collection_episodes_per_epoch: int = 4
    # Numerical choices below are preregistered adaptations, not author defaults.
    pretraining_epochs: int = 100
    batch_per_condition: int = 512
    policy_particles: int = 64
    critic_action_samples: int = 1
    critic_updates_per_epoch: int = 1
    m_step_iterations: int = 5
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    kl_multiplier_learning_rate: float = .01
    e_step_kl: float = .1
    m_step_kl: float = .01
    dual_max_iterations: int = 500
    dual_feasibility_tolerance: float = 1e-6
    max_backtracks: int = 20
    replay_capacity: int = 1_000_000

    def __post_init__(self):
        for name in ("hidden_size", "feature_dim", "condition_hidden_size", "pretraining_epochs",
                     "batch_per_condition", "policy_particles", "critic_action_samples",
                     "critic_updates_per_epoch", "m_step_iterations", "dual_max_iterations",
                     "max_backtracks", "replay_capacity", "collection_episodes_per_epoch"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive integer required: " + name)
        if self.policy_particles < 2 or self.gamma != 1. or not 0 < self.target_retention < 1:
            raise ValueError("Retain particle inference and native episodic gamma1")
        for name in ("feature_bound", "critic_learning_rate", "actor_learning_rate",
                     "kl_multiplier_learning_rate", "e_step_kl", "m_step_kl",
                     "dual_feasibility_tolerance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("Positive finite value required: " + name)
        for conditions in (self.behavior_conditions, self.versatile_conditions):
            if (len(conditions) != len(set(conditions)) or tuple(sorted(conditions)) != tuple(conditions)
                    or any(not np.isfinite(x) or not 0 < x <= 1 for x in conditions)):
                raise ValueError("Register ordered positive budget-fraction conditions")
        if self.fixed_full_budget:
            if self.behavior_conditions != (1.,) or self.versatile_conditions != (1.,):
                raise ValueError('Full-spend task only admits its fixed unit-budget constraint')
        elif (self.behavior_conditions != (.25, .5, .75, 1.0)
              or not set(self.behavior_conditions).issubset(self.versatile_conditions)
              or not set(self.versatile_conditions) - set(self.behavior_conditions)):
            raise ValueError("Retain all four behavior conditions and unseen fine-tuning conditions")


def _mlp(input_dim, widths, output_dim):
    layers = []
    for width in widths:
        layers.extend([nn.Linear(input_dim, width), nn.ReLU()])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class VersatileCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        self.psi_net = _mlp(obs_dim + action_dim, [config.hidden_size] * 2, config.feature_dim)
        self.z_net = _mlp(1, [config.condition_hidden_size] * 2, config.feature_dim)
        self.feature_bound = config.feature_bound

    def features(self, state, action):
        return self.feature_bound * torch.tanh(self.psi_net(torch.cat((state, action), -1)))

    def forward(self, state, action, condition):
        return (self.features(state, action) * self.z_net(condition)).sum(-1)


class ConditionedPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        self.net = _mlp(obs_dim + 1, [config.hidden_size] * 2, 2 * action_dim)

    def forward(self, state, condition):
        mean, log_std = self.net(torch.cat((state, condition), -1)).chunk(2, -1)
        return mean, log_std.clamp(-5., 2.)

    def sample(self, state, condition, count=1):
        mean, log_std = self(state, condition)
        raw = mean.unsqueeze(-2) + log_std.exp().unsqueeze(-2) * torch.randn(
            (*mean.shape[:-1], count, mean.shape[-1]), device=mean.device)
        return raw.tanh(), raw, mean, log_std


class CCPOCandidate:
    def __init__(self, observation_dim, action_dim, seed=0, *, device="cpu", config=None):
        if (any(type(v) is not int or v < 1 for v in (observation_dim, action_dim))
                or type(seed) is not int or seed < 0):
            raise ValueError("Positive neural dimensions and nonnegative seed required")
        provenance()
        self.observation_dim, self.action_dim, self.seed = observation_dim, action_dim, seed
        self.device, self.config = str(torch.device(device)), config or CCPOConfig()
        self._python_rng = random.Random(seed).getstate()
        self._numpy_rng = np.random.RandomState(seed).get_state()
        self._torch_rng = torch.Generator(device="cpu").manual_seed(seed).get_state()
        self._cuda_rng = (torch.Generator(device=self.device).manual_seed(seed).get_state()
                          if self.device.startswith("cuda") else None)
        with self._rng_scope():
            self.policy = ConditionedPolicy(observation_dim, action_dim, self.config).to(self.device)
            self.reward_critic = VersatileCritic(observation_dim, action_dim, self.config).to(self.device)
            self.cost_critic = VersatileCritic(observation_dim, action_dim, self.config).to(self.device)
        self.reward_target, self.cost_target = (copy.deepcopy(x).requires_grad_(False)
                                                for x in (self.reward_critic, self.cost_critic))
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.actor_learning_rate)
        self.reward_optimizer = torch.optim.Adam(self.reward_critic.parameters(), lr=self.config.critic_learning_rate)
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=self.config.critic_learning_rate)
        self.replay, self.replay_position = [], 0
        self._condition_replay_cache = None
        self.epoch = 0
        self.counters = dict(environment_steps=0, complete_episodes=0, pretraining_environment_steps=0,
            finetuning_environment_steps=0, reward_critic_updates=0, cost_critic_updates=0,
            actor_optimizer_steps=0, actor_accepted_steps=0, actor_backtracks=0,
            target_updates=0, e_step_calls=0, dual_function_evaluations=0, dual_failures=0,
            actor_blocks_skipped_infeasible=0, policy_particles=0)
        self.kl_multipliers = {}
        self.dual_initial = {}
        self.last_update = None
        self.contract, self._training_env, self._pending_environment = None, None, None
        self._resume_training_available = True

    @contextmanager
    def _rng_scope(self):
        old = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
        old_cuda = torch.cuda.get_rng_state(self.device) if self._cuda_rng is not None else None
        random.setstate(self._python_rng); np.random.set_state(self._numpy_rng)
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
                torch.cuda.set_rng_state(old_cuda, self.device)
            random.setstate(old[0]); np.random.set_state(old[1]); torch.random.set_rng_state(old[2])

    def _condition(self, conditions):
        # Retain the author's literal normalization (eps-eps_L)/eps_H.
        return ((conditions - self.config.behavior_conditions[0])
                / self.config.behavior_conditions[-1]).unsqueeze(-1)

    def _obs(self, observation):
        x = np.asarray(observation, np.float32)
        if x.shape != (self.observation_dim,) or not np.isfinite(x).all():
            raise ValueError("Reduced current public observation required")
        return x

    @property
    def stage(self):
        return "pretraining" if self.epoch < self.config.pretraining_epochs else "finetuning"

    @property
    def training_conditions(self):
        return (self.config.behavior_conditions if self.stage == "pretraining"
                else self.config.versatile_conditions)

    def act(self, observation, condition=1., deterministic=False):
        if condition not in self.config.versatile_conditions:
            raise ValueError("Unregistered candidate condition")
        state = torch.as_tensor(self._obs(observation), device=self.device).unsqueeze(0)
        c = self._condition(torch.tensor([condition], device=self.device))
        with torch.no_grad():
            if deterministic:
                action = self.policy(state, c)[0].tanh()[0]
            else:
                with self._rng_scope():
                    action = self.policy.sample(state, c)[0][0, 0]
        return action.cpu().numpy().copy()

    def observe(self, observation, action, reward, cost, next_observation, done, condition):
        if not self._resume_training_available:
            raise ValueError("Evaluation-only CCPO state cannot continue training")
        s, n, a = self._obs(observation), self._obs(next_observation), np.asarray(action, np.float32)
        if (a.shape != (self.action_dim,) or not np.isfinite(a).all() or np.any(np.abs(a) > 1)
                or not np.isfinite([reward, cost]).all() or cost < 0
                or not isinstance(done, (bool, np.bool_)) or condition not in self.config.behavior_conditions):
            raise ValueError("Invalid online CCPO transition")
        row = (s.copy(), a.copy(), float(reward), float(cost), n.copy(), bool(done), float(condition))
        self._condition_replay_cache = None
        if len(self.replay) < self.config.replay_capacity:
            self.replay.append(row)
        else:
            self.replay[self.replay_position] = row
        self.replay_position = (self.replay_position + 1) % self.config.replay_capacity
        self.counters["environment_steps"] += 1
        self.counters[self.stage + "_environment_steps"] += 1
        self.counters["complete_episodes"] += int(done)

    def _sample(self):
        if self._condition_replay_cache is None:
            grouped = {condition: [] for condition in self.config.behavior_conditions}
            for row in self.replay:
                grouped[row[-1]].append(row)
            self._condition_replay_cache = grouped
        with self._rng_scope():
            rows = []
            for condition in self.config.behavior_conditions:
                available = self._condition_replay_cache[condition]
                if not available:
                    raise ValueError("Collect online data for every behavior condition first")
                rows.extend(available[i] for i in np.random.randint(
                    len(available), size=self.config.batch_per_condition))
        fields = [np.asarray(x) for x in zip(*rows)]
        return [torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in fields]

    def _update_critics(self, batch):
        s, a, r, cost, nxt, done, eps = batch
        condition = self._condition(eps)
        with torch.no_grad(), self._rng_scope():
            actions = self.policy.sample(nxt, condition, self.config.critic_action_samples)[0]
            states = nxt.unsqueeze(1).expand(-1, actions.shape[1], -1)
            conditions = condition.unsqueeze(1).expand(-1, actions.shape[1], -1)
            reward_target = r + self.config.gamma * (1. - done) * self.reward_target(
                states, actions, conditions).mean(1)
            cost_target = cost + self.config.gamma * (1. - done) * self.cost_target(
                states, actions, conditions).mean(1)
        # Sum of conditional mean MSBEs, as in Appendix C (not pooled mean).
        multiplier = len(self.config.behavior_conditions)
        reward_loss = (self.reward_critic(s, a, condition) - reward_target).square().mean() * multiplier
        cost_loss = (self.cost_critic(s, a, condition) - cost_target).square().mean() * multiplier
        self.reward_optimizer.zero_grad(); reward_loss.backward(); self.reward_optimizer.step()
        self.cost_optimizer.zero_grad(); cost_loss.backward(); self.cost_optimizer.step()
        with torch.no_grad():
            rho = self.config.target_retention
            for target, source in ((self.reward_target, self.reward_critic), (self.cost_target, self.cost_critic)):
                for p, q in zip(target.parameters(), source.parameters()):
                    p.mul_(rho).add_(q, alpha=1. - rho)
        self.counters["reward_critic_updates"] += 1
        self.counters["cost_critic_updates"] += 1
        self.counters["target_updates"] += 1
        return dict(reward_loss=float(reward_loss.detach()), cost_loss=float(cost_loss.detach()),
                    terminal_targets_exact=True)

    def _e_step(self, states):
        values = self.training_conditions
        eps = torch.tensor(values, dtype=torch.float32, device=self.device)
        s = states.unsqueeze(0).expand(len(values), -1, -1)
        c = self._condition(eps).unsqueeze(1).expand(-1, len(states), -1)
        with torch.no_grad(), self._rng_scope():
            actions, raw, old_mean, old_log_std = self.policy.sample(s, c, self.config.policy_particles)
            expanded_s = s.unsqueeze(2).expand(-1, -1, self.config.policy_particles, -1)
            expanded_c = c.unsqueeze(2).expand(-1, -1, self.config.policy_particles, -1)
            qr, qc = (critic(expanded_s, actions, expanded_c)
                      for critic in (self.reward_critic, self.cost_critic))
            # One combined device-to-host transfer for all conditions and both Qs.
            host = torch.stack((qr, qc)).cpu().numpy()
        results = []
        for index, condition in enumerate(values):
            result = solve_conditioned_dual(host[0, index], host[1, index], condition,
                self.config.e_step_kl, initial=self.dual_initial.get(str(condition), (1., 0.)),
                max_iterations=self.config.dual_max_iterations,
                tolerance=self.config.dual_feasibility_tolerance)
            self.dual_initial[str(condition)] = (result["eta"], result["lam"])
            self.counters["e_step_calls"] += 1
            self.counters["dual_function_evaluations"] += result["function_evaluations"]
            self.counters["dual_failures"] += int(not result["success"])
            results.append(result)
        self.counters["policy_particles"] += int(actions.shape[0] * actions.shape[1] * actions.shape[2])
        weights = torch.as_tensor(np.stack([x["weights"] for x in results]),
                                  dtype=torch.float32, device=self.device)
        return (s, c, raw, old_mean, old_log_std, weights), results

    def _m_step(self, fixed):
        s, condition, raw, old_mean, old_log_std, weights = fixed
        values = self.training_conditions
        multipliers = torch.tensor([self.kl_multipliers.get(str(x), 1.) for x in values], device=self.device)
        old_distribution = Normal(old_mean, old_log_std.exp())
        # The E-step particles are fixed, detached inputs in every M-step.
        # Reuse exactly the same elementwise Jacobian values; retain subtraction
        # before the original action-axis sum to preserve floating-point order.
        log_jacobian = torch.log(1. - raw.tanh().square() + 1e-6)
        rows = []
        for _ in range(self.config.m_step_iterations):
            mean, log_std = self.policy(s, condition)
            distribution = Normal(mean, log_std.exp())
            kl = kl_divergence(old_distribution, distribution).sum(-1).mean(-1)
            log_prob = Normal(mean.unsqueeze(2), log_std.exp().unsqueeze(2)).log_prob(raw)
            log_prob = (log_prob - log_jacobian).sum(-1)
            objective = (weights * log_prob).sum(-1).mean()
            loss = -objective + (multipliers * (kl - self.config.m_step_kl)).mean()
            before = copy.deepcopy(self.policy.state_dict())
            self.actor_optimizer.zero_grad(); loss.backward(); self.actor_optimizer.step()
            self.counters["actor_optimizer_steps"] += 1
            proposal = copy.deepcopy(self.policy.state_dict())
            accepted, final_kl = False, None
            with torch.no_grad():
                for attempt in range(self.config.max_backtracks + 1):
                    if attempt:
                        fraction = .5 ** attempt
                        self.policy.load_state_dict({k: before[k] + fraction * (proposal[k] - before[k])
                                                     for k in before})
                        self.counters["actor_backtracks"] += 1
                    mu, ls = self.policy(s, condition)
                    final_kl = kl_divergence(old_distribution, Normal(mu, ls.exp())).sum(-1).mean(-1)
                    if torch.isfinite(final_kl).all() and torch.all(final_kl <= self.config.m_step_kl + 1e-7):
                        accepted = True
                        break
                if not accepted:
                    self.policy.load_state_dict(before)
                    mu, ls = self.policy(s, condition)
                    final_kl = kl_divergence(old_distribution, Normal(mu, ls.exp())).sum(-1).mean(-1)
                multipliers = (multipliers + self.config.kl_multiplier_learning_rate
                               * (final_kl - self.config.m_step_kl)).clamp_min(0.)
            self.counters["actor_accepted_steps"] += int(accepted)
            rows.append(dict(loss=float(loss.detach()), objective=float(objective.detach()),
                             accepted=accepted, per_condition_kl=final_kl.cpu().tolist()))
        self.kl_multipliers.update({str(x): float(a) for x, a in zip(values, multipliers.cpu())})
        return rows

    def update(self):
        if not self._resume_training_available:
            raise ValueError("Evaluation-only CCPO state cannot continue training")
        losses = [self._update_critics(self._sample()) for _ in range(self.config.critic_updates_per_epoch)]
        fixed, duals = self._e_step(self._sample()[0])
        if all(x["success"] for x in duals):
            msteps = self._m_step(fixed)
        else:
            # Appendix's feasible-in-trust-region assumption failed for these
            # learned Q estimates/particles. Never pretend a fallback solves it.
            msteps = []
            self.counters["actor_blocks_skipped_infeasible"] += 1
        self.last_update = dict(stage=self.stage, critics=losses,
            duals=[{k: v for k, v in x.items() if k != "weights"} for x in duals], m_steps=msteps)
        self.health_diagnostics()
        return copy.deepcopy(self.last_update)

    def collect_transition(self, observation, action, reward, cost, next_observation, done, condition):
        """One behavior-condition transition; finish_epoch performs joint updates."""
        self.observe(observation, action, reward, cost, next_observation, done, condition)
        return self.accounting

    def finish_epoch(self):
        """After the declared collection block: VVE/CVI, then stage advance."""
        result = self.update()
        self.epoch += 1
        return result

    def train_epoch(self, env, *, episodes_per_condition=None):
        if episodes_per_condition is None:
            episodes_per_condition=(self.config.collection_episodes_per_epoch
                                    if self.config.fixed_full_budget else 1)
        if (not self._resume_training_available or type(episodes_per_condition) is not int
                or episodes_per_condition < 1 or env.chapter != 3 or env.holdout
                or env.phase not in ("train", "engineering") or env.action_family != "total_space"):
            raise ValueError("Isolated chapter3 online training/engineering env required")
        contract = json.loads(json.dumps(env.get_contract()))
        if (len(env.policy_indices) != self.observation_dim or env.action_dim != self.action_dim
                or self.contract is not None and contract != self.contract):
            raise ValueError("CCPO native environment binding differs")
        if self._pending_environment is not None:
            env.import_state(self._pending_environment); self._pending_environment = None
        self.contract, self._training_env = contract, env
        stage, episodes = self.stage, []
        for condition in self.config.behavior_conditions:
            for _ in range(episodes_per_condition):
                obs, _ = (env.reset() if env.train_path_base is not None else
                          env.reset(seed=self.counters["complete_episodes"]))
                reward_sum = cost_sum = 0.
                for t in range(env.horizon):
                    policy_obs = obs[env.policy_indices]
                    action = self.act(policy_obs, condition)
                    nxt, reward, done, truncated, info = env.step(action)
                    if truncated or done != (t == env.horizon - 1) or not done and reward != 0.:
                        raise RuntimeError("Native terminal-only contract differs")
                    cost = float(np.asarray(info["physical_action"]).sum() / env.spec.initial_budget)
                    self.collect_transition(policy_obs, action, reward, cost, nxt[env.policy_indices], done, condition)
                    reward_sum += reward; cost_sum += cost; obs = nxt
                replay = env.replay_last_episode()
                if (cost_sum > 1. + 1e-12 or replay["error"] != 0. or
                        self.config.fixed_full_budget and abs(cost_sum-1.)>1e-9):
                    raise ArithmeticError("Native budget or action replay differs")
                episodes.append(dict(condition=condition, native_return=reward_sum,
                                     cost=cost_sum, native_replay_error=replay["error"]))
        update = self.finish_epoch()
        return dict(epoch=self.epoch, stage=stage, episodes=episodes, update=update,
                    accounting=self.accounting)

    @property
    def accounting(self):
        return dict(self.counters, epoch=self.epoch, stage=self.stage,
                    replay_transitions=len(self.replay), behavior_conditions=len(self.config.behavior_conditions),
                    versatile_conditions=len(self.config.versatile_conditions),
                    simulated_auxiliary_environment_steps=0)

    @property
    def metadata(self):
        return dict(revision=REVISION, method="CCPO", candidate_only=True, paper_derived=True,
                    source=provenance(), parameters=asdict(self.config), seed=self.seed, device=self.device,
                    native_reward="unchanged terminal reward", native_cost="executed resource / initial budget",
                    formal_evaluation_condition=1., physical_budget_unchanged=True,
                    offline_dataset_required=False, expected_cmdp_not_pathwise_budget_guarantee=True,
                    budget_constraint_redundant_at_condition1_with_exact_native_decoder=True,
                    m_step="joint conditional likelihood; per-condition exact Gaussian KL and feasible backtracking",
                    solver="Appendix B convex dual, scipy L-BFGS-B then SLSQP; explicit infeasibility",
                    source_limit="author algorithm TeX blocks are commented drafts; schedules and solvers preregistered",
                    fixed_full_budget_specialization=self.config.fixed_full_budget,
                    multi_threshold_performance_claimed=not self.config.fixed_full_budget,
                    endpoint_persistence="one evaluation model; no disk replay or optimizer state",
                    active_route_modified=False)

    def health_diagnostics(self):
        if self.device.startswith("cuda"):
            return self._cuda_health_diagnostics()
        result = {}
        for name in ("policy", "reward_critic", "cost_critic"):
            ps = list(getattr(self, name).parameters())
            gs = [p.grad for p in ps if p.grad is not None]
            if any(not torch.isfinite(x).all() for x in ps + gs):
                raise FloatingPointError("Nonfinite CCPO state: " + name)
            result[name] = dict(parameters=sum(p.numel() for p in ps), has_gradient=bool(gs))
        return result

    def _cuda_health_diagnostics(self):
        """Retain all finite checks; merge their host observation on CUDA."""
        result, flags, names = {}, [], []
        for name in ("policy", "reward_critic", "cost_critic"):
            ps = list(getattr(self, name).parameters())
            gs = [p.grad for p in ps if p.grad is not None]
            flat = torch.cat([p.detach().reshape(-1) for p in ps + gs])
            flags.append(torch.isfinite(flat).all()); names.append(name)
            result[name] = dict(parameters=sum(p.numel() for p in ps), has_gradient=bool(gs))
        bad = (~torch.stack(flags)).cpu().numpy()
        if bad.any():
            raise FloatingPointError("Nonfinite CCPO state: " + names[int(np.flatnonzero(bad)[0])])
        return result

    def state_dict(self, *, include_replay=True):
        if include_replay and not self._resume_training_available:
            raise ValueError("Evaluation-only CCPO state cannot be relabelled resumable")
        names = ("policy", "reward_critic", "cost_critic", "reward_target", "cost_target",
                 "actor_optimizer", "reward_optimizer", "cost_optimizer")
        return copy.deepcopy(dict(revision=REVISION, config=asdict(self.config), seed=self.seed,
            observation_dim=self.observation_dim, action_dim=self.action_dim, device=self.device,
            objects={name: getattr(self, name).state_dict() for name in names
                     if include_replay or not name.endswith("optimizer")}, epoch=self.epoch,
            counters=self.counters, kl_multipliers=self.kl_multipliers, dual_initial=self.dual_initial,
            replay=self.replay if include_replay else None, replay_position=self.replay_position,
            python_rng=self._python_rng, numpy_rng=self._numpy_rng, torch_rng=self._torch_rng,
            cuda_rng=self._cuda_rng, contract=self.contract,
            environment=self._training_env.export_state() if self._training_env else self._pending_environment,
            last_update=self.last_update, training_resume_available=bool(include_replay)))

    def load_state_dict(self, saved, *, env=None, restore_training=True):
        if (saved["revision"] != REVISION or saved["config"] != asdict(self.config)
                or saved["seed"] != self.seed or saved["observation_dim"] != self.observation_dim
                or saved["action_dim"] != self.action_dim):
            raise ValueError("CCPO checkpoint identity differs")
        if restore_training and (saved["device"] != self.device or saved["replay"] is None):
            raise ValueError("CCPO continuation requires same device and complete replay")
        if env is not None and json.loads(json.dumps(env.get_contract())) != saved["contract"]:
            raise ValueError("CCPO environment differs")
        for name, state in saved["objects"].items():
            getattr(self, name).load_state_dict(state)
        for name in ("epoch", "counters", "kl_multipliers", "dual_initial", "replay_position", "contract", "last_update"):
            setattr(self, name, copy.deepcopy(saved[name]))
        self.replay = copy.deepcopy(saved["replay"]) if saved["replay"] is not None else []
        self._condition_replay_cache = None
        self._python_rng, self._numpy_rng = saved["python_rng"], saved["numpy_rng"]
        self._torch_rng, self._cuda_rng = saved["torch_rng"], saved["cuda_rng"]
        self._resume_training_available = bool(saved["training_resume_available"])
        if env is not None:
            env.import_state(saved["environment"]); self._training_env = env
        else:
            self._pending_environment = copy.deepcopy(saved["environment"])
        return self

    def save(self, path, *, include_replay=False):
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError("CCPO checkpoints cannot overwrite artifacts")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("." + path.name + ".tmp-" + uuid.uuid4().hex); tmp.mkdir()
        try:
            torch.save(self.state_dict(include_replay=include_replay), tmp / "state.pt")
            manifest = dict(revision=REVISION, metadata=self.metadata,
                wrapper_source_sha256=_sha(__file__), vendor_source_sha256=_sha(VENDOR / "SOURCE_MANIFEST.json"),
                scientific_source_sha256=_scientific_sources(), software=_software(),
                payload_sha256=_sha(tmp / "state.pt"), training_resume_available=bool(include_replay))
            manifest['optimizer_state_retained'] = bool(include_replay)
            (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            os.rename(tmp, path)
        except BaseException:
            shutil.rmtree(tmp)
            raise
        return manifest

    @classmethod
    def load(cls, path, *, env=None, device=None, restore_training=True):
        path = Path(path).resolve(); manifest = json.loads((path / "manifest.json").read_text())
        if (manifest["revision"] != REVISION or manifest["wrapper_source_sha256"] != _sha(__file__)
                or manifest["vendor_source_sha256"] != _sha(VENDOR / "SOURCE_MANIFEST.json")
                or manifest["scientific_source_sha256"] != _scientific_sources()
                or manifest["software"] != _software()
                or manifest["payload_sha256"] != _sha(path / "state.pt")):
            raise ValueError("CCPO checkpoint source/payload differs")
        saved = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        if saved["training_resume_available"] != manifest["training_resume_available"]:
            raise ValueError("CCPO continuation flag differs")
        agent = cls(saved["observation_dim"], saved["action_dim"], saved["seed"],
                    device=saved["device"] if device is None else device,
                    config=CCPOConfig(**saved["config"]))
        return agent.load_state_dict(saved, env=env, restore_training=restore_training)


__all__ = ["CCPOCandidate", "CCPOConfig", "VersatileCritic", "ConditionedPolicy",
           "dual_value_gradient", "solve_conditioned_dual", "provenance"]
