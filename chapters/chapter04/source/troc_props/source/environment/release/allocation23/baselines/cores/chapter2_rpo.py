"""RPO-SAC for the chapter-2 state-dependent capped simplex.

Formula implementation of Ding et al., NeurIPS 2023, equations (4)--(9),
Algorithm 1 and the authors' ``rpo/algo/rpo_sac.py`` / ``agent/sac_pa.py``.
Source: https://github.com/wadx2019/rpo at
9cba44478f7800dbb6e73d06f425397a7ac0b9b6 (MIT).

The policy generates N-1 basic actions. The remaining action is completed
analytically from the equality, with its exact implicit derivative. Execution
and bootstrap targets use inequality-corrected actions. Following official
RPO-SAC, the actor uses the *uncorrected*, equality-completed action and an
adaptive positive-part constraint penalty, not a vanilla SAC projected action.

Declared task adaptations: allocation units are one object's mean quota;
the known linear sum/box constraints admit a finite active-set GRG correction
instead of small fixed steps for nonlinear manifolds. The fixed dependent
coordinate is the final object. Zero-width boxes contribute no entropy. Two
unshared MLPs, running input normalization, gamma=1 and normalized terminal
rewards follow the chapter-2 experiment contract. No transition/reward model,
future information, F allocator, or environment gradient is accessed.

SAC entropy refers to the sampled basic coordinates *before* correction, as
in the authors' RPO-SAC; it is not asserted to be an executed-action density.
The v3 task adaptation sets a target of -1 per nondegenerate standardized tanh
coordinate, then adds the current affine box log-Jacobian. This makes the
temperature residual independent of allocation units and permits a target
below the entropy ceiling imposed by the retained log_std_max=-2 constraint.
The common environment projection only reconciles observation/action floating
point roundoff; the RPO core already imposes caps and equality before step().
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import RunningObservationNorm
from .recent_replay import ReplayBuffer
from ..actions import neutral_mean_tensor


REVISION = "chapter23-rpo-sac-linear-grg-v5-declared-warmup"
SOURCE = dict(
    paper="https://proceedings.neurips.cc/paper_files/paper/2023/hash/7984e22a06eb5f0e35d745cb38345983-Abstract-Conference.html",
    repository="https://github.com/wadx2019/rpo",
    commit="9cba44478f7800dbb6e73d06f425397a7ac0b9b6",
    license="MIT", implementation="formula implementation, capped-simplex task adaptation")


def rpo_parameters(*, tiny=False, horizon=40):
    if type(horizon) is not int or horizon < 1:
        raise ValueError("positive integer horizon required")
    return dict(hidden_dim=32 if tiny else 256,
                buffer_size=max(256, 16 * horizon) if tiny else 200000,
                batch_size=16 if tiny else 256,
                learning_starts=2 * horizon if tiny else 8 * horizon,
                actor_learning_rate=1e-4, critic_learning_rate=3e-4,
                dual_learning_rate=1e-4, temperature_learning_rate=1e-4,
                tau=.005, gamma=1., policy_delay=2,
                initial_temperature=.01, target_entropy_coef=-1.,
                initial_dual=0., log_std_min=-23., log_std_max=-2.,
                log_std_bias=-3.)


class GeometryReplayBuffer(ReplayBuffer):
    """Compact replay with FP64 constraints and controls, FP32 neural features.

    Arrays 0..4 retain the common replay layout. Arrays 5 and 6 preserve the
    current/next public caps and budget before the neural-feature cast. Sampling
    reconstructs observations in FP64, so geometry and control decoding precede
    the final FP32 critic/actor boundary. All seven arrays are checkpointed by
    the common archive writer; no out-of-band state is needed for continuation.
    """
    def __init__(self, state_dim, action_dim, device, max_size, batch_size,
                 cap_field, budget_field):
        super().__init__(state_dim, action_dim, device, max_size, batch_size)
        self.geometry_columns = np.r_[
            np.arange(cap_field["start"], cap_field["stop"]),
            np.arange(budget_field["start"], budget_field["stop"])].astype(np.int64)
        self.geometry_index = torch.as_tensor(self.geometry_columns, device=self.device)
        self.arrays[1] = np.empty((self.max_size, action_dim), dtype=np.float64)
        self.arrays.extend(np.empty((self.max_size, len(self.geometry_columns)), dtype=np.float64)
                           for _ in range(2))

    def add(self, state, action, next_state, reward, done):
        self.arrays[5][self.ptr] = np.asarray(state)[self.geometry_columns]
        self.arrays[6][self.ptr] = np.asarray(next_state)[self.geometry_columns]
        super().add(state, action, next_state, reward, done)

    def sample(self):
        obs, control, nxt, reward, not_done, geometry, next_geometry = super().sample()
        obs, nxt = obs.double(), nxt.double()
        obs[:, self.geometry_index] = geometry
        nxt[:, self.geometry_index] = next_geometry
        return obs, control, nxt, reward, not_done

    def load_state_dict(self, state):
        saved = state.get("arrays", [])
        if (len(saved) != len(self.arrays)
                or any(np.asarray(a).dtype != b.dtype or np.asarray(a).shape != (state["size"], b.shape[1])
                       for a, b in zip(saved, self.arrays))):
            raise ValueError("RPO replay precision or geometry layout differs")
        super().load_state_dict(state)


def complete_equality(basic, target):
    """a_N=target-sum(a_B); d(a_N)/d(a_B)=-1, exactly (paper Eq.4)."""
    if basic.ndim != 2 or target.shape != (basic.shape[0], 1):
        raise ValueError("basic actions must be [batch,N-1] and target [batch,1]")
    return torch.cat((basic, target - basic.sum(-1, keepdim=True)), -1)


def inequality_violations(action, caps):
    """Positive parts of g=(-a,a-c), retained separately for dual weights."""
    return torch.cat(((-action).clamp_min(0.), (action - caps).clamp_min(0.)), -1)


def correct_inequalities(action, caps, target):
    """Finite active-set reduced-gradient correction for sum+box constraints.

    Basic coordinates already satisfy their boxes. Thus only the completed
    coordinate can violate a bound. Its L1 violation has reduced derivative
    +/-1 in every unconstrained basic coordinate. Move these coordinates by a
    common step, freeze each hit box, and recompute the dependent coordinate.
    Every move lies in the equality tangent space; at least one bound is hit
    each nonfinal iteration, so at most N rounds are needed. This is not the
    generic environment's Euclidean projection of an unconstrained N-vector.
    Operations are differentiable away from active-set changes, although
    official RPO actor optimization uses pre-correction actions instead.
    """
    if action.ndim != 2 or caps.shape != action.shape or target.shape != action[:, :1].shape:
        raise ValueError("invalid constraint tensor shapes")
    if action.shape[-1] < 2:
        raise ValueError("RPO needs at least two allocation objects")
    # Double precision keeps the analytical equality accurate at tight caps.
    original_dtype = action.dtype
    x, c, t = action.to(torch.float64), caps.to(torch.float64), target.to(torch.float64)
    basic, basic_cap = x[:, :-1], c[:, :-1]
    if (not torch.isfinite(x).all() or not torch.isfinite(c).all()
            or not torch.isfinite(t).all() or torch.any(c < 0.)
            or torch.any(t < 0.) or torch.any(t > c.sum(-1, keepdim=True) + 1e-9)
            or torch.any(basic < -1e-8) or torch.any(basic > basic_cap + 1e-8)):
        raise ValueError("invalid RPO basic actions or infeasible public constraints")
    # A positive remaining allowance can be arbitrarily small. Solve the same
    # GRG problem in unit-target coordinates, so an absolute stopping tolerance
    # cannot accept a negative dependent action comparable to the entire quota.
    # The registered constraint parser already tightens every cap to target.
    # A unit denominator protects exact-zero rows without flooring positive ones.
    denominator = torch.where(t > 0., t, torch.ones_like(t))
    basic, basic_cap = basic / denominator, basic_cap / denominator
    c, t = c / denominator, t / denominator
    basic = torch.minimum(basic.clamp_min(0.), basic_cap)
    dependent_goal = torch.minimum((t - basic.sum(-1, keepdim=True)).clamp_min(0.), c[:, -1:])
    desired_sum = t - dependent_goal
    for _ in range(action.shape[-1]):
        gap = desired_sum - basic.sum(-1, keepdim=True)
        if bool(torch.all(gap.abs() <= 32 * torch.finfo(torch.float64).eps)):
            break
        active = torch.where(gap > 0., basic < basic_cap, basic > 0.)
        count = active.sum(-1, keepdim=True).clamp_min(1)
        basic = torch.minimum((basic + active * (gap / count)).clamp_min(0.), basic_cap)
    result = complete_equality(basic, t)
    tolerance = 64 * torch.finfo(torch.float64).eps * action.shape[-1]
    if (not torch.isfinite(result).all() or torch.any(result < -tolerance)
            or torch.any(result > c + tolerance)
            or torch.any((result.sum(-1, keepdim=True) - t).abs() > tolerance)):
        raise RuntimeError("RPO finite GRG correction failed; no hidden fallback")
    # Remove only unit-simplex arithmetic roundoff, before converting back to
    # physical units. This never conceals a failed active-set correction.
    result = torch.minimum(result.clamp_min(0.), c)
    return (result * denominator).to(original_dtype)


class RPOActor(nn.Module):
    def __init__(self, obs_dim, action_dim, p):
        super().__init__()
        self.p = p
        w = p["hidden_dim"]
        self.encoder = nn.Sequential(nn.Linear(obs_dim, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU())
        self.mean = nn.Linear(w, action_dim - 1)
        self.log_std = nn.Linear(w, action_dim - 1)
        self.register_buffer('neutral_anchor_mean',torch.zeros(action_dim-1))

    def sample(self, obs, caps, deterministic=False, mean_adjustment=None):
        hidden = self.encoder(obs)
        mean = self.mean(hidden)
        if mean_adjustment is not None:
            mean=mean+mean_adjustment
        log_std = (self.log_std(hidden) + self.p["log_std_bias"]).clamp(
            self.p["log_std_min"], self.p["log_std_max"])
        normal = torch.distributions.Normal(mean, log_std.exp())
        pre = mean if deterministic else normal.rsample()
        scale = caps[:, :-1] / 2.
        basic = (pre.tanh() + 1.) * scale
        # Include tanh and nondegenerate box affine Jacobians. A zero-width
        # coordinate is deterministic and must not create log(0) or entropy.
        terms = normal.log_prob(pre) - 2. * (math.log(2.) - pre - F.softplus(-2. * pre))
        terms = terms - scale.clamp_min(1e-30).log()
        log_prob = torch.where(scale > 0., terms, torch.zeros_like(terms)).sum(-1, keepdim=True)
        return basic, log_prob


class RPOCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, width):
        super().__init__()
        self.qs = nn.ModuleList([nn.Sequential(nn.Linear(obs_dim + action_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1)) for _ in range(2)])

    def forward(self, obs, action):
        joined = torch.cat((obs, action.to(obs.dtype)), -1)
        return torch.cat([q(joined) for q in self.qs], -1)


class RPOCore:
    """Interaction-only RPO-SAC core with complete optimizer continuation."""
    def __init__(self, obs_dim, action_dim, parameters, device="cpu", observation_schema=None):
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        self.device, self.p = torch.device(device), copy.deepcopy(parameters)
        if self.obs_dim < 1 or self.action_dim < 2:
            raise ValueError("positive obs dimension and >=2 actions required")
        for key in ("hidden_dim", "buffer_size", "batch_size", "policy_delay"):
            if type(self.p.get(key)) is not int or self.p[key] < 1:
                raise ValueError(key + " must be a positive integer")
        if type(self.p.get("learning_starts")) is not int or self.p["learning_starts"] < 0:
            raise ValueError("learning_starts must be nonnegative")
        if self.p["buffer_size"] < max(self.p["batch_size"], self.p["learning_starts"]):
            raise ValueError("replay capacity must cover warmup and batch")
        if self.p["gamma"] != 1. or not 0. < self.p["tau"] <= 1.:
            raise ValueError("chapter-2 terminal task requires gamma=1 and valid tau")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in self.p.values()):
            raise ValueError("RPO parameters must be finite numbers")
        if (any(self.p[k] <= 0. for k in ("actor_learning_rate", "critic_learning_rate",
            "dual_learning_rate", "temperature_learning_rate", "initial_temperature"))
                or self.p["initial_dual"] < 0. or self.p["target_entropy_coef"] >= 0.
                or self.p["log_std_min"] >= self.p["log_std_max"]):
            raise ValueError("invalid RPO optimizer or distribution settings")
        self.observation_schema = copy.deepcopy(observation_schema)
        fields = {x["name"]: x for x in (observation_schema or [])}
        if "caps" not in fields or "budget" not in fields:
            raise ValueError("RPO requires the public caps/budget observation schema")
        self.cap_field, self.budget_field = fields["caps"], fields["budget"]
        for name, field, size in (("caps", self.cap_field, self.action_dim),
                                  ("budget", self.budget_field, 1)):
            if (field["stop"] - field["start"] != size or field["start"] < 0
                    or field["stop"] > self.obs_dim or not math.isfinite(field["divisor"])
                    or field["divisor"] <= 0):
                raise ValueError("invalid public " + name + " schema")
        self.actor = RPOActor(self.obs_dim, self.action_dim, self.p).to(self.device)
        self.critic = RPOCritic(self.obs_dim, self.action_dim, self.p["hidden_dim"]).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.p["actor_learning_rate"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.p["critic_learning_rate"])
        self.dual = nn.Parameter(torch.full((1, 2 * self.action_dim), self.p["initial_dual"], device=self.device))
        self.dual_optimizer = torch.optim.Adam([self.dual], lr=self.p["dual_learning_rate"], maximize=True)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(self.p["initial_temperature"]), device=self.device))
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=self.p["temperature_learning_rate"])
        self.obs_norm = RunningObservationNorm(self.obs_dim)
        self.replay_buffer = GeometryReplayBuffer(self.obs_dim, self.action_dim, self.device,
            self.p["buffer_size"], self.p["batch_size"], self.cap_field, self.budget_field)
        self.observed_steps = self.training_steps = self.actor_updates = 0
        self.policy_warmup = False
        self.neutral_context = None
        self.last_losses = {}

    def constraints(self, obs):
        """Only read public caps/budget; no native environment is held here."""
        caps = obs[:, self.cap_field["start"]:self.cap_field["stop"]].double()
        budget = obs[:, self.budget_field["start"]:self.budget_field["stop"]].double()
        budget = budget * (self.budget_field["divisor"] / self.cap_field["divisor"])
        if (not torch.isfinite(caps).all() or not torch.isfinite(budget).all()
                or torch.any(caps < 0.) or torch.any(budget < 0.)):
            raise ValueError("public caps and budget must be finite and nonnegative")
        target = torch.minimum(budget, caps.sum(-1, keepdim=True))
        # Individual allocations cannot exceed the total even when public cap
        # does; reducing that redundant box is exact, not a demand heuristic.
        return torch.minimum(caps, target), target

    def _sample(self, obs, deterministic=False, correction=True, *, normalized=None,
                geometry=None):
        # Callers may reuse preprocessing of this exact batch within one update.
        # No cached state survives a normalization/statistics or observation change.
        caps, target = self.constraints(obs) if geometry is None else geometry
        normalized = self.obs_norm(obs.float()) if normalized is None else normalized
        adjustment=(neutral_mean_tensor(obs,caps,target,self.neutral_context)-self.actor.neutral_anchor_mean[None]
                    if self.neutral_context is not None else None)
        basic, log_prob = self.actor.sample(normalized, caps, deterministic,
                                            mean_adjustment=adjustment)
        initial = complete_equality(basic, target)
        action = correct_inequalities(initial, caps, target) if correction else initial
        return action, log_prob, initial, caps, target

    def entropy_target(self, caps):
        """Desired pre-GRG basic-action entropy in the current box units.

        If a basic coordinate is a=(tanh(z)+1)*cap/2, then its density loses
        log(cap/2), whereas its target entropy gains that same quantity. Dirac
        (zero-width) coordinates contribute neither density nor target.
        """
        scale = caps[:, :-1].double() / 2.
        return torch.where(scale > 0., self.p["target_entropy_coef"]
                           + scale.clamp_min(1e-300).log(), 0.).sum(-1, keepdim=True)

    def entropy_upper_bound(self, caps):
        """Gaussian ceiling; tanh log-Jacobian can only reduce entropy."""
        scale = caps[:, :-1].double() / 2.
        ceiling = .5 * math.log(2. * math.pi * math.e) + self.p["log_std_max"]
        return torch.where(scale > 0., ceiling + scale.clamp_min(1e-300).log(),
                           0.).sum(-1, keepdim=True)

    def act(self, observation, deterministic=False):
        values = np.asarray(observation, dtype=np.float64)
        single = values.ndim == 1
        if values.shape[-1] != self.obs_dim or not np.isfinite(values).all():
            raise ValueError("invalid RPO observation")
        obs = torch.as_tensor(values.reshape(-1, self.obs_dim), device=self.device)
        with torch.no_grad():
            caps, target = self.constraints(obs)
            if (not deterministic and self.observed_steps < self.p["learning_starts"]
                    and not getattr(self, 'policy_warmup', False)):
                basic = torch.rand_like(caps[:, :-1]) * caps[:, :-1]
                allocation = correct_inequalities(complete_equality(basic, target), caps, target)
            else:
                allocation, _, _, _, _ = self._sample(obs, deterministic, geometry=(caps, target))
            # Any feasible allocation has a control representative in [-1,1].
            # Zeros at terminal produce -1 instead of NaN; bootstrap is masked.
            denominator = torch.where(target > 0, target, torch.ones_like(target)).double()
            control = 2. * allocation.double() / denominator - 1.
            result = control.clamp(-1., 1.).cpu().numpy()
        return result[0] if single else result

    def observe(self, obs, action, next_obs, reward, done):
        obs, nxt = np.asarray(obs, np.float64), np.asarray(next_obs, np.float64)
        action = np.asarray(action, np.float64)
        if (obs.shape != (self.obs_dim,) or nxt.shape != obs.shape or action.shape != (self.action_dim,)
                or not np.isfinite(obs).all() or not np.isfinite(nxt).all()
                or not np.isfinite(action).all() or np.any(np.abs(action) > 1. + 1e-7)
                or not math.isfinite(float(reward))):
            raise ValueError("invalid RPO transition")
        self.obs_norm.update(obs[None].astype(np.float32))
        self.replay_buffer.add(obs, action, nxt, reward, done)
        self.observed_steps += 1

    def train(self):
        if self.observed_steps < self.p["learning_starts"] or self.replay_buffer.size < self.p["batch_size"]:
            return False
        obs, control, next_obs, reward, not_done = self.replay_buffer.sample()
        normalized = self.obs_norm(obs.float())
        caps, target = self.constraints(obs)
        replay_action = target * (control + 1.) / 2.
        alpha = self.log_temperature.exp().detach()
        with torch.no_grad():
            next_normalized = self.obs_norm(next_obs.float())
            next_action, next_log, _, _, _ = self._sample(next_obs, normalized=next_normalized)
            next_q = self.critic_target(next_normalized, next_action).amin(-1, keepdim=True)
            td_target = reward + self.p["gamma"] * not_done * (next_q - alpha * next_log)
        q = self.critic(normalized, replay_action)
        critic_loss = sum(F.smooth_l1_loss(q[:, i:i + 1], td_target) for i in range(2))
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.training_steps += 1
        losses = dict(critic=float(critic_loss.detach()), temperature=float(alpha))
        if self.training_steps % self.p["policy_delay"] == 0:
            # Critic learns feasible replay actions; original RPO intentionally
            # evaluates the equality-completed initial action for actor training.
            self.critic.requires_grad_(False)
            initial, log_prob, _, caps, _ = self._sample(
                obs, correction=False, normalized=normalized, geometry=(caps, target))
            violation = inequality_violations(initial, caps)
            penalty = (violation * self.dual.detach()).sum(-1, keepdim=True)
            actor_loss = (alpha * log_prob - self.critic(normalized, initial).amin(-1, keepdim=True) + penalty).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.critic.requires_grad_(True)
            dual_objective = (self.dual * violation.detach()).sum(-1).mean()
            self.dual_optimizer.zero_grad(set_to_none=True)
            dual_objective.backward()
            self.dual_optimizer.step()
            with torch.no_grad():
                self.dual.clamp_(min=0.)
            entropy_target = self.entropy_target(caps)
            entropy_bound = self.entropy_upper_bound(caps)
            temperature_loss = -(self.log_temperature * (log_prob.detach() + entropy_target)).mean()
            self.temperature_optimizer.zero_grad(set_to_none=True)
            temperature_loss.backward()
            self.temperature_optimizer.step()
            self.actor_updates += 1
            losses.update(actor=float(actor_loss.detach()), positive_violation=float(violation.detach().sum(-1).mean()),
                          dual_max=float(self.dual.detach().max()), temperature_loss=float(temperature_loss.detach()),
                          log_prob=float(log_prob.detach().mean()), entropy=float(-log_prob.detach().mean()),
                          entropy_target=float(entropy_target.detach().mean()),
                          entropy_upper_bound=float(entropy_bound.detach().mean()),
                          entropy_target_bound_gap=float((entropy_target-entropy_bound).detach().mean()),
                          temperature_residual=float((log_prob.detach()+entropy_target).mean()))
        with torch.no_grad():
            for dest, source in zip(self.critic_target.parameters(), self.critic.parameters()):
                dest.lerp_(source, self.p["tau"])
        if not all(math.isfinite(v) for v in losses.values()):
            raise FloatingPointError("nonfinite RPO update")
        self.last_losses = losses
        return True

    def state_dict(self):
        """No replay/global RNG: common chapter-2 checkpoint owns those."""
        return copy.deepcopy(dict(revision=REVISION, obs_dim=self.obs_dim, action_dim=self.action_dim,
            parameters=self.p, observation_schema=self.observation_schema,
            actor=self.actor.state_dict(), critic=self.critic.state_dict(), critic_target=self.critic_target.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
            dual=self.dual.detach().clone(), dual_optimizer=self.dual_optimizer.state_dict(),
            log_temperature=self.log_temperature.detach().clone(), temperature_optimizer=self.temperature_optimizer.state_dict(),
            policy_warmup=self.policy_warmup,
            neutral_context=copy.deepcopy(self.neutral_context),
            obs_norm=dict(mean=self.obs_norm.mean.copy(), var=self.obs_norm.var.copy(), count=self.obs_norm.count),
            observed_steps=self.observed_steps, training_steps=self.training_steps,
            actor_updates=self.actor_updates, last_losses=self.last_losses))

    def load_state_dict(self, state):
        if (state.get("revision") != REVISION or state.get("obs_dim") != self.obs_dim
                or state.get("action_dim") != self.action_dim or state.get("parameters") != self.p
                or state.get("observation_schema") != self.observation_schema
                or state.get('policy_warmup') != self.policy_warmup
                or state.get('neutral_context') != self.neutral_context):
            raise ValueError("RPO checkpoint identity mismatch")
        for name in ("actor", "critic", "critic_target", "actor_optimizer", "critic_optimizer", "dual_optimizer", "temperature_optimizer"):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.dual.copy_(state["dual"].to(self.device))
            self.log_temperature.copy_(state["log_temperature"].to(self.device))
        self.obs_norm.mean = np.array(state["obs_norm"]["mean"], copy=True)
        self.obs_norm.var = np.array(state["obs_norm"]["var"], copy=True)
        self.obs_norm.count = float(state["obs_norm"]["count"])
        self.obs_norm._cache.clear()
        for name in ("observed_steps", "training_steps", "actor_updates"):
            setattr(self, name, int(state[name]))
        self.last_losses = copy.deepcopy(state["last_losses"])
