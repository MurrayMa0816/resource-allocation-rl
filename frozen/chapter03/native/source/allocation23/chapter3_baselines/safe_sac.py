"""Paper-derived Safe SAC candidate; this module has no training-route registration.

Source: Jiang et al., AAAI 2024, "Reward Penalties on Augmented States for
Solving Richly Constrained RL Effectively", Sec. 3, Sec. 4.2, Eq. (5), and
Safe SAC pseudocode in arXiv:2301.11592v2, Appendix B, Algorithm 2.

This is a numerical adaptation, not an official reproduction. In particular:
* Eq. (5) is solved approximately by actor candidates and gradient refinement.
  The final mask is still the Eq. (5) constraint. No global optimum is claimed.
* An empty feasible set uses the candidate with the smallest predicted total
  cost, and explicitly records that the constraint could not be satisfied.
* Both Bellman targets include the observed immediate term and terminal mask.
  The published pseudocode's overloaded alpha is resolved as reward discount
  gamma; cumulative resource cost is undiscounted, as in its RN-CMDP definition.
* The actor uses the paper's safe reward/entropy and unsafe cost score-gradient
  branches with w=1 and fresh actor samples at replay states. This specifies the
  otherwise undefined w without claiming behavior importance weights.
* The external environment reward is never changed. Reward penalties are kept
  in a separate, algorithm-internal replay field.

The local finite-horizon adaptation uses resource/B, budget=1, gamma=1 and
box actions [-1, 1]^(N+1). Its learned cost estimates are NOT hard safety
certificates. Native physical constraints remain the environment's definition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PAPER = {
    "title": "Reward Penalties on Augmented States for Solving Richly Constrained RL Effectively",
    "venue": "AAAI 2024",
    "doi": "10.1609/aaai.v38i18.29962",
    "paper": "https://ojs.aaai.org/index.php/AAAI/article/view/29962",
    "pseudocode": "https://arxiv.org/pdf/2301.11592v2",
    "official_repository": "https://github.com/xiaomeijiang258/aaaicrl",
    "inspected_commit": "f56cc1e2d47e26a680cd81d086da5ad06fb25bed",
    "implementation_status": "paper-derived continuous numerical adaptation; explicit staged numerical-learning diagnostics",
    "numerical_repair": "Actual sampled latents are retained; finite_latent search, unbiased score baseline and public cost range are separately configured; legacy defaults remain available for causal comparison",
    "hard_budget_guarantee": False,
}


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _ledger_roundoff(budget: float) -> float:
    """64 binary64 epsilons cover the 24/32-step normalized ledger addition.

    This is arithmetic tolerance, not an adjustable safety margin. Observed
    costs and history are retained unchanged; meaningful excess is penalized.
    """
    return 64 * np.finfo(np.float64).eps * max(1.0, abs(budget))


def penalized_reward(
    native_reward: float,
    history_cost: float,
    current_cost: float,
    budget: float,
    penalty: float,
    time_index: int = 0,
    reward_discount: float = 1.0,
) -> float:
    """Eq. (1), with history strictly before the current transition.

    The first crossing charges the whole accumulated cost; subsequent unsafe
    transitions charge their incremental cost. Thus the discounted trajectory
    penalty equals lambda * total_cost for a violating trajectory.
    """
    history_cost = _finite_nonnegative("history_cost", history_cost)
    current_cost = _finite_nonnegative("current_cost", current_cost)
    budget = _finite_nonnegative("budget", budget)
    penalty = _finite_nonnegative("penalty", penalty)
    if not math.isfinite(float(native_reward)):
        raise ValueError("native_reward must be finite")
    if not 0 < reward_discount <= 1:
        raise ValueError("reward_discount must be in (0, 1]")
    if not isinstance(time_index, (int, np.integer)) or time_index < 0:
        raise ValueError("time_index must be a nonnegative integer")
    tolerance = _ledger_roundoff(budget)
    if history_cost > budget + tolerance:
        charge = current_cost
    elif history_cost + current_cost > budget + tolerance:
        charge = history_cost + current_cost
    else:
        charge = 0.0
    return float(native_reward) - penalty * charge / reward_discount**time_index


def eq5_total_cost(
    cost_q1: Any,
    cost_q2: Any,
    cumulative_including_current: Any,
    current_cost: Any,
) -> Any:
    """Literal Eq. (5): max(Q_D1, Q_D2) + c - d.

    Q_D includes the current state/transition cost. Paper c also includes that
    cost, hence subtraction prevents double counting. In our replay convention
    c = history_cost + current_cost, so this is max(Q_D) + history_cost.
    """
    if torch.is_tensor(cost_q1) or torch.is_tensor(cost_q2):
        return torch.maximum(cost_q1, cost_q2) + cumulative_including_current - current_cost
    return np.maximum(cost_q1, cost_q2) + cumulative_including_current - current_cost


def select_finite_candidates(
    reward_q: np.ndarray,
    cost_q: np.ndarray,
    log_prob: np.ndarray,
    history_cost: float,
    budget: float = 1.0,
    entropy_weight: float = 0.2,
) -> tuple[int, dict[str, Any]]:
    """Exact Eq. (5) selection over the supplied finite set, no silent fallback.

    Q arrays have shape (K, 2). Ties are stable. No-feasible fallback minimizes
    predicted total cost, breaking ties by the paper's reward/entropy objective.
    This fallback is a disclosed numerical choice, not a hard safety layer.
    """
    reward_q = np.asarray(reward_q, dtype=np.float64)
    cost_q = np.asarray(cost_q, dtype=np.float64)
    log_prob = np.asarray(log_prob, dtype=np.float64).reshape(-1)
    if reward_q.ndim != 2 or reward_q.shape[1] != 2 or reward_q.shape != cost_q.shape:
        raise ValueError("reward_q and cost_q must have equal shape (K, 2)")
    if reward_q.shape[0] == 0 or len(log_prob) != reward_q.shape[0]:
        raise ValueError("a nonempty matching candidate set is required")
    if not all(np.isfinite(x).all() for x in (reward_q, cost_q, log_prob)):
        raise ValueError("candidate estimates must be finite")
    history_cost = _finite_nonnegative("history_cost", history_cost)
    budget = _finite_nonnegative("budget", budget)
    entropy_weight = _finite_nonnegative("entropy_weight", entropy_weight)
    objective = reward_q.min(axis=1) - entropy_weight * log_prob
    total_cost = cost_q.max(axis=1) + history_cost
    feasible = total_cost <= budget
    feasible_ids = np.flatnonzero(feasible)
    if len(feasible_ids):
        index = int(feasible_ids[np.argmax(objective[feasible_ids])])
        fallback = False
    else:
        smallest = np.flatnonzero(total_cost == total_cost.min())
        index = int(smallest[np.argmax(objective[smallest])])
        fallback = True
    return index, {
        "candidate_count": len(log_prob),
        "feasible_count": int(feasible.sum()),
        "fallback_used": fallback,
        "fallback_rule": "minimum_predicted_total_cost_then_maximum_eq5_objective" if fallback else None,
        "selected_predicted_total_cost": float(total_cost[index]),
        "selected_eq5_objective": float(objective[index]),
        "selected_satisfies_estimated_constraint": bool(feasible[index]),
        "hard_budget_guarantee": False,
    }


@dataclass(frozen=True)
class SafeSACConfig:
    budget: float = 1.0
    reward_discount: float = 1.0
    cost_discount: float = 1.0
    entropy_weight: float = 0.2
    hidden_sizes: tuple[int, ...] = (256, 256)
    learning_rate: float = 3e-4
    target_tau: float = 0.005
    replay_capacity: int = 100_000
    batch_size: int = 128
    initial_penalty: float = 10.0
    minimum_penalty: float = 0.1
    penalty_decay: float = 0.95
    penalty_window: int = 10
    candidate_count: int = 32
    refinement_steps: int = 5
    refinement_step_size: float = 0.03
    refinement_constraint_weight: float = 100.0
    max_grad_norm: float = 10.0
    device: str = "cpu"
    # Explicit local adaptation switches. Defaults retain the original pilot.
    candidate_solver: str = "legacy_action_refinement"
    score_baseline: str = "none"
    cost_range_mode: str = "none"
    actor_architecture: str = "flat"

    def __post_init__(self) -> None:
        if self.budget <= 0 or not math.isfinite(self.budget):
            raise ValueError("budget must be positive and finite")
        if not 0 < self.reward_discount <= 1 or self.cost_discount != 1.0:
            raise ValueError("reward discount must be in (0,1]; cumulative cost discount must be 1")
        if self.entropy_weight < 0 or self.initial_penalty < self.minimum_penalty or self.minimum_penalty < 0:
            raise ValueError("invalid entropy or penalty parameters")
        if not 0 < self.target_tau <= 1 or not 0 < self.penalty_decay <= 1:
            raise ValueError("invalid target or penalty decay")
        for value in (self.replay_capacity, self.batch_size, self.penalty_window, self.candidate_count):
            if value < 1:
                raise ValueError("capacities and candidate count must be positive")
        if self.refinement_steps < 0 or self.refinement_step_size <= 0:
            raise ValueError("invalid refinement configuration")
        if self.candidate_solver not in ("legacy_action_refinement", "finite_latent"):
            raise ValueError("unknown candidate solver")
        if self.score_baseline not in ("none", "leave_one_out", "state_mean_action"):
            raise ValueError("unknown score-gradient baseline")
        if self.cost_range_mode not in ("none", "sigmoid_remaining"):
            raise ValueError("unknown public cost range parameterization")
        if self.cost_range_mode != "none" and self.candidate_solver != "finite_latent":
            raise ValueError("Public remaining-cost bounds require the validated finite_latent solver")
        if self.candidate_solver == "finite_latent" and self.refinement_steps != 0:
            raise ValueError("finite_latent uses a declared finite candidate set, without gradient refinement")
        if self.actor_architecture not in ("flat", "shared_object_mean"):
            raise ValueError("unknown Actor mean architecture")


def _mlp(input_dim: int, sizes: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for size in sizes:
        layers.extend((nn.Linear(input_dim, size), nn.ReLU()))
        input_dim = size
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class _Actor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden: Sequence[int]):
        super().__init__()
        self.net = _mlp(observation_dim, hidden, 2 * action_dim)
        self.shared_mean = None

    def distribution(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(state).chunk(2, dim=-1)
        if self.shared_mean is not None:
            mean = self.shared_mean(state)
        return mean, log_std.clamp(-5.0, 2.0)

    def sample(self, state: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        action, log_prob, _ = self.sample_with_latent(state, generator)
        return action, log_prob

    def sample_with_latent(self, state: torch.Tensor, generator: torch.Generator):
        """Retain the actual Gaussian draw even when float32 tanh saturates.

        Inverting a clamped sampled action evaluates another latent point. It
        spuriously gives a sample near a large mean an enormous negative score.
        The pre-tanh change of variables is valid without that reconstruction.
        """
        mean, log_std = self.distribution(state)
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        raw = mean + log_std.exp() * noise
        action = raw.tanh()
        return action, self.log_prob_from_latent_stats(mean, log_std, raw), raw

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.distribution(state)
        return self.log_prob_from_stats(mean, log_std, action)

    def log_prob_from_latent(self, state: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.distribution(state)
        return self.log_prob_from_latent_stats(mean, log_std, raw)

    @staticmethod
    def log_prob_from_latent_stats(mean: torch.Tensor, log_std: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        """Exact squashed Gaussian log density at its stored latent point.

        log(1-tanh(x)^2) = 2*(log(2)-x-softplus(-2*x)). The score-gradient
        caller detaches raw: its Jacobian value then contributes no spurious
        pathwise derivative through the sampled action.
        """
        log_normal = -0.5 * (((raw - mean) / log_std.exp()) ** 2 + 2 * log_std + math.log(2 * math.pi))
        log_jacobian = 2 * (math.log(2.0) - raw - F.softplus(-2 * raw))
        return (log_normal - log_jacobian).sum(dim=-1, keepdim=True)

    @staticmethod
    def log_prob_from_stats(mean: torch.Tensor, log_std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(action)
        log_normal = -0.5 * (((raw - mean) / log_std.exp()) ** 2 + 2 * log_std + math.log(2 * math.pi))
        return (log_normal - torch.log1p(-action.square() + 1e-6)).sum(dim=-1, keepdim=True)


class _SharedObjectMean(nn.Module):
    """Generic shared object MLP, not the proposed method's response basis.

    It changes only the mean function. The original state-dependent flat
    Gaussian log-standard-deviation branch retains its initialization and full
    input. The added algorithm-owned cumulative cost is retained in context.
    """
    def __init__(self, augmented_observation_dim, action_dim):
        super().__init__()
        self.objects=action_dim-1
        public_dim=augmented_observation_dim-1
        if public_dim==3*self.objects+3:
            self.local_dim,self.public_global_dim=3,3
        elif public_dim==2*self.objects+2:
            self.local_dim,self.public_global_dim=2,2
        else:
            raise ValueError('Unrecognized public object layout; never drop augmented cost')
        self.local=_mlp(self.local_dim,(64,),32)
        context_dim=32+self.public_global_dim+1
        self.objects_head=_mlp(32+context_dim,(32,),1)
        self.total_head=_mlp(context_dim,(32,),1)

    def forward(self,state):
        split=self.objects*self.local_dim
        rows=state[...,:split].reshape(*state.shape[:-1],self.objects,self.local_dim)
        global_with_cost=state[...,split:]
        if global_with_cost.shape[-1]!=self.public_global_dim+1:
            raise ValueError('The public global input plus exact accumulated-cost feature is required')
        local=self.local(rows)
        context=torch.cat((local.mean(-2),global_with_cost),dim=-1)
        expanded=context.unsqueeze(-2).expand(*local.shape[:-1],context.shape[-1])
        object_mean=self.objects_head(torch.cat((local,expanded),dim=-1)).squeeze(-1)
        return torch.cat((self.total_head(context),object_mean),dim=-1)


class _TwinQ(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden: Sequence[int]):
        super().__init__()
        self.q1 = _mlp(observation_dim + action_dim, hidden, 1)
        self.q2 = _mlp(observation_dim + action_dim, hidden, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat((state, action), dim=-1)
        return self.q1(x), self.q2(x)


class SafeSACAgent:
    """Online, initially untrained continuous Safe SAC numerical candidate.

    The public API accepts native observations and box actions. Historical cost
    is an extra algorithm-owned feature; costs are actual consumed resources/B.
    """

    def __init__(
        self,
        observation_dim: int,
        action_low: Sequence[float],
        action_high: Sequence[float],
        config: SafeSACConfig | None = None,
        seed: int = 0,
    ):
        self.config = config or SafeSACConfig()
        self.device = torch.device(self.config.device)
        self.observation_dim = int(observation_dim)
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        if self.observation_dim <= 0 or not len(self.action_low):
            raise ValueError("observation and action dimensions must be positive")
        if self.action_low.shape != self.action_high.shape or not np.isfinite(self.action_low).all() or not np.isfinite(self.action_high).all() or not (self.action_low < self.action_high).all():
            raise ValueError("action bounds must be finite, matching, and nondegenerate")
        self.action_dim = len(self.action_low)
        self._scale = (self.action_high - self.action_low) / 2
        self._midpoint = (self.action_high + self.action_low) / 2
        self.rng = np.random.default_rng(seed)
        self.torch_rng = torch.Generator(device=self.device).manual_seed(seed)
        # Preserve the host process RNG during candidate-network initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.actor = _Actor(self.observation_dim + 1, self.action_dim, self.config.hidden_sizes).to(self.device)
            self.reward_q = _TwinQ(self.observation_dim + 1, self.action_dim, self.config.hidden_sizes).to(self.device)
            self.cost_q = _TwinQ(self.observation_dim + 1, self.action_dim, self.config.hidden_sizes).to(self.device)
        if self.config.actor_architecture == "shared_object_mean":
            # The original Actor and all Critics were initialized above with
            # identical RNG use. Construct only the new mean inside a separate
            # host-RNG fork, preserving the flat log_std and every Critic tensor.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed)+62491)
                self.actor.shared_mean = _SharedObjectMean(self.observation_dim+1,self.action_dim).to(self.device)
        self.reward_target = copy.deepcopy(self.reward_q).requires_grad_(False)
        self.cost_target = copy.deepcopy(self.cost_q).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.reward_optimizer = torch.optim.Adam(self.reward_q.parameters(), lr=self.config.learning_rate)
        self.cost_optimizer = torch.optim.Adam(self.cost_q.parameters(), lr=self.config.learning_rate)
        self.penalty = self.config.initial_penalty
        self._recent_episode_costs: list[float] = []
        self._replay: list[tuple[Any, ...]] = []
        self._replay_cursor = 0
        self.last_action_info: dict[str, Any] = {}
        self.evaluation_only = False
        self.total_updates = 0
        self.transitions_seen = 0
        self.penalized_transitions = 0

    def _state(self, observation: Sequence[float], history_cost: float) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        if observation.shape != (self.observation_dim,) or not np.isfinite(observation).all():
            raise ValueError("native observation has wrong dimension or nonfinite values")
        history_cost = _finite_nonnegative("history_cost", history_cost)
        return np.concatenate((observation, np.array([history_cost / self.config.budget], dtype=np.float32)))

    def _tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _candidate_estimates(self, states: torch.Tensor, actions: torch.Tensor, actor_stats=None):
        reward = self.reward_q(states, actions)
        cost = self._cost_values(self.cost_q, states, actions)
        log_prob = (self.actor.log_prob(states, actions) if actor_stats is None else
                    self.actor.log_prob_from_stats(*actor_stats, actions))
        return reward, cost, log_prob

    def act(self, observation: Sequence[float], history_cost: float, deterministic: bool = False) -> np.ndarray:
        if self.config.candidate_solver == "finite_latent":
            return self._finite_latent_act(observation, history_cost, deterministic)
        return self._legacy_act(observation, history_cost, deterministic)

    def _cost_values(self, network, states: torch.Tensor, actions: torch.Tensor):
        values = network(states, actions)
        if self.config.cost_range_mode == "sigmoid_remaining":
            # Current history is public and already part of the augmented state.
            # Native consumable-resource feasibility implies future total cost
            # cannot exceed B-history. This is a task-specific numerical prior,
            # never a hidden future trajectory or a learned safety certificate.
            remaining = (self.config.budget * (1.0 - states[:, -1:])).clamp(0.0, self.config.budget)
            return tuple(remaining * torch.sigmoid(q) for q in values)
        return values

    def _finite_latent_act(self, observation, history_cost, deterministic):
        """Finite Eq5 search over actor nodes, with their actual latent density.

        Continuous gradient ascent of Q-alpha*log(pi) need not have a bounded
        optimum on a tanh box: inverse-density diverges at the Gaussian tails.
        The numerical candidate problem is therefore stated as finite, without
        chasing arbitrary box boundaries or reconstructing another latent draw.
        """
        state = self._tensor(self._state(observation, history_cost)).unsqueeze(0)
        count = self.config.candidate_count
        states = state.expand(count, -1)
        with torch.no_grad():
            mean, log_std = self.actor.distribution(state)
            if deterministic:
                quantiles = (torch.arange(count, device=self.device, dtype=torch.float32) + 0.5) / count
                z = math.sqrt(2.0) * torch.erfinv(2 * quantiles - 1)
                offsets = torch.arange(self.action_dim, device=self.device)
                noise = z[(torch.arange(count, device=self.device)[:, None] + offsets[None, :]) % count]
            else:
                noise = torch.randn((count, self.action_dim), dtype=mean.dtype,
                                    device=self.device, generator=self.torch_rng)
            raw = mean + log_std.exp() * noise
            raw[0] = mean[0]
            actions = raw.tanh()
            log_prob = self.actor.log_prob_from_latent_stats(mean, log_std, raw)
            reward = self.reward_q(states, actions)
            cost = self._cost_values(self.cost_q, states, actions)
            host = torch.cat((*reward, *cost, log_prob, actions), dim=-1).cpu().numpy()
            guarded_history = (min(history_cost, self.config.budget)
                if history_cost <= self.config.budget + _ledger_roundoff(self.config.budget)
                else history_cost)
            selection_costs = host[:, 2:4]
            support_clip = 0.0
            if self.config.cost_range_mode == "sigmoid_remaining":
                # The NN history coordinate is float32, but the algorithm's
                # true accumulated cost is float64. Project IN FLOAT64 onto
                # the actual public remaining support, rather than adding an
                # arbitrary tolerance to the Eq5 budget. Assigning back into
                # the float32 host array would recreate the same false reject.
                raw_selection_costs = np.asarray(selection_costs, dtype=np.float64)
                remaining_exact = max(0.0, self.config.budget - guarded_history)
                selection_costs = np.clip(raw_selection_costs, 0.0, remaining_exact)
                support_clip = float(np.max(np.abs(raw_selection_costs - selection_costs)))
            index, info = select_finite_candidates(host[:, :2], selection_costs, host[:, 4],
                guarded_history, self.config.budget, self.config.entropy_weight)
            normalized_action = host[index, 5:]
        self.last_action_info = {
            **info, "solver": "finite_actor_latent_nodes", "refinement_steps": 0,
            "selection_is_global_optimum": False,
            "density_uses_actual_latent": True,
            "history_cost": float(history_cost), "penalty": float(self.penalty),
            "cost_range_mode": self.config.cost_range_mode,
            "selected_log_probability": float(host[index, 4]),
            "candidate_log_probability_min": float(host[:, 4].min()),
            "candidate_log_probability_max": float(host[:, 4].max()),
            "candidate_cost_prediction_max": float(np.max(selection_costs)),
            "candidate_cost_raw_before_support_projection_max": float(host[:, 2:4].max()),
            "known_cost_support_float64_clip_max": support_clip,
            "candidate_reward_prediction_absmax": float(np.abs(host[:, :2]).max()),
            "candidate_action_saturation_fraction": float((np.abs(host[:, 5:]) >= 1.0-1e-6).mean()),
        }
        return (self._midpoint + self._scale * normalized_action).astype(np.float32)

    def _legacy_act(self, observation: Sequence[float], history_cost: float, deterministic: bool = False) -> np.ndarray:
        state = self._tensor(self._state(observation, history_cost)).unsqueeze(0)
        count = self.config.candidate_count
        states = state.expand(count, -1)
        with torch.no_grad():
            mean, _ = self.actor.distribution(state)
            if deterministic:
                # Deterministic quantiles cover the same actor distribution with
                # no evaluation RNG consumption; they are numerical search nodes.
                quantiles = (torch.arange(count, device=self.device, dtype=torch.float32) + 0.5) / count
                z = math.sqrt(2.0) * torch.erfinv(2 * quantiles - 1)
                dimension_offsets = torch.arange(self.action_dim, device=self.device)
                noise = z[(torch.arange(count, device=self.device)[:, None] + dimension_offsets[None, :]) % count]
                mu, log_std = self.actor.distribution(states)
                actions = torch.tanh(mu + log_std.exp() * noise)
            else:
                actions, _ = self.actor.sample(states, self.torch_rng)
            actions[0] = mean.tanh()[0]
            original_actions = actions.clone()
            # State and actor weights do not change within action refinement.
            # Keep the same batch shape as each original distribution forward.
            refinement_stats = self.actor.distribution(states)
        finite_gradients = []
        for _ in range(self.config.refinement_steps):
            actions = actions.detach().requires_grad_(True)
            reward, cost, log_prob = self._candidate_estimates(states, actions, refinement_stats)
            score = torch.minimum(*reward) - self.config.entropy_weight * log_prob
            violation = F.relu(torch.maximum(*cost) + history_cost - self.config.budget)
            merit = score - self.config.refinement_constraint_weight * violation.square()
            gradient = torch.autograd.grad(merit.sum(), actions)[0]
            finite_gradients.append(torch.isfinite(gradient).all())
            actions = (actions + self.config.refinement_step_size * gradient).clamp(-1 + 1e-6, 1 - 1e-6)
        # Every refinement gradient is still checked before any action executes;
        # combine only their scalar transfer, avoiding G host synchronizations.
        if finite_gradients and not torch.stack(finite_gradients).all():
            raise FloatingPointError("nonfinite action-search gradient")
        with torch.no_grad():
            actions = torch.cat((original_actions, actions.detach()), dim=0)
            states = state.expand(len(actions), -1)
            reward, cost, log_prob = self._candidate_estimates(states, actions)
            # One transfer also carries every original float32 candidate action.
            host = torch.cat((*reward, *cost, log_prob, actions), dim=-1).cpu().numpy()
            index, info = select_finite_candidates(host[:, :2], host[:, 2:4], host[:, 4], history_cost, self.config.budget, self.config.entropy_weight)
            normalized_action = host[index, 5:]
        self.last_action_info = {
            **info,
            "solver": "actor_candidates_and_gradient_refinement",
            "refinement_steps": self.config.refinement_steps,
            "selection_is_global_optimum": False,
            "history_cost": float(history_cost),
            "penalty": float(self.penalty),
        }
        return (self._midpoint + self._scale * normalized_action).astype(np.float32)

    def observe(
        self,
        observation: Sequence[float],
        history_cost: float,
        action: Sequence[float],
        native_reward: float,
        cost: float,
        next_observation: Sequence[float],
        done: bool,
        time_index: int = 0,
    ) -> dict[str, float]:
        """Store the actually executed action, native reward, and observed cost."""
        if self.evaluation_only:
            raise RuntimeError("Evaluation-only Safe SAC snapshots cannot collect training replay")
        cost = _finite_nonnegative("cost", cost)
        state = self._state(observation, history_cost)
        next_state = self._state(next_observation, history_cost + cost)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all() or (action < self.action_low - 1e-6).any() or (action > self.action_high + 1e-6).any():
            raise ValueError("executed action is outside the declared box")
        normalized = np.clip((action - self._midpoint) / self._scale, -1, 1)
        training_reward = penalized_reward(native_reward, history_cost, cost, self.config.budget, self.penalty, time_index, self.config.reward_discount)
        transition = (state, normalized.copy(), float(native_reward), training_reward, cost, next_state, float(bool(done)), float(history_cost))
        if len(self._replay) < self.config.replay_capacity:
            self._replay.append(transition)
        else:
            self._replay[self._replay_cursor] = transition
        self._replay_cursor = (self._replay_cursor + 1) % self.config.replay_capacity
        self.transitions_seen += 1
        self.penalized_transitions += int(training_reward != float(native_reward))
        total = history_cost + cost
        return {"native_reward": float(native_reward), "algorithm_training_reward": training_reward, "observed_cost": cost, "penalty_at_collection": float(self.penalty),
                "ledger_roundoff_guard_used": float(self.config.budget < total <= self.config.budget + _ledger_roundoff(self.config.budget))}

    def update(self, batch_size: int | None = None) -> dict[str, float]:
        if self.evaluation_only:
            raise RuntimeError("Evaluation-only Safe SAC snapshots cannot resume training")
        batch_size = self.config.batch_size if batch_size is None else int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(self._replay) < batch_size:
            return {"updated": 0.0, "replay_size": float(len(self._replay))}
        indices = self.rng.choice(len(self._replay), batch_size, replace=False)
        columns = list(zip(*(self._replay[int(i)] for i in indices)))
        # A single exact-value upload; contiguous slices preserve the original
        # GEMM layouts. This changes neither replay selection nor float values.
        widths = (self.observation_dim + 1, self.action_dim, 1, 1, 1,
                  self.observation_dim + 1, 1, 1)
        packed = np.concatenate([
            np.asarray(column, dtype=np.float32).reshape(batch_size, width)
            for column, width in zip(columns, widths)
        ], axis=1)
        state, action, native_reward, training_reward, cost, next_state, done, history = (
            value.contiguous() for value in self._tensor(packed).split(widths, dim=1)
        )
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state, self.torch_rng)
            reward_next = torch.minimum(*self.reward_target(next_state, next_action))
            cost_next = torch.maximum(*self._cost_values(self.cost_target, next_state, next_action))
            reward_target = training_reward + self.config.reward_discount * (1 - done) * (reward_next - self.config.entropy_weight * next_log_prob)
            cost_target = cost + self.config.cost_discount * (1 - done) * cost_next
        reward_estimates = self.reward_q(state, action)
        cost_estimates = self._cost_values(self.cost_q, state, action)
        reward_loss = sum(F.mse_loss(q, reward_target) for q in reward_estimates)
        cost_loss = sum(F.mse_loss(q, cost_target) for q in cost_estimates)
        if not torch.isfinite(torch.stack((reward_loss.detach(), cost_loss.detach()))).all():
            raise FloatingPointError("nonfinite Safe SAC critic loss before optimizer step")
        self.reward_optimizer.zero_grad(set_to_none=True)
        reward_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_q.parameters(), self.config.max_grad_norm, error_if_nonfinite=True)
        self.reward_optimizer.step()
        self.cost_optimizer.zero_grad(set_to_none=True)
        cost_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cost_q.parameters(), self.config.max_grad_norm, error_if_nonfinite=True)
        self.cost_optimizer.step()

        # Fresh actor samples give the paper's score-gradient branches without
        # inventing a likelihood for the deterministic Eq. (5) search policy.
        with torch.no_grad():
            sampled_action, _, sampled_raw = self.actor.sample_with_latent(state, self.torch_rng)
            reward_weight = torch.minimum(*self.reward_q(state, sampled_action))
            cost_weight = torch.maximum(*self._cost_values(self.cost_q, state, sampled_action))
            if self.config.cost_range_mode == "sigmoid_remaining":
                # Compare in the same replay coordinate. Qc<=B-history is
                # algebraically equivalent to Qc+history<=B, without a second
                # floating-point addition that can cross the exact boundary.
                remaining_replay = self.config.budget-history
                estimated_safe = cost_weight <= remaining_replay
            else:
                estimated_safe = cost_weight + history <= self.config.budget
        actor_mean, actor_log_std = self.actor.distribution(state)
        log_prob = self.actor.log_prob_from_latent_stats(actor_mean, actor_log_std, sampled_raw.detach())
        safe_advantage = reward_weight - self.config.entropy_weight * log_prob.detach()
        safe_loss = -log_prob * safe_advantage
        unsafe_loss = log_prob * cost_weight
        if self.config.score_baseline == "state_mean_action":
            # A deterministic function of the current replay state is independent
            # of this iteration's fresh actor draw. Its parameter dependence is
            # detached: subtracting b(s) leaves E[score*b(s)] exactly zero.
            # Unlike a global batch mean, it cancels the large finite-horizon
            # soft-value variation between early and late states.
            with torch.no_grad():
                baseline_action=actor_mean.detach().tanh()
                baseline_logp=self.actor.log_prob_from_latent_stats(
                    actor_mean.detach(),actor_log_std.detach(),actor_mean.detach())
                baseline_reward=torch.minimum(*self.reward_q(state,baseline_action))
                baseline_cost=torch.maximum(*self._cost_values(self.cost_q,state,baseline_action))
                if self.config.cost_range_mode=="sigmoid_remaining":
                    baseline_safe=baseline_cost<=self.config.budget-history
                else:
                    baseline_safe=baseline_cost+history<=self.config.budget
                baseline=torch.where(baseline_safe,
                    baseline_reward-self.config.entropy_weight*baseline_logp,-baseline_cost)
                combined_weight=torch.where(estimated_safe,safe_advantage,-cost_weight).detach()
            diagnostic_weight=combined_weight-baseline
            actor_loss=(-log_prob*diagnostic_weight).mean()
        elif self.config.score_baseline == "leave_one_out" and batch_size > 1:
            # Baseline is subtracted AFTER combining the safe/unsafe weight.
            # Other samples' actions are independent of this sample's action;
            # hence E[score*b]=0. Subtracting inside the action-dependent safe
            # mask would instead change the policy-gradient expectation.
            combined_weight = torch.where(estimated_safe, safe_advantage, -cost_weight).detach()
            baseline = (combined_weight.sum() - combined_weight) / (batch_size - 1)
            actor_loss = (-log_prob * (combined_weight - baseline)).mean()
            diagnostic_weight = combined_weight - baseline
        else:
            actor_loss = torch.where(estimated_safe, safe_loss, unsafe_loss).mean()
            diagnostic_weight = torch.where(estimated_safe, safe_advantage, -cost_weight).detach()
        with torch.no_grad():
            # Distribution-parameter score dispersion, not an estimate of the
            # complete neural-network gradient variance or a causal guarantee.
            normalized_noise = (sampled_raw - actor_mean.detach()) / actor_log_std.detach().exp()
            distribution_score = torch.cat((normalized_noise / actor_log_std.detach().exp(),
                                            normalized_noise.square() - 1.0), dim=-1)
            weighted_score = diagnostic_weight * distribution_score
            score_variance_trace = weighted_score.var(dim=0, unbiased=False).sum()
            score_rms = weighted_score.square().sum(dim=-1).mean().sqrt()
            exact_remaining = (self.config.budget - history).clamp(0., self.config.budget)
            cost_prediction_mae = (torch.maximum(*cost_estimates).detach() - exact_remaining).abs().mean()
        if not torch.isfinite(actor_loss.detach()):
            raise FloatingPointError("nonfinite Safe SAC actor loss before optimizer step")
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm, error_if_nonfinite=True)
        self.actor_optimizer.step()
        with torch.no_grad():
            for source, target in ((self.reward_q, self.reward_target), (self.cost_q, self.cost_target)):
                for parameter, target_parameter in zip(source.parameters(), target.parameters()):
                    target_parameter.lerp_(parameter, self.config.target_tau)
        names = ("reward_critic_loss", "cost_critic_loss", "actor_loss",
                 "safe_actor_fraction", "unsafe_actor_fraction", "native_reward_mean",
                 "algorithm_training_reward_mean", "observed_cost_mean",
                 "reward_target_mean", "cost_target_mean", "reward_target_absmax",
                 "cost_target_max", "actor_log_probability_absmax",
                 "actor_raw_absmax", "actor_action_saturation_fraction",
                 "actor_gradient_norm_before_clip", "score_weight_mean", "score_weight_rms",
                 "distribution_score_gradient_rms", "distribution_score_sample_variance_trace",
                 "cost_prediction_mae_to_public_remaining")
        values = torch.stack((reward_loss.detach(), cost_loss.detach(), actor_loss.detach(),
                              estimated_safe.float().mean(), (~estimated_safe).float().mean(),
                              native_reward.mean(), training_reward.mean(), cost.mean(),
                              reward_target.mean(), cost_target.mean(), reward_target.abs().max(),
                              cost_target.max(), log_prob.detach().abs().max(),
                              sampled_raw.abs().max(), (sampled_action.abs() >= 1.0-1e-6).float().mean(),
                              actor_gradient_norm.detach(), diagnostic_weight.mean(), diagnostic_weight.square().mean().sqrt(),
                              score_rms, score_variance_trace, cost_prediction_mae)).cpu().tolist()
        metrics = {"updated": 1.0, **dict(zip(names, values)), "penalty": float(self.penalty)}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("nonfinite Safe SAC update")
        self.total_updates += 1
        return metrics

    def end_episode(self, total_cost: float) -> dict[str, Any]:
        total_cost = _finite_nonnegative("total_cost", total_cost)
        if self.evaluation_only:
            return {"total_cost": total_cost,
                    "violated_budget": total_cost > self.config.budget + _ledger_roundoff(self.config.budget),
                    "old_penalty": self.penalty, "penalty": self.penalty,
                    "evaluation_only": True}
        self._recent_episode_costs.append(total_cost)
        old_penalty = self.penalty
        if len(self._recent_episode_costs) >= self.config.penalty_window:
            if max(self._recent_episode_costs) < self.config.budget and self.config.penalty_decay * self.penalty > self.config.minimum_penalty:
                self.penalty *= self.config.penalty_decay
            self._recent_episode_costs.clear()
        return {"total_cost": total_cost, "violated_budget": total_cost > self.config.budget + _ledger_roundoff(self.config.budget), "old_penalty": old_penalty, "penalty": self.penalty}

    def state_dict(self, include_training: bool = False) -> dict[str, Any]:
        """An evaluation snapshot by default; no disk write occurs here."""
        if self.evaluation_only and include_training:
            raise RuntimeError("An evaluation snapshot cannot be promoted to a resumable training snapshot")
        result = {
            "paper": dict(PAPER), "config": asdict(self.config),
            "artifact_kind": "full_training_state" if include_training else "terminal_evaluation",
            "evaluation_only": not include_training,
            "observation_dim": self.observation_dim,
            "action_low": self.action_low.copy(), "action_high": self.action_high.copy(),
            "actor": self.actor.state_dict(), "reward_q": self.reward_q.state_dict(),
            "cost_q": self.cost_q.state_dict(), "penalty": self.penalty,
            "total_updates": self.total_updates,
            "transitions_seen": self.transitions_seen,
            "penalized_transitions": self.penalized_transitions,
            "last_action_info": copy.deepcopy(self.last_action_info),
        }
        if include_training:
            result.update({
                "reward_target": self.reward_target.state_dict(), "cost_target": self.cost_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "reward_optimizer": self.reward_optimizer.state_dict(), "cost_optimizer": self.cost_optimizer.state_dict(),
                "replay": copy.deepcopy(self._replay), "replay_cursor": self._replay_cursor,
                "numpy_rng": copy.deepcopy(self.rng.bit_generator.state), "torch_rng": self.torch_rng.get_state(),
                "recent_episode_costs": list(self._recent_episode_costs),
            })
        return result

    def load_state_dict(self, state: dict[str, Any], load_training: bool = False) -> None:
        if load_training and (state.get("evaluation_only", False) or
                              state.get("artifact_kind", "full_training_state") != "full_training_state"):
            raise ValueError("Terminal evaluation snapshots do not contain exact training continuation state")
        if load_training and not all(name in state for name in ("replay", "reward_target", "cost_target",
                "actor_optimizer", "reward_optimizer", "cost_optimizer", "numpy_rng", "torch_rng")):
            raise ValueError("Complete replay/optimizer/target/RNG state is required to resume training")
        if int(state["observation_dim"]) != self.observation_dim or not np.array_equal(state["action_low"], self.action_low) or not np.array_equal(state["action_high"], self.action_high):
            raise ValueError("snapshot observation/action contract mismatch")
        if state["config"] != asdict(self.config):
            raise ValueError("snapshot configuration mismatch")
        for name in ("actor", "reward_q", "cost_q"):
            getattr(self, name).load_state_dict(state[name])
        self.penalty = float(state["penalty"])
        self.total_updates = int(state["total_updates"])
        self.transitions_seen = int(state.get("transitions_seen", len(state.get("replay", []))))
        self.penalized_transitions = int(state.get("penalized_transitions", 0))
        self.last_action_info = copy.deepcopy(state.get("last_action_info", {}))
        self.evaluation_only = not load_training
        if load_training:
            for name in ("reward_target", "cost_target", "actor_optimizer", "reward_optimizer", "cost_optimizer"):
                getattr(self, name).load_state_dict(state[name])
            self._replay = copy.deepcopy(state["replay"])
            self._replay_cursor = int(state["replay_cursor"])
            self.rng.bit_generator.state = copy.deepcopy(state["numpy_rng"])
            self.torch_rng.set_state(state["torch_rng"])
            self._recent_episode_costs = list(state["recent_episode_costs"])
        else:
            self.reward_target.load_state_dict(state["reward_q"])
            self.cost_target.load_state_dict(state["cost_q"])
