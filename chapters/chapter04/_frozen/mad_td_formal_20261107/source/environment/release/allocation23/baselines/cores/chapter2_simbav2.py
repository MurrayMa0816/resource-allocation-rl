"""SAC--SimBaV2 (ICML 2025), a PyTorch port for chapter-2 interactions.

Formula/source reference: DAVIAN-Robotics/SimbaV2, Apache-2.0, commit
86899c277cdc697b2b02d827243de1ea93f20a1d; see SOURCE below. This is V2,
not the LayerNorm/AdamW SimBa-v1 architecture in recent_continuous.py.

Retained mechanisms: hyperspherical embedding and residual interpolation,
normalized HyperDense weights, learned scaling vectors, twin categorical
critics, running-return reward scaling, and actor -> temperature -> critic
updates. Adam moments are retained when online weights are renormalized;
the Polyak target is deliberately not renormalized (as in the author code).

Adaptations: PyTorch instead of JAX, one-step replay, undiscounted finite task,
single-environment running statistics, generic bounded control coordinates.
The categorical value support is separate from reward normalization. It is
registered before training using the horizon, action dimension and initial
entropy temperature, with the author's 0.1 atom spacing retained. This is an
expected soft-return scale estimate, not a bound on individual samples or on
later temperatures; upper/lower clipping diagnostics remain necessary.
The environment wrapper, not this learner, converts controls into feasible
resource quantities. No environment derivative or additional reward is used.
observe() updates raw observation moments once per consumed transition;
act() and evaluation never update statistics. Replay stores unscaled rewards.
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


REVISION = "chapter2-sac-simbav2-v2-finite-soft-support"
SOURCE = {
    "paper": "https://proceedings.mlr.press/v267/lee25u.html",
    "repository": "https://github.com/DAVIAN-Robotics/SimbaV2",
    "commit": "86899c277cdc697b2b02d827243de1ea93f20a1d",
    "license": "Apache-2.0",
    "source_sha256": {
        "scale_rl/agents/simbaV2/simbaV2_agent.py": "5c75fe55ce591b9440232b1b31534ce9fec6d16501486b3599dae9987526bfd6",
        "scale_rl/agents/simbaV2/simbaV2_layer.py": "b40e9761ff3a0750786e4d15d08e6529e020fc42e38c38231256fc73154f1fae",
        "scale_rl/agents/simbaV2/simbaV2_network.py": "76de34c3713eec537a885ac3f1b1372e3ceeed2a3d0e2cdfb91cd03d4036b4bf",
        "scale_rl/agents/simbaV2/simbaV2_update.py": "e32335dcd1384ff4de4e51d2a7c9d056b52dbd27364446a607715bdf2b17dd1a",
        "scale_rl/agents/wrappers/normalization.py": "7e8dcaa481ae86f103661883246c3b2cd610ae44692e8bbd2c7cecf05ce78f15",
        "configs/agent/simbaV2.yaml": "6b5f80407b4cee9a1dce6aa5bd02afd43ce8690c7219ace6306c0519ec24beb2",
    },
}


def finite_task_value_support(horizon, action_dim, *, reward_g_max=5.,
                              initial_temperature=.01, spacing=.1):
    """Register a fixed support without changing reward normalization.

    A density on [-1,1]^D has expected entropy at most D*log(2).
    With only a terminal reward, its normalized value is at most g_max;
    H-1 remaining entropy bonuses give an initial-temperature scale estimate.
    Reserve 25% headroom and round to ten value units. Neither pointwise
    log-density nor a learned temperature is bounded by this estimate.
    """
    for name, value in (("horizon", horizon), ("action_dim", action_dim)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    for value in (reward_g_max, initial_temperature, spacing):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("support calibration values must be finite and positive")
    scale = reward_g_max + initial_temperature * (horizon - 1) * action_dim * math.log(2.)
    radius = 10. * math.ceil(1.25 * scale / 10.)
    intervals = round(2. * radius / spacing)
    if not math.isclose(intervals * spacing, 2. * radius, rel_tol=0., abs_tol=1e-9):
        raise ValueError("support span must be divisible by atom spacing")
    return dict(critic_min_v=-radius, critic_max_v=radius, num_bins=intervals + 1)


def simbav2_parameters(*, tiny=False, horizon=40, action_dim=120):
    """Author network/optimizer defaults plus declared finite-task support.

    ``total_updates`` must be overwritten when the registered interaction
    budget changes. Tiny dimensions/UTD are engineering checks, not results.
    """
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    warmup = (2 if tiny else 8) * horizon
    utd = 1 if tiny else 2
    parameters = dict(
        gamma=1., batch_size=16 if tiny else 256,
        buffer_size=max(512 if tiny else 200000, warmup + horizon),
        learning_starts=warmup, updates_per_step=utd,
        actor_hidden_dim=32 if tiny else 128,
        critic_hidden_dim=32 if tiny else 512,
        actor_blocks=1, critic_blocks=2,
        learning_rate=1e-4, learning_rate_end=5e-5,
        total_updates=max(1, (320 * horizon - warmup) * utd),
        tau=.005, initial_temperature=.01, target_entropy_coef=-.5,
        c_shift=3., normalized_g_max=5.,
    )
    parameters.update(finite_task_value_support(horizon, action_dim,
        reward_g_max=parameters["normalized_g_max"],
        initial_temperature=parameters["initial_temperature"]))
    return parameters


def l2normalize(x, dim=-1):
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(1e-8)


class HyperDense(nn.Linear):
    """Unit rows in Torch correspond to unit columns of a Flax kernel."""
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.weight)
        self.renormalize()

    @torch.no_grad()
    def renormalize(self):
        self.weight.copy_(l2normalize(self.weight, dim=1))


@torch.no_grad()
def renormalize_network(network):
    for module in network.modules():
        if isinstance(module, HyperDense):
            module.renormalize()


class Scaler(nn.Module):
    def __init__(self, width, initial=1., scale=1.):
        super().__init__()
        self.value = nn.Parameter(torch.full((width,), float(scale)))
        self.forward_scale = float(initial / scale)

    def forward(self, x):
        return x * self.value * self.forward_scale


class HyperEmbedder(nn.Module):
    def __init__(self, input_dim, width, c_shift=3.):
        super().__init__()
        self.c_shift = float(c_shift)
        self.linear = HyperDense(input_dim + 1, width)
        scale = math.sqrt(2. / width)
        self.scale = Scaler(width, scale, scale)

    def forward(self, x):
        shift = torch.full_like(x[..., :1], self.c_shift)
        x = l2normalize(torch.cat((x, shift), -1))
        return l2normalize(self.scale(self.linear(x)))


class HyperLERPBlock(nn.Module):
    def __init__(self, width, blocks):
        super().__init__()
        self.expand = HyperDense(width, 4 * width)
        scale = math.sqrt(2. / width) / 2.
        self.scale = Scaler(4 * width, scale, scale)
        self.contract = HyperDense(4 * width, width)
        self.alpha = Scaler(width, 1. / (blocks + 1), 1. / math.sqrt(width))

    def forward(self, x):
        update = self.contract(F.relu(self.scale(self.expand(x))) + 1e-8)
        update = l2normalize(update)
        return l2normalize(x + self.alpha(update - x))


class HyperEncoder(nn.Module):
    def __init__(self, input_dim, width, blocks, c_shift):
        super().__init__()
        self.embedder = HyperEmbedder(input_dim, width, c_shift)
        self.blocks = nn.Sequential(*(HyperLERPBlock(width, blocks) for _ in range(blocks)))

    def forward(self, x):
        return self.blocks(self.embedder(x))


class HyperHead(nn.Module):
    def __init__(self, width, output_dim):
        super().__init__()
        self.first = HyperDense(width, width)
        self.scale = Scaler(width)
        self.last = HyperDense(width, output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        return self.last(self.scale(self.first(x))) + self.bias


class SimbaV2Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, p):
        super().__init__()
        width = p["actor_hidden_dim"]
        self.encoder = HyperEncoder(obs_dim, width, p["actor_blocks"], p["c_shift"])
        self.mean = HyperHead(width, action_dim)
        self.log_std = HyperHead(width, action_dim)

    def sample(self, obs, deterministic=False):
        hidden = self.encoder(obs)
        mean = self.mean(hidden)
        if deterministic:
            return mean.tanh(), None
        log_std = -10. + 6. * (1. + self.log_std(hidden).tanh())
        # Bounded log_std analytically makes the scale positive. Distribution
        # argument validation otherwise synchronizes CUDA with the host twice
        # per sample. Finite actions/losses are checked by the adapter/core.
        distribution = torch.distributions.Normal(mean, log_std.exp(), validate_args=False)
        raw = distribution.rsample()
        log_prob = (distribution.log_prob(raw) - 2. * (
            math.log(2.) - raw - F.softplus(-2. * raw))).sum(-1, keepdim=True)
        return raw.tanh(), log_prob


class CategoricalQ(nn.Module):
    def __init__(self, obs_dim, action_dim, p):
        super().__init__()
        width = p["critic_hidden_dim"]
        self.encoder = HyperEncoder(obs_dim + action_dim, width, p["critic_blocks"], p["c_shift"])
        self.head = HyperHead(width, p["num_bins"])
        self.register_buffer("support", torch.linspace(p["critic_min_v"],
                                                        p["critic_max_v"], p["num_bins"]))

    def forward(self, obs, action):
        log_prob = self.head(self.encoder(torch.cat((obs, action), -1))).log_softmax(-1)
        expected = (log_prob.exp() * self.support).sum(-1)
        return expected, log_prob


class SimbaV2Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, p):
        super().__init__()
        self.qs = nn.ModuleList([CategoricalQ(obs_dim, action_dim, p) for _ in range(2)])

    @property
    def support(self):
        return self.qs[0].support

    def forward(self, obs, action):
        outputs = [q(obs, action) for q in self.qs]
        return torch.stack([x[0] for x in outputs]), torch.stack([x[1] for x in outputs])


@torch.no_grad()
def categorical_projection(probabilities, reward, not_done, entropy_cost, support, gamma=1.):
    """C51 projection of r + gamma*(Z - alpha*log pi)*(1-done).

    Returns projected probabilities and *unclipped* target support, so callers
    can report support saturation rather than hide it. Integer-bin mass is
    preserved exactly, including the terminal delta distribution.
    """
    n_bins = support.numel()
    targets = reward.reshape(-1, 1) + gamma * not_done.reshape(-1, 1) * (
        support.reshape(1, -1) - entropy_cost.reshape(-1, 1))
    lo, hi = support[0], support[-1]
    b = ((targets.clamp(lo, hi) - lo) / ((hi - lo) / (n_bins - 1))).clamp(0, n_bins - 1)
    lower, upper = b.floor().long(), b.ceil().long()
    result = torch.zeros_like(probabilities)
    result.scatter_add_(1, lower, probabilities * (upper.to(b.dtype) - b + (lower == upper)))
    result.scatter_add_(1, upper, probabilities * (b - lower.to(b.dtype)))
    return result, targets


class RunningRewardScale:
    """Author wrapper recurrence, including terminal masking *before* addition.

    G <- gamma*(1-done)*G+r; scale=max(sqrt(Var[G]+eps),max_abs_G/gmax).
    Values are statistics only; raw rewards and terminal business metrics stay
    unchanged. Uniformly scaling all raw rewards leaves normalized scale equal.
    """
    def __init__(self, gamma, g_max):
        self.gamma, self.g_max = float(gamma), float(g_max)
        self.G = self.max_abs_return = 0.
        self.rms = RunningObservationNorm(1)

    def update(self, reward, done):
        self.G = self.gamma * (1. - float(done)) * self.G + float(reward)
        self.rms.update(np.array([[self.G]], dtype=np.float64))
        self.max_abs_return = max(self.max_abs_return, abs(self.G))

    @property
    def denominator(self):
        return max(float(np.sqrt(self.rms.var[0] + 1e-8)), self.max_abs_return / self.g_max)

    def __call__(self, reward):
        return reward / self.denominator

    def state_dict(self):
        return dict(G=self.G, max_abs_return=self.max_abs_return,
                    mean=self.rms.mean.copy(), var=self.rms.var.copy(), count=self.rms.count)

    def load_state_dict(self, state):
        self.G, self.max_abs_return = state["G"], state["max_abs_return"]
        self.rms.mean, self.rms.var = state["mean"].copy(), state["var"].copy()
        self.rms.count = state["count"]
        self.rms._cache.clear()


class SimbaV2Core:
    def __init__(self, obs_dim, action_dim, parameters, device="cpu", observation_schema=None):
        self.p = copy.deepcopy(parameters)
        required = set(simbav2_parameters())
        if set(self.p) != required:
            raise ValueError("SimBaV2 parameters differ: " + str(set(self.p) ^ required))
        for name in ("batch_size", "buffer_size", "actor_hidden_dim", "critic_hidden_dim",
                     "actor_blocks", "critic_blocks", "num_bins", "total_updates", "updates_per_step"):
            if type(self.p[name]) is not int or self.p[name] < 1:
                raise ValueError(name + " must be a positive integer")
        if self.p["num_bins"] < 2 or self.p["gamma"] != 1.:
            raise ValueError("at least two bins and undiscounted finite-task returns required")
        if (not math.isfinite(self.p["critic_min_v"])
                or not math.isfinite(self.p["critic_max_v"])
                or self.p["critic_min_v"] >= self.p["critic_max_v"]):
            raise ValueError("finite ordered categorical value support required")
        if self.p["buffer_size"] < self.p["batch_size"] or self.p["learning_starts"] < 0:
            raise ValueError("invalid replay capacity/warmup")
        for name in ("learning_rate", "learning_rate_end", "initial_temperature", "normalized_g_max", "c_shift"):
            if not math.isfinite(self.p[name]) or self.p[name] <= 0:
                raise ValueError(name + " must be finite and positive")
        if not 0 < self.p["tau"] <= 1 or self.p["target_entropy_coef"] >= 0:
            raise ValueError("invalid target update or entropy coefficient")
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        self.device = torch.device(device)
        self.observation_schema = copy.deepcopy(observation_schema)
        self.actor = SimbaV2Actor(obs_dim, action_dim, self.p).to(self.device)
        self.critic = SimbaV2Critic(obs_dim, action_dim, self.p).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.p["learning_rate"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.p["learning_rate"])
        self.log_temperature = nn.Parameter(torch.tensor(math.log(self.p["initial_temperature"]), device=self.device))
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=self.p["learning_rate"])
        self.target_entropy = self.p["target_entropy_coef"] * action_dim
        self.obs_norm = RunningObservationNorm(obs_dim)
        self.reward_scale = RunningRewardScale(self.p["gamma"], self.p["normalized_g_max"])
        self.replay_buffer = ReplayBuffer(obs_dim, action_dim, self.device,
                                         self.p["buffer_size"], self.p["batch_size"])
        self.training_steps = self.observed_steps = 0
        self.last_losses = {}

    @property
    def actor_updates(self):
        return self.training_steps

    @property
    def critic_updates(self):
        return self.training_steps

    @property
    def temperature_updates(self):
        return self.training_steps

    @property
    def optimizer_steps(self):
        return 3 * self.training_steps

    def act(self, observation, deterministic=False):
        values = np.asarray(observation, dtype=np.float32)
        single = values.ndim == 1
        values = values.reshape(-1, self.obs_dim)
        with torch.no_grad():
            action, _ = self.actor.sample(self.obs_norm(torch.as_tensor(values, device=self.device)), deterministic)
        result = action.cpu().numpy()
        return result[0] if single else result

    def observe(self, obs, action, next_obs, reward, done):
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        self.obs_norm.update(np.asarray(obs, dtype=np.float32).reshape(1, self.obs_dim))
        self.reward_scale.update(reward, done)
        self.replay_buffer.add(obs, action, next_obs, reward, done)
        self.observed_steps += 1

    def _learning_rate(self):
        fraction = min(1., self.training_steps / self.p["total_updates"])
        rate = self.p["learning_rate"] + fraction * (self.p["learning_rate_end"] - self.p["learning_rate"])
        for optimizer in (self.actor_optimizer, self.critic_optimizer, self.temperature_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = rate
        return rate

    def train(self, batch=None):
        """One update; caller schedules UTD and never learns on evaluation data."""
        if batch is None:
            if self.observed_steps <= self.p["learning_starts"] or not self.replay_buffer.size:
                return False
            batch = self.replay_buffer.sample()
        obs, action, next_obs, reward, not_done = [x.to(self.device) for x in batch]
        obs, next_obs = self.obs_norm(obs), self.obs_norm(next_obs)
        reward = self.reward_scale(reward)
        lr = self._learning_rate()
        # Match official order: policy uses pre-update critics and temperature.
        self.critic.requires_grad_(False)
        sampled, log_prob = self.actor.sample(obs)
        qs, _ = self.critic(obs, sampled)
        actor_alpha = self.log_temperature.exp().detach()
        actor_loss = (actor_alpha * log_prob.reshape(-1) - qs.min(0).values).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.linalg.vector_norm(torch.stack([
            parameter.grad.detach().norm() for parameter in self.actor.parameters()
            if parameter.grad is not None]))
        self.actor_optimizer.step()
        renormalize_network(self.actor)
        self.critic.requires_grad_(True)

        entropy = -log_prob.detach().mean()
        temperature_loss = self.log_temperature.exp() * (entropy - self.target_entropy)
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()

        with torch.no_grad():
            next_action, next_log = self.actor.sample(next_obs)
            target_q, target_log = self.critic_target(next_obs, next_action)
            choose = target_q.argmin(0)
            selected_prob = target_log[choose, torch.arange(len(obs), device=self.device)].exp()
            target_prob, raw_target = categorical_projection(selected_prob, reward, not_done,
                self.log_temperature.exp() * next_log, self.critic.support, self.p["gamma"])
        _, current_log = self.critic(obs, action)
        critic_loss = -(target_prob.unsqueeze(0) * current_log).sum(-1).mean(-1).sum()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.linalg.vector_norm(torch.stack([
            parameter.grad.detach().norm() for parameter in self.critic.parameters()
            if parameter.grad is not None]))
        self.critic_optimizer.step()
        renormalize_network(self.critic)
        with torch.no_grad():
            for target, current in zip(self.critic_target.parameters(), self.critic.parameters()):
                target.lerp_(current, self.p["tau"])
        self.training_steps += 1
        support = self.critic.support
        lower_clipped, upper_clipped = raw_target < support[0], raw_target > support[-1]
        clipped = lower_clipped | upper_clipped
        # Compute all scalar diagnostics before one device-to-host transfer;
        # this retains every metric while avoiding repeated GPU synchronizes.
        names = ("actor_loss", "critic_loss", "temperature_loss", "temperature",
                 "entropy", "scaled_reward_max", "target_clipped_atom_fraction",
                 "target_clipped_probability_mass", "target_clipped_lower_probability_mass",
                 "target_clipped_upper_probability_mass", "target_projected_edge_mass",
                 "raw_target_min", "raw_target_max", "target_mean_min", "target_mean_max",
                 "actor_gradient_norm", "critic_gradient_norm", "actor_q_mean",
                 "actor_entropy_bonus_mean")
        raw_target_means = (selected_prob * raw_target).sum(-1)
        scalars = torch.stack((actor_loss.detach(), critic_loss.detach(),
            temperature_loss.detach(), self.log_temperature.exp().detach(),
            entropy, reward.max(), clipped.float().mean(),
            (selected_prob * clipped).sum(-1).mean(),
            (selected_prob * lower_clipped).sum(-1).mean(),
            (selected_prob * upper_clipped).sum(-1).mean(),
            target_prob[:, 0].mean() + target_prob[:, -1].mean(),
            raw_target.min(), raw_target.max(), raw_target_means.min(), raw_target_means.max(),
            actor_grad_norm, critic_grad_norm, qs.detach().min(0).values.mean(),
            -actor_alpha * log_prob.detach().mean())).detach().cpu().tolist()
        self.last_losses = dict(zip(names, scalars))
        self.last_losses.update(reward_scale_denominator=self.reward_scale.denominator,
                                learning_rate=lr,
                                critic_atom_spacing=(self.p["critic_max_v"] - self.p["critic_min_v"])
                                                    / (self.p["num_bins"] - 1))
        if not all(math.isfinite(value) for value in self.last_losses.values()):
            raise FloatingPointError("non-finite SimBaV2 update")
        return True

    def state_dict(self):
        """Replay and global RNG are owned by the shared checkpoint wrapper."""
        return copy.deepcopy(dict(revision=REVISION, parameters=self.p,
            observation_schema=self.observation_schema, obs_dim=self.obs_dim, action_dim=self.action_dim,
            actor=self.actor.state_dict(), critic=self.critic.state_dict(),
            critic_target=self.critic_target.state_dict(), actor_optimizer=self.actor_optimizer.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(),
            temperature_optimizer=self.temperature_optimizer.state_dict(),
            log_temperature=self.log_temperature.detach().clone(),
            obs_norm=dict(mean=self.obs_norm.mean.copy(), var=self.obs_norm.var.copy(), count=self.obs_norm.count),
            reward_scale=self.reward_scale.state_dict(), training_steps=self.training_steps,
            observed_steps=self.observed_steps, last_losses=self.last_losses))

    def load_state_dict(self, state):
        identity = (state["revision"], state["parameters"], state["obs_dim"], state["action_dim"], state["observation_schema"])
        if identity != (REVISION, self.p, self.obs_dim, self.action_dim, self.observation_schema):
            raise ValueError("SimBaV2 checkpoint contract mismatch")
        for name in ("actor", "critic", "critic_target", "actor_optimizer", "critic_optimizer", "temperature_optimizer"):
            getattr(self, name).load_state_dict(copy.deepcopy(state[name]))
        with torch.no_grad():
            self.log_temperature.copy_(state["log_temperature"].to(self.device))
        for key in ("mean", "var", "count"):
            setattr(self.obs_norm, key, copy.deepcopy(state["obs_norm"][key]))
        self.obs_norm._cache.clear()
        self.reward_scale.load_state_dict(state["reward_scale"])
        self.training_steps, self.observed_steps = state["training_steps"], state["observed_steps"]
        self.last_losses = copy.deepcopy(state["last_losses"])


# Keep the common registry capitalization while using the author's spelling
# for architecture classes. Both names denote exactly the same implementation.
SimBaV2Core = SimbaV2Core
