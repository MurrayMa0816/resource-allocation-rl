"""PASPO adapted to the current chapter-2 state-dependent capped simplex.

Reference: Winkel et al., NeurIPS 2024, ``Autoregressive Policy Optimization
for Constrained Allocation Tasks``; author code niklasdbs/paspo, commit
81c65cd6046edd1507753819c1184169aa641d29 (model, distribution, uniform_init).

The independent conditional Beta heads, feasible-interval autoregression,
interval Jacobian, geometry-only de-biasing initialization and PPO update are
retained. For this polytope, each LP bound has an exact closed form. Unlike the
paper's fixed constraints, our bounds change with the *public current* caps.
Chapter-2 initialization fits Beta MLEs from approximately uniform pairwise
hit-and-run samples of the FIRST TRAINING polytope. Chapter 3 accepts a declared
reward-blind neutral amount/spatial prior on that same physical polytope, with
the unused coordinate first. That calibration is frozen after initialization.
Neither is a claim of exact/future geometry uniformity.
No rewards, future states or physical gradients enter this calibration.

The density is with respect to the independent allocation-fraction coordinates
(zero-width conditionals are Dirac masses). Control coordinates are 2*x-1;
their state-independent Jacobian cancels in PPO ratios. The native wrapper's
projection only corrects FP64 arithmetic roundoff. Public observations and the
actual returned controls retain float64 through rollout and probability
recomputation; neural feature computations use float32.
Deterministic deployment defaults to conditional Beta modes, following the
author's mode-based action selection. Exact uniform conditionals use their
midpoint and doubly singular conditionals choose the upper endpoint (the
author's Dirichlet first-coordinate convention). A separately registered
conditional-mean diagnostic is available; neither is the joint policy mode.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..actions import reference_fraction_tensor


REVISION = "chapter23-paspo-public-capped-v3-declared-geometry-prior"
AUTHOR_COMMIT = "81c65cd6046edd1507753819c1184169aa641d29"


def paspo_parameters(tiny=False, horizon=40):
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    return dict(hidden_width=32 if tiny else 128, branch_width=16 if tiny else 32,
                learning_rate=3e-4, n_steps=(2 if tiny else 8) * horizon,
                batch_size=min(16 if tiny else 128, (2 if tiny else 8) * horizon),
                epochs=1 if tiny else 4, gamma=1., gae_lambda=1., clip_range=.2,
                value_coef=.5, entropy_coef=0., max_grad_norm=.5,
                deterministic_rule="conditional_mode",
                debias_chains=32 if tiny else 64, debias_draws=4 if tiny else 8,
                debias_burn_sweeps=12 if tiny else 24, debias_thin_sweeps=2 if tiny else 4,
                debias_seed=71811)


def capped_polytope_samples(caps, target, *, chains, draws, burn_sweeps,
                           thin_sweeps, seed):
    """Geometry-only approximate uniform samples, with a uniform invariant law.

    A move chooses two coordinates and uniformly resplits their fixed sum over
    its feasible interval. Detailed balance holds for uniform volume on the
    capped simplex. Parallel chains start at a feasible relative interior;
    finite burn-in is explicitly approximate, rather than a convergence proof.
    """
    caps = np.asarray(caps, dtype=np.float64)
    if (caps.ndim != 1 or not np.isfinite(caps).all() or np.any(caps < 0)
            or not np.isfinite(target) or target < 0 or target > caps.sum() + 1e-12):
        raise ValueError("invalid capped-polytope geometry")
    for value in (chains, draws, burn_sweeps, thin_sweeps):
        if type(value) is not int or value < 1:
            raise ValueError("positive integer sampler settings required")
    target = min(float(target), float(caps.sum()))
    if target == 0:
        return np.zeros((chains * draws, len(caps)))
    if target == caps.sum():
        return np.broadcast_to(caps, (chains * draws, len(caps))).copy()
    active = np.flatnonzero(caps > 0)
    x = np.broadcast_to(caps * (target / caps.sum()), (chains, len(caps))).copy()
    if len(active) < 2:
        return np.repeat(x, draws, axis=0)
    rng = np.random.default_rng(seed)
    rows = np.arange(chains)

    def sweep(moves):
        for _ in range(moves):
            ii = rng.integers(len(active), size=chains)
            jj = (ii + rng.integers(1, len(active), size=chains)) % len(active)
            i, j = active[ii], active[jj]
            pair = x[rows, i] + x[rows, j]
            low, high = np.maximum(0., pair - caps[j]), np.minimum(caps[i], pair)
            first = low + rng.random(chains) * np.maximum(high - low, 0.)
            x[rows, i], x[rows, j] = first, pair - first

    sweep(burn_sweeps * len(active))
    samples = []
    for _ in range(draws):
        sweep(thin_sweeps * len(active))
        samples.append(x.copy())
    return np.concatenate(samples)


def fit_debias_parameters(caps, parameters, *, samples=None):
    """Fit the paper's normalized conditional Beta MLEs on initial geometry."""
    from scipy.stats import beta as beta_distribution

    caps = np.asarray(caps, dtype=np.float64)
    n = len(caps)
    target = min(1., float(caps.sum()))
    shape = np.ones((max(0, n - 1), 2), dtype=np.float64)
    supplied = samples is not None
    if samples is None:
        samples = capped_polytope_samples(caps, target,
            chains=parameters["debias_chains"], draws=parameters["debias_draws"],
            burn_sweeps=parameters["debias_burn_sweeps"],
            thin_sweeps=parameters["debias_thin_sweeps"], seed=parameters["debias_seed"])
    samples = np.asarray(samples, dtype=np.float64)
    if (samples.ndim != 2 or samples.shape[1] != n or len(samples) < 8
            or not np.isfinite(samples).all() or np.any(samples < -1e-13)
            or np.any(samples > caps + 1e-12)
            or np.any(np.abs(samples.sum(1)-target) > 1e-12)):
        raise ValueError('PASPO calibration samples must lie on the initial public polytope')
    remaining = np.full(len(samples), target)
    suffix = np.cumsum(caps[::-1])[::-1]
    fitted, degenerate = 0, 0
    for i in range(n - 1):
        low = np.maximum(0., remaining - suffix[i + 1])
        high = np.minimum(caps[i], remaining)
        width = np.maximum(high - low, 0.)
        active = width > 1e-13
        if np.count_nonzero(active) >= 8:
            u = np.clip((samples[active, i] - low[active]) / width[active], 1e-10, 1 - 1e-10)
            # This is the paper's Beta MLE, not an uncapped-simplex formula.
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                try:
                    a, b, _, _ = beta_distribution.fit(u, floc=0., fscale=1.)
                except (ValueError, RuntimeError, FloatingPointError, RuntimeWarning) as exc:
                    raise RuntimeError(f"PASPO de-bias MLE failed for coordinate {i}") from exc
            if not np.isfinite([a, b]).all() or min(a, b) <= 0:
                raise RuntimeError("invalid PASPO de-bias MLE")
            shape[i] = np.clip([a, b], math.exp(-6), math.exp(8))
            fitted += 1
        else:
            degenerate += 1
        remaining = np.maximum(0., remaining - samples[:, i])
    return shape, dict(kind=("declared_initial_training_geometry_prior_beta_mle" if supplied
                            else "initial_training_polytope_pairwise_hit_and_run_beta_mle"),
                       exact_uniform=False, future_geometry_uniform=False,
                       samples=len(samples), fitted_conditionals=fitted,
                       degenerate_conditionals=degenerate,
                       max_sum_error=float(np.max(np.abs(samples.sum(1) - target))),
                       max_cap_excess=float(np.maximum(samples - caps, 0.).max(initial=0.)),
                       mean_fraction=samples.mean(0).tolist(),
                       geometry_sha256=hashlib.sha256(caps.tobytes()).hexdigest())


def conditional_beta_mode(alpha, beta):
    """Finite exact Beta modes including boundary and uniform cases.

    Author StableBeta delegates to Dirichlet.mode after adding 1e-6. We retain
    our actual (unshifted) Beta law and handle its boundary cases explicitly.
    Both singular endpoints are modes when alpha,beta<1: choose x=1, matching
    the author's first-coordinate convention. Uniform Beta(1,1) uses x=.5.
    """
    left, right = (alpha - 1.).clamp_min(0.), (beta - 1.).clamp_min(0.)
    denom = left + right
    regular = left / denom.clamp_min(torch.finfo(alpha.dtype).tiny)
    no_interior = torch.where((alpha == 1.) & (beta == 1.), .5,
                              torch.where((alpha < 1.) & (beta == 1.), 0., 1.))
    return torch.where(denom > 0., regular, no_interior)


@torch.jit.script
def _autoregressive_sample(encoded: torch.Tensor, caps: torch.Tensor,
                           target: torch.Tensor, w1: torch.Tensor,
                           b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor,
                           w3: torch.Tensor, b3: torch.Tensor,
                           hidden: int, deterministic: bool,
                           conditional_mode: bool,
                           first_log_adjust: torch.Tensor) -> torch.Tensor:
    """Same sequential Beta draws in compiled tensor code, without Python dispatch.

    Beta.sample calls the Dirichlet sampling kernel on [alpha,beta] and takes
    its first coordinate. Calling that same kernel preserves draw shape/order
    and generator state; detach retains the original non-reparameterized draw.
    Parameters and constraints remain the original PASPO ones.
    """
    heads, branch = w1.shape[0], w1.shape[1]
    remaining = (target > 0).double()
    prefix = torch.zeros((encoded.shape[0], heads), dtype=encoded.dtype,
                         device=encoded.device)
    suffix = caps.flip(-1).cumsum(-1).flip(-1)
    values = torch.jit.annotate(list[torch.Tensor], [])
    # A scalar torch.where argument is materialized on CUDA on every call.
    # Keep the exact zero on the device across all conditional heads.
    zero = torch.zeros((), dtype=caps.dtype, device=caps.device)
    initial = F.linear(encoded, w1[:, :, :hidden].reshape(heads * branch, hidden),
                       b1.reshape(-1)).reshape(encoded.shape[0], heads, branch)
    for i in range(heads):
        h = initial[:, i]
        if i > 0:
            h = h + F.linear(prefix[:, :i], w1[i, :, hidden:hidden + i])
        h = F.relu(F.linear(F.relu(h), w2[i], b2[i]))
        raw_shapes = F.linear(h, w3[i], b3[i]).double()
        if i == 0:
            raw_shapes = raw_shapes + first_log_adjust
        shapes = raw_shapes.clamp(-6, 8).exp()
        if deterministic:
            a, b = shapes.unbind(-1)
            if conditional_mode:
                left, right = (a - 1.).clamp_min(0.), (b - 1.).clamp_min(0.)
                denom = left + right
                regular = left / denom.clamp_min(2.2250738585072014e-308)
                no_interior = torch.where((a == 1.) & (b == 1.), .5,
                                          torch.where((a < 1.) & (b == 1.), 0., 1.))
                unit = torch.where(denom > 0., regular, no_interior)
            else:
                unit = a / (a + b)
        else:
            # shapes already stores [alpha, beta] contiguously. Reusing it
            # avoids rebuilding the identical Dirichlet input at each head.
            unit = torch._sample_dirichlet(shapes.detach())[..., 0]
        low = (remaining - suffix[:, i + 1]).clamp_min(0.)
        high = torch.minimum(caps[:, i], remaining)
        width = (high - low).clamp_min(0.)
        selected = low + torch.where(width > 1e-14, width * unit, zero)
        values.append(selected)
        prefix[:, i] = selected.to(prefix.dtype)
        remaining = (remaining - selected).clamp_min(0.)
    values.append(remaining)
    return torch.stack(values, -1)


class CappedAutoregressiveBeta(nn.Module):
    """Independent per-coordinate heads with vectorized teacher forcing."""
    def __init__(self, obs_dim, action_dim, hidden, branch, observation_schema,
                 deterministic_rule="conditional_mode"):
        super().__init__()
        self.n, self.hidden = int(action_dim), int(hidden)
        if deterministic_rule not in ("conditional_mode", "conditional_mean"):
            raise ValueError("unknown PASPO deterministic deployment rule")
        self.deterministic_rule = deterministic_rule
        self.schema = deepcopy(observation_schema)
        fields = {field["name"]: field for field in self.schema}
        self.cap_field, self.budget_field = fields["caps"], fields["budget"]
        if (self.cap_field["stop"] - self.cap_field["start"] != self.n
                or self.budget_field["stop"] - self.budget_field["start"] != 1
                or self.cap_field["divisor"] <= 0 or self.budget_field["divisor"] <= 0
                or any(field["start"] < 0 or field["stop"] > obs_dim for field in self.schema)):
            raise ValueError("observation schema does not match caps/budget dimensions")
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        heads, din = self.n - 1, hidden + self.n - 1
        self.w1 = nn.Parameter(torch.zeros(heads, branch, din))
        self.b1 = nn.Parameter(torch.zeros(heads, branch))
        self.w2 = nn.Parameter(torch.empty(heads, branch, branch))
        self.b2 = nn.Parameter(torch.zeros(heads, branch))
        self.w3 = nn.Parameter(torch.zeros(heads, 2, branch))
        self.b3 = nn.Parameter(torch.zeros(heads, 2))
        self.register_buffer('neutral_anchor_log_shapes',torch.zeros(2,dtype=torch.float64))
        self.neutral_context = None
        mask = torch.cat((torch.ones(heads, hidden), torch.tril(torch.ones(heads, heads), diagonal=-1)), -1)
        self.register_buffer("prefix_mask", mask)
        with torch.no_grad():
            for i in range(heads):
                nn.init.kaiming_uniform_(self.w1[i, :, :hidden + i], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5))

    def geometry(self, obs):
        cf, bf = self.cap_field, self.budget_field
        caps = obs[:, cf["start"]:cf["stop"]].double() * cf["divisor"]
        budget = obs[:, bf["start"]].double() * bf["divisor"]
        if (not torch.isfinite(caps).all() or not torch.isfinite(budget).all()
                or torch.any(caps < 0) or torch.any(budget < 0)):
            raise ValueError("invalid public caps or budget")
        target = torch.minimum(budget, caps.sum(-1))
        # Every strictly positive target, including subnormal balances, defines
        # a unit simplex. Protect zero only; flooring a positive denominator
        # would shrink that simplex while sampling still allocates unit mass.
        denominator = torch.where(target > 0, target, torch.ones_like(target))
        fractions = torch.where(target[:, None] > 0,
                                caps / denominator[:, None], 0.)
        return fractions, target

    def _parameters_known(self, obs, fractions):
        encoded = self.encoder(obs.float())
        x = torch.cat((encoded, fractions[:, :-1].to(encoded.dtype)), -1)[:, None] * self.prefix_mask[None]
        hidden = F.relu(torch.einsum("bhi,hoi->bho", x, self.w1) + self.b1)
        hidden = F.relu(torch.einsum("bhi,hoi->bho", hidden, self.w2) + self.b2)
        raw=(torch.einsum("bhi,hoi->bho", hidden, self.w3) + self.b3).double()
        first_mask=torch.zeros((1,self.n-1,1),dtype=raw.dtype,device=raw.device)
        if self.n>1:
            first_mask[:,0,:]=1.
        raw=raw+first_mask*self._neutral_adjustment(obs)[:,None,:]
        return raw.clamp(-6,8).exp()

    def _neutral_adjustment(self,obs):
        if self.neutral_context is None:
            return torch.zeros((len(obs),2),dtype=torch.float64,device=obs.device)
        _,target=self.geometry(obs)
        r=reference_fraction_tensor(obs,target,self.neutral_context)
        shapes=torch.cat((1.+96.*(1.-r),1.+96.*r),-1)
        return shapes.log()-self.neutral_anchor_log_shapes[None]

    def known(self, obs, fractions):
        """Joint log-density with the affine interval Jacobian, no last density."""
        fractions = fractions.double()
        caps, target = self.geometry(obs)
        if fractions.shape != caps.shape or not torch.isfinite(fractions).all():
            raise ValueError("invalid allocation fractions")
        total = (target > 0).double()
        if (torch.any(fractions < -1e-10) or torch.any(fractions > caps + 1e-10)
                or torch.any(torch.abs(fractions.sum(-1) - total) > 1e-9)):
            raise ValueError("fractions outside current feasible polytope")
        if self.n == 1:
            zero = obs.sum(-1).double() * 0
            return zero, zero
        # Reverse sums avoid cancellation late in a nearly exhausted budget.
        remaining = fractions.flip(-1).cumsum(-1).flip(-1)[:, :-1]
        suffix_caps = caps.flip(-1).cumsum(-1).flip(-1)[:, 1:]
        low = (remaining - suffix_caps).clamp_min(0.)
        high = torch.minimum(caps[:, :-1], remaining)
        width = (high - low).clamp_min(0.)
        active = width > 1e-14
        unit = ((fractions[:, :-1] - low) / width.clamp_min(1e-300)).clamp(
            torch.finfo(torch.float64).eps, 1 - torch.finfo(torch.float64).eps)
        a, b = self._parameters_known(obs, fractions).unbind(-1)
        dist = torch.distributions.Beta(a, b, validate_args=False)
        log_width = width.clamp_min(1e-300).log()
        logp = torch.where(active, dist.log_prob(unit) - log_width, 0.).sum(-1)
        entropy = torch.where(active, dist.entropy() + log_width, 0.).sum(-1)
        return logp, entropy

    def sample(self, obs, deterministic=False):
        caps, target = self.geometry(obs)
        encoded = self.encoder(obs.float())
        return _autoregressive_sample(encoded, caps, target, self.w1, self.b1,
            self.w2, self.b2, self.w3, self.b3, self.hidden, deterministic,
            self.deterministic_rule == "conditional_mode",self._neutral_adjustment(obs))


class PASPOCore:
    """On-policy complete-rollout core with exact continuation state."""
    def __init__(self, obs_dim, action_dim, parameters, device="cpu", observation_schema=None):
        if observation_schema is None or action_dim < 1:
            raise ValueError("PASPO requires a public caps/budget observation schema")
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        self.p, self.device = deepcopy(parameters), torch.device(device)
        self.schema = deepcopy(observation_schema)
        p = self.p
        for key in ("n_steps", "batch_size", "epochs", "hidden_width", "branch_width"):
            if type(p[key]) is not int or p[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        self.actor = CappedAutoregressiveBeta(obs_dim, action_dim, p["hidden_width"],
                                             p["branch_width"], self.schema,
                                             p["deterministic_rule"]).to(self.device)
        self.critic = nn.Sequential(nn.Linear(obs_dim, p["hidden_width"]), nn.ReLU(),
                                   nn.Linear(p["hidden_width"], p["hidden_width"]), nn.ReLU(),
                                   nn.Linear(p["hidden_width"], 1)).to(self.device)
        self.optimizer = torch.optim.Adam([*self.actor.parameters(), *self.critic.parameters()],
                                          lr=p["learning_rate"])
        self.initialized = False
        self.debias_diagnostics = {}
        self.initial_caps = None
        self.rollout = []
        self.training_steps = self.actor_updates = self.critic_updates = self.environment_steps = 0
        self.last_losses = {}

    def _obs(self, observation):
        value = np.asarray(observation, dtype=np.float64)
        if value.shape != (self.obs_dim,) or not np.isfinite(value).all():
            raise ValueError("expected a finite single public observation")
        return value, torch.as_tensor(value, device=self.device).reshape(1, -1)

    def initialize(self, observation, *, calibration_samples=None):
        """Call on the first TRAINING state, before saving/evaluating a model."""
        if self.initialized:
            return
        _, obs = self._obs(observation)
        caps = self.actor.geometry(obs)[0][0].cpu().numpy()
        shapes, diagnostics = fit_debias_parameters(caps, self.p, samples=calibration_samples)
        with torch.no_grad():
            self.actor.b3.copy_(torch.as_tensor(np.log(shapes), dtype=torch.float32, device=self.device))
            if self.actor.neutral_context is not None:
                self.actor.neutral_anchor_log_shapes.copy_(self.actor.b3[0].double())
        self.initial_caps = caps.copy()
        self.debias_diagnostics = diagnostics
        self.initialized = True

    def act(self, observation, deterministic=False):
        if not self.initialized:
            raise RuntimeError("initialize PASPO on a training observation before prediction")
        _, obs = self._obs(observation)
        with torch.no_grad():
            fractions = self.actor.sample(obs, deterministic)[0].cpu().numpy()
        return np.clip(2. * fractions - 1., -1., 1.).astype(np.float64)

    def observe(self, observation, action, next_observation, reward, done):
        if not self.initialized:
            raise RuntimeError("initialize PASPO before observing transitions")
        if len(self.rollout) >= self.p["n_steps"]:
            raise RuntimeError("consume the completed PASPO rollout before further interaction")
        obs_np, obs = self._obs(observation)
        self._obs(next_observation)
        action = np.asarray(action, dtype=np.float64)
        if (action.shape != (self.action_dim,) or not np.isfinite(action).all()
                or np.any(np.abs(action) > 1.) or not np.isfinite(reward)
                or not isinstance(done, (bool, np.bool_))):
            raise ValueError("invalid PASPO transition")
        fractions = torch.as_tensor((action + 1.) / 2., device=self.device).reshape(1, -1)
        with torch.no_grad():
            logp, _ = self.actor.known(obs, fractions)
            value = self.critic(obs.float()).reshape(-1)
        self.rollout.append(dict(obs=obs_np.copy(), action=action.copy(), reward=float(reward),
                                 done=bool(done), logp=float(logp[0]), value=float(value[0])))
        self.environment_steps += 1

    def train(self):
        if len(self.rollout) < self.p["n_steps"]:
            return {}
        if not self.rollout[-1]["done"]:
            raise RuntimeError("PASPO n_steps must end on a complete native episode")
        p, rows = self.p, self.rollout
        returns, advantages = np.empty(len(rows)), np.empty(len(rows))
        next_value, carry = 0., 0.
        for i in reversed(range(len(rows))):
            row = rows[i]
            alive = 1. - row["done"]
            delta = row["reward"] + p["gamma"] * alive * next_value - row["value"]
            carry = delta + p["gamma"] * p["gae_lambda"] * alive * carry
            advantages[i], returns[i] = carry, carry + row["value"]
            next_value = row["value"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        obs = torch.as_tensor(np.stack([r["obs"] for r in rows]), device=self.device)
        fractions = torch.as_tensor((np.stack([r["action"] for r in rows]) + 1.) / 2., device=self.device)
        old_logp = torch.as_tensor([r["logp"] for r in rows], dtype=torch.float64, device=self.device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(advantages, dtype=torch.float64, device=self.device)
        metrics = []
        for _ in range(p["epochs"]):
            for ids in torch.randperm(len(rows), device=self.device).split(p["batch_size"]):
                logp, entropy = self.actor.known(obs[ids], fractions[ids])
                value = self.critic(obs[ids].float()).squeeze(-1)
                difference = logp - old_logp[ids]
                ratio = difference.exp()
                policy_loss = -torch.minimum(ratio * adv[ids],
                    ratio.clamp(1 - p["clip_range"], 1 + p["clip_range"]) * adv[ids]).mean()
                value_loss = (value - ret[ids]).square().mean()
                loss = policy_loss + p["value_coef"] * value_loss - p["entropy_coef"] * entropy.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite PASPO PPO objective")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()],
                                                p["max_grad_norm"], error_if_nonfinite=True)
                self.optimizer.step()
                self.training_steps += 1
                self.actor_updates += 1
                self.critic_updates += 1
                metrics.append([float(policy_loss.detach()), float(value_loss.detach()),
                                float(entropy.detach().mean()), float(norm),
                                float(((ratio - 1.) - difference).detach().mean())])
        mean = np.mean(metrics, axis=0)
        self.last_losses = dict(zip(("actor", "critic", "entropy", "grad_norm", "approx_kl"), mean.tolist()))
        self.rollout = []
        return self.last_losses

    def state_dict(self):
        return deepcopy(dict(revision=REVISION, obs_dim=self.obs_dim, action_dim=self.action_dim,
            parameters=self.p, schema=self.schema, actor=self.actor.state_dict(),
            critic=self.critic.state_dict(), optimizer=self.optimizer.state_dict(),
            initialized=self.initialized, initial_caps=self.initial_caps,
            debias_diagnostics=self.debias_diagnostics, rollout=self.rollout,
            neutral_context=deepcopy(self.actor.neutral_context),
            training_steps=self.training_steps, actor_updates=self.actor_updates,
            critic_updates=self.critic_updates, environment_steps=self.environment_steps,
            last_losses=self.last_losses))

    def load_state_dict(self, saved):
        if (saved.get("revision") != REVISION or saved.get("obs_dim") != self.obs_dim
                or saved.get("action_dim") != self.action_dim or saved.get("parameters") != self.p
                or saved.get("schema") != self.schema
                or saved.get('neutral_context') != self.actor.neutral_context):
            raise ValueError("PASPO continuation identity differs")
        value = deepcopy(saved)
        if len(value["rollout"]) > self.p["n_steps"]:
            raise ValueError("invalid PASPO pending rollout")
        self.actor.load_state_dict(value["actor"], strict=True)
        self.critic.load_state_dict(value["critic"], strict=True)
        self.optimizer.load_state_dict(value["optimizer"])
        for name in ("initialized", "initial_caps", "debias_diagnostics", "rollout", "training_steps",
                     "actor_updates", "critic_updates", "environment_steps", "last_losses"):
            setattr(self, name, value[name])
