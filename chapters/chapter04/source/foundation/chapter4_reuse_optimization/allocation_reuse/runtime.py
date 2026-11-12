"""CPU rollout mirrors with the unchanged source-device Gaussian random stream.

Only inference and ordinary action decoding move to CPU.  The source modules,
their gradients and their optimizers are never modified.  A caller refreshes
the pair once before collecting a complete frozen-policy batch or assessment.

For each stochastic rollout, the proxy draws ``horizon`` normal tensors on the
source sampling device, using exactly the per-step shapes of Actor.sample,
then copies the stacked noise to CPU once.  This preserves generator state at
complete-rollout boundaries; its state intentionally advances early while a
rollout is in progress.  Partial-rollout checkpointing/interleaved generators
are not supported.  CPU/GPU network arithmetic can still differ at roundoff.
"""
from __future__ import annotations

import copy
import time
from typing import Any

import torch


VERSION = "cpu-frozen-rollout-mirror-source-device-rng-v1"


def _cpu_state(module):
    return {key: value.detach().to("cpu").clone()
            for key, value in module.state_dict().items()}


class CPUActorMirror:
    """Transparent inference proxy for the released complete-event Actor.

    The proxy forwards all attributes except ``sample`` to a private CPU copy.
    In particular existing gaussian_parameters/log_prob/decoder helpers see
    the same joint quota/spatial policy, with CPU tensors and parameters.
    """

    def __init__(self, actor):
        self.model = copy.deepcopy(actor).to("cpu")
        self.model.requires_grad_(False)
        self.source_device = torch.device(actor.reference.device)
        self._noise = None
        self._position = 0
        self._generator = None
        self.prefetched_blocks = 0
        self.noise_copy_count = 0
        self.stochastic_steps = 0
        self.deterministic_steps = 0

    def __getattr__(self, name: str) -> Any:
        model = self.__dict__.get("model")
        if model is None:
            raise AttributeError(name)
        return getattr(model, name)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def pending_noise_steps(self):
        return 0 if self._noise is None else len(self._noise) - self._position

    def assert_rollout_complete(self):
        if self.pending_noise_steps:
            raise RuntimeError("Complete the stochastic rollout before refresh or mode change")

    @torch.no_grad()
    def refresh(self, actor):
        self.assert_rollout_complete()
        if torch.device(actor.reference.device) != self.source_device:
            raise ValueError("The source sampling device must remain fixed")
        if actor.event_dim != self.model.event_dim or actor.spec != self.model.spec:
            raise ValueError("The rollout policy/spec changed")
        self.model.load_state_dict(_cpu_state(actor), strict=True)
        self.model.train(actor.training)
        self._noise = None
        self._position = 0
        self._generator = None

    @torch.no_grad()
    def _next_noise(self, shape, dtype, generator):
        shape = tuple(shape)
        if self.pending_noise_steps:
            if generator is not self._generator:
                raise RuntimeError("Cannot interleave generators inside a prefetched rollout")
            if tuple(self._noise.shape[1:]) != shape or self._noise.dtype != dtype:
                raise RuntimeError("Batch shape/dtype changed inside a prefetched rollout")
        else:
            if generator is not None and torch.device(generator.device) != self.source_device:
                raise ValueError("Generator must use the original source sampling device")
            horizon = int(self.model.spec.horizon)
            if horizon < 1:
                raise ValueError("Positive complete rollout horizon required")
            # Separate randn calls preserve CUDA Philox consumption for each
            # exact per-step shape, including assessment tail batches.
            draws = [torch.randn(shape, generator=generator, dtype=dtype,
                                 device=self.source_device) for _ in range(horizon)]
            self._noise = torch.stack(draws).to("cpu")
            self._position = 0
            self._generator = generator
            self.prefetched_blocks += 1
            self.noise_copy_count += int(self.source_device.type != "cpu")
        result = self._noise[self._position]
        self._position += 1
        self.stochastic_steps += 1
        return result

    @torch.no_grad()
    def sample(self, state, generator=None, deterministic=False):
        # This follows the released Actor.sample, changing only where its
        # precomputed standard-normal noise is materialized and used.
        actor = self.model
        data = actor._prepare(state)
        spatial_mean = actor.spatial.mean(data)
        if actor.learned_quota:
            mean = torch.cat((actor.quota_mean(data), spatial_mean), -1)
            sigma = torch.cat((actor.quota_sigma.reshape(1), actor.spatial.sigma))
        else:
            mean, sigma = spatial_mean, actor.spatial.sigma
        if deterministic:
            self.assert_rollout_complete()
            raw = mean.detach().clone()
            self.deterministic_steps += 1
        else:
            raw = (mean + sigma * self._next_noise(mean.shape, mean.dtype, generator)).detach()
        eta, z = (raw[:, :1], raw[:, 1:]) if actor.learned_quota else (None, raw)
        selected = actor.quota(data, eta)
        actions = actor.decode(data, z, selected)
        control = actor.spatial_controls(data, z)
        active = (data["budget"] > 0 if actor.learned_quota else
                  ((selected > 0) & (selected < data["caps"].sum(-1))))
        return dict(raw=raw, log_prob=actor.log_prob(data, raw),
                    mask=active.to(mean.dtype)[:, None], actions=actions,
                    quota=selected.detach(), quota_sigma=actor.quota_sigma.detach(),
                    beta=control["beta"].detach(), share_mean=control["mean"].detach())


class CPURolloutPair:
    """Private CPU actor/critic mirrors for unchanged rollout/assess callers.

    Example::

        pair = CPURolloutPair(actor_cuda, critic_cuda)
        pair.refresh(actor_cuda, critic_cuda)  # after each learner update
        paths = rollout(pool, pair.actor, pair.critic, generator_cuda, ...)
        pair.assert_rollout_complete()

    The same proxy can be supplied to an existing assess function, which may
    create its own source-device generators.  Every batch must complete its
    native horizon; each tail batch gets its own exact randn shape.
    """

    def __init__(self, actor, critic):
        if actor.reference.dtype != torch.float64:
            raise TypeError("This study retains the released float64 policy")
        self.actor = CPUActorMirror(actor)
        self.critic = copy.deepcopy(critic).to("cpu")
        self.critic.requires_grad_(False)
        self.refresh_count = 0
        self.refresh_seconds = 0.0

    @torch.no_grad()
    def refresh(self, actor, critic):
        then = time.monotonic()
        self.actor.refresh(actor)
        self.critic.load_state_dict(_cpu_state(critic), strict=True)
        self.critic.train(critic.training)
        self.refresh_count += 1
        self.refresh_seconds += time.monotonic() - then
        return self

    def assert_rollout_complete(self):
        self.actor.assert_rollout_complete()

    def metadata(self):
        return dict(version=VERSION, inference_device="cpu",
                    source_sampling_device=str(self.actor.source_device),
                    optimizer_device="unchanged source device",
                    gaussian_random_stream="same per-step source-device randn shapes, horizon-prefetched",
                    exact_generator_scope="complete-rollout boundaries; not partial-rollout checkpoints",
                    numerical_scope="CPU/GPU forward/decode agree within float64 roundoff, not bitwise promise",
                    immutable_source_modules=True, refresh_count=self.refresh_count,
                    refresh_seconds=self.refresh_seconds,
                    prefetched_noise_blocks=self.actor.prefetched_blocks,
                    noise_device_to_cpu_copies=self.actor.noise_copy_count)
