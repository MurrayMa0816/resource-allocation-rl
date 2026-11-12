"""Independent Gymnasium simulator; no learner, history encoder or predictor.

Sequential TDMA slots, ordinary stop-and-wait ARQ, and one shared interference
process. Each action spends radio energy immediately; only received data
persists at receivers. The public observation is deliberately partial.
"""
from __future__ import annotations

from numbers import Integral

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.special import erfc

from .codec import decode_prefix, encode_image
from .config import Config
from .quality import batch_rgb_ssim
from ...full_spend import radio_minimum_input, budget_roundoff

OBSERVATION_VERSION = 'communication-image-global-interference-v1'
OBSERVATION_FIELDS = ('measured_link_strength', 'confirmed_packets', 'round_index')


def _cn(rng, shape):
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)


def packet_success_probability(snr, packet_bits=4128):
    """Conditional coherent BPSK, independent bit errors, ideal CRC/ACK.

    A zero-power slot is separately disabled by step(), even if a hypothetical
    all-correct random bit sequence had nonzero probability.
    """
    snr = np.asarray(snr, dtype=np.float64)
    if not np.isfinite(snr).all() or np.any(snr < 0):
        raise ValueError('SNR must be finite and nonnegative')
    if isinstance(packet_bits, bool) or not isinstance(packet_bits, Integral) or packet_bits <= 0:
        raise ValueError('packet_bits must be a positive integer')
    ber = .5 * erfc(np.sqrt(snr))
    return np.exp(packet_bits * np.log1p(-ber))


class _PowerSpace(spaces.Box):
    """Current feasible power set; sample() has no task-dependent priorities.

    Bounds reflect publicly known completion and a ledger of executed energy,
    not hidden channels. Sampling is for Gym interoperability, never used to
    repair step() actions or to implement an allocation policy.
    """

    def __init__(self, config, chapter):
        self.chapter = chapter
        # The public action space carries feasible-set geometry only, not a
        # copy of private link/interference model parameters.
        self._groups = config.groups
        self._slot_s = config.packet_slot_s
        self._maximum_power_W = config.maximum_power_W
        self._quota_J = config.period_quota_J
        self._power_tolerance_W = config.power_tolerance_W
        self._energy_tolerance_J = config.energy_tolerance_J
        self._source_packets=config.source_packets
        self._rounds=config.rounds
        self._full_spend_tolerance=min(config.energy_tolerance_J,budget_roundoff(config.episode_budget_J))
        super().__init__(0., config.maximum_power_W, shape=(config.groups,), dtype=np.float64)
        self.update(np.ones(config.groups, dtype=bool), config.episode_budget_J, False,
                    confirmed=np.zeros(config.groups,dtype=np.int32),round_index=0)

    def update(self, active, remaining_J, terminal, *, confirmed=None, round_index=0):
        self.high[:] = np.where(active & (not terminal), self._maximum_power_W, 0.)
        self.energy_limit_J = 0. if terminal else remaining_J
        self.target_J = None
        self.minimum_J=0.
        if self.chapter == 2:
            self.target_J = 0. if terminal else min(self._quota_J, int(active.sum()) * self._maximum_power_W * self._slot_s)
        elif self.chapter==3 and not terminal:
            if confirmed is None:raise ValueError('Full-spend radio geometry requires public confirmation counts')
            self.minimum_J=radio_minimum_input(remaining_J,confirmed,self._source_packets,
                self._rounds-round_index,self._maximum_power_W*self._slot_s,
                tolerance=self._full_spend_tolerance)

    def contains(self, x):
        try:
            raw = np.asarray(x)
            if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
                return False
            p = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError):
            return False
        if p.shape != self.shape or not np.isfinite(p).all() or np.any(p < 0):
            return False
        if np.any(p > self.high + self._power_tolerance_W) or np.any(p[self.high == 0] != 0):
            return False
        energy = float(p.sum() * self._slot_s)
        if energy > self.energy_limit_J + self._energy_tolerance_J:
            return False
        if self.chapter==3 and energy<self.minimum_J-self._full_spend_tolerance:
            return False
        return self.target_J is None or abs(energy - self.target_J) <= self._energy_tolerance_J

    def sample(self, mask=None, probability=None):
        if mask is not None or probability is not None:
            raise ValueError('Power space does not support sampling masks')
        p = np.zeros(self.shape, dtype=np.float64)
        if self.target_J is None and self.chapter!=3:
            p = self.np_random.random(self.shape) * self.high
            limit = self.energy_limit_J / self._slot_s
            if p.sum() > limit:
                p *= max(0., limit) * (1 - 64 * np.finfo(float).eps) / p.sum()
        else:
            target=(self.target_J if self.target_J is not None else self.minimum_J+
                    float(self.np_random.random())*(min(self.energy_limit_J,float(self.high.sum())*self._slot_s)-self.minimum_J))
            remaining = target / self._slot_s
            eligible = self.high > 0
            weights = self.np_random.random(self.shape) + np.finfo(float).eps
            # Bounded simplex sampling: cap coordinates and redistribute.
            for _ in range(self._groups + 1):
                if not np.any(eligible) or remaining <= 0:
                    break
                ids = np.flatnonzero(eligible)
                proposed = remaining * weights[ids] / weights[ids].sum()
                capped = proposed >= self.high[ids]
                if not np.any(capped):
                    p[ids] = proposed
                    break
                saturated = ids[capped]
                p[saturated] = self.high[saturated]
                remaining -= float(self.high[saturated].sum())
                eligible[saturated] = False
        if not self.contains(p):
            raise ArithmeticError('Could not sample feasible powers within numerical tolerance')
        return p


class CommunicationImageEnv(gym.Env):
    """Send one fixed image per receiver over a fixed number of rounds.

    Supply source images through reset(options={'images': ...}), constructor
    images=... (fixed fixtures), or image_source(rng) (episode task generator).
    No automatic fake images, data download, policy feature construction or
    intermediate quality evaluation takes place in this environment.
    """

    metadata = {'render_modes': []}
    observation_version = OBSERVATION_VERSION
    observation_fields = OBSERVATION_FIELDS
    reward_version = 'terminal-mean-rgb-ssim-v1'
    terminal_cost_version = 'terminal-ssim-shortfall-v1'
    reward_range = (-1., 1.)
    _config_type = Config

    def __init__(self, config=None, chapter=3, delta=None, epsilon=None, *,
                 tau=None, images=None, image_source=None, render_mode=None):
        super().__init__()
        self.config = self._config_type() if config is None else config
        if not isinstance(self.config, Config):
            raise TypeError('config must be a communication_image.Config')
        self.config.validate()
        if isinstance(chapter, bool) or chapter not in (2, 3, 4, 5):
            raise ValueError('chapter must be 2, 3, 4 or 5')
        self.chapter = int(chapter)
        c = self.config
        if chapter == 2 and c.episode_budget_J + c.energy_tolerance_J < c.rounds * c.period_quota_J:
            raise ValueError('Chapter 2 budget must cover all fixed period quotas')
        self._configure_constraints(chapter, delta, epsilon, tau)
        if render_mode is not None:
            raise ValueError('No rendering is exposed; source/decoded images are evaluator-only')
        self.render_mode = render_mode
        if images is not None and image_source is not None:
            raise ValueError('Choose constructor images or image_source, not both')
        if image_source is not None and not callable(image_source):
            raise TypeError('image_source must be callable(rng) -> uint8 image batch')
        self._fixed_images = None if images is None else self._validate_images(images).copy()
        self._image_source = image_source
        layout = np.linspace(c.reference_mean_snr_min_dB, c.reference_mean_snr_max_dB, c.groups)
        np.random.default_rng(c.deployment_seed).shuffle(layout)
        self._reference_snr = 10 ** (layout / 10)
        self.action_space = _PowerSpace(c, chapter)
        self.observation_space = spaces.Dict({
            'measured_link_strength': spaces.Box(0., np.inf, shape=(c.groups,), dtype=np.float64),
            'confirmed_packets': spaces.Box(0, c.source_packets, shape=(c.groups,), dtype=np.int32),
            'round_index': spaces.Discrete(c.rounds + 1),
        })
        self._initialized = False

    def _configure_constraints(self, chapter, delta, epsilon, tau):
        if delta is not None:
            raise ValueError('delta is a legacy MSE ceiling; use tau for an SSIM floor, '
                             'this release supports only the current SSIM reward')
        if chapter != 5 and (tau is not None or epsilon is not None):
            raise ValueError('Individual quality constraints belong to chapter 5')
        self.tau = self._threshold(tau, 'tau', 1.)
        # SSIM may be negative: a floor in [0,1] can have shortfall up to 2.
        self.epsilon = self._threshold(epsilon, 'epsilon', 2.)
        if chapter == 5 and (self.tau is None or self.epsilon is None):
            raise ValueError('Chapter 5 requires independently calibrated tau and epsilon')

    def _threshold(self, value, name, upper, *, legacy_broadcast=False):
        if value is None:
            return None
        try:
            raw = np.asarray(value)
            if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
                raise ValueError()
            if not legacy_broadcast and raw.ndim != 0 and raw.shape != (self.config.groups,):
                raise ValueError()
            array = np.broadcast_to(raw.astype(float), (self.config.groups,)).copy()
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{name} must be a scalar or length-N real vector') from exc
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > upper):
            raise ValueError(f'{name} must be finite within [0, {upper:g}]')
        array.setflags(write=False)
        return array

    def _validate_images(self, images):
        images = np.asarray(images)
        if images.shape != (self.config.groups, 64, 64, 3) or images.dtype != np.uint8:
            raise ValueError('Supply one uint8 RGB[64,64,3] source image per terminal')
        return images

    def reset(self, *, seed=None, options=None):
        if options is not None and (not isinstance(options, dict) or set(options) - {'images'}):
            raise ValueError("reset options supports only {'images': uint8_batch}")
        provided = options is not None and 'images' in options
        if not provided and self._fixed_images is None and self._image_source is None:
            raise ValueError("Provide reset(options={'images': batch}) or constructor image_source")
        images = self._validate_images(options['images']) if provided else self._fixed_images
        super().reset(seed=seed)
        actual_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**63, dtype=np.int64))
        if images is None:
            # Source selection has an independent RNG stream from radio physics.
            image_rng = np.random.default_rng(np.random.SeedSequence([actual_seed, 2026102906]))
            images = self._validate_images(self._image_source(image_rng))
        c = self.config
        self._source = images.copy()
        self._source_packets = np.stack([encode_image(x) for x in self._source])
        self._received_packets = np.zeros_like(self._source_packets)
        self._progress = np.zeros(c.groups, dtype=np.int32)
        channel_ss, csi_ss, packet_ss = np.random.SeedSequence(actual_seed).spawn(3)
        channel_rng, csi_rng, packet_rng = [np.random.default_rng(x) for x in (channel_ss, csi_ss, packet_ss)]
        h = np.empty((c.rounds, c.groups), dtype=np.complex128)
        h[0] = _cn(channel_rng, (c.groups,))
        for t in range(1, c.rounds):
            h[t] = c.channel_rho * h[t-1] + np.sqrt(1-c.channel_rho**2) * _cn(channel_rng, (c.groups,))
        self._true_channel = h * np.sqrt(self._reference_snr)[None, :]
        # Preserve the frozen G=1 prototype's stream while storing only ONE
        # common interference value per round. There are no regions or IDs.
        draws = np.random.default_rng(np.random.SeedSequence([actual_seed, c.global_seed_salt])).random((c.rounds, c.groups))[:, 0]
        bad = np.empty(c.rounds, dtype=bool)
        bad[0] = draws[0] < .5
        for t in range(1, c.rounds):
            bad[t] = bool(bad[t-1]) != bool(draws[t] > c.global_stay_probability)
        self._global_noise = np.where(bad, c.strong_noise_multiplier, c.weak_noise_multiplier)
        # Ratio form of 1 / (E[1/J] * J) avoids reciprocal overflow when both
        # multipliers are tiny. Equal multipliers always mean no common effect.
        ratio = c.weak_noise_multiplier / c.strong_noise_multiplier
        normalizer = .5 + .5 * ratio
        scales = np.where(bad, np.sqrt(ratio / normalizer), np.sqrt(1 / normalizer))
        self._true_channel *= scales[:, None]
        self._measured_channel = self._true_channel + np.sqrt(c.csi_complex_error_variance) * _cn(csi_rng, h.shape)
        self._packet_uniform = packet_rng.random((c.rounds, c.groups))
        self._round = 0
        self._remaining_J = c.episode_budget_J
        self._used_energy_J = 0.
        self._terminal_mse = None
        self._terminal_ssim = None
        self._initialized = True
        self.action_space.update(self._progress < c.source_packets, self._remaining_J, False,
                                 confirmed=self._progress,round_index=self._round)
        return self._observation(), {}

    def _observation(self):
        row = min(self._round, self.config.rounds - 1)
        return {
            'measured_link_strength': np.abs(self._measured_channel[row]).astype(np.float64) ** 2,
            'confirmed_packets': self._progress.copy(),
            'round_index': self._round,
        }

    def _validate_action(self, action):
        try:
            raw = np.asarray(action)
            if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
                raise ValueError('Power must be a real numeric vector')
            p = raw.astype(np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError('Expected finite real power vector[N] in watts') from exc
        c = self.config
        if p.shape != (c.groups,) or not np.isfinite(p).all():
            raise ValueError('Expected finite power vector[N]')
        if np.any(p < 0) or np.any(p > c.maximum_power_W + c.power_tolerance_W):
            raise ValueError('Power out of bounds')
        active = self._progress < c.source_packets
        if np.any(p[~active] != 0):
            raise ValueError('Completed terminals must use zero power')
        energy = float(p.sum() * c.packet_slot_s)
        if self.chapter == 2:
            quota = min(c.period_quota_J, int(active.sum()) * c.maximum_power_W * c.packet_slot_s)
            if abs(energy-quota) > c.energy_tolerance_J:
                raise ValueError('Chapter 2 requires min(period quota, active capacity) energy')
        if energy > self._remaining_J + c.energy_tolerance_J:
            raise ValueError('Episode data energy budget exceeded')
        if self.chapter==3:
            tolerance=min(c.energy_tolerance_J,budget_roundoff(c.episode_budget_J))
            lower=radio_minimum_input(self._remaining_J,self._progress,c.source_packets,
                c.rounds-self._round,c.maximum_power_W*c.packet_slot_s,tolerance=tolerance)
            if energy<lower-tolerance:
                raise ValueError('Power action leaves no guaranteed full-budget continuation')
            if energy>self._remaining_J+tolerance:
                raise ValueError('Power action exceeds the full-allocation resource ledger')
        return p, active, energy

    def step(self, power_W):
        if not self._initialized or self._round >= self.config.rounds:
            raise gym.error.ResetNeeded('Call reset before stepping; episode already ended or not started')
        p, active, energy = self._validate_action(power_W)
        c = self.config
        gamma = np.abs(self._true_channel[self._round]) ** 2 * p / c.reference_power_W
        probability = packet_success_probability(gamma, c.packet_bits)
        success = active & (p > 0) & (self._packet_uniform[self._round] < probability)
        for i in np.flatnonzero(success):
            j = int(self._progress[i])
            self._received_packets[i, j] = self._source_packets[i, j]
        self._progress += success.astype(np.int32)
        self._remaining_J = max(0., self._remaining_J - energy)
        self._used_energy_J += energy
        self._round += 1
        terminal = self._round == c.rounds
        if self.chapter==3 and terminal:
            tolerance=min(c.energy_tolerance_J,budget_roundoff(c.episode_budget_J))
            if self._remaining_J>tolerance or abs(self._used_energy_J-c.episode_budget_J)>tolerance:
                raise ArithmeticError('Radio full-allocation task ended with unspent energy')
        reward, info = 0., {}
        if terminal:
            reconstructed = np.stack([decode_prefix(self._received_packets[i, :n])
                                      for i, n in enumerate(self._progress)])
            self._terminal_mse = np.mean(((self._source.astype(float)-reconstructed)/255) ** 2, axis=(1, 2, 3))
            reward, info = self._terminal_feedback(reconstructed)
        self.action_space.update(self._progress < c.source_packets, self._remaining_J, terminal,
                                 confirmed=self._progress,round_index=self._round)
        return self._observation(), reward, terminal, False, info

    def _terminal_feedback(self, reconstructed):
        self._terminal_ssim = batch_rgb_ssim(self._source, reconstructed)
        info = {}
        if self.chapter == 5:
            # epsilon defines an expected-cost constraint for the learner; do
            # not subtract it or collapse the N individual costs into a mean.
            info['terminal_costs'] = np.maximum(self.tau - self._terminal_ssim, 0.)
        return float(self._terminal_ssim.mean()), info
