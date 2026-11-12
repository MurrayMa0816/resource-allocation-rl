"""Physical parameters of the progressive-image radio task (SI units)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral, Real


@dataclass(frozen=True)
class Config:
    groups: int = 100
    rounds: int = 32
    source_packets: int = 24
    payload_bytes: int = 512
    crc_bits: int = 32
    packet_slot_s: float = .001
    frame_s: float = .128
    control_and_guard_reserve_s: float = .028
    reference_power_W: float = .001
    # Budget scale is independent of the power used to define reference SNR.
    # None is retained only for the historical reference-power quota rule.
    quota_power_per_terminal_W: float | None = .00015
    maximum_power_W: float = .002
    reference_mean_snr_min_dB: float = 6.
    reference_mean_snr_max_dB: float = 18.
    channel_rho: float = .8
    csi_complex_error_variance: float = 1 / 16
    deployment_seed: int = 2026102904
    global_stay_probability: float = .8
    weak_noise_multiplier: float = 1.
    strong_noise_multiplier: float = 4.
    global_seed_salt: int = 2026102915
    # None scales quotas with population and horizon.
    period_quota_override_J: float | None = None
    episode_budget_override_J: float | None = None
    energy_tolerance_J: float = 1e-15
    power_tolerance_W: float = 1e-12

    def __post_init__(self):
        self.validate()

    @property
    def packet_bits(self):
        return 8 * self.payload_bytes + self.crc_bits

    @property
    def period_quota_J(self):
        if self.period_quota_override_J is not None:
            return self.period_quota_override_J
        quota_power = (self.reference_power_W if self.quota_power_per_terminal_W is None
                       else self.quota_power_per_terminal_W)
        return self.groups * quota_power * self.packet_slot_s

    @property
    def episode_budget_J(self):
        if self.episode_budget_override_J is not None:
            return self.episode_budget_override_J
        return self.rounds * self.period_quota_J

    @property
    def gross_symbol_rate_per_s(self):
        return self.packet_bits / self.packet_slot_s

    def to_dict(self):
        return asdict(self)

    def validate(self):
        for key in ('groups', 'rounds', 'source_packets', 'payload_bytes', 'crc_bits',
                    'deployment_seed', 'global_seed_salt'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
        if self.groups < 1 or self.rounds < 1:
            raise ValueError('groups and rounds must be positive')
        if (self.source_packets, self.payload_bytes, self.crc_bits) != (24, 512, 32):
            raise ValueError('This codec uses 24 packets of 512 bytes plus CRC32')
        for key, value in asdict(self).items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f'{key} must be a finite real number')
        for key in ('packet_slot_s', 'frame_s', 'reference_power_W', 'maximum_power_W',
                    'weak_noise_multiplier', 'strong_noise_multiplier'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        for key in ('control_and_guard_reserve_s', 'csi_complex_error_variance',
                    'energy_tolerance_J', 'power_tolerance_W'):
            if getattr(self, key) < 0:
                raise ValueError(f'{key} must be nonnegative')
        if self.quota_power_per_terminal_W is not None and self.quota_power_per_terminal_W < 0:
            raise ValueError('quota_power_per_terminal_W must be nonnegative or None')
        if self.reference_power_W > self.maximum_power_W:
            raise ValueError('reference_power_W exceeds maximum_power_W')
        if self.groups * self.packet_slot_s + self.control_and_guard_reserve_s > self.frame_s + 1e-12:
            raise ValueError('Sequential data slots and guard/control reserve exceed frame_s')
        if not -1 <= self.channel_rho <= 1:
            raise ValueError('channel_rho must be between -1 and 1')
        if not 0 <= self.global_stay_probability < 1:
            raise ValueError('global_stay_probability must be in [0, 1)')
        if self.weak_noise_multiplier > self.strong_noise_multiplier:
            raise ValueError('weak_noise_multiplier exceeds strong_noise_multiplier')
        # Prevent numerical overflow/underflow in the linear link model.
        if not -300 <= self.reference_mean_snr_min_dB <= self.reference_mean_snr_max_dB <= 300:
            raise ValueError('Require ordered mean SNR bounds within [-300, 300] dB')
        for key in ('period_quota_override_J', 'episode_budget_override_J'):
            value = getattr(self, key)
            if value is not None and value < 0:
                raise ValueError(f'{key} must be nonnegative or None')
        capacity = self.groups * self.maximum_power_W * self.packet_slot_s
        if not isfinite(capacity) or not isfinite(self.episode_budget_J):
            raise ValueError('Energy scales overflow')
        if self.period_quota_J > capacity + self.energy_tolerance_J:
            raise ValueError('Period quota exceeds all-terminal capacity')
        if self.power_tolerance_W > self.maximum_power_W * 1e-6:
            raise ValueError('Power tolerance must only cover numerical roundoff')
        if self.energy_tolerance_J > capacity * 1e-6:
            raise ValueError('Energy tolerance must only cover numerical roundoff')
