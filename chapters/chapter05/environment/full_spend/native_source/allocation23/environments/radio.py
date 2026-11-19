"""Current-state image transmission under the declared ideal telemetry."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from gymnasium import spaces

from ._radio.config import Config as _PhysicalConfig
from ._radio.env import CommunicationImageEnv as _PhysicalEnvironment


@dataclass(frozen=True)
class RadioConfig(_PhysicalConfig):
    strong_noise_multiplier: float = 16.


class RadioEnvironment(_PhysicalEnvironment):
    """Expose exact current CSI/J, confirmed packets, energy and time.

    This is an ideal current-measurement benchmark. Source pixels, future
    channels/interference, success draws, and intermediate quality are private.
    Images must be supplied at reset or through an independent image source.
    """

    _config_type = RadioConfig

    def __init__(self, config=None, chapter=3, *, images=None, image_source=None):
        if isinstance(chapter, bool) or chapter not in (2, 3):
            raise ValueError('This release supports chapters 2 and 3')
        super().__init__(config=config, chapter=chapter, images=images, image_source=image_source)
        c = self.config
        self.observation_version = f'communication-current-csi-interference-progress-chapter{chapter}-v1'
        self.observation_space = spaces.Dict({
            'channel_real': spaces.Box(-np.inf, np.inf, shape=(c.groups,), dtype=np.float64),
            'channel_imag': spaces.Box(-np.inf, np.inf, shape=(c.groups,), dtype=np.float64),
            'interference_multiplier': spaces.Box(c.weak_noise_multiplier, c.strong_noise_multiplier,
                                                  shape=(), dtype=np.float64),
            'confirmed_packets': spaces.Box(0, c.source_packets, shape=(c.groups,), dtype=np.int32),
            'remaining_energy_J': spaces.Box(0., c.episode_budget_J, shape=(), dtype=np.float64),
            'round_index': spaces.Discrete(c.rounds + 1),
        })
        self.observation_fields = tuple(self.observation_space.spaces)

    def _observation(self):
        c = self.config
        row = min(self._round, c.rounds - 1)
        noise = float(self._global_noise[row])
        ratio = c.weak_noise_multiplier / c.strong_noise_multiplier
        normalizer = .5 + .5 * ratio
        scale = np.sqrt((c.weak_noise_multiplier / noise) / normalizer)
        reference_channel = self._true_channel[row] / scale
        return {
            'channel_real': reference_channel.real.astype(np.float64, copy=True),
            'channel_imag': reference_channel.imag.astype(np.float64, copy=True),
            'interference_multiplier': np.asarray(noise, dtype=np.float64),
            'confirmed_packets': self._progress.copy(),
            'remaining_energy_J': np.asarray(self._remaining_J, dtype=np.float64),
            'round_index': np.asarray(self._round),
        }
