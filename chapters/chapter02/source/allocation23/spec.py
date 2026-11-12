"""Public dimensions, units and state contract shared by both methods.

The specification contains execution constraints and normalization constants.
It does not expose service efficiencies, arrival laws, radio transition laws,
packet success functions, source images, or quality predictions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Spec:
    domain: str
    chapter: int
    groups: int
    horizon: int
    local_dim: int
    global_dim: int
    nominal_total: float
    initial_budget: float
    per_object_cap: float | None
    action_cost_scale: float
    action_tolerance: float
    dictionary_kind: str
    dictionary_seed: int = 71331
    source_packets: int = 24
    feature_scale: float = 100.
    peak_total: float | None = None

    def __post_init__(self):
        if self.domain not in ('emergency_allocation_v5', 'communication_image_fullstate'):
            raise ValueError('Supported domain required')
        if isinstance(self.chapter, bool) or self.chapter not in (2, 3):
            raise ValueError('This release supports chapters 2 and 3')
        for name in ('groups', 'horizon', 'local_dim', 'global_dim', 'source_packets', 'dictionary_seed'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'Positive integer required: {name}')
        expected_local = 2 if self.study_domain == 'v5' else 3
        expected_global = (1 if self.study_domain == 'v5' else 2) + int(self.chapter == 3)
        if (self.local_dim, self.global_dim) != (expected_local, expected_global):
            raise ValueError('Dimensions must match the simplified public state')
        for name in ('nominal_total', 'initial_budget', 'action_cost_scale', 'action_tolerance', 'feature_scale'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'Positive finite execution scale required: {name}')
        if self.study_domain == 'v5':
            if self.dictionary_kind != 'raw' or self.per_object_cap is not None or self.peak_total is not None:
                raise ValueError('V5 has raw basis inputs and no extra regional or period cap')
        elif (self.dictionary_kind != 'channel_power' or self.per_object_cap is None
              or not math.isfinite(self.per_object_cap) or self.per_object_cap <= 0):
            raise ValueError('Radio requires the channel-power dictionary and public power limit')
        if self.action_tolerance > self.nominal_total * 1e-6:
            raise ValueError('Tolerance is restricted to numerical roundoff')

    @property
    def study_domain(self):
        return 'v5' if self.domain == 'emergency_allocation_v5' else 'radio'

    @property
    def budget_mode(self):
        return 'fixed' if self.chapter == 2 else 'adaptive'

    @property
    def feature_version(self):
        return f'simplified-{self.study_domain}-chapter{self.chapter}-no-action-history-v1'

    @property
    def observation_version(self):
        return (f'emergency-v5-demand-total-seen-chapter{self.chapter}-v1' if self.study_domain == 'v5'
                else f'communication-current-csi-interference-progress-chapter{self.chapter}-v1')

    @property
    def reward_range(self):
        return (0., 1.) if self.study_domain == 'v5' else (-1., 1.)

    @property
    def resource_unit(self):
        return 'equivalent diesel L' if self.study_domain == 'v5' else 'J'

    def to_dict(self):
        result = asdict(self)
        result.update(feature_interface='simplified', feature_version=self.feature_version,
                      observation_version=self.observation_version, budget_mode=self.budget_mode,
                      resource_unit=self.resource_unit, reward_range=list(self.reward_range),
                      memory_fields=[], neural_balance=self.chapter == 3,
                      observation_assumption=('native public total demand seen so far' if self.study_domain == 'v5'
                          else 'unchanged perfect current CSI and exact current common-interference telemetry'))
        return result
