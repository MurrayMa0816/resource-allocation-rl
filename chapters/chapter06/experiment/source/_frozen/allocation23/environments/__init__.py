"""Build the two supported environments and their public policy specification."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from ..spec import Spec
from ._emergency.config import load_config
from .emergency import EmergencyEnvironment
from .radio import RadioConfig, RadioEnvironment


def build_environment(domain: str, chapter: int, radio_groups: int = 100):
    """Return ``(environment, public_spec, physical_configuration_dict)``.

    ``physical_configuration_dict`` is experiment metadata, not a policy input.
    Radio image data are supplied separately to ``env.reset(options={'images':
    uint8_batch})`` so the environment has no dataset-directory dependency.
    """
    if isinstance(chapter, bool) or chapter not in (2, 3):
        raise ValueError('This release supports chapters 2 and 3')
    if domain == 'v5':
        directory = Path(__file__).resolve().parent / '_emergency' / 'configs'
        protocol = json.loads((directory / f'emergency_chapter{chapter}.json').read_text())
        expected_mode = 'fixed' if chapter == 2 else 'adaptive'
        if (protocol['paper_chapter'] != chapter or protocol['budget_mode'] != expected_mode
                or protocol['feedback'] != 'terminal' or protocol['gamma'] != 1.
                or protocol['objective'] != 'mean_regional_completion'):
            raise ValueError('The bundled protocol must retain the verified chapter contract')
        config = replace(load_config(directory / protocol['environment_config']), budget_mode=expected_mode)
        c = config.resolve()
        spec = Spec(domain='emergency_allocation_v5', chapter=chapter, groups=c.num_regions,
                    horizon=c.horizon, local_dim=2, global_dim=1 + int(chapter == 3),
                    nominal_total=c.fixed_period_budget, initial_budget=c.initial_budget,
                    per_object_cap=None, action_cost_scale=1., action_tolerance=c.action_tolerance,
                    dictionary_kind='raw')
        return EmergencyEnvironment(config, chapter), spec, config.to_dict()
    if domain == 'radio':
        config = RadioConfig(groups=radio_groups, frame_s=radio_groups * .001 + .028)
        spec = Spec(domain='communication_image_fullstate', chapter=chapter, groups=config.groups,
                    horizon=config.rounds, local_dim=3, global_dim=2 + int(chapter == 3),
                    nominal_total=config.period_quota_J / config.packet_slot_s,
                    initial_budget=config.episode_budget_J / config.packet_slot_s,
                    per_object_cap=config.maximum_power_W, action_cost_scale=config.packet_slot_s,
                    action_tolerance=config.power_tolerance_W, dictionary_kind='channel_power',
                    source_packets=config.source_packets)
        return RadioEnvironment(config, chapter), spec, config.to_dict()
    raise ValueError("domain must be 'v5' or 'radio'")


__all__ = ['build_environment', 'EmergencyEnvironment', 'RadioEnvironment', 'RadioConfig']
