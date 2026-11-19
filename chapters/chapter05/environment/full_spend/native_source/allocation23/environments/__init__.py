"""Build the two supported environments and their public policy specification."""
from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from ..spec import Spec
from ._emergency.config import EnvConfig, load_config
from .emergency import EmergencyEnvironment
from .radio import RadioConfig, RadioEnvironment


def _read_condition(condition_file, domain, chapter):
    """Read one explicit physical task, keeping its provenance outside state.

    Full configurations are required: a misspelled or missing physical field
    must fail instead of silently mixing a condition with dataclass defaults.
    The identity is content based so a durable copy may be relocated unchanged.
    """
    path = Path(condition_file).expanduser().resolve(strict=True)
    raw = path.read_bytes()
    payload = json.loads(raw)
    if (not isinstance(payload, dict)
            or set(payload) != {'schema', 'metadata', 'physical_config', 'invariants'}
            or payload['schema'] != 'allocation23-completion-physical-condition-v1'):
        raise ValueError('Expected a registered complete physical-condition file')
    metadata = payload['metadata']
    physical = payload['physical_config']
    if (not isinstance(metadata, dict) or metadata.get('domain') != domain
            or type(metadata.get('chapter')) is not int or metadata['chapter'] != chapter):
        raise ValueError('Physical condition chapter/domain differs from requested task')
    for name in ('condition_id', 'family', 'level'):
        if not isinstance(metadata.get(name), str) or not metadata[name]:
            raise ValueError(f'Condition metadata requires {name}')
    config_type = EnvConfig if domain == 'v5' else RadioConfig
    if not isinstance(physical, dict) or set(physical) != {f.name for f in fields(config_type)}:
        raise ValueError('Condition must supply exactly all physical configuration fields')
    encoded = (json.dumps(physical, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + '\n').encode('utf-8')
    identity = dict(schema=payload['schema'], config_sha256=hashlib.sha256(raw).hexdigest(),
                    physical_config_sha256=hashlib.sha256(encoded).hexdigest(),
                    **{k: metadata[k] for k in ('chapter', 'domain', 'condition_id', 'family', 'level')})
    return physical, identity, str(path)


def build_environment(domain: str, chapter: int, radio_groups: int = 100, *,
                      condition_file=None):
    """Return ``(environment, public_spec, physical_configuration_dict)``.

    ``physical_configuration_dict`` is experiment metadata, not a policy input.
    Radio image data are supplied separately to ``env.reset(options={'images':
    uint8_batch})`` so the environment has no dataset-directory dependency.
    """
    if isinstance(chapter, bool) or chapter not in (2, 3):
        raise ValueError('This release supports chapters 2 and 3')
    if domain not in ('v5', 'radio'):
        raise ValueError("domain must be 'v5' or 'radio'")
    physical, identity, path = (None, None, None)
    if condition_file is not None:
        physical, identity, path = _read_condition(condition_file, domain, chapter)
    if domain == 'v5':
        directory = Path(__file__).resolve().parent / '_emergency' / 'configs'
        protocol = json.loads((directory / f'emergency_chapter{chapter}.json').read_text())
        expected_mode = 'fixed' if chapter == 2 else 'adaptive'
        if (protocol['paper_chapter'] != chapter or protocol['budget_mode'] != expected_mode
                or protocol['feedback'] != 'terminal' or protocol['gamma'] != 1.
                or protocol['objective'] != 'mean_regional_completion'):
            raise ValueError('The bundled protocol must retain the verified chapter contract')
        config = (replace(load_config(directory / protocol['environment_config']), budget_mode=expected_mode)
                  if physical is None else EnvConfig.from_dict(physical))
        if config.budget_mode != expected_mode:
            raise ValueError('Physical condition budget mode differs from chapter')
        c = config.resolve()
        spec = Spec(domain='emergency_allocation_v5', chapter=chapter, groups=c.num_regions,
                    horizon=c.horizon, local_dim=2, global_dim=1 + int(chapter == 3),
                    nominal_total=c.fixed_period_budget, initial_budget=c.initial_budget,
                    per_object_cap=None, action_cost_scale=1., action_tolerance=c.action_tolerance,
                    dictionary_kind='raw')
        env = EmergencyEnvironment(config, chapter)
        env.condition_identity, env.condition_file = identity, path
        return env, spec, config.to_dict()
    if domain == 'radio':
        config = (RadioConfig(groups=radio_groups, frame_s=radio_groups * .001 + .028)
                  if physical is None else RadioConfig(**physical))
        spec = Spec(domain='communication_image_fullstate', chapter=chapter, groups=config.groups,
                    horizon=config.rounds, local_dim=3, global_dim=2 + int(chapter == 3),
                    nominal_total=config.period_quota_J / config.packet_slot_s,
                    initial_budget=config.episode_budget_J / config.packet_slot_s,
                    per_object_cap=config.maximum_power_W, action_cost_scale=config.packet_slot_s,
                    action_tolerance=config.power_tolerance_W, dictionary_kind='channel_power',
                    source_packets=config.source_packets)
        env = RadioEnvironment(config, chapter)
        env.condition_identity, env.condition_file = identity, path
        return env, spec, config.to_dict()


__all__ = ['build_environment', 'EmergencyEnvironment', 'RadioEnvironment', 'RadioConfig']
