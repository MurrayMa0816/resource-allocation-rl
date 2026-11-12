"""Additional C6 COLA comparison from immutable, accepted C5 formal285.

Only public I/O identity and deterministic fixed-point inference are adapted.
Native OADM/COR/SAC losses,8-step TD and true1-step OADM labels stay distinct.
No environment, rollout, reward, replay or model-selection operation is here.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
_SHARED=Path(__file__).resolve().parents[2]
if str(_SHARED) not in sys.path:sys.path.insert(0,str(_SHARED))
import c6_paths

import numpy as np
import torch

FORMAL = c6_paths.shared_formal285()
COLA = FORMAL / 'baselines/cola'
OBS_DIM, ACTION_DIM = 719, 49
PREFERENCE = (.2, .8)
VERSION = 'chapter6-additional-formal285-cola-fixed-oadm-io-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _formal_agent_class():
    """Absolute formal-root private package; portable from any HY snapshot."""
    package_name = '_irrigation_comparison_formal285_baselines'
    if package_name not in sys.modules:
        path = FORMAL / 'baselines/__init__.py'
        spec = importlib.util.spec_from_file_location(package_name, path,
            submodule_search_locations=[str(path.parent)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        try:
            spec.loader.exec_module(package)
        except BaseException:
            sys.modules.pop(package_name, None)
            raise
    module = importlib.import_module(package_name + '.cola.agent')
    if Path(module.__file__).resolve() != COLA / 'agent.py':
        raise RuntimeError('COLA must be loaded from the accepted formal285 closure')
    return module.Agent


def configuration():
    document = json.loads((COLA / 'configs.json').read_text())
    cfg = copy.deepcopy(document['configurations']['v5']['fixed_oadm'])
    cfg.update(method='cola', task_domain='hongmiao89_reasonable_depth_irrigation',
        historical_recipe_domain='v5', objective_order=['C', 'JA'],
        coverage_threshold=.9, deployment_preference=.2,
        evaluation_preference=list(PREFERENCE), horizon=81,
        observation_dim=OBS_DIM, action_dim=ACTION_DIM,
        learning_steps=259200, n_env=8,
        acknowledge_nstep_task_variant=True)
    return cfg


def make_agent(obs_dim, action_dim, seed, device, config=None):
    if int(obs_dim) != OBS_DIM or int(action_dim) != ACTION_DIM:
        raise ValueError('C6 COLA requires719 public coordinates and49 latent actions')
    cfg = configuration()
    cfg.update(dict(config or {}))
    expected = configuration()
    # Preserve the complete formal numerical recipe. Horizon/budget and
    # n_env/learning_steps are task/worker configuration, not new loss tuning.
    frozen = ('hidden_dim', 'policy_hidden_dim', 'batch_size', 'gamma', 'lr',
        'tau', 'updates_per_step', 'start_steps', 'learning_starts', 'alpha',
        'entropy_tuning', 'cola_latent_dim', 'cola_old_q_update_frequency',
        'cola_regular_alpha', 'cola_regular_bar', 'preference_floor',
        'target_entropy_scale', 'reward_scale', 'alpha_min', 'alpha_max',
        'actor_output_init_scale', 'observation_normalization',
        'observation_calibration_steps', 'actor_lr', 'actor_gradient_clip',
        'n_step', 'cola_batched_finite_checks', 'action_mode',
        'oadm_one_step_target', 'profile')
    for key in frozen:
        if cfg[key] != expected[key]:
            raise ValueError('Frozen C6 COLA recipe differs at ' + key)
    if float(cfg['deployment_preference']) != .2:
        raise ValueError('C6 deployment coverage preference is.2')
    parent = _formal_agent_class()

    class IrrigationCOLA(parent):
        def metadata(self):
            result = dict(super().metadata())
            result.pop('candidate_status', None)
            result.update(adapter_version=VERSION,
                implementation_authority='accepted C5 formal285; embedded candidate-preparation labels are historical',
                accepted_source_root=str(FORMAL),
                task_domain='hongmiao89_reasonable_depth_irrigation',
                objective_order=['C', 'JA'], coverage_threshold=.9,
                deployment_preference=.2, evaluation_preference=list(PREFERENCE),
                observation_dim=OBS_DIM, action_dim=ACTION_DIM,
                observation_encoding='same public89x8+5global+2ledger ratios; preference external',
                action_execution='native tanhGaussian coordinates; common bridge applies atanh clip1e-6 once then original C6 quota/area-depth map',
                fixed64='current OADM encoder→actorGaussian raw mean+std*eps→tanh; common bridge averages individually feasible physical vectors',
                train_preference_timing='native train_preferences once per complete8lane81day panel; preference fixed within each lane episode',
                true_one_step_oadm_label_required=True, replay_n_step=8,
                native_task_adaptation='gamma1 n-step raw reward accumulation; no intermediate entropy/importance correction, as accepted C5 recipe',
                learning_core_unchanged=True, normalization=False,
                extra_reward_or_state=False, CPU_sampling_mirror=False,
                task_threshold_is_not_polyak_tau=True)
            return result

    return IrrigationCOLA(int(obs_dim), int(action_dim), int(seed), device, cfg)


@torch.no_grad()
def fixed64_actions(agent, observations, eps, preference=PREFERENCE):
    """Native tanh-coordinate[B,64,49] points; no model/RNG update."""
    observations = np.asarray(observations)
    if observations.ndim == 1:
        observations = observations[None, :]
    points = np.asarray(eps)
    if observations.ndim != 2 or observations.shape[1] != OBS_DIM or not np.isfinite(observations).all():
        raise ValueError('Finite[B,719] public observations required')
    if points.shape != (64, ACTION_DIM) or not np.isfinite(points).all():
        raise ValueError('Common finite64x49 pre-tanh Gaussian points required')
    weights = np.asarray(preference, dtype=np.float32)
    if weights.shape == (2,):
        weights = np.broadcast_to(weights, (len(observations), 2)).copy()
    if weights.shape != (len(observations), 2) or not np.isfinite(weights).all():
        raise ValueError('One finite two-objective preference per observation required')
    if np.any(weights < 0) or not np.allclose(weights.sum(1), 1., rtol=0, atol=1e-6):
        raise ValueError('Normalized nonnegative objective preferences required')
    device = next(agent.actor.parameters()).device
    states = torch.as_tensor(observations, dtype=torch.float32, device=device)
    prefs = torch.as_tensor(weights, dtype=torch.float32, device=device)
    native_input = agent.actor_input(states, prefs)
    mean, log_std = agent.actor.net(native_input).chunk(2, dim=-1)
    log_std = log_std.clamp(-20., 2.)
    # Existing C6 fixed points are float64. Keep native forward/std float32,
    # then evaluate the declared finite-point inference operator in float64.
    raw = mean.to(torch.float64)[:, None, :] + log_std.exp().to(torch.float64)[:, None, :] * torch.as_tensor(points, dtype=torch.float64, device=device)[None, :, :]
    output = torch.tanh(raw).cpu().numpy()
    if output.shape != (len(observations), 64, ACTION_DIM) or not np.isfinite(output).all():
        raise FloatingPointError('Invalid native COLA quadrature candidates')
    if np.any(output < -1.) or np.any(output > 1.):
        raise FloatingPointError('Native COLA action escaped its tanh box')
    return output


def source_bindings():
    paths = [p for p in COLA.rglob('*') if p.is_file() and p.suffix in ('.py', '.json')]
    paths += [FORMAL / 'baselines/__init__.py']
    return {str(p): sha(p) for p in sorted(set(paths))}


__all__ = ['configuration', 'make_agent', 'fixed64_actions', 'source_bindings']
