"""PASTA for C6, importing the actual C5 formal285 frozen implementation.

Training sampling and update_rollout are inherited without replacement. This
module adapts configuration, identity and fixed64 latent distribution queries.
It does not import an irrigation environment or decode physical allocations.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
_SHARED=Path(__file__).resolve().parents[2]
if str(_SHARED) not in sys.path:sys.path.insert(0,str(_SHARED))
import c6_paths

import numpy as np
import torch

VERSION = 'c6-pasta-formal285-configuration-adapter-v1'
# This adapter is also copied into HY run snapshots. Resolve the scientific
# authority from its registered stable location, never from the copy's parents.
FORMAL_ROOT = c6_paths.shared_formal285()
PASTA_ROOT = FORMAL_ROOT / 'baselines/pasta'
NAMESPACE = '_c6_pasta_actual_formal285_20261109'
SOURCE_SHA256 = {
    '__init__.py': 'fa8dc959f25b8337506c3a1e7f717e1c76448341c1846aeb26804a90495fea3d',
    'agent.py': '3259f50527eb1aec8fcdb91f233eed960c64f5a2310685161a0e70afbc2f38c1',
    '_native.py': 'e97972992b3383ccd275ddced04b721ca1586c20cf13f4c956e574dacf97f92d',
    'configs.json': 'e3e29b01953acfdd379050108d8d74056e9b43561b83bc6707f989f0be77aad7',
    'README.rst': 'bebc5cb25d81d0d716c02b7daff40469284eb9c4d718a1d61f67c317ab986eb5',
    'RELEASE.json': '0452723ed5e8a0ae61c52495cd5f9cb54ddf850e6a268f4e421b931ff1676a0b',
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source():
    for name, expected in SOURCE_SHA256.items():
        path = PASTA_ROOT / name
        if _sha(path) != expected:
            raise RuntimeError(f'PASTA actual formal285 source mismatch: {path}')
    return dict(SOURCE_SHA256)


def _native_module():
    verify_source()
    if NAMESPACE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            NAMESPACE, PASTA_ROOT / '__init__.py',
            submodule_search_locations=[str(PASTA_ROOT)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[NAMESPACE] = module
        old_no_bytecode = sys.dont_write_bytecode
        try:
            # Frozen source is read-only; even import caches stay out of it.
            sys.dont_write_bytecode = True
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(NAMESPACE, None)
            raise
        finally:
            sys.dont_write_bytecode = old_no_bytecode
    return sys.modules[NAMESPACE]


def configuration(engineering=False):
    """Return the complete pre-fixed irrigation configuration as a fresh dict."""
    if not isinstance(engineering, bool):
        raise TypeError('engineering must be an explicit boolean')
    verify_source()
    recipe = json.loads((PASTA_ROOT / 'configs.json').read_text())[
        'configurations']['v5']['current_recipe']
    config = copy.deepcopy(recipe)
    config.update(domain='irrigation89', horizon=81,
                  learning_steps=5184 if engineering else 259200,
                  training_preference=[0.2, 0.8], dev_paths=2 if engineering else 32,
                  engineering=engineering)
    assert config['n_env'] == 8 and config['rollout_panels'] == 8
    assert config['minibatch_size'] == 64 and config['ppo_epochs'] == 10
    assert config['observation_normalization'] is False
    return config


NativeAgent = _native_module().Agent


class Agent(NativeAgent):
    """Transparent native Agent with accurate C6 metadata only.

    sample_info, train_preferences, update_rollout, vectorGAE, PCGrad,
    checkpoint contents, RNGs and optimizers are the frozen methods.
    """

    def __init__(self, obs_dim, action_dim, seed=0, device='cpu', config=None):
        engineering = False if config is None else config.get('engineering', False)
        expected = configuration(engineering=engineering)
        if config is not None:
            unknown = set(config) - set(expected)
            different = {key: value for key, value in config.items()
                         if key in expected and value != expected[key]}
            if unknown or different:
                raise ValueError(f'Unregistered PASTA C6 config overrides: unknown={sorted(unknown)}, different={different}')
            expected.update(copy.deepcopy(config))
        if int(obs_dim) != 719 or int(action_dim) != 49:
            raise ValueError('C6 public base observation719 and raw action49 required; preference is separate')
        super().__init__(obs_dim, action_dim, seed=seed, device=device, config=expected)

    def metadata(self):
        record = super().metadata()
        record['frozen_native_adapter_version'] = record['adapter_version']
        record.update(
            adapter_version=VERSION,
            actual_import_root=str(FORMAL_ROOT),
            actual_resolved_import_root=str(FORMAL_ROOT.resolve()),
            actual_frozen_source_sha256=verify_source(),
            source_recipe='actual C5 formal285 configs.json configurations.v5.current_recipe',
            inherited_source_document_status='README describes the adopted configuration; RELEASE retains the original preparation record. Implementation files remain the frozen formal285 source.',
            configuration=configuration(engineering=self.config['engineering']),
            engineering=self.config['engineering'],
            registered_native_update_groups=self.total_steps // self.rollout_size,
            public_observation='719 base features =89*8+5+2ledger, no appended preference; preference passed separately',
            reward='Actual terminal vector [C90, JA] on original0..1 scales; all intermediate rewards zero',
            task_adaptations=['C6 public observation719 and raw action49',
                              'complete81day true-terminal horizon',
                              'fixed preference[0.2,0.8] passed separately',
                              '8parallel paths times8complete panels =5184physical learning steps per native update',
                              f'{self.total_steps}learning steps ={self.total_steps // self.horizon}episodes ={self.total_steps // self.rollout_size}complete native update groups',
                              f"{self.config['dev_paths']}fixedDEVweather paths; fixed64standardized latent points decoded by the shared physical bridge"],
            sampling_and_update='Unmodified inherited frozen native methods on the requested device; no CPU mirror or altered RNG backend',
            action_adapter='Inherited unbounded raw Gaussian x=2*y-1, mean2*mu-1 andstd2*sigma; shared C6 quota/decode outside this module',
            fixed64='Native float32 distribution mean/std promoted to float64 and combined with64float64 standardized points; no ranking or trajectory-return averaging; training sampling/logp unchanged',
            preference_is_scalarized_PPO_replacement=False,
            external_physical_decoder_owned_by_runner=True,
        )
        return record


def make_agent(obs_dim, action_dim, seed, device, config=None):
    return Agent(obs_dim, action_dim, seed=seed, device=device, config=config)


@torch.no_grad()
def distribution_parameters(agent, obs, preference=(0.2, 0.8)):
    """Native raw-Gaussian parameters, no policy or RNG state mutation."""
    if not isinstance(agent, NativeAgent):
        raise TypeError('PASTA frozen-native Agent required')
    x, pref = agent._inputs(obs, preference)
    distribution = agent._dist(x, pref)
    return (distribution.mean.detach().cpu().numpy().copy(),
            distribution.stddev.expand_as(distribution.mean).detach().cpu().numpy().copy())


@torch.no_grad()
def fixed64_actions(agent, obs, eps, preference=(0.2, 0.8)):
    """Return[B,64,49]unbounded native raw actions for shared feasible decoding.

    eps contains the caller's registered64standardized points. This function
    neither samples RNG nor decodes/chooses actions or evaluates returns.
    """
    x, pref = agent._inputs(obs, preference)
    standard = np.asarray(eps, dtype=np.float64)
    if standard.shape != (64, agent.action_dim) or not np.isfinite(standard).all():
        raise ValueError('Exactly64finite standardized points of shape[64,49] required')
    distribution = agent._dist(x, pref)
    # Preserve the native network/distribution computation, then perform the
    # fixed-point arithmetic in float64 for the shared physical decoder.
    # Training sample_info/logp remain the inherited native float32 methods.
    mean = distribution.mean.to(torch.float64)
    std = distribution.stddev.expand_as(distribution.mean).to(torch.float64)
    noise = torch.as_tensor(standard, dtype=torch.float64, device=agent.device)
    actions = mean[:, None, :] + std[:, None, :] * noise[None, :, :]
    return actions.detach().cpu().numpy().copy()


def provenance():
    return dict(
        method='PASTA', adapter_version=VERSION,
        actual_formal_root=str(FORMAL_ROOT), resolved_formal_root=str(FORMAL_ROOT.resolve()),
        source_sha256=verify_source(), source_recipe='v5/current_recipe',
        inherited_original_commit='745d769740b6ce648c7ea8209933733dea8a1eab',
        inherited_repairs=['Actual done[t] successor masks for complete-path vectorGAE',
                           'PCGrad comparisons preserve original objective identity'],
        inherited_actor_update='Per-objectivePPO surrogate then sum of PCGrad-projected gradients plus entropy; not a fixed scalar actor loss',
        inherited_critic_update='Adaptive smoothTchebycheff attention with running min/max actualGAEreturns and native smoothness controller',
        inherited_distribution='Native yGaussian with sigmoid mean; rawx=2y-1 gives(mean2mu-1,std2sigma)',
        configuration=configuration(), engineering_configuration=configuration(engineering=True),
        allowed_learning_budgets={'formal': 259200, 'explicit_engineering_only': 5184},
        engineering_flag_required_for_5184=True,
        obs_dim=719, action_dim=49, objective_order=['C90','JA'],
        objective_scales='Original0..1, no external reward scaling',
        rollout_steps=5184, native_update_groups=50,
        actor_critic_minibatch_updates_per_group=810,
        observation_preference_interface='719base features; preference separate[.2,.8]',
        action_interface='49raw coefficients; shared quota/decode outside module',
        DEV='64registered standardized Gaussian points→native raw actions→shared physical feasible vectors→one meanphysicalaction',
        fixed64_precision='Native float32 mean/std are promoted without recomputation to float64, combined with float64 fixed standardized points; training sample_info and logp unchanged',
        no_teacher=True, no_future_weather=True, no_extra_action_selection=True,
        official_code_license_declared=False,
        source_readme_historical_candidate_label_preserved=False,
        new_environment_constructions=0, new_environment_steps=0,
        new_training_runs=0, new_model_loads=0,
    )
