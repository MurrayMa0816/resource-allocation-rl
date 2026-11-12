"""Five-seed formal baseline facade over the frozen native learner.

This file changes registration and provenance, not learner mathematics.  The
DEV-only native backend has the long worker's unchanged rollout, normalization,
replay, charged internal MORL/D interactions, and optimizer code.  The original
source closure is never edited.  A precisely bounded in-memory adapter permits
100608 interactions with regular512 DEV ticks plus the fixed off-grid endpoint.
Validation imports only the standard library and constructs no environment,
model, CUDA context, or optimizer.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
import traceback
import textwrap


VERSION = "chapter5-formal-five-seed-baseline-worker-v2-terminal-rollout"
SCHEMA = "chapter5-formal-random-5seed-contract-v1"
ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = ROOT / "execution/source"
SOURCE_MANIFEST = ROOT / "execution/source_manifest.json"
NATIVE_BACKEND_SHA = "498737237e2a6bf1f887429e9c1a7fd3037093b9e3056653452c2e5c534025fb"
NATIVE_PASTA_SHA = "3259f50527eb1aec8fcdb91f233eed960c64f5a2310685161a0e70afbc2f38c1"
PASTA_NAME = "baselines/pasta/agent.py"
BACKEND_NAME = "training/final_morld_worker.py"
BASELINE_PROFILES = {
    "capql": "fixed",
    "qpensieve": "legacy_recipe",
    "cola": "fixed_oadm",
    "morld": "critic_lr1e4",
    "pasta": "current_recipe",
}
DOMAINS = {
    "radio": {"horizon": 32, "N": 100, "task_tau": 0.2},
    "v5": {"horizon": 24, "N": 120, "task_tau": 0.7},
}
MORLD_PROFILE = {
    "actor_lr": 0.0003,
    "critic_lr": 0.0001,
    "morld_update_passes": 10,
    "morld_other_update_every_training_episodes": 1,
}

# The ten accepted native recipes are copied as values, with no dependency on an
# earlier cohort or its mutable configs. Formal fields such as seed and budget
# are validated separately; every native training option stays fixed.
NATIVE_RECIPES = {'capql__radio': {'acknowledge_nstep_task_variant': False,
                  'action_mode': 'tanh_latent',
                  'actor_lr': 0.0001,
                  'alpha': 0.005,
                  'batch_size': 256,
                  'critic_lr': 0.0001,
                  'domain': 'radio',
                  'gamma': 1.0,
                  'hidden_dim': 256,
                  'learning_starts': 2048,
                  'lr': 0.0003,
                  'method': 'capql',
                  'n_env': 1,
                  'observation_calibration_steps': 2048,
                  'observation_normalization': True,
                  'profile': 'fixed',
                  'reward_scale': 4.0,
                  'start_steps': 2048,
                  'task_tau': 0.2,
                  'tau': 0.005,
                  'updates_per_step': 1},
 'capql__v5': {'acknowledge_nstep_task_variant': False,
               'action_mode': 'tanh_latent',
               'actor_lr': 0.0001,
               'alpha': 0.01,
               'batch_size': 256,
               'critic_lr': 0.0001,
               'domain': 'v5',
               'gamma': 1.0,
               'hidden_dim': 256,
               'learning_starts': 2048,
               'lr': 0.0003,
               'method': 'capql',
               'n_env': 1,
               'observation_calibration_steps': 2048,
               'observation_normalization': False,
               'profile': 'fixed',
               'reward_scale': 1.0,
               'start_steps': 2048,
               'task_tau': 0.7,
               'tau': 0.005,
               'updates_per_step': 1},
 'cola__radio': {'acknowledge_nstep_task_variant': True,
                 'action_mode': 'tanh_latent',
                 'actor_gradient_clip': 1.0,
                 'actor_lr': 0.0001,
                 'actor_output_init_scale': 0.5,
                 'alpha': 0.001,
                 'alpha_max': 0.05,
                 'alpha_min': 1e-06,
                 'batch_size': 256,
                 'cola_batched_finite_checks': True,
                 'cola_latent_dim': 50,
                 'cola_old_q_update_frequency': 1,
                 'cola_regular_alpha': 0.1,
                 'cola_regular_bar': 0.25,
                 'deployment_preference': 0.8,
                 'development_cycle': 4,
                 'domain': 'radio',
                 'entropy_tuning': False,
                 'gamma': 1.0,
                 'hidden_dim': 256,
                 'holdout_selection': False,
                 'learning_starts': 2048,
                 'lr': 0.0003,
                 'method': 'cola',
                 'n_env': 1,
                 'n_step': 8,
                 'oadm_one_step_target': True,
                 'observation_calibration_steps': 2048,
                 'observation_normalization': True,
                 'policy_hidden_dim': 128,
                 'preference_floor': 0.02,
                 'profile': 'fixed_oadm',
                 'reward_scale': 1.0,
                 'start_steps': 2048,
                 'target_entropy_scale': 1.0,
                 'task_tau': 0.2,
                 'tau': 0.005,
                 'updates_per_step': 1},
 'cola__v5': {'acknowledge_nstep_task_variant': True,
              'action_mode': 'tanh_latent',
              'actor_gradient_clip': 1.0,
              'actor_lr': 0.0001,
              'actor_output_init_scale': 0.5,
              'alpha': 0.001,
              'alpha_max': 0.05,
              'alpha_min': 1e-06,
              'batch_size': 256,
              'cola_batched_finite_checks': True,
              'cola_latent_dim': 50,
              'cola_old_q_update_frequency': 1,
              'cola_regular_alpha': 0.1,
              'cola_regular_bar': 0.25,
              'deployment_preference': 0.8,
              'development_cycle': 5,
              'domain': 'v5',
              'entropy_tuning': False,
              'gamma': 1.0,
              'hidden_dim': 256,
              'holdout_selection': False,
              'learning_starts': 2048,
              'lr': 0.0001,
              'method': 'cola',
              'n_env': 8,
              'n_step': 8,
              'oadm_one_step_target': True,
              'observation_calibration_steps': 2048,
              'observation_normalization': False,
              'policy_hidden_dim': 128,
              'preference_floor': 0.02,
              'profile': 'fixed_oadm',
              'reward_scale': 1.0,
              'start_steps': 2048,
              'target_entropy_scale': 1.0,
              'task_tau': 0.7,
              'tau': 0.002,
              'updates_per_step': 1},
 'morld__radio': {'acknowledge_nstep_task_variant': True,
                  'action_mode': 'tanh_latent',
                  'actor_lr': 0.0003,
                  'alpha': 0.005,
                  'autotune': False,
                  'batch_size': 256,
                  'charged_physical_budget_includes_internal_psa': True,
                  'critic_lr': 0.0001,
                  'domain': 'radio',
                  'eval_interval': 512,
                  'gamma': 1.0,
                  'hidden_dim': 256,
                  'horizon': 32,
                  'internal_cases': 5,
                  'learning_starts': 2048,
                  'legacy_constructor_cuda_reseed': False,
                  'method': 'morld',
                  'morld_other_update_every_training_episodes': 1,
                  'morld_population': 3,
                  'morld_psa_every': 3072,
                  'morld_psa_internal_cases': 5,
                  'morld_psa_value_source': 'charged_internal',
                  'morld_update_passes': 10,
                  'n_env': 1,
                  'n_step': 32,
                  'observation_normalization': False,
                  'policy_freq': 2,
                  'profile': 'critic_lr1e4',
                  'start_steps': 2048,
                  'target_net_freq': 1,
                  'task_tau': 0.2,
                  'tau': 0.005,
                  'updates_per_step': 1},
 'morld__v5': {'acknowledge_nstep_task_variant': False,
               'action_mode': 'tanh_latent',
               'actor_lr': 0.0003,
               'alpha': 0.005,
               'autotune': False,
               'batch_size': 256,
               'charged_physical_budget_includes_internal_psa': True,
               'critic_lr': 0.0001,
               'domain': 'v5',
               'eval_interval': 512,
               'gamma': 1.0,
               'hidden_dim': 256,
               'horizon': 24,
               'internal_cases': 5,
               'learning_starts': 2048,
               'legacy_constructor_cuda_reseed': False,
               'method': 'morld',
               'morld_diagnostics_every_physical_steps': 512,
               'morld_other_update_every_training_episodes': 1,
               'morld_population': 3,
               'morld_psa_every': 3072,
               'morld_psa_internal_cases': 5,
               'morld_psa_value_source': 'charged_internal',
               'morld_update_passes': 10,
               'n_env': 1,
               'n_step': 1,
               'observation_normalization': False,
               'policy_freq': 2,
               'profile': 'critic_lr1e4',
               'start_steps': 2048,
               'task_tau': 0.7,
               'tau': 0.005,
               'updates_per_step': 1},
 'pasta__radio': {'acknowledge_nstep_task_variant': False,
                  'action_mode': 'raw',
                  'batch_size': 64,
                  'clip': 0.2,
                  'conflict_threshold': 0.4,
                  'domain': 'radio',
                  'entropy_coef': 0.01,
                  'gae_lambda': 0.95,
                  'gamma': 1.0,
                  'hidden_dim': 64,
                  'hidden_layers': 2,
                  'horizon': 32,
                  'initial_log_std': -0.5,
                  'learning_starts': 0,
                  'lr': 0.0001,
                  'maintenance_rate': 0.15,
                  'max_grad_norm': 0.5,
                  'max_mu': 10.0,
                  'method': 'pasta',
                  'min_mu': 0.05,
                  'minibatch_size': 64,
                  'n_env': 8,
                  'n_step': 1,
                  'observation_normalization': False,
                  'ppo_epochs': 10,
                  'profile': 'current_recipe',
                  'rollout_panels': 8,
                  'smoothness_ema': 0.05,
                  'start_steps': 0,
                  'task_tau': 0.2,
                  'training_preference': [0.8, 0.2],
                  'updates_per_step': 1,
                  'utopia_point': 1.05,
                  'value_loss_coef': 0.5},
 'pasta__v5': {'acknowledge_nstep_task_variant': False,
               'action_mode': 'raw',
               'batch_size': 64,
               'clip': 0.2,
               'conflict_threshold': 0.4,
               'domain': 'v5',
               'entropy_coef': 0.01,
               'gae_lambda': 0.95,
               'gamma': 1.0,
               'hidden_dim': 64,
               'hidden_layers': 2,
               'horizon': 24,
               'initial_log_std': -0.5,
               'learning_starts': 0,
               'lr': 0.0003,
               'maintenance_rate': 0.15,
               'max_grad_norm': 0.5,
               'max_mu': 10.0,
               'method': 'pasta',
               'min_mu': 0.05,
               'minibatch_size': 64,
               'n_env': 8,
               'n_step': 1,
               'observation_normalization': False,
               'ppo_epochs': 10,
               'profile': 'current_recipe',
               'rollout_panels': 8,
               'smoothness_ema': 0.05,
               'start_steps': 0,
               'task_tau': 0.7,
               'training_preference': [0.8, 0.2],
               'updates_per_step': 1,
               'utopia_point': 1.05,
               'value_loss_coef': 0.5},
 'qpensieve__radio': {'acknowledge_nstep_task_variant': True,
                      'action_mode': 'tanh_latent',
                      'actor_lr': 0.0001,
                      'alpha': 0.001,
                      'batch_size': 256,
                      'critic_lr': 0.0001,
                      'critic_objective_loss': 'full_vector_mean',
                      'domain': 'radio',
                      'entropy_tuning': False,
                      'gamma': 1.0,
                      'hidden_dim': 256,
                      'learning_starts': 2048,
                      'lr': 0.0003,
                      'method': 'qpensieve',
                      'n_env': 1,
                      'n_step': 8,
                      'network_initializer': 'legacy_torch_default',
                      'observation_calibration_steps': 2048,
                      'observation_normalization': True,
                      'preference_sampling': 'author_positive_uniform',
                      'profile': 'legacy_recipe',
                      'qp_preference_candidates': 4,
                      'qp_snapshot_capacity': 4,
                      'qp_snapshot_frequency': 2048,
                      'qp_snapshot_warmup': 4096,
                      'reward_scale': 1.0,
                      'start_steps': 2048,
                      'task_tau': 0.2,
                      'tau': 0.005,
                      'temperature_lr': 3e-05,
                      'updates_per_step': 1},
 'qpensieve__v5': {'acknowledge_nstep_task_variant': True,
                   'action_mode': 'tanh_latent',
                   'actor_lr': 0.0001,
                   'alpha': 0.001,
                   'batch_size': 256,
                   'critic_lr': 0.0001,
                   'critic_objective_loss': 'full_vector_mean',
                   'domain': 'v5',
                   'entropy_tuning': False,
                   'gamma': 1.0,
                   'gradient_clip_norm': 10.0,
                   'hidden_dim': 256,
                   'learning_starts': 2048,
                   'lr': 0.0003,
                   'method': 'qpensieve',
                   'n_env': 1,
                   'n_step': 16,
                   'network_initializer': 'legacy_torch_default',
                   'observation_calibration_steps': 2048,
                   'observation_normalization': True,
                   'preference_sampling': 'author_positive_uniform',
                   'profile': 'legacy_recipe',
                   'qp_preference_candidates': 4,
                   'qp_snapshot_capacity': 4,
                   'qp_snapshot_frequency': 2048,
                   'qp_snapshot_warmup': 4096,
                   'reward_scale': 1.0,
                   'start_steps': 2048,
                   'task_tau': 0.7,
                   'tau': 0.002,
                   'temperature_lr': 3e-05,
                   'updates_per_step': 1}}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same_path(value, expected, label):
    path = Path(value).expanduser().resolve()
    if path != expected.resolve():
        raise ValueError(label + " differs from this registered execution")
    return path


def integral(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("Invalid integral " + label)


def evaluation_grid(total):
    """Include zero, every512 interactions, and one exact fixed endpoint."""
    integral(total, "learning_steps", 1)
    result = list(range(0, total + 1, 512))
    if result[-1] != total:
        result.append(total)
    return result


SCHEDULE_SUBSTITUTIONS = (
    ("if total % 512 or 512 % ne or cfg['dev_paths'] < 1:",
     "if total % ne or 512 % ne or cfg['dev_paths'] < 1:"),
    ("evaluation_grid=list(range(0, total + 1, 512)),",
     "evaluation_grid=_formal_evaluation_grid(total),"),
    ("next_eval += 512",
     "next_eval = min(next_eval + 512, total) if next_eval < total else total + 512"),
)


def adapted_run_source(original):
    """Return only the registered native run body with three clock edits.

    Exact original file SHA and exact occurrence counts fail closed if the
    native implementation changes. No sampling, update, charging, PSA acceptance,
    replay, endpoint, or RNG-restoration instruction is changed.
    """
    if hashlib.sha256(original.encode()).hexdigest() != NATIVE_BACKEND_SHA:
        raise RuntimeError("Unrecognized native backend for the schedule adapter")
    tree = ast.parse(original)
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "run"]
    if len(functions) != 1:
        raise RuntimeError("Exactly one original native run function is required")
    function = functions[0]
    segment = ast.get_source_segment(original, function)
    adapted = segment
    for before, after in SCHEDULE_SUBSTITUTIONS:
        if adapted.count(before) != 1 or after in adapted:
            raise RuntimeError("The native schedule structure changed: " + before)
        adapted = adapted.replace(before, after)
    restored = adapted
    for before, after in reversed(SCHEDULE_SUBSTITUTIONS):
        if restored.count(after) != 1:
            raise RuntimeError("Ambiguous transformed schedule")
        restored = restored.replace(after, before)
    if (restored != segment or ast.dump(ast.parse(restored), include_attributes=False)
            != ast.dump(ast.parse(segment), include_attributes=False)):
        raise RuntimeError("An unregistered native instruction changed")
    # Preserve the original line numbers for meaningful native tracebacks.
    return "\n" * (function.lineno - 1) + adapted + "\n"


def schedule_adapter_proof(original):
    adapted = adapted_run_source(original)
    compile(adapted, str(SOURCE_ROOT / BACKEND_NAME), "exec")
    return {
        "version": "fixed-terminal-evaluation-clock-v1",
        "native_backend_sha256": NATIVE_BACKEND_SHA,
        "adapted_run_source_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
        "exact_substitutions": len(SCHEDULE_SUBSTITUTIONS),
        "original_source_file_modified": False,
        "rollout_update_replay_psa_rng_and_endpoint_instructions_changed": False,
        "terminal_tick_added_when_off_512_grid": True,
        "learning_or_psa_interactions_added": 0,
        "exact_inverse_ast_equal": True,
    }


def install_schedule_adapter(backend):
    original = (SOURCE_ROOT / BACKEND_NAME).read_text()
    adapted = adapted_run_source(original)
    backend._formal_evaluation_grid = evaluation_grid
    exec(compile(adapted, str(SOURCE_ROOT / BACKEND_NAME), "exec"), backend.__dict__)


PASTA_SUBSTITUTIONS = {
    "__init__": (
        ("if self.total_steps % self.rollout_size:",
         "if self.total_steps % (self.horizon * self.n_env):"),
    ),
    "update_rollout": (
        ("if len(self.pending_panels) < self.rollout_panels:",
         "if len(self.pending_panels) < self.rollout_panels and self.collected_steps != self.total_steps:"),
        ("if len(self.pending_panels) != self.rollout_panels:",
         "if len(self.pending_panels) != self.rollout_panels and not (self.collected_steps == self.total_steps and 0 < len(self.pending_panels) == (self.total_steps % self.rollout_size) // (self.horizon * self.n_env) < self.rollout_panels):"),
        ("if len(obs_t) != self.rollout_size:",
         "if len(obs_t) != len(self.pending_panels) * self.horizon * self.n_env:"),
        ("self.optimized_steps += self.rollout_size",
         "self.optimized_steps += len(obs_t)"),
    ),
}


def adapted_pasta_methods(original):
    """Permit one terminal short rollout of complete native episode panels.

    Only five guards/counts change. GAE, attention, decay, PCGrad, PPO losses,
    optimizer steps and64-example minibatches remain the original instructions.
    No artificial transitions, zero padding or omitted final samples are used.
    """
    if hashlib.sha256(original.encode()).hexdigest() != NATIVE_PASTA_SHA:
        raise RuntimeError("Unrecognized PASTA source for terminal-rollout adapter")
    tree = ast.parse(original)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Agent"]
    if len(classes) != 1:
        raise RuntimeError("Exactly one native PASTA Agent is required")
    result = {}
    for name, substitutions in PASTA_SUBSTITUTIONS.items():
        functions = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(functions) != 1:
            raise RuntimeError("The exact native PASTA method is required: " + name)
        node = functions[0]
        segment = textwrap.dedent(" " * node.col_offset + ast.get_source_segment(original, node))
        changed = segment
        for before, after in substitutions:
            if changed.count(before) != 1 or after in changed:
                raise RuntimeError("Native PASTA terminal structure changed: " + before)
            changed = changed.replace(before, after)
        inverse = changed
        for before, after in reversed(substitutions):
            if inverse.count(after) != 1:
                raise RuntimeError("Ambiguous PASTA terminal substitution")
            inverse = inverse.replace(after, before)
        if inverse != segment or ast.dump(ast.parse(inverse), include_attributes=False) != ast.dump(ast.parse(segment), include_attributes=False):
            raise RuntimeError("An unregistered native PASTA instruction changed")
        result[name] = "\n" * (node.lineno - 1) + changed + "\n"
        compile(result[name], str(SOURCE_ROOT / PASTA_NAME), "exec")
    return result


def pasta_adapter_proof(original):
    methods = adapted_pasta_methods(original)
    return {
        "version": "pasta-fixed-budget-terminal-short-rollout-v1",
        "native_pasta_sha256": NATIVE_PASTA_SHA,
        "adapted_method_source_sha256": {name: hashlib.sha256(source.encode()).hexdigest()
                                          for name, source in methods.items()},
        "exact_guard_or_sample_count_substitutions": 5,
        "exact_inverse_ast_equal": True,
        "complete_eight_panel_paths_equivalent": True,
        "ppo_gae_pcgrad_attention_and_decay_instructions_changed": False,
        "normalization_or_minibatch_equations_changed": False,
        "original_source_file_modified": False,
        "all_real_final_samples_optimized_without_padding": True,
        "final_batch_geometry_modified": True,
        "formal_terminal_samples": {"radio": 256, "v5": 768},
        "native_minibatch_size": 64,
    }


def install_pasta_adapter(backend):
    """Patch two class methods only within this new worker process."""
    module = backend.importlib.import_module("baselines.pasta.agent")
    originals = (SOURCE_ROOT / PASTA_NAME).read_text()
    for name, adapted in adapted_pasta_methods(originals).items():
        namespace = dict(module.__dict__)
        exec(compile(adapted, str(SOURCE_ROOT / PASTA_NAME), "exec"), namespace)
        setattr(module.Agent, name, namespace[name])


def checked_config(config_path):
    """Register fifty fresh formal baseline jobs, or explicitly disjoint engineering."""
    path = Path(config_path).resolve(strict=True)
    cfg = json.loads(path.read_text())
    contract_path = same_path(cfg["contract"], ROOT / "contract.json", "Contract")
    contract = json.loads(contract_path.read_text())
    contract_sha = sha(contract_path)
    if contract.get("schema") != SCHEMA:
        raise ValueError("The baseline-only formal retraining contract is required")
    if cfg.get("contract_sha256") != contract_sha:
        raise ValueError("Config contract SHA differs from the exact registered contract")
    same_path(contract["root"], ROOT, "Contract root")
    if not ROOT.is_relative_to("/hy-tmp"):
        raise ValueError("New scientific files must physically reside on HY")
    if contract.get("method_profiles") != BASELINE_PROFILES:
        raise ValueError("Only the five user-approved baseline profiles are registered")
    if contract.get("baseline_training_jobs") != 50:
        raise ValueError("This cohort must register fifty fresh baseline training jobs")
    if contract.get("science_seeds") != [0, 1, 2, 3, 4]:
        raise ValueError("The five formal training seeds 0 through 4 are required")
    terminal_contract = contract.get("pasta_terminal_rollout", {})
    for key, expected in {
        "profile": "complete-episodes-terminal-short-rollout-v1",
        "regular_panels": 8, "n_env": 8, "panel_episodes_complete": True,
        "terminal_panels": {"radio": 1, "v5": 4},
        "terminal_samples": {"radio": 256, "v5": 768}, "minibatch_size": 64,
    }.items():
        if terminal_contract.get(key) != expected:
            raise ValueError("The explicit fixed-budget PASTA terminal contract changed: " + key)
    if not isinstance(cfg.get("engineering"), bool):
        raise ValueError("Scientific versus engineering execution must be explicit")
    engineering = cfg["engineering"]
    scope = contract["engineering"] if engineering else contract
    if engineering and scope.get("permitted") is not True:
        raise ValueError("Engineering execution is not explicitly registered")
    method = cfg.get("method")
    if method not in BASELINE_PROFILES or cfg.get("profile") != BASELINE_PROFILES[method]:
        raise ValueError("Full, ablations, retired profiles and unregistered methods are excluded")
    domain = cfg.get("domain")
    if domain not in DOMAINS or contract.get("domains", {}).get(domain) != DOMAINS[domain]:
        raise ValueError("The unchanged native domain contract is required")
    if set(contract["domains"]) != set(DOMAINS):
        raise ValueError("Only the two registered native domains are supported")
    if cfg.get("task_tau") != DOMAINS[domain]["task_tau"]:
        raise ValueError("Task threshold differs from the registered native threshold")
    for key, expected in (("eval_every", 512), ("dev_paths", 8 if engineering else 32), ("holdout_paths", 0)):
        integral(cfg.get(key), key)
        bound = contract if key == "eval_every" else scope
        if cfg[key] != expected or bound.get(key) != expected:
            raise ValueError("The registered baseline contract differs at " + key)
    integral(cfg.get("learning_steps"), "learning_steps", 1)
    if engineering:
        if cfg["learning_steps"] not in scope["learning_steps"]:
            raise ValueError("Unregistered engineering interaction budget")
        permitted_budget = (4224 if method == "morld" else
                            4352 if method == "pasta" and domain == "radio" else
                            3840 if method == "pasta" and domain == "v5" else 3072)
        if cfg["learning_steps"] != permitted_budget and not (method == "capql" and cfg["learning_steps"] == 2688):
            raise ValueError("Only the exact registered native, PSA and final-rollout engineering budgets are allowed")
    elif cfg["learning_steps"] != 100608 or contract.get("learning_steps") != 100608:
        raise ValueError("The scientific charged interaction budget must be 100608")
    integral(cfg.get("seed"), "seed")
    approved_seeds = scope.get("seeds", scope.get("science_seeds_excluded", [])) if engineering else contract["science_seeds"]
    if cfg["seed"] not in approved_seeds or (engineering and cfg["seed"] in contract["science_seeds"]):
        raise ValueError("Unregistered scientific or disjoint engineering seed")
    for key in ("train_master", "dev_master", "test_master", "dev_rng_seed", "test_rng_seed"):
        integral(cfg.get(key), key)
        if cfg[key] != scope.get(key):
            raise ValueError("Scenario/RNG namespace differs from the contract: " + key)
    if len({cfg["train_master"], cfg["dev_master"], cfg["test_master"]}) != 3:
        raise ValueError("Training, DEV and unopened test namespaces must be distinct")
    if cfg.get("device") not in ("cpu", "cuda:0") or cfg["device"] not in contract["supported_devices"]:
        raise ValueError("Explicit registered CPU or UUID-assigned cuda:0 required; no fallback")
    if cfg.get("device") == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
        raise ValueError("CPU science may not inherit visible CUDA devices")
    expected_envs = 8 if method == "pasta" or (method == "cola" and domain == "v5") else 1
    expected_action = "raw" if method == "pasta" else "tanh_latent"
    if cfg.get("n_env") != expected_envs or cfg.get("gamma") != 1.0 or cfg.get("action_mode") != expected_action:
        raise ValueError("The native method-specific environment/action/gamma recipe changed")
    if cfg["learning_steps"] % (DOMAINS[domain]["horizon"] * expected_envs) or 512 % expected_envs:
        raise ValueError("Complete native episodes and reachable512 DEV ticks are required")
    expected_polyak = 0.002 if domain == "v5" and method in ("qpensieve", "cola") else 0.005
    if method != "pasta" and cfg.get("tau") != expected_polyak:
        raise ValueError("Polyak tau must remain distinct from the task threshold")
    for key in ("resume", "resume_from", "checkpoint", "checkpoint_path", "init_model", "pretrained_model"):
        if cfg.get(key) not in (None, False, ""):
            raise ValueError("Fresh retraining cannot load a prior model: " + key)
    if method == "morld":
        if any(key in cfg for key in ("policy_lr", "q_lr")):
            raise ValueError("Hidden optimizer aliases cannot override selected MORL/D")
        if contract.get("profiles", {}).get("critic_lr1e4") != MORLD_PROFILE:
            raise ValueError("The selected MORL/D profile contract changed")
        for key, expected in MORLD_PROFILE.items():
            if cfg.get(key) != expected:
                raise ValueError("Selected MORL/D profile changed: " + key)
        for key, expected in (("morld_psa_value_source", "charged_internal"),
                              ("charged_physical_budget_includes_internal_psa", True),
                              ("n_step", 32 if domain == "radio" else 1),
                              ("acknowledge_nstep_task_variant", domain == "radio"),
                              ("observation_normalization", False)):
            if cfg.get(key) != expected:
                raise ValueError("MORL/D charged native interaction recipe changed: " + key)
    recipe = NATIVE_RECIPES[method + "__" + domain]
    for key, expected in recipe.items():
        if cfg.get(key) != expected:
            raise ValueError("Accepted native training recipe changed: " + key)
    job_id = cfg.get("id")
    registered_id = f"{method}__{domain}__seed{cfg['seed']}"
    profile_id = f"{method}__{cfg['profile']}__{domain}__seed{cfg['seed']}"
    approved_ids = (("gate__" + registered_id + "__tail",)
                    if engineering and cfg["learning_steps"] == 2688 else
                    ("gate__" + registered_id,) if engineering else (registered_id, profile_id))
    if job_id not in approved_ids:
        raise ValueError("Job ID does not encode its registered method/domain/seed")
    same_path(cfg["output"], ROOT / ("engineering" if engineering else "jobs") / job_id, "Output")
    same_path(cfg["source_root"], SOURCE_ROOT, "Frozen source root")
    same_path(cfg["source_manifest"], SOURCE_MANIFEST, "Source manifest")
    if cfg.get("execution_worker_sha256") != sha(__file__):
        raise ValueError("The config must bind this exact execution facade SHA")
    if cfg.get("source_manifest_sha256") != contract.get("source_manifest_sha256"):
        raise ValueError("The exact registered new frozen science closure is required")
    data_root = Path(cfg["data_root"]).resolve(strict=True)
    if not data_root.is_relative_to("/hy-tmp"):
        raise ValueError("Scientific image inputs must physically reside on HY")
    if contract.get("data_root") is not None:
        same_path(cfg["data_root"], Path(contract["data_root"]), "Data root")
    for key, expected in (("optimizer_backup", False), ("replay_backup", False),
                          ("one_fixed_endpoint", True)):
        if contract.get(key) != expected:
            raise ValueError("Fixed-endpoint storage contract changed: " + key)
    return cfg, contract, contract_sha


def source_identity(cfg):
    """Verify the complete frozen closure plus the separate execution facade."""
    if cfg.get("execution_worker_sha256") != sha(__file__):
        raise RuntimeError("Execution facade changed")
    source_sha = sha(SOURCE_MANIFEST)
    if source_sha != cfg.get("source_manifest_sha256"):
        raise RuntimeError("Registered frozen source manifest changed")
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or BACKEND_NAME not in files:
        raise ValueError("The complete unchanged native backend source closure is required")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Invalid frozen source manifest entry: " + name)
        path = (SOURCE_ROOT / relative).resolve(strict=True)
        if not path.is_relative_to(SOURCE_ROOT) or sha(path) != expected:
            raise RuntimeError("Frozen native science source differs: " + name)
    if files.get(BACKEND_NAME) != NATIVE_BACKEND_SHA:
        raise RuntimeError("The approved native baseline backend changed")
    if files.get(PASTA_NAME) != NATIVE_PASTA_SHA:
        raise RuntimeError("The approved native PASTA implementation changed")
    return source_sha


def validate_only(config_path, source_manifest=None):
    cfg, contract, contract_sha = checked_config(config_path)
    if source_manifest is not None:
        same_path(source_manifest, SOURCE_MANIFEST, "CLI source manifest")
    source_sha = source_identity(cfg)
    return {
        "valid": True, "engineering": cfg["engineering"],
        "experiment_kind": "engineering_formal_baseline_preflight" if cfg["engineering"] else "formal_five_seed_baseline_training",
        "job_id": cfg["id"], "method": cfg["method"], "profile": cfg["profile"],
        "domain": cfg["domain"], "seed": cfg["seed"], "device": cfg["device"],
        "learning_steps": cfg["learning_steps"], "evaluation_points": len(evaluation_grid(cfg["learning_steps"])),
        "evaluation_grid": evaluation_grid(cfg["learning_steps"]),
        "eval_every": 512, "dev_paths": cfg["dev_paths"], "holdout_paths": 0,
        "contract_sha256": contract_sha, "source_manifest_sha256": source_sha,
        "execution_worker_sha256": sha(__file__),
        "backend": BACKEND_NAME, "native_backend_sha256": NATIVE_BACKEND_SHA,
        "evaluation_schedule_adapter": schedule_adapter_proof((SOURCE_ROOT / BACKEND_NAME).read_text()),
        "terminal_rollout_adapter": (pasta_adapter_proof((SOURCE_ROOT / PASTA_NAME).read_text())
                                     if cfg["method"] == "pasta" else None),
        "source_files_verified": len(json.loads(SOURCE_MANIFEST.read_text())["files"]),
        "environment_constructed": False, "cuda_initialized": False, "model_written": False,
    }


def run(config_path, source_manifest=None):
    """Run native learning with only the declared terminal-evaluation adapter."""
    validation = validate_only(config_path, source_manifest)
    sys.dont_write_bytecode = True
    specification = importlib.util.spec_from_file_location(
        "chapter5_formal_five_seed_baseline_native_backend", SOURCE_ROOT / BACKEND_NAME)
    backend = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(backend)
    native_write_json = backend.write_json

    def write_registered(path, payload):
        name = Path(path).name
        if name in ("protocol.json", "completed.json"):
            payload = dict(payload)
            payload.update(
                experiment_kind=validation["experiment_kind"], engineering=validation["engineering"],
                formal_training_run=not validation["engineering"], continuation_from_old_model=False,
                execution_worker_sha256=validation["execution_worker_sha256"],
                native_backend=BACKEND_NAME,
                native_backend_sha256=sha(SOURCE_ROOT / BACKEND_NAME),
                initialization="fresh_seeded_native_baseline_policy",
                loss_and_optimization_equations_modified=False,
                evaluation_schedule_adapter=validation["evaluation_schedule_adapter"],
                terminal_rollout_adapter=validation["terminal_rollout_adapter"],
                final_rollout_batch_geometry_modified=validation["method"] == "pasta",
                learning_budget_counts_training_plus_internal_psa=True,
                evaluation_grid=validation["evaluation_grid"],
                holdout_not_opened=True, holdout_paths=0,
            )
        elif name == "holdout_not_opened.json":
            payload = dict(payload, status="NOT_OPENED_ENGINEERING_PREFLIGHT" if validation["engineering"] else "NOT_OPENED_FORMAL_BASELINE_TRAINING",
                           formal_training_run=not validation["engineering"], engineering=validation["engineering"])
        return native_write_json(path, payload)

    backend.checked_config = checked_config
    backend.source_identity = source_identity
    backend.VERSION = VERSION
    backend.write_json = write_registered
    install_schedule_adapter(backend)
    if validation["method"] == "pasta":
        install_pasta_adapter(backend)
    try:
        backend.run(config_path, source_manifest)
    except BaseException:
        # Preserve the native ownership check: a rejected duplicate cannot
        # modify another invocation's output, and partial models stay unsealed.
        if backend._OWNED_OUTPUT is not None:
            if backend._UNVERIFIED_ENDPOINT is not None:
                backend._UNVERIFIED_ENDPOINT.unlink(missing_ok=True)
            write_registered(backend._OWNED_OUTPUT / "failure.json", {
                "traceback": traceback.format_exc(), "pid": os.getpid(),
                "engineering": validation["engineering"], "experiment_kind": validation["experiment_kind"],
                "execution_worker_sha256": validation["execution_worker_sha256"],
            })
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-manifest")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_only(args.config, args.source_manifest), sort_keys=True))
    else:
        run(args.config, args.source_manifest)
