"""Fresh registered Chapter5 Full: CPU collection and persistent CUDA updates.

The immutable scientific modules remain untouched. Collection, dense evaluation
and endpoint deployment use CPU models; native updates use persistent CUDA
models and Adam state, followed by complete synchronous inference-weight copies.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import time
import traceback
import sys

import numpy as np
import torch

# Bind the copied whole-budget Task/Actor BEFORE importing any legacy learner.
from . import bridge
bridge.bootstrap()
bridge.import_legacy('chapter5_cross_server_20261103')
from allocation23.learning import PPO
from allocation_morl.common import (
    atomic_torch_save, cpu_copy, file_hash, model_hash, optimizer_steps,
    state_hash, summary, weight_key,
)
from .bridge import make_models, make_pool
from allocation_morl.training import actual_kl, jsonable
from allocation_morl.initialization import initialization_record, validate_checkpoint_initialization
from . import common
from .rollout import rollout, result_row

WEIGHT = .8
LEARNING_STEPS = 61440
EVAL_EVERY = 512
PROPOSED_VARIANTS = ('full', 'only_j', 'only_c', 'scalar_value')


def save_trace(path, results):
    """Store original feasible actions, pre-guard decoder actions and latent events."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ('actions', 'raw_actions', 'latent_actions', 'quality')
    arrays = {key: np.stack([row[key] for row in results]) for key in names}
    arrays['native_rewards'] = np.stack([row['rewards'] for row in results])
    arrays['rewards'] = np.zeros(arrays['native_rewards'].shape+(2,), dtype=np.float64)
    for index, row in enumerate(results):
        native = arrays['native_rewards'][index]
        if np.any(native[:-1] != 0.) or abs(native[-1]-row['quality'].mean()) > 1e-10:
            raise AssertionError('Trace vector labels require a true native terminal result')
        arrays['rewards'][index, -1] = row['C'], row['native_return']
    for key in ('quota_lower','quota_upper','remaining_before','remaining_after','caps','period_totals'):
        arrays[key] = np.stack([row[key] for row in results])
    arrays['preference'] = np.array([row['preference'] for row in results])
    arrays['case_plan_json'] = np.array([
        json.dumps(row['plan'], sort_keys=True) for row in results])
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    return dict(path=str(path), sha256=file_hash(path),
                arrays={key: list(value.shape) for key, value in arrays.items()})


@contextmanager
def preserved_random_state(generator=None, rng=None):
    """Evaluation never advances a learning or process-global random stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    cuda_state = [state.clone() for state in torch.cuda.get_rng_state_all()]
    generator_state = generator.get_state().clone() if generator is not None else None
    rng_state = copy.deepcopy(rng.bit_generator.state) if rng is not None else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(cuda_state)
        if generator_state is not None:
            generator.set_state(generator_state)
        if rng_state is not None:
            rng.bit_generator.state = rng_state


def random_fingerprint(generator, rng):
    return state_hash(dict(python=random.getstate(), numpy=np.random.get_state(),
        torch=torch.get_rng_state(), generator=generator.get_state(),
        learner_numpy=copy.deepcopy(rng.bit_generator.state)))


def training_audit(actor, critic, ao, vo, learner, generator, rng, *, update_actor, update_critic):
    def module_runtime(module):
        return state_hash(dict(
            gradients={name: parameter.grad for name, parameter in module.named_parameters()},
            requires_grad={name: parameter.requires_grad for name, parameter in module.named_parameters()},
            training_modes={name: child.training for name, child in module.named_modules()}))
    result = dict(actor=model_hash(actor), critic=model_hash(critic),
        actor_runtime=module_runtime(actor), critic_runtime=module_runtime(critic),
        update_actor=model_hash(update_actor), update_critic=model_hash(update_critic),
        update_actor_runtime=module_runtime(update_actor), update_critic_runtime=module_runtime(update_critic),
        actor_optimizer=state_hash(ao.state_dict()),
        critic_optimizer=state_hash(vo.state_dict()),
        learner=state_hash(learner.state_dict()),
        random_streams=random_fingerprint(generator, rng),
        cuda_rng=state_hash(torch.cuda.get_rng_state_all()))
    if result['actor'] != result['update_actor'] or result['critic'] != result['update_critic']:
        raise AssertionError('CPU inference models differ from the current CUDA update weights')
    return result


def sync_inference_models(actor, critic, update_actor, update_critic):
    """Copy every parameter/buffer; keep Adam/history on the CUDA learner."""
    then = time.monotonic()
    actor.load_state_dict(update_actor.state_dict(), strict=True)
    critic.load_state_dict(update_critic.state_dict(), strict=True)
    torch.cuda.synchronize()
    return time.monotonic()-then


def gpu_proof(update_actor, update_critic, ao, vo, update_events_ms):
    devices = sorted({str(parameter.device) for model in (update_actor, update_critic)
                      for parameter in model.parameters()})
    if devices != ['cuda:0']:
        raise AssertionError('Amended neural update parameters are not on cuda:0')
    moment_devices = sorted({str(value.device) for optimizer in (ao, vo)
        for state in optimizer.state.values() for key, value in state.items()
        if torch.is_tensor(value) and key in ('exp_avg', 'exp_avg_sq')})
    if moment_devices != ['cuda:0'] or update_events_ms <= 0.:
        raise AssertionError('Persistent Adam moments left the CUDA device')
    return dict(actual_device='cuda:0', gpu_name=torch.cuda.get_device_name(0),
        torch=torch.__version__, cuda=torch.version.cuda,
        model_parameter_devices=devices, optimizer_moment_devices=moment_devices,
        collector_device='cpu', evaluation_device='cpu', collection_generator_device='cpu',
        dtype='float64', gpu_update_event_milliseconds=update_events_ms,
        cuda_training_verified=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved())


def deterministic_assess(pool, actor, critic, count, phase, *, tau,
                         trace_prefix=None, eval_seed=common.DEV_RNG_SEED,
                         step_callback=None):
    """Use latent Gaussian means on fixed cases, with no exploration or selection."""
    if count <= 0 or not pool:
        raise ValueError('Nonempty evaluation pool and positive case count required')
    generator = torch.Generator(device='cpu').manual_seed(eval_seed)
    rows = []
    # This local isolation also protects callers outside train(), including reload
    # verification. No eval()/train() calls change the models' execution modes.
    with preserved_random_state():
        for first in range(0, count, len(pool)):
            results = rollout(pool[:min(len(pool), count-first)], actor, critic,
                generator, phase, first, tau=tau, preferences=WEIGHT,
                deterministic=True, training=False, step_callback=step_callback)
            trace = save_trace(f'{trace_prefix}_{first:04d}.npz', results) if trace_prefix else None
            rows.extend(result_row(row, first+j, trace=trace, trace_index=j)
                        for j, row in enumerate(results))
    aggregate = summary(rows)
    # Keep native score_x100 for compatibility and an explicitly raw J for plots.
    aggregate['mean_J'] = aggregate['score'] / 100.
    aggregate['mean_F_0_8'] = WEIGHT*aggregate['mean_C'] + (1-WEIGHT)*aggregate['mean_J']
    return {weight_key(WEIGHT): dict(weight=WEIGHT,
        execution='deterministic latent Gaussian mean then native feasible decoder',
        deterministic=dict(**aggregate, rows=rows))}


class DenseEvaluationClock:
    """Assess exact ticks; complete-rollout ticks wait for their due update."""
    def __init__(self, interval, budget):
        if interval <= 0 or budget <= 0 or budget % interval:
            raise ValueError('Positive budget must contain complete assessment intervals')
        self.interval, self.budget = int(interval), int(budget)
        self.last_step = 0
        self.pending = None
        self.evaluated = [0]

    def observe(self, step, *, complete, assess, collector_fingerprint):
        if step <= self.last_step or step > self.budget:
            raise AssertionError('Learning-step clock must increase within its budget')
        if step // self.interval != self.last_step // self.interval:
            if step % self.interval:
                raise AssertionError('A vector collection tick skipped an exact evaluation boundary')
            if self.pending is not None:
                raise AssertionError('A complete-rollout evaluation was not flushed after its update')
            if complete:
                self.pending = step
            else:
                before = collector_fingerprint()
                assess(step, timing='during_rollout', collector=before)
                if collector_fingerprint() != before:
                    raise AssertionError('Assessment changed the active F1 collection state')
                self.evaluated.append(step)
        self.last_step = step

    def after_update(self, assess):
        if self.pending is not None:
            assess(self.pending, timing='after_due_update', collector=None)
            self.evaluated.append(self.pending)
            self.pending = None


def _args_valid(args):
    horizon = common.DOMAINS[args.domain]['horizon']
    if args.tau is None:
        args.tau = common.DOMAINS[args.domain]['tau']
    if not np.isfinite(args.tau) or not 0 <= args.tau <= 1:
        raise ValueError('The registered hard threshold must be finite and in [0,1]')
    if args.learning_steps <= 0 or args.learning_steps % (8*horizon):
        raise ValueError('Budget must comprise complete eight-episode native collections')
    if args.eval_every <= 0 or args.eval_every % 8 or args.learning_steps % args.eval_every:
        raise ValueError('Evaluation interval must be a positive multiple of 8 dividing the budget')
    if args.dev_paths <= 0 or args.holdout_paths < 0 or args.eval_batch <= 0:
        raise ValueError('Invalid evaluation case counts')
    if args.profile == 'formal':
        if (args.learning_steps != LEARNING_STEPS or args.eval_every != EVAL_EVERY
                or args.dev_paths != 32 or args.holdout_paths != 256
                or args.seed not in common.SEEDS):
            raise ValueError('Formal worker differs from the registered scientific budget')
        common.validate_worker(args, 'proposed')
    if args.device != 'cuda:0':
        raise ValueError('Use logical cuda:0; the scheduler assigns the physical GPU')
    if args.eval_modes != 'deterministic' or args.diagnostics_every < 0:
        raise ValueError('Deterministic evaluation and nonnegative diagnostics interval required')
    return horizon, args.learning_steps // horizon


def train(args):
    """Train a single independent fresh worker; never spawn a formal queue."""
    Learner = bridge.import_legacy('chapter5_validation_20261104.variants').Learner
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    horizon, episodes = _args_valid(args)
    execution = (common.validate_worker(args, 'proposed')['execution']
                 if args.profile == 'formal' else dict(
                     actual_device=args.device, execution_package='chapter5_full_spend_20261104',
                     config_sha256=common.sha(args.config_json) if args.config_json else None,
                     cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    config = common.read_json(args.config_json) if args.config_json else {'variant': args.variant, 'tau': args.tau}
    if config != {'variant': args.variant, 'tau': args.tau}:
        raise ValueError('Proposed worker accepts only the registered variant and tau; no silent algorithm overrides')
    if not torch.cuda.is_available():
        raise ValueError('This worker requires a functioning CUDA device')
    torch.cuda.init()
    torch.cuda.set_device(torch.device(args.device))
    folder = Path(args.output)
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    start = time.monotonic()
    pools = []
    counts = {key: dict(episodes=0, physical_steps=0)
              for key in ('train', 'development', 'holdout', 'reload_verification')}
    def counter(name):
        def increment(done):
            counts[name]['physical_steps'] += 1
            counts[name]['episodes'] += int(done)
        return increment
    try:
        random.seed(830000+args.seed)
        torch.manual_seed(840000+args.seed)
        rng = np.random.default_rng(820000+args.seed)
        generator = torch.Generator(device='cpu').manual_seed(810000+args.seed)
        pool = make_pool(args.domain, args.seed, 8, args.path_master)
        pools.extend(pool)
        devpool = make_pool(args.domain, args.seed, min(args.eval_batch, args.dev_paths), args.dev_master)
        pools.extend(devpool)
        if {id(task.env) for task in pool} & {id(task.env) for task in devpool}:
            raise AssertionError('Learning and assessment pools share an environment')
        spec = pool[0].spec
        if spec.horizon != horizon:
            raise AssertionError('Unexpected native horizon')
        tau = args.tau
        actor, critic, neural_spec = make_models(spec, args.seed, 'fixed')
        update_actor, update_critic = copy.deepcopy(actor).to('cuda:0'), copy.deepcopy(critic).to('cuda:0')
        ao = torch.optim.Adam(update_actor.parameters(), lr=PPO['actor_lr'])
        vo = torch.optim.Adam(update_critic.parameters(), lr=PPO['value_lr'])
        learner = Learner(neural_spec, update_actor.event_dim, args.seed, 'cpu',
            method='fixed', weight=WEIGHT, tau=tau, variant=args.variant)
        # Device adaptation is confined to this explicit amended execution
        # profile. Every native objective, batch and F1 control remains intact.
        learner.device = 'cuda:0'
        def parameter_audit():
            return training_audit(actor, critic, ao, vo, learner, generator, rng,
                update_actor=update_actor, update_critic=update_critic)
        torch.cuda.reset_peak_memory_stats()
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_update_event_ms = 0.
        hashes = common.source_manifest()
        initial_actor, initial_critic = model_hash(actor), model_hash(critic)
        protocol = dict(schema='chapter5-full-spend-proposed-dense-v1',
            profile=args.profile, domain=args.domain, variant=args.variant, seed=args.seed,
            learning_steps=args.learning_steps, episodes=episodes, horizon=horizon,
            evaluation_interval_physical_steps=args.eval_every, evaluation_ticks=list(range(
                0, args.learning_steps+1, args.eval_every)), dev_paths=args.dev_paths,
            holdout_paths=args.holdout_paths, train_master=args.path_master,
            dev_master=args.dev_master, test_master=args.test_master, dev_rng_seed=args.dev_rng_seed,
            test_rng_seed=args.test_rng_seed, tau=tau, actor_preference=WEIGHT, config=config,
            evaluation='deterministic latent Gaussian mean; fixed shared development cases',
            evaluation_timing='exact sum of physical train transitions across all lanes; complete-rollout tick follows its due update',
            training_pool='native fresh8 until32 episodes, then fresh4 plus historical4; no trajectory splitting',
            spec=spec.to_dict(), neural_spec=neural_spec.to_dict(),
            physical=pool[0].physical_dict, data=pool[0].data_identity,
            native_ppo=PPO, actor=actor.metadata(), implementation=learner.metadata(),
            initialization=dict(**initialization_record(actor),
                actor_sha256=initial_actor, critic_sha256=initial_critic),
            checkpoint=dict(policy='one fixed endpoint', model_selection=False,
                optimizer_backup=False, replay_backup=False, resume_supported=False),
            traces=dict(actions='actual native feasible actions', raw_actions='decoded physical actions before public guard',
                latent_actions='complete raw joint Gaussian event eta,z',
                native_rewards='unchanged native terminal-only J used by the original collector',
                rewards='logging-only terminal (C,J) vector, zeros before true done; training batches unchanged',
                quality='native terminal individual vector',
                public_execution_ledger='per step [L,U], remaining_before, period_total, remaining_after and per-object caps; no future private inputs',
                case_plan='full plan JSON plus hashes in every episode row'),
            full_spend_required=True,full_spend_bridge=bridge.protocol(),
            source_sha256=hashes, device='cuda_update/cpu_inference', threads=1, dtype='float64',
            update_execution='persistent nativeCUDAactor/critic/Adam; complete synchronousCPUinference mirrors after everyupdate',
            cpu_constructor_guard='fresh nativeLearner onCPU then explicit CUDA update device; scientific equations unchanged',
            cpu_cuda_bitwise_equivalence_claim=False)
        protocol.update(execution)
        common.write_json(folder/'protocol.json', protocol)
        clock = DenseEvaluationClock(args.eval_every, args.learning_steps)
        collection_seconds = update_seconds = evaluation_seconds = 0.
        last_update_statistics = {}
        diagnostics_events = 0
        diagnostics_path = folder/'diagnostics.jsonl'
        def diagnostic(step, evaluation, timing):
            # These are already-materialized scalars from the accepted native
            # update and ordinary development call. No forward/backward pass,
            # sampler, action guard or extra environment call is introduced.
            nonlocal diagnostics_events
            if not args.diagnostics_every or step % args.diagnostics_every:
                return
            keys = ('actor_loss', 'coverage_value_loss', 'quality_value_loss',
                    'value_loss', 'actual_joint_kl', 'tv', 'actor_lr_before',
                    'actor_lr_after', 'ess_fraction', 'history_only_ess_fraction',
                    'optimizer_steps', 'actor_committed_optimizer_steps',
                    'critic_committed_optimizer_steps', 'replay_episodes_read',
                    'fresh_C', 'fresh_J', 'update_device', 'learning_step')
            row = dict(schema='chapter5-full-spend-proposed-readonly-diagnostic-v1',
                learning_steps=step, stage=timing, policy_updates=learner.updates,
                completed_training_episodes=learner.episodes,
                native_statistics={key: last_update_statistics[key] for key in keys
                                   if key in last_update_statistics},
                development={key: evaluation[weight_key(WEIGHT)]['deterministic'][key]
                             for key in ('mean_K', 'mean_C', 'mean_J', 'mean_F_0_8')},
                additional_environment_steps=0, additional_optimizer_steps=0,
                changes_to_actions_objective_or_rng=False)
            with diagnostics_path.open('a') as handle:
                handle.write(json.dumps(jsonable(row), allow_nan=False)+'\n')
            diagnostics_events += 1
        def progress(phase):
            common.write_json(folder/'progress.json', dict(status=phase, pid=os.getpid(),
                variant=args.variant, domain=args.domain, seed=args.seed, counts=counts,
                learning_steps=counts['train']['physical_steps'], target_learning_steps=args.learning_steps,
                seconds=time.monotonic()-start))
            common.write_json(folder/'gpu_runtime.json', dict(schema='ch5-gpu-runtime-v1',
                pid=os.getpid(), actual_device='cuda:0', phase=phase, unix_time=time.time(),
                memory_allocated_bytes=torch.cuda.memory_allocated(),
                memory_reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                learning_steps=counts['train']['physical_steps']))
        def development(step, *, timing, collector=None):
            nonlocal evaluation_seconds
            progress('development')
            before = parameter_audit()
            then = time.monotonic()
            with preserved_random_state(generator, rng):
                evaluation = deterministic_assess(devpool, actor, critic, args.dev_paths, 'dev',
                    tau=tau, trace_prefix=folder/'traces'/f'development_{step:06d}',
                    eval_seed=args.dev_rng_seed, step_callback=counter('development'))
            evaluation_seconds += time.monotonic()-then
            after = parameter_audit()
            if before != after:
                raise AssertionError('Assessment changed model, optimizer, F1 history or a training RNG')
            value = dict(learning_step=step, checkpoint_episode=learner.episodes,
                policy_update=learner.updates, timing=timing, preferences=evaluation,
                parameter_audit=dict(unchanged=True, before=before, after=after),
                collector_sha256=collector, counts=copy.deepcopy(counts), model_selection=False)
            common.write_json(folder/f'evaluation_{step:06d}.json', value)
            diagnostic(step, evaluation, timing)
            print(json.dumps(dict(learning_step=step, domain=args.domain, variant=args.variant,
                seed=args.seed, evaluation=evaluation[weight_key(WEIGHT)]['deterministic']['mean_J'],
                seconds=time.monotonic()-start)), flush=True)
        development(0, timing='initial')
        with (folder/'training.jsonl').open('w') as ft, (folder/'updates.jsonl').open('w') as fu, \
                (folder/'training_ledger.jsonl').open('w') as fl:
            first = behavior_version = 0
            while first < episodes:
                batch_size = learner.collection_size(actor, max_new=min(8, episodes-first))
                if batch_size not in (4, 8):
                    raise AssertionError('Incomplete native fresh collection requested')
                def after_vector_step(*, complete, collector_fingerprint):
                    clock.observe(counts['train']['physical_steps'], complete=complete,
                        assess=development, collector_fingerprint=collector_fingerprint)
                progress('training')
                then = time.monotonic()
                evaluation_before = evaluation_seconds
                results = rollout(pool[:batch_size], actor, critic, generator, 'train', first,
                    tau=tau, preferences=np.full(batch_size, WEIGHT), behavior_version=behavior_version,
                    step_callback=counter('train'), after_vector_step=after_vector_step)
                collection = time.monotonic()-then-(evaluation_seconds-evaluation_before)
                collection_seconds += collection
                batches = [row['batch'] for row in results]
                before = copy.deepcopy(update_actor)
                old_steps = optimizer_steps(ao), optimizer_steps(vo)
                then = time.monotonic()
                gpu_start.record()
                stats = learner.update(update_actor, update_critic, ao, vo, batches, rng)
                gpu_end.record()
                gpu_end.synchronize()
                event_ms = gpu_start.elapsed_time(gpu_end)
                gpu_update_event_ms += event_ms
                weight_sync_seconds = sync_inference_models(actor, critic, update_actor, update_critic)
                elapsed = time.monotonic()-then
                update_seconds += elapsed
                stats.update(actor_committed_optimizer_steps=optimizer_steps(ao)-old_steps[0],
                    critic_committed_optimizer_steps=optimizer_steps(vo)-old_steps[1],
                    episode=first+batch_size, learning_step=counts['train']['physical_steps'],
                    new_episodes=batch_size, behavior_version=behavior_version,
                    collection_seconds=collection, update_seconds=elapsed,
                    measured_outer_actor_kl=actual_kl(before, update_actor, batches), counts=copy.deepcopy(counts),
                    update_device='cuda:0', weights_sync_seconds=weight_sync_seconds,
                    gpu_update_event_milliseconds=event_ms, CPU_inference_mirrors_synchronized=True)
                last_update_statistics = stats
                trace = save_trace(folder/'traces'/f'training_{first:04d}.npz', results)
                for j, result in enumerate(results):
                    row = result_row(result, first+j, trace=trace, trace_index=j)
                    row['episode'] = first+j+1
                    ft.write(json.dumps(jsonable(row), allow_nan=False)+'\n')
                    ledger=dict(row,phase='train',schema='chapter5-full-spend-training-episode-v1',
                                model_id=args.variant,variant=args.variant,domain=args.domain,seed=args.seed,tau=tau)
                    fl.write(json.dumps(jsonable(ledger), allow_nan=False)+'\n')
                fu.write(json.dumps(jsonable(stats), allow_nan=False)+'\n')
                ft.flush()
                fu.flush()
                fl.flush()
                first += batch_size
                behavior_version += 1
                clock.after_update(development)
        expected_ticks = list(range(0, args.learning_steps+1, args.eval_every))
        if clock.evaluated != expected_ticks or clock.pending is not None:
            raise AssertionError('Dense evaluation schedule is incomplete')
        if counts['train'] != dict(episodes=episodes, physical_steps=args.learning_steps):
            raise AssertionError('Training budget accounting differs from the registered endpoint')
        final_state = parameter_audit()
        endpoint_actor, endpoint_critic = model_hash(actor), model_hash(critic)
        endpoint = folder/'checkpoint_endpoint.pt'
        atomic_torch_save(endpoint, dict(format='chapter5-full-spend-fixed-endpoint-v1', episode=episodes,
            learning_step=args.learning_steps, actor=cpu_copy(actor.state_dict()),
            critic=cpu_copy(critic.state_dict()), actor_sha256=endpoint_actor,
            critic_sha256=endpoint_critic, protocol=protocol,
            optimizer_backup=False, replay_backup=False, resume_supported=False))
        # Reload into independent fresh objects. Construction can seed/allocate,
        # so preserve every training stream even though learning is finished.
        with preserved_random_state(generator, rng):
            loaded = torch.load(endpoint, map_location='cpu', weights_only=False)
            validate_checkpoint_initialization(loaded, seed=args.seed)
            reload_actor, reload_critic, _ = make_models(spec, args.seed, 'fixed')
            reload_actor.load_state_dict(loaded['actor'], strict=True)
            reload_critic.load_state_dict(loaded['critic'], strict=True)
            if (model_hash(reload_actor), model_hash(reload_critic)) != (endpoint_actor, endpoint_critic):
                raise AssertionError('Endpoint state failed strict reload verification')
            reload_evaluation = deterministic_assess(devpool, reload_actor, reload_critic,
                args.dev_paths, 'dev', tau=tau, eval_seed=args.dev_rng_seed,
                trace_prefix=folder/'traces'/'reload_endpoint',
                step_callback=counter('reload_verification'))
        final_dev = json.loads((folder/f'evaluation_{args.learning_steps:06d}.json').read_text())
        def without_trace(value):
            value = copy.deepcopy(value)
            for arm in value.values():
                for row in arm['deterministic']['rows']:
                    for key in ('trace_file', 'trace_sha256', 'trace_index'):
                        row.pop(key, None)
            return value
        if state_hash(without_trace(final_dev['preferences'])) != state_hash(without_trace(reload_evaluation)):
            raise AssertionError('Reloaded endpoint does not reproduce the final development evaluation')
        common.write_json(folder/'endpoint_reload_verification.json', dict(status='pass',
            actor_sha256=endpoint_actor, critic_sha256=endpoint_critic,
            final_evaluation_reproduced=True, full_spend_required=True,
            preferences=reload_evaluation, counts=counts['reload_verification']))
        if args.holdout_paths:
            progress('independent_evaluation')
            testpool = make_pool(args.domain, args.seed, min(args.eval_batch, args.holdout_paths),
                args.test_master, holdout=True)
            pools.extend(testpool)
            before = parameter_audit()
            then = time.monotonic()
            with preserved_random_state(generator, rng):
                evaluation = deterministic_assess(testpool, actor, critic, args.holdout_paths,
                    'final', tau=tau, trace_prefix=folder/'traces'/'holdout_endpoint',
                    eval_seed=args.test_rng_seed, step_callback=counter('holdout'))
            evaluation_seconds += time.monotonic()-then
            after = parameter_audit()
            if before != after:
                raise AssertionError('Independent evaluation changed learning state')
            common.write_json(folder/'holdout.json', dict(endpoint=dict(
                learning_step=args.learning_steps, checkpoint_episode=episodes,
                checkpoint_actor_sha256=endpoint_actor, checkpoint_critic_sha256=endpoint_critic,
                parameter_audit=dict(unchanged=True, before=before, after=after),
                preferences=evaluation), model_selection=False, holdout_selection=False))
        final_sources = common.source_manifest()
        if final_sources != hashes:
            common.write_json(folder/'runtime_source_change.json', dict(
                before=hashes, after=final_sources,
                changed=[name for name in sorted(set(hashes) | set(final_sources))
                    if hashes.get(name) != final_sources.get(name)]))
            raise AssertionError('Runtime source changed while worker was running')
        if parameter_audit() != final_state:
            raise AssertionError('Endpoint/reload/test changed the final learning state')
        completed = dict(status='complete', profile=args.profile,
            formal_training=args.profile == 'formal', variant=args.variant,
            full_spend_required=True,execution_ledger_verified=True,
            domain=args.domain, seed=args.seed, tau=tau, episodes=episodes,
            train_physical_steps=args.learning_steps, counts=counts,
            total_physical_steps=sum(row['physical_steps'] for row in counts.values()),
            evaluation_ticks=clock.evaluated, evaluation_points=len(clock.evaluated),
            training_state_sha256=state_hash(final_state), final_training_audit=final_state,
            seconds=time.monotonic()-start, training_collection_seconds=collection_seconds,
            training_update_seconds=update_seconds, evaluation_seconds=evaluation_seconds,
            endpoint_actor_sha256=endpoint_actor, endpoint_critic_sha256=endpoint_critic,
            endpoint_sha256=file_hash(endpoint), endpoint_retained=True,
            endpoint_reload_verified=True, source_unchanged=True, saved_model_count=1,
            training_ledger_sha256=file_hash(folder/'training_ledger.jsonl'),
            actor_changed=endpoint_actor != initial_actor, critic_changed=endpoint_critic != initial_critic,
            model_selection=False, holdout_selection=False, fresh_training=True,
            optimizer_backup=False, replay_backup=False, resume_supported=False)
        completed.update(execution, device='cuda_update/cpu_inference',
            gpu_proof=gpu_proof(update_actor, update_critic, ao, vo, gpu_update_event_ms),
            gpu_execution_source_unchanged=True, learning_steps=args.learning_steps,
            diagnostics_interval=args.diagnostics_every, diagnostics_events=diagnostics_events,
            diagnostics_sha256=file_hash(diagnostics_path) if diagnostics_events else None,
            diagnostic_scope='existing native update/development scalars only; no extra forward, action or RNG draws')
        progress('complete')
        common.write_json(folder/'completed.json', completed)
        return completed
    except BaseException:
        common.write_json(folder/'failure.json', dict(status='failed',
            traceback=traceback.format_exc(), counts=counts, seconds=time.monotonic()-start,
            resume_supported=False, restart='same registered seed from zero in a new attempt directory'))
        raise
    finally:
        for task in pools:
            task.env.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--variant', choices=PROPOSED_VARIANTS, required=True)
    result.add_argument('--domain', choices=tuple(common.DOMAINS), required=True)
    result.add_argument('--seed', type=int, required=True)
    result.add_argument('--tau', type=float)
    result.add_argument('--learning-steps', type=int, default=LEARNING_STEPS)
    result.add_argument('--eval-every', type=int, default=EVAL_EVERY)
    result.add_argument('--dev-paths', type=int, default=32)
    result.add_argument('--holdout-paths', type=int, default=256)
    result.add_argument('--eval-batch', type=int, default=8)
    result.add_argument('--output', required=True)
    result.add_argument('--profile', choices=('engineering', 'formal'), default='engineering')
    result.add_argument('--config-json')
    result.add_argument('--run-protocol')
    result.add_argument('--job-id')
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--path-master', type=int, default=common.TRAIN_MASTER)
    result.add_argument('--dev-master', type=int, default=common.DEV_MASTER)
    result.add_argument('--test-master', type=int, default=common.TEST_MASTER)
    result.add_argument('--dev-rng-seed', type=int, default=common.DEV_RNG_SEED)
    result.add_argument('--test-rng-seed', type=int, default=common.TEST_RNG_SEED)
    result.add_argument('--eval-modes', default='deterministic')
    result.add_argument('--diagnostics-every', type=int, default=512)
    return result


def main():
    train(parser().parse_args())


if __name__ == '__main__':
    main()
