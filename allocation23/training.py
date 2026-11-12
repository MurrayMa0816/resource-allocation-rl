"""Fresh training, exact checkpoint resume, and fixed-endpoint evaluation."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

from . import __version__
from .evaluation import evaluate
from .io import (SCHEMA, append_csv, compact_csv, digest, exclusive_lock, prune_csv, source_hashes,
                 state_hash, write_json)
from .learning import PPO, update
from .methods import make_actor, make_value
from .task import Task

CONFIG_KEYS = ('domain', 'chapter', 'method', 'seed', 'path_master', 'holdout_master',
               'radio_groups', 'variant', 'temporal', 'condition_file', 'quota_reference')


def configure_runtime():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def make_task(args, *, holdout=False):
    return Task(args.domain, args.chapter, args.method, args.seed,
                path_master=args.holdout_master if holdout else args.path_master,
                radio_groups=args.radio_groups, data_root=args.data_root, holdout=holdout,
                condition_file=getattr(args, 'condition_file', None))


def _verify_condition(task, identity):
    """Verify both the loaded task and the current bytes of its physical file."""
    expected = identity['data'].get('condition')
    actual = getattr(task, 'condition_identity', None)
    condition_file = getattr(task, 'condition_file', None)
    registered_file = identity['config'].get('condition_file')
    if expected is None and actual is None and condition_file is None and registered_file is None:
        return
    if (expected is None or actual != expected or not condition_file or not registered_file
            or digest(condition_file) != expected['config_sha256']):
        raise ValueError('Registered physical condition changed or differs from training protocol')


def fresh(task, args):
    random.seed(830000+args.seed)
    torch.manual_seed(840000+args.seed)
    actor = make_actor(task.spec, args.method, args.chapter, args.seed,
                       variant=args.variant, temporal=args.temporal,
                       quota_reference=getattr(args, 'quota_reference', 'pace'))
    critic = make_value(task.spec, args.seed)
    return (actor, critic, torch.optim.Adam(actor.parameters(), lr=PPO['actor_lr']),
            torch.optim.Adam(critic.parameters(), lr=PPO['value_lr']),
            torch.Generator().manual_seed(810000+args.seed), np.random.default_rng(820000+args.seed))


def save_checkpoint(folder, state, episode, identity, steps):
    actor, critic, actor_optimizer, value_optimizer, generator, rng = state
    payload = dict(schema=SCHEMA, episode=episode, identity=identity, optimizer_steps=steps,
                   actor=actor.state_dict(), critic=critic.state_dict(),
                   actor_optimizer=actor_optimizer.state_dict(), value_optimizer=value_optimizer.state_dict(),
                   generator=generator.get_state(), numpy_rng=rng.bit_generator.state,
                   torch_rng=torch.get_rng_state(), python_rng=random.getstate())
    temporary = folder/'checkpoint_latest.tmp'
    torch.save(payload, temporary)
    if (folder/'checkpoint_latest.pt').exists():
        previous = folder/'checkpoint_previous.tmp'
        previous.unlink(missing_ok=True)
        os.link(folder/'checkpoint_latest.pt', previous)
        previous.replace(folder/'checkpoint_previous.pt')
    temporary.replace(folder/'checkpoint_latest.pt')
    if episode in (960, 1920, identity['episodes']):
        target = folder/f'checkpoint_{episode}.pt'
        temporary = target.with_suffix('.tmp')
        temporary.unlink(missing_ok=True)
        os.link(folder/'checkpoint_latest.pt', temporary)
        temporary.replace(target)


def restore(state, payload):
    actor, critic, actor_optimizer, value_optimizer, generator, rng = state
    actor.load_state_dict(payload['actor'], strict=True)
    critic.load_state_dict(payload['critic'], strict=True)
    actor_optimizer.load_state_dict(payload['actor_optimizer'])
    value_optimizer.load_state_dict(payload['value_optimizer'])
    generator.set_state(payload['generator'])
    rng.bit_generator.state = payload['numpy_rng']
    torch.set_rng_state(payload['torch_rng'])
    random.setstate(payload['python_rng'])


def train(args):
    configure_runtime()
    if min(args.episodes, args.eval_every, args.development_paths) <= 0 or args.holdout_paths < 0:
        raise ValueError('Positive training/development counts and nonnegative test count required')
    batch = PPO['batch_episodes']
    checkpoint_every = getattr(args, 'checkpoint_every', None) or args.eval_every
    if checkpoint_every <= 0 or any(value % batch for value in (args.episodes, args.eval_every, checkpoint_every)):
        raise ValueError(f'Training and evaluation intervals must be multiples of {batch}')
    if args.path_master == args.holdout_master:
        raise ValueError('Training and independent test namespaces must differ')
    stop_after = getattr(args, 'stop_after', None)
    if stop_after is not None and (stop_after <= 0 or stop_after > args.episodes or stop_after % batch):
        raise ValueError('Stop point must be a positive whole batch within the training budget')
    folder = Path(args.output).expanduser().resolve()
    if folder.exists() and not args.resume:
        raise ValueError('Use a new output directory, or explicitly resume this run')
    if args.resume and not (folder/'protocol.json').is_file():
        raise ValueError('Resume requires an existing release protocol and checkpoint')
    folder.mkdir(parents=True, exist_ok=args.resume)
    if getattr(args, 'condition_file', None):
        args.condition_file = str(Path(args.condition_file).expanduser().resolve())
    with exclusive_lock(folder/'run.lock'):
        started = time.monotonic()
        try:
            task = make_task(args)
            state = fresh(task, args)
            actor, critic, actor_optimizer, value_optimizer, generator, rng = state
            config = {key: getattr(args, key, None) for key in CONFIG_KEYS}
            config['quota_reference'] = getattr(args, 'quota_reference', 'pace')
            identity = dict(schema=SCHEMA, release_version=__version__, config=config,
                            spec=task.spec.to_dict(), physical=task.physical_dict, data=task.data_identity,
                            reward_version=task.reward_version, ppo=PPO, actor=actor.metadata(),
                            source_sha256=source_hashes(), episodes=args.episodes, eval_every=args.eval_every,
                            checkpoint_every=checkpoint_every,
                            development_paths=args.development_paths, holdout_paths=args.holdout_paths,
                            feedback='native terminal return; gamma=1 Monte Carlo',
                            evaluation_rule='latent Gaussian mean followed by the same feasible decoder',
                            selection='fixed final checkpoint; independent test is never used for selection',
                            fresh_initialization=True,
                            software=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__))
            identity = json.loads(json.dumps(identity, allow_nan=False))
            _verify_condition(task, identity)
            first = steps = 0
            initial = state_hash(actor)
            if args.resume:
                old = json.loads((folder/'protocol.json').read_text())
                assert old == identity, 'Resume configuration, source, data, or software identity changed'
                payload = torch.load(folder/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
                assert payload['schema'] == SCHEMA and payload['identity'] == identity
                first, steps = payload['episode'], payload['optimizer_steps']
                assert first <= args.episodes
                restore(state, payload)
                initial = json.loads((folder/'initialization.json').read_text())['actor_sha256']
                for name in ('training.csv', 'updates.csv'):
                    prune_csv(folder/name, first)
                write_json(folder/'resumed.json', dict(from_episode=first, exact_rng_optimizer_resume=True))
            else:
                write_json(folder/'protocol.json', identity)
                write_json(folder/'initialization.json', dict(fresh=True, start_episode=0,
                           actor_sha256=initial, critic_sha256=state_hash(critic),
                           actor_optimizer_entries=len(actor_optimizer.state),
                           value_optimizer_entries=len(value_optimizer.state)))
                save_checkpoint(folder, state, 0, identity, steps)

            def progress(episode, phase):
                write_json(folder/'progress.json', dict(episode=episode, episodes=args.episodes, phase=phase,
                           optimizer_steps=steps, seconds=time.monotonic()-started, pid=os.getpid()))

            if first == 0:
                progress(0, 'development')
                evaluate(folder, task, actor, critic, 0, 'dev', args.development_paths)
                _verify_condition(task, identity)
            batches = []
            progress(first, 'training')
            for episode in range(first+1, args.episodes+1):
                result = task.episode('train', episode-1, actor, critic, generator, training=True)
                append_csv(folder/'training.csv', dict(episode=episode, score_x100=100*result['native_return'],
                           total_used=result['total_used'], remaining=result['remaining'], mean_hhi=result['mean_hhi'],
                           task_seed=result['plan']['task_seed'], path_sha256=result['path_sha256']))
                batches.append(result['batch'])
                if len(batches) == batch:
                    stats = update(actor, critic, actor_optimizer, value_optimizer, batches, rng)
                    steps += stats['optimizer_steps']
                    batches = []
                    append_csv(folder/'updates.csv', dict(episode=episode, **stats))
                if episode % 32 == 0:
                    progress(episode, 'training')
                if episode % args.eval_every == 0 or episode == args.episodes:
                    assert source_hashes() == identity['source_sha256'], 'Runtime sources changed during training'
                    _verify_condition(task, identity)
                    saved_generator = generator.get_state().clone()
                    saved_rng = copy.deepcopy(rng.bit_generator.state)
                    progress(episode, 'development')
                    evaluate(folder, task, actor, critic, episode, 'dev', args.development_paths)
                    _verify_condition(task, identity)
                    assert torch.equal(saved_generator, generator.get_state()) and saved_rng == rng.bit_generator.state
                    save_checkpoint(folder, state, episode, identity, steps)
                    print(json.dumps(dict(domain=args.domain, chapter=args.chapter, method=args.method,
                                         seed=args.seed, episode=episode)), flush=True)
                elif episode % checkpoint_every == 0:
                    assert source_hashes() == identity['source_sha256'], 'Runtime sources changed during training'
                    _verify_condition(task, identity)
                    save_checkpoint(folder, state, episode, identity, steps)
                if stop_after is not None and episode == stop_after:
                    assert not batches
                    _verify_condition(task, identity)
                    save_checkpoint(folder, state, episode, identity, steps)
                    progress(episode, 'stopped_at_checkpoint')
                    return
            assert not batches
            test = None
            if args.holdout_paths:
                progress(args.episodes, 'holdout')
                test_task = make_task(args, holdout=True)
                _verify_condition(test_task, identity)
                for key in ('images_sha256', 'split_manifest_sha256'):
                    assert test_task.data_identity.get(key) == identity['data'].get(key), 'Data changed during training'
                write_json(folder/'holdout_lock.json', dict(episode=args.episodes, actor_sha256=state_hash(actor),
                           path_master=args.holdout_master, count=args.holdout_paths))
                test = evaluate(folder, test_task, actor, critic, args.episodes, 'holdout', args.holdout_paths,
                                replay=True, traces=min(4, args.holdout_paths))
            assert source_hashes() == identity['source_sha256'], 'Runtime sources changed during evaluation'
            _verify_condition(task, identity)
            if task.domain == 'radio':
                assert digest(task.data_root/'images.npy') == identity['data']['images_sha256'], 'Image data changed'
            for name in ('training.csv', 'updates.csv'):
                compact_csv(folder/name)
            progress(args.episodes, 'complete')
            write_json(folder/'completed.json', dict(status='complete', config=config, episodes=args.episodes,
                       start_episode=first, optimizer_steps=steps, source_unchanged=True,
                       actor_changed=state_hash(actor) != initial, fresh_original_training=True,
                       holdout_score=None if test is None else test['score'],
                       replay_max_error=0. if test is None else test['replay_max_error'],
                       seconds=time.monotonic()-started))
        except BaseException:
            write_json(folder/'failure.json', dict(traceback=traceback.format_exc(), seconds=time.monotonic()-started))
            raise


def evaluate_checkpoint(args):
    configure_runtime()
    if args.paths <= 0:
        raise ValueError('A positive test path count is required')
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('schema') != SCHEMA:
        raise ValueError('Use a checkpoint created by this release')
    identity = payload['identity']
    assert identity['source_sha256'] == source_hashes(), 'Checkpoint runtime differs from this release'
    config = identity['config']
    master = config['holdout_master'] if args.path_master is None else args.path_master
    if master == config['path_master']:
        raise ValueError('Independent evaluation must use a separate path namespace')
    task = Task(config['domain'], config['chapter'], config['method'], config['seed'],
                path_master=master, radio_groups=config['radio_groups'], data_root=args.data_root, holdout=True,
                condition_file=config.get('condition_file'))
    _verify_condition(task, identity)
    assert task.physical_dict == identity['physical'], 'Evaluation physical condition differs from training'
    for key in ('images_sha256', 'split_manifest_sha256'):
        assert task.data_identity.get(key) == identity['data'].get(key), 'Evaluation data differs from training archive'
    actor = make_actor(task.spec, config['method'], config['chapter'], config['seed'],
                       variant=config['variant'], temporal=config['temporal'],
                       quota_reference=config.get('quota_reference', 'pace'))
    critic = make_value(task.spec, config['seed'])
    actor.load_state_dict(payload['actor'], strict=True)
    critic.load_state_dict(payload['critic'], strict=True)
    folder = Path(args.output).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/'evaluation_protocol.json', dict(schema=SCHEMA, config=config,
               checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
               episode=payload['episode'], paths=args.paths, path_master=master, data=task.data_identity))
    summary = evaluate(folder, task, actor, critic, payload['episode'], 'evaluation', args.paths,
                       replay=True, traces=min(4, args.paths))
    _verify_condition(task, identity)
    write_json(folder/'completed.json', dict(status='complete', config=config, episodes=payload['episode'],
               score=summary['score'], native_replays=summary['native_replays'],
               replay_max_error=summary['replay_max_error']))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
