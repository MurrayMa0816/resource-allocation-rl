"""Fresh native-task baseline runs and fixed-endpoint independent evaluation.

Algorithms keep their own rollout/replay and optimizer schedules. This module
only manages the shared native paths, declared budgets, checkpoint and reports.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time

import numpy as np
import torch

from ..evaluation import allocation_metrics
from ..io import (TRAIN_MASTER, TEST_MASTER, append_csv, compact_csv, digest,
                  exclusive_lock, source_hashes, write_json)
from .adapter import BaselineEnv
from .agent import (make_agent, dump_agent, load_agent, agent_parameter_vector,
                    algorithm_parameters)

METHODS = ('paspo', 'rpo', 'td7', 'simba_v2', 'dime')
SCHEMA = 'current-native-external-baselines-v2-resumable'
COMMIT_SCHEMA = 'native-baseline-checkpoint-commit-v1'
CHECKPOINT_KEEP = 2


def training_budget(method, env, *, profile, episodes):
    """Check the actual updater schedule without constructing neural networks.

    Replay sampling permits replacement, but RPO additionally waits for one
    full batch of stored transitions. TD7 updates at completed episode bounds;
    its first checkpoint phase updates after every post-warmup episode.
    """
    p = algorithm_parameters(method, env, profile=profile,
                             training_episodes=episodes)
    horizon, total = int(env.horizon), episodes * int(env.horizon)
    if method == 'paspo':
        rollout = p['n_steps']
        minimum_steps = rollout
        updates = (total // rollout) * p['epochs'] * (
            (rollout + p['batch_size'] - 1) // p['batch_size'])
    else:
        first_step = p['learning_starts'] + 1
        if method == 'rpo':
            first_step = max(first_step, p['batch_size'])
        delay = (p['policy_freq'] if method == 'td7' else
                 p.get('policy_delay', 1))
        utd = p.get('updates_per_step', 1)
        minimum_steps = first_step - 1 + (delay + utd - 1) // utd
        updates = max(0, total - first_step + 1) * utd // delay
    minimum_episodes = (minimum_steps + horizon - 1) // horizon
    result = dict(minimum_episodes=minimum_episodes,
                  scheduled_actor_updates=updates,
                  learning_starts=p.get('learning_starts', 0),
                  batch_size=p['batch_size'])
    if profile == 'formal' and episodes < minimum_episodes:
        raise ValueError(
            f'Formal {method} requires at least {minimum_episodes} complete '
            f'episodes for an actor update; received {episodes}. '
            'Set the registered training budget explicitly.')
    return result


def _policy_hash(agent):
    return hashlib.sha256(agent_parameter_vector(agent).tobytes()).hexdigest()


def _rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def _environment(config, *, evaluation=False, holdout=False, master=None):
    path_master = (config['holdout_master'] if holdout else config['path_master']) if master is None else master
    return BaselineEnv(config['domain'], config['chapter'], method=config['method'],
                       model_seed=config['seed'],
                       train_path_base=None if evaluation else path_master,
                       path_master=path_master, phase='final' if holdout else 'dev',
                       holdout=holdout, radio_groups=config['radio_groups'],
                       data_root=config.get('data_root'), condition_file=config.get('condition_file'))


def evaluate_agent(agent, env, count, *, replay=False, trace_arrays=None):
    if count <= 0:
        raise ValueError('Positive evaluation path count required')
    before, initial = _rng(), _policy_hash(agent)
    rows, totals, max_error = [], [], 0.
    try:
        for index in range(count):
            # Independent evaluation policy noise, also for DIME's genuine samples.
            stream = int(np.random.SeedSequence([env.path_master, 919, agent.seed, index])
                         .generate_state(1)[0])
            random.seed(stream)
            np.random.seed(stream)
            torch.manual_seed(stream)
            obs, _ = env.reset(seed=index)
            for _ in range(env.horizon):
                control, _ = agent.predict(obs, deterministic=agent.deployment_deterministic)
                obs, reward, done, truncated, _ = env.step(control)
                if truncated:
                    raise RuntimeError('Native evaluation unexpectedly truncated')
            if not done:
                raise RuntimeError('Native evaluation did not terminate')
            record = deepcopy(env.episode_records[-1])
            actions = np.asarray(env._actions)
            result = dict(actions=actions, total_used=float(actions.sum()),
                          remaining=float(env.decision_state['remaining']),
                          period_totals=actions.sum(1), observations=env._observations,
                          terminal_info=env._terminal_info,
                          mean_hhi=float(np.mean([np.square(a / a.sum()).sum() if a.sum() else 0.
                                                  for a in actions])))
            record.update(allocation_metrics(env.task, result))
            if replay:
                check = env.replay_last_episode()
                max_error = max(max_error, check['error'])
                record.update(check)
            if trace_arrays is not None and index < 4:
                prefix = f'path_{index}_'
                trace_arrays[prefix+'actions'] = actions.copy()
                trace_arrays[prefix+'rewards'] = np.asarray(env._rewards).copy()
                for key in env._observations[0]:
                    trace_arrays[prefix+key] = np.stack([observation[key] for observation in env._observations])
            rows.append(record)
            totals.append(result['period_totals'])
        if _policy_hash(agent) != initial:
            raise RuntimeError('Evaluation changed baseline actor parameters')
    finally:
        _restore_rng(before)
    scores = [r['score_x100'] for r in rows]
    summary = dict(score=dict(mean=float(np.mean(scores)), path_sd=float(np.std(scores)), n=count,
                              period_totals_mean=np.mean(totals, axis=0).tolist()),
                   execution=agent.metadata['deployment'], native_replays=count if replay else 0,
                   replay_max_error=max_error if replay else None)
    for key in ('total_used', 'remaining', 'budget_used_fraction', 'quality_p10', 'quality_min'):
        values = [row[key] for row in rows]
        summary['score'][key] = None if values[0] is None else float(np.mean(values))
    return summary, rows


def _save_evaluation(folder, label, episode, agent, env, count, *, replay=False):
    # An interrupted evaluation may have written only a prefix of its CSV.
    # Recreating this fixed endpoint must never append duplicate path rows.
    for suffix in ('.csv', '.csv.gz', '_summary.json'):
        (folder / f'{label}_{episode}{suffix}').unlink(missing_ok=True)
    trace_arrays = {} if replay else None
    summary, rows = evaluate_agent(agent, env, count, replay=replay, trace_arrays=trace_arrays)
    summary.update(episode=episode, phase=label, path_master=env.path_master,
                   test_split_opened=env.holdout, actor_sha256=_policy_hash(agent))
    for row in rows:
        append_csv(folder / f'{label}_{episode}.csv', row)
    compact_csv(folder / f'{label}_{episode}.csv')
    if trace_arrays:
        np.savez_compressed(folder / f'{label}_{episode}_sample_traces.npz', **trace_arrays)
    write_json(folder / f'{label}_{episode}_summary.json', summary)
    return summary


def _identity_digest(identity):
    raw = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _verify_condition(env, identity):
    """Reject changed metadata as well as changed physical values in task files."""
    expected = identity['environment'].get('data', {}).get('condition')
    task = getattr(env, 'task', None)
    actual = getattr(task, 'condition_identity', None)
    condition_file = getattr(task, 'condition_file', None)
    registered_file = identity.get('config', {}).get('condition_file')
    if expected is None and actual is None and condition_file is None and registered_file is None:
        return
    if (expected is None or actual != expected or not condition_file or not registered_file
            or digest(condition_file) != expected['config_sha256']):
        raise ValueError('Registered physical condition changed or differs from training protocol')


def _verify_data(env, identity):
    """Re-read physical and image files; cached identities alone are not a lock."""
    _verify_condition(env, identity)
    if getattr(env, 'domain', None) != 'radio':
        return
    expected = identity['environment']['data']
    for filename, key in (('images.npy', 'images_sha256'),
                          ('split_manifest.json', 'split_manifest_sha256')):
        if digest(env.task.data_root / filename) != expected[key]:
            raise ValueError(f'Registered radio data changed: {filename}')


def _checkpoint_path(folder, entry):
    relative = Path(entry['relative_path'])
    kind = entry.get('kind', 'full')
    suffix = '_evaluation' if kind == 'evaluation' else ''
    if (relative.is_absolute() or len(relative.parts) != 2
            or relative.parts[0] != '.checkpoints'
            or kind not in ('full', 'evaluation')
            or relative.name != f"episode_{entry['episode']:08d}{suffix}"):
        raise ValueError('Invalid checkpoint generation path')
    path = folder / relative
    if path.resolve().parent != (folder / '.checkpoints').resolve():
        raise ValueError('Checkpoint generation escapes this run')
    return path


def _read_commit(folder, identity):
    commit = json.loads((folder / 'checkpoint_latest.json').read_text())
    if (commit.get('schema') != COMMIT_SCHEMA
            or commit.get('identity_sha256') != _identity_digest(identity)
            or not 1 <= len(commit.get('generations', [])) <= CHECKPOINT_KEEP):
        raise ValueError('Checkpoint commit belongs to a different run protocol')
    episodes = [entry['episode'] for entry in commit['generations']]
    if (episodes != sorted(set(episodes), reverse=True)
            or commit.get('episode') != episodes[0]):
        raise ValueError('Checkpoint generations are not ordered committed endpoints')
    for entry in commit['generations']:
        _checkpoint_path(folder, entry)
    return commit


def _publish_checkpoint_alias(folder, entry):
    temporary = folder / '.checkpoint-link.tmp'
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(entry['relative_path'], target_is_directory=True)
    temporary.replace(folder / 'checkpoint')


def _prune_generations(folder, commit):
    """Remove only generated artifacts belonging to this run, after commit."""
    keep = {entry['relative_path'] for entry in commit['generations']}
    parent = folder / '.checkpoints'
    if parent.exists():
        for path in parent.iterdir():
            generated = (re.fullmatch(r'episode_[0-9]{8}(?:_evaluation)?', path.name)
                         or path.name.startswith('.recent-agent-'))
            if generated and str(path.relative_to(folder)) not in keep:
                if path.is_symlink() or path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)


def _checkpoint_reserve(agent):
    """Conservative uncompressed size including replay capacity and a margin."""
    def size(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, np.ndarray):
            return value.nbytes
        if isinstance(value, dict):
            return sum(size(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return sum(size(v) for v in value)
        return 0

    state_bytes = size(agent.core.state_dict())
    replay = getattr(agent.core, 'replay_buffer', None)
    replay_bytes = 0 if replay is None else sum(a.nbytes for a in replay.arrays)
    if replay is not None and replay.prioritized:
        replay_bytes += replay.priority.numel() * replay.priority.element_size()
    return 2 * state_bytes + replay_bytes + 64 * 1024**2


def _commit_checkpoint(folder, episode, agent, env, identity):
    if source_hashes() != identity['source_sha256']:
        raise RuntimeError('Registered runtime source changed before checkpoint')
    _verify_condition(env, identity)
    previous = (_read_commit(folder, identity)['generations']
                if (folder / 'checkpoint_latest.json').exists() else [])
    if previous and previous[0]['episode'] == episode:
        return _read_commit(folder, identity)
    parent = folder / '.checkpoints'
    parent.mkdir(exist_ok=True)
    reserve = _checkpoint_reserve(agent)
    if shutil.disk_usage(folder).free < reserve:
        raise OSError(f'Insufficient free space for atomic checkpoint; need {reserve} bytes')
    relative = f'.checkpoints/episode_{episode:08d}'
    path = folder / relative
    dump_agent(agent, path, env, identity)
    entry = dict(episode=episode, relative_path=relative,
                 manifest_sha256=digest(path / 'manifest.json'))
    entries = [entry, *[x for x in previous if x['episode'] != episode]][:CHECKPOINT_KEEP]
    commit = dict(schema=COMMIT_SCHEMA, episode=episode,
                  identity_sha256=_identity_digest(identity), generations=entries)
    # The manifest and full state exist before the single authoritative pointer
    # is replaced. If interrupted earlier, the old pointer remains restorable.
    write_json(folder / 'checkpoint_latest.json', commit)
    _publish_checkpoint_alias(folder, entry)
    _prune_generations(folder, commit)
    return commit


def _restore_committed(folder, env, identity, device, *, restore_rng=True, prune=True):
    commit = _read_commit(folder, identity)
    failures = []
    for index, entry in enumerate(commit['generations']):
        path = _checkpoint_path(folder, entry)
        try:
            if digest(path / 'manifest.json') != entry['manifest_sha256']:
                raise ValueError('Checkpoint generation manifest checksum differs')
            agent = load_agent(path, env, identity, restore_rng=restore_rng, device=device)
            if agent.num_timesteps != entry['episode'] * env.horizon:
                raise ValueError('Checkpoint endpoint differs from committed episode')
        except (OSError, ValueError, RuntimeError, KeyError, EOFError) as error:
            failures.append(dict(episode=entry['episode'], type=type(error).__name__, message=str(error)))
            continue
        if index:
            commit = dict(commit, episode=entry['episode'],
                          generations=commit['generations'][index:])
            write_json(folder / 'checkpoint_latest.json', commit)
        _publish_checkpoint_alias(folder, entry)
        if prune:
            _prune_generations(folder, commit)
        return agent, entry['episode'], failures
    raise ValueError(f'No valid committed baseline checkpoint is available: {failures}')


def _prune_uncommitted_outputs(folder, episode):
    raw, compressed = folder / 'training.csv', folder / 'training.csv.gz'
    if episode == 0:
        raw.unlink(missing_ok=True)
    else:
        source = raw if raw.exists() else compressed
        opener = gzip.open if source.suffix == '.gz' else open
        with opener(source, 'rt', newline='') as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, []
            if not fields or len(set(fields)) != len(fields) or 'episode' not in fields:
                raise ValueError('Committed training CSV schema is invalid')
            for index in range(episode):
                row = next(reader, None)
                if (row is None or set(row) != set(fields)
                        or any(value is None for value in row.values())
                        or int(row['episode']) != index + 1):
                    raise ValueError('Committed training CSV prefix is incomplete or unordered')
                rows.append(row)
            # A killed process can leave an arbitrarily torn suffix. It is
            # deliberately not parsed; only the committed prefix is durable.
        temporary = raw.with_suffix('.tmp')
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(raw)
    compressed.unlink(missing_ok=True)
    for path in folder.iterdir():
        match = re.fullmatch(r'(dev|holdout)_([0-9]+)(?:\.csv(?:\.gz)?|_summary\.json)', path.name)
        if match and (match[1] == 'holdout' or int(match[2]) > episode):
            path.unlink()
    (folder / 'holdout_lock.json').unlink(missing_ok=True)


def _prepare_final_checkpoint(folder, agent, env, identity):
    if identity['config']['final_checkpoint_mode'] == 'full':
        entry = _read_commit(folder, identity)['generations'][0]
        path = _checkpoint_path(folder, entry)
        return entry, json.loads((path / 'manifest.json').read_text())
    episode = identity['config']['episodes']
    relative = f'.checkpoints/episode_{episode:08d}_evaluation'
    path = folder / relative
    if path.exists():
        restored = load_agent(path, env, identity, restore_rng=False,
                              device=identity['config']['device'])
        if (restored.num_timesteps != episode * env.horizon
                or _policy_hash(restored) != _policy_hash(agent)):
            raise ValueError('Staged evaluation checkpoint differs from the trained endpoint')
        manifest = json.loads((path / 'manifest.json').read_text())
    else:
        reserve = _checkpoint_reserve(agent)
        if shutil.disk_usage(folder).free < reserve:
            raise OSError(f'Insufficient free space for final evaluation checkpoint; need {reserve} bytes')
        manifest = dump_agent(agent, path, env, identity, include_replay=False)
    entry = dict(episode=episode, relative_path=relative, kind='evaluation',
                 manifest_sha256=digest(path / 'manifest.json'))
    return entry, manifest


def _finalize_checkpoints(folder, identity, entry):
    """A completed run needs one final full state, with no replay duplicate."""
    commit = _read_commit(folder, identity)
    commit['episode'], commit['generations'] = entry['episode'], [entry]
    commit['complete'] = True
    write_json(folder / 'checkpoint_latest.json', commit)
    _publish_checkpoint_alias(folder, commit['generations'][0])
    _prune_generations(folder, commit)


def _config(args, every, checkpoint_every):
    keys = ('domain', 'chapter', 'method', 'seed', 'profile', 'episodes', 'path_master',
            'holdout_master', 'radio_groups', 'data_root', 'device')
    result = {key: getattr(args, key) for key in keys}
    result['condition_file'] = (str(Path(args.condition_file).expanduser().resolve())
                                if getattr(args, 'condition_file', None) else None)
    result.update(eval_every=every, checkpoint_every=checkpoint_every,
                  checkpoint_keep=CHECKPOINT_KEEP,
                  final_checkpoint_mode=getattr(args, 'final_checkpoint_mode', 'full'),
                  development_paths=args.development_paths, holdout_paths=args.holdout_paths)
    return result


def train(args):
    started = time.monotonic()
    if args.method not in METHODS or args.profile not in ('smoke', 'formal'):
        raise ValueError('Choose a registered baseline and profile')
    every = args.eval_every if args.eval_every is not None else (2 if args.profile == 'smoke' else 240)
    checkpoint_every = getattr(args, 'checkpoint_every', None) or every
    if getattr(args, 'checkpoint_keep', CHECKPOINT_KEEP) != CHECKPOINT_KEEP:
        raise ValueError('Running checkpoint retention is fixed at two complete generations')
    if (min(args.episodes, every, checkpoint_every, args.development_paths) <= 0
            or args.holdout_paths < 0):
        raise ValueError('Positive training/development counts and nonnegative holdout count required')
    rollout = 2 if args.profile == 'smoke' else 8
    if args.method == 'paspo' and any(n % rollout for n in (args.episodes, every, checkpoint_every)):
        raise ValueError(f'PASPO training, evaluation and checkpoint intervals must contain {rollout} complete episodes')
    if args.path_master == args.holdout_master:
        raise ValueError('Training and independent test namespaces must differ')
    stop_after = getattr(args, 'stop_after', None)
    if stop_after is not None and (stop_after <= 0 or stop_after > args.episodes
                                   or (args.method == 'paspo' and stop_after % rollout)):
        raise ValueError('Stop point must be a complete episode/rollout inside the registered budget')
    resume = getattr(args, 'resume', False)
    folder = Path(args.output).expanduser().resolve()
    if folder.exists() and not resume:
        raise FileExistsError('Use a new output directory, or explicitly resume this run')
    if resume and not (folder / 'protocol.json').is_file():
        raise ValueError('Resume requires an existing protocol and committed checkpoint')
    folder.mkdir(parents=True, exist_ok=resume)
    config = _config(args, every, checkpoint_every)
    if config['final_checkpoint_mode'] not in ('full', 'evaluation'):
        raise ValueError('Final checkpoint mode must be full or evaluation')
    with exclusive_lock(folder / 'run.lock'):
        env = dev = test = None
        try:
            env = _environment(config)
            budget_check = training_budget(args.method, env, profile=args.profile,
                                           episodes=args.episodes)
            dev = _environment(config, evaluation=True)
            agent = make_agent(args.method, env, args.seed, profile=args.profile,
                               training_episodes=args.episodes, device=args.device, cpu_threads=1)
            identity = dict(schema=SCHEMA, config=config, environment=env.get_contract(),
                            algorithm=agent.metadata, source_sha256=source_hashes(),
                            software=dict(python=sys.version, packages={
                                name: importlib.metadata.version(name)
                                for name in ('numpy', 'torch', 'gymnasium', 'scipy', 'Pillow')}),
                            budget_check=budget_check,
                            checkpoint_policy=dict(running_generations=CHECKPOINT_KEEP,
                                                   completed_generations=1,
                                                   final_mode=config['final_checkpoint_mode'],
                                                   commit='atomic full-state generation pointer'),
                            fresh_initialization=True, native_terminal_reward=True,
                            selection='fixed registered final training boundary; no development selection',
                            scope='engineering smoke' if args.profile == 'smoke' else 'registered experiment')
            identity = json.loads(json.dumps(identity, allow_nan=False))
            completed = 0
            if resume:
                old = json.loads((folder / 'protocol.json').read_text())
                if old != identity:
                    raise ValueError('Resume configuration, source, data, software or algorithm identity changed')
                _verify_data(env, identity)
                marker = folder / 'completed.json'
                agent, completed, fallback = _restore_committed(
                    folder, env, identity, args.device,
                    restore_rng=not marker.exists(), prune=not marker.exists())
                initial = json.loads((folder / 'initialization.json').read_text())['actor_sha256']
                if marker.exists():
                    result = json.loads(marker.read_text())
                    if (result.get('status') != 'complete' or result.get('config') != config
                            or result.get('episodes') != args.episodes
                            or completed != args.episodes or result.get('actor_sha256') != _policy_hash(agent)):
                        raise ValueError('Completed marker differs from restored final state')
                    entry, manifest = _prepare_final_checkpoint(folder, agent, env, identity)
                    if bool(manifest.get('training_resume_available', True)) != result['exact_restore_available']:
                        raise ValueError('Completed checkpoint continuation capability differs')
                    _finalize_checkpoints(folder, identity, entry)
                    return result
                _prune_uncommitted_outputs(folder, completed)
                write_json(folder / 'resumed.json', dict(from_episode=completed,
                           exact_rng_optimizer_replay_resume=True, fallback_failures=fallback))
            else:
                write_json(folder / 'protocol.json', identity)
                initial = _policy_hash(agent)
                write_json(folder / 'initialization.json', dict(fresh=True, start_episode=0, actor_sha256=initial))
                _commit_checkpoint(folder, 0, agent, env, identity)
            if completed == 0:
                _verify_condition(dev, identity)
                _save_evaluation(folder, 'dev', 0, agent, dev, args.development_paths)
                _verify_condition(dev, identity)
            if stop_after is not None and completed >= stop_after:
                if completed > stop_after:
                    raise ValueError('Requested stop point is earlier than the committed endpoint')
                return dict(status='stopped_at_checkpoint', episode=completed,
                            registered_episodes=args.episodes, holdout_opened=False)

            def record(_, row):
                append_csv(folder / 'training.csv', dict(episode=agent.num_timesteps // env.horizon, **row))

            while completed < args.episodes:
                next_dev = (completed // every + 1) * every
                next_checkpoint = (completed // checkpoint_every + 1) * checkpoint_every
                boundary = min(args.episodes, next_dev, next_checkpoint)
                if stop_after is not None and completed < stop_after:
                    boundary = min(boundary, stop_after)
                agent.learn((boundary - completed) * env.horizon, episode_callback=record)
                completed = boundary
                if completed % every == 0 or completed == args.episodes:
                    _verify_condition(dev, identity)
                    _save_evaluation(folder, 'dev', completed, agent, dev, args.development_paths)
                    _verify_condition(dev, identity)
                if completed == args.episodes:
                    if agent.accounting.get('actor_updates', 0) < 1 or _policy_hash(agent) == initial:
                        raise RuntimeError('Baseline training produced no actor updates or parameter change')
                if (completed % checkpoint_every == 0 or completed == args.episodes
                        or completed == stop_after):
                    _commit_checkpoint(folder, completed, agent, env, identity)
                write_json(folder / 'progress.json', dict(episode=completed, episodes=args.episodes,
                           accounting=agent.accounting, seconds=time.monotonic() - started,
                           health=agent.health_diagnostics(),
                           phase='stopped_at_checkpoint' if completed == stop_after else 'training'))
                if completed == stop_after:
                    return dict(status='stopped_at_checkpoint', episode=completed,
                                registered_episodes=args.episodes, holdout_opened=False)
            if source_hashes() != identity['source_sha256']:
                raise RuntimeError('Registered runtime source changed during baseline training')
            _verify_data(env, identity)
            accounting, changed = agent.accounting, _policy_hash(agent) != initial
            if accounting.get('actor_updates', 0) < 1 or not changed:
                raise RuntimeError('Baseline training produced no actor updates or parameter change')
            holdout = None
            if args.holdout_paths:
                test = _environment(config, evaluation=True, holdout=True)
                _verify_condition(test, identity)
                for key in ('images_sha256', 'split_manifest_sha256'):
                    if test.task.data_identity.get(key) != env.task.data_identity.get(key):
                        raise ValueError('Independent evaluation data differs from training archive')
                write_json(folder / 'holdout_lock.json', dict(episode=completed,
                           actor_sha256=_policy_hash(agent), count=args.holdout_paths,
                           path_master=config['holdout_master']))
                holdout = _save_evaluation(folder, 'holdout', args.episodes, agent, test,
                                           args.holdout_paths, replay=True)
            if source_hashes() != identity['source_sha256']:
                raise RuntimeError('Registered runtime source changed during baseline evaluation')
            _verify_data(env, identity)
            final_entry, final_manifest = _prepare_final_checkpoint(folder, agent, env, identity)
            resume_available = bool(final_manifest.get('training_resume_available', True))
            replay_retained = (_checkpoint_path(folder, final_entry) / 'replay.npz').is_file()
            compact_csv(folder / 'training.csv')
            result = dict(status='complete', schema=SCHEMA, config=config,
                          episodes=args.episodes, fresh_original_training=True,
                          actor_changed=changed, actor_sha256=_policy_hash(agent),
                          accounting=accounting,
                          holdout_score=None if holdout is None else holdout['score'],
                          replay_max_error=None if holdout is None else holdout['replay_max_error'],
                          test_split_opened=holdout is not None,
                          checkpoint='checkpoint', full_checkpoint='checkpoint' if resume_available else None,
                          checkpoint_commit='checkpoint_latest.json',
                          final_checkpoint_mode=config['final_checkpoint_mode'],
                          checkpoint_retention=dict(running=CHECKPOINT_KEEP, completed=1),
                          exact_restore_available=resume_available,
                          training_resume_available=resume_available, replay_retained=replay_retained,
                          optimizer_moments_retained=bool(final_manifest['optimizer_moments_retained']),
                          seconds=time.monotonic() - started, scope=identity['scope'])
            write_json(folder / 'progress.json', dict(episode=completed, episodes=args.episodes,
                       accounting=accounting, health=agent.health_diagnostics(),
                       seconds=time.monotonic() - started, phase='complete'))
            write_json(folder / 'completed.json', result)
            _finalize_checkpoints(folder, identity, final_entry)
            return result
        except BaseException as error:
            write_json(folder / 'failure.json', dict(type=type(error).__name__, message=str(error)))
            raise
        finally:
            for current in (env, dev, test):
                if current is not None:
                    current.close()


def evaluate(args):
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    identity = manifest['identity']
    if identity.get('schema') != SCHEMA:
        raise ValueError('Use a checkpoint from the native baseline runner')
    if identity.get('source_sha256') != source_hashes():
        raise ValueError('Checkpoint runtime differs from the registered baseline source')
    config = identity['config']
    checkpoint_episode = manifest['accounting']['complete_episodes']
    if checkpoint_episode != config['episodes']:
        raise ValueError('Independent test evaluation requires the registered final checkpoint')
    master = config['holdout_master'] if args.path_master is None else args.path_master
    if master == config['path_master']:
        raise ValueError('Independent test namespace must differ from training')
    training_env = _environment(config)
    env = None
    try:
        _verify_data(training_env, identity)
        agent = load_agent(checkpoint, training_env, identity, restore_rng=False, device=args.device)
        if agent.num_timesteps != checkpoint_episode * training_env.horizon:
            raise ValueError('Checkpoint state differs from its registered evaluation endpoint')
        _verify_data(training_env, identity)
        env = _environment(config, evaluation=True, holdout=True, master=master)
        _verify_condition(env, identity)
        folder = Path(args.output).expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / 'evaluation_protocol.json', dict(schema=SCHEMA, checkpoint=str(checkpoint),
                   checkpoint_episode=checkpoint_episode, registered_episodes=config['episodes'],
                   source_identity=identity, path_master=master, paths=args.paths))
        summary = _save_evaluation(folder, 'evaluation', checkpoint_episode, agent, env,
                                   args.paths, replay=True)
        _verify_data(training_env, identity)
        write_json(folder / 'completed.json', dict(status='complete', **summary))
        return summary
    finally:
        training_env.close()
        if env is not None:
            env.close()


def inspect(args):
    env = BaselineEnv(args.domain, args.chapter, path_master=args.path_master,
                      radio_groups=args.radio_groups, data_root=args.data_root,
                      condition_file=getattr(args, 'condition_file', None))
    try:
        return dict(methods=list(METHODS), environment=env.get_contract(),
                    training='Each algorithm retains its own on/off-policy updater',
                    inference='Registered method rule; DIME remains sampled diffusion')
    finally:
        env.close()


def parser():
    root = argparse.ArgumentParser(description='External baselines on the current native tasks')
    commands = root.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'train'):
        p = commands.add_parser(name)
        p.add_argument('--domain', choices=('v5', 'radio'), required=True)
        p.add_argument('--chapter', type=int, choices=(2, 3), required=True)
        p.add_argument('--path-master', type=int, default=TRAIN_MASTER)
        p.add_argument('--radio-groups', type=int, default=100)
        p.add_argument('--data-root')
        p.add_argument('--condition-file', help='Registered physical condition JSON')
        if name == 'train':
            p.add_argument('--method', choices=METHODS, required=True)
            p.add_argument('--seed', type=int, default=0)
            p.add_argument('--profile', choices=('smoke', 'formal'), default='smoke')
            p.add_argument('--episodes', type=int, default=4)
            p.add_argument('--eval-every', type=int)
            p.add_argument('--checkpoint-every', type=int)
            p.add_argument('--checkpoint-keep', type=int, choices=(CHECKPOINT_KEEP,), default=CHECKPOINT_KEEP)
            p.add_argument('--final-checkpoint-mode', choices=('full', 'evaluation'), default='full')
            p.add_argument('--development-paths', type=int, default=1)
            p.add_argument('--holdout-paths', type=int, default=1)
            p.add_argument('--holdout-master', type=int, default=TEST_MASTER)
            p.add_argument('--device', default='cpu')
            p.add_argument('--output', required=True)
            p.add_argument('--resume', action='store_true')
            p.add_argument('--stop-after', type=int, help='Engineering stop at a complete episode/rollout; resume keeps the registered budget')
    p = commands.add_parser('evaluate')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--paths', type=int, default=128)
    p.add_argument('--path-master', type=int)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    result = {'inspect': inspect, 'train': train, 'evaluate': evaluate}[args.command](args)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return result
