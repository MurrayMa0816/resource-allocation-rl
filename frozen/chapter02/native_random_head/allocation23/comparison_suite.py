"""Recorded, resource-aware A/B and external-baseline comparison queue.

All methods share native paths, training interactions and fixed evaluation
boundaries. Algorithm-specific optimizer schedules remain algorithm settings.
The dry run writes a resource plan and never starts training.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import psutil

from .image_data import DATA_ROOT
from .io import PROJECT_ROOT, digest, exclusive_lock, source_hashes, write_json
from .suite import terminate_workers

SCHEMA = 'allocation23-comparison-suite-v1'
METHODS = ('A', 'B', 'paspo', 'rpo', 'td7', 'simba_v2', 'dime')
MIB = 1024 ** 2


def cpu_quota():
    """Respect cpuset/affinity and both separate and combined cgroup v1 CPU."""
    limits = [float(os.cpu_count() or 1)]
    try:
        limits.append(float(len(os.sched_getaffinity(0))))
    except (AttributeError, OSError):
        pass
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            limits.append(int(quota) / int(period))
    except (OSError, ValueError):
        pass
    for folder in ('cpu', 'cpu,cpuacct'):
        base = Path('/sys/fs/cgroup') / folder
        try:
            quota = int((base / 'cpu.cfs_quota_us').read_text())
            period = int((base / 'cpu.cfs_period_us').read_text())
            if quota > 0:
                limits.append(quota / period)
        except (OSError, ValueError):
            pass
    return min(limits)


def memory_available():
    available = psutil.virtual_memory().available
    for folder, limit_name, usage_name in (
        ('/sys/fs/cgroup', 'memory.max', 'memory.current'),
        ('/sys/fs/cgroup/memory', 'memory.limit_in_bytes', 'memory.usage_in_bytes'),
    ):
        try:
            limit = (Path(folder) / limit_name).read_text().strip()
            usage = int((Path(folder) / usage_name).read_text())
            if limit != 'max':
                available = min(available, max(0, int(limit) - usage))
        except (OSError, ValueError):
            pass
    return available


def validate_plan(plan):
    if plan.get('schema') != SCHEMA:
        raise ValueError('Expected the registered comparison-suite schema')
    for key, allowed in (('domains', ('v5', 'radio')), ('chapters', (2, 3)),
                         ('methods', METHODS)):
        values = plan.get(key, [])
        if not values or len(values) != len(set(values)) or any(v not in allowed for v in values):
            raise ValueError('Invalid or duplicate comparison ' + key)
    seeds = plan.get('seeds', [])
    if (not seeds or len(seeds) != len(set(seeds)) or
            any(type(v) is not int or v < 0 for v in seeds)):
        raise ValueError('Nonnegative, distinct training seeds required')
    for key in ('episodes', 'eval_every', 'development_paths', 'radio_groups',
                'checkpoint_every'):
        if type(plan.get(key)) is not int or plan[key] <= 0:
            raise ValueError('Positive integer ' + key + ' required')
    if type(plan.get('holdout_paths')) is not int or plan['holdout_paths'] < 0:
        raise ValueError('Nonnegative independent-test count required')
    if plan.get('baseline_profile') not in ('formal', 'smoke'):
        raise ValueError('Declare the baseline network profile')
    if plan.get('checkpoint_keep') != 2:
        raise ValueError('Keep two committed in-progress generations')
    if plan.get('final_checkpoint_mode','full') not in ('full','evaluation'):
        raise ValueError('Declare full or evaluation final checkpoint retention')
    for key in ('path_master', 'holdout_master'):
        if type(plan.get(key)) is not int or plan[key] < 0:
            raise ValueError('Nonnegative namespace required: ' + key)
    if plan['path_master'] == plan['holdout_master']:
        raise ValueError('Training and test namespaces must differ')
    if plan.get('device', 'cpu') != 'cpu':
        raise ValueError('This registered comparison uses CPU for all methods')
    if any(plan[key] % 8 for key in ('episodes', 'eval_every', 'checkpoint_every')):
        raise ValueError('Comparison boundaries must contain complete eight-episode PPO rollouts')
    if plan.get('purpose') == 'formal_comparison' and not plan['holdout_paths']:
        raise ValueError('A formal comparison requires independent final testing')
    if plan.get('purpose') == 'engineering_preflight' and plan['holdout_paths']:
        raise ValueError('Engineering preflight must leave the holdout split closed')
    if plan.get('storage_reserve_bytes', 512*MIB) < 0 or plan.get('reserve_cpu', 2) < 0:
        raise ValueError('Resource reserves must be nonnegative')
    return plan


def registered_jobs(plan):
    validate_plan(plan)
    return [dict(domain=domain, chapter=chapter, method=method, seed=seed,
                 name=f'{domain}__chapter{chapter}__{method}__seed{seed}')
            for domain, chapter, method, seed in itertools.product(
                plan['domains'], plan['chapters'], plan['methods'], plan['seeds'])]


def job_command(plan, job, destination, *, data_root=None, resume=False):
    command = [sys.executable, '-m', 'allocation23']
    if job['method'] not in ('A', 'B'):
        command.append('baseline')
    command.append('train')
    for field in ('domain', 'chapter', 'method', 'seed'):
        command.extend(['--' + field.replace('_', '-'), str(job[field])])
    for field in ('episodes', 'eval_every', 'development_paths', 'holdout_paths',
                  'path_master', 'holdout_master', 'radio_groups'):
        command.extend(['--' + field.replace('_', '-'), str(plan[field])])
    if job['method'] not in ('A', 'B'):
        for flag, value in (('profile', plan['baseline_profile']),
                            ('checkpoint-every', plan['checkpoint_every']),
                            ('checkpoint-keep', plan['checkpoint_keep']),
                            ('final-checkpoint-mode',plan.get('final_checkpoint_mode','full')),
                            ('device', plan.get('device', 'cpu'))):
            command.extend(['--' + flag, str(value)])
    if data_root is not None:
        command.extend(['--data-root', str(Path(data_root).resolve())])
    command.extend(['--output', str(destination)])
    if resume:
        command.append('--resume')
    return command


def _tensor_bytes(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _optimizer_moment_bytes(core):
    import torch
    optimizers = []
    for value in vars(core).values():
        if isinstance(value, torch.optim.Optimizer):
            optimizers.append(value)
        elif isinstance(value, dict):
            optimizers.extend(v for v in value.values() if isinstance(v, torch.optim.Optimizer))
    # Every registered optimizer is Adam. Parameters shared by different
    # optimizers require different moments and deliberately count twice.
    return sum(2 * p.numel() * p.element_size() + 8
               for optimizer in optimizers for group in optimizer.param_groups
               for p in group['params'])


def _cell_storage(plan, domain, chapter, method):
    """Conservative full-state capacity; do not assume replay compresses well."""
    import torch
    torch.set_num_threads(1)
    from .task import Task
    if method in ('A', 'B'):
        from .methods import make_actor, make_value
        task = Task(domain, chapter, method, 0, radio_groups=plan['radio_groups'],
                    **({'condition_file': plan['condition_file']} if plan.get('condition_file') else {}))
        try:
            actor = make_actor(task.spec, method, chapter, 0,
                               **({k:plan[k] for k in ('variant','temporal','quota_reference')
                                   if plan.get(k) is not None}))
            critic = make_value(task.spec, 0)
            parameters = sum(p.numel()*p.element_size()
                             for module in (actor, critic) for p in module.parameters())
            model = _tensor_bytes(actor.state_dict()) + _tensor_bytes(critic.state_dict())
            full = model + 2*parameters + 2*MIB
        finally:
            task.env.close()
        return dict(model_optimizer_bytes=full, replay_bytes=0,
                    checkpoint_bytes=full,completed_checkpoint_bytes=full, completed_copies=1,
                    completed_optimizer_moments_retained=True,
                    worker_memory_bytes=256*MIB + 8*full)
    from .baselines.adapter import BaselineEnv
    from .baselines.agent import RecentChapter2Agent
    env = BaselineEnv(domain, chapter, train_path_base=plan['path_master'],
                      path_master=plan['path_master'], radio_groups=plan['radio_groups'],
                      **({'condition_file': plan['condition_file']} if plan.get('condition_file') else {}))
    try:
        agent = RecentChapter2Agent(method, env, 0, profile=plan['baseline_profile'],
                                   training_episodes=plan['episodes'], initialize=False, cpu_threads=1)
        core = agent.core
        while 'delegate' in vars(core):
            core = core.delegate
        base_model = _tensor_bytes(core.state_dict()) + 2*MIB
        moments = _optimizer_moment_bytes(core)
        model = base_model + moments
        replay = getattr(core, 'replay_buffer', None)
        replay_bytes = (sum(a.nbytes for a in replay.arrays) if replay is not None else 0)
        if replay is not None and replay.prioritized:
            replay_bytes += replay.priority.numel() * replay.priority.element_size()
        # Native state/actions and JSON records are small relative to replay;
        # serialization/ZIP headers and per-episode records get 10% headroom.
        full = math.ceil(1.1 * (model + replay_bytes))
        evaluation=plan.get('final_checkpoint_mode','full')=='evaluation'
        keep_moments=not(evaluation and replay is not None)
        completed = (math.ceil(1.1*(model if keep_moments else base_model))
                     if evaluation else full)
        return dict(model_optimizer_bytes=model, replay_bytes=replay_bytes,
                    checkpoint_bytes=full,completed_checkpoint_bytes=completed,completed_copies=1,
                    completed_optimizer_moments_retained=keep_moments,
                    worker_memory_bytes=512*MIB + 5*model + 2*replay_bytes)
    finally:
        env.close()
        gc.collect()


def storage_plan(plan, jobs, max_workers, *, cell_estimator=None):
    estimator = _cell_storage if cell_estimator is None else cell_estimator
    cells = {}
    for job in jobs:
        key = (job['domain'], job['chapter'], job['method'])
        if key not in cells:
            cells[key] = estimator(plan, *key)
    rows = []
    for job in jobs:
        estimate = cells[(job['domain'], job['chapter'], job['method'])]
        checkpoint = estimate['checkpoint_bytes']
        completed_checkpoint=estimate.get('completed_checkpoint_bytes',checkpoint)
        # Completed jobs keep the registered final state. At an in-progress commit,
        # latest + previous + atomic temporary need at most three states.
        rows.append(dict(name=job['name'], **estimate,
                         final_bytes=completed_checkpoint + 8*MIB,
                         extra_active_peak_bytes=3*checkpoint-completed_checkpoint))
    workers = max(1, min(max_workers, len(jobs),
                         math.floor(max(1., cpu_quota()-plan.get('reserve_cpu', 2)))))
    active_peak = sum(sorted((r['extra_active_peak_bytes'] for r in rows), reverse=True)[:workers])
    final = sum(r['final_bytes'] for r in rows)
    reserve = plan.get('storage_reserve_bytes', 512*MIB)
    return dict(assumption='uncompressed full running replay, model and Adam, 10% baseline serialization headroom; '
                'registered final retention, two committed active full generations plus atomic temporary',
                final_checkpoint_mode=plan.get('final_checkpoint_mode','full'),
                jobs=rows, job_count=len(jobs), max_effective_workers=workers,
                final_bytes=final, active_temporary_peak_bytes=active_peak,
                reserve_bytes=reserve, required_bytes=final+active_peak+reserve)


def _existing_bytes(path):
    # st_blocks reports allocated blocks. Count hardlinks once within the run.
    seen, total = set(), 0
    if not path.exists():
        return 0
    for current in path.rglob('*'):
        if current.is_file() and not current.is_symlink():
            stat = current.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity not in seen:
                seen.add(identity)
                total += stat.st_blocks * 512
    return total


def capacity_report(root, estimate):
    root = Path(root).resolve()
    existing = root
    while not existing.exists():
        existing = existing.parent
    disk = shutil.disk_usage(existing)
    already = _existing_bytes(root)
    additional = max(0, estimate['required_bytes']-already)
    return dict(path=str(root), filesystem_device=existing.stat().st_dev,
                free_bytes=disk.free, existing_run_bytes=already,
                required_additional_bytes=additional,
                capacity_sufficient=disk.free >= additional,
                estimate=estimate, cpu_quota=cpu_quota(),
                available_memory_bytes=memory_available())


def _filesystem_details(base):
    existing=base
    while not existing.exists():existing=existing.parent
    mounts=[]
    for mount in psutil.disk_partitions(all=True):
        mountpoint=Path(mount.mountpoint)
        if existing==mountpoint or mountpoint in existing.parents:
            mounts.append(mount)
    if mounts:
        mount=max(mounts,key=lambda m:len(m.mountpoint))
        if mount.fstype in ('tmpfs','ramfs','devtmpfs','proc','sysfs','cgroup','cgroup2'):
            raise ValueError('RAM/virtual filesystem is not a persistent result store')
        if 'ro' in mount.opts.split(','):
            raise ValueError('Read-only filesystem cannot store experiment outputs')
    if not os.access(existing,os.W_OK):
        raise ValueError('Experiment output store is not writable')
    return existing.stat().st_dev,shutil.disk_usage(existing).free


def placement_plan(root, extra_roots, estimate, max_workers, *, registered=None,
                   existing_credits=None):
    """Lock per-job persistent locations; never count one filesystem twice.

    The report/protocol tree stays under root. Extra persistent filesystems
    host individual job directories through permanent relative/absolute
    symlinks. RAM filesystems are rejected for final research artifacts.
    """
    root=Path(root).resolve()
    tag=root.name+'_'+hashlib.sha256(str(root).encode()).hexdigest()[:12]
    bases=[root]+[Path(p).expanduser().resolve()/tag for p in extra_roots]
    filesystems={}
    for base in bases:
        device,free=_filesystem_details(base)
        if device not in filesystems:
            filesystems[device]=dict(device=device,base=str(base),free_bytes=free,jobs=[])
        else:
            filesystems[device]['free_bytes']=min(filesystems[device]['free_bytes'],free)
    lookup={row['name']:row for row in estimate['jobs']}
    allocated={device:0 for device in filesystems}
    final_totals={device:0 for device in filesystems}
    credit_totals={device:0 for device in filesystems}
    resources={}
    placements={}
    if registered is not None and set(registered)!=set(lookup):
        raise ValueError('Registered storage placement does not cover exactly these jobs')
    for row in sorted(estimate['jobs'],key=lambda r:-r['final_bytes']):
        name=row['name']
        if registered is None:
            device=max(filesystems,key=lambda d:filesystems[d]['free_bytes']-allocated[d])
            target=Path(filesystems[device]['base'])/'jobs'/name
        else:
            target=Path(registered[name]).resolve()
            matching=[d for d,x in filesystems.items()
                      if target.parent==Path(x['base'])/'jobs']
            if len(matching)!=1:
                raise ValueError('Registered job destination is outside its persistent stores')
            device=matching[0]
        placements[name]=str(target)
        filesystems[device]['jobs'].append(name)
        credit=0 if existing_credits is None else existing_credits.get(name,0)
        if credit<0:
            raise ValueError('Existing-output credit cannot be negative')
        # Current committed states are already charged to disk.free. The base
        # must retain max(existing, future-final), while only the missing part
        # of a three-full-state active peak needs new allocation. Subtracting
        # all existing states from a global final estimate would be unsafe for
        # several paused jobs whose two full generations exceed their compact
        # final models.
        additional_final=max(0,row['final_bytes']-credit)
        peak=row['final_bytes']+row['extra_active_peak_bytes']
        additional_peak=max(0,peak-max(row['final_bytes'],credit))
        resources[name]=dict(existing_credited_bytes=credit,
                             final_bytes=additional_final,
                             extra_active_peak_bytes=additional_peak)
        allocated[device]+=additional_final
        final_totals[device]+=row['final_bytes']
        credit_totals[device]+=credit
    available_sum=sum(x['free_bytes'] for x in filesystems.values())
    chosen_rows=[]
    maximum=max(1,min(max_workers,estimate['max_effective_workers']))
    for workers in range(maximum,0,-1):
        rows=[]
        for device,entry in filesystems.items():
            count=max(1,math.ceil(workers*entry['free_bytes']/max(1,available_sum)))
            peak=sum(sorted((resources[n]['extra_active_peak_bytes'] for n in entry['jobs']),reverse=True)[:count])
            reserve=estimate['reserve_bytes'] if lookup else 0
            required=allocated[device]+peak+reserve
            rows.append(dict(entry,worker_slots=count,final_bytes=final_totals[device],
                             existing_credited_bytes=credit_totals[device],
                             final_additional_bytes=allocated[device],
                             active_temporary_peak_bytes=peak,
                             reserve_bytes=reserve,required_bytes=required,
                             capacity_sufficient=entry['free_bytes']>=required))
        chosen_rows=rows
        if all(x['capacity_sufficient'] for x in rows):break
    return dict(path=str(root),placements=placements,
                storage_roots=[str(Path(p).expanduser().resolve()) for p in extra_roots],
                filesystems=chosen_rows,unique_filesystems=len(filesystems),
                free_bytes=available_sum,required_additional_bytes=sum(x['required_bytes'] for x in chosen_rows),
                capacity_sufficient=all(x['capacity_sufficient'] for x in chosen_rows),
                storage_worker_limit=workers,estimate=estimate,cpu_quota=cpu_quota(),
                remaining_job_resources=resources,
                available_memory_bytes=memory_available())


def suite_identity(plan, jobs, data_root, *, placements=None, storage_roots=None):
    data = {}
    if 'radio' in plan['domains']:
        data = {name+'_sha256': digest(data_root/name) for name in
                ('images.npy', 'split_manifest.json')}
        if (data_root/'image_records.json').is_file():
            data['image_records.json_sha256']=digest(data_root/'image_records.json')
    result=dict(schema=SCHEMA, plan=plan, jobs=jobs,
                source_sha256=source_hashes(), data_sha256=data,
                software={name: importlib.metadata.version(name) for name in
                          ('torch', 'numpy', 'gymnasium', 'scipy')},
                python=sys.version)
    if placements is not None:
        result.update(storage_placement=placements,storage_roots=storage_roots or [])
    return result


def _assert_job_placement(folder,job,identity):
    placement=identity.get('storage_placement',{}).get(job['name'])
    if placement is not None and folder.resolve()!=Path(placement):
        raise ValueError('Job directory differs from its registered storage placement')


def _validate_job_protocol(folder,job,plan,identity):
    _assert_job_placement(folder,job,identity)
    protocol = json.loads((folder/'protocol.json').read_text())
    expected = {key: job[key] for key in ('domain', 'chapter', 'method', 'seed')}
    expected.update({key: plan[key] for key in ('path_master', 'holdout_master', 'radio_groups')})
    config = protocol.get('config', {})
    if (any(config.get(key) != value for key, value in expected.items()) or
            protocol.get('source_sha256') != identity['source_sha256']):
        raise ValueError('Job identity differs: '+job['name'])
    settings = config if job['method'] not in ('A', 'B') else protocol
    for key in ('episodes', 'eval_every', 'development_paths', 'holdout_paths'):
        if settings.get(key) != plan[key]:
            raise ValueError('Completed job comparison protocol differs: '+job['name']+' '+key)
    if job['method'] not in ('A', 'B'):
        expected_baseline=dict(profile=plan['baseline_profile'],
            checkpoint_every=plan['checkpoint_every'],checkpoint_keep=plan['checkpoint_keep'],
            final_checkpoint_mode=plan.get('final_checkpoint_mode','full'))
        if any(config.get(k)!=v for k,v in expected_baseline.items()):
            raise ValueError('Baseline network/checkpoint protocol differs')
    if identity.get('software'):
        software=protocol.get('software',{})
        if software.get('python')!=identity['python']:
            raise ValueError('Job Python runtime differs from the comparison')
        if job['method'] in ('A','B'):
            import torch
            if (software.get('numpy')!=identity['software']['numpy'] or
                    software.get('torch')!=str(torch.__version__)):
                raise ValueError('A/B numerical runtime differs from the comparison')
        elif any(software.get('packages',{}).get(k)!=v for k,v in identity['software'].items()):
            raise ValueError('Baseline numerical runtime differs from the comparison')
    if job['domain'] == 'radio':
        data = (protocol.get('data', {}) if job['method'] in ('A', 'B') else
                protocol.get('environment', {}).get('data', {}))
        for name in ('images.npy', 'split_manifest.json'):
            key = 'images_sha256' if name == 'images.npy' else 'split_manifest_sha256'
            if data.get(key) != identity['data_sha256'].get(name+'_sha256'):
                raise ValueError('Completed job image data differs')
    return protocol


def _validate_manifest(checkpoint,protocol,identity):
    manifest=json.loads((checkpoint/'manifest.json').read_text())
    if manifest.get('identity')!=protocol or manifest.get('source_hashes')!=identity['source_sha256']:
        raise ValueError('Baseline checkpoint source/identity differs')
    paths=[checkpoint/'manifest.json']
    for name,checksum in manifest.get('files',{}).items():
        if Path(name).name!=name or digest(checkpoint/name)!=checksum:
            raise ValueError('Baseline checkpoint checksum differs')
        paths.append(checkpoint/name)
    if not manifest.get('files'):
        raise ValueError('Baseline checkpoint is empty')
    return manifest,paths


def _completed_baseline_state(folder,job,plan,identity,protocol):
    from .baselines.runner import _read_commit,_checkpoint_path
    commit=_read_commit(folder,protocol)
    latest=commit['generations'][0]
    if latest['episode']!=plan['episodes']:
        raise ValueError('Completed marker has no committed final endpoint')
    committed=_checkpoint_path(folder,latest)
    if digest(committed/'manifest.json')!=latest['manifest_sha256']:
        raise ValueError('Completed generation manifest digest differs')
    _validate_manifest(committed,protocol,identity)
    mode=plan.get('final_checkpoint_mode','full')
    if mode=='evaluation':
        checkpoint=folder/'.checkpoints'/f"episode_{plan['episodes']:08d}_evaluation"
    else:
        if latest.get('kind','full')!='full':
            raise ValueError('Full final retention has no full committed endpoint')
        checkpoint=committed
    manifest,_=_validate_manifest(checkpoint,protocol,identity)
    if manifest.get('accounting',{}).get('complete_episodes')!=plan['episodes']:
        raise ValueError('Final checkpoint accounting differs from the fixed endpoint')
    pending=(commit.get('complete') is not True or len(commit['generations'])!=1 or
             committed.resolve()!=checkpoint.resolve() or
             (folder/'checkpoint').resolve()!=checkpoint.resolve())
    kept={entry['relative_path'] for entry in commit['generations']}
    for path in (folder/'.checkpoints').iterdir():
        generated=(re.fullmatch(r'episode_[0-9]{8}(?:_evaluation)?',path.name) or
                   path.name.startswith('.recent-agent-'))
        if generated and str(path.relative_to(folder)) not in kept:
            pending=True
    return checkpoint,manifest,pending


def validate_completed(folder, job, plan, identity):
    marker = folder/'completed.json'
    if not marker.exists():
        return False
    completed = json.loads(marker.read_text())
    protocol=_validate_job_protocol(folder,job,plan,identity)
    if completed.get('status')!='complete' or completed.get('episodes')!=plan['episodes']:
        raise ValueError('Completed job endpoint differs')
    expected_replay = 0. if plan['holdout_paths'] else (0. if job['method'] in ('A','B') else None)
    if completed.get('actor_changed') is not True or completed.get('replay_max_error') != expected_replay:
        raise ValueError('Completed job lacks successful update/replay evidence: '+job['name'])
    if not plan['holdout_paths'] and (completed.get('holdout_score') is not None or
                                     list(folder.glob('holdout_*'))):
        raise ValueError('Engineering preflight unexpectedly opened independent testing')
    if job['method'] in ('A','B'):
        import torch
        payload=torch.load(folder/f"checkpoint_{plan['episodes']}.pt",map_location='cpu',weights_only=False)
        if payload.get('identity')!=protocol or payload.get('episode')!=plan['episodes']:
            raise ValueError('Completed A/B checkpoint does not match its protocol')
    else:
        relative=Path(completed.get('checkpoint') or completed.get('full_checkpoint') or 'checkpoint')
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid final checkpoint reference')
        checkpoint,manifest,pending=_completed_baseline_state(folder,job,plan,identity,protocol)
        if relative!=Path('checkpoint') and (folder/relative).resolve()!=checkpoint.resolve():
            raise ValueError('Completed checkpoint reference differs from its generation')
        available=bool(manifest.get('training_resume_available',True))
        if any(completed.get(k)!=available for k in ('exact_restore_available','training_resume_available')):
            raise ValueError('Completed continuation capability differs from its checkpoint')
        retained=(checkpoint/'replay.npz').is_file()
        if completed.get('replay_retained')!=retained:
            raise ValueError('Completed replay retention differs from its checkpoint')
        mode=plan.get('final_checkpoint_mode','full')
        if completed.get('final_checkpoint_mode')!=mode:
            raise ValueError('Completed final retention mode differs')
        if mode=='evaluation' and (manifest.get('include_replay') is not False or retained):
            raise ValueError('Evaluation final checkpoint unexpectedly retained replay')
        if mode=='full' and (manifest.get('include_replay') is not True or
                              job['method']!='paspo' and (not available or not retained)):
            raise ValueError('Full final checkpoint omitted its continuation state')
        moments_expected=not(mode=='evaluation' and job['method']!='paspo')
        if (manifest.get('optimizer_moments_retained') is not moments_expected or
                completed.get('optimizer_moments_retained') is not moments_expected):
            raise ValueError('Completed optimizer moment retention differs from its registered mode')
        if pending:
            # The runner deliberately writes completed.json before atomically
            # switching/pruning the final generation. Its idempotent --resume
            # repairs this transaction; it must not be skipped or rejected as
            # a foreign retention policy.
            return False
    return True


def validate_partial(folder,job,plan,identity,*,finalization_only=False):
    """Return only checksum-verified files that can credit an existing run."""
    _assert_job_placement(folder,job,identity)
    if not folder.exists():
        return []
    protocol=_validate_job_protocol(folder,job,plan,identity)
    if job['method'] in ('A','B'):
        import torch
        path=folder/'checkpoint_latest.pt'
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload.get('identity')!=protocol or not 0<=payload.get('episode',-1)<=plan['episodes']:
            raise ValueError('Partial A/B checkpoint identity/endpoint differs')
        return [path]
    from .baselines.runner import _read_commit,_checkpoint_path
    commit=_read_commit(folder,protocol)
    valid=[]
    for entry in commit['generations']:
        try:
            path=_checkpoint_path(folder,entry)
            if digest(path/'manifest.json')!=entry['manifest_sha256']:
                raise ValueError('Partial checkpoint manifest digest differs')
            manifest,paths=_validate_manifest(path,protocol,identity)
            if (manifest.get('accounting',{}).get('complete_episodes')!=entry['episode'] or
                    not finalization_only and not manifest.get('training_resume_available',True)):
                raise ValueError('Partial checkpoint is not a resumable endpoint')
            valid.extend(paths)
        except (OSError,ValueError,KeyError):
            # A valid previous generation can repair a corrupt latest one.
            continue
    if not valid:
        raise ValueError('No verified committed partial state can be resumed')
    if finalization_only and plan.get('final_checkpoint_mode','full')=='evaluation':
        staged=folder/'.checkpoints'/f"episode_{plan['episodes']:08d}_evaluation"
        _,paths=_validate_manifest(staged,protocol,identity)
        valid.extend(paths)
    return valid


def credited_existing_bytes(paths_by_job):
    """Count allocated blocks once across jobs and filesystem hardlinks."""
    seen,credits=set(),{}
    for name,paths in paths_by_job.items():
        credit=0
        for path in paths:
            stat=path.stat()
            inode=(stat.st_dev,stat.st_ino)
            if inode not in seen:
                seen.add(inode)
                credit+=stat.st_blocks*512
        credits[name]=credit
    return credits


def _external_cpu(active, previous, previous_at):
    now = time.monotonic()
    own, totals, outside = {os.getpid()}, {}, 0.
    for item in active.values():
        own.add(item['process'].pid)
        try:
            own.update(c.pid for c in psutil.Process(item['process'].pid).children(recursive=True))
        except psutil.Error:
            pass
    for process in psutil.process_iter(['pid', 'cpu_times']):
        try:
            total = sum(process.info['cpu_times'][:2])
            totals[process.pid] = total
            if process.pid not in own and process.pid in previous:
                outside += max(0., total-previous[process.pid])/max(now-previous_at, 1e-6)
        except (psutil.Error, TypeError):
            pass
    return outside, totals, now


def _summary(root, plan, jobs):
    rows, grouped = [], defaultdict(list)
    for job in jobs:
        completed = json.loads((root/'jobs'/job['name']/'completed.json').read_text())
        score = completed.get('holdout_score')
        row = dict(**job, episodes=plan['episodes'],
                   holdout_mean=None if score is None else score['mean'],
                   holdout_paths=plan['holdout_paths'])
        rows.append(row)
        if score is not None:
            grouped[(job['domain'],job['chapter'],job['method'])].append(score['mean'])
    groups = [dict(domain=key[0],chapter=key[1],method=key[2],training_seeds=len(values),
                   mean=float(np.mean(values)),
                   seed_sd=None if len(values)<2 else float(np.std(values,ddof=1)))
              for key,values in sorted(grouped.items())]
    write_json(root/'comparison_results.json', dict(schema=SCHEMA, purpose=plan.get('purpose'),
               independent_test_opened=bool(plan['holdout_paths']), by_seed=rows, across_seeds=groups))
    with (root/'comparison_by_seed.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def run_suite(args):
    plan = validate_plan(json.loads(Path(args.config).read_text()))
    if args.max_workers < 1:
        raise ValueError('Positive worker limit required')
    root = Path(args.output).expanduser().resolve()
    jobs = registered_jobs(plan)
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else DATA_ROOT.resolve()
    extra_roots=getattr(args,'storage_root',None) or []
    done,pending,credit_paths,completion_repairs=[],[],{},set()
    if args.resume:
        original=root/'suite_protocol.json'
        if not original.is_file():
            raise ValueError('Resume requires an existing comparison protocol')
        old=json.loads(original.read_text())
        placements=old.get('storage_placement',{})
        if set(placements)!={j['name'] for j in jobs}:
            raise ValueError('Registered storage placement does not cover all comparison jobs')
        identity=suite_identity(plan,jobs,data_root,placements=placements,
            storage_roots=[str(Path(p).expanduser().resolve()) for p in extra_roots])
        if old!=identity:
            raise ValueError('Comparison source, data, software or registered plan changed')
        # Trust existing blocks only after protocol/data/source checks and
        # complete artifact or resumable-generation checksum validation.
        for job in jobs:
            destination=root/'jobs'/job['name']
            if validate_completed(destination,job,plan,identity):
                done.append(job['name'])
            else:
                repair=(destination/'completed.json').is_file()
                credit_paths[job['name']]=validate_partial(destination,job,plan,identity,
                                                          finalization_only=repair)
                if repair:completion_repairs.add(job['name'])
                pending.append(job)
        estimate=storage_plan(plan,pending,args.max_workers)
        credits=credited_existing_bytes(credit_paths)
        for row in estimate['jobs']:
            if row['name'] in completion_repairs:
                # Every model/state needed for finalization has been verified
                # above. Repair only replaces tiny JSON/link metadata and
                # prunes superseded generations; it does not train or write a
                # new replay. Keep existing allocation plus 64KiB headroom.
                row['final_bytes']=credits[row['name']]+64*1024
                row['extra_active_peak_bytes']=0
        capacity=placement_plan(root,extra_roots,estimate,args.max_workers,
            registered={j['name']:placements[j['name']] for j in pending},
            existing_credits=credits)
    else:
        pending=list(jobs)
        estimate=storage_plan(plan,jobs,args.max_workers)
        capacity=placement_plan(root,extra_roots,estimate,args.max_workers)
        identity=suite_identity(plan,jobs,data_root,placements=capacity['placements'],
            storage_roots=capacity['storage_roots'])
    if args.dry_run:
        report = dict(status='ready' if capacity['capacity_sufficient'] else 'insufficient_storage',
                      training_started=False, identity=identity, capacity=capacity,
                      already_completed=len(done),remaining_jobs=len(pending),
                      commands=[job_command(plan,j,root/'jobs'/j['name'],data_root=args.data_root,
                                  resume=(root/'jobs'/j['name']).exists()) for j in pending])
        if args.report:
            destination=Path(args.report).expanduser().resolve()
            destination.parent.mkdir(parents=True,exist_ok=True)
            write_json(destination,report)
        print(json.dumps(report,ensure_ascii=False))
        return 0 if capacity['capacity_sufficient'] else 2
    if args.report:
        raise ValueError('--report is only for dry-run planning')
    if root.exists() and not args.resume:
        raise ValueError('Use a new comparison output directory or --resume')
    if pending and not capacity['capacity_sufficient']:
        raise RuntimeError(f"Insufficient storage: need {capacity['required_additional_bytes']} additional bytes; "
                           f"available {capacity['free_bytes']}. No training started.")
    root.mkdir(parents=True,exist_ok=args.resume)
    with exclusive_lock(root/'queue.lock'):
        protocol=root/'suite_protocol.json'
        if args.resume:
            if not protocol.is_file() or json.loads(protocol.read_text())!=identity:
                raise ValueError('Comparison source, data, software or registered plan changed')
        else:
            write_json(protocol,identity)
        write_json(root/'capacity_plan.json',capacity)
        for name in ('jobs','logs'):
            (root/name).mkdir(exist_ok=True)
        failed,active=[],{}
        if args.resume:
            write_json(root/'resume_capacity_plan.json',capacity)
        estimates={r['name']:dict(r,**capacity['remaining_job_resources'][r['name']])
                   for r in estimate['jobs']}
        environment=dict(os.environ)
        environment.update({key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE')})
        environment['PYTHONPATH']=str(PROJECT_ROOT)
        started=time.monotonic()
        previous,previous_at,external={},started,0.
        state=dict(status='running',total=len(jobs),completed=len(done),failed=failed,
                   running=[],queued=len(pending),target_workers=0,
                   external_cpu_cores=0.,free_bytes=shutil.disk_usage(root).free,
                   elapsed_seconds=0.)
        def interrupted(signum,frame):
            raise SystemExit(128+signum)
        old_handler=signal.signal(signal.SIGTERM,interrupted)
        try:
            while pending or active:
                if source_hashes()!=identity['source_sha256']:
                    raise RuntimeError('Runtime source changed while comparison was running')
                outside,previous,previous_at=_external_cpu(active,previous,previous_at)
                external=.7*external+.3*outside
                target=max(1,min(args.max_workers,capacity['storage_worker_limit'],
                    math.floor(cpu_quota()-plan.get('reserve_cpu',2)-external)))
                while pending and len(active)<target:
                    # A space-limited filesystem must not block runnable jobs
                    # assigned to the other store.
                    choice=None
                    for index,candidate in enumerate(pending):
                        physical=Path(identity['storage_placement'][candidate['name']])
                        filesystem=next(x for x in capacity['filesystems']
                            if physical.parent==Path(x['base'])/'jobs')
                        same=sum(1 for item in active.values() if item['device']==filesystem['device'])
                        if same<filesystem['worker_slots']:
                            choice=(index,candidate,physical,filesystem);break
                    if choice is None:break
                    index,job,physical,filesystem=choice
                    job_est=estimates[job['name']]
                    memory_reserved=sum(estimates[n]['worker_memory_bytes'] for n in active)
                    disk_reserved=sum(estimates[n]['extra_active_peak_bytes'] for n,x in active.items()
                                      if x['device']==filesystem['device'])
                    physical.parent.mkdir(parents=True,exist_ok=True)
                    if (memory_available()<memory_reserved+job_est['worker_memory_bytes']+512*MIB or
                        shutil.disk_usage(physical.parent).free < estimate['reserve_bytes']+disk_reserved+
                            job_est['extra_active_peak_bytes']+job_est['final_bytes']):
                        if not active:
                            raise RuntimeError('Insufficient current memory/storage for next comparison job')
                        break
                    pending.pop(index)
                    destination=root/'jobs'/job['name']
                    existed=physical.exists()
                    if physical!=destination:
                        if destination.is_symlink():
                            if destination.resolve()!=physical:
                                raise ValueError('Existing job symlink changed its registered destination')
                        elif destination.exists():
                            raise ValueError('Existing job directory violates registered placement')
                        else:
                            destination.symlink_to(physical,target_is_directory=True)
                    command=job_command(plan,job,destination,data_root=args.data_root,resume=existed)
                    handle=(root/'logs'/(job['name']+'.log')).open('a' if existed else 'x')
                    process=subprocess.Popen(command,cwd=PROJECT_ROOT,env=environment,
                        stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                    active[job['name']]=dict(process=process,handle=handle,job=job,device=filesystem['device'])
                for name,item in list(active.items()):
                    process=item['process']
                    if process.poll() is None:
                        continue
                    item['handle'].close()
                    try:
                        okay=process.returncode==0 and validate_completed(root/'jobs'/name,item['job'],plan,identity)
                    except (OSError,ValueError) as error:
                        okay=False
                        failed.append(dict(name=name,error=str(error),exit_code=process.returncode))
                    if okay:
                        done.append(name)
                    elif not failed or failed[-1].get('name')!=name:
                        failed.append(dict(name=name,exit_code=process.returncode,
                                           log=str(root/'logs'/(name+'.log'))))
                    del active[name]
                state=dict(status='running',total=len(jobs),completed=len(done),failed=failed,
                           running=[dict(name=n,pid=x['process'].pid) for n,x in active.items()],
                           queued=len(pending),target_workers=target,external_cpu_cores=external,
                           free_bytes=shutil.disk_usage(root).free,elapsed_seconds=time.monotonic()-started)
                write_json(root/'live_status.json',state)
                if pending or active:
                    time.sleep(1)
            state.update(status='failed' if failed else 'complete',running=[],queued=0)
            write_json(root/'live_status.json',state)
            write_json(root/'suite_complete.json',state)
            if not failed:
                _summary(root,plan,jobs)
            return 1 if failed else 0
        finally:
            had_error=sys.exc_info()[0] is not None
            terminate_workers(active)
            if had_error:
                write_json(root/'suite_interrupted.json',dict(status='interrupted',
                    completed=done,owned_workers_stopped=list(active),
                    message='Resume from committed per-job checkpoints with the identical plan'))
            signal.signal(signal.SIGTERM,old_handler)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-workers',type=int,default=32)
    parser.add_argument('--data-root')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--report')
    parser.add_argument('--storage-root',action='append',
                        help='Additional persistent result directory; placements are protocol-locked')
    return run_suite(parser.parse_args(argv))


if __name__=='__main__':
    raise SystemExit(main())
