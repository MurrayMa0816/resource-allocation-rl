"""Resource-aware parallel execution of a fixed, recorded experiment plan."""
from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

from .io import PROJECT_ROOT, SCHEMA, digest, source_hashes, write_json
from .image_data import DATA_ROOT


def cpu_quota():
    limits = [float(os.cpu_count() or 1)]
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            limits.append(int(quota)/int(period))
    except (OSError, ValueError):
        pass
    try:
        quota = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
        period = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
        if quota > 0:
            limits.append(quota/period)
    except (OSError, ValueError):
        pass
    return min(limits)


def registered_jobs(plan):
    jobs = []
    if plan.get('arms'):
        arms = plan['arms']
    else:
        arms = [dict(chapter=chapter, method=method) for chapter, method in
                itertools.product(plan['chapters'], plan['methods'])]
    for domain, arm, seed in itertools.product(plan['domains'], arms, plan['seeds']):
        job = dict(domain=domain, seed=seed, **arm)
        name = f"{domain}__chapter{job['chapter']}__{job['method']}__seed{seed}"
        for field in ('variant', 'temporal'):
            if job.get(field):
                name += '__'+job[field]
        job['name'] = name
        jobs.append(job)
    if not jobs:
        raise ValueError('Suite must contain at least one job')
    if len({job['name'] for job in jobs}) != len(jobs):
        raise ValueError('Suite contains duplicate job identities')
    return jobs


def terminate_workers(active):
    """Only stop processes created by this queue, never unrelated experiments."""
    for item in active.values():
        process = item['process']
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic()+5.
    for item in active.values():
        process = item['process']
        try:
            process.wait(timeout=max(.01, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5.)
        item['handle'].close()


def _terminate_signal(signum, frame):
    raise SystemExit(128+signum)


def run_suite(args):
    plan = json.loads(Path(args.config).read_text())
    if plan.get('schema') != 'allocation23-suite-v1':
        raise ValueError('Expected an allocation23 suite plan')
    if args.max_workers < 1:
        raise ValueError('Positive concurrency required')
    root = Path(args.output).expanduser().resolve()
    if root.exists() and not args.resume:
        raise ValueError('Suite output must be new unless explicitly resuming')
    root.mkdir(parents=True, exist_ok=args.resume)
    lock = (root/'queue.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = registered_jobs(plan)
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else DATA_ROOT.resolve()
    data_identity = ({'images_sha256': digest(data_root/'images.npy'),
                      'split_manifest_sha256': digest(data_root/'split_manifest.json')}
                     if 'radio' in plan['domains'] else {})
    identity = dict(schema=SCHEMA, plan=plan, jobs=jobs, source_sha256=source_hashes(),
                    data_sha256=data_identity)
    if args.resume:
        if json.loads((root/'suite_protocol.json').read_text()) != identity:
            raise ValueError('Suite plan or runtime differs from its original protocol')
    else:
        write_json(root/'suite_protocol.json', identity)
    for name in ('jobs', 'logs'):
        (root/name).mkdir(exist_ok=True)
    pending, done, failed, active = [], [], [], {}
    for job in jobs:
        marker = root/'jobs'/job['name']/'completed.json'
        if marker.exists() and json.loads(marker.read_text()).get('status') == 'complete':
            completed = json.loads(marker.read_text())
            protocol = json.loads((marker.parent/'protocol.json').read_text())
            expected = {field: job.get(field) for field in ('domain', 'chapter', 'method', 'seed', 'variant', 'temporal')}
            expected.update({field: plan[field] for field in ('path_master', 'holdout_master', 'radio_groups')})
            if (completed['episodes'] != plan['episodes'] or completed['config'] != expected
                    or protocol['source_sha256'] != identity['source_sha256']):
                raise ValueError('Completed job does not belong to this suite: '+job['name'])
            if job['domain'] == 'radio' and any(protocol['data'].get(key) != value for key, value in data_identity.items()):
                raise ValueError('Completed job uses a different image dataset: '+job['name'])
            done.append(job['name'])
        else:
            pending.append(job)
    env = dict(os.environ)
    env.update({key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    # Support source-tree execution as well as installed packages.
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    previous, previous_at, external = {}, time.monotonic(), 0.
    started = time.monotonic()
    state = dict(status='running', total=len(jobs), completed=len(done), failed=failed,
                 running=[], queued=len(pending), target_workers=0,
                 external_cpu_cores=0., elapsed_seconds=0.)
    previous_termination_handler = signal.signal(signal.SIGTERM, _terminate_signal)
    try:
        while pending or active:
            now = time.monotonic()
            totals, outside = {}, 0.
            own = {os.getpid()}
            for item in active.values():
                own.add(item['process'].pid)
                try:
                    own.update(child.pid for child in psutil.Process(item['process'].pid).children(recursive=True))
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
            previous, previous_at = totals, now
            external = .7*external+.3*outside
            target = max(1, min(args.max_workers, math.floor(cpu_quota()-2-external)))
            while pending and len(active) < target:
                if shutil.disk_usage(root).free < 300*1024**2:
                    if not active:
                        raise RuntimeError('Insufficient free space to start the remaining jobs')
                    break
                job = pending.pop(0)
                destination = root/'jobs'/job['name']
                command = [sys.executable, '-m', 'allocation23', 'train']
                for field in ('domain', 'chapter', 'method', 'seed', 'variant', 'temporal'):
                    if job.get(field) is not None:
                        command.extend(['--'+field.replace('_', '-'), str(job[field])])
                for field in ('episodes', 'eval_every', 'development_paths', 'holdout_paths',
                              'path_master', 'holdout_master', 'radio_groups'):
                    command.extend(['--'+field.replace('_', '-'), str(plan[field])])
                if args.data_root:
                    command.extend(['--data-root', str(Path(args.data_root).resolve())])
                command.extend(['--output', str(destination)])
                if destination.exists():
                    command.append('--resume')
                handle = (root/'logs'/(job['name']+'.log')).open('a' if args.resume else 'x')
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env,
                                           stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                active[job['name']] = dict(process=process, handle=handle)
            for name, item in list(active.items()):
                process = item['process']
                if process.poll() is None:
                    continue
                item['handle'].close()
                marker = root/'jobs'/name/'completed.json'
                if process.returncode == 0 and marker.exists() and json.loads(marker.read_text()).get('status') == 'complete':
                    done.append(name)
                else:
                    failed.append(dict(name=name, exit_code=process.returncode, log=str(root/'logs'/(name+'.log'))))
                del active[name]
            progress = []
            for name, item in active.items():
                try:
                    info = json.loads((root/'jobs'/name/'progress.json').read_text())
                except (OSError, ValueError):
                    info = {}
                progress.append(dict(info, name=name, pid=item['process'].pid))
            state = dict(status='running', total=len(jobs), completed=len(done), failed=failed,
                         running=progress, queued=len(pending), target_workers=target,
                         external_cpu_cores=external, elapsed_seconds=time.monotonic()-started)
            write_json(root/'live_status.json', state)
            if pending or active:
                time.sleep(2)
        state.update(status='failed' if failed else 'complete', running=[], queued=0)
        write_json(root/'live_status.json', state)
        write_json(root/'suite_complete.json', state)
        if failed:
            return 1
        if not plan.get('arms'):
            from .plotting import plot_runs
            plot_runs(root/'jobs', root/'report')
        return 0
    finally:
        interrupted = sys.exc_info()[0] is not None
        terminate_workers(active)
        if interrupted:
            write_json(root/'suite_interrupted.json', dict(status='interrupted',
                       workers_stopped=list(active), completed=done,
                       message='Owned workers stopped; resume from committed checkpoints'))
        signal.signal(signal.SIGTERM, previous_termination_handler)
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-workers', type=int, default=40)
    parser.add_argument('--data-root')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    return run_suite(args)


if __name__ == '__main__':
    raise SystemExit(main())
