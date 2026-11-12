#!/usr/bin/env python3
"""Durable formal dispatch: bounded resources, explicit exits, no automatic retries."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

R = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def identity(pid):
    try:
        root = Path('/proc') / str(pid)
        stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] in ('Z', 'X'):
            return None
        argv = [x.decode() for x in (root / 'cmdline').read_bytes().split(b'\0') if x]
        if not argv:
            return None
        memory = dict(line.split(':', 1) for line in (root / 'status').read_text().splitlines() if ':' in line)
        return dict(pid=pid, start_ticks=int(stat[19]), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    argv=argv, state=stat[0], rss_anon=int(memory.get('RssAnon', '0 kB').split()[0]) * 1024)
    except (FileNotFoundError, ProcessLookupError):
        return None


def same(a, b):
    return bool(a and b) and all(a[k] == b[k] for k in ('pid', 'start_ticks', 'boot_id', 'argv'))


def stable_child(proc, expected):
    for _ in range(150):
        current = identity(proc.pid)
        if current and current['argv'] == expected:
            return current
        if proc.poll() is not None:
            break
        time.sleep(.02)
    raise RuntimeError('Child did not acquire its registered nonempty argv')


def cpu_usage():
    return int(Path('/sys/fs/cgroup/cpuacct/cpuacct.usage').read_text()) / 1e9


def resources():
    cpu_dir = Path('/sys/fs/cgroup/cpu')
    quota = int((cpu_dir / 'cpu.cfs_quota_us').read_text())
    period = int((cpu_dir / 'cpu.cfs_period_us').read_text())
    cores = quota / period if quota > 0 else len(os.sched_getaffinity(0))
    memory_dir = Path('/sys/fs/cgroup/memory')
    available_cgroup = int((memory_dir / 'memory.limit_in_bytes').read_text()) - int((memory_dir / 'memory.usage_in_bytes').read_text())
    meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = min(available_cgroup, int(meminfo['MemAvailable'].split()[0]) * 1024)
    usage, started = cpu_usage(), time.monotonic()
    time.sleep(2)
    used = (cpu_usage() - usage) / (time.monotonic() - started)
    gpu = {}
    if shutil.which('nvidia-smi'):
        for line in subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).splitlines():
            uuid, free = line.split(',')
            gpu[uuid.strip()] = int(free) * 1024 ** 2
    return dict(cpu_quota=cores, cpu_used=used, memory_available=available, gpu_free=gpu, disk_free=shutil.disk_usage(R).free)


def ram_peak(job):
    # Conservative whole-run native peaks; only verified anonymous RSS is credited.
    if job['method'] == 'full':
        return (3.5 if job['domain'] == 'radio' else 1.5) * GIB
    if job['domain'] == 'radio':
        return {'pasta': 6.5, 'morld': 4.5}.get(job['method'], 5.0) * GIB
    return {'pasta': 1.0, 'morld': 2.0}.get(job['method'], 1.75) * GIB


def gpu_peak(job):
    return (1.25 if job['method'] == 'morld' else 1.0) * GIB if job['gpu_uuid'] else 0


def output_peak(job):
    if job.get('kind') == 'fixed_evaluation':
        return 8 * 1024 ** 2
    return (160 if job['method'] == 'full' else 64) * 1024 ** 2


def allocated_output(job):
    total = 0
    for path in Path(job['output']).rglob('*'):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_blocks * 512
        except FileNotFoundError:
            # A native gzip output is allowed to be atomically finalized.
            continue
    return total


def verify_capsule(phase):
    name = {'gates': 'engineering_capsule.json', 'science': 'science_capsule.json', 'evaluation': 'postprocess_capsule.json'}[phase]
    capsule = read(R / 'deployment' / name)
    for rel, expected in capsule['files'].items():
        if sha(R / rel) != expected:
            raise RuntimeError('Execution capsule changed: ' + rel)


def supervise(args):
    job = read(args.job)
    lease = Path(args.lease)
    for _ in range(200):
        if lease.exists() and same(read(lease).get('supervisor'), identity(os.getpid())):
            break
        time.sleep(.05)
    else:
        raise RuntimeError('Supervisor lease not durably registered')
    try:
        verify_capsule(args.phase)
        if sha(job['config']) != job['config_sha256']:
            raise RuntimeError('Config SHA changed')
        if Path(job['output']).exists():
            raise RuntimeError('Refuse existing output; no automatic retry')
        cmd = [sys.executable, '-B', str(R / 'execution' / job['worker']), '--config', job['config']]
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                   XDG_CACHE_HOME=str(R / 'cache'), MPLCONFIGDIR=str(R / 'cache/matplotlib'),
                   TMPDIR=str(R / 'tmp'), TORCH_HOME=str(R / 'cache/torch'), CUDA_CACHE_PATH=str(R / 'cache/cuda'),
                   PYTORCH_KERNEL_CACHE_PATH=str(R / 'cache/torch_kernels'),
                   CUDA_VISIBLE_DEVICES=job['gpu_uuid'] or '', CH5_GPU_UUID=job['gpu_uuid'] or '')
        with (R / 'logs' / (job['id'] + '.log')).open('ab', buffering=0) as handle:
            child = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, cwd=R)
            rec = read(lease)
            try:
                rec.update(child=stable_child(child, cmd), child_command=cmd, child_started=now())
                write(lease, rec)
            except Exception:
                # Never abandon a living child or fabricate exit; wait for its actual exit.
                rec.update(child_registration_error=True, child_pid=child.pid)
                write(lease, rec)
            code = child.wait()
        rec = read(lease)
        rec.update(exit_code=code, exited_utc=now())
        write(lease, rec)
        return code
    except Exception as exc:
        rec = read(lease)
        rec.update(supervisor_error=str(exc), exit_code=125, exited_utc=now())
        write(lease, rec)
        raise


def controller(args):
    plan_path = R / {'gates': 'engineering_plan.json', 'science': 'plan.json', 'evaluation': 'fixed_evaluation_plan.json'}[args.phase]
    if sha(plan_path) != args.plan_sha:
        raise RuntimeError('Plan SHA mismatch')
    verify_capsule(args.phase)
    if args.phase == 'science':
        gate = read(R / 'deployment/gate_pass.json')
        if gate.get('status') != 'PASS' or gate['contract_sha256'] != sha(R / 'contract.json'):
            raise RuntimeError('Native gate not passed for this contract')
    jobs = [j for j in read(plan_path)['jobs'] if j['host'] == args.host]
    directory = R / 'dispatch' / args.phase / args.host
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(directory / 'controller.json', identity(os.getpid()))
    cpu_ewma = None
    while True:
        running, complete, failed, pending, uncertain = [], [], [], [], []
        for job in jobs:
            lease_path = directory / 'leases' / (job['id'] + '.json')
            done = Path(job['output']) / 'completed.json'
            if lease_path.exists():
                lease = read(lease_path)
                if 'exit_code' in lease:
                    target = complete if lease['exit_code'] == 0 and done.exists() and read(done).get('success') else failed
                    target.append(job)
                elif same(lease.get('supervisor'), identity(lease['supervisor']['pid'])):
                    child = identity(lease['child']['pid']) if lease.get('child') else None
                    if not same(lease.get('child'), child):
                        child = None
                    running.append((job, lease, child))
                else:
                    uncertain.append(job)
            elif Path(job['output']).exists():
                uncertain.append(job)
            else:
                pending.append(job)
        res = resources()
        cpu_ewma = res['cpu_used'] if cpu_ewma is None else .5 * res['cpu_used'] + .5 * cpu_ewma
        blocked = []
        future_outputs = sum(max(0, output_peak(job) - allocated_output(job))
                             for job in pending + [row[0] for row in running])
        res['future_output_reserve'] = future_outputs
        if not failed and not uncertain:
            for job in pending:
                memory_reserve = sum(max(0, ram_peak(j) - (child['rss_anon'] if child else 0)) for j, lease, child in running)
                gpu_reserve = sum(gpu_peak(j) for j, lease, child in running if j['gpu_uuid'] == job['gpu_uuid'])
                # New processes have not yet reached their steady usage; reserve a core for 30s.
                startup_cpu = sum(1.0 for j, lease, child in running if
                                  (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(lease['spawn_utc'])).total_seconds() < 30)
                cpu_limit = min(40.0, res['cpu_quota'] - 2.0)
                allowed = (len(running) < (8 if args.phase == 'gates' else int(cpu_limit))
                           and max(res['cpu_used'], cpu_ewma) + startup_cpu + 1 <= cpu_limit
                           and res['memory_available'] > memory_reserve + ram_peak(job) + 4 * GIB
                           and res['disk_free'] > future_outputs + (2 if args.phase == 'evaluation' else 4) * GIB)
                if job['gpu_uuid']:
                    allowed = allowed and res['gpu_free'].get(job['gpu_uuid'], 0) > gpu_reserve + gpu_peak(job) + 2 * GIB
                if not allowed:
                    blocked.append(job['id'])
                    continue
                if sha(job['config']) != job['config_sha256']:
                    raise RuntimeError('Pending config drift: ' + job['id'])
                job_path = directory / 'job_records' / (job['id'] + '.json')
                write(job_path, job)
                lease_path = directory / 'leases' / (job['id'] + '.json')
                lease_path.parent.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, '-B', str(Path(__file__).resolve()), 'supervise', '--job', str(job_path), '--lease', str(lease_path), '--phase', args.phase]
                with (directory / (job['id'] + '.supervisor.log')).open('ab', buffering=0) as handle:
                    proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=R, start_new_session=True)
                write(lease_path, dict(job_id=job['id'], supervisor=stable_child(proc, cmd), spawn_utc=now(), no_automatic_retry=True))
                break
        rows = []
        for job, lease, child in running:
            progress = Path(job['output']) / 'progress.json'
            rows.append(dict(id=job['id'], pid=child['pid'] if child else None, gpu_uuid=job['gpu_uuid'],
                             progress=read(progress) if progress.exists() else None))
        status = dict(schema='chapter5-formal-five-seed-live-status-v1', utc=now(), phase=args.phase, host=args.host,
                      plan_sha256=args.plan_sha, counts=dict(total=len(jobs), complete=len(complete), running=len(running),
                      pending=len(pending), failed=len(failed), uncertain=len(uncertain)), running=rows,
                      failed=[j['id'] for j in failed], uncertain=[j['id'] for j in uncertain], resource_blocked=blocked,
                      resources=res, cpu_ewma=cpu_ewma, automatic_retries=False)
        write(R / ('status_' + args.phase + '_' + args.host + '.json'), status)
        if len(complete) == len(jobs):
            return 0
        if failed or uncertain:
            return 2
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    item = sub.add_parser('controller')
    item.add_argument('--host', choices=['local', 'remote'], required=True)
    item.add_argument('--phase', choices=['gates', 'science', 'evaluation'], required=True)
    item.add_argument('--plan-sha', required=True)
    item = sub.add_parser('supervise')
    item.add_argument('--job', required=True)
    item.add_argument('--lease', required=True)
    item.add_argument('--phase', choices=['gates', 'science', 'evaluation'], required=True)
    args = parser.parse_args()
    return supervise(args) if args.command == 'supervise' else controller(args)


if __name__ == '__main__':
    sys.exit(main())
