"""Isolated Chapter 3 supplement dispatcher; adopts only exact scope identities."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BOOT = Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        raw = Path('/proc', str(pid), 'cmdline').read_bytes()
        return dict(start_ticks=fields[19], state=fields[0],
                    argv=raw.rstrip(b'\0').decode().split('\0'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def live(record):
    if record.get('boot_id') != BOOT:
        return False
    actual = identity(record['pid'])
    if not actual or actual['start_ticks'] != record['start_ticks'] or actual['state'] in ('Z', 'X'):
        return False
    if not any(actual['argv']):
        return None
    if actual['argv'] != record['command']:
        raise RuntimeError('Live PID argv mismatch; refusing to relaunch '+str(record['pid']))
    return True


def resources():
    # Container quotas matter; host-wide free memory is not an allocation.
    cpu = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    period = Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    quota = int(cpu.read_text())/int(period.read_text()) if cpu.exists() else len(os.sched_getaffinity(0))
    if quota <= 0:
        quota = len(os.sched_getaffinity(0))
    mmax = Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')
    mnow = Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')
    memfree = int(mmax.read_text())-int(mnow.read_text()) if mmax.exists() else 8*2**30
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    gpus = {line.split(',')[0].strip(): int(line.split(',')[1]) for line in gpu.splitlines()}
    disk = os.statvfs(ROOT)
    return dict(cpu_quota=quota, memory_free_bytes=memfree,
                disk_free_bytes=disk.f_bavail*disk.f_frsize, gpu_free_mib=gpus)


def validate(job, plan_sha):
    output = Path(job['output'])
    receipt = json.loads((output/'job_receipt.json').read_text())
    if (receipt.get('success') is not True or receipt.get('job_id') != job['id']
            or receipt.get('plan_sha256') != plan_sha):
        raise ValueError('Invalid job success identity')
    for relative, expected in receipt['artifacts_sha256'].items():
        path = (output/relative).resolve()
        if not path.is_relative_to(output.resolve()) or sha(path) != expected:
            raise ValueError('Job artifact changed: '+relative)
    return receipt


def seal_check(plan_sha):
    seal = json.loads((ROOT/'execution_seal.json').read_text())
    if sha(ROOT/'plan.json') != plan_sha or seal['plan_sha256'] != plan_sha:
        raise ValueError('Frozen plan changed')
    for relative, digest in seal['execution_sha256'].items():
        if sha(ROOT/relative) != digest:
            raise ValueError('Frozen execution file changed: '+relative)


def census(expected):
    # Durable intent plus exact-command census handles controller death around spawn.
    matches = []
    for item in Path('/proc').iterdir():
        if item.name.isdigit():
            actual = identity(int(item.name))
            if actual and actual['state'] not in ('Z', 'X') and actual['argv'] == expected:
                matches.append(dict(actual, pid=int(item.name), boot_id=BOOT, command=expected))
    return matches


def main():
    (ROOT/'logs').mkdir(exist_ok=True)
    lock = (ROOT/'queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((ROOT/'plan.json').read_text())
    plan_sha = sha(ROOT/'plan.json')
    seal_check(plan_sha)
    jobs = plan['jobs']
    state_path = ROOT/'dispatch_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'jobs': {}}
    children = {}
    logs = {}
    # One exact-command census at adoption, not 220 full /proc walks per tick.
    expected_commands = {tuple([plan['python'], str(ROOT/'execution/job_entry.py'), '--job-id', job['id']]): job['id']
                         for job in jobs}
    startup_matches = {job['id']: [] for job in jobs}
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        actual = identity(int(item.name))
        if actual and actual['state'] not in ('Z', 'X'):
            key = expected_commands.get(tuple(actual['argv']))
            if key:
                startup_matches[key].append(dict(actual, pid=int(item.name), boot_id=BOOT,
                                                 command=actual['argv']))
    write(ROOT/'queue_pid.json', dict(pid=os.getpid(), boot_id=BOOT, **identity(os.getpid())))
    write(ROOT/'recovery_launch.json', dict(status='running',
          command=[sys.executable, str(Path(__file__).resolve())], scope=str(ROOT)))
    while True:
        seal_check(plan_sha)
        for job in jobs:
            key = job['id']
            entry = state['jobs'].get(key)
            expected = [plan['python'], str(ROOT/'execution/job_entry.py'), '--job-id', key]
            if not entry:
                found = startup_matches.pop(key, [])
                if len(found) > 1:
                    raise RuntimeError('Multiple exact workers; refusing duplication '+key)
                if found:
                    entry = dict(found[0], status='running', output=job['output'])
                    state['jobs'][key] = entry
                else:
                    continue
            if entry['status'] in ('complete', 'failed', 'needs_identity_review'):
                continue
            if entry['status'] == 'starting' and not entry.get('pid'):
                found = census(expected)
                if len(found) == 1:
                    entry.update(found[0], status='running')
                else:
                    # A child may have completed after the durable intent but
                    # before the controller recorded its PID. Never respawn it.
                    try:
                        if found:
                            raise RuntimeError('Multiple exact workers')
                        validate(job, plan_sha)
                        entry.update(status='complete', ended=time.time())
                    except Exception:
                        entry.update(status='needs_identity_review', reason='Unresolved durable spawn intent')
                    continue
            child = children.get(key)
            if child is not None:
                code = child.poll()
                if code is None:
                    continue
                logs.pop(key).close()
                del children[key]
                if code != 0:
                    entry.update(status='failed', returncode=code, ended=time.time())
                    continue
            elif live(entry) is not False:
                continue
            try:
                validate(job, plan_sha)
                entry.update(status='complete', ended=time.time())
            except Exception as error:
                entry.update(status='failed', reason=repr(error), ended=time.time())
        counts = {name: sum(e['status'] == name for e in state['jobs'].values())
                  for name in ('running', 'complete', 'failed', 'starting', 'needs_identity_review')}
        counts['pending'] = len(jobs)-len(state['jobs'])
        resource = resources()
        running = [key for key, entry in state['jobs'].items()
                   if entry['status'] in ('running', 'starting', 'needs_identity_review')]
        cap = min(plan['max_concurrent'], max(1, int(resource['cpu_quota'])-2))
        available = max(0, cap-len(running))
        if counts['failed'] or counts['needs_identity_review']:
            available = 0  # Do not silently retry a scientific failure or ambiguous PID.
        # Use measured live resources with a fixed floor, including late CUDA contexts.
        if resource['memory_free_bytes'] < plan['memory_floor_bytes']:
            available = 0
        if resource['disk_free_bytes'] < plan['disk_floor_bytes']:
            available = 0
        for job in jobs:
            if available <= 0:
                break
            key = job['id']
            if key in state['jobs']:
                continue
            if any(state['jobs'].get(dep, {}).get('status') != 'complete' for dep in job.get('depends_on', [])):
                continue
            uuid = job.get('gpu_uuid')
            if uuid:
                reserved = sum(1 for entry in state['jobs'].values()
                    if entry.get('gpu_uuid') == uuid and entry['status'] in ('running', 'starting'))
                if resource['gpu_free_mib'].get(uuid, 0)-reserved*plan['gpu_job_reserve_mib'] < plan['gpu_floor_mib']+plan['gpu_job_reserve_mib']:
                    continue
            command = [plan['python'], str(ROOT/'execution/job_entry.py'), '--job-id', key]
            entry = dict(status='starting', started=time.time(), command=command,
                         output=job['output'], gpu_uuid=uuid, boot_id=BOOT)
            state['jobs'][key] = entry
            write(state_path, state)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=uuid or '-1', PYTHONDONTWRITEBYTECODE='1',
                       CH3_SUPPLEMENT_SCOPE=str(ROOT), CH3_SUPPLEMENT_JOB_ID=key)
            for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                env[name] = '1'
            stream = (ROOT/'logs'/(key+'.log')).open('ab', buffering=0)
            child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            entry.update(pid=child.pid, start_ticks=identity(child.pid)['start_ticks'], status='running')
            children[key] = child
            logs[key] = stream
            write(state_path, state)
            available -= 1
        counts = {name: sum(e['status'] == name for e in state['jobs'].values())
                  for name in ('running', 'complete', 'failed', 'starting', 'needs_identity_review')}
        counts['pending'] = len(jobs)-len(state['jobs'])
        view = []
        for job in jobs:
            entry = dict(state['jobs'].get(job['id'], {'status': 'pending'}))
            path = Path(job['output'])/'progress.json'
            if path.exists():
                try:
                    entry['progress'] = json.loads(path.read_text())
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
            view.append(dict(id=job['id'], kind=job['kind'], method=job['method'],
                             domain=job['domain'], seed=job['seed'], **entry))
        write(state_path, state)
        write(ROOT/'status.json', dict(updated=time.time(), plan_sha256=plan_sha,
              counts=counts, resources=resource, jobs=view))
        if counts['complete'] == len(jobs):
            write(ROOT/'completion_receipt.json', dict(success=True, jobs=len(jobs),
                  training_jobs=10, evaluation_jobs=210, plan_sha256=plan_sha, time=time.time()))
            write(ROOT/'recovery_launch.json', dict(status='complete', command=None,
                  automatic_extra_training=False))
            break
        time.sleep(5)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write(ROOT/'controller_error.json', dict(time=time.time(), error=traceback.format_exc()))
        raise
