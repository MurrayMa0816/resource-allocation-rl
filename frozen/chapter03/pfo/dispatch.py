"""Isolated, resource-admitted Chapter 3 queue. Never restores old queues."""
import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CAPSULE = Path(__file__).resolve().parent
PYTHON = '/hy-tmp/resource_allocation_results/chapter2/gpu0_five_seed_150k_20261104/runtime/python/bin/python'
SOURCE = ROOT/'execution/source'
DATA = ROOT/'data/eurosat_rgb'
BOOT = Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(start_ticks=fields[19], state=fields[0])
    except FileNotFoundError:
        return None


def scope_worker_alive(state, identity_reader=identity, argv_reader=None):
    """Verify a live worker without mistaking a natural-exit race for adoption.

    Linux can empty cmdline before stat reaches Z. Such a worker returns None
    (identity pending), remains reserved, and is rechecked without relaunching.
    A nonempty, stable live mismatch still rejects adoption.
    """
    if state['boot_id'] != BOOT:
        return False

    def live(record):
        return (record is not None and record['start_ticks'] == state['start_ticks']
                and record['state'] not in ('Z', 'X'))

    if argv_reader is None:
        def argv_reader(pid):
            raw = Path('/proc', str(pid), 'cmdline').read_bytes()
            return raw.rstrip(b'\0').decode().split('\0')
    if not live(identity_reader(state['pid'])):
        return False
    for attempt in range(2):
        try:
            argv = argv_reader(state['pid'])
        except (FileNotFoundError, ProcessLookupError):
            return False
        if not live(identity_reader(state['pid'])):
            return False
        if not any(argv):
            return None
        if argv == state['command']:
            return True
    raise RuntimeError('Scope worker PID command identity mismatch; refusing adoption: '
                       f"pid={state['pid']} expected={state['command']!r} observed={argv!r}")


def gpu_free():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    return {line.split(',')[0].strip(): int(line.split(',')[1]) for line in output.splitlines()}


def summary_score(run):
    with gzip.open(Path(run)/'development_curve.csv.gz', 'rt') as stream:
        rows = list(csv.DictReader(stream))
    # Only the ordinary, unique DEV points; reloaded endpoints are not extra points.
    unique = {}
    for row in rows:
        if row['evaluation_kind'] not in ('endpoint_reload', 'reload'):
            value = float(row['native_mean'])
            if not math.isfinite(value):
                raise ValueError('Nonfinite registered DEV score')
            unique[int(row['training_step'])] = value
    points = sorted(unique.items())
    if len(points) < 3:
        raise ValueError('Insufficient registered DEV points')
    return sum(value for step, value in points[-3:])/3


def validated_completion(job, state, plan):
    """Admit only a scope-bound successful native run, including actual actor updates."""
    output = Path(state['output']).resolve()
    if not output.is_relative_to(ROOT/'jobs'/job['id']):
        raise ValueError('Job output escaped its registered scope')
    completed = output/'completed.json'
    receipt = json.loads(completed.read_text())
    if receipt.get('success') is not True:
        raise ValueError('Native success receipt is required')
    protocol = json.loads((output/'protocol.json').read_text())
    if job['kind'] == 'train_A':
        arguments = protocol['arguments']
        if (any(receipt.get(k) != job[k] for k in ('method', 'domain', 'seed'))
            or any(arguments.get(k) != job[k] for k in ('domain', 'seed', 'steps'))
            or arguments.get('arm') != job['arm']):
            raise ValueError('A training identity mismatch')
        if receipt.get('source_unchanged') is not True or protocol['source_identity']['package'] != plan['source_sha256']:
            raise ValueError('A frozen scientific source differs')
        if protocol['source_identity']['worker_sha256'] != sha(ROOT/'execution/a_native_formal_worker.py'):
            raise ValueError('A adapter identity differs')
        if (receipt['training_steps'] != job['steps'] or receipt['accounting']['environment_steps'] != job['steps']
            or arguments['path_master'] != job['master'] or arguments['final_master'] != plan['final_master']
            or receipt.get('actual_device') != 'cpu' or receipt.get('endpoint_reload_pass') is not True
            or receipt.get('full_spend_required') is not True or receipt.get('native_replay_error') != 0.
            or receipt.get('maximum_resource_fraction_error', 1.) > 1e-9
            or receipt.get('accounting', {}).get('actor_update_calls', 0) <= 0
            or receipt.get('final_evaluation', {}).get('paths') != plan['final_paths']
            or receipt.get('gzip_complete_before_success') is not True):
            raise ValueError('A scientific budget/replay/full-spend/update/final gate failed')
        if receipt['model_sha256'] != sha(output/'model/state.pt'):
            raise ValueError('A model digest differs')
        if not receipt.get('artifacts_sha256'):
            raise ValueError('A numeric artifact manifest missing')
        for name, digest in receipt.get('artifacts_sha256', {}).items():
            p = (output/name).resolve()
            if not p.is_relative_to(output) or sha(p) != digest:
                raise ValueError('A artifact digest differs '+name)
        return receipt
    if job['kind'] == 'evaluate_A':
        if (receipt.get('domain') != job['domain'] or receipt.get('seed') != job['seed']
                or receipt.get('arm') != job['arm']
                or receipt.get('learning_environment_steps') != 0
                or receipt.get('old_dev_equivalence', {}).get('status') != 'PASS'
                or receipt.get('actor_tensor_bytes_unchanged') is not True
                or receipt.get('origin_critic_tensor_bytes_unchanged') is not True
                or receipt.get('rng_unchanged') is not True
                or receipt.get('summary', {}).get('paths') != 128
                or receipt.get('summary', {}).get('full_spend_pass') is not True
                or receipt.get('summary', {}).get('replay_max_error') != 0):
            raise ValueError('Reused A evaluation gates failed')
        expected = sha(ROOT/'conditions'/job['domain']/(job['condition']+'.json')) if job.get('condition') else None
        if (receipt.get('condition_identity') or {}).get('config_sha256') != expected:
            raise ValueError('Reused A evaluation condition identity mismatch')
        if receipt.get('configuration', {}).get('path_master') != plan['final_master']:
            raise ValueError('Reused A evaluation namespace differs from registration')
        for name, expected_sha in receipt.get('artifacts_sha256', {}).items():
            artifact = (output/name).resolve()
            if not artifact.is_relative_to(output) or sha(artifact) != expected_sha:
                raise ValueError('Reused A evaluation artifact identity mismatch')
        return receipt
    if receipt.get('source_unchanged') is not True or protocol.get('source_sha256') != plan['source_sha256']:
        raise ValueError('Native receipt is not bound to the frozen source')
    if receipt.get('protocol_sha256') != sha(output/'protocol.json'):
        raise ValueError('Native protocol identity changed')
    arguments = protocol['arguments']
    if arguments.get('device') != 'cuda' or arguments.get('storage_profile') != 'lean':
        raise ValueError('Scope worker must use the registered CUDA lean contract')
    statistics = json.loads((ROOT/'runtime_statistics'/(job['id']+'.json')).read_text())
    if (statistics.get('pid') != state['pid']
            or statistics.get('cuda_visible_devices') != state['gpu_uuid']
            or statistics.get('max_memory_allocated', 0) <= 0):
        raise ValueError('Worker has no matching actual CUDA execution evidence')
    validation = receipt.get('evaluation_validation', {})
    paths = validation.get('evaluated_paths', 0)
    if paths <= 0 or validation.get('native_replayed_paths') != paths or validation.get('full_spend_checked_paths') != paths:
        raise ValueError('Every native evaluation path must pass replay and full spend')
    if job['kind'] in ('gate', 'diagnostic', 'main'):
        if (any(receipt.get(key) != job[key] for key in ('method', 'domain', 'seed'))
                or any(arguments.get(key) != job[key] for key in ('method', 'domain', 'seed'))):
            raise ValueError('Training receipt job identity mismatch')
        if (receipt.get('training_steps') != job['steps'] or arguments.get('steps') != job['steps']
                or receipt.get('accounting', {}).get('environment_steps') != job['steps']
                or arguments.get('path_master') != job['master']
                or receipt.get('engineering_gate') != (job['kind'] == 'gate')):
            raise ValueError('Training budget, namespace or engineering status differs from registration')
        accounting = receipt['accounting']
        actor_updates = accounting.get('policy_updates') if job['method'] == 'c_ppo' else accounting.get('actor_updates')
        if not isinstance(actor_updates, (int, float)) or actor_updates <= 0:
            raise ValueError('Actual actor optimizer updates are required before a gate or science job is complete')
        if receipt.get('endpoint_reload_gate', {}).get('status') != 'PASS':
            raise ValueError('Native fixed endpoint save/load equivalence failed')
        for key, value in state.get('effective_overrides', job.get('config', {})).items():
            if protocol['effective_config'].get(key) != value:
                raise ValueError('Native effective configuration differs from registered override '+key)
        for name, entry in receipt['model_files'].items():
            model = (output/'model'/name).resolve()
            if not model.is_relative_to(output/'model') or sha(model) != entry['sha256']:
                raise ValueError('Fixed endpoint model identity mismatch')
    elif job['kind'] == 'evaluate_external':
        parent = json.loads((Path(state['parent_output'])/'completed.json').read_text())
        if (any(receipt.get(key) != job[key] for key in ('method', 'domain'))
                or any(parent.get(key) != job[key] for key in ('method', 'domain', 'seed'))
                or receipt.get('learning_environment_steps') != 0 or receipt.get('summary', {}).get('paths') != 128
                or receipt.get('all_native_replays_exact') is not True
                or arguments.get('phase') != 'final' or arguments.get('path_master') != plan['final_master']
                or receipt.get('training_completion_sha256') != sha(Path(state['parent_output'])/'completed.json')):
            raise ValueError('Independent evaluation differs from its registered fixed endpoint or paths')
        expected = sha(ROOT/'conditions'/job['domain']/(job['condition']+'.json')) if job.get('condition') else None
        if receipt.get('condition_file_sha256') != expected:
            raise ValueError('Evaluation condition identity mismatch')
    return receipt


def make_command(job, state, plan):
    out = ROOT/'jobs'/job['id']/('attempt%d' % state.get('attempt', 0))
    state['output'] = str(out)
    if job['kind'] == 'train_A':
        return [PYTHON, str(ROOT/'execution/a_native_formal_worker.py'),
                '--source-root', str(SOURCE), '--domain', job['domain'], '--arm', job['arm'],
                '--seed', str(job['seed']), '--steps', str(job['steps']), '--eval-every', '512',
                '--development-paths', '8', '--final-paths', str(plan['final_paths']),
                '--path-master', str(job['master']), '--final-master', str(plan['final_master']),
                '--data-root', str(DATA), '--output', str(out)]
    if job['kind'] in ('gate', 'diagnostic', 'main'):
        config = dict(job.get('config', {}))
        if job.get('selection_method'):
            selected = json.loads((ROOT/'selections'/f"{job['method']}.json").read_text())
            config['initial_total_std'] = selected['selected_initial_total_std']
        state['effective_overrides'] = config
        args = ['train', '--method', job['method'], '--domain', job['domain'],
                '--seed', str(job['seed']), '--steps', str(job['steps']),
                '--eval-every', '512', '--development-paths', str(job.get('development_paths', 8)),
                '--path-master', str(job['master']), '--config', json.dumps(config),
                '--storage-profile', 'lean', '--device', 'cuda', '--cpu-threads', '1',
                '--data-root', str(DATA), '--output', str(out)]
        if job['kind'] == 'gate':
            args.append('--engineering-gate')
        return [PYTHON, str(ROOT/'execution/worker.py'), '--job-id', job['id'], *args]
    if job['kind'] == 'evaluate_external':
        args = ['evaluate', '--run', state['parent_output'], '--output', str(out),
                '--paths', '128', '--phase', 'final', '--path-master', str(plan['final_master']),
                '--storage-profile', 'lean', '--device', 'cuda', '--cpu-threads', '1',
                '--data-root', str(DATA)]
        if job.get('condition'):
            args += ['--condition-file', str(ROOT/'conditions'/job['domain']/(job['condition']+'.json'))]
        return [PYTHON, str(ROOT/'execution/worker.py'), '--job-id', job['id'], *args]
    if job['kind'] == 'evaluate_A':
        args = [PYTHON, str(ROOT/'execution/evaluate_reused_A.py'), '--run', str(ROOT/'reused_A'/job['reuse_id']),
                '--output', str(out), '--paths', '128', '--path-master', str(plan['final_master']),
                '--data-root', str(DATA), '--source-root', str(SOURCE)]
        if job.get('condition'):
            args += ['--condition-file', str(ROOT/'conditions'/job['domain']/(job['condition']+'.json'))]
        return args
    raise ValueError(job['kind'])


def selections(plan, states):
    # Formal configuration is fixed before launch; no retuning on final tests.
    return


def cpu_usage():
    p = Path('/sys/fs/cgroup/cpuacct/cpuacct.usage')
    return int(p.read_text()) / 1e9


def available_cpu_cores(active):
    def own_cpu():
        total = 0.
        ticks = os.sysconf('SC_CLK_TCK')
        for state in active:
            try:
                f = Path('/proc',str(state['pid']),'stat').read_text().rsplit(')',1)[1].split()
                if f[19] == state['start_ticks']:
                    total += (int(f[11])+int(f[12]))/ticks
            except (FileNotFoundError, ProcessLookupError):
                pass
        return total
    before, own_before, stamp = cpu_usage(), own_cpu(), time.monotonic()
    time.sleep(.3)
    elapsed = time.monotonic()-stamp
    used = (cpu_usage()-before)/elapsed
    own_used = max(0.,(own_cpu()-own_before)/elapsed)
    other_used = max(0.,used-own_used)
    quota = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
    period = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
    # Every accepted single-thread worker retains a full core reservation even
    # during CUDA waits or initial import. Other scopes are read-only observations.
    return max(0, int(quota/period - other_used - len(active) - 1.5))


def child_exit_reaped(process, state, clock=time.time):
    """Wait for the owned child's real status. None is never a nonzero exit.

    /proc can already report Z or disappear after an earlier poll returned
    None. Retain the same running reservation until a fresh poll reaps it.
    No validation, failure, signal or duplicate launch occurs while pending.
    Adopted workers without a Popen keep the previous receipt-audit path.
    """
    if process is None:
        return True
    exitcode = process.poll()
    if exitcode is None:
        state['exit_reap_pending'] = dict(
            reason='owned child not yet reaped; retain running reservation',
            observed=clock())
        return False
    state.pop('exit_reap_pending', None)
    state['observed_child_exitcode'] = exitcode
    return True


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((ROOT/'plan.json').read_text())
    assert plan['execution_scope'] == str(ROOT)
    for relative, expected in plan['source_sha256'].items():
        if sha(SOURCE/relative) != expected:
            raise ValueError('Frozen source mismatch '+relative)
    for relative, expected in plan.get('full_source_sha256', {}).items():
        if sha(SOURCE/relative) != expected:
            raise ValueError('Full frozen source mismatch '+relative)
    for relative, expected in plan.get('execution_sha256', {}).items():
        if sha(ROOT/'execution'/relative) != expected:
            raise ValueError('Frozen execution mismatch '+relative)
    statefile = ROOT/'dispatch_state.json'
    states = json.loads(statefile.read_text()) if statefile.exists() else {}
    # An interrupted spawn is deliberately not retried: an unrecorded child
    # may still exist, so a fresh launch needs an explicit identity audit.
    if any(state.get('phase') == 'spawning' for state in states.values()):
        raise RuntimeError('Interrupted scope worker spawn requires an identity audit; refusing a duplicate launch')
    for job in plan['jobs']:
        state = states.get(job['id'], {})
        if state.get('phase') == 'complete':
            validated_completion(job, state, plan)
            if state.get('receipt_sha256') != sha(Path(state['output'])/'completed.json'):
                raise ValueError('Previously completed scope receipt changed: '+job['id'])
    write(ROOT/'queue_pid.json', dict(pid=os.getpid(), boot_id=BOOT, **identity(os.getpid())))
    write(ROOT/'recovery_launch.json', dict(status='running', command=[PYTHON, str(CAPSULE/'launch_guard.py')],
                                           adopts_only_identity_verified_scope_workers=True))
    children = {}
    while True:
        for job in plan['jobs']:
            state = states.setdefault(job['id'], dict(phase='pending', attempt=0))
            if state['phase'] != 'running':
                continue
            process = children.get(job['id'])
            exitcode = process.poll() if process is not None else None
            if process is not None and exitcode is not None:
                alive = False
            else:
                try:
                    alive = scope_worker_alive(state)
                except RuntimeError as error:
                    state.update(phase='needs_identity_review', identity_error=str(error),
                                 identity_review_requested=time.time())
                    write(ROOT/'identity_reviews'/(job['id']+'.json'),
                          dict(job_id=job['id'], pid=state['pid'], boot_id=state['boot_id'],
                               start_ticks=state['start_ticks'], expected_command=state['command'],
                               observed_identity=identity(state['pid']), error=str(error),
                               action='reserved and isolated; no signal, relaunch or automatic adoption'))
                    continue
            if alive is None:
                state['identity_pending'] = dict(reason='same boot/start but cmdline empty; await natural exit',
                                                 observed=time.time())
                continue
            state.pop('identity_pending', None)
            if alive:
                continue
            if not child_exit_reaped(process, state):
                continue
            completed = Path(state['output'])/'completed.json'
            if completed.exists():
                try:
                    if process is not None and process.returncode != 0:
                        raise ValueError('Worker returned a nonzero exit status after writing a receipt')
                    receipt = validated_completion(job, state, plan)
                    state.update(phase='complete', finished=time.time(), receipt_sha256=sha(completed))
                    continue
                except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as error:
                    state['validation_error'] = repr(error)
            state.update(phase='failed', finished=time.time(), exitcode=process.returncode if process else None)
        selections(plan, states)
        active = [s for s in states.values() if s['phase'] in ('running', 'needs_identity_review')]
        cpu_slots = available_cpu_cores(active)
        for job in sorted(plan['jobs'], key=lambda j: (j['priority'], j['id'])):
            state = states[job['id']]
            if state['phase'] != 'pending':
                continue
            if not all(states.get(dep, {}).get('phase') == 'complete' for dep in job.get('depends', [])):
                continue
            if job.get('selection_method') and not (ROOT/'selections'/f"{job['method']}.json").exists():
                continue
            if job['kind'] == 'evaluate_A' and not (ROOT/'execution/evaluate_reused_A.py').exists():
                continue
            if cpu_slots <= 0:
                break
            if len(active) >= plan['max_workers']:
                break
            free_disk = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
            if free_disk < 2*1024**3 + (len(plan['jobs'])-sum(s['phase']=='complete' for s in states.values()))*16*1024**2:
                continue
            if os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') < 4*1024**3:
                continue
            uuid = None
            if job['cuda']:
                free = gpu_free()
                reservations = {u: sum(s['reserve_mib'] for s in active if s.get('gpu_uuid') == u) for u in plan['gpu_uuids']}
                options = [u for u in plan['gpu_uuids'] if free.get(u, 0)-reservations[u] >= job['reserve_mib']+1536]
                if not options:
                    continue
                uuid = max(options, key=lambda u: free[u]-reservations[u])
            if job.get('parent_job'):
                state['parent_output'] = states[job['parent_job']]['output']
            command = make_command(job, state, plan)
            env = dict(os.environ, PYTHONPATH=str(SOURCE), PYTHONDONTWRITEBYTECODE='1',
                       ALLOCATION23_DATA_ROOT=str(DATA), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                       CUDA_VISIBLE_DEVICES=uuid if uuid else '',
                       CH3_SCOPE_ROOT=str(ROOT))
            for key, leaf in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('CUDA_CACHE_PATH','cuda'),
                              ('MPLCONFIGDIR','mpl'),('TORCH_HOME','torch'),('TORCH_EXTENSIONS_DIR','extensions')]:
                directory = ROOT/'runtime_cache'/job['id']/leaf
                directory.mkdir(parents=True, exist_ok=True)
                env[key] = str(directory)
            logfile = ROOT/'logs'/(job['id']+'.log')
            logfile.parent.mkdir(exist_ok=True)
            with logfile.open('ab') as stream:
                state.update(phase='spawning', command=command, spawn_requested=time.time())
                write(statefile, states)
                process = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
            record = identity(process.pid)
            if record is None:
                state.update(phase='failed', issued_pid=process.pid,
                             error='Worker exited before identity capture; no automatic retry')
                continue
            state.update(phase='running', pid=process.pid, boot_id=BOOT, start_ticks=record['start_ticks'],
                         gpu_uuid=uuid, reserve_mib=job['reserve_mib'], command=command, started=time.time())
            children[job['id']] = process
            active.append(state)
            cpu_slots -= 1
            write(statefile, states)
            time.sleep(.15)
        counts = {phase:sum(s['phase']==phase for s in states.values())
                  for phase in ('pending','running','complete','failed','needs_identity_review')}
        progress = []
        for job in plan['jobs']:
            state = states[job['id']]
            row = dict(id=job['id'], kind=job['kind'], phase=state['phase'], output=state.get('output'), gpu_uuid=state.get('gpu_uuid'))
            if state.get('output'):
                f=Path(state['output'])/'progress.json'
                try:
                    row['worker_progress']=json.loads(f.read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
            progress.append(row)
        write(statefile, states)
        write(ROOT/'status.json', dict(updated=time.time(), counts=counts, jobs=progress,
                                      phase='running' if counts['running'] else 'complete' if counts['complete']==len(plan['jobs']) else 'needs_review',
                                      training_jobs=10, engineering_jobs=0, evaluation_jobs=10,
                                      no_old_queues_restored=True,
                                      dispatcher_sidecar=str(Path(__file__).resolve())))
        if counts['complete'] == len(plan['jobs']):
            write(ROOT/'completion_receipt.json', dict(status='complete', jobs=len(plan['jobs']),
                  training_jobs=10, engineering_jobs=0, evaluation_jobs=10, plan_sha256=sha(ROOT/'plan.json')))
            write(ROOT/'recovery_launch.json', dict(status='complete', command=None, automatic_extra_training=False))
            return
        if not counts['running'] and (counts['failed'] or counts['needs_identity_review']):
            write(ROOT/'recovery_launch.json', dict(status='needs_review', command=None, automatic_retry=False,
                                                   reason='Preserve failures and identity reservations; no silent retries or parameter changes'))
            return
        time.sleep(3)


if __name__ == '__main__':
    try:
        run()
    except Exception:
        write(ROOT/'controller_failure.json', dict(time=time.time(), traceback=traceback.format_exc()))
        raise
