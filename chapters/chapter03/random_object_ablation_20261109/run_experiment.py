"""One fixed two-domain random-object ablation; no retries or old-file writes."""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

PYTHON = str(_resolve_path('/root/gamewar/resource_allocation_rl/python'))
ARM = 'A_RANDOM_OBJECT_ALLOCATION'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, data):
    p = Path(p)
    t = p.with_name(p.name + '.tmp')
    t.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    t.replace(p)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    pp = Path(args.plan).resolve(strict=True)
    plan = read(pp)
    digest = sha(pp)
    root = Path(plan['experiment_root']).resolve(strict=True)
    if digest != (root/'registration/plan.sha256').read_text().strip():
        raise ValueError('Plan changed')
    if (plan['arm'] != ARM or plan['steps'] != 100608 or plan['seeds'] != list(range(5))
            or plan['domains'] != ['radio', 'v5'] or plan['eval_every'] != 512
            or plan['development_paths'] != 8 or plan['final_paths'] != 0
            or plan['device'] != 'cpu' or plan['max_parallel'] != 10):
        raise ValueError('Only the fixed ten CPU endpoints with original DEV protocol are allowed')
    for name in ['adapter_static_verification.json', 'implementation_independent_review.json',
                 'registration_validation.json']:
        if read(root/'audit'/name)['status'] != 'PASS':
            raise ValueError('Missing pre-execution check: ' + name)
    identity = read(root/'registration/execution_identity.json')['files']
    protected = read(root/'registration/protected_before.json')['files']

    def guard(original=False):
        if sha(pp) != digest:
            raise ValueError('Plan changed during execution')
        for name, item in identity.items():
            if sha(item['execution']) != item['sha256']:
                raise ValueError('Execution source changed: ' + name)
        if original:
            for p, h in protected.items():
                if sha(p) != h:
                    raise ValueError('Original protected evidence changed: ' + p)

    guard(original=True)
    with os.fdopen(os.open(root/'controller.pid', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), 'w') as f:
        f.write(str(os.getpid()) + '\n')
    jobs, children = [], []
    start = time.time()

    def status(phase, **extra):
        data = []
        for job in jobs:
            item = dict(job)
            p = Path(job['output'])/'progress.json'
            if p.exists():
                try:
                    item['progress'] = read(p)
                except (OSError, ValueError):
                    pass
            data.append(item)
        write(root/'status.json', dict(status=phase, elapsed_seconds=time.time()-start,
            controller_pid=os.getpid(), jobs=data, new_independent_TEST_paths=0, **extra))

    def command(job, engineering):
        cmd = [PYTHON, '-B', identity['train.py']['execution'], '--source-root', plan['source_root'],
            '--domain', job['domain'], '--arm', ARM, '--device', 'cpu', '--seed', str(job['seed']),
            '--steps', str(job['steps']), '--eval-every', '0' if engineering else '512',
            '--development-paths', '1' if engineering else '8', '--final-paths', '0',
            '--path-master', str(plan['training_master']), '--final-master', str(plan['unused_final_master']),
            '--data-root', plan['data_root'], '--output', job['output']]
        if engineering:
            cmd.append('--engineering-gate')
        return cmd

    def group(phase, records, parallel, engineering):
        jobs.extend(records)
        queue, pending, failures = list(records), [], []
        while queue or pending:
            if failures:
                queue = []
            while queue and len(pending) < parallel:
                job = queue.pop(0)
                if Path(job['output']).exists():
                    raise ValueError('Never reuse an existing output directory')
                cmd = command(job, engineering)
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
                    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                    NUMEXPR_NUM_THREADS='1')
                env.pop('CH3_RANDOM_TIMING_AUDIT_ROOT', None)
                with open(root/'logs'/(job['id']+'.log'), 'wb') as log:
                    child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                        env=env, start_new_session=True)
                job.update(pid=child.pid, started_unix=time.time(), command=cmd)
                pending.append((child, job))
                children.append((child, job))
            alive = []
            for child, job in pending:
                rc = child.poll()
                if rc is None:
                    alive.append((child, job))
                else:
                    job.update(exit_code=rc, finished_unix=time.time())
                    if rc:
                        failures.append(dict(id=job['id'], exit_code=rc))
            pending = alive
            status(phase, active=len(pending), queued=len(queue), failures=failures)
            if pending or queue:
                time.sleep(5)
        guard()
        if failures:
            raise RuntimeError('Failed attempts retained; no automatic retry: ' + str(failures))

    try:
        write(root/'recovery.json', dict(status='RUNNING', command=None,
            policy='No exact optimizer resume or automatic retry; fixed native endpoints'))
        engineering = []
        for e in plan['engineering']:
            name = 'engineering_' + e['domain']
            output = root/'engineering'/name
            engineering.append(dict(id=name, kind='engineering', method=ARM, domain=e['domain'],
                seed=e['seed'], steps=e['steps'], run=str(output), output=str(output)))
        write(root/'registration/engineering_manifest.json', dict(records=engineering))
        group('ENGINEERING', engineering, 2, True)
        for job in engineering:
            d = read(Path(job['output'])/'completed.json')
            if not (d['success'] and d['source_unchanged'] and d['endpoint_reload_pass']
                    and d['training_steps'] == job['steps'] and d['evaluation_steps'] == 0):
                raise ValueError('Engineering acceptance failed')
        write(root/'registration/engineering_manifest.json', dict(records=engineering))
        write(root/'audit/engineering_acceptance.json', dict(status='PASS', jobs=len(engineering),
            physical_environment_steps=sum(e['steps'] for e in engineering), model_payload_loads=len(engineering)))
        training = []
        for domain in plan['domains']:
            for seed in plan['seeds']:
                name = f'{ARM}__{domain}__seed{seed}'
                output = root/'training'/name
                training.append(dict(id=name, kind='training', method=ARM, domain=domain,
                    seed=seed, steps=100608, run=str(output), output=str(output)))
        write(root/'registration/training_manifest.json', dict(records=training))
        group('TRAINING', training, 10, False)
        write(root/'registration/training_manifest.json', dict(records=training))
        for job in training:
            run = Path(job['run'])
            d, p = read(run/'completed.json'), read(run/'protocol.json')
            if not (d['success'] and d['source_unchanged'] and d['endpoint_reload_pass']
                    and d['training_steps'] == 100608 and p['arguments']['final_paths'] == 0
                    and not p['arguments']['engineering_gate']):
                raise ValueError('Training did not complete the exact registered endpoint')
        guard(original=True)
        status('ANALYZING', active=0)
        for name, extra in [('analyze.py', ['--root', str(root), '--output', str(root/'analysis')]),
                            ('plot.py', ['--analysis', str(root/'analysis'), '--output', str(root/'figures')])]:
            with open(root/'logs'/(name+'.log'), 'wb') as log:
                subprocess.run([PYTHON, '-B', identity[name]['execution'], *extra],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
        guard(original=True)
        status('COMPLETE_EXECUTION_AND_ANALYSIS', active=0)
        write(root/'recovery.json', dict(status='COMPLETE', command=None, controller_pid=None))
    except BaseException as exc:
        active = [dict(pid=c.pid, id=j['id']) for c, j in children if c.poll() is None]
        write(root/'controller_failed.json', dict(error=repr(exc), traceback=traceback.format_exc(),
            active_children=active))
        status('FAILED', active=len(active), error=repr(exc))
        write(root/'recovery.json', dict(status='FAILED', command=None, active_children=active))
        raise


if __name__ == '__main__':
    main()
