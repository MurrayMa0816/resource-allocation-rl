"""Launch the five fixed runs once and observe them without automatic retries."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import time
import datetime

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    temp = path.with_name(path.name + '.writing')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def main():
    lock = (ROOT/'runtime/formal_controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT/'runtime/formal_launch.json').exists() or any((ROOT/'jobs').iterdir()):
        raise FileExistsError('This fixed cohort has already started; inspect its existing state, never duplicate it.')
    gate = json.loads((ROOT/'audit/formal_start_gate.json').read_text())
    if gate['status'] != 'PASS':
        raise RuntimeError('Engineering and independent gates must pass before formal execution')
    seal = json.loads((ROOT/'protocol/execution_seal.json').read_text())
    worker = ROOT/'source/worker.py'
    assert sha(worker) == seal['worker_sha256']
    assert sha(ROOT/'protocol/PLAN.json') == seal['plan_sha256']
    protected = json.loads((ROOT/'protocol/protected_original_inputs.json').read_text())['files']
    for record in protected:
        assert sha(record['path']) == record['sha256'], record['path']
    runtime = json.loads(Path('/hy-tmp/resource_allocation_results/chapter04/experiments/double_random_formal_20261107/protocol/runtime_environment.json').read_text())
    started = time.monotonic()
    processes = []
    streams = []
    for seed in range(5):
        device = 'cuda:0' if seed == 2 else 'cpu'
        output = ROOT/'jobs'/f'props__v5__seed{seed}'
        environment = os.environ.copy()
        environment.update(runtime['env'])
        environment['CUDA_VISIBLE_DEVICES'] = 'GPU-a3130b13-8288-caaa-0f96-f2916c92d8e3' if seed == 2 else ''
        environment.update(PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='disabled')
        for key in ('TMPDIR', 'XDG_CACHE_HOME', 'MPLCONFIGDIR', 'TORCH_HOME', 'CUDA_CACHE_PATH'):
            path = ROOT/'runtime/bootstrap'/f'formal_seed{seed}'/key.lower()
            path.mkdir(parents=True, exist_ok=True)
            environment[key] = str(path)
        command = [runtime['python'], str(worker), '--seed', str(seed), '--device', device,
                   '--output', str(output), '--plan-sha256', seal['plan_sha256']]
        stream = (ROOT/'logs'/f'formal_props__v5__seed{seed}.log').open('x')
        streams.append(stream)
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        processes.append((seed, process, command))
    launch = dict(schema='props-stochastic-curve-formal-launch-v1', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  controller_pid=os.getpid(), controller_sha256=sha(__file__), worker_sha256=sha(worker),
                  plan_sha256=seal['plan_sha256'], formal_gate_sha256=sha(ROOT/'audit/formal_start_gate.json'),
                  workers=[dict(seed=seed, pid=process.pid, command=command) for seed, process, command in processes],
                  automatic_retry=False, old_queues_touched=False)
    dump(ROOT/'runtime/formal_launch.json', launch)
    dump(ROOT/'runtime/recovery.json', dict(status='running', command=None,
         note='Existing processes are observed. Do not relaunch this fixed cohort or recover any historical queue.'))
    last_print = -60.
    while True:
        rows = []
        for seed, process, command in processes:
            output = ROOT/'jobs'/f'props__v5__seed{seed}'
            progress = {}
            if (output/'progress.json').exists():
                try:
                    progress = json.loads((output/'progress.json').read_text())
                except json.JSONDecodeError:
                    pass
            rows.append(dict(seed=seed, pid=process.pid, returncode=process.poll(),
                             phase=progress.get('phase', 'initializing'),
                             training_steps=progress.get('training_steps', 0),
                             completed_nodes=progress.get('completed_nodes', 0),
                             physical_counts=progress.get('physical_counts', {}),
                             worker_elapsed_seconds=progress.get('elapsed_seconds', 0)))
        status = dict(status='running' if any(r['returncode'] is None for r in rows) else
                      ('completed' if all(r['returncode'] == 0 for r in rows) else 'failed'),
                      elapsed_seconds=time.monotonic()-started, workers=rows,
                      completed=sum(r['returncode'] == 0 for r in rows),
                      failed=sum(r['returncode'] not in (None, 0) for r in rows),
                      running=sum(r['returncode'] is None for r in rows))
        dump(ROOT/'runtime/status.json', status)
        if status['elapsed_seconds']-last_print >= 60 or status['status'] != 'running':
            print(json.dumps(status, ensure_ascii=False), flush=True)
            last_print = status['elapsed_seconds']
        if status['status'] != 'running':
            break
        time.sleep(2)
    for stream in streams:
        stream.close()
    dump(ROOT/'runtime/recovery.json', dict(status=status['status'], command=None, automatic_retry=False))
    if status['status'] != 'completed':
        raise RuntimeError('A fixed run failed; preserved partial costs require review. No automatic retries were started.')


if __name__ == '__main__':
    main()
