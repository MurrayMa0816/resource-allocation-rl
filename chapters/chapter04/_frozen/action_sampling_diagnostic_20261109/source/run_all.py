"""Run each predeclared endpoint once; never resume or retry failed evaluations."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def main():
    if (ROOT/'runtime/launch.json').exists() or (ROOT/'jobs').exists():
        raise FileExistsError('This diagnostic has already been launched; do not repeat endpoints.')
    review = json.loads((ROOT/'audit/preflight.json').read_text())
    if review['status'] != 'PASS_FOR_DIAGNOSTIC_EXECUTION':
        raise RuntimeError('Independent preflight has not passed')
    code_seal = json.loads((ROOT/'protocol/execution_seal.json').read_text())
    for path, expected in code_seal['source_sha256'].items():
        assert sha(path) == expected, path
    runtime = json.loads(Path('/hy-tmp/resource_allocation_results/chapter04/experiments/double_random_formal_20261107/protocol/runtime_environment.json').read_text())
    environment = os.environ.copy()
    environment.update(runtime['env'])
    environment.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='disabled')
    python = runtime['python']
    started = time.monotonic()
    launch = dict(status='running', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  pid=os.getpid(), scope='DEV endpoint execution diagnostic; zero training',
                  python=python, planned_workers=10, source_seal_sha256=sha(ROOT/'protocol/execution_seal.json'),
                  preflight_sha256=sha(ROOT/'audit/preflight.json'), recovery_command=None)
    dump(ROOT/'runtime/launch.json', launch)
    processes = []
    streams = []
    for method in ('f1', 'props'):
        for seed in range(5):
            stream = (ROOT/'logs'/f'{method}__v5__seed{seed}.log').open('x')
            streams.append(stream)
            command = [python, str(ROOT/'source/evaluate_endpoint.py'), '--method', method, '--seed', str(seed)]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       cwd=ROOT, env=environment)
            processes.append((method, seed, process))
    while True:
        states = [dict(method=method, seed=seed, pid=process.pid, returncode=process.poll())
                  for method, seed, process in processes]
        dump(ROOT/'runtime/status.json', dict(workers=states, completed=sum(s['returncode'] == 0 for s in states),
              failed=sum(s['returncode'] not in (None, 0) for s in states),
              running=sum(s['returncode'] is None for s in states), elapsed_seconds=time.monotonic()-started))
        if all(state['returncode'] is not None for state in states):
            break
        time.sleep(2)
    for stream in streams:
        stream.close()
    launch.update(status='complete' if all(s['returncode'] == 0 for s in states) else 'failed',
                  elapsed_seconds=time.monotonic()-started, workers=states)
    dump(ROOT/'runtime/launch.json', launch)
    if launch['status'] != 'complete':
        raise RuntimeError('At least one worker failed; preserve attempt and inspect costs. No automatic retry.')
    print(json.dumps(launch), flush=True)


if __name__ == '__main__':
    main()
