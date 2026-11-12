"""Run one exact registered command and seal successful scientific artifacts."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    plan = json.loads((ROOT/'plan.json').read_text())
    plan_sha = sha(ROOT/'plan.json')
    seal = json.loads((ROOT/'execution_seal.json').read_text())
    if plan_sha != seal['plan_sha256']:
        raise ValueError('Plan seal mismatch')
    for name, digest in seal['execution_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Execution capsule changed '+name)
    job, = [job for job in plan['jobs'] if job['id'] == args.job_id]
    output = Path(job['output'])
    if output.exists():
        raise FileExistsError('Refuse to overwrite or silently resume '+str(output))
    expected_uuid = job.get('gpu_uuid') or '-1'
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_uuid:
        raise ValueError('Registered physical CUDA UUID visibility mismatch')
    for name in ('TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','CUDA_CACHE_PATH',
                 'TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):
        cache = ROOT/'runtime_cache'/job['id']/name.lower()
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(cache)
    started = time.time()
    process = subprocess.Popen(job['command'], env=os.environ.copy())
    write(ROOT/'logs'/(job['id']+'.child.json'), dict(pid=process.pid, parent=os.getpid(),
          command=job['command'], gpu_uuid=job.get('gpu_uuid'), started=started))
    code = process.wait()
    if code:
        raise RuntimeError('Scientific worker exit '+str(code))
    completed = json.loads((output/'completed.json').read_text())
    if completed.get('success') is not True:
        raise ValueError('Scientific success missing')
    for key in ('domain','seed','method'):
        if completed.get(key) != job[key]:
            raise ValueError('Scientific job identity mismatch '+key)
    if completed.get('source_unchanged') is not True:
        raise ValueError('Scientific source immutability gate missing')
    for name, digest in completed.get('artifacts_sha256', {}).items():
        path = (output/name).resolve()
        if not path.is_relative_to(output.resolve()) or sha(path) != digest:
            raise ValueError('Native sealed artifact mismatch '+name)
    if job['kind'] == 'train_direct':
        if completed.get('training_steps') != job['steps']:
            raise ValueError('Learning step budget mismatch')
        if completed.get('actual_device') not in ('cuda', 'cuda:0'):
            raise ValueError('Direct training is not registered CUDA execution')
        if completed.get('maximum_resource_fraction_error', 1.) > 1e-9:
            raise ValueError('Training full-spend check failed')
        if not (output/'model/state.pt').is_file():
            raise ValueError('One terminal Direct model missing')
        if (completed.get('profile') == 'engineering_gate'
                or completed.get('endpoint_reload_pass') is not True
                or completed.get('actor_changed') is not True
                or completed.get('native_replay_error') != 0.
                or completed.get('accounting', {}).get('actor_update_calls', 0) <= 0
                or completed.get('final_evaluation', {}).get('paths') != 128):
            raise ValueError('Formal learning/update/reload/final gates failed')
        if sha(output/'model/state.pt') != completed['model_sha256']:
            raise ValueError('Direct endpoint digest mismatch')
    else:
        if completed.get('paths') != job['paths']:
            raise ValueError('Evaluation path budget mismatch')
        if not completed.get('all_full_spend') or not completed.get('all_native_replays_exact'):
            raise ValueError('Evaluation full-spend/native replay failed')
        if not math.isfinite(completed['native_mean']):
            raise ValueError('Nonfinite evaluation')
        if (completed.get('engineering_gate') is not False
                or completed.get('learning_environment_steps') != 0
                or completed.get('policy_and_normalization_unchanged') is not True):
            raise ValueError('Frozen formal evaluation gates failed')
        if completed.get('condition') != job['condition']:
            raise ValueError('Evaluation condition mismatch')
    artifacts = {}
    for path in output.rglob('*'):
        if not path.is_file() or 'runtime' in path.relative_to(output).parts:
            continue
        relative = path.relative_to(output).as_posix()
        if path.suffix == '.gz':
            with gzip.open(path,'rb') as stream:
                for _ in iter(lambda: stream.read(2**20), b''):
                    pass
        artifacts[relative] = sha(path)
    forbidden = [name for name in artifacts if any(token in name.lower() for token in
                 ('optimizer.pt','replay.pt','periodic_checkpoint'))]
    if forbidden:
        raise ValueError('Unexpected disk training state '+str(forbidden))
    write(output/'job_receipt.json', dict(success=True, job_id=job['id'], kind=job['kind'],
          method=job['method'], domain=job['domain'], seed=job['seed'],
          plan_sha256=plan_sha, execution_sha256=seal['execution_sha256'],
          artifacts_sha256=artifacts, started=started, ended=time.time(),
          gpu_uuid=job.get('gpu_uuid'), native_completed_sha256=artifacts['completed.json']))


if __name__ == '__main__':
    main()
