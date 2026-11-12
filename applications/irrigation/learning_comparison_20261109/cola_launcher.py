#!/usr/bin/env python3
"""Explicit COLA engineering/formal launch; never retries a failed/old output."""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, subprocess, sys, time

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP)); sys.dont_write_bytecode = True
from paths import STUDY as BASE
STUDY = BASE / 'additional_cola_20261109'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def get(path): return json.loads(Path(path).read_text())
def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'); tmp.replace(path)
def index_path(phase): return STUDY / 'protocol' / (phase + '_attempt0_jobs.json')


def config_paths(phase): return get(index_path(phase))['configs']


def inspect(phase):
    records = []
    if index_path(phase).exists():
        for path in config_paths(phase):
            config = get(path); output = Path(config['output'])
            progress = get(output/'progress.json') if (output/'progress.json').exists() else dict(status='NOT_STARTED')
            if (output/'failed.json').exists(): progress = get(output/'failed.json')
            records.append(dict(job=config['job_id'], config=path, output=str(output), progress=progress))
    return dict(study=str(STUDY), phase=phase, method='COLA', jobs=records, automatic_retries=False)


def engineering_gate():
    output = STUDY / 'engineering/cola__seed7901/attempt0'
    if (output/'failed.json').exists(): raise ValueError('Failed engineering attempt retained; no formal launch')
    receipt = get(output/'completed.json')
    if (receipt['status'] != 'COMPLETE' or receipt['learning_steps'] != 5184 or receipt['episodes'] != 64 or
            not receipt['endpoint_reload_exact'] or not receipt['source_unchanged'] or
            receipt['device_proof']['cuda_optimizer_tensors'] < 1):
        raise ValueError('COLA physical CUDA engineering prerequisite not satisfied')
    return receipt


def validate(phase, require_engineering=True):
    paths = config_paths(phase); configs = [get(path) for path in paths]
    expected = {7901} if phase == 'engineering' else {6400, 6401, 6402}
    if len(configs) != len(expected) or {c['seed'] for c in configs} != expected:
        raise ValueError('Wrong COLA stage roster')
    for config in configs:
        engine = phase == 'engineering'; total = 5184 if engine else 259200
        native = config['algorithm_config']
        if (config['method'] != 'cola' or config['engineering'] != engine or config['learning_steps'] != total or
                config['episodes'] != (64 if engine else 3200) or config['dev_paths'] != (2 if engine else 32) or
                config['eval_every'] != (5184 if engine else 10368) or config['device'] != 'cuda:0' or
                native['n_step'] != 8 or native['oadm_one_step_target'] is not True or native['tau'] != .002 or
                native['learning_steps'] != total or native['horizon'] != 81 or native['deployment_preference'] != .2):
            raise ValueError('COLA registered task/native configuration changed')
        if not Path(config['output']).resolve().is_relative_to(STUDY.resolve()): raise ValueError('Foreign output root')
        for path, value in config['source_files_sha256'].items():
            if sha(path) != value: raise ValueError('Changed source/input: ' + path)
    if len({c['gpu_uuid'] for c in configs}) != 1: raise ValueError('COLA jobs must use one registered GPU UUID')
    if phase == 'formal' and require_engineering: engineering_gate()
    return configs


def run(phase):
    directory = STUDY / ('engineering_controller' if phase == 'engineering' else 'training')
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    launch = directory/'launch.json'
    if launch.exists(): raise FileExistsError('COLA phase already launched; no automatic retry/recovery')
    configs = validate(phase)
    for config in configs:
        if Path(config['output']).exists(): raise FileExistsError('Existing output retained: ' + config['output'])
    children = []; records = []
    # Reserve the launch before spawning. A partial launch is never retried.
    write(launch, dict(status='LAUNCH_RESERVED', phase=phase, controller_pid=os.getpid(),
                      started_epoch=time.time(), automatic_retries=False, jobs=[]))
    for config, path in zip(configs, config_paths(phase)):
        logfile = directory/(config['job_id']+'.log'); handle = logfile.open('w')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=config['gpu_uuid'], PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        process = subprocess.Popen([sys.executable, '-B', config['worker'], '--config', path],
                                   stdout=handle, stderr=subprocess.STDOUT, env=env)
        children.append((process, handle)); records.append(dict(job=config['job_id'], pid=process.pid,
            config=path, config_sha256=sha(path), output=config['output'], log=str(logfile), gpu_uuid=config['gpu_uuid']))
        write(launch, dict(status='LAUNCHED', phase=phase, controller_pid=os.getpid(),
                          jobs=records, maximum_concurrency=len(configs), all_jobs_launched=len(records)==len(configs),
                          automatic_retries=False, recovery=dict(command=None)))
    while True:
        exits = {r['job']: p.poll() for r,(p,h) in zip(records,children)}
        state = inspect(phase); state.update(controller_pid=os.getpid(), process_exit_codes=exits,
                                             status='RUNNING' if any(v is None for v in exits.values()) else 'FINISHED')
        write(directory/'status.json', state)
        if all(v is not None for v in exits.values()):
            for process, handle in children: handle.close()
            passed = all(v == 0 for v in exits.values()) and all((Path(r['output'])/'completed.json').exists() for r in records)
            write(directory/'controller_completed.json', dict(status='COMPLETE' if passed else 'FAILED_JOBS_RETAINED',
                phase=phase, jobs=records, process_exit_codes=exits, automatic_retries=False,
                ended_epoch=time.time(), recovery=dict(command=None)))
            return 0 if passed else 2
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=['prepare','inspect','check','train'], default='inspect')
    parser.add_argument('--phase', choices=['engineering','formal'], required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        from execution.prepare_cola import prepare
        print(json.dumps(prepare(args.phase), ensure_ascii=False, indent=2))
    elif args.command == 'train': return run(args.phase)
    elif args.command == 'check':
        print(json.dumps(dict(status='PASS', phase=args.phase, validated_jobs=len(validate(args.phase))), indent=2))
    else: print(json.dumps(inspect(args.phase), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__': sys.exit(main())
