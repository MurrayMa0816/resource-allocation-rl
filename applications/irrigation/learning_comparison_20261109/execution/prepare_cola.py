"""Prepare a separate immutable COLA phase; never starts a worker or environment."""
from pathlib import Path
import argparse, copy, hashlib, importlib, json, shutil, sys

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP)); sys.dont_write_bytecode = True
from paths import STUDY as BASE, C5, C6

STUDY = BASE / 'additional_cola_20261109'
FULL = Path('/hy-tmp/resource_allocation_results/chapter06/experiments/reasonable_allocation89_20261109')
GPU_UUID = 'GPU-a3130b13-8288-caaa-0f96-f2916c92d8e3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def prepare(phase):
    if phase not in ('engineering', 'formal'):
        raise ValueError('Explicit engineering/formal phase required')
    for child in ('protocol', 'source', 'engineering', 'formal', 'runtime_cache', 'reports', 'audit', 'training'):
        (STUDY / child).mkdir(parents=True, exist_ok=True)
    tag = phase + '_attempt0'; frozen = STUDY / 'source' / tag
    index = STUDY / 'protocol' / (tag + '_jobs.json')
    if frozen.exists() or index.exists():
        raise FileExistsError('COLA phase already prepared; no overwrite/automatic retry')
    module = importlib.import_module('methods.cola_adapter')
    native = module.configuration()
    if (native['n_step'] != 8 or native['oadm_one_step_target'] is not True or
            native['tau'] != .002 or native['deployment_preference'] != .2 or native['horizon'] != 81):
        raise ValueError('COLA fixed_oadm/v5 task contract differs')
    worker = APP / 'execution/cola_worker.py'
    if not worker.is_file(): raise FileNotFoundError(worker)
    # The complete maintenance Python closure is copied into a distinct source
    # tree. Existing CAPQL/PASTA scientific snapshots and outputs are untouched.
    frozen.mkdir()
    for path in APP.rglob('*.py'):
        if '__pycache__' in path.parts: continue
        dest = frozen / path.relative_to(APP)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    deps = {str(path): sha(path) for root in (C5, C6) for path in root.rglob('*')
            if path.is_file() and path.suffix in ('.py', '.json', '.csv')}
    originals = {}
    for seed in (6400, 6401, 6402):
        path = FULL / f'formal/training/budget_4000000__seed{seed}/attempt0/protocol.json'
        value = json.loads(path.read_text())
        if value['seed'] != seed or value['episodes'] != 3200 or len(value['train_weather_seeds']) != 3200 or len(value['dev_weather_seeds']) != 32:
            raise ValueError('Original Full weather protocol differs')
        originals[seed] = value; deps[str(path)] = sha(path)
    own = {str(path): sha(path) for path in frozen.rglob('*.py')}
    sources = {**deps, **own}
    manifest = frozen / 'source_manifest.json'
    write(manifest, dict(schema='c6-COLA-independent-source-v1', phase=phase,
        source_files_sha256=sources, read_only_dependencies=[str(C5), str(C6)],
        prior_comparison_source_or_output_modified=False, original_Full_protocols_reused_readonly=True))
    engineering = phase == 'engineering'; episodes = 64 if engineering else 3200
    paths = 2 if engineering else 32; every = 5184 if engineering else 10368
    native = copy.deepcopy(native); native['learning_steps'] = episodes * 81
    configs = []
    for seed in ((7901,) if engineering else (6400, 6401, 6402)):
        reference = originals[6400 if engineering else seed]
        train_weather = list(range(4211000000, 4211000000 + 64)) if engineering else reference['train_weather_seeds']
        job = f'cola__seed{seed}'; output = STUDY / phase / job / 'attempt0'
        if output.exists(): raise FileExistsError(output)
        config = dict(schema='chapter6-COLA-additional-job-v1', job_id=job, method='cola', seed=seed,
            engineering=engineering, learning_steps=episodes * 81, episodes=episodes,
            eval_every=every, dev_paths=paths, device='cuda:0', gpu_uuid=GPU_UUID,
            train_weather_seeds=train_weather, dev_weather_seeds=reference['dev_weather_seeds'][:paths],
            algorithm_config=copy.deepcopy(native), source_files_sha256=sources, source_release=sha(manifest),
            output=str(output), worker=str(frozen / 'execution/cola_worker.py'),
            new_TEST=False, selection='Fixed endpoint only; DEV never selects parameters/checkpoint/budget',
            candidate_history='Additional COLA selected after observing previous CAPQL results; original CAPQL/PASTA evidence retained',
            native_TD_n_step=8, native_OADM_target='true one-step successor',
            coverage_threshold=.9, polyak_tau=.002, no_new_rule_runs=True)
        target = STUDY / 'protocol' / f'{tag}_{job}.json'; write(target, config); configs.append(str(target))
    result = dict(phase=phase, configs=configs, source_manifest=str(manifest), source_release=sha(manifest),
                  method='cola', fresh_attempt=0, automatic_retries=False, physical_steps_by_preparation=0)
    write(index, result)
    return dict(**result, frozen=str(frozen), source_count=len(sources), index=str(index))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['engineering', 'formal'])
    print(json.dumps(prepare(parser.parse_args().phase), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
