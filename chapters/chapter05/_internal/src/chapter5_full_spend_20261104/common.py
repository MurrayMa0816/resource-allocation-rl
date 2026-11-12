"""Pure-file identities for the independent sixteen-job full-spend scope."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket

from . import WORKSPACE,HERE

TRAIN_MASTER=2026110481101
DEV_MASTER=2026110481201
TEST_MASTER=2026110481699
DEV_RNG_SEED=118201
TEST_RNG_SEED=118699
SEEDS=(420,421)
DOMAINS={'radio':{'N':100,'horizon':32,'tau':.2},'v5':{'N':120,'horizon':24,'tau':.7}}
VARIANTS=('full','scalar_value','only_j','only_c')
TICKS=list(range(0,61441,512))
REGISTRATION_SCHEMA='chapter5-full-spend-registration-v1'
READINESS_SCHEMA='chapter5-full-spend-readiness-v1'
ENVIRONMENT_REVISION='chapter5-shared-native-full-spend-bridge-v1'


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temporary.replace(path)


def masters():return dict(train=TRAIN_MASTER,development=DEV_MASTER,test=TEST_MASTER,dev_rng=DEV_RNG_SEED,test_rng=TEST_RNG_SEED)


def source_manifest():
    # Filesystem-only: never import a historical Env before bridge.bootstrap.
    legacy=read_json(HERE/'development_source_manifest.json')['files_sha256'];result={}
    for name,digest in legacy.items():
        actual=sha(WORKSPACE/name);require(actual==digest,'Read-only legacy dependency changed: '+name);result[name]=actual
    native=read_json(HERE/'native_source_manifest.json')['files_sha256']
    require(len(native)==175,'Exactly175 frozen native full-spend source files required')
    for name,digest in native.items():
        path=HERE/'native_source'/name;actual=sha(path)
        require(actual==digest,'New native full-spend source changed: '+name)
        result['chapter5_full_spend_20261104/native_source/'+name]=actual
    for path in HERE.rglob('*'):
        relative=path.relative_to(HERE)
        if 'native_source' in relative.parts or 'tests' in relative.parts or '__pycache__' in relative.parts:continue
        if path.is_file() and path.suffix in ('.py','.json','.rst'):result[str(path.relative_to(WORKSPACE))]=sha(path)
    return dict(sorted(result.items()))


def data_manifest():
    manifest=read_json(HERE/'data_manifest.json');result={}
    for name,item in manifest['files'].items():
        path=Path(item['path']);actual=sha(path)
        require(actual==item['sha256'],'Prepared full-spend data changed: '+name);result[str(path)]=actual
    return dict(sorted(result.items()))


def software_manifest():
    value={'python':platform.python_version()}
    for package in ('torch','numpy','scipy','gymnasium','psutil','Pillow','matplotlib'):
        try:value[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:value[package]=None
    return value


def environment_fingerprint():
    return object_sha(dict(revision=ENVIRONMENT_REVISION,native_manifest_sha256=sha(HERE/'native_source_manifest.json'),
        bridge_sha256=sha(HERE/'bridge.py'),env_sha256=sha(HERE/'env.py'),data_manifest_sha256=sha(HERE/'data_manifest.json'),
        full_spend_required=True,native_task_chapter=3))


def validate_readiness(path):
    value=read_json(path)
    require(value.get('schema')==READINESS_SCHEMA and value.get('status')=='pass' and value.get('formal_training_started') is False,
        'A new passing full-spend readiness receipt is required')
    require(value.get('target_tests',0)>0 and value.get('worker_short_gate_tests',0)>0,
        'The new worker/collector/controller target short gates are required')
    require(value.get('runtime_source_sha256')==source_manifest() and value.get('data_sha256')==data_manifest()
        and value.get('software')==software_manifest() and value.get('environment_fingerprint')==environment_fingerprint(),
        'Readiness source/data/software/full-spend environment changed')
    for item in value.get('gate_receipts',[]):require(sha(item['path'])==item['sha256'],'Bound engineering gate changed')
    if value.get('junit_path'):
        import xml.etree.ElementTree as ET
        require(sha(value['junit_path'])==value['junit_sha256'],'Target test JUnit changed')
        rows=ET.parse(value['junit_path']).getroot().findall('.//testcase')
        require(len(rows)==value['target_tests'] and all(not any(row.find(k) is not None for k in ('failure','error','skipped')) for row in rows),
            'Full-spend target tests failed, skipped or changed')
    return value


def validate_registration(path):
    value=read_json(path)
    require(value.get('schema')==REGISTRATION_SCHEMA and value.get('masters')==masters()
        and value.get('seeds')==list(SEEDS) and value.get('full_spend_required') is True,'Wrong full-spend scope')
    require(value.get('intended_execution_hostname')==socket.gethostname(),'Wrong full-spend execution host')
    require(value.get('runtime_source_sha256')==source_manifest() and value.get('data_sha256')==data_manifest()
        and value.get('software')==software_manifest() and value.get('environment_fingerprint')==environment_fingerprint(),
        'Registered source/data/software/full-spend environment changed')
    require(sha(value['readiness'])==value['readiness_sha256'],'Registered readiness changed');validate_readiness(value['readiness'])
    expected={(variant,domain,seed) for variant in VARIANTS for domain in DOMAINS for seed in SEEDS}
    jobs=value['jobs'];require(len(jobs)==16 and len({j['id'] for j in jobs})==16
        and {(j['variant'],j['domain'],j['seed']) for j in jobs}==expected,'Exactly sixteen predeclared proposed/control jobs required')
    require(value.get('automatic_more_seeds') is False and value.get('automatic_retry') is False
        and value.get('external_baseline_jobs')==0,'No extra seed, retry or external-baseline expansion permitted')
    return value


def validate_worker(args,family='proposed'):
    registration=validate_registration(args.run_protocol);jobs=[j for j in registration['jobs'] if j['id']==args.job_id]
    require(family=='proposed' and len(jobs)==1,'One registered proposed/control worker required');job=jobs[0]
    require((args.variant,args.domain,args.seed)==(job['variant'],job['domain'],job['seed']) and args.profile=='formal','Worker identity changed')
    require((args.learning_steps,args.eval_every,args.dev_paths,args.holdout_paths,args.eval_batch)==(61440,512,32,256,8),
        'Registered train-only/development/final budget changed')
    require((args.path_master,args.dev_master,args.test_master,args.dev_rng_seed,args.test_rng_seed)==tuple(masters().values()),'Worker case namespaces changed')
    require(args.device=='cuda:0' and args.eval_modes=='deterministic' and args.diagnostics_every==512
        and float(args.tau)==job['tau'],'Worker device, threshold or deployment changed')
    require(Path(args.output).resolve()==Path(job['output']).resolve() and Path(args.config_json).resolve()==Path(job['config']).resolve()
        and sha(args.config_json)==job['config_sha256'] and read_json(args.config_json)=={'variant':args.variant,'tau':args.tau},
        'Registered config/output changed')
    execution=dict(registered_job_id=job['id'],registration_path=str(Path(args.run_protocol).resolve()),registration_sha256=sha(args.run_protocol),
        config_path=job['config'],config_sha256=job['config_sha256'],runtime_source_sha256=registration['runtime_source_sha256'],
        data_sha256=registration['data_sha256'],readiness_path=registration['readiness'],readiness_sha256=registration['readiness_sha256'],
        actual_device='cuda:0',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),execution_package='chapter5_full_spend_20261104',
        tau=job['tau'],environment_fingerprint=registration['environment_fingerprint'],full_spend_required=True,
        real_training_preferences={'mode':'fixed','coverage_weight':.8},evaluated_preferences=[.8],
        learning_budget_basis='true_training_transitions_only')
    return dict(registration=registration,job=job,execution=execution)


def validate_completed(job,registration):
    folder=Path(job['output']);value=read_json(folder/'completed.json');protocol=read_json(folder/'protocol.json')
    initialization=protocol.get('initialization',{})
    require(initialization.get('profile')==read_json(HERE/'development_source_manifest.json')['initialization_profile']
        and initialization.get('native_default_draw_replay_exact') is True
        and set(initialization.get('heads',{}))=={'spatial','quota'},
        'Current completion requires both random output heads; read historical runs with their frozen source')
    require(value.get('status')=='complete' and value.get('formal_training') is True and value.get('source_unchanged') is True
        and value.get('endpoint_reload_verified') is True and value.get('fresh_training') is True
        and value.get('full_spend_required') is True and value.get('execution_ledger_verified') is True,
        'Completion, fresh training or actual full-spend proof missing')
    for row in (value,protocol):
        require(row.get('registered_job_id')==job['id'] and row.get('registration_sha256')==sha(Path(registration['root'])/'registration.json')
            and row.get('config_sha256')==job['config_sha256'] and row.get('runtime_source_sha256')==registration['runtime_source_sha256']
            and row.get('data_sha256')==registration['data_sha256'] and row.get('environment_fingerprint')==registration['environment_fingerprint']
            and row.get('actual_device')==job['device'] and (row.get('variant'),row.get('domain'),row.get('seed'),row.get('tau'))==
                (job['variant'],job['domain'],job['seed'],job['tau']),'Completed identity/source/environment changed')
    require(sha(job['config'])==job['config_sha256'],'Completed config changed')
    counts=value['counts'];h=DOMAINS[job['domain']]['horizon']
    require(counts['train']['physical_steps']==61440 and counts['development']['physical_steps']==121*32*h
        and counts['holdout']['physical_steps']==256*h and counts['holdout']['episodes']==256
        and counts.get('selection',{}).get('physical_steps',0)==0 and value.get('evaluation_ticks')==TICKS,
        'Incomplete exact train/development/final populations')
    require(value.get('optimizer_backup') is False and value.get('replay_backup') is False
        and value.get('saved_model_count')==1 and value.get('model_selection') is False and value.get('holdout_selection') is False,
        'Unexpected disk learning backup, selected checkpoint or extra model')
    require(value.get('gpu_proof',{}).get('cuda_training_verified') is True,'Actual CUDA training proof missing')
    require(sha(folder/'checkpoint_endpoint.pt')==value['endpoint_sha256'] and sha(folder/'training_ledger.jsonl')==value['training_ledger_sha256']
        and (folder/'holdout.json').is_file(),'Missing or changed endpoint/training ledger/final result')
    return value
