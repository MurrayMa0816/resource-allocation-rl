"""Immutable sixteen-job registration; no external method or extra seed route."""
from __future__ import annotations
import math
from pathlib import Path
import socket
from . import common as c

GIB=1024**3;MIB=1024**2


def immutable(path,value):
    path=Path(path)
    if path.exists():c.require(c.read_json(path)==value,'Refuse to overwrite immutable full-spend registration/config: '+str(path))
    else:c.write_json(path,value)


def estimate_bytes(domain):
    n,h,d=c.DOMAINS[domain]['N'],c.DOMAINS[domain]['horizon'],19 if domain=='radio' else 13
    physical=61440+121*32*h+256*h+32*h;episodes=physical//h
    # actions[N], pre-guard actions[N], caps[N], latent[d], native/vector
    # rewards[3], lower/upper/before/after/period-total[5], quality[N].
    arrays=physical*(3*n+d+8)*8+episodes*n*8
    plans=episodes*8192
    json_bytes=(episodes+61440//h)*16384 # training rows plus separate ledger
    zip_bytes=(math.ceil(episodes/8)+128)*8192
    logs=math.ceil(61440/(8*h))*16384+16*MIB
    raw=arrays+plans+json_bytes+zip_bytes+logs+8*MIB+8*MIB
    bound=math.ceil(raw*1.15/(16*MIB))*16*MIB
    return dict(estimated_disk_bytes=bound,estimated_gpu_bytes=1536*MIB,estimated_memory_bytes=2048*MIB,estimated_cpu_cores=1.,
        storage_components=dict(uncompressed_arrays=arrays,case_plan_allowance=plans,json_and_duplicate_training_ledger=json_bytes,
            zip_allowance=zip_bytes,update_logs_diagnostics=logs,endpoint=8*MIB,metadata=8*MIB,margin=1.15),
        expected_physical_steps=physical)


def build(root,python,readiness):
    root=Path(root).resolve();c.require(root.is_relative_to(Path('/hy-tmp')),'All new outputs must physically remain in HY')
    ready=c.validate_readiness(readiness);python=str(Path(python).expanduser().absolute());jobs=[]
    for domain,spec in c.DOMAINS.items():
        for variant in c.VARIANTS:
            for seed in c.SEEDS:
                identifier=f'{domain}__{variant}__seed{seed}';config=root/'configs'/(identifier+'.json')
                immutable(config,{'variant':variant,'tau':spec['tau']});output=root/'runs'/identifier
                argv=[python,'-m','chapter5_full_spend_20261104.worker','--variant',variant,'--domain',domain,'--seed',str(seed),
                    '--tau',str(spec['tau']),'--profile','formal','--device','cuda:0','--learning-steps','61440','--eval-every','512',
                    '--dev-paths','32','--holdout-paths','256','--eval-batch','8','--diagnostics-every','512','--eval-modes','deterministic',
                    '--path-master',str(c.TRAIN_MASTER),'--dev-master',str(c.DEV_MASTER),'--test-master',str(c.TEST_MASTER),
                    '--dev-rng-seed',str(c.DEV_RNG_SEED),'--test-rng-seed',str(c.TEST_RNG_SEED),
                    '--config-json',str(config),'--run-protocol',str(root/'registration.json'),'--job-id',identifier,'--output',str(output)]
                jobs.append(dict(id=identifier,family='proposed',method='proposed',variant=variant,domain=domain,seed=seed,tau=spec['tau'],
                    cohort='proposed',config=str(config),config_sha256=c.sha(config),output=str(output),argv=argv,device='cuda:0',threads=1,
                    full_spend_required=True,**estimate_bytes(domain)))
    value=dict(schema=c.REGISTRATION_SCHEMA,root=str(root),masters=c.masters(),seeds=list(c.SEEDS),jobs=jobs,
        learning_steps=61440,evaluation_interval=512,evaluation_ticks=c.TICKS,dev_paths=32,holdout_paths=256,
        full_spend_required=True,environment_fingerprint=c.environment_fingerprint(),native_source_manifest_sha256=c.sha(c.HERE/'native_source_manifest.json'),
        intended_execution_hostname=socket.gethostname(),runtime_source_sha256=ready['runtime_source_sha256'],data_sha256=ready['data_sha256'],software=ready['software'],
        readiness=str(Path(readiness).resolve()),readiness_sha256=c.sha(readiness),automatic_more_seeds=False,automatic_retry=False,external_baseline_jobs=0,
        resources=dict(physical_gpu_indices=[0,1],cpu_reserve_cores=1.,max_cpu_demand_ratio=1.15,memory_reserve_bytes=8*GIB,
            gpu_reserve_bytes=1536*MIB,disk_reserve_bytes=5*GIB,report_reserve_bytes=512*MIB,
            external_scope='/hy-tmp/resource_allocation_results/chapter5/five_round_baseline_20261104',external_max_jobs=60,
            external_max_concurrent=10,external_worker_gpu_bytes=1536*MIB,external_worker_memory_bytes=2560*MIB),
        postprocess_argv=[python,'-m','chapter5_full_spend_20261104.report','--root',str(root)],report_receipt=str(root/'reports/report.json'),
        after_report='awaiting_user_review',automatic_thesis_update=False)
    immutable(root/'registration.json',value)
    immutable(root/'master_manifest.json',dict(schema='chapter5-full-spend-master-v1',root=str(root),registration=str(root/'registration.json'),
        registration_sha256=c.sha(root/'registration.json'),exact_job_ids=[j['id'] for j in jobs],automatic_more_seeds=False,automatic_retry=False))
    return value
