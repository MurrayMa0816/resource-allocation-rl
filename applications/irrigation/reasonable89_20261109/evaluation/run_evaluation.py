#!/usr/bin/env python3
"""Resume-safe CPU dispatch for the36 registered jobs; explicit TEST release required."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('reasonable89_preparation',HERE/'prepare_evaluation.py')
prep=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(prep)
ev=prep.ev;ROOT=ev.RUN;DISPATCH=ROOT/'dispatch/evaluation'
CAPSULE_NAMES=prep.CAPSULE_NAMES


def read_json(path):return json.loads(Path(path).read_text())


def cpu_limits():
    quota=None
    p=Path('/sys/fs/cgroup/cpu.max')
    if p.exists():
        words=p.read_text().split()
        if words[0]!='max':quota=float(words[0])/float(words[1])
    else:
        q=Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us');period=Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
        if q.exists() and int(q.read_text())>0:quota=int(q.read_text())/int(period.read_text())
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
    return min(float(affinity),float(quota or affinity))


def cpu_seconds():
    p=Path('/sys/fs/cgroup/cpu.stat')
    if p.exists():
        stats=dict(row.split() for row in p.read_text().splitlines())
        if 'usage_usec' in stats:return int(stats['usage_usec'])/1e6
    p=Path('/sys/fs/cgroup/cpuacct/cpuacct.usage')
    return int(p.read_text())/1e9 if p.exists() else None


def available_memory():
    mem={row.split(':')[0]:int(row.split()[1])*1024 for row in Path('/proc/meminfo').read_text().splitlines()}
    result=mem['MemAvailable']
    pairs=[('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
           ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]
    for limit,current in pairs:
        p=Path(limit);q=Path(current)
        if p.exists() and q.exists() and p.read_text().strip()!='max':result=min(result,max(0,int(p.read_text())-int(q.read_text())))
    return result


def process_identity(pid):
    try:
        raw=Path('/proc/'+str(pid)+'/stat').read_text();fields=raw.rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        return dict(pid=int(pid),start_ticks=int(fields[19]),argv=Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(b'\0')[:-1])
    except OSError:return None


def same_process(record):
    now=process_identity(record['pid'])
    return bool(now and now['start_ticks']==record['start_ticks'] and now['argv']==[x.encode() for x in record['argv']])


def claim_busy(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
    return False


def launch_attempt(entry,info,state,state_path,handles,env):
    name=entry['job']['job_id'];attempt=make_attempt(entry,len(info['attempts']))
    log=DISPATCH/'console'/name/('attempt'+str(attempt['number'])+'.txt');log.parent.mkdir(parents=True,exist_ok=True)
    claim=DISPATCH/'worker_claims'/name/('attempt'+str(attempt['number'])+'.lock');claim.parent.mkdir(parents=True,exist_ok=True)
    stream=claim.open('a')
    try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close();raise RuntimeError('Attempt already owned by an unregistered live worker:'+str(claim))
    argv=[sys.executable,'-B',str(HERE/'evaluate.py'),'--config',attempt['config']]
    attempt.update(status='launching',claim_path=str(claim),argv=argv,launch_intent_at=time.time(),console=str(log))
    info['attempts'].append(attempt);info['status']='running'
    # Atomic durable intent precedes Popen. Inherited flock prevents duplicate recovery.
    ev.write_json(state_path,state)
    try:
        try:
            with log.open('ab') as output:
                process=subprocess.Popen(argv,stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True,pass_fds=(stream.fileno(),))
        except OSError:
            # Popen failed before a usable evaluator exec. Interrupts after a
            # successful fork retain the durable launching intent and child claim.
            attempt.update(status='launch_failed',launch_confirmed=False,actual_work=dict(environment_steps=0,paths=0,model_inferences=0),work_exact=True)
            info['status']='pending';ev.write_json(state_path,state);raise
        handles[name]=process;identity=process_identity(process.pid)
        attempt.update(status='running',pid=process.pid,start_ticks=identity['start_ticks'] if identity else -1,started_at=time.time(),launch_confirmed=True)
    finally:
        # Close, never LOCK_UN: the exec'd evaluator owns the inherited open description.
        stream.close()
    ev.write_json(state_path,state)


def validate_complete(output,job):
    output=ev.require_hy(output);done=read_json(output/'complete.json')
    if (done.get('status')!='COMPLETE' or done.get('formal') is not True or
        done.get('job_sha256')!=ev.hashlib.sha256(ev.canonical(job)).hexdigest()):
        raise ValueError('Invalid formal completed attempt:'+str(output))
    if set(done['files_sha256'])!={'results.npz','metadata.json','execution_work.json'}:
        raise ValueError('Incomplete retained output set')
    for name,digest in done['files_sha256'].items():
        if ev.sha(output/name)!=digest:raise ValueError('Completed result changed:'+str(output/name))
    expected=32 if job['kind']=='feedback' else 1 if job['kind'] in ('history','descriptive_feedback') else 128
    if done.get('paths')!=expected or done.get('environment_steps')!=81*expected or done.get('learning_updates')!=0:
        raise ValueError('Completion has wrong actual work')
    meta=read_json(output/'metadata.json')
    if (meta.get('evaluation_execution_sha256')!=ev.sha(HERE/'evaluate.py') or
        meta.get('evaluation_files_sha256')!=job['evaluation_files_sha256'] or
        meta.get('scientific_source_release')!=job['source_release'] or
        meta.get('actor_sha256')!=meta.get('executed_actor_sha256') or
        meta.get('test_namespace_amendment_sha256')!=job['test_namespace_amendment_sha256'] or
        meta.get('execution_operator')!=ev.M64_OPERATOR or meta.get('execution_description',{}).get('points_array_sha256')!=job['execution_points_array_sha256']):
        raise ValueError('Result source or actual actor identity changed')
    if job['kind']=='initial' and (meta.get('executed_policy_episode')!=0 or not meta.get('initialization_reconstruction',{}).get('exact_actual_training_initialization')):
        raise ValueError('Initial-policy parameter identity is not proven')
    if job['kind']!='initial' and meta.get('executed_policy_episode')!=3200:
        raise ValueError('Noninitial job did not use fixed3200 endpoint')
    expected_calls=0 if job['kind'] in ('feedback','descriptive_feedback') and job.get('controller_mode')=='replay' else 81*expected
    if done.get('parameter_evaluations')!=expected_calls or done.get('raw_feasible_action_vectors')!=64*expected_calls or done.get('policy_parameter_batch_calls')!=expected_calls:
        raise ValueError('Actual M64 parameter/feasible-vector counters differ from evaluation role')
    return done


def publish_success(job_id,attempt):
    config_path=Path(attempt['config']);job=read_json(config_path);output=ev.require_hy(job['output_dir']);done=validate_complete(output,job)
    base=ROOT/'evaluation'/job_id;base.mkdir(parents=True,exist_ok=True)
    proof=dict(status='PASS',job_id=job_id,attempt=attempt['number'],actual_config_path=str(config_path),actual_config_sha256=ev.sha(config_path),
        output_dir=str(output),complete_sha256=ev.sha(output/'complete.json'),base_config_path=attempt['base_config'],base_config_sha256=ev.sha(attempt['base_config']))
    prep.immutable_json(base/'successful_attempt.json',proof)
    for name in ('results.npz','metadata.json','execution_work.json','complete.json'):
        target=output/name;alias=base/name
        if alias.exists() or alias.is_symlink():
            if alias.resolve()!=target:raise FileExistsError('Retain existing different publication alias:'+str(alias))
        else:alias.symlink_to(target.relative_to(base))
    return done


def load_plan():
    path=ROOT/'configs/evaluation/matrix.json';matrix=read_json(path)
    if matrix.get('case_id')!=ev.CASE_ID or matrix.get('complete_paths')!=2694 or len(matrix.get('jobs',[]))!=36:
        raise ValueError('Expected the exact registered36-job matrix')
    expected_capsule={str(HERE/name):ev.sha(HERE/name) for name in CAPSULE_NAMES}
    if matrix.get('evaluation_files_sha256')!=expected_capsule:raise ValueError('Execution capsule differs from bound matrix')
    jobs={}
    for record in matrix['jobs']:
        p=ev.require_hy(record['path'])
        if ev.sha(p)!=record['sha256']:raise ValueError('Immutable bound config changed:'+str(p))
        job=read_json(p)
        if job['job_id']!=record['job_id'] or job.get('evaluation_files_sha256')!=expected_capsule:
            raise ValueError('Wrong registered config identity')
        jobs[job['job_id']]=dict(record=record,path=str(p),job=job)
    if len(jobs)!=36:raise ValueError('Duplicate job registration')
    return matrix,jobs



def load_descriptive_plan(jobs,state):
    path=ROOT/'configs/descriptive_feedback/matrix.json'
    if not path.exists():return
    matrix=read_json(path)
    if (matrix.get('base_evaluation_matrix_sha256')!=state['matrix_sha256'] or
        matrix.get('representative_feedback_contract_sha256')!=prep.REPRESENTATIVE_SHA or
        matrix.get('evaluation_files_sha256')!={str(HERE/name):ev.sha(HERE/name) for name in CAPSULE_NAMES} or
        len(matrix.get('jobs',[])) not in (0,4) or matrix.get('complete_paths')!=len(matrix['jobs']) or matrix.get('included_in_feedback_statistics') is not False):
        raise ValueError('Invalid conditional descriptive registration')
    if state.get('descriptive_matrix_sha256') not in (None,ev.sha(path)):
        raise ValueError('Conditional descriptive matrix changed')
    case_path=Path(matrix['representative_case_path'])
    if ev.sha(case_path)!=matrix['representative_case_sha256']:raise ValueError('Representative case registration changed')
    for record in matrix['jobs']:
        p=ev.require_hy(record['path'])
        if ev.sha(p)!=record['sha256']:raise ValueError('Descriptive config changed')
        job=read_json(p)
        if job['kind']!='descriptive_feedback' or job['representative_case_sha256']!=matrix['representative_case_sha256'] or job['job_id']!=record['job_id']:
            raise ValueError('Foreign descriptive job')
        jobs[job['job_id']]=dict(record=record,path=str(p),job=job)
        state['jobs'].setdefault(job['job_id'],dict(status='pending',attempts=[]))
    state['descriptive_matrix_sha256']=ev.sha(path);state['descriptive_registered']=True


def preflight():
    matrix,jobs=load_plan()
    # This gate hashes all9 endpoints and checks root release without opening panel.
    first=next(iter(jobs.values()))['job'];ev.verify_test_barrier(first)
    identity=prep.read_identity();expected={j['job_id']:j for j in prep.bound_jobs(identity)}
    if any(entry['job']!=expected[name] for name,entry in jobs.items()):
        raise ValueError('Bound config differs from selected application/protocol')
    return matrix,jobs


def make_attempt(entry,number):
    base=entry['job'];actual=dict(base);actual['output_dir']=str(ROOT/'evaluation'/base['job_id']/('attempt'+str(number)))
    # No scientific config field may vary across retries.
    if {k:v for k,v in actual.items() if k!='output_dir'}!={k:v for k,v in base.items() if k!='output_dir'}:
        raise AssertionError('Scientific retry config changed')
    p=DISPATCH/'attempt_configs'/base['job_id']/('attempt'+str(number)+'.json')
    prep.immutable_json(p,actual)
    return dict(number=number,config=str(p),base_config=entry['path'],output_dir=actual['output_dir'],status='prepared')


def status():
    path=DISPATCH/'status.json'
    return read_json(path) if path.exists() else dict(status='NOT_STARTED',run_root=str(ROOT),TEST_execution_started=False)


def execute(max_workers=32,max_attempts=2):
    if not 1<=max_workers<=32 or not 1<=max_attempts<=2:raise ValueError('Bounded1..32 workers and1..2 attempts required')
    DISPATCH.mkdir(parents=True,exist_ok=True)
    lock=(DISPATCH/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    matrix,jobs=preflight();state_path=DISPATCH/'state.json'
    if state_path.exists():
        state=read_json(state_path)
        if state['matrix_sha256']!=ev.sha(ROOT/'configs/evaluation/matrix.json'):raise ValueError('Never resume another matrix')
    else:state=dict(matrix_sha256=ev.sha(ROOT/'configs/evaluation/matrix.json'),jobs={name:dict(status='pending',attempts=[]) for name in jobs})
    load_descriptive_plan(jobs,state)
    if set(state['jobs'])!=set(jobs):raise ValueError('Resume state differs from immutable base/descriptive job registration')
    handles={};verified=set();started=time.time();sample_time=time.monotonic();sample_cpu=cpu_seconds()
    time.sleep(3)
    sample_end=time.monotonic();sample_cpu_end=cpu_seconds()
    usage=max(0.,(sample_cpu_end-sample_cpu)/(sample_end-sample_time)) if sample_cpu is not None and sample_cpu_end is not None else 0.
    cpu_previous=(sample_end,sample_cpu_end)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    quota=cpu_limits()
    if quota<=2:raise ValueError('CPU quota must exceed the2-core reserve')
    ceiling=min(max_workers,int(quota-2))
    while True:
        for name,info in state['jobs'].items():
            base=ROOT/'evaluation'/name;proof=base/'successful_attempt.json'
            if info['status']=='complete':
                if name not in verified:
                    chosen=read_json(proof);validate_complete(chosen['output_dir'],read_json(chosen['actual_config_path']));verified.add(name)
                continue
            if info['attempts']:
                attempt=info['attempts'][-1];out=Path(attempt['output_dir'])
                if (out/'complete.json').exists():
                    done=publish_success(name,attempt);verified.add(name);attempt.update(status='complete',launch_confirmed=True);info.update(status='complete',paths=done['paths'],environment_steps=done['environment_steps']);continue
                if attempt['status'] in ('launching','running'):
                    busy=claim_busy(attempt['claim_path'])
                    alive=bool(attempt.get('pid') and same_process(attempt))
                    if busy or alive:
                        attempt['status']='running';attempt['launch_confirmed']=True;info['status']='running'
                    else:
                        code=handles[name].poll() if name in handles else None;attempt.update(status='failed',exit_code=code,ended_at=time.time())
                        failure=out/'failure.json';progress=out/'progress.json'
                        if failure.exists():attempt['actual_work']=read_json(failure).get('actual_work');attempt['work_exact']=read_json(failure).get('operator_work_exact',False)
                        elif progress.exists():attempt['actual_work_lower_bound']=read_json(progress).get('actual_work');attempt['work_exact']=False
                        else:attempt['work_exact']=False
                        info['status']='failed' if len(info['attempts'])>=max_attempts else 'pending'
                elif attempt['status']=='launch_failed':
                    if claim_busy(attempt['claim_path']) or (attempt.get('pid') and same_process(attempt)):
                        attempt.update(status='running',launch_confirmed=True);info['status']='running'
                    else:info['status']='failed' if len(info['attempts'])>=max_attempts else 'pending'
        running=[(name,info['attempts'][-1]) for name,info in state['jobs'].items() if info['status']=='running']
        now=time.monotonic();current=cpu_seconds()
        if current is not None and cpu_previous[1] is not None and now>cpu_previous[0]:
            observed=max(0.,(current-cpu_previous[1])/(now-cpu_previous[0]));usage=.7*usage+.3*observed
        cpu_previous=(now,current)
        startup=sum(1 for _,a in running if time.time()-a.get('started_at',0)<15)
        memory=available_memory();free=shutil.disk_usage(ROOT).free
        remaining_paths=sum(jobs[name]['record']['paths'] for name,info in state['jobs'].items() if info['status']!='complete')
        disk_required=(1536+remaining_paths)*1024**2
        slots=max(0,min(ceiling-len(running),max(0,int(quota-2-usage-startup)),max(0,int((memory-4*1024**3)/(768*1024**2)))))
        if free<disk_required:slots=0
        # Feedback must wait for the intact same-model main donor; no extra rollout.
        candidates=[]
        for name,entry in jobs.items():
            if state['jobs'][name]['status']!='pending':continue
            if entry['job']['kind'] in ('feedback','descriptive_feedback'):
                dependency='main__seed'+str(entry['job']['training_identity']['seed'])
                if state['jobs'][dependency]['status']!='complete':continue
            candidates.append(name)
        candidates.sort(key=lambda name:(jobs[name]['job']['kind']!='main',name))
        for name in candidates[:slots]:
            launch_attempt(jobs[name],state['jobs'][name],state,state_path,handles,env)
        counts={label:sum(info['status']==label for info in state['jobs'].values()) for label in ('pending','running','complete','failed')}
        summary=dict(status='COMPLETE' if counts['complete']==len(jobs) and state.get('descriptive_registered') else 'FAILED' if counts['failed'] and not counts['running'] else 'RUNNING',
            run_root=str(ROOT),counts=counts,successful_paths=sum(info.get('paths',0) for info in state['jobs'].values() if info['status']=='complete'),
            successful_environment_steps=sum(info.get('environment_steps',0) for info in state['jobs'].values() if info['status']=='complete'),
            learning_updates=0,TEST_release_accepted=True,TEST_jobs_launched=sum(bool(a.get('launch_confirmed')) for info in state['jobs'].values() for a in info['attempts']),
            TEST_execution_started=any(a.get('launch_confirmed') for info in state['jobs'].values() for a in info['attempts']),controller_pid=os.getpid(),seconds=time.time()-started,
            resources=dict(cpu_quota=quota,cpu_usage_cores_EWMA=usage,worker_ceiling=ceiling,memory_available_bytes=memory,disk_free_bytes=free,disk_required_bytes=disk_required),
            execution_capsule_sha256=matrix['evaluation_files_sha256'])
        ev.write_json(state_path,state);ev.write_json(DISPATCH/'status.json',summary)
        if counts['complete']==len(jobs):
            if not state.get('descriptive_registered'):
                prep.prepare_descriptive_feedback(jobs);load_descriptive_plan(jobs,state);ev.write_json(state_path,state);continue
            expected_paths=2694+len(jobs)-36
            if len(jobs) not in (36,40) or summary['successful_paths']!=expected_paths or summary['successful_environment_steps']!=81*expected_paths:
                raise AssertionError('Complete statistical/descriptive scientific count differs')
            summary.update(base_jobs=36,base_paths=2694,descriptive_jobs=len(jobs)-36,descriptive_paths=len(jobs)-36,descriptive_excluded_from_feedback_statistics=True)
            if {str(HERE/name):ev.sha(HERE/name) for name in CAPSULE_NAMES}!=matrix['evaluation_files_sha256']:
                raise ValueError('Execution capsule changed during dispatch')
            for name in jobs:
                proof=read_json(ROOT/'evaluation'/name/'successful_attempt.json');validate_complete(proof['output_dir'],read_json(proof['actual_config_path']))
            ev.write_json(DISPATCH/'completion.json',summary);return summary
        if counts['failed'] and not counts['running']:return summary
        if not counts['running'] and candidates and slots==0 and ceiling<1:raise RuntimeError('No viableCPU quota')
        time.sleep(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--max-workers',type=int,default=32);parser.add_argument('--max-attempts',type=int,default=2);args=parser.parse_args()
    if args.run:
        try:result=execute(args.max_workers,args.max_attempts)
        except (OSError,ValueError,KeyError) as exc:
            print(json.dumps(dict(status='DISPATCH_REJECTED_OR_INTERRUPTED',reason=str(exc),execution_status=status(),new_work_claim='Inspect state and worker receipts; no zero-work assumption after dispatch errors'),ensure_ascii=False));raise SystemExit(2)
    else:result=status()
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
