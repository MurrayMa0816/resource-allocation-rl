"""Execute only pre-registered independent development jobs; never auto-retry."""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys,os,time,datetime

SCOPE=Path(__file__).resolve().parents[1]
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
    p=argparse.ArgumentParser()
    p.add_argument('--registry',type=Path,required=True);p.add_argument('--worker',type=Path,required=True)
    p.add_argument('--max-parallel',type=int,default=16);p.add_argument('--stage',choices=['engineering','development'],required=True)
    a=p.parse_args();registry=json.loads(a.registry.read_text());digest=sha(a.registry)
    status_path=SCOPE/'runtime'/f'{a.stage}_status.json'
    assert not status_path.exists(),'A completed/started registry is not automatically replayable'
    jobs=registry['jobs'];queue=list(jobs);active={};done=[];failed=[]
    start=time.time();logs=SCOPE/'runtime'/f'{a.stage}_logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='GPU-a3130b13-8288-caaa-0f96-f2916c92d8e3')
    runtime=json.loads(Path('/hy-tmp/resource_allocation_results/chapter04/experiments/double_random_formal_20261107/protocol/runtime_environment.json').read_text())
    env['LD_LIBRARY_PATH']=runtime['env']['LD_LIBRARY_PATH']
    env['PYTHONPATH']='/root/gamewar/resource_allocation_rl/frozen/dependencies/chapter04/recent_baselines_formal_20261102_source'
    env['PYTHONNOUSERSITE']='1'
    def status():
        progress=[]
        for cid,item in active.items():
            pp=Path(item['job']['output'])/'progress.json'
            row=dict(id=cid,pid=item['process'].pid)
            if pp.exists():
                try:row.update(json.loads(pp.read_text()))
                except json.JSONDecodeError:pass
            progress.append(row)
        return dict(stage=a.stage,controller_pid=os.getpid(),registry=str(a.registry),registry_sha256=digest,worker=str(a.worker),worker_sha256=sha(a.worker),
            elapsed_seconds=time.time()-start,queued=[x['id'] for x in queue],active=progress,completed=done,failed=failed,
            completed_count=len(done),failed_count=len(failed),total_count=len(jobs),recovery=dict(command=None),
            status='complete' if not queue and not active and not failed else 'failed' if not queue and not active else 'running')
    while queue or active:
        while queue and len(active)<a.max_parallel:
            j=queue.pop(0)
            assert not (Path(j['output'])/'protocol.json').exists(),'Immutable registered attempt already exists'
            argv=[sys.executable,str(a.worker),'--registry',str(a.registry),'--case-id',j['id'],'--expected-registry-sha',digest,
                '--domain',j['domain'],'--method',j['method'],'--seed',str(j['seed']),'--steps',str(j['steps']),
                '--output',j['output'],'--device',j['device'],'--train-master',str(j['train_master']),'--dev-master',str(j['dev_master']),
                '--eval-every',str(j['eval_every']),'--dev-cases',str(j['dev_cases'])]
            if j.get('engineering_gate'):argv+=['--engineering-gate']
            log=(logs/(j['id']+'.log')).open('x')
            proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=str(SCOPE))
            active[j['id']]=dict(job=j,process=proc,log=log,started_at=time.time(),argv=argv)
            write(logs/(j['id']+'.launch.json'),dict(pid=proc.pid,argv=argv,started_at=time.time(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().split()[21]))
        for cid,item in list(active.items()):
            code=item['process'].poll()
            if code is None:continue
            item['log'].close();out=Path(item['job']['output']);receipt=out/'completed.json'
            row=dict(id=cid,returncode=code,elapsed_seconds=time.time()-item['started_at'],output=str(out),receipt=str(receipt))
            if code==0 and receipt.exists():
                c=json.loads(receipt.read_text());row.update(training_steps=c.get('training_steps'),development_env_steps=c.get('development_env_steps'),endpoint_reload_pass=c.get('endpoint_reload_pass'),actual_device=c.get('actual_device'))
                done.append(row)
            else:failed.append(row)
            del active[cid]
        value=status();write(status_path,value)
        if not active and not queue:break
        time.sleep(10)
    print(json.dumps(status(),ensure_ascii=False),flush=True)
    return int(bool(failed))
if __name__=='__main__':sys.exit(run())
