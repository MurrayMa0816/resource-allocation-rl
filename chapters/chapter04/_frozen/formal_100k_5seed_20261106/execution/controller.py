"""Resource-aware, recoverable local C4 queue. Never launches historical scopes."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1]
BOOT=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
POLICY={'target_cpu':40.,'max_jobs':90,'gpu_floor':1536,'disk_floor':2048*2**20,'ram_floor':16384*2**20}
def read(p,default=None):
    try:return json.loads(Path(p).read_text())
    except FileNotFoundError:return default
def write(p,d):
    p=Path(p);t=Path(str(p)+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def process(pid):
    try:
        s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        return {'state':s[0],'start_ticks':int(s[19]),'cpu_ticks':int(s[11])+int(s[12]),'argv':[x.decode()for x in argv]}
    except (FileNotFoundError,ProcessLookupError):return None
def alive(runtime):
    s=process(runtime['pid'])
    return bool(s and s['state']!='Z' and s['start_ticks']==runtime['start_ticks'] and BOOT==runtime['boot_id'] and s['argv']==runtime['argv'])
def gpu():
    p=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    if p.returncode:raise RuntimeError('GPU capacity query failed; do not assume idle')
    v=[float(x.strip())for x in p.stdout.strip().splitlines()[0].split(',')]
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    processes=[]
    if apps.returncode==0:
        for l in apps.stdout.strip().splitlines():
            x=l.split(',')
            try:processes.append({'pid':int(x[0]),'used_mib':float(x[1])})
            except (ValueError,IndexError):pass
    return {'total_mib':v[0],'used_mib':v[1],'utilization':v[2],'processes':processes}

def exact_workers(argv):
    found=[]
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            s=process(int(p.name))
            if s and s['state']!='Z'and s['argv']==argv:
                found.append(dict(pid=int(p.name),start_ticks=s['start_ticks'],boot_id=BOOT,argv=argv))
    assert len(found)<=1,('duplicate workers',found)
    return found[0]if found else None
def ram_available():
    values={}
    for l in Path('/proc/meminfo').read_text().splitlines():
        x=l.split();values[x[0].rstrip(':')]=int(x[1])*1024
    limit=int(Path('/sys/fs/cgroup/memory/memory.limit_in_bytes').read_text());used=int(Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text())
    return min(values['MemAvailable'],max(0,limit-used))
def complete(job):
    p=Path(job['output'])/'completed.json';d=read(p)
    if not d or not (d.get('success')is True or (job['family']=='offpolicy'and d.get('status')=='PASS')):return None
    if job['kind']=='training':
        assert int(d.get('training_steps',d.get('train_steps',-1)))==100608,(job['id'],'wrong steps')
        assert d['domain']==job['domain'] and int(d['seed'])==job['seed']
        model=model_path(job,d);assert model.is_file()
        expected=d.get('model_sha256',d.get('checkpoint_sha256'))
        if expected:assert sha(model)==expected,(job['id'],'endpoint SHA')
    return d
def model_path(job,d):
    v=d.get('checkpoint',d.get('model',d.get('model_path')))
    assert v, (job['id'],'no endpoint model path')
    p=Path(v);return p if p.is_absolute()else Path(job['output'])/p
def approval(job,cache):
    p=Path(job['approval']);d=read(p)
    if not d or d.get('status')!='PASS':return None
    digest=sha(p)
    if cache.get(str(p))!=digest:
        assert job['method'] in d['methods'],(job['id'],'not approved method')
        for path,expected in d['source_sha256'].items():assert sha(path)==expected,('family source changed',path)
        cache[str(p)]=digest
    return d
def command(job,byid):
    if job['argv']:return list(job['argv'])
    train=byid[job['training_job_id']];done=complete(train);assert done
    original=train['argv'];worker=original[1]
    args=[original[0],worker,'--method',job['method'],'--domain',job['domain'],'--seed',str(job['seed']),
          '--steps','100608','--output',job['output'],'--device',job['device'],
          '--evaluate-model',str(model_path(train,done)),'--eval-master','2026110694199','--eval-cases','256',
          '--environment-root',str(R/'source/environment')]
    args+=['--evaluation-phase'if job['family']=='f1'else'--eval-phase','final']
    return args
def folder_size(p):
    total=0
    if not p.exists():return 0
    for d,ds,fs in os.walk(p):
        for n in fs:
            try:total+=(Path(d)/n).stat().st_blocks*512
            except FileNotFoundError:pass
    return total
def runtime_env(job):
    env=dict(os.environ,**read(R/'protocol/runtime_environment.json')['env'])
    env['CUDA_VISIBLE_DEVICES']=''if job['device']=='cpu'else'GPU-a3130b13-8288-caaa-0f96-f2916c92d8e3'
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',WANDB_MODE='disabled',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    env.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
    for k in ['TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','JAX_COMPILATION_CACHE_DIR']:
        p=R/'runtime'/'worker_caches'/job['id']/k.lower();p.mkdir(parents=True,exist_ok=True);env[k]=str(p)
    return env
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.parse_args()
    (R/'runtime').mkdir(exist_ok=True);lock=open(R/'runtime/controller.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(R/'plan.json');seal=read(R/'plan_seal.json');assert sha(R/'plan.json')==seal['sha256']
    assert plan['new_training_jobs']==90 and plan['independent_endpoint_evaluations']==100
    jobs=plan['jobs'];byid={j['id']:j for j in jobs};state=read(R/'runtime/queue_state.json',{'schema':'chapter4-live-queue-v1','jobs':{},'attempts':{}})
    cache={};children={};cpu_samples={};hz=os.sysconf('SC_CLK_TCK');last_start=0.;last_hashcheck=0.
    for j in jobs:state['jobs'].setdefault(j['id'],{'status':'pending','attempts':0})
    # A launch intent is durable before Popen. Recover its exact process instead
    # of starting a second worker after a coordinator crash.
    for j in jobs:
        st=state['jobs'][j['id']]
        if st['status']=='launching':
            rt=exact_workers(st['launch_intent']['argv'])
            if rt:
                rt.update(device=j['device'],log=st['launch_intent']['log'],started_at=st['launch_intent']['time'],approval_sha256=sha(j['approval']))
                st.update(status='running',runtime=rt)
            else:st.update(status='failed',reason='Interrupted launch has no live exact worker; never blindly duplicate or merge partial outputs.')
    identity=process(os.getpid());write(R/'runtime/controller_identity.json',dict(pid=os.getpid(),start_ticks=identity['start_ticks'],boot_id=BOOT,argv=identity['argv'],controller_sha256=sha(__file__),plan_sha256=seal['sha256']))
    write(R/'recovery_launch.json',dict(status='running',scope=str(R),command=[sys.executable,str(Path(__file__).resolve()),'--resume'],precondition='Only after exact registered controller exited; live workers are adopted, never duplicated.',controller_pid=os.getpid(),plan_sha256=seal['sha256'],old_scopes_never_resumed=True))
    while True:
        tick=time.time();active=[];done_ids=set();failed=[]
        for j in jobs:
            st=state['jobs'][j['id']];receipt=complete(j)
            if receipt:
                st.update(status='complete',completed_at=tick);done_ids.add(j['id']);continue
            rt=st.get('runtime')
            if rt and alive(rt):
                st['status']='running';active.append(j);continue
            if st['status']=='running':
                proc=children.get(j['id']);code=proc.poll()if proc else None
                st.update(status='failed',exit_code=code,failed_at=tick,reason='Worker exited without valid fixed-endpoint completion; no automatic scientific retry.')
            if st['status']=='failed':failed.append(j['id'])
        g=gpu();ram=ram_available();free=shutil.disk_usage(R).free
        cpu_est=0.;gpu_reserved=0;ram_reserved=0
        for j in active:
            rt=state['jobs'][j['id']]['runtime'];ps=process(rt['pid']);prior=cpu_samples.get(j['id'])
            estimate=1.
            if prior and ps:
                interval=tick-prior['time'];estimate=max(.05,min(4.,(ps['cpu_ticks']-prior['ticks'])/hz/max(.01,interval)))
                estimate=.6*prior['ewma']+.4*estimate
            if ps:cpu_samples[j['id']]={'time':tick,'ticks':ps['cpu_ticks'],'ewma':estimate}
            cpu_est+=max(.2,estimate);gpu_reserved+=j['gpu_reserve_mib'];ram_reserved+=j['ram_reserve_mib']*2**20
        owned={state['jobs'][j['id']]['runtime']['pid']:j for j in active if j['device']!='cpu'}
        known_memory={x['pid']:x['used_mib']for x in g['processes']if x['pid']in owned}
        external_used=max(0.,g['used_mib']-sum(known_memory.values()))
        # Namespace-unmatched NVML PIDs are not guessed: their memory remains
        # external, conservatively in addition to all pending own peak growth.
        gpu_peak_reserved=external_used+sum(max(j['gpu_reserve_mib'],known_memory.get(pid,0.))for pid,j in owned.items())
        future_disk=0
        for j in jobs:
            if j['id']not in done_ids and state['jobs'][j['id']]['status']!='failed':
                future_disk+=max(0,j['output_reserve_mib']*2**20-folder_size(Path(j['output'])))
        pending=sorted((j for j in jobs if state['jobs'][j['id']]['status']=='pending'),key=lambda j:(j['priority'],j['seed'],j['domain'],j['method']))
        blocked={};launched=0
        for j in pending:
            reason=None
            if not set(j['depends'])<=done_ids:reason='dependency'
            elif not approval(j,cache):reason='family_preflight'
            elif len(active)>=POLICY['max_jobs']or cpu_est+1.>POLICY['target_cpu']:reason='cpu_throughput'
            elif ram<j['ram_reserve_mib']*2**20+POLICY['ram_floor']:reason='ram'
            elif free<POLICY['disk_floor']+future_disk:reason='disk_future_outputs'
            elif j['method']=='pgr'and sum(x['method']=='pgr'and x['kind']=='training'for x in active)>=2 and any(x['device']!='cpu'and x['method']not in ['pgr']and x['kind']=='training'for x in pending):reason='pgr_balance_other_gpu_families'
            elif j['device']!='cpu' and max(g['used_mib'],gpu_peak_reserved)+j['gpu_reserve_mib']+POLICY['gpu_floor']>g['total_mib']:reason='gpu_peak_capacity'
            if reason:blocked[reason]=blocked.get(reason,0)+1;continue
            args=command(j,byid);log=R/'runtime'/f"{j['id']}.log"
            prior=exact_workers(args)
            if prior:
                prior.update(device=j['device'],log=str(log),started_at=tick,approval_sha256=sha(j['approval']))
                state['jobs'][j['id']].update(status='running',runtime=prior)
                active.append(j);cpu_est+=1.;gpu_peak_reserved+=j['gpu_reserve_mib'];continue
            out=Path(j['output']);assert not out.exists()or not any(out.iterdir()),('unregistered output exists',str(out))
            st=state['jobs'][j['id']];st.update(status='launching',launch_intent={'argv':args,'log':str(log),'time':tick})
            write(R/'runtime/queue_state.json',state)
            f=log.open('a')
            proc=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=runtime_env(j),start_new_session=True);f.close()
            ps=None
            for _ in range(50):
                ps=process(proc.pid)
                if ps:break
                time.sleep(.01)
            assert ps
            # Limit host library thread pools while retaining two execution cores.
            allowed=sorted(os.sched_getaffinity(0));idx=jobs.index(j)%len(allowed)
            affinity=sorted({allowed[idx],allowed[(idx+len(allowed)//2)%len(allowed)]})
            try:os.sched_setaffinity(proc.pid,affinity)
            except ProcessLookupError:pass
            rt=dict(pid=proc.pid,start_ticks=ps['start_ticks'],boot_id=BOOT,argv=args,device=j['device'],cpu_affinity=affinity,log=str(log),started_at=tick,approval_sha256=sha(j['approval']))
            st=state['jobs'][j['id']];st.update(status='running',runtime=rt,attempts=st.get('attempts',0)+1);children[j['id']]=proc
            active.append(j);cpu_est+=1.;gpu_reserved+=j['gpu_reserve_mib'];gpu_peak_reserved+=j['gpu_reserve_mib'];launched+=1;last_start=tick
            write(R/'runtime/queue_state.json',state)
            # Stagger CUDA context and model allocation without a fixed two-job cap.
            if j['device']!='cpu':break
            if launched>=8:break
        progress=[]
        for j in active:
            p=read(Path(j['output'])/'progress.json',{})
            progress.append({'id':j['id'],'kind':j['kind'],'method':j['method'],'device':j['device'],'pid':state['jobs'][j['id']]['runtime']['pid'],
                             'training_steps':p.get('training_steps',p.get('train_steps',0)),'phase':p.get('phase',p.get('status','starting'))})
        counts={s:sum(st['status']==s for st in state['jobs'].values())for s in ['pending','running','complete','failed']}
        train_counts={s:sum(state['jobs'][j['id']]['status']==s for j in jobs if j['kind']=='training')for s in counts}
        status=dict(schema='chapter4-local-formal-live-status-v1',updated_unix=tick,pid=os.getpid(),phase='running'if active or pending else'failed'if failed else'training_and_evaluation_complete',
                    counts=counts,training_counts=train_counts,evaluation_counts={s:counts[s]-train_counts[s]for s in counts},blocked_reasons=blocked,
                    active=progress,failed=failed,resources={'cpu_effective_quota':42,'cpu_estimated_current_workers':cpu_est,'gpu':g,'ram_available':ram,'HY_free_bytes':free,'remaining_output_reservation_bytes':future_disk},
                    scientific_plan_sha256=seal['sha256'],new_training_started=bool(active or done_ids),automatic_old_queue_recovery=False)
        write(R/'status.json',status);write(R/'runtime/queue_state.json',state)
        if tick-last_hashcheck>120:
            assert sha(R/'plan.json')==seal['sha256'];cache.clear();last_hashcheck=tick
        if counts['complete']==190:
            write(R/'postprocess_ready.json',dict(status='READY_FOR_INDEPENDENT_ACCEPTANCE',plan_sha256=seal['sha256'],training=90,evaluation=100))
            script=R/'execution/postprocess.py'
            if script.exists():
                f=(R/'runtime/postprocess.log').open('a');code=subprocess.call([sys.executable,str(script)],stdout=f,stderr=subprocess.STDOUT,env=runtime_env({'id':'postprocess','device':'cpu'}));f.close()
                status.update(phase='complete'if code==0 else'postprocess_failed',postprocess_exit_code=code);write(R/'status.json',status)
            else:
                status['phase']='postprocess_preparation_missing';write(R/'status.json',status)
            rec=read(R/'recovery_launch.json');rec['status']=status['phase'];rec['history_command']=rec.pop('command');rec['command']=None;write(R/'recovery_launch.json',rec);return
        if failed and not active and all(state['jobs'][j['id']]['status']!='pending'or any(state['jobs'][d]['status']=='failed'for d in j['depends'])for j in jobs):return
        time.sleep(3 if launched else 10)

if __name__=='__main__':main()
