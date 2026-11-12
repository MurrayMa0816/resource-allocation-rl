"""Exact16 full-spend execution with durable launch leases and external growth guards."""
from __future__ import annotations
import contextlib,fcntl,math,os,shlex,socket,subprocess,sys,time,uuid,stat
from pathlib import Path
from . import common as c
sha,read_json,write_json=c.sha,c.read_json,c.write_json
GIB=1024**3
THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
def require(condition, message):
    if not condition:
        raise ValueError(message)

def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def pid_identity(pid):
    folder = Path('/proc')/str(pid)
    stat = (folder/'stat').read_text()
    fields = stat[stat.rfind(')')+2:].split()
    return dict(pid=int(pid), start_ticks=int(fields[19]), state=fields[0],
        argv=[value.decode() for value in (folder/'cmdline').read_bytes().split(b'\0') if value])

def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
    devices = {}
    for line in result.stdout.splitlines():
        index, identifier, total, free, utilization = [value.strip() for value in line.split(',')]
        devices[int(index)] = dict(index=int(index), uuid=identifier, total_bytes=int(float(total)*1024**2),
            free_bytes=int(float(free)*1024**2), utilization=float(utilization))
    require(bool(devices), 'No actual NVIDIA GPU is available')
    return devices

def cgroup_resources():
    """Respect the container quota rather than host-visible processor count."""
    import psutil
    candidates = [(Path('/sys/fs/cgroup/cpu.max'), 'v2'),
        (Path('/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us'), 'v1'),
        (Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'), 'v1')]
    limit = float(len(os.sched_getaffinity(0)))
    cpu_usage = None
    for path, kind in candidates:
        if not path.exists():
            continue
        if kind == 'v2':
            quota, period = path.read_text().split()
            if quota != 'max':
                limit = min(limit, int(quota)/int(period))
            stat = dict(line.split() for line in path.with_name('cpu.stat').read_text().splitlines())
            cpu_usage = int(stat['usage_usec'])/1e6
        else:
            quota = int(path.read_text())
            period = int(path.with_name('cpu.cfs_period_us').read_text())
            if quota > 0:
                limit = min(limit, quota/period)
            usage_path = path.with_name('cpuacct.usage')
            if usage_path.exists():
                cpu_usage = int(usage_path.read_text())/1e9
        break
    available = int(psutil.virtual_memory().available)
    for limit_path, current_path in ((Path('/sys/fs/cgroup/memory.max'),Path('/sys/fs/cgroup/memory.current')),
            (Path('/sys/fs/cgroup/memory/memory.limit_in_bytes'),Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'))):
        if limit_path.exists() and current_path.exists():
            raw = limit_path.read_text().strip()
            if raw != 'max':
                bound = int(raw)
                if 0 < bound < 2**60:
                    available = min(available, max(0, bound-int(current_path.read_text())))
            break
    return dict(cpu_limit=limit, memory_available_bytes=available, cgroup_cpu_seconds=cpu_usage)

class ResourceMonitor:
    def __init__(self):
        self.last = None
        self.previous = {}
        self.demand = {}

    def sample(self, running):
        import psutil
        now = time.monotonic()
        value = cgroup_resources()
        observed, rss = {}, {}
        for identifier, entry in running.items():
            record, job = entry['record'], entry['job']
            if not process_live(record):
                continue
            try:
                process = psutil.Process(record['pid'])
                cpu = process.cpu_times()
                seconds = cpu.user+cpu.system
                memory = process.memory_info().rss
            except psutil.NoSuchProcess:
                continue
            previous = self.previous.get(record['pid'])
            rate = max(0., (seconds-previous[1])/max(.01, now-previous[0])) if previous else job['estimated_cpu_cores']
            self.previous[record['pid']] = (now, seconds)
            self.demand[identifier] = max(self.demand.get(identifier, job['estimated_cpu_cores']), min(1.25, rate))
            observed[identifier], rss[identifier] = rate, memory
        cgroup = value.pop('cgroup_cpu_seconds')
        total = None
        if self.last is not None and cgroup is not None and self.last[1] is not None:
            total = max(0., (cgroup-self.last[1])/max(.01, now-self.last[0]))
        self.last = now, cgroup
        value.update(cpu_observed_by_job=observed, cpu_demand_by_job={identifier:self.demand[identifier] for identifier in observed},
            worker_rss_bytes=rss, external_cpu=max(0., (total or sum(observed.values()))-sum(observed.values())),
            gpu=gpu_snapshot())
        return value

def gpu_memory_profile(job, record):
    observed = 0
    path = Path(job['output'])/'gpu_runtime.json'
    if path.exists():
        value = read_json(path)
        runtime = value.get('gpu_runtime', value)
        observed = int(runtime.get('peak_cuda_reserved_bytes', runtime.get('peak_reserved_bytes', 0)))
        if observed:
            observed += 384*1024**2  # Context/allocator memory outside torch tensors.
    return observed, max(job['estimated_gpu_bytes'], math.ceil(observed*1.25))

def select_gpu(job, registration, snapshot, running):
    demand = sum(snapshot['cpu_demand_by_job'].values())
    if demand+job['estimated_cpu_cores']+snapshot['external_cpu']+registration['cpu_reserve_cores'] > snapshot['cpu_limit']*registration['max_cpu_demand_ratio']:
        return None
    growth = sum(max(0, entry['job']['estimated_memory_bytes']-snapshot['worker_rss_bytes'].get(identifier, 0)) for identifier, entry in running.items())
    if snapshot['memory_available_bytes'] < registration['memory_reserve_bytes']+growth+job['estimated_memory_bytes']:
        return None
    candidates = []
    for physical in registration['physical_gpu_indices']:
        device = snapshot['gpu'].get(physical)
        if device is None:
            raise ValueError('A registered physical GPU disappeared: '+str(physical))
        device_jobs = [entry for entry in running.values() if entry['record']['physical_gpu_index'] == physical]
        future = sum(max(0, bound-observed) for observed, bound in (gpu_memory_profile(entry['job'], entry['record']) for entry in device_jobs))
        free_after_reserves = device['free_bytes']-future-registration['gpu_reserve_bytes']
        if free_after_reserves >= job['estimated_gpu_bytes']:
            candidates.append((free_after_reserves-job['estimated_gpu_bytes'], -device['utilization'], -len(device_jobs), -physical, device))
    return max(candidates, key=lambda row:row[:4])[-1] if candidates else None

def checkpoint_rate(job):
    prefix = 'evaluation_' if job['family'] == 'proposed' else 'development_'
    values = []
    for path in Path(job['output']).glob(prefix+'*.json'):
        try:
            values.append((int(path.stem.removeprefix(prefix)), path.stat().st_mtime))
        except (ValueError, FileNotFoundError):
            continue
    values.sort()
    if len(values) < 3:
        return None
    first, last = values[-3], values[-1]
    if last[1] <= first[1]:
        return None
    config=read_json(job['config'])
    boundary=0 if job['family']=='proposed' or job['method']=='cmorl' else max(
        int(config.get('start_steps',0)),int(config.get('learning_starts',0)))
    if first[0]<boundary<=last[0]:
        return None
    return dict(step=last[0], rate=(last[0]-first[0])/(last[1]-first[1]),
        phase='learning' if first[0]>=boundary else 'warmup')

class ThroughputGuard:
    def __init__(self):
        self.reference = {}
        self.bad_since = None

    def observe(self, samples, now):
        comparable, slow = [], []
        for identifier, sample in samples.items():
            # Require two updated policies before comparing training throughput.
            if sample['step'] < 1024:
                continue
            key=identifier,sample.get('phase','learning')
            previous = self.reference.get(key)
            self.reference[key] = max(previous or 0., sample['rate'])
            if previous:
                comparable.append(identifier)
                if sample['rate'] < .45*previous:
                    slow.append(identifier)
        severe = len(comparable) >= 3 and len(slow) >= math.ceil(.6*len(comparable))
        self.bad_since = (self.bad_since if self.bad_since is not None else now) if severe else None
        return dict(paused=self.bad_since is not None and now-self.bad_since >= 120,
            severe=severe, comparable=len(comparable), slow=slow, bad_since=self.bad_since, samples=samples)


_raw_identity=pid_identity
def pid_identity(pid):
    boot=boot_id();start=None
    for attempt in range(11):
        value=_raw_identity(pid)
        if start is None:start=value['start_ticks']
        require(boot_id()==boot and value['start_ticks']==start,'PID/kernel changed during identity sampling')
        if value['state']=='Z':raise ProcessLookupError('Original process ended')
        if value['argv']:return value
        if attempt<10:time.sleep(.025)
    raise ValueError('Never record an alive empty procfs command line: '+str(pid))


def process_live(record,*,inspect_environment=False):
    require(record['boot_id']==boot_id(),'Kernel boot changed; do not adopt stale worker state')
    try:
        value=pid_identity(record['pid'])
        if value['start_ticks']!=record['start_ticks']:return False
        require(value['argv']==record['argv'],'Owned worker command changed')
        if inspect_environment:
            values=(Path('/proc')/str(record['pid'])/'environ').read_bytes().split(b'\0')
            env=dict(x.decode().split('=',1) for x in values if b'=' in x)
            require(all(env.get(k)==v for k,v in record['environment'].items()),'Owned GPU/thread environment changed')
        return True
    except (FileNotFoundError,ProcessLookupError):return False


@contextlib.contextmanager
def lock(root):
    with (Path(root)/'master.lock').open('a+') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another validation controller owns this scope')
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def valid_record(record,job,owner,master_sha):
    require(record['owner']==owner and record['master_manifest_sha256']==master_sha and record['hostname']==socket.gethostname()
        and record['job_id']==job['id'] and record['argv']==record['requested_argv']==job['argv']
        and record['registration_sha256']==job['registration_sha256'],'Unowned or changed validation worker')
    require(record['logical_device']==job['device'],'Logical device changed')
    visible='' if job['device']=='cpu' else str(record['physical_gpu_index'])
    require(record['environment']['CUDA_VISIBLE_DEVICES']==visible
        and all(record['environment'][key]=='1' for key in THREAD_ENV),'Physical GPU mapping or threads changed')


def completed(job):
    reg=c.read_json(job['registration_path'])
    require(c.sha(job['registration_path'])==job['registration_sha256'],'Worker registration changed')
    return c.validate_completed(job,reg)


def census(job):
    """Independent procfs discovery; an unrecorded process never means pending."""
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:value=_raw_identity(int(path.name))
        except (FileNotFoundError,ProcessLookupError):continue
        argv=value['argv']
        if str(job['output']) not in argv or 'chapter5_full_spend_20261104.worker' not in argv or value['state']=='Z':continue
        require(argv==job['argv'],'A live full-spend worker has changed/unregistered arguments')
        value=pid_identity(value['pid']);value['boot_id']=boot_id();result.append(value)
    require(len(result)<=1,'Duplicate live full-spend workers: '+job['id'])
    return result


def exclusive_json(path,value):
    """Durable intent must reach disk before a child can exist."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        import json
        json.dump(value,handle,ensure_ascii=False,allow_nan=False,indent=2);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    directory=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


def inventory(jobs,root,owner,master_sha):
    root=Path(root);done,running,failed,pending={},{},{},[]
    for job in jobs:
        record_path=root/'worker_pids'/(job['id']+'.json');lease_path=root/'launch_leases'/(job['id']+'.json')
        record=read_json(record_path) if record_path.exists() else None
        lease=read_json(lease_path) if lease_path.exists() else None
        if lease:valid_record(lease,job,owner,master_sha)
        if record:
            require(lease is not None,'A worker record lost its durable launch lease');valid_record(record,job,owner,master_sha)
        matches=census(job)
        if not matches and record and process_live(record,inspect_environment=True):
            # Known live identity cannot be declared dead because one procfs
            # census sampled a transient empty command line.
            time.sleep(.025);matches=census(job)
            require(bool(matches),'Independent census is uncertain for a known live worker')
        if matches:
            require(lease is not None,'Unowned live worker; never relaunch it')
            first=matches[0];time.sleep(.025);again=census(job)
            require(len(again)==1 and all(again[0][k]==first[k] for k in ('pid','start_ticks','boot_id','argv')),
                'Worker identity is uncertain during recovery')
            if record:require(all(record[k]==first[k] for k in ('pid','start_ticks','boot_id','argv')),'Durable worker identity changed')
            if lease.get('pid') is not None:require(lease['pid']==first['pid'],'Launch lease points to another PID')
            require(lease['boot_id']==first['boot_id'] and (lease.get('start_ticks') is None or lease['start_ticks']==first['start_ticks']),
                'Launch lease kernel/start identity changed')
            adopted=dict(lease,**first,status='running');require(process_live(adopted,inspect_environment=True),'Adopted environment changed')
            write_json(record_path,adopted);running[job['id']]=dict(job=job,record=adopted,process=None)
        elif (Path(job['output'])/'completed.json').exists():
            require(record is not None and lease is not None,'Completion lacks durable ownership proof')
            done[job['id']]=completed(job)
        elif record:failed[job['id']]='Recorded worker ended without a valid completion; no automatic retry'
        elif lease:raise ValueError('Unresolved durable launch intent; no automatic replacement: '+job['id'])
        elif Path(job['output']).exists():raise ValueError('Unowned existing output: '+job['id'])
        else:pending.append(job)
    return done,running,failed,pending


def spawn(job,root,owner,master_sha,device):
    root=Path(root);record_path=root/'worker_pids'/(job['id']+'.json');lease_path=root/'launch_leases'/(job['id']+'.json')
    require(not record_path.exists() and not lease_path.exists() and not Path(job['output']).exists() and not census(job),
        'Never overwrite, retry or duplicate a started full-spend job')
    require(c.sha(job['registration_path'])==job['registration_sha256'] and c.sha(job['config'])==job['config_sha256'],'Registered recipe changed')
    env=dict(os.environ);env.update({key:'1' for key in THREAD_ENV});env['CUDA_VISIBLE_DEVICES']=str(device['index'])
    env['CH5_PHYSICAL_GPU_ID']=str(device['index']);env['CH5_PHYSICAL_GPU_UUID']=device['uuid']
    environment={key:env[key] for key in THREAD_ENV+('CUDA_VISIBLE_DEVICES','CH5_PHYSICAL_GPU_ID','CH5_PHYSICAL_GPU_UUID')}
    record=dict(job_id=job['id'],owner=owner,master_manifest_sha256=master_sha,registration_sha256=job['registration_sha256'],
        hostname=socket.gethostname(),boot_id=boot_id(),argv=job['argv'],requested_argv=job['argv'],status='spawn_intent',
        physical_gpu_index=device['index'],logical_device=job['device'],environment=environment,started_unix_time=time.time())
    exclusive_json(lease_path,record)
    log=root/'worker_logs'/(job['id']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('xb') as handle:
        process=subprocess.Popen(job['argv'],cwd=str(c.WORKSPACE),env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    # Persist the actual returned PID before the stable procfs sampling retry.
    record.update(pid=process.pid,status='child_created');write_json(lease_path,record)
    identity=pid_identity(process.pid);require(identity['argv']==job['argv'],'Spawned canonical command changed')
    record.update(identity,status='running');write_json(record_path,record);write_json(lease_path,record)
    return dict(job=job,record=record,process=process)


def poll(root,running,done,failed):
    for identifier,entry in list(running.items()):
        process,record=entry['process'],entry['record']
        if (process.poll() is None if process is not None else process_live(record)):continue
        code=process.returncode if process is not None else None
        record.update(status='exited',returncode=code,exited_unix_time=time.time());write_json(Path(root)/'worker_pids'/(identifier+'.json'),record)
        del running[identifier]
        try:
            require(process is None or code==0,'Worker exit code '+str(code));done[identifier]=completed(entry['job'])
        except (ValueError,KeyError,OSError) as error:failed[identifier]=type(error).__name__+': '+str(error)


def allocated_bytes(folder):
    total=0
    if Path(folder).exists():
        for path in Path(folder).rglob('*'):
            try:value=path.lstat()
            except FileNotFoundError:continue
            if stat.S_ISREG(value.st_mode):total+=value.st_blocks*512
    return total


def external_job_bound(job):
    """Old five-round clock=train+selection, no final holdout, no new ledger."""
    argv=job['argv'];option=lambda name,default:int(argv[argv.index(name)+1]) if name in argv else default
    domain=job['domain'];n,h,d=c.DOMAINS[domain]['N'],c.DOMAINS[domain]['horizon'],19 if domain=='radio' else 13
    budget=int(job['learning_steps']);dev=option('--dev-paths',32);interval=option('--eval-every',512)
    physical=budget+(budget//interval+1)*dev*h;episodes=physical//h
    arrays=physical*(n+2*d+3)*8+episodes*n*8
    endpoint=(192 if job['method']=='cmorl' else 24)*1024**2
    raw=arrays+episodes*16384+(budget//h+256)*8192+budget*2048+endpoint+24*1024**2
    return math.ceil(raw*1.15/(16*1024**2))*16*1024**2


def external_reservations(policy):
    """Read-only accounting for up to60 old jobs, including unregistered tail."""
    import psutil
    root=Path(policy['external_scope']);require(root.is_dir(),'Protected old five-round scope is unavailable')
    jobs={}
    for path in sorted((root/'plans').glob('*.json')):
        plan=read_json(path)
        if plan.get('status')!='reviewed_for_execution':continue
        for job in plan['jobs']:
            require(job['id'] not in jobs,'Duplicate job in old five-round reviewed plans');jobs[job['id']]=job
    require(len(jobs)<=policy['external_max_jobs'],'Old queue exceeded its protected60-job scope; stop new admission')
    remaining=0;live=[];not_started=0;closed=0
    for identifier,job in jobs.items():
        folder=Path(job['output']);complete=folder/'completed.json';record=root/'worker_pids'/(identifier+'.json')
        if complete.exists():
            value=read_json(complete)
            require(value.get('status')=='complete' and value.get('source_unchanged') is True and value.get('endpoint_reload_verified') is True
                and value.get('config_sha256')==job['config_sha256']
                and sum(value['counts'][k]['physical_steps'] for k in ('train','selection'))==job['learning_steps'],
                'Old completion is not a closed growth reservation: '+identifier)
            closed+=1;continue
        remaining+=max(0,external_job_bound(job)-allocated_bytes(folder))
        if record.exists():
            identity=read_json(record)
            try:
                actual=pid_identity(identity['pid'])
                if actual['start_ticks']!=identity['start_ticks']:continue
                require(actual['argv']==identity['argv']==job['argv'] and identity['boot_id']==boot_id(),'Old worker identity changed')
                env=(Path('/proc')/str(identity['pid'])/'environ').read_bytes().split(b'\0')
                env=dict(x.decode().split('=',1) for x in env if b'=' in x)
                physical=int(env['CUDA_VISIBLE_DEVICES']);require(physical in policy['physical_gpu_indices'],'Protected old physical GPU changed')
                rss=psutil.Process(identity['pid']).memory_info().rss
                runtime=read_json(folder/'gpu_runtime.json') if (folder/'gpu_runtime.json').exists() else {}
                observed=int(runtime.get('current_cuda_reserved_bytes',0))+384*1024**2
                live.append(dict(id=identifier,pid=identity['pid'],physical=physical,rss=rss,gpu_observed=observed))
            except (FileNotFoundError,ProcessLookupError,psutil.NoSuchProcess):continue
        elif not (folder/'failure.json').exists():not_started+=1
    unknown=policy['external_max_jobs']-len(jobs)
    worst=0
    for domain in c.DOMAINS:
        dummy=dict(domain=domain,method='cmorl',learning_steps=61440,argv=['--dev-paths','32','--eval-every','512'])
        worst=max(worst,external_job_bound(dummy))
    remaining+=unknown*worst
    future_slots=min(max(0,policy['external_max_concurrent']-len(live)),unknown+not_started)
    memory_growth=sum(max(0,policy['external_worker_memory_bytes']-v['rss']) for v in live)+future_slots*policy['external_worker_memory_bytes']
    gpu_growth={index:future_slots*policy['external_worker_gpu_bytes'] for index in policy['physical_gpu_indices']}
    for value in live:gpu_growth[value['physical']]+=max(0,policy['external_worker_gpu_bytes']-value['gpu_observed'])
    return dict(scope=str(root),observed_jobs=len(jobs),closed_jobs=closed,maximum_jobs=policy['external_max_jobs'],
        unknown_future_jobs=unknown,remaining_disk_bytes=remaining,live=live,future_worker_slots=future_slots,
        future_cpu_cores=float(future_slots),memory_growth_bytes=memory_growth,gpu_growth_bytes=gpu_growth,
        utc_unix=time.time(),read_only=True)


def storage_ok(jobs,done,policy,external):
    import shutil
    root=Path(jobs[0]['output']).parent
    while not root.exists():root=root.parent
    remaining=sum(max(0,j['estimated_disk_bytes']-allocated_bytes(j['output'])) for j in jobs if j['id'] not in done)
    free=shutil.disk_usage(root).free
    required=remaining+external['remaining_disk_bytes']+policy['disk_reserve_bytes']+policy['report_reserve_bytes']
    return required<=free,dict(free_bytes=free,new_remaining_bytes=remaining,old_five_round_remaining_bytes=external['remaining_disk_bytes'],
        reserve_bytes=policy['disk_reserve_bytes'],report_bytes=policy['report_reserve_bytes'],required_bytes=required)


def run(root,registration,jobs,finalize,*,poll_seconds=2.):
    root=Path(root);master_sha=sha(root/'master_manifest.json');reg_sha=sha(root/'registration.json')
    with lock(root):
        path=root/'owner.json'
        if path.exists():
            owner_record=read_json(path);require(owner_record['boot_id']==boot_id() and owner_record['master_manifest_sha256']==master_sha,'Scope owner/kernel changed')
        else:
            owner_record=dict(owner=str(uuid.uuid4()),boot_id=boot_id(),hostname=socket.gethostname(),master_manifest_sha256=master_sha);write_json(path,owner_record)
        owner=owner_record['owner'];done,running,failed,pending=inventory(jobs,root,owner,master_sha)
        write_json(root/'controller_pid.json',dict(pid_identity(os.getpid()),boot_id=boot_id(),owner=owner,registration_sha256=reg_sha))
        monitor,guard=ResourceMonitor(),ThroughputGuard();external=None;external_at=0.;throughput=None;sample_at=0.
        def emit(phase,**extra):
            value=dict(schema='chapter5-full-spend-controller-v1',phase=phase,pid=os.getpid(),owner=owner,unix_time=time.time(),registration_sha256=reg_sha,
                planned=16,completed=len(done),running={i:e['record'] for i,e in running.items()},pending=[j['id'] for j in pending],failed=failed,
                automatic_more_seeds=False,automatic_retry=False,external_baseline_jobs=0)
            value.update(extra);write_json(root/'status.json',value);return value
        while True:
            require(sha(root/'master_manifest.json')==master_sha and sha(root/'registration.json')==reg_sha,'Registered16 scope changed')
            poll(root,running,done,failed)
            if failed and not running:return emit('failed_jobs_require_review',controller_stopped=True)
            if not running and not pending:return finalize(done,emit)
            now=time.time()
            if external is None or now-external_at>=10:external=external_reservations(registration['resources']);external_at=now
            snapshot=monitor.sample(running)
            if now-sample_at>=10:
                samples={i:v for i,e in running.items() if (v:=checkpoint_rate(e['job'])) is not None}
                throughput=guard.observe(samples,now);sample_at=now
            ok,storage=storage_ok(jobs,{**done,**failed},registration['resources'],external)
            emit('parallel_full_spend' if ok else 'storage_admission_held',resources=snapshot,storage=storage,protected_external=external,throughput=throughput)
            if ok and not (throughput and throughput['paused']):
                adjusted=dict(snapshot,cpu_demand_by_job=dict(snapshot['cpu_demand_by_job']),worker_rss_bytes=dict(snapshot['worker_rss_bytes']),
                    gpu={i:dict(v) for i,v in snapshot['gpu'].items()})
                adjusted['external_cpu']+=external['future_cpu_cores']
                adjusted['memory_available_bytes']=max(0,adjusted['memory_available_bytes']-external['memory_growth_bytes'])
                for index,growth in external['gpu_growth_bytes'].items():adjusted['gpu'][index]['free_bytes']=max(0,adjusted['gpu'][index]['free_bytes']-growth)
                active={(e['job']['variant'],e['job']['domain']) for e in running.values()}
                for job in sorted(pending,key=lambda j:((j['variant'],j['domain']) in active,j['seed'])):
                    device=select_gpu(job,registration['resources'],adjusted,running)
                    if device is None:continue
                    c.validate_registration(job['registration_path']);running[job['id']]=spawn(job,root,owner,master_sha,device);pending.remove(job);break
            time.sleep(poll_seconds)

