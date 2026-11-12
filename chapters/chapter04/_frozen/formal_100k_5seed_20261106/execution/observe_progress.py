"""Read-only C4 observer: reports only; never controls workers or scientific state."""
from __future__ import annotations
import argparse,collections,datetime,fcntl,hashlib,json,os,re,shutil,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'progress_report'
BOOT=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
TAIL_LIMIT=65536
TERMINAL={'complete','postprocess_failed','postprocess_preparation_missing','failed','partial_failed','partialfailed'}

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError,OSError):return default

def write(path,value):
    path=Path(path)
    if not path.is_relative_to(REPORT):raise ValueError('Observer writes only progress_report')
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.writing')
    text=value if isinstance(value,str)else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    temp.write_text(text);os.replace(temp,path)

def process(pid):
    try:
        p=Path('/proc')/str(int(pid));s=(p/'stat').read_text().rsplit(')',1)[1].split()
        argv=[x.decode(errors='replace')for x in(p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]
        return dict(pid=int(pid),state=s[0],start_ticks=int(s[19]),boot_id=BOOT,argv=argv)
    except (FileNotFoundError,ProcessLookupError,ValueError,PermissionError):return None

def controller_state(identity):
    if not identity:return dict(alive=False,reason='controller identity file missing')
    p=process(identity.get('pid',-1))
    alive=bool(p and p['state']!='Z'and p['start_ticks']==identity.get('start_ticks')and p['boot_id']==identity.get('boot_id')and p['argv']==identity.get('argv'))
    return dict(alive=alive,registered_identity=identity,observed_identity=p,
                reason='exact boot/start_ticks/argv match'if alive else'exact registered controller is not alive')

def tail(path,limit=TAIL_LIMIT):
    path=Path(path)
    try:
        with path.open('rb')as f:
            f.seek(0,os.SEEK_END);size=f.tell();offset=max(0,size-limit);f.seek(offset);data=f.read(limit)
        return data.decode(errors='replace'),dict(path=str(path),bytes_read=len(data),file_bytes=size,offset=offset,mtime_ns=path.stat().st_mtime_ns,tail_sha256=hashlib.sha256(data).hexdigest())
    except (FileNotFoundError,OSError):return '',dict(path=str(path),bytes_read=0,file_missing=True)

def last_match(pattern,text):
    found=list(re.finditer(pattern,text,re.MULTILINE))
    if not found:return None
    x=found[-1]
    return dict(groups=x.groups(),offset=x.start(),raw=x.group(0)[:220])

def pgr_details(job,progress,runtime):
    output=Path(job['output']);latest=None;completed=[]
    for folder in sorted(output.glob('diffusion_cycle_*')):
        if not folder.is_dir():continue
        row=dict(directory=str(folder),cycle_name=folder.name)
        for filename in ('diffusion_training_completed.json','cycle_receipt.json'):
            value=read(folder/filename)
            if value is not None:row[filename]=value
        latest=row
        if'cycle_receipt.json'in row:completed.append(row)
    text,evidence=tail(runtime.get('log',ROOT/'runtime'/(job['id']+'.log')))
    diff=last_match(r'\[(\d+)\s*/\s*(\d+)\]\s*loss:\s*[^\r\n]*',text)
    block=last_match(r'Generating split\s+(\d+)\s+of\s+(\d+)',text)
    denoise=last_match(r'sampling time step:\s*[^\r\n]*?(\d+)\s*/\s*(\d+)\s*\[',text)
    out=dict(id=job['id'],worker_phase=progress.get('phase','unknown'),
             latest_recorded_training_steps=progress.get('training_steps',progress.get('train_steps',0)),
             source='actual worker progress, native cycle JSON, final64KiB log only',
             completed_diffusion_cycles=len(completed),latest_cycle=latest,log_tail_evidence=evidence,
             diffusion_progress_from_worker={k:progress[k]for k in ('diffusion_gradient_step','diffusion_gradient_target')if k in progress},
             latest_native_diffusion_log=diff,latest_native_generation_block=block,latest_native_denoising_log=denoise)
    # Preserve both worker phase and raw log evidence. A log-derived hint is
    # never substituted for sampled-step counters or a computed ETA.
    if progress.get('phase')=='diffusion_training'and block and(not diff or block['offset']>diff['offset']):
        out['phase_hint_from_native_log']='conditional_generation_or_last_generation_record'
    elif progress.get('phase')=='diffusion_training':out['phase_hint_from_native_log']='diffusion_training'
    else:out['phase_hint_from_native_log']=progress.get('phase','unknown')
    return out

def table(headers,rows):
    widths=[max([len(str(h))]+[len(str(row[i]))for row in rows])for i,h in enumerate(headers)]
    line=' '.join('='*w for w in widths)
    result=[line,' '.join(str(h).ljust(w)for h,w in zip(headers,widths)),line]
    result+=[' '.join(str(x).ljust(w)for x,w in zip(row,widths))for row in rows]
    return '\n'.join(result+[line])+'\n'

def render(s):
    counts=s['counts'];train=s['training_counts'];evaluation=s['evaluation_counts'];resources=s['resources']
    lines=['第四章正式训练进度','==================','',
           '更新时间：'+s['observed_local_time']+'。本文件为只读观察结果，不启动、恢复或停止任何训练。','',
           '本轮口径：90次新增完整训练＋100次独立终点评价；第三章A的10个已有模型复用，不计新增训练。工程预检不计入以下进度。','',
           '总进度','------','',
           table(['类别','完成','运行','待运行','失败'],[
             ['正式训练',train.get('complete',0),train.get('running',0),train.get('pending',0),train.get('failed',0)],
             ['独立终测',evaluation.get('complete',0),evaluation.get('running',0),evaluation.get('pending',0),evaluation.get('failed',0)]]),
           '控制器阶段：'+str(s['controller_phase'])+'；精确身份存活：'+str(s['controller']['alive'])+'。','',
           '当前并发与资源','--------------','',
           f"当前任务并发{s['concurrency']['total']}项：CPU任务{s['concurrency']['cpu']}项、GPU任务{s['concurrency']['gpu']}项。",
           f"控制器测得本队列CPU约{resources.get('cpu_estimated_current_workers',0):.2f}核；有效额度{resources.get('cpu_effective_quota','未知')}核。",
           f"最近GPU利用率{resources.get('gpu',{}).get('utilization','未知')}%；显存{resources.get('gpu',{}).get('used_mib','未知')}/{resources.get('gpu',{}).get('total_mib','未知')} MiB。",
           f"当前HY可用{s['HY_free_bytes_now']/2**30:.2f} GiB；控制器尚未产出文件预留{resources.get('remaining_output_reservation_bytes',0)/2**30:.2f} GiB。",
           f"正在运行的训练任务最近可核验学习步范围：{s['learning_steps']['active_min']}—{s['learning_steps']['active_max']}；每项正式预算100608步。",
           f"已完成及正在运行任务的学习步记录合计下界：{s['learning_steps']['recorded_total_lower_bound']}步，不包括开发、终测和工程预检。",'',
           '方法进度','--------','',
           table(['方法','训练完成','训练运行','终测完成'],[[x['method'],f"{x['training_complete']}/10",x['training_running'],f"{x['evaluation_complete']}/10"]for x in s['methods']]),
           f"第三章A：复用10个已有模型，独立终测完成{s['inherited_A_evaluations_complete']}/10。",'',
           'PGR原生计算阶段','---------------','',
           'PGR会在已采集的真实数据上训练扩散模型、生成回放数据。此阶段新增环境步数可暂时不增加，GPU仍可能在计算。下面只列实际日志和计数，不推算总进度百分比或完成时间。','']
    if not s['pgr']:lines+=['当前没有运行中的PGR任务。','']
    for p in s['pgr']:
        lines +=[p['id'], '~'*len(p['id']), '',f"worker记录阶段：{p['worker_phase']}；最近学习步：{p['latest_recorded_training_steps']}；已写成的周期收据：{p['completed_diffusion_cycles']}份。"]
        if p['latest_cycle']:lines.append('最近周期目录：'+p['latest_cycle']['cycle_name']+'。')
        x=p['diffusion_progress_from_worker']
        if x:lines.append('worker实际扩散梯度记录：'+str(x.get('diffusion_gradient_step','未知'))+'/'+str(x.get('diffusion_gradient_target','未知'))+'。')
        for key,title in [('latest_native_diffusion_log','最近扩散日志'),('latest_native_generation_block','最近生成分块日志'),('latest_native_denoising_log','最近去噪日志')]:
            if p[key]:lines.append(title+'：``'+p[key]['raw'].replace('\n',' ').replace('\r',' ')+'``。')
        if not x and not any(p[k]for k in ('latest_native_diffusion_log','latest_native_generation_block','latest_native_denoising_log')):lines.append('尚无可读取的扩散/生成进度计数；不据此猜测进度。')
        lines+=['']
    lines+=['异常与边界','----------','']
    lines+=['* '+a for a in s['alerts']]if s['alerts']else['当前未检测到失败或控制器退出告警。']
    lines+=['','该观察器不代替完整科研验收。训练/终测全部完成后，论文是否发布以独立验收与发布收据为准。','',
            '当前运行任务','------------','',
            table(['任务','设备','最近学习步','阶段'],[[x['id'],x['device'],x['training_steps'],x['phase']]for x in s['active']]),
            '依据：status.json、runtime/queue_state.json、各任务progress.json及PGR日志末尾最多65536字节；未读取模型或运行科学库。','']
    return '\n'.join(lines)

def snapshot():
    stamp=time.time();status=read(ROOT/'status.json',{});plan=read(ROOT/'plan.json',{});queue=read(ROOT/'runtime/queue_state.json',{})
    controller=controller_state(read(ROOT/'runtime/controller_identity.json'))
    jobs=plan.get('jobs',[]);states=queue.get('jobs',{});byid={j['id']:j for j in jobs}
    active=[];pgr=[];alerts=[];method_rows=[];recorded_byid={};live_steps=[]
    current=status.get('active',[])
    for row in current:
        job=byid.get(row['id'])
        if not job:alerts.append('状态文件含计划外任务：'+str(row['id']));continue
        p=read(Path(job['output'])/'progress.json',{})
        step=int(p.get('training_steps',p.get('train_steps',row.get('training_steps',0)))or 0)
        actual=dict(row,training_steps=step,phase=p.get('phase',p.get('status',row.get('phase','unknown'))))
        active.append(actual)
        if job['kind']=='training':recorded_byid[job['id']]=step;live_steps.append(step)
        if job['method']=='pgr':pgr.append(pgr_details(job,p,states.get(job['id'],{}).get('runtime',{})))
    for method in plan.get('main_methods',[])+plan.get('ablations',[]):
        train=[j for j in jobs if j['method']==method and j['kind']=='training'];final=[j for j in jobs if j['method']==method and j['kind']=='evaluation']
        count=lambda group,value:sum(states.get(j['id'],{}).get('status')==value for j in group)
        complete=count(train,'complete')
        for job in train:
            if states.get(job['id'],{}).get('status')=='complete':recorded_byid[job['id']]=100608
        method_rows.append(dict(method=method,training_complete=complete,training_running=count(train,'running'),evaluation_complete=count(final,'complete')))
    phase=status.get('phase','status_missing');counts=status.get('counts',{})
    full=counts.get('complete')==190
    if not status:alerts.append('缺少控制器状态文件，无法确认正式队列进度。')
    age=max(0.,stamp-float(status.get('updated_unix',stamp)))
    if age>120:alerts.append(f'控制器状态已{age:.0f}秒未更新；不把旧快照当成实时进度。')
    if counts.get('failed',0):alerts.append('正式任务失败：'+', '.join(status.get('failed',[]))+'；观察器不会自动重试或修改配置。')
    if not controller['alive']and not full:alerts.append('精确登记的控制器已退出，正式任务尚未全部完成；这不是等待资源。请核对控制器日志和登记的恢复入口，观察器不会恢复队列。')
    if phase=='postprocess_failed':alerts.append('训练及终测后处理失败；不能宣称PDF已经更新。')
    if phase=='postprocess_preparation_missing':alerts.append('训练及终测已完成，但后处理入口未准备；PDF更新尚未完成。')
    if full and phase!='complete':alerts.append('190项训练/终测已收齐；仍等待独立验收和论文发布确认。')
    if not controller['alive']and full and phase!='complete':alerts.append('控制器已退出，后处理/论文发布尚未确认。')
    terminal=phase in TERMINAL or(not controller['alive'])or(bool(counts.get('failed',0))and not active)
    inherited=[j for j in jobs if j['family']=='a_reference']
    s=dict(schema='chapter4-readonly-progress-observation-v1',observed_unix=stamp,
           observed_local_time=datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'),
           controller_phase=phase,controller=controller,status_age_seconds=age,
           counts=counts,training_counts=status.get('training_counts',{}),evaluation_counts=status.get('evaluation_counts',{}),
           plan_training_jobs=90,plan_evaluation_jobs=100,engineering_excluded=True,
           concurrency=dict(total=len(active),cpu=sum(x['device']=='cpu'for x in active),gpu=sum(x['device']!='cpu'for x in active)),
           resources=status.get('resources',{}),HY_free_bytes_now=shutil.disk_usage(ROOT).free,
           learning_steps=dict(active_min=min(live_steps)if live_steps else None,active_max=max(live_steps)if live_steps else None,recorded_total_lower_bound=sum(recorded_byid.values())),
           methods=method_rows,inherited_A_evaluations_complete=sum(states.get(j['id'],{}).get('status')=='complete'for j in inherited),
           active=active,pgr=pgr,alerts=alerts,terminal=terminal,
           scientific_plan_sha256=status.get('scientific_plan_sha256'),log_tail_max_bytes=TAIL_LIMIT,
           no_training_or_model_or_status_mutations=True,no_scientific_libraries_imported=True)
    write(REPORT/'progress_snapshot.json',s);write(REPORT/'当前训练进度.rst',render(s));return s

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');parser.add_argument('--interval',type=float,default=30.)
    args=parser.parse_args()
    if args.interval<5 or args.interval>60:raise ValueError('Use an interval between5 and60 seconds')
    REPORT.mkdir(exist_ok=True);lock=(REPORT/'observer.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=process(os.getpid());command=[sys.executable,str(Path(__file__).resolve()),'--watch','--interval',str(args.interval)]
    recovery=dict(schema='chapter4-readonly-observer-recovery-v1',status='watching'if args.watch else'one_snapshot',observer_identity=identity,
                  command=command if args.watch else None,precondition='Only after exact registered observer has exited; this command observes only, never controls training.',does_not_restore_training=True)
    write(REPORT/'observer_recovery.json',recovery)
    try:
        while True:
            s=snapshot()
            if not args.watch or s['terminal']:
                recovery.update(status='finished'if not s['alerts']or s['controller_phase']=='complete'else'finished_with_alert',last_controller_phase=s['controller_phase'],last_alerts=s['alerts'],history_command=recovery['command'],command=None)
                write(REPORT/'observer_recovery.json',recovery)
                return 0 if not s['alerts']or s['controller_phase']=='complete'else 2
            time.sleep(args.interval)
    except BaseException as e:
        write(REPORT/'observer_error.json',dict(error=repr(e),time=time.time(),no_training_control_attempted=True));raise

if __name__=='__main__':sys.exit(main())
