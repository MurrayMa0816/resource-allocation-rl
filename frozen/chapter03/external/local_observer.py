#!/usr/bin/env python3
"""Observe and mirror sealed remote results; never control remote training."""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

SCOPE = Path(__file__).resolve().parents[1]
AUDIT = SCOPE / 'report/observer'
META = AUDIT / 'remote_metadata'
RESULTS = SCOPE / 'remote_results'
SSH = ['ssh', '-p', '10706', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
       '-o', 'StrictHostKeyChecking=yes', '-o',
       'UserKnownHostsFile=/hy-tmp/resource_allocation_results/chapter4/baseline_completion_20261104/audit/remote_known_hosts',
       'review@compute.invalid']

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)

def proc_identity(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=fields[19], state=fields[0],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except FileNotFoundError:
        return None

def ssh_json(code, timeout=55):
    r = subprocess.run(SSH + ['python3 -'], input=code, text=True, capture_output=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError('Read-only remote probe failed: ' + r.stderr[-4000:])
    return json.loads(r.stdout)

# This probe reads small queue metadata and hashes only newly completed outputs.
# No scientific package import, remote write, signal or training command occurs.
REMOTE_SNAPSHOT = r'''
from pathlib import Path
import hashlib,json,os,stat,time,datetime
root=Path(ROOT)
known=KNOWN
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def identity(pid):
 try:
  f=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
  return {'start_ticks':f[19],'state':f[0]}
 except FileNotFoundError:return None
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
plan=json.loads((root/'plan.json').read_text())
assert sha(root/'plan.json')==PLAN_SHA,'Plan changed from the local registered plan'
states=json.loads((root/'dispatch_state.json').read_text())
metadata={}
names=['status.json','queue_pid.json','recovery_launch.json','launch_receipt.json','dispatch_state.json',
       'completion_receipt.json','controller_failure.json','controller_failure_resolved.json',
       'dispatch_updates/exit_race_20261105/resume_receipt.json']
names += [str(p.relative_to(root)) for p in sorted((root/'selections').glob('*.json'))]
names += [str(p.relative_to(root)) for p in sorted(root.glob('watchdog*.json'))]
for name in names:
 p=root/name
 if p.is_file():
  text=p.read_text();metadata[name]={'text':text,'sha256':hashlib.sha256(text.encode()).hexdigest(),'bytes':len(text.encode())}
q=json.loads(metadata['queue_pid.json']['text']) if 'queue_pid.json' in metadata else {}
actual=identity(q.get('pid',-1)) if q.get('boot_id')==boot else None
controller_alive=bool(actual and actual['start_ticks']==str(q.get('start_ticks')) and actual['state'] not in ['Z','X'])
status_data=json.loads(metadata['status.json']['text']) if 'status.json' in metadata else {}
unfinished=sum(int(status_data.get('counts',{}).get(k,0)) for k in ['pending','running'])
status_age_seconds=max(0.,time.time()-float(status_data.get('updated',time.time())))
failure=json.loads(metadata['controller_failure.json']['text']) if 'controller_failure.json' in metadata else {}
resolution=json.loads(metadata['controller_failure_resolved.json']['text']) if 'controller_failure_resolved.json' in metadata else {}
try:resolved_epoch=datetime.datetime.fromisoformat(resolution.get('created_utc','1970-01-01T00:00:00+00:00')).timestamp()
except ValueError:resolved_epoch=0.
controller_health={'pid':q.get('pid'),'identity_alive':controller_alive,'actual':actual,'unfinished_jobs_in_snapshot':unfinished,'status_age_seconds':status_age_seconds,'status_stale':bool(unfinished and status_age_seconds>90),'missing_with_unfinished_work':bool(unfinished and not controller_alive),'current_controller_failure':bool(unfinished and not controller_alive and float(failure.get('time',0))>resolved_epoch)}
eligible=[];issues=[];complete=[]
for job in plan['jobs']:
 jid=job['id'];state=states.get(jid,{})
 if state.get('phase')!='complete':continue
 complete.append(jid)
 if jid in known:
  if state.get('receipt_sha256')!=known[jid]:issues.append({'job_id':jid,'error':'Previously mirrored completion fingerprint changed'})
  continue
 try:
  actual=identity(state['pid']) if state.get('boot_id')==boot else None
  if actual and actual['start_ticks']==state.get('start_ticks') and actual['state'] not in ['Z','X']:
   issues.append({'job_id':jid,'error':'Dispatch complete but original worker is still live; not eligible'})
   continue
  output=Path(state['output']).resolve(strict=True)
  assert output.is_relative_to(root/'jobs'/jid)
  completed=output/'completed.json';fingerprint=sha(completed)
  assert fingerprint==state['receipt_sha256'],'Dispatch receipt fingerprint mismatch'
  receipt=json.loads(completed.read_text());assert receipt.get('success') is True
  protocol=output/'protocol.json';assert protocol.is_file()
  if receipt.get('protocol_sha256'):assert sha(protocol)==receipt['protocol_sha256']
  if receipt.get('config_sha256'):assert sha(output/'config.json')==receipt['config_sha256']
  protocol_data=json.loads(protocol.read_text())
  if job['kind']=='train_A':
   assert receipt.get('source_unchanged') is True and protocol_data['source_identity']['package']==plan['source_sha256']
  elif job['kind']!='evaluate_A':
   assert receipt.get('source_unchanged') is True and protocol_data.get('source_sha256')==plan['source_sha256']
  files={}
  def add(p,relative):
   s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink()
   assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
   digest=sha(p);after=p.lstat()
   assert (s.st_ino,s.st_dev,s.st_size,s.st_mtime_ns)==(after.st_ino,after.st_dev,after.st_size,after.st_mtime_ns)
   files[relative]={'relative_path':relative,'remote_path':str(p),'bytes':s.st_size,'sha256':digest,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'device':s.st_dev,'mode':stat.S_IMODE(s.st_mode)}
  for p in sorted(output.iterdir()):
   if p.is_file() and (p.suffix in ['.json','.npz'] or p.name.endswith('.csv.gz')):add(p,p.name)
  if (output/'model').is_dir():
   for p in sorted((output/'model').rglob('*')):
    if p.is_file():add(p,str(p.relative_to(output)))
  for relative,expected in receipt.get('artifacts_sha256',{}).items():
   assert relative in files and files[relative]['sha256']==expected,'Scientific numeric/trajectory artifact binding mismatch'
  for relative,entry in receipt.get('model_files',{}).items():
   name='model/'+relative;assert name in files and files[name]['sha256']==entry['sha256'],'Fixed endpoint binding mismatch'
  numerical={p.name for p in output.iterdir() if p.is_file() and (p.name.endswith('.csv.gz') or p.suffix=='.npz')}
  assert numerical <= set(receipt.get('artifacts_sha256',{})),'Unbound numerical output'
  if job['kind'] in ['main','diagnostic','gate']:
   assert receipt.get('model_files') and (output/'endpoint_reload_gate.json').is_file()
  statistics=root/'runtime_statistics'/(jid+'.json')
  if statistics.is_file():add(statistics,'_remote/runtime_statistics.json')
  assert fingerprint==sha(completed)
  eligible.append({'job_id':jid,'job':job,'state':state,'output':str(output),'completed_sha256':fingerprint,'files':list(files.values()),'remote_process_exited':True,'remote_job_phase':'complete'})
 except Exception as e:issues.append({'job_id':jid,'error':repr(e)})
print(json.dumps({'remote_time':time.time(),'boot_id':boot,'plan_sha256':sha(root/'plan.json'),'metadata':metadata,'eligible':eligible,'issues':issues,'complete_job_ids':complete,'controller_health':controller_health}))
'''

REMOTE_ARCHIVE = r'''
from pathlib import Path
import hashlib,json,os,stat,sys,tarfile
root=Path(ROOT);candidate=CANDIDATE
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
state=json.loads((root/'dispatch_state.json').read_text())[candidate['job_id']]
assert state['phase']=='complete' and state['receipt_sha256']==candidate['completed_sha256']
assert sha(Path(candidate['output'])/'completed.json')==candidate['completed_sha256']
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 for row in candidate['files']:
  p=Path(row['remote_path']);s=p.lstat()
  assert stat.S_ISREG(s.st_mode) and not p.is_symlink()
  assert (s.st_ino,s.st_dev,s.st_size,s.st_mtime_ns)==(row['inode'],row['device'],row['bytes'],row['mtime_ns'])
  assert sha(p)==row['sha256']
  tar.add(p,arcname=row['relative_path'],recursive=False)
assert sha(Path(candidate['output'])/'completed.json')==candidate['completed_sha256']
'''

def remote_code(template, **bindings):
    return '\n'.join(k + '=' + repr(v) for k,v in bindings.items()) + '\n' + template

def verify_local(directory, receipt):
    if receipt.get('status') != 'PASS' or receipt.get('remote_job_phase') != 'complete' or receipt.get('remote_process_exited') is not True:
        raise ValueError('Local transfer commit is not an exited completed remote job')
    for r in receipt['files']:
        p = directory / r['relative_path']
        if not p.is_file() or p.is_symlink() or p.stat().st_size != r['bytes'] or sha(p) != r['sha256']:
            raise ValueError('Mirrored artifact changed: ' + str(p))
    if sha(directory / 'completed.json') != receipt['completed_sha256']:
        raise ValueError('Mirrored completion changed')
    completed = json.loads((directory / 'completed.json').read_text())
    for relative, digest in completed.get('artifacts_sha256', {}).items():
        if sha(directory / relative) != digest:
            raise ValueError('Native numeric/trajectory SHA mismatch')
    for relative, entry in completed.get('model_files', {}).items():
        if sha(directory / 'model' / relative) != entry['sha256']:
            raise ValueError('Native endpoint SHA mismatch')
    return True

def mirror(candidate, plan_sha):
    jid = candidate['job_id']
    target = RESULTS / jid
    if target.exists():
        old = json.loads((target / 'transfer_receipt.json').read_text())
        verify_local(target, old)
        if old['completed_sha256'] != candidate['completed_sha256']:
            raise ValueError('Refuse to overwrite an existing scientific result')
        return old
    incoming = AUDIT / 'incoming'
    incoming.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=jid + '.', dir=incoming) as temp:
        temp = Path(temp);archive = temp / 'sealed.tar';stage = temp / 'result';stage.mkdir()
        code = remote_code(REMOTE_ARCHIVE, ROOT=str(SCOPE), CANDIDATE=candidate)
        with archive.open('wb') as f:
            r = subprocess.run(SSH + ['python3 -'], input=code.encode(), stdout=f, stderr=subprocess.PIPE, timeout=300)
        if r.returncode:
            raise RuntimeError('Sealed result transfer failed: ' + r.stderr.decode(errors='replace')[-3000:])
        expected = {r['relative_path']: r for r in candidate['files']}
        with tarfile.open(archive, 'r:') as tar:
            members = tar.getmembers()
            if len(members) != len(expected) or {m.name for m in members} != set(expected):
                raise ValueError('Archive members differ from the sealed manifest')
            for member in members:
                if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                    raise ValueError('Only exact regular sealed output files can be extracted')
                row = expected[member.name];p = stage / member.name;p.parent.mkdir(parents=True, exist_ok=True)
                h=hashlib.sha256();count=0
                with tar.extractfile(member) as src, p.open('xb') as dst:
                    for data in iter(lambda:src.read(1024*1024),b''):
                        dst.write(data);h.update(data);count+=len(data)
                if h.hexdigest()!=row['sha256'] or count!=row['bytes']:
                    raise ValueError('Transferred file SHA/size mismatch')
                p.chmod(row['mode'])
        receipt = dict(schema='chapter3-local-sealed-remote-result-transfer-v1', status='PASS',
                       transferred_utc=utc(), job_id=jid, kind=candidate['job']['kind'],
                       method=candidate['job']['method'], domain=candidate['job']['domain'], seed=candidate['job']['seed'],
                       remote_output=candidate['output'], remote_job_phase='complete', remote_process_exited=True,
                       completed_sha256=candidate['completed_sha256'],remote_dispatch_receipt_sha256=candidate['state']['receipt_sha256'],
                       source_plan_sha256=plan_sha, remote_dispatch_state=candidate['state'],
                       files=[{k:r[k] for k in ['relative_path','sha256','bytes']} for r in candidate['files']],
                       archive_sha256=sha(archive), copied_runtime_dependencies=False, copied_images=False,
                       remote_writes=0, signals_sent=0, training_control=False)
        write(stage / 'transfer_receipt.json', receipt)
        verify_local(stage,receipt)
        RESULTS.mkdir(parents=True,exist_ok=True)
        stage.rename(target)
        return receipt

def report_callback():
    script = SCOPE / 'execution/postprocess_results.py'
    capsule=json.loads((SCOPE/'postprocess_capsule.json').read_text())
    if sha(script)!=capsule['files']['postprocess_results.py']:
        raise ValueError('Frozen reporting module SHA mismatch')
    if not script.is_file():
        return dict(status='waiting_for_report_module', updated_utc=utc())
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    for key,leaf in [('TMPDIR','tmp'),('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg')]:
        path=SCOPE/'report/runtime'/leaf;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    result=subprocess.run([sys.executable,str(script),'--scope',str(SCOPE)],env=env,capture_output=True,text=True,timeout=240)
    return dict(status='PASS' if result.returncode==0 else 'ERROR',updated_utc=utc(),returncode=result.returncode,
                stdout_tail=result.stdout[-3000:],stderr_tail=result.stderr[-3000:])

def run(args):
    AUDIT.mkdir(parents=True,exist_ok=True)
    lock=(AUDIT/'observer.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another local observer holds the exclusive lock; no duplicate started.')
    identity=proc_identity(os.getpid());command=[sys.executable,str(Path(__file__).resolve()),'--interval',str(args.interval)]
    write(AUDIT/'pid.json',dict(**identity,command=command,script_sha256=sha(__file__),started_utc=utc(),observation_only=True))
    write(AUDIT/'recovery_launch.json',dict(status='running',command=command,lock=str(AUDIT/'observer.lock'),
          verifies_exclusive_lock=True,remote_training_control=False,signals_sent=0,
          note='Run this command only to restore local observation. The exclusive lock prevents a second observer; never run a remote recovery command here.'))
    capsule=json.loads((SCOPE/'postprocess_capsule.json').read_text())
    if sha(__file__)!=capsule['files']['local_observer.py'] or sha(SCOPE/'plan.json')!=capsule['plan_sha256']:
        raise ValueError('Frozen local observer or plan binding differs')
    plan_sha=sha(SCOPE/'plan.json');known={};errors={};futures={};last_report=None;last_snapshot=None
    for p in sorted(RESULTS.glob('*/transfer_receipt.json')):
        try:
            r=json.loads(p.read_text());verify_local(p.parent,r);known[r['job_id']]=r['completed_sha256']
        except Exception as e:errors[p.parent.name]=repr(e)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as transfers, concurrent.futures.ThreadPoolExecutor(max_workers=1) as reports:
        while True:
            tick=time.monotonic()
            for jid,f in list(futures.items()):
                if f.done():
                    try:
                        r=f.result();known[jid]=r['completed_sha256'];errors.pop(jid,None)
                    except Exception as e:errors[jid]=repr(e)
                    del futures[jid]
            if last_report is not None and last_report.done():
                try:write(AUDIT/'report_callback.json',last_report.result())
                except Exception as e:write(AUDIT/'report_callback.json',dict(status='ERROR',updated_utc=utc(),error=repr(e)))
                last_report=None
            try:
                code=remote_code(REMOTE_SNAPSHOT,ROOT=str(SCOPE),KNOWN=known,PLAN_SHA=plan_sha)
                snapshot=ssh_json(code)
                for name,r in snapshot['metadata'].items():
                    content=r['text'].encode()
                    if hashlib.sha256(content).hexdigest()!=r['sha256']:raise ValueError('Metadata transport SHA failed')
                    p=META/name;p.parent.mkdir(parents=True,exist_ok=True)
                    tmp=p.with_name(p.name+'.tmp');tmp.write_bytes(content);tmp.replace(p)
                for candidate in snapshot['eligible']:
                    jid=candidate['job_id']
                    if jid not in known and jid not in futures:
                        futures[jid]=transfers.submit(mirror,candidate,plan_sha)
                remote_status=json.loads(snapshot['metadata']['status.json']['text'])
                last_snapshot=dict(observed_utc=utc(),remote_time=snapshot['remote_time'],remote_status=remote_status,
                   remote_validation_issues=snapshot['issues'],remote_complete_job_ids=snapshot['complete_job_ids'],controller_health=snapshot.get('controller_health',{}))
                if snapshot.get('controller_health',{}).get('missing_with_unfinished_work'):
                    errors['_remote_controller']='Controller not alive with unfinished tasks; snapshot is not live running progress'
                else:errors.pop('_remote_controller',None)
                errors.pop('_snapshot',None)
            except Exception as e:
                errors['_snapshot']=repr(e)
            if last_report is None:last_report=reports.submit(report_callback)
            kinds=collections.Counter()
            for jid in known:
                try:kinds[jid.split('__',1)[0]]+=1
                except Exception:pass
            state=dict(schema='chapter3-local-read-only-observer-status-v1',updated_utc=utc(),pid=os.getpid(),
               interval_seconds=args.interval,plan_sha256=plan_sha,mirrored_verified_jobs=len(known),mirrored_by_prefix=dict(kinds),
               submitted_or_transferring_jobs=sorted(futures),errors=errors,last_remote_snapshot=last_snapshot,
               remote_mutations=0,remote_signals=0,active_gzip_transferred=False,
               historical_controller_failure_is_current_failure=bool((last_snapshot or {}).get('controller_health',{}).get('current_controller_failure',False)))
            write(AUDIT/'latest_status.json',state)
            print(json.dumps(dict(time=utc(),mirrored=len(known),transfers=len(futures),errors=errors,
                   remote_counts=(last_snapshot or {}).get('remote_status',{}).get('counts')),ensure_ascii=False),flush=True)
            if args.once:
                for jid,f in futures.items():
                    try:r=f.result();known[jid]=r['completed_sha256']
                    except Exception as e:errors[jid]=repr(e)
                report_callback()
                return
            if len(known)==len(json.loads((SCOPE/'plan.json').read_text())['jobs']) and not futures and not errors:
                result=report_callback()
                write(AUDIT/'report_callback.json',result)
                report_progress=json.loads((SCOPE/'report/progress.json').read_text())
                if result['status']=='PASS' and report_progress.get('status')=='ALL_RESULTS_COMPLETE':
                    state['completion_state']='all_130_verified_and_reports_ready'
                    write(AUDIT/'latest_status.json',state)
                    write(AUDIT/'recovery_launch.json',dict(status='complete',command=None,remote_mutations=0))
                    return
            time.sleep(max(.1,args.interval-(time.monotonic()-tick)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval',type=int,default=45)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if not 30<=args.interval<=60:parser.error('Metadata interval must be 30–60 seconds')
    for key,leaf in [('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg'),('MPLCONFIGDIR','mpl')]:
        p=AUDIT/'runtime'/leaf;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    sys.dont_write_bytecode=True
    try:run(args)
    except Exception:
        write(AUDIT/'observer_failure.json',dict(updated_utc=utc(),traceback=traceback.format_exc()))
        raise
