import argparse,json,os,queue,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from allocation_efficiency.common import ROOT,write_json,source_hashes

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['engineering','pilot','formal'],required=True)
 p.add_argument('--workers',type=int,default=12);args=p.parse_args()
 protocol=json.loads((ROOT/'protocol.json').read_text());cfg=protocol[args.stage]
 if source_hashes()!=json.loads((ROOT/'implementation_lock.json').read_text()):raise RuntimeError('Frozen source changed')
 if args.stage in ('pilot','formal'):
  if json.loads((ROOT/'prelaunch_gate.json').read_text())['status']!='PASS':raise RuntimeError('Engineering gate not PASS')
 if args.stage=='formal':
  gate=json.loads((ROOT/'analysis/pilot_learning_gate.json').read_text())
  if gate['status']!='PASS' or not gate.get('advance_to_formal',False):raise RuntimeError('Pilot learning gate failed; stop as registered')
 jobs=[(d,m,s) for s in cfg['seeds'] for m in cfg['methods'] for d in protocol['domains']]
 assert len(jobs)==cfg['jobs'];(ROOT/'logs').mkdir(parents=True,exist_ok=True);slots=queue.Queue()
 for i in range(args.workers):slots.put(i)
 cpus=sorted(os.sched_getaffinity(0))
 def one(job):
  d,m,s=job;slot=slots.get();name=f'{d}__{m}__seed{s}'
  try:
   affinity=[cpus[(slot*4+i)%len(cpus)] for i in range(min(4,len(cpus)))];out=ROOT/args.stage/name
   log=ROOT/'logs'/f'{args.stage}_{name}.log'
   if out.exists():raise FileExistsError(out)
   cmd=['taskset','-c',','.join(map(str,affinity)),sys.executable,str(Path(__file__).parent/'run.py'),
    '--domain',d,'--method',m,'--seed',str(s),'--episodes',str(cfg['episodes']),'--eval-every',str(cfg['evaluation_interval']),
    '--dev-paths',str(cfg['dev_paths']),'--holdout-paths',str(cfg['holdout_paths']),
    '--path-master',str(cfg['train_master']),'--test-master',str(cfg['test_master']),'--output',str(out)]
   env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
   with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
   return dict(name=name,returncode=r.returncode,output=str(out),log=str(log),slot=slot,cpus=affinity)
  finally:slots.put(slot)
 start=time.monotonic();results=[]
 with ThreadPoolExecutor(max_workers=args.workers) as ex:
  for f in as_completed([ex.submit(one,j) for j in jobs]):
   r=f.result();results.append(r);print(json.dumps(r),flush=True)
   write_json(ROOT/f'{args.stage}_status.json',dict(finished=len(results),total=len(jobs),results=results,seconds=time.monotonic()-start))
 write_json(ROOT/f'{args.stage}_summary.json',dict(complete=all(r['returncode']==0 for r in results),results=results,seconds=time.monotonic()-start))
 if any(r['returncode'] for r in results):raise SystemExit(1)
if __name__=='__main__':main()
