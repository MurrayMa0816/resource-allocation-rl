"""Prepare immutable execution snapshot and source-bound fresh job configs."""
from pathlib import Path
import argparse,hashlib,importlib,json,shutil,sys
APP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP))
from paths import STUDY,C5,C6

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['engineering','formal'],required=True);parser.add_argument('--attempt',type=int,default=0);args=parser.parse_args()
    tag=args.phase+f'_attempt{args.attempt}';frozen=STUDY/'source'/tag
    if frozen.exists():raise FileExistsError(frozen)
    frozen.mkdir()
    for p in APP.rglob('*.py'):
        if '__pycache__'in p.parts:continue
        dest=frozen/p.relative_to(APP);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    deps={str(p):sha(p)for root in [C5,C6]for p in root.rglob('*')if p.is_file()and p.suffix in ['.py','.json','.csv']}
    own={str(p):sha(p)for p in frozen.rglob('*.py')}
    sources={**deps,**own};write(frozen/'source_manifest.json',{'source_files_sha256':sources,'read_only_dependencies':[str(C5),str(C6)]})
    engineering=args.phase=='engineering';episodes=64 if engineering else 3200;every=5184 if engineering else 10368;paths=2 if engineering else 32
    configs=[]
    for method in ['capql','pasta']:
        module=importlib.import_module('methods.'+method+'_adapter');native=module.configuration()
        if method=='pasta':native.update(engineering=engineering,learning_steps=episodes*81,dev_paths=paths)
        for seed in ([7900]if engineering else [6400,6401,6402]):
            first=4210000000 if engineering else 2611101000+100000*(seed-6400)
            job=f'{method}__seed{seed}';output=STUDY/args.phase/job/f'attempt{args.attempt}'
            cfg=dict(schema='chapter6-external-comparison-job-v1',job_id=job,method=method,seed=seed,
                engineering=engineering,learning_steps=episodes*81,episodes=episodes,eval_every=every,dev_paths=paths,
                device='cuda:0',gpu_uuid='GPU-a3130b13-8288-caaa-0f96-f2916c92d8e3',
                train_weather_seeds=list(range(first,first+episodes)),dev_weather_seeds=list(range(2611095000,2611095000+paths)),
                algorithm_config=native,source_files_sha256=sources,source_release=sha(frozen/'source_manifest.json'),
                output=str(output),worker=str(frozen/'execution/worker.py'),new_TEST=False,selection='fixed endpoint only')
            target=STUDY/'protocol'/f'{tag}_{job}.json';write(target,cfg);configs.append(str(target))
    write(STUDY/'protocol'/f'{tag}_jobs.json',dict(phase=args.phase,configs=configs,source_manifest=str(frozen/'source_manifest.json'),source_release=sha(frozen/'source_manifest.json')))
    print(json.dumps(dict(tag=tag,configs=configs,source_count=len(sources),frozen=str(frozen)),indent=2))
if __name__=='__main__':main()
