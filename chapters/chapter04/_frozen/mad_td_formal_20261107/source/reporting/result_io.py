"""Read-only normalization and independently checkable C4 result arithmetic."""
from __future__ import annotations
from collections import defaultdict
import csv,gzip,hashlib,json,math,statistics
from decimal import Decimal
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
FROZEN=Path('/root/resource_allocation_run_storage_20261101/chapter4_recent_baselines_formal_20261102_source')
MAIN=['f1','props','mad_td','redo','mrq','bro']
ABLATIONS=['no_history','no_offpolicy_correction','fixed_actor_lr']
SEEDS=list(range(5));DOMAINS=['radio','v5'];BUDGET=100608;INTERVAL=768;CASES=32;FINAL_CASES=256

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb')as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def load_plan():
    """Normalize190 explicit records without changing either source registry."""
    raw=read(ROOT/'protocol/formal_registry.json')
    previous=read(ROOT.parent/'double_random_formal_20261107/plan.json')
    byid={x['id']:x for x in previous['jobs']}
    jobs=[]
    for current in raw['jobs']:
        source=byid.get(current.get('source_job_id',current['id']),{})
        job={**source,**current}
        job['scientific_origin']=current.get('origin',current.get('historical_origin',source.get('scientific_origin')))
        if job['method']=='mad_td':
            job.update(family='mad_td',steps=BUDGET if job['kind']=='training'else 0,approval=str(ROOT/'preflight/approval_madtd.json'))
        jobs.append(job)
    plan={**previous,**raw}
    plan.update(jobs=jobs,scope=str(ROOT),new_training_jobs=raw['new_MADTD_training'],reused_external_trainings=raw['reused_complete_training'],
        unique_training_runs=raw['training_curve_count'],reused_A_models=raw['A_reference_models'],independent_endpoint_evaluations=raw['TEST_count'],
        new_endpoint_evaluations=raw['new_TEST'],reused_endpoint_evaluations=raw['reused_complete_TEST'],
        training_seeds=raw['seeds'],training_master=raw['train_master'],dev_master=raw['DEV_master'],final_master=raw['TEST_master'],
        eval_every=raw['DEV_interval'],dev_cases=raw['DEV_cases'],dev_points=raw['DEV_points'],final_cases=raw['TEST_cases'],
        main_methods=MAIN,ablations=ABLATIONS,postprocess_normalization_only=True,
        registry_path=str(ROOT/'protocol/formal_registry.json'),registry_sha256=sha(ROOT/'protocol/formal_registry.json'))
    return plan
def write(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.writing')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def csv_rows(path):
    p=Path(path);op=gzip.open if p.suffix=='.gz'else open
    with op(p,'rt',newline='')as f:return list(csv.DictReader(f))
def json_rows(path):
    p=Path(path);op=gzip.open if p.suffix=='.gz'else open
    with op(p,'rt')as f:return [json.loads(line)for line in f if line.strip()]
def num(v):
    v=float(v);assert math.isfinite(v),('nonfinite number',v);return v
def integer(v):
    # Scenario seeds are genuine 64-bit integers. A float roundtrip silently
    # merges adjacent seeds above2**53, invalidating common-panel checks.
    assert not isinstance(v,bool),'Boolean is not an integer identifier'
    if isinstance(v,int):return v
    if isinstance(v,str):
        d=Decimal(v);assert d.is_finite()and d==d.to_integral_value();return int(d)
    if isinstance(v,float):
        assert math.isfinite(v)and abs(v)<=2**53 and v==int(v),'Large integer fields cannot arrive as imprecise floats'
        return int(v)
    raise TypeError('Expected exact integer or exact decimal integer string')
def close(a,b,atol=2e-11,rtol=2e-10):
    return math.isclose(num(a),num(b),abs_tol=atol,rel_tol=rtol)
def true(v):return v is True or v in ('True','true',1,'1')
def all_finite(value):
    if isinstance(value,dict):
        for v in value.values():all_finite(v)
    elif isinstance(value,list):
        for v in value:all_finite(v)
    elif isinstance(value,float):assert math.isfinite(value),value
def moments(values):
    v=[num(x)for x in values];assert v
    return dict(mean=statistics.fmean(v),sample_sd=statistics.stdev(v)if len(v)>1 else 0.,n=len(v),values=v)
def attainment(curve,q,budget=BUDGET):
    previous=False
    for row in sorted(curve,key=lambda r:r['training_step']):
        good=row['mean_return']>=q
        if previous and good:return dict(reached=True,confirmed_step=row['training_step'],restricted_step=row['training_step'],censored=False)
        previous=good
    return dict(reached=False,confirmed_step=None,restricted_step=budget,censored=True)
def auc(curve,budget=BUDGET):
    rows=sorted(curve,key=lambda r:r['training_step']);assert rows[0]['training_step']==0 and rows[-1]['training_step']==budget
    return math.fsum((b['training_step']-a['training_step'])*(a['mean_return']+b['mean_return'])/2 for a,b in zip(rows,rows[1:]))/budget
def receipt(out):
    p=Path(out)/'completed.json'
    if not p.exists():return None
    d=read(p)
    assert d.get('success')is True or d.get('status')=='PASS',(str(p),'not successful')
    all_finite(d);return d
def endpoint_path(job,d):
    name=d.get('checkpoint')or d.get('model')or d.get('model_path')or d.get('model_loaded_from')or d.get('source_model')
    assert name,(job['id'],'no model path')
    p=Path(name);return p if p.is_absolute()else Path(job['output'])/p
def curve_rows(out):
    out=Path(out)
    p=out/'evaluation_curve.csv.gz'
    if p.exists():
        return [dict(training_step=integer(x['training_step']),phase=x['phase'],mean_return=num(x['native_mean']),
            sample_sd=num(x.get('native_std',0)),cases=integer(x['paths']))for x in csv_rows(p)]
    p=out/'evaluations.json'
    assert p.exists(),(str(out),'missing curve')
    rows=read(p);result=[]
    for x in rows:
        vals=[num(v)for v in x['returns']]
        result.append(dict(training_step=integer(x.get('train_steps',x.get('train_env_steps',x.get('training_step')))),phase=x.get('phase','dev'),
            mean_return=statistics.fmean(vals),sample_sd=statistics.stdev(vals)if len(vals)>1 else 0.,cases=len(vals),
            reported_mean_return=num(x.get('native_mean',x.get('mean_return',num(x['mean_score'])/100)))))
    return result
def case_rows(out):
    out=Path(out);p=out/'evaluation_paths.csv.gz'
    rows=csv_rows(p)if p.exists()else json_rows(out/'evaluation_paths.jsonl.gz')
    fallback_phase='final'
    if any('phase'not in x for x in rows)and(out/'protocol.json').exists():fallback_phase=read(out/'protocol.json').get('phase','final')
    result=[]
    for x in rows:
        step=x.get('training_step',x.get('train_steps',BUDGET));phase=x.get('phase',fallback_phase)
        result.append(dict(training_step=integer(step),phase=phase,case_index=integer(x.get('path_index',x.get('case'))),
            path_seed=integer(x.get('path_seed',x.get('task_seed'))),native_return=num(x['native_return']),
            resource_fraction=num(x.get('resource_fraction',1.)),steps=integer(x['steps']),
            replay_error=num(x['replay_error'])if x.get('replay_error')not in(None,'')else None,raw=x))
    return result
def resource_rows(out):
    out=Path(out)
    for n in ['resource_episodes.csv.gz','resource_audit.csv.gz']:
        p=out/n
        if p.exists():return csv_rows(p),n
    p=out/'resource_ledger.jsonl.gz'
    if p.exists():return json_rows(p),p.name
    return [],None
def train_rows(out):
    out=Path(out)
    for n in ['training_episodes.csv.gz','train_episodes.jsonl.gz','training.jsonl']:
        p=out/n
        if p.exists():
            rr=csv_rows(p)if n.endswith('.csv.gz')else json_rows(p)
            if n=='training.jsonl':
                rr=[x for x in rr if x.get('record_type')=='episode'or'episode_return'in x or('episode'in x and'native_return'in x)]
                return [dict(training_step=integer(x['train_env_steps']),native_return=num(x.get('episode_return',x.get('native_return'))),raw=x)for x in rr],n
            return [dict(training_step=integer(x.get('training_step',x.get('train_steps'))),native_return=num(x.get('native_return',x.get('return_raw',0))),raw=x)for x in rr],n
    return [],None
def update_rows(out):
    out=Path(out)
    for n in ['updates.jsonl.gz','training.jsonl']:
        p=out/n
        if p.exists():return json_rows(p)
    return []
def grouped_cases(rows):
    groups=defaultdict(list)
    for x in rows:groups[(x['phase'],x['training_step'])].append(x)
    return groups
def verify_means(curves,cases,grid,cases_each,phase):
    assert [r['training_step']for r in curves]==grid
    assert all(r['phase']==phase and r['cases']==cases_each for r in curves)
    groups=grouped_cases(cases);assert set(groups)=={(phase,n)for n in grid}
    for r in curves:
        group=groups[(phase,r['training_step'])]
        assert len(group)==cases_each and sorted(x['case_index']for x in group)==list(range(cases_each))
        vals=[x['native_return']for x in group]
        assert close(statistics.fmean(vals),r['mean_return']),(r,'case mean mismatch')
        assert close(statistics.stdev(vals)if len(vals)>1 else 0.,r['sample_sd']),(r,'sample SD mismatch')
        if 'reported_mean_return'in r:assert close(r['reported_mean_return'],r['mean_return'])
def source_map_check(mapping,origin_scope=None):
    checked=0
    for key,value in mapping.items():
        if not(isinstance(value,str)and len(value)==64 and all(c in'0123456789abcdef'for c in value)):continue
        p=Path(key)
        if not p.is_absolute():
            origin=Path(origin_scope)if origin_scope else ROOT
            if key=='formal/f1_worker.py':p=origin/'execution/f1_worker.py'
            elif key=='formal/f1_extension.py':p=origin/'source/f1_extension.py'
            elif key.startswith('chapter23_final/'):
                p=origin/'source/environment/release'/key[len('chapter23_final/'):]
            else:p=FROZEN/p
        assert p.exists()and sha(p)==value,('source changed',str(p));checked+=1
    return checked
