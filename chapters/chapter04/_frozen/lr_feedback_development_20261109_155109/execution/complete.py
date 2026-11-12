"""Record a completed development study; never promotes scientific paper results."""
from pathlib import Path
import json,hashlib,datetime

ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.complete_tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');q.replace(p)

def run():
    assert not (ROOT/'completion_receipt.json').exists(),'Do not replay completed development publication'
    status=read(ROOT/'runtime/development_status.json')
    assert status['status']=='complete' and status['completed_count']==16 and not status['failed']
    for p,d in read(ROOT/'protocol/development_source_seal.json')['files'].items():assert sha(p)==d
    protected=read(ROOT/'audit/original_protection_manifest.json')
    for x in protected['files']:assert sha(x['path'])==x['sha256'],x['path']
    for name in ['development_review.json','reload_results.json']:
        assert read(ROOT/'audit/independent'/name)['status']=='PASS'
    assert read(ROOT/'audit/root_report_visual_review.json')['status']=='PASS'
    metrics=read(ROOT/'results/summary.json')
    assert metrics['status']=='COMPLETE'
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    paths=[ROOT/'protocol/plan.json',ROOT/'protocol/development_registry.json',ROOT/'protocol/development_source_seal.json',
           ROOT/'audit/independent/development_review.json',ROOT/'audit/independent/reload_results.json',ROOT/'audit/root_report_visual_review.json']
    for p in sorted((ROOT/'results').glob('*')):
        if p.is_file():paths.append(p)
    for p in sorted((ROOT/'report').glob('*')):
        if p.is_file():paths.append(p)
    evidence={str(p.relative_to(ROOT)):sha(p) for p in paths}
    final=dict(schema='chapter04-lr-feedback-development-completion-v1',status='COMPLETE',completed_at=now,
        development_training_jobs=16,engineering_training_jobs=10,development_steps_per_job=49152,
        domains=['radio','v5'],paired_training_seeds=[7100,7101],arms=['original_feedback','fixed_3e4','fixed_1e4','relaxed_feedback'],
        fixed_endpoint_models=26,independent_endpoint_reloads=16,old_models_loaded=0,new_independent_TEST_cases=0,
        automatic_formal_adoption=False,main_paper_modified=False,main_paper_builds=0,
        original_scientific_pointers_modified=False,original_protected_files_SHA_unchanged=len(protected['files']),
        original_models_SHA_unchanged=protected['model_count'],recovery=dict(command=None),
        summary=metrics['summary'],per_seed=metrics['per_seed'],
        scientific_execution_steps=dict(engineering_learning=15360,engineering_DEV=26880,development_learning=786432,development_DEV=931840,independent_DEV_reload=14336,total=1774848),
        evidence=evidence)
    # Derive physical costs once more from actual receipts, not the expected budget.
    eng=[read(Path(x['output'])/'completed.json') for x in read(ROOT/'protocol/engineering_registry.json')['jobs']]
    dev=[read(Path(x['output'])/'completed.json') for x in read(ROOT/'protocol/development_registry.json')['jobs']]
    actual=sum(x['total_physical_environment_steps'] for x in eng+dev)+read(ROOT/'audit/independent/reload_results.json')['total_environment_steps']
    assert actual==final['scientific_execution_steps']['total']
    dump(ROOT/'FINAL_RESULTS.json',final)
    dump(ROOT/'completion_receipt.json',dict(final,final_results=str(ROOT/'FINAL_RESULTS.json'),final_results_sha256=sha(ROOT/'FINAL_RESULTS.json')))
    docs=Path('/root/gamewar/resource_allocation_rl/docs/chapter4_lr_feedback_development_20261109')
    dump(docs/'RESULTS.json',dict(scope=str(ROOT),completion=str(ROOT/'completion_receipt.json'),final_results=str(ROOT/'FINAL_RESULTS.json'),scientific_role='development only; no formal adoption'))
    print(json.dumps(dict(status='COMPLETE',scope=str(ROOT),physical_steps=actual),ensure_ascii=False))

if __name__=='__main__':run()
