"""Seal reviewed postprocess code only; never train, evaluate or publish."""
from pathlib import Path
import argparse,ast,datetime,hashlib,json,os
R=Path(__file__).resolve().parents[1]
RELATIVE=[
 'execution/publish_thesis.py','source/reporting/thesis_content.py',
 'source/reporting/contract_patches.py','source/reporting/citations.py',
 'source/reporting/figure_plot.py','source/reporting/PUBLISHER_README.rst',
 'preflight/publisher_tests/run_synthetic_checks.py','preflight/publisher_tests/run_guard_checks.py',
 'execution/independent_acceptance.py','execution/aggregate_results.py',
 'execution/postprocess.py','source/reporting/result_io.py']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.new');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seal',action='store_true');a=p.parse_args()
 paths=[R/x for x in RELATIVE]+[Path('/root/gamewar/latex_thesis_resource_allocation/scripts/build.sh')]
 for path in paths:
  assert path.is_file(),str(path)
  if path.suffix=='.py':ast.parse(path.read_text())
 plan=read(R/'plan.json');assert sha(R/'plan.json')==read(R/'plan_seal.json')['sha256']
 assert plan['initialization']=='both_random'and(plan['new_training_jobs'],plan['reused_external_trainings'],plan['unique_training_runs'])==(50,40,90)
 review=R/'preflight/publication_pipeline_independent_review.json';assert review.exists()
 rev=read(review);assert str(rev['status']).startswith('PASS'),rev
 for path,digest in rev.get('files_sha256',rev.get('source_sha256',{})).items():
  q=Path(path);q=q if q.is_absolute()else R/q
  assert sha(q)==digest,('Reviewed source changed',str(q))
 tests=[R/'preflight/publisher_tests/selftest.json',R/'preflight/publisher_tests/SYNTHETIC_ONLY/NOT_SCIENTIFIC_RESULTS.json',R/'preflight/publisher_tests/SYNTHETIC_GUARD_ONLY/NOT_SCIENTIFIC_RESULTS.json']
 for path in tests:
  d=read(path);assert 'PASS'in d['status']and d['new_environment_steps']==0 and d['live_thesis_modified']is False
 approval_path=R/'preflight/publisher_approval.json';approval=read(approval_path)
 assert approval['status']=='PASS_ENGINEERING_POSTPROCESS_READY_SCIENTIFIC_RESULTS_PENDING'and approval['thesis_or_live_PDF_updated']is False
 for path,digest in approval['source_sha256'].items():assert sha(path)==digest,('Publisher approved source changed',path)
 hashes={str(x):sha(x)for x in paths}
 record=dict(schema='chapter4-double-random-postprocess-capsule-v1',status='SEALED_CODE_READY_SCIENTIFIC_RESULTS_PENDING',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(R/'plan.json'),files_sha256=hashes,independent_review=str(review),independent_review_sha256=sha(review),publisher_approval_sha256=sha(approval_path),engineering_test_receipt_sha256={str(p):sha(p)for p in tests},sealer_sha256=sha(__file__),initialization='both_random',new_trainings=50,retained_external_trainings=40,inherited_new_C3_A_endpoints=10,scientific_result_entries=190,publication_trigger='All90 registered mixed-origin training results and100 endpoint evaluations complete; independent semantic acceptance PASS then aggregate and five actual whole-thesis builds.',scientific_results_complete=False,publication_performed=False,automatic_extra_training=False,human_visual_review_not_claimed=True)
 target=R/'protocol/postprocess_capsule.json'
 if not a.seal:
  print('REVIEWED_CODE_READY_NOT_SEALED',len(hashes));return
 if target.exists():
  previous=read(target);assert previous['plan_sha256']==record['plan_sha256']and previous['files_sha256']==hashes,'Never overwrite a sealed code capsule';print('EXISTING_IDENTICAL_CAPSULE',target);return
 write(target,record);write(R/'protocol/postprocess_capsule_seal.json',dict(schema='chapter4-postprocess-capsule-sha256-v1',path=str(target),sha256=sha(target),plan_sha256=record['plan_sha256'],new_environment_steps=0,publication_performed=False));print('SEALED_ENGINEERING_CODE_ONLY',target)
if __name__=='__main__':main()
