"""Read-only all100-model barrier; never imports ML or opens new TEST data."""
from pathlib import Path
import datetime,importlib.util,sys,traceback
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_madtd_formal_release_acceptance',R/'execution/independent_acceptance.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def main():
    io=a.io;target=R/'protocol/endpoint_release_gate.json';plan=io.load_plan();a.assert_plan(plan)
    missing=[j['id']for j in plan['jobs']if j['kind']=='training'and not(Path(j['output'])/'completed.json').is_file()]
    if missing:
        io.write(R/'audit/endpoint_release_review.json',dict(status='PENDING',missing_training_jobs=missing,all_training_completed=False,no_new_TEST_opened_by_auditor=True));print('PENDING',len(missing));return 3
    original=io.read(R/'audit/retained170_inventory.json');seal=io.read(R/'audit/retained170_inventory_seal.json')
    assert io.sha(R/'audit/retained170_inventory.json')==seal['sha256']
    for p,h in original['bound_files_sha256'].items():assert io.sha(p)==h,('retained result changed',p)
    endpoints=original['training_curve_endpoints']+original['A_reference_endpoints'];shared={};models={};approval_cache={};new=[]
    # Fix the same DEV scenario panel as the retained current F1; no TEST returns
    # are loaded or used to select a model, parameter or stopping point here.
    for domain in ['radio','v5']:
        job=next(j for j in plan['jobs']if j['kind']=='training'and j['method']=='f1'and j['domain']==domain and j['seed']==0)
        for x in io.case_rows(job['output']):
            k=(domain,x['phase'],x['case_index'])
            if k in shared:assert shared[k]==x['path_seed']
            else:shared[k]=x['path_seed']
    for job in plan['jobs']:
        if job['method']!='mad_td'or job['kind']!='training':continue
        approved,_=a.verify_approval(job,approval_cache);report=a.verify_job(job,approved,shared,models);new.append(report)
        d=io.receipt(job['output']);assert d['formal_training_performed']is True and d['full_formal_training_budget_executed']is True
        endpoints.append(dict(method='mad_td',domain=job['domain'],seed=job['seed'],model=report['model_path'],model_sha256=report['model_sha256'],completion=str(Path(job['output'])/'completed.json'),completion_sha256=report['completed_sha256'],origin='new_MADTD_formal_from_zero'))
    assert len(new)==10 and len(endpoints)==100 and len({(e['method'],e['domain'],e['seed'])for e in endpoints})==100
    for e in endpoints:
        assert io.sha(e['model'])==e['model_sha256']and io.sha(e['completion'])==e['completion_sha256']
    if target.exists():
        previous=io.read(target)
        assert previous['status']=='PASS'and previous['fixed_endpoints']==endpoints
        assert previous['registry_sha256']==io.sha(R/'protocol/formal_registry.json')and previous['science_code_seal_sha256']==io.sha(R/'protocol/science_code_seal.json')
        print('PASS already released',io.sha(target));return 0
    # No newly registered TEST may start before this one-time barrier. Historical
    #30 panels are retained records and are explicitly outside this new guard.
    for j in plan['jobs']:
        if j['kind']=='evaluation'and j['execution_required']:
            out=Path(j['output']);assert not out.exists()or not any(out.iterdir()),('new TEST output predates all-model release',j['id'])
    result=dict(schema='chapter4-madtd-formal-all-endpoints-release-v1',status='PASS',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),registry_sha256=io.sha(R/'protocol/formal_registry.json'),science_code_seal_sha256=io.sha(R/'protocol/science_code_seal.json'),all_training_completed=True,endpoint_count=100,training_curve_count=90,test_master=2026110694199,test_cases=256,test_phase='final',no_PGR=True,fixed_endpoints=endpoints,new_MADTD_training_reports=new,retained170_inventory_sha256=io.sha(R/'audit/retained170_inventory.json'),independent_auditor_source_sha256=io.sha(__file__),independent_acceptance_source_sha256=io.sha(R/'execution/independent_acceptance.py'),new_TEST_jobs_released=70,historical_TEST_panels_retained=30,no_TEST_results_used_for_release=True,new_environment_steps=0,new_optimizer_steps=0,model_deserialization=False)
    io.write(target,result);io.write(R/'audit/endpoint_release_review.json',dict(status='PASS_ALL100_ENDPOINTS_FROZEN_BEFORE_NEW70_TEST',endpoint_release_gate_sha256=io.sha(target),new_MADTD_models=10,retained_training_models=80,reused_A_reference_models=10,new_TEST_opened_by_auditor=False));print('PASS all100 endpoints',io.sha(target));return 0
if __name__=='__main__':
    try:code=main()
    except Exception as e:
        a.io.write(R/'audit/endpoint_release_review.json',dict(status='FAIL',error=repr(e),traceback=traceback.format_exc(),new_environment_steps=0,new_TEST_opened_by_auditor=False));raise
    sys.exit(code)
