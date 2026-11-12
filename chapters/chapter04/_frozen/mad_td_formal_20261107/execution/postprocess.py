"""Automatic complete-cohort acceptance, aggregation and authorized C4 publication."""
import datetime,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1]

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def verified_publication(seal):
    publication=R/'publication/FINAL_THESIS_PUBLICATION.json'
    assert publication.exists(),'Registered publication receipt is required'
    pub=read(publication)
    assert pub['status']=='PUBLISHED_AUTOMATED_CHECKS_PASS'and pub['published']is True
    assert pub['plan_sha256']==seal
    assert pub['acceptance_sha256']==sha(R/'independent/acceptance.json')
    assert pub['summary_sha256']==sha(R/'reports/summary.json')
    assert int(pub['whole_thesis_build_passes'])==5
    pdf=Path(pub['pdf_path'])
    assert pdf.is_relative_to(R/'publication')and pdf.is_file()
    assert pdf.read_bytes()[:5]==b'%PDF-'and sha(pdf)==pub['pdf_sha256']
    return publication
def main():
    lock=open(R/'runtime/postprocess.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    done=R/'completion_receipt.json'
    status={'schema':'chapter4-postprocess-live-v1','phase':'initial_acceptance','started_unix':time.time(),'new_training_steps':0,'new_environment_evaluation_steps':0}
    try:
        plan=read(R/'protocol/formal_registry.json');seal=read(R/'protocol/science_code_seal.json')['registry_sha256'];assert sha(R/'protocol/formal_registry.json')==seal
        ready=read(R/'postprocess_ready.json');assert ready['plan_sha256']==seal
        assert ready['training']==90 and ready['evaluation']==100
        capsule=read(R/'protocol/postprocess_capsule.json')
        for path,digest in capsule['files_sha256'].items():assert sha(path)==digest,('postprocess source changed',path)
        if done.exists():
            previous=read(done)
            assert previous['status']=='COMPLETE_SCIENTIFIC_RESULTS_AND_THESIS_PUBLISHED'
            assert previous['plan_sha256']==seal
            assert read(R/'independent/acceptance.json')['status']=='PASS'
            # Existing FINAL takes the publisher's full current190/raw-input
            # gate and verifies its immutable receipt; --prepare never rebuilds.
            code=subprocess.call([sys.executable,str(R/'execution/publish_thesis.py'),'--scope',str(R),'--prepare'])
            assert code==0,'Existing completion did not pass current scientific/publication gates'
            publication=verified_publication(seal)
            assert previous['thesis_publication_sha256']==sha(publication)
            status.update(phase='complete',idempotent_current_gates_verified=True,updated_unix=time.time())
            write(R/'runtime/postprocess_status.json',status)
            return
        for step,argv in [
            ('independent_acceptance',[sys.executable,str(R/'execution/independent_acceptance.py'),'--scope',str(R)]),
            ('aggregate_results',[sys.executable,str(R/'execution/aggregate_results.py'),'--scope',str(R)]),
            ('publish_thesis',[sys.executable,str(R/'execution/publish_thesis.py'),'--scope',str(R),'--publish'])]:
            status.update(phase=step,updated_unix=time.time());write(R/'runtime/postprocess_status.json',status)
            log=R/'runtime'/f'{step}.log'
            with log.open('a')as f:
                code=subprocess.call(argv,stdout=f,stderr=subprocess.STDOUT)
            if code!=0:raise RuntimeError(f'{step} failed with exit{code}; see{log}')
            if step=='independent_acceptance':assert read(R/'independent/acceptance.json')['status']=='PASS'
        publication=verified_publication(seal)
        end={'schema':'chapter4-formal-full-delivery-v1','status':'COMPLETE_SCIENTIFIC_RESULTS_AND_THESIS_PUBLISHED',
             'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':seal,
             'new_trainings':10,'retained_external_trainings':80,'unique_training_runs':90,
             'inherited_A_models':10,'independent_model_evaluations':100,'initialization':'both_random',
             'new_endpoint_evaluations':70,'retained_endpoint_evaluations':30,
             'independent_acceptance':str(R/'independent/acceptance.json'),'summary':str(R/'reports/summary.json'),
             'thesis_publication_receipt':str(publication),'thesis_publication_sha256':sha(publication),
             'manual_visual_review_status':'Read actual publisher receipt; automatic rendering is not human inspection.',
             'postprocess_new_environment_steps':0,'automatic_extra_training':False,'no_extra_seeds_or_budget_extensions':True}
        write(done,end);status.update(phase='complete',updated_unix=time.time());write(R/'runtime/postprocess_status.json',status)
        lease=Path('/hy-tmp/resource_allocation_results/resource_runtime_leases/CHAPTER4_MADTD_FORMAL_20261107.json')
        if lease.exists():
            record=read(lease);assert record['scope']==str(R)
            record.update(status='COMPLETE_KEEP_SCIENTIFIC_ARTIFACTS',runtime_resource_reservation_active=False,
                          completion_receipt=str(done),completed_unix=time.time())
            write(lease,record)
        pointer=R.parent/'LATEST_FORMAL_TRAINING.json'
        if pointer.exists()and read(pointer).get('scope')==str(R):
            record=read(pointer);record.update(status='complete',completion_receipt=str(done));write(pointer,record)
        print(json.dumps(end,ensure_ascii=False),flush=True)
    except BaseException:
        status.update(phase='failed',updated_unix=time.time(),traceback=traceback.format_exc())
        write(R/'runtime/postprocess_status.json',status);raise

if __name__=='__main__':main()
