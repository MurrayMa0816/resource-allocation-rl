"""Run only the registered sixteen fresh jobs, report, then stop for review."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from . import common as c,register as r,dispatch as d


def prepare(root,python,readiness):
    root=Path(root).resolve();path=root/'registration.json'
    if path.exists():
        value=c.validate_registration(path)
        c.require(value['readiness']==str(Path(readiness).resolve()) and value['jobs'][0]['argv'][0]==str(Path(python).expanduser().absolute()),
            'Recovery interpreter/readiness changed')
        master=c.read_json(root/'master_manifest.json')
        c.require(master['registration_sha256']==c.sha(path) and sorted(master['exact_job_ids'])==sorted(j['id'] for j in value['jobs']),
            'Registered master identity changed')
        return value
    return r.build(root,python,readiness)


def verify_report(root,registration):
    path=Path(registration['report_receipt']);value=c.read_json(path)
    c.require(value.get('status') in ('pass','needs_review') and value.get('completed_training_jobs')==16
        and sorted(value.get('training_job_ids',[]))==sorted(j['id'] for j in registration['jobs'])
        and value.get('registration_sha256')==c.sha(Path(root)/'registration.json'),'Report scope/completion binding changed')
    c.require(bool(value.get('artifacts')),'Report has no bound artifacts')
    for name,digest in value['artifacts'].items():
        p=Path(name).resolve();c.require(p.is_relative_to(path.parent.resolve()) and c.sha(p)==digest,'Report artifact changed or escaped reports')
    return value


def run(root,*,python,readiness,prepare_only=False,poll_seconds=2.):
    root=Path(root).resolve();registration=prepare(root,python,readiness)
    jobs=[dict(j,registration_path=str(root/'registration.json'),registration_sha256=c.sha(root/'registration.json')) for j in registration['jobs']]
    argv=[str(Path(python).expanduser().absolute()),'-m','chapter5_full_spend_20261104.pipeline','--root',str(root),'--python',str(Path(python).expanduser().absolute()),'--readiness',str(Path(readiness).resolve())]
    c.write_json(root/'recovery_launch.json',dict(argv=argv,command=shlex.join(argv),registration_sha256=c.sha(root/'registration.json'),
        scope='exact registered16 only; no retry, extra seed, external baseline expansion or automatic thesis publishing'))
    if prepare_only:
        external=d.external_reservations(registration['resources']);ok,storage=d.storage_ok(jobs,{},registration['resources'],external)
        value=dict(status='ready' if ok else 'storage_admission_held',registered_jobs=16,external=external,storage=storage,new_workers_started=0)
        c.write_json(root/'capacity_preflight.json',value);return value
    def finalize(done,emit):
        c.require(set(done)=={j['id'] for j in jobs},'All sixteen registered completions required before report')
        path=Path(registration['report_receipt'])
        if not path.exists():
            emit('postprocessing_full_spend')
            with (root/'postprocess.log').open('ab') as handle:
                result=subprocess.run(registration['postprocess_argv'],cwd=str(c.WORKSPACE),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
            if result.returncode not in (0,3) or not path.exists():return emit('report_failed_requires_review',controller_stopped=True)
        report=verify_report(root,registration)
        return emit('awaiting_user_review',controller_stopped=True,report_receipt=str(path),report_status=report['status'],
            automatic_more_seeds=False,automatic_retry=False,automatic_thesis_update=False)
    return d.run(root,registration,jobs,finalize,poll_seconds=poll_seconds)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--python',required=True)
    parser.add_argument('--readiness',required=True);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--poll-seconds',type=float,default=2.)
    args=parser.parse_args()
    try:value=run(args.root,python=args.python,readiness=args.readiness,prepare_only=args.prepare_only,poll_seconds=args.poll_seconds)
    except Exception as error:
        value=dict(phase='controller_error_requires_review',controller_stopped=True,error=type(error).__name__+': '+str(error),automatic_retry=False)
        c.write_json(Path(args.root)/'status.json',value);c.write_json(Path(args.root)/'controller_error.json',value);raise
    print(json.dumps(value,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
