#!/usr/bin/env python3
"""Independent local reports for the fresh PFO 10-training/10-final Chapter 3 scope.

Reads only closed, SHA-verified result mirrors. No model selection, training,
remote access, source mutation, interpolation, or thesis publication occurs.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

DEFAULT_SCOPE=Path('/hy-tmp/resource_allocation_results/chapter3/pfo_formal_100k_5seed_20261106')
METHODS=('pfo',)
ARMS=('A_FULL','A_PACE','A_BUDGET_ONLY','A_NO_BUDGET','A_CONSTANT_REFERENCE')
LABELS={'pfo':'PFO','A_FULL':'A','c_ppo':'C-PPO','gradient_ppo':'Gradient PPO','reflective_ppo':'Reflective PPO',
        'ppo_plus':'PPO+','A_PACE':'Rule total','A_BUDGET_ONLY':'Without task summary',
        'A_NO_BUDGET':'Without direct budget','A_CONSTANT_REFERENCE':'Constant reference'}
DOMAINS=('radio','v5')
STEPS=100608
FINAL_MASTER=202611058199

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024**2),b''):h.update(part)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def number(value):
    value=float(value);require(math.isfinite(value),'Nonfinite scientific value');return value

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)

def rows(path):
    with gzip.open(path,'rt',newline='') if str(path).endswith('.gz') else Path(path).open(newline='') as f:
        return list(csv.DictReader(f))

def write_rows(path,values):
    require(bool(values),'Cannot write an empty scientific table')
    fields=list(dict.fromkeys(k for row in values for k in row));path=Path(path)
    temp=path.with_name(path.name+'.tmp')
    with gzip.open(temp,'wt',newline='') if path.name.endswith('.gz') else temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(values)
    temp.replace(path)

def artifact(root,relative):
    path=(root/relative).resolve(strict=True)
    require(path.is_relative_to(root.resolve()) and path.is_file(),'Unsafe/missing artifact '+relative)
    return path

def closed_bundle(scope,job,plan_sha):
    root=scope/'remote_results'/job['id'];t=read(root/'transfer_receipt.json')
    require(t.get('status')=='PASS' and t.get('job_id')==job['id'],'Transfer identity is not PASS')
    require(t.get('remote_job_phase')=='complete' and t.get('remote_process_exited') is True,
            'Transfer was not taken after the worker exited successfully')
    require(t.get('source_plan_sha256')==plan_sha,'Transfer plan SHA differs')
    files=t.get('files',[])
    if isinstance(files,dict):
        files=[dict(relative_path=n,**(v if isinstance(v,dict) else {'sha256':v})) for n,v in files.items()]
    names=set()
    for entry in files:
        name=entry['relative_path'];require(name not in names,'Repeated transfer file');names.add(name)
        p=artifact(root,name);require(sha(p)==entry['sha256'],'Transfer SHA differs '+name)
        if 'bytes' in entry:require(p.stat().st_size==entry['bytes'],'Transfer byte count differs')
        if name.endswith('.gz'):
            with gzip.open(p,'rb') as f:
                for _ in iter(lambda:f.read(1024**2),b''):pass
    require({'completed.json','protocol.json'}<=names,'Missing completed/protocol transfer binding')
    c,p=read(root/'completed.json'),read(root/'protocol.json');digest=sha(root/'completed.json')
    require(c.get('success') is True,'Native unsuccessful result')
    require(t.get('completed_sha256')==t.get('remote_dispatch_receipt_sha256')==digest,'Completion transfer binding differs')
    for key in ('artifacts_sha256','artifact_sha256'):
        for name,h in c.get(key,{}).items():
            require(name in names and sha(artifact(root,name))==h,'Native artifact differs '+name)
    for name,entry in c.get('model_files',{}).items():
        require('model/'+name in names and sha(artifact(root,'model/'+name))==entry['sha256'],'Native model differs')
    for name,h in c.get('model_files_sha256',{}).items():
        require('model/'+name in names and sha(artifact(root,'model/'+name))==h,'A model differs')
    return dict(root=root,completed=c,protocol=p,transfer=t,names=names,completion_sha=digest,
                fingerprint=sha(root/'transfer_receipt.json'),job=job)

def validate_training(bundle,plan):
    c,p,j,root=(bundle[k] for k in ('completed','protocol','job','root'));a=p['arguments']
    require(c.get('source_unchanged') is True,'Changed scientific source')
    require(c.get('training_steps')==a.get('steps')==j['steps']==STEPS==c.get('accounting',{}).get('environment_steps'),
            'Native learning budget differs from 100608')
    require(a.get('path_master')==j['master'],'Training namespace differs')
    require(all(c.get(k)==j[k] for k in ('method','domain','seed')),'Training identity differs')
    if j['kind']=='train_A':
        require(c.get('schema')=='chapter3-A-native-formal-lean-v3','Unexpected A adapter schema')
        require(p['source_identity']['package']==c['source_identity']['package']==plan['source_sha256'],'A source closure differs')
        require(p['source_identity']['worker_sha256']==plan['execution_sha256']['a_native_formal_worker.py'],'A worker differs')
        require(a.get('arm')==j['arm'] and a.get('domain')==j['domain'] and a.get('seed')==j['seed'],'A arm/seed differs')
        require(a.get('final_master')==plan['final_master'] and a.get('final_paths')==plan['final_paths'],'A final registration differs')
        require(c.get('actual_device')=='cpu' and c.get('endpoint_reload_pass') is True,'A CPU/load gate differs')
        require(c.get('full_spend_required') is True and c.get('native_replay_error')==0.
                and number(c.get('maximum_resource_fraction_error',1.))<=1e-9,'A full-spend/replay gate failed')
        require(c.get('gzip_complete_before_success') is True and c.get('artifacts_sha256'),'A scientific outputs are not sealed')
        require(c['artifacts_sha256'].get('protocol.json')==sha(root/'protocol.json'),'A protocol digest differs')
        require(c.get('model_sha256')==sha(root/'model/state.pt'),'A endpoint model differs')
        require(c['accounting'].get('actor_update_calls',0)>0 and c['accounting'].get('ppo_optimizer_steps',0)>0,'A had no actual updates')
        require(c['accounting'].get('complete_episodes')==STEPS//(32 if j['domain']=='radio' else 24),'A episode accounting differs')
    else:
        require(c.get('engineering_gate') is False and p.get('source_sha256')==plan['source_sha256'],'External scope/source differs')
        require(c.get('protocol_sha256')==sha(root/'protocol.json'),'External protocol digest differs')
        require(a.get('device')=='cuda' and a.get('storage_profile')=='lean','External CUDA lean contract differs')
        require(all(a.get(k)==j[k] for k in ('method','domain','seed')),'External arguments differ')
        override=bundle['transfer'].get('remote_dispatch_state',{}).get('effective_overrides')
        require(override==j.get('config',{}),'Unregistered native configuration override')
        require(all(p['effective_config'].get(k)==v for k,v in override.items()),'Native effective configuration differs')
        gate=c.get('endpoint_reload_gate',{});v=c.get('evaluation_validation',{})
        require(gate.get('status')=='PASS' and gate.get('physical_actions_bitwise_equal') is True
                and gate.get('all_native_replay_errors_zero') is True and gate.get('same_path_seeds') is True,
                'External native save/load/replay gate failed')
        count=v.get('evaluated_paths',0)
        require(count>0 and v.get('native_replayed_paths')==v.get('full_spend_checked_paths')==count,'External DEV path gate failed')
        require(c['accounting'].get('policy_updates' if j['method']=='c_ppo' else 'actor_updates',0)>0,'External had no actor updates')
    horizon=32 if j['domain']=='radio' else 24
    train=rows(root/'training_episodes.csv.gz');steps=rows(root/'training_steps.csv.gz')
    require(len(train)==STEPS//horizon and [int(r['training_step']) for r in train]==list(range(horizon,STEPS+1,horizon)),
            'Actual complete training episodes do not cover the registered learning budget')
    require(len(steps)==STEPS and all(int(r['training_step'])==i for i,r in enumerate(steps,1)),'Actual step CSV has gaps or duplicates')
    require(all(abs(number(r['resource_fraction'])-1.)<=1e-9 for r in train),'Training episode did not allocate all resource')
    filename='evaluation_curve.csv.gz' if j['kind']=='train_A' else 'development_curve.csv.gz'
    dev=[r for r in rows(root/filename) if r.get('phase','dev')=='dev' and r.get('evaluation_kind') not in ('endpoint_reload','reload')]
    unique={int(r['training_step']):r for r in dev};require(len(unique)==len(dev),'Repeated ordinary DEV point')
    require(min(unique)==0 and max(unique)==STEPS and len(unique)==198,'DEV grid lacks initial/final or registered evaluations')
    require(all(int(r['paths'])==8 for r in dev),'DEV path count differs')
    bundle.update(training=train,development=sorted(dev,key=lambda r:int(r['training_step'])))
    return bundle

def validate_final(bundle,plan,parent=None):
    c,p,j,root=(bundle[k] for k in ('completed','protocol','job','root'))
    if j['kind']=='train_A':
        summary=c['final_evaluation'];raw=[r for r in rows(root/'evaluation_paths.csv.gz') if r['phase']=='final']
    else:
        require(parent is not None,'Parent endpoint has not independently passed')
        require(c.get('source_unchanged') is True and c.get('source_sha256')==p.get('source_sha256')==plan['source_sha256'],
                'Final evaluation source differs')
        require(c.get('protocol_sha256')==sha(root/'protocol.json'),'Final protocol digest differs')
        require(c.get('method')==j['method'] and c.get('training_completion_sha256')==parent['completion_sha'],
                'Final result parent/method binding differs')
        a=p['arguments'];reg=p['path_registration']
        require(a.get('phase')==reg.get('phase')=='final' and a.get('path_master')==reg.get('master')==FINAL_MASTER
                and a.get('paths')==reg.get('paths')==128 and reg.get('independent_from_training_namespace') is True,
                'Final test namespace/paths differ')
        require(c.get('learning_environment_steps')==0 and c.get('all_native_replays_exact') is True
                and c.get('fixed_policy_geometry_matches_training') is True,'Final contains learning or replay/geometry mismatch')
        val=c['evaluation_validation']
        require(val.get('evaluated_paths')==val.get('native_replayed_paths')==val.get('full_spend_checked_paths')==128,
                'Not every final test path passed the physical gates')
        require(c.get('condition_file_sha256') is None,'Unexpected changed final environment')
        summary=c['summary'];raw=[r for r in rows(root/'evaluation_paths.csv.gz') if r.get('evaluation_kind')=='final']
    require(summary.get('paths')==len(raw)==128,'Exactly 128 final paths required')
    index='path_index' if 'path_index' in raw[0] else 'path';raw.sort(key=lambda r:int(r[index]))
    require([int(r[index]) for r in raw]==list(range(128)) and len({r['path_seed'] for r in raw})==128,'Final path indices/seeds differ')
    require(all(number(r['replay_error'])==0 and abs(number(r['resource_fraction'])-1.)<=1e-9 for r in raw),
            'A raw final path failed replay/full spend')
    values=[number(r['native_return']) for r in raw]
    require(abs(statistics.mean(values)-number(summary['native_mean']))<=1e-12,'Final mean disagrees with raw paths')
    bundle.update(final_raw=raw,final_mean=statistics.mean(values),parent=parent or bundle)
    return bundle

def curve_tables(evaluations):
    train,dev=[],[]
    for e in evaluations:
        j=e['parent']['job'];base=dict(method=j['method'],domain=j['domain'],seed=j['seed'])
        train.extend(dict(**r,**base,score_x100=100*number(r['native_return'])) for r in e['parent']['training'])
        dev.extend(dict(**r,**base,score_x100=100*number(r['native_mean'])) for r in e['parent']['development'])
    grouped=[]
    for kind,records in [('training',train),('development',dev)]:
        groups=collections.defaultdict(list)
        for r in records:groups[(r['method'],r['domain'],int(r['training_step']))].append(r)
        for (method,domain,step),rs in sorted(groups.items()):
            require(len(rs)==5 and {r['seed'] for r in rs}==set(range(5)),'Curves do not share the actual five-seed step grid')
            values=[r['score_x100'] for r in rs]
            grouped.append(dict(kind=kind,method=method,domain=domain,training_step=step,
                score_mean=statistics.mean(values),score_seed_sample_sd=statistics.stdev(values),n_training_seeds=5))
    return train,dev,grouped

def report_group(scope,name,evaluations,methods,plan_sha):
    folder=scope/'report'/name;folder.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(json.dumps(dict(plan_sha256=plan_sha,report_module_sha256=sha(__file__),
        inputs={e['job']['id']:dict(completion=e['completion_sha'],transfer=e.get('fingerprint'),
            parent_completion=e['parent'].get('completion_sha'),parent_transfer=e['parent'].get('fingerprint'))
            for e in evaluations}),sort_keys=True).encode()).hexdigest()
    old=read(folder/'completion_receipt.json') if (folder/'completion_receipt.json').exists() else {}
    if old.get('fingerprint')==fingerprint and old.get('status')=='PASS' and old.get('artifacts_sha256') and all((folder/f).is_file() and sha(folder/f)==h for f,h in old.get('artifacts_sha256',{}).items()):
        return str(folder/'completion_receipt.json')
    seed_rows=[];raw=[]
    for e in evaluations:
        j=e['parent']['job'];base=dict(method=j['method'],domain=j['domain'],seed=j['seed'])
        seed_rows.append(dict(**base,score_mean=100*e['final_mean'],native_mean=e['final_mean'],training_steps=STEPS,
            paths=128,final_master=FINAL_MASTER,completion_sha256=e['completion_sha']))
        raw.extend(dict(**r,**base) for r in e['final_raw'])
    means=[]
    for domain in DOMAINS:
        for method in methods:
            selected=[r for r in seed_rows if r['domain']==domain and r['method']==method]
            require(len(selected)==5 and {r['seed'] for r in selected}==set(range(5)),'Final cohort lacks five seeds')
            values=[r['score_mean'] for r in selected]
            means.append(dict(method=method,domain=domain,n_training_seeds=5,paths_per_seed=128,
                score_mean=statistics.mean(values),score_seed_sample_sd=statistics.stdev(values)))
    train,dev,aggregate=curve_tables(evaluations);tails=[]
    for domain in DOMAINS:
        for method in methods:
            for seed in range(5):
                rs=sorted([r for r in dev if (r['domain'],r['method'],r['seed'])==(domain,method,seed)],key=lambda r:int(r['training_step']))
                last=rs[-10:];before=rs[-20:-10];ys=[r['score_x100'] for r in last]
                xs=[int(r['training_step']) for r in last];xm,ym=statistics.mean(xs),statistics.mean(ys)
                slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)*10000
                tails.append(dict(method=method,domain=domain,seed=seed,initial_dev=rs[0]['score_x100'],endpoint_dev=rs[-1]['score_x100'],
                    tail10_mean=ym,previous10_mean=statistics.mean(r['score_x100'] for r in before),
                    tail_mean_change=ym-statistics.mean(r['score_x100'] for r in before),
                    tail_slope_points_per_10000_steps=slope,tail_range=max(ys)-min(ys),
                    convergence='observation_only; fixed budget does not prove convergence'))
    for filename,table in [('holdout_seed_results.csv',seed_rows),('holdout_paths_raw.csv.gz',raw),
        ('holdout_means_sample_sd.csv',means),('training_episodes_raw.csv.gz',train),('development_curve_raw.csv.gz',dev),
        ('curve_means_sample_sd.csv.gz',aggregate),('tail_diagnostics.csv',tails)]:write_rows(folder/filename,table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.25})
    def curves(kind,per_seed=False):
        fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),constrained_layout=True)
        for ax,domain in zip(axes,DOMAINS):
            for k,method in enumerate(methods):
                color=plt.get_cmap('tab10')(k)
                if per_seed:
                    records=dev if kind=='development' else train
                    for seed in range(5):
                        rs=sorted([r for r in records if (r['method'],r['domain'],r['seed'])==(method,domain,seed)],key=lambda r:int(r['training_step']))
                        ax.plot([int(r['training_step']) for r in rs],[r['score_x100'] for r in rs],color=color,alpha=.35,lw=.65,label=LABELS[method] if seed==0 else None)
                else:
                    rs=[r for r in aggregate if (r['kind'],r['method'],r['domain'])==(kind,method,domain)]
                    x=np.asarray([r['training_step'] for r in rs]);y=np.asarray([r['score_mean'] for r in rs]);sd=np.asarray([r['score_seed_sample_sd'] for r in rs])
                    ax.plot(x,y,color=color,lw=1.3,label=LABELS[method]);ax.fill_between(x,y-sd,y+sd,color=color,alpha=.12)
            ax.set(xlabel='Learning environment steps',ylabel='SSIM x100' if domain=='radio' else 'Completion (%)',
                title=('Image transmission' if domain=='radio' else 'Emergency energy allocation')+' / '+kind,xlim=(0,STEPS))
            ax.legend(fontsize=8)
        return fig
    with PdfPages(folder/'curves_mean_sample_sd.pdf') as pdf:
        for kind in ('development','training'):
            f=curves(kind);pdf.savefig(f);plt.close(f)
    with PdfPages(folder/'curves_per_seed.pdf') as pdf:
        for kind in ('development','training'):
            f=curves(kind,True);pdf.savefig(f);plt.close(f)
    with PdfPages(folder/'independent_results_report.pdf') as pdf:
        for kind in ('development','training'):
            f=curves(kind);f.suptitle(('PFO formal add-on' if name=='pfo_only' else name)+' | five seeds, fixed 100608-step endpoints')
            pdf.savefig(f);plt.close(f)
        fig,axes=plt.subplots(2,1,figsize=(11.7,8.3),constrained_layout=True)
        for ax,domain in zip(axes,DOMAINS):
            selected=[r for r in means if r['domain']==domain];ax.axis('off')
            table=ax.table(cellText=[[LABELS[r['method']],f"{r['score_mean']:.3f}",f"{r['score_seed_sample_sd']:.3f}"] for r in selected],
                colLabels=['Method','Final score mean','Sample SD across 5 seeds'],cellLoc='center',loc='center')
            table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,1.8)
            ax.set_title(domain+' | 128 paired independent final paths per model')
        fig.suptitle('Fixed endpoint results; no final-test model selection');pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off')
        text=['Tail diagnostics (descriptive; these numbers do not establish convergence).',
              'Slopes: arithmetic mean over 5 training seeds, last 10 unsmoothed DEV points; score points / 10k steps.','']
        for domain in DOMAINS:
            text.append(domain)
            for method in methods:
                rs=[r for r in tails if r['method']==method and r['domain']==domain]
                text.append(f"  {LABELS[method]:26s} tail slope {statistics.mean(r['tail_slope_points_per_10000_steps'] for r in rs):+.3f}; tail mean change {statistics.mean(r['tail_mean_change'] for r in rs):+.3f}")
        text+=['','Learning, DEV, independent final tests, and native physical replay are accounted separately.',
            'Lines join observed measurements only; no interpolation, fabricated points, smoothing, or best-test checkpoint choice.',
            'Shading is sample standard deviation across 5 training seeds; it is not a confidence interval.',
            'Only PFO is newly trained here; original five-method and ablation results remain unchanged.',
            'PFO final256 positions are actually updated; actual learning budget is exactly100608 steps.',
            'PFO replaces CrossQ by user decision. This independent report does not itself modify thesis files.']
        ax.text(.04,.96,'\n'.join(text),va='top',family='monospace',fontsize=9);pdf.savefig(fig);plt.close(fig)
    manifest={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='completion_receipt.json'}
    atomic_json(folder/'completion_receipt.json',dict(status='PASS',fingerprint=fingerprint,plan_sha256=plan_sha,
        report_module_sha256=sha(__file__),artifacts_sha256=manifest,independent_training_endpoints=len(evaluations),
        seeds=list(range(5)),training_steps=STEPS,final_master=FINAL_MASTER,paths_per_seed=128,
        actual_step_axes=True,interpolation=False,smoothing=False,main_pdf_modified=False))
    return str(folder/'completion_receipt.json')

def _run_postprocess(scope=DEFAULT_SCOPE):
    scope=Path(scope).resolve();report=scope/'report';report.mkdir(parents=True,exist_ok=True)
    plan=read(scope/'plan.json');plan_sha=sha(scope/'plan.json');jobs={j['id']:j for j in plan['jobs']}
    require(plan.get('final_master')==FINAL_MASTER and plan.get('final_paths')==128,'Unexpected independent final registration')
    require(len(jobs)==20 and plan.get('new_training_jobs')==10 and plan.get('crossq_included') is False,'Unexpected formal cohort')
    require(collections.Counter(j['kind'] for j in jobs.values())=={'main':10,'evaluate_external':10},'Formal job types differ')
    for relative,digest in plan['source_sha256'].items():require(sha(scope/'execution/source'/relative)==digest,'Frozen source changed '+relative)
    errors,pending,bundles,training,evaluations={},[],{},{},{}
    for jid,job in jobs.items():
        if not (scope/'remote_results'/jid/'transfer_receipt.json').exists():pending.append(jid);continue
        try:
            bundle=closed_bundle(scope,job,plan_sha);bundles[jid]=bundle
            if job['kind'] in ('main','train_A'):training[jid]=validate_training(bundle,plan)
        except (OSError,ValueError,KeyError,TypeError) as e:errors[jid]=str(e)
    for jid,bundle in bundles.items():
        job=jobs[jid]
        try:
            if job['kind']=='train_A' and jid in training:evaluations[jid]=validate_final(bundle,plan)
            elif job['kind']=='evaluate_external':evaluations[jid]=validate_final(bundle,plan,training.get(job['parent_job']))
        except (OSError,ValueError,KeyError,TypeError) as e:errors[jid]=str(e)
    for domain in DOMAINS:
        complete=[e for e in evaluations.values() if e['job']['domain']==domain]
        if complete:
            reference=[r['path_seed'] for r in complete[0]['final_raw']]
            if not all([r['path_seed'] for r in e['final_raw']]==reference for e in complete):errors['paired_final_'+domain]='Final path seeds differ across models/seeds'
    groups={'pfo_only':list(evaluations.values())}
    complete=not errors and not pending and len(training)==10 and len(evaluations)==10
    readiness={name:dict(expected=10,independently_verified=len(es),ready=complete and len(es)==10) for name,es in groups.items()}
    generated={}
    if complete:
        for name,es in groups.items():
            try:generated[name]=report_group(scope,name,es,METHODS,plan_sha)
            except (OSError,ValueError,KeyError,TypeError) as e:errors['report:'+name]=str(e);readiness[name]['ready']=False
    result=dict(schema='chapter3-formal-independent-results-progress-v1',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='ALL_RESULTS_COMPLETE' if complete and not errors else 'PARTIAL_AWAITING_VERIFIED_ENDPOINTS',
        scientific_cohorts=readiness,local_closed_transfers_verified=len(bundles),training_verified=len(training),
        final_endpoints_verified=len(evaluations),pending_local_transfer_count=len(pending),validation_errors=errors,
        generated_reports=generated,plan_sha256=plan_sha,final_master=FINAL_MASTER,main_pdf_modified=False)
    atomic_json(report/'progress.json',result)
    if result['status']=='ALL_RESULTS_COMPLETE':
        receipts={name:dict(path=path,sha256=sha(path)) for name,path in generated.items()}
        require(set(receipts)=={'pfo_only'},'PFO sealed scientific report must exist before publication')
        completed=dict(schema='chapter3-formal-local-independent-completion-v1',status='PASS',
            scope=str(scope),plan_sha256=plan_sha,report_module_sha256=sha(__file__),
            training_jobs_verified=10,external_final_jobs_verified=10,unique_final_endpoints_verified=10,
            complete_closed_transfers_verified=20,training_steps_per_job=STEPS,seeds=list(range(5)),
            final_paths_per_model=128,final_master=FINAL_MASTER,cohort_receipts=receipts,
            report_progress_sha256=sha(report/'progress.json'),main_pdf_modified=False,
            automatic_extra_training=False)
        target=scope/'completion_receipt_local.json';atomic_json(target,completed)
        # A training pointer is maintained separately by the launcher.  The
        # latest scientific-results pointer advances only after all 130 jobs
        # and both independent reports are closed and independently verified.
        atomic_json(scope.parent/'LATEST_PFO_FORMAL_RESULTS.json',dict(
            schema='chapter3-latest-independent-results-v1',status='complete',scope=str(scope),
            completion_receipt=str(target),completion_receipt_sha256=sha(target),
            plan_sha256=plan_sha,cohort_receipts=receipts,main_pdf_modified=False))
    return result

def run_postprocess(scope=DEFAULT_SCOPE):
    scope=Path(scope).resolve();folder=scope/'report';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'postprocess.lock').open('a') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(postprocess_already_running=True)
        return _run_postprocess(scope)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scope',type=Path,default=DEFAULT_SCOPE)
    result=run_postprocess(p.parse_args().scope)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if result.get('validation_errors'):raise SystemExit(1)

if __name__=='__main__':main()
