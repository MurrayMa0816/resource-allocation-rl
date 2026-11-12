"""Independent audit and two-seed review; no further training or paper update."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

import numpy as np

SEEDS=(420,421)
VARIANTS=('full','scalar_value','only_j','only_c')
DOMAINS={'radio':dict(N=100,horizon=32,tau=.2),'v5':dict(N=120,horizon=24,tau=.7)}
TICKS=list(range(0,61441,512))
LABELS={'full':'本章方法','scalar_value':'标量价值监督','only_j':'仅原任务目标','only_c':'仅达标覆盖','zero_rule':'固定规则参考'}
REPORT_SCHEMA='chapter5-full-spend-two-seed-review-v1'


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def csv_write(path,rows):
    require(bool(rows),'Refuse an empty review CSV')
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def paired_summary(rows):
    require(len(rows)==2 and sorted(r['seed'] for r in rows)==list(SEEDS),'Exactly two registered training seeds required')
    result={}
    for key in ('K','C','J','F_0_8'):
        values=np.array([row[key] for row in rows],dtype=float)
        require(np.isfinite(values).all(),'Nonfinite result')
        result[key+'_mean']=float(values.mean());result[key+'_sd']=float(values.std(ddof=1))
    return result


def metrics(rows,domain,tau):
    q=np.asarray([row['quality'] for row in rows],dtype=np.float64)
    require(q.shape==(len(rows),DOMAINS[domain]['N']) and np.isfinite(q).all(),'Invalid individual quality array')
    k=(q>=tau).sum(1);c=k/DOMAINS[domain]['N'];j=q.mean(1)
    require(np.array_equal(k,[row['K'] for row in rows]),'Recorded K differs from true inclusive threshold')
    require(np.allclose(c,[row['C'] for row in rows],atol=1e-12,rtol=0),'Coverage differs from K/N')
    require(np.allclose(j,[row.get('J',row['native_return']) for row in rows],atol=1e-10,rtol=0),'Original native J differs from quality mean')
    require(np.allclose(j,[row['native_return'] for row in rows],atol=1e-10,rtol=0),'Native terminal reward changed')
    return dict(K=float(k.mean()),C=float(c.mean()),J=float(j.mean()),F_0_8=float((.8*c+.2*j).mean()))


class TraceAudit:
    """Hash each archive and decompress each required member once per archive.

    Keep compact verified episode results, not the large action/cap arrays.
    Repeated references to the same archive reuse the verified episode cache.
    """
    MEMBERS=('actions','raw_actions','latent_actions','quality','native_rewards','rewards',
             'quota_lower','quota_upper','remaining_before','remaining_after','caps','period_totals')

    def __init__(self,root):
        self.root=Path(root).resolve();self.cache={};self.files={};self.fingerprints={};self.episodes=0

    def _load(self,path,digest,domain,spec):
        path=Path(path).resolve()
        require(path.is_relative_to(self.root),'Trace escaped its registered run directory')
        stat=path.stat();fingerprint=(stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if path in self.cache:
            require(self.files[str(path)]==digest,'Conflicting SHA references to one trace')
            require(self.fingerprints[path]==fingerprint,'A verified trace changed during the report audit')
            return self.cache[path]
        require(sha(path)==digest,'Trace archive SHA mismatch: '+str(path))
        with np.load(path,allow_pickle=False) as archive:
            require(set(self.MEMBERS).issubset(archive.files),'Incomplete raw/physical ledger archive')
            require({'preference','case_plan_json'}.issubset(archive.files),'Trace lacks original preference/case identity')
            data={name:archive[name] for name in self.MEMBERS}
            trace_preferences=archive['preference'];case_plans=archive['case_plan_json']
        a=data['actions'];n,h,N=a.shape
        require((h,N)==(DOMAINS[domain]['horizon'],DOMAINS[domain]['N']),'Wrong real horizon or object count')
        budget=float(spec['initial_budget']);cap_tol=float(spec['action_tolerance'])
        tolerance=min(cap_tol,512*np.finfo(np.float64).eps*max(abs(budget),np.finfo(float).tiny))
        require(data['raw_actions'].shape==a.shape and data['quality'].shape==(n,N),'Wrong raw physical action/quality shape')
        require(data['latent_actions'].shape==(n,h,19 if domain=='radio' else 13),'Wrong original Gaussian event dimension')
        require(data['caps'].shape==a.shape,'Wrong per-object cap shape')
        require(trace_preferences.shape==(n,) and np.all(trace_preferences==.8) and case_plans.shape==(n,),'Foreign deployment preference/case axes')
        for name in self.MEMBERS:
            require(np.isfinite(data[name]).all(),'Nonfinite raw trace member: '+name)
        for name in ('quota_lower','quota_upper','remaining_before','remaining_after','period_totals'):
            require(data[name].shape==(n,h),'Incomplete per-step public ledger: '+name)
        before,after=data['remaining_before'],data['remaining_after']
        totals=a.sum(-1);lower,upper=data['quota_lower'],data['quota_upper'];caps=data['caps']
        require(np.all(a>=0) and np.all(caps>=0) and np.all(a<=caps+cap_tol),'Physical per-object bound violated')
        require(np.all(a[caps==0]==0),'A completed or zero-cap object received resources')
        require(np.all(lower>=-tolerance) and np.all(upper>=lower-tolerance),'Invalid public quota interval')
        require(np.all(totals>=lower-tolerance) and np.all(totals<=upper+tolerance),'Executed period total escaped [L,U]')
        require(np.all(np.abs(totals-data['period_totals'])<=tolerance),'Recorded period totals differ from executed actions')
        require(np.all(np.abs(before[:,0]-budget)<=tolerance),'Wrong initial resource budget')
        require(np.all(np.abs(before[:,1:]-after[:,:-1])<=tolerance),'Resource ledger is not continuous')
        require(np.all(np.abs(before-totals-after)<=tolerance),'Per-step resource balance failed')
        require(np.all(after>=-tolerance) and np.all(np.abs(after[:,-1])<=tolerance),'Terminal resource balance is not zero')
        require(np.all(np.abs(a.sum((1,2))-budget)<=tolerance),'Whole episode did not spend the full initial budget')
        require(np.all(np.abs(lower[:,-1]-before[:,-1])<=tolerance) and np.all(np.abs(upper[:,-1]-before[:,-1])<=tolerance),'Last public quota must equal the remaining budget')
        native=data['native_rewards'];vector=data['rewards']
        require(native.shape==(n,h) and vector.shape==(n,h,2),'Original native and C/J logging rewards have wrong axes')
        require(np.all(native[:,:-1]==0) and np.all(vector[:,:-1]==0),'Intermediate outcome reward was introduced')
        require(np.allclose(native[:,-1],data['quality'].mean(1),atol=1e-10,rtol=0),'Native true terminal J changed')
        verified=[]
        for i in range(n):
            verified.append(dict(quality=data['quality'][i].copy(),total=float(a[i].sum()),remaining=float(after[i,-1]),
                period_totals=totals[i].copy(),terminal_reward=vector[i,-1].copy(),initial_budget=budget,tolerance=tolerance,
                max_ledger_error=float(np.abs(before[i]-totals[i]-after[i]).max()),
                max_quota_violation=float(np.maximum(lower[i]-totals[i],totals[i]-upper[i]).clip(0).max()),
                max_cap_excess=float((a[i]-caps[i]).clip(0).max())))
            verified[-1]['plan']=json.loads(str(case_plans[i]))
        self.cache[path]=verified;self.files[str(path)]=digest;self.episodes+=n
        self.fingerprints[path]=fingerprint
        return verified

    def rows(self,rows,domain,tau,spec):
        for row in rows:
            item=self._load(row['trace_file'],row['trace_sha256'],domain,spec)[int(row['trace_index'])]
            tol=item['tolerance']
            require(np.array_equal(item['quality'],np.asarray(row['quality'])),'JSON and raw individual outcomes differ')
            require(item['plan']==row['plan'] and hashlib.sha256(json.dumps(row['plan'],sort_keys=True).encode()).hexdigest()==row['plan_sha256'],'Raw trace and registered external case identity differ')
            require(abs(item['total']-row['total_used'])<=tol and abs(item['remaining']-row['remaining'])<=tol,'JSON resource ledger differs from physical actions')
            require(abs(float(row['initial_budget'])-item['initial_budget'])<=tol and row['horizon']==DOMAINS[domain]['horizon'],'Episode budget/horizon identity changed')
            require(row.get('full_spend_required') is True and row['full_spend']['full_spend_required'] is True,'Missing whole-budget execution proof')
            require(abs(row['full_spend']['tolerance']-tol)<=np.finfo(float).eps*max(tol,1e-300),'Unregistered resource tolerance')
            require(row['ledger_audit']['status']=='pass','Worker public ledger audit failed')
            require(np.allclose(item['terminal_reward'],[row['C'],row.get('J',row['native_return'])],atol=1e-10,rtol=0),'Terminal logged vector is not [C,J]')
            require(np.allclose(item['period_totals'],row['period_totals'],atol=tol,rtol=0),'Period-by-period logged spending changed')
        return metrics(rows,domain,tau)


def model_payload_audit(path,protocol,complete):
    import torch
    payload=torch.load(path,map_location='cpu',weights_only=False)
    require(payload['format']=='chapter5-full-spend-fixed-endpoint-v1','Old task/model format is not a full-spend endpoint')
    require(payload['protocol']==protocol,'Endpoint embeds another registered scientific protocol')
    require(payload.get('optimizer_backup') is False and payload.get('replay_backup') is False,'Unexpected disk optimizer/replay backup')
    def state_hash(state):
        h=hashlib.sha256()
        for key,value in state.items():
            require(value.device.type=='cpu' and torch.isfinite(value).all(),'Endpoint parameters must be finite portable CPU tensors')
            h.update(key.encode());h.update(value.detach().numpy().tobytes())
        return h.hexdigest()
    for kind in ('actor','critic'):
        actual=state_hash(payload[kind])
        require(actual==payload[kind+'_sha256']==complete['endpoint_'+kind+'_sha256'],'Retained endpoint learned-state hash failed')
    require(payload['learning_step']==61440 and payload['episode']==61440//protocol['spec']['horizon'],'Endpoint is not the fixed registered final model')


def rule_reference(root,expected_plans):
    """No training: one zero raw event at every step of the shared new Env."""
    from .env import Env
    from . import bridge
    from . import common
    out=Path(root)/'zero_rule';out.mkdir(exist_ok=True)
    rows_by_cell={};costs=[];artifacts={}
    for domain in DOMAINS:
        for seed in SEEDS:
            folder=out/f'{domain}__seed{seed}';folder.mkdir(exist_ok=True)
            env=Env(domain,seed,phase='dev',action_mode='raw',path_master=common.DEV_MASTER,observation_dtype=np.float64)
            spec=env.native_spec.to_dict();env.close()
            audit=TraceAudit(folder)
            for phase,count,master,holdout in [('dev',32,common.DEV_MASTER,False),('final',256,common.TEST_MASTER,True)]:
                path=folder/(phase+'.json');start=time.monotonic();records=[]
                if path.exists():
                    previous=read(path)
                    require(previous.get('schema')=='chapter5-full-spend-zero-rule-v1' and previous['master']==master and previous['cases']==count,'Stale or foreign zero-rule result')
                    records=previous['rows'];seconds=previous['seconds']
                else:
                    pool=[Env(domain,seed,phase=phase,action_mode='raw',path_master=master,holdout=holdout,observation_dtype=np.float64) for _ in range(8)]
                    try:
                        for first in range(0,count,8):
                            selected=pool[:min(8,count-first)];results=[]
                            ledgers=[defaultdict(list) for unused in selected]
                            for j,e in enumerate(selected):e.reset(options={'episode_index':first+j})
                            for tick in range(DOMAINS[domain]['horizon']):
                                for e,ledger in zip(selected,ledgers):
                                    lo,hi=bridge.interval(e.state,e.native_spec)
                                    ledger['quota_lower'].append(lo);ledger['quota_upper'].append(hi)
                                    ledger['remaining_before'].append(float(e.state['remaining']));ledger['caps'].append(e.state['caps'].copy())
                                    unused,reward,done,truncated,info=e.step(np.zeros(e.action_dim))
                                    require(not truncated and not info and done==(tick==e.horizon-1),'Zero rule changed the real native horizon')
                                    ledger['remaining_after'].append(float(e.state['remaining']))
                            for e,ledger in zip(selected,ledgers):
                                result=e.terminal_result;result.update({key:np.asarray(value) for key,value in ledger.items()})
                                result['period_totals']=result['actions'].sum(1)
                                # The common archive uses raw_actions for the
                                # decoded physical action before the guard.
                                # This zero-rule Env's original raw event stays
                                # in latent_actions; no latent is relabelled Q.
                                result['raw_actions']=result['actions'].copy()
                                # Exact double-precision logging labels, as in
                                # the Full collector; native rewards stay raw.
                                logged=np.zeros((e.horizon,2),dtype=np.float64)
                                logged[-1]=result['C'],result['native_return']
                                result['rewards']=logged
                                result.update(initial_budget=e.native_spec.initial_budget,horizon=e.horizon,
                                    full_spend=bridge.assert_full_episode(result['remaining'],result['total_used'],e.native_spec),
                                    full_spend_required=True,ledger_audit=dict(status='pass'))
                                results.append(result)
                            trace=folder/f'{phase}_{first:04d}.npz'
                            arrays={key:np.stack([result[key] for result in results]) for key in TraceAudit.MEMBERS}
                            arrays['preference']=np.full(len(results),.8)
                            arrays['case_plan_json']=np.asarray([json.dumps(result['plan'],sort_keys=True) for result in results])
                            np.savez_compressed(trace,**arrays);digest=sha(trace)
                            for j,result in enumerate(results):
                                row={key:result[key] for key in ('K','C','J','native_return','total_used','remaining','initial_budget','horizon','full_spend','full_spend_required','ledger_audit','plan','plan_sha256')}
                                row.update(index=first+j,quality=result['quality'].tolist(),period_totals=result['period_totals'].tolist(),trace_file=str(trace),trace_sha256=digest,trace_index=j)
                                records.append(row)
                    finally:
                        for e in pool:e.close()
                    seconds=time.monotonic()-start
                    write(path,dict(schema='chapter5-full-spend-zero-rule-v1',domain=domain,seed=seed,cases=count,master=master,training_updates=0,
                        definition='all original raw eta,z coordinates equal zero at every step of the common full-spend decoder',seconds=seconds,rows=records))
                require([row['plan_sha256'] for row in records]==expected_plans[(domain,seed,phase)],'Zero rule uses different comparison cases')
                value=audit.rows(records,domain,DOMAINS[domain]['tau'],spec)
                rows_by_cell[domain,seed,phase]=(records,value)
                costs.append(dict(domain=domain,seed=seed,phase=phase,cases=count,physical_steps=count*DOMAINS[domain]['horizon'],seconds=seconds,training_updates=0))
                artifacts[str(path)]=sha(path)
            artifacts.update(audit.files)
    write(out/'receipt.json',dict(schema='chapter5-full-spend-zero-rule-receipt-v1',status='pass',training_updates=0,
        costs=costs,artifacts=artifacts,not_a_seventeenth_training_job=True))
    return rows_by_cell,costs,dict(artifacts,**{str(out/'receipt.json'):sha(out/'receipt.json')})


def analyze(root,*,render=True,include_rule=True):
    from . import common
    root=Path(root).resolve();registration_path=root/'registration.json'
    registration=common.validate_registration(registration_path)
    jobs=registration['jobs']
    require(len(jobs)==16 and len({job['id'] for job in jobs})==16,'Exactly16 freshly registered training jobs required')
    expected={(variant,domain,seed) for variant in VARIANTS for domain in DOMAINS for seed in SEEDS}
    require({(j['variant'],j['domain'],j['seed']) for j in jobs}==expected,'Foreign or missing training condition')
    output=root/'reports';output.mkdir(exist_ok=True)
    curves=[];endpoints=[];costs=[];budgets=[];quality={};plans={};learning_flags=[];model_artifacts={};all_trace_files={}
    for job in jobs:
        complete=common.validate_completed(job,registration)
        folder=Path(job['output']);protocol=read(folder/'protocol.json');spec=protocol['spec']
        require(spec['chapter']==3 and complete['full_spend_required'] and complete['execution_ledger_verified'],'Old budget condition cannot enter the new report')
        require(float(job['tau'])==DOMAINS[job['domain']]['tau'],'Unregistered threshold in the first16')
        audit=TraceAudit(folder);identity=dict(job_id=job['id'],variant=job['variant'],domain=job['domain'],seed=job['seed'],tau=job['tau'])
        ledger=folder/'training_ledger.jsonl';require(sha(ledger)==complete['training_ledger_sha256'],'Training episode ledger SHA changed')
        train=[json.loads(line) for line in ledger.open() if line.strip()]
        require(len(train)==61440//DOMAINS[job['domain']]['horizon'],'Incomplete actual training episodes')
        require([row['index'] for row in train]==list(range(len(train))),'Missing/reordered training episodes')
        require(all(row['variant']==job['variant'] and row['domain']==job['domain'] and row['seed']==job['seed'] for row in train),'Foreign training ledger identity')
        audit.rows(train,job['domain'],job['tau'],spec)
        dev_budget=[]
        for tick in TICKS:
            record=read(folder/f'evaluation_{tick:06d}.json')
            require(record['learning_step']==tick and set(record['preferences'])=={'w080'},'Missing actual512 tick or unexpected preference')
            rows=record['preferences']['w080']['deterministic']['rows'];require(len(rows)==32,'Incomplete development population')
            value=audit.rows(rows,job['domain'],job['tau'],spec)
            dev_budget.extend((row['total_used']/spec['initial_budget'],abs(row['remaining'])) for row in rows)
            curves.append(dict(**identity,train_steps=tick,**value))
            key=(job['domain'],job['seed'],'dev');row_plans=[row['plan_sha256'] for row in rows]
            require(key not in plans or plans[key]==row_plans,'Development arms do not use identical external cases')
            plans[key]=row_plans
            if tick==61440:final_rows=rows
        held=read(folder/'holdout.json');require(held.get('model_selection') is False and held.get('holdout_selection') is False,'Independent test was used for selection')
        require(set(held['endpoint']['preferences'])=={'w080'},'Unexpected endpoint preference')
        rows=held['endpoint']['preferences']['w080']['deterministic']['rows'];require(len(rows)==256,'Incomplete independent endpoint cases')
        value=audit.rows(rows,job['domain'],job['tau'],spec);endpoints.append(dict(**identity,**value))
        quality[job['id']]=np.asarray([row['quality'] for row in rows],dtype=np.float64)
        key=(job['domain'],job['seed'],'final');row_plans=[row['plan_sha256'] for row in rows]
        require(key not in plans or plans[key]==row_plans,'Endpoint arms do not use identical independent cases')
        plans[key]=row_plans
        reload=read(folder/'endpoint_reload_verification.json')
        require(reload['status']=='pass' and reload['final_evaluation_reproduced'] is True,'Endpoint reload proof missing')
        for kind in ('actor','critic'):require(reload[kind+'_sha256']==complete['endpoint_'+kind+'_sha256'],'Reload proof identifies another model')
        reloaded=reload['preferences']['w080']['deterministic']['rows'];require(len(reloaded)==32,'Incomplete real reload evaluation')
        audit.rows(reloaded,job['domain'],job['tau'],spec)
        require([r['plan_sha256'] for r in reloaded]==[r['plan_sha256'] for r in final_rows] and
            np.array_equal([r['quality'] for r in reloaded],[r['quality'] for r in final_rows]),'Retained endpoint does not reproduce the last development point')
        endpoint=folder/'checkpoint_endpoint.pt';require(sha(endpoint)==complete['endpoint_sha256'],'Endpoint file SHA changed')
        model_payload_audit(endpoint,protocol,complete)
        updates=[json.loads(line) for line in (folder/'updates.jsonl').open() if line.strip()]
        counts=complete['counts'];require(counts['train']['physical_steps']==61440,'Unequal real training interaction budget')
        for phase,cases in [('development',121*32),('holdout',256),('reload_verification',32)]:
            require(counts[phase]['episodes']==cases and counts[phase]['physical_steps']==cases*DOMAINS[job['domain']]['horizon'],'Incorrect charged evaluation cost')
        gpu=complete['gpu_proof'];require(gpu.get('cuda_training_verified') is True,'Actual GPU update/Adam proof missing')
        costs.append(dict(**identity,train_steps=61440,development_steps=counts['development']['physical_steps'],
            holdout_steps=counts['holdout']['physical_steps'],reload_steps=counts['reload_verification']['physical_steps'],
            native_outer_updates=len(updates),actor_optimizer_steps=sum(r.get('actor_committed_optimizer_steps',r.get('optimizer_steps',0)) for r in updates),
            critic_optimizer_steps=sum(r.get('critic_committed_optimizer_steps',r.get('optimizer_steps',0)) for r in updates),seconds=complete['seconds']))
        own=[r for r in curves if r['job_id']==job['id']]
        objective='J' if job['variant']=='only_j' else 'C' if job['variant']=='only_c' else 'F_0_8'
        if np.mean([r[objective] for r in own[-3:]])<own[0][objective]-.05:
            learning_flags.append(dict(job_id=job['id'],kind='late_training_objective_below_initial',objective=objective,
                initial=own[0][objective],late_three_point_mean=float(np.mean([r[objective] for r in own[-3:]]))))
        for phase,phase_rows in [('train',train),('development',None),('holdout',rows),('reload',reloaded)]:
            if phase_rows is not None:
                budgets.append(dict(**identity,phase=phase,episodes=len(phase_rows),max_terminal_residue=max(abs(r['remaining']) for r in phase_rows),
                    min_budget_usage=min(r['total_used']/spec['initial_budget'] for r in phase_rows),max_budget_usage=max(r['total_used']/spec['initial_budget'] for r in phase_rows),status='pass'))
        budgets.append(dict(**identity,phase='development',episodes=len(dev_budget),max_terminal_residue=max(x[1] for x in dev_budget),
            min_budget_usage=min(x[0] for x in dev_budget),max_budget_usage=max(x[0] for x in dev_budget),status='pass'))
        all_trace_files.update(audit.files)
        for path in [endpoint,folder/'protocol.json',folder/'completed.json',folder/'endpoint_reload_verification.json',Path(job['config'])]:model_artifacts[str(path)]=sha(path)
    rule_costs=[];rule_artifacts={};rule_development=[]
    if include_rule:
        rule,rule_costs,rule_artifacts=rule_reference(root,plans)
        for (domain,seed,phase),(rows,value) in rule.items():
            if phase=='final':
                endpoints.append(dict(job_id='zero_rule__'+domain+'__seed'+str(seed),variant='zero_rule',domain=domain,seed=seed,tau=DOMAINS[domain]['tau'],**value))
                quality['zero_rule__'+domain+'__seed'+str(seed)]=np.asarray([row['quality'] for row in rows])
            else:rule_development.append(dict(domain=domain,seed=seed,**value))
    summaries=[];paired=[]
    for domain in DOMAINS:
        for variant in VARIANTS+(('zero_rule',) if include_rule else ()):
            selected=[r for r in endpoints if r['domain']==domain and r['variant']==variant]
            summaries.append(dict(domain=domain,variant=variant,seeds=2,cases_per_seed=256,**paired_summary(selected)))
            if variant!='full':
                for row in selected:
                    full=next(r for r in endpoints if r['domain']==domain and r['variant']=='full' and r['seed']==row['seed'])
                    paired.append(dict(domain=domain,variant=variant,seed=row['seed'],**{key:row[key]-full[key] for key in ('K','C','J','F_0_8')}))
    for name,rows in [('development_curves',curves),('independent_endpoint',endpoints),('endpoint_summary',summaries),
                      ('paired_delta_vs_full',paired),('physical_compute_costs',costs),('full_spend_audit',budgets)]:csv_write(output/(name+'.csv'),rows)
    if rule_costs:csv_write(output/'zero_rule_costs.csv',rule_costs);csv_write(output/'zero_rule_development.csv',rule_development)
    paired_summaries=[]
    for domain in DOMAINS:
        for variant in VARIANTS[1:]+(('zero_rule',) if include_rule else ()):
            selected=[r for r in paired if r['domain']==domain and r['variant']==variant]
            paired_summaries.append(dict(domain=domain,variant=variant,sign='variant minus Full',**paired_summary(selected)))
    csv_write(output/'paired_delta_summary.csv',paired_summaries)
    np.savez_compressed(output/'independent_quality.npz',**quality)
    write(output/'trace_audit_receipt.json',dict(status='pass',verified_trace_files=len(all_trace_files),
        trace_sha256=all_trace_files,archive_members_loaded_once=True,full_spend_every_recorded_training_development_holdout_reload_episode=True))
    if render:figures(output,curves,summaries,paired_summaries,rule_development,learning_flags)
    artifacts={str(p):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='report.json'}
    result=dict(schema=REPORT_SCHEMA,status='needs_review' if learning_flags else 'pass',integrity_status='pass',
        completed_training_jobs=16,training_job_ids=sorted(j['id'] for j in jobs),seeds=list(SEEDS),
        registration_sha256=sha(registration_path),artifacts=artifacts,model_artifacts=model_artifacts,
        zero_rule_artifacts=rule_artifacts,zero_rule_additional_physical_steps=sum(row['physical_steps'] for row in rule_costs),
        learning_review_flags=learning_flags,full_spend_required=True,actual_evaluation_ticks=TICKS,
        statistical_unit='two independent training seeds; within-seed case means, then mean and sample SD; no significance test',
        integrity_is_not_algorithm_stability=True,raw_curves_unsmoothed=True,model_selection=False,holdout_selection=False,
        more_seeds_started=False,automatic_next_stage=False,paper_updated=False,next_step='await_user_review_then_stop')
    write(output/'report.json',result)
    return result


def figures(folder,curves,summary,paired,rule_development,flags):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, ft2font
    from matplotlib.backends.backend_pdf import PdfPages
    candidates=[Path(os.environ['CH5_REPORT_FONT'])] if os.environ.get('CH5_REPORT_FONT') else [
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),Path('/root/.local/share/fonts/microsoft/simsun.ttc'),Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')]
    font=next((p for p in candidates if p.is_file() and all(ft2font.FT2Font(str(p)).get_char_index(ord(c)) for c in 'K0达标±')),None)
    require(font is not None,'A complete Chinese and Latin report font must be installed or set through CH5_REPORT_FONT')
    font_manager.fontManager.addfont(str(font));font_name=font_manager.FontProperties(fname=str(font)).get_name()
    write(folder/'figure_environment.json',dict(matplotlib=matplotlib.__version__,numpy=np.__version__,font_path=str(font),font_sha256=sha(font)))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[font_name,'DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':11,'pdf.fonttype':42})
    colors={'full':'#b32025','scalar_value':'#1565c0','only_j':'#2e7d32','only_c':'#8e24aa'}
    pdf=PdfPages(folder/'两种子_全部预算用尽_结果审阅.pdf')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex='col')
    for column,domain in enumerate(DOMAINS):
        for variant in VARIANTS:
            rows=sorted([r for r in curves if r['domain']==domain and r['variant']==variant],key=lambda r:(r['seed'],r['train_steps']))
            for line,key in enumerate(('K','J')):
                matrix=np.asarray([[r[key] for r in rows if r['seed']==seed] for seed in SEEDS]);require(matrix.shape==(2,121),'Learning plot must retain all actual121 points')
                scale=100 if key=='J' and domain=='v5' else 1
                mean=matrix.mean(0)*scale;sd=matrix.std(0,ddof=1)*scale
                axes[line,column].plot(TICKS,mean,color=colors[variant],label=LABELS[variant],lw=1.6)
                axes[line,column].fill_between(TICKS,mean-sd,mean+sd,color=colors[variant],alpha=.13)
        if rule_development:
            rr=[r for r in rule_development if r['domain']==domain]
            for line,key in enumerate(('K','J')):
                value=np.mean([r[key] for r in rr])*(100 if key=='J' and domain=='v5' else 1)
                axes[line,column].axhline(value,color='#666666',ls='--',lw=1.2,label='固定规则参考（同开发情景，无训练）')
        axes[0,column].set_title('图像下行传输' if domain=='radio' else '应急资源配置')
        axes[0,column].set_ylabel('达标对象数 K');axes[1,column].set_ylabel('原始平均 SSIM' if domain=='radio' else '平均区域完成比例（%）')
        axes[1,column].set_xlabel('实际训练交互步数（每512步真实评价）')
        for ax in axes[:,column]:ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('两训练种子的均值与样本标准差；原始121点，无曲线平滑')
    fig.tight_layout(rect=(0,.1,1,.95));fig.savefig(folder/'学习曲线_四子图.pdf');fig.savefig(folder/'学习曲线_四子图.png',dpi=180);pdf.savefig(fig);plt.close(fig)
    for domain in DOMAINS:
        fig,ax=plt.subplots(figsize=(11,6));ax.axis('off');data=[]
        for row in [r for r in summary if r['domain']==domain]:
            scale=100 if domain=='v5' else 1
            data.append([LABELS[row['variant']],f"{row['K_mean']:.2f} ± {row['K_sd']:.2f}",
                f"{scale*row['J_mean']:.3f} ± {scale*row['J_sd']:.3f}",f"{row['F_0_8_mean']:.4f} ± {row['F_0_8_sd']:.4f}"])
        ax.table(cellText=data,colLabels=['方法','K','原始SSIM' if domain=='radio' else '平均区域完成比例（%）','辅助 F0.8'],loc='center',cellLoc='center').scale(1,2)
        ax.set_title(('图像下行传输' if domain=='radio' else '应急资源配置')+'：256独立情景后跨两种子汇总')
        fig.tight_layout();fig.savefig(folder/(domain+'_终点表.png'),dpi=180);pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,6));ax.axis('off');delta=[]
        for row in [r for r in paired if r['domain']==domain]:
            scale=100 if domain=='v5' else 1
            delta.append([LABELS[row['variant']],f"{row['K_mean']:+.3f} ± {row['K_sd']:.3f}",
                f"{scale*row['J_mean']:+.4f} ± {scale*row['J_sd']:.4f}",f"{row['F_0_8_mean']:+.5f} ± {row['F_0_8_sd']:.5f}"])
        ax.table(cellText=delta,colLabels=['对照减Full','ΔK','ΔSSIM' if domain=='radio' else 'Δ完成比例（百分点）','ΔF0.8'],loc='center',cellLoc='center').scale(1,2)
        ax.set_title(('图像下行传输' if domain=='radio' else '应急资源配置')+'：同训练种子与情景配对差（均值±样本标准差）')
        fig.tight_layout();fig.savefig(folder/(domain+'_对Full配对差.png'),dpi=180);pdf.savefig(fig);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6));ax.axis('off')
    text=['预算条件与验收范围','全部16项从零训练；每回合累计投入等于初始预算。','原始动作/逐对象质量/逐步[L,U]/caps/资源账本与SHA逐项核查。',
          '完整性通过只说明数据与执行协议成立，不等同算法稳定或性能提高。','消融与Full差值见paired_delta_vs_full.csv，同一训练种子和独立情景配对。',
          '固定规则参考无训练：输入固定零潜变量，经同一可行映射按剩余预算分配，并非零投入。',
          '其开发横线来自相同固定情景，额外评价32,256步单独列账。',f'训练目标末三点评价低于初值0.05以上的记录：{len(flags)}项。',
          '两种子仅作描述，不作显著性或普遍优势结论；不自动增加种子或改论文。']
    ax.text(.02,.95,'\n\n'.join(text),va='top',fontsize=12);pdf.savefig(fig);plt.close(fig);pdf.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    args=parser.parse_args();value=analyze(args.root)
    print(json.dumps({key:value[key] for key in ('status','integrity_status','completed_training_jobs','learning_review_flags')},ensure_ascii=False))
