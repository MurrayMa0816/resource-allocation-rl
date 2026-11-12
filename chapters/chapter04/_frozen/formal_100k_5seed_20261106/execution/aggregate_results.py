"""Aggregate only the independently accepted fixed C4 cohort; stdlib only."""
from __future__ import annotations
import argparse,csv,datetime,json,math,statistics,sys
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'source/reporting'))
import result_io as io

def save_csv(path,rows,fields=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            w.writerow({k:(json.dumps(v,ensure_ascii=False,separators=(',',':'))if isinstance(v,(dict,list))else v)for k,v in row.items()})

def stats(values):
    r=io.moments(values);r['score_x100_mean']=100*r['mean'];r['score_x100_sample_sd']=100*r['sample_sd'];return r

def counts_and_times(job,d):
    native=d.get('native_receipt',{});diag=d.get('final_diagnostics',{})
    if job['method']=='props':
        mechanism=native.get('mechanism_checks',{})
        actor=critic=mechanism.get('target_minibatches')
        behavior=mechanism.get('behavior_minibatches')
        shared=actor
    else:
        actor=d.get('actor_updates',native.get('actor_updates',diag.get('actor_updates',diag.get('policy_optimizer_steps',diag.get('optimizer_iterations',{}).get('actor')))))
        critic=d.get('critic_updates',native.get('q_optimizer_steps',diag.get('critic_updates',diag.get('rl_updates',diag.get('optimizer_iterations',{}).get('critic')))))
        behavior=shared=None
    return dict(outer_updates=d.get('outer_updates',native.get('target_updates')),
        actor_updates=actor,critic_updates=critic,joint_target_optimizer_steps=shared,
        behavior_optimizer_steps=behavior,encoder_optimizer_steps=diag.get('encoder_optimizer_steps'),
        diffusion_optimizer_steps=native.get('diffusion_training_steps'),curiosity_optimizer_steps=native.get('curiosity_updates'),
        optimizer_count_note='actor/value share one target Adam step; do not sum them as two distinct optimizer calls'if shared is not None else None,
        collection_seconds=d.get('collection_seconds'),update_seconds=d.get('update_seconds',d.get('observe_update_seconds')),
        development_seconds=d.get('evaluation_seconds',d.get('development_wall_seconds')),
        wall_seconds=d.get('elapsed_seconds',d.get('wall_seconds')),actual_device=d.get('actual_device',d.get('device',job['device'])))

def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',type=Path,default=R);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    assert a.scope.resolve()==R,'This aggregator is bound to this registered scope'
    if a.self_test:return self_test()
    accepted=io.read(R/'independent/acceptance.json');assert accepted['status']=='PASS'and accepted['verified_jobs']==190
    accept_sha=io.sha(R/'independent/acceptance.json');plan=io.read(R/'plan.json')
    assert accepted['plan_sha256']==io.sha(R/'plan.json')and accepted['quality_targets_sha256']==io.sha(R/'protocol/quality_targets.json')
    verified={x['id']:x for x in accepted['jobs']};assert set(verified)=={j['id']for j in plan['jobs']}
    for job in plan['jobs']:
        entry=verified[job['id']];out=Path(job['output'])
        for rel,digest in entry['input_sha256'].items():assert io.sha(out/rel)==digest,('accepted input changed',job['id'],rel)
    thresholds=plan['quality_targets']['thresholds'];curves=[];metrics=[];endpoints=[];internal=[];prefixes=[];configs={}
    bykey={};endpoint_bykey={};endpoint_jobs={}
    for job in plan['jobs']:
        out=Path(job['output']);d=io.receipt(out);key=(job['domain'],job['method'],job['seed'])
        if job['kind']=='training':
            curve=io.curve_rows(out);bykey[key]=curve
            for row in curve:curves.append(dict(domain=key[0],method=key[1],seed=key[2],training_step=row['training_step'],mean_return=row['mean_return'],score_x100=100*row['mean_return'],development_cases=32))
            metric=dict(domain=key[0],method=key[1],seed=key[2],auc=io.auc(curve),auc_score_x100=100*io.auc(curve),**counts_and_times(job,d))
            for q in ['0.9','0.95','1.0']:
                label={'0.9':'q90','0.95':'q95','1.0':'q100'}[q];hit=io.attainment(curve,thresholds[key[0]][q])
                metric['nq_'+label]=hit['confirmed_step'];metric['nq_'+label+'_censored']=hit['censored'];metric['nq_'+label+'_restricted']=hit['restricted_step']
            metric['training_steps']=100608;metric['development_steps']=132*32*(32 if key[0]=='radio'else 24)
            metrics.append(metric)
            if job['family']=='f1':
                for row in io.update_rows(out):
                    internal.append({**row,'domain':key[0],'method':key[1],'seed':key[2]})
            for budget in plan['budget_windows']:
                prefix=[x for x in curve if x['training_step']<=budget]
                assert prefix[-1]['training_step']==budget
                value=io.auc(prefix,budget)
                prefixes.append(dict(domain=key[0],method=key[1],seed=key[2],training_budget=budget,
                    normalized_auc=value,normalized_auc_score_x100=100*value,endpoint_mean_return=prefix[-1]['mean_return'],same_long_run_prefix=True))
            if key[2]==0:
                protocol=io.read(out/'protocol.json');cp=out/'config.json';cfg=io.read(cp)if cp.exists()else protocol.get('arguments',{})
                cfgkey=key[1]+'__'+key[0]
                configs[cfgkey]=dict(protocol_path=str(out/'protocol.json'),protocol_sha256=io.sha(out/'protocol.json'),
                    configuration_path=str(cp)if cp.exists()else str(out/'protocol.json'),
                    configuration=cfg,gamma=(1. if job['family']in('f1','props','pgr')else .99),
                    actual_final_diagnostics=d.get('final_diagnostics'),native_receipt_configuration=d.get('native_receipt',{}).get('native_config'),
                    ReDo_recycle_period_units='native optimizer updates'if job['method']=='redo'else None,
                    frozen_task_adapter_source_and_registered_hyperparameters_preserved=True,
                    implementation_boundary=({'redo':'Paper SAC+ReDo TensorFlow-Agents-to-Torch adapter; not a bitwise TensorFlow reproduction.',
                        'mrq':'Pinned author MR.Q Torch core with public-transition/priority-initialization adapter.',
                        'bro':'Pinned author native JAX BRO(Fast), optimistic collector/pessimistic deployment, RR2 and reset15001.',
                        'props':'Author PROPS update functions with matched set actor and flat critic; user-selected target half learning rate.',
                        'pgr':'Native PGR update/generation with known terminal-only synthetic reward support; actual rewards unchanged.'}.get(job['method'],'F1 native complete-trajectory recent-history update with explicit named ablation.')),
                    PGR_terminal_support_adaptation=job['method']=='pgr',PROPS_user_selected_not_validation_optimal=job['method']=='props')
        else:
            cases=io.case_rows(out);vals=[x['native_return']for x in sorted(cases,key=lambda x:x['case_index'])]
            endpoint_bykey[key]=statistics.fmean(vals);endpoint_jobs[key]=job
            for row in cases:endpoints.append(dict(domain=key[0],method=key[1],seed=key[2],case_index=row['case_index'],path_seed=row['path_seed'],native_return=row['native_return'],score_x100=100*row['native_return'],resource_fraction=row['resource_fraction'],steps=row['steps']))
    metricmap={(x['domain'],x['method'],x['seed']):x for x in metrics}
    for key,metric in metricmap.items():metric['endpoint_return']=endpoint_bykey[key];metric['endpoint_score_x100']=100*endpoint_bykey[key]
    methods={domain:{}for domain in io.DOMAINS}
    for domain in io.DOMAINS:
        for method in io.MAIN+io.ABLATIONS+['A_inherited']:
            vals=[endpoint_bykey[(domain,method,seed)]for seed in io.SEEDS]
            group=dict(n=5,seeds=io.SEEDS,endpoint=stats(vals))
            if method!='A_inherited':
                mm=[metricmap[(domain,method,s)]for s in io.SEEDS];group['auc']=stats([x['auc']for x in mm]);group['nq']={}
                for label in ['q90','q95','q100']:
                    restricted=[x['nq_'+label+'_restricted']for x in mm];censored=[x['nq_'+label+'_censored']for x in mm]
                    group['nq'][label]=dict(threshold=thresholds[domain][{'q90':'0.9','q95':'0.95','q100':'1.0'}[label]],
                        seeds=io.SEEDS,reached_steps=[x['nq_'+label]for x in mm],censor_flags=censored,
                        restricted_steps=restricted,reached_n=sum(not x for x in censored),rate=sum(not x for x in censored)/5,
                        censored_n=sum(censored),restricted_mean_steps=statistics.fmean(restricted),sample_sd=statistics.stdev(restricted),
                        right_censor_limit=100608,confirmation='second consecutive unsmoothed DEV point >=threshold')
                group['costs']={k:io.moments([x[k]for x in mm if x[k]is not None])for k in ['actor_updates','critic_updates','joint_target_optimizer_steps','behavior_optimizer_steps','encoder_optimizer_steps','diffusion_optimizer_steps','curiosity_optimizer_steps','collection_seconds','update_seconds','development_seconds','wall_seconds']if any(x[k]is not None for x in mm)}
            methods[domain][method]=group
    aggregate_curves=[]
    cg=defaultdict(list)
    for row in curves:cg[(row['domain'],row['method'],row['training_step'])].append(row['mean_return'])
    for (domain,method,step),values in sorted(cg.items()):
        assert len(values)==5;r=stats(values)
        aggregate_curves.append(dict(domain=domain,method=method,training_step=step,mean_return=r['mean'],sample_sd=r['sample_sd'],score_x100=r['score_x100_mean'],sample_sd_score_x100=r['score_x100_sample_sd'],training_seeds=5))
    pairs=[]
    for domain in io.DOMAINS:
        for method in io.MAIN[1:]+io.ABLATIONS:
            for seed in io.SEEDS:
                f=metricmap[(domain,'f1',seed)];b=metricmap[(domain,method,seed)]
                pairs.append(dict(domain=domain,comparison_method=method,seed=seed,
                    endpoint_gain_F1_minus_comparator=f['endpoint_return']-b['endpoint_return'],
                    auc_gain_F1_minus_comparator=f['auc']-b['auc'],
                    restricted_q95_cost_saved=b['nq_q95_restricted']-f['nq_q95_restricted'],
                    F1_censored=f['nq_q95_censored'],comparator_censored=b['nq_q95_censored'],
                    cost_saving_is_not_claimed_actual_attainment_when_censored=True))
    out=R/'reports';out.mkdir(exist_ok=True)
    outputs={'learning_curves.csv':curves,'learning_curve_aggregate.csv':aggregate_curves,
        'run_metrics.csv':metrics,'endpoint_cases.csv':endpoints,'f1_internal.csv':internal,
        'budget_prefixes.csv':prefixes,'paired_differences.csv':pairs}
    for name,rows in outputs.items():save_csv(out/name,rows)
    summary=dict(schema='chapter4-accepted-formal-summary-v1',status='PASS',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope_root=str(R),acceptance_sha256=accept_sha,plan_sha256=accepted['plan_sha256'],quality_targets_sha256=accepted['quality_targets_sha256'],
        main_methods=io.MAIN,ablation_methods=['f1']+io.ABLATIONS,A_reference_method='A_inherited',
        training_steps=100608,seeds=io.SEEDS,training_master=202611043101,dev_master=202611043101,final_master=2026110694199,
        evaluation_interval=768,development_cases=32,development_points=132,final_cases=256,
        thresholds=thresholds,methods=methods,method_configs=configs,
        native_return_units=True,display_scores_are_native_return_times100=True,
        statistics='Training seed is independent unit: mean and sample SD across five fixed endpoints; no path pooling for seed SD.',
        primary_metric='q95: second consecutive raw DEV threshold crossing; right-censored at100608, report rate and restricted confirmation cost together.',
        A_inherited_new_training_steps=0,A_historical8_case_DEV_excluded_from_main_curves=True,
        budget_windows=plan['budget_windows'],budget_windows_are_same_long_run_prefixes=True,
        no_forced_monotonicity_or_best_checkpoint_selection=True,
        mixed_devices_wallclock_not_a_fair_algorithm_speed_ranking=True,
        factual_scope='Results of the registered task adapters and finite budget; no assertion every native baseline was hyperparameter-optimal.',
        physical_costs={k:accepted[k]for k in ['unique_new_training_steps','development_environment_steps','independent_endpoint_environment_steps','inherited_A_replay_environment_steps']},
        csv_sha256={name:io.sha(out/name)for name in outputs},source_sha256={str(Path(__file__)):io.sha(__file__),str(R/'source/reporting/result_io.py'):io.sha(R/'source/reporting/result_io.py')})
    io.write(out/'summary.json',summary)
    io.write(out/'aggregation_receipt.json',dict(status='PASS',summary_sha256=io.sha(out/'summary.json'),acceptance_sha256=accept_sha,
        unique_training_runs=90,inherited_models=10,endpoint_panels=100,curve_rows=len(curves),endpoint_case_rows=len(endpoints),
        methods=io.MAIN,ablation_methods=io.ABLATIONS,raw_results_unchanged=True,new_environment_steps=0))
    print('PASS summary',out/'summary.json');return 0

def self_test():
    curve=[dict(training_step=0,mean_return=.2),dict(training_step=768,mean_return=.8),dict(training_step=1536,mean_return=.6)]
    assert io.close(io.auc(curve,1536),.6)
    assert io.attainment(curve,.7,1536)['censored']
    assert io.attainment(curve,.5,1536)['confirmed_step']==1536
    assert io.moments([1,2,3,4,5])['sample_sd']==statistics.stdev([1,2,3,4,5])
    io.write(R/'preflight/aggregate_results_selftest.json',dict(status='PASS',synthetic_only=True,tests=['trapezoidal_normalized_AUC','single_hit_censor','second_consecutive_confirmation','sample_SD_not_population_SD'],new_environment_steps=0))
    print('aggregate selftest PASS');return 0
if __name__=='__main__':raise SystemExit(main())
