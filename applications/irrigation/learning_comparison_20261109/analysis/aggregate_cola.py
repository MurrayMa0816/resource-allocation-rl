#!/usr/bin/env python3
"""Separate offline DEV report: existing SeDVal/PASTA and additional COLA.

Original CAPQL/PASTA outputs and reports are never rewritten. This imports only
the pure offline report helper, not policies, models, replay or environments.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo

APP = Path(__file__).resolve().parents[1]
BASE = Path('/hy-tmp/resource_allocation_results/chapter06/experiments/external_learning_comparison_20261109')
STUDY = BASE / 'additional_cola_20261109'
REPORTS = STUDY / 'reports'
HELPER = APP / 'analysis/aggregate.py'
spec = importlib.util.spec_from_file_location('_c6_additional_cola_offline_helpers', HELPER)
shared = importlib.util.module_from_spec(spec); spec.loader.exec_module(shared)
shared.REPORTS = REPORTS
shared.NAMES = {'full': 'SeDVal', 'pasta': 'PASTA', 'cola': 'COLA'}
shared.COLORS = {'full': '#236d9e', 'pasta': '#4a8269', 'cola': '#cf7627'}
shared.STYLES = {'full': ('-', 'o'), 'pasta': ('-.', '^'), 'cola': ('--', 's')}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def formal_configs(inputs=None):
    old_index = shared.read_json(BASE/'protocol/formal_attempt0_jobs.json', inputs)
    pasta = []
    for path in old_index['configs']:
        config = shared.read_json(path, inputs)
        if config['method'] == 'pasta': pasta.append(config)
    if len(pasta) != 3 or {c['seed'] for c in pasta} != {6400,6401,6402}:
        raise ValueError('Exactly three existing PASTA seeds required')
    index = STUDY/'protocol/formal_attempt0_jobs.json'
    if index.exists():
        manifest = shared.read_json(index, inputs)
        cola = [shared.read_json(path, inputs) for path in manifest['configs']]
        if len(cola) != 3 or {(c['method'],c['seed']) for c in cola} != {('cola',s) for s in (6400,6401,6402)}:
            raise ValueError('Exactly three registered additional COLA seeds required')
    else:
        # Planned paths supply no observations or metrics. They appear only as
        # waiting rows until actual formal configuration/output is registered.
        cola = [dict(method='cola',seed=seed,engineering=False,learning_steps=259200,
                     eval_every=10368,dev_paths=32,dev_weather_seeds=pasta[0]['dev_weather_seeds'],
                     output=str(STUDY/f'formal/cola__seed{seed}/attempt0'),formal_config_pending=True)
                for seed in (6400,6401,6402)]
    configs = pasta + cola
    for config in configs:
        if config['engineering'] or config['learning_steps'] != 259200 or config['dev_paths'] != 32 or config['eval_every'] != 10368:
            raise ValueError('Foreign formal DEV condition')
    return configs


shared.formal_configs = formal_configs
BASE_COLLECT = shared.collect


def collect():
    state, rows, seeds = BASE_COLLECT()
    cola = [job for job in state['jobs'] if job['method']=='cola']
    for job in state['jobs']:
        if job['method']=='pasta' and job['status']=='COMPLETE': job['status']='REUSED_COMPLETE'
    failed = [job for job in cola if job['status']=='FAILED']
    completed = sum(job['status']=='COMPLETE' for job in cola)
    finished = completed==3 and not failed
    state.update(schema='c6-additional-COLA-offline-DEV-v1',source_sha256=digest(__file__),
        helper_source_sha256=digest(HELPER),completed_new_jobs=completed,total_new_jobs=3,
        status='FAILED_PARTIAL_DEV' if failed else 'COMPLETE_FIXED_ENDPOINT_DEV' if finished else 'UNFINISHED_DEV',
        unfinished=not finished,
        failed_jobs=[dict(method=j['method'],seed=j['seed'],failure_record=j['failure_record']) for j in failed],
        COLA_formal_configuration_registered=(STUDY/'protocol/formal_attempt0_jobs.json').exists(),
        reference_methods_reused=['SeDVal','PASTA'],main_methods=['SeDVal','PASTA','COLA'],
        CAPQL_history=dict(retained_unchanged=True,original_study=str(BASE),original_reports=str(BASE/'reports'),
            note='CAPQL results motivated the additional method choice; they remain historical evidence and are not deleted or retested by this report.'),
        selection_history='Additional COLA chosen after observed CAPQL results, not an originally simultaneous three-method comparison.',
        original_CAPQL_PASTA_reports_modified=False)
    if not finished: state['fixed_259200_DEV_endpoints']=None
    return state,rows,seeds


def render_report(plt,state,path):
    from matplotlib.backends.backend_pdf import PdfPages
    tmp=path.with_name(path.stem+'.tmp.'+str(os.getpid())+'.pdf')
    with PdfPages(tmp,metadata=dict(CreationDate=None,ModDate=None,Creator='C6 additional COLA offline DEV report')) as pdf:
        fig=plt.figure(figsize=(210/25.4,297/25.4))
        shared.text_at(fig,282,'第六章追加COLA学习比较（DEV）',14,center=True)
        status='COLA固定259200步端点完成，仍仅为DEV' if not state['unfinished'] else f"COLA完成{state['completed_new_jobs']}/3项，训练尚未全部结束"
        if not state['COLA_formal_configuration_registered']: status='COLA正式训练尚未登记，等待新增记录'
        if state['failed_jobs']: status='COLA存在失败任务，保留当前DEV并停止自动汇总'
        shared.text_at(fig,270,status,11,center=True)
        shared.paragraph(fig,258,'SeDVal与PASTA复用既有开发记录。COLA每项策略先对32条天气取均值，再汇总三训练种子的均值与样本标准差。')
        shared.draw_curves(fig,state,(27/210,136/297,156/210,100/297))
        shared.text_at(fig,127,'最新三种子共同完整评价点（各方法步数可能不同）',10.5,center=True)
        columns=[27,66,103,145]
        for x,text in zip(columns,['方法','训练步数','达标率/%','平均相对产量/%']): shared.text_at(fig,117,text,x_mm=x)
        for i,method in enumerate(shared.NAMES):
            row=state['latest_method_complete_common_points'][method]
            values=[shared.NAMES[method],str(row['learning_steps'])if row else'等待',
                shared.fmt(row['C_mean_percent'],row['C_sample_SD_percent'])if row else'等待三种子',
                shared.fmt(row['JA_mean_percent'],row['JA_sample_SD_percent'])if row else'等待三种子']
            for x,text in zip(columns,values):shared.text_at(fig,107-i*8,text,x_mm=x)
        shared.text_at(fig,79,'COLA各任务当前进度',10.5,center=True)
        for i,job in enumerate(j for j in state['jobs'] if j['method']=='cola'):
            word={'COMPLETE':'固定终点完成','RUNNING':'运行中','PENDING':'尚未开始','FAILED':'失败已保留'}[job['status']]
            point=str(job['latest_complete_DEV_step'])if job['latest_complete_DEV_step']is not None else'无完整DEV'
            shared.text_at(fig,70-i*7,f"seed{job['seed']}   已训练{job['learning_steps']}步   最近完整DEV {point}   {word}",x_mm=27)
        shared.paragraph(fig,44,'只画三种子共同完整点，不补尾段，不按峰值挑选。不同步数不作最终排序。本次追加选择发生于已知CAPQL结果后，原CAPQL记录保留。')
        shared.text_at(fig,23,'当前仅DEV，未生成独立TEST，也未加载任何模型或运行环境。',9,center=True)
        beijing=datetime.fromisoformat(state['generated_utc']).astimezone(ZoneInfo('Asia/Shanghai'))
        shared.text_at(fig,14,'北京时间 '+beijing.isoformat(sep=' ',timespec='seconds'),9,center=True)
        shared.check_figure(fig);pdf.savefig(fig);plt.close(fig)
    tmp.replace(path)


shared.render_report=render_report


def write_outputs(state,rows,seeds):
    shared.write_outputs(state,rows,seeds)
    path=REPORTS/'aggregate_manifest.json';value=json.loads(path.read_text())
    value.update(source=str(Path(__file__).resolve()),source_sha256=digest(__file__),
        shared_report_helper=str(HELPER),shared_report_helper_sha256=digest(HELPER),
        main_methods=['SeDVal','PASTA','COLA'],original_CAPQL_PASTA_outputs_modified=False,
        additional_method_selected_after_CAPQL_results=True)
    shared.atomic_json(path,value)


def signature():
    entries=[];index=STUDY/'protocol/formal_attempt0_jobs.json'
    entries.append((str(index),index.stat().st_mtime_ns if index.exists() else None))
    for config in formal_configs():
        for name in ['development.npz','completed.json','failed.json']:
            path=Path(config['output'])/name
            stat=path.stat()if path.exists()else None
            entries.append((str(path),stat.st_mtime_ns if stat else None,stat.st_size if stat else None))
    return tuple(entries)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--once',action='store_true');mode.add_argument('--watch',action='store_true');args=parser.parse_args()
    previous=None
    while True:
        try:
            current=signature()
            if current!=previous:
                state,rows,seeds=collect();write_outputs(state,rows,seeds);previous=current
                event=dict(generated_utc=state['generated_utc'],status=state['status'],completed_COLA_jobs=state['completed_new_jobs'],
                    latest_common_steps={m:r['learning_steps']if r else None for m,r in state['latest_method_complete_common_points'].items()})
                with(REPORTS/'aggregate_watch.jsonl').open('a',encoding='utf-8')as stream:stream.write(json.dumps(event,ensure_ascii=False)+'\n')
                print(json.dumps(event,ensure_ascii=False),flush=True)
                if state['failed_jobs']:return 2
                if not state['unfinished']:return 0
            if not args.watch:return 0
            time.sleep(60)
        except KeyboardInterrupt:return 130
        except Exception:
            import traceback
            shared.atomic_json(REPORTS/'aggregation_error.json',dict(status='OFFLINE_AGGREGATION_ERROR',error=traceback.format_exc(),
                environment_steps=0,model_loads=0,training_restarted=False))
            raise


if __name__=='__main__':sys.exit(main())
