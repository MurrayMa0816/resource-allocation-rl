#!/usr/bin/env python3
"""Offline DEV-only progress aggregation. Never imports policies/environments.

--once writes one current report; --watch checks every60seconds and exits when
all6formal jobs complete or any formal job fails. No training is restarted.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import textwrap
import time
from zoneinfo import ZoneInfo

import numpy as np

APP = Path(__file__).resolve().parents[1]
STUDY = Path('/hy-tmp/resource_allocation_results/chapter06/experiments/external_learning_comparison_20261109')
OLD = Path('/hy-tmp/resource_allocation_results/chapter06/experiments/reasonable_allocation89_20261109')
REPORTS = STUDY / 'reports'
SEEDS = (6400, 6401, 6402)
TARGET = 259200
INTERVAL = 10368
NAMES = {'full': 'SeDVal', 'capql': 'CAPQL', 'pasta': 'PASTA'}
COLORS = {'full': '#236d9e', 'capql': '#cf7627', 'pasta': '#4a8269'}
STYLES = {'full': ('-', 'o'), 'capql': ('--', 's'), 'pasta': ('-.', '^')}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def atomic_csv(path, rows, fields):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def read_json(path, inputs=None):
    path = Path(path); raw = path.read_bytes()
    if inputs is not None: inputs[str(path)] = sha(raw)
    return json.loads(raw)


def read_npz(path, inputs):
    """Read one coherent atomic NPZ generation, even while worker replaces it."""
    path = Path(path); raw = path.read_bytes(); inputs[str(path)] = sha(raw)
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def formal_configs(inputs=None):
    jobs_path = STUDY / 'protocol/formal_attempt0_jobs.json'
    jobs = read_json(jobs_path, inputs)
    configs = [read_json(path, inputs) for path in jobs['configs']]
    if len(configs) != 6 or {(c['method'], int(c['seed'])) for c in configs} != {
            (m, s) for m in ('capql', 'pasta') for s in SEEDS}:
        raise ValueError('Exactly six registered formal CAPQL/PASTA jobs required')
    for config in configs:
        if (config['engineering'] or config['learning_steps'] != TARGET or
                config['eval_every'] != INTERVAL or config['dev_paths'] != 32):
            raise ValueError('Foreign or engineering configuration in DEV report')
    return configs


def metrics_from_Y(arrays, area):
    quality = np.asarray(arrays['quality'], dtype=np.float64)
    if quality.ndim != 2 or quality.shape[1] != 89 or not np.isfinite(quality).all():
        raise ValueError('All89 finite regional final outcomes required')
    return (np.mean(quality >= .9, axis=1), np.sum(quality * area, axis=1) / np.sum(area))


def complete_points(steps, weather, C, JA, expected_weather):
    if not (steps.shape == weather.shape == C.shape == JA.shape):
        raise ValueError('DEV flat row axes disagree')
    if not np.isfinite(C).all() or not np.isfinite(JA).all():
        raise ValueError('Nonfinite DEV metrics')
    points = {}; pending = []
    for value in np.unique(steps):
        value = int(value); mask = steps == value; count = int(mask.sum())
        if value < 0 or value > TARGET or value % INTERVAL:
            raise ValueError('DEV position is outside the registered actual learning steps')
        if count != 32:
            pending.append(dict(learning_steps=value, rows=count, expected_rows=32)); continue
        if not np.array_equal(np.sort(weather[mask]), np.sort(expected_weather)):
            raise ValueError('DEV weather identity differs from the registered fixed32')
        points[value] = dict(C=float(C[mask].mean() * 100), JA=float(JA[mask].mean() * 100), weather_count=32)
    return points, pending


def collect():
    inputs = {}; configs = formal_configs(inputs)
    weather = np.asarray(configs[0]['dev_weather_seeds'], dtype=np.int64)
    if len(weather) != 32 or len(np.unique(weather)) != 32:
        raise ValueError('Exactly32 distinct DEV weather seeds required')
    if any(c['dev_weather_seeds'] != configs[0]['dev_weather_seeds'] for c in configs):
        raise ValueError('Registered formal jobs do not share the fixed DEV panel')
    original = read_npz(OLD / 'formal/evaluation/main__seed6400/attempt0/results.npz', inputs)
    ids = original['region_ids'].copy()
    area_path = OLD / 'reports/plot_data/cross_day_all89.csv'
    raw = area_path.read_bytes(); inputs[str(area_path)] = sha(raw)
    area_by_id = {r['region_id']: float(r['area_mu']) / 15 for r in csv.DictReader(io.StringIO(raw.decode()))}
    area = np.asarray([area_by_id[str(x)] for x in ids], dtype=np.float64)
    runs = {}; jobs = []
    for seed in SEEDS:
        folder = OLD / f'formal/training/budget_4000000__seed{seed}/attempt0'
        arrays = read_npz(folder / 'development.npz', inputs)
        C, JA = metrics_from_Y(arrays, area)
        names = list(arrays['metric_names'])
        np.testing.assert_allclose(C, arrays['metrics'][:, names.index('C')], rtol=0, atol=1e-12)
        np.testing.assert_allclose(JA, arrays['metrics'][:, names.index('JA')], rtol=0, atol=1e-12)
        points, pending = complete_points(arrays['episode'] * 81, arrays['weather_seed'], C, JA, weather)
        if pending or len(points) != 26 or max(points) != TARGET:
            raise ValueError('Existing Full DEV must retain all26 complete fixed positions')
        runs[('full', seed)] = points
        jobs.append(dict(method='full', display_method=NAMES['full'], seed=seed, status='REUSED_COMPLETE',
                         learning_steps=TARGET, latest_complete_DEV_step=TARGET, complete_DEV_points=26,
                         output=str(folder), pending_DEV_rows=[]))
    for config in configs:
        method, seed = config['method'], int(config['seed']); output = Path(config['output'])
        if not output.resolve().is_relative_to(STUDY.resolve()):
            raise ValueError('Formal output outside the current study')
        receipt = None; failed = None; progress = {}
        if (output / 'completed.json').exists(): receipt = read_json(output / 'completed.json', inputs)
        if (output / 'failed.json').exists(): failed = read_json(output / 'failed.json', inputs)
        if (output / 'progress.json').exists(): progress = read_json(output / 'progress.json', inputs)
        points, pending = {}, []
        if (output / 'development.npz').exists():
            arrays = read_npz(output / 'development.npz', inputs)
            if not np.array_equal(arrays['region_ids'], ids): raise ValueError('Regional ordering differs')
            np.testing.assert_allclose(arrays['area_ha'], area, atol=1e-12, rtol=0)
            C, JA = metrics_from_Y(arrays, area)
            np.testing.assert_allclose(C, arrays['C'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(JA, arrays['JA'], atol=1e-12, rtol=0)
            points, pending = complete_points(arrays['learning_steps'], arrays['weather_seed'], C, JA, weather)
        status = 'FAILED' if failed else 'COMPLETE' if receipt else 'RUNNING' if progress else 'PENDING'
        if receipt and (receipt['learning_steps'] != TARGET or receipt['episodes'] != 3200 or
                        len(points) != 26 or max(points, default=-1) != TARGET or pending):
            raise ValueError('Completion receipt does not match all26 complete fixed DEV positions')
        runs[(method, seed)] = points
        jobs.append(dict(method=method, display_method=NAMES[method], seed=seed, status=status,
                         learning_steps=int(progress.get('learning_steps', receipt.get('learning_steps', 0) if receipt else 0)),
                         latest_complete_DEV_step=max(points, default=None), complete_DEV_points=len(points),
                         output=str(output), pending_DEV_rows=pending,
                         failure_record=str(output / 'failed.json') if failed else None,
                         progress_phase=progress.get('phase'), training_updates=progress.get('updates')))
    summaries = {}; rows = []; seeds = []
    for method in NAMES:
        common = sorted(set.intersection(*(set(runs[(method, seed)]) for seed in SEEDS)))
        summaries[method] = []
        for step in common:
            C = np.array([runs[(method, seed)][step]['C'] for seed in SEEDS])
            JA = np.array([runs[(method, seed)][step]['JA'] for seed in SEEDS])
            row = dict(method=method, display_method=NAMES[method], learning_steps=step,
                       C_mean_percent=float(C.mean()), C_sample_SD_percent=float(C.std(ddof=1)),
                       JA_mean_percent=float(JA.mean()), JA_sample_SD_percent=float(JA.std(ddof=1)),
                       training_seed_count=3, DEV_weather_per_seed=32)
            summaries[method].append(row); rows.append(row)
        for seed in SEEDS:
            for step, point in sorted(runs[(method, seed)].items()):
                seeds.append(dict(method=method, training_seed=seed, learning_steps=step,
                                  C_weather_mean_percent=point['C'], JA_weather_mean_percent=point['JA'], DEV_weather_count=32,
                                  included_in_three_seed_common_curve=step in common))
    current_jobs = [job for job in jobs if job['method'] != 'full']
    failed_jobs = [job for job in current_jobs if job['status'] == 'FAILED']
    completed = len([job for job in current_jobs if job['status'] == 'COMPLETE'])
    finished = completed == 6 and not failed_jobs
    status = 'FAILED_PARTIAL_DEV' if failed_jobs else 'COMPLETE_FIXED_ENDPOINT_DEV' if finished else 'UNFINISHED_DEV'
    mutual = sorted(set.intersection(*(set(r['learning_steps'] for r in summaries[m]) for m in NAMES)))
    latest = {m: summaries[m][-1] if summaries[m] else None for m in NAMES}
    endpoint_ready = finished and all(latest[m] is not None and latest[m]['learning_steps'] == TARGET for m in NAMES)
    return dict(schema='c6-offline-DEV-comparison-v1', status=status, generated_utc=utcnow(),
                target_learning_steps_per_seed=TARGET, completed_new_jobs=completed, total_new_jobs=6,
                jobs=jobs, curves=summaries, latest_method_complete_common_points=latest,
                latest_three_method_shared_step=mutual[-1] if mutual else None,
                fixed_259200_DEV_endpoints={m: latest[m] for m in NAMES} if endpoint_ready else None,
                failed_jobs=[dict(method=j['method'], seed=j['seed'], failure_record=j['failure_record']) for j in failed_jobs],
                is_DEV=True, is_independent_TEST=False, unfinished=not finished, no_rule_methods=True,
                statistical_unit='Each policy32fixedweather mean, then3trainingseeds mean/sampleSD. Onlymethod-specific complete3seedcommonsteps plotted.',
                no_missing_point_imputation=True, no_smoothing=True, no_peak_selection=True,
                input_snapshot_sha256=inputs, source_sha256=sha(Path(__file__).read_bytes()),
                scientific_execution=dict(environment_steps=0, model_loads=0, inference=0, training=0)), rows, seeds


def plotting():
    os.environ['MPLCONFIGDIR'] = str(STUDY / 'runtime_cache/offline_report_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path('/root/.local/share/fonts/thesis/ThesisFigureText-Regular.otf')
    font_manager.fontManager.addfont(str(font)); family = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'font.family': family, 'font.size': 10.5, 'axes.labelsize': 10.5,
        'axes.titlesize': 10.5, 'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5, 'legend.fontsize': 10.5,
        'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'c6-external-DEV-offline-v1', 'savefig.facecolor': 'white'})
    return plt


def draw_curves(fig, state, box):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple
    left, bottom, width, height = box
    axes = [fig.add_axes([left + .13 * width, bottom + .25 * height, .35 * width, .59 * height]),
            fig.add_axes([left + .63 * width, bottom + .25 * height, .35 * width, .59 * height])]
    for method in NAMES:
        rows = state['curves'][method]; x = np.array([r['learning_steps'] / 1000 for r in rows])
        for ax, metric in zip(axes, ['C', 'JA']):
            if not rows: continue
            mean = np.array([r[f'{metric}_mean_percent'] for r in rows]); sd = np.array([r[f'{metric}_sample_SD_percent'] for r in rows])
            linestyle, marker = STYLES[method]
            ax.fill_between(x, mean - sd, mean + sd, color=COLORS[method], alpha=.16, linewidth=0)
            ax.plot(x, mean, color=COLORS[method], linestyle=linestyle, marker=marker, ms=3.0, lw=1.15)
    for ax, metric, title, ylabel in zip(axes, ['C', 'JA'], ['(a) 区域产量达标率', '(b) 平均相对产量'], ['区域产量达标率（%）', '平均相对产量（%）']):
        # A descriptive mean±sampleSD can extend beyond the metric's0–100
        # support. Display the entire band rather than clipping its low tail.
        lows = [r[f'{metric}_mean_percent'] - r[f'{metric}_sample_SD_percent']
                for rows in state['curves'].values() for r in rows]
        highs = [r[f'{metric}_mean_percent'] + r[f'{metric}_sample_SD_percent']
                 for rows in state['curves'].values() for r in rows]
        limits = (min([0.] + lows) - 1.5, max([100.] + highs) + 1.5)
        ax.set(xlim=(-3, 264), ylim=limits, xlabel='训练步数（×10³）', ylabel=ylabel)
        ax.set_xticks([0, 50, 100, 150, 200, 250]); ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_title(title, pad=8); ax.grid(axis='y', alpha=.18, linewidth=.6)
    handles = [(Patch(fc=COLORS[m], alpha=.16, ec='none'), Line2D([], [], color=COLORS[m],
                linestyle=STYLES[m][0], marker=STYLES[m][1], ms=3, lw=1.15)) for m in NAMES]
    fig.legend(handles, list(NAMES.values()), handler_map={tuple: HandlerTuple(ndivide=1)},
               loc='lower center', bbox_to_anchor=(left + .56 * width, bottom + .015 * height),
               frameon=False, ncol=3, handlelength=2.0, handletextpad=.5, columnspacing=1.2)
    return axes


def check_figure(fig):
    from matplotlib.text import Text
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text(): continue
        b = text.get_window_extent(renderer)
        if b.x0 < -.5 or b.y0 < -.5 or b.x1 > fig.bbox.x1 + .5 or b.y1 > fig.bbox.y1 + .5:
            raise ValueError('Report text outside canvas: ' + text.get_text())
    for ax in fig.axes:
        if ax.get_title():
            b = ax.title.get_window_extent(renderer)
            if abs((b.x0 + b.x1 - ax.bbox.x0 - ax.bbox.x1) / 2) > .5:
                raise ValueError('Subfigure title is not centered on its axes')


def text_at(fig, y_mm, text, size=10.5, x_mm=27, weight='normal', center=False):
    fig.text((105 if center else x_mm) / 210, y_mm / 297, text,
             ha='center' if center else 'left', va='top', fontsize=size, fontweight=weight)


def paragraph(fig, y, text, width=42, line_mm=5.6):
    for line in textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False):
        text_at(fig, y, line); y -= line_mm
    return y - 2


def fmt(value, spread):
    return f'{value:.2f}±{spread:.2f}'


def render_report(plt, state, path):
    from matplotlib.backends.backend_pdf import PdfPages
    tmp = path.with_name(path.stem + '.tmp.' + str(os.getpid()) + '.pdf')
    metadata = dict(CreationDate=None, ModDate=None, Creator='C6 offline DEV-only aggregation')
    with PdfPages(tmp, metadata=metadata) as pdf:
        fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
        text_at(fig, 282, '第六章三种学习方法比较（DEV）', 14, center=True)
        condition = '六项新训练已完成，固定259200步端点DEV' if not state['unfinished'] else '训练尚未全部完成，以下为当前已落盘DEV'
        if state['failed_jobs']: condition = '存在失败任务，保留已有DEV，自动汇总停止'
        text_at(fig, 270, condition, 11, center=True)
        y = paragraph(fig, 257, '89区、81日、400万m³。每个策略先对32条固定开发天气取均值，再汇总三个训练种子的均值与样本标准差。')
        paragraph(fig, y, 'SeDVal使用既有记录。每种方法只画三个种子均已完成的共同评价点。缺少的后续点留空，不平滑、不补尾段。')
        draw_curves(fig, state, (27 / 210, 125 / 297, 156 / 210, 95 / 297))
        text_at(fig, 117, '各方法最新的三种子共同完整点（步数可能不同）', 10.5, center=True)
        columns = [28, 66, 103, 145]
        for x, text in zip(columns, ['方法', '训练步数', '达标率/%', '平均相对产量/%']): text_at(fig, 106, text, x_mm=x)
        for index, method in enumerate(NAMES):
            row = state['latest_method_complete_common_points'][method]; yy = 96 - index * 9
            values = [NAMES[method], str(row['learning_steps']) if row else '等待',
                      fmt(row['C_mean_percent'], row['C_sample_SD_percent']) if row else '等待三种子',
                      fmt(row['JA_mean_percent'], row['JA_sample_SD_percent']) if row else '等待三种子']
            for x, text in zip(columns, values): text_at(fig, yy, text, x_mm=x)
        y = paragraph(fig, 63, '平均相对产量仍按各区种植面积比例计算。不同方法最新步数不同时，表中结果不能直接当作最终性能排序。')
        y = paragraph(fig, y, '本报告仅使用开发评价。259200步是预先固定的训练终点，最终DEV结果也不属于新的独立TEST。')
        beijing = datetime.fromisoformat(state['generated_utc']).astimezone(ZoneInfo('Asia/Shanghai'))
        text_at(fig, 17, '生成时间（北京时间） ' + beijing.isoformat(sep=' ', timespec='seconds'), 9, center=True)
        check_figure(fig); pdf.savefig(fig); plt.close(fig)
        fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
        text_at(fig, 282, '任务进度与数据范围', 14, center=True)
        text_at(fig, 268, f"本轮完成 {state['completed_new_jobs']}/6 项，三方法共有最近评价步为 {state['latest_three_method_shared_step']}", 10.5, center=True)
        columns = [22, 57, 77, 112, 150]
        for x, text in zip(columns, ['方法', '种子', '已训练步数', '最近完整DEV', '任务状态']): text_at(fig, 251, text, x_mm=x)
        translations = {'REUSED_COMPLETE': '既有完整记录', 'COMPLETE': '固定终点完成', 'RUNNING': '运行中', 'PENDING': '等待开始', 'FAILED': '失败已保留'}
        for index, job in enumerate(state['jobs']):
            yy = 240 - index * 10
            values = [job['display_method'], str(job['seed']), str(job['learning_steps']),
                      str(job['latest_complete_DEV_step']) if job['latest_complete_DEV_step'] is not None else '尚无完整点', translations[job['status']]]
            for x, text in zip(columns, values): text_at(fig, yy, text, x_mm=x)
        y = paragraph(fig, 139, '三次训练分别使用种子6400至6402。每10368个实际学习交互步评价一次，每个完整回合81步，3200回合共259200学习步。开发交互和优化器迭代不并入横轴。')
        y = paragraph(fig, y, 'SeDVal的三个完整开发记录仅读复用。CAPQL与PASTA分别是本轮新增的三次训练。原始记录中每个评价点有32条天气，未满32条或未被三个种子共同完成的点不进入均值曲线。')
        y = paragraph(fig, y, '读取全部预先登记的种子和固定终点，不按峰值挑模型，不调整训练预算，不新增规则。任务失败时保留已生成的结果并停止自动汇总，不重启训练。')
        y = paragraph(fig, y, '后续独立TEST属于另外的评价阶段，本脚本既不生成TEST，也不加载模型。当前报告不能据不完整曲线推定最终排序。')
        if state['failed_jobs']:
            paragraph(fig, y, '失败任务为 ' + '，'.join(j['method'] + '/' + str(j['seed']) for j in state['failed_jobs']) + '。')
        text_at(fig, 17, '仅离线读取现有DEV数组和进度收据，0新环境步，0模型载入。', 9, center=True)
        check_figure(fig); pdf.savefig(fig); plt.close(fig)
    tmp.replace(path)


def write_outputs(state, summary_rows, seed_rows):
    plt = plotting(); REPORTS.mkdir(parents=True, exist_ok=True)
    folder = REPORTS / 'learning_curves'; folder.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(156 / 25.4, 100 / 25.4)); draw_curves(fig, state, (0, 0, 1, 1)); check_figure(fig)
    for extension in ('pdf', 'svg', 'png'):
        tmp = folder / f'figure.tmp.{os.getpid()}.{extension}'
        kwargs = dict(dpi=200) if extension == 'png' else {}
        if extension == 'pdf': kwargs['metadata'] = dict(CreationDate=None, ModDate=None, Creator='C6 offline DEV curves')
        fig.savefig(tmp, **kwargs); tmp.replace(folder / f'figure.{extension}')
    plt.close(fig)
    atomic_csv(folder / 'three_seed_common_points.csv', summary_rows,
        ['method','display_method','learning_steps','C_mean_percent','C_sample_SD_percent','JA_mean_percent','JA_sample_SD_percent','training_seed_count','DEV_weather_per_seed'])
    atomic_csv(folder / 'seed_weather_means.csv', seed_rows,
        ['method','training_seed','learning_steps','C_weather_mean_percent','JA_weather_mean_percent','DEV_weather_count','included_in_three_seed_common_curve'])
    render_report(plt, state, REPORTS / 'DEV_学习比较进度.pdf')
    atomic_json(REPORTS / 'latest_status.json', state)
    atomic_json(REPORTS / 'latest_endpoints.json', dict(status=state['status'], generated_utc=state['generated_utc'],
        latest_method_complete_common_points=state['latest_method_complete_common_points'],
        fixed_259200_DEV_endpoints=state['fixed_259200_DEV_endpoints'],
        is_independent_TEST=False, unfinished=state['unfinished'], no_rule_methods=True))
    paths = [folder/f'figure.{x}' for x in ('pdf','svg','png')] + [folder/'three_seed_common_points.csv', folder/'seed_weather_means.csv', REPORTS/'DEV_学习比较进度.pdf']
    atomic_json(REPORTS / 'aggregate_manifest.json', dict(status=state['status'], source=str(Path(__file__).resolve()),
        source_sha256=state['source_sha256'], generated_utc=state['generated_utc'],
        input_snapshot_sha256=state['input_snapshot_sha256'],
        assets={str(p.relative_to(REPORTS)):dict(path=str(p),sha256=sha(p.read_bytes())) for p in paths},
        scientific_execution=state['scientific_execution'], shared_curve_training_count=3, weather_per_training=32,
        all_curves_use_only_complete_three_seed_common_points=True, no_smoothing=True, no_rule_methods=True,
        comparison_role='DEV only, no endpoint selection, no independentTEST claim'))


def signature():
    configs = formal_configs(); files = []
    for config in configs:
        output = Path(config['output'])
        for name in ['development.npz', 'completed.json', 'failed.json']:
            path = output / name
            if path.exists():
                st = path.stat(); files.append((str(path), st.st_mtime_ns, st.st_size))
            else: files.append((str(path), None, None))
    return tuple(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--once', action='store_true'); modes.add_argument('--watch', action='store_true')
    args = parser.parse_args(); last = None
    while True:
        try:
            current = signature()
            if current != last:
                state, rows, seeds = collect(); write_outputs(state, rows, seeds); last = current
                event = dict(generated_utc=state['generated_utc'], status=state['status'], completed_new_jobs=state['completed_new_jobs'],
                             latest_common_steps={m: r['learning_steps'] if r else None for m,r in state['latest_method_complete_common_points'].items()})
                with (REPORTS/'aggregate_watch.jsonl').open('a', encoding='utf-8') as stream: stream.write(json.dumps(event, ensure_ascii=False)+'\n')
                print(json.dumps(event, ensure_ascii=False), flush=True)
                if state['failed_jobs']: return 2
                if not state['unfinished']: return 0
            if not args.watch: return 0
            time.sleep(60)
        except KeyboardInterrupt: return 130
        except Exception:
            import traceback
            atomic_json(REPORTS/'aggregation_error.json', dict(status='OFFLINE_AGGREGATION_ERROR', utc=utcnow(),
                error=traceback.format_exc(), environment_steps=0, model_loads=0, training_restarted=False))
            raise


if __name__ == '__main__':
    sys.exit(main())
