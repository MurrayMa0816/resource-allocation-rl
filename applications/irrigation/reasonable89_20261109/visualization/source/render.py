#!/usr/bin/env python3
"""Eight figure groups for the NEW reasonable89 study, from accepted data only.

No result is rendered without the new frozen-run acceptance and independent
numeric audit.  No old trajectory, model, experiment, publisher, or TeX is used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from data_adapter import Evidence, ROOT, MAP_DAYS, SEEDS, aggregate, dump, rows_csv, sha, public_deficit, select_cross_day_examples
import display_helpers as display
import process_panels as process

SOURCE = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
THESIS = Path('/root/gamewar/latex_thesis_resource_allocation')
FIGURES = {
    'scene': 'figure06-01_洪庙支渠89区灌溉推演场景',
    'workflow': 'figure06-02_灌溉推演环境交互流程',
    'overview': 'figure06-03_全期用水安排与供水余额',
    'maps': 'figure06-04_区域配水过程',
    'learning': 'figure06-05_自身学习与终点成效',
    'regions': 'figure06-06_区域与面积保障',
    'resources': 'figure06-07_供水条件与配置成效',
    'feedback': 'figure06-08_天气变化后的重新配置',
}
WIDTH = 156 / 25.4
BLUE, ORANGE, GREEN, GREY = '#236d9e', '#cf7627', '#4a8269', '#999999'
SEED_COLORS = (BLUE, ORANGE, GREEN)
SEED_MARKERS = ('o', 's', '^')


def setup():
    plt = display.plotting(ROOT)
    plt.rcParams.update({'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5, 'legend.fontsize': 10.5,
                         'svg.hashsalt': 'reasonable89-new-frozen-evidence'})
    return plt


def source_snapshot():
    names = ('render.py', 'data_adapter.py', 'process_panels.py', 'display_helpers.py')
    hashes = {name: sha(SOURCE / name) for name in names}
    import hashlib
    key = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    folder = OUT / 'source_versions' / key
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = folder / name
        if not target.exists():
            shutil.copy2(SOURCE / name, target)
        if sha(target) != hashes[name]:
            raise ValueError('An immutable plotting-source snapshot changed')
    for name in ('geometry', 'diagrams', 'channel_style.json', 'static_scene'):
        target = folder / name
        if not target.exists():
            target.symlink_to(SOURCE / name, target_is_directory=(SOURCE / name).is_dir())
    return folder, hashes


def save(fig, key, plt, evidence=None, extra=None, name='figure'):
    folder = OUT / FIGURES[key]; folder.mkdir(parents=True, exist_ok=True)
    from matplotlib.text import Text
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); bad = []
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text() or item.get_clip_on():
            continue
        box = item.get_window_extent(renderer)
        if (box.x0 < fig.bbox.x0 - .5 or box.y0 < fig.bbox.y0 - .5 or
                box.x1 > fig.bbox.x1 + .5 or box.y1 > fig.bbox.y1 + .5):
            bad.append(item.get_text())
    if bad:
        raise ValueError('Visible text outside the figure: ' + repr(bad))
    result = display.save(fig, folder / name, plt)
    snapshot, hashes = source_snapshot()
    result.update(key=key, paper_group=list(FIGURES).index(key) + 1,
                  source=str(snapshot / 'render.py'), source_sha256=hashes['render.py'],
                  complete_source_snapshot_sha256=hashes, active_maintenance_source=str(SOURCE / 'render.py'),
                  input_files_sha256=evidence.bindings if evidence else {},
                  objective_order=['C90', 'JA'], numeric_result_origin='new accepted reasonable89 run only',
                  new_environment_steps=0, new_model_inferences=0, root_tex_modified=False)
    if evidence:
        result['selected_application_identity'] = dict(adapter=evidence.adapter_id, training_weight=evidence.weight,
            spatial_sigma_scale=evidence.sigma_scale, execution=evidence.execution)
    if extra:
        result.update(extra)
    dump(folder / (name + '_source.json'), result)
    if evidence:
        recipe = ('from pathlib import Path\nimport runpy,sys\n' +
            f'code=Path({str(snapshot / "render.py")!r})\nsys.path.insert(0,str(code.parent))\n' +
            f'sys.argv=[str(code),"--index",{str(evidence.index_path)!r},"--groups",{key!r}]\n' +
            'runpy.run_path(str(code),run_name="__main__")\n')
        (folder / 'plot.py').write_text(recipe)
    return result


def static_assets(groups):
    result = []
    if 'scene' in groups:
        folder = OUT / FIGURES['scene']
        shutil.copytree(SOURCE / 'static_scene', folder, dirs_exist_ok=True)
        result.append(dict(key='scene', pdf=str(folder / 'figure.pdf'), sha256=sha(folder / 'figure.pdf'),
                           interpretation='Frozen categorical scene, not old training result'))
    if 'workflow' in groups:
        folder = OUT / FIGURES['workflow']; folder.mkdir(parents=True, exist_ok=True)
        origin = SOURCE / 'diagrams/figure06-02_灌溉仿真环境交互流程/source.drawio'
        target = folder / 'source.drawio'
        if not target.exists() or sha(target) != sha(origin):
            shutil.copy2(origin, target)
        spec = importlib.util.spec_from_file_location('reasonable89_native_export', THESIS / 'scripts/export_method_figures.py')
        exporter = importlib.util.module_from_spec(spec); spec.loader.exec_module(exporter)
        args = SimpleNamespace(output_root=None, force=False, drawio=None, work_dir=ROOT / 'runtime_cache/drawio')
        record = exporter.export(target, args)
        # Official native source/PDF/font receipt is kept in source.json.
        import fitz
        with fitz.open(folder / 'figure.pdf') as document:
            document[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(folder / 'figure.png')
            # Vector sidecar from the official calibrated PDF; no raster image
            # or independent redraw can change its geometry/font appearance.
            (folder / 'figure.svg').write_text(document[0].get_svg_image())
        dump(folder / 'svg_source.json', dict(svg_sha256=sha(folder / 'figure.svg'),
            origin='vector extraction of the official native-source PDF, glyph outlines; no raster',
            native_pdf_sha256=sha(folder / 'figure.pdf'), editable_source='source.drawio',
            drawio_source_sha256=sha(target), pdf_cache_untouched=True))
        result.append(dict(key='workflow', source=str(target), source_sha256=sha(target),
                           pdf=str(folder / 'figure.pdf'), sha256=sha(folder / 'figure.pdf'), export=record))
    return result


def learning(evidence, plt):
    from matplotlib.lines import Line2D
    curves = evidence.development_curves()
    height_mm = 180.
    fig = plt.figure(figsize=(WIDTH, height_mm / 25.4))
    axes = [fig.add_axes([.14, .71, .35, .235]), fig.add_axes([.62, .71, .35, .235])]
    # Exact square, common complete 0--100% scales for the two TEST outcomes.
    endpoint = fig.add_axes([.32, .21, .44, .44 * 156 / height_mm])
    data, joint_rows, pairs = [], [], []
    for order, (seed, color, marker) in enumerate(zip(SEEDS, SEED_COLORS, SEED_MARKERS), 1):
        curve = next(item for item in curves if item['seed'] == seed)
        initial = next(item for item in evidence.initial if item.seed == seed)
        final = next(item for item in evidence.main if item.seed == seed)
        for column, metric in enumerate(('C90', 'JA')):
            axes[column].plot(curve['episodes'], curve[metric] * 100, color=color, marker=marker, ms=2.4,
                              lw=1, label=f'第{order}次训练')
            for episode, value in zip(curve['episodes'], curve[metric]):
                data.append(dict(source='fixed32_DEV', training_seed=seed, episode=int(episode), metric=metric, value=float(value)))
        points = np.asarray([[record.metrics[metric].mean() * 100 for metric in ('C90', 'JA')]
                             for record in (initial, final)])
        pairs.append(points)
        endpoint.annotate('', xy=points[1], xytext=points[0],
            arrowprops=dict(arrowstyle='->', color=color, alpha=.65, lw=.9, shrinkA=5, shrinkB=5),
            annotation_clip=False)
        for phase, point, face in zip(('initial', 'final'), points, ('white', color)):
            endpoint.plot(point[0], point[1], marker=marker, ms=5, linestyle='none', color=color,
                          markerfacecolor=face, markeredgewidth=1, clip_on=False, zorder=5)
            joint_rows.append(dict(source='paired128_TEST', training_seed=seed, independent_training=order,
                phase=phase, episode=0 if phase == 'initial' else 3200,
                C90=float(point[0] / 100), JA=float(point[1] / 100), weather_count=128))
            for metric, value in zip(('C90', 'JA'), point):
                data.append(dict(source='paired128_TEST', training_seed=seed,
                    episode=0 if phase == 'initial' else 3200, metric=metric, value=float(value / 100)))
    for column, metric in enumerate(('C90', 'JA')):
        ylabel = '区域产量达标率（%）' if metric == 'C90' else '面积加权相对产量（%）'
        axes[column].set(xlabel='已完成训练回合', ylabel=ylabel, xlim=(0, 3200), ylim=(0, 100))
        axes[column].set_xticks([0, 800, 1600, 2400, 3200])
        axes[column].set_title(f'({chr(97 + column)}) ' + ('区域产量达标率的学习过程' if metric == 'C90' else '平均相对产量的学习过程'), pad=7)
        axes[column].grid(axis='y', alpha=.17)
    mean_points = np.asarray(pairs).mean(axis=0)
    endpoint.annotate('', xy=mean_points[1], xytext=mean_points[0],
        arrowprops=dict(arrowstyle='->', color='black', lw=1.3, shrinkA=7, shrinkB=7), annotation_clip=False)
    for point, face in zip(mean_points, ('white', 'black')):
        endpoint.plot(point[0], point[1], marker='D', ms=7, linestyle='none', color='black',
                      markerfacecolor=face, markeredgewidth=1.1, clip_on=False, zorder=7)
    endpoint.set(xlim=(0, 100), ylim=(0, 100), xlabel='区域产量达标率（%）', ylabel='面积加权相对产量（%）')
    endpoint.set_xticks([0, 25, 50, 75, 100]); endpoint.set_yticks([0, 25, 50, 75, 100])
    endpoint.set_title('(c) 独立终测中的训练前后成效', pad=7)
    endpoint.grid(alpha=.17)
    seeds = [Line2D([], [], color=color, marker=marker, ms=4, label=f'第{order}次训练')
             for order, (color, marker) in enumerate(zip(SEED_COLORS, SEED_MARKERS), 1)]
    phases = [Line2D([], [], color='black', marker='o', markerfacecolor='white', ms=5,
                    linestyle='none', label='训练前'),
              Line2D([], [], color='black', marker='o', markerfacecolor='black', ms=5,
                    linestyle='none', label='训练后'),
              Line2D([], [], color='black', marker='D', markerfacecolor='white', ms=6,
                    linestyle='none', label='训练前（三次均值）'),
              Line2D([], [], color='black', marker='D', markerfacecolor='black', ms=6,
                    linestyle='none', label='训练后（三次均值）')]
    fig.legend(handles=seeds, loc='lower center', bbox_to_anchor=(.55, .102), ncol=3, frameon=False,
               handletextpad=.4, columnspacing=1.)
    fig.legend(handles=phases, loc='lower center', bbox_to_anchor=(.55, .014), ncol=2, frameon=False,
               handletextpad=.4, columnspacing=1.)
    folder = OUT / FIGURES['learning']; folder.mkdir(parents=True, exist_ok=True)
    rows_csv(folder / 'learning_display_values.csv', data)
    rows_csv(folder / 'paired_TEST_joint_C90_JA.csv', joint_rows)
    return save(fig, 'learning', plt, evidence, dict(panel_count=3,
        DEV_role='within-training diagnostic, not an independent TEST estimate',
        TEST_role='three before/after policy pairs; every small point is its full128-weather mean; diamond is the three-policy mean',
        TEST_joint_axes=['C90 percent, complete0to100', 'JA percent, complete0to100'],
        all_true0_and_every128_DEV_position_kept=True, initial_comparison_budget_m3=4000000,
        secondary_budgets_have_no_initialization_comparator=True))


def regional_guarantee(evidence, plt):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(WIDTH, 195 / 25.4))
    top = fig.add_axes([.14, .71, .83, .22])
    depth = fig.add_axes([.14, .40, .56, .18])
    yield_top = fig.add_axes([.80, .40, .17, .18])
    worst = fig.add_axes([.14, .15, .83, .14])
    phases = [('initial', evidence.initial, GREY, '训练前'), ('final', evidence.main, BLUE, '训练后')]
    order = np.asarray(sorted(range(89), key=lambda i: (evidence.area[i], int(evidence.ids[i]))))
    top5 = evidence.top5; values = {}; rows = []
    for phase, records, color, label in phases:
        regional_by_policy = np.stack([record.arrays['Y'].mean(axis=0) for record in records])
        region_mean = regional_by_policy.mean(axis=0)
        top.scatter(np.arange(89), region_mean[order] * 100, color=color, marker='o' if phase == 'final' else 'x',
                    s=12 if phase == 'final' else 14, linewidths=.65, zorder=4 if phase == 'final' else 3)
        values[phase] = regional_by_policy
        for row, record in enumerate(records):
            minimum = int(np.argmin(regional_by_policy[row]))
            for i, region in enumerate(evidence.ids):
                rows.append(dict(phase=phase, training_seed=record.seed, region_id=region,
                    area_mu=float(evidence.area[i] * 15), weather_mean_Y=float(regional_by_policy[row, i]),
                    weather_mean_total_I_mm=float(record.metrics['I_mm_by_region'][:, i].mean()),
                    weather_mean_zero_irrigation_days=float(record.metrics['zero_days_by_region'][:, i].mean()),
                    fixed_top5=bool(i in top5), is_worst_weather_mean_region=bool(i == minimum)))
    for order_seed, (seed, color, marker) in enumerate(zip(SEEDS, SEED_COLORS, SEED_MARKERS), 1):
        row = next(i for i, record in enumerate(evidence.main) if record.seed == seed)
        worst.plot([0, 1], [values[phase][row].min() * 100 for phase in ('initial', 'final')],
                   color=color, marker=marker, ms=4, lw=1)
    top.axhline(90, color='#777777', lw=.9, ls='--')
    top.set(xlim=(-1.5, 90), ylim=(0, 105), ylabel='平均相对产量（%）', xlabel='区域编号（按面积从小到大排列）')
    ticks = [0, 22, 44, 66, 88]; top.set_xticks(ticks); top.set_xticklabels([evidence.ids[order[i]] for i in ticks])
    top.set_title('(a) 89区的平均相对产量', pad=7)
    # These two aligned panels describe the same predeclared complete case;
    # the upper/lower panels summarize the whole formal weather population.
    case = evidence.case()
    daily_I = case['a_m3'][:, top5].T / (10 * evidence.area[top5, None])
    upper = float(np.ceil(float(daily_I.max()) / 5.) * 5.) if daily_I.max() > 0 else 1.
    cmap = plt.get_cmap('Blues').copy(); cmap.set_bad(display.ZERO_COLOR)
    painted = np.ma.masked_where(daily_I == 0, daily_I)
    image = depth.imshow(painted, origin='upper', aspect='auto', interpolation='nearest',
                         extent=(-.5, 80.5, 4.5, -.5), vmin=0, vmax=upper, cmap=cmap)
    depth.set_yticks(np.arange(5)); depth.set_yticklabels([evidence.ids[i] + '区' for i in top5], fontsize=9.5)
    depth.set_ylabel('区域')
    ticks = [0, 32, 62, 80]
    dates = [str(case['dates'][day]).split('-') for day in ticks]
    depth.set_xticks(ticks); depth.set_xticklabels([f'{int(parts[1])}/{int(parts[2])}' for parts in dates], fontsize=9.5)
    depth.set_xlabel('灌溉日期')
    depth.set_title('(b) 较大区域逐日灌水深度（mm）', pad=7)
    color_axis = fig.add_axes([.72, .40, .008, .18])
    bar = fig.colorbar(image, cax=color_axis); bar.ax.tick_params(labelsize=9.5)
    representative_y = case['Y'][top5] * 100
    yield_top.barh(np.arange(5), representative_y, height=.60, color=BLUE, zorder=3)
    yield_top.axvline(90, color='#777777', lw=.9, ls='--')
    yield_top.set(xlim=(0, 105), ylim=(4.5, -.5), xlabel='相对产量（%）')
    yield_top.set_xticks([0, 50, 100]); yield_top.set_yticks(np.arange(5)); yield_top.set_yticklabels([])
    yield_top.set_title('(c) 终点相对产量', pad=7); yield_top.grid(axis='x', alpha=.17); yield_top.set_axisbelow(True)
    worst.set(xlim=(-.12, 1.12), ylim=(0, 105), ylabel='最低区域均值（%）')
    worst.set_xticks([0, 1]); worst.set_xticklabels(['训练前', '训练后'])
    worst.set_title('(d) 最低区域平均相对产量', pad=7)
    for ax in (top, worst):
        ax.grid(axis='y', alpha=.17)
    handles = [Line2D([], [], color=GREY, marker='x', linestyle='none', ms=4, label='训练前'),
               Line2D([], [], color=BLUE, marker='o', linestyle='none', ms=4, label='训练后'),
               Line2D([], [], color='#777777', ls='--', label='产量要求（90%）'),
               Patch(fc=display.ZERO_COLOR, label='零灌水')]
    handles += [Line2D([], [], color=color, marker=marker, ms=4, label=f'第{order_seed}次训练')
                for order_seed, (color, marker) in enumerate(zip(SEED_COLORS, SEED_MARKERS), 1)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.55, .012), ncol=4, frameon=False,
               handletextpad=.35, columnspacing=.8, handlelength=1.3)
    folder = OUT / FIGURES['regions']; folder.mkdir(parents=True, exist_ok=True)
    rows_csv(folder / 'all89_initial_final_regional_means.csv', rows)
    rows_csv(folder / 'fixed_largest5_regional_means.csv', [row for row in rows if row['fixed_top5']])
    rows_csv(folder / 'each_policy_worst_region.csv', [row for row in rows if row['is_worst_weather_mean_region']])
    rows_csv(folder / 'representative_largest5_daily_I_mm.csv', (
        dict(day=day, date=str(case['dates'][day]), region_id=evidence.ids[region],
             area_mu=float(evidence.area[region] * 15), actual_I_mm=float(daily_I[row, day]),
             actual_a_m3=float(case['a_m3'][day, region]))
        for row, region in enumerate(top5) for day in range(81)))
    rows_csv(folder / 'representative_largest5_terminal_Y.csv', (
        dict(region_id=evidence.ids[region], area_mu=float(evidence.area[region] * 15),
             final_Y=float(case['Y'][region]), training_seed=case['training_seed'], weather_seed=case['weather_seed'])
        for region in top5))
    return save(fig, 'regions', plt, evidence, dict(all89_regions_retained=True,
        top5_original_ids=[evidence.ids[i] for i in top5], top5_selection='largest fixed physical area; tie originalID',
        minimum_definition='for each policy: min across89 of128-weather regional meanY',
        panel_b='same representative case, full81dailyI_mm; no accumulation, date screening or overlaid lines',
        panel_c='same5regions and same representative case terminalY; not a population mean',
        panels_a_d='population weather means; all89regions and all3independent policies retained',
        largest5_daily_zero_exact=True, daily_scale_shared_across_all5x81=True,
        displayed_largest5_daily_I_mm_max=float(daily_I.max()), daily_depth_colour_upper_mm=upper,
        grouping_uses_real_areas_not_map_hex_counts=True))


def resources(evidence, plt):
    from matplotlib.lines import Line2D
    final = {budget: evidence.select('final', budget) for budget in evidence.levels}
    fixed = {budget: evidence.main if budget == 4000000 else evidence.select('fixed_budget', budget)
             for budget in evidence.levels}
    fig = plt.figure(figsize=(WIDTH, 100 / 25.4))
    axes = [fig.add_axes([.14, .30, .35, .58]), fig.add_axes([.62, .30, .35, .58])]
    raw_rows = []
    for mode, groups, color, marker, line in [('trained', final, BLUE, 's', '-'), ('fixed', fixed, ORANGE, 'o', '--')]:
        for column, metric in enumerate(('C90', 'JA')):
            x, means = [], []
            for budget in evidence.levels:
                records = groups[budget]; summary = aggregate(records, metric); raw = summary['by_policy'] * 100
                x.append(budget / 10000); means.append(float(summary['mean'] * 100))
                axes[column].scatter(np.repeat(budget / 10000, len(raw)), raw, color=color, s=13, alpha=.55, zorder=3)
                axes[column].errorbar(budget / 10000, summary['mean'] * 100, yerr=summary['sd'] * 100,
                    color=color, marker=marker, ms=4.5, capsize=3, lw=1., mfc='white' if mode == 'fixed' else color, zorder=4)
            axes[column].plot(x, means, color=color, ls=line, lw=1., zorder=2)
        for budget, records in groups.items():
            for record in records:
                raw_rows.append(dict(execution=mode, budget_m3=budget, training_seed=record.seed,
                    trained_budget_m3=record.info['trained_budget_m3'],
                    **{key: float(record.metrics[key].mean()) for key in ('C90', 'JA', 'J', 'CA90', 'used_water_m3')}))
    for ax, metric, title in zip(axes, ('C90', 'JA'), ('(a) 供水条件与产量达标率', '(b) 供水条件与面积加权产量')):
        ax.set(xlabel='全期供水量（万m³）', ylabel='区域产量达标率（%）' if metric == 'C90' else '面积加权相对产量（%）')
        ax.set_xticks(np.asarray(evidence.levels) / 10000); ax.set_title(title, pad=7); ax.grid(axis='y', alpha=.17)
    historical_rows, historical_summaries = [], []
    for year in (2022, 2024):
        records = sorted([record for record in evidence.records if record.role == 'historical_final'
                          and int(record.info['year']) == year], key=lambda record: record.seed)
        if [record.seed for record in records] != list(SEEDS):
            raise ValueError('Historical cases must retain all three actual main-policy endpoints')
        for record in records:
            evidence.paired(records[0], record)
            historical_rows.append(dict(year=year, training_seed=record.seed,
                **{metric: float(record.metrics[metric].mean()) for metric in ('C90', 'JA', 'J', 'CA90', 'used_water_m3')}))
        summary_row = dict(year=year, independent_training_count=len(records), historical_weather_paths_per_training=1)
        for metric in ('C90', 'JA', 'J', 'CA90', 'used_water_m3'):
            summary = aggregate(records, metric)
            summary_row[metric + '_mean'] = float(summary['mean'])
            summary_row[metric + '_sample_SD3'] = float(summary['sd'])
        historical_summaries.append(summary_row)
    fig.legend(handles=[Line2D([], [], color=BLUE, marker='s', lw=1, label='对应预算独立训练'),
        Line2D([], [], color=ORANGE, marker='o', mfc='white', ls='--', lw=1, label='固定400万策略'),
        Line2D([], [], color=GREY, marker='o', ls='none', ms=3, label='单次独立训练结果'),
        Line2D([], [], color='black', marker='s', ls='none', ms=4, label='均值±标准差')],
        ncol=2, frameon=False, loc='lower center', bbox_to_anchor=(.55, .012), columnspacing=.9, handletextpad=.4)
    folder = OUT / FIGURES['resources']; folder.mkdir(parents=True, exist_ok=True)
    rows_csv(folder / 'three_budget_policy_means.csv', raw_rows)
    rows_csv(folder / 'historical_fixed_main_policy_means.csv', historical_rows)
    rows_csv(folder / 'historical_fixed_main_summary.csv', historical_summaries)
    return save(fig, 'resources', plt, evidence, dict(no_secondary_budget_initial_results_invented=True,
        separately_trained_vs_fixed_main_explicit=True, sample_SD_across3_training_policies=True,
        x_coordinates_exact_budget_no_jitter=True, historical_years=[2022, 2024], no_rule_comparison=True,
        panel_count=2, historical_data='all2years/3seeds retained as fullCSV plus mean/sampleSD3; nearceiling endpoint panels omitted for information density, not data removed'))


def supporting_tables(evidence, case):
    folder = ROOT / 'reports/plot_data'; folder.mkdir(parents=True, exist_ok=True)
    main = []
    for budget in evidence.levels:
        phases = [('final', evidence.select('final', budget))] + ([('initial', evidence.initial)] if budget == 4000000 else [])
        for phase, records in phases:
            for record in records:
                regional_mean = record.arrays['Y'].mean(axis=0)
                top = evidence.top5
                main.append(dict(budget_m3=budget, phase=phase, training_seed=record.seed,
                    **{key: float(record.metrics[key].mean()) for key in ('C90', 'JA', 'J', 'CA90', 'used_water_m3')},
                    minimum_regional_weather_mean_Y=float(regional_mean.min()),
                    top5_area_weighted_weather_mean_Y=float(np.sum(evidence.area[top] * regional_mean[top]) / evidence.area[top].sum())))
    rows_csv(folder / 'main_outcomes_per_policy.csv', main)
    history = []
    for record in evidence.records:
        if record.role == 'historical_final':
            history.append(dict(year=int(record.info['year']), training_seed=record.seed,
                **{key: float(record.metrics[key].mean()) for key in ('C90', 'JA', 'J', 'CA90', 'used_water_m3')}))
    if history:
        rows_csv(folder / 'historical_fixed_policy_auxiliary.csv', history)
    # Day0 deficits are already public; this descriptive relation is not a causal test.
    deficit = public_deficit(case['B_mm'], evidence.fixed)
    rows_csv(folder / 'representative_key_days_public_deficit_and_allocation.csv', (
        dict(day=day, date=str(case['dates'][day]), region_id=region, area_ha=float(evidence.area[i]),
             beginning_public_deficit_mm=float(deficit[day, i]), actual_a_m3=float(case['a_m3'][day, i]),
             actual_I_mm=float(case['a_m3'][day, i] / (10 * evidence.area[i])),
             next_day_public_deficit_mm=float(deficit[day + 1, i]))
        for day in MAP_DAYS for i, region in enumerate(evidence.ids)))
    rows_csv(folder / 'cross_day_budget.csv', (dict(day=day, date=str(case['dates'][day]),
        beginning_balance_m3=float(case['balance_m3'][day]), actual_q_m3=float(case['q_m3'][day]),
        ending_balance_m3=float(case['balance_m3'][day + 1])) for day in (62, 63)))
    # All89 remain in the cross-day table; large/deficit examples can be identified without hiding other rows.
    rows_csv(folder / 'cross_day_all89.csv', (dict(region_id=region, area_mu=float(evidence.area[i] * 15),
        day62_B_mm=float(case['B_mm'][62, i]), day62_deficit_mm=float(deficit[62, i]), day62_a_m3=float(case['a_m3'][62, i]),
        day63_B_mm=float(case['B_mm'][63, i]), day63_deficit_mm=float(deficit[63, i]), day63_a_m3=float(case['a_m3'][63, i]),
        day64_B_mm=float(case['B_mm'][64, i]), day64_deficit_mm=float(deficit[64, i]), fixed_top5=bool(i in evidence.top5))
        for i, region in enumerate(evidence.ids)))
    selected = select_cross_day_examples(case, evidence.area, evidence.fixed)
    dump(folder / 'cross_day_objective_example_selection.json', selected)
    example_csv = folder / 'cross_day_objective_examples.csv'
    if selected['selected']:
        rows_csv(example_csv, selected['selected'])
    elif example_csv.exists():
        # This is only a derived mutable table in the new output scope, never
        # a scientific source file. Do not leave stale examples for a zero-change case.
        example_csv.unlink()
    dump(folder / 'representative_selection.json', {key: value for key, value in evidence.representative.items() if key not in ('record', 'row')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True, type=Path, help='New accepted and independently verified data index')
    parser.add_argument('--groups', default=','.join(FIGURES), help='Comma-separated exact groups; rerender only what changed')
    args = parser.parse_args(); groups = args.groups.split(',')
    if any(group not in FIGURES for group in groups):
        parser.error('Unknown requested group')
    evidence = Evidence(args.index); case = evidence.case(); plt = setup()
    process.FIGURES = FIGURES; process.SAVE = save
    assets = static_assets(groups)
    if any(key in groups for key in ('overview', 'maps', 'learning', 'regions', 'resources', 'feedback')):
        supporting_tables(evidence, case)
    functions = {'overview': lambda: process.overview(evidence, case, plt),
                 'maps': lambda: process.maps(evidence, case, plt),
                 'learning': lambda: learning(evidence, plt),
                 'regions': lambda: regional_guarantee(evidence, plt),
                 'resources': lambda: resources(evidence, plt),
                 'feedback': lambda: process.feedback(evidence, case, plt)}
    for key in groups:
        if key in functions:
            assets.append(functions[key]())
    dump(OUT / 'reasonable89_last_render_manifest.json', dict(status='RENDERED_NOT_PUBLISHED_NEEDS_ACTUAL_REVIEW',
        groups=groups, assets=assets, input_files_sha256=evidence.bindings,
        source_files_sha256={p.name: sha(p) for p in SOURCE.glob('*.py')},
        representative={key: value for key, value in evidence.representative.items() if key not in ('record', 'row')},
        new_environment_steps=0, new_model_inferences=0, old_result_arrays_used=False, root_tex_modified=False))


if __name__ == '__main__':
    main()
