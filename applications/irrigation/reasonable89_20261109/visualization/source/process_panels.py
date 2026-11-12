"""Frozen-style display functions for NEW reasonable89 data only.

Copied pure drawing structure, never an earlier evaluation result.
"""
from __future__ import annotations
from datetime import date,timedelta
import json
from pathlib import Path
import numpy as np
from data_adapter import ROOT,MAP_DAYS,SEEDS as TRAINING_SEEDS,rows_csv as csv_rows,dump,sha,common_volume_edges as map_edges
import display_helpers as display
SOURCE=Path(__file__).resolve().parent
OUT=ROOT/'figures'
BLUE,ORANGE,GREEN,GREY='#236d9e','#cf7627','#4a8269','#999999'
WIDTH=156/25.4
STAGES=[('返青',0,9),('分蘖',9,46),('孕穗',46,62),('抽穗',62,72),('乳熟',72,81)]
FIGURES={}
SAVE=None
def save(*args,**kwargs):
    if SAVE is None:raise RuntimeError('Configure the new-run save callback first')
    return SAVE(*args,**kwargs)


def date_ticks(ax, labels=True):
    positions = (0, 17, 32, 46, 62, 72, 81)
    ax.set_xlim(-.8, 81.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([(date(2022, 5, 30) + timedelta(days=day)).strftime('%-m/%-d') for day in positions])
    if not labels:
        ax.tick_params(labelbottom=False)
    for _, start, _ in STAGES[1:]:
        ax.axvline(start, color='#b1b1b1', ls=':', lw=.6, zorder=0)
    ax.grid(axis='y', alpha=.17)

def overview(campaign, case, plt):
    fig = plt.figure(figsize=(WIDTH, 80 / 25.4))
    top = fig.add_axes([.125, .5375, .85, .3125])
    bottom = fig.add_axes([.125, .1625, .85, .25])
    fig.text(.55, .95625, '(a) 每日配置总量', ha='center', va='center', fontsize=10.5)
    fig.text(.55, .475, '(b) 日初供水余额', ha='center', va='center', fontsize=10.5)
    q, b = case['q_m3'] / 10000, case['balance_m3'] / 10000
    top.plot(np.arange(81), q, color=BLUE)
    bottom.plot(np.arange(82), b, color=BLUE)
    for ax, values in ((top, q), (bottom, b)):
        ax.scatter(MAP_DAYS, values[list(MAP_DAYS)], color=ORANGE, s=17, zorder=5, edgecolors='white', linewidths=.35)
        ax.axvspan(61.6, 63.4, color=ORANGE, alpha=.12, zorder=0)
        date_ticks(ax, labels=ax is bottom)
        ax.tick_params(labelsize=9.5)
    top.set(ylabel='当日总量（万m³）', ylim=(0, max(q) * 1.3))
    from data_adapter import nice_ceiling
    step=nice_ceiling(top.get_ylim()[1]/4)
    top.set_yticks(np.arange(0,top.get_ylim()[1]+1e-12,step))
    bottom.set(ylabel='日初余额（万m³）', xlabel='灌溉日期', ylim=(-.02 * b[0], 1.08 * b[0]))
    step=nice_ceiling(bottom.get_ylim()[1]/3)
    bottom.set_yticks(np.arange(0,bottom.get_ylim()[1]+1e-12,step))
    for name, start, stop in STAGES:
        top.text((start + stop) / 2, 1.035, name, ha='center', va='bottom', transform=top.get_xaxis_transform(), fontsize=9.5)
    # Date annotations use fixed x positions, independent of whether a date is a peak.
    for day, text, target in ((0, '5/30', 8), (46, '7/15', 40), (62, '7/31—8/1', 61), (72, '8/10', 75)):
        height = .80 if day in (0, 46) else .94 if day == 62 else .56
        top.annotate(text, xy=(day, q[day]), xytext=(target, height), textcoords=top.get_xaxis_transform(),
                     ha='center', va='center', fontsize=9.5,
                     arrowprops=dict(arrowstyle='-', color='#646464', lw=.65, shrinkA=3, shrinkB=4))
    folder = OUT / FIGURES['overview']
    folder.mkdir(parents=True, exist_ok=True)
    csv_rows(folder / 'daily_total_and_balance_m3.csv', (dict(day=day, date=str(case['dates'][day]),
        beginning_balance_m3=float(case['balance_m3'][day]), allocation_m3=float(case['q_m3'][day]),
        ending_balance_m3=float(case['balance_m3'][day + 1])) for day in range(81)))
    return save(fig, 'overview', plt, campaign, {'selected_days': list(MAP_DAYS)})

def maps(campaign, case, plt):
    from matplotlib.patches import Rectangle
    geometry = {path.stem: json.loads(path.read_text()) for path in (SOURCE / 'geometry').glob('*.json')}
    style = json.loads((SOURCE / 'channel_style.json').read_text())
    if [region['id'] for region in geometry['layout']['regions']] != case['ids']:
        raise ValueError('Map geometry does not have the new unchanged89 region identities')
    edges = map_edges(case['a_m3'])
    display.BINS = edges
    # Show kilocubic metres if labels in m3 would exceed the approved legend width.
    unit, divisor = '千m³', 1000.
    folder = OUT / FIGURES['maps']
    folder.mkdir(parents=True, exist_ok=True)
    frame = geometry['layout']['board_rectangle']
    ratio = (frame['y_max'] - frame['y_min']) / (frame['x_max'] - frame['x_min'])
    width, board, margin, top = 119., 118., .5, .2
    height = board * ratio + 2 * margin + top
    assets = []
    for day in MAP_DAYS:
        fig = plt.figure(figsize=(width / 25.4, height / 25.4))
        ax = fig.add_axes([margin / width, margin / height, board / width, board * ratio / height])
        display.draw_map(ax, case['a_m3'][day], geometry)
        texts = [text for text in ax.texts if '渠' in text.get_text()]
        if len(texts) != 6:
            raise AssertionError('All six approved channel labels must remain')
        for text, label in zip(texts, style):
            text.set_position(label['position'])
            text.set_fontsize(label['font_size_pt'])
        assets.append(save(fig, 'maps', plt, campaign, {'day': day, 'date': str(case['dates'][day]),
            'actual_unit': 'm3', 'board_width_mm': board,
            'fixed_region_id_font_pt': 9.5, 'channel_style': style,
            'geometry_sha256': {path.name: sha(path) for path in (SOURCE / 'geometry').glob('*.json')}}, f'day{day:02}'))
    legend_h = 11.8
    fig = plt.figure(figsize=(width / 25.4, legend_h / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, legend_h)); ax.axis('off')
    ax.text(width / 2, 9.1, f'区域当日配水量（{unit}）', ha='center', va='center', fontsize=10.5)
    zero_x, zero_w, gap, bottom, swatch_h = .5, 8.6, 1.8, 4.35, 2.2
    positive_x = zero_x + zero_w + gap
    positive_w = width - .5 - positive_x
    ax.add_patch(Rectangle((zero_x, bottom), zero_w, swatch_h, fc=display.ZERO_COLOR, ec='#80786b', lw=.35))
    ax.text(zero_x + zero_w / 2, 1.85, '0', ha='center', va='center', fontsize=9.5)
    labels = []
    for index, color in enumerate(display.POSITIVE_COLORS):
        w = positive_w / 6
        ax.add_patch(Rectangle((positive_x + index * w, bottom), w, swatch_h, fc=color, ec=color, lw=.15))
        label = f'({edges[index] / divisor:g},{edges[index + 1] / divisor:g}]'
        labels.append(label)
        ax.text(positive_x + (index + .5) * w, 1.85, label, ha='center', va='center', fontsize=9.5)
    ax.add_patch(Rectangle((positive_x, bottom), positive_w, swatch_h, fc='none', ec='#80786b', lw=.35))
    legend = save(fig, 'maps', plt, campaign, {'color_unit': unit, 'positive_edges_m3': edges.tolist()}, 'legend_horizontal')
    csv_rows(folder / 'all89_selected_days_m3.csv', (dict(day=day, date=str(case['dates'][day]), region_id=region,
        allocation_m3=float(case['a_m3'][day, index])) for day in MAP_DAYS for index, region in enumerate(case['ids'])))
    csv_rows(folder / 'selected_dates_budget_m3.csv', (dict(day=day, date=str(case['dates'][day]),
        beginning_balance_m3=float(case['balance_m3'][day]), allocation_m3=float(case['q_m3'][day]),
        ending_balance_m3=float(case['balance_m3'][day + 1])) for day in MAP_DAYS))
    dump(folder / 'colour_scale.json', dict(unit=unit, divisor=divisor, positive_edges_m3=edges.tolist(),
        zero_exact=True, zero_color=display.ZERO_COLOR, positive_colors=list(display.POSITIVE_COLORS),
        labels=labels, positive_intervals='left-open/right-closed', source_scope='entire representative81x89 allocation array'))
    first = folder / 'figure.pdf'
    if not first.exists():
        first.symlink_to('day00.pdf')
    return dict(maps=assets, legend=legend)

def feedback_pairs(campaign, multiplier, include_descriptive=False):
    allowed = {'feedback', 'representative_feedback'} if include_descriptive else {'feedback'}
    records = [record for record in campaign.records if record.role in allowed
               and float(record.info['et0_multiplier']) == multiplier]
    pairs = []
    for seed in TRAINING_SEEDS:
        selected = [record for record in records if record.seed == seed]
        for role in ('feedback', 'representative_feedback'):
            if role not in allowed:
                continue
            by_mode = {record.info['execution_mode']: record for record in selected if record.role == role}
            if not by_mode:
                continue
            if set(by_mode) != {'new_observation_feedback', 'original_actions_replay'}:
                raise ValueError('Incomplete new-weather feedback/replay pair')
            first, second = by_mode['new_observation_feedback'], by_mode['original_actions_replay']
            campaign.assert_paired(first, second)
            if role == 'feedback' and not np.array_equal(first.arrays['weather_seed'], campaign.test_weather[:32]):
                raise ValueError('Formal feedback sample is not the predeclared first32 TEST weather paths')
            pairs.append((first, second))
    if not include_descriptive and len(pairs) != 3:
        raise ValueError('Each feedback direction needs three independent main policies')
    return pairs

def window_storage(record, row, days):
    if 'B_mm' in record.arrays:
        return record.arrays['B_mm'][row, days]
    if 'window_B_mm' not in record.arrays:
        raise ValueError('Feedback data must retain genuine field storage at plotting boundaries')
    boundaries = record.arrays.get('window_boundaries', record.info.get('window_boundaries'))
    if boundaries is None:
        raise ValueError('Feedback window boundaries must be explicit')
    boundaries = list(np.asarray(boundaries).astype(int))
    return record.arrays['window_B_mm'][row, [boundaries.index(int(day)) for day in days]]

def feedback(campaign, case, plt):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 182 / 25.4))
    days = np.arange(61, 68)
    daily_rows, pair_rows, series = [], [], []
    for column, multiplier in enumerate((.85, 1.15)):
        candidates = []
        for first, second in feedback_pairs(campaign, multiplier, include_descriptive=True):
            if first.seed == case['trained_seed'] and int(case['weather_seed']) in first.arrays['weather_seed']:
                candidates.append((first, second))
        if len(candidates) != 1:
            raise ValueError('Representative feedback must match the same declared main seed and weather exactly once')
        first, second = candidates[0]
        row, other_row = first.weather_row(case['weather_seed']), second.weather_row(case['weather_seed'])
        q = first.arrays['q_m3'][row, days] - second.arrays['q_m3'][other_row, days]
        storage = 10 * np.sum(campaign.area[None, :] *
            (window_storage(first, row, days) - window_storage(second, other_row, days)), axis=1)
        series.append((q, storage))
        axes[0, column].plot(days, q, marker='o', ms=3, color=BLUE)
        axes[1, column].plot(days, storage, marker='o', ms=3, color=BLUE)
        for day, qdelta, bdelta in zip(days, q, storage):
            daily_rows.append(dict(et0_multiplier=multiplier, trained_seed=case['trained_seed'],
                weather_seed=case['weather_seed'], day=int(day), date=str(case['dates'][day]),
                q_change_m3=float(qdelta), beginning_storage_change_m3=float(bdelta)))
        for first, second in feedback_pairs(campaign, multiplier):
            for row in range(32):
                pair_rows.append(dict(et0_multiplier=multiplier, trained_seed=first.seed,
                    weather_seed=int(first.arrays['weather_seed'][row]),
                    C90_change_pp=float((first.metrics['C90'][row] - second.metrics['C90'][row]) * 100),
                    JA_change_pp=float((first.metrics['JA'][row] - second.metrics['JA'][row]) * 100),
                    J_change_pp=float((first.metrics['J'][row] - second.metrics['J'][row]) * 100)))
        points = [item for item in pair_rows if item['et0_multiplier'] == multiplier]
        if len(points) != 96:
            raise ValueError('There must be96 formal feedback pairs per direction')
        axes[2, column].scatter([item['C90_change_pp'] for item in points],
                                [item['JA_change_pp'] for item in points], s=10, alpha=.6, color=BLUE)
    qmax = max(float(abs(q).max()) for q, _ in series)
    bmax = max(float(abs(storage).max()) for _, storage in series)
    cmax = max(abs(item['C90_change_pp']) for item in pair_rows)
    jmax = max(abs(item['JA_change_pp']) for item in pair_rows)
    for column, name in enumerate(('较低需水', '较高需水')):
        for row in (0, 1):
            axes[row, column].axhspan(0, 0, color='none')
            axes[row, column].axvspan(62, 64, color='#d5d8d9', alpha=.4, zorder=0)
            axes[row, column].axvline(63, color=GREY, ls='--', lw=.8)
            axes[row, column].set_xticks([61, 63, 65, 67])
            axes[row, column].set_xticklabels([str(case['dates'][day])[5:] for day in (61, 63, 65, 67)])
            axes[row, column].set_xlabel('日期', labelpad=3)
        axes[0, column].set(ylabel='日配水增减量（m³）', ylim=(-max(1., qmax * 1.10), max(1., qmax * 1.10)))
        axes[1, column].set(ylabel='储水增减量（m³）', ylim=(-max(1., bmax * 1.10), max(1., bmax * 1.10)))
        axes[2, column].set(xlabel='产量达标率变化（百分点）', ylabel='面积加权相对产量变化（百分点）',
                            xlim=(-max(.01, cmax * 1.10), max(.01, cmax * 1.10)),
                            ylim=(-max(.001, jmax * 1.10), max(.001, jmax * 1.10)))
        for row, action in enumerate(('配水调整', '储水响应', '终点成效')):
            axes[row, column].set_title(f'({chr(97 + row * 2 + column)}) {name}情景的{action}', pad=7)
            axes[row, column].axhline(0, color='#a5a5a5', lw=.6)
            axes[row, column].grid(alpha=.15)
    fig.subplots_adjust(left=.16, right=.985, top=.95, bottom=.15, hspace=.68, wspace=.67)
    fig.legend(handles=[Line2D([], [], color=BLUE, marker='o', ms=3, label='重新配置减去原动作回放'),
                        Patch(fc='#d5d8d9', alpha=.4, label='需水变化时段')],
               loc='lower center', bbox_to_anchor=(.56, .018), frameon=False, ncol=2,
               columnspacing=.9, handletextpad=.4, handlelength=1.3)
    folder = OUT / FIGURES['feedback']; folder.mkdir(parents=True, exist_ok=True)
    csv_rows(folder / 'representative_feedback_storage.csv', daily_rows)
    csv_rows(folder / 'all192_feedback_pairs.csv', pair_rows)
    training_means = []
    for multiplier in (.85, 1.15):
        for seed in TRAINING_SEEDS:
            selected = [item for item in pair_rows if item['et0_multiplier'] == multiplier and item['trained_seed'] == seed]
            training_means.append(dict(et0_multiplier=multiplier, trained_seed=seed, pairs=len(selected),
                **{metric: float(np.mean([item[metric] for item in selected]))
                   for metric in ('C90_change_pp', 'JA_change_pp', 'J_change_pp')}))
    csv_rows(folder / 'feedback_training_means.csv', training_means)
    return save(fig, 'feedback', plt, campaign, {'paired_process_count': 192,
        'representative_extra_excluded_from_192': True, 'terminal_vertical_axis': 'JA, not unweighted J',
        'feedback_difference_order': 'new_observation_feedback minus original_actions_replay'})
