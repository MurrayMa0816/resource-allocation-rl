"""Standalone random-object comparison; only completed numerical data is read."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import warnings

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np

RANDOM = 'A_RANDOM_OBJECT_ALLOCATION'
ARMS = ('A_FULL', 'A_PACE', RANDOM, 'A_DIRECT_TOTAL')
DOMAINS = ('radio', 'v5')
LABELS = dict(A_FULL='ReJoP（本文）', A_PACE='去总量学习（余额均分）',
              A_RANDOM_OBJECT_ALLOCATION='随机对象分配', A_DIRECT_TOTAL='去动态参照残差')
STYLES = dict(
    A_FULL=dict(color='#0072B2', linestyle='-', linewidth=1.4),
    A_PACE=dict(color='#A85A00', linestyle='--', linewidth=1.1),
    A_RANDOM_OBJECT_ALLOCATION=dict(color='#7A3E87', linestyle='-.', linewidth=1.1),
    A_DIRECT_TOTAL=dict(color='#444444', linestyle=':', linewidth=1.3))
FONTS = (Path('/root/.local/share/fonts/microsoft/simsun.ttc'),
         Path('/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'))
SIZE = 10.5
PARENT = Path('/hy-tmp/resource_allocation_results/chapter03/figures/figure35_three_ablations_preview_20261109_191528/execution/plot.py')
PARENT_SHA = 'cb7733116caeb188891604173dae092c44bcd2c894f079625d1cd18f98a085a7'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def rows(p):
    with Path(p).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(p, data):
    with Path(p).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def load(directory):
    directory = Path(directory).resolve(strict=True)
    receipt = read(directory / 'completion_receipt.json')
    if (receipt['status'] != 'PASS' or receipt.get('schema')
            != 'chapter3-random-object-ablation-analysis-completion-v1'):
        raise ValueError('Completed analysis required')
    sources = {str(directory / 'completion_receipt.json'): sha(directory / 'completion_receipt.json')}
    for name, digest in receipt['output_sha256'].items():
        if sha(directory / name) != digest:
            raise ValueError('Changed completed analysis: ' + name)
        sources[str(directory / name)] = digest
    report = read(directory/'summary.json')
    if (report.get('schema') != 'chapter3-random-object-ablation-analysis-v1'
            or report.get('status') != 'PASS' or report.get('methods') != list(ARMS)
            or report.get('smoothing') is not False or report.get('nodes') != 198
            or report.get('new_training_models') != 10 or report.get('reused_training_models') != 30
            or report.get('no_new_TEST') is not True):
        raise ValueError('Expected complete five-seed random-object DEV analysis')
    raw = rows(directory / 'curves.csv')
    grid = np.r_[np.arange(0, 100608, 512), 100608].astype(int)
    groups = {}
    for x in raw:
        key = x['domain'], x['arm'], int(x['seed'])
        groups.setdefault(key, []).append((int(x['training_step']), float(x['score'])))
    expected = {(d, a, s) for d in DOMAINS for a in ARMS for s in range(5)}
    if set(groups) != expected or len(raw) != 7920:
        raise ValueError('Precisely forty complete learned curves required')
    matrices = {}
    for d in DOMAINS:
        for a in ARMS:
            data = []
            for s in range(5):
                x = sorted(groups[d, a, s])
                if [t for t, _ in x] != grid.tolist():
                    raise ValueError('DEV grid missing or duplicated')
                data.append([v for _, v in x])
            matrix = np.asarray(data)
            if matrix.shape != (5, 198) or not np.isfinite(matrix).all():
                raise ValueError('Expected five finite 198-point curves')
            matrices[d, a] = matrix
            stated=report['settings'][d][a]
            for metric, values in [('initial',matrix[:,0]),('DEV_fixed_endpoint',matrix[:,-1]),
                ('normalized_AUC',np.trapezoid(matrix,grid,axis=1)/100608)]:
                np.testing.assert_allclose(values,stated[metric]['by_training_seed'],rtol=0,atol=2e-11)
                np.testing.assert_allclose([values.mean(),values.std(ddof=1)],
                    [stated[metric]['mean'],stated[metric]['training_seed_sample_SD']],rtol=0,atol=2e-11)
    return grid, matrices, raw, sources


def fonts():
    import matplotlib as mpl
    from matplotlib import font_manager
    for p in FONTS:
        font_manager.fontManager.addfont(str(p))
    mpl.rcParams.update({'font.family': ['Times New Roman', 'SimSun'], 'font.size': SIZE,
        'axes.titlesize': SIZE, 'axes.labelsize': SIZE, 'xtick.labelsize': SIZE,
        'ytick.labelsize': SIZE, 'legend.fontsize': SIZE, 'figure.titlesize': SIZE,
        'axes.unicode_minus': True, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
        'axes.linewidth': .65, 'xtick.major.width': .65, 'ytick.major.width': .65,
        'xtick.direction': 'out', 'ytick.direction': 'out', 'mathtext.fontset': 'stix'})
    return {str(p): sha(p) for p in FONTS}


def plot(out, name, arms, grid, matrices):
    import matplotlib.pyplot as plt
    width, height = 156., 98.
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    bottom = height - 65.
    axes, title_artists, panels = [], [], []
    for d, left, title in [('radio', 15., '(a) 通信功率分配'),
                            ('v5', 92., '(b) 应急柴油分配')]:
        ax = fig.add_axes([left / width, bottom / height, 57. / width, 53. / height])
        axes.append(ax)
        series = []
        low, high = math.inf, -math.inf
        for a in arms:
            matrix = matrices[d, a]
            mean, sd = matrix.mean(0), matrix.std(0, ddof=1)
            ax.fill_between(grid / 1000, mean-sd, mean+sd, color=STYLES[a]['color'],
                            alpha=.10, edgecolor='none', linewidth=0)
            line, = ax.plot(grid / 1000, mean, label=LABELS[a], **STYLES[a])
            np.testing.assert_array_equal(line.get_ydata(), mean)
            np.testing.assert_array_equal(line.get_xdata(), grid / 1000)
            low, high = min(low, (mean-sd).min()), max(high, (mean+sd).max())
            series.append(dict(arm=a, label=LABELS[a], x_training_steps=grid.tolist(),
                plotted_x=line.get_xdata().tolist(), mean=line.get_ydata().tolist(),
                sample_SD=sd.tolist(), lower=(mean-sd).tolist(), upper=(mean+sd).tolist(),
                style=STYLES[a]))
        ax.set_xlim(0, 100.608)
        ax.set_xticks([0, 25, 50, 75, 100.608], ['0', '25', '50', '75', '100.61'])
        # Retain the parent axis range unless complete SD bands require expansion.
        base_low,base_high=(15.,50.) if d=='radio' else (50.,85.)
        lower=min(base_low,5*math.floor((low-.5)/5))
        upper=max(base_high,5*math.ceil((high+.5)/5))
        ax.set_ylim(lower,upper)
        tick0=10*math.ceil(lower/10)
        ax.set_yticks(np.arange(tick0,upper+.01,10))
        if low < ax.get_ylim()[0] or high > ax.get_ylim()[1]:
            raise ValueError('Mean or sample SD would be clipped')
        ax.set_xlabel('训练步数（×10³）', labelpad=4)
        ax.set_ylabel('平均图像质量得分（分）' if d == 'radio' else '平均需求完成比例（%）', labelpad=4)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', linewidth=.45, color='#DDDDDD')
        ax.set_axisbelow(True)
        title_artist = ax.set_title(title, pad=8, loc='center')
        title_artists.append(title_artist)
        panels.append(dict(domain=d, title=title, axis_mm=[left, bottom, 57., 53.],
            xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()), series=series))
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.53, 2. / height),
        ncol=2, frameon=False, handlelength=2.3, handletextpad=.55,
        columnspacing=1.0, labelspacing=.5, borderaxespad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    text_boxes = []
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():
            continue
        box = t.get_window_extent(renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        if (box.x0 < -.1 or box.y0 < -.1 or box.x1 > figure_box.x1 + .1
                or box.y1 > figure_box.y1 + .1):
            raise ValueError('Text outside native figure: ' + t.get_text())
        if not np.isclose(t.get_fontsize(), SIZE):
            raise ValueError('Nonstandard text size: ' + t.get_text())
        text_boxes.append(dict(text=t.get_text(), fontsize_bp=t.get_fontsize(),
            bbox_mm=[box.x0/fig.dpi*25.4, box.y0/fig.dpi*25.4,
                     box.x1/fig.dpi*25.4, box.y1/fig.dpi*25.4]))
    for ax, t in zip(axes, title_artists):
        box, abox = t.get_window_extent(renderer), ax.get_window_extent(renderer)
        if abs((box.x0+box.x1)/2 - (abox.x0+abox.x1)/2) > .05:
            raise ValueError('Subfigure title is not centered on its own axes')
    lb = legend.get_window_extent(renderer)
    for ax in axes:
        xb = ax.xaxis.label.get_window_extent(renderer)
        if lb.y1 + .5 > xb.y0:
            raise ValueError('Legend collides with axis labels')
    paths = []
    for suffix in ('pdf', 'svg', 'png'):
        p = out / (name + '.' + suffix)
        fig.savefig(p, dpi=600 if suffix == 'png' else 300)
        paths.append(p)
    plt.close(fig)
    return dict(name=name, size_mm=[width, height], ordinary_font_bp=SIZE,
        panels=panels, text_boxes=text_boxes, artifacts={str(p): sha(p) for p in paths})


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    out = Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out / 'artists.json').exists():
        raise ValueError('Completed graphics may not be overwritten')
    if sha(PARENT)!=PARENT_SHA:
        raise ValueError('Reviewed layout parent changed')
    grid, matrices, raw, sources = load(args.analysis)
    font_identity = fonts()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        plots = [plot(out, 'figure35_random_object_preview', ARMS, grid, matrices)]
    if caught:
        write(out/'render_warnings.json', [str(x.message) for x in caught])
        raise ValueError('Figure rendering warnings require review')
    write(out/'artists.json', dict(status='GENERATED_NEEDS_INDEPENDENT_AND_VISUAL_REVIEW',
        scientific_identity='retrained_random_object_DEV_learning_curves', source_sha256=sources,
        reviewed_layout_parent=dict(path=str(PARENT),sha256=PARENT_SHA),
        font_sha256=font_identity, smoothing=False, interpolation=False,
        original_zero_step_behavior_preserved=True, plots=plots,
        object_execution='Fixed independent U(0,1) object weights remain sampled during DEV; eta only uses its mean',
        new_environment_steps=0, model_loads=0, new_training_steps=0))
    print(out/'figure35_random_object_preview.pdf')


if __name__ == '__main__':
    main()
