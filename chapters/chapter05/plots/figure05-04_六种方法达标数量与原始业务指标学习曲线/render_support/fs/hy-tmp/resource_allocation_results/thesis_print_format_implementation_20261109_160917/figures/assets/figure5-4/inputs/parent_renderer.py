"""Chapter 5 presentation only: read sealed CSV, write only a new HY asset.

No environment, policy, tensor, optimiser, training or publisher is imported.
The source CSV and original presentation directories are never written.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
METHODS = ('full', 'capql', 'qpensieve', 'cola', 'morld', 'pasta')
ABLATIONS = ('full', 'only_j', 'only_c', 'scalar_value')
NAMES = dict(full='SeDVal（本文）', capql='CAPQL', qpensieve='Q-Pensieve',
             cola='COLA', morld='MORL/D', pasta='PASTA', only_j='仅平均成效目标',
             only_c='仅达标目标', scalar_value='标量价值监督')
COLORS = dict(full='#154f80', capql='#dd873a', qpensieve='#319664', cola='#9368a6',
              morld='#c9494d', pasta='#498f9d', only_j='#d98b36', only_c='#429a73', scalar_value='#9768a2')
MARKERS = dict(zip(METHODS, ('o', 's', '^', 'D', 'P', 'X')))
MARKERS.update(only_j='s', only_c='^', scalar_value='D')
SEEDS = tuple(range(5))
PANELS = (('radio', 'K'), ('v5', 'K'), ('radio', 'J'), ('v5', 'J'))
LABELS = {'learning': 'fig:ch5-validation-learning', 'joint_endpoint': 'fig:ch5-validation-tradeoff',
          'ablations': 'fig:ch5-validation-ablation', 'threshold_recount': 'fig:ch5-validation-threshold',
          'budget_transfer': 'fig:ch5-validation-budget'}
NUMBERS = dict(zip(LABELS, ('5.4', '5.5', '5.6', '5.7', '5.8')))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load_csv(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for k in ('K', 'C', 'J', 'F_0_8', 'scale', 'task_threshold'):
            if k in row:
                row[k] = float(row[k])
        for k in ('seed', 'learning_steps'):
            if k in row:
                row[k] = int(row[k])
    return rows


def smooth(values):
    import numpy as np
    raw = np.asarray(values, dtype=float)
    result = raw.copy()
    for i in range(1, len(raw)-1):
        result[i] = raw[max(0, i-2):min(len(raw), i+3)].mean()
    assert result[0] == raw[0] and result[-1] == raw[-1]
    return result


def moments(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    assert values.shape[0] == 5 and np.isfinite(values).all()
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def ylabel(domain, metric):
    if metric == 'K':
        return '达标终端数' if domain == 'radio' else '达标区域数'
    return '平均图像质量（SSIM）' if domain == 'radio' else '平均需求完成比例（%）'


def scale(domain, metric):
    return 100. if domain == 'v5' and metric == 'J' else 1.


def arrays_digest(values):
    import numpy as np
    arr = np.ascontiguousarray(values, dtype='<f8')
    return {'shape': list(arr.shape), 'sha256': hashlib.sha256(arr.tobytes()).hexdigest()}


def render(role, source, destination):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager as fm
    from matplotlib.text import Text
    destination = Path(destination).resolve()
    assert destination.is_relative_to(ROOT) and destination != source.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    fonts = Path('/root/gamewar/latex_thesis_resource_allocation/assets/fonts/windows')
    for font in ('TIMES.TTF', 'simsun.ttc'):
        fm.fontManager.addfont(str(fonts/font))
    plt.rcParams.update({'font.family': ['Times New Roman', 'SimSun'], 'font.size': 10.5,
                         'axes.labelsize': 10.5, 'axes.titlesize': 10.5, 'axes.titleweight': 'normal',
                         'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5, 'legend.fontsize': 10.5,
                         'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'path', 'axes.grid': False,
                         'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    raw_rows = load_csv(source/'learning_raw.csv')
    display_rows = load_csv(source/'learning_display.csv')
    endpoints = load_csv(source/'fixed_endpoints_by_seed.csv')
    thresholds = load_csv(source/'threshold_recounts_by_seed.csv')
    series = defaultdict(list)
    display = defaultdict(list)
    for row in raw_rows:
        series[row['algorithm'], row['domain'], row['seed']].append(row)
    for row in display_rows:
        display[row['algorithm'], row['domain'], row['seed']].append(row)
    end = {(r['algorithm'], r['domain'], r['seed'], r['scale']): r for r in endpoints}
    for key in series:
        assert len(series[key]) == 198
        for metric in ('K', 'C', 'J', 'F_0_8'):
            assert np.allclose(smooth([r[metric] for r in series[key]]),
                               [r[metric] for r in display[key]], rtol=0, atol=1e-12)
    proof = {'schema': 'chapter5-presentation-alignment-proof-v1', 'role': role,
             'actual_number': NUMBERS[role], 'label': LABELS[role], 'series': [],
             'raw_ssim_not_scaled': True, 'diesel_J_display_percent_only': True,
             'scientific_execution': dict(training_runs=0, model_loads=0, environment_steps=0, policy_evaluations=0)}
    if role in ('learning', 'ablations', 'budget_transfer'):
        fig, axes2 = plt.subplots(2, 2, figsize=(15.6/2.54, 5.12))
        axes = list(axes2.flat)
        panels = PANELS
        fig.subplots_adjust(left=.125, right=.975, top=.93, bottom=.225, wspace=.53, hspace=.68)
    else:
        fig, axes2 = plt.subplots(1, 2, figsize=(15.6/2.54, 3.46))
        axes = list(axes2.flat)
        panels = (('radio', 'J'), ('v5', 'J'))
        fig.subplots_adjust(left=.125, right=.975, top=.865, bottom=.32, wspace=.55)
    methods = ABLATIONS if role == 'ablations' else METHODS
    for index, (ax, (domain, metric)) in enumerate(zip(axes, panels)):
        ax.set_title(f'({chr(97+index)}) '+('通信功率分配' if domain == 'radio' else '应急柴油分配'), pad=8)
        ax.set_ylabel(ylabel(domain, 'K' if role == 'threshold_recount' else metric), labelpad=5)
        ax.tick_params(length=3, width=.7, pad=3)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_linewidth(.7)
        for method in methods:
            entry = {'panel': chr(97+index), 'method': method}
            style = dict(color=COLORS[method], label=NAMES[method], marker=MARKERS[method],
                         markerfacecolor=COLORS[method], markeredgecolor=COLORS[method],
                         markersize=3.4 if method == 'full' else 3.1, linestyle='-',
                         linewidth=1.45 if method == 'full' else 1.1)
            if role in ('learning', 'ablations'):
                values = np.asarray([[r[metric] for r in display[method, domain, seed]] for seed in SEEDS])*scale(domain, metric)
                x = np.asarray([r['learning_steps'] for r in series[method, domain, 0]])/1000.
                mean, sd = moments(values)
                ax.plot(x, mean, markevery=list(range(0, len(x), 24))+[len(x)-1], **style)
                ax.fill_between(x, mean-sd, mean+sd, color=COLORS[method], alpha=.12, linewidth=0)
                entry.update(x=arrays_digest(x), per_seed_display=arrays_digest(values), mean=arrays_digest(mean), sd=arrays_digest(sd))
                ax.set_xlabel('训练步数（×10³）', labelpad=5)
                ax.set_xlim(0, 100.608)
                ax.set_xticks([0, 25, 50, 75, 100])
            elif role == 'joint_endpoint':
                rows = [end[method, domain, seed, 1.] for seed in SEEDS]
                k, ksd = moments([r['K'] for r in rows])
                j, jsd = moments([r['J']*scale(domain, 'J') for r in rows])
                ax.errorbar(k, j, xerr=ksd, yerr=jsd, fmt=MARKERS[method],
                            color=COLORS[method], markerfacecolor=COLORS[method], markeredgecolor=COLORS[method],
                            label=NAMES[method], markersize=5.2, capsize=2.5, elinewidth=.85, linestyle='none', zorder=3)
                ax.set_xlabel(ylabel(domain, 'K'), labelpad=5)
                entry.update(x=float(k), y=float(j), x_sd=float(ksd), y_sd=float(jsd))
            elif role == 'threshold_recount':
                x = sorted({r['task_threshold'] for r in thresholds if r['domain'] == domain})
                values = [[next(r['K'] for r in thresholds if r['algorithm']==method and r['domain']==domain
                                and r['seed']==seed and r['task_threshold']==threshold) for threshold in x] for seed in SEEDS]
                mean, sd = moments(values)
                x = np.asarray(x)*(100. if domain == 'v5' else 1.)
                ax.plot(x, mean, **style)
                ax.fill_between(x, mean-sd, mean+sd, color=COLORS[method], alpha=.10, linewidth=0)
                ax.set_xticks(x)
                ax.set_xlabel('图像质量达标要求（SSIM）' if domain == 'radio' else '需求完成比例要求（%）', labelpad=5)
                entry.update(x=arrays_digest(x), per_seed=arrays_digest(values), mean=arrays_digest(mean), sd=arrays_digest(sd))
            else:
                x = np.array([.8, 1., 1.2])
                values = [[end[method, domain, seed, float(s)][metric]*scale(domain, metric) for s in x] for seed in SEEDS]
                mean, sd = moments(values)
                ax.errorbar(x, mean, yerr=sd, capsize=2, elinewidth=.7, **style)
                ax.set_xticks(x, ['0.8', '1.0', '1.2'])
                ax.set_xlabel('发射能量相对值' if domain == 'radio' else '柴油供给相对值', labelpad=5)
                ax.set_xlim(.77, 1.23)
                entry.update(x=arrays_digest(x), per_seed=arrays_digest(values), mean=arrays_digest(mean), sd=arrays_digest(sd))
            proof['series'].append(entry)
        if role == 'threshold_recount':
            ax.axvline(.2 if domain == 'radio' else 70., linestyle='--', color='#555555', linewidth=.8, zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, ncol=2 if role=='ablations' else 3, loc='lower center',
                        bbox_to_anchor=(.5, .008), frameon=False, handlelength=1.6,
                        columnspacing=1.35, handletextpad=.5, labelspacing=.5, borderaxespad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    def box(artist):
        return list(artist.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted()).bounds)
    # Locators also create labels outside the view interval; Matplotlib does
    # not draw those ticks, so omit them from the rendered-text extent audit.
    hidden_tick_ids=set()
    for ax in axes:
        for axis,lim in ((ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())):
            for tick in axis.get_major_ticks():
                if not min(lim)-1e-9 <= tick.get_loc() <= max(lim)+1e-9:
                    hidden_tick_ids.update((id(tick.label1),id(tick.label2)))
    text_proof=[]
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text() and id(t) not in hidden_tick_ids:
            text_proof.append(dict(text=t.get_text(), fontsize=t.get_fontsize(), family=t.get_fontfamily(), weight=t.get_fontweight(), bbox_inches=box(t)))
    assert all(t['fontsize']==10.5 for t in text_proof), 'Save-stage font override detected'
    width, height = fig.get_size_inches()
    # Visible labels/legend must fit the exported uncropped canvas.
    for item in text_proof:
        x,y,w,h=item['bbox_inches']
        assert x>=-.002 and y>=-.002 and x+w<=width+.002 and y+h<=height+.002, item
    proof.update(text=text_proof, legend_bbox_inches=box(legend), figure_inches=[width,height],
                 axes=[dict(position=list(ax.get_position().bounds), xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
                            xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(), title=ax.get_title(), xlabel_bbox_inches=box(ax.xaxis.label),
                            lines=[dict(x=arrays_digest(line.get_xdata()),y=arrays_digest(line.get_ydata()),
                                        label=line.get_label(),linestyle=line.get_linestyle(),marker=line.get_marker(),
                                        markerfacecolor=line.get_markerfacecolor(),color=line.get_color())for line in ax.lines]) for ax in axes],
                 font_files={str(fonts/f):sha(fonts/f) for f in ('TIMES.TTF','simsun.ttc')},
                 smoothing='centered5_per_seed_internal_only_raw_start_and_endpoint')
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(destination/('figure.'+ext), dpi=180)
    proof['outputs']={ext:sha(destination/('figure.'+ext)) for ext in ('pdf','svg','png')}
    dump(destination/'presentation_proof.json',proof)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--role', choices=list(LABELS), required=True)
    parser.add_argument('--data', type=Path, default=ROOT/'data')
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
    os.environ['XDG_CACHE_HOME']=str(ROOT/'cache')
    manifest=json.loads((args.data/'data_manifest.json').read_text())
    for name,info in manifest['files'].items():
        assert sha(args.data/name)==info['sha256'], name
    render(args.role,args.data.resolve(),args.output)


if __name__=='__main__':
    main()
