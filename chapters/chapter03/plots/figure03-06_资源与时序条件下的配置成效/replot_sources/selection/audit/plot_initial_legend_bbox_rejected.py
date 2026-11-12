"""Select the existing Full/Pace/external condition curves for Chapter 3.

This renderer reads saved scalar CSV observations only. It first reproduces the
accepted parent PNG exactly, then removes the obsolete Direct/NoTask artists.
No scientific evaluation is performed and the old complete evidence is kept.
"""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
FULL, PACE = 'A_FULL', 'A_PACE'
EXTERNAL = {'radio': 'c_ppo', 'v5': 'gradient_ppo'}
LABELS = {'ReJoP（本文）': 'ReJoP', '按剩余期数均分': '去总量学习'}
REMOVED = {'直接预测本期总量', '总量仅看资源与时间'}
PATTERNS = {'#0072b2': '-', '#d55e00': '--', '#009e73': '-.',
            '#7b5ea7': (0, (3, 1, 1, 1, 1, 1)), '#555555': '--'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def rows(path):
    with gzip.open(path, 'rt', newline='') as stream:
        return list(csv.DictReader(stream))


def csv_write(path, records):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--asset', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    asset = args.asset.resolve(strict=True)
    out = (args.output_dir or asset).resolve()
    if not out.is_relative_to(Path('/hy-tmp')):
        raise ValueError('Output and all render caches must be in /hy-tmp')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'figure.pdf').exists():
        raise ValueError('Use a fresh output directory')
    spec = json.loads((asset/'source.json').read_text())
    for name, expected in spec['inherited_inputs_sha256'].items():
        assert sha(asset/'inputs'/name) == expected, name
    assert sha(__file__) == spec['renderer_sha256']
    assert sha(spec['parent_png']) == spec['parent_png_sha256']
    inp = asset/'inputs'
    cache = out/'runtime'
    for key, name in [('MPLCONFIGDIR', 'mpl'), ('XDG_CACHE_HOME', 'xdg'), ('TMPDIR', 'tmp')]:
        folder = cache/name
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    (out/'audit').mkdir(exist_ok=True)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox
    from PIL import Image

    def plain(value):
        if isinstance(value, np.ndarray):
            return plain(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        return value

    def series(container):
        line, caps, bars = container.lines
        return dict(x=plain(line.get_xdata()), y=plain(line.get_ydata()),
                    color=line.get_color(), linewidth=line.get_linewidth(),
                    linestyle=plain(line._unscaled_dash_pattern), marker=line.get_marker(),
                    markersize=line.get_markersize(), markerfacecolor=line.get_markerfacecolor(),
                    markeredgecolor=line.get_markeredgecolor(),
                    caps=[plain(cap.get_xydata()) for cap in caps],
                    errorbar_segments=[plain(bar.get_segments()) for bar in bars])

    def geometry(fig):
        return [dict(position=list(ax.get_position().bounds), xlim=list(ax.get_xlim()),
                     ylim=list(ax.get_ylim()), xticks=plain(ax.get_xticks()),
                     yticks=plain(ax.get_yticks()), xlabel=ax.get_xlabel(),
                     ylabel=ax.get_ylabel(), title=ax.get_title()) for ax in fig.axes]

    native_spec = importlib.util.spec_from_file_location('inherited_conditions', inp/'plot_source.py')
    native = importlib.util.module_from_spec(native_spec)
    native_spec.loader.exec_module(native)
    sys.path.insert(0, str(inp))
    sys.modules.pop('presentation', None)
    native.setup(cache, Path(spec['thesis']))
    original_save = Figure.savefig
    completed = set()
    proof = {}
    artists = []

    def save(fig, destination, *save_args, **save_kwargs):
        if id(fig) not in completed:
            completed.add(id(fig))
            for line in fig.findobj(Line2D):
                color = matplotlib.colors.to_hex(line.get_color()).lower()
                if color in PATTERNS and line.get_linestyle().lower() not in ('none', '') and len(np.atleast_1d(line.get_xdata())) >= 2:
                    line.set_linestyle(PATTERNS[color])
            # The print derivative changed only dash patterns. Its full current
            # pixels must be reproduced before removing any obsolete artist.
            parent_reproduction = out/'audit/current_parent_reproduction.png'
            original_save(fig, parent_reproduction, format='png', dpi=240)
            before_pixels = np.asarray(Image.open(parent_reproduction).convert('RGBA'))
            parent_pixels = np.asarray(Image.open(spec['parent_png']).convert('RGBA'))
            assert before_pixels.shape == parent_pixels.shape and np.array_equal(before_pixels, parent_pixels), 'Parent reproduction differs'
            before_geometry = geometry(fig)
            retained = []
            removed = []
            for axis_index, ax in enumerate(fig.axes):
                expected_keep = {container.get_label(): series(container) for container in ax.containers
                                 if container.get_label() not in REMOVED}
                axis_removed = []
                for container in list(ax.containers):
                    label = container.get_label()
                    if label in REMOVED:
                        axis_removed.append(label)
                        container.remove()
                    else:
                        container.set_label(LABELS.get(label, label))
                actual_keep = {container.get_label(): series(container) for container in ax.containers}
                renamed_expected = {LABELS.get(label, label): value for label, value in expected_keep.items()}
                assert actual_keep == renamed_expected, (axis_index, 'Retained errorbar changed')
                assert len(actual_keep) == 3 and len(axis_removed) == 1
                retained.append(actual_keep)
                removed.append(axis_removed)
            assert geometry(fig) == before_geometry, 'Axes or coordinate changed'
            for legend in list(fig.legends):
                legend.remove()
            handles = {}
            for ax in fig.axes:
                for handle, label in zip(*ax.get_legend_handles_labels()):
                    handles.setdefault(label, handle)
            legend = fig.legend(handles.values(), handles.keys(), loc='lower center', ncol=2,
                                frameon=False, bbox_to_anchor=(.5, 0), handlelength=1.4,
                                columnspacing=1.0, handletextpad=.4, fontsize=10.5)
            for text in legend.get_texts():
                text.set_fontfamily(['Times New Roman', 'SimSun'])
            # Balance the visible remaining whitespace with the same accepted
            # 9.6pt caption allowance; plotting rectangles stay fixed.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            labels = [ax.xaxis.label.get_window_extent(renderer) for ax in fig.axes]
            visible = list(legend.get_texts()) + list(getattr(legend, 'legend_handles', []))
            boxes = [item.get_window_extent(renderer) for item in visible if item.get_visible()]
            ink = Bbox.union(boxes)
            k = 72/fig.dpi
            down = (ink.y0*k + 9.6 - (min(box.y0 for box in labels)-ink.y1)*k)/2
            legend.set_bbox_to_anchor((.5, -down/(fig.get_figheight()*72)), transform=fig.transFigure)
            fig.canvas.draw()
            boxes = [item.get_window_extent(fig.canvas.get_renderer()) for item in visible if item.get_visible()]
            ink = Bbox.union(boxes)
            assert ink.x0 >= 0 and ink.x1 <= fig.bbox.x1 and ink.y0 >= 0
            assert all(abs(text.get_fontsize()-10.5) < 1e-12 for text in fig.findobj(Text))
            assert geometry(fig) == before_geometry
            proof.update(status='PASS', parent_png_pixel_exact=True,
                         retained_errorbar_series_exact=True, retained_series=12,
                         removed_by_axis=removed, geometry_before=before_geometry,
                         geometry_after=geometry(fig), axes_ranges_ticks_positions_exact=True,
                         figure_inches=list(fig.get_size_inches()), ordinary_font_bp=10.5,
                         legend_labels=list(handles), legend_within_canvas=True,
                         legend_ink_to_bottom_pt=float(ink.y0*k),
                         new_environment_steps=0, model_loads=0, policy_inferences=0,
                         new_training_steps=0, new_independent_test_paths=0)
            artists.extend(retained)
        suffix = Path(destination).suffix
        if suffix == '.png':
            save_kwargs['dpi'] = 600
        if suffix == '.pdf':
            save_kwargs['metadata'] = {'CreationDate': None, 'ModDate': None}
        if suffix == '.svg':
            save_kwargs['metadata'] = {'Date': None}
        return original_save(fig, destination, *save_args, **save_kwargs)

    Figure.savefig = save
    try:
        native.render(inp, out)
    finally:
        Figure.savefig = original_save
    assert len(completed) == 1
    original_aggregates = rows(out/'display_aggregates.csv.gz')
    selected_aggregates = [r for r in original_aggregates if r['method'] in (FULL, PACE, EXTERNAL[r['domain']])]
    assert len(original_aggregates) == 48 and len(selected_aggregates) == 36
    with gzip.open(out/'display_aggregates.csv.gz', 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, list(selected_aggregates[0]))
        writer.writeheader()
        writer.writerows(selected_aggregates)
    csv_write(out/'display_aggregates.csv', selected_aggregates)
    original_seed = rows(inp/'data.csv.gz')
    selected_seed = [r for r in original_seed if r['method'] in (FULL, PACE, EXTERNAL[r['domain']])]
    assert len(original_seed) == 250 and len(selected_seed) == 150
    csv_write(out/'display_seed_data.csv', selected_seed)
    target = out/'data.csv.gz'
    if not target.exists():
        target.symlink_to(inp/'data.csv.gz')
    conditions = [('radio', c) for c in ['budget08', 'default', 'budget12', 'persistence02', 'persistence095']]
    conditions += [('v5', c) for c in ['budget08', 'default', 'budget12', 'arrivals_front', 'arrivals_middle']]
    differences = []
    for domain, condition in conditions:
        arm = {}
        for method in (FULL, PACE):
            arm[method] = np.array([float(next(r['score_mean'] for r in original_seed
                                if r['domain'] == domain and r['condition'] == condition
                                and r['method'] == method and int(r['seed']) == seed)) for seed in range(5)])
        delta = arm[FULL] - arm[PACE]
        differences.append(dict(domain=domain, condition=condition,
                                full_mean=float(arm[FULL].mean()), pace_mean=float(arm[PACE].mean()),
                                full_minus_pace_mean=float(delta.mean()), paired_sample_sd=float(delta.std(ddof=1)),
                                positive_seeds=int((delta > 0).sum()), seed_count=5,
                                unit='score points' if domain == 'radio' else 'percentage points'))
    csv_write(out/'full_minus_pace.csv', differences)
    write(out/'audit/retained_artists.json', artists)
    write(out/'audit/presentation_verification.json', proof)
    write(out/'full_minus_pace.json', differences)
    outputs = {name: sha(out/name) for name in ['figure.pdf', 'figure.png', 'figure.svg',
               'display_aggregates.csv', 'display_aggregates.csv.gz', 'display_seed_data.csv',
               'full_minus_pace.csv', 'full_minus_pace.json', 'audit/retained_artists.json',
               'audit/presentation_verification.json']}
    write(out/'render_receipt.json', dict(status='PASS', outputs_sha256=outputs,
                                        renderer_sha256=sha(__file__), input_asset=str(asset),
                                        original_250_seed_rows_preserved=True,
                                        displayed_aggregate_rows=36, displayed_unique_seed_rows=150,
                                        new_environment_steps=0, model_loads=0,
                                        policy_inferences=0, training_runs=0, thesis_builds=0))
    print(json.dumps(dict(status='PASS', output=str(out), retained_series=12,
                         pdf_sha256=outputs['figure.pdf']), ensure_ascii=False))


if __name__ == '__main__':
    main()
