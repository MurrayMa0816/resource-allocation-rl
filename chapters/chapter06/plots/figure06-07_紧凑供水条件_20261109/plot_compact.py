#!/usr/bin/env python3
"""Reproduce native compact Figure 6.7 from its two local published CSVs.

Derived from the accepted_compact.py plotting calls.  The current native
layout's values, limits, ticks, error bars, colors, and three annotation offsets
are preserved; only the physical canvas/axes dimensions change.  No environment,
model, training log, or evaluation code is loaded.
"""
from pathlib import Path
import argparse, csv, hashlib, io, json, os

HERE = Path(__file__).resolve().parent
INPUTS = ('budget_joint_mean_sample_SD.csv', 'budget_all3_seed_weather_means.csv')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--input-dir', type=Path, default=HERE)
    parser.add_argument('--font', type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    inputs = args.input_dir.resolve()
    font = args.font
    if font is None:
        candidates = [p/'assets/fonts/figure/ThesisFigureText-Regular.otf' for p in HERE.parents]
        candidates.append(Path('/root/gamewar/latex_thesis_resource_allocation/assets/fonts/figure/ThesisFigureText-Regular.otf'))
        font = next(p for p in candidates if p.is_file())
    font = font.resolve()
    os.environ.setdefault('MPLCONFIGDIR', str(out/'.matplotlib-cache'))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox
    import fitz
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({
        'font.family': family, 'font.size': 10.5, 'axes.labelsize': 10.5,
        'axes.titlesize': 10.5, 'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
        'legend.fontsize': 10.5, 'axes.unicode_minus': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'lines.linewidth': 1.15,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'c6-figure67-native-compact-20261109', 'savefig.facecolor': 'white',
    })
    protected_inputs = {name: sha(inputs/name) for name in INPUTS}
    with (inputs/INPUTS[0]).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row['budget_m3']) for row in rows] == [3500000, 4000000, 4500000]
    width, height = 105., 82.
    fig = plt.figure(figsize=(width/25.4, height/25.4))
    ax = fig.add_axes([18/width, 15/height, 85/width, 65/height])
    offsets = {3500000: (0, 28), 4000000: (-20, 12), 4500000: (-23, 12)}
    annotations, errors = [], []
    for row in rows:
        x, y, sx, sy = [float(row[k]) for k in ('C90_mean_percent', 'JA_mean_percent', 'C90_sample_SD_percent', 'JA_sample_SD_percent')]
        errors.append(ax.errorbar(x, y, xerr=sx, yerr=sy, color='#236d9e', marker='o', ms=5, ls='none', capsize=4, capthick=.9, elinewidth=1, zorder=3))
        annotations.append(ax.annotate(f"{int(row['budget_m3'])//10000}万m³", (x, y), xytext=offsets[int(row['budget_m3'])], textcoords='offset points', ha='center', va='center'))
    ax.set(xlim=(35, 103.5), ylim=(76, 100.5), xlabel='区域产量达标率（%）', ylabel='平均相对产量（%）')
    ax.set_xticks([40, 60, 80, 100]); ax.set_yticks([80, 85, 90, 95, 100])
    ax.grid(alpha=.18, linewidth=.6)
    ax.xaxis.labelpad = 4; ax.yaxis.labelpad = 4
    ax.tick_params(axis='both', which='major', pad=3)

    def plain(value):
        if isinstance(value, np.ndarray): return plain(value.tolist())
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, (list, tuple)): return [plain(x) for x in value]
        if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
        return value

    def numeric():
        return [dict(xlim=plain(a.get_xlim()), ylim=plain(a.get_ylim()), xticks=plain(a.get_xticks()), yticks=plain(a.get_yticks()), xscale=a.get_xscale(), yscale=a.get_yscale(), lines=[dict(x=plain(l.get_xdata()), y=plain(l.get_ydata()), color=plain(l.get_color()), alpha=l.get_alpha(), linewidth=l.get_linewidth(), linestyle=l.get_linestyle(), marker=l.get_marker(), markersize=l.get_markersize(), markevery=plain(l.get_markevery())) for l in a.lines], collections=[dict(paths=[dict(vertices=plain(p.vertices), codes=plain(p.codes)) for p in c.get_paths()], offsets=plain(c.get_offsets()), array=plain(c.get_array()), facecolors=plain(c.get_facecolors()), edgecolors=plain(c.get_edgecolors()), linewidths=plain(c.get_linewidths()), alpha=c.get_alpha()) for c in a.collections], patches=[dict(vertices=plain(p.get_path().vertices), codes=plain(p.get_path().codes), facecolor=plain(p.get_facecolor()), edgecolor=plain(p.get_edgecolor()), linewidth=p.get_linewidth(), alpha=p.get_alpha()) for p in a.patches], annotations=[dict(text=t.get_text(), position=plain(t.get_position()), xy=plain(getattr(t, 'xy', None)), font=t.get_fontsize(), rotation=t.get_rotation()) for t in a.texts]) for a in fig.axes]

    before = numeric()
    # Native PDF-frame crop: retain 85x65mm axes and ordinary 10.5bp fonts,
    # with the accepted 1.5mm top and 2mm lower content margins.
    fig.canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', metadata={'CreationDate': None, 'ModDate': None})
    with fitz.open(stream=buf.getvalue(), filetype='pdf') as doc:
        page = doc[0]
        frames = [(line, fitz.Rect(line['bbox'])) for block in page.get_text('dict')['blocks'] for line in block.get('lines', [])]
        xlabel = next(box for line, box in frames if ''.join(s['text'] for s in line['spans']).replace(' ', '') == ax.get_xlabel().replace(' ', ''))
        top = min(min(box.y0 for line, box in frames), page.rect.height-ax.bbox.y1*72/fig.dpi)
        top_crop = top-1.5*72/25.4
        bottom_crop = page.rect.height-xlabel.y1-2*72/25.4
    pos = np.asarray(ax.get_position().bounds)*[width, height, width, height]
    height -= (top_crop+bottom_crop)*25.4/72
    fig.set_size_inches(width/25.4, height/25.4, forward=True)
    ax.set_position([pos[0]/width, (pos[1]-bottom_crop*25.4/72)/height, pos[2]/width, pos[3]/height])
    fig.canvas.draw()
    after = numeric(); assert before == after
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text(): continue
        box = item.get_window_extent(renderer)
        assert item.get_fontsize() == 10.5
        assert box.x0 >= -.5 and box.y0 >= -.5 and box.x1 <= fig.bbox.x1+.5 and box.y1 <= fig.bbox.y1+.5, (item.get_text(), box.bounds)
    for i, item in enumerate(annotations):
        box = item.get_window_extent(renderer)
        for other in annotations[i+1:]: assert not box.overlaps(other.get_window_extent(renderer))
        for container in errors:
            for collection in container.lines[2]:
                for path in collection.get_paths(): assert not path.transformed(ax.transData).intersects_bbox(box, filled=False), item.get_text()
            for line in container.lines[:1]:
                for xy in line.get_xydata():
                    cx, cy = ax.transData.transform(xy)
                    assert not box.overlaps(Bbox.from_bounds(cx-4, cy-4, 8, 8)), item.get_text()
    for ext in ('pdf', 'svg', 'png'):
        kw = {'metadata': {'CreationDate': None, 'ModDate': None}} if ext == 'pdf' else {'metadata': {'Date': None}} if ext == 'svg' else {'dpi': 200}
        fig.savefig(out/f'compact.{ext}', **kw)
    with fitz.open(out/'compact.pdf') as doc:
        page = doc[0]; traces = page.get_texttrace()
        assert all(abs(t['size']-10.5)<1e-5 and page.rect.contains(fitz.Rect(t['bbox'])) for t in traces)
        actual_mm = [page.rect.width*25.4/72, page.rect.height*25.4/72]
        pdf_frames = [dict(text=''.join(chr(c[0]) for c in t['chars']), size_bp=t['size'], bbox_bp=t['bbox']) for t in traces]
    assert protected_inputs == {name: sha(inputs/name) for name in INPUTS}
    geometry = dict(schema='c6-figure67-native-compact-105-v1', status='PASS_REQUIRES_VISUAL_REVIEW', native_mm=actual_mm, axes_sizes_mm=[[85.,65.]], font_bp=10.5, annotation_offsets_points={str(k):list(v) for k,v in offsets.items()}, display_changes_relative_to_current_layout=[], all_scientific_artists_preserved=True, all_text_inside_canvas=True, annotations_clear_errorbars_and_markers=True, source_CSV_sha256=protected_inputs, source_sha256=sha(__file__), font_sha256=sha(font), output_sha256={f'compact.{e}':sha(out/f'compact.{e}') for e in ('pdf','svg','png')}, actual_pdf_text_frames=pdf_frames, science_execution=dict(environment_steps=0,model_loads=0,policy_inference=0,training_runs=0,new_TEST=0))
    (out/'compact_geometry.json').write_text(json.dumps(geometry, ensure_ascii=False, indent=2)+'\n')
    (out/'compact_numeric_artists.json').write_text(json.dumps(dict(before=before,after=after), ensure_ascii=False, indent=2)+'\n')
    plt.close(fig)
    print(json.dumps(dict(status=geometry['status'],native_mm=actual_mm,axes_mm=geometry['axes_sizes_mm'],output=str(out)),ensure_ascii=False))


if __name__ == '__main__':
    main()
