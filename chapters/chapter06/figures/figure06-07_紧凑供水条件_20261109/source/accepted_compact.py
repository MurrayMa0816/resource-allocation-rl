#!/usr/bin/env python3
"""Native compact Figure 6.7 using the two previously published CSVs only.

Run this file to reproduce the PDF, SVG, PNG and provenance manifest beside it.
No model, environment, training log or raw evaluation loading is performed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
THESIS=next(p for p in HERE.parents if (p/'assets/fonts').is_dir())
OUTPUT = HERE
HERE = OUTPUT
PARENT = OUTPUT
FONT_PATH = THESIS/'assets/fonts/figure/ThesisFigureText-Regular.otf'
BLUE = '#236d9e'
WIDTH_MM, HEIGHT_MM = 82., 73.
INPUT_NAMES = ('budget_joint_mean_sample_SD.csv', 'budget_all3_seed_weather_means.csv')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    # Preserve the exact published source bytes rather than rewrite the CSVs.
    protected = {str(path): sha(path) for path in PARENT.iterdir() if path.is_file()}
    for name in INPUT_NAMES:
        target = HERE / name
        if target.exists():
            assert target.read_bytes() == (PARENT / name).read_bytes()
        else:
            shutil.copyfile(PARENT / name, target)
    rows = read_csv(HERE / INPUT_NAMES[0])
    raw = read_csv(HERE / INPUT_NAMES[1])
    assert [int(row['budget_m3']) for row in rows] == [3500000, 4000000, 4500000]
    assert len(raw) == 9
    for row in rows:
        selected = sorted((r for r in raw if r['budget_m3'] == row['budget_m3']), key=lambda r: int(r['training_seed']))
        assert [int(r['training_seed']) for r in selected] == [6400, 6401, 6402]
        values = np.asarray([[float(r[k]) * 100 for k in ('C90', 'JA')] for r in selected])
        np.testing.assert_allclose(values.mean(axis=0), [float(row[k]) for k in ('C90_mean_percent', 'JA_mean_percent')], rtol=0, atol=1e-12)
        np.testing.assert_allclose(values.std(axis=0, ddof=1), [float(row[k]) for k in ('C90_sample_SD_percent', 'JA_sample_SD_percent')], rtol=0, atol=1e-12)

    os.environ.setdefault('MPLCONFIGDIR', str(HERE.parent / 'runtime_cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox
    font_manager.fontManager.addfont(str(FONT_PATH))
    family = font_manager.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams.update({
        'font.family': family, 'font.size': 10.5, 'axes.labelsize': 10.5,
        'axes.titlesize': 10.5, 'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
        'legend.fontsize': 10.5, 'axes.unicode_minus': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'lines.linewidth': 1.15,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'c6-figure67-native-compact-20261109', 'savefig.facecolor': 'white',
    })
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    # 62 x 55 mm plotting field, near the 54.6 x 51.3 mm field of one 6.5 panel.
    ax = fig.add_axes([18 / WIDTH_MM, 15 / HEIGHT_MM, 62 / WIDTH_MM, 55 / HEIGHT_MM])
    offsets = {3500000: (0, 20), 4000000: (-20, 12), 4500000: (-23, 12)}
    annotations, errors, numeric = [], [], []
    for row in rows:
        x, y, sx, sy = [float(row[k]) for k in ('C90_mean_percent', 'JA_mean_percent', 'C90_sample_SD_percent', 'JA_sample_SD_percent')]
        errors.append(ax.errorbar(x, y, xerr=sx, yerr=sy, color=BLUE, marker='o', ms=5, ls='none', capsize=4, capthick=.9, elinewidth=1, zorder=3))
        annotations.append(ax.annotate(f"{int(row['budget_m3']) // 10000}万m³", (x, y), xytext=offsets[int(row['budget_m3'])], textcoords='offset points', ha='center', va='center'))
        numeric.append(dict(budget_m3=int(row['budget_m3']), x=x, y=y, xerr=sx, yerr=sy))
    ax.set(xlim=(35, 103.5), ylim=(76, 100.5), xlabel='区域产量达标率（%）', ylabel='平均相对产量（%）')
    ax.set_xticks([40, 60, 80, 100]); ax.set_yticks([80, 85, 90, 95, 100])
    ax.grid(alpha=.18, linewidth=.6)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    text_checks = []
    for item in fig.findobj(match=Text):
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer)
            assert item.get_fontsize() == 10.5, item.get_text()
            assert box.x0 >= -.5 and box.y0 >= -.5 and box.x1 <= fig.bbox.x1 + .5 and box.y1 <= fig.bbox.y1 + .5, (item.get_text(), tuple(box.bounds))
            text_checks.append(dict(text=item.get_text(), size_bp=item.get_fontsize(), box_pixels=list(box.bounds)))
    for i, item in enumerate(annotations):
        box = item.get_window_extent(renderer)
        for other in annotations[i + 1:]:
            assert not box.overlaps(other.get_window_extent(renderer))
        for container in errors:
            for collection in container.lines[2]:
                for path in collection.get_paths():
                    assert not path.transformed(ax.transData).intersects_bbox(box, filled=False), item.get_text()
            for line in container.lines[:1]:
                for xy in line.get_xydata():
                    cx, cy = ax.transData.transform(xy)
                    marker_box = Bbox.from_bounds(cx - 4, cy - 4, 8, 8)
                    assert not box.overlaps(marker_box), item.get_text()
    metadata = dict(CreationDate=None, ModDate=None, Creator='C6 Figure 6.7 native compact existing-CSV renderer')
    for ext in ('pdf', 'svg', 'png'):
        kwargs = dict(dpi=200) if ext == 'png' else {}
        if ext == 'pdf':
            kwargs['metadata'] = metadata
        fig.savefig(HERE / f'figure.{ext}', **kwargs)
    import fitz
    with fitz.open(HERE / 'figure.pdf') as document:
        page = document[0]
        pdf_size = [page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72]
        spans = [span for block in page.get_text('dict')['blocks'] if 'lines' in block for line in block['lines'] for span in line['spans'] if span['text'].strip()]
        assert all(abs(span['size'] - 10.5) < 1e-5 for span in spans)
        pdf_text = [dict(text=s['text'], size_bp=s['size'], bbox=s['bbox']) for s in spans]
    assert protected == {path: sha(path) for path in protected}, 'Published parent was modified'
    manifest = dict(
        status='RENDERED_REQUIRES_VISUAL_REVIEW', figure_number='6.7', group='budgets',
        width_mm=WIDTH_MM, height_mm=HEIGHT_MM, measured_pdf_size_mm=pdf_size,
        axes_size_mm=[62., 55.], comparison_figure65_single_panel_axes_mm=[54.6, 51.3],
        font=dict(path=str(FONT_PATH), sha256=sha(FONT_PATH), family=family, ordinary_font_size_bp=10.5),
        all_visible_text_bp=10.5, text_bounds_pass=True, annotation_errorbar_no_collision=True,
        visible_text_checks=text_checks, pdf_text_checks=pdf_text,
        source=str(Path(__file__).resolve()), source_sha256=sha(__file__),
        parent_manifest=dict(path=str(PARENT / 'manifest.json'), sha256=protected[str(PARENT / 'manifest.json')]),
        source_csv_exact_bytes_preserved={name: dict(parent_path=str(PARENT / name), local_path=str(HERE / name), sha256=sha(HERE / name)) for name in INPUT_NAMES},
        plotted_values=numeric, xlim=[35., 103.5], ylim=[76., 100.5], xticks=[40, 60, 80, 100], yticks=[80, 85, 90, 95, 100],
        full_horizontal_and_vertical_sample_SD=True, no_connecting_lines=True, color=BLUE,
        errorbar_unit='sample SD across three training-seed weather means, each 128 common TEST cases',
        JA_is_unchanged_area_weighted_metric=True, original_marker_line_cap_style_preserved=True,
        original_parent_files_sha256=protected, parent_files_unchanged=True,
        assets={ext: dict(path=str(HERE / f'figure.{ext}'), sha256=sha(HERE / f'figure.{ext}')) for ext in ('pdf', 'svg', 'png')},
        scientific_execution=dict(environment_steps=0, model_loads=0, inference=0, training=0, new_TEST=0),
        TeX_edits=0, paper_builds=0,
    )
    (HERE / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    plt.close(fig)
    print(json.dumps(dict(native_size_mm=pdf_size, pdf=str(HERE / 'figure.pdf'), png=str(HERE / 'figure.png')), ensure_ascii=False))


if __name__ == '__main__':
    main()
