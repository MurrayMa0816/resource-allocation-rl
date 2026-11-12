#!/usr/bin/env python3
"""Render Chapter 3 cases from saved CSVs, preserving scientific artists.

The accepted parent is reproduced pixel-for-pixel before presentation changes.
This module never imports a policy, model, environment, or training package.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent.parent
THESIS=next(p for p in HERE.parents if (p/'assets/fonts/windows').is_dir())
PUBLICATION=HERE
PARENT=HERE/'replot_sources/parent'
MAINTAINED=Path(__file__)
CASES = {'3.7': ('figure37', 'figure3-7'), '3.8': ('figure38', 'figure3-8')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def clean(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_manifest(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def prepare(number):
    return HERE,HERE/'replot_sources/parent',HERE/'inputs'


def scientific_artists(fig):
    records = []
    for ax in fig.axes:
        records.append(dict(
            xlim=clean(ax.get_xlim()), ylim=clean(ax.get_ylim()),
            xscale=ax.get_xscale(), yscale=ax.get_yscale(),
            xticks=clean(ax.get_xticks()), yticks=clean(ax.get_yticks()),
            lines=[dict(x=clean(line.get_xdata()), y=clean(line.get_ydata()),
                        color=clean(line.get_color()), alpha=line.get_alpha(),
                        marker=line.get_marker(), markevery=clean(line.get_markevery()))
                   for line in ax.lines],
            collections=[dict(paths=[dict(vertices=clean(path.vertices), codes=clean(path.codes))
                                     for path in collection.get_paths()],
                              offsets=clean(collection.get_offsets()), facecolors=clean(collection.get_facecolors()),
                              edgecolors=clean(collection.get_edgecolors()), linewidths=clean(collection.get_linewidths()),
                              alpha=collection.get_alpha(), array=clean(collection.get_array()),
                              clim=clean(collection.get_clim())) for collection in ax.collections],
            patches=[dict(vertices=clean(patch.get_path().vertices), codes=clean(patch.get_path().codes),
                          facecolor=clean(patch.get_facecolor()), edgecolor=clean(patch.get_edgecolor()),
                          linewidth=patch.get_linewidth()) for patch in ax.patches],
            images=[dict(array=clean(im.get_array()), extent=clean(im.get_extent()), clim=clean(im.get_clim()))
                    for im in ax.images],
            object_IDs=[dict(text=t.get_text(), position=clean(t.get_position()), fontsize=t.get_fontsize())
                        for t in ax.texts if t.get_text().isdigit()]))
    return records


def physical_axes(fig):
    import numpy as np
    wh = fig.get_size_inches() * 25.4
    return [clean(np.asarray(ax.get_position().bounds) * np.tile(wh, 2)) for ax in fig.axes]


def text_and_hex_audit(fig):
    import numpy as np
    from matplotlib.patches import Polygon
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer, k = fig.canvas.get_renderer(), 72 / fig.dpi
    hidden = set()
    for ax in fig.axes:
        for axis, lim in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            for tick in axis.get_major_ticks():
                if not min(lim) - 1e-8 <= tick.get_loc() <= max(lim) + 1e-8:
                    hidden.update((id(tick.label1), id(tick.label2)))
    texts, ids, collisions = [], [], []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text() or id(t) in hidden:
            continue
        bb = t.get_window_extent(renderer)
        rect = clean(np.asarray(bb.bounds) * k)
        record = dict(text=t.get_text(), font_bp=t.get_fontsize(), bbox_bp=rect,
                      font_family=t.get_fontfamily(), color=clean(t.get_color()))
        texts.append(record)
        assert bb.x0 >= -.5 / k and bb.y0 >= -.5 / k and bb.x1 <= fig.bbox.x1 + .5 / k and bb.y1 <= fig.bbox.y1 + .5 / k, ('text outside canvas', record)
    for axid, ax in enumerate(fig.axes[1:5], 1):
        pairs = []
        for t in ax.texts:
            if not t.get_text().isdigit():
                continue
            assert t.get_fontsize() == 9.0
            bb = t.get_window_extent(renderer)
            mid = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
            cells = []
            for patch in ax.patches:
                if isinstance(patch, Polygon) and len(patch.get_path().vertices) == 7:
                    path = patch.get_path().transformed(patch.get_transform())
                    cells.append((float(np.linalg.norm(path.vertices[:-1].mean(0) - mid)), path))
            path = min(cells, key=lambda c: c[0])[1]
            corners = np.array([[bb.x0, bb.y0], [bb.x0, bb.y1], [bb.x1, bb.y0], [bb.x1, bb.y1]])
            contained = bool(path.contains_points(corners, radius=1e-7).all())
            assert contained, (axid, t.get_text(), 'ID outside hex')
            ids.append(dict(axis=axid, ID=t.get_text(), font_bp=t.get_fontsize(),
                            position_data=clean(t.get_position()), bbox_bp=clean(np.asarray(bb.bounds) * k),
                            hex_vertices_bp=clean(path.vertices * k), bbox_inside_hex=contained))
            pairs.append((t.get_text(), bb))
        for i, (name, bb) in enumerate(pairs):
            for name2, bb2 in pairs[i + 1:]:
                ox = max(0, min(bb.x1, bb2.x1) - max(bb.x0, bb2.x0))
                oy = max(0, min(bb.y1, bb2.y1) - max(bb.y0, bb2.y0))
                if ox > 1e-7 and oy > 1e-7:
                    collisions.append([axid, name, name2, ox * oy])
    assert not collisions, collisions[:10]
    for t in texts:
        expected = 9.0 if t['text'].isdigit() and t['font_bp'] == 9.0 else 10.5
        assert t['font_bp'] == expected, ('font size', t)
    return dict(visible_texts=texts, object_IDs=ids, object_ID_count=len(ids), ID_bbox_collisions=collisions,
                all_object_ID_bboxes_inside_complete_hexagons=True, visible_text_outside_canvas=0)


def render(number, output_dir=None):
    asset, parent, inputs = prepare(number)
    out = Path(output_dir or asset).resolve()
    assert out!=HERE, 'Choose a fresh output directory'
    assert not out.is_relative_to(parent)
    assert not (out / 'figure.pdf').exists(), 'Existing render is immutable; choose a new output directory'
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'runtime'
    for key, leaf in [('MPLCONFIGDIR', 'mpl'), ('XDG_CACHE_HOME', 'xdg'), ('TMPDIR', 'tmp')]:
        p = cache / leaf
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib import pyplot as plt
    from PIL import Image
    module_spec = importlib.util.spec_from_file_location('saved_case_' + number.replace('.', '_'), inputs / 'plot_source.py')
    native = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(native)
    save_original = Figure.savefig
    captured = []

    class FigureCaptured(Exception):
        pass

    def capture(fig, *args, **kwargs):
        captured.append(fig)
        raise FigureCaptured()

    Figure.savefig = capture
    try:
        native.render(native.DEFAULT_DOMAIN, out)
    except FigureCaptured:
        pass
    finally:
        Figure.savefig = save_original
    assert len(captured) == 1
    fig = captured[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Apply the accepted print derivative exactly before any new changes.
    for line in fig.findobj(Line2D):
        color = matplotlib.colors.to_hex(line.get_color()).lower()
        if color in {'#0072b2', '#d55e00', '#009e73'} and line.get_linestyle().lower() not in ('none', '') and len(np.atleast_1d(line.get_xdata())) >= 2:
            line.set_linestyle({'#0072b2': '-', '#d55e00': '--', '#009e73': '-.'}[color])
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text().isdigit() and t.get_fontsize() == 7.5:
                bb = t.get_window_extent(renderer)
                oldmid = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
                t.set_fontsize(9.0)
                bb = t.get_window_extent(renderer)
                newmid = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
                transform = t.get_transform()
                t.set_position(transform.inverted().transform(transform.transform(np.asarray(t.get_position())) + oldmid - newmid))
    fig.canvas.draw()
    save_original(fig, out / 'accepted_parent_reproduction.png', dpi=240)
    with Image.open(out / 'accepted_parent_reproduction.png') as im:
        before_pixels = np.asarray(im.convert('RGBA'))
    with Image.open(parent / 'figure.png') as im:
        accepted_pixels = np.asarray(im.convert('RGBA'))
    assert before_pixels.shape == accepted_pixels.shape and np.array_equal(before_pixels, accepted_pixels), (number, 'accepted PNG not reproduced', before_pixels.shape, accepted_pixels.shape)
    before = scientific_artists(fig)
    before_axes = physical_axes(fig)
    old_wh = fig.get_size_inches() * 25.4
    assert len(fig.axes) == 6
    curve_before = before_axes[0]
    new_height = curve_before[1] + 26.0 + 6.5
    fig.set_layout_engine(None)
    fig.set_size_inches(156.0 / 25.4, new_height / 25.4, forward=True)
    for i, (ax, rect) in enumerate(zip(fig.axes, before_axes)):
        x, y, w, h = rect
        if i == 0:
            h = 26.0
        ax.set_position([x / 156.0, y / new_height, w / 156.0, h / new_height])
        ax.set_xticks(before[i]['xticks'])
        ax.set_yticks(before[i]['yticks'])
        ax.set_xlim(before[i]['xlim'])
        ax.set_ylim(before[i]['ylim'])
    # The anchor is expressed in physical mm and stays fixed relative to maps.
    for leg in fig.legends:
        bbox = leg.get_bbox_to_anchor()._bbox
        anchor = bbox.bounds
        leg.set_bbox_to_anchor((anchor[0] * old_wh[0] / 156.0,
                               anchor[1] * old_wh[1] / new_height), transform=fig.transFigure)
    changed_lines = []
    for line in fig.findobj(Line2D):
        color = matplotlib.colors.to_hex(line.get_color()).lower()
        if color not in {'#0072b2', '#d55e00', '#009e73'} or line.get_linestyle().lower() in ('none', '') or len(np.atleast_1d(line.get_xdata())) < 2:
            continue
        changed_lines.append(dict(label=line.get_label(), color=color, marker=line.get_marker(),
                                  old_linewidth_bp=line.get_linewidth(), old_linestyle=line.get_linestyle(),
                                  old_marker_size_bp=line.get_markersize()))
        line.set_linestyle('-')
        line.set_linewidth(1.25)
        line.set_markersize(4.0)
        line.set_markeredgewidth(.6)
        line.set_markerfacecolor(line.get_color())
        line.set_markeredgecolor(line.get_color())
    fig.canvas.draw()
    after = scientific_artists(fig)
    after_axes = physical_axes(fig)
    assert digest(before) == digest(after), (number, 'scientific artists changed')
    assert np.allclose(before_axes[1:], after_axes[1:], atol=1e-11, rtol=0), 'Map/colorbar physical geometry changed'
    assert np.allclose(before_axes[0][:3], after_axes[0][:3], atol=1e-11, rtol=0)
    assert np.allclose(after_axes[0][2:], [125, 26], atol=1e-10, rtol=0)
    audits = text_and_hex_audit(fig)
    assert audits['object_ID_count'] == (400 if number == '3.7' else 480)
    renderer, k = fig.canvas.get_renderer(), 72 / fig.dpi
    ax, legend = fig.axes[0], fig.legends[0]
    xlabel_bb, legend_bb = ax.xaxis.label.get_window_extent(renderer), legend.get_window_extent(renderer)
    topmap_title_bbs = [a.title.get_window_extent(renderer) for a in fig.axes[1:3]]
    bottom_label_bb = fig.axes[-1].xaxis.label.get_window_extent(renderer)
    spacing = dict(
        xlabel_to_legend_bp=(xlabel_bb.y0 - legend_bb.y1) * k,
        legend_to_first_map_title_bp=(legend_bb.y0 - max(bb.y1 for bb in topmap_title_bbs)) * k,
        colorbar_label_to_figure_bottom_bp=bottom_label_bb.y0 * k,
        curve_to_first_map_top_mm=(after_axes[0][1] - after_axes[1][1] - after_axes[1][3]),
        map_rows_frame_gap_mm=(after_axes[1][1] - after_axes[3][1] - after_axes[3][3]),
        bottom_maps_to_colorbar_mm=after_axes[3][1] - after_axes[-1][1] - after_axes[-1][3],
        legend_rows=1, legend_columns=2)
    assert spacing['xlabel_to_legend_bp'] > 0
    assert spacing['legend_to_first_map_title_bp'] > 0
    input_hashes = {str(p.relative_to(inputs)): sha(p) for p in sorted(inputs.rglob('*')) if p.is_file()}
    dump(out / 'audit' / 'scientific_before.json', before)
    dump(out / 'audit' / 'scientific_after.json', after)
    dump(out / 'audit' / 'text_and_hex_bboxes.json', audits)
    proof = dict(schema='chapter3-case-style-unification-v1', status='PASS', number=number,
                 accepted_parent_directory=str(parent), accepted_parent_png_reproduction_exact=True,
                 scientific_before_sha256=digest(before), scientific_after_sha256=digest(after),
                 scientific_arrays_ranges_ticks_patches_colors_IDs_exact=True,
                 original_input_data_sha256={n: sha(inputs / n) for n in ['data.csv', 'period_totals.csv', 'data/provenance.json']},
                 input_files_sha256=input_hashes, figure_size_mm=[156.0, new_height],
                 parent_figure_size_mm=clean(old_wh), parent_axes_mm=before_axes, axes_mm=after_axes,
                 map_and_colorbar_physical_rectangles_unchanged=True,
                 curve_axes_size_mm=[125.0, 26.0], curve_linewidth_bp=1.25,
                 curve_linestyle='solid', marker_size_bp=4.0, marker_edgewidth_bp=.6,
                 method_marker_identities_preserved=True, modified_line_artists=changed_lines,
                 fonts_bp=dict(main=10.5, object_ID=9.0), spacing=spacing,
                 visible_text_outside_canvas=audits['visible_text_outside_canvas'],
                 object_ID_count=audits['object_ID_count'], ID_bbox_collision_count=0,
                 all_object_ID_bboxes_inside_complete_hexagons=True,
                 environment_steps=0, model_loads=0, policy_inferences=0, training_runs=0)
    for ext in ['pdf', 'svg', 'png']:
        metadata = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {'Date': None} if ext == 'svg' else None
        save_original(fig, out / ('figure.' + ext), dpi=240, metadata=metadata)
    plt.close(fig)
    with Image.open(out / 'figure.png') as im:
        im.convert('L').save(out / 'figure_grayscale.png')
    for name in ['data.csv', 'period_totals.csv']:
        dst = out / name
        if dst.resolve() != (asset / name).resolve() and not dst.exists():
            shutil.copyfile(asset / name, dst)
        assert sha(dst) == sha(asset / name)
    # Local CSV hashes are verified by plot.py before loading the native source.
    proof['outputs_sha256'] = {f'figure.{ext}': sha(out / f'figure.{ext}') for ext in ['pdf', 'svg', 'png']}
    dump(out / 'presentation_proof.json', proof)
    source = dict(schema='chapter3-case-style-unification-v1', number=number,
                  parent_directory=str(parent), parent_pdf_sha256=sha(parent / 'figure.pdf'),
                  editable_source='plot.py', renderer=str(MAINTAINED), renderer_sha256=sha(MAINTAINED),
                  inherited_native_source='inputs/plot_source.py', input_files_sha256=input_hashes,
                  scientific_values_changed=False, changes=['equal solid 1.25bp method lines with filled 4bp markers',
                      'common 125 x 26mm total-allocation axes; fixed 58mm map width and inherited internal map/colorbar spacing'],
                  figure_size_mm=proof['figure_size_mm'], scientific_before_sha256=digest(before),
                  scientific_after_sha256=digest(after), outputs=proof['outputs_sha256'],
                  proof_sha256=sha(out / 'presentation_proof.json'), environment_steps=0,
                  model_loads=0, policy_inferences=0, training_runs=0)
    dump(out / 'source.json', source)
    dump(out / 'spec.json', dict(number=number, kind='chapter3_case', figure_size_mm=proof['figure_size_mm'],
                               axes_mm=after_axes, fonts_bp=proof['fonts_bp'], spacing=spacing,
                               color_scales_and_map_geometry='inherited exactly', curve_axes_size_mm=[125.0, 26.0]))
    (out / 'README.rst').write_text('第三章案例图风格统一\n==================\n\n只读本地保存的CSV重绘，无模型、环境、推理或训练。先逐像素复现接受父PNG，再统一为1.25bp实线、4bp实心marker和125×26mm总量绘图区。四地图的原始数值、原58mm宽正六边形几何、颜色/色标、ID9bp、坐标范围与刻度保持。地图对象行数不同，因此总高度随任务变化。详细图元与无裁切核验见presentation_proof.json及audit。\n')
    print(json.dumps(dict(number=number, output=str(out), size_mm=proof['figure_size_mm'], spacing=spacing,
                          scientific_exact=True, accepted_parent_pixels_exact=True), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--number', choices=CASES, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    render(args.number, args.output_dir)


if __name__ == '__main__':
    main()
