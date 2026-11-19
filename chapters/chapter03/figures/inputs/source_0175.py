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


CURVE_AXES_MM = (125.0, 26.13563961876694)
TARGET_INK_XLABEL_LEGEND_BP = 15.0
TARGET_INK_LEGEND_MAPTITLE_BP = 12.0
TARGET_INK_BOTTOM_MM = 2.0
TARGET_INK_TOP_BP = 4.5
ALLOCATION_LABELS_BY_CASE = {'3.7': {'本期投入占\n初始资源（%）': '本期发射能量占\n初始发射能量（%）', '(a) 各期总投入': '(a) 各期发射能量分配总量', '对象投入占初始资源（%）': '各终端发射能量占初始发射能量（%）'}, '3.8': {'本期投入占\n初始资源（%）': '本期分配的额度占\n初始总额度（%）', '(a) 各期总投入': '(a) 各期分配的柴油额度', '对象投入占初始资源（%）': '各区域获得的额度占初始总额度（%）'}}



# 2026-11-17: explicit time labels after all retained scientific/native guards.
def apply_time_labels(fig, out, number):
    import re
    from matplotlib.text import Text
    before = scientific_artists(fig)
    before_axes = physical_axes(fig)
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    properties = lambda: [(id(t),t.get_fontsize(),t.get_fontfamily(),t.get_fontweight(),
                           t.get_fontstyle(),t.get_position(),t.get_rotation(),t.get_ha(),t.get_va()) for t in texts]
    old_properties = properties()
    changes = []
    for text in texts:
        original = text.get_text()
        value = original.replace('决策期', '传输轮次' if number == '3.7' else '分配步骤')
        value = value.replace('本期', '本轮' if number == '3.7' else '本步')
        value = value.replace('各期', '各轮' if number == '3.7' else '各步')
        value = re.sub(r'第(\d+)期', lambda m: '第'+m.group(1)+('轮' if number == '3.7' else '步'), value)
        if value != original:
            text.set_text(value)
            changes.append(dict(before=original,after=value,font_bp=text.get_fontsize()))
    assert changes, 'Expected time labels not found'
    fig.canvas.draw()
    assert digest(scientific_artists(fig)) == digest(before)
    assert digest(physical_axes(fig)) == digest(before_axes)
    assert properties() == old_properties
    audit = text_and_hex_audit(fig)
    assert not audit['visible_text_outside_canvas'], audit['visible_text_outside_canvas']
    dump(Path(out)/'time_expression_native_proof.json', dict(
        status='PASS',number=number,changes=changes,
        scientific_before_sha256=digest(before),scientific_after_sha256=digest(scientific_artists(fig)),
        physical_axes_canvas_fonts_anchors_exact=True,
        all_labels_inside_canvas=True,display_numbers_preserved=True,
        model_index_unchanged=True,environment_steps=0,training_runs=0,model_loads=0))

def visible_ink_spacing(pdf, legend_labels, bottom_label_text):
    """Measure actual720dpi raster ink inside independently identified PDF text regions."""
    import fitz
    import numpy as np
    doc = fitz.open(pdf)
    page = doc[0]
    lines = []
    for b in page.get_text('dict')['blocks']:
        if b['type'] != 0:
            continue
        for line in b['lines']:
            text = ''.join(s['text'] for s in line['spans']).strip()
            if text:
                lines.append((text, fitz.Rect(line['bbox'])))
    def line_for(text):
        normalized = text.replace(' ', '')
        matches = [r for t, r in lines if t.replace(' ', '') == normalized]
        assert len(matches) == 1, (text, [(t, list(r)) for t,r in lines if normalized in t.replace(' ', '')])
        return matches[0]
    def union(rects):
        result = fitz.Rect(rects[0])
        for rect in rects[1:]:
            result |= rect
        return result
    def ink(rect, pad=1.0):
        r = fitz.Rect(rect.x0-pad, rect.y0-pad, rect.x1+pad, rect.y1+pad) & page.rect
        pix = page.get_pixmap(matrix=fitz.Matrix(10,10), clip=r, colorspace=fitz.csRGB, alpha=False)
        array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height,pix.width,3)
        yy, xx = np.nonzero(array.min(axis=2) < 220)
        assert len(xx), ('no ink', list(rect))
        return [(pix.x+xx.min())/10, (pix.y+yy.min())/10,
                (pix.x+xx.max()+1)/10, (pix.y+yy.max()+1)/10]
    xlabel_rect = line_for('决策期')
    legend_rect = union([line_for(label) for label in legend_labels])
    # Include the legend's line/marker keys as well as its text.
    legend_rect.x0 = max(0, legend_rect.x0-40)
    titles = [r for text,r in lines if text.startswith('(b)') or text.startswith('(c)')]
    assert len(titles) == 2
    xlabel = ink(xlabel_rect)
    legend = ink(legend_rect)
    map_titles = [ink(r) for r in titles]
    bottom_label = ink(line_for(bottom_label_text))
    top_visible = ink(fitz.Rect(0,0,page.rect.width,30),pad=0)
    return dict(schema='actual-raster-ink-spacing-v1', dpi=720, threshold_min_rgb_less_than=220,
                measurement_basis='actual raster pixels in independent PDF text/key crops; not font layout boxes',
                xlabel_bbox_bp=list(xlabel_rect), legend_bbox_bp=list(legend_rect),
                xlabel_ink_bp=xlabel, legend_ink_bp=legend, top_map_title_ink_bp=map_titles,
                bottom_scale_label_ink_bp=bottom_label, top_visible_ink_bp=top_visible,
                top_visible_content_to_file_top_bp=top_visible[1],
                xlabel_to_first_legend_ink_bp=legend[1]-xlabel[3],
                last_legend_to_first_map_title_ink_bp=min(r[1] for r in map_titles)-legend[3],
                scale_label_ink_to_file_bottom_bp=page.rect.height-bottom_label[3],
                native_canvas_bp=list(page.rect)[2:])


def apply_physical_geometry(fig, rects_mm, height_mm, legend_anchor_mm):
    """Change canvas/positions only; physical axes width/height and font sizes stay explicit."""
    fig.set_size_inches(156.0/25.4, height_mm/25.4, forward=True)
    for ax, (x,y,w,h) in zip(fig.axes, rects_mm):
        ax.set_position([x/156.0, y/height_mm, w/156.0, h/height_mm])
    fig.legends[0].set_bbox_to_anchor((legend_anchor_mm[0]/156.0, legend_anchor_mm[1]/height_mm),
                                    transform=fig.transFigure)
    fig.canvas.draw()


def calibrate_visible_spacing(fig, out, save_original):
    import numpy as np
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    probe = out/'runtime'/'spacing_probe.pdf'
    records = []
    # PDF's coordinates point down; Matplotlib's physical coordinates point up.
    for iteration in range(3):
        save_original(fig, probe, metadata={'CreationDate':None,'ModDate':None})
        measured = visible_ink_spacing(probe, labels, fig.axes[-1].xaxis.label.get_text())
        records.append(measured)
        if (abs(measured['xlabel_to_first_legend_ink_bp']-TARGET_INK_XLABEL_LEGEND_BP) <= .15
            and abs(measured['last_legend_to_first_map_title_ink_bp']-TARGET_INK_LEGEND_MAPTITLE_BP) <= .15
            and abs(measured['scale_label_ink_to_file_bottom_bp']-TARGET_INK_BOTTOM_MM*72/25.4) <= .15
            and abs(measured['top_visible_content_to_file_top_bp']-TARGET_INK_TOP_BP) <= .15):
            break
        wh = fig.get_size_inches()*25.4
        rects = [list(r) for r in physical_axes(fig)]
        anchor = fig.legends[0].get_bbox_to_anchor()._bbox.bounds
        legend_anchor = [anchor[0]*wh[0], anchor[1]*wh[1]]
        legend_up = TARGET_INK_LEGEND_MAPTITLE_BP-measured['last_legend_to_first_map_title_ink_bp']
        curve_up = TARGET_INK_XLABEL_LEGEND_BP-measured['xlabel_to_first_legend_ink_bp']+legend_up
        bottom_down = measured['scale_label_ink_to_file_bottom_bp']-TARGET_INK_BOTTOM_MM*72/25.4
        rects[0][1] += curve_up*25.4/72
        legend_anchor[1] += legend_up*25.4/72
        new_height = float(wh[1]+curve_up*25.4/72)
        for rect in rects:
            rect[1] -= bottom_down*25.4/72
        legend_anchor[1] -= bottom_down*25.4/72
        new_height -= bottom_down*25.4/72
        new_height += (TARGET_INK_TOP_BP-measured['top_visible_content_to_file_top_bp'])*25.4/72
        apply_physical_geometry(fig,rects,new_height,legend_anchor)
    save_original(fig, probe, metadata={'CreationDate':None,'ModDate':None})
    final = visible_ink_spacing(probe, labels, fig.axes[-1].xaxis.label.get_text())
    assert abs(final['xlabel_to_first_legend_ink_bp']-15) <= .2, final
    assert abs(final['last_legend_to_first_map_title_ink_bp']-12) <= .2, final
    assert abs(final['scale_label_ink_to_file_bottom_bp']-2*72/25.4) <= .2, final
    assert abs(final['top_visible_content_to_file_top_bp']-TARGET_INK_TOP_BP) <= .2, final
    return dict(iterations=records,final=final,targets=dict(xlabel_to_legend_ink_bp=15,
                legend_to_top_map_title_ink_bp=12,bottom_scale_label_to_file_bottom_mm=2,top_visible_content_to_file_top_bp=4.5))


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
    new_height = curve_before[1] + CURVE_AXES_MM[1] + 6.5
    fig.set_layout_engine(None)
    fig.set_size_inches(156.0 / 25.4, new_height / 25.4, forward=True)
    for i, (ax, rect) in enumerate(zip(fig.axes, before_axes)):
        x, y, w, h = rect
        if i == 0:
            h = CURVE_AXES_MM[1]
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
    ink_spacing_proof = calibrate_visible_spacing(fig, out, save_original)
    # Current labels change only after the historical parent has been reproduced.
    # Keep the saved source, scientific artists, geometry and font sizes intact.
    from matplotlib.text import Text
    label_changes = []
    allocation_labels = ALLOCATION_LABELS_BY_CASE[number]
    for text in fig.findobj(Text):
        old_label = text.get_text()
        if old_label in allocation_labels:
            new_label = allocation_labels[old_label]
            text.set_text(new_label)
            label_changes.append(dict(before=old_label, after=new_label,
                                      font_bp=text.get_fontsize()))
    assert len(label_changes) == 3, (number, label_changes)
    compact_layout = None
    if number == '3.8':
        # Current layout: reuse the curve's empty upper-left area for the legend.
        # Remove only its former separate band; keep all physical data axes unchanged.
        shrink_bp = 20.0
        shrink_mm = shrink_bp * 25.4 / 72.0
        prior_wh = fig.get_size_inches() * 25.4
        prior_rects = [list(v) for v in physical_axes(fig)]
        compact_height = float(prior_wh[1] - shrink_mm)
        fig.set_size_inches(156.0 / 25.4, compact_height / 25.4, forward=True)
        for index, (axes, rect) in enumerate(zip(fig.axes, prior_rects)):
            x, y, width, height = rect
            if index == 0:
                y -= shrink_mm
            axes.set_position([x / 156.0, y / compact_height, width / 156.0, height / compact_height])
        legend = fig.legends[0]
        legend._loc = 2  # Matplotlib upper-left location, retaining the existing handles/fonts.
        legend.set_bbox_to_anchor((.025, .97), transform=fig.axes[0].transAxes)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer)
        legend_data = legend_bbox.transformed(fig.axes[0].transData.inverted())
        assert legend_data.x0 >= 1 and legend_data.x1 <= 16, list(legend_data.bounds)
        assert legend_data.y0 > 15, list(legend_data.bounds)
        for line in fig.axes[0].lines:
            x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
            within = (x >= legend_data.x0 - 1) & (x <= legend_data.x1 + 1)
            if np.any(within):
                assert float(np.max(y[within])) + 1 < legend_data.y0, ('legend overlaps data', line.get_label())
        compact_layout = dict(canvas_height_before_mm=float(prior_wh[1]),
            canvas_height_after_mm=compact_height, removed_external_legend_band_bp=shrink_bp,
            legend_location='upper left inside curve axes', legend_bbox_data=list(legend_data.bounds),
            data_and_map_axes_physical_sizes_unchanged=True, legend_data_overlap=False)
    # Preserve label geometry during terminology changes; the 3.8 layout exception is explicit above.
    final_probe = out / 'runtime' / 'term_spacing_probe.pdf'
    save_original(fig, final_probe, metadata={'CreationDate': None, 'ModDate': None})
    measured_terms = visible_ink_spacing(final_probe,
                        [t.get_text() for t in fig.legends[0].get_texts()],
                        fig.axes[-1].xaxis.label.get_text())
    if compact_layout is None:
        assert abs(measured_terms['xlabel_to_first_legend_ink_bp']-15) <= .2
        assert abs(measured_terms['last_legend_to_first_map_title_ink_bp']-12) <= .2
    else:
        xlabel_to_map = min(v[1] for v in measured_terms['top_map_title_ink_bp']) - measured_terms['xlabel_ink_bp'][3]
        assert 15 <= xlabel_to_map <= 18, xlabel_to_map
        compact_layout['xlabel_to_top_map_title_ink_bp'] = xlabel_to_map
    assert abs(measured_terms['scale_label_ink_to_file_bottom_bp']-2*72/25.4) <= .2
    assert abs(measured_terms['top_visible_content_to_file_top_bp']-TARGET_INK_TOP_BP) <= .2
    ink_spacing_proof['final'] = measured_terms
    ink_spacing_proof['wording_change_preserves_accepted_canvas_and_axes'] = True
    if compact_layout is not None:
        ink_spacing_proof['current_layout_exception'] = compact_layout
    new_height = float(fig.get_size_inches()[1]*25.4)
    fig.canvas.draw()
    after = scientific_artists(fig)
    after_axes = physical_axes(fig)
    assert digest(before) == digest(after), (number, 'scientific artists changed')
    assert np.allclose(np.array(before_axes)[1:, [0,2,3]], np.array(after_axes)[1:, [0,2,3]], atol=1e-9, rtol=0), 'Map/colorbar physical sizes or x positions changed'
    assert np.ptp(np.array(after_axes)[1:,1]-np.array(before_axes)[1:,1]) < 1e-9, 'Map/colorbar internal spacing changed'
    assert np.allclose(after_axes[0][[0,2]] if isinstance(after_axes[0],np.ndarray) else [after_axes[0][0],after_axes[0][2]], [before_axes[0][0],before_axes[0][2]], atol=1e-9, rtol=0)
    assert np.allclose(after_axes[0][2:], CURVE_AXES_MM, atol=1e-9, rtol=0)
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
    if compact_layout is None:
        assert spacing['xlabel_to_legend_bp'] > 0
        assert spacing['legend_to_first_map_title_bp'] > 0
    else:
        spacing['legend_location'] = 'inside upper-left curve area'
        spacing['xlabel_to_top_map_title_ink_bp'] = compact_layout['xlabel_to_top_map_title_ink_bp']
    input_hashes = {str(p.relative_to(inputs)): sha(p) for p in sorted(inputs.rglob('*')) if p.is_file()}
    dump(out / 'audit' / 'scientific_before.json', before)
    dump(out / 'audit' / 'scientific_after.json', after)
    dump(out / 'audit' / 'text_and_hex_bboxes.json', audits)
    proof = dict(schema='chapter3-case-native-restoration-spacing15-v1', status='PASS', number=number,
                 accepted_parent_directory=str(parent), accepted_parent_png_reproduction_exact=True,
                 scientific_before_sha256=digest(before), scientific_after_sha256=digest(after),
                 scientific_arrays_ranges_ticks_patches_colors_IDs_exact=True,
                 original_input_data_sha256={n: sha(inputs / n) for n in ['data.csv', 'period_totals.csv', 'data/provenance.json']},
                 input_files_sha256=input_hashes, figure_size_mm=[156.0, new_height],
                 parent_figure_size_mm=clean(old_wh), parent_axes_mm=before_axes, axes_mm=after_axes,
                 map_and_colorbar_physical_sizes_preserved=True, map_and_colorbar_spacing_preserved=True,
                 actual_ink_spacing_720dpi=ink_spacing_proof,
                 prior_radio_curve_axes_mm=[125.0,26.13563961876694],
                 curve_axes_size_mm=list(CURVE_AXES_MM), curve_linewidth_bp=1.25,
                 curve_linestyle='solid', marker_size_bp=4.0, marker_edgewidth_bp=.6,
                 method_marker_identities_preserved=True, modified_line_artists=changed_lines,
                 fonts_bp=dict(main=10.5, object_ID=9.0), spacing=spacing,
                 visible_text_outside_canvas=audits['visible_text_outside_canvas'],
                 object_ID_count=audits['object_ID_count'], ID_bbox_collision_count=0,
                 all_object_ID_bboxes_inside_complete_hexagons=True,
                 allocation_term_label_changes=label_changes,
                 environment_steps=0, model_loads=0, policy_inferences=0, training_runs=0)
    if compact_layout is not None:
        proof['schema'] = 'chapter3-case-compact-legend-v1'
        proof['compact_legend_layout'] = compact_layout
    apply_time_labels(fig, out, number)
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
    proof['outputs_sha256'] = {name: sha(out / name) for name in
                              ['figure.pdf', 'figure.svg', 'figure.png', 'figure_grayscale.png']}
    dump(out / 'audit' / 'actual_ink_spacing_720dpi.json', ink_spacing_proof)
    dump(out / 'presentation_proof.json', proof)
    source = dict(schema='chapter3-case-native-restoration-spacing15-v1', number=number,
                  parent_directory=str(parent), parent_pdf_sha256=sha(parent / 'figure.pdf'),
                  editable_source='plot.py', renderer=str(MAINTAINED), renderer_sha256=sha(MAINTAINED),
                  inherited_native_source='inputs/plot_source.py', input_files_sha256=input_hashes,
                  scientific_values_changed=False, changes=['equal solid 1.25bp method lines with filled 4bp markers',
                      'restore common125x26.1356396188mm curve axes,58mm wide maps at original equal-hex scale; actual720dpi xlabel-to-legend15bp and legend-to-map-title12bp',
                      'replace three labels with explicit transmitter-energy or diesel-quota quantities; initial-budget denominators, saved inputs and scientific artists unchanged'],
                  allocation_term_label_changes=label_changes,
                  figure_size_mm=proof['figure_size_mm'], scientific_before_sha256=digest(before),
                  scientific_after_sha256=digest(after), outputs=proof['outputs_sha256'],
                  proof_sha256=sha(out / 'presentation_proof.json'), environment_steps=0,
                  model_loads=0, policy_inferences=0, training_runs=0)
    if compact_layout is not None:
        source['schema'] = 'chapter3-case-compact-legend-v1'
        source['compact_legend_layout'] = compact_layout
        source['changes'].append('2026-11-13: move legend inside empty upper-left curve area and remove its former20bp band; data axes, maps, fonts and data unchanged; old15/12bp calibration only applies to historical reproduction stage')
    dump(out / 'source.json', source)
    dump(out / 'spec.json', dict(number=number, kind='chapter3_case', figure_size_mm=proof['figure_size_mm'],
                               axes_mm=after_axes, fonts_bp=proof['fonts_bp'], spacing=spacing, actual_ink_spacing_720dpi=ink_spacing_proof['final'],
                               color_scales_and_map_geometry='inherited exactly', curve_axes_size_mm=list(CURVE_AXES_MM)))
    (out / 'README.rst').write_text('第三章案例图风格统一\n==================\n\n只读本地保存的CSV重绘，无模型、环境、推理或训练。先逐像素复现接受父PNG，再统一为1.25bp实线、4bp实心marker和125×26.1356396188mm总量绘图区；横坐标至图例实际墨迹15bp，图例至上排地图标题实际墨迹12bp。四地图的原始数值、原58mm宽正六边形几何、颜色/色标、ID9bp、坐标范围与刻度保持。地图对象行数不同，因此总高度随任务变化。详细图元与无裁切核验见presentation_proof.json及audit。\n')
    with (out / 'README.rst').open('a') as readme:
        readme.write('\n2026-11-12术语统一：当前活动renderer在父图复现后，仅将三处可见标签写明通信发射能量或柴油分配额度，并明确分母为初始发射能量或初始总额度。历史inputs和父图不变；画布、绘图区、科学图元、字体与间距保持。\n')
    if compact_layout is not None:
        (out / 'README.rst').write_text('第三章柴油实例图紧凑图例\n==========================\n\n2026-11-13按用户要求前移图后小结。两方法图例移入曲线左上空区，数据无交叠，取消原独立图例带20bp。曲线125×26.135640mm、四地图58mm宽、色标与图内字体、数据和对象编号保持。地图内部间距保持，横轴标题至上排地图标题实际墨迹约16.7bp。画布宽156mm，高193.452503mm；没有缩小曲线或地图。\n\n活动入口plot.py和replot_sources/renderer.py只读本地CSV重绘，无训练、模型载入、推理或环境运行。父图精确复现与旧15/12bp校准仍作前置核验，随后明确执行本轮图例布局，不将旧外图例间距冒称为当前布局。scientific artist摘要、地图几何、字体、无裁切及图例不遮线的证据见presentation_proof.json与audit。\n')
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
