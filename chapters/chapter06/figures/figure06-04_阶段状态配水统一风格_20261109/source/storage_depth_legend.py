#!/usr/bin/env python3
"""Native redraw of the C6.4 legend with an explicit depth label.

Only the storage title differs from the sealed parent legend.  This program
does not load a model, environment or scientific result array.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE
HELPER=HERE/'source/display_helpers.py'
OLD_TITLE = '日初储水量（mm）'
NEW_TITLE = '日初储水深度（mm）'
WIDTH = 156.
HEIGHT = 28.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(output, title=NEW_TITLE):
    spec = importlib.util.spec_from_file_location('sealed_c64_display', HELPER)
    display = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(display)
    plt = display.plotting(ROOT)
    plt.rcParams.update({'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5, 'legend.fontsize': 10.5,
                         'svg.hashsalt': 'chapter6-storage-depth-legend-20261110'})
    from matplotlib.patches import Circle, Rectangle
    import matplotlib.patheffects as pe
    fig = plt.figure(figsize=(WIDTH/25.4, HEIGHT/25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT))
    ax.axis('off')
    edges = [80., 100., 120., 140., 160., 180.]
    colours = ['#fff9df', '#ffedc5', '#ffdc9c', '#ffc775', '#f6ac52']
    ax.text(WIDTH/2, 25., title, ha='center', va='center', fontsize=10.5)
    left, right, y, h = 5., 151., 20., 2.3
    step = (right-left)/len(colours)
    for i, colour in enumerate(colours):
        ax.add_patch(Rectangle((left+i*step, y), step, h, fc=colour, ec='none'))
    ax.add_patch(Rectangle((left, y), right-left, h, fc='none', ec='#65725e', lw=.8))
    for i, value in enumerate(edges):
        ax.text(left+i*step, 17.2, f'{value:g}', ha='center', va='center', fontsize=10.5)
    ax.text(5., 7.3, '当日配水量\n（m³）', ha='left', va='center', fontsize=10.5, linespacing=1.05)
    effects = [pe.Stroke(linewidth=3., foreground='white'), pe.Normal()]
    circles = []
    for x, value in ((45., 125.), (90., 1000.), (138.1916666667, 4000.)):
        radius = 1.6*math.sqrt(value/1000)
        circle = Circle((x, 7.3), radius, fc='none', ec='#A80058', lw=1.5)
        circle.set_path_effects(effects)
        ax.add_patch(circle)
        if value == 4000:
            ax.text(151., 7.3, f'{value:g}', ha='right', va='center', fontsize=10.5)
        else:
            ax.text(x+radius+2.2, 7.3, f'{value:g}', ha='left', va='center', fontsize=10.5)
        circles.append({'value_m3': value, 'radius_mm': radius, 'center_mm': [x, 7.3]})
    output.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    from matplotlib.text import Text
    for item in fig.findobj(match=Text):
        if item.get_visible() and item.get_text():
            bb = item.get_window_extent(renderer)
            assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= fig.bbox.x1 and bb.y1 <= fig.bbox.y1, item.get_text()
    for ext in ('pdf', 'svg', 'png'):
        kwargs = {'dpi': 240} if ext == 'png' else {}
        if ext == 'pdf':
            kwargs['metadata'] = {'CreationDate': None, 'ModDate': None}
        elif ext == 'svg':
            kwargs['metadata'] = {'Date': None}
        fig.savefig(output/('legend.'+ext), **kwargs)
    plt.close(fig)
    return {'width_mm': WIDTH, 'height_mm': HEIGHT, 'storage_depth_edges_mm': edges,
            'storage_colours': colours, 'circles': circles, 'circle_line_bp': 1.5,
            'white_halo_bp': 3., 'text_bp': 10.5, 'storage_title': title,
            'storage_bar': {'left_mm': left, 'right_mm': right, 'height_mm': h, 'y_mm': y},
            'scientific_execution': {'environment_steps': 0, 'model_loads': 0, 'training_steps': 0, 'inference_calls': 0}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '--output-dir', dest='output', type=Path, default=HERE)
    parser.add_argument('--parent-label', action='store_true', help='Verification-only native parent reconstruction')
    args = parser.parse_args()
    record = render(args.output, OLD_TITLE if args.parent_label else NEW_TITLE)
    (args.output/'render_geometry.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': 'GENERATED', 'output': str(args.output), 'title': record['storage_title']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
