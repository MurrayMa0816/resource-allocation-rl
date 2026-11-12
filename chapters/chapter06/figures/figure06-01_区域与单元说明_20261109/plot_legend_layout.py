#!/usr/bin/env python3
"""Draw the compact, nonquantitative legend of the unchanged C6.1 scene.

Only four existing explanations and their categorical icons are laid out.
No environment, policy, scientific result or scene-body renderer is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
SHARED_DATA = next(parent for parent in Path(__file__).resolve().parents if parent.name == "figures") / "shared/data"

import tempfile

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--source-dir', type=Path, default=HERE)
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thesis = next(p for p in source_dir.parents if (p / 'assets/fonts/windows').is_dir())
    palette_path = SHARED_DATA/'palette.json'
    fonts = [thesis / 'assets/fonts/windows' / name for name in ('simsun.ttc', 'TIMES.TTF')]
    inputs = {str(p): sha(p) for p in [palette_path, *fonts]}
    palette = json.loads(palette_path.read_text())
    assert len(palette) == 6

    with tempfile.TemporaryDirectory(prefix='.legend-layout-runtime-', dir=output_dir) as runtime:
        os.environ['MPLCONFIGDIR'] = str(Path(runtime) / 'matplotlib')
        os.environ['XDG_CACHE_HOME'] = str(Path(runtime) / 'xdg')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.collections import PolyCollection, LineCollection
        from matplotlib.patches import RegularPolygon, Rectangle
        from matplotlib.text import Text
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        for font in fonts:
            font_manager.fontManager.addfont(str(font))
        plt.rcParams.update({
            'font.family': ['Times New Roman', 'SimSun'],
            'font.size': 10.5, 'pdf.fonttype': 42,
            'svg.fonttype': 'none', 'svg.hashsalt': 'c6-scene-compact-legend-15mm-v1',
            'savefig.facecolor': 'white', 'axes.unicode_minus': False,
        })
        width, height = 156., 15.
        upper, lower = 10.5, 4.5
        fig = plt.figure(figsize=(width / 25.4, height / 25.4), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, width), ylim=(0, height))
        ax.axis('off')

        # Six same-colour cells with a closed outer boundary illustrate one region.
        unit = 2.
        first_y = upper - unit * math.sqrt(3) / 4
        centres = [(7.8 + unit * x, first_y + unit * y)
                   for x, y in [(0, 0), (1, 0), (2, 0),
                                (.5, math.sqrt(3) / 2),
                                (1.5, math.sqrt(3) / 2),
                                (2.5, math.sqrt(3) / 2)]]
        polygons = [[(round(cx + unit / math.sqrt(3) * math.cos(math.radians(30 + 60 * k)), 6),
                      round(cy + unit / math.sqrt(3) * math.sin(math.radians(30 + 60 * k)), 6))
                     for k in range(6)] for cx, cy in centres]
        ax.add_collection(PolyCollection(polygons, facecolors=palette[0],
                                        edgecolors='#7c9167', linewidths=.25, zorder=1))
        region = unary_union([Polygon(p) for p in polygons])
        assert region.geom_type == 'Polygon'
        ax.add_collection(LineCollection([list(region.exterior.coords)],
                                        colors='#435c41', linewidths=.9, zorder=2))
        ax.text(22., upper, '多个单元组成一个区域', ha='left', va='center', fontsize=10.5)

        # The six categorical swatches have no scale and retain their exact colours.
        for i, colour in enumerate(palette):
            ax.add_patch(RegularPolygon((85. + 3.2 * i, upper), numVertices=6,
                                        radius=1.15, orientation=math.pi / 6,
                                        facecolor=colour, edgecolor='#65563e', linewidth=.35))
        ax.text(105., upper, '颜色辅助区分区域', ha='left', va='center', fontsize=10.5)

        # The existing total-gate glyph is scaled uniformly to 90% as an icon.
        gate_scale = .9
        for x in (8.5, 12.1):
            ax.add_patch(Rectangle((x, lower - 2.5 * gate_scale),
                                   1.5 * gate_scale, 5. * gate_scale,
                                   fc='#cf7627', ec='none'))
        ax.plot([10.975, 10.975], [lower - 1.9 * gate_scale, lower + 1.9 * gate_scale],
                color='#cf7627', lw=2. * gate_scale)
        ax.text(22., lower, '总闸', ha='left', va='center', fontsize=10.5)

        # The inlet keeps its blue body and brown side rails/crossbar.
        inlet_x = 91.2
        inlet_width, inlet_height = 4. * gate_scale, 5. * gate_scale
        inlet_bottom = lower - inlet_height / 2
        ax.add_patch(Rectangle((inlet_x, inlet_bottom), inlet_width, inlet_height,
                               fc='#399bc1', ec='none'))
        for x in (inlet_x, inlet_x + inlet_width):
            ax.plot([x, x], [inlet_bottom, inlet_bottom + inlet_height],
                    color='#a35f22', lw=.8 * gate_scale)
        ax.plot([inlet_x + gate_scale, inlet_x + 3. * gate_scale], [lower, lower],
                color='#a35f22', lw=.7 * gate_scale)
        ax.text(105., lower, '区域入口', ha='left', va='center', fontsize=10.5)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
        expected = ['多个单元组成一个区域', '颜色辅助区分区域', '总闸', '区域入口']
        assert [t.get_text() for t in texts] == expected
        checks = []
        for t in texts:
            bbox = t.get_window_extent(renderer)
            assert t.get_fontsize() == 10.5
            assert bbox.x0 >= 0 and bbox.y0 >= 0
            assert bbox.x1 <= fig.bbox.width and bbox.y1 <= fig.bbox.height
            checks.append({'text': t.get_text(), 'font_bp': 10.5,
                           'bbox_pixels': list(bbox.bounds)})
        assert not any(texts[i].get_window_extent(renderer).overlaps(texts[j].get_window_extent(renderer))
                       for i in range(len(texts)) for j in range(i + 1, len(texts)))
        for ext in ('pdf', 'svg', 'png'):
            kwargs = {'dpi': 260} if ext == 'png' else {}
            if ext == 'pdf':
                kwargs['metadata'] = {'CreationDate': None, 'ModDate': None,
                                      'Creator': 'C6 unchanged scene compact categorical legend'}
            elif ext == 'svg':
                kwargs['metadata'] = {'Date': None}
            fig.savefig(output_dir / f'legend_layout.{ext}', **kwargs)
        plt.close(fig)

    assert {str(p): sha(p) for p in [palette_path, *fonts]} == inputs
    proof = {
        'schema': 'c6-scene-compact-legend-layout-v1', 'status': 'PASS',
        'canvas_mm': [156., 15.], 'text_font_bp': 10.5,
        'four_original_texts_preserved': expected,
        'text_checks': checks,
        'rows_centres_mm': [upper, lower],
        'nominal_row_text_gap_mm': (upper - lower) - 10.5 * 25.4 / 72,
        'same_colour_region_group_cells': 6,
        'categorical_swatch_count': 6, 'categorical_palette': palette,
        'gate_and_inlet_icon_scale': gate_scale,
        'quantitative_scale': False,
        'map_body_edited': False,
        'input_files_sha256': inputs,
        'input_files_unchanged': True,
        'source_sha256': sha(__file__),
        'outputs_sha256': {f'legend_layout.{ext}': sha(output_dir / f'legend_layout.{ext}')
                           for ext in ('pdf', 'svg', 'png')},
        'scientific_execution': 0,
    }
    (output_dir / 'legend_layout_proof.json').write_text(
        json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'GENERATED', 'canvas_mm': [156., 15.],
                      'output_dir': str(output_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
