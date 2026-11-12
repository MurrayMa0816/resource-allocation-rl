"""Frozen display functions only; no environment, model, or result is imported."""
from __future__ import annotations
import csv,hashlib,json,math,os
from pathlib import Path
import numpy as np
FONT_ROOT=Path('/root/gamewar/latex_thesis_resource_allocation/assets/fonts/windows')
FONT_NAMES=('simsun.ttc','TIMES.TTF','TIMESI.TTF','TIMESBD.TTF')
POSITIVE_COLORS=('#ffffcc','#d9f0a3','#addd8e','#78c679','#31a354','#006837')
ZERO_COLOR='#cdd3d8'
ORANGE='#cf7627'
BINS=None  # Set only from the new representative trajectory before rendering.

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def plotting(root):
    os.environ['MPLCONFIGDIR'] = str(root / 'runtime_cache/plot_matplotlib')
    os.environ['XDG_CACHE_HOME'] = str(root / 'runtime_cache/plot_xdg')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in FONT_NAMES:
        font_manager.fontManager.addfont(str(FONT_ROOT / name))
    plt.rcParams.update({
        'font.family': ['Times New Roman', 'SimSun'],
        'font.size': 10.5, 'axes.labelsize': 10.5, 'axes.titlesize': 10.5,
        'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 9.5,
        'axes.unicode_minus': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'lines.linewidth': 1.25, 'savefig.facecolor': 'white',
    })
    return plt

def hexagon(cell):
    return [(round(cell['x'] + math.cos(math.radians(30 + 60 * k)) / math.sqrt(3), 6),
             round(cell['y'] + math.sin(math.radians(30 + 60 * k)) / math.sqrt(3), 6))
            for k in range(6)]

def map_path(geom):
    from matplotlib.path import Path as MplPath
    parts = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
    vertices, codes = [], []
    for part in parts:
        for ring in [part.exterior] + list(part.interiors):
            points = list(ring.coords)
            vertices.extend(points)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 2) + [MplPath.CLOSEPOLY])
    return MplPath(vertices, codes)

def allocation_colors(values):
    # Exactly zero has its own gray category; no positive value is suppressed.
    values = np.asarray(values)
    band = np.searchsorted(BINS[1:], values, side='left')
    return [ZERO_COLOR if value == 0 else POSITIVE_COLORS[int(index)]
            for value, index in zip(values, band)]

def draw_map(ax, values, geometry):
    from matplotlib.collections import PolyCollection, LineCollection
    from matplotlib.patches import PathPatch, Rectangle
    from shapely.geometry import shape
    layout, symbols = geometry['layout'], geometry['compact_gate_symbols']
    lookup = {int(region['owner']): i for i, region in enumerate(layout['regions'])}
    shades = allocation_colors(values)
    for kind in ('background', 'service', 'canal', 'upstream_canal', 'canal_connector'):
        cells = [cell for cell in layout['cells'] if cell['kind'] == kind]
        facecolors = [shades[lookup[cell['owner']]] if kind == 'service' else '#f4f1e8' for cell in cells]
        ax.add_collection(PolyCollection([hexagon(cell) for cell in cells], facecolors=facecolors,
            edgecolors='#a59c8a' if kind == 'service' else '#deded1', linewidths=.16, zorder=1))
    edges = {}
    for cell in layout['cells']:
        if cell['kind'] != 'service':
            continue
        polygon = hexagon(cell)
        for first, last in zip(polygon, polygon[1:] + polygon[:1]):
            edges.setdefault(tuple(sorted((first, last))), []).append(cell['owner'])
    boundaries = [edge for edge, owners in edges.items() if len(owners) != 2 or owners[0] != owners[1]]
    ax.add_collection(LineCollection(boundaries, colors='#65563e', linewidths=.6, zorder=2))
    water = shape(geometry['combined_water_geometry'])
    ax.add_patch(PathPatch(map_path(water), fc='#399bc1', ec='none', zorder=3))
    grid = geometry['water_hex_edges']
    ax.add_collection(LineCollection(grid['edges'], colors=grid['color'], linewidths=grid['linewidth_pt'], zorder=3.1))
    banks = shape(symbols['main_bank_lines_geometry'])
    pieces = list(banks.geoms) if hasattr(banks, 'geoms') else [banks]
    ax.add_collection(LineCollection([list(piece.coords) for piece in pieces], colors='#287ba0', linewidths=.4, zorder=3.2))
    for label in symbols['channel_labels']:
        text = '\n'.join(label['text']) if label.get('stacked') else label['text']
        position = list(label['position'])
        if label.get('stacked'):
            position[1] -= .3
        ax.text(*position, text, ha='center', va='center', fontsize=9.5,
                fontweight='bold', color='black', linespacing=.95, zorder=4)
    for gate in symbols['regional_gates']:
        for line in gate['side_rails']:
            ax.plot(*zip(*line), color='#a35f22', lw=.8, zorder=5)
        ax.plot(*zip(*gate['gate_crossbar']), color='#a35f22', lw=.7, zorder=5)
    gate = symbols['total_gate']
    for polygon in gate.get('pier_polygons', []):
        ax.add_collection(PolyCollection([polygon], facecolors=ORANGE, edgecolors='none', zorder=6))
    if 'gate_crossbar' in gate:
        ax.plot(*zip(*gate['gate_crossbar']), color=ORANGE, lw=2, zorder=6)
    for label in symbols['region_labels']:
        ax.text(*label['position'], label['id'], ha='center', va='center', fontsize=9.5,
                color='#1d2426', fontweight='normal',
                bbox=dict(fc='white', ec='none', pad=.25, alpha=.88), zorder=7)
    frame = layout['board_rectangle']
    ax.set(xlim=(frame['x_min'], frame['x_max']), ylim=(frame['y_min'], frame['y_max']))
    ax.add_patch(Rectangle((frame['x_min'], frame['y_min']), frame['x_max'] - frame['x_min'],
        frame['y_max'] - frame['y_min'], fill=False, edgecolor='#65725e', linewidth=.8, zorder=8))
    ax.set_aspect('equal')
    ax.axis('off')

def save(fig, base, plt):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix('.pdf'))
    fig.savefig(base.with_suffix('.svg'))
    fig.savefig(base.with_suffix('.png'), dpi=200)
    size = dict(width_mm=float(fig.get_size_inches()[0] * 25.4), height_mm=float(fig.get_size_inches()[1] * 25.4))
    assets = {extension: dict(path=str(base.with_suffix('.' + extension)), sha256=sha(base.with_suffix('.' + extension)))
              for extension in ('pdf', 'svg', 'png')}
    plt.close(fig)
    return dict(**size, assets=assets)
