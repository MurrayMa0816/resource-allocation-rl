#!/usr/bin/env python3
"""One local81-day replay of the SAME accepted and audited median-JA case.

Reads saved arrays only.  No environment/model, interpolation, smoothing,
network dependency or additional result selection is part of this exporter.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from data_adapter import ROOT, Evidence, common_volume_edges, dump, sha
import display_helpers as display

SOURCE = Path(__file__).resolve().parent
DEFAULT_SCALE = ROOT / 'figures/figure06-04_区域配水过程/colour_scale.json'


def points_path(points, close=False):
    points = list(points)
    if not points:
        return ''
    # Preserve the source coordinates; do not reshape/repack any field.
    return ('M' + 'L'.join(f'{float(x)!r},{float(y)!r}' for x, y in points) + ('Z' if close else ''))


def geometry_path(geometry):
    kind, coordinates = geometry['type'], geometry['coordinates']
    if kind == 'Polygon':
        return ''.join(points_path(ring, True) for ring in coordinates)
    if kind == 'MultiPolygon':
        return ''.join(points_path(ring, True) for polygon in coordinates for ring in polygon)
    if kind == 'LineString':
        return points_path(coordinates)
    if kind == 'MultiLineString':
        return ''.join(points_path(line) for line in coordinates)
    raise ValueError('Unsupported unchanged map geometry: ' + kind)


def build_svg(geometry, channel_style, ids):
    """Same cells, banks, blue water, orange gates and white ID labels as PDF."""
    layout, symbols = geometry['layout'], geometry['compact_gate_symbols']
    if [str(region['id']) for region in layout['regions']] != list(ids):
        raise ValueError('Replay geometry must preserve all89 original region IDs/order')
    frame = layout['board_rectangle']
    xmin, xmax, ymin, ymax = (frame[key] for key in ('x_min', 'x_max', 'y_min', 'y_max'))
    width, height = xmax - xmin, ymax - ymin
    pt = lambda value: float(value) * 25.4 / 72 / 119 * width
    lookup = {int(region['owner']): i for i, region in enumerate(layout['regions'])}
    cells, edges, outlines = defaultdict(list), {}, defaultdict(list)
    background = []
    for cell in layout['cells']:
        polygon = display.hexagon(cell)
        if cell['kind'] != 'service':
            background.append(points_path(polygon, True))
            continue
        owner = int(cell['owner']); cells[owner].append(points_path(polygon, True))
        for first, last in zip(polygon, polygon[1:] + polygon[:1]):
            edges.setdefault(tuple(sorted((first, last))), []).append(owner)
    for edge, owners in edges.items():
        if len(owners) != 2 or owners[0] != owners[1]:
            for owner in set(owners):
                outlines[owner].append(points_path(edge))
    parts = [f'<svg id="scene" xmlns="http://www.w3.org/2000/svg" '
             f'viewBox="{xmin} {-ymax} {width} {height}" role="img" '
             'aria-label="洪庙支渠89区当日配水量地图">',
             '<g transform="scale(1,-1)">',
             f'<path d="{"".join(background)}" fill="#f4f1e8" stroke="#deded1" stroke-width="{pt(.16)}"/>']
    for owner, i in sorted(lookup.items(), key=lambda pair: pair[1]):
        parts.extend([f'<g class="region" data-region="{i}" tabindex="0" role="button" aria-label="{escape(ids[i])}区">',
            f'<path class="region-fill" d="{"".join(cells[owner])}" fill="{display.ZERO_COLOR}" '
            f'stroke="#a59c8a" stroke-width="{pt(.16)}"/>',
            f'<path class="region-outline" d="{"".join(outlines[owner])}" fill="none" '
            f'stroke="#65563e" stroke-width="{pt(.6)}" pointer-events="none"/>', '</g>'])
    parts.append(f'<path d="{geometry_path(geometry["combined_water_geometry"])}" fill="#399bc1" fill-rule="evenodd" pointer-events="none"/>')
    grid = geometry['water_hex_edges']
    parts.append(f'<path d="{"".join(points_path(edge) for edge in grid["edges"])}" fill="none" '
                 f'stroke="{grid["color"]}" stroke-width="{pt(grid["linewidth_pt"])}" pointer-events="none"/>')
    parts.append(f'<path d="{geometry_path(symbols["main_bank_lines_geometry"])}" fill="none" stroke="#287ba0" '
                 f'stroke-width="{pt(.4)}" pointer-events="none"/>')
    for gate in symbols['regional_gates']:
        for line in gate['side_rails']:
            parts.append(f'<path d="{points_path(line)}" fill="none" stroke="#a35f22" stroke-width="{pt(.8)}" pointer-events="none"/>')
        parts.append(f'<path d="{points_path(gate["gate_crossbar"])}" fill="none" stroke="#a35f22" stroke-width="{pt(.7)}" pointer-events="none"/>')
    gate = symbols['total_gate']
    for polygon in gate.get('pier_polygons', []):
        parts.append(f'<path d="{points_path(polygon, True)}" fill="{display.ORANGE}" pointer-events="none"/>')
    if 'gate_crossbar' in gate:
        parts.append(f'<path d="{points_path(gate["gate_crossbar"])}" fill="none" stroke="{display.ORANGE}" stroke-width="{pt(2)}" pointer-events="none"/>')
    parts.extend([f'<rect x="{xmin}" y="{ymin}" width="{width}" height="{height}" fill="none" '
                  f'stroke="#65725e" stroke-width="{pt(.8)}" pointer-events="none"/>', '</g>',
                  '<g text-anchor="middle" dominant-baseline="central" pointer-events="none" font-family="Times New Roman,SimSun,serif">'])
    for label, style in zip(symbols['channel_labels'], channel_style):
        x, y = style['position']; size = pt(style['font_size_pt']); lines = list(label['text']) if label.get('stacked') else [label['text']]
        for row, text in enumerate(lines):
            yy = -y + (row - (len(lines) - 1) / 2) * size * .95
            parts.append(f'<text x="{x}" y="{yy}" font-size="{size}" font-weight="700" fill="black">{escape(text)}</text>')
    size = pt(9.5)
    for label in symbols['region_labels']:
        x, y = label['position']; text = str(label['id']); boxw = size * (.55 * len(text) + .24)
        parts.extend([f'<rect x="{x-boxw/2}" y="{-y-size*.55}" width="{boxw}" height="{size*1.1}" fill="white" fill-opacity=".88"/>',
                      f'<text x="{x}" y="{-y}" font-size="{size}" fill="#1d2426">{escape(text)}</text>'])
    parts.extend(['</g>', '</svg>'])
    return ''.join(parts)


def validate_arrays(case, area):
    """Pure saved-array shape/balance gate; no synthetic data are published."""
    a, q, b, state = (np.asarray(case[key], dtype=np.float64) for key in ('a_m3', 'q_m3', 'balance_m3', 'B_mm'))
    if (a.shape != (81, 89) or q.shape != (81,) or b.shape != (82,) or state.shape != (82, 89)
            or len(case['dates']) != 81 or len(case['ids']) != 89 or np.asarray(area).shape != (89,)):
        raise ValueError('Replay requires the genuine complete81 allocations and82 state boundaries')
    if any(not np.isfinite(value).all() for value in (a, q, b, state, area)) or (a < 0).any() or (np.asarray(area) <= 0).any():
        raise ValueError('Nonfinite, negative allocations or invalid physical service areas')
    np.testing.assert_allclose(a.sum(axis=1), q, rtol=0, atol=1e-7)
    np.testing.assert_allclose(b[:-1] - q, b[1:], rtol=0, atol=1e-7)


def validate_scale(scale, case):
    edges = common_volume_edges(case['a_m3'])
    if (not np.array_equal(np.asarray(scale['positive_edges_m3']), edges)
            or scale['positive_colors'] != list(display.POSITIVE_COLORS)
            or scale['zero_color'] != display.ZERO_COLOR or scale['zero_exact'] is not True
            or scale['unit'] != '千m³' or float(scale['divisor']) != 1000):
        raise ValueError('Replay and static maps must use the same full81 scale, palette and exact-zero semantics')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True, type=Path)
    parser.add_argument('--colour-scale', type=Path, default=DEFAULT_SCALE)
    parser.add_argument('--output', type=Path, default=ROOT / 'figures/replay/洪庙支渠89区_逐日配置推演.html')
    args = parser.parse_args()
    # This is deliberately before map/template work: no scientific data or HTML
    # are exported without the identical accepted-run and independent-audit gate.
    evidence = Evidence(args.index); case = evidence.case()
    output = args.output.resolve(); scale_path = args.colour_scale.resolve()
    if not output.is_relative_to(ROOT / 'figures') or not scale_path.is_relative_to(ROOT / 'figures'):
        raise ValueError('Supplementary artifacts must remain inside the new figures scope')
    validate_arrays(case, evidence.area)
    scale = json.loads(scale_path.read_text()); validate_scale(scale, case)
    geometry = {p.stem: json.loads(p.read_text()) for p in (SOURCE / 'geometry').glob('*.json')}
    style = json.loads((SOURCE / 'channel_style.json').read_text())
    svg = build_svg(geometry, style, evidence.ids)
    payload = dict(ids=evidence.ids, dates=[str(value) for value in case['dates']], area_ha=evidence.area.tolist(),
        a_m3=case['a_m3'].tolist(), q_m3=case['q_m3'].tolist(), balance_m3=case['balance_m3'].tolist(),
        B_mm=case['B_mm'].tolist(), scale=scale)
    template = (SOURCE / 'replay_template.html').read_text()
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    html = template.replace('__MAP_SVG__', svg).replace('__TRAJECTORY_JSON__', encoded)
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(html)
    dump(output.with_suffix('.source.json'), dict(status='GENERATED_NEEDS_ACTUAL_BROWSER_REVIEW',
        output=str(output), output_sha256=sha(output), output_bytes=output.stat().st_size,
        accepted_evidence_files_sha256=evidence.bindings,
        representative={k:v for k,v in evidence.representative.items() if k not in ('record','row')},
        static_colour_scale=dict(path=str(scale_path),sha256=sha(scale_path)),
        source_files_sha256={str(p):sha(p) for p in (SOURCE / 'replay.py', SOURCE / 'replay_template.html', SOURCE / 'data_adapter.py', SOURCE / 'display_helpers.py')},
        geometry_sha256={p.name:sha(p) for p in (SOURCE / 'geometry').glob('*.json')},
        channel_style_sha256=sha(SOURCE / 'channel_style.json'),
        frames=81, regions=89, stored_state_boundaries=82, new_environment_steps=0,
        new_model_inferences=0, interpolation=False, smoothing=False, scientific_array_rounding=False,
        locally_self_contained=True, deployment=False))
    print(json.dumps(dict(status='REPLAY_GENERATED_NOT_BROWSER_REVIEWED',path=str(output),bytes=output.stat().st_size), ensure_ascii=False))


if __name__ == '__main__':
    main()
