#!/usr/bin/env python3
"""Compose two map panels per page with centered external titles.

Map coordinates and circles use one uniform physical scale. Text is rendered
at its established native size, rather than scaling an existing PDF.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile

import fitz

HERE = Path(__file__).resolve().parent
LAYER = HERE / 'centered_titles_20261113'
MM = 72 / 25.4
SCALE = 0.83
DAYS = (0, 46, 62, 72)
MAP_WIDTH_MM = 156 * SCALE
TITLES = {0: '(a) 5月30日（返青首日）', 46: '(b) 7月15日（孕穗首日）',
          62: '(c) 7月31日（抽穗首日）', 72: '(d) 8月10日（乳熟首日）'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def ink(page):
    boxes = []
    for drawing in page.get_drawings():
        fill, stroke = drawing.get('fill'), drawing.get('color')
        if fill is not None and all(v > .999 for v in fill) and stroke is None:
            continue
        if fill is None and stroke is None:
            continue
        width = float(drawing.get('width') or 0) if stroke is not None else 0.
        boxes.append(fitz.Rect(drawing['rect']) + (-width / 2, -width / 2, width / 2, width / 2))
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            for span in line['spans']:
                if span['text'].strip():
                    boxes.append(fitz.Rect(span['bbox']))
    return fitz.Rect(min(b.x0 for b in boxes), min(b.y0 for b in boxes),
                     max(b.x1 for b in boxes), max(b.y1 for b in boxes))


def identifier_frames(page, ids):
    frames = []
    for trace in page.get_texttrace():
        text = ''.join(chr(c[0]) for c in trace['chars'])
        if text in ids and abs(trace['size'] - 9.5) < 1e-4:
            frames.append(dict(region_id=text, bbox=list(trace['bbox']), font_bp=trace['size']))
    assert len(frames) == 89 and {x['region_id'] for x in frames} == set(ids)
    assert all(abs(x['font_bp'] - 9.5) < 1e-4 for x in frames)
    distances = []
    for i, first in enumerate(frames):
        for second in frames[i + 1:]:
            a, b = first['bbox'], second['bbox']
            distances.append(math.hypot(max(a[0] - b[2], b[0] - a[2], 0),
                                        max(a[1] - b[3], b[1] - a[3], 0)))
    assert min(distances) > 1.19, min(distances)
    return dict(count=89, font_bp=9.5, minimum_character_frame_gap_bp=min(distances))


def shared_legend(work):
    helper = module(HERE / 'source/display_helpers.py', 'pair_display_fonts')
    plt = helper.plotting(work)
    from matplotlib.patches import Circle, Rectangle
    import matplotlib.patheffects as pe
    fig = plt.figure(figsize=(MAP_WIDTH_MM / 25.4, 14 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, MAP_WIDTH_MM), ylim=(0, 14))
    ax.axis('off')
    ax.text(32, 11.4, '分配前田间储水深度（mm）', ha='center', va='center', fontsize=10.5)
    # The complete 80 tick frame and 4000 value frame align with the map border.
    frame_left, frame_right = 5 * SCALE, 151 * SCALE
    left, right = frame_left + 5.25 / MM, 58.
    step = (right - left) / 5
    for i, color in enumerate(('#fff9df', '#ffedc5', '#ffdc9c', '#ffc775', '#f6ac52')):
        ax.add_patch(Rectangle((left + i * step, 6.1), step, 1.7, fc=color, ec='none'))
    ax.add_patch(Rectangle((left, 6.1), right - left, 1.7, fc='none', ec='#65725e', lw=.8))
    for i, value in enumerate((80, 100, 120, 140, 160, 180)):
        ax.text(left + i * step, 3.5, str(value), ha='center', va='center', fontsize=10.5)
    ax.text(95.5, 11.4, '各区当天分得灌溉水量（m³）', ha='center', va='center', fontsize=10.5)
    examples = []
    last_radius = 1.6 * SCALE * 2
    last_center = frame_right - 21 / MM - 1.2 - last_radius
    for x, value in ((70., 125), (91.5, 1000), (last_center, 4000)):
        radius = 1.6 * SCALE * math.sqrt(value / 1000)
        circle = Circle((x, 4.8), radius, fc='none', ec='#A80058', lw=1.5)
        circle.set_path_effects([pe.Stroke(linewidth=3., foreground='white'), pe.Normal()])
        ax.add_patch(circle)
        ax.text(frame_right if value == 4000 else x + radius + 1.2, 4.8, str(value),
                ha='right' if value == 4000 else 'left', va='center', fontsize=10.5)
        examples.append(dict(value_m3=value, radius_mm=radius))
    fig.canvas.draw()
    for item in ax.texts:
        box = item.get_window_extent(fig.canvas.get_renderer())
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), item.get_text()
    fig.savefig(work / 'shared_legend.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    return examples


def centered_headers(work):
    helper = module(HERE / 'source/display_helpers.py', 'centered_header_fonts')
    plt = helper.plotting(work)
    paths = {}
    for day, title in TITLES.items():
        fig = plt.figure(figsize=(156 / 25.4, 6 / 25.4))
        fig.text(.5, .5, title, ha='center', va='center', fontsize=10.5)
        path = work / f'title_{day:02}.pdf'
        fig.savefig(path, metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)
        paths[day] = path
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixed = json.loads((HERE / 'data/mapping.json').read_text())
    source_inputs = sorted(p for p in (HERE / 'data').rglob('*') if p.is_file())
    protected = {str(p.relative_to(HERE)): sha(p) for p in source_inputs}
    from matplotlib.patches import Circle
    circles = []
    original = Circle.__init__

    def capture(self, *a, **kw):
        original(self, *a, **kw)
        if kw.get('ec') == '#A80058':
            circles.append(self)

    with tempfile.TemporaryDirectory(prefix='.two-page-pairs-', dir=output) as directory:
        work = Path(directory)
        (work / 'audit').mkdir()
        renderer = module(LAYER / 'source/render_maps.py', 'two_page_current_maps')
        assert renderer.GEOMETRY_SCALE == SCALE
        renderer.ROOT = work
        argv = sys.argv[:]
        Circle.__init__ = capture
        try:
            sys.argv = [str(__file__), '--figure', '6.4']
            renderer.main()
        finally:
            Circle.__init__ = original
            sys.argv = argv
        stage = work / 'figures/stage_overlay'
        circle_records = []
        index = 0
        for day in DAYS:
            for row in (r for r in fixed['rows'] if r['day'] == day):
                record = {k: row[k] for k in ('day', 'region_id', 'allocation_m3',
                          'beginning_storage_mm', 'circle_x', 'circle_y', 'radius_geometry_units', 'drawn')}
                if row['drawn']:
                    circle = circles[index]
                    index += 1
                    assert max(abs(circle.center[j] - row[('circle_x', 'circle_y')[j]]) for j in (0, 1)) < 1e-10
                    assert abs(circle.radius - row['radius_geometry_units']) < 1e-10
                    record['actual_radius_mm'] = circle.radius * fixed['mm_per_geometry_unit'] * SCALE
                circle_records.append(record)
        assert len(circle_records) == 356 and len(circles) - index == 3
        examples = shared_legend(work)
        header_paths = centered_headers(work)
        geometry = {}
        maps = {day: fitz.open(stage / f'day{day:02}.pdf') for day in DAYS}
        headers = {day: fitz.open(header_paths[day]) for day in DAYS}
        legend = fitz.open(work / 'shared_legend.pdf')
        lb = ink(legend[0])
        for key, days in (('ab', (0, 46)), ('cd', (62, 72))):
            boxes = [ink(maps[day][0]) for day in days]
            header_boxes = [ink(headers[day][0]) for day in days]
            width, height = 156 * MM, 227 * MM
            top, bottom = 1.5 * MM, 2 * MM
            title_map_gap, between_panels = 5., 9.5
            legend_gap = (height - top - bottom - sum(x.height for x in boxes)
                          - sum(x.height for x in header_boxes) - lb.height
                          - 2 * title_map_gap - between_panels)
            assert legend_gap >= 10, ('Insufficient legend spacing', legend_gap)
            doc = fitz.open()
            page = doc.new_page(width=width, height=height)
            y = top
            placements = []
            for index, (day, box, hb) in enumerate(zip(days, boxes, header_boxes)):
                title_clip = fitz.Rect(0, hb.y0, width, hb.y1)
                page.show_pdf_page(fitz.Rect(0, y, width, y + hb.height), headers[day],
                                   0, clip=title_clip, keep_proportion=True)
                assert abs((hb.x0 + hb.x1) / 2 - width / 2) < .25
                title_target = [0, y, width, y + hb.height]
                y += hb.height + title_map_gap
                src = maps[day]
                clip = fitz.Rect(0, box.y0, src[0].rect.width, box.y1)
                x = (width - clip.width) / 2
                target = fitz.Rect(x, y, x + clip.width, y + clip.height)
                page.show_pdf_page(target, src, 0, clip=clip, keep_proportion=True)
                placements.append(dict(day=day, target=list(target), source_clip=list(clip),
                                       title=TITLES[day], title_target=title_target,
                                       identifiers=identifier_frames(src[0], fixed['ids'])))
                y += clip.height + (between_panels if index == 0 else legend_gap)
            clip = fitz.Rect(0, lb.y0, legend[0].rect.width, lb.y1)
            legend_x = (width - clip.width) / 2
            page.show_pdf_page(fitz.Rect(legend_x, y, legend_x + clip.width, y + clip.height),
                               legend, 0, clip=clip, keep_proportion=True)
            frame_left = (156 - MAP_WIDTH_MM) / 2 * MM + 5 * SCALE * MM
            frame_right = (156 - MAP_WIDTH_MM) / 2 * MM + 151 * SCALE * MM
            assert abs(legend_x + lb.x0 - frame_left) < .1
            assert abs(legend_x + lb.x1 - frame_right) < .1
            assert abs(height - y - clip.height - bottom) < .01
            name = 'paired_' + key
            doc.set_metadata({'creator': 'Native two-date maps, fixed data and shared legend',
                              'producer': 'PyMuPDF vector composition, native 9.5/10.5 bp text'})
            doc.save(output / (name + '.pdf'), garbage=4, deflate=True, no_new_id=True)
            doc.close()
            with fitz.open(output / (name + '.pdf')) as final:
                spans = [s for b in final[0].get_text('dict')['blocks'] for l in b.get('lines', [])
                         for s in l['spans'] if s['text'].strip()]
                fonts = sorted({round(s['size'], 3) for s in spans})
                assert set(fonts) <= {9.5, 10.5}, fonts
                assert len([t for t in final[0].get_texttrace()
                            if abs(t['size'] - 9.5) < 1e-4
                            and ''.join(chr(c[0]) for c in t['chars']) in fixed['ids']]) == 178
                final[0].get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(output / (name + '.png'))
                (output / (name + '.svg')).write_text(final[0].get_svg_image(text_as_path=False))
            geometry[name] = dict(canvas_mm=[156, 227], map_geometry_scale=SCALE,
                                  maps=placements, title_to_map_gap_bp=title_map_gap,
                                  first_map_to_second_title_gap_bp=between_panels,
                                  second_map_to_legend_gap_bp=legend_gap,
                                  map_border_horizontal_bp=[frame_left, frame_right],
                                  legend_complete_horizontal_bp=[legend_x + lb.x0, legend_x + lb.x1],
                                  shared_legend_complete_ink_height_mm=lb.height / MM,
                                  actual_font_sizes_bp=fonts,
                                  outputs_sha256={e: sha(output / (name + '.' + e)) for e in ('pdf', 'png', 'svg')})
        for item in maps.values():
            item.close()
        for item in headers.values():
            item.close()
        legend.close()
    assert protected == {p: sha(HERE / p) for p in protected}
    proof = dict(schema='c6-two-page-centered-titles-v1', status='PASS', geometry=geometry,
                 circle_records=circle_records, circle_legend_examples=examples,
                 scientific_data_unchanged=True, geographic_world_geometry_unchanged=True,
                 uniform_paper_scale_relative_to_original156=SCALE,
                 uniform_paper_scale_relative_to_previous=SCALE / .875, region_ID_font_bp=9.5,
                 major_font_bp=10.5, source_data_sha256=protected,
                 source_files_sha256={str(p.relative_to(HERE)): sha(p) for p in
                                      (Path(__file__), LAYER / 'source/render_maps.py',
                                       LAYER / 'label_adjustments.json', LAYER / 'origin.json')},
                 scientific_execution=0)
    (output / 'centered_pairs_proof.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'output': str(output), 'geometry': geometry}, ensure_ascii=False))


if __name__ == '__main__':
    main()
