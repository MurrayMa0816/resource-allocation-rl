#!/usr/bin/env python3
"""Split the current paired vector maps into four native, unscaled panels.

Only existing paired PDF components are composed. No plotting, scientific
runtime, model, or environmental calculation is invoked by this entry point.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

import fitz

HERE = Path(__file__).resolve().parent
MM = 72 / 25.4
SPECS = (('a', 'paired_ab', 0), ('b', 'paired_ab', 1),
         ('c', 'paired_cd', 0), ('d', 'paired_cd', 1))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_crop(page, rect):
    """Render each vector clip at the same local origin and physical scale."""
    rect = fitz.Rect(rect)
    doc = fitz.open()
    target = doc.new_page(width=rect.width + 2, height=rect.height + 2)
    target.show_pdf_page(fitz.Rect(1, 1, rect.width + 1, rect.height + 1),
                         page.parent, page.number, clip=rect, keep_proportion=False)
    pix = target.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
    result = (pix.width, pix.height, pix.samples)
    doc.close()
    return result


def pixel_comparison(parent, source, child, target):
    first = canonical_crop(parent, source)
    second = canonical_crop(child, target)
    assert first[:2] == second[:2], (first[:2], second[:2])
    differences = [abs(a - b) for a, b in zip(first[2], second[2])]
    maximum = max(differences, default=0)
    changed = sum(value != 0 for value in differences)
    return dict(scale=3, pixel_dimensions=list(first[:2]),
                identical=changed == 0, changed_channels=changed,
                max_channel_difference=maximum,
                parent_rgb_sha256=hashlib.sha256(first[2]).hexdigest(),
                single_rgb_sha256=hashlib.sha256(second[2]).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--source-dir', type=Path, default=HERE)
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pair_proof = source / 'centered_pairs_proof.json'
    accepted = json.loads(pair_proof.read_text())
    ids = set(json.loads((HERE / 'data/mapping.json').read_text())['ids'])
    protected = {str(p.relative_to(HERE)): sha(p)
                 for p in (HERE / 'data').rglob('*') if p.is_file()}
    protected['all89_four_dates_state_action.csv'] = sha(HERE / 'all89_four_dates_state_action.csv')
    parents = {name: sha(source / (name + '.pdf')) for name in ('paired_ab', 'paired_cd')}
    panels = {}
    for label, name, index in SPECS:
        geometry = accepted['geometry'][name]
        assert abs(geometry['map_geometry_scale'] - .83) < 1e-12
        assert parents[name] == geometry['outputs_sha256']['pdf']
        selected = geometry['maps'][index]
        parent = fitz.open(source / (name + '.pdf'))
        page = parent[0]
        width = page.rect.width
        title = fitz.Rect(selected['title_target']) & page.rect
        map_clip = fitz.Rect(selected['target'])
        last_map = fitz.Rect(geometry['maps'][-1]['target'])
        legend_y = last_map.y1 + geometry['second_map_to_legend_gap_bp']
        legend_height = geometry['shared_legend_complete_ink_height_mm'] * MM
        # Shared legend uses the same native canvas width as each map.
        legend = fitz.Rect(map_clip.x0, legend_y, map_clip.x1, legend_y + legend_height)
        y = 1.5 * MM
        title_target = fitz.Rect(0, y, width, y + title.height)
        y += title.height + geometry['title_to_map_gap_bp']
        map_target = fitz.Rect(map_clip.x0, y, map_clip.x1, y + map_clip.height)
        y += map_clip.height + geometry['second_map_to_legend_gap_bp']
        legend_target = fitz.Rect(legend.x0, y, legend.x1, y + legend.height)
        height = legend_target.y1 + 2 * MM
        doc = fitz.open()
        child = doc.new_page(width=width, height=height)
        parts = [('title', title, title_target), ('map', map_clip, map_target),
                 ('legend', legend, legend_target)]
        placements = {}
        for role, clip, target in parts:
            assert abs(clip.width - target.width) < 1e-7
            assert abs(clip.height - target.height) < 1e-7
            child.show_pdf_page(target, parent, 0, clip=clip, keep_proportion=False)
            placements[role] = dict(source_clip_bp=list(clip), target_bp=list(target),
                                    physical_scale_xy=[1., 1.])
        stem = output / ('single_' + label)
        doc.set_metadata(dict(creator='Unscaled vector components from current figure 6.4 paired PDFs',
                              producer='PyMuPDF native vector composition; no data or font scaling'))
        doc.save(stem.with_suffix('.pdf'), garbage=4, deflate=True, no_new_id=True)
        doc.close()
        with fitz.open(stem.with_suffix('.pdf')) as final:
            child = final[0]
            spans = [span for block in child.get_text('dict')['blocks']
                     for line in block.get('lines', []) for span in line['spans']
                     if span['text'].strip()]
            fonts = sorted({round(span['size'], 3) for span in spans})
            assert set(fonts) <= {9.5, 10.5}, fonts
            # Text traces expose clipped-away parent forms. Visible extraction
            # respects those clips and is therefore used to count identifiers.
            frames = [dict(region_id=word[4], bbox_bp=list(word[:4]), font_bp=9.5)
                      for word in child.get_text('words', clip=map_target)
                      if word[4] in ids]
            identifier_trace_sizes = {round(span['size'], 3)
                                      for block in child.get_text('dict', clip=map_target)['blocks']
                                      for line in block.get('lines', []) for span in line['spans']
                                      if span['text'].strip() and
                                      all(word in ids for word in span['text'].split())}
            assert identifier_trace_sizes == {9.5}, identifier_trace_sizes
            assert len(frames) == 89 and {item['region_id'] for item in frames} == ids, len(frames)
            comparisons = {role: pixel_comparison(page, clip, child, target)
                           for role, clip, target in parts}
            # PDF float coordinates may cause subpixel antialiasing differences;
            # preserve their measured magnitude rather than claiming identity.
            assert all(item['max_channel_difference'] <= 12 for item in comparisons.values()), comparisons
            assert not child.get_images(full=True), 'Native maps must remain vector-only'
            child.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(stem.with_suffix('.png'))
            stem.with_suffix('.svg').write_text(child.get_svg_image(text_as_path=False))
            panels[label] = dict(day=selected['day'], title=selected['title'],
                                 parent_pdf=name + '.pdf', parent_pdf_sha256=parents[name],
                                 canvas_mm=[child.rect.width / MM, child.rect.height / MM],
                                 map_geometry_scale=.83, native_font_sizes_bp=fonts,
                                 region_ID_count=89, region_ID_frames=frames,
                                 all_components_unscaled=True, placements=placements,
                                 vector_only=True, raster_component_checks=comparisons,
                                 outputs_sha256={extension: sha(stem.with_suffix('.' + extension))
                                                 for extension in ('pdf', 'svg', 'png')})
        parent.close()
        print(json.dumps(dict(panel=label, canvas_mm=panels[label]['canvas_mm'],
                              ID_count=89,
                              raster_identical={key: value['identical'] for key, value in comparisons.items()})))
    assert parents == {name: sha(source / (name + '.pdf')) for name in parents}
    assert protected == {name: sha(HERE / name) for name in protected}
    proof = dict(schema='c6-single-panels-unscaled-vector-v1', status='PASS', panels=panels,
                 source_pair_proof_sha256=sha(pair_proof), source_paired_PDFs_sha256=parents,
                 source_data_sha256=protected, data_unchanged=True, parent_PDFs_unchanged=True,
                 source_files_sha256={'plot_single_panels.py': sha(Path(__file__))},
                 scientific_execution=0, model_loads=0, environment_steps=0, thesis_builds=0)
    (output / 'single_panels_proof.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
