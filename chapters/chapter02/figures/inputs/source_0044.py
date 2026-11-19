"""Patch only the approved 2026-11-19 SVG labels after a native plot export.

No plotting data, axes, paths, fonts, y positions, or other XML are changed.
Editable text keeps all attributes.  Matplotlib glyph text keeps its center and
baseline by removing four 100-font-unit advances and moving its parent by 21pt.
"""
from pathlib import Path
from decimal import Decimal
import copy
import hashlib
import re
import xml.etree.ElementTree as ET

OLD_AXIS = '训练环境交互步数（×10³）'
NEW_AXIS = '训练步数（×10³）'
OLD_LEGEND = '比例映射'
NEW_LEGEND = '按比例分配'
REMOVED_GLYPHS = {'SimSun-73af', 'SimSun-5883', 'SimSun-4ea4', 'SimSun-4e92'}
NS = '{http://www.w3.org/2000/svg}'
USE = re.compile(r'<use\b[^>]*/>')
TRANSLATE = re.compile(r'translate\(([-\d.]+) ([-\d.]+)\)')
PARENT = re.compile(r'<g transform="translate\(([-\d.]+) ([-\d.]+)\) scale\(([-\d.]+) ([-\d.]+)\)">')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _num(value):
    text = format(value, 'f')
    return text.rstrip('0').rstrip('.') if '.' in text else text


def _parse(text):
    return ET.fromstring(text, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))


def _signature(node, targets=()):
    tag = node.tag if isinstance(node.tag, str) else '#comment'
    attrs = tuple(sorted(node.attrib.items()))
    if node.get('id') in targets:
        return ('APPROVED_TEXT_GROUP', tag, attrs)
    return (tag, attrs, node.text or '', node.tail or '',
            tuple(_signature(child, targets) for child in node))


def _all_paths(root):
    return [tuple(sorted(node.attrib.items())) for node in root.iter()
            if isinstance(node.tag, str) and node.tag == NS + 'path']


def _group_fragment(text, comment_start):
    candidates = list(re.finditer(r'<g id="(text_\d+)">', text[:comment_start]))
    if not candidates:
        raise ValueError('No enclosing Matplotlib text group')
    opening = candidates[-1]
    depth = 0
    for token in re.finditer(r'<g\b[^>]*>|</g\s*>', text[opening.start():]):
        depth += -1 if token[0].startswith('</') else 1
        if depth == 0:
            end = opening.start() + token.end()
            return opening.start(), end, opening[1], text[opening.start():end]
    raise ValueError('Unclosed SVG target text group')


def _patch_glyph_group(fragment):
    parent = PARENT.search(fragment)
    if not parent or parent.groups()[2:] != ('0.105', '-0.105'):
        raise ValueError('Unexpected glyph scale/parent transform')
    before_uses = USE.findall(fragment)
    if len(before_uses) != 14:
        raise ValueError('Expected fourteen source glyph uses')
    removed = []
    expected = []
    for use in before_uses:
        href = re.search(r'xlink:href="#([^"]+)"', use)
        if not href:
            raise ValueError('Missing glyph href')
        if href[1] in REMOVED_GLYPHS:
            removed.append(href[1])
            continue
        move = TRANSLATE.search(use)
        if move and Decimal(move[1]) >= 600:
            if Decimal(move[2]) != 0:
                raise ValueError('Unexpected glyph y shift')
            x = _num(Decimal(move[1]) - 400)
            use = use[:move.start(1)] + x + use[move.end(1):]
        expected.append(use)
    if len(removed) != 4 or set(removed) != REMOVED_GLYPHS:
        raise ValueError('Source label does not contain four expected glyphs')
    # Replace individual uses, preserving path definitions and surrounding XML.
    remaining = iter(expected)
    def change_use(match):
        href = re.search(r'xlink:href="#([^"]+)"', match[0])[1]
        return '' if href in REMOVED_GLYPHS else next(remaining)
    result = USE.sub(change_use, fragment)
    old_x = Decimal(parent[1]); new_x = old_x + 21
    # There are no preceding length changes before the first parent transform.
    result = result[:parent.start(1)] + _num(new_x) + result[parent.end(1):]
    result = result.replace(OLD_AXIS, NEW_AXIS)
    after_parent = PARENT.search(result)
    assert after_parent.groups()[1:] == parent.groups()[1:]
    assert USE.findall(result) == expected
    # Font definitions and glyph outlines are byte-identical.
    defs = lambda t: re.findall(r'<defs\b[^>]*>[\s\S]*?</defs>', t)
    assert defs(fragment) == defs(result)
    assert Decimal(after_parent[1]) - old_x == 21
    return result, {'old_parent_x_pt': str(old_x), 'new_parent_x_pt': str(new_x),
                    'parent_y_pt': parent[2], 'scale_unchanged': list(parent.groups()[2:]),
                    'removed_glyphs': removed, 'source_glyph_uses': 14,
                    'target_glyph_uses': len(expected), 'removed_advance_font_units': 400,
                    'remaining_glyph_x_shift_font_units': -400,
                    'center_delta_pt': '0', 'baseline_delta_pt': '0',
                    'font_definitions_byte_unchanged': True}


def patch_svg(svg_path):
    """Update an exported SVG in place and return JSON-serializable proof.

    Idempotent for SVGs whose approved labels have already been updated.
    A different or partial target layout is rejected before writing.
    """
    path = Path(svg_path)
    source_bytes = path.read_bytes()
    source = source_bytes.decode('utf-8')
    before_root = _parse(source)
    axis_count = source.count(OLD_AXIS)
    legend_count = source.count(OLD_LEGEND)
    if not axis_count:
        if source.count(NEW_AXIS) not in (2, 4) or legend_count:
            raise ValueError('No complete approved source/updated SVG labels')
        return {'svg': str(path), 'status': 'already_updated', 'axis_edits': 0,
                'legend_edits': 0, 'changed_axis_labels': 0, 'axis_count': source.count(NEW_AXIS),
                'changed_legend_labels': 0, 'sha256_before': _sha(source_bytes),
                'sha256_after': _sha(source_bytes), 'non_target_xml_unchanged': True,
                'all_path_geometry_unchanged': True, 'page_geometry_unchanged': True}
    if axis_count not in (2, 4) or legend_count not in (0, 1):
        raise ValueError('Unexpected source SVG target counts')
    parents = {child: parent for parent in before_root.iter() for child in parent}
    text_nodes = [node for node in before_root.iter()
                  if isinstance(node.tag, str) and node.tag == NS + 'text'
                  and node.text in (OLD_AXIS, OLD_LEGEND)]
    targets = {parents[node].get('id') for node in text_nodes}
    if None in targets:
        raise ValueError('Editable target text has no identified group')
    glyph_proof = []
    result = source
    if '<!-- ' + OLD_AXIS + ' -->' in source:
        if axis_count != 4 or legend_count or text_nodes:
            raise ValueError('Unexpected mixed glyph/editable SVG targets')
        # Work from right to left so source offsets remain valid.
        comments = list(re.finditer(re.escape('<!-- ' + OLD_AXIS + ' -->'), source))
        for comment in reversed(comments):
            start, end, ident, fragment = _group_fragment(source, comment.start())
            patched, proof = _patch_glyph_group(fragment)
            result = result[:start] + patched + result[end:]
            targets.add(ident)
            glyph_proof.append({'text_group_id': ident, **proof})
        mode = 'glyph_paths'
    else:
        if len([node for node in text_nodes if node.text == OLD_AXIS]) != axis_count:
            raise ValueError('Expected editable SVG axis text elements')
        if len([node for node in text_nodes if node.text == OLD_LEGEND]) != legend_count:
            raise ValueError('Expected editable SVG legend text element')
        result = source.replace(OLD_AXIS, NEW_AXIS).replace(OLD_LEGEND, NEW_LEGEND)
        mode = 'editable_text'
    assert OLD_AXIS not in result and OLD_LEGEND not in result
    after_root = _parse(result)
    assert before_root.attrib == after_root.attrib
    assert _signature(before_root, targets) == _signature(after_root, targets)
    assert _all_paths(before_root) == _all_paths(after_root)
    if mode == 'editable_text':
        before_text = [node.attrib for node in before_root.iter() if node.tag == NS + 'text']
        after_text = [node.attrib for node in after_root.iter() if node.tag == NS + 'text']
        assert before_text == after_text
    target_bytes = result.encode('utf-8')
    proof = {'svg': str(path), 'status': 'updated', 'mode': mode,
             'axis_edits': axis_count, 'legend_edits': legend_count,
             'changed_axis_labels': axis_count, 'axis_count': axis_count,
             'changed_legend_labels': legend_count,
             'sha256_before': _sha(source_bytes), 'sha256_after': _sha(target_bytes),
             'target_group_ids': sorted(targets),
             'non_target_xml_unchanged': True, 'all_path_geometry_unchanged': True,
             'page_geometry_unchanged': True,
             'editable_text_attributes_unchanged': mode == 'editable_text',
             'glyph_groups': list(reversed(glyph_proof))}
    path.write_bytes(target_bytes)
    return proof
