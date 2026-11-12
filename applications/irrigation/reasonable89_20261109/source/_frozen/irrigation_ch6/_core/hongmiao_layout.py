"""Published Hongmiao outlet order and rice area, drawn without invented GIS.

Rectangles represent outlet service units, not individual fields. Row wrapping
is only typography. Both rectangle dimensions share global scale factors, so
rectangle area / published rice area is identical across every row.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from importlib.resources import files
import json
import math
from numbers import Real

import numpy as np


LAYOUT_ID = "hongmiao93_published_chainage_area_schematic"
LAYOUT_NOTE = "1条洪庙支渠，分4行连续展开；11对同桩号分水口在同一沿渠位置向图上两侧独立引出，尾端为明渠。矩形大小与水稻面积成比例，两侧、折返、间距和渠宽均为示意，不表示实际岸别、走向或边界。"
DISPLAY_RESOURCE = "data/display/hongmiao93_opposite_banks.json"
REFERENCE_WIDTH, REFERENCE_HEIGHT = 2240., 1302.


@lru_cache(maxsize=1)
def _source_rows():
    path = files("irrigation_ch6._core").joinpath("data/hongmiao93/regions_93.csv")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        km, metre = row["section_code"].split("+")
        area = float(row["midseason_rice_area_mu"])
        result.append((row["gate_id"], row["outlet_name"], row["section_code"],
                       int(km) * 1000 + int(metre), area, int(row["source_order"]),
                       int(row["source_pdf_page"])))
    if len(result) != 93 or len({r[0] for r in result}) != 93 or any(r[4] <= 0 for r in result):
        raise ValueError("invalid packaged Hongmiao identity/area table")
    return tuple(result)


def matches_hongmiao(region_ids, areas_ha):
    """Recognize this published case by all identities and matching areas."""
    ids = list(region_ids)
    if len(ids) != 93 or len(set(ids)) != 93:
        return False
    source = {r[0]: r[4] / 15. for r in _source_rows()}
    areas = np.asarray(areas_ha, dtype=float)
    return (set(ids) == set(source) and areas.shape == (93,)
            and np.allclose(areas, [source[i] for i in ids], rtol=0., atol=1e-9))


@lru_cache(maxsize=1)
def _approved_rows():
    """Load fixed display geometry and cross-check published identities/areas.

    The display resource contains no actions, water states or case parameters.
    Its above/below arrangement is an approved schematic, not measured banks.
    """
    payload = json.loads(files("irrigation_ch6._core").joinpath(DISPLAY_RESOURCE).read_text(encoding="utf-8"))
    if (payload.get("schema") != "hongmiao93-opposite-banks-display-v1"
            or payload.get("width") != REFERENCE_WIDTH
            or payload.get("height") != REFERENCE_HEIGHT):
        raise ValueError("unsupported Hongmiao display geometry")
    rows = payload.get("regions", [])
    source = sorted(_source_rows(), key=lambda row: (row[3], row[5]))
    if len(rows) != len(source):
        raise ValueError("Hongmiao display must contain all published service units")
    ratios, sections = [], {}
    for i, (row, published) in enumerate(zip(rows, source)):
        identity, _, section, _, area, _, _ = published
        expected_label = "尾端明渠" if identity == "terminal" else f"{identity}#"
        if (row.get("index") != i or row.get("region_id") != identity
                or row.get("label") != expected_label or row.get("section_code") != section
                or row.get("area_mu") != area or row.get("row") not in range(4)
                or row.get("side") not in ("above", "below")
                or "allocation_m3" in row):
            raise ValueError("Hongmiao display identity, area or ordering disagrees with published data")
        rect = np.asarray(row.get("rect"), dtype=float)
        if rect.shape != (4,) or not np.isfinite(rect).all() or np.any(rect[2:] <= 0):
            raise ValueError("invalid Hongmiao display rectangle")
        x, y, width, height = rect
        cx, cy = row["cx"], row["cy"]
        side = 1 if row["side"] == "below" else -1
        edge = y if side == 1 else y + height
        if (not math.isfinite(cx) or not math.isfinite(cy)
                or not math.isclose(x + width / 2, cx, abs_tol=1e-10)
                or not math.isclose(edge, cy + side * 46., abs_tol=1e-10)
                or x < 0 or y < 0 or x + width > REFERENCE_WIDTH
                or y + height > REFERENCE_HEIGHT):
            raise ValueError("invalid Hongmiao display connector or bounds")
        ratios.append(width * height / area)
        sections.setdefault(section, []).append(row)
    if not np.allclose(ratios, ratios[0], rtol=1e-14, atol=0.):
        raise ValueError("Hongmiao display rectangles must preserve published area ratios")
    pairs = [group for group in sections.values() if len(group) > 1]
    if len(sections) != 82 or len(pairs) != 11:
        raise ValueError("Hongmiao display requires eleven same-section pairs")
    for pair in pairs:
        if (len(pair) != 2 or {row["side"] for row in pair} != {"above", "below"}
                or len({(row["row"], row["cx"], row["cy"]) for row in pair}) != 1):
            raise ValueError("same-section outlets must share one coordinate on opposite schematic sides")
    return rows


def build_hongmiao_layout(region_ids, areas_ha, *, width=1208., height=640., rows=4):
    """Return the approved geometry, scaled independently in x and y.

    Regions retain input-vector indices. Positions attach to published IDs,
    and repeated section codes share a canal coordinate with independent
    connectors above and below. These sides never imply actual bank identity.
    """
    ids = list(region_ids)
    if not matches_hongmiao(ids, areas_ha):
        raise ValueError("Hongmiao layout requires all 93 published IDs and matching rice areas")
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
            or not math.isfinite(v) or v <= 0 for v in (width, height)) or rows != 4):
        raise ValueError("Hongmiao drawing needs positive dimensions and four display rows")
    sx, sy = float(width) / REFERENCE_WIDTH, float(height) / REFERENCE_HEIGHT
    scale = min(sx, sy)
    point = lambda x, y: [float(x) * sx, float(y) * sy]
    rectangle = lambda x, y, w, h: [float(x) * sx, float(y) * sy, float(w) * sx, float(h) * sy]
    by_id = {identity: i for i, identity in enumerate(ids)}
    source = {row[0]: row for row in _source_rows()}
    approved = _approved_rows()
    regions, groups = [None] * 93, {}
    for row in approved:
        groups.setdefault(row["section_code"], []).append(row)
    for row in approved:
        identity = row["region_id"]
        _, name, code, chainage, area, order, page = source[identity]
        index = by_id[identity]
        x, y, w, h = row["rect"]
        cx, cy = row["cx"], row["cy"]
        side = 1 if row["side"] == "below" else -1
        edge = y if side == 1 else y + h
        pad = min(3., max(1.6, w * .035))
        bar_height = min(5.2, max(3.2, h * .11))
        outside = w < 43 or h < 24
        label_y = (y + h + 14 if side == 1 else y - 6) if outside else y + (h - 7) / 2 + 4
        tip = cy + side * 38.
        head = tip - side * 3.8
        junction, center = point(cx, cy), point(cx, cy + side * 8)
        regions[index] = {
            "index": index, "region_id": identity, "label": row["label"],
            "outlet_name": name, "area_ha": area / 15., "area_mu": area,
            "section_code": code, "chainage_m": chainage,
            "source_order": order, "source_pdf_page": page, "row": row["row"],
            "side": row["side"], "cx": cx * sx, "cy": cy * sy,
            "rect": rectangle(x, y, w, h),
            "same_section_ids": [item["region_id"] for item in groups[code]],
            "area_label": {"position": point(cx, label_y), "outside": outside,
                           "font_size": 12. * scale},
            "storage_bar": {"rect": rectangle(x + pad, y + h - pad - bar_height,
                                               w - 2 * pad, bar_height),
                            "corner_radius": scale},
            "outlet": {
                "kind": "open_terminal" if identity == "terminal" else "culvert_gate",
                "center": center, "size": 12.4 * scale,
                "half_width": 6.2 * sx, "half_height": 4.5 * sy,
                "channel_connection": junction,
                "service_path": [junction.copy(), center.copy(), point(cx, edge)],
                "connector_start": point(cx, cy + side * 5.5),
                "connector_width": 8.6 * sx,
                "supply_arrow": [point(cx - 2.6, head), point(cx, tip), point(cx + 2.6, head)],
                "supply_arrow_shaft": [point(cx, cy + side * 28), point(cx, tip)],
                "label_position": point(cx + 11, cy + 26 if side == 1 else cy - 18),
                "label_font_size": 16. * scale, "label_align": "start",
            },
        }
    guides, sections, pairs, arrows, channel_path = [], [], [], [], []
    for code, group in groups.items():
        item = {"section_code": code, "indices": [by_id[row["region_id"]] for row in group],
                "junction": point(group[0]["cx"], group[0]["cy"])}
        sections.append(item)
        if len(group) == 2:
            pairs.append({**item, "region_ids": [row["region_id"] for row in group],
                          "sides": [row["side"] for row in group]})
    for row_number in range(4):
        band = [row for row in approved if row["row"] == row_number]
        cy = band[0]["cy"]
        direction = 1 if row_number % 2 == 0 else -1
        start_x, end_x = (360., 2190.) if direction == 1 else (2190., 360.)
        channel_path.extend([point(start_x, cy), point(end_x, cy)])
        above = max((r["rect"][3] for r in band if r["side"] == "above"), default=0.)
        guides.append({"row": row_number, "x": 390. * sx,
                       "y": (cy - above - 46. - 22.) * sy, "direction": direction,
                       "text": f'{band[0]["section_code"]} — {band[-1]["section_code"]}'
                               + (" · 尾端" if row_number == 3 else ""),
                       "visible": row_number > 0, "font_size": 14. * scale,
                       "indices": [by_id[row["region_id"]] for row in band]})
        xs = sorted({row["cx"] for row in band})
        for fraction in (.22, .52, .8):
            index = min(len(xs) - 2, int((len(xs) - 1) * fraction))
            ax = (xs[index] + xs[index + 1]) / 2
            arrows.append([point(ax - 5 * direction, cy - 3.5),
                           point(ax + 5 * direction, cy), point(ax - 5 * direction, cy + 3.5)])
    node = {
        "center": point(165, 179),
        "diamond": [point(165, 132), point(224, 179), point(165, 226), point(106, 179)],
        "label": {"position": point(165, 113), "font_size": 18. * scale, "text": "总量配置"},
        "symbol": {"position": point(165, 191), "font_size": 33. * scale, "text": "Vₜ"},
        "value": {"position": point(165, 266), "font_size": 28. * scale},
        "equation": {"position": point(165, 310), "font_size": 22. * scale, "text": "Vₜ = Σᵢ aₜ,ᵢ"},
        "connector": [point(225, 179), point(342, 179)],
    }
    return {
        "identity": LAYOUT_ID, "layout_variant": "opposite_banks_v1",
        "width": float(width), "height": float(height),
        "reference_canvas": [REFERENCE_WIDTH, REFERENCE_HEIGHT],
        "total_area_ha": math.fsum(row[4] for row in source.values()) / 15.,
        "total_area_mu": math.fsum(row[4] for row in source.values()),
        "regions": regions, "display_order": [by_id[row["region_id"]] for row in approved],
        "row_guides": guides, "same_section_groups": [], "same_section_pairs": pairs,
        "channel_count": 1, "channel_name": "洪庙支渠", "channel_path": channel_path,
        "channel_arrows": arrows, "channel_sections": sections, "outlet_count": len(regions),
        "channel_color": "#B4D8E9", "channel_bank_color": "#3478A5",
        "channel_highlight_color": "#B4D8E9", "channel_arrow_color": "#286A92",
        "channel_width": 12. * scale, "channel_bank_width": 15. * scale,
        "channel_ripples": [], "total_allocation_node": node,
        "inlet": {"path": [point(360, 160), channel_path[0].copy()],
                  "gate_center": point(360, 179), "gate_size": 22. * scale,
                  "gate_half_width": 11. * sx, "gate_half_height": 4.5 * sy,
                  "label_position": point(406, 145), "label_font_size": 17. * scale,
                  "label_x": 406. * sx, "label_y": 145. * sy, "gate_label_y": 145. * sy,
                  "name": "洪庙支渠进水闸", "gate_name": "三分干18#涵闸"},
        "supply_chain": ["漳河水库", "总干渠", "三干渠", "三分干", "洪庙支渠"],
        "supply_source": "CN118521431B [0176]—[0182]",
        "service_example": {"gate": "1#涵闸", "channel": "毛渠", "service": "水稻灌溉区",
                            "note": "一个灌溉区可包含多块农田"},
        "topology_note": "同一支渠沿线口门服务关联；图上两侧与折返仅为排版，不加入渠道输水计算。",
        "pixel_area_per_mu": approved[0]["rect"][2] * approved[0]["rect"][3] / approved[0]["area_mu"] * sx * sy,
        "note": LAYOUT_NOTE, "source": "CN119514977A 表2，PDF第29—34页",
        "area_identity": "published_midseason_rice_area_not_measured_polygon",
        "tie_note": "同断面编码的两口在同一沿渠位置向图上两侧独立引出；上下不代表实际岸别。",
    }

