#!/usr/bin/env python3
"""Render the local extracted C3 CSVs; no raw NPZ, policy, or environment reads."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1]/"cache"/"figure_3_8"/"mpl"))
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager, ticker
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
DEFAULT_DOMAIN = "v5"
FONT_DIR=Path(os.environ['THESIS_FIGURE_FONT_DIR'])
PALETTE = ["#F8FCFF", "#DDF1FB", "#B0DFF3", "#78C5E7"]
TEXT_COLOR = "#263445"
TASKS = {
    "radio": dict(N=100, H=32, periods=[8, 24], budget=.48,
                  external="c_ppo", external_label="C-PPO", unit="W", suffix="期",
                  task_label="通信功率分配", figure_height_cm=17.7),
    "v5": dict(N=120, H=24, periods=[6, 18], budget=20160.,
               external="gradient_ppo", external_label="Gradient PPO", unit="L", suffix="期",
               task_label="应急柴油分配", figure_height_cm=18.9),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_csv(path):
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def paths(domain):
    if domain != DEFAULT_DOMAIN:
        raise ValueError("This self-contained asset is for " + DEFAULT_DOMAIN)
    return {"maps": BASE / "data.csv", "totals": BASE / "period_totals.csv",
            "source": BASE / "data" / "provenance.json"}


def setup_fonts():
    for filename in ["TIMES.TTF", "simsun.ttc"]:
        font_manager.fontManager.addfont(str(FONT_DIR / filename))
    plt.rcParams.update({"font.family": ["Times New Roman", "SimSun"], "font.size": 10.5,
                         "axes.labelsize": 10.5, "axes.titlesize": 10.5,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                         "legend.fontsize": 10.5, "text.color": TEXT_COLOR,
                         "axes.labelcolor": TEXT_COLOR, "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.unicode_minus": False, "savefig.facecolor": "white", "svg.hashsalt": "formal-hex-c3-" + DEFAULT_DOMAIN})


def geometry(n):
    ids = np.arange(n)
    centers = np.column_stack([np.sqrt(3) * (ids % 10 + .5 * ((ids // 10) % 2)), -1.5 * (ids // 10)])
    theta = np.pi / 6 + np.arange(6) * np.pi / 3
    unit = np.column_stack([np.cos(theta), np.sin(theta)])
    polygons = centers[:, None, :] + unit[None, :, :]
    all_vertices = polygons.reshape(-1, 2)
    low, high = all_vertices.min(axis=0), all_vertices.max(axis=0)
    return centers, polygons, (float(low[0]), float(high[0]), float(low[1]), float(high[1]))


def render(domain, output_dir):
    cfg, p = TASKS[domain], paths(domain)
    maps, totals = read_csv(p["maps"]), read_csv(p["totals"])
    source = json.loads(p["source"].read_text())
    assert sha(p["maps"]) == source["map_actions_sha256"]
    assert sha(p["totals"]) == source["period_totals_sha256"]
    setup_fonts()
    centers, polygons, bounds = geometry(cfg["N"])
    xmin, xmax, ymin, ymax = bounds
    width, height, map_width = 15.6, cfg["figure_height_cm"], 5.8
    map_height = map_width * (ymax - ymin) / (xmax - xmin)
    lower_bottom = 1.70
    upper_bottom = lower_bottom + map_height + .75
    curve_bottom = upper_bottom + map_height + 2.10
    curve_height = height - .65 - curve_bottom
    assert curve_height > 1.5
    fig = plt.figure(figsize=(width / 2.54, height / 2.54), facecolor="white", dpi=240)
    rect = lambda x, y, w, h: [x / width, y / height, w / width, h / height]
    ax = fig.add_axes(rect(1.65, curve_bottom, 12.50, curve_height))
    curve_colors = {"A_FULL": "#0072B2", "c_ppo": "#D55E00", "gradient_ppo": "#009E73"}
    curve_markers = {"A_FULL": "o", "c_ppo": "s", "gradient_ppo": "D"}
    for method, label, line in [("A_FULL", "ReJoP（本文）", "-"), (cfg["external"], cfg["external_label"], "-")]:
        rr = sorted([row for row in totals if row["method"] == method], key=lambda row: int(row["period"]))
        xx = np.array([int(row["period"]) for row in rr])
        yy = np.array([float(row["q_percent_initial"]) for row in rr])
        assert list(xx) == list(range(1, cfg["H"] + 1))
        ax.plot(xx, yy, line, color=curve_colors[method], lw=1.25, label=label, zorder=3,
                marker=curve_markers[method], markevery=max(1, cfg["H"] // 8), markersize=3.6,
                markerfacecolor=curve_colors[method], markeredgecolor=curve_colors[method], markeredgewidth=.45)
    ax.set_xlim(.5, cfg["H"] + .5)
    ax.set_ylim(bottom=0)
    ax.set_xticks([1, cfg["H"] // 4, cfg["H"] // 2, cfg["H"] * 3 // 4, cfg["H"]])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    ax.set_xlabel("决策期" if domain == "radio" else "决策期", labelpad=2)
    ax.set_ylabel("本期投入占\n初始资源（%）", labelpad=5)
    ax.tick_params(width=.7, length=3, pad=2)
    for spine in ax.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    task_title = ax.text(.5, 1.02, "(a) 各期总投入", transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5)
    curve_legend = fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
                            bbox_to_anchor=((1.65 + 12.50 / 2) / width, (curve_bottom - 1.10) / height), ncol=2,
                            frameon=False, handlelength=1.7, columnspacing=.9,
                            handletextpad=.4, borderaxespad=0, borderpad=0)
    vmax = max(float(row["allocation_percent_initial"]) for row in maps)
    norm, cmap = colors.Normalize(0, vmax), colors.LinearSegmentedColormap.from_list("c3_blue", PALETTE)
    idfont = font_manager.FontProperties(fname=str(FONT_DIR / "TIMES.TTF"), size=7.5)
    panel_checks, id_artists = [], []
    for ri, (method, label) in enumerate([("A_FULL", "ReJoP"), (cfg["external"], cfg["external_label"])]):
        for ci, period in enumerate(cfg["periods"]):
            rr = sorted([row for row in maps if row["method"] == method and int(row["period"]) == period],
                        key=lambda row: int(row["object_id"]))
            assert [int(row["object_id"]) for row in rr] == list(range(1, cfg["N"] + 1))
            values = np.array([float(row["allocation_percent_initial"]) for row in rr])
            bottom = upper_bottom if ri == 0 else lower_bottom
            aa = fig.add_axes(rect([1.45, 8.35][ci], bottom, map_width, map_height))
            for index, (center, polygon, value) in enumerate(zip(centers, polygons, values), 1):
                aa.add_patch(Polygon(polygon, closed=True, facecolor="#FFFFFF" if value == 0 else cmap(norm(value)),
                                     edgecolor="#718695", linewidth=.35, clip_on=False, zorder=1))
                text = aa.text(*center, str(index), ha="center", va="center", color=TEXT_COLOR,
                               fontproperties=idfont, clip_on=False, zorder=4)
                id_artists.append((aa, text, center))
            aa.set_xlim(xmin, xmax)
            aa.set_ylim(ymin, ymax)
            aa.set_aspect("equal", adjustable="box")
            aa.axis("off")
            aa.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                                   edgecolor="#000000", linewidth=.75, clip_on=False, zorder=3))
            aa.set_title(f"({'bcde'[2 * ri + ci]}) {label} · 第{period}{cfg['suffix']}", fontsize=10.5, pad=3.5)
            period_row = next(row for row in totals if row["method"] == method and int(row["period"]) == period)
            discrepancy = abs(float(values.sum()) - float(period_row["q_percent_initial"]))
            assert discrepancy < 1e-10
            panel_checks.append(dict(row=ri, column=ci, method=method, method_label=label, period=period,
                                     N=cfg["N"], object_ids="1..N in original order", zero_objects=int((values == 0).sum()),
                                     object_share_sum=float(values.sum()), curve_total_share=float(period_row["q_percent_initial"]),
                                     sum_discrepancy=discrepancy, min=float(values.min()), max=float(values.max())))
    cax = fig.add_axes(rect(1.65, 1.20, 12.50, .20))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([0, vmax / 2, vmax])
    cb.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.3g}"))
    cb.ax.tick_params(labelsize=10.5, length=3, width=.65, pad=2)
    cb.set_label("对象投入占初始资源（%）", fontsize=10.5, labelpad=3)
    cb.outline.set_edgecolor("#000000")
    cb.outline.set_linewidth(.65)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_pt = 72 / fig.dpi
    header_clearance_pt = min(fig.bbox.y1 - task_title.get_window_extent(renderer).y1,
                              fig.bbox.y1 - curve_legend.get_window_extent(renderer).y1) * to_pt
    colorbar_label_clearance_pt = (cb.ax.xaxis.label.get_window_extent(renderer).y0 - fig.bbox.y0) * to_pt
    assert header_clearance_pt >= 3, (domain, "header margin", header_clearance_pt)
    assert colorbar_label_clearance_pt >= 3, (domain, "bottom label margin", colorbar_label_clearance_pt)
    normals = np.column_stack([np.cos(np.arange(6) * np.pi / 3), np.sin(np.arange(6) * np.pi / 3)])
    minimum_clearance = float("inf")
    for aa, text, center in id_artists:
        bb = text.get_window_extent(renderer=renderer)
        corners = aa.transData.inverted().transform([[bb.x0, bb.y0], [bb.x0, bb.y1], [bb.x1, bb.y0], [bb.x1, bb.y1]])
        clearance = float(np.min(np.sqrt(3) / 2 - (corners - center) @ normals.T))
        assert clearance > 0, (domain, text.get_text(), clearance)
        minimum_clearance = min(minimum_clearance, clearance * map_width / (xmax - xmin) * 72 / 2.54)
    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {"Date": None} if ext == "svg" else None
        fig.savefig(folder / f"figure.{ext}", dpi=240, metadata=metadata)
    plt.close(fig)
    with Image.open(folder / "figure.png") as image:
        image.convert("L").save(folder / "figure_grayscale.png")
    checks = dict(status="RENDERED_PENDING_FIVE_ROUND_REVIEW", domain=domain,
                  figure_size_cm=[width, height], fonts_bp={"main": 10.5, "ticks": 10.5, "object_ID": 7.5},
                  map_frame_size_cm=[map_width, map_height], map_columns=10,
                  full_hexagons_per_map=cfg["N"], all_four_maps_same_positions_and_original_IDs=True,
                  complete_regular_hexagons=True, clipped_or_stretched_hexagons=False,
                  minimum_actual_ID_bbox_to_hex_boundary_clearance_pt=minimum_clearance,
                  header_top_clearance_pt=header_clearance_pt,
                  colorbar_label_bottom_clearance_pt=colorbar_label_clearance_pt,
                  map_color_meaning="Current individual allocation / initial interface budget * 100; not cumulative.",
                  map_color_limits=[0, vmax], exact_zero_fill="#FFFFFF", positive_palette=PALETTE,
                  map_color_scale_shared_across_all_four_panels=True, curve_meaning="Current total allocation / initial budget * 100.",
                  full_period_count=cfg["H"], selected_periods_one_based=cfg["periods"], panels=panel_checks,
                  data_files={key: {"path": str(path), "sha256": sha(path)} for key, path in p.items()},
                  script_sha256=sha(__file__), outputs={ext: sha(folder / f"figure.{ext}") for ext in ["pdf", "svg", "png"]},
                  grayscale_sha256=sha(folder / "figure_grayscale.png"), model_loads=0,
                  new_environment_steps=0, new_policy_inferences=0, new_learning_steps=0)
    write_json(folder / "checks.json", checks)
    write_json(folder / "source.json", dict(
        editable_source=str(Path(__file__)), rerender_command=f"python plot.py --output-dir OUTPUT",
        source_data=str(p["source"]), source_data_sha256=sha(p["source"]), status="formal figure asset; offline saved-data rendering",
        caption=f"{'通信' if domain == 'radio' else '柴油'}任务固定情景中的跨期总量与对象投入",
        caption_note="既有训练种子0、默认测试情景0；两方法同外生情景。上方为完整周期本期总投入，下方为固定期限分位点的当期对象投入。四张地图共用同一线性色标，编号/位置固定，排列没有地理含义。地图值逐对象求和等于所示时点总量。单条情景不代替五种子总体评价。",
        raw_provenance=source["trajectories"], checks_file=str(folder / "checks.json"), checks_sha256=sha(folder / "checks.json")))
    print(json.dumps(dict(domain=domain, png=str(folder / "figure.png"), pdf=str(folder / "figure.pdf"),
                          size_cm=[width, height], vmax=vmax, minimum_ID_clearance_pt=minimum_clearance), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=[DEFAULT_DOMAIN], default=DEFAULT_DOMAIN)
    parser.add_argument("--output-dir", type=Path, default=BASE)
    args = parser.parse_args()
    domains = list(TASKS) if args.domain == "all" else [args.domain]
    for domain in domains:
        missing = [str(path) for path in paths(domain).values() if not path.exists()]
        if missing:
            raise FileNotFoundError("This renderer requires the delivered local CSV/source files: " + ", ".join(missing))
        render(domain, args.output_dir)


if __name__ == "__main__":
    main()
