"""Align the accepted Chapter 3 conditions asset with the Figure 3.4 style.

The accepted renderer first completes its original source, parent pixel and
retained-series checks. This wrapper changes only final artist presentation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy
import sys

sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent.parent
THESIS=next(p for p in HERE.parents if (p/'assets/fonts/windows').is_dir())
ROOT=HERE
PARENT=HERE/'replot_sources/selection'
REFERENCE=HERE/'replot_sources/style_reference.json'
WIDTH, HEIGHT = 156.0, 164.0
LABELS = ('ReJoP', '去总量学习', 'C-PPO / Gradient PPO')
MARKERS = ('o', 's', '^')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'assets/figure36')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out==HERE or (out/'figure.pdf').exists():
        raise ValueError('Use a fresh output directory below this revision root')
    out.mkdir(parents=True, exist_ok=True)
    for key, leaf in [('MPLCONFIGDIR', 'mpl'), ('XDG_CACHE_HOME', 'xdg'), ('TMPDIR', 'tmp')]:
        folder = out/'runtime'/leaf
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    def plain(value):
        if isinstance(value, np.ndarray):
            return plain(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        return value

    def series(container):
        line, caps, bars = container.lines
        return dict(label=container.get_label(), x=plain(line.get_xdata()),
                    y=plain(line.get_ydata()), color=line.get_color(),
                    caps=[dict(xy=plain(c.get_xydata()), width=c.get_linewidth(),
                               marker=c.get_marker(), markersize=c.get_markersize()) for c in caps],
                    bars=[dict(segments=plain(b.get_segments()),
                               widths=plain(b.get_linewidths())) for b in bars])

    def numeric(fig):
        return [dict(xlim=plain(a.get_xlim()), ylim=plain(a.get_ylim()),
                     xticks=plain(a.get_xticks()), yticks=plain(a.get_yticks()),
                     xlabels=[t.get_text() for t in a.get_xticklabels()],
                     xlabel=a.get_xlabel(), ylabel=a.get_ylabel(), title=a.get_title(),
                     series=[series(c) for c in a.containers]) for a in fig.axes]

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                         allow_nan=False).encode()).hexdigest()

    original_save = Figure.savefig
    styled = set()
    proof, before_after = {}, {}

    def style(fig):
        assert len(fig.axes) == 4
        fig.canvas.draw()
        before = numeric(fig)
        before_after['before'] = before
        before_positions = [plain(a.get_position().bounds) for a in fig.axes]
        for legend in list(fig.legends):
            legend.remove()
        fig.set_layout_engine(None)
        fig.set_size_inches(WIDTH/25.4, HEIGHT/25.4, forward=True)
        fig.set_dpi(200)
        primary = []
        for i, ax in enumerate(fig.axes):
            x, y = (17.0 if i % 2 == 0 else 94.0), (105.0 if i < 2 else 29.0)
            ax.set_position([x/WIDTH, y/HEIGHT, 59.2/WIDTH, 51.0/HEIGHT])
            title = ax.get_title()
            ax.set_title('')
            ax.set_title(title, loc='center', pad=6, fontsize=10.5,
                         fontfamily=['Times New Roman', 'SimSun'], fontweight='normal')
            ax.xaxis.labelpad = 4.0
            ax.yaxis.labelpad = 4.0
            ax.tick_params(axis='both', which='major', length=3, width=.7,
                           pad=3, labelsize=10.5, direction='out')
            assert len(ax.containers) == 3
            for j, container in enumerate(ax.containers):
                line = container.lines[0]
                line.set_linestyle('-')
                line.set_linewidth(1.25)
                line.set_marker(MARKERS[j])
                line.set_markersize(4.0)
                line.set_markeredgewidth(.6)
                line.set_markerfacecolor(line.get_color())
                line.set_markeredgecolor(line.get_color())
                if i == 0:
                    primary.append(line)
        for text in fig.findobj(Text):
            text.set_fontsize(10.5)
            text.set_fontfamily(['Times New Roman', 'SimSun'])
            text.set_fontweight('normal')
        # The longer last categorical tick retains its wording and coordinate;
        # right alignment avoids exceeding the shared 156 mm canvas.
        fig.axes[3].get_xticklabels()[-1].set_ha('right')
        legend = fig.legend(primary, LABELS, loc='lower center',
                            bbox_to_anchor=(.5, 2.0/HEIGHT), ncol=3,
                            frameon=False, handlelength=1.8, handletextpad=.5,
                            columnspacing=1.25, labelspacing=.5,
                            borderpad=0, borderaxespad=0)
        for text in legend.get_texts():
            text.set_fontfamily(['Times New Roman', 'SimSun'])
            text.set_fontsize(10.5)
        fig.canvas.draw()
        after = numeric(fig)
        assert before == after, 'Numeric data, uncertainty or coordinate limits changed'
        before_after.update(after=after, before_sha256=digest(before), after_sha256=digest(after))
        ren = fig.canvas.get_renderer()
        k = 72/fig.dpi

        def bb(artist):
            return plain(np.asarray(artist.get_window_extent(ren).bounds)*k)

        hidden = set()
        for ax in fig.axes:
            for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
                for tick in axis.get_major_ticks():
                    if not min(limits)-1e-8 <= tick.get_loc() <= max(limits)+1e-8:
                        hidden.update((id(tick.label1), id(tick.label2)))
        texts = []
        for text in fig.findobj(Text):
            if not text.get_visible() or not text.get_text() or id(text) in hidden:
                continue
            bounds = bb(text)
            texts.append(dict(text=text.get_text(), font_bp=text.get_fontsize(), bbox_bp=bounds))
            assert bounds[0] >= -.5 and bounds[1] >= -.5 and bounds[0]+bounds[2] <= WIDTH*72/25.4+.5 and bounds[1]+bounds[3] <= HEIGHT*72/25.4+.5, ('Text outside canvas', texts[-1])
        lb = legend.get_window_extent(ren)
        xlabel_bottom = min(ax.xaxis.label.get_window_extent(ren).y0 for ax in fig.axes[2:])
        assert xlabel_bottom > lb.y1, 'Legend overlaps lower axis labels'
        proof.update(schema='chapter3-unified-conditions-style-v1', status='PASS',
                     reference=str(REFERENCE), reference_sha256=sha(REFERENCE),
                     figure_mm=[WIDTH, HEIGHT], axes_mm=[59.2, 51.0],
                     axes_position_before=before_positions,
                     axes_position_mm=[plain(np.asarray(ax.get_position().bounds)*np.array([WIDTH,HEIGHT,WIDTH,HEIGHT])) for ax in fig.axes],
                     primary_line_bp=1.25, primary_lines_solid=True,
                     filled_marker_pt=4.0, markers=list(MARKERS),
                     marker_edge_pt=.6, font_bp=10.5, title_pad_bp=6,
                     axis_label_pad_bp=4, tick_length_width_pad_bp=[3,.7,3],
                     legend_columns=3, legend_rows=1, legend_labels=list(LABELS),
                     legend_bbox_bp=bb(legend), legend_bottom_mm=float(lb.y0*k*25.4/72),
                     xlabel_to_legend_ink_bp=float((xlabel_bottom-lb.y1)*k),
                     long_last_tick_alignment='right; text and tick coordinate unchanged',
                     all_text_inside_canvas=True, texts=texts,
                     numeric_before_sha256=digest(before), numeric_after_sha256=digest(after),
                     all_means_sd_errorbar_segments_and_widths_exact=True,
                     series_count=12, scalar_points=36,
                     domain_reference={'radio':'C-PPO','diesel':'Gradient PPO'},
                     environment_steps=0, model_loads=0, policy_inferences=0,
                     training_steps=0, thesis_builds=0)

    def save(fig, destination, *save_args, **save_kwargs):
        dest = Path(destination)
        # The inherited renderer must reproduce and audit its parent before any
        # presentation changes. Only final output files pass through style().
        if dest.name in ('figure.pdf', 'figure.png', 'figure.svg'):
            if id(fig) not in styled:
                style(fig)
                styled.add(id(fig))
        return original_save(fig, destination, *save_args, **save_kwargs)

    Figure.savefig = save
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(PARENT/'plot.py'), '--asset', str(PARENT), '--output-dir', str(out)]
        runpy.run_path(str(PARENT/'plot.py'), run_name='__main__')
    finally:
        Figure.savefig = original_save
        sys.argv = previous_argv
    assert len(styled) == 1
    # The inherited checks describe the exact parent stage, not the derivative.
    for name in ['retained_artists.json', 'presentation_verification.json']:
        path = out/'audit'/name
        path.rename(out/'audit'/('inherited_parent_stage_'+name))
    (out/'render_receipt.json').rename(out/'audit/inherited_parent_stage_render_receipt.json')
    write(out/'artist_values_before_after.json', before_after)
    write(out/'render_geometry.json', proof)
    # Scientific scalar files are copied exactly from the accepted parent after
    # its independently reproduced aggregation has passed the same data checks.
    for name in ['data.csv.gz','display_aggregates.csv','display_aggregates.csv.gz',
                 'display_seed_data.csv','full_minus_pace.csv','full_minus_pace.json']:
        p, q = PARENT/name, out/name
        if name.endswith('.gz'):
            import gzip
            assert gzip.decompress(p.read_bytes()) == gzip.decompress(q.read_bytes()), name
        else:
            assert p.read_bytes() == q.read_bytes(), name
        if q.is_symlink():
            q.unlink()
        q.write_bytes(p.read_bytes())
    (out/'plot.py').write_text('from pathlib import Path\nimport runpy,sys\nsys.argv=[str(Path(__file__))]+sys.argv[1:]\nrunpy.run_path('+repr(str(Path(__file__).resolve()))+',run_name="__main__")\n')
    write(out/'source.json', dict(schema='chapter3-conditions-style-derivative-v1',
          number='3.6', parent_directory=str(PARENT), parent_pdf_sha256=sha(PARENT/'figure.pdf'),
          parent_renderer_sha256=sha(PARENT/'plot.py'), maintained_renderer=str(Path(__file__).resolve()),
          maintained_renderer_sha256=sha(__file__), reference_geometry=str(REFERENCE),
          reference_geometry_sha256=sha(REFERENCE),
          scientific_data_sha256={n:sha(out/n) for n in ['data.csv.gz','display_aggregates.csv','display_aggregates.csv.gz','display_seed_data.csv','full_minus_pace.csv','full_minus_pace.json']},
          numeric_before_sha256=proof['numeric_before_sha256'], numeric_after_sha256=proof['numeric_after_sha256'],
          domain_reference={'radio':'C-PPO','diesel':'Gradient PPO'},
          changes=['59.2x51mm axes in 156x164mm canvas','1.25bp solid means with filled distinct 4pt markers','one-row three-column legend bottom2mm','title, axis label and tick padding aligned to Figure3.4'],
          errorbar_numeric_values_and_widths_changed=False, environment_steps=0, model_loads=0,
          policy_inferences=0, training_steps=0, thesis_builds=0))
    write(out/'render_receipt.json', dict(status='PASS', renderer_sha256=sha(__file__),
          outputs_sha256={n:sha(out/n) for n in ['figure.pdf','figure.png','figure.svg','render_geometry.json','artist_values_before_after.json','source.json','plot.py']},
          inherited_checks_stage='Exact accepted parent, before derivative styles',
          environment_steps=0, model_loads=0, policy_inferences=0, training_steps=0, thesis_builds=0))
    print(json.dumps(dict(status='PASS', output=str(out), figure_mm=[WIDTH,HEIGHT],
                         legend_bottom_mm=proof['legend_bottom_mm']), ensure_ascii=False))


if __name__ == '__main__':
    main()
