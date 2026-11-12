"""Balance a saved plot's shared legend; no observations or axes are changed."""
from pathlib import Path
import os
import sys

# The supported replot entry sets these before importing matplotlib.  Keep a
# safe fallback for direct local use as well.
_ROOT = Path(__file__).resolve().parents[2] if Path(__file__).parent.parent.name == 'assets' else Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(_ROOT/'cache'/'matplotlib'))
sys.dont_write_bytecode = True
CAPTION_EXTRA_PT = 9.6


def measure_spacing(fig, renderer):
    """Measure visible label/legend ink in physical points, independent of DPI."""
    from matplotlib.transforms import Bbox
    if len(fig.legends) != 1:
        raise ValueError('Expected one shared figure legend')
    legend = fig.legends[0]
    labels = [ax.xaxis.label.get_window_extent(renderer) for ax in fig.axes
              if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text()]
    if not labels:
        raise ValueError('No x-axis labels were found')
    objects = list(legend.get_texts())
    handles = getattr(legend, 'legend_handles', None)
    if handles is None:
        handles = getattr(legend, 'legendHandles', [])
    objects += list(handles)
    boxes = [obj.get_window_extent(renderer) for obj in objects if obj.get_visible()]
    boxes = [box for box in boxes if box.width >= 0 and box.height >= 0]
    ink = Bbox.union(boxes)
    full = legend.get_window_extent(renderer)
    scale = 72.0/fig.dpi
    xlabel_bottom = min(box.y0 for box in labels)
    gap = (xlabel_bottom-ink.y1)*scale
    bottom = ink.y0*scale
    return {'xlabel_to_legend_pt': float(gap),
            'legend_to_canvas_bottom_pt': float(bottom),
            'caption_extra_pt': CAPTION_EXTRA_PT,
            'legend_to_caption_estimated_pt': float(bottom+CAPTION_EXTRA_PT),
            'legend_ink_box_display': list(ink.bounds),
            'legend_full_box_display': list(full.bounds)}


def balance_legend_spacing(fig):
    """Move only the legend so the two adjacent visible gaps are comparable."""
    if getattr(fig, '_presentation_spacing_applied', False):
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    before = measure_spacing(fig, renderer)
    down = (before['legend_to_canvas_bottom_pt']+CAPTION_EXTRA_PT
            -before['xlabel_to_legend_pt'])/2.0
    legend = fig.legends[0]
    anchor = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
    height_pt = fig.get_figheight()*72.0
    legend.set_bbox_to_anchor((anchor.x0, anchor.y0-down/height_pt),
                             transform=fig.transFigure)
    fig.canvas.draw()
    after = measure_spacing(fig, fig.canvas.get_renderer())
    if after['legend_to_canvas_bottom_pt'] < 1.0:
        raise ValueError('Legend ink would be clipped by the figure canvas')
    fig._presentation_spacing_applied = True
    fig._presentation_spacing = {'before_move': before, 'after_move': after,
                                 'downward_shift_pt': float(down),
                                 'caption_extra_pt': CAPTION_EXTRA_PT}
