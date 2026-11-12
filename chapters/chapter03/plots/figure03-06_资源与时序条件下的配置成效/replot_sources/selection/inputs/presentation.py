"""Presentation-only labels, fonts and shared-legend spacing."""
from matplotlib import pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Bbox

def finish(fig,role):
 if role=='conditions':
  old_height=fig.get_figheight();extra=1.20/2.54;new_height=old_height+extra
  positions=[ax.get_position().bounds for ax in fig.axes]
  fig.set_size_inches(fig.get_figwidth(),new_height)
  for ax,(x,y,w,h) in zip(fig.axes,positions):ax.set_position([x,(y*old_height+extra)/new_height,w,h*old_height/new_height])
 main_axes=fig.axes[:8] if role=='configuration' else fig.axes
 if role.endswith('development'):
  titles=['(a) 通信功率分配','(b) 应急柴油分配']
 elif role=='conditions':
  titles=['(a) 通信可用能量变化','(b) 通信干扰变化','(c) 柴油可用总量变化','(d) 柴油需求出现时间变化']
 elif role=='configuration':
  titles=['(a) 通信功率分配','(b) 应急柴油分配']
 else:raise ValueError(role)
 titled=main_axes[:2] if role=='configuration' else main_axes
 for ax,title in zip(titled,titles):
  ax.set_title('');ax.set_title(title,loc='center',pad=9,fontfamily=['Times New Roman','SimSun'],fontsize=10.5,fontweight='normal')
 for text in fig.findobj(match=Text):
  text.set_fontsize(10.5);text.set_fontfamily(['Times New Roman','SimSun']);text.set_fontweight('normal')
 # Preserve the actual observation positions, limits and accepted plot areas.
 # Only the legend's anchor moves to balance the visible adjacent whitespace.
 assert len(fig.legends)==1
 legend=fig.legends[0];fig.canvas.draw();renderer=fig.canvas.get_renderer()
 labels=[ax.xaxis.label.get_window_extent(renderer) for ax in fig.axes if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text()]
 handles=getattr(legend,'legend_handles',getattr(legend,'legendHandles',[]));objects=list(legend.get_texts())+list(handles)
 boxes=[o.get_window_extent(renderer) for o in objects if o.get_visible()];boxes=[b for b in boxes if b.width>=0 and b.height>=0];ink=Bbox.union(boxes)
 k=72/fig.dpi;gap=(min(b.y0 for b in labels)-ink.y1)*k;bottom=ink.y0*k;extra=9.6
 down=(bottom+extra-gap)/2;anchor=legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted());legend.set_bbox_to_anchor((anchor.x0,anchor.y0-down/(fig.get_figheight()*72)),transform=fig.transFigure)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();boxes=[o.get_window_extent(renderer) for o in objects if o.get_visible()];boxes=[b for b in boxes if b.width>=0 and b.height>=0];ink=Bbox.union(boxes)
 fig._presentation_spacing={'xlabel_to_legend_pt':float((min(b.y0 for b in labels)-ink.y1)*k),'legend_ink_to_canvas_bottom_pt':float(ink.y0*k),'legend_to_caption_estimate_pt':float(ink.y0*k+extra),'caption_extra_pt':extra}
 if ink.y0*k<1:raise ValueError('Visible shared legend would be clipped')
