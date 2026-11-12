"""Presentation only: readable labels/layout and numeric-artist evidence."""
from pathlib import Path
import hashlib
import json
import numpy as np


def finalize_presentation(fig,config):
    kind=config['kind']
    internal=kind in ('history','control')
    if internal:
        fig.subplots_adjust(left=.145,right=.97,bottom=.18,top=.91,wspace=.43,hspace=.80)
    elif kind=='cost':
        fig.subplots_adjust(left=.12,right=.98,bottom=.32,top=.88,wspace=.40)
    else:
        fig.subplots_adjust(left=.12,right=.98,bottom=.29,top=.88,wspace=.40)
    for index,axis in enumerate(fig.axes):
        title=axis.get_title()
        axis.set_title('',loc='center')
        axis.set_title(f'({chr(97+index)}) {title}',loc='center',pad=8,
                       fontfamily=['Times New Roman','SimSun'],fontsize=10.5,fontweight='normal')
        if kind!='cost':
            axis.set_xticks([0,25,50,75,100]);axis.set_xticklabels(['0','25','50','75','100'])
            axis.get_xticklabels()[-1].set_horizontalalignment('right')
    if fig.legends:
        from matplotlib.transforms import Bbox
        legend=fig.legends[0]
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        labels=[axis.xaxis.label.get_window_extent(renderer) for axis in fig.axes if axis.xaxis.label.get_text()]
        objects=list(legend.get_texts())+list(legend.legend_handles)
        ink=Bbox.union([obj.get_window_extent(renderer) for obj in objects if obj.get_visible()])
        xlabel_bottom=min(box.y0 for box in labels)
        point=72/fig.dpi
        gap=(xlabel_bottom-ink.y1)*point;bottom=ink.y0*point
        shift=(bottom+9.6-gap)/2
        anchor=legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        legend.set_bbox_to_anchor((anchor.x0,anchor.y0-shift/(fig.get_figheight()*72)),transform=fig.transFigure)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        ink=Bbox.union([obj.get_window_extent(renderer) for obj in objects if obj.get_visible()])
        fig._alignment_spacing={'xlabel_to_legend_pt':(xlabel_bottom-ink.y1)*point,
                                'legend_to_canvas_pt':ink.y0*point,
                                'caption_gap_assumption_pt':9.6,
                                'legend_to_caption_estimate_pt':ink.y0*point+9.6}
        if ink.y0*point<1:raise ValueError('Legend ink crosses the new canvas edge')


def fingerprint(values):
    values=np.asarray(values)
    return {'shape':list(values.shape),'dtype':str(values.dtype),
            'sha256':hashlib.sha256(values.tobytes()).hexdigest()}


def export_geometry(fig,destination):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    def box(obj):
        value=obj.get_window_extent(renderer)
        return [float(value.x0),float(height-value.y1),float(value.x1),float(height-value.y0)]
    def text(key,kind,obj):
        return {'key':key,'kind':kind,'text':obj.get_text(),'box':box(obj),'font_size':obj.get_fontsize(),
                'font_family':obj.get_fontfamily(),'font_weight':obj.get_fontweight(),'font_style':obj.get_fontstyle()}
    records=[];legends=[];geometry=[]
    for index,axis in enumerate(fig.axes):
        named=[(f'ax{index}.xlabel','xlabel',axis.xaxis.label),(f'ax{index}.ylabel','ylabel',axis.yaxis.label),
               (f'ax{index}.title','title',axis.title)]
        for family,coordinate in [('xtick',axis.xaxis),('ytick',axis.yaxis)]:
            for tick_index,tick in enumerate(coordinate._update_ticks()):
                named.append((f'ax{index}.{family}{tick_index}',family,tick.label1))
            named.append((f'ax{index}.{family}.offset','offset',coordinate.get_offset_text()))
        named.extend((f'ax{index}.annotation{n}','annotation',obj) for n,obj in enumerate(axis.texts))
        records.extend(text(key,kind,obj) for key,kind,obj in named if obj.get_visible() and obj.get_text().strip())
        geometry.append({'xlim':list(axis.get_xlim()),'ylim':list(axis.get_ylim()),'xscale':axis.get_xscale(),
                         'yscale':axis.get_yscale(),'xticks':axis.get_xticks().tolist(),'yticks':axis.get_yticks().tolist(),
                         'lines':[fingerprint(line.get_xydata()) for line in axis.lines],
                         'collections':[[{'vertices':fingerprint(path.vertices),'codes':None if path.codes is None else fingerprint(path.codes)}
                                         for path in collection.get_paths()]for collection in axis.collections],
                         'patches':[list(patch.get_bbox().bounds) for patch in axis.patches if hasattr(patch,'get_bbox')]})
    for index,legend in enumerate(fig.legends):
        legends.append({'key':f'legend{index}','box':box(legend),'texts':[obj.get_text() for obj in legend.get_texts()]})
        records.extend(text(f'legend{index}.text{n}','legend',obj)for n,obj in enumerate(legend.get_texts()))
    report={'canvas_shape':[height,width],'figure_inches':fig.get_size_inches().tolist(),'dpi':fig.dpi,
            'axes_positions':[list(axis.get_position().bounds)for axis in fig.axes],
            'geometry':geometry,'texts':records,'legends':legends,'spacing':getattr(fig,'_alignment_spacing',None)}
    report['artist_geometry_sha256']=hashlib.sha256(json.dumps(geometry,sort_keys=True).encode()).hexdigest()
    Path(destination).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
