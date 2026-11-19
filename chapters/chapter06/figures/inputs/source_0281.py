#!/usr/bin/env python3
"""C6 native uncompressed two-row template with 15bp legend spacing.

Both axes are118.8x52.8mm in a144mm canvas. The existing local CSV renderer
and original spacing source remain read only. Four guarded layout-only source
substitutions in memory change the two existing8bp legend targets to15bp.
All scientific artists, current annotations and native10.5bp fonts are checked.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, shutil, sys, tempfile, types
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.axes import Axes

NUMBER='6.8'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):return len(a)==len(b) and all(abs(float(x)-y)<1e-9 for x,y in zip(a,b))



# RL terminology revision 2026-11-14: final text layer after original guards.
RL_WORDING_MAP = [('变化后天气', '蒸散率扰动后'), ('相同变化天气下的每日配水', '同一条扰动后的天气序列下的每日配水')]
RL_WORDING_AXIS = None

def _rl_plain(v):
    import numpy as np
    if isinstance(v,np.ndarray):return _rl_plain(v.tolist())
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,(tuple,list)):return [_rl_plain(x) for x in v]
    if isinstance(v,dict):return {str(k):_rl_plain(x) for k,x in v.items()}
    return v

def _rl_scientific(fig):
    return _rl_plain(dict(canvas=fig.get_size_inches(),axes=[dict(
        position=ax.get_position().bounds,xlim=ax.get_xlim(),ylim=ax.get_ylim(),
        xscale=ax.get_xscale(),yscale=ax.get_yscale(),xticks=ax.get_xticks(),yticks=ax.get_yticks(),
        lines=[dict(x=x.get_xdata(),y=x.get_ydata(),color=x.get_color(),width=x.get_linewidth(),
                    style=x.get_linestyle(),marker=x.get_marker(),ms=x.get_markersize(),
                    alpha=x.get_alpha(),zorder=x.get_zorder())for x in ax.lines],
        collections=[dict(paths=[dict(vertices=p.vertices,codes=p.codes)for p in x.get_paths()],
                          offsets=x.get_offsets(),array=x.get_array(),clim=x.get_clim(),
                          fc=x.get_facecolors(),ec=x.get_edgecolors(),lw=x.get_linewidths(),
                          alpha=x.get_alpha(),zorder=x.get_zorder())for x in ax.collections],
        patches=[dict(vertices=x.get_path().vertices,codes=x.get_path().codes,
                      transform=x.get_patch_transform().get_matrix(),fc=x.get_facecolor(),
                      ec=x.get_edgecolor(),lw=x.get_linewidth(),alpha=x.get_alpha(),zorder=x.get_zorder())for x in ax.patches],
        images=[dict(array=x.get_array(),extent=x.get_extent(),clim=x.get_clim())for x in ax.images]
        )for ax in fig.axes]))

def _rl_export_wording(base,fig,target,*args,**kwargs):
    from matplotlib.text import Text
    from matplotlib.ticker import FuncFormatter
    import json,hashlib
    from pathlib import Path
    try:path=Path(target)
    except TypeError:return base(fig,target,*args,**kwargs)
    if path.name not in ('figure.pdf','figure.svg','figure.png'):
        return base(fig,target,*args,**kwargs)
    fig.canvas.draw();before=_rl_scientific(fig)
    artists=[t for t in fig.findobj(Text)if t.get_visible()and t.get_text()]
    fonts=lambda:[(id(t),t.get_fontsize(),t.get_fontfamily(),t.get_fontweight(),
                   t.get_fontstyle(),t.get_position(),t.get_rotation(),t.get_ha(),t.get_va())for t in artists]
    font_before=fonts();changed=[];formatters=[];axis_records=[]
    try:
        if RL_WORDING_AXIS:
            ax=fig.axes[RL_WORDING_AXIS['axis_index']]
            formatters.append((ax.xaxis,ax.xaxis.get_major_formatter()))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:format(v/RL_WORDING_AXIS['divisor'],'g')))
            old=ax.get_xlabel();new=RL_WORDING_AXIS['xlabel']
            changed.append((ax.xaxis.label,old,new));ax.xaxis.label.set_text(new)
            axis_records.append(dict(axis_index=RL_WORDING_AXIS['axis_index'],
                                     numeric_ticks_before=list(ax.get_xticks()),divisor=RL_WORDING_AXIS['divisor']))
        for t in artists:
            old=t.get_text();new=old
            for a,b in RL_WORDING_MAP:new=new.replace(a,b)
            if new!=old:changed.append((t,old,new));t.set_text(new)
        assert changed,'Expected final wording labels were not found'
        fig.canvas.draw()
        assert _rl_scientific(fig)==before,'Non-text artists, axes, ticks or canvas changed'
        assert fonts()==font_before,'Font, rotation or text anchor changed'
        ren=fig.canvas.get_renderer();crop=kwargs.get('bbox_inches')
        bounds=fig.bbox
        if crop is not None and hasattr(crop,'bounds'):
            from matplotlib.transforms import Bbox
            bounds=Bbox.from_bounds(*(x*fig.dpi for x in crop.bounds))
        for t,old,new in changed:
            bb=t.get_window_extent(ren)
            assert bb.x0>=bounds.x0-1 and bb.x1<=bounds.x1+1 and bb.y0>=bounds.y0-1 and bb.y1<=bounds.y1+1,('Reflow wording; do not shrink fonts',new,bb.bounds,bounds.bounds)
        result=base(fig,target,*args,**kwargs)
        assert _rl_scientific(fig)==before,'Save changed numerical or geometric artists'
        proof=dict(schema='rl-wording-final-native-label-layer-v1',status='PASS',
                   original_render_guards_retained=True,only_final_labels_changed=True,
                   numeric_artists_axes_canvas_fonts_exact=True,
                   scientific_before_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
                   scientific_after_sha256=hashlib.sha256(json.dumps(_rl_scientific(fig),sort_keys=True).encode()).hexdigest(),
                   labels=[dict(before=old,after=new,font_bp=t.get_fontsize(),bbox_pixels=list(t.get_window_extent(ren).bounds))for t,old,new in changed],
                   axis_formatter=axis_records,environment_steps=0,model_loads=0,policy_inferences=0,training_runs=0)
        (path.parent/'rl_wording_label_proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
        return result
    finally:
        for axis,formatter in formatters:axis.set_major_formatter(formatter)
        for t,old,new in reversed(changed):t.set_text(old)

def _rl_capture_final(directory):
    from matplotlib.figure import Figure
    from pathlib import Path
    base=Figure.savefig;captured={}
    def capture(fig,target,*args,**kwargs):
        try:path=Path(target)
        except TypeError:return base(fig,target,*args,**kwargs)
        if path.name in ('figure.pdf','figure.svg','figure.png')and path.parent.resolve()==Path(directory).resolve():
            captured[path.name]=(fig,args,dict(kwargs))
        return base(fig,target,*args,**kwargs)
    Figure.savefig=capture
    return base,captured

def _rl_restore_capture(base):
    from matplotlib.figure import Figure
    Figure.savefig=base

def _rl_emit_captured(base,captured,directory):
    from pathlib import Path
    assert 'figure.pdf'in captured and 'figure.png'in captured,'No current final native figure captured'
    for name,(fig,args,kwargs)in captured.items():
        _rl_export_wording(base,fig,Path(directory)/name,*args,**kwargs)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=HERE);ap.add_argument('--source-dir',type=Path,default=HERE);a=ap.parse_args()
    source=a.source_dir.resolve();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    original_init,original_add,original_legend=Figure.__init__,Figure.add_axes,Figure.legend
    original_annotate,original_ticklabels=Axes.annotate,Axes.set_xticklabels
    changes=[];canvas=240. if NUMBER=='6.8' else 200.
    top_y=lower_y=None
    if NUMBER=='6.5':
        old=json.loads((source/'render_geometry.json').read_text());oldH=old['native_mm'][1];lower_y=old['axes_positions_mm'][0][1];old_ax_h=old['axes_sizes_mm'][0][1]
        title=old['pdf_frames_after']['title_frames_bp'][0];xlabel=old['pdf_frames_after']['xlabel_frames_bp'][0]
        title_above=oldH-title[1]*25.4/72-(lower_y+old_ax_h);xlabel_below=lower_y-(oldH-xlabel[3]*25.4/72)
        top_y=lower_y+52.8+title_above+12*25.4/72+xlabel_below
    def init(self,*args,**kw):
        dims={'6.3':[144/25.4,80/25.4],'6.5':[156/25.4,81/25.4],'6.8':[144/25.4,132/25.4]}[NUMBER]
        if kw.get('figsize') is not None and close(kw['figsize'],dims):
            kw['figsize']=(144/25.4,canvas/25.4);changes.append({'role':'temporary_native_canvas','after_mm':[144,canvas]})
        return original_init(self,*args,**kw)
    def add(self,*args,**kw):
        if args and isinstance(args[0],(list,tuple)) and len(args[0])==4:
            before=list(args[0]);after=None
            if NUMBER=='6.3':
                if close(before,[.16,.5375,.825,.3125]):after=[.16,.5375,.825,52.8/canvas]
                if close(before,[.16,.1625,.825,.25]):after=[.16,.1625,.825,52.8/canvas]
            elif NUMBER=='6.5':
                if close(before,[17/156,22/81,59.2/156,51/81]):after=[.16,top_y/canvas,.825,52.8/canvas]
                if close(before,[94/156,22/81,59.2/156,51/81]):after=[.16,lower_y/canvas,.825,52.8/canvas]
            elif NUMBER=='6.8':
                if close(before,[.16,.755,.825,.18]):after=[.16,.755,.825,52.8/canvas]
                if close(before,[.16,.195,.825,.40]):after=[.16,.195,.825,52.8/canvas]
            if after is not None:args=(after,)+args[1:];changes.append({'role':'restore_clear_axes','before_fraction':before,'after_fraction':after})
        return original_add(self,*args,**kw)
    def legend(self,*args,**kw):
        if NUMBER=='6.5' and close(kw.get('bbox_to_anchor',[]),[.5,2/81]):kw['bbox_to_anchor']=(.5,2/canvas)
        return original_legend(self,*args,**kw)
    def annotate(self,text,*args,**kw):
        if NUMBER=='6.8' and text=='水量未变' and args and int(args[0][0])==61 and kw.get('xytext')==(0,-17):kw['xytext']=(0,-9)
        return original_annotate(self,text,*args,**kw)
    def ticklabels(self,labels,*args,**kw):
        result=original_ticklabels(self,labels,*args,**kw)
        if NUMBER=='6.3' and len(self.figure.axes)==2 and self is self.figure.axes[1] and len(labels)==7 and str(labels[-1])=='8/19':result[-1].set_ha('right')
        return result
    Figure.__init__,Figure.add_axes,Figure.legend=init,add,legend
    Axes.annotate,Axes.set_xticklabels=annotate,ticklabels
    substitutions=[
        ("move_axes([0],8-f['upper_axis_including_ticks_to_legend_bp'])","move_axes([0],15-f['upper_axis_including_ticks_to_legend_bp'])"),
        ("move_legend(len(fig.legends)-1,8-f['xlabel_to_legend_bp'])","move_legend(len(fig.legends)-1,15-f['xlabel_to_legend_bp'])"),
        ("abs(final['xlabel_to_legend_bp']-8)","abs(final['xlabel_to_legend_bp']-15)"),
        ("abs(final['upper_axis_including_ticks_to_legend_bp']-8)","abs(final['upper_axis_including_ticks_to_legend_bp']-15)")]
    try:
        with tempfile.TemporaryDirectory(prefix='.template15-',dir=out) as tmp:
            work=Path(tmp);path=source/'source/spacing.py';code=path.read_text()
            for old,new in substitutions:assert code.count(old)==1,(old,code.count(old));code=code.replace(old,new)
            m=types.ModuleType('c6_guarded_spacing15');m.__file__=str(path);exec(compile(code,str(path)+'[layout15-memory]','exec'),m.__dict__);m.ROOT=work
            rendered=work/'rendered';oldargv=sys.argv[:]
            _rl_base_save,_rl_captured=_rl_capture_final(rendered/'accepted_stage'/'figures'/'feedback')
            try:sys.argv=[__file__,'--number',NUMBER,'--output-dir',str(rendered)];m.main()
            finally:sys.argv=oldargv;_rl_restore_capture(_rl_base_save)
            assert len(changes)==3,changes
            values=json.loads((rendered/'numeric_artist_values_before_after.json').read_text())
            parentfile={'6.3':'layout_numeric_artists.json','6.5':'narrow_numeric_artists.json','6.8':'equal_numeric_artists.json'}[NUMBER]
            parent=json.loads((source/parentfile).read_text())
            expected=json.loads(json.dumps(parent['after']))
            for axis in expected:
                for annotation in axis['annotations']:
                    annotation['text']=annotation['text'].replace('标注为同日较原安排增减量（m³）','标注为同日较原安排增减水量（m³）').replace('未调整','水量未变').replace('多配','增加').replace('少配','减少')
            assert values['before']==values['after']==expected
            geo=json.loads((rendered/'render_geometry.json').read_text());assert abs(geo['native_mm'][0]-144)<1e-6
            assert all(abs(w-118.8)<1e-8 and abs(h-52.8)<1e-8 for w,h in geo['axes_sizes_mm'])
            if NUMBER in['6.5','6.8']:assert abs(geo['xlabel_to_legend_bp']-15)<.002
            if NUMBER=='6.8':assert abs(geo['upper_axis_including_ticks_to_legend_bp']-15)<.002 and abs(geo['upper_legend_to_lower_title_bp']-12)<.002
            _rl_emit_captured(_rl_base_save,_rl_captured,rendered)
            shutil.copyfile(rendered/'rl_wording_label_proof.json',out/'rl_wording_label_proof.json')
            for ext in['pdf','svg','png']:shutil.copyfile(rendered/f'figure.{ext}',out/f'template15.{ext}')
            import fitz
            with fitz.open(out/'template15.pdf') as doc:
                page=doc[0];text=page.get_texttrace();assert all(abs(t['size']-10.5)<1e-5 and page.rect.contains(fitz.Rect(t['bbox'])) for t in text)
                ink=[fitz.Rect(t['bbox']) for t in text]
                for block in page.get_text('dict')['blocks']:
                    for line in block.get('lines',[]):
                        ink.extend(fitz.Rect(span['bbox']) for span in line['spans'] if span['text'].strip())
                for drawing in page.get_drawings():
                    box=fitz.Rect(drawing['rect'])
                    if drawing['fill'] is not None and min(drawing['fill'])>.999 and abs(box.width-page.rect.width)<.01 and abs(box.height-page.rect.height)<.01:continue
                    if drawing['color'] is not None and drawing['stroke_opacity']>0:
                        half=(drawing['width'] or 0)/2;box=fitz.Rect(box.x0-half,box.y0-half,box.x1+half,box.y1+half)
                    ink.append(box)
                bounds=fitz.Rect(min(b.x0 for b in ink),min(b.y0 for b in ink),max(b.x1 for b in ink),max(b.y1 for b in ink))
                visible={'measurement':'union of actual PDF text span/character frames and drawn path bounds, including legend handles/strokes; full-page white background excluded','bounds_bp':list(bounds),'top_white_bp':bounds.y0,'bottom_white_bp':page.rect.height-bounds.y1,'last_complete_legend_or_xlabel_y_bp':bounds.y1,'PDF_page_height_bp':page.rect.height}
            record={'schema':'c6-restored-template15-v1','status':'PASS_REQUIRES_VISUAL_REVIEW','figure':NUMBER,'native_mm':geo['native_mm'],'axes_sizes_mm':geo['axes_sizes_mm'],'native_fonts_bp':10.5,'scientific_artists_exact_current_parent':True,'parent_numeric_record_sha256':sha(source/parentfile),'new_data_or_annotation_exceptions':[],'retained_display_attributes':'6.3 last8/19 right alignment;6.8 first水量未变 offset(0,-9)bp','authorized_geometry_changes':changes,'layout_only_in_memory_substitutions':[{'before':x,'after':y} for x,y in substitutions],'unchanged_local_spacing_source_sha256':sha(path),'entry_sha256':sha(__file__),'CSV_sha256':{p.name:sha(p) for p in source.glob('*.csv')},'geometry':geo,'complete_visible_content':visible,'output_sha256':{f'template15.{e}':sha(out/f'template15.{e}') for e in['pdf','svg','png']},'scientific_execution':{'environment_steps':0,'model_loads':0,'policy_inference':0,'training_runs':0,'new_TEST':0}}
            (out/'template15_geometry.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');(out/'template15_numeric_artists.json').write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n')
    finally:
        Figure.__init__,Figure.add_axes,Figure.legend=original_init,original_add,original_legend
        Axes.annotate,Axes.set_xticklabels=original_annotate,original_ticklabels
    print(json.dumps({'status':'PASS_REQUIRES_VISUAL_REVIEW','figure':NUMBER,'native_mm':record['native_mm'],'axes_sizes_mm':record['axes_sizes_mm'],'output':str(out)},ensure_ascii=False))


if __name__=='__main__':main()
