#!/usr/bin/env python3
"""C6 native uncompressed two-row template with 15bp legend spacing.

Both axes are79.2x52.8mm in a104.4mm canvas, as requested by the user. The existing local CSV renderer
and original spacing source remain read only. Four guarded layout-only source
substitutions in memory change the two existing8bp legend targets to15bp.
All scientific artists, current annotations and native10.5bp fonts are checked.
"""
from pathlib import Path

import sys
_FIGURE_SHARED = next(p for p in Path(__file__).resolve().parents if p.name == "figures") / "shared"
if str(_FIGURE_SHARED) not in sys.path:
    sys.path.insert(0, str(_FIGURE_SHARED))
from figure_paths import configure_fonts, fonts_root, validate_output_dir
SHARED_DATA = _FIGURE_SHARED / "data"
import argparse, hashlib, importlib.util, json, math, shutil, sys, tempfile, types

NUMBER='6.5'
WIDTH=104.4
AXIS_WIDTH=79.2
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):return len(a)==len(b) and all(abs(float(x)-y)<1e-9 for x,y in zip(a,b))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--source-dir',type=Path,default=HERE);ap.add_argument('--font-dir',type=Path,help='Original font root containing windows/ and figure/');a=ap.parse_args()
    try:
        validated_output=validate_output_dir(__file__,a.output_dir,sources=(a.source_dir,),font_directory=a.font_dir)
    except ValueError as exc:
        ap.error(str(exc))
    configure_fonts(__file__,a.font_dir,('windows/simsun.ttc','windows/TIMES.TTF'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    source=a.source_dir.resolve();out=validated_output;out.mkdir(parents=True,exist_ok=True)
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
            kw['figsize']=(WIDTH/25.4,canvas/25.4);changes.append({'role':'temporary_native_canvas','after_mm':[WIDTH,canvas]})
        return original_init(self,*args,**kw)
    def add(self,*args,**kw):
        if args and isinstance(args[0],(list,tuple)) and len(args[0])==4:
            before=list(args[0]);after=None
            if NUMBER=='6.3':
                if close(before,[.16,.5375,.825,.3125]):after=[.16,.5375,.825,52.8/canvas]
                if close(before,[.16,.1625,.825,.25]):after=[.16,.1625,.825,52.8/canvas]
            elif NUMBER=='6.5':
                if close(before,[17/156,22/81,59.2/156,51/81]):after=[23.04/WIDTH,top_y/canvas,AXIS_WIDTH/WIDTH,52.8/canvas]
                if close(before,[94/156,22/81,59.2/156,51/81]):after=[23.04/WIDTH,lower_y/canvas,AXIS_WIDTH/WIDTH,52.8/canvas]
            elif NUMBER=='6.8':
                if close(before,[.16,.755,.825,.18]):after=[.16,.755,.825,52.8/canvas]
                if close(before,[.16,.195,.825,.40]):after=[.16,.195,.825,52.8/canvas]
            if after is not None:args=(after,)+args[1:];changes.append({'role':'restore_clear_axes','before_fraction':before,'after_fraction':after})
        return original_add(self,*args,**kw)
    def legend(self,*args,**kw):
        if NUMBER=='6.5' and close(kw.get('bbox_to_anchor',[]),[.5,2/81]):kw['bbox_to_anchor']=(.5,2/canvas)
        return original_legend(self,*args,**kw)
    def annotate(self,text,*args,**kw):
        if NUMBER=='6.8' and text=='未调整' and args and int(args[0][0])==61 and kw.get('xytext')==(0,-17):kw['xytext']=(0,-9)
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
            try:sys.argv=[__file__,'--number',NUMBER,'--output-dir',str(rendered)];m.main()
            finally:sys.argv=oldargv
            assert len(changes)==3,changes
            values=json.loads((rendered/'numeric_artist_values_before_after.json').read_text())
            parentfile={'6.3':'layout_numeric_artists.json','6.5':'narrow_numeric_artists.json','6.8':'equal_numeric_artists.json'}[NUMBER]
            parent=json.loads((source/parentfile).read_text());assert values['before']==values['after']==parent['after']
            geo=json.loads((rendered/'render_geometry.json').read_text());assert abs(geo['native_mm'][0]-WIDTH)<1e-6
            assert all(abs(w-AXIS_WIDTH)<1e-8 and abs(h-52.8)<1e-8 for w,h in geo['axes_sizes_mm'])
            if NUMBER in['6.5','6.8']:assert abs(geo['xlabel_to_legend_bp']-15)<.002
            if NUMBER=='6.8':assert abs(geo['upper_axis_including_ticks_to_legend_bp']-15)<.002 and abs(geo['upper_legend_to_lower_title_bp']-12)<.002
            for ext in['pdf','svg','png']:shutil.copyfile(rendered/f'figure.{ext}',out/f'narrow15.{ext}')
            import fitz
            with fitz.open(out/'narrow15.pdf') as doc:
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
            record={'schema':'c6-user-narrowed-figure65-v1','status':'PASS_REQUIRES_VISUAL_REVIEW','figure':NUMBER,'native_mm':geo['native_mm'],'axes_sizes_mm':geo['axes_sizes_mm'],'native_fonts_bp':10.5,'scientific_artists_exact_current_parent':True,'parent_numeric_record_sha256':sha(source/parentfile),'new_data_or_annotation_exceptions':[],'retained_display_attributes':'6.3 last8/19 right alignment;6.8 first未调整 offset(0,-9)bp','authorized_geometry_changes':changes,'layout_only_in_memory_substitutions':[{'before':x,'after':y} for x,y in substitutions],'unchanged_local_spacing_source_sha256':sha(path),'entry_sha256':sha(__file__),'CSV_sha256':{p.name:sha(p) for p in source.glob('*.csv')},'geometry':geo,'complete_visible_content':visible,'output_sha256':{f'narrow15.{e}':sha(out/f'narrow15.{e}') for e in['pdf','svg','png']},'scientific_execution':{'environment_steps':0,'model_loads':0,'policy_inference':0,'training_runs':0,'new_TEST':0}}
            (out/'narrow15_geometry.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');(out/'narrow15_numeric_artists.json').write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n')
    finally:
        Figure.__init__,Figure.add_axes,Figure.legend=original_init,original_add,original_legend
        Axes.annotate,Axes.set_xticklabels=original_annotate,original_ticklabels
    print(json.dumps({'status':'PASS_REQUIRES_VISUAL_REVIEW','figure':NUMBER,'native_mm':record['native_mm'],'axes_sizes_mm':record['axes_sizes_mm'],'output':str(out)},ensure_ascii=False))


if __name__=='__main__':main()
