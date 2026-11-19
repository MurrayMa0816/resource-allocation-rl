#!/usr/bin/env python3
"""Native C6 learning artists in two118.8x34.338mm rows from local CSV.

Default sources are local to this figure. --source-dir supports isolated HY
staging. Existing scientific artists and original spacing source stay intact.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,sys,tempfile
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import fitz
HERE=Path(__file__).resolve().parent
SOURCE=HERE

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):return len(a)==len(b)and all(abs(float(x)-y)<1e-9 for x,y in zip(a,b))
def main():
    global SOURCE
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=HERE);p.add_argument('--source-dir',type=Path,default=HERE);a=p.parse_args();SOURCE=a.source_dir.resolve();a.output_dir.mkdir(parents=True,exist_ok=True)
    parent_geom=json.loads((SOURCE/'render_geometry.json').read_text());oldH=parent_geom['native_mm'][1];lower_y=parent_geom['axes_positions_mm'][0][1];old_ax_h=parent_geom['axes_sizes_mm'][0][1];title=parent_geom['pdf_frames_after']['title_frames_bp'][0];xlabel=parent_geom['pdf_frames_after']['xlabel_frames_bp'][0]
    title_above=oldH-title[1]*25.4/72-(lower_y+old_ax_h);xlabel_below=lower_y-(oldH-xlabel[3]*25.4/72);height=34.338;top_y=lower_y+height+title_above+12*25.4/72+xlabel_below;canvas=140.;changes=[]
    original_init,original_add,original_legend=Figure.__init__,Figure.add_axes,Figure.legend
    def init(self,*args,**kw):
        if kw.get('figsize')is not None and close(kw['figsize'],[156/25.4,81/25.4]):kw['figsize']=(144/25.4,canvas/25.4);changes.append('native_canvas144mm')
        return original_init(self,*args,**kw)
    def add(self,*args,**kw):
        if args and isinstance(args[0],(list,tuple)):
            before=list(args[0]);after=None
            if close(before,[17/156,22/81,59.2/156,51/81]):after=[.16,top_y/canvas,.825,height/canvas];changes.append('upper_axes118.8x34.338mm')
            elif close(before,[94/156,22/81,59.2/156,51/81]):after=[.16,lower_y/canvas,.825,height/canvas];changes.append('lower_axes118.8x34.338mm')
            if after is not None:args=(after,)+args[1:]
        return original_add(self,*args,**kw)
    def legend(self,*args,**kw):
        if close(kw.get('bbox_to_anchor',[]),[.5,2/81]):kw['bbox_to_anchor']=(.5,2/canvas);changes.append('legend_initial2mm')
        return original_legend(self,*args,**kw)
    Figure.__init__,Figure.add_axes,Figure.legend=init,add,legend
    try:
        with tempfile.TemporaryDirectory(prefix='.render-vertical-',dir=a.output_dir)as tmp:
            work=Path(tmp);sp=importlib.util.spec_from_file_location('accepted_spacing',SOURCE/'source/spacing.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);m.ROOT=work;out=work/'rendered';sys.argv=[__file__,'--number','6.5','--output-dir',str(out)];m.main();assert len(changes)==4,changes
            numeric=json.loads((out/'numeric_artist_values_before_after.json').read_text());parent=json.loads((SOURCE/'numeric_artist_values_before_after.json').read_text());assert numeric['before']==numeric['after']==parent['after']
            geo=json.loads((out/'render_geometry.json').read_text());frames=geo['pdf_frames_after'];xl=sorted(frames['xlabel_frames_bp'],key=lambda b:b[1]);titles=sorted(frames['title_frames_bp'],key=lambda b:b[1]);gap=titles[1][1]-xl[0][3];assert abs(gap-12)<.002,(gap,frames);assert abs(geo['xlabel_to_legend_bp']-8)<.002;assert all(abs(w-118.8)<1e-6 and abs(h-34.338)<1e-6 for w,h in geo['axes_sizes_mm'])
            for ext in['pdf','svg','png']:shutil.copyfile(out/f'figure.{ext}',a.output_dir/f'vertical.{ext}')
            (a.output_dir/'vertical_numeric_artists.json').write_text(json.dumps(numeric,ensure_ascii=False,indent=2)+'\n');(a.output_dir/'vertical_geometry.json').write_text(json.dumps({'status':'PASS','native_mm':geo['native_mm'],'axes_sizes_mm':geo['axes_sizes_mm'],'upper_xlabel_to_lower_title_bp':gap,'bottom_xlabel_to_legend_bp':geo['xlabel_to_legend_bp'],'all_scientific_artists_exact_parent':True,'all26points_3methods_2axes_and_full_SD_band':all(len(l['x'])==26 for ax in numeric['after']for l in ax['lines']),'native_fonts_bp':10.5,'source_renderer_sha256':sha(SOURCE/'source/spacing.py'),'entry_sha256':sha(__file__),'geometry':geo,'outputs_sha256':{f'vertical.{e}':sha(a.output_dir/f'vertical.{e}')for e in['pdf','svg','png']},'environment_steps':0,'model_loads':0,'training':0},ensure_ascii=False,indent=2)+'\n')
    finally:Figure.__init__,Figure.add_axes,Figure.legend=original_init,original_add,original_legend
    print(json.dumps({'status':'PASS','output':str(a.output_dir),'native_mm':geo['native_mm'],'axes_mm':geo['axes_sizes_mm'],'row_gap_bp':gap},ensure_ascii=False))
if __name__=='__main__':main()
