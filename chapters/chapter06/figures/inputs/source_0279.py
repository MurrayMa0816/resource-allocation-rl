#!/usr/bin/env python3
"""Native taller Figure 6.8 from the unchanged local accepted CSV renderer.

Weather/allocation axes are 118.8x33.76 and118.8x62.8mm.  The existing
10.5bp text, all scientific artists, and the accepted7/30 display offset
remain unchanged. Existing local source is read only; no scientific run occurs.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, shutil, sys, tempfile
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.axes import Axes

HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def close(a, b): return len(a)==len(b) and all(abs(float(x)-y)<1e-9 for x,y in zip(a,b))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=HERE);p.add_argument('--source-dir',type=Path,default=HERE)
    a=p.parse_args();source=a.source_dir.resolve();output=a.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    original_init,original_add,original_annotate=Figure.__init__,Figure.add_axes,Axes.annotate;changes=[]
    def init(self,*args,**kwargs):
        before=kwargs.get('figsize')
        if before is not None and close(before,[144/25.4,132/25.4]):
            kwargs['figsize']=(144/25.4,162/25.4)
            changes.append({'role':'temporary_canvas_height_to_fit_taller_axes','before_mm':[144,132],'after_mm':[144,162]})
        original_init(self,*args,**kwargs)
    def add(self,*args,**kwargs):
        if args and isinstance(args[0],(list,tuple)) and len(args[0])==4:
            before=list(args[0]);after=None
            if close(before,[.16,.755,.825,.18]):after=[.16,.755,.825,33.76/162]
            if close(before,[.16,.195,.825,.40]):after=[.16,.195,.825,62.8/162]
            if after is not None:
                args=(after,)+args[1:];changes.append({'role':'axes_physical_layout','before_fraction':before,'after_fraction':after})
        return original_add(self,*args,**kwargs)
    def annotate(self,text,*args,**kwargs):
        # Retain the already accepted7/30 offset; this is not a new exception.
        if text=='未调整' and args and int(args[0][0])==61 and kwargs.get('xytext')==(0,-17):kwargs['xytext']=(0,-9)
        return original_annotate(self,text,*args,**kwargs)
    Figure.__init__,Figure.add_axes,Axes.annotate=init,add,annotate
    try:
        with tempfile.TemporaryDirectory(prefix='.render-flow-',dir=output) as tmp:
            work=Path(tmp);sp=importlib.util.spec_from_file_location('accepted_c6_flow_spacing',source/'source/spacing.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);m.ROOT=work
            rendered=work/'rendered';oldargv=sys.argv[:]
            try:sys.argv=[__file__,'--number','6.8','--output-dir',str(rendered)];m.main()
            finally:sys.argv=oldargv
            assert len(changes)==3,changes
            values=json.loads((rendered/'numeric_artist_values_before_after.json').read_text());parent=json.loads((source/'layout_numeric_artists.json').read_text())
            assert values['before']==values['after']==parent['after'],'A current scientific artist or display attribute changed'
            geometry=json.loads((rendered/'render_geometry.json').read_text())
            assert abs(geometry['native_mm'][0]-144)<1e-6 and abs(geometry['native_mm'][1]-143.760003)<1e-4
            assert all(abs(v-w)<1e-8 for a,b in zip(geometry['axes_sizes_mm'],[[118.8,33.76],[118.8,62.8]]) for v,w in zip(a,b))
            for ext in ['pdf','svg','png']:shutil.copyfile(rendered/('figure.'+ext),output/('flow.'+ext))
            record={'schema':'c6-native-flow68-v1','status':'PASS','native_mm':geometry['native_mm'],'axes_sizes_mm':geometry['axes_sizes_mm'],'current_layout_numeric_artists_exact':True,'current_parent_numeric_record_sha256':sha(source/'layout_numeric_artists.json'),'new_annotation_exceptions':[],'accepted_7_30_offset_points':[0,-9],'authorized_geometry_changes':changes,'font_and_spacing_measurements':geometry,'CSV_sha256':{p.name:sha(p) for p in source.glob('*.csv')},'source_files_sha256':{'plot_flow.py':sha(__file__),'source/spacing.py':sha(source/'source/spacing.py'),'source/accepted_statistical.py':sha(source/'source/accepted_statistical.py')},'output_sha256':{f'flow.{e}':sha(output/f'flow.{e}') for e in ['pdf','svg','png']},'scientific_execution':{'environment_steps':0,'model_loads':0,'policy_inference':0,'training_runs':0,'new_TEST':0}}
            (output/'flow_geometry.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            (output/'flow_numeric_artists.json').write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n')
    finally:Figure.__init__,Figure.add_axes,Axes.annotate=original_init,original_add,original_annotate
    print(json.dumps({'status':'PASS','native_mm':record['native_mm'],'axes_sizes_mm':record['axes_sizes_mm'],'output':str(output)},ensure_ascii=False))


if __name__=='__main__':main()
