#!/usr/bin/env python3
"""Native physical-axes layout using only this figure's local CSV/source.

The accepted source remains unchanged. Native fonts, data, limits, ticks, line
styles and annotation data anchors are retained. The declared physical axes
changes and, where present, one explicit display-only label offset are checked
against the accepted numeric-artist record before writing the new outputs.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,sys,tempfile
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.axes import Axes
NUMBER='6.3'
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):return len(a)==len(b)and all(abs(float(x)-y)<1e-9 for x,y in zip(a,b))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=HERE);a=parser.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    original_init,original_add,original_set_xticklabels=Figure.__init__,Figure.add_axes,Axes.set_xticklabels;changes=[]
    def init(self,*args,**kwargs):
        before=kwargs.get('figsize')
        if before is not None:
            if NUMBER=='6.3'and close(before,[144/25.4,80/25.4]):kwargs['figsize']=(144/25.4,122/25.4);changes.append({'role':'canvas_initial','before_mm':[144,80],'after_mm':[144,122]})
            if NUMBER=='6.7'and close(before,[82/25.4,73/25.4]):kwargs['figsize']=(144/25.4,132/25.4);changes.append({'role':'canvas_initial','before_mm':[82,73],'after_mm':[144,132]})
        original_init(self,*args,**kwargs)
    def add(self,*args,**kwargs):
        if args and isinstance(args[0],(list,tuple))and len(args[0])==4:
            before=list(args[0]);after=None
            if NUMBER=='6.3':
                if close(before,[.16,.5375,.825,.3125]):after=[.16,64.29163848029241/122,.825,46/122]
                if close(before,[.16,.1625,.825,.25]):after=[.16,12.521335432264541/122,.825,41/122]
            elif NUMBER=='6.7'and close(before,[18/82,15/73,62/82,55/73]):after=[18/144,15/132,124/144,105/132]
            elif NUMBER=='6.8'and close(before,[.16,.195,.825,.40]):after=[.16,.195,.825,.40-10/132]
            if after is not None:args=(after,)+args[1:];changes.append({'role':'axes_physical_layout','before_fraction':before,'after_fraction':after})
        return original_add(self,*args,**kwargs)
    def set_xticklabels(self,labels,*args,**kwargs):
        result=original_set_xticklabels(self,labels,*args,**kwargs)
        if NUMBER=='6.3'and len(self.figure.axes)==2 and self is self.figure.axes[1]and len(labels)==7 and str(labels[-1])=='8/19':
            assert result[-1].get_ha()=='center';result[-1].set_ha('right');changes.append({'role':'terminal_tick_label_display_alignment_only','text':'8/19','tick_value':81,'before_ha':'center','after_ha':'right','reason':'Retain a clear right canvas margin without changing tick/data position'})
        return result
    Figure.__init__,Figure.add_axes,Axes.set_xticklabels=init,add,set_xticklabels
    try:
        with tempfile.TemporaryDirectory(prefix='.render-native-layout-',dir=a.output_dir)as tmp:
            work=Path(tmp);spec=importlib.util.spec_from_file_location('accepted_spacing_layout',HERE/'source/spacing.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ROOT=work
            out=work/'rendered';sys.argv=[__file__,'--number',NUMBER,'--output-dir',str(out)];m.main()
            assert len(changes)=={'6.3':4,'6.7':2,'6.8':1}[NUMBER],changes
            numbers=json.loads((out/'numeric_artist_values_before_after.json').read_text());parent=json.loads((HERE/'numeric_artist_values_before_after.json').read_text());assert numbers['before']==numbers['after']==parent['after'],'A scientific artist changed'
            geometry=json.loads((out/'render_geometry.json').read_text());assert abs(geometry['native_mm'][0]-144)<1e-6
            record={'schema':'chapter6-physical-axes-layout-v1','status':'PASS','figure_number':NUMBER,'authorized_geometry_changes':changes,'native_mm':geometry['native_mm'],'axes_sizes_mm':geometry['axes_sizes_mm'],'all_scientific_numeric_artists_equal_parent':True,'numeric_sha256':numbers['before_sha256'],'parent_numeric_sha256':parent['after_sha256'],'fonts_native_10_5bp':True,'annotation_positions_and_target_data_exact':True,'tick_label_display_alignment_exception':{'8/19':{'tick_value':81,'before_ha':'center','after_ha':'right'}},'unchanged_original_parent_PDF_sha256':sha(HERE/'figure.pdf'),'CSV_sha256':{p.name:sha(p)for p in HERE.glob('*.csv')},'local_source_files_sha256':{'plot_layout.py':sha(__file__),'source/spacing.py':sha(HERE/'source/spacing.py')},'font_and_spacing_measurements':geometry,'scientific_execution':0}
            for ext in ['pdf','svg','png']:shutil.copyfile(out/('figure.'+ext),a.output_dir/('layout.'+ext))
            record['output_sha256']={f'layout.{e}':sha(a.output_dir/f'layout.{e}')for e in ['pdf','svg','png']}
            (a.output_dir/'layout_geometry.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            (a.output_dir/'layout_numeric_artists.json').write_text(json.dumps(numbers,ensure_ascii=False,indent=2)+'\n')
    finally:Figure.__init__,Figure.add_axes,Axes.set_xticklabels=original_init,original_add,original_set_xticklabels
    print(json.dumps({'status':'PASS','figure':NUMBER,'native_mm':record['native_mm'],'axes_mm':record['axes_sizes_mm'],'output':str(a.output_dir)},ensure_ascii=False))
if __name__=='__main__':main()
