"""Replot this figure from its local offline sources and saved observations.

Historical renderer bytes are kept in render_support/fs. This adapter rebases
historical names into a disposable local workspace. It reads no original HY
results or external project source; only this figure and thesis fonts are used.
"""
from pathlib import Path
import argparse,builtins,io,json,os,runpy,shutil,sys,tempfile,hashlib,importlib.util,importlib.machinery
HERE=Path(__file__).resolve().parent
THESIS=next(p for p in HERE.parents if (p/'assets/fonts').is_dir())
CONFIG=json.loads((HERE/'render_support/config.json').read_text())
FONT_FILES={p.name.lower():str(p) for p in (THESIS/'assets/fonts/windows').iterdir() if p.is_file()}


# RL terminology revision 2026-11-14: final text layer after original guards.
RL_WORDING_MAP = [('训练步数', '训练环境交互步数')]
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
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output-dir',type=Path,default=HERE/'replot_output')
    args=parser.parse_args();destination=args.output_dir.resolve()
    if destination==HERE:raise SystemExit('Use --output-dir with a separate directory to preserve the accepted figure.')
    destination.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='thesis-figure-replot-'));fs=work/'fs'
    shutil.copytree(HERE/'render_support/fs',fs)
    prefixes=['/hy-tmp/resource_allocation_results','/root/gamewar/resource_allocation_rl']
    raw_open=builtins.open;raw_io_open=io.open;raw_scandir=os.scandir
    raw_pathnew=Path.__new__;pathclass=type(Path());raw_posixnew=pathclass.__new__
    raw_runpath=runpy.run_path;raw_spec=importlib.util.spec_from_file_location
    raw_loaderdata=importlib.machinery.SourceFileLoader.get_data;raw_loads=json.loads
    raw_relative=Path.is_relative_to
    def remap(value):
        try:s=os.fspath(value)
        except TypeError:return value
        if isinstance(s,bytes):return s
        if s==str(HERE) or s.startswith(str(HERE)+'/') or s==str(destination) or s.startswith(str(destination)+'/'):return s
        if s.startswith('/root/.local/share/fonts/') and os.path.basename(s).lower() in FONT_FILES:
            return FONT_FILES[os.path.basename(s).lower()]
        base='/root/gamewar/latex_thesis_resource_allocation'
        if s==base or s.startswith(base+'/'):return str(THESIS)+s[len(base):]
        for prefix in prefixes:
            if s==prefix or s.startswith(prefix+'/'):return str(fs)+s
        return s
    def pathnew(cls,*args,**kwargs):
        if args:args=(remap(args[0]),*args[1:])
        return raw_pathnew(cls,*args,**kwargs)
    def posixnew(cls,*args,**kwargs):
        if args:args=(remap(args[0]),*args[1:])
        return raw_posixnew(cls,*args,**kwargs)
    def fopen(file,*args,**kwargs):return raw_open(remap(file),*args,**kwargs)
    def iopen(file,*args,**kwargs):return raw_io_open(remap(file),*args,**kwargs)
    def scandir(path='.'):return raw_scandir(remap(path))
    def runpath(path,*args,**kwargs):return raw_runpath(remap(path),*args,**kwargs)
    def spec(name,location,*args,**kwargs):return raw_spec(name,remap(location),*args,**kwargs)
    def loaderdata(self,path):return raw_loaderdata(self,remap(path))
    def decoded(v):
        if isinstance(v,str):return remap(v)
        if isinstance(v,list):return [decoded(x) for x in v]
        if isinstance(v,dict):return {remap(k):decoded(x) for k,x in v.items()}
        return v
    def loads(*args,**kwargs):return decoded(raw_loads(*args,**kwargs))
    def is_relative_to(self,*other):return raw_relative(self,*[remap(x) for x in other])
    def deny_external(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            p=os.fsdecode(args[0])
            if p==str(HERE) or p.startswith(str(HERE)+'/') or p==str(destination) or p.startswith(str(destination)+'/'):return
            blocked=prefixes+['/root/.local/share/fonts']
            original_thesis='/root/gamewar/latex_thesis_resource_allocation'
            if str(THESIS)!=original_thesis:blocked.append(original_thesis)
            if any(p==q or p.startswith(q+'/') for q in blocked):
                raise RuntimeError('External figure dependency refused: '+p)
    sys.addaudithook(deny_external)
    # The compatibility adapter exists only for this disposable renderer process.
    Path.__new__=pathnew;pathclass.__new__=posixnew;Path.is_relative_to=is_relative_to
    builtins.open=fopen;io.open=iopen;os.scandir=scandir;runpy.run_path=runpath
    importlib.util.spec_from_file_location=spec
    importlib.machinery.SourceFileLoader.get_data=loaderdata;json.loads=loads
    entry=Path(CONFIG['entry']);output=Path(CONFIG['output_root'])/'local_replot_output'
    sys.argv=[str(entry),'--output-dir',str(output)]
    _rl_base_save,_rl_captured=_rl_capture_final(output)
    try:
        runpath(str(entry),run_name='__main__')
        for name in ['figure.pdf','figure.png','figure.svg','plotted_points.csv','plotted_values.json','numeric_artist_values_before_after.json','render_geometry.json','spacing_proof.json']:
            source=output/name
            if source.is_file():shutil.copyfile(source,destination/name)
        for required in ['figure.pdf','figure.png']:
            if not (destination/required).is_file():raise RuntimeError('Missing renderer output: '+required)
        result={}
        for name in ['figure.pdf','figure.png']:
            p=destination/name
            if p.exists():
                result[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'accepted_sha256':hashlib.sha256((HERE/name).read_bytes()).hexdigest()}
                result[name]['bytes_equal']=result[name]['sha256']==result[name]['accepted_sha256']
        if not result['figure.pdf']['bytes_equal']:
            # Relocated font names may reorder embedded PDF font xref blocks.
            # Require every decoded stream plus all visible/vector/text content.
            import fitz
            accepted=fitz.open(HERE/'figure.pdf');rendered=fitz.open(destination/'figure.pdf')
            streams=lambda d:sorted(hashlib.sha256(d.xref_stream(i)).hexdigest() for i in range(1,d.xref_length()) if d.xref_is_stream(i))
            same=len(accepted)==len(rendered) and streams(accepted)==streams(rendered) and accepted.metadata==rendered.metadata
            for a,b in zip(accepted,rendered):
                same=same and a.rect==b.rect and a.get_text('rawdict')==b.get_text('rawdict') and a.get_drawings()==b.get_drawings()
                same=same and a.get_pixmap(matrix=fitz.Matrix(300/72,300/72)).samples==b.get_pixmap(matrix=fitz.Matrix(300/72,300/72)).samples
            result['figure.pdf']['decoded_streams_text_vectors_and_300dpi_pixels_equal']=bool(same)
            accepted.close();rendered.close()
            if not same:raise RuntimeError('Replot PDF content differs: '+str(result))
        if not result['figure.png']['bytes_equal']:raise RuntimeError('Replot PNG differs: '+str(result))
        # Original byte/pixel/vector checks above have passed before this label layer.
        _rl_parent_reproduction=dict(result)
        _rl_emit_captured(_rl_base_save,_rl_captured,destination)
        result={'accepted_parent_reproduction':_rl_parent_reproduction,
                'intentional_label_revision':True,
                'outputs_sha256':{name:hashlib.sha256((destination/name).read_bytes()).hexdigest()for name in _rl_captured},
                'final_label_proof':'rl_wording_label_proof.json'}
        with raw_open(destination/'local_replot_receipt.json','w') as f:
            f.write(json.dumps({'status':'PASS','outputs':result,'historical_paths_rebased_locally':True,'external_source_reads_forbidden':True,'science_runs':0},ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'local_render':'PASS','outputs':result},ensure_ascii=False))
    finally:
        _rl_restore_capture(_rl_base_save)
        Path.__new__=raw_pathnew;pathclass.__new__=raw_posixnew;Path.is_relative_to=raw_relative
        builtins.open=raw_open;io.open=raw_io_open;os.scandir=raw_scandir;runpy.run_path=raw_runpath
        importlib.util.spec_from_file_location=raw_spec;importlib.machinery.SourceFileLoader.get_data=raw_loaderdata;json.loads=raw_loads
        shutil.rmtree(work)
if __name__=='__main__':main()
