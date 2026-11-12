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

FIGURE_NUMBER='5.6'
LABEL_REPLACEMENTS={'覆盖预测\n$C$': '达标覆盖率\n$C$', '成效预测\n$J$': '平均成效\n$J$', '加权预测\n$F_{0.8}$': '加权评价\n$F_{0.8}$'}

def install_language_labels():
    """Change only final output labels, after the parent renderer's guards.

    Saved observations, scientific artists, native dimensions and10.5bp fonts
    remain untouched. The parent reproduction and layout checks still run with
    their own historical labels. No original scientific source is modified.
    """
    import copy
    import numpy as np
    from matplotlib.figure import Figure
    original_save = Figure.savefig
    def plain(value):
        if isinstance(value, np.ndarray): return plain(value.tolist())
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, (tuple,list)): return [plain(x) for x in value]
        if isinstance(value, dict): return {str(k):plain(v) for k,v in value.items()}
        return value
    def scientific(fig):
        return plain(dict(native_inches=fig.get_size_inches(), axes=[dict(
            position=ax.get_position().bounds, xlim=ax.get_xlim(),ylim=ax.get_ylim(),
            xscale=ax.get_xscale(),yscale=ax.get_yscale(),xticks=ax.get_xticks(),yticks=ax.get_yticks(),
            lines=[dict(x=l.get_xdata(),y=l.get_ydata(),color=l.get_color(),width=l.get_linewidth(),
                        style=l.get_linestyle(),marker=l.get_marker(),ms=l.get_markersize(),alpha=l.get_alpha())for l in ax.lines],
            collections=[dict(paths=[dict(vertices=p.vertices,codes=p.codes)for p in c.get_paths()],
                              offsets=c.get_offsets(),array=c.get_array(),clim=c.get_clim(),
                              fc=c.get_facecolors(),ec=c.get_edgecolors(),lw=c.get_linewidths(),alpha=c.get_alpha())for c in ax.collections],
            patches=[dict(vertices=p.get_path().vertices,codes=p.get_path().codes,
                          fc=p.get_facecolor(),ec=p.get_edgecolor(),lw=p.get_linewidth(),alpha=p.get_alpha())for p in ax.patches],
            images=[dict(array=i.get_array(),extent=i.get_extent(),clim=i.get_clim())for i in ax.images])for ax in fig.axes]))
    def save(fig,path,*args,**kwargs):
        try: target=Path(path)
        except TypeError: return original_save(fig,path,*args,**kwargs)
        if target.name not in ['figure.pdf','figure.png','figure.svg']:
            return original_save(fig,path,*args,**kwargs)
        def pdf_label_gap():
            import fitz
            buf=io.BytesIO();original_save(fig,buf,format='pdf',metadata={'CreationDate':None,'ModDate':None})
            with fitz.open(stream=buf.getvalue(),filetype='pdf') as document:
                page=document[0]
                spans=[span for block in page.get_text('dict')['blocks'] for line in block.get('lines',[]) for span in line['spans']]
                legend_top=min(span['bbox'][1] for span in spans if 'SeDVal' in span['text'])
                axes_bottom=max(page.rect.height-ax.get_window_extent().y0*72/fig.dpi for ax in fig.axes)
                label_bottom=max(span['bbox'][3] for span in spans if span['bbox'][1]>axes_bottom+3 and span['bbox'][3]<legend_top-2)
                return legend_top-label_bottom
        original_gap=pdf_label_gap()
        original_leading=[(t,t._linespacing) for ax in fig.axes for t in ax.get_xticklabels()]
        original_labelpad=[(ax,ax.xaxis.labelpad) for ax in fig.axes]
        before=scientific(fig); changed=[]; formatters=[]
        for ax in fig.axes:
            artists = ax.get_xticklabels() if FIGURE_NUMBER=='5.6' else [ax.xaxis.label]
            for artist in artists:
                old=artist.get_text();new=LABEL_REPLACEMENTS.get(old,old)
                if new != old:
                    assert artist.get_fontsize()==10.5,(old,artist.get_fontsize())
                    changed.append((artist,old,new));artist.set_text(new)
        assert len(changed)==(6 if FIGURE_NUMBER=='5.6' else 4),[(a,b)for _,a,b in changed]
        for ax in fig.axes:
            if FIGURE_NUMBER=='5.6':
                from matplotlib.ticker import FixedFormatter
                formatter=ax.xaxis.get_major_formatter()
                original_labels=[old for t,old,new in changed if t in ax.get_xticklabels()]
                formatters.append((ax.xaxis,formatter))
                ax.xaxis.set_major_formatter(FixedFormatter([LABEL_REPLACEMENTS.get(label,label) for label in original_labels]))
        fig.canvas.draw()
        assert all(t.get_text()==new for t,old,new in changed)
        initial_gap=pdf_label_gap()
        for iteration in range(4):
            gap=pdf_label_gap();difference=gap-original_gap
            if abs(difference)<.0001:break
            if FIGURE_NUMBER=='5.6':
                for t,old,new in changed:t.set_linespacing(t._linespacing+.01)
                probe_gap=pdf_label_gap();slope=(probe_gap-gap)/.01
                assert slope<0
                for t,old,new in changed:t.set_linespacing(t._linespacing-.01-difference/slope)
            else:
                for ax in fig.axes:ax.xaxis.labelpad+=difference
            fig.canvas.draw()
        final_gap=pdf_label_gap()
        assert abs(final_gap-original_gap)<.003,(original_gap,initial_gap,final_gap)
        # Remove only the colloquial direction note from the existing y label.
        ylabel_changes=0
        for artist in fig.texts:
            old=artist.get_text()
            if old=='预测均方误差（越低越好）':
                assert artist.get_fontsize()==10.5
                new='预测均方误差';changed.append((artist,old,new));artist.set_text(new)
                ylabel_changes+=1
        assert ylabel_changes==1
        fig.canvas.draw()
        assert abs(pdf_label_gap()-final_gap)<.003
        geometry=[dict(before=old,after=new,font_bp=t.get_fontsize(),
                       bbox_pixels=list(t.get_window_extent(fig.canvas.get_renderer()).bounds))for t,old,new in changed]
        for t,old,new in changed:
            b=t.get_window_extent(fig.canvas.get_renderer())
            assert b.x0>=-.8 and b.x1<=fig.bbox.x1+.8,(new,b.bounds,fig.bbox.bounds)
        try:
            assert scientific(fig)==before,'Numerical artists or axes changed during label revision'
            result=original_save(fig,path,*args,**kwargs)
            after=scientific(fig);assert after==before
            proof=dict(status='PASS',number=FIGURE_NUMBER,change='Only final labels clarified',
                       labels=geometry,numeric_artists_and_axes_before=before,
                       numeric_artists_and_axes_after=after,numeric_artists_and_axes_exact=True,
                       native_canvas_and_fonts_exact=True,original_xlabel_to_legend_bp=original_gap,
                       initial_revised_xlabel_to_legend_bp=initial_gap,final_xlabel_to_legend_bp=final_gap,
                       only_label_glyph_leading_or_padding_calibrated=True,
                       environment_steps=0,model_loads=0,training_runs=0)
            (target.parent/'language_label_proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
            return result
        finally:
            for axis,formatter in formatters:axis.set_major_formatter(formatter)
            for t,leading in original_leading:t.set_linespacing(leading)
            for ax,pad in original_labelpad:ax.xaxis.labelpad=pad
            for t,old,new in changed:t.set_text(old)
    Figure.savefig=save
    return original_save

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
    original_figure_save = install_language_labels()
    try:
        runpath(str(entry),run_name='__main__')
        for name in ['figure.pdf','figure.png','figure.svg','language_label_proof.json','plotted_points.csv','plotted_values.json','numeric_artist_values_before_after.json','render_geometry.json','spacing_proof.json','math_font_proof.json','figure.before_unicode.pdf']:
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
        proof=json.loads((destination/'language_label_proof.json').read_text())
        assert proof['status']=='PASS' and proof['numeric_artists_and_axes_exact']
        import fitz
        with fitz.open(HERE/'figure.pdf') as accepted, fitz.open(destination/'figure.pdf') as rendered:
            assert len(accepted)==len(rendered)==1 and accepted[0].rect==rendered[0].rect
        result['intentional_label_revision']=True
        result['native_canvas_exact']=True
        result['numeric_artists_and_axes_exact']=True
        with raw_open(destination/'local_replot_receipt.json','w') as f:
            f.write(json.dumps({'status':'PASS','outputs':result,'historical_paths_rebased_locally':True,'external_source_reads_forbidden':True,'science_runs':0},ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'local_render':'PASS','outputs':result},ensure_ascii=False))
    finally:
        from matplotlib.figure import Figure
        Figure.savefig=original_figure_save
        Path.__new__=raw_pathnew;pathclass.__new__=raw_posixnew;Path.is_relative_to=raw_relative
        builtins.open=raw_open;io.open=raw_io_open;os.scandir=raw_scandir;runpy.run_path=raw_runpath
        importlib.util.spec_from_file_location=raw_spec;importlib.machinery.SourceFileLoader.get_data=raw_loaderdata;json.loads=raw_loads
        shutil.rmtree(work)
if __name__=='__main__':main()
