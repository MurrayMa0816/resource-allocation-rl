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
        with raw_open(destination/'local_replot_receipt.json','w') as f:
            f.write(json.dumps({'status':'PASS','outputs':result,'historical_paths_rebased_locally':True,'external_source_reads_forbidden':True,'science_runs':0},ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'local_render':'PASS','outputs':result},ensure_ascii=False))
    finally:
        Path.__new__=raw_pathnew;pathclass.__new__=raw_posixnew;Path.is_relative_to=raw_relative
        builtins.open=raw_open;io.open=raw_io_open;os.scandir=raw_scandir;runpy.run_path=raw_runpath
        importlib.util.spec_from_file_location=raw_spec;importlib.machinery.SourceFileLoader.get_data=raw_loaderdata;json.loads=raw_loads
        shutil.rmtree(work)
if __name__=='__main__':main()
