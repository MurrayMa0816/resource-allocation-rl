"""Portable offline replot of the accepted even-spacing figure."""
from pathlib import Path
import argparse,hashlib,json,os,runpy,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
THESIS=next((p for p in HERE.parents if (p/'assets/fonts/windows').is_dir()),None)
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-dir',type=Path,default=HERE/'regenerated')
p.add_argument('--font-dir',type=Path)
a=p.parse_args();out=a.output_dir.resolve()
if out==HERE or (out/'figure.pdf').exists():raise ValueError('Choose a new output directory; accepted figure.pdf is preserved')
fonts=a.font_dir or (THESIS/'assets/fonts/windows' if THESIS is not None else None)
if fonts is None or not fonts.is_dir():raise ValueError('Pass --font-dir when rendering outside the thesis')
os.environ['THESIS_FIGURE_FONT_DIR']=str(fonts.resolve())
for key,leaf in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
 q=out/'runtime'/leaf;q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
for name,h in json.loads((HERE/'local_inputs_sha256.json').read_text()).items():
 assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h,name
sys.argv=[str(HERE/'replot_sources/renderer.py'),'--number','2.5','--output-dir',str(out),'--font-dir',str(fonts.resolve())]
runpy.run_path(str(HERE/'replot_sources/renderer.py'),run_name='__main__')

# User-approved final experiment wording layer, after existing native checks.
runpy.run_path(str(HERE/'wording_layer_20261119/label_export.py'))['finalize'](out,HERE,fonts)
