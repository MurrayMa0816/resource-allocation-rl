"""Offline saved-data replot; all code and data are local to this figure."""
from pathlib import Path
import argparse, hashlib, json, os, runpy, sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
THESIS=next(p for p in HERE.parents if (p/'assets/fonts/windows').is_dir())
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=HERE/'regenerated')
args=parser.parse_args()
out=args.output_dir.resolve()
if out==HERE or (out/'figure.pdf').exists():
    raise ValueError('Choose a new output directory; accepted figure.pdf is preserved')
os.environ['THESIS_FIGURE_FONT_DIR']=str(THESIS/'assets/fonts/windows')
for key,leaf in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
    p=out/'runtime'/leaf;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
manifest=json.loads((HERE/'local_inputs_sha256.json').read_text())
for name,expected in manifest.items():
    assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected, name
sys.argv=[str(HERE/'replot_sources/renderer.py'),'--number','3.4','--output-dir',str(out)]
runpy.run_path(str(HERE/'replot_sources/renderer.py'),run_name='__main__')
