"""Render the adjacent CSV only; output to a separate requested HY directory."""
from pathlib import Path
import argparse,os,subprocess,sys
base=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args()
env=dict(os.environ)
env['PYTHONDONTWRITEBYTECODE']='1'
env['MPLCONFIGDIR']=str(base.parents[1]/'cache'/'c2_multitime_mpl')
subprocess.run([sys.executable,'-B',str(base/'plot.py'),'--output-dir',str(a.output_dir)],check=True,env=env)
