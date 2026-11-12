#!/usr/bin/env python3
"""Reproduce the current figure from the CSVs and local rendering source."""
from pathlib import Path
import argparse,subprocess,sys,tempfile,shutil
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=HERE);args=p.parse_args()
args.output_dir.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='.render-',dir=HERE)as tmp:
    out=Path(tmp)/'output'
    subprocess.run([sys.executable,str(HERE/'source/spacing.py'),'--number','6.5','--output-dir',str(out)],check=True)
    for name in ['figure.pdf','figure.svg','figure.png','numeric_artist_values_before_after.json','render_geometry.json']:
        shutil.copyfile(out/name,args.output_dir/name)
