#!/usr/bin/env python3
"""Reproduce the current maps from local geometry and saved numerical values."""
from pathlib import Path
import argparse,importlib.util,sys,tempfile,shutil,json
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=HERE);args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/'source'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
with tempfile.TemporaryDirectory(prefix='.render-map-',dir=HERE)as tmp:
    work=Path(tmp);m=module('map_family');m.ROOT=work;(work/'audit').mkdir();sys.argv=[str(HERE/'plot.py'),'--figure','6.6'];m.main()
    stage=work/'figures'/'regional'
    if '6.6'=='6.1':
        legend=module('scene_clarity');legend.PARENT=stage;legend.REVISION=work;sys.argv=[str(HERE/'plot.py'),'--output-dir',str(work/'scene_clarity')];legend.main();stage=work/'scene_clarity'
    for source in stage.iterdir():
        if source.is_file()and source.suffix in ('.pdf','.png','.svg'):
            shutil.copyfile(source,args.output_dir/source.name)

# Additional native map uses the same physical geometry as Figure 6.4.
import subprocess as _native_subprocess
_native_subprocess.run([sys.executable,str(HERE/'plot_native_map.py'),'--output-dir',str(args.output_dir)],check=True)

# Restored native dimensions and measured15bp image spacing.
subprocess.run([sys.executable,str(HERE/'plot_fullsize15.py'),'--output-dir',str(args.output_dir)],check=True)
