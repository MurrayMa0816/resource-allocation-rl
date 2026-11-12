"""Reproduce the saved real stochastic curve from adjacent CSV; no science runs."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

here = Path(__file__).resolve().parent
publication = here.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=publication / 'reproduced' / here.name)
args = parser.parse_args()
env = os.environ.copy()
env['PYTHONDONTWRITEBYTECODE'] = '1'
subprocess.run([sys.executable, '-B', str(here / 'plot.py'), '--output-dir', str(args.output_dir)], check=True, env=env)
