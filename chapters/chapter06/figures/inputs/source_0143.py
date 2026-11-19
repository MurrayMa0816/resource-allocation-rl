#!/usr/bin/env python3
"""Reproduce the current centered two-page paired layout of figure 6.4."""
from pathlib import Path
import argparse
import subprocess
import sys

HERE = Path(__file__).resolve().parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    subprocess.run([sys.executable, str(HERE / 'plot_centered_pairs.py'),
                    '--output-dir', str(args.output_dir.resolve())], check=True)
