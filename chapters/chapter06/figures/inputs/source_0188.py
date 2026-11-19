#!/usr/bin/env python3
"""Reproduce the currently inserted horizontal learning figure from local CSVs."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / 'plot_horizontal15.py'), run_name='__main__')
