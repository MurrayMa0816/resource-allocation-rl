#!/usr/bin/env python3
"""Current read-only C6 entry; historical training dispatch is not exposed here."""
from pathlib import Path
import os,sys
PROJECT=Path(__file__).resolve().parents[3]
os.execv(sys.executable,[sys.executable,'-B',str(PROJECT/'chapters/chapter06/launcher.py'),*sys.argv[1:]])
