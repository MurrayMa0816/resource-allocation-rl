"""Stable entry point for the independent mixed-origin190-record validator."""
from pathlib import Path
import subprocess,sys
if __name__=='__main__':
    p=Path(__file__).resolve().parents[1]/'execution/independent_acceptance.py'
    raise SystemExit(subprocess.call([sys.executable,'-B',str(p),*sys.argv[1:]]))
