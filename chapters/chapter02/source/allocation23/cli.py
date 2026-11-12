"""The chapter runner is the public CLI; algorithms retain their package names."""
from pathlib import Path
import runpy, sys
def main(argv=None):
    entry=Path(__file__).resolve().parents[2]/'run.py'
    sys.argv=[str(entry),*(sys.argv[1:] if argv is None else argv)]
    runpy.run_path(str(entry),run_name='__main__')
