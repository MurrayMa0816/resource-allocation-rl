from pathlib import Path
import runpy, sys
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'conditions.py'), run_name='__main__')
