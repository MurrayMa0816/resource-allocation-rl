from pathlib import Path
import runpy, sys
sys.argv = [str(Path(__file__)), '--number', '3.4', *sys.argv[1:]]
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'main_learning.py'), run_name='__main__')
