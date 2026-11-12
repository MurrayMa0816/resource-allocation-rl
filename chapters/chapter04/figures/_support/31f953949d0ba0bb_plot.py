from pathlib import Path
import runpy,sys
sys.argv=[str(Path(__file__)),"--number",'4.4']+sys.argv[1:]
runpy.run_path('/hy-tmp/resource_allocation_results/thesis_main_learning_style_20261109/render_learning_figures.py',run_name="__main__")
