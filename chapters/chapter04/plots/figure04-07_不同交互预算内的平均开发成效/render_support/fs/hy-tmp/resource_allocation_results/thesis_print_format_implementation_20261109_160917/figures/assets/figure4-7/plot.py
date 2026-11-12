from pathlib import Path
import runpy,sys
sys.argv=[str(Path(__file__)),"--number",'4.7']+sys.argv[1:]
runpy.run_path('/hy-tmp/resource_allocation_results/thesis_print_format_implementation_20261109_160917/figures/render_print_figures.py',run_name="__main__")
