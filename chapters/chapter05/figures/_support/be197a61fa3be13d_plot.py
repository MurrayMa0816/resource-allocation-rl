from pathlib import Path
import hashlib,runpy
p=Path('/root/gamewar/resource_allocation_rl/tools/paper_experiments/visualization/chapter05_supervision_boxplot_20261110.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='60593a3b8c055319b2fe5dd5f83a79cdb2ad5189512213cd70907786364f2ff8'
runpy.run_path(str(p),run_name="__main__")
