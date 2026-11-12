from pathlib import Path
import hashlib,runpy
p=Path('/root/gamewar/resource_allocation_rl/tools/paper_experiments/visualization/chapter05_supervision_boxplot_20261110.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='aa488539a6d9d89e144046cb5c4fbeb8157987f4da982f87e37b61cd91b7ca82'
runpy.run_path(str(p),run_name="__main__")
