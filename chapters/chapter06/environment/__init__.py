"""C6 source lookup without environment import or construction."""
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[3]
if str(PROJECT)not in sys.path:sys.path.insert(0,str(PROJECT))
from portable_paths import project_root,chapter_data_root

def source_paths():
 base=project_root()/'frozen/chapter06/reasonable89_20261109'
 return {'environment':base/'source/hongmiao89','physical_core':base/'source/_frozen/irrigation_ch6','frozen_inputs':base/'source/_frozen/irrigation_ch6/_core/data/hongmiao93','saved_fixed_inputs':chapter_data_root(6)/'studies/reasonable_allocation89_20261109/formal/fixed_inputs89.json'}
