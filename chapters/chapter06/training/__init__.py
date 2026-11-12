"""Saved C6 training/DEV lookup; completed controllers are not dispatched."""
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[3]
if str(PROJECT)not in sys.path:sys.path.insert(0,str(PROJECT))
from portable_paths import project_root,chapter_data_root

def record_paths():
 base=chapter_data_root(6)/'studies'
 return {'SeDVal':base/'reasonable_allocation89_20261109/formal/training','PASTA':base/'external_learning_comparison_20261109/formal','COLA':base/'external_learning_comparison_20261109/additional_cola_20261109/formal'}
