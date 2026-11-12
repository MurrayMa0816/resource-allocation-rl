"""Saved C6 evaluation lookup; no environment reset, inference or TEST runs."""
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[3]
if str(PROJECT)not in sys.path:sys.path.insert(0,str(PROJECT))
from portable_paths import project_root,chapter_data_root

def record_paths():
 base=chapter_data_root(6)
 return {'completed_evaluations':base/'studies/reasonable_allocation89_20261109/formal/evaluation','figures':base/'figures','model_index':base/'MODELS.json'}
