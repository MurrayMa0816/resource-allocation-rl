"""Current C6 maintenance paths. Frozen execution sources retain original bytes."""
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from portable_paths import project_root, final_data_root, chapter_data_root, data_path, resolve_path

def primary_study():return chapter_data_root(6)/'studies/reasonable_allocation89_20261109'
def external_study():return chapter_data_root(6)/'studies/external_learning_comparison_20261109'
def primary_source():return project_root()/'frozen/chapter06/reasonable89_20261109'
def shared_formal285():return project_root()/'chapters/chapter05/_frozen/formal285'
def figures():return chapter_data_root(6)/'figures'
def fonts():return final_data_root()/'assets/fonts'
def original_input(value):return data_path(value)
