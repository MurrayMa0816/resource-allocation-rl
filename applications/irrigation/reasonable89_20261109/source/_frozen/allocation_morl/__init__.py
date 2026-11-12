"""Two terminal objectives on the frozen A resource allocation task."""
import sys
from pathlib import Path
_workspace=Path(__file__).resolve().parents[2]
for _path in (_workspace/'chapter4_sample_efficiency_v5',_workspace):
    if str(_path) not in sys.path:sys.path.insert(0,str(_path))
import allocation_efficiency
