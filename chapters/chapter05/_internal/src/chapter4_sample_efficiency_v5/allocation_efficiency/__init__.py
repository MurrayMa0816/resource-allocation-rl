"""Public-data sample reuse on the frozen chapter23 A task."""
import sys
from pathlib import Path
_workspace = Path(__file__).resolve().parents[2]
for _root in (_workspace/'chapter4_credit_assignment', _workspace):
    _p = str(_root)
    if _p not in sys.path: sys.path.insert(0,_p)
import allocation4  # pins allocation23 to the released standalone package
