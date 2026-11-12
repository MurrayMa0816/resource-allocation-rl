"""Explicit immutable dependencies for the Chapter6 external comparison."""
from pathlib import Path
import sys
_SHARED=Path(__file__).resolve().parents[1]
if str(_SHARED) not in sys.path:sys.path.insert(0,str(_SHARED))
import c6_paths
APP=Path(__file__).resolve().parent
STUDY=c6_paths.external_study()
C6=c6_paths.primary_source()
C5=c6_paths.shared_formal285()

def bootstrap():
    for p in [APP,C6/'source',C5]:
        if str(p) not in sys.path:sys.path.insert(0,str(p))
    import hongmiao89,baselines
    if Path(hongmiao89.__file__).resolve()!=C6/'source/hongmiao89/__init__.py':raise RuntimeError('Wrong Chapter6 source')
    if Path(baselines.__file__).resolve()!=C5/'baselines/__init__.py':raise RuntimeError('Wrong frozen Chapter5 baselines')
    return hongmiao89
