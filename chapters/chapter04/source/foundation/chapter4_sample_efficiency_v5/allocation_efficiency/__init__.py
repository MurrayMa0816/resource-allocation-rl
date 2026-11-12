"""Public-data sample reuse on the frozen chapter23 A task."""

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root
import sys
for _p in [str(_project_root()/'chapters/chapter04/source/foundation'/'chapter4_credit_assignment'),str(_project_root()/'chapters/chapter04/source/foundation')]:
    if _p not in sys.path: sys.path.insert(0,_p)
import allocation4  # pins allocation23 to the released standalone package
