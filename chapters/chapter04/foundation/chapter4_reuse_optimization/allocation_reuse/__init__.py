"""Frozen-prefix replay study; native environment and A decision generator retained."""

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root
import sys
V5=str(_project_root()/'chapters/chapter04/foundation'/'chapter4_sample_efficiency_v5')
if V5 not in sys.path:sys.path.insert(0,V5)
import allocation_efficiency
