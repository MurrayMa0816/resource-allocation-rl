
import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root
from pathlib import Path
import torch
from allocation4.common import (write_json,cpu_copy,model_hash,file_hash,
                                source_hashes as native_hashes)

ROOT=Path(str(_resolve_path('/hy-tmp/resource_allocation_results/chapter4/calibrated_snapshot_20261101')))
SOURCE=Path(__file__).resolve().parents[1]

def source_hashes():
    result=native_hashes()
    for p in (SOURCE/'allocation_efficiency').rglob('*.py'):
        result[str(p)]=file_hash(p)
    result[str(SOURCE/'run.py')]=file_hash(SOURCE/'run.py')
    return result

def optimizer_steps(optimizer):
    if optimizer is None:return 0
    return max((int(s.get('step',0)) for s in optimizer.state.values()),default=0)

def require_finite(value):
    if torch.is_tensor(value) and not torch.isfinite(value).all():
        raise FloatingPointError('Nonfinite tensor')
