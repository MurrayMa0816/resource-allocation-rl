
import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root
from pathlib import Path
from allocation_efficiency.common import (write_json,cpu_copy,model_hash,file_hash,optimizer_steps,source_hashes as base_hashes)
ROOT=Path(str(_resolve_path('/hy-tmp/resource_allocation_results/chapter4/reuse_optimization_20261102')))
SOURCE=Path(__file__).resolve().parents[1]
def source_hashes():
    result=base_hashes()
    for p in (SOURCE/'allocation_reuse').glob('*.py'):result[str(p)]=file_hash(p)
    for n in ('run.py','register.py','launch.py'):
        p=SOURCE/n
        if p.exists():result[str(p)]=file_hash(p)
    return result
