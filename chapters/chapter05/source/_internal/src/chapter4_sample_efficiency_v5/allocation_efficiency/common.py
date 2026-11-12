from pathlib import Path
import torch
from allocation4.common import (write_json,cpu_copy,model_hash,file_hash,
                                source_hashes as native_hashes)

ROOT=Path('/hy-tmp/resource_allocation_results/chapter4/calibrated_snapshot_20261101')
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
