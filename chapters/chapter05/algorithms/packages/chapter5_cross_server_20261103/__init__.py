"""Independent Chapter 5 comparison/ablation execution scope.

Historical and registered science modules remain untouched. This package adds
explicit ablation learning objectives, dense evaluation and a new run manifest.
"""
from pathlib import Path
import os
import sys

WORKSPACE = Path(__file__).resolve().parent.parent
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _folder in ('chapter4_credit_assignment',
                'chapter4_sample_efficiency_v5', 'chapter5_multiobjective'):
    _path = str(WORKSPACE / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

__version__ = '2026.11.03.1'
