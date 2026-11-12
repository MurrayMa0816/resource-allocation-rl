"""Independent complete Chapter5 validation; no historical scope is modified."""
from pathlib import Path
import os
import sys

WORKSPACE=Path(os.environ.get('CH5_WORKSPACE',Path(__file__).resolve().parents[1])).resolve()
for name in ('chapter4_credit_assignment','chapter4_sample_efficiency_v5','chapter5_multiobjective'):
    path=str(WORKSPACE/name)
    if path not in sys.path:sys.path.insert(0,path)
__version__='2026.11.04.validation.1'
