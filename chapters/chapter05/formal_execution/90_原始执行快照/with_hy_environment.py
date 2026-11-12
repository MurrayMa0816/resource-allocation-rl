"""Execute one registered postprocess entry with all new caches on HY."""
from pathlib import Path
import os
import sys

R = Path(__file__).resolve().parents[1]
if len(sys.argv) < 2:
    raise SystemExit('A registered execution script is required')
script = (R / 'execution' / sys.argv[1]).resolve(strict=True)
if script.parent != R / 'execution' or script.suffix != '.py':
    raise SystemExit('Postprocess entry escaped the registered execution directory')
paths = dict(XDG_CACHE_HOME='cache', MPLCONFIGDIR='cache/matplotlib', TMPDIR='tmp',
             TORCH_HOME='cache/torch', CUDA_CACHE_PATH='cache/cuda',
             PYTORCH_KERNEL_CACHE_PATH='cache/torch_kernels', TRITON_CACHE_DIR='cache/triton',
             NUMBA_CACHE_DIR='cache/numba', TEXMFVAR='cache/texmf-var', TEXMFCONFIG='cache/texmf-config')
for key, sub in paths.items():
    path = R / sub
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                  PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='', CH5_GPU_UUID='')
os.execv(sys.executable, [sys.executable, '-B', str(script), *sys.argv[2:]])
