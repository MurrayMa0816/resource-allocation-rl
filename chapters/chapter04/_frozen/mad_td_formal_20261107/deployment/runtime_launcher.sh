#!/bin/bash
set -euo pipefail
# CPU/GPU choice is explicit in the frozen registry; no implicit CUDA fallback.
if [ "$#" -lt 4 ]; then echo 'usage: runtime_launcher.sh JOB_ID cpu|cuda GPU_UUID_OR_none WORKER.py [args...]' >&2; exit 64; fi
MAD_JOB_ID="$1"; MAD_DEVICE="$2"; MAD_UUID="$3"; MAD_WORKER="$4"; shift 4
case "$MAD_JOB_ID" in *[!a-zA-Z0-9_.-]*|'') echo 'invalid job id' >&2; exit 65;; esac
MAD_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
case "${MAD_RUNTIME_KIND:-jax}" in
 jax) MAD_PYTHON='/hy-tmp/resource_allocation_results/chapter4/recent_baselines_ready_20261102/runtime/jax/bin/python';;
 torch) MAD_PYTHON='/hy-tmp/resource_allocation_results/chapter4/recent_baselines_ready_20261102/runtime/torch/bin/python';;
 *) echo 'unsupported registered runtime' >&2; exit 69;;
esac
MAD_AFFINITY="${MAD_CPU_AFFINITY:-}"
if [ -z "$MAD_AFFINITY" ]; then MAD_AFFINITY="$(/usr/local/bin/python3 -B -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0))[:4])) )')"; fi
case "$MAD_AFFINITY" in *[!0-9,-]*|'') echo 'invalid explicit CPU affinity' >&2; exit 68;; esac
export MAD_CPU_AFFINITY="$MAD_AFFINITY"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_force_host_platform_device_count=1'
export WANDB_MODE=disabled WANDB_DISABLED=true
export PYTHONPATH="/hy-tmp/resource_allocation_results/chapter4/mad_td_pilot_20261107/runtime/hydra_overlay"
# The local JAX executable is a symlink; the remote one adds a portable library
# shim. Discover the existing library directories without importing Torch/JAX.
mapfile -t MAD_LIBRARY_INFO < <("$MAD_PYTHON" -B - <<'PY'
import importlib.util,sys
from pathlib import Path
paths=[];cuda=[]
for item in sys.path:
    p=Path(item)
    for q in p.glob('nvidia/*/lib'):
        if q.is_dir() and str(q) not in paths:paths.append(str(q))
    q=p/'nvidia/cuda_nvcc'
    if(q/'nvvm/libdevice').is_dir():cuda.append(str(q))
spec=importlib.util.find_spec('torch')
if spec and spec.origin:
    q=Path(spec.origin).parent/'lib'
    if q.is_dir():paths.append(str(q))
print(':'.join(paths));print(cuda[0] if cuda else '')
PY
)
if [ -n "${MAD_LIBRARY_INFO[0]:-}" ]; then export LD_LIBRARY_PATH="${MAD_LIBRARY_INFO[0]}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
if [ -n "${MAD_LIBRARY_INFO[1]:-}" ]; then export XLA_FLAGS="$XLA_FLAGS --xla_gpu_cuda_data_dir=${MAD_LIBRARY_INFO[1]}"; fi
MAD_CACHE="$MAD_ROOT/runtime/worker_caches/$MAD_JOB_ID"
mkdir -p "$MAD_CACHE"
for MAD_CACHE_NAME in tmpdir xdg_cache_home mplconfigdir torch_home cuda_cache_path torch_extensions_dir triton_cache_dir torchinductor_cache_dir jax_compilation_cache_dir; do mkdir -p "$MAD_CACHE/$MAD_CACHE_NAME"; done
export TMPDIR="$MAD_CACHE/tmpdir" XDG_CACHE_HOME="$MAD_CACHE/xdg_cache_home" MPLCONFIGDIR="$MAD_CACHE/mplconfigdir"
export TORCH_HOME="$MAD_CACHE/torch_home" CUDA_CACHE_PATH="$MAD_CACHE/cuda_cache_path" TORCH_EXTENSIONS_DIR="$MAD_CACHE/torch_extensions_dir"
export TRITON_CACHE_DIR="$MAD_CACHE/triton_cache_dir" TORCHINDUCTOR_CACHE_DIR="$MAD_CACHE/torchinductor_cache_dir" JAX_COMPILATION_CACHE_DIR="$MAD_CACHE/jax_compilation_cache_dir"
case "$MAD_DEVICE" in
 cpu) export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu;;
 cuda) case "$MAD_UUID" in GPU-*) export CUDA_VISIBLE_DEVICES="$MAD_UUID" JAX_PLATFORMS=cuda;; *) echo 'explicit physical GPU UUID required' >&2; exit 66;; esac;;
 *) echo 'device must be cpu or cuda' >&2; exit 67;;
esac
cd "$MAD_ROOT"
exec taskset -c "$MAD_AFFINITY" "$MAD_PYTHON" -B "$MAD_WORKER" "$@"
