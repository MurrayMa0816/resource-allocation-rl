#!/usr/bin/env python3
"""Minimal explicit-runtime worker for a fresh full-spend Chapter 4 attempt.

This file is an entry point, not a scheduler. It never resumes an old process or
opens a holdout. Call only after the parent admits that specific training job.
"""
from pathlib import Path
import argparse
import importlib
import json
import os
import sys

ENV_ROOT=Path(__file__).resolve().parent
SCOPE=ENV_ROOT.parents[1]
RUNTIME=Path('/hy-tmp/resource_allocation_results/chapter4/recent_baselines_ready_20261102/runtime')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    method=str(config['method']).lower()
    if method not in ('a','f1','props','fog','addq','tdmpc2','pgr'):
        raise ValueError('Expected a registered Chapter 4 method')
    out=Path(config['output_dir']).resolve()
    if not out.is_relative_to(SCOPE):
        raise ValueError('Only fresh output directories inside this new scope are allowed')
    family='jax' if method=='fog' else 'torch' if method in ('tdmpc2','pgr') else None
    expected=RUNTIME/family if family else None
    if expected is not None and Path(sys.prefix).resolve()!=expected.resolve():
        python=expected/'bin/python'
        libraries=[]
        for site in (expected/'lib/python3.11/site-packages',Path('/usr/local/lib/python3.11/dist-packages')):
            libraries.extend(str(p) for p in (site/'nvidia').glob('*/lib'))
        process_env={**os.environ,'LD_LIBRARY_PATH':':'.join(libraries+[os.environ.get('LD_LIBRARY_PATH','')])}
        # Re-execute THIS bootstrap-aware file. The old baseline CLI would
        # re-execute its own module and silently lose a prior in-memory binding.
        os.execve(str(python),[str(python),str(Path(__file__).resolve()),'--config',str(args.config.resolve())],process_env)
    temporary=SCOPE/'tmp/env'/method
    temporary.mkdir(parents=True,exist_ok=True)
    for name in ('TMPDIR','TMP','TEMP'):
        os.environ[name]=str(temporary)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(temporary/'cache'),WANDB_MODE='disabled')
    if str(config.get('device','cpu'))=='cpu':
        os.environ['CUDA_VISIBLE_DEVICES']=''
    if method=='fog':
        config.setdefault('jax_cache_dir',str(temporary/'jax_cache'))
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(ENV_ROOT))
    from chapter4_full_spend_env.bootstrap import install
    install(audit_dir=out/'resource_audit',context={
        'method':method,'seed':int(config['seed']),'domain':config['domain'],
        'variant':config.get('variant','full_spend_only')})
    module='reference_runner' if method in ('a','f1') else method+'_runner'
    runner=importlib.import_module('chapter4_recent_baselines.runners.'+module)
    runner.run(config)


if __name__=='__main__':
    main()
