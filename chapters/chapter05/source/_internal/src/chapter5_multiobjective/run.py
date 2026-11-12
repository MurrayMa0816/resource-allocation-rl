"""CPU-only multi-objective training entry."""
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import allocation_morl
if Path(allocation_morl.__file__).resolve().parent!=Path(__file__).resolve().parent/'allocation_morl':
    raise RuntimeError('Wrong allocation_morl package')
from allocation_morl.training import main
if __name__=='__main__':main()
