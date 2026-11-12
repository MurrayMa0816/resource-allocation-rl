"""One scope-owned CUDA worker; the learner's native CLI runs in this process."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--job-id', required=True)
args, command=parser.parse_known_args()
sys.path.insert(0,str(ROOT/'execution/source'))
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
start=time.time()
try:
    from allocation23.chapter3_candidates.cli import main
    main(command)
finally:
    record=dict(job_id=args.job_id, seconds=time.time()-start, pid=os.getpid(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                torch_version=torch.__version__)
    if torch.cuda.is_initialized():
        record.update(device=torch.cuda.get_device_name(0),
                      max_memory_allocated=torch.cuda.max_memory_allocated(),
                      max_memory_reserved=torch.cuda.max_memory_reserved())
    p=ROOT/'runtime_statistics'
    p.mkdir(exist_ok=True)
    (p/(args.job_id+'.json')).write_text(json.dumps(record,indent=2)+'\n')
