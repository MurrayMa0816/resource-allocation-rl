"""Shared compact artifacts and explicit terminal-objective accounting."""
from pathlib import Path
import os
import numpy as np
import torch
from allocation_efficiency.common import (write_json,cpu_copy,model_hash,file_hash,
    optimizer_steps,source_hashes as frozen_source_hashes)

WEIGHTS=(0.,.5,.8,1.)
METHODS=('fixed','conditioned')
SOURCE=Path(__file__).resolve().parents[1]

def weight_key(value):return f"w{round(100*float(value)):03d}"

def source_hashes():
    result=frozen_source_hashes()
    for name in ('__init__.py','common.py','models.py','values.py','learner.py','rollout.py','training.py'):
        path=SOURCE/'allocation_morl'/name
        result[str(path)]=file_hash(path)
    path=SOURCE/'run.py';result[str(path)]=file_hash(path)
    return result

def coverage_metrics(quality,tau):
    y=np.asarray(quality,dtype=np.float64)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all():raise ValueError('Finite terminal quality vector required')
    n=len(y);k=int(np.count_nonzero(y>=tau))
    return dict(K=k,N=n,C=k/n,Q=k/n,quality_mean=float(y.mean()),quality_min=float(y.min()),quality_p10=float(np.quantile(y,.1)))

def summary(rows):
    if not rows:raise ValueError('Nonempty evaluation required')
    scores=np.array([r['score_x100'] for r in rows])
    return dict(count=len(rows),score=float(scores.mean()),sd=float(scores.std(ddof=1)) if len(rows)>1 else 0.,
        mean_K=float(np.mean([r['K'] for r in rows])),mean_C=float(np.mean([r['C'] for r in rows])),
        mean_quality_min=float(np.mean([r['quality_min'] for r in rows])),mean_quality_p10=float(np.mean([r['quality_p10'] for r in rows])))

def state_hash(value):
    import hashlib
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous().numpy();h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
        elif isinstance(x,np.ndarray):h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=lambda item:(type(item).__name__,repr(item))):visit(k);visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for item in x:visit(item)
        else:h.update(repr((type(x).__name__,x)).encode())
    visit(value);return h.hexdigest()

def atomic_torch_save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
    torch.save(payload,temporary);os.replace(temporary,path)

def save_trace(path,results):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays=dict(actions=np.stack([x['actions'] for x in results]),quality=np.stack([x['quality'] for x in results]),
        rewards=np.stack([x['rewards'] for x in results]),preference=np.array([x['preference'] for x in results]))
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(temporary,path)
    return dict(path=str(path),sha256=file_hash(path),arrays={k:list(v.shape) for k,v in arrays.items()})
