"""Shared candidate runtime helpers, copied without algorithm changes."""
import ast,copy,hashlib,json
from pathlib import Path
import numpy as np
import torch

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def cpu_state(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_state(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_state(v) for v in value)
    return copy.deepcopy(value)


def digest_state(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous().numpy();h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
        elif isinstance(x,np.ndarray):h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=repr):visit(k);visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def load_definitions(path, namespace, names=None):
    """Compile the pinned authors' definitions, excluding only import plumbing.

    The unmodified downloaded file and its SHA remain in vendor.  Caller supplies
    its original torch/numpy imports under private names, avoiding global-module
    collisions and obsolete logging or MuJoCo imports.
    """
    path=Path(path); tree=ast.parse(path.read_text(),filename=str(path))
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.Assign,ast.AnnAssign))
           and (names is None or isinstance(n,(ast.Assign,ast.AnnAssign)) or n.name in names)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def finite_stats(stats):
    for key,value in stats.items():
        if isinstance(value,(int,float,np.number)) and not np.isfinite(value):raise FloatingPointError(key)
    return stats


