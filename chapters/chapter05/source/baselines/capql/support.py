"""Private author-source loader and output conversion; no task imports."""
from pathlib import Path
import ast,copy,hashlib,math
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def cpu_state(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return type(value)(cpu_state(v) for v in value)
    if isinstance(value,np.ndarray): return value.copy()
    return copy.deepcopy(value)
def finite_stats(values):
    for name,value in values.items():
        if isinstance(value,(float,np.floating)) and not math.isfinite(float(value)): raise FloatingPointError(name)
    return values
def load_definitions(path,namespace,names=None):
    tree=ast.parse(Path(path).read_text());nodes=[]
    for node in tree.body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and (names is None or node.name in names): nodes.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and (names is None or x.id in names) for x in node.targets): nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),namespace)
