"""RNG and CUDA accounting helpers, preserved from prior tested runtime."""
import copy,random,json
import numpy as np
import torch

def safe(x):
    if torch.is_tensor(x): return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [safe(v) for v in x]
    return x


def line(f, value):
    f.write(json.dumps(safe(value),separators=(',',':'),allow_nan=False)+'\n')


def rng_state(agent):
    private=[];seen=set()
    def visit(o):
        if id(o) in seen:return
        seen.add(id(o))
        if isinstance(o,torch.Generator):private.append((o,'torch',o.get_state().clone()))
        elif isinstance(o,np.random.Generator):private.append((o,'numpy',copy.deepcopy(o.bit_generator.state)))
        elif isinstance(o,np.random.RandomState):private.append((o,'legacy_numpy',o.get_state()))
        elif isinstance(o,random.Random):private.append((o,'python',o.getstate()))
        elif isinstance(o,(list,tuple,dict)):
            for v in (o.values() if isinstance(o,dict) else o):visit(v)
        elif hasattr(o,'__dict__') and not isinstance(o,(torch.Tensor,torch.nn.Module,torch.optim.Optimizer,np.ndarray)):
            for v in vars(o).values():visit(v)
    visit(agent)
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],private)


def restore_rng(s):
    random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2])
    if s[3]:torch.cuda.set_rng_state_all(s[3])
    for obj,kind,value in s[4]:
        if kind in ('torch','legacy_numpy'):obj.set_state(value)
        elif kind=='python':obj.setstate(value)
        else:obj.bit_generator.state=value


def parameter_devices(agent):
    found=set();seen=set()
    def visit(o):
        if id(o) in seen:return
        seen.add(id(o))
        if isinstance(o,torch.nn.Module):found.update(str(p.device) for p in o.parameters())
        elif isinstance(o,(dict,list,tuple)):
            for v in (o.values() if isinstance(o,dict) else o):visit(v)
        elif hasattr(o,'__dict__') and not isinstance(o,(torch.Tensor,torch.optim.Optimizer,np.ndarray)):
            for v in vars(o).values():visit(v)
    visit(agent);return sorted(found)


def optimizer_cuda_count(agent):
    count=0;seen=set()
    def visit(o):
        nonlocal count
        if id(o) in seen:return
        seen.add(id(o))
        if isinstance(o,torch.optim.Optimizer):
            for state in o.state.values():
                for key,value in state.items():
                    if key in ('exp_avg','exp_avg_sq','max_exp_avg_sq') and torch.is_tensor(value):
                        count+=int(value.is_cuda)
        elif isinstance(o,(dict,list,tuple)):
            for v in (o.values() if isinstance(o,dict) else o):visit(v)
        elif hasattr(o,'__dict__') and not isinstance(o,(torch.Tensor,torch.nn.Module,np.ndarray)):
            for v in vars(o).values():visit(v)
    visit(agent);return count

