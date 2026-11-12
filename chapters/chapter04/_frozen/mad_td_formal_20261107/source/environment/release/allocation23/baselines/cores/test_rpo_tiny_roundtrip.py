"""Positive-small-target RPO action/control inverse, including terminal zero."""
import json
from pathlib import Path
import types
import numpy as np
import torch
from .chapter2_rpo import RPOCore, rpo_parameters


def run(output):
    schema=[dict(name='caps',start=0,stop=4,divisor=.00015),
            dict(name='budget',start=4,stop=5,divisor=.015)]
    torch.manual_seed(0)
    core=RPOCore(5,4,rpo_parameters(tiny=True,horizon=4),observation_schema=schema)
    core.observed_steps=core.p['learning_starts']
    weights=torch.as_tensor([[.1,.2,.3,.4]],dtype=torch.float64)
    def controlled_sample(self,obs,deterministic=False,correction=True,normalized=None,geometry=None):
        caps,target=geometry
        allocation=target*weights
        return allocation,None,allocation,caps,target
    # Isolate the public coordinate inverse; retain the real constraint parser.
    core._sample=types.MethodType(controlled_sample,core)
    rows=[]
    for physical in (1e-3,1e-40,1e-305,0.):
        obs=np.r_[np.full(4,physical/.00015),physical/.015]
        raw=torch.as_tensor(obs,dtype=torch.float64)[None]
        caps,target=core.constraints(raw)
        original=(target*weights).numpy()[0]
        for deterministic in (False,True):
            control=core.act(obs,deterministic)
            decoded=float(target.item())*(control+1)/2
            if target.item()>0:
                relative=float(np.max(np.abs(decoded-original))/float(target.item()))
                assert relative<1e-14,(physical,relative)
                assert np.allclose((control+1)/2,weights.numpy()[0],rtol=1e-14,atol=1e-14)
            else:
                relative=0.;assert np.array_equal(control,np.full(4,-1.));assert np.array_equal(decoded,np.zeros(4))
            assert np.all(np.isfinite(control)) and np.max(np.abs(control))<=1.
            rows.append(dict(physical_target=physical,internal_target=float(target.item()),
                             deterministic=deterministic,relative_roundtrip_error=relative))
    result=dict(status='passed',cases=rows,scope='RPO action/control arithmetic only; updates/physical constraints/reward unchanged')
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    import sys
    run(sys.argv[1])
