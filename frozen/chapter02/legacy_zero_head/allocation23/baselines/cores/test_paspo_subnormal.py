"""Regression for legal positive balances below the old 1e-300 divisor floor."""
import json
from pathlib import Path
import numpy as np
import torch
from .chapter2_paspo import CappedAutoregressiveBeta


def run(output):
    n=101
    schema=[dict(name='caps',start=0,stop=n,divisor=.00015),
            dict(name='budget',start=n,stop=n+1,divisor=.015)]
    torch.manual_seed(0)
    actor=CappedAutoregressiveBeta(n+1,n,8,8,schema)
    rows=[]
    for target in (1e-3,1e-305,np.finfo(np.float64).tiny,np.nextafter(0.,1.),0.):
        obs=torch.as_tensor(np.r_[np.full(n,target/.00015),target/.015],dtype=torch.float64)[None]
        cap_fractions,actual=actor.geometry(obs)
        expected_total=float(actual.item()>0)
        assert torch.isfinite(cap_fractions).all()
        if expected_total:
            assert float(cap_fractions.sum())>=1.
        for deterministic in (False,True):
            raw=actor.sample(obs,deterministic=deterministic)
            # Reproduce exact deployed-control and observation storage roundtrip.
            control=np.clip(2*raw.detach().numpy()[0]-1,-1,1)
            fractions=torch.as_tensor((control+1)/2,dtype=torch.float64)[None]
            logp,entropy=actor.known(obs,fractions)
            assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()
            assert abs(float(fractions.sum())-expected_total)<1e-9
            assert float((fractions-cap_fractions).max())<1e-10
            rows.append(dict(target=target,actual_target=float(actual.item()),deterministic=deterministic,
                             fraction_sum=float(fractions.sum()),cap_sum=float(cap_fractions.sum()),
                             logprob=float(logp.item()),entropy=float(entropy.item())))
    result=dict(status='passed',cases=rows,scope='PASPO public-geometry arithmetic only; native constraints/reward unchanged')
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    import sys
    run(sys.argv[1])
