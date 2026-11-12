"""Inference-only native physical-action expectation; no PPO sampling changes.

A fixed finite Gaussian quadrature is applied at each unique public state.
Every point is decoded through native quota and projection; the physical action
is their arithmetic mean. This is a distinct deterministic closed-loop policy,
not an exact expectation of the stochastic policy's nonlinear return.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from allocation4.methods.lax import gaussian_parameters

M64_OPERATOR='fixed64_antithetic_mean_feasible_actions'
M64_SEED=2026110964

def fixed_antithetic_points(event_dim):
    if int(event_dim)!=event_dim or event_dim<1:raise ValueError('Positive event dimension required')
    generator=torch.Generator(device='cpu').manual_seed(M64_SEED)
    positive=torch.randn((32,int(event_dim)),generator=generator,dtype=torch.float64)
    return torch.cat((positive,-positive),dim=0).numpy()

class ActionExpectation:
 def __init__(self,actor,eps):
  self.actor=actor;self.eps=torch.as_tensor(eps,dtype=torch.float64,device='cpu');self.parameter_evaluations=0;self.feasible_vectors=0;self.batch_calls=0;self.max_mass_error=0.
 def parameters(self):return self.actor.parameters()
 def sample(self,states,generator=None,deterministic=True):
  if generator is not None or deterministic is not True:raise ValueError('This is a deterministic inference-only conversion')
  with torch.no_grad():
   data=self.actor._prepare(states);mean,sigma=gaussian_parameters(self.actor,data)
   self.parameter_evaluations+=len(mean);self.batch_calls+=1;actions=[];P=len(self.eps)
   for j in range(len(mean)):
    repeated={k:v[j:j+1].expand(P,*v.shape[1:]) for k,v in data.items()}
    raw=mean[j:j+1]+sigma[j:j+1]*self.eps
    eta,z=raw[:,:1],raw[:,1:]
    quotas=self.actor.quota(repeated,eta);candidates=self.actor.decode(repeated,z,quotas)
    self.feasible_vectors+=P
    caps=data['caps'][j].detach().cpu().numpy();budget=float(data['budget'][j])
    if candidates.shape!=(P,89) or np.any(candidates<0) or np.any(candidates>caps[None,:]+1e-7) or not np.isfinite(candidates).all():raise ValueError('Illegal native candidate action')
    error=max(abs(math.fsum(row.tolist())-float(q)) for row,q in zip(candidates,quotas))
    self.max_mass_error=max(self.max_mass_error,error)
    actual=candidates.mean(0,dtype=np.float64)
    q=math.fsum(actual.tolist());target=float(quotas.mean())
    if abs(q-target)>1e-7 or q>budget+1e-7 or np.any(actual<0) or np.any(actual>caps+1e-7):raise ArithmeticError('Mean physical action infeasible')
    actions.append(actual)
   return dict(actions=np.stack(actions))

def make_execution_actor(actor, operator='latent_mean'):
    """Return an inference actor and immutable execution contract description."""
    if operator=='latent_mean':
        return actor,dict(operator='latent_mean',training=False)
    if operator!=M64_OPERATOR:raise ValueError('Unknown declared execution operator')
    if next(actor.parameters()).device.type!='cpu':
        raise ValueError('Application execution uses the native CPU sampling mirror')
    points=fixed_antithetic_points(actor.event_dim)
    description=dict(operator=M64_OPERATOR,training=False,point_count=64,
        positive_points=32,antithetic=True,seed=M64_SEED,
        event_dim=int(actor.event_dim),point_generation='torch.randn CPU float64 using isolated generator; positive thennegative',
        points_array_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        averaging='native feasible physical action vectors before one physical transition',
        meaning='Finite-point conditional physical-action mean; not original latentmean or exact stochastic expectedreturn')
    return ActionExpectation(actor,points),description
