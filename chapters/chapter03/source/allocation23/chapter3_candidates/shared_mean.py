"""Generic object-sharing mean network; full N+1 action support, no response basis."""
import torch
from torch import nn

def network(d,h,o):
 return nn.Sequential(nn.Linear(d,h),nn.Tanh(),nn.Linear(h,o))

class SharedObjectMean(nn.Module):
 def __init__(self,observation_dim,action_dim,condition_dim=0):
  super().__init__();self.objects=action_dim-1;self.condition_dim=condition_dim
  if observation_dim==self.objects*3+3:self.local_dim,self.global_dim=3,3
  elif observation_dim==self.objects*2+2:self.local_dim,self.global_dim=2,2
  else:raise ValueError('Public observation layout does not match object rows')
  self.local=network(self.local_dim,64,32)
  context_dim=32+self.global_dim+condition_dim
  self.object_head=network(32+context_dim,32,1)
  self.amount_head=network(context_dim,32,1)
 def forward(self,obs,condition=None):
  split=self.objects*self.local_dim
  local=obs[...,:split].reshape(*obs.shape[:-1],self.objects,self.local_dim)
  embedded=self.local(local)
  pieces=[embedded.mean(-2),obs[...,split:]]
  if self.condition_dim:
   if condition is None or condition.shape[-1]!=self.condition_dim:raise ValueError('Condition input required')
   pieces.append(condition)
  context=torch.cat(pieces,-1)
  expanded=context.unsqueeze(-2).expand(*embedded.shape[:-1],context.shape[-1])
  objects=self.object_head(torch.cat([embedded,expanded],-1)).squeeze(-1)
  return torch.cat([self.amount_head(context),objects],-1)
