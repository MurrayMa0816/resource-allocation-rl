"""Pinned authors' CAPQL class and networks, with only batched I/O plumbing."""
from __future__ import annotations
import ast
import copy
from functools import lru_cache
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.optim import Adam
from gymnasium.spaces import Box
from .support import HERE, cpu_state, finite_stats, load_definitions, sha

UPSTREAM=HERE/'vendor'


@lru_cache(maxsize=1)
def native_classes():
    ns=dict(torch=torch,nn=nn,F=F,Normal=Normal,copy=copy,Adam=Adam)
    load_definitions(UPSTREAM/'models/networks.py',ns)
    load_definitions(UPSTREAM/'utils/utils.py',ns,{'update_Q','Weight_Sampler_pos','Weight_Sampler_angle'})
    load_definitions(UPSTREAM/'models/capql.py',ns,{'CAPQL'})
    return ns


@lru_cache(maxsize=1)
def actor_backward_update():
    # The actor loss also traverses Q, but the Q parameter gradients it used
    # to accumulate are discarded by the next critic zero_grad(). Restrict
    # this backward to the actor leaves, retaining the exact action gradient.
    # Vendor remains untouched and all Bellman/Adam/target calculations stay
    # in the pinned author function.
    ns=dict(native_classes())
    tree=ast.parse((UPSTREAM/'models/capql.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CAPQL')
    update=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update_parameters')
    changed=0
    for n in ast.walk(update):
        if (isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                and n.func.attr=='backward' and isinstance(n.func.value,ast.Name)
                and n.func.value.id=='policy_loss'):
            if n.args or n.keywords:raise RuntimeError('Unexpected author actor backward signature')
            n.keywords=[ast.keyword(arg='inputs',value=ast.Attribute(
                value=ast.Name(id='self',ctx=ast.Load()),attr='_policy_backward_inputs',ctx=ast.Load()))]
            changed+=1
    if changed!=1:raise RuntimeError('Pinned CAPQL actor backward no longer matches')
    module=ast.fix_missing_locations(ast.Module(body=[update],type_ignores=[]))
    exec(compile(module,str(UPSTREAM/'models/capql.py'),'exec'),ns)
    return ns['update_parameters']


class Agent:
    on_policy=False
    action_mode='tanh_latent'
    resample_preference_each_step=True
    def __init__(self,obs_dim,action_dim,seed=0,device='cpu',config=None):
        self.config=dict(config or {});self.device=torch.device(device)
        torch.manual_seed(seed);self.seed=seed;self.updates=0
        cfg=NS(training=NS(gamma=self.config.get('gamma',1.),tau=self.config.get('tau',.005),
            alpha=self.config.get('alpha',.2),target_update_interval=1,lr=self.config.get('lr',3e-4)),
            model=NS(type='CAPQL',hidden_size=self.config.get('hidden_dim',256)),
            gpu=NS(cuda=self.device.type=='cuda'))
        self.native=native_classes()['CAPQL'](obs_dim,Box(-1.,1.,(action_dim,),dtype=np.float32),2,cfg)
        self.native.device=self.device
        for name in ('policy','critic','critic_target'):getattr(self.native,name).to(self.device)
        self.native._policy_backward_inputs=tuple(self.native.policy.parameters())
        self.native.update_parameters=MethodType(actor_backward_update(),self.native)
        self.sampler=native_classes()['Weight_Sampler_pos'](2)

    def sample_weights(self,count,rng):
        # Official positive spherical sampler: abs Gaussian / L1, not an
        # arbitrary uniform lambda grid. Sampling uses caller's recorded RNG.
        x=np.abs(rng.normal(size=(count,2)));return (x/np.maximum(x.sum(1,keepdims=True),1e-12)).astype(np.float32)

    @torch.no_grad()
    def act(self,obs,prefs,deterministic=False):
        s=torch.as_tensor(obs,dtype=torch.float32,device=self.device)
        w=torch.as_tensor(prefs,dtype=torch.float32,device=self.device)
        mean,log_std=self.native.policy.forward(s,w)
        # Keep the original draw even for deterministic actions, so evaluation
        # and training retain exactly the original RNG stream. No log-density
        # is needed on this inference-only path.
        draw=Normal(mean,log_std.exp()).rsample()
        raw=mean if deterministic else draw
        return (torch.tanh(raw)*self.native.policy.action_scale+self.native.policy.action_bias).cpu().numpy()

    def update(self,batch):
        class FixedBatch:
            def sample(self,batch_size):
                return (batch['obs'],batch['actions'],batch['weights'],batch['rewards'],
                        batch['next_obs'],1.-np.asarray(batch['dones']).reshape(-1))
        losses=self.native.update_parameters(FixedBatch(),len(batch['obs']),self.updates)
        self.updates+=1
        return finite_stats(dict(critic1_loss=losses[0],critic2_loss=losses[1],actor_loss=losses[2],
            updates=self.updates,official_update_calls=self.updates,vector_objectives=2))

    def state_dict(self):
        return cpu_state(dict(policy=self.native.policy.state_dict(),critic=self.native.critic.state_dict(),
            critic_target=self.native.critic_target.state_dict(),updates=self.updates))

    def load_state_dict(self,value):
        for key in ('policy','critic','critic_target'):getattr(self.native,key).load_state_dict(value[key])
        self.updates=value['updates']

    def metadata(self):
        return dict(method='CAPQL',venue='ICLR2023',implementation='authors update definitions with exact actor-leaf-only backward; inference omits unused log-density',
            source_commit='0ef02f3d3b3d0c512d282cd5ec0f00475349beee',
            core_sha256=sha(UPSTREAM/'models/capql.py'),networks_sha256=sha(UPSTREAM/'models/networks.py'),
            gamma=self.native.gamma,alpha=self.native.alpha,core='vector twin Q, entropy-concave Bellman and preference-conditioned SAC',
            finite_horizon_adaptation='gamma1; true terminal bootstrap mask0',
            modifications=['batched policy forward','external RAM replay sampling','no tensorboard/MuJoCo imports',
                'actor-only backward inputs; unused critic parameter gradient buffers skipped',
                'inference skips unused log-density while preserving original Gaussian draws',
                'cached private compiled author definitions'],
            license='author repository has no explicit root license; private research snapshot, not relicensed')
