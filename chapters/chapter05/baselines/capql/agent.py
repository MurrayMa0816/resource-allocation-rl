"""Current full-spend candidate API; never selects checkpoints or launches jobs."""
from __future__ import annotations
import copy,json
from pathlib import Path
import numpy as np
import torch
from .core import CAPQLAgent
HERE=Path(__file__).resolve().parent

def load_config(domain,profile='fixed'):
    document=json.loads((HERE/'configs.json').read_text())
    if domain not in ('radio','v5') or profile not in document['profiles']:
        raise ValueError('Unknown domain or candidate profile')
    return copy.deepcopy(document['profiles'][profile][domain])

def _cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,dict):return {k:_cpu(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(_cpu(v)for v in value)
    return copy.deepcopy(value)

def _normalization_class():
    # Both normal package imports and a standalone baselines PYTHONPATH work.
    try:
        from .._support.normalization import NormalizedOffPolicy
    except (ImportError,ValueError):
        from _support.normalization import NormalizedOffPolicy
    return NormalizedOffPolicy

class Agent:
    on_policy=False
    action_mode='tanh_latent'
    manages_observation_normalization=True
    resample_preference_each_step=True
    name='capql'

    def __init__(self,obs_dim,action_dim,seed=0,device='cpu',config=None):
        self.config=dict(config or {})
        self.device=torch.device(device)
        self.obs_dim,self.action_dim=int(obs_dim),int(action_dim)
        context=torch.cuda.device(self.device)if self.device.type=='cuda'else __import__('contextlib').nullcontext()
        with context:
            self.core=CAPQLAgent(obs_dim,action_dim,seed,self.device,self.config)
        self.native=(_normalization_class()(self.core,obs_dim,self.config)
                     if self.config.get('observation_normalization',False)else self.core)

    def __getattr__(self,name):
        return getattr(self.native,name)

    @property
    def actor(self):return self.core.native.policy

    @property
    def critic(self):return self.core.native.critic

    def sample_weights(self,count,rng):return self.core.sample_weights(count,rng)

    def observe_training(self,observations):
        if self.native is not self.core:self.native.observe_training(observations)

    def act(self,obs,preference,deterministic=False):
        obs=np.asarray(obs);preference=np.asarray(preference)
        if obs.ndim==1:obs=obs[None]
        if preference.ndim==1:preference=np.broadcast_to(preference,(len(obs),2)).copy()
        if obs.shape!=(len(obs),self.obs_dim)or preference.shape!=(len(obs),2):raise ValueError('Wrong flat observation or preference shape')
        result=self.native.act(obs,preference,deterministic)
        if np.asarray(result).shape!=(len(obs),self.action_dim)or not np.isfinite(result).all():raise FloatingPointError('Invalid complete latent action')
        return result

    def update(self,batch):
        return self.native.update(batch)

    def state_dict(self):
        # One inference endpoint; no optimizer/replay backup or resume claim.
        return _cpu(self.native.state_dict())

    def load_state_dict(self,payload):self.native.load_state_dict(payload)
    def endpoint(self):return self.state_dict()

    def rng_state_dict(self):
        numpy_state={};torch_state={}
        for name,value in vars(self.core).items():
            if isinstance(value,np.random.Generator):numpy_state[name]=copy.deepcopy(value.bit_generator.state)
            elif isinstance(value,torch.Generator):torch_state[name]=value.get_state().clone()
        return dict(numpy=numpy_state,torch=torch_state)

    def load_rng_state_dict(self,state):
        for name,value in state['numpy'].items():getattr(self.core,name).bit_generator.state=copy.deepcopy(value)
        for name,value in state['torch'].items():getattr(self.core,name).set_state(value)

    def metadata(self):
        result=dict(self.native.metadata())
        result.update(candidate_status='revalidation_not_formal_roster',
                      profile=self.config.get('profile','unspecified'),
                      current_environment_required='chapter5_full_spend_20261104.env.Env',
                      owns_training_observation_normalizer=True,
                      environment_reward_or_task_changed=False,
                      checkpoint_selection=False,
                      legacy_policy_recovery=False,
                      implementation_directory=str(HERE))
        return result
