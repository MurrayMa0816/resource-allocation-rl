# Adaptation of CC-BY-NC-4.0 Meta FAIR MR.Q; original license in vendor/LICENSE.
"""Public transition adapter; complete native MR.Q updates remain untouched."""
from __future__ import annotations
import copy
from contextlib import contextmanager
import dataclasses
import hashlib
import math

import numpy as np
import torch
from torch import nn

from .native_loader import load_native, replay_class, verify_vendor

PIN='073adcd6b30f84694c241e00bd684db50242a215'
PRIORITY_EXECUTION_PROFILE='mrq-priority-index-put-strict-v1'
native=load_native()


def _hash(value):
    out=hashlib.sha256()
    def add(x):
        if isinstance(x,torch.Tensor):
            a=x.detach().cpu().contiguous().numpy();out.update(str(a.dtype).encode());out.update(str(a.shape).encode());out.update(a.tobytes())
        elif isinstance(x,np.ndarray):
            out.update(str(x.dtype).encode());out.update(str(x.shape).encode());out.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):out.update(str(k).encode());add(x[k])
        elif isinstance(x,(list,tuple)):
            out.update(type(x).__name__.encode())
            for y in x:add(y)
        else:out.update(repr(x).encode())
    add(value);return out.hexdigest()


@contextmanager
def priority_write_determinism():
    """Choose Torch's deterministic indexed-write kernel, then restore flags.

    Only this operation is controlled. Global strict mode would reject native
    floating-point CUDA cumsum in Torch2.5.1. No aggregation/priority formula is
    substituted, and this is not a whole-stack CUDA reproducibility claim.
    """
    enabled=torch.are_deterministic_algorithms_enabled()
    warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
    record=dict(profile=PRIORITY_EXECUTION_PROFILE,enabled_before=enabled,warn_only_before=warn_only)
    try:
        torch.use_deterministic_algorithms(True,warn_only=False)
        record['strict_inside']=torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
        yield record
    finally:
        torch.use_deterministic_algorithms(enabled,warn_only=warn_only)
        record['enabled_after']=torch.are_deterministic_algorithms_enabled()
        record['warn_only_after']=torch.is_deterministic_algorithms_warn_only_enabled()
        record['fully_restored']=record['enabled_after']==enabled and record['warn_only_after']==warn_only


class ObservedReplay(native.buffer.ReplayBuffer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Author priority=torch.empty is read over the full capacity by
        # cumsum(priority*mask). Define never-written/mask-zero slots as zero:
        # NaN*0 is NaN. Native add still sets every valid transition's priority.
        # No RNG, hyperparameter, valid-slot priority or update rule is changed.
        if self.prioritized:self.priority.zero_()
        self.sample_calls=dict(encoder=0,value=0,other=0)
        self.priority_update_calls=0

    def sample(self,horizon,include_intermediate=False):
        batch=super().sample(horizon,include_intermediate)
        key='encoder' if horizon==5 and include_intermediate else 'value' if horizon==3 and not include_intermediate else 'other'
        self.sample_calls[key]+=1
        return batch

    def update_priority(self,priority):
        # Preserve the author's indexed assignment and priority values. Only
        # dispatch this call under the explicit deterministic Torch profile.
        with priority_write_determinism() as record:
            result=super().update_priority(priority)
        self.last_priority_guard=record;self.priority_update_calls+=1
        return result


class ObservedCore(native.Agent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.encoder_updates=0;self.value_updates=0;self.policy_updates=0
        self.target_syncs=0;self.sync_environment_steps=[];self.physical_step_for_observer=0
        self.last_Q=None;self.last_Q_target=None

    def train_encoder(self,*args,**kwargs):
        result=super().train_encoder(*args,**kwargs);self.encoder_updates+=1
        return result

    def train_rl(self,*args,**kwargs):
        result=super().train_rl(*args,**kwargs)
        self.value_updates+=1;self.policy_updates+=1
        self.last_Q=result[0].detach();self.last_Q_target=result[1].detach()
        return result

    def train(self):
        before=self.training_steps;result=super().train()
        if self.training_steps>before and before%self.target_update_freq==0:
            self.target_syncs+=1;self.sync_environment_steps.append(self.physical_step_for_observer)
        return result


class DeploymentActor(nn.Module):
    """Only the state encoder and policy needed by the native deployment path."""
    def __init__(self,observation_dim,action_dim,hp):
        super().__init__()
        self.state_encoder=native.models.BaseMLP(observation_dim,hp['zs_dim'],hp['enc_hdim'],hp['enc_activ'])
        self.policy=native.models.Policy(action_dim,False,hp['gumbel_tau'],hp['zs_dim'],hp['policy_hdim'],hp['policy_activ'])
        self.encoder_activation=getattr(torch.nn.functional,hp['enc_activ'])

    def forward(self,obs):
        zs=native.models.ln_activ(self.state_encoder(obs),self.encoder_activation)
        return self.policy.act(zs)


class Agent:
    def __init__(self,cfg,observation_dim,action_dim,device='cuda:0'):
        verify_vendor();values=dict(cfg)
        self.seed=int(values.pop('seed',0));self.total_steps=int(values.pop('total_steps',49152))
        reward_scale=values.pop('reward_scale',1.)
        if reward_scale!=1.:raise ValueError('external reward shaping/scaling is not in this profile')
        if 'gamma' in values:
            gamma=values.pop('gamma')
            if 'discount' in values and values['discount']!=gamma:raise ValueError('discount and gamma disagree')
            values['discount']=gamma
        defaults=dataclasses.asdict(native.Hyperparameters())
        hp=dataclasses.asdict(native.Hyperparameters(**values))
        if hp!=defaults:raise ValueError('this prepared profile preserves all author hyperparameters')
        if not (10000<self.total_steps<defaults['buffer_size']//2):raise ValueError('budget must cover native warmup without ring wrap')
        self.hp=hp;self.observation_dim=int(observation_dim);self.action_dim=int(action_dim)
        self.device=torch.device(device)
        if self.device.type=='cuda' and self.device.index is None:self.device=torch.device('cuda:0')
        self.steps=0;self.first_learning_step=None;self._inference_only=False
        self.rng=np.random.RandomState(self.seed)
        with replay_class(native,ObservedReplay):
            self.core=ObservedCore((self.observation_dim,),self.action_dim,1.,False,False,
                self.device,history=1,hp=hp)
        self.replay=self.core.replay_buffer
        self.actor=self.core.policy;self.critic=self.core.value;self.model=self.core.encoder
        self.encoder=self.core.encoder

    @property
    def n_updates(self):return self._loaded_updates if self._inference_only else self.core.training_steps

    @property
    def updates(self):return self.n_updates

    def _observation(self,obs):
        x=np.asarray(obs,dtype=np.float64)
        if x.shape!=(self.observation_dim,) or not np.isfinite(x).all():raise ValueError('invalid public observation')
        return x

    @torch.no_grad()
    def policy_action(self,obs,deterministic=True):
        obs=self._observation(obs)
        if self._inference_only:
            return self.actor(torch.as_tensor(obs,dtype=torch.float32,device=self.device).reshape(1,-1)).cpu().numpy().reshape(-1)
        return self.core.select_action(obs,use_exploration=False)

    def act(self,obs,deterministic=False):
        if self._inference_only or deterministic:return self.policy_action(obs,True)
        action=self.core.select_action(self._observation(obs),use_exploration=True)
        if action is None:return self.rng.uniform(-1.,1.,size=self.action_dim).astype(np.float32)
        return action

    def observe(self,obs,action,reward,next_obs,done):
        if self._inference_only:raise RuntimeError('inference endpoint cannot continue training')
        if self.steps>=self.total_steps:raise RuntimeError('registered physical budget exceeded')
        obs=self._observation(obs);nxt=self._observation(next_obs);a=np.asarray(action,dtype=np.float64)
        if a.shape!=(self.action_dim,) or not np.isfinite(a).all() or np.max(np.abs(a))>1.000001 or not math.isfinite(reward):raise ValueError('invalid transition')
        # The task H cutoff is a true terminal, not a continuing-MDP truncation.
        self.replay.add(obs,a,nxt,float(reward),bool(done),False)
        self.steps+=1;self.core.physical_step_for_observer=self.steps
        before=self.n_updates;expected=int(self.replay.size>self.core.buffer_size_before_training)
        self.core.train()
        assert self.n_updates-before==expected
        if self.n_updates and self.first_learning_step is None:self.first_learning_step=self.steps
        n=self.n_updates;freq=self.core.target_update_freq;bursts=(n+freq-1)//freq
        assert self.core.encoder_updates==bursts*freq
        assert self.core.policy_updates==self.core.value_updates==n
        assert self.core.target_syncs==bursts
        return dict(steps=self.steps,updates=n)

    def get_rng_state(self):
        return dict(uniform=copy.deepcopy(self.rng.get_state()),numpy=copy.deepcopy(np.random.get_state()),
            torch=torch.get_rng_state().clone(),cuda=[x.clone() for x in torch.cuda.get_rng_state_all()] if self.device.type=='cuda' else [])

    def set_rng_state(self,x):
        self.rng.set_state(x['uniform']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'])
        if x['cuda']:torch.cuda.set_rng_state_all(x['cuda'])

    def native_state_fingerprint(self):
        x=dict(steps=self.steps,updates=self.n_updates,inference_only=self._inference_only,rng=_hash(self.get_rng_state()))
        if self._inference_only:
            x['actor']=_hash(self.actor.state_dict());return x
        for name in ['encoder','encoder_target','policy','policy_target','value','value_target','encoder_optimizer','policy_optimizer','value_optimizer']:
            x[name]=_hash(getattr(self.core,name).state_dict())
        x.update(reward_scale=self.core.reward_scale,target_reward_scale=self.core.target_reward_scale,
            encoder_updates=self.core.encoder_updates,policy_updates=self.core.policy_updates,
            value_updates=self.core.value_updates,target_syncs=self.core.target_syncs,
            buffer_size=self.replay.size,buffer_index=self.replay.ind,episode_steps=self.replay.ep_timesteps,
            terminal_seen=self.replay.env_terminates,history_queue=list(self.replay.history_queue),
            max_priority=float(self.replay.max_priority),sample_calls=dict(self.replay.sample_calls),
            priority_update_calls=self.replay.priority_update_calls,
            sampled_indices=_hash(getattr(self.replay,'sampled_ind',None)))
        return x

    def diagnostics(self):
        if self._inference_only:return {**self._loaded_diagnostics,'inference_only':True}
        x=dict(train_steps=self.steps,steps=self.steps,updates=self.n_updates,
            rl_updates=self.core.value_updates,policy_optimizer_steps=self.core.policy_updates,
            encoder_optimizer_steps=self.core.encoder_updates,target_hard_syncs=self.core.target_syncs,
            target_sync_environment_steps=list(self.core.sync_environment_steps),
            first_learning_environment_step=self.first_learning_step,
            replay_slots_size=self.replay.size,replay_insert_index=self.replay.ind,
            replay_capacity=self.replay.max_size,observation_storage_device=str(self.replay.storage_device),
            priority_storage_device=str(self.replay.priority.device),replay_sample_calls=dict(self.replay.sample_calls),
            priority_updates=self.replay.priority_update_calls,replay_has_true_terminal=self.replay.env_terminates,
            priority_execution_profile=PRIORITY_EXECUTION_PROFILE,
            enc_horizon=self.core.enc_horizon,Q_horizon=self.core.Q_horizon,
            replay_size_warmup_threshold=self.core.buffer_size_before_training,
            reward_scale=self.core.reward_scale,target_reward_scale=self.core.target_reward_scale,
            external_reward_scale=1.,native_device=str(self.device),native_core_commit=PIN,
            deterministic_deployment=True,inference_only=False)
        if self.core.last_Q is not None:
            x.update(last_Q_mean=float(self.core.last_Q.mean().cpu()),last_Q_target_mean=float(self.core.last_Q_target.mean().cpu()))
        return x

    def save(self,path):
        if self._inference_only:raise RuntimeError('save the original registered inference endpoint')
        torch.save(dict(format='mrq-deployment-inference-v1',source_commit=PIN,
            observation_dim=self.observation_dim,action_dim=self.action_dim,hp=self.hp,
            seed=self.seed,total_steps=self.total_steps,steps=self.steps,updates=self.n_updates,
            state_encoder_state={k:v.detach().cpu() for k,v in self.core.encoder.zs_mlp.state_dict().items()},
            policy_state={k:v.detach().cpu() for k,v in self.core.policy.state_dict().items()},
            diagnostics=self.diagnostics(),optimizer_value_model_or_replay_included=False),path)

    @classmethod
    def load(cls,path,device='cuda:0'):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data['format']!='mrq-deployment-inference-v1' or data['source_commit']!=PIN:raise ValueError('inference identity mismatch')
        verify_vendor();self=cls.__new__(cls);self.device=torch.device(device)
        self.observation_dim=int(data['observation_dim']);self.action_dim=int(data['action_dim'])
        self.hp=data['hp'];self.seed=int(data['seed']);self.total_steps=int(data['total_steps'])
        self.steps=int(data['steps']);self._loaded_updates=int(data['updates']);self._inference_only=True
        self._loaded_diagnostics=data['diagnostics'];self.rng=np.random.RandomState(self.seed)
        with torch.random.fork_rng(devices=[]):
            self.actor=DeploymentActor(self.observation_dim,self.action_dim,self.hp).to(self.device)
        self.actor.state_encoder.load_state_dict(data['state_encoder_state'])
        self.actor.policy.load_state_dict(data['policy_state']);self.actor.eval()
        return self
