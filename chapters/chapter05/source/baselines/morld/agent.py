"""MORL/D-SB-PSA using the authors' continuous MOSAC update and networks.

MORL-Baselines code is MIT licensed; the unmodified license and source remain
in vendor/morld at the pinned commit. This adapter replaces experiment I/O,
not the native SAC equations. Task adaptations are stated in metadata().
"""
from __future__ import annotations
import ast
import copy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn, optim
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
VENDOR=ROOT/'vendor'
COMMIT='ee60a9bd16a6a77e4d0a28239925ad349ba9bb38'
SELECTED_PROFILE='critic_lr1e4'
VERSION='morld-native-mosac-sb-psa-selected-critic1e4-v1'
INFERENCE_VERSION='morld-selected-critic1e4-three-actors-inference-v1'


def selected_config(config=None):
    """The sole current recipe; retired experiment settings cannot be resumed here."""
    requested=dict(config or {})
    domain=requested.get('domain','radio')
    if domain not in ('radio','v5'):
        raise ValueError('Unknown selected MORL/D task')
    current=json.loads((ROOT/'configs.json').read_text())[domain][SELECTED_PROFILE]
    locked=('method','profile','domain','actor_lr','critic_lr','hidden_dim','batch_size',
            'start_steps','learning_starts','updates_per_step','gamma','alpha','autotune',
            'tau','policy_freq','target_net_freq','morld_population','morld_update_passes',
            'morld_other_update_every_training_episodes','morld_psa_value_source',
            'morld_psa_every','morld_psa_internal_cases','internal_cases',
            'legacy_constructor_cuda_reseed','observation_normalization','horizon','n_step')
    for key in locked:
        if key in requested and (requested[key]!=current[key] or
                isinstance(requested[key],bool)!=isinstance(current[key],bool)):
            raise ValueError('Only the selected MORL/D configuration is active: '+key)
    aliases={'q_lr':current['critic_lr'],'policy_lr':current['actor_lr'],
             'other_update_every_training_episodes':1,
             'net_arch':[current['hidden_dim']]*2,
             'morld_psa_delta':.1,'morld_return_window':8,
             'morld_weights':[[0.,1.],[.8,.2],[1.,0.]]}
    for key,value in aliases.items():
        if key in requested and requested[key]!=value:
            raise ValueError('Conflicting selected MORL/D setting: '+key)
    return {**current,**requested}


def _selected_state_metadata(state):
    metadata=state.get('metadata',{})
    if (metadata.get('selected_profile')!=SELECTED_PROFILE or
            metadata.get('actor_lr')!=3e-4 or metadata.get('critic_lr')!=1e-4 or
            metadata.get('other_update_every_training_episodes')!=1 or
            metadata.get('extra_update_passes_for_other_members')!=10):
        raise ValueError('This endpoint is not the selected MORL/D implementation')


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _cpu(value):
    if th.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_cpu(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(_cpu(v)for v in value)
    return copy.deepcopy(value)


def _native_stats(values):
    stats={}
    for key in ('qf_loss','qf1_loss','qf2_loss','actor_loss','alpha_loss'):
        value=values.get(key)
        stats[key]=None if value is None else float(value.detach())
    policy=values['self']
    stats.update(alpha=float(policy.alpha),actor_update_events=policy.policy_freq if policy.global_step%policy.policy_freq==0 else 0,
        target_update_event=int(policy.global_step%policy.target_net_freq==0))
    done=values['mb_dones'].flatten()==1
    stats['terminal_target_error']=float((values['next_q_value'][done]-values['scalarized_rewards'].flatten()[done]).abs().max())if bool(done.any())else 0.
    if any(not math.isfinite(v)for v in stats.values()if v is not None):raise FloatingPointError('Nonfinite native MOSAC statistics')
    return stats


class _PolicyBase:
    """Only the authors' device/id base plumbing; no numerical learner code."""
    def __init__(self,id=None,device='auto'):
        self.id=id;self.device=th.device('cuda'if th.cuda.is_available()else'cpu')if device=='auto'else th.device(device)
        self.global_step=0


class _SharedBatch:
    """Expose the worker's uniformly sampled shared RAM batch to native MOSAC.

    Each call comes from the external shared replay. No replay/optimizer is
    created on disk and no population-sized replay copies are allocated.
    """
    def __init__(self,*args,**kwargs):self.current=None
    def set(self,batch):self.current=batch
    def sample(self,batch_size,to_tensor=False,device=None,**kwargs):
        if self.current is None:raise RuntimeError('Native update requires a real externally sampled shared replay batch')
        value=self.current
        if len(value['obs'])!=batch_size:raise ValueError('Actual native batch size differs from configured batch')
        keys=('obs','actions','rewards','next_obs','dones')
        arrays=[np.asarray(value[k],np.float32)for k in keys]+[np.arange(batch_size)]
        return tuple(th.as_tensor(x,device=device)for x in arrays)if to_tensor else tuple(arrays)
    def __len__(self):return 0 if self.current is None else len(self.current['obs'])


def _compile(path,namespace,names,*,return_statistics=False):
    nodes=[]
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef))and node.name in names:
            if return_statistics and isinstance(node,ast.ClassDef)and node.name=='MOSAC':
                node=copy.deepcopy(node)
                update=next(m for m in node.body if isinstance(m,ast.FunctionDef)and m.name=='update')
                update.body.append(ast.Return(value=ast.Call(func=ast.Name(id='_native_stats',ctx=ast.Load()),
                    args=[ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[])],keywords=[])))
            nodes.append(node)
        elif isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id in names for t in node.targets):nodes.append(node)
    code=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    exec(compile(code,str(path),'exec'),namespace)


@lru_cache(maxsize=1)
def native_definitions():
    provenance=json.loads((ROOT/'provenance.json').read_text())
    if provenance['commit']!=COMMIT:raise RuntimeError('MORL/D vendor commit differs from reviewed source')
    files=('morl_baselines/common/networks.py','morl_baselines/common/pareto.py',
           'morl_baselines/single_policy/ser/mosac_continuous_action.py','morl_baselines/multi_policy/morld/morld.py')
    for name in files:
        if _sha(VENDOR/name)!=provenance['files_sha256'][name]:raise RuntimeError('Native MORL/D source changed: '+name)
    ns=dict(th=th,nn=nn,F=F,np=np,optim=optim,gym=gym,math=math,time=__import__('time'),os=os,
        Optional=Optional,Tuple=Tuple,Union=Union,List=List,Dict=Dict,Type=Type,Iterable=Iterable,Callable=Callable,
        deepcopy=copy.deepcopy,override=lambda fn:fn,MOPolicy=_PolicyBase,MOAgent=object,
        ReplayBuffer=_SharedBatch,_native_stats=_native_stats)
    _compile(VENDOR/files[0],ns,{'mlp','polyak_update','layer_init'})
    _compile(VENDOR/files[1],ns,{'get_non_pareto_dominated_inds','filter_pareto_dominated'})
    _compile(VENDOR/files[2],ns,{'MOSoftQNetwork','MOSACActor','MOSAC','LOG_STD_MIN','LOG_STD_MAX'},return_statistics=True)
    _compile(VENDOR/files[3],ns,{'Policy','MORLD'})
    ns['source_sha256']={str(VENDOR/name):_sha(VENDOR/name)for name in files}
    return ns


class _CompactArchive:
    """Native Pareto pruning with actor-only immutable snapshots, not replay copies."""
    def __init__(self,filter_function):
        self.filter_function=filter_function;self.individuals=[];self.evaluations=[];self.insertions=0
    def add(self,policy,evaluation):
        value=np.asarray(evaluation,np.float64)
        if value.shape!=(2,)or not np.isfinite(value).all():raise ValueError('Finite two-objective archive value required')
        snapshot=dict(policy_id=policy.id,weights=policy.weights.copy(),actor=_cpu(policy.wrapped.actor.state_dict()))
        values=self.evaluations+[value.copy()];individuals=self.individuals+[snapshot]
        allowed={tuple(v)for v in self.filter_function(values)}
        kept_values=[];kept_individuals=[];seen=set()
        for v,item in zip(values,individuals):
            key=tuple(v)
            if key in allowed and key not in seen:
                kept_values.append(v);kept_individuals.append(item);seen.add(key)
        self.evaluations=kept_values;self.individuals=kept_individuals;self.insertions+=1


class Agent:
    on_policy=False
    action_mode='tanh_latent'
    resample_preference_each_step=False
    name='morld'

    def __init__(self,obs_dim,action_dim,seed=0,device='cpu',config=None):
        self.config=selected_config(config);self.obs_dim=int(obs_dim);self.action_dim=int(action_dim)
        self.seed=int(seed);self.device=th.device(device);native=native_definitions()
        self.population_size=int(self.config.get('morld_population',3))
        if self.population_size!=3:raise ValueError('This reviewed task instantiation fixes three genuine population members')
        self.horizon=int(self.config.get('horizon',32 if self.config.get('domain','radio')=='radio'else 24))
        self.initial_weights=np.asarray(self.config.get('morld_weights',[[0.,1.],[.8,.2],[1.,0.]]),np.float64)
        if self.initial_weights.shape!=(3,2)or np.any(self.initial_weights<0)or not np.allclose(self.initial_weights.sum(1),1.):
            raise ValueError('Three valid two-objective weight vectors required')
        self.deployment_weights=np.asarray([.8,.2],np.float64)
        self.deployment_member=int(np.argmin(((self.initial_weights-self.deployment_weights)**2).sum(1)))
        if not np.allclose(self.initial_weights[self.deployment_member],self.deployment_weights):
            raise ValueError('Population must include the predeclared .8 deployment member')
        self.shared_buffer=_SharedBatch();self.population=[]
        spaces=SimpleNamespace(observation_space=gym.spaces.Box(-np.inf,np.inf,(self.obs_dim,),np.float32),
            action_space=gym.spaces.Box(-1.,1.,(self.action_dim,),np.float32),reward_space=gym.spaces.Box(-np.inf,np.inf,(2,),np.float32),
            spec=SimpleNamespace(id='chapter5-full-spend-'+self.config.get('domain','radio')))
        spaces.unwrapped=spaces
        kwargs=dict(buffer_size=1,gamma=float(self.config.get('gamma',1.)),tau=float(self.config.get('tau',.005)),
            batch_size=int(self.config.get('batch_size',256)),learning_starts=int(self.config.get('learning_starts',2048)),
            net_arch=list(self.config.get('net_arch',[int(self.config.get('hidden_dim',256))]*2)),
            policy_lr=float(self.config.get('policy_lr',self.config.get('actor_lr',3e-4))),
            q_lr=float(self.config.get('q_lr',self.config.get('critic_lr',1e-4))),
            policy_freq=int(self.config.get('policy_freq',2)),target_net_freq=int(self.config.get('target_net_freq',1)),
            alpha=float(self.config.get('alpha',.005)),autotune=bool(self.config.get('autotune',False)),
            scalarization=th.matmul,device=self.device,log=False,seed=self.seed)
        if kwargs['gamma']!=1.:raise ValueError('This complete-terminal benchmark fixes gamma1')
        self.batch_size=kwargs['batch_size']
        self.learning_starts=kwargs['learning_starts']
        self.cross_update_passes=10
        with th.random.fork_rng(devices=[]):
            # Networks are initialized on CPU before .to(device). Seed only
            # that generator: torch.manual_seed also reseeds every CUDA stream
            # despite fork_rng(devices=[]), leaking across environment/DEV setup.
            th.random.default_generator.manual_seed(self.seed)
            for i,weights in enumerate(self.initial_weights):
                policy=native['MOSAC'](env=spaces,weights=weights.copy(),id=i,**kwargs)
                policy.set_buffer(self.shared_buffer)
                self.population.append(native['Policy'](i,weights.copy(),policy))
        self.native=object.__new__(native['MORLD'])
        self.native.population=self.population;self.native.pop_size=3;self.native.current_policy=0;self.native.iteration=0
        self.native.weights=self.initial_weights.copy();self.native.weight_adaptation_method='PSA'
        self.native.delta=float(self.config.get('morld_psa_delta',.1))
        self.native.archive=_CompactArchive(native['filter_pareto_dominated']);self.native.transfer=False
        self.native.sharing_mechanism=['shared_buffer'];self.native.shared_buffer=True
        self.actor=nn.ModuleList([p.wrapped.actor for p in self.population])
        self.critic=nn.ModuleList([q for p in self.population for q in (p.wrapped.qf1,p.wrapped.qf2)])
        self.critic_target=nn.ModuleList([q for p in self.population for q in (p.wrapped.qf1_target,p.wrapped.qf2_target)])
        # Existing runtime diagnostics can inspect actual optimizer tensors;
        # iter_optimizers() additionally exposes all population members.
        self.actor_optimizer=self.population[0].wrapped.actor_optimizer
        self.critic_optimizer=self.population[0].wrapped.q_optimizer
        self.collect_member=0;self.last_selected_episode=None;self.physical_step=0
        self.updates=0;self.member_update_counts=[0,0,0];self.member_actor_updates=[0,0,0]
        self.selection_changes=0;self.psa_calls=0;self.last_psa_step=0
        self.psa_every=int(self.config.get('morld_psa_every',3072))
        self.return_window=int(self.config.get('morld_return_window',8));self.training_returns=[[],[],[]]
        self.return_counts=[0,0,0]
        self.value_source='charged_internal'
        self.cross_updates=0;self.exchange_events=0;self.last_exchange_step=None
        if self.value_source not in ('training_returns','charged_internal')or self.psa_every<1 or self.return_window<1:
            raise ValueError('Invalid PSA value source/period/window')

    def iter_optimizers(self):
        for p in self.population:
            for key in ('actor_optimizer','q_optimizer','a_optimizer'):
                value=getattr(p.wrapped,key,None)
                if value is not None:yield value

    @property
    def alpha(self):return self.population[self.collect_member].wrapped.alpha

    def select_policy(self,step):
        if isinstance(step,bool)or int(step)!=step or step<0:raise ValueError('Nonnegative actual training step required')
        episode=int(step)//self.horizon
        if episode!=self.last_selected_episode:
            self.native.current_policy=episode%3
            candidate=self.native._MORLD__select_candidate()
            self.collect_member=candidate.id;self.last_selected_episode=episode;self.selection_changes+=1
        self.physical_step=int(step)
        return self.collect_member

    def set_physical_step(self,step):
        if isinstance(step,bool)or int(step)!=step or step<0:raise ValueError('Nonnegative actual training step required')
        self.physical_step=int(step)

    def sample_weights(self,count,rng=None):return np.repeat(self.population[self.collect_member].weights[None],int(count),axis=0).astype(np.float32)

    @th.no_grad()
    def act_policy(self,policy_id,obs,deterministic=False):
        if int(policy_id)not in range(3):raise ValueError('Unknown population policy')
        x=th.as_tensor(obs,dtype=th.float32,device=self.device)
        if x.ndim==1:x=x[None]
        if x.shape[1]!=self.obs_dim:raise ValueError('Public observation dimension changed')
        model=self.population[int(policy_id)].wrapped.actor
        if deterministic:
            mean,unused=model(x);action=mean.tanh()*model.action_scale+model.action_bias
        else:action=model.get_action(x)[0]
        if not bool(th.isfinite(action).all()):raise FloatingPointError('Nonfinite native action')
        return action.cpu().numpy()

    def act(self,obs,prefs,deterministic=False):
        if deterministic:
            weights=np.asarray(prefs)
            if not np.allclose(weights,self.deployment_weights):
                raise ValueError('This experiment declares only fixed .8-member deployment; no hidden archive/DEV preference selection')
        return self.act_policy(self.deployment_member if deterministic else self.collect_member,obs,deterministic)

    def update_policy(self,policy_id,batch):
        policy=self.population[int(policy_id)].wrapped
        self.shared_buffer.set(batch);policy.global_step=int(self.physical_step)
        stats=policy.update();self.member_update_counts[int(policy_id)]+=1
        self.member_actor_updates[int(policy_id)]+=stats['actor_update_events']
        return stats

    def update(self,batch):
        stats=self.update_policy(self.collect_member,batch);self.updates+=1
        return dict(critic_loss=stats['qf_loss'],actor_loss=stats['actor_loss'],alpha=stats['alpha'],
            alpha_loss=stats['alpha_loss'],updates=self.updates,actual_policy_id=self.collect_member,
            native_critic_updates=sum(self.member_update_counts),native_actor_updates=sum(self.member_actor_updates),
            native_actor_updates_this_call=stats['actor_update_events'],terminal_target_error=stats['terminal_target_error'],
            population_size=3,shared_buffer=True,psa_calls=self.psa_calls,archive_policies=len(self.native.archive.evaluations),
            native_core='authors MOSAC.update; return-only diagnostic instrumentation',method='MORL/D-SB-PSA')

    def after_exchange(self,sample_batch,*,replay_ready=True):
        """Call once after a complete *training* episode, never DEV/internal PSA.

        The historical short-budget adapter exchanges members every32/24 real
        training steps (one complete episode), unlike the paper's50000-step
        exchange. Keeping that declared schedule, the selected implementation executes
        the author's10 passes, each updating every *other* member on a fresh
        uniformly sampled batch from the same shared RAM replay. No physical
        interaction or reward relabelling occurs here. ``sample_batch`` is a
        zero-argument callback owned by the runner; it retains the same n-step
        convention as ordinary updates. No alternative update cadence is exposed.
        """
        if self.physical_step%self.horizon:
            raise ValueError('Exchange hook requires a complete episode boundary')
        if self.last_exchange_step is not None and self.physical_step<=self.last_exchange_step:
            raise ValueError('Exchange hook duplicated or physical clock moved backwards')
        self.last_exchange_step=self.physical_step;self.exchange_events+=1
        regular_ready=bool(replay_ready)and self.physical_step>=self.learning_starts
        result=dict(exchange_event=self.exchange_events,physical_step=self.physical_step,
            trained_member=self.collect_member,update_passes=self.cross_update_passes,
            extra_native_critic_updates_this_exchange=0,extra_native_actor_updates_this_exchange=0,
            extra_native_critic_updates_total=self.cross_updates,
            replay_ready=regular_ready,physical_interaction_steps=0)
        if not regular_ready:return result
        if not callable(sample_batch):raise TypeError('Ready replay requires a sample_batch callback')
        actor_updates=0;max_terminal_error=0.
        for unused in range(self.cross_update_passes):
            for policy in self.population:
                if policy.id==self.collect_member:continue
                stats=self.update_policy(policy.id,sample_batch())
                self.cross_updates+=1;actor_updates+=stats['actor_update_events']
                max_terminal_error=max(max_terminal_error,stats['terminal_target_error'])
        result.update(extra_native_critic_updates_this_exchange=self.cross_update_passes*(self.population_size-1),
            extra_native_actor_updates_this_exchange=actor_updates,
            extra_native_critic_updates_total=self.cross_updates,terminal_target_error=max_terminal_error,
            member_update_counts=self.member_update_counts.copy())
        return result

    def observe_episode(self,return_vector,policy_id=None,physical_step=None):
        index=self.collect_member if policy_id is None else int(policy_id)
        values=np.asarray(return_vector,np.float64)
        if values.shape==(2,):values=values[None,:]
        if index not in range(3)or values.ndim!=2 or values.shape[1]!=2 or not len(values)or not np.isfinite(values).all():raise ValueError('Actual terminal C/J vector(s) required')
        for value in values:self.training_returns[index].append(value.copy())
        self.training_returns[index]=self.training_returns[index][-self.return_window:]
        self.return_counts[index]+=len(values)
        if physical_step is not None:self.set_physical_step(physical_step)

    def needs_internal_evaluation(self,learning):
        return self.value_source=='charged_internal'and learning-self.last_psa_step>=self.psa_every

    def accept_internal_evaluation(self,return_vectors,physical_step):
        values=np.asarray(return_vectors,np.float64)
        if values.shape!=(3,2)or not np.isfinite(values).all():raise ValueError('One real objective-vector mean for each population member required')
        for policy,value in zip(self.population,values):self.native.archive.add(policy,value)
        self.native._MORLD__adapt_weights([v.copy()for v in values])
        self.psa_calls+=1;self.last_psa_step=int(physical_step)
        return dict(psa_applied=True,psa_calls=self.psa_calls,archive_policies=len(self.native.archive.evaluations),
            weights=[p.weights.tolist()for p in self.population],value_source=self.value_source)

    def maybe_adapt(self,physical_step):
        if self.value_source!='training_returns'or physical_step-self.last_psa_step<self.psa_every or not all(self.training_returns):
            return dict(psa_applied=False,psa_calls=self.psa_calls)
        values=np.stack([np.mean(rows,axis=0)for rows in self.training_returns])
        return self.accept_internal_evaluation(values,physical_step)

    def _state(self,with_optimizers):
        population=[]
        for p in self.population:
            item=p.wrapped
            state={name:_cpu(getattr(item,name).state_dict())for name in ('actor','qf1','qf2','qf1_target','qf2_target')}
            state.update(weights=p.weights.copy(),alpha=float(item.alpha),global_step=item.global_step)
            if item.autotune:state['log_alpha']=_cpu(item.log_alpha)
            if with_optimizers:state['optimizers']={name:_cpu(getattr(item,name).state_dict())for name in ('actor_optimizer','q_optimizer','a_optimizer')if hasattr(item,name)}
            population.append(state)
        return dict(format=VERSION,population=population,initial_weights=self.initial_weights.copy(),
            collect_member=self.collect_member,last_selected_episode=self.last_selected_episode,physical_step=self.physical_step,
            updates=self.updates,member_update_counts=self.member_update_counts.copy(),member_actor_updates=self.member_actor_updates.copy(),
            selection_changes=self.selection_changes,psa_calls=self.psa_calls,last_psa_step=self.last_psa_step,
            training_returns=_cpu(self.training_returns),return_counts=self.return_counts.copy(),
            archive_individuals=_cpu(self.native.archive.individuals),archive_evaluations=_cpu(self.native.archive.evaluations),
            archive_insertions=self.native.archive.insertions,native_iteration=self.native.iteration,native_current_policy=self.native.current_policy,
            cross_updates=self.cross_updates,exchange_events=self.exchange_events,last_exchange_step=self.last_exchange_step,
            optimizer_backup=with_optimizers,replay_backup=False,metadata=self.metadata())

    def state_dict(self):return self._state(True)
    def checkpoint_state_dict(self):return self.inference_state_dict()

    def inference_state_dict(self):
        """One fixed endpoint,3current actors; no Q/Adam/replay/old archive actors."""
        return dict(format=INFERENCE_VERSION,obs_dim=self.obs_dim,action_dim=self.action_dim,
            initial_weights=self.initial_weights.copy(),deployment_member=self.deployment_member,
            actors=[_cpu(p.wrapped.actor.state_dict())for p in self.population],
            current_weights=[p.weights.copy()for p in self.population],metadata=self.metadata(),
            optimizer_backup=False,replay_backup=False,critic_backup=False,
            archive_actor_backup=False,exact_continuation_claimed=False)

    def load_inference_state_dict(self,state):
        if state.get('format')!=INFERENCE_VERSION or int(state['obs_dim'])!=self.obs_dim or int(state['action_dim'])!=self.action_dim:
            raise ValueError('Different MORL/D inference geometry/format')
        _selected_state_metadata(state)
        if len(state['actors'])!=3 or not np.array_equal(state['initial_weights'],self.initial_weights):
            raise ValueError('Different MORL/D inference population')
        if int(state['deployment_member'])!=self.deployment_member:
            raise ValueError('Different predeclared deployment member')
        saved_weights=np.asarray(state['current_weights'],np.float64)
        if saved_weights.shape!=(3,2)or not np.isfinite(saved_weights).all()or np.any(saved_weights<0)or not np.allclose(saved_weights.sum(1),1.):
            raise ValueError('Invalid retained population weights')
        if int(state['metadata']['extra_update_passes_for_other_members'])!=self.cross_update_passes:
            raise ValueError('Different MORL/D retained update profile')
        self.cross_updates=int(state['metadata']['extra_native_critic_updates'])
        for policy,actor,weights in zip(self.population,state['actors'],saved_weights):
            policy.wrapped.actor.load_state_dict(actor)
            policy.weights=np.asarray(weights,np.float64).copy()
            policy.wrapped.set_weights(policy.weights)
        return self

    def load_state_dict(self,state):
        if state.get('format')==INFERENCE_VERSION:return self.load_inference_state_dict(state)
        if state['format']!=VERSION or not np.array_equal(state['initial_weights'],self.initial_weights):raise ValueError('Different MORL/D task/population contract')
        _selected_state_metadata(state)
        for p,saved in zip(self.population,state['population']):
            item=p.wrapped
            for name in ('actor','qf1','qf2','qf1_target','qf2_target'):getattr(item,name).load_state_dict(saved[name])
            p.weights=np.asarray(saved['weights'],np.float64).copy();item.set_weights(p.weights)
            item.global_step=int(saved['global_step']);item.alpha=float(saved['alpha'])
            if item.autotune:
                with th.no_grad():item.log_alpha.copy_(saved['log_alpha'].to(self.device))
                item.alpha_tensor=item.log_alpha.exp()
            else:item.alpha_tensor=th.scalar_tensor(item.alpha,device=self.device)
            for name,value in saved.get('optimizers',{}).items():getattr(item,name).load_state_dict(value)
        for name in ('collect_member','last_selected_episode','physical_step','updates','member_update_counts','member_actor_updates',
                     'selection_changes','psa_calls','last_psa_step','training_returns','return_counts'):
            setattr(self,name,copy.deepcopy(state[name]))
        for name in ('cross_updates','exchange_events','last_exchange_step'):
            setattr(self,name,copy.deepcopy(state[name]))
        self.native.archive.individuals=_cpu(state['archive_individuals']);self.native.archive.evaluations=_cpu(state['archive_evaluations'])
        self.native.archive.insertions=int(state['archive_insertions']);self.native.iteration=int(state['native_iteration'])
        self.native.current_policy=int(state['native_current_policy'])

    def metadata(self):
        return dict(method='MORL/D-SB-PSA',version=VERSION,selected_profile=SELECTED_PROFILE,
            actor_lr=3e-4,critic_lr=1e-4,other_update_every_training_episodes=1,
            sole_current_implementation=True,publication_year=2024,venue='JAIR79:679–723',
            paper='https://jair.org/index.php/jair/article/view/15702',official_repository='https://github.com/LucasAlegre/morl-baselines',
            source_commit=COMMIT,license='MIT',native_sources=native_definitions()['source_sha256'],
            native_core='author MOSAC actor/vector twin-Q/update, round-robin selector and PSA; Pareto pruning from author source',
            diagnostic_instrumentation='one return statement appended after unchanged native update body',
            population_size=3,initial_weights=self.initial_weights.tolist(),current_weights=[p.weights.tolist()for p in self.population],
            shared_replay='one worker-owned uniformly sampled RAM replay; no per-policy replay or on-disk replay',
            deployment='fixed member initially trained at [.8,.2]; never selected by DEV or archive peak',deployment_member=self.deployment_member,
            gamma=1.,neural_dtype='float32',hidden=list(self.population[0].wrapped.net_arch),
            critic_update='native scalarized vector-Q MSE',policy_frequency=self.population[0].wrapped.policy_freq,
            delayed_actor_compensation='native policy_freq actor updates on each eligible actor event',
            autotune=self.population[0].wrapped.autotune,configured_alpha=self.config.get('alpha',.005),
            archive='native Pareto criterion, actor-only snapshots; no optimizer/replay clones',
            psa_value_source=self.value_source,psa_every=self.psa_every,psa_delta=self.native.delta,
            psa_internal_cases_per_policy=int(self.config.get('morld_psa_internal_cases',5))if self.value_source=='charged_internal'else 0,
            psa_internal_evaluation='fresh deterministic full episodes, independent training-master indexes>=1000000, worker charges every physical step to total budget'if self.value_source=='charged_internal'else 'not enabled',
            training_return_adaptation='disabled; fresh charged deterministic population evaluation used'if self.value_source=='charged_internal'else'latest up to8 actually collected stochastic terminal returns per member; not fresh deterministic evaluation; mixed-policy-age estimates',
            population_exchange='one complete training episode per current member in native round-robin order; shared replay available to each member',
            extra_update_passes_for_other_members=self.cross_update_passes,
            extra_native_critic_updates=self.cross_updates,
            task_exchange_steps=self.horizon,paper_table1_exchange_steps=50000,
            pinned_code_default_exchange_steps=40000,
            exchange_schedule_is_short_budget_adaptation=True,
            after_exchange_hook='once after each completed training episode; never DEV/internal PSA',
            extra_environment_steps_in_agent=0,holdout_used=False,reward_normalization=False,
            constructor_seed_scope='CPU_default_generator_only_inside_CPU_fork',
            caller_owns_stochastic_training_rng=True,
            publication_date_caveat='JAIR2024 expanded article; arXiv2023 and extended abstract2022 precede this window',
            native_paper_default_claimed=False,exact_continuation_claimed=False)
