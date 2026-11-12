"""Single collection/evaluation/persistence facade for five ONLINE comparators."""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np

METHODS = ('cal', 'ccpo', 'c_ccpo', 'safe_sac', 'safedreamer')


def make(method, domain, *, seed=0, device='cpu', profile='formal', overrides=None):
    if method not in METHODS or domain not in ('v5','radio'):
        raise ValueError('Registered Chapter 3 method and domain required')
    if profile not in ('formal','engineering'):
        raise ValueError('formal or engineering profile required')
    D,A,H = (242,121,24) if domain=='v5' else (303,101,32)
    params = dict(overrides or {})
    if 'device' in params and params['device'] != device:
        raise ValueError('Configuration and explicit execution device disagree')
    params.pop('device',None)
    if 'horizon' in params and params['horizon'] != H:
        raise ValueError('The native physical horizon cannot be changed by an algorithm config')
    for field in ('gamma','cost_gamma','reward_discount','cost_discount','discount','safety_gamma'):
        if field in params and params[field] != 1.:
            raise ValueError('This release uses the unchanged finite-horizon undiscounted objective')
    for field in ('cost_limit','cost_lim','budget'):
        if field in params and params[field] != 1.:
            raise ValueError('The physical resource cost/B threshold is fixed at one')
    if method=='cal':
        from .cal import CALCandidate, CALConfig
        if profile=='engineering':
            params={**dict(hidden_size=32,policy_train_batch_size=8,init_exploration_steps=0,
                           min_pool_size=8,num_train_repeat=1,replay_size=1024),**params}
        agent=CALCandidate(D,A,seed,device=device,config=CALConfig(**params))
    elif method=='ccpo':
        from .ccpo import CCPOCandidate, CCPOConfig
        params={**dict(fixed_full_budget=True,behavior_conditions=(1.,),versatile_conditions=(1.,),
                       collection_episodes_per_epoch=4,batch_per_condition=512),**params}
        if not params['fixed_full_budget'] or tuple(params['behavior_conditions'])!=(1.,) or tuple(params['versatile_conditions'])!=(1.,):
            raise ValueError('This full-spend release compares the fixed full-budget CCPO specialization')
        for name in ('behavior_conditions','versatile_conditions'):
            if name in params:params[name]=tuple(params[name])
        if profile=='engineering':
            params={**dict(hidden_size=32,condition_hidden_size=8,feature_dim=8,
                pretraining_epochs=1,batch_per_condition=4,policy_particles=8,
                m_step_iterations=2,replay_capacity=1024),**params}
        agent=CCPOCandidate(D,A,seed,device=device,config=CCPOConfig(**params))
    elif method=='c_ccpo':
        from .c_ccpo import Agent, Config
        expected_normalizer=1.5-0.5/H
        if 'weighted_cost_normalizer' in params and params['weighted_cost_normalizer']!=expected_normalizer:
            raise ValueError('Full-spend SPF costs use their analytic finite-horizon upper bound')
        params['weighted_cost_normalizer']=expected_normalizer
        if 'hidden_sizes' in params:params['hidden_sizes']=tuple(params['hidden_sizes'])
        if profile=='engineering':
            params={**dict(rollout_steps=16,policy_iterations=3,value_iterations=3),**params}
        agent=Agent(D,A,Config(**dict(params,device=device)),seed)
    elif method=='safe_sac':
        from .safe_sac import SafeSACAgent, SafeSACConfig
        if 'hidden_sizes' in params:params['hidden_sizes']=tuple(params['hidden_sizes'])
        if profile=='engineering':
            params={**dict(hidden_sizes=(32,32),batch_size=8,replay_capacity=1024,
                           candidate_count=8,refinement_steps=1),**params}
        agent=SafeSACAgent(D,-np.ones(A,np.float32),np.ones(A,np.float32),
                          SafeSACConfig(**dict(params,device=device)),seed)
    else:
        from .safedreamer import SafeDreamerCandidate, SafeDreamerConfig
        config=(SafeDreamerConfig.engineering(H) if profile=='engineering'
                else SafeDreamerConfig(horizon=H))
        config=replace(config,**params)
        agent=SafeDreamerCandidate(D,A,seed,device=('gpu' if device.startswith('cuda') else 'cpu'),config=config)
    return OnlineAgent(method,domain,agent,seed,device,profile)


class OnlineAgent:
    def __init__(self,method,domain,agent,seed,device,profile):
        self.method,self.domain,self.agent=method,domain,agent
        self.seed,self.device,self.profile=int(seed),device,profile
        self.environment_steps=self.episodes=0
        self.condition_index=0
        self.evaluation_only=False

    @property
    def condition(self):
        return self.agent.config.behavior_conditions[self.condition_index] if self.method=='ccpo' else 1.

    @property
    def metadata(self):
        meta=getattr(self.agent,'metadata',None)
        meta=meta() if callable(meta) else deepcopy(meta)
        if meta is None:
            meta=dict(config=asdict(self.agent.config),implementation='paper-derived')
        # Canonical active identity belongs to this facade and the release registry;
        # remove the standalone study's historical routing labels.
        for key in ('candidate_only','active_registry_changed','active_route_modified'):
            meta.pop(key,None)
        return dict(method=self.method,domain=self.domain,seed=self.seed,device=self.device,
                    profile=self.profile,online_from_scratch=True,offline_data=False,
                    environment_steps=self.environment_steps,episodes=self.episodes,
                    condition_index=self.condition_index,algorithm=meta)

    @property
    def accounting(self):
        a=getattr(self.agent,'accounting',None)
        a=a() if callable(a) else deepcopy(a)
        if a is None and self.method=='safe_sac':
            a=dict(environment_steps=self.agent.transitions_seen,optimizer_blocks=self.agent.total_updates,
                   penalized_transitions=self.agent.penalized_transitions)
        if a is None and self.method=='c_ccpo':
            a=self.agent.metadata()['counters']
        return dict(environment_steps=self.environment_steps,complete_episodes=self.episodes,
                    algorithm=a)

    def start_episode(self):
        if self.method=='safedreamer':self.agent.reset_episode()

    def act(self,observation,history_cost=0.,deterministic=False):
        if self.method=='ccpo':return self.agent.act(observation,1. if deterministic else self.condition,deterministic)
        if self.method=='safe_sac':return self.agent.act(observation,history_cost,deterministic)
        if self.method=='c_ccpo':return self.agent.act(observation,history_cost,deterministic,record=not deterministic)
        return self.agent.act(observation,deterministic)

    def observe(self,obs,action,reward,cost,nxt,done,*,history_cost=0.,episode_step=0):
        if self.evaluation_only:raise ValueError('This fixed terminal model is evaluation-only')
        a,m=self.agent,self.method
        result={}
        if m=='cal':
            result=a.collect_transition(obs,action,reward,cost,nxt,done) or {}
        elif m=='ccpo':
            a.collect_transition(obs,action,reward,cost,nxt,done,self.condition)
        elif m=='c_ccpo':
            result={'learning_reward':a.observe(obs,action,reward,cost,nxt,done)}
        elif m=='safe_sac':
            result=a.observe(obs,history_cost,action,reward,cost,nxt,done,episode_step)
            if a.transitions_seen>=a.config.batch_size:
                result.update(a.update())
            if done:result.update(a.end_episode(history_cost+cost))
        else:
            a.observe(obs,action,reward,cost,nxt,done)
            if done:
                count=max(1,round(a.config.train_ratio*a.config.horizon/
                                 (a.config.batch_size*a.config.batch_length)))
                metrics=a.train_updates(count)
                result={'train_updates_requested':count,'train_updates_completed':len(metrics)}
        self.environment_steps+=1
        if done:
            self.episodes+=1
            if m=='ccpo':
                self.condition_index=(self.condition_index+1)%len(a.config.behavior_conditions)
                if self.condition_index==0 and (not a.config.fixed_full_budget or
                        self.episodes%a.config.collection_episodes_per_epoch==0):
                    result={'epoch':a.finish_epoch()}
        return result

    def finish(self):
        if self.method=='c_ccpo':return self.agent.update(terminal=True)
        return {}

    @contextmanager
    def evaluation_scope(self):
        if self.method=='safedreamer':
            with self.agent.evaluation_scope():yield
        elif self.method=='safe_sac':
            # Deterministic numeric search consumes no policy RNG, but diagnostics
            # belong to the training stream and must be restored too.
            info=deepcopy(getattr(self.agent,'last_action_info',None))
            try:yield
            finally:self.agent.last_action_info=info
        else:
            yield

    def evaluation_observe(self,obs,action,reward,cost,nxt,done):
        if self.method=='safedreamer':
            self.agent.observe(obs,action,reward,cost,nxt,done,training=False)

    def save(self,directory):
        directory=Path(directory)
        if directory.exists():raise FileExistsError('Never overwrite a terminal model')
        if self.method in ('cal','ccpo'):
            self.agent.save(directory,include_replay=False)
            payload=directory/'state.pt'
        elif self.method=='safedreamer':
            payload=self.agent.save(directory)
        else:
            import torch
            directory.mkdir(parents=True)
            state=(self.agent.state_dict(full=False) if self.method=='c_ccpo'
                   else self.agent.state_dict(include_training=False))
            payload=directory/'state.pt';torch.save(state,payload)
        (directory/'online_manifest.json').write_text(json.dumps(self.metadata,ensure_ascii=False,
                                                               indent=2,allow_nan=False)+'\n')
        return payload

    @classmethod
    def load(cls,directory,*,expected_domain=None):
        directory=Path(directory)
        meta=json.loads((directory/'online_manifest.json').read_text())
        m,d=meta['method'],meta['domain']
        if expected_domain is not None and d!=expected_domain:raise ValueError('Model domain differs')
        if m in ('cal','ccpo'):
            if m=='cal':from .cal import CALCandidate as K
            else:from .ccpo import CCPOCandidate as K
            agent=K.load(directory,device=meta['device'],restore_training=False)
            result=cls(m,d,agent,meta['seed'],meta['device'],meta['profile'])
        elif m=='safedreamer':
            from .safedreamer import SafeDreamerCandidate
            agent=SafeDreamerCandidate.load(directory,restore_training=False)
            result=cls(m,d,agent,meta['seed'],meta['device'],meta['profile'])
        else:
            import torch
            state=torch.load(directory/'state.pt',map_location=meta['device'],weights_only=False)
            params=dict(meta['algorithm']['config'])
            params.pop('device',None)
            result=make(m,d,seed=meta['seed'],device=meta['device'],profile='formal',overrides=params)
            result.profile=meta['profile']
            if m=='c_ccpo':result.agent.load_state_dict(state,full=False)
            else:result.agent.load_state_dict(state,load_training=False)
        result.environment_steps=meta['environment_steps'];result.episodes=meta['episodes']
        result.condition_index=meta['condition_index'];result.evaluation_only=True
        return result
