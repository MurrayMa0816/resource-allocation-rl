"""Focused migration checks, using tiny synthetic geometry rather than new results."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
from gymnasium import spaces
from allocation23.baselines.agent import make_agent, dump_agent, load_agent, agent_parameter_vector
from allocation23.baselines.cores.neural_input import FeatureCore
from allocation23.baselines.actions import action_family_for, amount_from_control, reference_amount

METHODS=('paspo','rpo','td7','simba_v2','dime')

class TinyEnv:
    """Small fully public finite task, only for core API and continuation checks."""
    def __init__(self, chapter=2):
        self.chapter=chapter;self.horizon=4;self.n_objects=3;self.action_dim=3+int(chapter==3)
        self.train_path_base=17;self.environment_steps=0;self.episode_records=[];self.next_episode=0
        sizes=[('local',(3,2)),('global',(1+int(chapter==3),)),('caps',(self.action_dim,)),('budget',())]
        self.schema=[];offset=0
        for n,shape in sizes:
            count=int(np.prod(shape,dtype=int));self.schema.append(dict(name=n,shape=list(shape),start=offset,stop=offset+count,divisor=1.));offset+=count
        self.observation_space=spaces.Box(-np.inf,np.inf,(offset,),dtype=np.float64)
        self.action_space=spaces.Box(-1.,1.,(self.action_dim,),dtype=np.float32)
        self.geometry=[i for f in self.schema if f['name'] in ('caps','budget') for i in range(f['start'],f['stop'])]
        self.policy=[i for f in self.schema if f['name'] in ('local','global') for i in range(f['start'],f['stop'])]
        self.t=0;self.remaining=4.;self.quality=0.
        self.spec=SimpleNamespace(initial_budget=4.)
        self.action_family='polytope'
    def configure_method(self,method):
        self.action_family=action_family_for(method,self.chapter)
    @property
    def decision_state(self):
        global_=np.array([self.t/4]+([self.remaining/4] if self.chapter==3 else []))
        return dict(local=(np.arange(6).reshape(3,2)+self.next_episode+self.t)/10,
                    **{'global':global_},caps=np.full(3,2.),
                    budget=np.asarray(min(self.remaining,6.)),remaining=np.asarray(self.remaining))
    def get_contract(self):
        return dict(schema='tiny-calibrated-core-check-v2',chapter=self.chapter,action_family=self.action_family,
                    observation_schema=self.schema,policy_indices=self.policy)
    def _obs(self):
        local=(np.arange(6).reshape(3,2)+self.next_episode+self.t)/10
        global_=np.array([self.t/4]+([self.remaining/4] if self.chapter==3 else []))
        target=min(self.remaining,1.) if self.chapter==2 else self.remaining
        caps=np.array([target]+[2.]*3) if self.chapter==3 else np.full(3,2.)
        return np.concatenate([local.ravel(),global_,caps,[target]])
    def reset(self,seed=None):
        self.next_episode+=1;self.t=0;self.remaining=4.;self.quality=0.
        return self._obs(),{}
    def step(self,action):
        from allocation23.methods.constraints import capped_simplex
        target=min(self.remaining,1.) if self.chapter==2 else self.remaining
        caps=np.array([target]+[2.]*3) if self.chapter==3 else np.full(3,2.)
        if self.chapter==3 and self.action_family=='total_space':
            amount=amount_from_control(float(action[0]),target,reference_amount(self.decision_state,self.horizon))
            physical=capped_simplex(amount*(np.asarray(action)[1:]+1)/2,caps[1:],amount)
        else:
            allocation=capped_simplex(target*(np.asarray(action)+1)/2,caps,target)
            physical=allocation[1:] if self.chapter==3 else allocation
        self.remaining=max(0.,self.remaining-physical.sum())
        self.quality+=float(physical @ np.array([.1,.2,.3]));self.t+=1;self.environment_steps+=1
        done=self.t==4;reward=self.quality if done else 0.
        if done:self.episode_records.append(dict(raw_return=reward,steps=4))
        return self._obs(),reward,done,False,{}
    def export_state(self):
        return copy.deepcopy(dict(next_episode=self.next_episode,t=self.t,remaining=self.remaining,quality=self.quality,environment_steps=self.environment_steps,episode_records=self.episode_records))
    def import_state(self,state):
        for key,value in copy.deepcopy(state).items():setattr(self,key,value)


def source_updates_unchanged():
    root=Path(__file__).parent
    expected=json.loads((root/'migration_sources.json').read_text())['update_methods_ast_sha256']
    for key,digest in expected.items():
        filename,target=key.split(':');classname,methodname=target.split('.')
        tree=ast.parse((root/filename).read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==methodname)
        actual=hashlib.sha256(ast.dump(method,include_attributes=False).encode()).hexdigest()
        assert actual==digest,key
    return len(expected)


def run(output):
    rows=[]
    for chapter in (2,3):
        for method in METHODS:
            env=TinyEnv(chapter);agent=make_agent(method,env,5,profile='smoke',training_episodes=8)
            first=agent_parameter_vector(agent).copy()
            obs=agent._last_obs if agent._last_obs is not None else env._obs()
            altered=obs.copy();altered[env.geometry]*=2
            core=agent.core
            if isinstance(core,FeatureCore):
                before=torch.random.get_rng_state();a=core.act(obs,deterministic=method!='dime');torch.random.set_rng_state(before);b=core.act(altered,deterministic=method!='dime');assert np.array_equal(a,b)
            elif method=='paspo':
                a=torch.as_tensor(obs,dtype=torch.float32).reshape(1,-1);b=torch.as_tensor(altered,dtype=torch.float32).reshape(1,-1)
                assert torch.equal(core.actor.encoder(a),core.actor.encoder(b));assert torch.equal(core.critic(a),core.critic(b))
            else:
                a=torch.as_tensor(obs,dtype=torch.float32).reshape(1,-1);b=torch.as_tensor(altered,dtype=torch.float32).reshape(1,-1)
                assert torch.equal(core.obs_norm(a),core.obs_norm(b))
            agent.learn(16)
            with tempfile.TemporaryDirectory() as td:
                path=Path(td)/'checkpoint';dump_agent(agent,path,env,dict(test='core'))
                agent.learn(16);expected=agent_parameter_vector(agent).copy();expected_account=agent.accounting
                reenv=TinyEnv(chapter);reenv.configure_method(method)
                restored=load_agent(path,reenv,dict(test='core'),restore_rng=True)
                restored.learn(16);actual=agent_parameter_vector(restored)
                assert np.array_equal(actual,expected),(chapter,method,float(np.max(np.abs(actual-expected))))
                assert restored.num_timesteps==32
                assert not np.array_equal(first,actual),(chapter,method,'no training change')
            rows.append(dict(chapter=chapter,method=method,steps=32,updates=int(expected_account['training_updates']),geometry_neural_isolation=True,checkpoint_resume_exact=True,parameters_changed=True))
    report=dict(update_methods_ast_equal=source_updates_unchanged(),rows=rows,status='passed',scope='synthetic API/update/continuation checks only; not research results')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    import sys
    run(sys.argv[1])
