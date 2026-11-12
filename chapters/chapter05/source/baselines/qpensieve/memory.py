"""Q-Pensieve (ICLR 2023), adapted from the authors' MIT implementation.

Source: vendor/qpensieve/agent.py:368-486 and base.py:QMemory.
The essential max over old Q snapshots AND alternate preference evaluations is
preserved.  The same sampled current-preference action is used for every Q.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
from .primitives import SACBase, VectorCritic, as_tensor, freeze


class Agent(SACBase):
    name = 'qpensieve'

    def __init__(self, obs_dim, action_dim, seed=0, device='cpu', config=None):
        super().__init__(obs_dim, action_dim, seed, device, config)
        c=self.config
        self.snapshot_warmup = int(c.get('qp_snapshot_warmup', c.get('snapshot_warmup', 20000)))
        self.snapshot_frequency = int(c.get('qp_snapshot_frequency', c.get('q_frequency', 1000)))
        self.snapshot_capacity = int(c.get('qp_snapshot_capacity', c.get('buf_num', 4)))
        self.preference_candidates = int(c.get('qp_preference_candidates', c.get('prefer_num', 4)))
        if self.snapshot_frequency <= 0 or self.snapshot_capacity < 0 or self.preference_candidates < 1:
            raise ValueError('Invalid Q-Pensieve memory configuration')
        self.snapshots=[]
        self.snapshot_position=0
        self.snapshot_creations=0
        self.memory_actor_updates=0
        self.last_candidate_count=0

    def policy_value(self, obs, preferences, actions):
        # Official PREF contains the current target preference, then alternatives.
        pref=[preferences[0]]
        pref += [as_tensor(self.random_preference(), self.device)
                 for _ in range(self.preference_candidates-1)]
        candidates=torch.stack(pref)
        b, k=len(obs), len(candidates)
        states=obs[None].expand(k,-1,-1).reshape(k*b,-1)
        sampled=actions[None].expand(k,-1,-1).reshape(k*b,-1)
        eval_prefs=candidates[:,None].expand(-1,b,-1).reshape(k*b,2)
        target_prefs=preferences[None].expand(k,-1,-1).reshape(k*b,2)
        # All snapshot critics see this identical tensor.  Concatenate once;
        # the current critic and frozen snapshots still execute independently.
        critic_inputs=torch.cat([states,sampled,eval_prefs],-1)
        values=[]
        critics=[self.critic]+self.snapshots
        for critic in critics:
            q1,q2=critic(critic_inputs)
            q=torch.minimum((q1*target_prefs).sum(-1),(q2*target_prefs).sum(-1))
            values.append(q.reshape(k,b))
        # Preserve all forward candidates and their original GEMM shapes. A
        # critic with no winning candidate receives only zeros from this max;
        # detach that whole graph instead of executing its all-zero backward.
        # Retain the original path for public non-actor/custom-critic calls,
        # other devices, and invalid weights (0*Inf must not hide a NaN guard).
        if (len(critics)>1 and actions.device.type=='cpu' and actions.requires_grad
                and torch.is_grad_enabled()
                and all(isinstance(critic,VectorCritic) for critic in critics)
                and all(not p.requires_grad and np.isfinite(p.detach().numpy()).all()
                        for critic in critics for p in critic.parameters())):
            with torch.no_grad():
                winners=torch.cat([value.detach() for value in values],0).max(0).indices//k
                active=set(winners.unique().tolist())
            values=[value if i in active else value.detach() for i,value in enumerate(values)]
        self.last_candidate_count=k*(1+len(self.snapshots))
        self.memory_actor_updates += int(bool(self.snapshots))
        return torch.cat(values,0).max(0).values

    def after_update(self):
        created=False
        if self.snapshot_capacity and self.updates>self.snapshot_warmup and self.updates%self.snapshot_frequency==0:
            critic=copy.deepcopy(self.critic).eval();freeze(critic)
            if len(self.snapshots)<self.snapshot_capacity:
                self.snapshots.append(critic)
            else:
                self.snapshots[self.snapshot_position]=critic
            self.snapshot_position=(self.snapshot_position+1)%self.snapshot_capacity
            self.snapshot_creations+=1;created=True
        return dict(q_snapshot_count=len(self.snapshots),q_snapshot_created=created,
                    q_snapshot_creations=self.snapshot_creations,
                    q_memory_actor_updates=self.memory_actor_updates,
                    q_actor_candidate_count=self.last_candidate_count)

    def metadata(self):
        m=super().metadata()
        m.update(paper='Q-Pensieve: Boosting Sample Efficiency of Multi-Objective RL Through Memory Sharing of Q-Snapshots',
                 venue='ICLR2023',source='https://github.com/rl-bandits-lab/Q-Pensieve',license='MIT',
                 core='max scalarized conservative twin-Q over current/historical critics and sampled preference set',
                 snapshot_warmup=getattr(self,'snapshot_warmup',20000),
                 snapshot_frequency=getattr(self,'snapshot_frequency',1000),
                 snapshot_capacity=getattr(self,'snapshot_capacity',4),
                 preference_candidates=getattr(self,'preference_candidates',4),
                 vectorization='preference evaluations batched; Q snapshot identities retained')
        return m

    def state_dict(self):
        p=super().state_dict()
        p.update(q_snapshots=[x.state_dict() for x in self.snapshots],snapshot_position=self.snapshot_position,
                 snapshot_creations=self.snapshot_creations,memory_actor_updates=self.memory_actor_updates)
        return p

    def load_state_dict(self,payload):
        super().load_state_dict(payload);self.snapshots=[]
        for state in payload.get('q_snapshots',[]):
            critic=copy.deepcopy(self.critic).eval();critic.load_state_dict(state);freeze(critic)
            self.snapshots.append(critic)
        self.snapshot_position=int(payload.get('snapshot_position',0))
        self.snapshot_creations=int(payload.get('snapshot_creations',0))
        self.memory_actor_updates=int(payload.get('memory_actor_updates',0))
