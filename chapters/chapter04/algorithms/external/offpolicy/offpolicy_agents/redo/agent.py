# SPDX-License-Identifier: Apache-2.0
"""SAC+ReDo, paper SAC profile, explicit TF-Agents-to-Torch task adapter.

No Env is constructed. One observe() collects one paid transition and performs
RR8 updates after the paid10000-step prefix. See PROVENANCE.json for boundaries.
"""
from __future__ import annotations
import copy
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn

try:
    from .networks import DenseTFAdam, SquashedActor, TwinCritic
except ImportError:
    from networks import DenseTFAdam, SquashedActor, TwinCritic


def tensor_hash(module):
    h=hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(key.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


class RAMReplay:
    def __init__(self, capacity, observations, actions):
        self.capacity, self.observations, self.actions = int(capacity), int(observations), int(actions)
        self.data=np.empty((self.capacity, 2*observations+actions+2),dtype=np.float32)
        self.size=0

    def add(self, obs, action, reward, nxt, done):
        if self.size>=self.capacity:
            raise RuntimeError('registered ReDo replay capacity exceeded')
        d,a=self.observations,self.actions
        row=self.data[self.size]
        row[:d]=obs;row[d:d+a]=action;row[d+a]=reward
        row[d+a+1:d+a+1+d]=nxt;row[-1]=done
        self.size+=1

    def fingerprint(self):
        return hashlib.sha256(memoryview(self.data[:self.size]).cast('B')).hexdigest()


class Agent:
    def __init__(self, cfg, observation_dim, action_dim, device='cpu'):
        self.cfg=dict(cfg);self.observation_dim=int(observation_dim);self.action_dim=int(action_dim)
        self.device=torch.device(device)
        self.warmup=int(self.cfg.get('warmup_steps',10000))
        self.rr=int(self.cfg.get('replay_ratio',8));self.batch_size=int(self.cfg.get('batch_size',256))
        self.recycle_period=int(self.cfg.get('recycle_period',200000))
        self.threshold=float(self.cfg.get('dormant_threshold',0.))
        self.reset_mode=self.cfg.get('reset_mode','neurons')
        if self.reset_mode!='neurons' or self.cfg.get('reset_actor',True) is not True or self.cfg.get('reset_critic',True) is not True:
            raise ValueError('registered ReDo requires neuron recycling in actor and both critics')
        if self.cfg.get('reset_target_models',False) or self.cfg.get('scale_recycled_weights',False):
            raise ValueError('registered ReDo keeps direct target reset/scaled recycling disabled')
        if self.rr<1 or self.warmup<1 or self.recycle_period<1 or self.threshold<0:
            raise ValueError('invalid explicit ReDo clock/threshold')
        self.gamma=float(self.cfg.get('gamma',.99));self.tau=float(self.cfg.get('tau',.005))
        self.reward_scale=float(self.cfg.get('reward_scale',1.))
        self.target_entropy=-self.action_dim/2.
        seed=int(self.cfg.get('seed',0));torch.manual_seed(seed)
        self.rng=np.random.default_rng(seed+1947)
        self.actor=SquashedActor(self.observation_dim,self.action_dim).to(self.device)
        self.critic=TwinCritic(self.observation_dim,self.action_dim).to(self.device)
        self.critic_target=copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha=nn.Parameter(torch.tensor(float(self.cfg.get('initial_log_alpha',0.)),device=self.device))
        lr=float(self.cfg.get('learning_rate',3e-4))
        self.actor_optimizer=DenseTFAdam(self.actor.parameters(),lr=lr)
        self.critic_optimizer=DenseTFAdam(self.critic.parameters(),lr=lr)
        self.alpha_optimizer=DenseTFAdam([self.log_alpha],lr=lr)
        self.replay=RAMReplay(int(self.cfg.get('replay_capacity',self.cfg.get('total_steps',49152))),self.observation_dim,self.action_dim)
        self.steps=0;self.n_updates=0;self.events=[];self._inference_only=False;self._last_losses=None

    def _observation(self, obs):
        x=np.asarray(obs,dtype=np.float32)
        if x.shape!=(self.observation_dim,) or not np.isfinite(x).all():
            raise ValueError('finite public observation with registered shape required')
        return x

    def act(self, obs, deterministic=False):
        obs=self._observation(obs)
        if not self._inference_only and not deterministic and self.steps<self.warmup:
            return self.rng.uniform(-1,1,self.action_dim).astype(np.float32)
        return self.policy_action(obs,deterministic=deterministic)

    @torch.no_grad()
    def policy_action(self, obs, deterministic=False):
        obs=self._observation(obs)
        action,_,_=self.actor.sample(torch.as_tensor(obs,device=self.device),deterministic=deterministic)
        if not torch.isfinite(action).all():
            raise FloatingPointError('nonfinite ReDo policy action; no silent std clipping')
        return action.cpu().numpy()

    def observe(self, obs, action, reward, next_obs, done):
        if self._inference_only:
            raise RuntimeError('inference endpoint is not a continuation checkpoint')
        o=self._observation(obs);no=self._observation(next_obs);a=np.asarray(action,dtype=np.float32)
        if a.shape!=(self.action_dim,) or not np.isfinite(a).all() or np.max(np.abs(a))>1.000001 or not math.isfinite(reward):
            raise ValueError('invalid native transition')
        self.replay.add(o,a,float(reward),no,bool(done));self.steps+=1
        before=len(self.events)
        if self.steps>self.warmup:
            # All8 independent uniform batches condition on exactly the same
            # unchanged replay. One contiguous upload; no free data/UTD change.
            idx=self.rng.integers(0,self.replay.size,size=(self.rr,self.batch_size))
            batches=torch.as_tensor(self.replay.data[idx],device=self.device)
            for batch in batches:
                self._update_batch(batch)
        return {'steps':self.steps,'updates':self.n_updates,'recycle_events':self.events[before:]}

    def _parts(self,batch):
        d,a=self.observation_dim,self.action_dim
        return batch[:,:d],batch[:,d:d+a],batch[:,d+a],batch[:,d+a+1:d+a+1+d],batch[:,-1]

    def _update_batch(self,batch,noises=None):
        obs,action,reward,nxt,done=self._parts(batch)
        before=self.n_updates
        due=before>0 and before%self.recycle_period==0
        noise=(None,None,None) if noises is None else noises
        with torch.no_grad():
            next_action,next_logp,_=self.actor.sample(nxt,noise=noise[0])
            t1,t2=self.critic_target(nxt,next_action)
            target=self.reward_scale*reward+(1-done)*self.gamma*(torch.minimum(t1,t2)-self.log_alpha.exp()*next_logp)
        self.critic_optimizer.zero_grad()
        if due:
            q1,q2,critic_activations=self.critic(obs,action,capture=True)
        else:
            q1,q2=self.critic(obs,action)
        critic_loss=.5*((q1-target).square()+(q2-target).square()).mean()
        torch._assert_async(torch.isfinite(critic_loss),'ReDo critic loss nonfinite')
        critic_loss.backward();self.critic_optimizer.step()
        for p in self.critic.parameters():p.requires_grad_(False)
        self.actor_optimizer.zero_grad()
        proposed,logp,actor_activations=self.actor.sample(obs,noise=noise[1],capture=due)
        qa,qb=self.critic(obs,proposed)
        actor_loss=(self.log_alpha.detach().exp()*logp-torch.minimum(qa,qb)).mean()
        torch._assert_async(torch.isfinite(actor_loss),'ReDo actor loss nonfinite')
        actor_loss.backward();self.actor_optimizer.step()
        for p in self.critic.parameters():p.requires_grad_(True)
        # Official hook is after critic+actor updates, before alpha update and
        # BEFORE incrementing the TF train_step_counter.
        if due:
            self._recycle(critic_activations,actor_activations,before)
        self.alpha_optimizer.zero_grad()
        with torch.no_grad():
            _,alpha_logp,_=self.actor.sample(obs,noise=noise[2])
        alpha_loss=-(self.log_alpha*(alpha_logp+self.target_entropy)).mean()
        torch._assert_async(torch.isfinite(alpha_loss),'ReDo alpha loss nonfinite')
        alpha_loss.backward();self.alpha_optimizer.step()
        self.n_updates+=1
        with torch.no_grad():
            for online,target_parameter in zip(self.critic.parameters(),self.critic_target.parameters()):
                target_parameter.copy_((1-self.tau)*target_parameter+self.tau*online)
        self._last_losses=torch.stack([critic_loss.detach(),actor_loss.detach(),alpha_loss.detach()])

    @torch.no_grad()
    def _recycle_network(self,name,network,optimizer,activations,counter):
        # Freeze every layer mask before changing adjacent-layer weights.
        masks=[];scores=[]
        for act in activations:
            magnitude=act[:256].abs().mean(0)
            score=magnitude/(magnitude.mean()+1e-9)
            if not torch.isfinite(score).all():raise FloatingPointError('nonfinite ReDo activity score')
            masks.append(score<=self.threshold);scores.append(score)
        row={'network':name,'layers':[],'optimizer_iterations_before':optimizer.iterations}
        # Author performs ALL incoming replacements before ALL outgoing zeros.
        for i,(mask,score) in enumerate(zip(masks,scores)):
            layer=network.layers[i]
            generator=torch.Generator(device=self.device).manual_seed(counter)
            new=torch.empty_like(layer.weight)
            limit=math.sqrt(6/(layer.in_features+layer.out_features))
            new.uniform_(-limit,limit,generator=generator)
            incoming=mask[:,None].expand_as(layer.weight)
            layer.weight.copy_(torch.where(incoming,new,layer.weight))
            optimizer.clear_kernel_mask(layer.weight,incoming)
            layer.bias.masked_fill_(mask,0)
            # Intentionally preserve bias Adam slots and the optimizer's global
            # iteration: this matches reset_dead_neurons/reset_momentum source.
            row['layers'].append({'hidden_layer':i,'neurons':len(mask),'dead_indices':mask.nonzero().flatten().cpu().tolist(),
                'dead_count':int(mask.sum()),'score_min':float(score.min()),'score_max':float(score.max()),
                'score_mean':float(score.mean()),'incoming_kernel_entries':int(incoming.sum()),
                'bias_slots_reset':False})
        for i,mask in enumerate(masks):
            outgoing_layer=network.layers[i+1]
            outgoing=mask[None,:].expand_as(outgoing_layer.weight)
            outgoing_layer.weight.masked_fill_(outgoing,0)
            optimizer.clear_kernel_mask(outgoing_layer.weight,outgoing)
            row['layers'][i]['outgoing_kernel_entries']=int(outgoing.sum())
        checked_slots=0
        for i,layer in enumerate(network.layers):
            changed=torch.zeros_like(layer.weight,dtype=torch.bool)
            if i<len(masks):changed|=masks[i][:,None]
            if i>0:changed|=masks[i-1][None,:]
            for slot in ['m','v']:
                nonzero=int(torch.count_nonzero(optimizer.state[layer.weight][slot][changed]))
                if nonzero:raise RuntimeError('masked kernel Adam slot was not cleared')
                checked_slots+=int(changed.sum())
        row['masked_kernel_Adam_nonzero_after_reset']=0
        row['masked_kernel_Adam_entries_checked']=checked_slots
        row['optimizer_iterations_after']=optimizer.iterations
        return row

    def _recycle(self,critic_activations,actor_activations,counter):
        target_before=tensor_hash(self.critic_target);replay_before=self.replay.fingerprint()
        rows=[self._recycle_network('critic1',self.critic.q1,self.critic_optimizer,critic_activations[0],counter),
              self._recycle_network('critic2',self.critic.q2,self.critic_optimizer,critic_activations[1],counter),
              self._recycle_network('actor',self.actor,self.actor_optimizer,actor_activations,counter)]
        target_after=tensor_hash(self.critic_target);replay_after=self.replay.fingerprint()
        if target_before!=target_after or replay_before!=replay_after:
            raise RuntimeError('ReDo must not directly reset target or replay')
        event={'environment_step':self.steps,'update_counter_before':counter,'update_number':counter+1,
               'event_index':len(self.events)+1,'networks':rows,
               'total_dead_neurons':sum(l['dead_count'] for r in rows for l in r['layers']),
               'replay_size':self.replay.size,'replay_before_sha256':replay_before,'replay_after_sha256':replay_after,
               'target_before_sha256':target_before,'target_after_direct_recycle_sha256':target_after,
               'target_direct_reset':False,'target_regular_polyak_after_hook':True,
               'alpha_direct_reset':False,'kernel_Adam_slots_masked_reset':True,'bias_Adam_slots_reset':False}
        event['recycle_initializer_seed']=counter
        event['optimizer_global_iterations_reset']=False
        self.events.append(event)

    def diagnostics(self):
        return {'steps':self.steps,'env_steps':self.steps,'updates':self.n_updates,'n_updates':self.n_updates,
            'replay_size':self.replay.size,'replay_ratio':self.rr,'warmup_steps':self.warmup,
            'recycle_period':self.recycle_period,'recycle_event_count':len(self.events),
            'reset_mode':self.reset_mode,
            'last_recycle_event':self.events[-1] if self.events else None,
            'alpha':float(self.log_alpha.detach().exp().cpu()),
            'critic_actor_alpha_losses':None if self._last_losses is None else self._last_losses.cpu().tolist(),
            'optimizer_iterations':{'actor':self.actor_optimizer.iterations,'critic':self.critic_optimizer.iterations,'alpha':self.alpha_optimizer.iterations}}

    def native_state_fingerprint(self):
        return {'actor':tensor_hash(self.actor),'critic':tensor_hash(self.critic),
                'target':tensor_hash(self.critic_target),'log_alpha':float(self.log_alpha.detach().cpu()),
                'steps':self.steps,'updates':self.n_updates,'replay_size':self.replay.size,'events':len(self.events)}

    def save(self,path):
        torch.save({'schema':'redo-actor-inference-v1','cfg':self.cfg,'observation_dim':self.observation_dim,
                    'action_dim':self.action_dim,'actor':{k:v.detach().cpu() for k,v in self.actor.state_dict().items()},
                    'diagnostics':self.diagnostics()},Path(path))

    @classmethod
    def load(cls,path,device='cpu'):
        x=torch.load(path,map_location='cpu',weights_only=False)
        if x['schema']!='redo-actor-inference-v1':raise ValueError('unknown ReDo endpoint')
        a=cls(x['cfg'],x['observation_dim'],x['action_dim'],device)
        a.actor.load_state_dict(x['actor'],strict=True);a.steps=int(x['diagnostics']['steps'])
        a.n_updates=int(x['diagnostics']['updates']);a._inference_only=True
        return a
