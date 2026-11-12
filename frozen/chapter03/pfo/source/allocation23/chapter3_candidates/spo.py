"""Independent paper-derived ICML2025 Simple Policy Optimization (Eq16).

Inspected numerical facts: author repo9fdeda315b00cee82d1dc4e10c01db200c97e909.
No author implementation copied. Native quadratic ratio loss, combined Adam,
Gaussian latent likelihood, GAE, value clipping and optimization clocks retained.
Task adaptations: public SharedObjectMean and one serial trajectory/RMS stream
instead of native eight 256-step worker streams. Native population RMS with
Gymnasium0.29.1 terminal-before-add recurrence and +/-10 clips is explicit.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import copy, csv, gzip, json, math
import numpy as np
import torch
from torch import nn

from .shared_mean import SharedObjectMean


class Moments:
    """Independent single-sample parallel population-moment recurrence."""
    def __init__(self,shape=()):
        self.mean=np.zeros(shape,np.float64);self.var=np.ones(shape,np.float64);self.count=1e-4
    def update(self,x):
        x=np.asarray(x,np.float64);n=self.count+1.;d=x-self.mean
        self.mean=self.mean+d/n
        self.var=(self.var*self.count+d*d*self.count/n)/n;self.count=n
    def normalize(self,x):return (np.asarray(x,np.float64)-self.mean)/np.sqrt(self.var+1e-8)
    def state(self):return dict(mean=torch.from_numpy(np.asarray(self.mean).copy()),var=torch.from_numpy(np.asarray(self.var).copy()),count=self.count)
    def restore(self,state):
        self.mean=state['mean'].detach().cpu().numpy().copy();self.var=state['var'].detach().cpu().numpy().copy();self.count=float(state['count'])


def linear(i,o,gain):
    layer=nn.Linear(i,o);nn.init.orthogonal_(layer.weight,gain);nn.init.zeros_(layer.bias);return layer


def mlp(dimensions,last_gain):
    layers=[]
    for i,(a,b) in enumerate(zip(dimensions[:-1],dimensions[1:])):
        layers.append(linear(a,b,last_gain if i==len(dimensions)-2 else math.sqrt(2)))
        if i<len(dimensions)-2:layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self,D,A):
        super().__init__();self.mean=mlp([D,256,256,128,128,64,64,A],.01)
        self.log_std=nn.Parameter(torch.zeros(1,A))
    def distribution(self,x):
        mean=self.mean(x);return torch.distributions.Normal(mean,self.log_std.exp().expand_as(mean))


def spo_objective(ratio,advantage,epsilon=.2):
    """Eq16 per-sample objective, with no ratio/KL clip or early stop."""
    return ratio*advantage-advantage.abs()*(ratio-1).square()/(2*float(epsilon))


def value_loss(value,old_value,target,epsilon=.2):
    clipped=old_value+(value-old_value).clamp(-epsilon,epsilon)
    return .5*torch.maximum((value-target).square(),(clipped-target).square()).mean()


def gae(rewards,dones,values,last_value,gamma=.99,lam=.95):
    """All real rows credited; terminal masks cut both bootstrap and recursion."""
    r=np.asarray(rewards,np.float32);d=np.asarray(dones,np.float32);v=np.asarray(values,np.float32)
    out=np.empty(len(r),np.float32);carry=np.float32(0);following=np.float32(last_value)
    g,l=np.float32(gamma),np.float32(lam)
    for i in range(len(r)-1,-1,-1):
        mask=np.float32(1-d[i]);delta=np.float32(r[i]+g*mask*following-v[i])
        carry=np.float32(delta+g*l*mask*carry);out[i]=carry;following=v[i]
    return out


class Candidate:
    DEFAULTS=dict(horizon=32,actor_architecture='shared_mean',rollout_steps=2048,total_steps=6144,
        epochs=10,minibatches=4,gamma=.99,gae_lambda=.95,epsilon=.2,learning_rate=3e-4,
        learning_rate_decay=True,value_coefficient=.5,entropy_coefficient=0.,max_grad_norm=.5,
        reward_scale=1.,normalize_observation=True,normalize_reward=True,normalization_clip=10.,
        adam_epsilon=1e-8)
    def __init__(self,obs_dim,action_dim,seed,device='cuda',config=None):
        self.D,self.A,self.seed=int(obs_dim),int(action_dim),int(seed)
        self.horizon={(303,101):32,(242,121):24}[(self.D,self.A)]
        unknown=set(config or {})-set(self.DEFAULTS)
        if unknown:raise ValueError('Unknown SPO config: '+str(sorted(unknown)))
        self.config={**self.DEFAULTS,'horizon':self.horizon,**(config or {})};c=self.config
        if c['horizon']!=self.horizon or c['actor_architecture'] not in ('shared_mean','flat'):
            raise ValueError('SPO public layout/horizon/architecture mismatch')
        if c['minibatches']<1 or c['rollout_steps']//c['minibatches']<2 or c['rollout_steps']%c['minibatches'] or c['total_steps']%c['rollout_steps']:
            raise ValueError('SPO requires whole evenly partitioned rollout batches')
        if not(0<c['gamma']<1 and 0<c['gae_lambda']<=1 and c['epsilon']>0 and c['epochs']>0 and
               c['learning_rate']>0 and c['max_grad_norm']>0 and c['reward_scale']==1.):
            raise ValueError('Invalid SPO numerical contract')
        self.device=torch.device(device)
        with torch.random.fork_rng(devices=[]):
            cpu=torch.Generator(device='cpu');cpu.manual_seed(self.seed);torch.set_rng_state(cpu.get_state())
            # Author entry initializes the native critic before its actor.
            self.critic=mlp([self.D,64,64,1],1.)
            self.actor=Actor(self.D,self.A)
            post_native_init=torch.get_rng_state().clone()
            if c['actor_architecture']=='shared_mean':
                with torch.random.fork_rng(devices=[]):
                    private=torch.Generator(device='cpu');private.manual_seed(self.seed+930000)
                    torch.set_rng_state(private.get_state())
                    self.actor.mean=SharedObjectMean(self.D,self.A)
        self.actor.to(self.device);self.critic.to(self.device)
        self.generator=torch.Generator(device=self.device)
        if self.device.type=='cpu':self.generator.set_state(post_native_init)
        else:self.generator.manual_seed(self.seed)
        self.shuffle=np.random.RandomState(self.seed)
        # Native Agent traversal: direct logstd, critic module, then mean module.
        self.parameters=[self.actor.log_std]+list(self.critic.parameters())+list(self.actor.mean.parameters())
        self.optimizer=torch.optim.Adam(self.parameters,lr=c['learning_rate'],betas=(.9,.999),eps=c['adam_epsilon'])
        self.observation_moments=Moments((self.D,));self.return_moments=Moments();self.discounted_return=0.
        self.environment_steps=self.episodes=self.updates=self.rollout_updates=0
        self.terminal_rows=self.optimized_positions=self.optimizer_terminal_positions=0
        self.buffer=[];self.pending=None;self.cached_next=None;self.last={};self.evaluation_only=False;self.optimizer_trace=[]

    def _observation(self,obs,update=False):
        x=np.asarray(obs,np.float32)
        if x.shape!=(self.D,) or not np.isfinite(x).all():raise ValueError('Invalid public SPO observation')
        if self.config['normalize_observation']:
            if update:self.observation_moments.update(x)
            x=self.observation_moments.normalize(x)
        return np.clip(x,-self.config['normalization_clip'],self.config['normalization_clip']).astype(np.float32)

    @torch.no_grad()
    def act(self,obs,deterministic=False):
        if deterministic:
            x=self._observation(obs,False)
            mean=self.actor.distribution(torch.as_tensor(x,device=self.device).unsqueeze(0)).mean[0]
            return mean.clamp(-1,1).cpu().numpy().astype(np.float32)
        if self.evaluation_only:raise RuntimeError('SPO endpoint is evaluation-only')
        if self.pending is not None:raise RuntimeError('SPO must observe the preceding actual action')
        raw=np.asarray(obs,np.float32)
        if self.cached_next is not None and np.array_equal(raw,self.cached_next[0]):x=self.cached_next[1].copy()
        else:x=self._observation(raw,True)
        self.cached_next=None;s=torch.as_tensor(x,device=self.device).unsqueeze(0)
        dist=self.actor.distribution(s)
        latent=dist.mean[0]+dist.stddev[0]*torch.randn((self.A,),generator=self.generator,device=self.device)
        logp=dist.log_prob(latent.unsqueeze(0)).sum(-1)[0];v=self.critic(s).flatten()[0]
        packed=torch.cat((latent,logp.reshape(1),v.reshape(1),dist.mean[0],dist.stddev[0])).cpu().numpy()
        action=packed[:self.A].clip(-1,1).astype(np.float32)
        self.pending=(raw.copy(),x,packed[:self.A].copy(),float(packed[self.A]),float(packed[self.A+1]),action.copy(),packed[self.A+2:self.A*2+2].copy(),packed[self.A*2+2:].copy())
        return action

    def observe(self,obs,action,native_reward,next_obs,done):
        if self.evaluation_only:raise RuntimeError('SPO endpoint contains no optimizer/replay contract')
        if self.pending is None:raise RuntimeError('SPO observe requires its cached raw Gaussian action')
        raw,x,latent,logp,v,bounded,old_mean,old_std=self.pending
        if not(np.array_equal(raw,np.asarray(obs,np.float32)) and np.array_equal(bounded,np.asarray(action,np.float32))):
            raise ValueError('SPO transition differs from the actual sampled state/action')
        if bool(done)!=(self.environment_steps%self.horizon==self.horizon-1):raise ValueError('SPO native episode changed')
        if not math.isfinite(float(native_reward)):raise ValueError('Nonfinite native reward')
        self.pending=None;next_raw=np.asarray(next_obs,np.float32);nxt=self._observation(next_raw,True)
        self.cached_next=None if done else (next_raw.copy(),nxt.copy())
        reward=float(native_reward)
        if self.config['normalize_reward']:
            # Gymnasium0.29.1: termination mask BEFORE addition, carrying terminal r onward.
            self.discounted_return=self.config['gamma']*(1-float(done))*self.discounted_return+reward
            self.return_moments.update(self.discounted_return)
            reward=reward/float(np.sqrt(self.return_moments.var+1e-8))
        reward=float(np.clip(reward,-self.config['normalization_clip'],self.config['normalization_clip']))
        self.buffer.append((x,latent,logp,v,reward,float(done),nxt,old_mean,old_std))
        self.environment_steps+=1;self.episodes+=int(bool(done));self.terminal_rows+=int(bool(done))
        if len(self.buffer)==int(self.config['rollout_steps']):self._update()
        return dict(self.last,learning_reward=reward)

    def _update(self):
        c=self.config;rows=self.buffer;N=len(rows)
        if N!=c['rollout_steps']:raise RuntimeError('Partial SPO rollout would change native clocks')
        if self.rollout_updates>=c['total_steps']//c['rollout_steps']:raise RuntimeError('SPO declared training budget exhausted')
        obs=np.stack([r[0] for r in rows]);latent=np.stack([r[1] for r in rows])
        oldp=np.asarray([r[2] for r in rows],np.float32);oldv=np.asarray([r[3] for r in rows],np.float32)
        reward=np.asarray([r[4] for r in rows],np.float32);done=np.asarray([r[5] for r in rows],np.float32)
        with torch.no_grad():last=0. if done[-1] else float(self.critic(torch.as_tensor(rows[-1][6],device=self.device).unsqueeze(0)).flatten()[0].cpu())
        advantage=gae(reward,done,oldv,last,c['gamma'],c['gae_lambda']);target=oldv+advantage
        old_mean=np.stack([r[7] for r in rows]);old_std=np.stack([r[8] for r in rows])
        data=[torch.as_tensor(x,device=self.device) for x in [obs,latent,oldp,oldv,advantage,target,old_mean,old_std]]
        if c['learning_rate_decay']:
            fraction=1.-self.rollout_updates/(c['total_steps']//c['rollout_steps'])
            lr=c['learning_rate']*fraction
        else:lr=c['learning_rate']
        for group in self.optimizer.param_groups:group['lr']=lr
        indices=np.arange(N);B=N//int(c['minibatches']);packed_metrics=[];labels=[]
        for epoch in range(int(c['epochs'])):
            self.shuffle.shuffle(indices)
            for first in range(0,N,B):
                ix=torch.as_tensor(indices[first:first+B],dtype=torch.long,device=self.device)
                s,u,op,ov,adv,ret,old_mu,old_sigma=[x[ix] for x in data]
                # Native per-minibatch SAMPLE std, epsilon outside the std.
                adv=(adv-adv.mean())/(adv.std(unbiased=True)+1e-8)
                distribution=self.actor.distribution(s);logp=distribution.log_prob(u).sum(-1)
                ratio=(logp-op).exp();policy=-spo_objective(ratio,adv,c['epsilon']).mean()
                value=value_loss(self.critic(s).flatten(),ov,ret,c['epsilon'])
                entropy=distribution.entropy().sum(-1).mean()
                loss=policy+c['value_coefficient']*value-c['entropy_coefficient']*entropy
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=nn.utils.clip_grad_norm_(self.parameters,c['max_grad_norm'],error_if_nonfinite=True)
                self.optimizer.step();self.updates+=1
                with torch.no_grad():
                    new_mu,new_sigma=distribution.mean,distribution.stddev
                    kl=(torch.log(new_sigma/old_sigma)+((old_mu-new_mu).square()+old_sigma.square()-new_sigma.square())/(2*new_sigma.square())).sum(-1).mean()
                packed_metrics.append(torch.stack((policy.detach(),value.detach(),entropy.detach(),norm.detach(),
                    ratio.detach().max(),(ratio.detach()-1).abs().mean(),logp.detach().sub(op).abs().max(),kl.detach())))
                labels.append((self.updates,epoch,first//B))
        all_metrics=torch.stack(packed_metrics).cpu().numpy();metrics=all_metrics[-1].tolist()
        if not np.isfinite(all_metrics).all():raise FloatingPointError('Nonfinite native SPO update')
        for label,values in zip(labels,all_metrics):
            self.optimizer_trace.append(dict(update=label[0],rollout=self.rollout_updates+1,epoch=label[1],minibatch=label[2],batch_size=B,learning_rate=lr,**dict(zip(('policy_loss','value_loss','entropy','gradient_norm','ratio_max','ratio_deviation','abs_logratio_max','analytic_KL'),map(float,values)))))
        self.rollout_updates+=1;self.optimized_positions+=N*int(c['epochs'])
        self.optimizer_terminal_positions+=int(done.sum())*int(c['epochs'])
        self.last=dict(policy_loss=metrics[0],value_loss=metrics[1],entropy=metrics[2],gradient_norm=metrics[3],
            analytic_KL=metrics[7],final_minibatch_ratio_max=metrics[4],final_minibatch_ratio_deviation=metrics[5],final_minibatch_abs_logratio_max=metrics[6],
            learning_rate=lr,rollout_updates=self.rollout_updates,optimizer_updates=self.updates,rollout_size=N,
            rollout_terminal_rows=int(done.sum()),optimized_positions=self.optimized_positions,
            optimized_terminal_positions=self.optimizer_terminal_positions)
        self.buffer=[]

    def finish(self):
        if self.pending is not None or self.buffer:raise RuntimeError('SPO endpoint requires complete registered rollouts')
        return dict(self.last)

    def stats(self):
        return dict(self.last)

    def accounting(self):
        return dict(environment_steps=self.environment_steps,complete_episodes=self.episodes,
            updates=self.updates,actor_updates=self.updates,value_updates=self.updates,
            rollout_updates=self.rollout_updates,terminal_rows=self.terminal_rows,
            effective_training_positions=self.optimized_positions,optimized_terminal_positions=self.optimizer_terminal_positions,
            partial_rollout_steps=len(self.buffer),pending_behavior_action=int(self.pending is not None),
            observation_normalizer_count=float(self.observation_moments.count),reward_normalizer_count=float(self.return_moments.count),
            evaluation_only=self.evaluation_only)

    @contextmanager
    def evaluation_scope(self,common_seed=None):
        # Deterministic execution already never updates RMS, private RNG or pending behavior.
        yield

    def save(self,directory):
        if self.evaluation_only:raise RuntimeError('Loaded SPO endpoint cannot be relabeled a training trace')
        p=Path(directory);p.mkdir(parents=True,exist_ok=False)
        state=dict(actor={k:v.detach().cpu().clone() for k,v in self.actor.state_dict().items()},
            config=copy.deepcopy(self.config),obs_dim=self.D,action_dim=self.A,seed=self.seed,
            observation_moments=self.observation_moments.state(),return_moments=self.return_moments.state(),
            discounted_return=self.discounted_return,accounting=self.accounting(),evaluation_only=True)
        torch.save(state,p/'model.pt')
        (p/'metadata.json').write_text(json.dumps(dict(method='spo',
            implementation='independent-paper-derived ICML2025 Simple Policy Optimization Eq16',
            no_optimizer_or_replay=True,critic_saved=False,config=self.config),indent=2)+'\n')
        # Complete scalar trace is scientific data, not Adam/replay/critic state.
        if self.optimizer_trace:
            with gzip.open(p.parent/'optimizer_updates.csv.gz','wt',newline='',compresslevel=5) as f:
                writer=csv.DictWriter(f,fieldnames=list(self.optimizer_trace[0]));writer.writeheader();writer.writerows(self.optimizer_trace)
        return p/'model.pt'

    @classmethod
    def load(cls,directory,device='cuda'):
        state=torch.load(Path(directory)/'model.pt',map_location='cpu',weights_only=True)
        if state['evaluation_only'] is not True:raise ValueError('Not an evaluation-only SPO endpoint')
        result=cls(state['obs_dim'],state['action_dim'],state['seed'],device,state['config'])
        result.actor.load_state_dict(state['actor']);result.observation_moments.restore(state['observation_moments'])
        result.return_moments.restore(state['return_moments']);result.discounted_return=float(state['discounted_return'])
        a=state['accounting'];result.environment_steps=a['environment_steps'];result.episodes=a['complete_episodes']
        result.updates=a['updates'];result.rollout_updates=a['rollout_updates'];result.terminal_rows=a['terminal_rows']
        result.optimized_positions=a['effective_training_positions'];result.optimizer_terminal_positions=a['optimized_terminal_positions']
        result.evaluation_only=True;return result
