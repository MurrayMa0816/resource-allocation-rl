"""Paper-derived online Gradient PPO / Reflective PPO, explicit task adaptations.

The two paper objectives are implemented separately below. Shared boilerplate
only handles online normalization, true chronological RAM rollouts, native
Gaussian latent likelihoods, and compact evaluation-only serialization.
No upstream implementation is copied into this file. See provenance manifests.
"""
from __future__ import annotations
from pathlib import Path
import copy, json, math
import numpy as np
import torch
from torch import nn

DEFAULT_ALGORITHM = 'gradient_ppo'
from .shared_mean import SharedObjectMean


class RunningMoments:
    """Population moments, including the ordinary 1e-4 cold-start prior."""
    def __init__(self, shape=()):
        self.mean=np.zeros(shape,np.float64)
        self.var=np.ones(shape,np.float64)
        self.count=1e-4
    def update(self,x):
        x=np.asarray(x,np.float64)
        # A single real online observation/discounted return, not a new dataset.
        delta=x-self.mean; total=self.count+1.
        self.mean=self.mean+delta/total
        self.var=(self.var*self.count+delta*delta*self.count/total)/total
        self.count=total
    def normalize(self,x):
        return ((np.asarray(x,np.float64)-self.mean)/np.sqrt(self.var+1e-8)).astype(np.float32)
    def state(self):
        return dict(mean=self.mean.copy(),var=self.var.copy(),count=self.count)
    def restore(self,state):
        self.mean=np.asarray(state['mean'],np.float64).copy()
        self.var=np.asarray(state['var'],np.float64).copy();self.count=float(state['count'])


def orthogonal_linear(i,o,gain):
    layer=nn.Linear(i,o)
    nn.init.orthogonal_(layer.weight,gain);nn.init.zeros_(layer.bias)
    return layer


class GaussianActor(nn.Module):
    def __init__(self,D,A,initial_total_std=1.):
        super().__init__()
        initial_total_std=float(initial_total_std)
        if not math.isfinite(initial_total_std) or initial_total_std<=0:
            raise ValueError('initial_total_std must be finite and positive')
        self.body=nn.Sequential(orthogonal_linear(D,64,math.sqrt(2)),nn.Tanh(),
                                orthogonal_linear(64,64,math.sqrt(2)),nn.Tanh())
        self.mean=orthogonal_linear(64,A,.01)
        self.log_std=nn.Parameter(torch.zeros(A))
        # Public total_space decoding reads amount from control[0]; space is [1:].
        # This only initializes the ordinary learnable Gaussian parameter.
        if initial_total_std!=1.:
            with torch.no_grad():self.log_std[0]=math.log(initial_total_std)
        self.shared=False
    def attach_shared(self,D,A,seed):
        # Attach after all original actor/value/h initializations; no parent RNG drift.
        with torch.random.fork_rng(devices=[]):
            generator=torch.Generator(device='cpu').manual_seed(int(seed)+930000)
            torch.set_rng_state(generator.get_state())
            self.mean=SharedObjectMean(D,A)
        self.shared=True
    def distribution(self,x):
        mean=self.mean(x) if self.shared else self.mean(self.body(x))
        return torch.distributions.Normal(mean,self.log_std.exp().expand_as(mean))
    def log_probability(self,x,latent):
        return self.distribution(x).log_prob(latent).sum(-1)


class Value(nn.Module):
    def __init__(self,D):
        super().__init__()
        self.net=nn.Sequential(orthogonal_linear(D,64,math.sqrt(2)),nn.Tanh(),
            orthogonal_linear(64,64,math.sqrt(2)),nn.Tanh(),orthogonal_linear(64,1,1.))
    def forward(self,x):return self.net(x).squeeze(-1)


def lambda_advantages(reward,done,value,next_value,gamma,lam):
    """Differentiable finite-sequence lambda TD errors, with true-terminal cuts.

All inputs are [sequences,length]. Matrix multiplication is exactly the
unrolled reverse recurrence; it avoids allocating a per-parameter Jacobian.
The prefix-done mask removes every product crossing a terminal transition.
"""
    if not (reward.shape==done.shape==value.shape==next_value.shape):
        raise ValueError('Every lambda-return operand must have the same [B,L] shape')
    L=reward.shape[-1]
    delta=reward+float(gamma)*(1-done)*next_value-value
    prefix=torch.cat([torch.zeros_like(done[:,:1]),done[:,:-1].cumsum(-1)],-1)
    positions=torch.arange(L,device=reward.device)
    distance=positions[None,:]-positions[:,None]
    coefficients=(float(gamma)*float(lam))**distance.clamp_min(0).to(reward.dtype)
    weights=(distance>=0).to(reward.dtype)*coefficients
    weights=weights.unsqueeze(0)*(prefix.unsqueeze(-1)==prefix.unsqueeze(-2)).to(reward.dtype)
    return torch.bmm(weights,delta.unsqueeze(-1)).squeeze(-1)


def stale_gae(rewards,dones,values,last_value,gamma,lam):
    """One CPU reverse pass per rollout; all stored values are already on CPU."""
    output=np.empty(len(rewards),np.float32)
    carry=np.float32(0.);following=np.float32(last_value)
    g,l=np.float32(gamma),np.float32(lam)
    for t in range(len(rewards)-1,-1,-1):
        mask=np.float32(1.-float(dones[t]))
        delta=np.float32(rewards[t]+g*mask*following-values[t])
        carry=np.float32(delta+g*l*mask*carry);output[t]=carry
        following=np.float32(values[t])
    return output


def clipped_objective(logp,old_logp,adv,epsilon):
    ratio=(logp-old_logp).exp()
    return torch.minimum(ratio*adv,ratio.clamp(1-epsilon,1+epsilon)*adv)


def reflective_objective(logp,old_logp,adv,next_logp,next_old_logp,next_adv,
                         active,epsilon=.2,next_epsilon=.1,beta=.3):
    """Paper L0+beta*L1: separately clipped ratios; BOTH receive gradients."""
    ratio=(logp-old_logp).exp();next_ratio=(next_logp-next_old_logp).exp()
    base=torch.minimum(ratio*adv,ratio.clamp(1-epsilon,1+epsilon)*adv)
    future=torch.minimum(ratio*next_ratio*next_adv,
        ratio.clamp(1-epsilon,1+epsilon)*next_ratio.clamp(1-next_epsilon,1+next_epsilon)*next_adv)
    return base+float(beta)*active*future


class Candidate:
    DEFAULTS=dict(horizon=32,rollout_steps=2048,shared_mean=True,initial_total_std=1.,normalize_observation=True,
        normalize_reward=True,actor_lr=3e-4,gamma=.99,gae_lambda=.95,clip_epsilon=.2,
        max_grad_norm=.5,adam_epsilon=1e-5,adam_epsilon_convention='bias_corrected_torch',metric_interval=1,reward_scale=1.,
        critic_lr=3e-3,h_lr=3e-3,epochs=4,minibatches=8,sequence_length=32,
        h_regularization=1.,reflective_beta=.3,next_epsilon=.1,
        value_clip=True,normalizer_only_warmup=0,complete_transition_windows=True)
    def __init__(self,obs_dim,action_dim,seed,device='cuda',config=None):
        self.algorithm=DEFAULT_ALGORITHM
        defaults=dict(self.DEFAULTS)
        if self.algorithm=='reflective_ppo':
            defaults.update(gamma=.995,gae_lambda=.97,critic_lr=3e-4,epochs=10,minibatches=32,value_clip=False,
                            adam_epsilon=1e-7,adam_epsilon_convention='keras_epsilon_hat')
        unknown=set(config or {})-set(defaults)
        if unknown:raise ValueError('Unknown on-policy configuration: '+str(sorted(unknown)))
        self.config={**defaults,**(config or {})};c=self.config
        if c['normalizer_only_warmup']!=0:
            raise ValueError('This registered quick profile counts all rollouts; no hidden warmup')
        if not (0<c['gamma']<1 and 0<=c['gae_lambda']<=1):raise ValueError('Invalid discount/lambda')
        if int(c['rollout_steps'])<2 or int(c['sequence_length'])<2:raise ValueError('Empty rollout/sequence')
        self.D,self.A,self.seed=int(obs_dim),int(action_dim),int(seed)
        self.device=torch.device(device);torch.manual_seed(seed);self.rng=np.random.default_rng(seed+32001)
        # Every native base initialization precedes the local, shared-mean addition.
        self.actor=GaussianActor(self.D,self.A,c['initial_total_std'])
        self.critic=Value(self.D)
        self.h=Value(self.D) if self.algorithm=='gradient_ppo' else None
        if c['shared_mean']:self.actor.attach_shared(self.D,self.A,self.seed)
        self.actor.to(self.device);self.critic.to(self.device)
        if self.h is not None:self.h.to(self.device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=c['actor_lr'],eps=c['adam_epsilon'])
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=c['critic_lr'],eps=c['adam_epsilon'])
        self.h_optimizer=(torch.optim.Adam(self.h.parameters(),lr=c['h_lr'],eps=c['adam_epsilon'])
                          if self.h is not None else None)
        self.observation_moments=RunningMoments((self.D,));self.return_moments=RunningMoments()
        self.discounted_return=0.;self.environment_steps=self.updates=self.actor_updates=0
        self.h_updates=self.rollout_updates=self.effective_td_positions=0
        self.critic_terminal_positions=self.critic_nonzero_reward_positions=0
        self.buffer=[];self.pending=None;self.cached_next=None;self.last={};self.evaluation_only=False

    def _normalized(self,obs,update=False):
        x=np.asarray(obs,np.float32)
        if x.shape!=(self.D,) or not np.isfinite(x).all():raise ValueError('Public observation shape/nonfinite')
        if self.config['normalize_observation']:
            if update:self.observation_moments.update(x)
            return self.observation_moments.normalize(x)
        return x.copy()

    @torch.no_grad()
    def act(self,obs,deterministic=False):
        if deterministic:
            x=self._normalized(obs,False)
            mean=self.actor.distribution(torch.as_tensor(x,device=self.device).unsqueeze(0)).mean[0]
            return mean.clamp(-1,1).cpu().numpy().astype(np.float32)
        if self.evaluation_only:raise RuntimeError('Endpoint model is evaluation only')
        if self.pending is not None:raise RuntimeError('Observe the preceding real action before sampling again')
        raw=np.asarray(obs,np.float32)
        if self.cached_next is not None and np.array_equal(raw,self.cached_next[0]):
            x=self.cached_next[1].copy()
        else:x=self._normalized(raw,True)
        self.cached_next=None
        s=torch.as_tensor(x,device=self.device).unsqueeze(0)
        distribution=self.actor.distribution(s);latent=distribution.sample()[0]
        logp=distribution.log_prob(latent.unsqueeze(0)).sum(-1)[0];value=self.critic(s)[0]
        # One small D2H transfer per real decision, rather than three synchronizations.
        packed=torch.cat([latent,logp.reshape(1),value.reshape(1)]).cpu().numpy()
        bounded=packed[:self.A].clip(-1,1).astype(np.float32)
        self.pending=(raw.copy(),x,packed[:self.A].copy(),float(packed[-2]),float(packed[-1]),bounded.copy())
        return bounded

    def observe(self,obs,action,native_reward,next_obs,done):
        if self.evaluation_only:raise RuntimeError('Endpoint model contains no replay/optimizer')
        if self.pending is None:raise RuntimeError('No sampled raw Gaussian latent to credit')
        raw,x,latent,logp,value,bounded=self.pending
        if not np.array_equal(raw,np.asarray(obs,np.float32)) or not np.array_equal(bounded,np.asarray(action,np.float32)):
            raise ValueError('Observed public transition differs from the actual sampled action')
        self.pending=None
        next_raw=np.asarray(next_obs,np.float32);next_normalized=self._normalized(next_raw,True)
        self.cached_next=None if done else (next_raw.copy(),next_normalized.copy())
        reward=float(native_reward)*float(self.config['reward_scale'])
        if self.config['normalize_reward']:
            # Gym-style discounted-return population normalization, terminal mask BEFORE addition.
            self.discounted_return=self.config['gamma']*self.discounted_return*(1-float(done))+reward
            self.return_moments.update(np.asarray(self.discounted_return))
            learning_reward=reward/float(np.sqrt(self.return_moments.var+1e-8))
        else:learning_reward=reward
        self.buffer.append((x,latent,logp,value,learning_reward,float(done),next_normalized))
        self.environment_steps+=1
        if len(self.buffer)>=int(self.config['rollout_steps']):self._update_rollout()
        return dict(self.last,learning_reward=learning_reward)

    def _step(self,optimizer,parameters,loss,clip=True):
        optimizer.zero_grad(set_to_none=True);loss.backward()
        parameters=list(parameters)
        if clip:
            norm=nn.utils.clip_grad_norm_(parameters,float(self.config['max_grad_norm']))
        else:
            norm=torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in parameters if p.grad is not None]))
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite actual policy/value/trace gradient')
        if self.config['adam_epsilon_convention']=='keras_epsilon_hat':
            # Algebraically match Keras2.1 epsilon on UNCORRECTED sqrt(v).
            iteration=(self.actor_updates if optimizer is self.actor_optimizer else self.updates)+1
            for group in optimizer.param_groups:
                group['eps']=float(self.config['adam_epsilon'])/math.sqrt(1-group['betas'][1]**iteration)
        optimizer.step();return norm.detach()

    def _update_rollout(self):
        c=self.config;rows=self.buffer;N=len(rows)
        observations=np.stack([r[0] for r in rows]);latents=np.stack([r[1] for r in rows])
        next_observations=np.stack([r[6] for r in rows])
        logp=np.asarray([r[2] for r in rows],np.float32);old_values=np.asarray([r[3] for r in rows],np.float32)
        rewards=np.asarray([r[4] for r in rows],np.float32);dones=np.asarray([r[5] for r in rows],np.float32)
        with torch.no_grad():last_value=(0. if dones[-1] else float(self.critic(torch.as_tensor(rows[-1][6],device=self.device).unsqueeze(0))[0].cpu()))
        advantages=stale_gae(rewards,dones,old_values,last_value,c['gamma'],c['gae_lambda'])
        targets=advantages+old_values
        # One transfer per rollout of each stacked contiguous field.
        data={k:torch.as_tensor(v,device=self.device) for k,v in dict(obs=observations,next_obs=next_observations,latent=latents,
            old_logp=logp,old_value=old_values,reward=rewards,done=dones,advantage=advantages,target=targets).items()}
        last_losses=None
        for epoch in range(int(c['epochs'])):
            if self.algorithm=='gradient_ppo':
                length=int(c['sequence_length'])
                if N%length:raise ValueError('Gradient PPO needs complete registered chronological sequences')
                order=self.rng.permutation(N//length)
                batches=[(sequence[:,None]*length+np.arange(length)[None,:]).reshape(-1)
                         for sequence in np.array_split(order,min(int(c['minibatches']),len(order))) if len(sequence)]
            else:
                order=self.rng.permutation(N)
                batches=[a for a in np.array_split(order,min(int(c['minibatches']),N)) if len(a)]
            for indices in batches:
                index=torch.as_tensor(indices,device=self.device,dtype=torch.long)
                s,latent=data['obs'][index],data['latent'][index]
                adv=data['advantage'][index]
                center=adv.mean();scale=adv.std(unbiased=False)+1e-8
                normalized=(adv-center)/scale
                live_logp=self.actor.log_probability(s,latent)
                if self.algorithm=='reflective_ppo':
                    next_index=(index+1).clamp_max(N-1)
                    active=((index<N-1)&(data['done'][index]==0)).to(adv.dtype)
                    next_adv=(data['advantage'][next_index]-center)/scale
                    next_logp=self.actor.log_probability(data['obs'][next_index],data['latent'][next_index])
                    policy_loss=-reflective_objective(live_logp,data['old_logp'][index],normalized,
                        next_logp,data['old_logp'][next_index],next_adv,active,
                        c['clip_epsilon'],c['next_epsilon'],c['reflective_beta']).mean()
                else:policy_loss=-clipped_objective(live_logp,data['old_logp'][index],normalized,c['clip_epsilon']).mean()
                actor_norm=self._step(self.actor_optimizer,self.actor.parameters(),policy_loss)
                self.actor_updates+=1
                if self.algorithm=='gradient_ppo':
                    L=int(c['sequence_length'])
                    sequence_reward=data['reward'][index].reshape(-1,L)
                    sequence_done=data['done'][index].reshape(-1,L)
                    if c['complete_transition_windows']:
                        # 32 REAL transitions plus their actual observed next states.
                        # No reward row is discarded as an artificial bootstrap row.
                        combined=self.critic(torch.cat([s,data['next_obs'][index]],dim=0))
                        values=combined[:len(index)].reshape(-1,L)
                        next_values=combined[len(index):].reshape(-1,L)
                        live_delta=lambda_advantages(sequence_reward,sequence_done,
                            values,next_values,c['gamma'],c['gae_lambda'])
                        h_value=self.h(s).reshape(-1,L)
                        td_values=values
                        covered=np.asarray(indices)
                    else:
                        # Explicit legacy diagnostic branch; never used by corrected runs.
                        values=self.critic(s).reshape(-1,L)
                        live_delta=lambda_advantages(sequence_reward[:,:-1],sequence_done[:,:-1],
                            values[:,:-1],values[:,1:],c['gamma'],c['gae_lambda'])
                        h_value=self.h(s).reshape(-1,L)[:,:-1]
                        td_values=values[:,:-1]
                        covered=np.asarray(indices).reshape(-1,L)[:,:-1].reshape(-1)
                    self.critic_terminal_positions+=int(np.count_nonzero(dones[covered]))
                    self.critic_nonzero_reward_positions+=int(np.count_nonzero(rewards[covered]))
                    # -Delta_w = -delta.detach()*gradV + h.detach()*grad(V+delta).
                    value_loss=(-live_delta.detach()*td_values+h_value.detach()*(td_values+live_delta)).mean()
                    h_loss=.5*(h_value-live_delta.detach()).square().mean()
                    h_loss=h_loss+.5*float(c['h_regularization'])*sum(p.square().sum() for p in self.h.parameters())
                    # Form both updates at the SAME old V/h parameters, then apply each.
                    self.critic_optimizer.zero_grad(set_to_none=True);self.h_optimizer.zero_grad(set_to_none=True)
                    value_loss.backward();h_loss.backward()
                    value_norm=nn.utils.clip_grad_norm_(self.critic.parameters(),c['max_grad_norm'])
                    h_norm=nn.utils.clip_grad_norm_(self.h.parameters(),c['max_grad_norm'])
                    if not (torch.isfinite(value_norm) and torch.isfinite(h_norm)):
                        raise FloatingPointError('Nonfinite TDRC(lambda) update')
                    self.critic_optimizer.step();self.h_optimizer.step();self.h_updates+=1
                    self.effective_td_positions+=int(live_delta.numel())
                    trace_error=live_delta.detach().square().mean().sqrt()
                else:
                    values=self.critic(s);target=data['target'][index]
                    if c['value_clip']:
                        clipped=data['old_value'][index]+(values-data['old_value'][index]).clamp(-c['clip_epsilon'],c['clip_epsilon'])
                        value_loss=.5*torch.maximum((values-target).square(),(clipped-target).square()).mean()
                    else:value_loss=.5*(values-target).square().mean()
                    value_norm=self._step(self.critic_optimizer,self.critic.parameters(),value_loss,clip=False)
                    h_loss=trace_error=torch.zeros((),device=self.device)
                self.updates+=1
                last_losses=torch.stack([policy_loss.detach(),value_loss.detach(),h_loss.detach(),trace_error,
                    actor_norm,value_norm,live_logp.detach().mean()])
        numbers=last_losses.cpu().tolist()
        if not all(math.isfinite(a) for a in numbers):raise FloatingPointError('Nonfinite rollout summary')
        self.last=dict(zip(['actor_loss','critic_loss','h_loss','td_lambda_rms','actor_gradient_norm',
                           'critic_gradient_norm','mean_latent_log_probability'],numbers))
        self.rollout_updates+=1;self.buffer=[]

    def finish(self):
        if self.pending is not None:raise RuntimeError('Unobserved final action')
        if self.buffer:self._update_rollout()
        return {}
    def stats(self):return dict(self.last)
    def accounting(self):
        return dict(environment_steps=self.environment_steps,updates=self.updates,actor_updates=self.actor_updates,
            reward_model_updates=0,h_updates=self.h_updates,rollout_updates=self.rollout_updates,
            effective_td_positions=self.effective_td_positions,algorithm=self.algorithm,
            critic_terminal_positions=self.critic_terminal_positions,
            critic_nonzero_reward_positions=self.critic_nonzero_reward_positions,
            observation_moment_count=self.observation_moments.count,
            return_moment_count=self.return_moments.count)
    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        path=directory/'model.pt'
        torch.save(dict(algorithm=self.algorithm,actor=self.actor.state_dict(),D=self.D,A=self.A,seed=self.seed,
            config=self.config,observation_moments=self.observation_moments.state(),
            accounting=self.accounting(),evaluation_only=True),path)
        (directory/'metadata.json').write_text(json.dumps(dict(algorithm=self.algorithm,config=self.config,
            saved_networks=['actor'],contains_optimizer=False,contains_replay=False,contains_critic=False,
            native_likelihood='raw unbounded Gaussian latent; fixed physical clip[-1,1]',
            architecture='generic public shared object mean (task adaptation); native constant logstd retained',
            interpretation='online development screen, not author-default numerical replication'),indent=2)+'\n')
        return path
    @classmethod
    def load(cls,directory,device='cuda'):
        state=torch.load(Path(directory)/'model.pt',map_location='cpu',weights_only=False)
        obj=cls.__new__(cls);obj.algorithm=state['algorithm'];obj.config=dict(state['config'])
        obj.config.setdefault('initial_total_std',1.)
        obj.D,obj.A,obj.seed=state['D'],state['A'],state['seed'];obj.device=torch.device(device)
        obj.actor=GaussianActor(obj.D,obj.A,obj.config['initial_total_std'])
        if obj.config['shared_mean']:obj.actor.attach_shared(obj.D,obj.A,obj.seed)
        obj.actor.load_state_dict(state['actor']);obj.actor.to(obj.device).eval()
        obj.observation_moments=RunningMoments((obj.D,));obj.observation_moments.restore(state['observation_moments'])
        obj.return_moments=RunningMoments();obj.return_moments.count=state['accounting']['return_moment_count']
        for key in ['environment_steps','updates','actor_updates','h_updates','rollout_updates','effective_td_positions',
                    'critic_terminal_positions','critic_nonzero_reward_positions']:
            setattr(obj,key,int(state['accounting'][key]))
        obj.evaluation_only=True;obj.pending=None;obj.cached_next=None;obj.last={};obj.buffer=[]
        return obj
