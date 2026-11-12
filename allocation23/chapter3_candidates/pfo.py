"""PFO (Moalla et al., NeurIPS2024), independent minimal Torch task port.

Pinned official models use TanhNormal and state-dependent NormalParamExtractor
scale. This port retains both, clipped PPO, official population-normalized GAE
advantages, and detached behavior PRE-activation anchors. Shared object/total
heads are a declared task architecture: each head's penultimate representation
feeds BOTH loc and scale. Equal-head weighting prevents100/120 objects from
diluting the total head. No SPO objective or state-independent sigma is used.
"""
from contextlib import contextmanager
from pathlib import Path
import json,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .spo import Candidate as CollectionBase, gae


def normalize_advantage(advantage):
    """Official PPOLoss: population variance, denominator clamp after sqrt."""
    centered=advantage-advantage.mean()
    return centered/centered.square().mean().sqrt().clamp_min(1e-6)


def safe_tanh(latent):
    eps=torch.finfo(latent.dtype).resolution
    return latent.tanh().clamp(-1.+eps,1.-eps)


def inverse_tanh(action):
    eps=torch.finfo(action.dtype).resolution
    a=action.clamp(-1.+eps,1.-eps)
    return .5*(torch.log1p(a)-torch.log1p(-a))


def tanh_log_prob_from_latent(latent,loc,scale):
    """Joint density of the executed tanh action; transform Jacobian included."""
    log_base=-.5*((latent-loc)/scale).square()-scale.log()-.5*math.log(2.*math.pi)
    log_jac=2.*(math.log(2.)-latent-F.softplus(-2.*latent))
    return (log_base-log_jac).sum(-1)


def positive_scale(raw):
    # NormalParamExtractor default biased_softplus_1.0: min_val=.01, then lb1e-4.
    shift=math.log(math.expm1(.99))
    return (F.softplus(raw+shift)+.01).clamp_min(1e-4)


def policy_and_feature_loss(ratio,advantage,new_object,old_object,new_total,old_total,epsilon=.2,coefficient=1.):
    policy=-torch.minimum(ratio*advantage,ratio.clamp(1-epsilon,1+epsilon)*advantage).mean()
    # Official mean over feature coordinates then batch; object axis averaged
    # within its head, followed by the explicit equal-head task adaptation.
    object_drift=(new_object-old_object.detach()).square().mean(-1).mean(-1)
    total_drift=(new_total-old_total.detach()).square().mean(-1)
    features=.5*(object_drift+total_drift).mean()
    return policy+coefficient*features,policy,features


class PFOActor(nn.Module):
    """Shared two-head policy; every head projects its features to loc AND scale."""
    def __init__(self,D,A):
        super().__init__();self.objects=A-1
        if D==self.objects*3+3:self.local_dim,self.global_dim=3,3
        elif D==self.objects*2+2:self.local_dim,self.global_dim=2,2
        else:raise ValueError('PFO compact public layout mismatch')
        self.local=nn.Sequential(nn.Linear(self.local_dim,64),nn.Tanh(),nn.Linear(64,32))
        context=32+self.global_dim
        self.object_head=nn.Sequential(nn.Linear(32+context,32),nn.Tanh(),nn.Linear(32,2))
        self.amount_head=nn.Sequential(nn.Linear(context,32),nn.Tanh(),nn.Linear(32,2))

    def parameters_and_features(self,obs):
        split=self.objects*self.local_dim
        local=obs[...,:split].reshape(*obs.shape[:-1],self.objects,self.local_dim)
        embedded=self.local(local);context=torch.cat((embedded.mean(-2),obs[...,split:]),-1)
        expanded=context.unsqueeze(-2).expand(*embedded.shape[:-1],context.shape[-1])
        object_pre=self.object_head[0](torch.cat((embedded,expanded),-1))
        total_pre=self.amount_head[0](context)
        objects=self.object_head[2](self.object_head[1](object_pre))
        total=self.amount_head[2](self.amount_head[1](total_pre))
        loc=torch.cat((total[...,0:1],objects[...,0]),-1)
        raw_scale=torch.cat((total[...,1:2],objects[...,1]),-1)
        return loc,positive_scale(raw_scale),object_pre,total_pre


class Candidate(CollectionBase):
    DEFAULTS={**CollectionBase.DEFAULTS,'rollout_steps':512,'total_steps':24576,
        'minibatches':8,'epochs':10,'learning_rate_decay':False,
        'gae_lambda':1.,'normalize_observation':False,'normalize_reward':False,
        'pfo_coefficient':1.,'feature_scope':'heads_penultimate',
        'policy_distribution':'tanh_state_dependent_scale'}

    def __init__(self,obs_dim,action_dim,seed,device='cuda',config=None):
        # Reuse only the validated serial collector/GAE/value initialization and
        # its private seeds, replacing the legacy actor and combined optimizer.
        requested={**self.DEFAULTS,**(config or {})}
        rollout,batches,total=(int(requested[k]) for k in ('rollout_steps','minibatches','total_steps'))
        if (rollout<=0 or batches<=0 or total<=0 or rollout%batches or
                rollout//batches<2 or total%(rollout//batches)):
            raise ValueError('PFO requires positive real batches divisible by the nominal minibatch size')
        # The inherited collector's initializer validates whole-rollout budgets.
        # It performs no clock-dependent work. Supply its validation ceiling,
        # then restore the exact real budget before any action or learning.
        # The SPO collector and all old complete-rollout profiles stay unchanged.
        initialization={**(config or {}),'total_steps':math.ceil(total/rollout)*rollout}
        super().__init__(obs_dim,action_dim,seed,device,initialization)
        self.config['total_steps']=total
        c=self.config
        if c['actor_architecture']!='shared_mean' or c['feature_scope']!='heads_penultimate' or c['policy_distribution']!='tanh_state_dependent_scale':
            raise ValueError('PFO registered shared mean/scale head architecture required')
        if c['normalize_observation'] or c['normalize_reward'] or c['reward_scale']!=1.:
            raise ValueError('Registered PFO task profile uses raw terminal reward and frozen raw public inputs')
        if not math.isfinite(c['pfo_coefficient']) or c['pfo_coefficient']<=0:raise ValueError('Positive finite PFO coefficient required')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed+230007);self.actor=PFOActor(self.D,self.A)
        self.actor.to(self.device)
        self.actor_parameters=list(self.actor.parameters());self.value_parameters=list(self.critic.parameters())
        self.parameters=self.actor_parameters+self.value_parameters
        self.actor_optimizer=torch.optim.Adam(self.actor_parameters,lr=c['learning_rate'],eps=c['adam_epsilon'])
        self.value_optimizer=torch.optim.Adam(self.value_parameters,lr=c['learning_rate'],eps=c['adam_epsilon'])
        del self.optimizer
        self.learn_generator=torch.Generator(device=self.device);self.learn_generator.manual_seed(self.seed+330007)
        self.feature_buffer=[];self.pending_features=None;self.anchored_transitions=0

    @torch.no_grad()
    def act(self,obs,deterministic=False):
        if deterministic:
            x=self._observation(obs,False)
            loc,_,_,_=self.actor.parameters_and_features(torch.as_tensor(x,device=self.device).unsqueeze(0))
            return safe_tanh(loc[0]).cpu().numpy().astype(np.float32)
        if self.evaluation_only:raise RuntimeError('PFO endpoint is evaluation-only')
        if self.pending is not None or self.pending_features is not None:raise RuntimeError('PFO preceding actual action must be observed first')
        raw=np.asarray(obs,np.float32)
        if self.cached_next is not None and np.array_equal(raw,self.cached_next[0]):x=self.cached_next[1].copy()
        else:x=self._observation(raw,True)
        self.cached_next=None;s=torch.as_tensor(x,device=self.device).unsqueeze(0)
        loc,scale,objects,total=self.actor.parameters_and_features(s)
        sample=loc+scale*torch.randn(loc.shape,generator=self.generator,device=self.device)
        action=safe_tanh(sample)
        # Store inverse of the ACTUALLY executed safe-tanh action. This agrees
        # with TanhNormal.log_prob(action), including finite-precision clipping.
        latent=inverse_tanh(action);logp=tanh_log_prob_from_latent(latent,loc,scale)
        value=self.critic(s).flatten()
        packed=torch.cat((latent[0],logp,value,loc[0],scale[0],action[0])).cpu().numpy()
        bounded=packed[3*self.A+2:4*self.A+2].copy()
        self.pending=(raw.copy(),x,packed[:self.A].copy(),float(packed[self.A]),float(packed[self.A+1]),bounded.copy(),packed[self.A+2:2*self.A+2].copy(),packed[2*self.A+2:3*self.A+2].copy())
        self.pending_features=(objects[0].detach().clone(),total[0].detach().clone())
        return bounded

    def observe(self,obs,action,native_reward,next_obs,done):
        if self.evaluation_only or self.pending is None or self.pending_features is None:raise RuntimeError('PFO requires a live sampled behavior transition')
        if not np.array_equal(np.asarray(obs,np.float32),self.pending[0]) or not np.array_equal(np.asarray(action,np.float32),self.pending[5]):raise ValueError('PFO state/action differs from actual behavior')
        if bool(done)!=(self.environment_steps%self.horizon==self.horizon-1) or not math.isfinite(float(native_reward)):raise ValueError('PFO actual done/reward contract differs')
        # Actual collection-time detached values, never recomputed after training.
        self.feature_buffer.append(self.pending_features);self.pending_features=None;self.anchored_transitions+=1
        return super().observe(obs,action,native_reward,next_obs,done)

    def _update(self):
        c=self.config;rows=self.buffer;N=len(rows)
        remaining=c['total_steps']-self.rollout_updates*c['rollout_steps']
        expected=min(c['rollout_steps'],remaining)
        if (N!=expected or N<=0 or len(self.feature_buffer)!=N or
                self.environment_steps!=self.rollout_updates*c['rollout_steps']+N):
            raise RuntimeError('PFO real rollout or behavior anchors differ from the registered budget')
        clocks=math.ceil(c['total_steps']/c['rollout_steps'])
        if self.rollout_updates>=clocks:raise RuntimeError('PFO registered clock exhausted')
        obs=np.stack([r[0] for r in rows]);latent=np.stack([r[1] for r in rows]);oldp=np.asarray([r[2] for r in rows],np.float32)
        oldv=np.asarray([r[3] for r in rows],np.float32);reward=np.asarray([r[4] for r in rows],np.float32);done=np.asarray([r[5] for r in rows],np.float32)
        with torch.no_grad():last=0. if done[-1] else float(self.critic(torch.as_tensor(rows[-1][6],device=self.device).unsqueeze(0)).flatten()[0].cpu())
        advantages=gae(reward,done,oldv,last,c['gamma'],c['gae_lambda']);targets=oldv+advantages
        old_mu=np.stack([r[7] for r in rows]);old_sigma=np.stack([r[8] for r in rows])
        data=[torch.as_tensor(x,device=self.device) for x in [obs,latent,oldp,advantages,targets,old_mu,old_sigma]]
        old_objects=torch.stack([x[0] for x in self.feature_buffer]).detach();old_total=torch.stack([x[1] for x in self.feature_buffer]).detach()
        if old_objects.requires_grad or old_total.requires_grad:raise RuntimeError('Behavior anchors must be detached')
        lr=c['learning_rate']*(1.-self.rollout_updates/clocks) if c['learning_rate_decay'] else c['learning_rate']
        for opt in [self.actor_optimizer,self.value_optimizer]:
            for group in opt.param_groups:group['lr']=lr
        # Keep the normal minibatch size on the last real partial rollout.
        # E.g. final256/nominal64 gives four minibatches, never padding/dropping.
        indices=np.arange(N);B=c['rollout_steps']//int(c['minibatches']);diagnostics=[];labels=[]
        if N%B:raise RuntimeError('PFO final real rollout must partition into nominal minibatches')
        for epoch in range(int(c['epochs'])):
            self.shuffle.shuffle(indices)
            for first in range(0,N,B):
                ix=torch.as_tensor(indices[first:first+B],device=self.device,dtype=torch.long)
                s,u,op,adv,target,omu,osigma=[x[ix] for x in data];adv=normalize_advantage(adv)
                loc,scale,objects,total=self.actor.parameters_and_features(s)
                logp=tanh_log_prob_from_latent(u,loc,scale);ratio=(logp-op).exp()
                actor,policy,drift=policy_and_feature_loss(ratio,adv,objects,old_objects[ix],total,old_total[ix],c['epsilon'],c['pfo_coefficient'])
                # Official TanhNormal entropy fallback: one current-policy sample.
                sample=(loc+scale*torch.randn(loc.shape,generator=self.learn_generator,device=self.device)).detach()
                sampled_u=inverse_tanh(safe_tanh(sample))
                entropy=-tanh_log_prob_from_latent(sampled_u,loc,scale).mean()
                actor=actor-c['entropy_coefficient']*entropy
                value=c['value_coefficient']*(self.critic(s).flatten()-target).square().mean()
                gaussian_kl=(torch.log(scale/osigma)+(osigma.square()+(omu-loc).square())/(2*scale.square())-.5).sum(-1).mean()
                self.actor_optimizer.zero_grad(set_to_none=True);actor.backward()
                actor_norm=nn.utils.clip_grad_norm_(self.actor_parameters,c['max_grad_norm'],error_if_nonfinite=True);self.actor_optimizer.step()
                self.value_optimizer.zero_grad(set_to_none=True);value.backward()
                value_norm=nn.utils.clip_grad_norm_(self.value_parameters,c['max_grad_norm'],error_if_nonfinite=True);self.value_optimizer.step()
                self.updates+=1
                diagnostics.append(torch.stack([policy.detach(),value.detach(),drift.detach(),entropy.detach(),actor_norm.detach(),value_norm.detach(),ratio.detach().max(),(ratio.detach()-1).abs().mean(),gaussian_kl.detach()]))
                labels.append((self.updates,epoch,first//B))
        numbers=torch.stack(diagnostics).cpu().numpy()
        if not np.isfinite(numbers).all():raise FloatingPointError('Nonfinite PFO update')
        names=['policy_loss','value_loss','feature_loss','entropy_MC','actor_grad_norm','value_grad_norm','ratio_max','ratio_deviation','gaussian_analytic_KL_pre_step']
        for label,vals in zip(labels,numbers):self.optimizer_trace.append(dict(update=label[0],rollout=self.rollout_updates+1,epoch=label[1],minibatch=label[2],batch_size=B,learning_rate=lr,**dict(zip(names,map(float,vals)))))
        self.rollout_updates+=1;self.optimized_positions+=N*int(c['epochs']);self.optimizer_terminal_positions+=int(done.sum())*int(c['epochs'])
        self.last=dict(zip(names,map(float,numbers[-1])));self.last.update(actor_updates=self.updates,value_updates=self.updates,rollout_updates=self.rollout_updates,feature_anchor_positions=N)
        self.buffer=[];self.feature_buffer=[]

    def finish(self):
        if self.pending is not None or self.pending_features is not None:raise RuntimeError('Unmatched PFO action or anchors')
        if self.environment_steps!=self.config['total_steps']:raise RuntimeError('PFO fixed endpoint requires its exact registered real learning budget')
        if self.buffer:self._update()
        if self.feature_buffer:raise RuntimeError('Unmatched PFO anchors')
        return super().finish()

    def accounting(self):
        return dict(super().accounting(),feature_anchor_buffer_positions=len(self.feature_buffer),pending_feature_anchor=int(self.pending_features is not None),anchored_transitions=self.anchored_transitions,Adam_step_calls=2*self.updates)

    def save(self,directory):
        if self.pending_features is not None or self.feature_buffer:raise RuntimeError('PFO fixed endpoint requires a completed rollout')
        result=super().save(directory)
        (Path(directory)/'metadata.json').write_text(json.dumps(dict(method='pfo',paper='No Representation, No Trust (NeurIPS2024)',config=self.config,
            evaluation_only=True,optimizer_or_replay_saved=False,critic_saved=False,distribution='TanhNormal with state-dependent scale',
            feature_scope='actual two shared mean+scale head penultimate preactivations; equal head weighting'),indent=2)+'\n')
        return result

    @classmethod
    def load(cls,directory,device='cuda'):
        meta=json.loads((Path(directory)/'metadata.json').read_text())
        if meta['method']!='pfo':raise ValueError('Not a PFO endpoint')
        result=super().load(directory,device);result.anchored_transitions=result.environment_steps
        result.actor.eval();result.critic.eval();return result
