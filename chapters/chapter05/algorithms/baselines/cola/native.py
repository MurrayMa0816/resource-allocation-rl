"""COLA (NeurIPS 2025), author-code based continuous MORL adapter.

MIT source: vendor/cola/agent.py:632-708, 904-954, 976-1035, 1057-1118;
vendor/cola/model.py:363-385.  Both indispensable mechanisms are retained:
OADM temporal latent consistency and conflict-gated preservation of old Q.
Auxiliary dynamics learning consumes observed transitions only; it creates no
extra reward, privileged physical input, or simulated interaction.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .sac_base import (SACBase, GaussianActor, VectorCritic, as_tensor,
                       conservative_vector, finite, freeze, polyak)


class LatentEncoder(nn.Module):
    """Objective-agnostic E(s), F(E(s),a), as in the authors' Latent_Encoder."""
    def __init__(self, states, actions, latent, hidden=256, normalize=True):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(states,hidden),nn.ELU(),nn.Linear(hidden,latent))
        self.dynamics=nn.Sequential(nn.Linear(latent+actions,hidden),nn.ELU(),
            nn.Linear(hidden,hidden),nn.ELU(),nn.Linear(hidden,latent))
        self.normalize=bool(normalize)

    def encode(self, states):
        z=self.encoder(states)
        return z/(z.abs().mean(-1,keepdim=True).clamp_min(1e-8)) if self.normalize else z

    def predict(self,z,actions):
        return self.dynamics(torch.cat([z,actions],-1))


class Agent(SACBase):
    name='cola'

    def __init__(self,obs_dim,action_dim,seed=0,device='cpu',config=None):
        super().__init__(obs_dim,action_dim,seed,device,config)
        c=self.config
        self.latent_dim=int(c.get('cola_latent_dim',c.get('latent_dim',50)))
        self.policy_hidden=int(c.get('policy_hidden_dim',c.get('hidden_dim',128)))
        # Lock one author-published recipe before seeing this benchmark's scores.
        # README Quick Start uses .001; run.sh also contains task-specific 1.0
        # profiles.  We use the README recipe uniformly in both environments.
        self.regular_alpha=float(c.get('cola_regular_alpha',c.get('regular_alpha',.001)))
        self.regular_bar=float(c.get('cola_regular_bar',c.get('regular_bar',.25)))
        self.preference_candidates=int(c.get('cola_preference_candidates',1))
        self.old_q_frequency=int(c.get('cola_old_q_update_frequency',c.get('old_Q_update_freq',1)))
        self.dynamic_coefficient=float(c.get('cola_dynamic_coefficient',1.))
        self.normalize_latent=bool(c.get('cola_normalize_latent',True))
        if self.latent_dim<1 or self.preference_candidates<1 or self.old_q_frequency<1:
            raise ValueError('Invalid COLA mechanism configuration')
        if self.regular_alpha<0 or self.dynamic_coefficient<=0:
            raise ValueError('COLA regularization nonnegative, dynamics loss positive required')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed+23001)
            self.encoder=LatentEncoder(obs_dim,action_dim,self.latent_dim,self.hidden,self.normalize_latent).to(self.device)
            self.actor=GaussianActor(obs_dim+self.latent_dim+2,action_dim,self.policy_hidden).to(self.device)
            # Author run.sh: Critic_use_both, Critic_use_s, Critic_use_a enabled.
            self.critic=VectorCritic(2*self.latent_dim+obs_dim+action_dim+2,self.hidden).to(self.device)
        self.encoder_target=copy.deepcopy(self.encoder);freeze(self.encoder_target)
        self.critic_target=copy.deepcopy(self.critic);freeze(self.critic_target)
        self.old_critic=copy.deepcopy(self.critic);freeze(self.old_critic)
        self.encoder_optimizer=torch.optim.Adam(self.encoder.parameters(),lr=self.lr)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=self.lr)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=self.lr)
        self.oadm_updates=0
        self.cor_trigger_count=0
        self.cor_nonzero_penalty_count=0
        self.cor_gradient_conflict_count=0

    def actor_input(self,obs,preferences,encoder=None):
        enc=self.encoder if encoder is None else encoder
        return torch.cat([enc.encode(obs),obs,preferences],-1)

    def critic_input(self,obs,actions,preferences,encoder=None,detach=False):
        enc=self.encoder if encoder is None else encoder
        if detach:
            # The original result detached both encoder outputs immediately.
            # Avoid constructing that discarded graph in the first place.
            with torch.no_grad():
                z=enc.encode(obs)
                za=enc.predict(z,actions)
        else:
            z=enc.encode(obs)
            za=enc.predict(z,actions)
        return torch.cat([z,za,obs,actions,preferences],-1)

    @torch.no_grad()
    def act(self,obs,prefs,deterministic=False):
        states,preferences=as_tensor(obs,self.device),as_tensor(prefs,self.device)
        if states.ndim==1:states=states[None]
        if preferences.ndim==1:preferences=preferences[None].expand(len(states),-1)
        assert states.shape[1]==self.obs_dim and preferences.shape==(len(states),2)
        actions=self.actor.actions(self.actor_input(states,preferences),self.action_generator,deterministic)
        finite(actions,'COLA action')
        return actions.cpu().numpy()

    @torch.no_grad()
    def target(self,next_states,preference,rewards,dones,next_latent=None):
        z=self.encoder_target.encode(next_states) if next_latent is None else next_latent
        actions,logp=self.actor.sample(torch.cat([z,next_states,preference],-1),self.learn_generator)
        za=self.encoder_target.predict(z,actions)
        q1,q2=self.critic_target(torch.cat([z,za,next_states,actions,preference],-1))
        q=conservative_vector(q1,q2,preference)-self.alpha*logp
        return rewards+(1-dones)*self.gamma*q

    @staticmethod
    def stiffness(loss,other_loss,network):
        # Official code compares gradients of the final Linear weight+bias.
        parameters=list(network.parameters())[-2:]
        a=torch.autograd.grad(loss,parameters,retain_graph=True)
        b=torch.autograd.grad(other_loss,parameters,retain_graph=True)
        aa=torch.cat([x.reshape(-1) for x in a]).detach()
        bb=torch.cat([x.reshape(-1) for x in b]).detach()
        return F.cosine_similarity(aa,bb,dim=0,eps=1e-8).detach()

    def update(self,batch):
        batch=self.batch(batch)
        states,actions,next_states=batch['obs'],batch['actions'],batch['next_obs']
        b=len(states)
        current=self.random_preference()
        preference=as_tensor(current,self.device)[None].expand(b,-1)
        other=as_tensor(self.random_preference(exclude=current),self.device)[None].expand(b,-1)

        # OADM: no rewards, preferences or hidden model quantities enter this loss.
        latent=self.encoder.encode(states)
        with torch.no_grad():
            next_latent=self.encoder_target.encode(next_states)
            # Bellman labels may be n-step; OADM predicts exactly one observed
            # action, so its supervised target must remain the actual s[t+1].
            oadm_next_latent=next_latent
            if getattr(self, 'oadm_one_step_target', False):
                oadm_next_states=batch['oadm_next_obs']
                if oadm_next_states is not next_states:
                    oadm_next_latent=self.encoder_target.encode(oadm_next_states)
        dynamic_loss=F.mse_loss(self.encoder.predict(latent,actions),oadm_next_latent)
        finite(dynamic_loss,'OADM loss')
        self.encoder_optimizer.zero_grad(set_to_none=True)
        (self.dynamic_coefficient*dynamic_loss).backward()
        for parameter in self.encoder.parameters():
            if parameter.grad is not None:finite(parameter.grad,'OADM gradient')
        self.encoder_optimizer.step();self.oadm_updates+=1

        # Critic targets use the target OADM; critic inputs are detached from E/F.
        target=self.target(next_states,preference,batch['rewards'],batch['dones'],next_latent)
        other_target=self.target(next_states,other,batch['rewards'],batch['dones'],next_latent)
        features=self.critic_input(states,actions,preference,detach=True)
        # Preferences enter after E/F, so their state-action features coincide.
        other_features=torch.cat([features[:,:-2],other],-1)
        q1,q2=self.critic(features)
        oq1,oq2=self.critic(other_features)
        loss1=((q1-target).square()*preference).sum(-1).mean()
        loss2=((q2-target).square()*preference).sum(-1).mean()
        other_loss1=((oq1-other_target).square()*other).sum(-1).mean()
        other_loss2=((oq2-other_target).square()*other).sum(-1).mean()
        cos1=self.stiffness(loss1,other_loss1,self.critic.q1)
        cos2=self.stiffness(loss2,other_loss2,self.critic.q2)
        with torch.no_grad():
            old1,old2=self.old_critic(other_features)
            # The old-Q outputs are now materialized.  Copy the still-unmodified
            # current critic directly, preserving the exact one-update lag and
            # frequency without a temporary full clone/state-dict load later.
            if (self.updates+1)%self.old_q_frequency==0:
                self.old_critic.load_state_dict(self.critic.state_dict())
        regular1=F.mse_loss(oq1,old1);regular2=F.mse_loss(oq2,old2)
        coefficient1=(self.regular_bar-cos1).clamp_min(0)*self.regular_alpha
        coefficient2=(self.regular_bar-cos2).clamp_min(0)*self.regular_alpha
        penalty=coefficient1*regular1+coefficient2*regular2
        critic_loss=loss1+loss2+penalty
        finite(critic_loss,'COLA critic loss')
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        for parameter in self.critic.parameters():
            if parameter.grad is not None:finite(parameter.grad,'COLA critic gradient')
        self.critic_optimizer.step()
        triggered=bool((coefficient1+coefficient2)>0)
        self.cor_trigger_count+=int(triggered)
        self.cor_nonzero_penalty_count+=int(float(penalty.detach())>0)
        self.cor_gradient_conflict_count+=int(bool((cos1<0)|(cos2<0)))

        # Envelope-SAC policy update; preserve gradients through F(z,a) to a.
        freeze(self.encoder);freeze(self.critic)
        z=features[:,:self.latent_dim]
        preferences=[preference]
        preferences += [as_tensor(self.random_preference(),self.device)[None].expand(b,-1)
                        for _ in range(self.preference_candidates-1)]
        values=[];original_logp=None
        for i,candidate in enumerate(preferences):
            actor_input=torch.cat([z,states,candidate],-1)
            sampled,logp=self.actor.sample(actor_input,self.learn_generator)
            if i==0:original_logp=logp
            za=self.encoder.predict(z,sampled)
            q1,q2=self.critic(torch.cat([z,za,states,sampled,preference],-1))
            scalar=torch.minimum((q1*preference).sum(-1),(q2*preference).sum(-1))
            values.append(scalar-self.alpha*logp.squeeze(-1))
        actor_loss=-torch.stack(values,0).max(0).values.mean()
        finite(actor_loss,'COLA actor loss')
        self.actor_optimizer.zero_grad(set_to_none=True);actor_loss.backward()
        for parameter in self.actor.parameters():
            if parameter.grad is not None:finite(parameter.grad,'COLA actor gradient')
        self.actor_optimizer.step()
        freeze(self.encoder,False);freeze(self.critic,False)
        alpha_loss=self.update_alpha(original_logp)
        polyak(self.encoder_target,self.encoder,self.tau)
        polyak(self.critic_target,self.critic,self.tau)
        self.updates+=1
        return dict(mechanism=self.name,updates=self.updates,actor_loss=float(actor_loss.detach()),
            critic_loss=float(critic_loss.detach()),alpha=float(self.alpha),alpha_loss=alpha_loss,
            oadm_loss=float(dynamic_loss.detach()),oadm_updates=self.oadm_updates,
            cor_penalty=float(penalty.detach()),cor_coefficient_max=float(torch.maximum(coefficient1,coefficient2)),
            cor_cosine_q1=float(cos1),cor_cosine_q2=float(cos2),
            cor_trigger_count=self.cor_trigger_count,cor_nonzero_penalty_count=self.cor_nonzero_penalty_count,
            cor_gradient_conflict_count=self.cor_gradient_conflict_count)

    def metadata(self):
        m=super().metadata()
        m.update(paper='COLA: Towards Efficient Multi-Objective Reinforcement Learning with Conflict Objective Regularization in Latent Space',
            venue='NeurIPS2025',source='https://github.com/yeshenpy/COLA',license='MIT',
            core=['objective-agnostic latent temporal dynamics consistency',
                  'cross-preference final-layer gradient cosine conflict gate',
                  'conflict-weighted preservation of previous critic on other preferences'],
            latent_dim=getattr(self,'latent_dim',50),regular_alpha=getattr(self,'regular_alpha',.001),
            regular_bar=getattr(self,'regular_bar',.25),
            actor_features='current latent state + raw public state + preference',
            critic_features='latent state + latent state-action prediction + raw state + action + preference',
            author_configuration='README Quick Start regular_alpha=.001 and regular_bar=.25 selected uniformly before benchmark outcomes; latent actor/raw skips/both latent critic views, Q snapshot buffer0',
            alternative_author_profiles='run.sh includes task-specific regular_alpha1.0, latent50/use_avg variants; not selected using target benchmark performance',
            no_synthetic_interactions=True,no_intermediate_reward_created=True,
            conflict_gate_layer='last critic Linear layer, matching original gradient slice')
        return m

    def state_dict(self):
        p=super().state_dict()
        p.update(encoder=self.encoder.state_dict(),encoder_target=self.encoder_target.state_dict(),
            old_critic=self.old_critic.state_dict(),oadm_updates=self.oadm_updates,
            cor_trigger_count=self.cor_trigger_count,cor_nonzero_penalty_count=self.cor_nonzero_penalty_count,
            cor_gradient_conflict_count=self.cor_gradient_conflict_count)
        return p

    def load_state_dict(self,payload):
        super().load_state_dict(payload)
        self.encoder.load_state_dict(payload['encoder'])
        self.encoder_target.load_state_dict(payload['encoder_target'])
        self.old_critic.load_state_dict(payload['old_critic'])
        for k in ('oadm_updates','cor_trigger_count','cor_nonzero_penalty_count','cor_gradient_conflict_count'):
            setattr(self,k,int(payload.get(k,0)))
