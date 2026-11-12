"""Native-PROPS Agent API with a matched actor/critic representation2x2.

No environment, checkpoint, decoder or reward model is imported. Set modules
are copied from fresh A_FULL/adaptive actor and Value templates supplied by the
new full-spend bridge. Every cell has a linear unbounded Gaussian mean, zero
initial mean/value and the same learnable diagonal logstd initialization.
"""
from __future__ import annotations
import copy,math
from dataclasses import dataclass
import torch
from torch import nn
from torch.distributions import Normal

@dataclass(frozen=True)
class FlatSchema:
    groups:int
    local_dim:int
    global_dim:int
    spatial_dim:int
    @property
    def observation_dim(self):return self.groups*self.local_dim+self.global_dim+2
    @property
    def action_dim(self):return 1+self.spatial_dim
    def unpack(self,x):
        if x.ndim!=2 or x.shape[1]!=self.observation_dim:raise ValueError('Shared public flat observation schema required')
        stop=self.groups*self.local_dim
        return x[:,:stop].reshape(-1,self.groups,self.local_dim),x[:,stop:stop+self.global_dim],x[:,-2:]

def linear(layer,gain):
    nn.init.orthogonal_(layer.weight,gain);nn.init.zeros_(layer.bias);return layer

class FlatMean(nn.Sequential):
    def __init__(self,schema,seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31000+int(seed))
            super().__init__(linear(nn.Linear(schema.observation_dim,64),math.sqrt(2)),nn.Tanh(),linear(nn.Linear(64,64),math.sqrt(2)),nn.Tanh(),nn.Linear(64,schema.action_dim))
            nn.init.zeros_(self[-1].weight);nn.init.zeros_(self[-1].bias)

class SetMean(nn.Module):
    def __init__(self,schema,actor_template):
        super().__init__();self.schema=schema
        if actor_template.study_variant!='A_FULL' or actor_template.temporal_mode!='adaptive' or actor_template.spatial.event_dim!=schema.spatial_dim:raise ValueError('Fresh canonical A_FULL adaptive actor template required')
        self.object_net=copy.deepcopy(actor_template.spatial.object_net)
        self.context_net=copy.deepcopy(actor_template.spatial.context_net)
        self.spatial_head=copy.deepcopy(actor_template.spatial.head)
        self.quota_head=copy.deepcopy(actor_template.quota_head)
        for head in [self.spatial_head,self.quota_head[-1]]:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    def forward(self,x):
        local,global_,ledger=self.schema.unpack(x)
        pooled=self.object_net(local).mean(1)
        context=self.context_net(torch.cat((pooled,global_),-1))
        quota=self.quota_head(torch.cat((context,global_[:,:1],ledger),-1))
        return torch.cat((quota,self.spatial_head(context)),-1)

class FlatValue(nn.Sequential):
    def __init__(self,schema,seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(41000+int(seed))
            super().__init__(linear(nn.Linear(schema.observation_dim,64),math.sqrt(2)),nn.Tanh(),linear(nn.Linear(64,64),math.sqrt(2)),nn.Tanh(),nn.Linear(64,1))
            nn.init.zeros_(self[-1].weight);nn.init.zeros_(self[-1].bias)

class SetValue(nn.Module):
    def __init__(self,schema,value_template):
        super().__init__();self.schema=schema
        self.encoder=copy.deepcopy(value_template.encoder)
        self.head=copy.deepcopy(value_template.head)
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x):
        local,global_,_ledger=self.schema.unpack(x)
        _encoded,context=self.encoder({'local':local,'global':global_})
        return self.head(context)

class MatchedAgent(nn.Module):
    """Preserves author six/five-tuple APIs and ordinary joint Gaussian density."""
    def __init__(self,spec,seed,actor_kind,critic_kind,*,actor_template=None,value_template=None,dtype=torch.float32):
        super().__init__()
        if actor_kind not in ['flat','set'] or critic_kind not in ['flat','set']:raise ValueError('Registered matched2x2 cell required')
        self.schema=FlatSchema(spec.groups,spec.local_dim,spec.global_dim,6*spec.local_dim)
        self.actor_kind,self.critic_kind,self.seed=actor_kind,critic_kind,int(seed)
        self.actor_mean=(FlatMean(self.schema,seed) if actor_kind=='flat' else SetMean(self.schema,actor_template))
        self.critic=(FlatValue(self.schema,seed) if critic_kind=='flat' else SetValue(self.schema,value_template))
        sigma=torch.tensor([.8]+[1/math.sqrt(self.schema.spatial_dim)]*self.schema.spatial_dim,dtype=dtype)
        self.actor_logstd=nn.Parameter(sigma.log()[None]);self.to(dtype=dtype)
    def _input(self,x):return x.to(device=self.actor_logstd.device,dtype=self.actor_logstd.dtype)
    def _distribution(self,x,clamp=False):
        x=self._input(x);mean=self.actor_mean(x);logstd=self.actor_logstd.expand_as(mean)
        if clamp:logstd=logstd.clamp(-3,1)
        std=logstd.exp();return x,mean,std,Normal(mean,std)
    def get_value(self,x):return self.critic(self._input(x))
    def get_action_and_value(self,x,action=None):
        x,mean,std,dist=self._distribution(x)
        action=dist.sample() if action is None else action.to(device=mean.device,dtype=mean.dtype)
        return action,mean,std,dist.log_prob(action).sum(1),dist.entropy().sum(1),self.critic(x)
    def get_action_and_info(self,x,action=None,clamp=False):
        _x,mean,std,dist=self._distribution(x,clamp)
        action=dist.sample() if action is None else action.to(device=mean.device,dtype=mean.dtype)
        return action,mean,std,dist.log_prob(action).sum(1),dist.entropy().sum(1)
    def get_action(self,x,noise=False):
        _x,mean,_std,dist=self._distribution(x);return dist.sample() if noise else mean
    def sample_actions(self,x):return self.get_action(x,noise=True)
    def sample_actions_unif(self,x):
        with torch.no_grad():
            _x,mean,std,_dist=self._distribution(x)
            return torch.distributions.Uniform((mean-3*std).clamp(-1,1),(mean+3*std).clamp(-1,1)).sample()
    def metadata(self):
        return {'schema':'native-PROPS-matched-actor-critic-representation2x2-v1','actor_representation':self.actor_kind,'critic_representation':self.critic_kind,'observation_dim':self.schema.observation_dim,'action_dim':self.schema.action_dim,'mean_output':'unbounded linear in every cell','initial_mean':0,'initial_value':0,'initial_quota_sigma':.8,'initial_spatial_sigma':1/math.sqrt(self.schema.spatial_dim),'sigma_learning':'all diagonal logstd parameters learnable, identical native PROPS rule; not canonical A fixed-spatial-sigma dynamics','dtype':str(self.actor_logstd.dtype),'physical_decoder':'outside Agent; supplied only by common full-spend environment','original_native_anchor':'separate flat/Tanh/sigma1; not one of these matched cells','no_trained_checkpoint_initialization':True}

def create_agent(spec,seed,actor_kind,critic_kind,*,actor_factory,value_factory,dtype=torch.float32):
    """Factories must construct fresh modules from the new release, never load weights."""
    actor=actor_factory(spec,seed) if actor_kind=='set' else None
    value=value_factory(spec,seed) if critic_kind=='set' else None
    return MatchedAgent(spec,seed,actor_kind,critic_kind,actor_template=actor,value_template=value,dtype=dtype)
