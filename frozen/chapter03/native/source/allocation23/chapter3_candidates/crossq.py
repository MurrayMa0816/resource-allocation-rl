"""CrossQ: explicit PyTorch port of the pinned MIT author's JAX update.

Joint current/next batch renormalization, no target network, twin wide critics,
delayed actor and entropy updates. Finite-horizon/unit/warmup changes are config.
Original reference and MIT license are preserved in ./reference.
"""
from pathlib import Path
import copy,json,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Explicit generic public-object mean adaptation, separate from paper mechanism.
# This exact module belongs to the collector-verified frozen native source closure.
from .shared_mean import SharedObjectMean


def _attach_shared_mean(actor, obs_dim, action_dim, seed, device):
    """Actor-mean-only task adaptation; preserve every base RNG initialization."""
    # fork_rng with an empty device list protects CPU state without touching CUDA.
    # A CPU-local Generator seeds only this new CPU module, never all devices.
    with torch.random.fork_rng(devices=[]):
        local_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 930000)
        torch.set_rng_state(local_generator.get_state())
        head = SharedObjectMean(obs_dim, action_dim)
    actor.shared_mean = head.to(device)


def dense(inp,out):
    layer=nn.Linear(inp,out)
    # Flax Dense defaults to LeCun truncated-normal; framework RNGs differ.
    with torch.no_grad():
        nn.init.trunc_normal_(layer.weight,std=1.,a=-2.,b=2.)
        layer.weight.mul_(1/math.sqrt(inp)/.87962566103423978)
        layer.bias.zero_()
    return layer


class BatchRenorm(nn.Module):
    """Pinned population-variance EMA and literal pre-normalization correction."""
    def __init__(self,size,momentum=.99,eps=.001,warmup=100000):
        super().__init__();self.momentum,self.eps,self.warmup=momentum,eps,int(warmup)
        self.weight=nn.Parameter(torch.ones(size));self.bias=nn.Parameter(torch.zeros(size))
        self.register_buffer('mean',torch.zeros(size));self.register_buffer('var',torch.ones(size))
        self.calls=0
    def get_extra_state(self):return dict(calls=self.calls)
    def set_extra_state(self,state):self.calls=int(state['calls'])
    def forward(self,x):
        if self.training:
            mean=x.mean(0)
            var=(x.square().mean(0)-mean.square()).clamp_min(0.)
            custom_mean,custom_var=mean,var
            if self.calls>=self.warmup:
                r=(torch.sqrt(var+self.eps)/torch.sqrt(self.var+self.eps)).detach().clamp(1/3,3)
                d=((mean-self.mean)/torch.sqrt(self.var+self.eps)).detach().clamp(-5,5)
                custom_var=var/r.square()
                # Literal author implementation: sqrt(var), not sqrt(var+eps).
                custom_mean=mean-d*torch.sqrt(var)/r
            with torch.no_grad():
                self.mean.mul_(self.momentum).add_(mean,alpha=1-self.momentum)
                self.var.mul_(self.momentum).add_(var,alpha=1-self.momentum)
            self.calls+=1
        else:custom_mean,custom_var=self.mean,self.var
        return (x-custom_mean)*torch.rsqrt(custom_var+self.eps)*self.weight+self.bias


class Actor(nn.Module):
    def __init__(self,D,A,width=256,momentum=.99,brn_warmup=100000):
        super().__init__();self.D,self.A=D,A
        self.bn0=BatchRenorm(D,momentum,warmup=brn_warmup)
        self.fc1=dense(D,width);self.bn1=BatchRenorm(width,momentum,warmup=brn_warmup)
        self.fc2=dense(width,width);self.bn2=BatchRenorm(width,momentum,warmup=brn_warmup)
        self.mu=dense(width,A);self.logstd=dense(width,A)
    def forward(self,x):
        raw_public_obs=x
        x=self.bn0(x);x=self.bn1(F.relu(self.fc1(x)));x=self.bn2(F.relu(self.fc2(x)))
        mean=self.shared_mean(raw_public_obs) if hasattr(self,'shared_mean') else self.mu(x)
        return mean,self.logstd(x).clamp(-20,2)
    def sample(self,x,deterministic=False):
        mean,logstd=self(x)
        if deterministic:return torch.tanh(mean),None
        raw=mean+logstd.exp()*torch.randn_like(mean)
        logp=(-.5*((raw-mean)*(-logstd).exp()).square()-logstd-.5*math.log(2*math.pi)
              -2*(math.log(2)-raw-F.softplus(-2*raw))).sum(-1,keepdim=True)
        return torch.tanh(raw),logp


class QNet(nn.Module):
    def __init__(self,D,A,width=2048,momentum=.99,brn_warmup=100000):
        super().__init__();self.bn0=BatchRenorm(D+A,momentum,warmup=brn_warmup)
        self.fc1=dense(D+A,width);self.bn1=BatchRenorm(width,momentum,warmup=brn_warmup)
        self.fc2=dense(width,width);self.bn2=BatchRenorm(width,momentum,warmup=brn_warmup)
        self.out=dense(width,1)
    def forward(self,s,a):
        x=self.bn0(torch.cat([s,a],-1));x=self.bn1(F.relu(self.fc1(x)))
        return self.out(self.bn2(F.relu(self.fc2(x))))


class Replay:
    def __init__(self,D,A,capacity,seed):
        self.data=np.empty((int(capacity),2*D+A+2),np.float32)
        self.D,self.A,self.capacity=D,A,int(capacity);self.index=self.size=0
        self.rng=np.random.default_rng(seed)
    def add(self,s,a,r,n,d):
        self.data[self.index]=np.concatenate([s,a,[r],n,[d]])
        self.index=(self.index+1)%self.capacity;self.size=min(self.capacity,self.size+1)
    def sample(self,B,device):
        x=torch.from_numpy(self.data[self.rng.integers(self.size,size=B)]).to(device)
        D,A=self.D,self.A
        return x[:,:D],x[:,D:D+A],x[:,D+A:D+A+1],x[:,D+A+1:2*D+A+1],x[:,-1:]


class Candidate:
    DEFAULTS=dict(actor_width=256,critic_width=2048,batch_size=256,
        random_action_steps=512,learning_starts=512,reward_scale=100.,gamma=1.,
        replay_capacity=65536,utd=1,actor_delay=3,lr=.001,temp_lr=.001,
        init_alpha=1.,target_entropy_factor=1.,momentum=.99,brn_warmup=100000,
        adam_beta1=.5,adam_beta2=.999,metric_interval=64,actor_architecture='flat')
    def __init__(self,obs_dim,action_dim,seed,device='cuda',config=None):
        self.config={**self.DEFAULTS,**(config or {})}
        allowed=set(self.DEFAULTS)|{'horizon'}
        if set(self.config)-allowed:raise ValueError('Unknown CrossQ config: '+str(set(self.config)-allowed))
        c=self.config
        if c['actor_architecture'] not in ('flat','shared_mean'):raise ValueError('Unknown actor architecture')
        if c['gamma'] not in (1.0,0.99) or c['utd']<1 or c['actor_delay']<1:raise ValueError('Invalid registered task/config')
        self.D,self.A,self.seed=int(obs_dim),int(action_dim),int(seed);self.device=torch.device(device)
        torch.manual_seed(seed);self.rng=np.random.default_rng(seed)
        self.actor=Actor(self.D,self.A,c['actor_width'],c['momentum'],c['brn_warmup']).to(self.device)
        self.critics=nn.ModuleList([QNet(self.D,self.A,c['critic_width'],c['momentum'],c['brn_warmup']) for _ in range(2)]).to(self.device)
        betas=(c['adam_beta1'],c['adam_beta2'])
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=c['lr'],betas=betas)
        self.q_opt=torch.optim.Adam(self.critics.parameters(),lr=c['lr'],betas=betas)
        self.logalpha=nn.Parameter(torch.tensor(math.log(c['init_alpha']),device=self.device))
        self.alpha_opt=torch.optim.Adam([self.logalpha],lr=c['temp_lr'])
        self.replay=Replay(self.D,self.A,c['replay_capacity'],seed+991)
        self.environment_steps=self.updates=self.actor_updates=self.episodes=0
        self.last={};self.evaluation_only=False
        # Attach AFTER every native base initialization, preserving Critic/std/RNG.
        if c['actor_architecture']=='shared_mean':
            _attach_shared_mean(self.actor,self.D,self.A,self.seed,self.device)
            self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=c['lr'],betas=betas)

    @torch.no_grad()
    def act(self,obs,deterministic=False):
        if not deterministic and self.environment_steps<self.config['random_action_steps']:
            return self.rng.uniform(-1,1,self.A).astype(np.float32)
        training=self.actor.training;self.actor.eval()
        try:
            action,_=self.actor.sample(torch.as_tensor(obs,device=self.device,dtype=torch.float32).reshape(1,-1),deterministic)
            return action[0].cpu().numpy()
        finally:self.actor.train(training)

    def update(self,batch):
        s,a,r,n,d=batch;c=self.config;alpha=self.logalpha.exp().detach()
        self.actor.eval();self.critics.train()
        with torch.no_grad():na,nlogp=self.actor.sample(n)
        allq=torch.stack([q(torch.cat([s,n]),torch.cat([a,na])) for q in self.critics])
        B=len(s);current,nextq=allq[:,:B],allq[:,B:]
        target=r+c['gamma']*(1-d)*(nextq.min(0).values-alpha*nlogp)
        loss=.5*(current-target.detach()).square().mean(1).sum()
        self.q_opt.zero_grad(set_to_none=True);loss.backward();self.q_opt.step();self.updates+=1
        actor_loss=torch.zeros((),device=self.device);entropy=torch.zeros((),device=self.device)
        if self.updates%c['actor_delay']==0:
            self.actor.train();self.critics.eval()
            for p in self.critics.parameters():p.requires_grad_(False)
            try:
                pa,logp=self.actor.sample(s)
                qpi=torch.stack([q(s,pa) for q in self.critics]).min(0).values
                actor_loss=(alpha*logp-qpi).mean()
                self.actor_opt.zero_grad(set_to_none=True);actor_loss.backward();self.actor_opt.step()
            finally:
                for p in self.critics.parameters():p.requires_grad_(True)
                self.critics.train()
            entropy=-logp.detach().mean()
            temp_loss=self.logalpha.exp()*(entropy+c['target_entropy_factor']*self.A)
            self.alpha_opt.zero_grad(set_to_none=True);temp_loss.backward();self.alpha_opt.step()
            self.actor_updates+=1
        if self.updates<=3 or self.updates%c['metric_interval']==0:
            values=torch.stack([loss.detach(),actor_loss.detach(),self.logalpha.exp().detach(),
                current.detach().mean(),target.detach().mean(),entropy]).cpu().tolist()
            if not all(math.isfinite(x) for x in values):raise FloatingPointError('Nonfinite CrossQ learning state')
            self.last=dict(zip(['critic_loss','actor_loss','alpha','current_q','target_q','entropy'],values))
        return self.last

    def observe(self,obs,action,native_reward,next_obs,done):
        if self.evaluation_only:raise ValueError('Evaluation model has no replay/optimizer continuation')
        r=self.config['reward_scale']*float(native_reward)
        self.replay.add(obs,action,r,next_obs,float(done));self.environment_steps+=1
        self.episodes+=int(done)
        if self.environment_steps>=self.config['learning_starts'] and self.replay.size>=self.config['batch_size']:
            for _ in range(self.config['utd']):self.update(self.replay.sample(self.config['batch_size'],self.device))
        return dict(self.last,learning_reward=r)
    def finish(self):return {}
    def stats(self):return self.last
    def accounting(self):
        return dict(environment_steps=self.environment_steps,updates=self.updates,
                    actor_updates=self.actor_updates,episodes=self.episodes,
                    critic_has_target_network=False,declared_critic_width=self.config['critic_width'])
    def save(self,directory):
        p=Path(directory);p.mkdir(parents=True,exist_ok=False)
        target=p/'model.pt'
        torch.save(dict(actor=self.actor.state_dict(),config=self.config,obs_dim=self.D,
            action_dim=self.A,seed=self.seed,accounting=self.accounting(),evaluation_only=True),target)
        (p/'metadata.json').write_text(json.dumps(dict(method='crossq',implementation='explicit MIT-author-math PyTorch port',
            no_optimizer_or_replay=True,critic_saved=False,config=self.config),indent=2)+'\n')
        return target
    @classmethod
    def load(cls,directory,device='cuda'):
        state=torch.load(Path(directory)/'model.pt',map_location='cpu',weights_only=False)
        # Construct inference directly; a 40MB critic is not needed to read an actor.
        obj=cls.__new__(cls);obj.config=state['config'];obj.D=state['obs_dim'];obj.A=state['action_dim']
        obj.seed=state['seed'];obj.device=torch.device(device);c=obj.config
        obj.actor=Actor(obj.D,obj.A,c['actor_width'],c['momentum'],c['brn_warmup']).to(obj.device)
        if c.get('actor_architecture','flat')=='shared_mean':
            _attach_shared_mean(obj.actor,obj.D,obj.A,obj.seed,obj.device)
        obj.actor.load_state_dict(state['actor']);obj.actor.eval();obj.evaluation_only=True
        obj.rng=np.random.default_rng(obj.seed)
        acc=state['accounting'];obj.environment_steps=acc['environment_steps'];obj.updates=acc['updates']
        obj.actor_updates=acc['actor_updates'];obj.episodes=acc['episodes'];obj.last={}
        return obj
