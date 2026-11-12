"""Official MAD-TD JAX learner with an explicit finite-terminal task contract."""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root
import copy,dataclasses,hashlib,json,math,os
from pathlib import Path
import numpy as np
from .native_loader import load_native,verify_vendor,PATCH_PROFILE

native=load_native()
import jax
import jax.numpy as jnp
from flax import serialization
from mad_td.nn.actor import TDMPC2Actor
from mad_td.nn.critic import BinnedTDMPC2Critic
from mad_td.nn.model import TDMPC2Encoder,TDMPC2LatentModel
from mad_td.rl_types import get_batch_seed_shape

def _hash(value):
    h=hashlib.sha256()
    for x in jax.tree_util.tree_leaves(value):
        a=np.asarray(jax.device_get(x));h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()

def _hyperparams():
    return native.hp.MadTdHyperparams(batch_size=512,actor_learning_rate=3e-4,critic_learning_rate=3e-4,encoder_learning_rate=1e-4,model_learning_rate=3e-4,gradient_clip=10.,gamma=1.,tau=.99,length_training_rollout=1,num_bins=151,num_quantile=151,is_binned=True,binned_regression_encoding='hl-gauss',use_symlog=False,vmin=-1.,vmax=1.,update_encoder_with_critic_loss=True,use_target_encoder=False,proportion_real=.95,use_resetting=False,slow_actor=False,slow_model=True,use_daml=True,use_model_loss=True,use_mpc=False,num_policy_mpc_samples=1,num_mpc_samples=512,mpc_iterations=6,mpc_top_k=64,mpc_temperature=.5)

class Agent:
    def __init__(self,cfg,observation_dim,action_dim,device='cuda:0'):
        verify_vendor();cfg=dict(cfg)
        self.seed=int(cfg.pop('seed',0));self.total_steps=int(cfg.pop('total_steps',24576));self.domain=cfg.pop('domain',None)
        if cfg.pop('gamma',1.)!=1. or cfg.pop('reward_scale',1.)!=1.:raise ValueError('Registered terminal task uses gamma1 and unchanged real rewards')
        if cfg.pop('value_support',[-1.,1.])!=[-1.,1.]:raise ValueError('Public task support fixed before development results')
        if cfg:raise ValueError('Unregistered MAD-TD overrides: '+repr(cfg))
        if self.domain not in (None,'radio','v5'):raise ValueError('Unknown task')
        if int(action_dim)not in (13,19):raise ValueError('Expected registered compact full-spend action dimensions19/13')
        self.observation_dim=int(observation_dim);self.action_dim=int(action_dim);self.device=str(device)
        self.hp=_hyperparams();self.warmup=5000;self.utd=8;self.internal_validation_fraction=.05
        self.rng=np.random.RandomState(self.seed);self._key=jax.random.PRNGKey(self.seed)
        self.steps=0;self.learning_calls=0;self.first_learning_step=None;self.training_insertions=0;self.validation_insertions=0
        self.real_terminal_transitions=0;self.training_terminal_transitions=0;self.nonterminal_nonzero_rewards=0
        self.sampled_terminal_rows=0;self.sampled_real_terminal_rows=0;self.sampled_model_terminal_rows=0
        self._inference_only=False;self._last_info=None
        requested='gpu'if str(device).startswith(('cuda','gpu'))else'cpu'
        devices=jax.devices(requested)
        if len(devices)!=1 and requested=='gpu':raise RuntimeError('Exactly one registered physical GPU must be visible')
        self.jax_device=devices[0]
        with jax.default_device(self.jax_device):
            self.replay=native.replay.ReplayBuffer((self.observation_dim,),(self.action_dim,),None,1,self.total_steps+1,1)
            self.validation_replay=native.replay.ReplayBuffer((self.observation_dim,),(self.action_dim,),None,1,max(2,int(2*self.total_steps*self.internal_validation_fraction)+1),1)
            encoder=TDMPC2Encoder(hidden_dim=(512,),feature_dim=512)
            actor=TDMPC2Actor(feature_dim=(512,512),output_dim=self.action_dim)
            critic=BinnedTDMPC2Critic(feature_dim=(512,512),num_bins=151,vmin=-1.,vmax=1.,use_symlog=False,add_zero_prior=True,single_critic=False,use_residual=False)
            model=TDMPC2LatentModel(feature_dim=(512,512),output_dim=512)
            self.factory=native.factory.MadTdFactory(critic,critic,actor,encoder,encoder,model,self.replay,self.hp.encoder_learning_rate,self.hp.model_learning_rate,self.hp.critic_learning_rate,self.hp.actor_learning_rate,self.hp.gradient_clip,self.seed)
            self.models=self.factory.init()
            shape=get_batch_seed_shape(self.replay.get_dummy_batch(1,batch_size=10))
            self._update=jax.jit(native.updates.vmaped_utd_update,static_argnums=(2,3,4,5))
            self._batch_shape=shape
            self._action=jax.jit(jax.vmap(lambda s,e,a,c,m,noise,random,k:native.policy_action(s,e,a,c,m,noise,random,self.hp,k),in_axes=(0,0,0,0,0,None,None,0)),static_argnums=(6,))
            self._encoder_module=encoder;self._actor_module=actor
            self._deployment=jax.jit(lambda ep,ap,obs:actor.apply(ap,encoder.apply(ep,obs)))

    @property
    def n_updates(self):return self._loaded_updates if self._inference_only else self.learning_calls*self.utd
    @property
    def updates(self):return self.n_updates

    def _observation(self,obs):
        x=np.asarray(obs,dtype=np.float32)
        if x.shape!=(self.observation_dim,)or not np.isfinite(x).all():raise ValueError('Invalid public vector observation')
        return x

    def policy_action(self,obs,deterministic=True):
        x=self._observation(obs)
        if self._inference_only:ep,ap=self._deployment_params
        else:ep,ap=jax.tree_util.tree_map(lambda x:x[0],(self.models.encoder.params,self.models.actor.params))
        value=np.asarray(jax.device_get(self._deployment(ep,ap,x)),dtype=np.float32)
        if value.shape!=(self.action_dim,)or not np.isfinite(value).all()or np.max(np.abs(value))>1.000001:raise FloatingPointError('MAD-TD deployment action')
        return value

    def act(self,obs,deterministic=False):
        if deterministic or self._inference_only:return self.policy_action(obs,True)
        self._key,action_key=jax.random.split(self._key)
        action,_=self._action(self._observation(obs)[None],self.models.encoder,self.models.actor,self.models.critic,self.models.latent_model,.1,self.steps<self.warmup,jax.random.split(action_key,1))
        value=np.asarray(jax.device_get(action[0]),dtype=np.float32)
        if not np.isfinite(value).all():raise FloatingPointError('MAD-TD exploration action')
        return value

    def observe(self,obs,action,reward,next_obs,done):
        if self._inference_only:raise RuntimeError('Inference endpoint cannot resume learning')
        if self.steps>=self.total_steps:raise RuntimeError('Registered physical budget exceeded')
        state=self._observation(obs);nxt=self._observation(next_obs);a=np.asarray(action,dtype=np.float32);reward=float(reward);done=bool(done)
        if a.shape!=(self.action_dim,)or not np.isfinite(a).all()or np.max(np.abs(a))>1.000001 or not math.isfinite(reward):raise ValueError('Invalid task transition')
        if not done and reward!=0.:self.nonterminal_nonzero_rewards+=1;raise AssertionError('Only genuine terminal rewards allowed')
        if not -1.-1e-8<=reward<=1.+1e-8:raise AssertionError('Public terminal reward bound violated')
        self.real_terminal_transitions+=int(done)
        # The author's independent5% diagnostic replay receives paid transitions;
        # it is never the benchmark DEV/holdout panel and never trains networks.
        training=bool(self.rng.choice([True,False],p=[.95,.05]))
        buffer=self.replay if training else self.validation_replay
        buffer.insert(state[None],a[None],np.array([reward]),np.array([done]),nxt[None])
        self.training_insertions+=int(training);self.validation_insertions+=int(not training);self.training_terminal_transitions+=int(training and done)
        self.steps+=1
        if self.steps>self.warmup:
            self._key,model_key=jax.random.split(self._key)
            batch=self.replay.sample(self.hp.batch_size,model_key,self.utd)
            terminal=np.asarray(batch.mask)==0
            self.sampled_terminal_rows+=int(terminal.sum());self.sampled_real_terminal_rows+=int(terminal[:,:,:486].sum());self.sampled_model_terminal_rows+=int(terminal[:,:,486:].sum())
            with jax.default_device(self.jax_device):
                self.models,self._last_info=self._update(self.models,batch,self.hp,self._batch_shape,1,self.utd,self._key)
            self.learning_calls+=1
            if self.first_learning_step is None:self.first_learning_step=self.steps
        return dict(steps=self.steps,updates=self.n_updates)

    def get_rng_state(self):return dict(key=np.asarray(jax.device_get(self._key)).copy(),uniform=copy.deepcopy(self.rng.get_state()),numpy=copy.deepcopy(np.random.get_state()))
    def set_rng_state(self,state):
        self._key=jax.device_put(np.asarray(state['key']),self.jax_device);self.rng.set_state(state['uniform']);np.random.set_state(state['numpy'])

    def native_state_fingerprint(self):
        result=dict(steps=self.steps,updates=self.n_updates,key=_hash(self._key),uniform_state=hashlib.sha256(repr(self.rng.get_state()).encode()).hexdigest())
        if self._inference_only:result['deployment']=_hash(self._deployment_params)
        else:
            for name in ['critic','critic_target','actor','encoder','encoder_target','latent_model']:
                model=getattr(self.models,name);result[name]=_hash((model.params,model.opt_state,model.step))
            result.update(training_insertions=self.training_insertions,validation_insertions=self.validation_insertions)
        return result

    def diagnostics(self):
        if self._inference_only:return dict(self._loaded_diagnostics,inference_only=True)
        actual={name:int(np.asarray(jax.device_get(getattr(self.models,name).step)).reshape(-1)[0])for name in ['critic','actor','encoder','latent_model']}
        assert actual['critic']==actual['actor']==self.n_updates and actual['encoder']==actual['latent_model']==self.learning_calls
        result=dict(method='MAD-TD',native_core_commit=native.manifest['commit'],patch_profile=PATCH_PROFILE,train_steps=self.steps,updates=self.n_updates,learning_calls=self.learning_calls,critic_optimizer_steps=actual['critic'],policy_optimizer_steps=actual['actor'],encoder_optimizer_steps=actual['encoder'],model_optimizer_steps=actual['latent_model'],first_learning_environment_step=self.first_learning_step,UTD=self.utd,batch_size=512,critic_real_rows_per_batch=486,critic_model_generated_rows_per_batch=26,generated_on_policy_transitions=self.n_updates*26,real_critic_references=self.n_updates*486,training_insertions=self.training_insertions,internal_validation_insertions=self.validation_insertions,internal_validation_fraction=.05,internal_validation_is_benchmark_holdout=False,real_terminal_transitions=self.real_terminal_transitions,training_terminal_transitions=self.training_terminal_transitions,sampled_terminal_rows=self.sampled_terminal_rows,sampled_real_terminal_rows=self.sampled_real_terminal_rows,sampled_model_terminal_rows=self.sampled_model_terminal_rows,nonterminal_nonzero_rewards=self.nonterminal_nonzero_rewards,replay_size=self.replay.filled,replay_capacity=self.replay.capacity,gamma=1.,external_reward_scale=1.,value_support=[-1.,1.],num_bins=151,use_mpc=False,slow_model=True,slow_actor=False,deterministic_deployment=True,native_device=str(self.jax_device),inference_only=False,task_terminal_supervised=True,terminal_bootstrap_zero=True)
        if self._last_info is not None:
            for group,fields in jax.device_get(self._last_info).items():
                for name,value in fields.items():
                    v=float(np.asarray(value).mean())
                    # Native corrcoef can be undefined on a constant minibatch;
                    # retain JSON null only for that diagnostic, never losses.
                    if not math.isfinite(v):
                        if name!='ood_correlation':raise FloatingPointError((group,name,v))
                        result[f'{group}/{name}']=None
                    else:result[f'{group}/{name}']=v
        return result

    def save(self,path):
        if self._inference_only:raise RuntimeError('Preserve original inference endpoint')
        p=Path(path).resolve()
        if not p.is_relative_to(str(_resolve_path('/hy-tmp'))):raise ValueError('All generated models belong in HY')
        ep,ap=jax.tree_util.tree_map(lambda x:np.asarray(jax.device_get(x[0])),(self.models.encoder.params,self.models.actor.params))
        metadata=dict(format='madtd-deployment-inference-v1',source_commit=native.manifest['commit'],patch_profile=PATCH_PROFILE,observation_dim=self.observation_dim,action_dim=self.action_dim,seed=self.seed,total_steps=self.total_steps,steps=self.steps,updates=self.n_updates,domain=self.domain,hyperparams=dataclasses.asdict(self.hp),diagnostics=self.diagnostics(),optimizer_model_critic_or_replay_included=False)
        payload=serialization.msgpack_serialize(dict(metadata=json.dumps(metadata,allow_nan=False),encoder=ep,actor=ap))
        p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.writing');tmp.write_bytes(payload);os.replace(tmp,p)

    @classmethod
    def load(cls,path,device='cuda:0'):
        verify_vendor();data=serialization.msgpack_restore(Path(path).read_bytes());meta=json.loads(data['metadata'])
        if meta['format']!='madtd-deployment-inference-v1'or meta['source_commit']!=native.manifest['commit']or meta['patch_profile']!=PATCH_PROFILE:raise ValueError('MAD-TD inference identity mismatch')
        self=cls.__new__(cls);self._inference_only=True;self.device=str(device);self.jax_device=jax.devices('gpu'if str(device).startswith(('cuda','gpu'))else'cpu')[0]
        for key in ['seed','total_steps','steps','observation_dim','action_dim','domain']:setattr(self,key,meta[key])
        self._loaded_updates=meta['updates'];self._loaded_diagnostics=meta['diagnostics'];self.hp=_hyperparams();self.rng=np.random.RandomState(self.seed);self._key=jax.random.PRNGKey(self.seed)
        encoder=TDMPC2Encoder(hidden_dim=(512,),feature_dim=512);actor=TDMPC2Actor(feature_dim=(512,512),output_dim=self.action_dim)
        self._deployment=jax.jit(lambda ep,ap,obs:actor.apply(ap,encoder.apply(ep,obs)));self._deployment_params=jax.device_put((data['encoder'],data['actor']),self.jax_device)
        return self
