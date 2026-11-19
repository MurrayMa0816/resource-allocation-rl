"""Native BRO (Fast): author RR2 profile with one audited original reset.

All losses, models, optimizer updates, quantiles, dual policies and reset
mathematics are executed by byte-identical vendor code.  This file bridges
the original single-environment loop to the common transition interface.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', '/hy-tmp/resource_allocation_results/chapter4/candidate_pool_expansion_20261105/bro/cache')
os.environ.setdefault('JAX_ENABLE_COMPILATION_CACHE', 'false')

import numpy as np
from .native_loader import load_native

PIN = '5870a13d4bf4f66eeddfc291a59df6d2009de600'
MODEL_NAMES = ('actor', 'actor_o', 'critic', 'target_critic', 'temp', 'optimism', 'regularizer')


@dataclass(frozen=True)
class Config:
    seed: int = 0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    adj_lr: float = 3e-5
    gamma: float = .99
    tau: float = .005
    target_entropy: float | None = None
    init_temperature: float = 1.
    init_optimism: float = 1.
    init_regularizer: float = .25
    pessimism: float = 0.
    n_quantiles: int = 100
    kl_target: float = .05
    std_multiplier: float = .75
    start_training: int = 2500
    updates_per_step: int = 2
    batch_size: int = 128
    replay_capacity: int = 1000000
    reward_scale: float = 1.
    total_steps: int | None = None


def parse_config(cfg):
    values = dict(cfg) if isinstance(cfg, dict) else (asdict(cfg) if isinstance(cfg, Config) else vars(cfg).copy())
    if 'discount' in values and 'gamma' not in values:
        values['gamma'] = values['discount']
    c = Config(**{k:v for k,v in values.items() if k in Config.__dataclass_fields__})
    expected = Config(seed=c.seed, target_entropy=c.target_entropy, total_steps=c.total_steps)
    if c != expected:
        raise ValueError('BRO (Fast) author profile fixed; no silent core/scale/RR/clock changes')
    return c


def tree_hash(tree):
    import jax
    h = hashlib.sha256()
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    for path, leaf in leaves:
        a = np.asarray(leaf)
        h.update(repr(path).encode());h.update(str(a.dtype).encode());h.update(repr(a.shape).encode())
        h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()


class Agent:
    method = 'BRO (Fast)'
    implementation = 'unmodified_author_JAX_BRO_Fast_RR2_with_explicit_loop_adapter'

    def __init__(self, cfg, observation_dim, action_dim, device='cuda:0'):
        self.cfg = parse_config(cfg)
        self.observation_dim, self.action_dim = int(observation_dim), int(action_dim)
        if self.cfg.target_entropy is not None and self.cfg.target_entropy != -self.action_dim / 2:
            raise ValueError('native entropy target is -action_dim/2')
        self._bro, Replay, self._Batch = load_native()
        import jax
        import gym
        self.device = jax.devices('cpu' if str(device).startswith('cpu') else 'gpu')[0]
        c = self.cfg
        with jax.default_device(self.device):
            self._native = self._bro.BRO(c.seed,
                np.zeros((1,self.observation_dim),np.float32),np.zeros((1,self.action_dim),np.float32),
                actor_lr=c.actor_lr,critic_lr=c.critic_lr,temp_lr=c.temp_lr,adj_lr=c.adj_lr,
                discount=c.gamma,tau=c.tau,target_entropy=c.target_entropy,
                init_temperature=c.init_temperature,init_optimism=c.init_optimism,
                init_regularizer=c.init_regularizer,pessimism=c.pessimism,num_seeds=1,
                updates_per_step=c.updates_per_step,distributional=True,n_quantiles=c.n_quantiles,
                kl_target=c.kl_target,std_multiplier=c.std_multiplier)
        if self._native.reset_list != [15001]:
            raise RuntimeError('native BRO Fast reset schedule changed')
        obs_space = SimpleNamespace(shape=(self.observation_dim,),dtype=np.dtype(np.float32))
        self.replay = Replay(obs_space,self.action_dim,c.replay_capacity,num_seeds=1)
        self._warmup_space = gym.spaces.Box(-1.,1.,shape=(self.action_dim,),dtype=np.float32)
        self._warmup_space.seed(c.seed)
        self.train_steps = self.gradient_iteration_count = self.updates_since_reset = 0
        self.reset_events = []
        self._last_info = {}
        self._inference_only = False
        self._initial_learning_hash = self._learning_hash()
        self._parameter_counts = {name:sum(int(x.size) for x in jax.tree_util.tree_leaves(getattr(self._native,name).params)) for name in MODEL_NAMES}
        self._original_native_reset = self._native.reset
        self._native.reset = self._audited_reset

    @property
    def n_updates(self):
        return self.gradient_iteration_count

    @property
    def updates(self):
        return self.gradient_iteration_count

    def _obs(self, obs):
        x = np.asarray(obs,dtype=np.float32)
        if x.shape != (self.observation_dim,) or not np.isfinite(x).all():
            raise ValueError('one finite public flat observation required')
        return x

    def act(self, obs, deterministic=False):
        x = self._obs(obs)
        if self._inference_only or deterministic:
            return self.policy_action(x,deterministic=deterministic)
        # Original loop starts at i=1, i<2500 uniform; i=2500 optimistic policy.
        if self.train_steps + 1 < self.cfg.start_training:
            return self._warmup_space.sample().copy()
        import jax
        with jax.default_device(self.device):
            a = self._native.sample_actions_o(x[None],temperature=1.)
        return np.asarray(a)[0].copy()

    def policy_action(self, obs, deterministic=False):
        """Deployment pessimistic policy, never uniform or optimistic collector.

        The root study uses deterministic=True, matching the author evaluator.
        False remains an explicit stochastic deployment-policy diagnostic.
        """
        import jax
        x = self._obs(obs)
        with jax.default_device(self.device):
            a = self._native.sample_actions(x[None],temperature=0. if deterministic else 1.)
        return np.asarray(a)[0].copy()

    def get_rng_state(self):
        return {'jax_rng':np.array(self._native.rng,copy=True),
            'warmup_rng':copy.deepcopy(self._warmup_space.np_random.bit_generator.state)}

    def set_rng_state(self, state):
        import jax
        key = np.asarray(state['jax_rng'],dtype=np.uint32)
        if key.shape != np.asarray(self._native.rng).shape:
            raise ValueError('native PRNG shape mismatch')
        self._native.rng = jax.device_put(key,self.device)
        self._warmup_space.np_random.bit_generator.state = copy.deepcopy(state['warmup_rng'])

    def _learning_hash(self):
        states = {'rng':self._native.rng,'native_step':int(self._native.step)}
        for name in MODEL_NAMES:
            model = getattr(self._native,name,None)
            if model is not None:
                states[name] = {'params':model.params,'opt_state':model.opt_state,'step':model.step}
        return tree_hash(states)

    def native_state_fingerprint(self):
        h = hashlib.sha256(self._learning_hash().encode())
        h.update(json.dumps({'train_steps':self.train_steps,'updates':self.n_updates,
            'updates_since_reset':self.updates_since_reset,'reset_events':self.reset_events,
            'replay_size':self.replay.size if self.replay is not None else 0,
            'replay_insert_index':self.replay.insert_index if self.replay is not None else 0,
            'warmup_rng':self._warmup_space.np_random.bit_generator.state},sort_keys=True).encode())
        return h.hexdigest()

    def initial_fingerprints(self):
        return {'native_actor':tree_hash(self._native.actor.params),
            'native_optimistic_actor':tree_hash(self._native.actor_o.params) if hasattr(self._native,'actor_o') else None,
            'native_learning_state':self._learning_hash() if not self._inference_only else None}

    def _replay_hash(self):
        n = self.replay.size
        return tree_hash({k:getattr(self.replay,k)[:,:n] for k in
            ('observations','actions','rewards','masks','dones_float','next_observations')})

    def _audited_reset(self):
        """Instrumentation of the one reset called inside the unchanged native update."""
        before = self._replay_hash()
        size,index = self.replay.size,self.replay.insert_index
        native_step = int(self._native.step)
        cycle = self.updates_since_reset
        self._original_native_reset()
        after = self._replay_hash()
        if before != after or (size,index) != (self.replay.size,self.replay.insert_index):
            raise RuntimeError('native BRO reset modified historical replay')
        if self._learning_hash() != self._initial_learning_hash:
            raise RuntimeError('BRO reset did not restore complete original learning state')
        self.reset_events.append({'env_step':self.train_steps,
            'lifetime_updates_before':self.n_updates,'update_clock_before':cycle,'update_clock_after':0,
            'native_step_before':native_step,'native_step_after':int(self._native.step),
            'replay_size':size,'replay_insert_index':index,'replay_sha256_before':before,
            'replay_sha256_after':after,'complete_learning_state_equals_initial':True})
        self.updates_since_reset = 0

    def observe(self, obs, action, reward, next_obs, done):
        if self._inference_only:
            raise RuntimeError('single inference endpoint cannot resume training')
        import jax
        x,xp = self._obs(obs),self._obs(next_obs)
        a = np.asarray(action,dtype=np.float32)
        if a.shape != (self.action_dim,) or not np.isfinite(a).all() or np.any(np.abs(a)>1.000001):
            raise ValueError('replay requires the executed native bounded latent action')
        if not np.isfinite(reward):raise ValueError('nonfinite raw reward')
        self.replay.insert(x[None],a[None],np.array([reward*self.cfg.reward_scale],np.float32),
            np.array([0. if done else 1.],np.float32),np.array([float(done)],np.float32),xp[None])
        self.train_steps += 1
        if self.train_steps >= self.cfg.start_training:
            batch = self.replay.sample_parallel_multibatch(self.cfg.batch_size,self.cfg.updates_per_step)
            with jax.default_device(self.device):
                self._last_info = self._native.update(batch,self.cfg.updates_per_step,self.train_steps)
            self.gradient_iteration_count += self.cfg.updates_per_step
            self.updates_since_reset += self.cfg.updates_per_step
        return None

    def diagnostics(self):
        import jax
        info = {k:float(np.asarray(v).reshape(-1)[0]) for k,v in jax.device_get(self._last_info).items()}
        return {'train_steps':self.train_steps,'steps':self.train_steps,'updates':self.n_updates,
            'gradient_iteration_count':self.n_updates,'actor_updates':self.n_updates,
            'optimistic_actor_updates':self.n_updates,'critic_updates':self.n_updates,
            'target_updates':self.n_updates,'temperature_updates':self.n_updates,
            'optimism_updates':self.n_updates,'regularizer_updates':self.n_updates,
            'updates_since_reset':self.updates_since_reset,'native_step':int(self._native.step),
            'reset_event_count':len(self.reset_events),'reset_events':copy.deepcopy(self.reset_events),
            'reset_times':[e['env_step'] for e in self.reset_events],
            'replay_size':self.replay.size if self.replay is not None else 0,
            'native_jax_backend':self.device.platform,'native_jax_device':str(self.device),
            'native_jax_device_kind':self.device.device_kind,
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'parameter_counts':self._parameter_counts,'inference_only':self._inference_only,
            'deployment_policy':'pessimistic actor','training_collector':'optimistic actor after uniform warmup',**info}

    def save(self, path):
        import flax
        import jax
        meta = {'config':asdict(self.cfg),'observation_dim':self.observation_dim,'action_dim':self.action_dim,
            'train_steps':self.train_steps,'updates':self.n_updates,'updates_since_reset':self.updates_since_reset,
            'reset_events':self.reset_events,'source_pin':PIN,'deployment_policy':'pessimistic actor',
            'native_step':int(self._native.step),'parameter_counts':self._parameter_counts,
            'format_note':'inference-only Flax msgpack, not Torch pickle or resumable learning state'}
        payload = {'schema':'bro-fast-native-inference-v1','metadata_json':json.dumps(meta),
            'actor_params':jax.tree_util.tree_map(np.asarray,self._native.actor.params),
            'rng':np.asarray(self._native.rng)}
        Path(path).write_bytes(flax.serialization.msgpack_serialize(payload))

    @classmethod
    def load(cls, path, device='cuda:0'):
        import flax
        import jax
        from .native_loader import load_native
        load_native()
        from jaxrl.networks.common import Model
        from jaxrl.networks.policies import NormalTanhPolicy
        import gym
        payload = flax.serialization.msgpack_restore(Path(path).read_bytes())
        if payload['schema'] != 'bro-fast-native-inference-v1':raise ValueError('unexpected BRO endpoint schema')
        meta = json.loads(payload['metadata_json']);obj = cls.__new__(cls)
        obj.cfg = parse_config(meta['config']);obj.observation_dim = meta['observation_dim'];obj.action_dim = meta['action_dim']
        obj.device = jax.devices('cpu' if str(device).startswith('cpu') else 'gpu')[0]
        learner,_,_ = load_native()
        obj._bro = learner
        params = jax.tree_util.tree_map(lambda a:jax.device_put(a,obj.device),payload['actor_params'])
        # Construct only the existing native deployment actor; no initialization,
        # optimistic actor, critic, replay, temperature or optimizer is allocated.
        actor = Model(step=1,apply_fn=NormalTanhPolicy(obj.action_dim,use_bronet=True),params=params,tx=None,opt_state=None)
        obj._native = SimpleNamespace(actor=actor,rng=jax.device_put(payload['rng'],obj.device),step=meta['native_step'])
        # Reuse the native method (no copied action distribution mathematics).
        obj._native.sample_actions = lambda observations,temperature=1.:learner.BRO.sample_actions(obj._native,observations,temperature)
        obj._warmup_space = gym.spaces.Box(-1.,1.,shape=(obj.action_dim,),dtype=np.float32);obj._warmup_space.seed(obj.cfg.seed)
        obj.train_steps = meta['train_steps'];obj.gradient_iteration_count = meta['updates'];obj.updates_since_reset = meta['updates_since_reset']
        obj.reset_events = meta['reset_events'];obj._parameter_counts = meta['parameter_counts']
        obj._last_info = {};obj.replay = None;obj._inference_only = True
        return obj
