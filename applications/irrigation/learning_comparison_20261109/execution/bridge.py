"""Shared public observation and native feasible-action bridge; no reward shaping."""
import math
import numpy as np
import torch
from paths import bootstrap
h=bootstrap()
from hongmiao89.interface import IrrigationSpec,IrrigationEncoder,append_preference
from hongmiao89.execution_operator import fixed_antithetic_points

ENV_CONFIG=dict(total_budget_m3=4_000_000.,adapter='depth_area_physical',spatial_sigma_scale=1.,coverage_threshold=.9,coverage_weight=.2,reward_mode='area_weighted_performance')
PREFERENCE=np.array([.2,.8],np.float32)

class PublicPool:
    def __init__(self,count):
        self.counters=dict(step_attempts=0,physical_steps=0,episodes=0,reset_attempts=0,resets=0)
        self.envs=h.make_pool(ENV_CONFIG,count)
        self.encoders=[IrrigationEncoder(IrrigationSpec.from_env(e))for e in self.envs]
        saved=torch.get_rng_state()
        try:self.decoder,unused,self.neural_spec=h.make_models_for_arm('full',h.FullSpendSpec.from_env(self.envs[0]),0)
        finally:torch.set_rng_state(saved)
        del unused
        self.decoder.requires_grad_(False)
        self.action_dim=self.decoder.event_dim;self.obs_dim=89*8+5+2
        assert self.action_dim==49 and self.obs_dim==719
        self.points=fixed_antithetic_points(self.action_dim)
        self.seeds=[];self.weather=[];self.q=[];self.initial_previous=[];self.maximum_mass_error=0.
    def flatten(self):
        states=[e.state()for e in self.encoders]
        out=np.stack([np.concatenate([s['local'].ravel(),s['global'],[float(s['remaining'])/4e6,float(s['budget'])/4e6]])for s in states]).astype(np.float32)
        assert out.shape==(len(self.envs),719)and np.isfinite(out).all()
        return out
    def reset(self,seeds):
        assert len(seeds)==len(self.envs)
        self.seeds=list(map(int,seeds));self.weather=[[]for _ in seeds];self.q=[[]for _ in seeds];self.initial_previous=[]
        for env,encoder,seed in zip(self.envs,self.encoders,seeds):
            self.counters['reset_attempts']+=1
            obs,info=env.reset(seed=seed);self.counters['resets']+=1;assert not info;encoder.reset(obs)
            self.initial_previous.append(np.array(obs['previous_weather'],copy=True))
        return self.flatten()
    def decode(self,actions,action_mode):
        x=np.asarray(actions,dtype=np.float64)
        if x.ndim==2:x=x[:,None,:]
        B,P,A=x.shape
        if B!=len(self.envs)or A!=49 or not np.isfinite(x).all():raise ValueError('Complete finite49-coordinate events required')
        if action_mode=='tanh_latent':
            if np.any(abs(x)>1.):raise ValueError('Outside tanh action interval')
            x=np.arctanh(np.clip(x,-1+1e-6,1-1e-6))
        elif action_mode!='raw':raise ValueError('Unknown action coordinates')
        states=[append_preference(e.state(),.2)for e in self.encoders]
        data={k:np.repeat(np.stack([s[k]for s in states]),P,axis=0)for k in states[0]}
        raw=torch.as_tensor(x.reshape(B*P,49),dtype=torch.float64)
        with torch.no_grad():
            quotas=self.decoder.quota(data,raw[:,:1]);vectors=self.decoder.decode(data,raw[:,1:],quotas)
        actual=vectors.reshape(B,P,89).mean(axis=1,dtype=np.float64)
        totals=np.asarray(quotas).reshape(B,P).mean(axis=1)
        for i,row in enumerate(actual):
            error=abs(math.fsum(row.tolist())-float(totals[i]));self.maximum_mass_error=max(self.maximum_mass_error,error)
            state=states[i];R=float(state['remaining']);day=int(round(float(state['global'][0])*81))
            if error>1e-7 or np.any(row<0)or math.fsum(row.tolist())>R+1e-7:raise ArithmeticError('Illegal feasible mean')
            if day==80 and abs(math.fsum(row.tolist())-R)>5e-7:raise ArithmeticError('Last day not full spent')
        return actual
    def step(self,actions,action_mode=None,physical=False):
        actual=np.asarray(actions,dtype=np.float64)if physical else self.decode(actions,action_mode)
        rewards=[];dones=[]
        for i,(env,enc,row)in enumerate(zip(self.envs,self.encoders,actual)):
            self.counters['step_attempts']+=1
            obs,reward,done,truncated,info=env.step(row)
            self.counters['physical_steps']+=1;self.counters['episodes']+=int(done)
            if truncated or (not done and(reward!=0 or info)):raise AssertionError('Native terminal-only contract')
            enc.update(row,obs);self.weather[i].append(np.asarray(obs['previous_weather']).copy());self.q[i].append(math.fsum(row.tolist()))
            if done:
                m=env.terminal_metrics();r=[float(m['C']),float(m['JA'])]
                if abs(float(reward)-r[1])>1e-12:raise AssertionError('JA reward changed')
            else:r=[0.,0.]
            rewards.append(r);dones.append([float(done)])
        return self.flatten(),np.asarray(rewards,np.float32),np.asarray(dones,np.float32)
    def rows(self):
        rows=[]
        for i,e in enumerate(self.envs):
            m=e.terminal_metrics();y=np.asarray(m['quality'],dtype=np.float64);area=np.asarray(e.scenario.areas_ha)
            C=float(np.mean(y>=.9));JA=float(np.dot(area,y)/area.sum())
            if abs(C-float(m['C']))>1e-12 or abs(JA-float(m['JA']))>1e-12:raise AssertionError('Terminal metric recomputation differs')
            q=np.asarray(self.q[i]);weather=np.asarray(self.weather[i]);previous=self.initial_previous[i]
            if q.shape!=(81,)or abs(math.fsum(q.tolist())-4e6)>1e-6:raise ArithmeticError('Seasonal water account')
            rows.append(dict(weather_seed=self.seeds[i],C=C,JA=JA,J=float(y.mean()),F=.2*C+.8*JA,
                quality=y,period_totals=q,weather_trace=weather,initial_previous_weather=previous,
                water_used_m3=math.fsum(q.tolist()),remaining_m3=float(m['remaining_budget_m3']),task_fingerprint=e.task_fingerprint))
        return rows
    def close(self):
        for e in self.envs:e.close()
