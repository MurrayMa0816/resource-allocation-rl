"""Agent factory and detached read-only signal audits for the native loop."""
import math
import numpy as np
import torch
from matched_agent import create_agent

VARIANTS=('native','native_no_tanh','matched_ff','matched_fs','matched_sf','matched_ss')
def configure_agent_factory(NativeAgent,spec,seed,config):
    variant=config['representation_variant']
    if variant not in VARIANTS:raise ValueError('Unregistered communication diagnostic variant')
    if variant=='native':return NativeAgent
    if variant=='native_no_tanh':
        class NoTanhAgent(NativeAgent):
            def __init__(self,facade):
                super().__init__(facade)
                if not isinstance(self.actor_mean[-1],torch.nn.Tanh):raise TypeError('Expected exact native final Tanh')
                self.actor_mean=torch.nn.Sequential(*list(self.actor_mean.children())[:-1])
        return NoTanhAgent
    actor_kind='flat' if variant[-2]=='f' else 'set';critic_kind='flat' if variant[-1]=='f' else 'set'
    def factory(facade):
        # Bootstrap has already bound allocation23 to the copied full-spend release.
        from allocation23.methods.policy import Actor
        from allocation23.methods.networks import Value
        agent=create_agent(spec,seed,actor_kind,critic_kind,actor_factory=lambda s,k:Actor(s,'A_FULL','adaptive',k),value_factory=lambda s,k:Value(s,k),dtype=torch.float32)
        if tuple(facade.single_observation_space.shape)!=(agent.schema.observation_dim,) or tuple(facade.single_action_space.shape)!=(agent.schema.action_dim,):raise ValueError('Agent and common environment dimensions differ')
        return agent
    return factory

def readonly_signal_audit(means,stds,advantages,returns,values,rewards,dones,previous_done,config):
    """Observed behavior-trajectory MC audit; never used as a training target."""
    arr=lambda x:np.asarray(x.detach().cpu(),dtype=np.float64).reshape(-1)
    v,ret,adv,r,done=map(arr,[values,returns,advantages,rewards,dones])
    if len({len(v),len(ret),len(adv),len(r),len(done)})!=1:raise ValueError('Audit requires aligned native buffer')
    after=np.r_[done[1:],float(previous_done)];mc=np.empty_like(r);running=0.
    for i in reversed(range(len(r))):running=r[i]+(0. if after[i] else running);mc[i]=running
    if after[-1]!=1.:raise ValueError('Complete behavior episodes are required for MC audit')
    mu,sd=arr(means),arr(stds)
    return {'MC_value_RMSE':float(np.sqrt(np.mean((mc-v)**2))),
        'GAE_target_value_residual_RMSE':float(np.sqrt(np.mean((ret-v)**2))),
        'GAE_adv_vs_MCminusV_RMSE':float(np.sqrt(np.mean((adv-(mc-v))**2))),
        'value_variance':float(np.var(v)),'GAE_return_variance':float(np.var(ret)),
        'MC_return_variance':float(np.var(mc)),'MC_return_mean':float(np.mean(mc)),
        'advantages_std':float(np.std(adv)),'MC_minusV_std':float(np.std(mc-v)),
        'actor_mean_abs_mean':float(np.mean(np.abs(mu))),'actor_mean_abs_max':float(np.max(np.abs(mu))),
        'actor_mean_abs_ge_099_fraction':float(np.mean(np.abs(mu)>=.99)),
        'mean_boundary_saturation_applicable':config['representation_variant']=='native',
        'actor_std_mean':float(np.mean(sd)),'actor_std_min':float(np.min(sd)),
        'actor_std_max':float(np.max(sd)),'MC_audit_only_not_training_target':True,
        'MC_is_observed_behavior_return_not_true_current_policy_value':True}

def readonly_gradient_audit(agent):
    result={}
    for prefix in ['actor','critic']:
        total=math.fsum(float(p.grad.detach().double().square().sum()) for name,p in agent.named_parameters() if name.startswith(prefix) and p.grad is not None)
        result[prefix+'_gradient_norm_after_native_clip']=math.sqrt(total)
    return result
