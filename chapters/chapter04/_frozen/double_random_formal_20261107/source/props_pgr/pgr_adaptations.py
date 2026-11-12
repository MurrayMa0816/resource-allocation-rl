"""Pinned native adapters; preserve TD/PGR updates and declare task changes."""
import importlib.util,sys,time,types
import numpy as np
from formal_support import ACCEL,dump,read,sha

def reward_support(generated,domain):
    """Return copied rewards; all other generated fields remain native objects."""
    if domain not in ('radio','v5') or len(generated)!=5:raise ValueError('Five native transition arrays required')
    obs,act,reward,nextobs,done=generated
    reward=np.asarray(reward);done=np.asarray(done)
    if reward.shape!=done.shape or reward.ndim!=1 or not np.isfinite(reward).all():raise ValueError('Finite aligned scalar native rewards required')
    if not np.isfinite(done).all() or not np.isin(done,[0.,1.]).all():raise ValueError('Keep native thresholded binary terminal flags')
    corrected=reward.copy();corrected[done==0]=0.
    low=-1. if domain=='radio' else 0.
    corrected[done==1]=np.clip(corrected[done==1],low,1.)
    nonterminal=done==0
    metadata=dict(samples=int(len(done)),native_terminal_count=int(done.sum()),
        nonterminal_count=int(nonterminal.sum()),
        raw_nonterminal_nonzero_reward_count=int(np.count_nonzero(reward[nonterminal])),
        corrected_nonterminal_nonzero_reward_count=int(np.count_nonzero(corrected[nonterminal])),
        raw_reward_min=float(reward.min()),raw_reward_max=float(reward.max()),
        corrected_reward_min=float(corrected.min()),corrected_reward_max=float(corrected.max()),
        changed_rewards=int(np.count_nonzero(corrected!=reward)),done_fields_changed=False,
        rule='Known task support only: synthetic nonterminal reward0; terminal prediction clipped to public real reward range',
        no_real_reward_rewrite=True,no_hidden_label_or_environment_evaluation=True)
    return (obs,act,corrected,nextobs,done),metadata

def load_file(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module

def verified_acceleration():
    if sha(ACCEL/'capsule.json')!='d273286138897c191e1a679bd4bb0049e60685d9c8fccd488e1c575dd1fc85c6':
        raise RuntimeError('Pinned v3 acceleration capsule identity differs')
    capsule=read(ACCEL/'capsule.json')
    for path,expected in capsule['files_sha256'].items():
        if sha(path)!=expected:raise RuntimeError('Reviewed PGR acceleration changed: '+path)
    parity=read(capsule['engineering_parity'])
    if not parity.get('passed') or sha(capsule['engineering_parity'])!=capsule['engineering_parity_sha256']:
        raise RuntimeError('Reviewed full-size IO/EMA parity gate changed')
    io=load_file(capsule['patch_module'],'_new_scope_pgr_exact_io')
    ema=load_file(capsule['ema_patch_module'],'_new_scope_pgr_exact_ema')
    return capsule,io,ema

def install_pgr(runner,domain,out,*,engineering=False):
    """Native model/core stays unchanged; patch generation output and exact IO."""
    capsule,io,ema=verified_acceleration()
    original_author=runner._load_author;original_agent=runner._agent
    original_trainer=runner.install_safe_trainer_optimizations
    records={'support':[],'io':[],'ema':[],'diffusion_update_seconds':[],'generation':[]}
    def native_author():
        native=original_author();Generator=native['Generator']
        class SupportedGenerator(Generator):
            def sample(self,*args,**kwargs):
                import torch
                torch.cuda.synchronize();started=time.perf_counter()
                generated=super().sample(*args,**kwargs)
                torch.cuda.synchronize()
                timing=dict(wall_seconds=time.perf_counter()-started,samples=int(len(generated[-1])),
                    denoising_steps=int(self.num_sample_steps) if hasattr(self,'num_sample_steps') else 128,
                    cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                    cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
                records['generation'].append(timing)
                supported,receipt=reward_support(generated,domain)
                receipt['cycle']=len(records['support'])+1;records['support'].append(receipt)
                dump(out/'synthetic_support.json',records['support'])
                dump(out/'pgr_timing.json',dict(diffusion_update_seconds=records['diffusion_update_seconds'],generation=records['generation'],engineering_only=engineering))
                return supported
        native['Generator']=SupportedGenerator
        return native
    def accelerated_agent(*args,**kwargs):
        agent=original_agent(*args,**kwargs)
        before=runner._rng();identities=[id(agent.policy_net),id(agent.cond_net),id(agent.policy_optimizer),id(agent.cond_optimizer),id(agent.alpha_optim)]
        result=dict(io.install_approved(agent));after=runner._rng()
        assert before[0]==after[0] and np.array_equal(before[1][1],after[1][1]) and before[1][2:]==after[1][2:]
        import torch
        assert torch.equal(before[2],after[2]) and all(torch.equal(a,b) for a,b in zip(before[3],after[3]))
        assert identities==[id(agent.policy_net),id(agent.cond_net),id(agent.policy_optimizer),id(agent.cond_optimizer),id(agent.alpha_optim)]
        result.update(installation_rng_unchanged=True,model_optimizer_objects_unchanged=True)
        records['io'].append(result);dump(out/'exact_io_installations.json',records['io'])
        return agent
    def accelerated_trainer(trainer):
        original_trainer(trainer);before=runner._rng()
        result=dict(ema.install_cuda(trainer.ema));after=runner._rng()
        import torch
        assert before[0]==after[0] and np.array_equal(before[1][1],after[1][1]) and before[1][2:]==after[1][2:]
        assert torch.equal(before[2],after[2]) and all(torch.equal(a,b) for a,b in zip(before[3],after[3]))
        result['installation_rng_unchanged']=True;records['ema'].append(result)
        dump(out/'exact_ema_installations.json',records['ema'])
        if engineering:
            native_batch=trainer.train_on_batch
            def timed_batch(self,*args,**kwargs):
                torch.cuda.synchronize();stamp=time.perf_counter()
                value=native_batch(*args,**kwargs)
                torch.cuda.synchronize();records['diffusion_update_seconds'].append(time.perf_counter()-stamp)
                return value
            trainer.train_on_batch=types.MethodType(timed_batch,trainer)
    runner._load_author=native_author;runner._agent=accelerated_agent
    runner.install_safe_trainer_optimizations=accelerated_trainer
    dump(out/'pgr_adaptation.json',dict(
        scientific_change='synthetic reward support at native generator output before replay insertion',
        real_replay_rewards_unchanged=True,all_native_sample_and_update_budgets_unchanged=True,
        acceleration_capsule=str(ACCEL/'capsule.json'),acceleration_capsule_sha256=sha(ACCEL/'capsule.json'),
        old_registered_job_whitelist_not_claimed=True,
        new_scope_authorization='External-baseline adaptation under the fixed full native interaction budget',
        original_float_math_and_rng_sequence_acceleration='Pinned IO+EMA v3 reused after full capsule/parity hash verification'))
    return records
