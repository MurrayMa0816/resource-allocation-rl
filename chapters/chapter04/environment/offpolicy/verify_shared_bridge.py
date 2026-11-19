#!/usr/bin/env python3
"""CPU-only, no-training acceptance of the unified full-spend task bridge."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
SCOPE = ROOT.parents[1]
TMP = SCOPE / 'tmp/env'
TMP.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(TMP), TMP=str(TMP), TEMP=str(TMP),
                  PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  XDG_CACHE_HOME=str(TMP/'cache'))
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
torch.set_num_threads(1)
from chapter4_full_spend_env.bootstrap import install

OUT = ROOT / 'acceptance/attempt04'


def same(a, b):
    assert set(a) == set(b)
    assert all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)


def expect_error(fn, types=(ArithmeticError, ValueError, RuntimeError)):
    try:
        fn()
    except types:
        return True
    raise AssertionError('Expected a rejected invalid operation')


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    identity = install(audit_dir=OUT/'resource_audit', context={'purpose':'no-training-full-spend-gate'})
    from chapter4_full_spend_env import native
    from chapter4_recent_baselines.env_adapter import Env, Task, Actor
    from chapter4_recent_baselines.runners.reference_runner import _native
    refs = _native()
    assert refs['make_actor'].__module__.startswith(native._RELEASE_PACKAGE)
    assert sys.modules['allocation23.task'].Task is Task
    assert sys.modules['allocation23.methods.policy'].Actor is Actor
    prior_actor = ROOT/'release/allocation23/methods/policy.py'
    old_actor = Path('/root/resource_allocation_run_storage_20261101/chapter4_recent_baselines_formal_20261102_source/chapter23_final/allocation23/methods/policy.py')
    funcs = lambda p: {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(p.read_text()))
                        if isinstance(n,ast.FunctionDef)}
    current_funcs, old_funcs = funcs(prior_actor), funcs(old_actor)
    unchanged = {name: current_funcs[name] == old_funcs[name] for name in ('sample','log_prob')}
    assert all(unchanged.values())
    old_tree, new_tree = ast.parse(old_actor.read_text()), ast.parse(prior_actor.read_text())
    old_decode = next(n for n in ast.walk(old_tree) if isinstance(n,ast.FunctionDef) and n.name=='decode')
    new_decode = next(n for n in ast.walk(new_tree) if isinstance(n,ast.FunctionDef) and n.name=='decode')
    additions = [n for n in new_decode.body if isinstance(n,ast.If)
                 and ast.unparse(n.test)=='self.spec.chapter == 3']
    assert len(additions)==1
    new_decode.body = [n for n in new_decode.body if n not in additions]
    assert ast.dump(new_decode,include_attributes=False)==ast.dump(old_decode,include_attributes=False)
    records = []
    pure_checks = []
    original_steps = replay_steps = 0
    negative_checks = 0
    for domain in ('radio','v5'):
        env = Env(domain,730,phase='engineering',action_mode='raw',observation_dtype=np.float64)
        spec = env.native_spec
        for case in range(4):
            obs, info = env.reset(options={'episode_index':case})
            assert info == {} and obs.shape == env.observation_space.shape
            assert not env.holdout
            actions, observations, rewards = [], [], []
            snapshot = env.task.env.encoder.state()
            bad = np.full(env.action_dim,np.nan)
            assert expect_error(lambda:env.step(bad));negative_checks += 1
            same(snapshot,env.task.env.encoder.state())
            violation = np.asarray(snapshot['caps']).copy()
            violation[0] += max(spec.initial_budget,1.)
            assert expect_error(lambda:env.task.env.step(violation));negative_checks += 1
            same(snapshot,env.task.env.encoder.state())
            if case == 0:
                actor = refs['make_actor'](spec,'A',3,730)
                g1 = torch.Generator().manual_seed(9101)
                g2 = torch.Generator().manual_seed(9101)
                sampled = actor.sample(env.state,generator=g1)
                mirror = refs['CPURolloutPair'](actor,refs['make_value'](spec,730))
                mirrored = mirror.actor.sample(env.state,generator=g2)
                for key in ('raw','log_prob','mask','quota'):
                    assert torch.equal(sampled[key],mirrored[key]),key
                assert np.array_equal(sampled['actions'],mirrored['actions'])
                mu, sigma = refs['gaussian_parameters'](actor,env.state)
                independent_lp = torch.distributions.Normal(mu,sigma).log_prob(sampled['raw']).sum(-1,keepdim=True)
                assert torch.max(torch.abs(independent_lp-sampled['log_prob'])) < 1e-10
                pure_checks.append({'domain':domain,'joint_Gaussian_density_exact':True,
                    'sample_fields_unchanged':True,'CPU_mirror_single_step_exact':True,
                    'event_dim':actor.event_dim,'mask_unchanged':True})
            for t in range(spec.horizon):
                raw = np.zeros(env.action_dim)
                if case == 1:
                    raw[0] = 30.
                elif case == 2:
                    raw[0] = -30.
                elif case == 3:
                    raw[0] = -20. if t % 3 else 20.
                    raw[1:] = np.where((np.arange(env.action_dim-1)+t)%2,8.,-8.)
                if t == spec.horizon-1 and float(env.state['remaining']) > 0:
                    lower,upper = native.full_spend.full_spend_bounds(env.state,spec)
                    tolerance=native.full_spend.terminal_budget_tolerance(spec)
                    assert abs(lower-upper) <= tolerance, (domain,case,lower,upper,tolerance)
                    assert abs(upper-float(env.state['remaining'])) <= tolerance
                    if lower > native.full_spend.terminal_budget_tolerance(spec):
                        before = env.task.env.encoder.state()
                        assert expect_error(lambda:env.task.env.step(np.zeros(spec.groups)))
                        negative_checks += 1
                        same(before,env.task.env.encoder.state())
                obs,reward,done,truncated,info = env.step(raw)
                assert not truncated and done == (t==spec.horizon-1)
                assert info == {}
                assert reward == 0. if not done else np.isfinite(reward)
                actions.append(env.last_physical_action)
                observations.append(env.task.env.encoder.observation())
                rewards.append(float(reward))
                original_steps += 1
            ledger = env.task.env.last_full_spend_audit
            assert ledger['terminal_full_spend_assertion']
            assert ledger['relative_allocation_error'] <= 1e-10
            assert all(r['quota']+native.full_spend.terminal_budget_tolerance(spec)>=r['lower']
                       and r['quota']<=r['upper']+native.full_spend.terminal_budget_tolerance(spec)
                       for r in ledger['rounds'])
            # Exact submitted physical actions are replayed on the same public
            # task stream, without constructing or updating any learning model.
            replay = Task(domain,3,'A',730,path_master=env.path_master)
            replay.reset('engineering',case)
            for t,a in enumerate(actions):
                ro,rr,rd,rt,ri = replay.env.step(a)
                same(ro,observations[t]);assert rr == rewards[t]
                assert rd == (t==spec.horizon-1) and not rt
                replay_steps += 1
            assert replay.env.last_full_spend_audit['total_used']==ledger['total_used']
            records.append({'domain':domain,'case':case,'raw_mode':['zero','quota+30','quota-30','alternating-extreme'][case],
                'horizon':spec.horizon,'initial_budget':spec.initial_budget,
                'total_used':ledger['total_used'],'remaining':ledger['remaining'],
                'relative_allocation_error':ledger['relative_allocation_error'],
                'total_allocation_absolute_error':ledger['total_allocation_absolute_error'],
                'maximum_step_balance_absolute_error':ledger['maximum_step_balance_absolute_error'],
                'maximum_negative_allocation_violation':ledger['maximum_negative_allocation_violation'],
                'maximum_per_object_cap_violation':ledger['maximum_per_object_cap_violation'],
                'maximum_completed_object_allocation':ledger['maximum_completed_object_allocation'],
                'native_observations_actions_rewards_replay_exact':True,
                'last_lower':ledger['rounds'][-1]['lower'],'last_upper':ledger['rounds'][-1]['upper']})
            replay.env.close()
        if domain == 'radio':
            state = {k:np.asarray(v).copy() for k,v in snapshot.items()}
            state['local'][:,2]=1.
            state['local'][0,2]=(spec.source_packets-1)/spec.source_packets
            state['caps'][:]=0.;state['caps'][0]=spec.per_object_cap
            state['remaining']=np.asarray(spec.per_object_cap/2)
            state['budget']=np.asarray(spec.per_object_cap/2)
            state['global'][0]=0.
            lower,upper=native.full_spend.full_spend_bounds(state,spec)
            assert lower==upper==spec.per_object_cap/2
            physical=env.decoder.decode(state,np.zeros(env.decoder.spatial.event_dim),np.asarray([lower]))[0]
            assert physical[0]>0 and not np.any(physical[1:])
            pure_checks.append({'domain':'radio','early_last_packet_forces_spend_now':True,
                                'completed_receivers_zero_power':True,'lower_equals_upper':lower})
            impossible=copy.deepcopy(state)
            impossible['remaining']=np.asarray(2*spec.per_object_cap)
            impossible['budget']=np.asarray(spec.per_object_cap)
            assert expect_error(lambda:native.full_spend.full_spend_bounds(impossible,spec));negative_checks+=1
        env.close()
    assert original_steps==replay_steps==4*(32+24)
    assert not torch.cuda.is_initialized()
    # Reuse the already completed C3 C2 regression evidence; no C2 process,
    # environment, model, RNG stream or queue is accessed or changed here.
    c3=Path('/hy-tmp/resource_allocation_results/chapter3/full_spend_revision_20261104/preflight')
    evidence=json.loads((c3/'chapter2_unchanged_acceptance.json').read_text())
    old=json.loads((c3/'chapter2_old_physical_traces.json').read_text())
    new=json.loads((c3/'chapter2_new_physical_traces.json').read_text())
    assert evidence['status']=='pass' and old==new and len(old)==8
    assert native.verify_sources()=={k:identity[k] for k in native.verify_sources()}
    result=dict(status='PASS',scope='No-training engineering acceptance, not learning efficacy',
        original_development_or_training_paths=0,engineering_paths=8,physical_replay_paths=8,
        actual_environment_steps=original_steps+replay_steps,training_updates=0,holdout_paths=0,
        cuda_initialized=False,signals_sent=0,negative_checks=negative_checks,
        complete_Gaussian_methods_AST_unchanged=unchanged,spatial_decode_only_adds_C3_feasibility_guard=True,pure_checks=pure_checks,
        chapter2_existing_8trace_byte_regression_reused=True,
        chapter2_evidence_sha256=native.digest(c3/'chapter2_unchanged_acceptance.json'),
        all_paths_full_spend=True,max_relative_allocation_error=max(r['relative_allocation_error'] for r in records),
        max_total_allocation_absolute_error_by_domain={domain:max(r['total_allocation_absolute_error'] for r in records if r['domain']==domain) for domain in ('radio','v5')},
        max_step_balance_absolute_error_by_domain={domain:max(r['maximum_step_balance_absolute_error'] for r in records if r['domain']==domain) for domain in ('radio','v5')},
        maximum_negative_allocation_violation=max(r['maximum_negative_allocation_violation'] for r in records),
        maximum_per_object_cap_violation=max(r['maximum_per_object_cap_violation'] for r in records),
        maximum_completed_object_allocation=max(r['maximum_completed_object_allocation'] for r in records),
        cases=records,identity=identity,
        copied_core_source_sha256=native.verify_sources()['full_spend_core_sha256'],
        script_sha256=native.digest(__file__))
    (OUT/'shared_bridge_acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','actual_environment_steps','training_updates','all_paths_full_spend','max_relative_allocation_error')},ensure_ascii=False))


if __name__=='__main__':
    try:
        run()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'failure.json').write_text(json.dumps({'status':'FAIL','traceback':traceback.format_exc()},indent=2)+'\n')
        raise
