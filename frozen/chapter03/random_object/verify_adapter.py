"""Synthetic-only adapter checks: no Task, real model, environment or optimizer."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def module(name, p):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, p)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
    return sys.modules[name]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    policy = module('_chapter03_random_object_policy', HERE/'policy.py')
    fixtures = module('_chapter03_core_random_verify_fixtures', policy.CORE/'verify_adapter.py')
    import numpy as np
    import torch
    from allocation23.spec import Spec
    from allocation23.methods.constraints import gaussian_logprob, capped_simplex
    from allocation23.io import state_hash
    torch.set_num_threads(1)
    root = Path('/hy-tmp/resource_allocation_results/chapter03')
    models = json.loads((root/'models/index.json').read_text())['models']
    records = {(x['domain'], x['seed']): x for x in models if x['method']=='ReJoP'}
    outputs, constructors, states = [], 0, 0
    for name in ('communication', 'emergency_diesel'):
        for seed in (0, 1):
            protocol = json.loads((root/records[name,seed]['training_records']/'protocol.json').read_text())
            spec = Spec(**{f.name: protocol['spec'][f.name] for f in dataclasses.fields(Spec)})
            global_torch = torch.get_rng_state().clone()
            global_numpy = copy.deepcopy(np.random.get_state())
            global_python = random.getstate()
            actor = policy.RandomObjectActor(spec, seed)
            equal = policy.core.NoObjectActor(spec, seed)
            constructors += 2
            assert state_hash(actor) == state_hash(equal)
            assert actor.event_dim == 1 and actor.spatial.event_dim == 0
            assert not hasattr(actor.spatial, 'head')
            assert all(not k.startswith('spatial.head.') for k, _ in actor.named_parameters())
            before_hash = state_hash(actor)
            checked = []
            for state in fixtures.cases(spec):
                states += 1
                actor._object_rng = np.random.default_rng(71491)
                result = actor.sample(state, deterministic=True)
                first_actions = result['actions'].copy()
                first_uniforms = result['object_random_uniforms'].copy()
                actor._object_rng = np.random.default_rng(71491)
                repeated = actor.sample(state, deterministic=True)
                np.testing.assert_array_equal(first_actions, repeated['actions'])
                np.testing.assert_array_equal(first_uniforms, repeated['object_random_uniforms'])
                actor._object_rng = np.random.default_rng(71492)
                changed = actor.sample(state, deterministic=True)
                assert not np.array_equal(first_uniforms, changed['object_random_uniforms'])
                reference = equal.sample(state, deterministic=True)
                torch.testing.assert_close(result['raw'], reference['raw'], rtol=0, atol=0)
                torch.testing.assert_close(result['quota'], reference['quota'], rtol=0, atol=0)
                torch.testing.assert_close(result['log_prob'], gaussian_logprob(result['raw'],
                    actor.quota_mean(state), actor.quota_sigma), rtol=0, atol=0)
                action, total = first_actions[0], float(result['quota'][0])
                assert np.all(action >= 0.) and np.all(action <= state['caps']+1e-14)
                assert abs(action.sum()-total) <= max(spec.action_tolerance, 1e-14)
                if spec.study_domain == 'radio':
                    assert np.all(action[state['caps']==0.] == 0.)
                active = state['caps'] > 0.
                if total > 0.:
                    weights = np.where(active, np.maximum(first_uniforms[0], np.finfo(float).tiny), 0.)
                    expected = capped_simplex(total*weights/weights.sum(), state['caps'], total)
                    np.testing.assert_array_equal(action, expected)
                else:
                    np.testing.assert_array_equal(action, np.zeros(spec.groups))
                checked.append(dict(period=int(round(state['global'][0]*spec.horizon)),
                    total=total, quota_equal_to_native_initial=True,
                    random_draw_reproducible=True, distinct_seed_changes_draw=True,
                    physical_allocation_valid=True))
            actor._object_rng = None
            state = fixtures.synthetic(spec)
            eta = actor.quota_mean(state).detach()+.3
            lp = actor.log_prob(state, eta)
            params = list(actor.quota_head.parameters())+list(actor.spatial.context_net.parameters())
            gradients = torch.autograd.grad(lp.sum(), params)
            assert all(x is not None and torch.isfinite(x).all() for x in gradients)
            assert any(torch.any(x != 0) for x in gradients)
            assert state_hash(actor) == before_hash

            class FakeTask:
                domain = spec.study_domain
                path_master = 202611043101
                def __init__(self, seed, nested=False):
                    self.seed, self.nested = seed, nested
                def episode(self, phase, index, owner, **kwargs):
                    first = owner.sample(state, deterministic=True)
                    if self.nested:
                        outer_rng = owner._object_rng
                        old = copy.deepcopy(outer_rng.bit_generator.state)
                        inner = FakeTask(991)
                        policy.episode_with_random_object(inner, 'dev', 3, owner)
                        assert owner._object_rng is outer_rng
                        assert owner._object_rng.bit_generator.state == old
                    second = owner.sample(state, deterministic=True)
                    return dict(actions=np.concatenate((first['actions'], second['actions'])))

            baseline = policy.episode_with_random_object(FakeTask(seed), 'train', 2, actor)
            nested = policy.episode_with_random_object(FakeTask(seed, nested=True), 'train', 2, actor)
            np.testing.assert_array_equal(baseline['actions'], nested['actions'])
            np.testing.assert_array_equal(baseline['random_object_uniforms'], nested['random_object_uniforms'])
            left = policy.episode_with_random_object(FakeTask(0), 'dev', 4, actor)
            right = policy.episode_with_random_object(FakeTask(4), 'dev', 4, actor)
            np.testing.assert_array_equal(left['random_object_uniforms'], right['random_object_uniforms'])
            different = policy.episode_with_random_object(FakeTask(seed), 'train', 3, actor)
            assert not np.array_equal(baseline['random_object_uniforms'], different['random_object_uniforms'])
            assert actor._object_rng is None and actor._object_rng_identity is None and actor._object_draw_trace is None
            assert torch.equal(global_torch, torch.get_rng_state())
            now = np.random.get_state()
            assert global_numpy[0] == now[0] and np.array_equal(global_numpy[1], now[1]) and global_numpy[2:] == now[2:]
            assert global_python == random.getstate()
            assert state_hash(actor) == before_hash
            outputs.append(dict(domain=spec.study_domain, seed=seed, states=checked,
                exact_native_no_object_initial_tensors=True, eta_only_logprob=True,
                finite_eta_gradient_through_retained_context=True, no_object_head=True,
                nested_DEV_restores_outer_object_rng_by_reference=True,
                DEV_random_object_stream_common_across_training_seeds=True,
                global_torch_numpy_python_RNG_unchanged=True,
                actor_tensors_unchanged=True))
    train = module('_chapter03_random_object_verify_train', HERE/'train.py')
    namespace, identity = train.derive_worker()
    assert namespace['ARMS'] == {policy.ARM: dict(variant='A_FULL', temporal='adaptive', quota_reference='pace')}
    assert namespace['SCHEMA'] == policy.SCHEMA
    assert states <= 24
    return dict(schema='chapter3-random-object-synthetic-adapter-verification-v1', status='PASS',
        domains_seeds=outputs, synthetic_state_cases=states, synthetic_actor_constructors=constructors,
        worker_derivation=identity, source_sha256={n:sha(HERE/n) for n in ('policy.py','train.py','verify_adapter.py')},
        random_namespace=policy.RANDOM_NAMESPACE, environment_constructions=0, environment_steps=0,
        real_model_payload_loads=0, synthetic_optimizer_updates=0, new_TEST_paths=0)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.output).resolve()
    if not path.is_relative_to(Path('/hy-tmp')) or path.exists():
        raise ValueError('A new HY synthetic-verification receipt is required')
    result = verify()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','synthetic_state_cases', 'synthetic_actor_constructors',
        'environment_steps','real_model_payload_loads','synthetic_optimizer_updates')}))


if __name__ == '__main__':
    main()
