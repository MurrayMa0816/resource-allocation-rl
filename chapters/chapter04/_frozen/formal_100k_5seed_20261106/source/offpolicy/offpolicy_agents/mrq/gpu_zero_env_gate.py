"""Independent actual-CUDA MR.Q gate, with synthetic transitions and zero Env calls.

No author or adapter learning method is replaced.  Three learners per domain
exercise the actual first 250-step encoder burst followed by two native RL
updates: adapter, direct author implementation, direct author repeat.  A repeat
mismatch is reported as native nondeterminism, never hidden by a CPU result.
The module's --dry-run path imports only the standard library and uses no GPU.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
SCOPE = Path('/hy-tmp/resource_allocation_results/chapter4/mrq_reserve_validation_20261106')
PIN = '073adcd6b30f84694c241e00bd684db50242a215'
GPU_UUID = 'GPU-a3130b13-8288-caaa-0f96-f2916c92d8e3'
DOMAINS = [('radio', 305, 19, 32, 9698), ('v5', 244, 13, 24, 9601)]
NETS = ['encoder', 'encoder_target', 'policy', 'policy_target', 'value', 'value_target']
OPTS = ['encoder_optimizer', 'value_optimizer', 'policy_optimizer']


def source_hashes():
    return {p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(HERE.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def actual_run(args, result):
    # Intentionally reached only without --dry-run.  Do not move Torch imports
    # above this function: preparation/review must not create a CUDA context.
    import numpy as np
    import torch
    from types import MethodType
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    sys.path.insert(0, str(HERE.parent))
    from mrq.agent import (Agent, ObservedReplay, PRIORITY_EXECUTION_PROFILE,
                           priority_write_determinism, _hash, native)
    from mrq.native_loader import verify_vendor

    checks = result['checks']

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    def rng():
        return dict(numpy=copy.deepcopy(np.random.get_state()),
                    torch_cpu=torch.get_rng_state().clone(),
                    torch_cuda=[x.clone() for x in torch.cuda.get_rng_state_all()])

    def restore_rng(state):
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch_cpu'])
        torch.cuda.set_rng_state_all(state['torch_cuda'])

    def seed(value):
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu(v) for v in value)
        return copy.deepcopy(value)

    def compare(a, b):
        """Bitwise comparison, with bounded numeric diagnostics on any mismatch."""
        differences = []
        tensors = 0

        def walk(x, y, path):
            nonlocal tensors
            if type(x) is not type(y):
                differences.append(dict(path=path, reason='type'))
            elif isinstance(x, torch.Tensor):
                tensors += 1
                if x.dtype != y.dtype or x.shape != y.shape:
                    differences.append(dict(path=path, reason='shape_or_dtype'))
                elif x.contiguous().numpy().tobytes() != y.contiguous().numpy().tobytes():
                    entry = dict(path=path, reason='tensor_values',
                                 unequal_elements=int((x != y).sum()))
                    if x.numel():
                        delta = (x.to(torch.float64) - y.to(torch.float64)).abs()
                        if bool(torch.isfinite(delta).all()):
                            entry['max_abs_difference'] = float(delta.max())
                        else:
                            entry['nonfinite_difference'] = True
                    differences.append(entry)
            elif isinstance(x, np.ndarray):
                if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                    differences.append(dict(path=path, reason='numpy_values'))
            elif isinstance(x, dict):
                if x.keys() != y.keys():
                    differences.append(dict(path=path, reason='keys'))
                else:
                    for k in x:
                        walk(x[k], y[k], path + '/' + str(k))
            elif isinstance(x, (list, tuple)):
                if len(x) != len(y):
                    differences.append(dict(path=path, reason='length'))
                else:
                    for i, (xx, yy) in enumerate(zip(x, y)):
                        walk(xx, yy, path + '/' + str(i))
            elif x != y:
                differences.append(dict(path=path, reason='scalar_value'))

        walk(a, b, '')
        return dict(exact=not differences, tensors_compared=tensors,
                    difference_count=len(differences), first_differences=differences[:12],
                    reference_sha256=_hash(a), candidate_sha256=_hash(b))

    result['torch_version'] = torch.__version__
    result['torch_cuda_version'] = torch.version.cuda
    result['cudnn_version'] = torch.backends.cudnn.version()
    result['runtime_environment'] = {k: os.environ.get(k) for k in
        ['CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'OMP_NUM_THREADS',
         'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG']}
    result['numeric_flags'] = dict(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        changed_globally_by_gate=False,
        strict_determinism_temporarily_enabled_only_for_native_priority_write=True)
    result['determinism_control'] = dict(
        profile=PRIORITY_EXECUTION_PROFILE,
        scope='native ReplayBuffer.update_priority call only',
        global_deterministic_algorithms_changed=False,
        native_cumsum_unchanged=True, native_priority_formula_and_assignment_unchanged=True,
        direct_native_reference_uses_same_scoped_profile=True,
        full_stack_default_CUDA_reproducibility_claimed=False,
        CUBLAS_WORKSPACE_CONFIG_changed=False)
    check('vendor_blobs_exact', verify_vendor() == PIN)
    check('visible_device_contract', os.environ.get('CUDA_VISIBLE_DEVICES') == args.expected_gpu_uuid)
    check('actual_CUDA_available', torch.cuda.is_available())
    check('exactly_one_visible_CUDA_device', torch.cuda.device_count() == 1)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    # Query physical identity at actual execution, not while preparing the code.
    query = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader',
                            '--id=' + args.expected_gpu_uuid], check=True,
                           text=True, capture_output=True, timeout=15)
    identity = query.stdout.strip()
    result['physical_gpu_uuid'] = identity
    result['torch_device_uuid'] = str(getattr(properties, 'uuid', 'unavailable_in_this_torch'))
    result['device_name'] = properties.name
    result['device_total_memory_bytes'] = int(properties.total_memory)
    check('actual_physical_GPU_UUID', identity == args.expected_gpu_uuid)
    check('actual_cuda_tensor_device', str(torch.zeros(1, device=device).device) == 'cuda:0')
    result['CUDA_used'] = True
    free, total = torch.cuda.mem_get_info(device)
    result['memory_at_entry'] = dict(free_bytes=free, total_bytes=total)
    check('minimum_free_memory_before_gate', free >= args.minimum_free_gib * (1024 ** 3))
    torch.cuda.reset_peak_memory_stats(device)

    # Independent CUDA formulas: terminal rewards are not propagated past done;
    # these are not calls to a physical environment or learned-value references.
    reward = torch.tensor([[[0.], [.7], [9.]], [[0.], [0.], [.9]],
                           [[.2], [.3], [.4]]], device=device)
    mask = torch.tensor([[[1.], [0.], [1.]], [[1.], [1.], [0.]],
                         [[1.], [1.], [1.]]], device=device)
    actual, discount = native.multi_step_reward(reward, mask, .99)
    expected = torch.tensor([[.99 * .7], [.99 ** 2 * .9],
                             [.2 + .99 * .3 + .99 ** 2 * .4]], device=device)
    check('CUDA_three_step_true_terminal_mask',
          torch.allclose(actual, expected, atol=1e-7, rtol=0) and
          torch.allclose(discount, torch.tensor([[0.], [0.], [.99 ** 3]], device=device),
                         atol=1e-7, rtol=0))
    pred = torch.tensor([[2., 3.], [1000., 1000.]], device=device)
    target = torch.ones(2, 2, device=device)
    check('CUDA_masked_dynamics_after_terminal',
          bool(native.masked_mse(pred, target, torch.tensor([[1.], [0.]], device=device)) == 1.25))
    two = native.TwoHot(device, -10, 10, 65)
    targets = torch.tensor([[0.], [.3], [.7], [1.]], device=device)
    hot = two.transform(targets)
    check('CUDA_two_hot_conserves_mass',
          torch.allclose(hot.sum(1), torch.ones(4, device=device), atol=1e-7, rtol=0))
    check('CUDA_two_hot_expected_raw_reward',
          torch.allclose((hot * two.bins).sum(1, keepdim=True), targets, atol=1e-7, rtol=0))

    # Explicit engineering repair: the adapter defines torch.empty's invalid
    # priority slots as zero. This consumes no RNG and preserves the native
    # finite, mask-weighted sampling distribution. NaN*0 is a real counterexample
    # to treating unspecified torch.empty values as mathematically zero.
    initial_rng = rng()
    fixed = ObservedReplay((3,), 2, 1., False, device, history=1,
                           horizon=5, max_size=512, batch_size=256)
    check('CUDA_priority_initialization_consumes_zero_RNG', _hash(initial_rng) == _hash(rng()))
    check('CUDA_adapter_invalid_priorities_initially_zero_and_finite',
          bool((fixed.priority == 0).all()) and bool(torch.isfinite(fixed.priority).all()))
    # A small independent replay fixture checks sampling/masking and the exact
    # LAP priority formula, without shrinking any learner's native replay.
    replay = native.buffer.ReplayBuffer((3,), 2, 1., False, device, history=1,
                                      horizon=5, max_size=512, batch_size=256)
    replay.priority.zero_()  # Defined fixture tail, not an algorithm/source change.
    for i in range(32):
        done = (i + 1) % 8 == 0
        replay.add(np.full(3, i), np.zeros(2), np.full(3, i + 1),
                   .7 if done else 0., done, False)
        fixed.add(np.full(3, i), np.zeros(2), np.full(3, i + 1),
                  .7 if done else 0., done, False)
    check('CUDA_priority_fix_preserves_native_valid_transition_priorities',
          torch.equal(replay.priority[replay.mask.bool()], fixed.priority[fixed.mask.bool()]) and
          torch.equal(replay.mask, fixed.mask))
    replay.priority[replay.mask == 0] = float('nan')
    invalid_cumulative = torch.cumsum(replay.priority * replay.mask, 0)
    check('CUDA_NaN_times_zero_tail_counterexample_is_reproduced',
          bool(torch.isnan(invalid_cumulative).any()))
    replay.priority[replay.mask == 0] = 0.
    check('CUDA_priority_fix_preserves_finite_masked_sampling_distribution',
          torch.equal(torch.cumsum(replay.priority * replay.mask, 0),
                      torch.cumsum(fixed.priority * fixed.mask, 0)) and
          bool(torch.isfinite(torch.cumsum(fixed.priority * fixed.mask, 0)).all()))
    sample_rng = rng()
    native_indices = replay.sample_ind().copy()
    after_native_rng = rng()
    restore_rng(sample_rng)
    fixed_indices = fixed.sample_ind().copy()
    check('CUDA_priority_fix_same_RNG_same_native_sample_indices',
          np.array_equal(native_indices, fixed_indices) and _hash(after_native_rng) == _hash(rng()))
    # The production guard must restore enabled AND warn_only even if the
    # enclosed operation fails. Exercise all original flag pairs; no cumsum or
    # matrix operation is moved under global strict mode.
    original_flags=(torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled())
    guard_cases=[]
    guard_priority=torch.linspace(1.,2.,256,device=device)
    fixed.sampled_ind=np.arange(256)
    try:
        for enabled in [False,True]:
            for warn_only in [False,True]:
                for inject_exception in [False,True]:
                    torch.use_deterministic_algorithms(enabled,warn_only=warn_only)
                    guard_rng=rng();record=None
                    try:
                        with priority_write_determinism() as record:
                            check('CUDA_priority_guard_strict_inside_'+str(len(guard_cases)),
                                  torch.are_deterministic_algorithms_enabled() and
                                  not torch.is_deterministic_algorithms_warn_only_enabled())
                            native.buffer.ReplayBuffer.update_priority(fixed,guard_priority)
                            if inject_exception:raise LookupError('intentional guard restoration fixture')
                    except LookupError:
                        if not inject_exception:raise
                    check('CUDA_priority_guard_complete_restoration_'+str(len(guard_cases)),
                          (torch.are_deterministic_algorithms_enabled(),
                           torch.is_deterministic_algorithms_warn_only_enabled())==(enabled,warn_only)
                          and record['fully_restored'])
                    check('CUDA_priority_guard_zero_RNG_'+str(len(guard_cases)),_hash(guard_rng)==_hash(rng()))
                    guard_cases.append(dict(**record,intentional_exception=inject_exception))
    finally:
        torch.use_deterministic_algorithms(original_flags[0],warn_only=original_flags[1])
    result['priority_guard_restoration_cases']=guard_cases
    eligible = torch.nonzero(replay.mask).reshape(-1).cpu().numpy()
    scripted = np.resize(eligible, 256)
    replay.sample_ind = MethodType(lambda self: scripted, replay)
    state, action, nxt, rewards, not_done = replay.sample(5, include_intermediate=True)
    check('CUDA_native_five_step_sample_shape',
          tuple(state.shape) == (256, 5, 3) and tuple(action.shape) == (256, 5, 2)
          and tuple(nxt.shape) == (256, 5, 3) and str(state.device) == 'cuda:0')
    check('CUDA_true_terminal_transition_retained',
          replay.env_terminates and bool((not_done[:, 0] == 0).any()))
    r3, d3 = native.multi_step_reward(rewards[:, :3], not_done[:, :3], .99)
    manual_r = torch.zeros_like(r3)
    manual_d = torch.ones_like(d3)
    for i in range(3):
        manual_r += manual_d * rewards[:, i]
        manual_d *= .99 * not_done[:, i]
    check('CUDA_native_replay_multistep_formula',
          torch.equal(r3, manual_r) and torch.equal(d3, manual_d))
    replay.sampled_ind = np.arange(256)
    Q = torch.arange(512, dtype=torch.float, device=device).reshape(256, 2) / 100
    Qt = torch.full((256, 1), .7, device=device)
    priority = (Q - Qt.expand(-1, 2)).abs().max(1).values.clamp(min=1).pow(.4)
    replay.update_priority(priority)
    check('CUDA_native_LAP_exact_priority_assignment',
          torch.equal(replay.priority[:256], priority) and replay.max_priority == float(priority.max()))
    result['independent_replay_fixture'] = dict(
        synthetic_transition_calls=32, physical_environment_steps=0,
        eligible_true_transition_slots=int(len(eligible)), sampling_batch=256,
        scripted_indices_for_terminal_coverage_only=True,
        optimizer_steps=0, capacity=512,
        explicit_priority_initialization_engineering_repair=True,
        fix_changes_valid_slot_priorities_or_sampling_math=False,
        native_reference_has_defined_invalid_priority_slots=True)
    del replay, fixed, state, action, nxt, rewards, not_done, r3, d3, manual_r, manual_d, Q, Qt, priority

    result['domains'] = []
    comparisons = []
    totals = dict(encoder=0, value=0, policy=0)
    result['actual_synthetic_optimizer_steps'] = totals

    for domain, obs, act, H, first in DOMAINS:
        domain_result = dict(domain=domain, observation_dim=obs, action_dim=act,
            true_terminal_horizon=H, physical_environment_steps=0,
            native_batch_size=256, native_encoder_width=512, native_value_width=512,
            native_policy_width=512, encoder_horizon=5, Q_horizon=3,
            native_replay_capacity=1000000, native_warmup_replay_slots=10000,
            fixture_priority_tail_zeroed=True, fixture_target_offset_before_hard_sync=.125,
            learners=[], comparisons=[])
        result['domains'].append(domain_result)
        learners = []
        for kind in ['adapter', 'direct_native', 'native_repeat']:
            seed(981)
            obj = Agent({'seed': 981, 'total_steps': 49152}, obs, act, device) if kind == 'adapter' else \
                native.Agent((obs,), act, 1., False, False, device)
            core = obj.core if kind == 'adapter' else obj
            buffer = core.replay_buffer
            check(domain + '_' + kind + '_actual_author_defaults',
                  core.batch_size == 256 and core.enc_horizon == 5 and core.Q_horizon == 3 and
                  core.enc_hdim == core.value_hdim == core.policy_hdim == 512 and
                  core.zs_dim == core.zsa_dim == 512 and core.za_dim == 256 and
                  core.target_update_freq == 250 and core.buffer_size_before_training == 10000 and
                  buffer.max_size == 1000000 and buffer.horizon == 5)
            check(domain + '_' + kind + '_all_six_actual_networks_on_CUDA',
                  all(p.device == device for name in NETS for p in getattr(core, name).parameters()))
            # Adapter now defines invalid priorities in ObservedReplay.__init__.
            # Give direct author references that same defined initial input;
            # do not modify vendor or native sampling/update mathematics.
            if kind == 'adapter':
                check(domain + '_adapter_full_native_capacity_priority_initialization',
                      bool((buffer.priority == 0).all()) and bool(torch.isfinite(buffer.priority).all()))
            else:
                buffer.priority.zero_()
            counts = dict(encoder=0, value=0, policy=0)
            sync = []
            priority_writes=[]
            original_priority_update=buffer.update_priority
            def traced_priority_write(this, incoming, original=original_priority_update,
                                      rows=priority_writes, reference=(kind!='adapter'),owner=core):
                indices=this.sampled_ind.copy()
                values=incoming.detach().cpu().numpy().reshape(-1).copy()
                unique,frequencies=np.unique(indices,return_counts=True)
                conflicts=[]
                for index in unique[frequencies>1]:
                    positions=np.flatnonzero(indices==index)
                    candidates=values[positions]
                    conflicts.append(dict(index=int(index),positions=positions.tolist(),
                        incoming_priorities=candidates.tolist(),
                        distinct_priority_count=int(len(np.unique(candidates)))))
                before_rng=rng()
                if reference:
                    with priority_write_determinism() as guard:out=original(incoming)
                else:
                    out=original(incoming);guard=this.last_priority_guard
                for conflict in conflicts:
                    conflict['written_priority']=float(this.priority[conflict['index']].detach().cpu())
                rows.append(dict(native_RL_clock=owner.training_steps,batch_size=int(len(indices)),
                    distinct_sample_indices=int(len(unique)),duplicate_sample_positions=int((frequencies-1).sum()),
                    conflicting_duplicate_index_count=sum(x['distinct_priority_count']>1 for x in conflicts),
                    duplicate_indices=conflicts,guard=dict(guard),
                    RNG_exact_across_priority_write=_hash(before_rng)==_hash(rng())))
                return out
            buffer.update_priority=MethodType(traced_priority_write,buffer)
            handles = []
            for component in counts:
                def counted(optimizer, unused_args, unused_kwargs, key=component, values=counts):
                    values[key] += 1
                    totals[key] += 1
                handles.append(getattr(core, component + '_optimizer').register_step_post_hook(counted))
            for component in ['encoder', 'value', 'policy']:
                def synced(module, unused_keys, key=component, records=sync, owner=core):
                    records.append(dict(component=key, native_RL_clock=owner.training_steps,
                        equal_at_hard_copy=_hash(module.state_dict()) ==
                        _hash(getattr(owner, key).state_dict())))
                handles.append(getattr(core, component + '_target').register_load_state_dict_post_hook(synced))
            # A deliberately different starting target detects a missing first
            # hard copy. It is fixture data, never a production initializer.
            with torch.no_grad():
                for name in ['encoder_target', 'value_target', 'policy_target']:
                    for p in getattr(core, name).parameters():
                        p.add_(.125)
            fixture_rng = np.random.RandomState(7081)
            calls = 0
            for step in range(1, first):
                o = fixture_rng.standard_normal(obs).astype(np.float32) * .2
                a = fixture_rng.uniform(-1., 1., act).astype(np.float32)
                n = fixture_rng.standard_normal(obs).astype(np.float32) * .2
                done = step % H == 0
                r = float(.5 + .2 * np.sin(step)) if done else 0.
                if kind == 'adapter':
                    obj.observe(o, a, r, n, done)
                else:
                    buffer.add(o, a, n, r, done, False)
                    core.train()
                calls += 1
            check(domain + '_' + kind + '_zero_optimizer_before_warmup',
                  core.training_steps == 0 and counts == dict(encoder=0, value=0, policy=0))
            learner = dict(kind=kind, obj=obj, core=core, buffer=buffer, counts=counts,
                           sync=sync, handles=handles, snapshots=[], fixture_rng=fixture_rng,
                           synthetic_transition_calls=calls,priority_writes=priority_writes)
            learners.append(learner)
            row = dict(kind=kind, optimizer_step_hooks=counts, target_hard_copy_events=sync,
                       synthetic_transition_calls=calls, optimizer_steps_during_prefill=0,
                       prefill_replay_slots=buffer.size, prefill_insert_index=buffer.ind,
                       priority_write_records=priority_writes,
                       actual_native_hyperparameters=dict(vars(core.hp)))
            learner['report'] = row
            domain_result['learners'].append(row)

        # size excludes the extra terminal-observation slot when the most
        # recent transition itself ended an episode. v5 has size9999/ind10000
        # here, whereas radio has size10000/ind10000. Both learn on the next
        # genuine synthetic transition (size10001); do not conflate the clocks.
        check(domain + '_actual_prefill_same_slot_clock',
              all(x['buffer'].size == (first - 1) + (first - 2) // H and
                  x['buffer'].ind == (first - 1) + (first - 1) // H for x in learners))
        initial = rng()
        for learner in learners:
            restore_rng(initial)
            core, obj, buffer = learner['core'], learner['obj'], learner['buffer']
            kind = learner['kind']
            start = time.perf_counter()
            for offset in range(2):
                step = first + offset
                fixture_rng = learner['fixture_rng']
                o = fixture_rng.standard_normal(obs).astype(np.float32) * .2
                a = fixture_rng.uniform(-1., 1., act).astype(np.float32)
                n = fixture_rng.standard_normal(obs).astype(np.float32) * .2
                done = step % H == 0
                r = float(.5 + .2 * np.sin(step)) if done else 0.
                if kind == 'adapter':
                    obj.observe(o, a, r, n, done)
                else:
                    buffer.add(o, a, n, r, done, False)
                    core.train()
                learner['synthetic_transition_calls'] += 1
                torch.cuda.synchronize(device)
                check(domain + '_' + kind + '_actual_first_RL_clock_' + str(offset + 1),
                      core.training_steps == offset + 1 and
                      (kind != 'adapter' or obj.first_learning_step == first))
                snap = dict(
                    parameters={name: cpu(getattr(core, name).state_dict()) for name in NETS},
                    Adam={name: cpu(getattr(core, name).state_dict()) for name in OPTS},
                    gradients={name: {k: cpu(v.grad) for k, v in getattr(core, name).named_parameters()}
                               for name in NETS},
                    RNG=cpu(rng()),
                    native_clock=dict(RL=core.training_steps, reward_scale=core.reward_scale,
                                      target_reward_scale=core.target_reward_scale),
                    replay=dict(size=buffer.size, index=buffer.ind, ep_timesteps=buffer.ep_timesteps,
                        env_terminates=buffer.env_terminates, history=list(buffer.history_queue),
                        max_priority=buffer.max_priority,
                        sampled_ind=cpu(buffer.sampled_ind), priority=cpu(buffer.priority), mask=cpu(buffer.mask),
                        obs_written=cpu(buffer.obs[:buffer.ind]),
                        action_reward_notdone_written=cpu(buffer.action_reward_notdone[:buffer.ind]),
                        state_indices=buffer.state_ind[:buffer.ind].copy(),
                        next_indices=buffer.next_ind[:buffer.ind].copy()),
                    actual_optimizer_step_hooks=dict(learner['counts']))
                learner['snapshots'].append(snap)
                check(domain + '_' + kind + '_real_update_counts_' + str(offset + 1),
                      learner['counts'] == dict(encoder=250, value=offset + 1, policy=offset + 1))
                for opt in OPTS:
                    states = getattr(core, opt).state_dict()['state'].values()
                    expected_step = 250 if opt == 'encoder_optimizer' else offset + 1
                    check(domain + '_' + kind + '_' + opt + '_Adam_clock_' + str(offset + 1),
                          len(getattr(core, opt).state_dict()['state']) > 0 and
                          all(int(s['step'].item()) == expected_step for s in states))
                    check(domain + '_' + kind + '_' + opt + '_Adam_finite_' + str(offset + 1),
                          all(bool(torch.isfinite(t).all())
                              for s in getattr(core, opt).state_dict()['state'].values()
                              for t in s.values() if isinstance(t, torch.Tensor)))
                for name in NETS:
                    check(domain + '_' + kind + '_' + name + '_finite_' + str(offset + 1),
                          all(bool(torch.isfinite(t).all()) for t in getattr(core, name).state_dict().values()))
                for name in NETS:
                    check(domain + '_' + kind + '_' + name + '_gradient_finite_' + str(offset + 1),
                          all(p.grad is None or bool(torch.isfinite(p.grad).all())
                              for p in getattr(core, name).parameters()))
            learner['report'].update(synthetic_transition_calls=learner['synthetic_transition_calls'],
                elapsed_seconds=time.perf_counter() - start,
                native_RL_clock=core.training_steps,
                observation_storage_device=str(buffer.storage_device),
                priority_storage_device=str(buffer.priority.device),
                actual_optimizer_steps=dict(learner['counts']),
                final_replay_slots=buffer.size, true_terminal_seen=buffer.env_terminates)
            check(domain + '_' + kind + '_one_actual_three_network_hard_sync',
                  len(learner['sync']) == 3 and
                  {x['component'] for x in learner['sync']} == {'encoder', 'value', 'policy'} and
                  all(x['native_RL_clock'] == 1 and x['equal_at_hard_copy'] for x in learner['sync']))
            check(domain+'_'+kind+'_both_priority_writes_strict_guarded_restored_without_RNG',
                  len(learner['priority_writes'])==2 and
                  all(x['guard']['strict_inside'] and x['guard']['fully_restored'] and
                      x['RNG_exact_across_priority_write'] for x in learner['priority_writes']))
            check(domain + '_' + kind + '_targets_unchanged_on_second_RL',
                  all(_hash(learner['snapshots'][0]['parameters'][name]) ==
                      _hash(learner['snapshots'][1]['parameters'][name])
                      for name in ['encoder_target', 'value_target', 'policy_target']))

        for offset in range(2):
            for label, left, right in [('adapter_vs_direct_native', 1, 0),
                                       ('direct_native_vs_native_repeat', 1, 2)]:
                a, b = learners[left]['snapshots'][offset], learners[right]['snapshots'][offset]
                row = dict(comparison=label, native_RL_clock=offset + 1,
                           full_state=compare(a, b), categories={})
                for category in ['parameters', 'Adam', 'gradients', 'RNG', 'native_clock', 'replay',
                                 'actual_optimizer_step_hooks']:
                    row['categories'][category] = compare(a[category], b[category])
                domain_result['comparisons'].append(row)
                comparisons.append(row)

        adapter = learners[0]['obj']
        check(domain + '_adapter_observed_native_sample_and_update_counts',
              adapter.replay.sample_calls == dict(encoder=250, value=2, other=0) and
              adapter.replay.priority_update_calls == 2 and adapter.core.target_syncs == 1)
        before = adapter.native_state_fingerprint()
        observation = np.zeros(obs, dtype=np.float32)
        action = adapter.policy_action(observation, True)
        check(domain + '_deterministic_deployment_and_DEV_no_mutation',
              np.array_equal(action, adapter.policy_action(observation, True)) and
              np.array_equal(action, adapter.act(observation, True)) and
              before == adapter.native_state_fingerprint())
        with tempfile.TemporaryDirectory(dir=args.output.parent, prefix='temporary_inference_') as temporary:
            endpoint = Path(temporary) / 'endpoint.pt'
            adapter.save(endpoint)
            disk = torch.load(endpoint, weights_only=True, map_location='cpu')
            check(domain + '_endpoint_contains_inference_only',
                  disk['optimizer_value_model_or_replay_included'] is False and
                  set(disk) == {'format', 'source_commit', 'observation_dim', 'action_dim', 'hp', 'seed',
                                'total_steps', 'steps', 'updates', 'state_encoder_state', 'policy_state',
                                'diagnostics', 'optimizer_value_model_or_replay_included'})
            loaded = Agent.load(endpoint, device)
            check(domain + '_actual_CUDA_inference_roundtrip_exact',
                  np.array_equal(action, loaded.policy_action(observation, True)) and
                  next(loaded.actor.parameters()).device == device)
            check(domain + '_loaded_endpoint_has_no_learning_state',
                  not hasattr(loaded, 'core') and not hasattr(loaded, 'replay') and not hasattr(loaded, 'critic'))
            rejected = False
            try:
                loaded.observe(observation, action, 0., observation, False)
            except RuntimeError:
                rejected = True
            check(domain + '_loaded_endpoint_rejects_training', rejected)
            check(domain + '_save_load_preserves_original_learner_and_RNG',
                  before == adapter.native_state_fingerprint())
            domain_result['temporary_inference_file_bytes'] = endpoint.stat().st_size
            temporary_path = str(endpoint)
            del loaded, disk
        check(domain + '_temporary_inference_file_removed', not Path(temporary_path).exists())
        for learner in learners:
            for handle in learner['handles']:
                handle.remove()
        # Release both native full-capacity CPU observations and actual CUDA
        # states between domains; no learning state is written to disk.
        del learner, learners, obj, core, buffer, adapter, a, b, snap, synced, counted, handles, handle
        del traced_priority_write,original_priority_update,states,p
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps(dict(domain=domain, phase='CUDA_math_complete',
                              physical_environment_steps=0, accumulated_optimizer_steps=totals)), flush=True)

    repeat = [x for x in comparisons if x['comparison'] == 'direct_native_vs_native_repeat']
    adapted = [x for x in comparisons if x['comparison'] == 'adapter_vs_direct_native']
    check('all_eight_actual_update_comparisons_present', len(repeat) == 4 and len(adapted) == 4)
    result['exact_comparisons'] = dict(
        native_repeat_all_full_state_exact=all(x['full_state']['exact'] for x in repeat),
        adapter_native_all_full_state_exact=all(x['full_state']['exact'] for x in adapted),
        parameters_and_targets=all(x['categories']['parameters']['exact'] for x in comparisons),
        Adam=all(x['categories']['Adam']['exact'] for x in comparisons),
        gradients=all(x['categories']['gradients']['exact'] for x in comparisons),
        RNG=all(x['categories']['RNG']['exact'] for x in comparisons),
        replay_and_LAP=all(x['categories']['replay']['exact'] for x in comparisons))
    check('actual_optimizer_counts_across_two_domains',
          totals == dict(encoder=1500, value=12, policy=12))
    check('no_Gym_Atari_environment_modules_imported',
          not any(k in sys.modules for k in ['gym', 'gymnasium', 'ale_py', 'env_preprocessing']))
    check('all_sources_unchanged_during_gate', source_hashes() == result['source_sha256'])
    result['memory_peak'] = dict(allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                                 reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                                 scope='this gate only; not a scientific-training capacity certificate')
    result['persistent_models'] = 0
    result['temporary_inference_files_remaining'] = []
    result['native_determinism_not_forced_or_repaired'] = False
    check('native_global_determinism_flags_restored_after_all_updates',
          (torch.are_deterministic_algorithms_enabled(),
           torch.is_deterministic_algorithms_warn_only_enabled())==original_flags)
    if not result['exact_comparisons']['native_repeat_all_full_state_exact']:
        result['status'] = 'FAIL_NATIVE_NONDETERMINISM'
    elif not result['exact_comparisons']['adapter_native_all_full_state_exact']:
        result['status'] = 'FAIL_ADAPTER_MISMATCH'
    else:
        result['status'] = 'PASS'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=SCOPE / 'preflight/cuda_attempt02/gate.json')
    parser.add_argument('--expected-gpu-uuid', default=GPU_UUID)
    parser.add_argument('--minimum-free-gib', type=float, default=6.)
    parser.add_argument('--dry-run', action='store_true',
                        help='stdlib-only contract check; does not import Torch or touch CUDA')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((SCOPE / 'preflight').resolve()):
        parser.error('gate output must stay in the independent MR.Q preflight scope')
    result = dict(schema='mrq-actual-cuda-zero-env-gate-v1', status='RUNNING',
                  physical_environment_steps=0, environments_constructed=0,
                  device='cuda:0', physical_gpu_uuid=None, author_commit=PIN,
                  source_root=str(HERE), source_sha256=source_hashes(),
                  checks={}, CUDA_used=False,
                  clock_only_optimizer_fixtures_used=False,
                  fixture_note='Synthetic replay.add calls and adapter.steps are fixture clocks, never real environment interactions.',
                  actual_synthetic_optimizer_steps=dict(encoder=0, value=0, policy=0),
                  production_hyperparameters_changed=False,
                  explicit_engineering_adaptations=[
                      'ObservedReplay.__init__: zero initially invalid priority slots; native reference receives same defined initial data'],
                  wall_start_unix=time.time())
    if args.dry_run:
        result.update(status='PREPARED_NOT_EXECUTED', CUDA_used=False,
                      Torch_imported='torch' in sys.modules,
                      CUDA_initialization_attempted=False,
                      expected_physical_gpu_uuid=args.expected_gpu_uuid,
                      expected_actual_optimizer_steps=dict(encoder=1500, value=12, policy=12),
                      actual_cuda_gate_still_required=True)
        write_json(args.output, result)
        print(json.dumps({k: result[k] for k in ['status', 'physical_environment_steps', 'CUDA_used']}))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    try:
        actual_run(args, result)
        result['CUDA_used'] = True
    except Exception as error:
        result['status'] = 'FAIL_RUNTIME_OR_ASSERTION'
        result['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        # Preserve actual partial execution evidence; a failing CUDA gate must
        # not be misreported as zero GPU work or zero optimizer calls.
        torch_module = sys.modules.get('torch')
        if torch_module is not None and torch_module.cuda.is_initialized():
            result['CUDA_used'] = True
            result['memory_peak'] = dict(
                allocated_bytes=int(torch_module.cuda.max_memory_allocated()),
                reserved_bytes=int(torch_module.cuda.max_memory_reserved()),
                scope='partial failing gate execution only')
    result['wall_elapsed_seconds'] = time.time() - result['wall_start_unix']
    result['source_sha256_after'] = source_hashes()
    write_json(args.output, result)
    print(json.dumps({k: result.get(k) for k in
        ['status', 'physical_environment_steps', 'device', 'physical_gpu_uuid',
         'torch_version', 'actual_synthetic_optimizer_steps', 'memory_peak', 'wall_elapsed_seconds']}), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
