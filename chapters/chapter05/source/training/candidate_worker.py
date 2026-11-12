"""Paired candidate repair screen; native Full task, fixed endpoint, real512 DEV."""
from pathlib import Path
import argparse, copy, gzip, importlib, json, os, random, sys, time, traceback
import numpy as np
import torch

CHAPTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHAPTER / '_internal/src'))
sys.path.insert(0, str(CHAPTER))
from chapter5_full_spend_20261104.env import Env
from baselines._support.common import sha, write_json, cpu_state, digest_state, finite_stats
from baselines._support.runtime import line, rng_state, restore_rng, parameter_devices, optimizer_cuda_count
from baselines._support.normalization import wrap_off_policy
from baselines._support.replay import NStepReplay
from baselines._support.update_clock import physical_update_target


def make_agent(cfg, obs_dim, action_dim):
    module = importlib.import_module('baselines.' + cfg['method'])
    agent = module.Agent(obs_dim, action_dim, cfg['seed'], cfg['device'], cfg)
    if not getattr(agent, 'on_policy', False) and not getattr(agent, 'manages_observation_normalization', False):
        agent = wrap_off_policy(agent, obs_dim, cfg)
    return agent


def source_identity(cfg):
    manifest = json.loads(Path(cfg['source_manifest']).read_text())
    for name, expected in manifest['files'].items():
        if sha(CHAPTER / name) != expected:
            raise RuntimeError('Frozen candidate source changed: ' + name)
    return sha(cfg['source_manifest'])


def parameter_hash(agent):
    states = {}; seen = set()
    def visit(value, path):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.nn.Module):
            states[path] = {k: v.detach() for k, v in value.named_parameters()}
        elif isinstance(value, (list, tuple, dict)):
            for key, child in (value.items() if isinstance(value, dict) else enumerate(value)):
                visit(child, path + '/' + str(key))
        elif hasattr(value, '__dict__') and not isinstance(value, (np.ndarray, torch.Tensor, torch.optim.Optimizer)):
            for key, child in vars(value).items():
                visit(child, path + '/' + str(key))
    visit(agent, 'agent')
    return digest_state(states)


def rows_of(results, phase):
    rows = []
    for r in results:
        row = {key: r[key] for key in ('domain','index','plan_sha256','tau','N','K','C','J',
            'native_return','total_used','remaining','full_spend')}
        row['phase'] = phase
        row['initial_budget'] = float(r['total_used']) + float(r['remaining'])
        if phase != 'train':
            row['quality'] = np.asarray(r['quality']).tolist()
            row['period_totals'] = np.asarray(r['period_totals']).tolist()
        rows.append(row)
    return rows


def endpoint_state(agent):
    exporter = getattr(agent, 'checkpoint_state_dict', None) or getattr(agent, 'endpoint', None)
    return cpu_state(exporter() if exporter else agent.state_dict())


def run(config_path):
    cfg = json.loads(Path(config_path).read_text()); out = Path(cfg['output']).expanduser().resolve()
    if out.exists():
        raise FileExistsError('Do not overwrite or recover a screening job: ' + str(out))
    if not out.is_relative_to('/hy-tmp'):
        raise ValueError('All new scientific outputs must be on HY')
    total, ne, seed = int(cfg['learning_steps']), int(cfg.get('n_env', 1)), int(cfg['seed'])
    if total % 512 or 512 % ne or cfg['dev_paths'] < 1:
        raise ValueError('Exact512 evaluation geometry required')
    device = torch.device(cfg['device'])
    if device.type != 'cuda' and not cfg.get('engineering'):
        raise ValueError('Scientific candidate screen requires native CUDA updates')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    random.seed(seed + 91000); np.random.seed(seed + 92000); torch.manual_seed(seed + 93000)
    rng = np.random.default_rng(seed + 94000)
    if device.type == 'cuda':
        if device.index != 0 or torch.cuda.device_count() != 1:
            raise ValueError('One scheduler-assigned GPU UUID, exposed as cuda:0, required')
        torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats()
        probe = torch.ones((8, 8), device=device); assert float((probe @ probe)[0, 0]) == 8; del probe
    original_rng = rng_state(None)
    try:
        def env(phase, master):
            return Env(cfg['domain'], seed, phase=phase, path_master=master,
                       action_mode=cfg['action_mode'], data_root=cfg['data_root'])
        pool = [env('train', cfg['train_master']) for _ in range(ne)]
        dev = [env('dev', cfg['dev_master']) for _ in range(cfg['dev_paths'])]
        internal = ([env('train', cfg['train_master']) for _ in range(5)]
                    if cfg.get('morld_psa_value_source') == 'charged_internal' else [])
    finally:
        restore_rng(original_rng)
    h = pool[0].horizon
    if total % (h * ne):
        raise ValueError('Complete native episodes required')
    agent = make_agent(cfg, pool[0].observation_dim, pool[0].action_dim)
    if cfg['action_mode'] != agent.action_mode:
        raise ValueError('Learner and public latent coordinates disagree')
    onpolicy = bool(agent.on_policy)
    if cfg['method'] == 'morld' and ne != 1:
        raise ValueError('The registered MORL/D exchange adaptation uses one complete episode')
    nstep = int(cfg.get('n_step', 1))
    if nstep != 1 and not cfg.get('acknowledge_nstep_task_variant'):
        raise ValueError('Original-reward n-step task adaptation must be declared')
    replay = None if onpolicy else NStepReplay(total, pool[0].observation_dim,
        pool[0].action_dim, n_step=nstep, n_env=ne)
    source_sha = source_identity(cfg)
    initial = digest_state(agent.state_dict()); initial_params = parameter_hash(agent)
    devices = parameter_devices(agent)
    if not devices or devices != [str(device)]:
        raise RuntimeError('Neural parameters are not on the registered device: ' + repr(devices))
    out.mkdir(parents=True)
    write_json(out / 'protocol.json', dict(config=cfg, config_sha256=sha(config_path),
        native=pool[0].protocol, implementation=agent.metadata(), source_manifest_sha256=source_sha,
        initial_model_sha256=initial, initial_parameters_sha256=initial_params,
        full_spend_required=True, evaluation_grid=list(range(0, total + 1, 512)),
        holdout_paths=0, dev_used_only_for_screening_not_formal_ranking=True,
        environment_construction_all_global_rng_isolated=True,
        auxiliary_one_step_label_retained=True, replay_n_step=nstep,
        n_step_is_uncorrected_original_reward_task_adaptation=nstep != 1,
        gpu_uuid=os.environ.get('CH5_GPU_UUID'), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        endpoint='one fixed endpoint; no periodic model, disk replay or optimizer sidecar'))
    counts = {k: dict(physical_steps=0, episodes=0) for k in ('train','development','internal')}
    learning = updates = episode = next_eval = extra_updates = 0
    calibrated_reference_taken = False
    started = time.monotonic(); last_progress = 0.; gpu_events = []; timed_calls = 0

    def progress(phase):
        nonlocal last_progress
        if phase == 'training' and time.monotonic() - last_progress < 3:
            return
        row = dict(phase=phase, pid=os.getpid(), learning_steps=learning, target=total,
            updates=updates, extra_exchange_updates=extra_updates, counts=copy.deepcopy(counts),
            seconds=time.monotonic()-started)
        if device.type == 'cuda':
            row.update(peak_reserved=torch.cuda.max_memory_reserved(), allocated=torch.cuda.memory_allocated())
        write_json(out / 'progress.json', row); last_progress = time.monotonic()

    def timed_update(call):
        nonlocal timed_calls
        timed_calls += 1
        if device.type == 'cuda' and (timed_calls <= 32 or timed_calls % 128 == 0):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); stats = call(); end.record(); gpu_events.append((begin, end))
        else:
            stats = call()
        return finite_stats(stats)

    with gzip.open(out/'development.jsonl.gz','wt',compresslevel=1) as fe, \
         gzip.open(out/'training.jsonl.gz','wt',compresslevel=1) as ft, \
         gzip.open(out/'updates.jsonl.gz','wt',compresslevel=1) as fu, \
         gzip.open(out/'internal.jsonl.gz','wt',compresslevel=1) as fi:
        def evaluate(kind='curve'):
            nonlocal next_eval
            assert learning == next_eval
            progress('development'); saved = rng_state(agent)
            before = digest_state(agent.state_dict())
            observations = np.stack([e.reset(options={'episode_index':j})[0] for j,e in enumerate(dev)])
            preferences = np.tile([.8,.2], (len(dev),1)).astype(np.float32)
            try:
                for _ in range(h):
                    actions = agent.act(observations, preferences, deterministic=True)
                    data = [e.step(a) for e,a in zip(dev,actions)]
                    observations = np.stack([x[0] for x in data])
                    counts['development']['physical_steps'] += len(dev)
            finally:
                restore_rng(saved)
            if before != digest_state(agent.state_dict()):
                raise RuntimeError('DEV changed a learning network, normalizer or population state')
            rows = rows_of([e.terminal_result for e in dev], 'development')
            counts['development']['episodes'] += len(dev)
            line(fe, dict(kind=kind, learning_steps=learning, updates=updates, rows=rows,
                mean_K=float(np.mean([r['K'] for r in rows])),
                mean_J=float(np.mean([r['J'] for r in rows])),
                mean_F=float(np.mean([.8*r['C']+.2*r['J'] for r in rows])),
                model_sha256=before, rng_and_learning_state_unchanged=True))
            fe.flush()
            if kind == 'curve':
                next_eval += 512
            progress('training')

        def internal_evaluation():
            nonlocal learning
            if not internal or total-learning < 3*len(internal)*h or not agent.needs_internal_evaluation(learning):
                return
            saved = rng_state(agent); before = digest_state(agent.state_dict()); values = []
            packet = counts['internal']['episodes'] // (3*len(internal))
            result_rows = []
            try:
                for policy in range(3):
                    for j,e in enumerate(internal):
                        obs_i = e.reset(options={'episode_index':1000000+packet*len(internal)+j})[0][None]
                        for tick in range(h):
                            # Internal PSA uses the same train-fitted observation map.
                            native = agent.native if hasattr(agent, '_observations') else agent
                            obs_native = agent._observations(obs_i) if hasattr(agent, '_observations') else obs_i
                            action = native.act_policy(policy, obs_native, True)[0]
                            item = e.step(action); obs_i = item[0][None]; learning += 1
                            counts['internal']['physical_steps'] += 1
                            counts['internal']['episodes'] += int(item[2])
                            last = policy == 2 and j == len(internal)-1 and tick == h-1
                            if learning == next_eval and not last:
                                evaluate()
                    rows = rows_of([e.terminal_result for e in internal], 'internal')
                    result_rows.extend(dict(policy=policy, **row) for row in rows)
                    values.append(np.mean([[r['C'],r['J']] for r in rows], axis=0))
            finally:
                restore_rng(saved)
            if before != digest_state(agent.state_dict()):
                raise RuntimeError('PSA rollout changed the learner before acceptance')
            stats = agent.accept_internal_evaluation(np.asarray(values), learning)
            line(fi, dict(learning_steps=learning, rows=result_rows, values=values, stats=stats)); fi.flush()
            if learning == next_eval:
                evaluate()

        evaluate()
        while learning < total:
            if total-learning < h*ne:
                raise RuntimeError('Charged internal interaction accounting left an incomplete training episode')
            obs = np.stack([e.reset(options={'episode_index':episode+j})[0] for j,e in enumerate(pool)])
            if hasattr(agent, 'select_policy'):
                agent.select_policy(learning)
            if hasattr(agent, 'train_preferences'):
                weights = agent.train_preferences(learning, total, rng)
            elif hasattr(agent, 'sample_weights'):
                weights = agent.sample_weights(ne, rng)
            else:
                weights = np.tile([.8,.2], (ne,1)).astype(np.float32)
            trajectory = {k:[] for k in ('obs','actions','logp','values','rewards','dones')}
            for tick in range(h):
                if getattr(agent, 'resample_preference_each_step', False):
                    weights = agent.sample_weights(ne, rng)
                if onpolicy:
                    sampled = agent.sample_info(obs, weights); actions = np.asarray(sampled['actions'])
                else:
                    actions = (rng.uniform(-1,1,(ne,pool[0].action_dim)).astype(np.float32)
                        if counts['train']['physical_steps'] < cfg['start_steps'] else agent.act(obs, weights, False))
                if hasattr(agent, 'observe_training'):
                    agent.observe_training(obs)
                data = [e.step(a) for e,a in zip(pool,actions)]
                nxt = np.stack([x[0] for x in data]); rewards = np.stack([x[1] for x in data])
                dones = np.asarray([x[2] for x in data], np.float32)[:,None]
                assert all(x[2] == (tick == h-1) and not x[3] and not x[4] for x in data)
                assert tick == h-1 or not rewards.any()
                learning += ne; counts['train']['physical_steps'] += ne
                counts['train']['episodes'] += int(dones.sum())
                if onpolicy:
                    for key,val in dict(obs=obs,actions=actions,logp=sampled['logp'],values=sampled['values'],rewards=rewards,dones=dones).items():
                        trajectory[key].append(np.asarray(val).copy())
                else:
                    if hasattr(agent, 'set_physical_step'):
                        agent.set_physical_step(learning)
                    replay.add(obs=obs, actions=actions, weights=np.asarray(weights,np.float32),
                               rewards=rewards, next_obs=nxt, dones=dones)
                    if replay.size >= cfg['batch_size'] and counts['train']['physical_steps'] >= cfg['learning_starts']:
                        if (updates == 0 and cfg.get('observation_normalization')
                                and not calibrated_reference_taken):
                            if learning != next_eval:
                                raise RuntimeError('The calibrated reference must use an existing512 tick')
                            evaluate(kind='frozen_map_before_first_update')
                            calibrated_reference_taken = True
                        target = physical_update_target(counts['train']['physical_steps'], cfg['learning_starts'], cfg['batch_size'], cfg['updates_per_step'])
                        while updates < target:
                            stats = timed_update(lambda:agent.update(replay.sample(cfg['batch_size'], rng)))
                            updates += 1; line(fu, dict(stats, learning_steps=learning, worker_updates=updates))
                obs = nxt
                if learning == next_eval and tick < h-1:
                    evaluate()
                progress('training')
            rows = rows_of([e.terminal_result for e in pool], 'train')
            for row in rows:
                line(ft, dict(learning_steps=learning, **row))
            if onpolicy:
                stats = timed_update(lambda:agent.update_rollout({k:np.stack(v) for k,v in trajectory.items()}, weights, learning))
                updates += int(stats.get('update_groups',stats.get('optimizer_updated',True)))
                line(fu, dict(stats, learning_steps=learning, worker_updates=updates))
            elif hasattr(agent, 'observe_episode'):
                returns = np.asarray([[r['C'],r['J']] for r in rows])
                stats = agent.observe_episode(returns[0] if ne == 1 else returns, physical_step=learning)
                if stats:
                    line(fu, dict(stats, learning_steps=learning))
            if hasattr(agent, 'after_exchange'):
                agent.set_physical_step(learning)
                stats = timed_update(lambda:agent.after_exchange(lambda:replay.sample(cfg['batch_size'],rng),
                    replay_ready=(replay.size>=cfg['batch_size'] and
                                  counts['train']['physical_steps']>=cfg['learning_starts'])))
                extra_updates += int(stats.get('extra_native_critic_updates_this_exchange',0))
                line(fu, dict(stats, learning_steps=learning))
            if hasattr(agent, 'maybe_adapt'):
                stats = agent.maybe_adapt(learning)
                if stats:
                    line(fu, dict(stats, learning_steps=learning))
            if learning == next_eval:
                evaluate()
            internal_evaluation(); episode += ne; ft.flush(); fu.flush()
        assert learning == total and next_eval == total+512
    if replay is not None:
        replay.assert_episode_boundary()
    changed = parameter_hash(agent) != initial_params
    if not changed or updates < 1:
        raise RuntimeError('No genuine neural update completed')
    saved = rng_state(agent)
    endpoint = endpoint_state(agent); digest = digest_state(endpoint)
    clone = make_agent(cfg, pool[0].observation_dim, pool[0].action_dim)
    loader = getattr(clone,'load_checkpoint_state_dict',clone.load_state_dict); loader(endpoint)
    restore_rng(saved)
    preferences = np.tile([.8,.2], (ne,1)).astype(np.float32)
    np.testing.assert_array_equal(agent.act(obs,preferences,True),clone.act(obs,preferences,True))
    if digest_state(endpoint_state(clone)) != digest:
        raise RuntimeError('Fixed endpoint did not reload exactly')
    with gzip.open(out/'endpoint.pt.gz','wb',compresslevel=1) as handle:
        torch.save(dict(state=endpoint, config=cfg, model_sha256=digest, inference_only=True,
                        training_restore_supported=False),handle)
    if source_identity(cfg) != source_sha:
        raise RuntimeError('Source manifest changed during candidate screen')
    cuda_ms = 0.
    if device.type == 'cuda':
        torch.cuda.synchronize(); cuda_ms = sum(a.elapsed_time(b) for a,b in gpu_events)
        if optimizer_cuda_count(agent) < 1 or cuda_ms <= 0:
            raise RuntimeError('No native CUDA optimizer/event evidence')
    import resource
    write_json(out/'completed.json', dict(success=True, learning_steps=learning,
        updates=updates, extra_exchange_updates=extra_updates, counts=counts,
        seconds=time.monotonic()-started, model_sha256=digest, endpoint_sha256=sha(out/'endpoint.pt.gz'),
        config_sha256=sha(config_path), source_manifest_sha256=source_sha, source_unchanged=True,
        initial_model_sha256=initial, initial_parameters_sha256=initial_params,
        endpoint_reload_exact=True, neural_parameters_changed=changed, optimizer_backup=False,
        replay_backup=False, holdout_paths=0, method=cfg['method'], profile=cfg['profile'],
        domain=cfg['domain'], seed=seed, engineering=bool(cfg.get('engineering')),
        gpu_proof=dict(device=str(device),parameter_devices=parameter_devices(agent),
            cuda_optimizer_tensors=optimizer_cuda_count(agent), cuda_update_event_ms=cuda_ms,
            cuda_timing_is_sampled=True, cuda_event_pairs=len(gpu_events),
            gpu_uuid=os.environ.get('CH5_GPU_UUID'), peak_reserved=torch.cuda.max_memory_reserved() if cuda_ms else 0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)))
    progress('complete')
    for e in pool+dev+internal:
        e.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    args = parser.parse_args()
    try:
        run(args.config)
    except BaseException:
        cfg = json.loads(Path(args.config).read_text()); out = Path(cfg['output'])
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'failure.json',dict(traceback=traceback.format_exc(),pid=os.getpid()))
        raise
