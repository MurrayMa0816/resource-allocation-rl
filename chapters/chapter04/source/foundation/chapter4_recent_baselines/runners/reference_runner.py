"""Unchanged mature original A / F1, with the new cohort's outer IO protocol.

This adapter does not implement another PPO variant. NativePPO(extra=False),
FrequentPPO(replay=True), complete-trajectory recording and CPU mirrors are
imported from the reviewed mature source packages. No prior result is loaded.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

from ..common import json_write
from ..env_adapter import Env, Task, _RELEASE_PACKAGE, RELEASE_ROOT

MASTER = 202610303101
SOURCE_WORKSPACE = Path(__file__).resolve().parents[2]


def _native():
    """Bind mature imports to precisely the release already used by Env."""
    importlib.import_module(_RELEASE_PACKAGE + '.learning')
    for name, module in list(sys.modules.items()):
        if name == _RELEASE_PACKAGE or name.startswith(_RELEASE_PACKAGE + '.'):
            alias = 'allocation23' + name[len(_RELEASE_PACKAGE):]
            if alias in sys.modules and sys.modules[alias] is not module:
                raise RuntimeError('Conflicting allocation23 release: ' + alias)
            sys.modules[alias] = module
    for folder in ('chapter4_credit_assignment', 'chapter4_sample_efficiency_v5',
                   'chapter4_reuse_optimization'):
        path = str(SOURCE_WORKSPACE / folder)
        if path not in sys.path:
            sys.path.insert(0, path)
    from allocation23.methods import make_actor, make_value
    from allocation23.learning import PPO
    from allocation_efficiency.baseline import NativePPO
    from allocation_efficiency.frequent import FrequentPPO
    from allocation_efficiency.rollout import rollout
    from allocation_efficiency.common import optimizer_steps
    from allocation4.common import cpu_copy, model_hash, tensor_batch
    from allocation4.methods.lax import gaussian_parameters, gaussian_kl
    from allocation_reuse.runtime import CPURolloutPair
    return locals()


def _source_lineage():
    records = {}
    for folder, paths in {
        'chapter4_credit_assignment': ['allocation4/common.py', 'allocation4/rollout.py',
                                     'allocation4/methods/baseline.py', 'allocation4/methods/lax.py'],
        'chapter4_sample_efficiency_v5': ['allocation_efficiency/baseline.py',
                                       'allocation_efficiency/frequent.py',
                                       'allocation_efficiency/geppo.py',
                                       'allocation_efficiency/rollout.py'],
        'chapter4_reuse_optimization': ['allocation_reuse/runtime.py'],
    }.items():
        for relative in paths:
            path = SOURCE_WORKSPACE / folder / relative
            records[str(path.relative_to(SOURCE_WORKSPACE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((RELEASE_ROOT / 'allocation23').rglob('*.py')):
        if '__pycache__' not in path.parts:
            records['chapter23_final/' + str(path.relative_to(RELEASE_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return records


def _rng_snapshot():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def _rng_restore(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def _action(actor, env, *, action_seed):
    generator = torch.Generator(device='cpu').manual_seed(int(action_seed))
    @torch.no_grad()
    def action(observation, t0=False, deterministic=False):
        if np.shape(observation) != env.observation_space.shape:
            raise ValueError('Unexpected public observation shape')
        result = actor.sample(env.state, generator=generator, deterministic=bool(deterministic))
        return result['raw'][0].detach().cpu().numpy().copy()
    action.seed_rng = lambda seed: generator.manual_seed(int(seed))
    action.rng_state = lambda: generator.get_state().clone()
    action.restore_rng = lambda state: generator.set_state(state)
    action.parameter_state = actor.state_dict
    return action


def evaluate(actor, domain, seed, paths, *, path_master=MASTER, phase='dev', engineering=False):
    """Fixed public paths and independent stochastic action RNG; restore training."""
    before = _rng_snapshot()
    mode = actor.training
    env = None
    try:
        action_seed = (8_000_000 if phase == 'final' else 4_000_000) + int(seed)
        random.seed(action_seed); np.random.seed(action_seed); torch.manual_seed(action_seed)
        env = Env(domain, seed, phase=('engineering' if engineering else phase),
                  action_mode='raw', path_master=path_master,
                  holdout=(phase == 'final' and not engineering), observation_dtype=np.float64)
        actor.eval()
        action = _action(actor, env, action_seed=action_seed)
        scores, plan_hashes = [], []
        for index in range(int(paths)):
            obs, _ = env.reset(options={'episode_index': index})
            plan_hashes.append(hashlib.sha256(json.dumps(env.task.plan(env.phase, index), sort_keys=True).encode()).hexdigest())
            ended = False; score = 0.
            while not ended:
                obs, reward, terminated, truncated, _ = env.step(action(obs))
                score += float(reward); ended = terminated or truncated
            scores.append(100. * score)
        return dict(mean=float(np.mean(scores)), std=float(np.std(scores, ddof=1)) if len(scores)>1 else 0.,
                    episode_scores=scores, evaluation_steps=env.real_steps,
                    phase=env.phase, holdout=env.holdout, paths=list(range(int(paths))),
                    path_master=int(path_master), action_seed=action_seed,
                    stochastic=True, plan_sha256=plan_hashes)
    finally:
        if env is not None:
            env.close()
        actor.train(mode)
        _rng_restore(before)


def run(config):
    config = dict(config)
    name = str(config['method']).lower()
    if name not in ('a', 'f1'):
        raise ValueError('Reference method must be original a or f1, never b1')
    if config.get('device', 'cpu') != 'cpu':
        raise ValueError('The registered mature references run on CPU')
    domain, seed = config['domain'], int(config['seed'])
    episodes = int(config.get('episodes', 960))
    interval = int(config.get('eval_every_episodes', 48))
    paths = int(config.get('eval_episodes', 32))
    smoke = bool(config.get('smoke', False))
    master = int(config.get('path_master', MASTER))
    if episodes % 8 or interval % 8:
        raise ValueError('Complete reference update/checkpoint budgets must be divisible by eight')
    if not smoke and (episodes != 960 or interval != 48 or paths != 32 or master != MASTER):
        raise ValueError('Formal reference cohort is fixed: 960ep, dev0+48ep/32paths, native master')
    out = Path(config['output_dir']); out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise FileExistsError('Never overwrite a reference attempt: ' + str(out))
    torch.set_num_threads(int(config.get('torch_threads', 1)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    native = _native()
    random.seed(830000 + seed); torch.manual_seed(840000 + seed)
    rng = np.random.default_rng(820000 + seed)
    generator = torch.Generator(device='cpu').manual_seed(810000 + seed)
    pool = []
    start = time.monotonic(); training_steps = 0; updates = 0
    eval_rows = []; collection_seconds = update_seconds = evaluation_seconds = 0.
    try:
        pool = [Task(domain, 3, 'A', seed, path_master=master) for _ in range(8)]
        spec = pool[0].spec
        actor = native['make_actor'](spec, 'A', 3, seed).cpu()
        critic = native['make_value'](spec, seed).cpu()
        ppo = native['PPO']
        ao = torch.optim.Adam(actor.parameters(), lr=ppo['actor_lr'])
        vo = torch.optim.Adam(critic.parameters(), lr=ppo['value_lr'])
        method = (native['NativePPO'](spec, actor.event_dim, seed, 'cpu', extra=False) if name == 'a'
                  else native['FrequentPPO'](spec, actor.event_dim, seed, 'cpu', replay=True))
        pair = native['CPURolloutPair'](actor, critic)
        source = _source_lineage(); initial_actor = native['model_hash'](actor)
        protocol = dict(schema='chapter4-new-formal-native-A-F1-v1', method=name, domain=domain, seed=seed,
                        fresh_training=True, prior_checkpoint_loaded=False, smoke_only=smoke,
                        device='cpu', dtype='float64', episodes=episodes, path_master=master,
                        initial_actor_sha256=initial_actor, spec=spec.to_dict(), native_ppo=ppo,
                        actor=actor.metadata(), implementation=method.metadata(),
                        collection='A: always8new; F1: warm32 8new, then4new+immediatelyPrevious4',
                        public_information='same released local/global/caps/budget/remaining; flat adapter exposes quota ledgers',
                        terminal_feedback='native0 until true horizon terminalR; gamma1; no private labels',
                        dev=dict(initial=True, every_episodes=interval, paths=paths, phase='dev',
                                 action_seed=4_000_000+seed, holdout=False, indices=list(range(paths))),
                        final_holdout='separate worker: fixed960, phase final,128paths,seed8M+seed,holdoutTrue',
                        source_sha256=source, source_workspace=str(SOURCE_WORKSPACE), release_root=str(RELEASE_ROOT),
                        checkpoint_policy='one fixed-endpoint lightweight actor+critic; no replay/optimizer copies',
                        runtime=pair.metadata())
        json_write(out/'config.json', config); json_write(out/'protocol.json', protocol)

        def dev(first):
            nonlocal evaluation_seconds
            before = time.monotonic()
            result = evaluate(actor, domain, seed, paths, path_master=master, engineering=smoke)
            evaluation_seconds += time.monotonic() - before
            row = dict(episode=first, train_env_steps=first*spec.horizon,
                       mean=result['mean'], std=result['std'], evaluation=result,
                       training_collection_seconds=collection_seconds, training_update_seconds=update_seconds,
                       evaluation_seconds=evaluation_seconds, wall_seconds=time.monotonic()-start)
            eval_rows.append(row); json_write(out/'evaluations.json', eval_rows)
            print(json.dumps(dict(method=name, domain=domain, seed=seed, episode=first,
                                  score=result['mean'], wall_seconds=time.monotonic()-start)), flush=True)

        dev(0)
        with (out/'training.jsonl').open('x') as ft, (out/'updates.jsonl').open('x') as fu:
            first = 0
            while first < episodes:
                batch_size = 4 if name == 'f1' and first >= 32 else 8
                stop = first + batch_size
                if stop > episodes or (first//interval != (stop-1)//interval):
                    raise AssertionError('Reference collection crossed fixed development boundary')
                json_write(out/'progress.json', dict(episode=first, episodes=episodes, phase='collecting',
                                                     pid=os.getpid(), wall_seconds=time.monotonic()-start))
                pair.refresh(actor, critic)
                before = time.monotonic()
                results = native['rollout'](pool[:batch_size], pair.actor, pair.critic, generator,
                                           'engineering' if smoke else 'train', first, behavior_version=updates)
                pair.assert_rollout_complete()
                collection_elapsed = time.monotonic()-before; collection_seconds += collection_elapsed
                batches = [row['batch'] for row in results]
                actor_before = copy.deepcopy(actor)
                old_a, old_v = native['optimizer_steps'](ao), native['optimizer_steps'](vo)
                before = time.monotonic()
                stats = method.update(actor, critic, ao, vo, batches, rng)
                update_elapsed = time.monotonic()-before; update_seconds += update_elapsed
                training_steps += batch_size * spec.horizon
                with torch.no_grad():
                    data = native['tensor_batch'](batches, 'cpu')
                    mu0, sd0 = native['gaussian_parameters'](actor_before, data['states'])
                    mu1, sd1 = native['gaussian_parameters'](actor, data['states'])
                    actual_kl = float(native['gaussian_kl'](mu0, sd0, mu1, sd1).mean())
                stats.update(episode=stop, real_steps=training_steps, new_episodes=batch_size,
                             behavior_version=updates, actor_committed_optimizer_steps=native['optimizer_steps'](ao)-old_a,
                             native_critic_optimizer_steps=native['optimizer_steps'](vo)-old_v,
                             collection_seconds=collection_elapsed, update_seconds=update_elapsed,
                             actual_joint_kl=actual_kl)
                for j, row in enumerate(results):
                    ft.write(json.dumps(dict(episode=first+j+1, score_x100=row['score_x100'],
                                             period_totals=row['period_totals'].tolist(), total_used=row['total_used'],
                                             remaining=row['remaining'], plan_sha256=row['plan_sha256']), allow_nan=False)+'\n')
                fu.write(json.dumps(stats, allow_nan=False)+'\n'); ft.flush(); fu.flush()
                first = stop; updates += 1
                if first % interval == 0 or first == episodes:
                    dev(first)
        # Only the fixed final model is eligible for holdout; no selected-best checkpoint.
        checkpoint = dict(schema='native-reference-lightweight-v1', method=name, domain=domain, seed=seed,
                          episode=episodes, train_env_steps=training_steps, smoke_only=smoke,
                          actor=native['cpu_copy'](actor.state_dict()), critic=native['cpu_copy'](critic.state_dict()),
                          protocol=protocol, actor_sha256=native['model_hash'](actor))
        temporary = out/'checkpoint_final.tmp'
        torch.save(checkpoint, temporary); temporary.replace(out/'checkpoint_final.pt')
        if _source_lineage() != source:
            raise AssertionError('Mature reference source changed during run')
        receipt = dict(method=name, domain=domain, seed=seed, device='cpu', smoke_only=smoke,
                       status='completed', train_env_steps=training_steps, episodes=episodes,
                       actor_updates=native['optimizer_steps'](ao), critic_updates=native['optimizer_steps'](vo),
                       outer_updates=updates, development_points=len(eval_rows),
                       development_env_steps=sum(x['evaluation']['evaluation_steps'] for x in eval_rows),
                       training_collection_seconds=collection_seconds, training_update_seconds=update_seconds,
                       evaluation_seconds=evaluation_seconds, wall_seconds=time.monotonic()-start,
                       actor_changed=native['model_hash'](actor)!=initial_actor, source_unchanged=True,
                       checkpoint='checkpoint_final.pt', checkpoint_actual_train_steps=training_steps,
                       distinctiveness=dict(original_a=name=='a', extra_epochs=False,
                                            replay=name=='f1', warmup_episodes=32))
        json_write(out/'receipt.json', receipt)
        json_write(out/'progress.json', dict(episode=episodes, episodes=episodes, phase='complete',
                                             pid=os.getpid(), wall_seconds=time.monotonic()-start))
        return receipt
    except BaseException:
        json_write(out/'failure.json', dict(traceback=traceback.format_exc(), wall_seconds=time.monotonic()-start))
        raise
    finally:
        for task in pool:
            task.env.close()


def load_final(job_dir, env, device='cpu'):
    if str(device) != 'cpu':
        raise ValueError('Mature reference checkpoints are evaluated on CPU')
    native = _native()
    folder = Path(job_dir)
    checkpoint = torch.load(folder/'checkpoint_final.pt', map_location='cpu', weights_only=False)
    config = json.loads((folder/'config.json').read_text())
    smoke = bool(checkpoint.get('smoke_only'))
    expected = int(config['episodes']) * int(env.horizon)
    if checkpoint['train_env_steps'] != expected or checkpoint['episode'] != int(config['episodes']):
        raise ValueError('Reference checkpoint does not match its fixed endpoint')
    if not smoke and (checkpoint['episode'] != 960 or expected != 960*env.horizon):
        raise ValueError('Only fixed960 formal references are eligible for holdout')
    if checkpoint['domain'] != env.domain or int(checkpoint['seed']) != int(env.seed_value):
        raise ValueError('Reference environment/checkpoint mismatch')
    actor = native['make_actor'](env.native_spec, 'A', 3, checkpoint['seed']).cpu()
    actor.load_state_dict(checkpoint['actor'], strict=True); actor.eval()
    if native['model_hash'](actor) != checkpoint['actor_sha256']:
        raise AssertionError('Restored reference actor hash mismatch')
    action = _action(actor, env, action_seed=8_000_000 + checkpoint['seed'])
    return action, dict(method=checkpoint['method'], domain=checkpoint['domain'],
                        checkpoint=str(folder/'checkpoint_final.pt'),
                        checkpoint_actual_train_steps=checkpoint['train_env_steps'],
                        checkpoint_episode=checkpoint['episode'], engineering_fixture=smoke,
                        actor_sha256=checkpoint['actor_sha256'], stochastic=True,
                        action_seed=8_000_000+checkpoint['seed'], dtype='float64', device='cpu')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text())), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
