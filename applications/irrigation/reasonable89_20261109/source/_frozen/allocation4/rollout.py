"""Batched policy inference over independent unchanged physical environments.

The eight episodes in a production update share a frozen policy. Batching their
inference changes only computation, not the sequential decision task.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
import torch
from allocation23.task import Task
from .common import STATE_KEYS

def make_pool(domain, seed, size, master, *, holdout=False):
    return [Task(domain, 3, 'A', seed, path_master=master, holdout=holdout)
            for _ in range(size)]

def rollout(pool, actor, critic, generator, phase, first, *, count=None,
            training=False, deterministic=False):
    count = len(pool) if count is None else count
    tasks = pool[:count]
    encoders = []
    plans = []
    for j, task in enumerate(tasks):
        observation, info = task.reset(phase, first+j)
        if info:
            raise AssertionError('Reset exposed information outside the observation')
        encoder = task.encoder()
        encoder.reset(observation)
        encoders.append(encoder)
        plans.append(task.plan(phase, first+j))
    horizon = tasks[0].spec.horizon
    saved = [dict(states={k: [] for k in STATE_KEYS}, raw=[], old_logp=[], mask=[], values=[])
             for _ in tasks]
    actions = [[] for _ in tasks]
    rewards = [[] for _ in tasks]
    for t in range(horizon):
        states = [enc.state() for enc in encoders]
        combined = {k: np.stack([s[k] for s in states]) for k in STATE_KEYS}
        with torch.no_grad():
            sampled = actor.sample(combined, generator=generator, deterministic=deterministic)
            value = critic(combined) if training else torch.zeros_like(sampled['log_prob'])
        arrays = {k: sampled[k].detach().cpu().numpy().copy()
                  for k in ('raw', 'log_prob', 'mask')}
        values = value.detach().cpu().numpy().copy()
        for j, (task, enc, state) in enumerate(zip(tasks, encoders, states)):
            action, delta = task.validate(sampled['actions'][j], state)
            if not np.isfinite(action).all() or np.any(action < 0):
                raise AssertionError('Invalid executed resource allocation')
            observation, reward, done, truncated, info = task.env.step(action)
            if truncated or done != (t == horizon-1):
                raise AssertionError('Unexpected physical horizon')
            if not done and (reward != 0. or info):
                raise AssertionError('Nonterminal feedback must remain zero and empty')
            # The emergency simulator has terminal diagnostic info. It is
            # deliberately discarded rather than supplied to any learner.
            if training:
                b = saved[j]
                for k in STATE_KEYS:
                    b['states'][k].append(np.asarray(state[k]).copy())
                b['raw'].append(arrays['raw'][j])
                b['old_logp'].append(arrays['log_prob'][j])
                b['mask'].append(arrays['mask'][j])
                b['values'].append(values[j])
            enc.update(action, observation)
            actions[j].append(action.copy())
            rewards[j].append(float(reward))
    results = []
    for j, task in enumerate(tasks):
        native = float(sum(rewards[j]))
        a = np.asarray(actions[j])
        result = dict(native_return=native, score_x100=100*native,
                      total_used=float(a.sum()), remaining=float(encoders[j].state()['remaining']),
                      period_totals=a.sum(1), rewards=np.asarray(rewards[j]), plan=plans[j],
                      plan_sha256=hashlib.sha256(json.dumps(plans[j], sort_keys=True).encode()).hexdigest())
        if training:
            b = saved[j]
            result['batch'] = dict(states={k: np.stack(v) for k, v in b['states'].items()},
                                  **{k: np.asarray(v) for k, v in b.items() if k != 'states'},
                                  returns=np.full(horizon, native))
        results.append(result)
    return results

def evaluate(pool, actor, critic, paths, phase, device):
    result = {}
    for mode, deterministic in [('stochastic', False), ('mean_control', True)]:
        generator = torch.Generator(device=device).manual_seed(995103 if phase == 'dev' else 995199)
        rows = []
        for first in range(0, paths, len(pool)):
            episodes = rollout(pool, actor, critic, generator, phase, first,
                               count=min(len(pool), paths-first), deterministic=deterministic)
            rows.extend(dict(index=first+j, score_x100=e['score_x100'],
                             total_used=e['total_used'], remaining=e['remaining'],
                             plan_sha256=e['plan_sha256']) for j, e in enumerate(episodes))
        scores = np.asarray([r['score_x100'] for r in rows])
        result[mode] = dict(score=float(scores.mean()), sd=float(scores.std(ddof=1)) if paths>1 else 0.,
                            rows=rows)
    return result
