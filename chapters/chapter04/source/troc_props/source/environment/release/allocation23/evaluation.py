"""Fixed-rule policy evaluation and evaluator-only allocation diagnostics."""
from __future__ import annotations

import csv
import gzip
import hashlib
from pathlib import Path

import numpy as np
import torch

from .io import state_hash, task_seed, write_json


def allocation_metrics(task, result):
    actions, used = result['actions'], result['total_used']
    metrics = dict(total_used=used, remaining=result['remaining'], mean_hhi=result['mean_hhi'],
                   mean_positive_objects=float((actions > 1e-12).sum(axis=1).mean()),
                   budget_used_fraction=used/task.spec.initial_budget,
                   last_quarter_fraction=float(result['period_totals'][-task.spec.horizon//4:].sum()
                                               / task.spec.initial_budget),
                   last_period_fraction=float(result['period_totals'][-1]/task.spec.initial_budget),
                   zero_demand_fraction=None, ineffective_fraction=None,
                   quality_p10=None, quality_min=None, quality_below_50=None,
                   quality_below_20=None, aggregate_completion=None)
    if task.domain == 'v5':
        demand = np.stack([observation['remaining_demand'] for observation in result['observations']])
        metrics['zero_demand_fraction'] = float(actions[demand <= 0].sum()/max(used, 1e-300))
        metrics['ineffective_fraction'] = float(result['terminal_info']['total_ineffective']/max(used, 1e-300))
        quality = np.asarray(result['terminal_info']['per_region_completion'])
        metrics['aggregate_completion'] = float(result['terminal_info']['aggregate_completion'])
    else:
        quality = np.asarray(task.env._terminal_ssim)
    metrics.update(quality_p10=float(np.quantile(quality, .1)), quality_min=float(quality.min()),
                   quality_below_50=float(np.mean(quality < .5)),
                   quality_below_20=float(np.mean(quality < .2)))
    return metrics


def evaluate(folder, task, actor, critic, episode, label, count, *, replay=False, traces=0):
    if count <= 0:
        return None
    if label not in ('dev', 'holdout', 'evaluation'):
        raise ValueError('Unknown evaluation phase')
    folder = Path(folder)
    before = (state_hash(actor), state_hash(critic))
    phase = 'dev' if label == 'dev' else 'final'
    temporary = folder/f'{label}_{episode}.csv.tmp'
    scores, records, period_totals, trace_arrays = [], [], [], {}
    replay_error = 0.
    with gzip.open(temporary, 'wt', newline='') as stream:
        writer = None
        for index in range(count):
            generator = torch.Generator().manual_seed(task_seed(phase, 0, index, task.path_master)+100000)
            result = task.episode(phase, index, actor, generator=generator, capture=True)
            metrics = allocation_metrics(task, result)
            observed_hash, error = None, None
            if replay:
                reward, observed_hash = task.replay(phase, index, result)
                error = abs(reward-result['native_return'])
                replay_error = max(replay_error, error)
                assert error < 1e-12, 'Native reward replay mismatch'
            row = dict(episode=episode, path=index, score_x100=100*result['native_return'], **metrics,
                       task_seed=result['plan']['task_seed'], path_sha256=result['path_sha256'],
                       actions_sha256=hashlib.sha256(result['actions'].tobytes()).hexdigest(),
                       observed_trace_sha256=observed_hash, replay_error=error)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            scores.append(row['score_x100'])
            records.append(metrics)
            period_totals.append(result['period_totals'])
            if index < traces:
                prefix = f'path_{index}_'
                trace_arrays[prefix+'actions'] = result['actions']
                trace_arrays[prefix+'rewards'] = result['rewards']
                for key in result['observations'][0]:
                    trace_arrays[prefix+key] = np.stack([observation[key] for observation in result['observations']])
    temporary.replace(folder/f'{label}_{episode}.csv.gz')
    assert before == (state_hash(actor), state_hash(critic)), 'Evaluation changed model parameters'
    if trace_arrays:
        np.savez_compressed(folder/f'{label}_{episode}_sample_traces.npz', **trace_arrays)
    score = dict(mean=float(np.mean(scores)), path_sd=float(np.std(scores)), n=count,
                 period_totals_mean=np.mean(period_totals, axis=0).tolist())
    for key in records[0]:
        values = [record[key] for record in records]
        score[key] = None if values[0] is None else float(np.mean(values))
    summary = dict(episode=episode, phase=label, path_master=task.path_master, score=score,
                   actor_sha256=before[0], critic_sha256=before[1],
                   native_replays=count if replay else 0, replay_max_error=replay_error if replay else None,
                   test_split_opened=task.holdout)
    write_json(folder/f'{label}_{episode}_summary.json', summary)
    return summary
