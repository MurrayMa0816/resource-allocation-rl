"""Plot A/B main-method development evaluations from release-format runs."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

SCHEMA = 'allocation23-release-v1'
DOMAIN_NAMES = {'v5': '应急任务', 'radio': '通信任务'}
METHOD_COLORS = {'A': '#1768AC', 'B': '#D97706'}
METRICS = ('quality_p10', 'quality_min', 'quality_below_50', 'quality_below_20',
           'aggregate_completion', 'zero_demand_fraction', 'ineffective_fraction',
           'mean_hhi', 'remaining', 'mean_positive_objects', 'budget_used_fraction',
           'total_used', 'last_quarter_fraction', 'last_period_fraction')


def _json(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def _write_csv(path, rows, first):
    columns = list(dict.fromkeys(first + [key for row in rows for key in row]))
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    tmp.replace(path)


def _same(values, description):
    unique = {json.dumps(value, sort_keys=True) for value in values}
    if len(unique) > 1:
        raise ValueError(f'Cannot combine runs with different {description}; use separate --runs directories')


def _read_runs(root):
    runs, skipped = [], []
    paths = sorted(root.rglob('protocol.json'))
    if (root / 'protocol.json').exists() and root / 'protocol.json' not in paths:
        paths.insert(0, root / 'protocol.json')
    seen = set()
    for path in paths:
        protocol = _json(path)
        config = protocol.get('config', {})
        if protocol.get('schema') != SCHEMA:
            skipped.append({'path': str(path), 'reason': 'different schema'})
            continue
        if config.get('variant') is not None or config.get('temporal') is not None:
            skipped.append({'path': str(path), 'reason': 'ablation; main A/B plots only'})
            continue
        if config.get('domain') not in DOMAIN_NAMES or config.get('method') not in METHOD_COLORS:
            raise ValueError(f'Unknown main method/domain in {path}')
        if config.get('chapter') not in (2, 3):
            raise ValueError(f'Chapter must be2 or3 in {path}')
        key = (config['domain'], config['chapter'], config['method'], config['seed'])
        if key in seen:
            raise ValueError(f'Duplicate main run for domain/chapter/method/seed: {key}')
        seen.add(key)
        runs.append({'directory': path.parent, 'protocol_path': path, 'protocol': protocol,
                     'config': config, 'episodes': int(protocol['episodes'])})
    if not runs:
        raise ValueError(f'No main {SCHEMA} runs found under {root}')
    for domain in DOMAIN_NAMES:
        domain_runs = [run for run in runs if run['config']['domain'] == domain]
        for field in ('images_sha256', 'split_manifest_sha256'):
            values = [run['protocol'].get('data', {}).get(field) for run in domain_runs]
            if domain == 'radio' and any(not isinstance(value, str) or
                                         re.fullmatch(r'[0-9a-f]{64}', value) is None
                                         for value in values):
                raise ValueError(f'Radio runs require a valid data.{field} content hash')
            _same(values, f'{domain} data.{field} content hash')
    for chapter in sorted({run['config']['chapter'] for run in runs}):
        group = [run for run in runs if run['config']['chapter'] == chapter]
        for field in ('episodes', 'eval_every', 'development_paths', 'holdout_paths'):
            _same([run['protocol'].get(field) for run in group], f'chapter{chapter} {field}')
        for field in ('path_master', 'holdout_master', 'quota_reference'):
            _same([run['config'].get(field) for run in group], f'chapter{chapter} {field}')
        _same([run['protocol'].get('ppo') for run in group], f'chapter{chapter} PPO settings')
        for domain in DOMAIN_NAMES:
            domain_runs = [run for run in group if run['config']['domain'] == domain]
            # Method-specific metadata is deliberately excluded; the declared public
            # state and execution geometry must match for the same physical task.
            _same([run['protocol'].get('spec') for run in domain_runs], f'{domain}/chapter{chapter} public state/resource specification')
            _same([run['protocol'].get('physical') for run in domain_runs], f'{domain}/chapter{chapter} physical environment')
    return runs, skipped


def _collect(runs):
    development, holdout = [], []
    for run in runs:
        config, protocol, folder = run['config'], run['protocol'], run['directory']
        base = {key: config[key] for key in ('domain', 'chapter', 'method', 'seed')}
        base.update(run=str(folder), training_episodes=run['episodes'])
        paths = []
        for path in sorted(folder.glob('dev_*_summary.json')):
            match = re.fullmatch(r'dev_(\d+)_summary\.json', path.name)
            if match:
                paths.append(('dev', int(match.group(1)), path))
        endpoint = folder / f"holdout_{run['episodes']}_summary.json"
        if endpoint.is_file():
            paths.append(('holdout', run['episodes'], endpoint))
        for phase, episode, path in paths:
            summary = _json(path)
            if episode > run['episodes'] or int(summary.get('episode', -1)) != episode:
                raise ValueError(f'Evaluation checkpoint differs from registered budget: {path}')
            if summary.get('phase') != phase:
                raise ValueError(f'Evaluation phase mismatch: {path}')
            expected_master = config['path_master'] if phase == 'dev' else config['holdout_master']
            expected_count = int(protocol['development_paths'] if phase == 'dev' else protocol['holdout_paths'])
            if summary.get('path_master') != expected_master:
                raise ValueError(f'Evaluation path master mismatch: {path}')
            score = summary['score']
            if int(score['n']) != expected_count or expected_count < 1:
                raise ValueError(f'Evaluation path count mismatch: {path}')
            mean = float(score['mean'])
            if not math.isfinite(mean):
                raise ValueError(f'Nonfinite evaluation mean: {path}')
            row = dict(base, episode=episode, score=mean, paths=expected_count,
                       path_master=expected_master, path_sd=score.get('path_sd'), source=str(path))
            if phase == 'holdout':
                row.update({metric: score.get(metric) for metric in METRICS})
                row['period_totals_mean'] = score.get('period_totals_mean')
                holdout.append(row)
            else:
                development.append(row)
    for domain in DOMAIN_NAMES:
        domain_rows = [row for row in holdout if row['domain'] == domain]
        _same([row['path_master'] for row in domain_rows],
              f'{domain} cross-chapter independent-test path master')
        _same([row['paths'] for row in domain_rows],
              f'{domain} cross-chapter independent-test path count')
    return development, holdout


def _aggregate(rows):
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in ('domain', 'chapter', 'method', 'episode'))
        grouped.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(grouped.items()):
        if len({row['seed'] for row in group}) != len(group):
            raise ValueError(f'Duplicate seed at evaluation point: {key}')
        values = np.asarray([row['score'] for row in group])
        item = dict(zip(('domain', 'chapter', 'method', 'episode'), key))
        item.update(mean=float(values.mean()), seed_sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                    n_seeds=len(values), seeds=sorted(row['seed'] for row in group),
                    paths_per_model=group[0]['paths'], path_master=group[0]['path_master'],
                    training_episodes=group[0]['training_episodes'])
        for metric in METRICS:
            samples = [row[metric] for row in group if row.get(metric) is not None]
            if samples:
                item[metric + '_mean'] = float(np.mean(samples))
                item[metric + '_seed_sd'] = float(np.std(samples, ddof=1)) if len(samples) > 1 else None
        series = [row['period_totals_mean'] for row in group if row.get('period_totals_mean')]
        if series:
            item['period_totals_mean'] = np.mean(series, axis=0).tolist()
        result.append(item)
    return result


def _font(plt, font_manager):
    bundled = Path(__file__).parent / 'assets' / 'DroidSansFallbackFull.ttf'
    system_droid = Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    droid = bundled if bundled.is_file() else system_droid
    noto = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if droid.is_file():
        font_manager.fontManager.addfont(str(droid))
        family = ['DejaVu Sans', font_manager.FontProperties(fname=str(droid)).get_name()]
        fonttype = 42
    elif noto.is_file():
        font_manager.fontManager.addfont(str(noto))
        family = [font_manager.FontProperties(fname=str(noto)).get_name()]
        fonttype = 3  # CFF-based Noto TTC must not be embedded as TrueType.
    else:
        raise RuntimeError('Chinese plotting requires fonts-droid-fallback or fonts-noto-cjk')
    plt.rcParams.update({'font.family': family, 'pdf.fonttype': fonttype, 'ps.fonttype': fonttype,
                         'axes.unicode_minus': True, 'savefig.dpi': 180, 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18})
    return {'family': family, 'pdf_fonttype': fonttype,
            'file': str(droid if droid.is_file() else noto), 'bundled': droid == bundled}


def _plot(runs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D

    registered, skipped = _read_runs(runs)
    raw_dev, raw_holdout = _collect(registered)
    development, holdout = _aggregate(raw_dev), _aggregate(raw_holdout)
    font = _font(plt, font_manager)
    artifacts, missing = [], []
    for chapter in sorted({run['config']['chapter'] for run in registered}):
        chapter_runs = [run for run in registered if run['config']['chapter'] == chapter]
        episodes = chapter_runs[0]['episodes']
        paths = int(chapter_runs[0]['protocol']['development_paths'])
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8))
        counts = []
        for ax, domain in zip(axes, DOMAIN_NAMES):
            domain_runs = [run for run in chapter_runs if run['config']['domain'] == domain]
            available_steps = sorted({row['episode'] for row in development if row['chapter'] == chapter and row['domain'] == domain})
            domain_counts = []
            if not domain_runs:
                ax.text(.5, .5, '尚无该环境的主方法结果', transform=ax.transAxes, ha='center', va='center')
                missing.append({'chapter': chapter, 'domain': domain, 'reason': 'environment absent'})
            for method, color in METHOD_COLORS.items():
                selected = {row['episode']: row for row in development if row['chapter'] == chapter and row['domain'] == domain and row['method'] == method}
                domain_counts.append(f"{method} n={selected[max(selected)]['n_seeds'] if selected else 0}")
                y = np.asarray([selected[step]['mean'] if step in selected else np.nan for step in available_steps])
                sd = np.asarray([selected[step]['seed_sd'] if step in selected and selected[step]['seed_sd'] is not None else np.nan for step in available_steps])
                ax.plot(available_steps, y, color=color, marker='o', markersize=3.5, linewidth=2, label=method)
                ax.fill_between(available_steps, y - sd, y + sd, where=np.isfinite(sd), color=color, alpha=.15)
                if not selected:
                    missing.append({'chapter': chapter, 'domain': domain, 'method': method, 'reason': 'development evaluations absent'})
            counts.append(DOMAIN_NAMES[domain] + '：' + '，'.join(domain_counts))
            ax.set(title=DOMAIN_NAMES[domain], xlabel='训练回合', ylabel='策略评估回报 ×100', xlim=(0, episodes))
            if len(available_steps) <= 9:
                ax.set_xticks(sorted(set([0, episodes] + available_steps)))
        title = '第二章：固定总量' if chapter == 2 else '第三章：自适应总量'
        fig.suptitle(title + ' · A/B评估曲线', fontsize=17)
        fig.subplots_adjust(left=.065, right=.985, top=.86, bottom=.27, wspace=.22)
        handles = [Line2D([0], [0], color=color, marker='o', linewidth=2, label=method) for method, color in METHOD_COLORS.items()]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .15), ncol=2, frameon=False, fontsize=12)
        notes = [f'曲线为固定训练点的策略评估，非逐回合训练奖励；登记训练预算{episodes}回合，每模型{paths}条开发任务。',
                 '跨种子均值±样本SD；单种子不估计SD，缺失不补值。末个已显示点：' + '；'.join(counts),
                 '本图不选择最佳检查点；固定终点独立测试结果另见holdout_summary.csv。']
        for index, note in enumerate(notes):
            fig.text(.065, .125 - index * .035, note, fontsize=9, color='#444444')
        for suffix in ('.png', '.pdf'):
            target = output / f'chapter{chapter}_learning_curves{suffix}'
            temporary = target.with_name(target.stem + '.tmp' + suffix)
            fig.savefig(temporary, facecolor='white')
            temporary.replace(target)
            artifacts.append(str(target))
        plt.close(fig)
    identity = ['domain', 'chapter', 'method', 'episode']
    for name, rows in [('development_by_seed', raw_dev), ('development_summary', development),
                       ('holdout_by_seed', raw_holdout), ('holdout_summary', holdout)]:
        _write_json(output / (name + '.json'), rows)
        _write_csv(output / (name + '.csv'), rows, identity)
    metadata = {'schema': 'allocation23-plot-v1', 'generated_at': datetime.now(timezone.utc).isoformat(),
                'runs_root': str(runs), 'output': str(output), 'run_count': len(registered),
                'source_protocols': [{'path': str(run['protocol_path']), 'sha256': _digest(run['protocol_path'])} for run in registered],
                'skipped': skipped, 'missing': missing, 'font': font, 'artifacts': artifacts,
                'development_rows': len(raw_dev), 'holdout_rows': len(raw_holdout),
                'statistics': 'Training seeds are repetitions; sampleSD(ddof=1), no band for one seed; missing evaluations not filled',
                'units': 'Score is native terminal return x100; quality/fraction metrics remain0..1; resource fields retain native units'}
    _write_json(output / 'plot_metadata.json', metadata)
    return metadata


def plot_runs(runs, output):
    """Write two-panel A/B figures and development/holdout numeric tables.

    Main release runs may be incomplete. Different training budgets, public task
    specifications or evaluation path protocols are rejected rather than pooled.
    """
    runs, output = Path(runs).expanduser().resolve(), Path(output).expanduser().resolve()
    if not runs.is_dir():
        raise ValueError(f'Run directory does not exist: {runs}')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.plot.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return _plot(runs, output)
