"""Read completed random-object runs and original controls; no policy execution."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

RANDOM = 'A_RANDOM_OBJECT_ALLOCATION'
ARMS = ('A_FULL', 'A_PACE', RANDOM, 'A_DIRECT_TOTAL')
CONTROLS = ('A_FULL', 'A_PACE', 'A_DIRECT_TOTAL')
DOMAINS = ('radio', 'v5')
SEEDS = tuple(range(5))
GRID = np.r_[np.arange(0, 100608, 512), 100608].astype(np.int64)
ORIGINAL = Path('/hy-tmp/resource_allocation_results/chapter03/experiments/mechanism_suite_20261109_163158/analysis')
ORIGINAL_CURVES_SHA = '81f081c5d36b20053d6452a3e9a6554cb9cde35b4eba7ae009226bf69bddaf2a'
INPUTS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def track(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = sha(path)
    if expected is not None and expected != digest:
        raise ValueError('SHA mismatch: ' + str(path))
    if INPUTS.setdefault(str(path), digest) != digest:
        raise ValueError('Input changed while analyzing: ' + str(path))
    return path


def read(path, expected=None):
    return json.loads(track(path, expected).read_text())


def rows(path, expected=None):
    path = track(path, expected)
    with (gzip.open(path, 'rt', newline='') if path.suffix == '.gz'
          else path.open(newline='')) as f:
        return list(csv.DictReader(f))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def write_csv(path, data):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def close(actual, expected, label, atol=2e-11):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != expected.shape or not np.isfinite(actual).all()
            or not np.isfinite(expected).all()
            or not np.allclose(actual, expected, rtol=0, atol=atol)):
        raise ValueError('Numerical mismatch: ' + label)


def summarize(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError('Five finite training-seed values required')
    return dict(by_training_seed=values.tolist(), training_seeds=list(SEEDS),
        mean=float(values.mean()), training_seed_sample_SD=float(values.std(ddof=1)),
        positive_seed_count=int((values > 0).sum()),
        negative_seed_count=int((values < 0).sum()),
        zero_seed_count=int((values == 0).sum()),
        minimum=float(values.min()), maximum=float(values.max()))


def metrics(matrix):
    late = GRID >= 2 * GRID[-1] / 3
    return dict(initial=matrix[:, 0], DEV_fixed_endpoint=matrix[:, -1],
        normalized_AUC=np.trapezoid(matrix, GRID, axis=1) / GRID[-1],
        last_third_normalized_AUC=np.trapezoid(matrix[:, late], GRID[late], axis=1)
            / (GRID[-1] - GRID[late][0]),
        endpoint_change_from_initial=matrix[:, -1] - matrix[:, 0])


def validate_run(item, method, common_plans, reference=None):
    domain, seed = item['domain'], int(item['seed'])
    run = Path(item.get('run') or item['output']).resolve(strict=True)
    p = read(run / 'protocol.json', item.get('protocol_sha256'))
    d = read(run / 'completed.json', item.get('completion_sha256'))
    a = p['arguments']
    if (p.get('method') != method or p.get('domain') != domain or a.get('seed') != seed
            or d.get('method') != method or d.get('domain') != domain or d.get('seed') != seed
            or d.get('success') is not True or d.get('endpoint_reload_pass') is not True
            or d.get('source_unchanged') is not True or d.get('training_steps') != 100608
            or a.get('steps') != 100608 or a.get('engineering_gate')
            or a.get('eval_every') != 512 or a.get('development_paths') != 8
            or a.get('final_paths') != 0 or a.get('path_master') != 202611043101):
        raise ValueError('Wrong completed fixed training identity: ' + str(run))
    schema = ('chapter3-random-object-allocation-v1' if method == RANDOM
              else 'chapter3-double-random-native-formal-v1')
    if p.get('schema') != schema or d.get('schema') != schema:
        raise ValueError('Wrong policy schema: ' + str(run))
    if reference is not None:
        for name in ('spec', 'physical', 'ppo'):
            if p[name] != reference[name]:
                raise ValueError('New policy changed shared ' + name)
        if p['source_identity']['package'] != reference['source_identity']['package']:
            raise ValueError('New policy changed original scientific package')
        if domain == 'radio':
            for name in ('images_sha256', 'split_manifest_sha256', 'image_shape', 'groups'):
                if p['data'][name] != reference['data'][name]:
                    raise ValueError('Shared real-image data differs: ' + name)
            if p['data']['image_sources']['dev'] != reference['data']['image_sources']['dev']:
                raise ValueError('Shared DEV image stream differs')
    curve = sorted(rows(run/'evaluation_curve.csv.gz',
        d['artifacts_sha256']['evaluation_curve.csv.gz']), key=lambda x:int(x['training_step']))
    if (len(curve) != 198 or [int(x['training_step']) for x in curve] != GRID.tolist()
            or any(x['phase'] != 'dev' or int(x['paths']) != 8 for x in curve)):
        raise ValueError('Complete 198-node DEV8 required')
    path_rows = rows(run/'evaluation_paths.csv.gz',
        d['artifacts_sha256']['evaluation_paths.csv.gz'])
    by_step = defaultdict(list)
    for x in path_rows:
        by_step[int(x['training_step'])].append(x)
    if len(path_rows) != 1584 or set(by_step) != set(GRID.tolist()):
        raise ValueError('Complete 198 times eight DEV paths required')
    horizon = 32 if domain == 'radio' else 24
    values = []
    for x in curve:
        step = int(x['training_step'])
        cases = sorted(by_step[step], key=lambda y:int(y['path_index']))
        if [int(y['path_index']) for y in cases] != list(range(8)):
            raise ValueError('Exactly eight ordered DEV cases required')
        seeds = tuple(int(y['path_seed']) for y in cases)
        if domain not in common_plans:
            common_plans[domain] = seeds
        if common_plans[domain] != seeds:
            raise ValueError('All original and random-object curves must use the same DEV cases')
        if any(y['phase'] != 'dev' or int(y['steps']) != horizon
               or float(y['replay_error']) != 0
               or abs(float(y['resource_fraction'])-1) > 1e-9 for y in cases):
            raise ValueError('Native replay/resource/path validation differs')
        raw = np.asarray([float(y['native_return']) for y in cases])
        close(raw.mean(), float(x['native_mean']), 'eight-path native mean')
        close(raw.std(ddof=1), float(x['native_std']), 'eight-path native sample SD')
        close([raw.min(), raw.max()], [float(x['min_return']), float(x['max_return'])],
              'eight-path range')
        if int(x['evaluation_environment_steps']) != 8*horizon:
            raise ValueError('Wrong per-node native evaluation work')
        values.append(100*raw.mean())
    expected_eval = 198*8*horizon
    for key, value in [('evaluation_steps', expected_eval),
                       ('native_replay_environment_steps', expected_eval),
                       ('total_physical_environment_steps', 100608+2*expected_eval)]:
        if d.get(key) != value:
            raise ValueError('Wrong physical work: ' + key)
    return np.asarray(values), p, d, dict(domain=domain, method=method, seed=seed,
        run=str(run), original_model_payload_loaded=False, complete_training_steps=100608,
        curve_nodes=198, DEV_case_records=1584, new_training=(method==RANDOM))


def analyze(root, output, original):
    root, output, original = Path(root).resolve(strict=True), Path(output).resolve(), Path(original).resolve(strict=True)
    if output.exists():
        raise ValueError('Analysis output must be new')
    receipt = read(original/'completion_receipt.json')
    if receipt.get('status') != 'PASS':
        raise ValueError('Original completed analysis required')
    original_rows = rows(original/'curves.csv', ORIGINAL_CURVES_SHA)
    if len(original_rows) != 9900:
        raise ValueError('Expected original full fifty-curve aggregation')
    old_grouped = defaultdict(list)
    for x in original_rows:
        if x['arm'] in CONTROLS:
            old_grouped[x['domain'],x['arm'],int(x['seed'])].append(x)
    old_manifest = read(original.parent/'registration/training_manifest.json')['records']
    original_by_key = {(x['domain'], x['method'], int(x['seed'])):x for x in old_manifest}
    old_items = read(root/'registration/reused_training_manifest.json')['records']
    new_items = read(root/'registration/training_manifest.json')['records']
    expected_old = {(d,a,s) for d in DOMAINS for a in CONTROLS for s in SEEDS}
    expected_new = {(d,RANDOM,s) for d in DOMAINS for s in SEEDS}
    if (len(old_items)!=30 or {(x['domain'],x['method'],int(x['seed'])) for x in old_items}!=expected_old
            or len(new_items)!=10 or {(x['domain'],x['method'],int(x['seed'])) for x in new_items}!=expected_new):
        raise ValueError('Exactly thirty reused controls and ten new random-object endpoints required')
    for item in old_items:
        old = original_by_key[item['domain'], item['method'], int(item['seed'])]
        for name in ('run', 'protocol_sha256', 'completion_sha256', 'model_sha256'):
            if item.get(name) != old.get(name):
                raise ValueError('A registered original control differs: ' + name)
    grouped, references, common_plans, evidence, new_done = {}, {}, {}, [], []
    # Full is checked before other arms so each domain has its physical reference.
    for item in sorted(old_items, key=lambda x:(x['method']!='A_FULL',x['domain'],int(x['seed']))):
        key = item['domain'],item['method'],int(item['seed'])
        values, p, d, e = validate_run(item, item['method'], common_plans, references.get(item['domain']))
        references.setdefault(item['domain'], p)
        source_rows = sorted(old_grouped[key], key=lambda x:int(x['training_step']))
        if [int(x['training_step']) for x in source_rows] != GRID.tolist():
            raise ValueError('Original aggregate grid differs')
        close(values, [float(x['score']) for x in source_rows], 'original controls raw paths versus original aggregate')
        grouped[key] = values
        evidence.append(e)
    for item in new_items:
        key = item['domain'],RANDOM,int(item['seed'])
        values,p,d,e = validate_run(item,RANDOM,common_plans,references[item['domain']])
        grouped[key] = values
        evidence.append(e)
        new_done.append(d)
    matrices = {(d,a):np.stack([grouped[d,a,s] for s in SEEDS]) for d in DOMAINS for a in ARMS}
    curves,statistics,metric_rows,settings,contrasts = [],[],[],{},{}
    for d in DOMAINS:
        settings[d],contrasts[d] = {},{}
        for a in ARMS:
            m = matrices[d,a]
            ms = metrics(m)
            settings[d][a] = {k:summarize(v) for k,v in ms.items()}
            mean,sd = m.mean(0),m.std(0,ddof=1)
            for s in SEEDS:
                for i,t in enumerate(GRID):
                    curves.append(dict(domain=d,arm=a,seed=s,training_step=int(t),score=float(m[s,i])))
                metric_rows.append(dict(domain=d,arm=a,seed=s,**{k:float(v[s]) for k,v in ms.items()}))
            for i,t in enumerate(GRID):
                statistics.append(dict(domain=d,arm=a,training_step=int(t),mean=float(mean[i]),
                    training_seed_sample_SD=float(sd[i]),n_training_seeds=5,DEV_paths_per_node=8))
        for a in ARMS[1:]:
            contrasts[d][a] = {k:summarize(v) for k,v in metrics(matrices[d,'A_FULL']-matrices[d,a]).items()}
    ledger_keys = ('training_steps','evaluation_steps','native_replay_environment_steps','total_physical_environment_steps')
    ledger = {k:sum(x[k] for x in new_done) for k in ledger_keys}
    summary = dict(schema='chapter3-random-object-ablation-analysis-v1',status='PASS',
        scope='Fixed100608 retrained random-object ablation DEV curves; three original controls reused',
        methods=list(ARMS),domains=list(DOMAINS),training_seeds=list(SEEDS),nodes=198,
        DEV_paths_per_node=8,training_steps_per_model=100608,new_training_models=10,reused_training_models=30,
        new_DEV_path_records=15840,reused_DEV_path_records=47520,total_curve_rows=7920,
        statistical_unit='Five independent original training seeds; eight fixed DEV cases at each node',
        random_object_execution='Eta evaluated at its learned mean; independent normalized U(0,1) object weights still sampled from fixed case/period streams',
        no_new_TEST=True,new_independent_TEST_paths=0,endpoint_confirmation_performed=False,
        smoothing=False,interpolation=False,initial_scores_preserved=True,
        native_DEV_task_seed_lists={d:list(v) for d,v in common_plans.items()},
        settings=settings,full_minus_removed=contrasts,new_formal_training_and_DEV_work=ledger,
        limits=['New random-object allocation is not the previous equal-object ablation.',
                'DEV results describe finite fixed environmental and object-randomness scenarios, not an exact random-policy expectation.',
                'No independent128 endpoint evaluation was performed in this experiment.',
                'Random allocation retains no trainable object action event; eta and the remaining quota policy are trained.',
                'New and reused training backend identities are retained; source package, task and PPO protocol are checked equal.',
                'No outcome-based seed, checkpoint, distribution or smoothing selection was performed.'])
    output.mkdir(parents=True)
    write_csv(output/'curves.csv',curves)
    write_csv(output/'curve_statistics.csv',statistics)
    write_csv(output/'metrics_by_seed.csv',metric_rows)
    write(output/'summary.json',summary)
    write(output/'training_evidence.json',dict(records=evidence))
    text = ['第三章随机对象分配消融：完整DEV训练结果','='*42,'',
        '新随机对象分配双域五种子从零100608步；完整/Pace/Direct三组30条既有控制曲线复用。',
        '每节点8共同DEV情景，含0与固定终点共198节点；均值和样本标准差按5训练种子统计，未平滑。',
        '随机对象权重在DEV仍执行；只有总量eta取均值。固定对象随机流用于可复现有限情景评价，不等于精确随机期望。','']
    for d in DOMAINS:
        unit = '分' if d=='radio' else '%'
        text.append('通信任务' if d=='radio' else '柴油任务')
        for a in ARMS:
            x=settings[d][a]['DEV_fixed_endpoint']
            text.append(f"{a}: 终点 {x['mean']:.6f} ± {x['training_seed_sample_SD']:.6f}{unit}")
        x=contrasts[d][RANDOM]
        for name in ('DEV_fixed_endpoint','normalized_AUC','last_third_normalized_AUC'):
            v=x[name]
            text.append(f"完整减随机分配 {name}: {v['mean']:+.6f}；{v['positive_seed_count']}/5为正。")
        text.append('')
    text.extend(['上述均为DEV学习过程，未新增独立TEST或128情景终点确认。',
        '随机分配与对象均分是不同对照；既有均分结果保留历史身份，不能将旧曲线重标成随机。',
        '结论以完整数据为准，不预设蓝线必须最高；论文正文和科学指针未由本分析改写。'])
    (output/'结论.rst').write_text('\n'.join(text)+'\n')
    artifacts=['curves.csv','curve_statistics.csv','metrics_by_seed.csv','summary.json','training_evidence.json','结论.rst']
    for p,digest in INPUTS.items():
        if sha(p)!=digest:
            raise ValueError('An input changed before completion')
    write(output/'completion_receipt.json',dict(schema='chapter3-random-object-ablation-analysis-completion-v1',
        status='PASS',inputs_sha256=INPUTS,output_sha256={p:sha(output/p) for p in artifacts},
        analysis_environment_steps=0,analysis_model_loads=0,analysis_inference_calls=0,new_TEST_paths=0))
    print(output/'summary.json')


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--original-analysis',default=str(ORIGINAL))
    a=p.parse_args()
    analyze(a.root,a.output,a.original_analysis)


if __name__=='__main__':
    main()
