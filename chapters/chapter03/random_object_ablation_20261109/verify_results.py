"""Independent NumPy/SciPy verification of saved random-object DEV experiments.

This file never imports Torch, Actor, Task, an environment, or learned payloads.
Model bytes may be SHA-hashed, without deserializing or executing them.  New
physical work and previously completed/reused evidence are kept separate.
"""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root

import argparse
import collections
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import erfc

sys.dont_write_bytecode = True
ARM = 'A_RANDOM_OBJECT_ALLOCATION'
REUSED = ('A_FULL', 'A_PACE', 'A_DIRECT_TOTAL')
ARMS = ('A_FULL', 'A_PACE', ARM, 'A_DIRECT_TOTAL')
DOMAINS = ('radio', 'v5')
GRID = np.r_[np.arange(0, 100608, 512), 100608].astype(np.int64)
HELPER = Path(str(_resolve_path('/root/gamewar/resource_allocation_rl/chapters/chapter03/core_ablation_20261109/verify_results.py')))
HELPER_SHA = '355d4aa9318469ff9d32debfea339a9d7fa923846f09d46ceeaefde8d868f82e'
INPUTS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def track(path, expected=None):
    path = Path(path).resolve(strict=True)
    value = sha(path)
    if expected is not None and value != expected:
        raise AssertionError('SHA differs: ' + str(path))
    if INPUTS.setdefault(str(path), value) != value:
        raise AssertionError('Input changed during verification: ' + str(path))
    return path


def read(path):
    return json.loads(track(path).read_text())


def rows(path):
    p = track(path)
    with (gzip.open(p, 'rt', newline='') if p.suffix == '.gz' else p.open(newline='')) as f:
        return list(csv.DictReader(f))


def close(a, b, label, atol=1e-12, rtol=0):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError('Shape/finite: '+label)
    error = float(np.max(np.abs(a-b), initial=0.))
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f'{label}: max error {error}')
    return error


def native_helper():
    track(HELPER, HELPER_SHA)
    spec = importlib.util.spec_from_file_location('_random_object_pure_array_helper', HELPER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def record_directory(x):
    return Path(x.get('run', x.get('output'))).resolve(strict=True)


def verify_curve(run, method, domain, seed, helper, pools):
    p, d = read(run/'protocol.json'), read(run/'completed.json')
    args = p['arguments']
    assert p['method'] == method and p['domain'] == domain and args['seed'] == seed
    schema=('chapter3-random-object-allocation-v1' if method==ARM
            else 'chapter3-double-random-native-formal-v1')
    assert p['schema']==d['schema']==schema
    assert d['success'] and d['source_unchanged'] and d['endpoint_reload_pass']
    assert args['steps'] == d['training_steps'] == 100608 and args['final_paths'] == 0
    assert not args['engineering_gate'] and args['eval_every'] == 512 and args['development_paths'] == 8
    assert args['path_master'] == 202611043101
    curves, paths = rows(run/'evaluation_curve.csv.gz'), rows(run/'evaluation_paths.csv.gz')
    track(run/'evaluation_curve.csv.gz', d['artifacts_sha256']['evaluation_curve.csv.gz'])
    track(run/'evaluation_paths.csv.gz', d['artifacts_sha256']['evaluation_paths.csv.gz'])
    assert len(curves) == 198 and len(paths) == 198*8
    assert [int(x['training_step']) for x in curves] == GRID.tolist()
    horizon = 32 if domain == 'radio' else 24
    mapped = collections.defaultdict(list)
    for x in paths:
        assert x['phase'] == 'dev' and int(x['steps']) == horizon
        assert abs(float(x['resource_fraction'])-1.) < 1e-9 and float(x['replay_error']) == 0.
        index = int(x['path_index'])
        expected = (helper.task_seed('dev', seed, index, args['path_master']) if domain == 'v5'
                    else helper.radio_plan(args['path_master'], 'dev', seed, index,
                                           pools['validation'], p['spec']['groups'])['task_seed'])
        assert int(x['path_seed']) == expected
        mapped[int(x['training_step'])].append(x)
    values = []
    for x in curves:
        assert x['phase'] == 'dev' and int(x['paths']) == 8
        assert int(x['evaluation_environment_steps']) == 8*horizon
        group = sorted(mapped[int(x['training_step'])], key=lambda y:int(y['path_index']))
        assert [int(y['path_index']) for y in group] == list(range(8))
        a = np.asarray([float(y['native_return']) for y in group])
        for key, z in dict(native_mean=a.mean(), native_std=a.std(ddof=1),
                           min_return=a.min(), max_return=a.max()).items():
            close(float(x[key]), z, 'DEV path aggregation '+key, atol=3e-14)
        values.append(100*a.mean())
    return np.asarray(values), p, d


def verify_exports(root, matrices):
    analysis = root/'analysis'
    raw = rows(analysis/'curves.csv')
    assert len(raw) == 40*198
    expected = {(d,a,s,int(t)):float(matrices[d,a][s,i])
                for d in DOMAINS for a in ARMS for s in range(5) for i,t in enumerate(GRID)}
    seen = set()
    for x in raw:
        k = x['domain'], x.get('arm',x.get('method')), int(x['seed']), int(x['training_step'])
        assert k in expected and k not in seen
        seen.add(k)
        close(float(x['score']), expected[k], 'all exported raw curves')
    assert seen == set(expected)
    stat = rows(analysis/'curve_statistics.csv')
    assert len(stat) == 2*4*198
    seen = set()
    for x in stat:
        d,a,t = x['domain'],x.get('arm',x.get('method')),int(x['training_step'])
        k = d,a,t
        assert k not in seen
        seen.add(k)
        idx = GRID.tolist().index(t)
        v = matrices[d,a][:,idx]
        close(float(x['mean']),v.mean(),'curve means')
        key = 'training_seed_sample_SD' if 'training_seed_sample_SD' in x else 'sample_SD'
        close(float(x[key]),v.std(ddof=1),'curve sample SD')
    assert seen == {(d,a,int(t)) for d in DOMAINS for a in ARMS for t in GRID}
    metrics = rows(analysis/'metrics_by_seed.csv')
    assert len(metrics) == 40
    seen = set()
    for x in metrics:
        d,a,s = x['domain'],x.get('arm',x.get('method')),int(x['seed'])
        k = d,a,s
        assert k not in seen
        seen.add(k)
        v = matrices[d,a][s]
        late=GRID>=2*GRID[-1]/3
        calculated = dict(initial=v[0], DEV_fixed_endpoint=v[-1],
                          normalized_AUC=np.trapezoid(v,GRID)/100608,
                          full_horizon_normalized_AUC=np.trapezoid(v,GRID)/100608,
                          last_third_normalized_AUC=np.trapezoid(v[late],GRID[late])/(GRID[-1]-GRID[late][0]),
                          endpoint_change_from_initial=v[-1]-v[0])
        checked = 0
        for key,z in calculated.items():
            if key in x:
                close(float(x[key]),z,'seed metric '+key)
                checked += 1
        assert checked >= 5
    assert seen == {(d,a,s) for d in DOMAINS for a in ARMS for s in range(5)}
    summary = read(analysis/'summary.json')
    assert summary.get('status') in ('PASS','COMPLETE','COMPLETE_DEV_ANALYSIS')
    assert summary['schema']=='chapter3-random-object-ablation-analysis-v1'
    assert summary['methods']==list(ARMS) and summary['no_new_TEST'] is True
    assert summary['new_independent_TEST_paths']==0 and summary['endpoint_confirmation_performed'] is False
    assert summary['smoothing'] is summary['interpolation'] is False
    late=GRID>=2*GRID[-1]/3
    def measures(m):
        return dict(initial=m[:,0],DEV_fixed_endpoint=m[:,-1],
                    normalized_AUC=np.trapezoid(m,GRID,axis=1)/GRID[-1],
                    last_third_normalized_AUC=np.trapezoid(m[:,late],GRID[late],axis=1)/(GRID[-1]-GRID[late][0]),
                    endpoint_change_from_initial=m[:,-1]-m[:,0])
    summary_items=0
    for domain in DOMAINS:
        for arm in ARMS:
            for name,values in measures(matrices[domain,arm]).items():
                x=summary['settings'][domain][arm][name]
                close(x['by_training_seed'],values,'summary seed metric')
                close([x['mean'],x['training_seed_sample_SD'],x['minimum'],x['maximum']],
                      [values.mean(),values.std(ddof=1),values.min(),values.max()],
                      'summary statistics')
                assert x['positive_seed_count']==int((values>0).sum())
                assert x['negative_seed_count']==int((values<0).sum())
                assert x['zero_seed_count']==int((values==0).sum())
                summary_items+=1
            if arm=='A_FULL':continue
            for name,values in measures(matrices[domain,'A_FULL']-matrices[domain,arm]).items():
                x=summary['full_minus_removed'][domain][arm][name]
                close(x['by_training_seed'],values,'paired summary seed metric')
                close([x['mean'],x['training_seed_sample_SD']],
                      [values.mean(),values.std(ddof=1)],'paired summary statistics')
                assert x['positive_seed_count']==int((values>0).sum())
                assert x['negative_seed_count']==int((values<0).sum())
                assert x['zero_seed_count']==int((values==0).sum())
                close([x['minimum'],x['maximum']],[values.min(),values.max()],
                      'paired summary range')
                summary_items+=1
    return dict(raw_curve_rows=len(raw), aggregated_curve_rows=len(stat), seed_metric_rows=len(metrics),
                recomputed_summary_conditions=summary_items)


def verify_plot(root,matrices):
    """Read artists and final PDF; no plotting or model execution."""
    import fitz
    p=read(root/'figures/artists.json')
    for path,h in p['source_sha256'].items():track(path,h)
    for path,h in p['font_sha256'].items():track(path,h)
    assert p['smoothing'] is p['interpolation'] is False
    assert len(p['plots'])==1
    plot=p['plots'][0]
    close(plot['size_mm'],[156,98],'figure native dimensions',atol=1e-8)
    assert plot['ordinary_font_bp']==10.5 and len(plot['panels'])==2
    lines=0
    for panel in plot['panels']:
        domain=panel['domain']
        assert {x['arm'] for x in panel['series']}==set(ARMS)
        for s in panel['series']:
            m=matrices[domain,s['arm']]
            close(s['x_training_steps'],GRID,'artist training steps',atol=0)
            close(s['plotted_x'],GRID/1000,'artist plotted steps',atol=0)
            close(s['mean'],m.mean(0),'artist means')
            close(s['sample_SD'],m.std(0,ddof=1),'artist sample SD')
            close(s['lower'],m.mean(0)-m.std(0,ddof=1),'artist lower band')
            close(s['upper'],m.mean(0)+m.std(0,ddof=1),'artist upper band')
            assert min(s['lower'])>=panel['ylim'][0] and max(s['upper'])<=panel['ylim'][1]
            lines+=1
    for path,h in plot['artifacts'].items():track(path,h)
    pdf=root/'figures/figure35_random_object_preview.pdf'
    with fitz.open(track(pdf)) as doc:
        assert len(doc)==1
        page=doc[0]
        close([page.rect.width*25.4/72,page.rect.height*25.4/72],[156,98],
              'actual PDF dimensions',atol=1e-5)
        assert not page.get_images()
        spans=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']]
        assert spans
        for s in spans:
            assert abs(s['size']-10.5)<1e-5
            box=fitz.Rect(s['bbox'])
            assert box.x0>=-.1 and box.y0>=-.1 and box.x1<=page.rect.width+.1 and box.y1<=page.rect.height+.1
    return dict(lines=lines,nodes_per_line=198,ordinary_font_bp=10.5,native_mm=[156,98],
                PDF_pure_vector=True,PDF_sha256=sha(pdf))


def project_independently(proposal, caps, total):
    """Bisection of the projection multiplier; no native allocation import."""
    p, c = np.asarray(proposal), np.asarray(caps)
    if total <= 0:
        return np.zeros_like(c)
    if total >= c.sum():
        return c.copy()
    lo = float(np.min(p-c))
    hi = float(np.max(p))
    for _ in range(200):
        mid = (lo+hi)/2
        if np.clip(p-mid, 0, c).sum() > total:
            lo = mid
        else:
            hi = mid
    return np.clip(p-(lo+hi)/2, 0, c)


def verify_initial_random_kernel(run, protocol, plan, pools, helper):
    """Rebuild saved initial DEV path0 random allocations and radio packet caps."""
    path = track(run/'initial_allocation_examples.npz')
    with np.load(path,allow_pickle=False) as z:
        actions=np.asarray(z['allocations'])
    domain=protocol['domain']
    n,horizon=(100,32) if domain=='radio' else (120,24)
    assert actions.shape==(1,horizon,n)
    actions=actions[0]
    words=[202611094901,1 if domain=='radio' else 2,plan['training_master'],2,0,0]
    rng=np.random.default_rng(np.random.SeedSequence(words))
    packet_count=np.zeros(n,dtype=np.int64)
    remaining=float(protocol['spec']['initial_budget'])
    max_error=0.
    nonuniform=0
    if domain=='radio':
        ep=helper.radio_plan(plan['training_master'],'dev',0,0,pools['validation'],n)
        _,true,_,uniform=helper.radio_exogenous(ep['task_seed'],protocol['physical'])
    for t,a in enumerate(actions):
        q=float(a.sum())
        draws=rng.random((1,n))[0]
        caps=(np.where(packet_count<protocol['spec']['source_packets'],
                       protocol['spec']['per_object_cap'],0.) if domain=='radio'
              else np.full(n,remaining))
        active=caps>0
        weights=np.where(active,np.maximum(draws,np.finfo(float).tiny),0.)
        proposal=q*weights/weights.sum() if q>0 else np.zeros(n)
        rebuilt=project_independently(proposal,caps,q)
        error=close(a,rebuilt,'independent initial random object kernel',
                    atol=2e-15 if domain=='radio' else 2e-9)
        max_error=max(max_error,error)
        if q>0 and np.max(np.abs(a[active]-q/active.sum()))>(1e-12 if domain=='radio' else 1e-7):
            nonuniform+=1
        if domain=='radio':
            physical=protocol['physical']
            gamma=np.abs(true[t])**2*a/physical['reference_power_W']
            probability=np.exp((physical['payload_bytes']*8+physical['crc_bits'])*np.log1p(-.5*erfc(np.sqrt(gamma))))
            packet_count+=((packet_count<protocol['spec']['source_packets']) & (a>0) &
                           (uniform[t]<probability)).astype(np.int64)
        remaining-=q
    assert nonuniform>horizon//2
    return dict(domain=domain,seed=protocol['arguments']['seed'],DEV_path_index=0,
                periods=horizon,nonuniform_periods=nonuniform,
                independent_random_kernel_max_abs_error=max_error,
                reconstructed_allocation_seed_words=words,
                radio_progress_and_packet_draws_rebuilt=domain=='radio')


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve(strict=True)
    plan = read(root/'registration/plan.json')
    assert plan['arm'] == ARM and plan['steps'] == 100608 and plan['final_paths'] == 0
    assert plan['new_independent_TEST_paths'] == 0 and not plan['thesis_update']
    assert plan['object_random_namespace'] == 202611094901
    for name,digest in plan['source_identity'].items():
        track(Path(plan['source_root'])/name,digest)
    pools = read(Path(plan['data_root'])/'split_manifest.json')['splits']
    helper = native_helper()
    native_plan = dict(plan, _pools=pools,
        engineering={'steps':{x['domain']:x['steps'] for x in plan['engineering']}})
    new = read(root/'registration/training_manifest.json')['records']
    reused = read(root/'registration/reused_training_manifest.json')['records']
    assert len(new) == 10 and len(reused) == 30
    assert {(x['domain'],x['method'],x['seed']) for x in new} == {(d,ARM,s) for d in DOMAINS for s in range(5)}
    assert {(x['domain'],x['method'],x['seed']) for x in reused} == {(d,a,s) for d in DOMAINS for a in REUSED for s in range(5)}
    matrices = {(d,a):np.empty((5,198)) for d in DOMAINS for a in ARMS}
    execution, reused_checks, random_kernels = [], [], []
    for x in new:
        run = record_directory(x)
        audited, values = helper.verify_training(run, ARM, x['domain'], x['seed'], native_plan, formal=True)
        matrices[x['domain'],ARM][x['seed']] = values
        _,protocol,_=verify_curve(run,ARM,x['domain'],x['seed'],helper,pools)
        assert protocol['initial_algorithm']['event_dimension']==1
        assert protocol['initial_algorithm']['object_output_head_present'] is False
        assert protocol['initial_algorithm']['random_object_namespace']==202611094901
        assert protocol['initial_algorithm']['random_object_probability_in_PPO'] is False
        random_kernels.append(verify_initial_random_kernel(run,protocol,plan,pools,helper))
        execution.append(audited)
    engineering = sorted((root/'engineering').glob('*/completed.json'))
    assert len(engineering) == 2
    for done_path in engineering:
        p = read(done_path.parent/'protocol.json')
        audited,_ = helper.verify_training(done_path.parent, ARM, p['domain'],p['arguments']['seed'],native_plan,formal=False)
        execution.append(audited)
    for x in reused:
        run = record_directory(x)
        for filename,key in [('protocol.json','protocol_sha256'),('completed.json','completion_sha256')]:
            if key in x:track(run/filename,x[key])
        values,p,d = verify_curve(run,x['method'],x['domain'],x['seed'],helper,pools)
        assert p['source_identity']['package'] == plan['source_identity']
        matrices[x['domain'],x['method']][x['seed']] = values
        reused_checks.append(dict(domain=x['domain'],method=x['method'],seed=x['seed'],DEV_points=198,DEV_paths=1584))
    export_checks = verify_exports(root,matrices)
    figure_checks=verify_plot(root,matrices)
    protected = read(root/'registration/protected_before.json')['files']
    for p,h in protected.items():track(p,h)
    for p,h in helper.INPUTS.items():
        if p in INPUTS and INPUTS[p]!=h:raise AssertionError('Helper input changed: '+p)
        INPUTS[p]=h
    ledger = dict(training_environment_steps=sum(x['training_rows'] for x in execution),
        new_complete_DEV_paths=sum(x['DEV_paths'] for x in execution),
        total_physical_environment_steps=sum(x['physical_steps'] for x in execution),
        reused_complete_DEV_paths=sum(x['DEV_paths'] for x in reused_checks),
        endpoint_reload_receipts=len(execution),new_independent_TEST_paths=0)
    planned = plan['planned_work']
    assert ledger['training_environment_steps'] == planned['new_learning_steps']+planned['engineering_learning_steps']
    assert ledger['new_complete_DEV_paths'] == planned['new_complete_DEV_paths'] == 15840
    assert ledger['reused_complete_DEV_paths'] == planned['reused_complete_DEV_paths'] == 47520
    assert ledger['total_physical_environment_steps'] == planned['total_physical_environment_steps'] == 1893568
    assert ledger['endpoint_reload_receipts'] == planned['expected_new_endpoint_payload_loads'] == 12
    for p,h in INPUTS.items():
        if sha(p)!=h:raise AssertionError('Input changed before final receipt: '+p)
    output = dict(schema='chapter3-random-object-independent-results-verification-v1',status='PASS',
        scope='Saved original records and pure-array recomputation; no Actor/Task/Torch import or execution',
        execution=execution,reused_evidence=reused_checks,initial_random_kernel_checks=random_kernels,
        exports=export_checks,figure=figure_checks,ledger=ledger,
        protected_files=len(protected),frozen_native_sources=len(plan['source_identity']),
        inputs_sha256=INPUTS,model_deserializations=0,environment_steps=0,inference_calls=0,optimizer_steps=0)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS',ledger=ledger,exports=export_checks),ensure_ascii=False))


if __name__=='__main__':
    main()
