"""Publish five Chapter 5 figures and four tables from the sealed 90/210 cohort.

No environment, policy, optimizer, CUDA or training module is imported. All raw
statistics remain unchanged; the common five-point display filter is separate.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import warnings


MAIN = ('full', 'capql', 'qpensieve', 'cola', 'morld', 'pasta')
ABLATIONS = ('full', 'only_j', 'only_c', 'scalar_value')
DOMAINS = ('radio', 'v5')
SEEDS = (0, 1, 2, 3, 4)
NAMES = {'full': '本章方法', 'capql': 'CAPQL', 'qpensieve': 'Q-Pensieve', 'cola': 'COLA',
         'morld': 'MORL/D', 'pasta': 'PASTA', 'only_j': '仅原任务目标',
         'only_c': '仅达标数量', 'scalar_value': '标量价值监督'}
COLORS = {'full': '#154f80', 'capql': '#dd873a', 'qpensieve': '#319664', 'cola': '#9368a6',
          'morld': '#c9494d', 'pasta': '#498f9d', 'only_j': '#d98b36',
          'only_c': '#429a73', 'scalar_value': '#9768a2'}
STYLES = dict(zip(MAIN, ('-', '--', '-.', ':', (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)))))
STYLES.update(only_j='--', only_c='-.', scalar_value=':')
MARKERS = dict(zip(MAIN, ('o', 's', '^', 'D', 'P', 'X')))
SCHEMA = 'chapter5-formal-five-seed-report-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def jsonl(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            require(bool(line.strip()), 'Blank raw data record')
            yield json.loads(line)


def output_csv(path, rows):
    require(bool(rows), 'Cannot publish an empty CSV')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def algorithm(model):
    return model['variant'] if model['method'] == 'full' else model['method']


def five_point_display(values):
    """Centered five-point mean of internal points; true start/end remain exact."""
    import numpy as np
    raw = np.asarray(values, dtype=float)
    require(raw.ndim == 1 and len(raw) >= 3 and np.isfinite(raw).all(), 'Malformed curve')
    result = raw.copy()
    for index in range(1, len(raw)-1):
        result[index] = raw[max(0, index-2):min(len(raw), index+3)].mean()
    require(result[0] == raw[0] and result[-1] == raw[-1], 'Display filter moved a true endpoint')
    return result


def moments(values):
    import numpy as np
    raw = np.asarray(values, dtype=float)
    require(raw.shape[0] == 5 and np.isfinite(raw).all(), 'Exactly five registered training seeds required')
    return raw.mean(axis=0), raw.std(axis=0, ddof=1)


def development_metrics(row, domain):
    """Derive the common C field from actual cases, including native baselines."""
    n = {'radio': 100, 'v5': 120}[domain]
    cases = row['rows']
    require(cases and all(case['N'] == n for case in cases), 'DEV case geometry differs')
    k = math.fsum(case['K'] for case in cases)/len(cases)
    c = math.fsum(case['C'] for case in cases)/len(cases)
    j = math.fsum(case['J'] for case in cases)/len(cases)
    require(abs(k-row['mean_K']) < 1e-10 and abs(j-row['mean_J']) < 1e-10 and abs(c-k/n) < 1e-12,
            'DEV means differ from their actual case outcomes')
    if 'mean_C' in row:
        require(abs(c-row['mean_C']) < 1e-12, 'Recorded DEV C differs from K/N')
    f = .8*c+.2*j
    require(abs(row['mean_F']-f) < 1e-10, 'Recorded DEV auxiliary value differs')
    return dict(K=k, C=c, J=j, F_0_8=f)


def audit_inputs(root):
    bundle_path = root/'report_bundle.json'
    final = read(root/'final_evaluation_audit.json')
    training = read(root/'training_audit.json')
    require(final['status'] == training['status'] == 'PASS', 'Completed independent training and evaluation audits required')
    require(final['report_bundle_sha256'] == sha(bundle_path), 'Report input seal differs')
    bundle = read(bundle_path)
    require(bundle['schema'] == 'chapter5-formal-report-input-bundle-v1', 'Unknown report input schema')
    require(Path(bundle['root']).resolve() == root, 'Foreign report root')
    require(bundle['contract_sha256'] == sha(root/'contract.json') == final['contract_sha256'], 'Contract changed')
    require(bundle['training_audit_sha256'] == sha(root/'training_audit.json'), 'Training seal changed')
    require(bundle['model_registry_sha256'] == sha(root/'model_registry.json'), 'Ninety-model registry changed')
    require(bundle['training_steps_per_model'] == 100608 and bundle['training_seeds'] == list(SEEDS), 'Formal budget/seeds differ')
    grid = list(range(0, 100609, 512)) + [100608]
    require(bundle['evaluation_grid'] == grid and len(grid) == bundle['curve_points'] == 198, 'Incomplete DEV grid')
    require(bundle['raw_data_only'] is True and bundle['model_or_seed_selection'] is False, 'Selected or transformed input rejected')
    models = bundle['models']
    evaluations = bundle['evaluations']
    require(len(models) == 90 and len(evaluations) == 210, 'All ninety models and 210 fixed panels must be present')
    lookup = {model['model_id']: model for model in models}
    require(len(lookup) == 90, 'Duplicated fixed model')
    groups = defaultdict(set)
    for model in models:
        key = algorithm(model)
        require(key in set(MAIN) | set(ABLATIONS) and model['domain'] in DOMAINS, 'Foreign algorithm/domain')
        groups[key, model['domain']].add(model['seed'])
        for path_key, hash_key in [('development_path', 'development_sha256'),
                                   ('training_completed_path', 'training_completed_sha256'),
                                   ('config_path', 'config_sha256'), ('endpoint_path', 'endpoint_sha256')]:
            require(sha(model[path_key]) == model[hash_key], 'Bound training file changed: '+path_key)
    require(len(groups) == 18 and all(values == set(SEEDS) for values in groups.values()), 'Incomplete five-seed algorithm groups')
    seen = set()
    for evaluation in evaluations:
        require(evaluation['model_id'] in lookup and evaluation['paths'] == 256, 'Foreign or incomplete fixed panel')
        model = lookup[evaluation['model_id']]
        require(all(evaluation[key] == model[key] for key in ('method', 'variant', 'domain', 'seed')), 'Fixed panel model identity differs')
        key = evaluation['model_id'], float(evaluation['scale'])
        require(key not in seen, 'Duplicated fixed panel')
        seen.add(key)
        require(float(evaluation['scale']) in ([.8, 1., 1.2] if algorithm(model) in MAIN else [1.]), 'Unregistered budget panel')
        require(sha(evaluation['evaluation_path']) == evaluation['evaluation_sha256'] and
                sha(evaluation['completed_path']) == evaluation['completed_sha256'], 'Fixed panel data changed')
        done = read(evaluation['completed_path'])
        require(done['success'] is True and done['engineering'] is False and done['model_unchanged'] is True and
                done['rng_unchanged'] is True and done['normalization_state_unchanged'] is True and
                done['new_training_steps'] == done['optimizer_steps'] == done['PSA_calls'] == 0,
                'Fixed evaluation performed learning or changed its model')
        for metric in ('K', 'C', 'J', 'F_0_8'):
            require(math.isfinite(evaluation['means'][metric]) and
                    abs(evaluation['means'][metric]-done['mean_'+metric]) < 1e-10, 'Sealed fixed aggregate changed')
    require(sum(e['physical_steps'] for e in evaluations) == 1505280, 'Fixed panel physical work differs')
    return bundle, lookup


def assemble(root, bundle, lookup):
    import numpy as np
    raw_curves, references, display_curves, endpoints, thresholds, costs = [], [], [], [], [], []
    series = {}
    endpoint_lookup = {}
    contract = read(root/'contract.json')
    for model in bundle['models']:
        key = algorithm(model)
        curve = []
        for row in jsonl(model['development_path']):
            values = dict(model_id=model['model_id'], algorithm=key, domain=model['domain'], seed=model['seed'],
                          learning_steps=row['learning_steps'], kind=row['kind'],
                          **development_metrics(row, model['domain']))
            (raw_curves if row['kind'] == 'curve' else references).append(values)
            if row['kind'] == 'curve':
                curve.append(values)
        require([row['learning_steps'] for row in curve] == bundle['evaluation_grid'], 'Actual raw DEV curve incomplete')
        series[key, model['domain'], model['seed']] = curve
        display = {metric: five_point_display([row[metric] for row in curve]) for metric in ('K', 'C', 'J', 'F_0_8')}
        for index, row in enumerate(curve):
            display_curves.append(dict(row, **{metric: float(display[metric][index]) for metric in display},
                                       display_filter='centered5_internal_only_raw_start_and_endpoint'))
        done = read(model['training_completed_path'])
        counts = done['counts']
        if model['method'] == 'full':
            reload_steps = counts['reload_verification']['physical_steps']
            reload_scope = 'independent_environment_panel'
        else:
            require(counts.get('reload_verification', {}).get('physical_steps', 0) == 0,
                    'Native baseline reload unexpectedly incurred new environment steps')
            reload_steps = 0
            reload_scope = 'existing_observation_action_comparison_no_new_environment_steps'
        costs.append(dict(model_id=model['model_id'], algorithm=key, domain=model['domain'], seed=model['seed'],
                          device=model['device'], host=model['host'], train_steps=counts['train']['physical_steps'],
                          internal_steps=counts.get('internal', {}).get('physical_steps', 0),
                          charged_steps=counts['train']['physical_steps']+counts.get('internal', {}).get('physical_steps', 0),
                          development_steps=counts['development']['physical_steps'],
                          reload_steps=reload_steps, reload_verification_scope=reload_scope,
                          default_test_steps=256*contract['domains'][model['domain']]['horizon'],
                          transfer_test_steps=(512 if key in MAIN else 0)*contract['domains'][model['domain']]['horizon'],
                          native_update_records=done['updates'], extra_exchange_updates=done.get('extra_exchange_updates', 0),
                          seconds=done['seconds']))
    for evaluation in bundle['evaluations']:
        model = lookup[evaluation['model_id']]
        key = algorithm(model)
        values = dict(model_id=model['model_id'], algorithm=key, domain=model['domain'], seed=model['seed'],
                      scale=float(evaluation['scale']), **evaluation['means'])
        endpoints.append(values)
        endpoint_lookup[key, model['domain'], model['seed'], float(evaluation['scale'])] = values
        if float(evaluation['scale']) != 1. or key not in MAIN:
            continue
        cases = list(jsonl(evaluation['evaluation_path']))
        require(len(cases) == 256, 'Threshold recount requires all 256 fixed cases')
        q = np.asarray([row['quality'] for row in cases], dtype=float)
        require(q.shape == (256, contract['domains'][model['domain']]['N']) and np.isfinite(q).all(), 'Incomplete saved per-object task results')
        for threshold in contract['evaluation']['thresholds'][model['domain']]:
            k = (q >= threshold).sum(axis=1)
            thresholds.append(dict(model_id=model['model_id'], algorithm=key, domain=model['domain'], seed=model['seed'],
                                   task_threshold=threshold, K=float(k.mean()), C=float(k.mean()/q.shape[1]),
                                   J=float(q.mean()), policy_fixed=True, new_environment_steps=0))
    require(len(raw_curves) == 90*198 and len(endpoints) == 210 and len(thresholds) == 300,
            'Published source arrays are incomplete')
    require(sum(row['charged_steps'] for row in costs) == 9054720, 'Formal charged learning total changed')
    return dict(raw=raw_curves, references=references, display=display_curves, series=series,
                endpoints=endpoints, endpoint_lookup=endpoint_lookup, thresholds=thresholds, costs=costs)


def setup_fonts(output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager as fm
    font_root = Path(os.environ.get('CH5_PAPER_FONT_ROOT', '/root/gamewar/latex_thesis_resource_allocation/assets/fonts/windows'))
    paths = [font_root/'TIMES.TTF', font_root/'simsun.ttc', font_root/'simhei.ttf']
    require(all(path.is_file() for path in paths), 'Times New Roman and SimSun font assets are required')
    for path in paths:
        fm.fontManager.addfont(str(path))
    names = [fm.FontProperties(fname=str(path)).get_name() for path in paths]
    plt.rcParams.update({'font.family': names[:2], 'font.size': 10.5, 'axes.labelsize': 10.5,
                         'axes.titlesize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
                         'legend.fontsize': 9.5, 'pdf.fonttype': 3, 'ps.fonttype': 3,
                         'axes.unicode_minus': False, 'pdf.compression': 6})
    warnings.filterwarnings('error', message=r'Glyph .* missing from font', category=UserWarning)
    return plt, dict(files={str(path): sha(path) for path in paths}, families=names)


def metric_label(domain, metric):
    return '达标对象数 K' if metric == 'K' else ('平均SSIM J' if domain == 'radio' else '平均任务完成比例 J（%）')


def metric_scale(domain, metric):
    return 100. if domain == 'v5' and metric == 'J' else 1.


def display_figure(plt, data, methods, title):
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(15.6/2.54, 5.1), sharex=True)
    for index, (ax, (domain, metric)) in enumerate(zip(axes.flat, (('radio', 'K'), ('v5', 'K'), ('radio', 'J'), ('v5', 'J')))):
        for method in methods:
            seed_values = [[row[metric] for row in data['series'][method, domain, seed]] for seed in SEEDS]
            smoothed = np.stack([five_point_display(values) for values in seed_values])*metric_scale(domain, metric)
            mean, sd = moments(smoothed)
            steps = [row['learning_steps'] for row in data['series'][method, domain, 0]]
            ax.plot(np.asarray(steps)/10000, mean, label=NAMES[method], color=COLORS[method], lw=1.35, ls=STYLES[method])
            ax.fill_between(np.asarray(steps)/10000, mean-sd, mean+sd, color=COLORS[method], alpha=.12, lw=0)
        ax.set_xlabel('学习交互步数（万步）')
        ax.set_ylabel(metric_label(domain, metric))
        ax.set_title(f"({chr(97+index)}) "+('无线通信' if domain == 'radio' else '应急供电'))
        ax.set_xlim(0, 10.0608)
        ax.grid(alpha=.2, linewidth=.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3 if len(methods) == 6 else 2, loc='lower center', bbox_to_anchor=(.5, .057), frameon=False)
    fig.suptitle(title, y=.995, fontsize=11)
    fig.text(.5, .012, '五种子均值±样本标准差；统一5点展示平滑，真实起点和终点保留；原始198点完整保存。', ha='center', fontsize=7)
    fig.subplots_adjust(left=.12, right=.98, top=.925, bottom=.235, wspace=.48, hspace=.6)
    return fig


def joint_figure(plt, data):
    fig, axes = plt.subplots(1, 2, figsize=(15.6/2.54, 3.65))
    markers = ('o', 's', '^', 'D', 'P', 'X')
    for index, (domain, ax) in enumerate(zip(DOMAINS, axes)):
        for method, marker in zip(MAIN, markers):
            rows = [data['endpoint_lookup'][method, domain, seed, 1.] for seed in SEEDS]
            k, ksd = moments([row['K'] for row in rows])
            j, jsd = moments([row['J']*metric_scale(domain, 'J') for row in rows])
            ax.errorbar(k, j, xerr=ksd, yerr=jsd, fmt=marker, color=COLORS[method], label=NAMES[method],
                        markersize=5.8, capsize=2, elinewidth=.8, lw=0, zorder=3)
        ax.set_xlabel('达标对象数 K')
        ax.set_ylabel(metric_label(domain, 'J'))
        ax.set_title(f"({chr(97+index)}) "+('无线通信' if domain == 'radio' else '应急供电'))
        ax.grid(alpha=.2, linewidth=.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(.5, .08), frameon=False)
    fig.suptitle('固定终点：达标数量与原任务指标的关系', y=.99, fontsize=11)
    fig.text(.5, .015, '每法一个五种子均值点，误差线为样本标准差；右上方向表示两项同时改善，交叉高低表示取舍。', ha='center', fontsize=7)
    fig.subplots_adjust(left=.12, right=.98, top=.84, bottom=.35, wspace=.5)
    return fig


def threshold_figure(plt, data, contract):
    fig, axes = plt.subplots(1, 2, figsize=(15.6/2.54, 3.65))
    for index, (domain, ax) in enumerate(zip(DOMAINS, axes)):
        x = contract['evaluation']['thresholds'][domain]
        for method in MAIN:
            rows = [row for row in data['thresholds'] if row['algorithm'] == method and row['domain'] == domain]
            values = [[next(row['K'] for row in rows if row['seed'] == seed and row['task_threshold'] == threshold) for threshold in x] for seed in SEEDS]
            mean, sd = moments(values)
            ax.plot(x, mean, marker=MARKERS[method], markersize=3, color=COLORS[method], label=NAMES[method], lw=1.2, ls=STYLES[method])
            ax.fill_between(x, mean-sd, mean+sd, color=COLORS[method], alpha=.1, lw=0)
        ax.axvline(contract['domains'][domain]['task_tau'], ls='--', color='#555555', lw=.8)
        ax.set_xlabel('SSIM达标门槛' if domain == 'radio' else '任务完成比例达标门槛')
        ax.set_ylabel('达标对象数 K')
        ax.set_title(f"({chr(97+index)}) "+('无线通信' if domain == 'radio' else '应急供电'))
        ax.grid(alpha=.2, linewidth=.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(.5, .08), frameon=False)
    fig.suptitle('固定策略在不同统计门槛下的达标数量', y=.99, fontsize=11)
    fig.text(.5, .015, '对同一批固定策略的既有任务结果重新计数；虚线为训练门槛；此图不表示更换门槛后的重训或策略适应。', ha='center', fontsize=6.9)
    fig.subplots_adjust(left=.12, right=.98, top=.84, bottom=.35, wspace=.5)
    return fig


def budget_figure(plt, data):
    fig, axes = plt.subplots(2, 2, figsize=(15.6/2.54, 5.1))
    for index, (ax, (domain, metric)) in enumerate(zip(axes.flat, (('radio', 'K'), ('v5', 'K'), ('radio', 'J'), ('v5', 'J')))):
        for method in MAIN:
            values = [[data['endpoint_lookup'][method, domain, seed, scale][metric]*metric_scale(domain, metric)
                       for scale in (.8, 1., 1.2)] for seed in SEEDS]
            mean, sd = moments(values)
            ax.errorbar([.8, 1., 1.2], mean, yerr=sd, label=NAMES[method], color=COLORS[method],
                        marker=MARKERS[method], markersize=3, capsize=2, lw=1.2, elinewidth=.7, ls=STYLES[method])
        ax.set_xticks([.8, 1., 1.2], ['80%', '100%', '120%'])
        ax.set_xlabel('可用预算相对训练预算的比例')
        ax.set_ylabel(metric_label(domain, metric))
        ax.set_title(f"({chr(97+index)}) "+('无线通信' if domain == 'radio' else '应急供电'))
        ax.grid(alpha=.2, linewidth=.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(.5, .057), frameon=False)
    fig.suptitle('固定模型在预算变化后的表现', y=.995, fontsize=11)
    fig.text(.5, .012, '模型和已学归一化冻结，在80%/100%/120%预算下重新运行同一256情景；未追加训练。', ha='center', fontsize=7)
    fig.subplots_adjust(left=.12, right=.98, top=.925, bottom=.235, wspace=.48, hspace=.6)
    return fig


def metric_table(data, methods):
    rows = []
    for domain in DOMAINS:
        for method in methods:
            cases = [data['endpoint_lookup'][method, domain, seed, 1.] for seed in SEEDS]
            cells = []
            for metric, scale in [('K', 1.), ('C', 100.), ('J', metric_scale(domain, 'J')), ('F_0_8', 1.)]:
                mean, sd = moments([row[metric]*scale for row in cases])
                cells.append(f'{mean:.3f} ± {sd:.3f}')
            rows.append(['无线通信' if domain == 'radio' else '应急供电', NAMES[method], *cells])
    return ['场景', '方法', '达标对象数 K', '达标比例 C（%）', '原任务指标 J', '辅助 F₀.₈'], rows


def table_figure(plt, title, columns, rows, note, fontsize=10):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=columns, cellLoc='center', loc='center', bbox=[0, .18, 1, .76],
                     colWidths=[.17, .83] if len(columns) == 2 else None)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(.35)
        cell.set_facecolor('#dce7ef' if r == 0 else ('#f3f6f8' if r % 2 == 0 else 'white'))
    fig.suptitle(title, y=.97, fontsize=15)
    fig.text(.045, .07, note, fontsize=10)
    fig.subplots_adjust(left=.035, right=.985, top=.92, bottom=.06)
    return fig


def latex_table(path, title, label, columns, rows, note):
    def safe(value):
        return str(value).replace('_', r'\_').replace('%', r'\%').replace('±', r'$\pm$').replace('F₀.₈', r'$F_{0.8}$')
    lines = [r'\begin{table}[htbp]', r'\centering', '\\caption{'+title+'}', '\\label{'+label+'}', r'\small']
    if len(columns) != 2:
        lines.append(r'\resizebox{\textwidth}{!}{%')
    specification = r'p{2.4cm}p{11.6cm}' if len(columns) == 2 else 'l'*len(columns)
    lines += ['\\begin{tabular}{'+specification+'}',
             r'\toprule', ' & '.join(map(safe, columns))+r' \\', r'\midrule']
    lines += [' & '.join(map(safe, row))+r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}'+('}' if len(columns) != 2 else ''), r'\par\smallskip',
              r'\parbox{0.97\textwidth}{\footnotesize '+safe(note)+'}', r'\end{table}']
    Path(path).write_text('\n'.join(lines)+'\n')


def render_pdf(pdf, output):
    import fitz
    import numpy as np
    document = fitz.open(pdf)
    require(len(document) == 9, 'The independent report must have nine actual pages')
    pages = []
    folder = output/'rendered_pages'
    folder.mkdir(exist_ok=True)
    for index, page in enumerate(document, 1):
        image = page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False)
        destination = folder/f'page{index:02d}.png'
        image.save(destination)
        array = np.frombuffer(image.samples, dtype=np.uint8)
        require(min(image.width, image.height) > 300 and float(array.std()) > 5, 'Blank or malformed rendered PDF page')
        pages.append(dict(page=index, image=str(destination), image_sha256=sha(destination),
                          width=image.width, height=image.height, pdf_text_characters=len(page.get_text())))
    document.close()
    return dict(status='PASS_ACTUAL_PDF_RENDER_NONBLANK', pages=pages,
                agent_visual_review='pending inspection of actual rendered pages')


def make_figures(plt, data, contract):
    return {'learning': display_figure(plt, data, MAIN, '六方法的学习过程：达标数量与原任务指标'),
            'joint_endpoint': joint_figure(plt, data),
            'ablations': display_figure(plt, data, ABLATIONS, '关键部分消融：数量与原任务指标同时观察'),
            'threshold_recount': threshold_figure(plt, data, contract),
            'budget_transfer': budget_figure(plt, data)}


def save_paper_figure(figure, destination):
    texts = list(figure.texts)
    for text in texts:
        text.set_visible(False)
    for ax in figure.axes:
        ax.title.set_fontfamily(['Times New Roman', 'SimHei'])
    for extension in ('pdf', 'svg', 'png'):
        figure.savefig(str(destination)+'.'+extension, dpi=200)
    for text in texts:
        text.set_visible(True)


def replot_from_csv(source, output, selected=None):
    """Recreate assets solely from the adjacent sealed small plotting inputs."""
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    require(output.is_relative_to('/hy-tmp'), 'Generated plotting assets and caches must remain on HY')
    os.environ['MPLCONFIGDIR'] = str(output/'cache/matplotlib')
    os.environ['XDG_CACHE_HOME'] = str(output/'cache')
    manifest = read(source/'plot_inputs_manifest.json')
    for filename, expected in manifest['files'].items():
        require(sha(source/filename) == expected, 'Local plotting data changed: '+filename)
    def load(filename):
        with (source/filename).open(encoding='utf-8-sig') as stream:
            values = list(csv.DictReader(stream))
        for row in values:
            for key in ('K', 'C', 'J', 'F_0_8', 'scale', 'task_threshold'):
                if key in row:
                    row[key] = float(row[key])
            for key in ('seed', 'learning_steps'):
                if key in row:
                    row[key] = int(row[key])
        return values
    data = dict(series=defaultdict(list), endpoint_lookup={}, thresholds=load('threshold_recounts_by_seed.csv'))
    for row in load('learning_raw.csv'):
        data['series'][row['algorithm'], row['domain'], row['seed']].append(row)
    for row in load('fixed_endpoints_by_seed.csv'):
        data['endpoint_lookup'][row['algorithm'], row['domain'], row['seed'], row['scale']] = row
    plt, _ = setup_fonts(output)
    figures = make_figures(plt, data, read(source/'plot_contract.json'))
    require(selected is None or selected in figures, 'Unknown figure selected')
    output.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        if selected is None or name == selected:
            save_paper_figure(figure, output/name)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-preparation-only', action='store_true')
    parser.add_argument('--replot-csv', type=Path)
    parser.add_argument('--figure', choices=['learning', 'joint_endpoint', 'ablations', 'threshold_recount', 'budget_transfer'])
    args = parser.parse_args()
    if args.replot_csv is not None:
        replot_from_csv(args.replot_csv, args.output or args.replot_csv/'regenerated', args.figure)
        return
    root = args.root.resolve(strict=True)
    output = (args.output or root/'reports/final').resolve()
    require(root.is_relative_to('/hy-tmp') and output.is_relative_to(root/'reports'), 'All report outputs must remain under registered HY reports')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['MPLCONFIGDIR'] = str(root/'cache/matplotlib_report')
    os.environ['XDG_CACHE_HOME'] = str(root/'cache')
    plt, fonts = setup_fonts(output)
    if args.validate_preparation_only:
        import numpy as np
        example = np.asarray([8., 4., 7., 2., 9., 1., 6.])
        displayed = five_point_display(example)
        require(displayed[0] == example[0] and displayed[-1] == example[-1], 'Filter endpoint check failed')
        mean, sd = moments([[float(seed), 1.] for seed in SEEDS])
        require(mean.tolist() == [2., 1.] and abs(sd[0]-math.sqrt(2.5)) < 1e-12 and sd[1] == 0., 'Five-seed sample SD check failed')
        print(json.dumps(dict(status='PASS_REPORT_CODE_PREPARATION_ONLY', environment_steps=0,
                              formal_data_read=False, figures_published=0, fonts=fonts), ensure_ascii=False))
        return
    require(not (output/'publication_ready.json').exists(), 'Published reports are immutable; choose a new report directory for revised presentation')
    bundle, lookup = audit_inputs(root)
    data = assemble(root, bundle, lookup)
    output.mkdir(parents=True, exist_ok=True)
    assets = output/'paper_assets'
    assets.mkdir(exist_ok=True)
    for filename, rows in [('learning_raw.csv', data['raw']), ('learning_display.csv', data['display']),
                           ('fixed_endpoints_by_seed.csv', data['endpoints']), ('threshold_recounts_by_seed.csv', data['thresholds']),
                           ('costs_by_seed.csv', data['costs'])]:
        output_csv(output/filename, rows)
    if data['references']:
        output_csv(output/'calibration_events_raw.csv', data['references'])
    contract = read(root/'contract.json')
    figures = make_figures(plt, data, contract)
    for name, figure in figures.items():
        save_paper_figure(figure, assets/name)
    for filename in ('learning_raw.csv', 'learning_display.csv', 'fixed_endpoints_by_seed.csv',
                     'threshold_recounts_by_seed.csv', 'costs_by_seed.csv'):
        shutil.copy2(output/filename, assets/filename)
    write(assets/'plot_contract.json', contract)
    write(assets/'plot_inputs_manifest.json', dict(schema='chapter5-formal-sealed-plot-inputs-v1',
          files={p.name: sha(p) for p in sorted(assets.glob('*.csv'))} | {'plot_contract.json': sha(assets/'plot_contract.json')},
          source_bundle_sha256=sha(root/'report_bundle.json'), display_filter='centered5_internal_only_raw_endpoints'))
    shutil.copy2(__file__, assets/'plot_assets.py')
    settings = [['训练方法', '本章方法、三项消融、五个外部方法'], ['场景', '无线通信（100对象/32期）、应急供电（120对象/24期）'],
                ['训练种子', '0、1、2、3、4；每种方法每场景五个种子'], ['每模型训练长度', '100608个学习交互步'],
                ['开发评价', '每512步32情景；另含0步与100608终点，共198点'], ['固定终点测试', '90个模型；每模型256个共同独立情景'],
                ['预算迁移', '六种主方法，80%/100%/120%预算，固定模型重新评价'], ['门槛分析', '对固定策略已有任务结果按不同门槛重新计数'],
                ['初始化', '本章方法/消融的总量与区域两头随机初始化；外部方法按各自原生种子初始化'],
                ['保存与评价原则', '每项一个固定终点；训练期间不打开终测；不选峰值或训练种子']]
    tables = {}
    tables['settings'] = ('正式实验设置', ['项目', '设置'], settings, '完整原生方法配置、逐种子设备与源文件身份见本实验合同及可重绘数据。')
    columns, rows = metric_table(data, MAIN)
    note = '数值为五个训练种子的均值±样本标准差，每种子先平均256共同测试情景。无线J为SSIM，应急J为百分比。'
    tables['endpoints'] = ('六方法固定终点测试结果', columns, rows, note)
    columns, rows = metric_table(data, ABLATIONS)
    tables['ablation_endpoints'] = ('本章方法及三项消融的固定终点结果', columns, rows, note)
    cost_rows = []
    for domain in DOMAINS:
        for method in MAIN+ABLATIONS[1:]:
            rows = [row for row in data['costs'] if row['algorithm'] == method and row['domain'] == domain]
            require(len(rows) == 5, 'Cost group incomplete')
            avg = lambda key: sum(row[key] for row in rows)/5
            cost_rows.append(['无线' if domain == 'radio' else '应急', NAMES[method],
                              *[f'{avg(key):.0f}' for key in ('train_steps', 'internal_steps', 'development_steps', 'reload_steps',
                                                           'default_test_steps', 'transfer_test_steps', 'native_update_records')],
                              f'{avg("seconds")/3600:.2f}'])
    tables['costs'] = ('交互与更新开销：每训练种子的平均值',
                       ['场景', '方法', '训练步', '内部采样', 'DEV步', '重载步', '默认终测', '迁移终测', '原生更新组', '小时'],
                       cost_rows, '训练步与内部采样之和均为100608；更新组依各方法原生定义。共享CPU/GPU、混合精度与并行条件下，耗时不作算法速度排名。')
    table_figures = {}
    for name, (title, columns, rows, note) in tables.items():
        latex_table(assets/(name+'.tex'), title, 'tab:ch5-formal-'+name.replace('_', '-'), columns, rows, note)
        table_figures[name] = table_figure(plt, title, columns, rows, note, 8.1 if name == 'costs' else (9.1 if name == 'settings' else 10))
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = output/'第五章_正式五种子_主比较消融与条件分析.pdf'
    order = [table_figures['settings'], figures['learning'], figures['joint_endpoint'], table_figures['endpoints'],
             figures['ablations'], table_figures['ablation_endpoints'], figures['threshold_recount'],
             figures['budget_transfer'], table_figures['costs']]
    with PdfPages(pdf) as pages:
        for figure in order:
            pages.savefig(figure)
    for figure in order:
        plt.close(figure)
    render = render_pdf(pdf, output)
    write(output/'actual_pdf_render_audit.json', render)
    receipt = dict(schema=SCHEMA, status='PASS',
                   root=str(root), models=90, fixed_evaluations=210, formal_training_seeds=list(SEEDS),
                   learning_steps=100608, raw_curve_points=198, report_bundle_sha256=sha(root/'report_bundle.json'),
                   independent_evaluation_audit_sha256=sha(root/'final_evaluation_audit.json'),
                   script_sha256=sha(__file__), fonts=fonts, independent_pdf=dict(path=str(pdf), sha256=sha(pdf)),
                   paper_assets={p.name: dict(path=str(p), sha256=sha(p)) for p in sorted(assets.iterdir())},
                   source_csv={p.name: dict(path=str(p), sha256=sha(p)) for p in sorted(output.glob('*.csv'))},
                   display_smoothing=dict(window=5, mode='centered_internal_points_per_seed',
                                          raw_initial_and_endpoint_preserved=True, sample_sd_after_same_per_seed_filter=True),
                   raw_numbers_changed=False, best_checkpoint_or_seed_selection=False, extra_environment_steps=0,
                   new_optimizer_steps=0, model_files_modified=False, thesis_modified=False,
                   threshold_semantics=bundle['threshold_plot_semantics'],
                   manual_or_agent_visual_review='pending; actual PDF render and nonblank checks completed',
                   render_audit=dict(path=str(output/'actual_pdf_render_audit.json'),
                                                                 sha256=sha(output/'actual_pdf_render_audit.json')))
    write(output/'publication_ready.json', receipt)
    print(json.dumps(dict(status=receipt['status'], pdf=str(pdf), figures=5, tables=4, new_environment_steps=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
