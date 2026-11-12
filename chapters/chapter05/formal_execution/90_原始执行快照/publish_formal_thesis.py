"""Publish the actually completed formal cohort into the live Chapter 5 only.

Preparation never edits the manuscript. Actual publication requires the complete
90-model/210-panel audit chain, reads live prose under the shared build lock,
preserves other chapters, and builds the whole thesis through scripts/build.sh.
"""
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict
import argparse
import csv
import datetime
import fcntl
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = Path('chapters/chapter05_dual_value')
SECTION = CHAPTER/'section63_experiments.tex'
SUMMARY = CHAPTER/'section64_summary.tex'
FIGURES = (
    ('5.4', 'learning', '六种方法达标数量与原任务指标的学习过程', 'fig:ch5-validation-learning'),
    ('5.5', 'joint_endpoint', '六种方法固定终点的达标数量与原任务指标', 'fig:ch5-validation-tradeoff'),
    ('5.6', 'ablations', '目标构成与价值监督的消融学习过程', 'fig:ch5-validation-ablation'),
    ('5.7', 'threshold_recount', '固定策略在不同统计门槛下的达标数量', 'fig:ch5-validation-threshold'),
    ('5.8', 'budget_transfer', '固定模型在预算变化后的达标数量与原任务指标', 'fig:ch5-validation-budget'))
spec = importlib.util.spec_from_file_location('chapter5_formal_report_publication_helpers', ROOT/'execution/report_formal.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
require, read, sha, write = report.require, report.read, report.sha, report.write


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.formal-tmp')
    temporary.write_bytes(payload)
    temporary.replace(path)


def snapshot(thesis):
    result = {}
    for folder in ('chapters', 'misc', 'reference', 'scripts', 'assets'):
        for path in (thesis/folder).rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.tex', '.bib', '.py', '.sh', '.drawio', '.ttf', '.ttc'):
                result[str(path.relative_to(thesis))] = sha(path)
    return result


def compilation_inputs(thesis):
    """Hash active sources and every persistent actual input in the latest FLS."""
    suffixes = {'.tex', '.bib', '.cls', '.sty', '.cfg', '.def', '.fd', '.clo', '.ttf', '.ttc',
                '.otf', '.pfb', '.enc', '.map', '.pdf', '.png', '.jpg', '.jpeg', '.eps',
                '.bst', '.bbx', '.cbx', '.lbx', '.drawio', '.sh', '.py'}
    paths = set()
    for folder in ('chapters', 'misc', 'reference', 'scripts', 'assets'):
        for path in (thesis/folder).rglob('*'):
            if path.is_file() and path.suffix.lower() in suffixes and path.suffix.lower() not in {'.pdf', '.png', '.jpg', '.jpeg', '.eps'}:
                paths.add(path.resolve())
    for path in thesis.iterdir():
        if path.is_file() and (path.suffix.lower() in suffixes or path.name == '.latexmkrc'):
            paths.add(path.resolve())
    fls = thesis/'build/main.fls'
    require(fls.is_file(), 'Actual prior whole-thesis FLS input list required')
    for line in fls.read_text(errors='replace').splitlines():
        if line.startswith('INPUT '):
            path = Path(line[6:].strip().strip('"'))
            if not path.is_absolute():
                path = thesis/path
            # Include every real FLS input, including the actual .bbl/.aux/.toc
            # and format files; their first-build change legitimately requires
            # a second stable build before the PDF can be sealed.
            if path.is_file():
                paths.add(path.resolve())
    require(any(path.suffix.lower() == '.pdf' for path in paths), 'FLS contains no actual inserted figure PDFs')
    require(any(path.suffix.lower() in ('.ttf', '.ttc', '.otf', '.pfb') for path in paths), 'Actual font inputs missing from closure')
    return {str(path): sha(path) for path in sorted(paths)}


@contextmanager
def stable_whole_build(thesis, planned, destination):
    """Yield with the whole lock held only for an input-consistent actual PDF."""
    rounds = []
    for attempt in range(1, 4):
        with locked(thesis/'build/.whole_thesis_build.lock'):
            for relative, payload in planned.items():
                require((thesis/relative).read_bytes() == payload, 'Concurrent writer changed controlled Chapter5 content')
            before = compilation_inputs(thesis)
            before_pdf = sha(thesis/'build/main.pdf')
            write(destination/f'build_round{attempt}_input_before.json', before)
        started_ns = time.time_ns()
        log = destination/f'whole_thesis_build_round{attempt}.log'
        with log.open('w') as stream:
            completed = subprocess.run(['bash', 'scripts/build.sh'], cwd=thesis, stdout=stream, stderr=subprocess.STDOUT)
        require(completed.returncode == 0, 'Whole-thesis build failed; originals/log retained: '+str(log))
        with locked(thesis/'build/.whole_thesis_build.lock'):
            for relative, payload in planned.items():
                require((thesis/relative).read_bytes() == payload, 'Concurrent writer changed controlled Chapter5 content during build')
            after = compilation_inputs(thesis)
            write(destination/f'build_round{attempt}_input_after.json', after)
            pdf, fls = thesis/'build/main.pdf', thesis/'build/main.fls'
            entry = dict(round=attempt, build_command=['bash', 'scripts/build.sh'], returncode=completed.returncode,
                         input_closure_equal=before == after, input_files=len(after),
                         before_closure_sha256=sha(destination/f'build_round{attempt}_input_before.json'),
                         after_closure_sha256=sha(destination/f'build_round{attempt}_input_after.json'),
                         build_log_sha256=sha(log), current_pdf_sha256=sha(pdf), pdf_sha256_before_call=before_pdf,
                         build_call_started_unix_ns=started_ns, pdf_mtime_ns=pdf.stat().st_mtime_ns,
                         fls_mtime_ns=fls.stat().st_mtime_ns, final_fls_sha256=sha(fls),
                         actual_fls_inputs_including_generated_auxiliaries_hashed=True,
                         pdf_regenerated_since_call_started=pdf.stat().st_mtime_ns >= started_ns,
                         equivalent_build_with_identical_inputs_by_another_writer_acceptable=True,
                         unique_build_authorship_claim=False)
            rounds.append(entry)
            write(destination/'build_rounds.json', rounds)
            if before == after:
                log_text = (thesis/'build/main.log').read_text(errors='replace')
                require('There were undefined references' not in log_text and 'There were undefined citations' not in log_text,
                        'Final whole-thesis log has unresolved references')
                yield rounds
                return
    raise ValueError('Live compilation inputs changed during three builds; latest other sources preserved, publication remains unsealed')


def statistics(directory):
    groups = defaultdict(list)
    with (directory/'fixed_endpoints_by_seed.csv').open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            groups[row['algorithm'], row['domain'], float(row['scale'])].append(row)
    result = {}
    for key, rows in groups.items():
        require(sorted(int(row['seed']) for row in rows) == list(range(5)), 'Five-seed endpoint group incomplete')
        result[key] = {metric: sum(float(row[metric]) for row in rows)/5 for metric in ('K', 'C', 'J', 'F_0_8')}
    require(len(result) == 42, 'Eighteen default and twenty-four changed-budget groups required')
    return result


def domain_name(domain):
    return '图像下行传输任务' if domain == 'radio' else '应急资源配置任务'


def names(methods):
    return '、'.join('SeDVal' if method == 'full' else report.NAMES[method] for method in methods)


def main_result(data, domain):
    full = data['full', domain, 1.]
    text = domain_name(domain)+f"中，SeDVal的平均达标数为{full['K']:.2f}，"
    text += (f"平均SSIM为{full['J']:.3f}。" if domain == 'radio' else f"平均任务完成比例为{full['J']*100:.2f}\\%。")
    higher, lower, trade, tied = [], [], [], []
    for method in report.MAIN[1:]:
        other = data[method, domain, 1.]
        if full['K'] == other['K'] and full['J'] == other['J']:
            tied.append(method)
        elif full['K'] >= other['K'] and full['J'] >= other['J'] and (full['K'] > other['K'] or full['J'] > other['J']):
            higher.append(method)
        elif other['K'] >= full['K'] and other['J'] >= full['J'] and (other['K'] > full['K'] or other['J'] > full['J']):
            lower.append(method)
        else:
            trade.append(method)
    if higher:
        text += '相对'+names(higher)+'，SeDVal的两项终点均值均不低于对照，且至少一项更高。'
    if lower:
        text += names(lower)+'的两项终点均值均不低于SeDVal，且至少一项更高，本组结果未显示SeDVal对这些方法的联合优势。'
    if trade:
        text += '与'+names(trade)+'的比较中，两项均值各有高低，体现了覆盖数量与原任务指标之间的取舍。'
    if tied:
        text += '与'+names(tied)+'的两项终点均值相同，本组均值不区分其表现。'
    return text


def ablation_result(data, domain):
    full = data['full', domain, 1.]
    parts = [domain_name(domain)+'的同结构比较中，']
    scale = 100 if domain == 'v5' else 1
    unit = '个百分点' if domain == 'v5' else ''
    for method in report.ABLATIONS[1:]:
        row = data[method, domain, 1.]
        parts.append(report.NAMES[method]+f"相对完整方法的平均达标数差为{row['K']-full['K']:+.2f}，"
                     +('平均SSIM差为' if domain == 'radio' else '平均任务完成比例差为')
                     +f"{(row['J']-full['J'])*scale:+.3f}{unit}。")
    parts.append('上述差值分别保留目标取舍与监督形式的结果，均值接近不构成等效证明；未进行显著性检验，不据此宣称稳定优势。')
    return ''.join(parts)


def figure(directory, title, label):
    return '\\begin{figure}[H]\n \\centering\n \\includegraphics{'+str(directory/'figure.pdf')+'}\n \\caption{'+title+'}\n \\label{'+label+'}\n\\end{figure}'


def environment_and_methods(current):
    environment = r'\subsubsection{实验环境}'
    methods = r'\subsubsection{对比方法}'
    training = r'\subsubsection{训练过程与算法参数设置}'
    require(all(current.count(marker) == 1 for marker in (environment, methods, training)), 'Current C5 setting anchors changed; merge must be reviewed')
    return current[current.index(environment):current.index(training)]


def make_section(current, data, directories, input_data):
    settings = environment_and_methods(current)
    graphics = {stem: figure(directories[number], title, label) for number, stem, title, label in FIGURES}
    prefix = r'''% !TeX root = ../../main.tex
% Actual formal cohort: five seeds, 100608 steps, ninety fixed models and 210 audited panels.
\clearpage
\section{实验结果与分析}
\label{sec:ch6-experiments}
实验在图像下行传输与应急资源配置任务中评价SeDVal对达标数量和原任务指标的联合优化。方法比较分析两项目标的配置结果，同结构对照考察目标构成与价值监督的作用，门槛和预算条件进一步说明评价标准与资源变化对结果的影响。

\subsection{实验设置}
\label{subsec:ch6-experiments-1}
'''
    training = r'''
\subsubsection{训练过程与算法参数设置}
完整方法与三个同结构对照均采用总量分支和区域分支输出头的随机初始化，五个对比方法采用各自原生的随机种子初始化。所有模型均从头训练100608个学习交互步，采用0至4五个训练种子。六种主方法在两个任务中形成60个模型，三个消融对照另有30个模型，完整方法的10个模型在两类分析中共用。MORL/D的内部采样计入同一学习预算，开发与独立测试交互另计。

SeDVal保留两个独立价值网络与共同策略更新，采用Adam优化器。首32回合每批采集8条新轨迹，之后每批使用4条新轨迹与最近4条旧轨迹。MORL/D固定采用价值学习率0.0001的实现，策略学习率为0.0003，各对比方法的其余原生设置保持。训练中间奖励为零，折扣系数为1，在真实终点停止自举。模型均取固定训练终点，不按开发峰值选择检查点，主要设置见表\ref{tab:ch5-formal-settings}。
\input{chapters/chapter05_dual_value/tables/formal_settings.tex}

\subsubsection{评价指标与统计方法}
\label{par:ch6-experiment-metrics}
终点达标对象数\(K\)与原任务指标\(J\)分别报告。通信任务的\(J\)为平均SSIM，SSIM不低于0.20时记为达标。应急任务的\(J\)为平均任务完成比例，完成比例不低于0.70时记为达标，无需求区域视为已完成。平均SSIM保留负值，应急完成比例以百分比报告。应急门槛仍为探索性实验设置，不作为经业务认证的服务标准，覆盖率\(C=K/N\)和\(F_{0.8}=0.8C+0.2J\)作为辅助统计。

开发评价每512步采用32条固定情景，保留真实0步并补充100608步固定终点，共198个数据点。展示曲线对每个训练种子的内部点统一作5点居中滑动平均，起点和终点保留原值，再计算五种子均值与样本标准差；原始数据完整保存。所有90个模型训练完成并固定后才打开共同独立测试，每个模型采用256条情景，先求模型内情景均值，再汇总五个训练种子的均值与样本标准差。训练、开发与测试采用不同外生随机序列，各方法共享开发与测试情景，统计重复单位为训练种子。

\subsection{双目标配置性能与结构分析}
\label{subsec:ch6-experiments-2}
\subsubsection{配置性能比较}
六种方法在同一学习预算下的开发过程同时保留达标数量与原任务指标，见图\ref{fig:ch5-validation-learning}。两项指标的变化共同说明学习中的取舍，单项下降不能单独代表另一项目标也退化。固定终点的联合表现见图\ref{fig:ch5-validation-tradeoff}，各方法以五种子均值点和样本标准差表示，右上方向对应两项共同改善。
'''
    parts = [prefix, settings, training, graphics['learning'], graphics['joint_endpoint'],
             r'\input{chapters/chapter05_dual_value/tables/formal_endpoints.tex}',
             main_result(data, 'radio'), main_result(data, 'v5'),
             '两项各优一方时保留其取舍，辅助组合值不替代单项结果。固定偏好下的一个部署点不构成完整多目标前沿。',
             r'\subsubsection{目标构成与价值监督分析}', r'\label{subsec:ch6-experiments-3}',
             r'仅原任务目标按\(J\)更新策略并冻结覆盖价值网络，仅达标数量按\(C\)更新策略并冻结原任务价值网络，两组同时改变目标与有效价值监督。标量价值监督保留两座价值网络与共同初始化，只对按0.8和0.2加权的预测和施加误差损失，不再分别计算两项价值损失。三个对照与完整方法采用相同的随机初始化、五个种子和训练预算，学习过程见图\ref{fig:ch5-validation-ablation}，固定终点结果见表\ref{tab:ch5-formal-ablation-endpoints}。',
             graphics['ablations'], r'\input{chapters/chapter05_dual_value/tables/formal_ablation_endpoints.tex}',
             ablation_result(data, 'radio'), ablation_result(data, 'v5'),
             r'各方法的交互与更新开销见表\ref{tab:ch5-formal-costs}。原生更新组包含的优化小步不同，共享CPU与GPU的并发耗时也不作为算法速度排名。',
             r'\input{chapters/chapter05_dual_value/tables/formal_costs.tex}',
             r'\subsection{评价门槛与资源条件分析}', r'\subsubsection{评价门槛变化分析}',
             r'对同一批固定策略的既有任务结果按不同门槛重新计数，达标数量的变化见图\ref{fig:ch5-validation-threshold}。通信门槛为0.10至0.30，应急门槛为0.50至0.90；策略与实际配置保持不变，原任务指标\(J\)也保持不变。这一分析说明评价标准的敏感性，不表示在新门槛下重新训练或策略适应。', graphics['threshold_recount']]
    for domain in report.DOMAINS:
        threshold_values = input_data['thresholds']
        xs = sorted({row['task_threshold'] for row in threshold_values if row['domain'] == domain})
        full = [sum(row['K'] for row in threshold_values if row['algorithm'] == 'full' and row['domain'] == domain and row['task_threshold'] == x)/5 for x in xs]
        others = [method for method in report.MAIN[1:] if sum(row['K'] for row in threshold_values if row['algorithm'] == method and row['domain'] == domain and row['task_threshold'] == xs[-1])/5 > full[-1]]
        text = domain_name(domain)+f'中，门槛由{xs[0]:.2f}提高到{xs[-1]:.2f}时，SeDVal平均达标数由{full[0]:.2f}变为{full[-1]:.2f}。'
        if others:
            text += '在最严格的所测门槛下，'+names(others)+'的达标均值高于SeDVal，方法排序应结合评价标准理解。'
        parts.append(text)
    parts.extend([r'\subsubsection{资源预算变化分析}',
                  r'固定模型分别在标准预算的0.8倍、1倍和1.2倍下重新运行共同测试情景，结果见图\ref{fig:ch5-validation-budget}。物理响应、需求与对象上限保持不变，模型参数、观测归一化以及部署成员均冻结，各条件仍满足期末全额分配。该实验检验预算变化后的部署表现，不加入新的梯度更新。', graphics['budget_transfer']])
    for domain in report.DOMAINS:
        full = [data['full', domain, scale] for scale in (.8, 1., 1.2)]
        js = [row['J']*(100 if domain == 'v5' else 1) for row in full]
        text = domain_name(domain)+f"中，三档预算下SeDVal平均达标数依次为{full[0]['K']:.2f}、{full[1]['K']:.2f}和{full[2]['K']:.2f}，"
        text += ('平均SSIM依次为' if domain == 'radio' else '平均任务完成比例依次为')+f'{js[0]:.3f}、{js[1]:.3f}和{js[2]:.3f}'+('。' if domain == 'radio' else r'\%。')
        for scale in (.8, 1.2):
            own = data['full', domain, scale]
            higher = [m for m in report.MAIN[1:] if data[m, domain, scale]['K'] >= own['K'] and data[m, domain, scale]['J'] >= own['J'] and (data[m, domain, scale]['K'] > own['K'] or data[m, domain, scale]['J'] > own['J'])]
            if higher:
                text += f'在{scale:.1f}倍预算下，'+names(higher)+'的两项均值均不低于SeDVal，且至少一项更高。'
        parts.append(text)
    parts.append('这些结果限于本组任务、五个训练种子和三档预算，不推断任意资源条件下的统一排序，也不构成逐对象达标保证。')
    result = '\n\n'.join(parts)+'\n'
    require(result.count(r'\subsection{') == 3, 'Exactly three experiment subsections required')
    require(all(word not in result for word in ('两个训练种子', '98304', '9840', '等权平均', '经验累积', 'CDF')), 'Obsolete identity or unnecessary terminology entered the new section')
    return result


def make_summary(current, data):
    start = current.index(r'\label{sec:ch6-summary}')+len(r'\label{sec:ch6-summary}')
    text = current[start:].lstrip()
    theory = text.split('\n\n', 1)[0]
    return current[:start]+'\n'+theory+'\n\n'+main_result(data, 'radio')+'\n\n'+main_result(data, 'v5')+'\n\n'+ablation_result(data, 'radio')+'\n\n'+ablation_result(data, 'v5')+'\n\n固定策略的门槛重新计数说明评价标准对达标数量的影响，三档预算评价说明资源变化后的部署表现。结论基于100608步、五个训练种子和本轮共同独立情景，未作逐对象硬保障或任意条件下统一占优的判断。\n'


def check_preparation(thesis):
    require((thesis/'scripts/build.sh').is_file(), 'Sole whole-thesis build entry missing')
    agents = thesis/'AGENTS.md'
    rules = thesis.parent/'论文写作规范汇总_20261014/写作强制规则_完整执行版.rst'
    require(agents.is_file() and rules.is_file(), 'Current manuscript instructions missing')
    current = (thesis/SECTION).read_text()
    environment_and_methods(current)
    require(current.count(r'\subsection{') == 3, 'Live manuscript three-subsection structure changed')
    with (thesis/'chapters/FIGURE_INDEX.csv').open(encoding='utf-8-sig') as stream:
        index = list(csv.DictReader(stream))
    for number, _, _, _ in FIGURES:
        require(sum(row['图号'] == number for row in index) == 1, 'Current C5 figure registry changed')
    return dict(status='PASS_PUBLICATION_CODE_PREPARATION_ONLY', manuscript_writes=0,
                agents_sha256=sha(agents), writing_rules_sha256=sha(rules),
                live_experiments_sha256=sha(thesis/SECTION), live_summary_sha256=sha(thesis/SUMMARY))


def render_thesis(pdf, output):
    import fitz
    document = fitz.open(pdf)
    targets = set()
    for index, page in enumerate(document):
        text = page.get_text()
        if any(token in text for token in ('100608', '图5.4', '图5.5', '图5.6', '图5.7', '图5.8')) and 'SeDVal' in text:
            targets.update([i for i in (index-1, index, index+1) if 0 <= i < len(document)])
    require(bool(targets), 'Cannot locate actual updated Chapter 5 PDF pages')
    output.mkdir(parents=True, exist_ok=True)
    pages = []
    for index in sorted(targets):
        page = document[index]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        destination = output/f'page{index+1:03d}.png'
        pix.save(destination)
        pages.append(dict(reader_page=index+1, image=str(destination), sha256=sha(destination),
                          text_characters=len(page.get_text()), embedded_fonts=page.get_fonts()))
    result = dict(status='PASS_ACTUAL_AFFECTED_AND_ADJACENT_PAGE_RENDER', total_pages=len(document), pages=pages,
                  review_scope='Actual rasterization, text location and font records; no claim of full-thesis manual review')
    document.close()
    return result


def publish(thesis, directory):
    check_preparation(thesis)
    ready = read(directory/'publication_ready.json')
    require(ready['status'] == 'PASS' and ready['models'] == 90 and ready['fixed_evaluations'] == 210,
            'Completed formal report required')
    bundle, lookup = report.audit_inputs(ROOT)
    require(ready['report_bundle_sha256'] == sha(ROOT/'report_bundle.json'), 'Formal report belongs to another cohort')
    for section in ('paper_assets', 'source_csv'):
        for item in ready[section].values():
            require(sha(item['path']) == item['sha256'], 'Audited report asset changed')
    data = statistics(directory)
    input_data = report.assemble(ROOT, bundle, lookup)
    destination = ROOT/'publication'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    destination.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='PREPARING', root=str(ROOT), report_ready_sha256=sha(directory/'publication_ready.json'),
                   new_environment_steps=0, model_files_modified=False, other_chapters_rewritten=False)
    try:
        with locked(thesis/'build/.chapter05_formal_publication.lock'):
            with locked(thesis/'build/.whole_thesis_build.lock'):
                before = snapshot(thesis)
                current = (thesis/SECTION).read_text()
                old_summary = (thesis/SUMMARY).read_text()
                with (thesis/'chapters/FIGURE_INDEX.csv').open(encoding='utf-8-sig', newline='') as stream:
                    reader = csv.DictReader(stream)
                    fields, index = reader.fieldnames, list(reader)
                planned, directories, replaced = {}, {}, []
                assets = Path(ready['paper_assets']['learning.pdf']['path']).parent
                for number, stem, title, label in FIGURES:
                    row = next(row for row in index if row['图号'] == number)
                    relative = Path(row['图目录'])
                    require(relative.is_relative_to(CHAPTER/'figures'), 'Figure escaped Chapter 5')
                    directories[number] = relative
                    physical = destination/'assets'/stem
                    physical.mkdir(parents=True)
                    for item in assets.iterdir():
                        if item.suffix in ('.csv', '.json') or item.name == 'plot_assets.py':
                            shutil.copy2(item, physical/item.name)
                    for extension in ('pdf', 'svg', 'png'):
                        shutil.copy2(assets/(stem+'.'+extension), physical/('figure.'+extension))
                    wrapper = ('from pathlib import Path\nimport subprocess,sys\np=Path(__file__).resolve().parent\n'
                               f"subprocess.run([sys.executable,str(p/'plot_assets.py'),'--replot-csv',str(p),'--figure','{stem}','--output',str(p/'regenerated')],check=True)\n"
                               f"for extension in ('pdf','svg','png'):(p/'regenerated'/('{stem}.'+extension)).replace(p/('figure.'+extension))\n")
                    (physical/'plot.py').write_text(wrapper)
                    (physical/'README.rst').write_text(title+'\n'+'='*len(title)+'\n\n'
                        '正式五种子0—4、每模型100608学习交互步、198个原始DEV点。\n'
                        '固定终点每模型256个共同独立情景，误差为训练种子样本标准差。\n'
                        'python plot.py 使用相邻封印CSV重绘；不运行模型或环境。\n')
                    write(physical/'source.json', dict(schema='thesis-figure-local-assets-v2', number=number,
                          caption=title, label=label, experiment_root=str(ROOT), independent_audit=str(ROOT/'final_evaluation_audit.json'),
                          font_policy=dict(chinese='SimSun', latin='Times New Roman', group_titles='SimHei', main_bp=10.5, secondary_bp=9.5, natural_width_cm=15.6),
                          files={p.name: sha(p) for p in physical.iterdir() if p.is_file()}))
                    row.update({'图题': title, '正文PDF': 'figure.pdf', '绘图程序': 'plot.py', '可编辑图源': '',
                                '绘图数据': 'learning_raw.csv;fixed_endpoints_by_seed.csv;threshold_recounts_by_seed.csv;plot_contract.json'})
                    original = thesis/relative
                    backup = destination/'before'/relative
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    if original.exists() or original.is_symlink():
                        shutil.move(str(original), str(backup))
                    original.symlink_to(physical, target_is_directory=True)
                    replaced.append(dict(path=str(relative), backup=str(backup), new_target=str(physical)))
                planned[SECTION] = make_section(current, data, directories, input_data).encode()
                planned[SUMMARY] = make_summary(old_summary, data).encode()
                for stem in ('settings', 'endpoints', 'ablation_endpoints', 'costs'):
                    planned[CHAPTER/'tables'/('formal_'+stem+'.tex')] = (assets/(stem+'.tex')).read_bytes()
                string = io.StringIO()
                writer = csv.DictWriter(string, fieldnames=fields, lineterminator='\n')
                writer.writeheader()
                writer.writerows(index)
                planned[Path('chapters/FIGURE_INDEX.csv')] = string.getvalue().encode('utf-8-sig')
                for relative, payload in planned.items():
                    original = thesis/relative
                    if original.exists():
                        backup = destination/'before'/relative
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(original, backup)
                    atomic(original, payload)
                if (thesis/'build/main.pdf').exists():
                    shutil.copy2(thesis/'build/main.pdf', destination/'before/main.pdf')
            with stable_whole_build(thesis, planned, destination) as build_rounds:
                for relative, payload in planned.items():
                    require((thesis/relative).read_bytes() == payload, 'Concurrent writer changed controlled C5 publication file')
                after = snapshot(thesis)
                own = {str(path) for path in planned}
                others = {key: dict(before=before.get(key), after=after.get(key)) for key in set(before)|set(after)
                          if before.get(key) != after.get(key) and key not in own and not any(key.startswith(str(row['path'])+'/') for row in replaced)}
                require(all((thesis/key).exists() for key in others), 'An unrelated live source disappeared during publication')
                render = render_thesis(thesis/'build/main.pdf', destination/'rendered_pages')
                pdf = destination/'main.pdf'
                shutil.copy2(thesis/'build/main.pdf', pdf)
                write(destination/'actual_pdf_page_audit.json', render)
                receipt.update(status='PUBLISHED', thesis_pdf=str(thesis/'build/main.pdf'), immutable_pdf=str(pdf),
                               pdf_sha256=sha(pdf), pdf_pages=render['total_pages'], report_pdf=ready['independent_pdf'],
                               controlled_source_sha256={str(path): sha(thesis/path) for path in planned},
                               preserved_original_figures=replaced, concurrent_changes_preserved=others,
                               theory_sections_61_62_not_written=True, other_chapters_not_written=True,
                               build_command=['bash', 'scripts/build.sh'], build_rounds=build_rounds,
                               actual_pdf_page_audit_sha256=sha(destination/'actual_pdf_page_audit.json'),
                               scientific_results_relabelled=False, manual_or_agent_visual_review='pending; actual affected pages rendered',
                               publication_script_sha256=sha(__file__), instructions=check_preparation(thesis))
                target = ROOT/'FINAL_THESIS_PUBLICATION.json'
                write(destination/'FINAL_THESIS_PUBLICATION.json', receipt)
                write(target, receipt)
                docs = thesis/'docs/chapter5_formal_random_five_seed_20261107'
                docs.mkdir(parents=True, exist_ok=True)
                write(docs/'FINAL_THESIS_PUBLICATION.json', receipt)
                latest = thesis/'docs/LATEST_THESIS_PUBLICATION.json'
                if latest.exists():
                    shutil.copy2(latest, destination/'previous_latest_publication.json')
                write(latest, dict(status='PUBLISHED', scope='Chapter5 formal five-seed experiments and summary',
                                  receipt=str(target), pdf=str(thesis/'build/main.pdf'), pdf_sha256=sha(pdf), pages=render['total_pages']))
        return receipt
    except BaseException:
        receipt.update(status='FAILED', traceback=traceback.format_exc(), originals_preserved=True,
                       automatic_training_retry=False, other_live_sources_not_rolled_back=True)
        write(destination/'failure_receipt.json', receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thesis', type=Path, default=Path('/root/gamewar/latex_thesis_resource_allocation'))
    parser.add_argument('--report', type=Path, default=ROOT/'reports/final')
    parser.add_argument('--validate-preparation-only', action='store_true')
    args = parser.parse_args()
    if args.validate_preparation_only:
        print(json.dumps(check_preparation(args.thesis.resolve()), ensure_ascii=False))
        return
    target = ROOT/'FINAL_THESIS_PUBLICATION.json'
    if target.exists():
        previous = read(target)
        require(previous['status'] == 'PUBLISHED', 'Existing publication receipt is not successful')
        print(json.dumps(dict(status='ALREADY_PUBLISHED_IMMUTABLE_RECEIPT', receipt=str(target))))
        return
    print(json.dumps(publish(args.thesis.resolve(), args.report.resolve()), ensure_ascii=False))


if __name__ == '__main__':
    main()
