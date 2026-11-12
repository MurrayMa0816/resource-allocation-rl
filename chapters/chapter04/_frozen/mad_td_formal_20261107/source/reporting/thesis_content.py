"""Result-conditioned C4 text. Never supplies substitute experimental values."""
from __future__ import annotations
import math, statistics
from collections import defaultdict

MAIN=('f1','props','mad_td','redo','mrq','bro')
ABLATIONS=('f1','no_history','no_offpolicy_correction','fixed_actor_lr')
NAMES={'f1':'TRoC','props':'PROPS','mad_td':'MAD-TD','redo':'SAC+ReDo','mrq':'MR.Q','bro':'BRO(Fast)',
 'no_history':'不使用旧轨迹','no_offpolicy_correction':'不作离策略训练校正','fixed_actor_lr':'固定Actor学习率','A_inherited':'第三章联合策略'}
DOMAINS={'radio':'通信','v5':'应急'}
FIGURES=[
 ('4.4','TRoC与外部方法的开发学习曲线','fig:ch4-formal-main','learning',MAIN,'learning_curves.csv'),
 ('4.5','TRoC与外部方法的受限交互成本及确认次数','fig:ch4-completed-development','cost',MAIN,None),
 ('4.6','TRoC与三种消融变体的开发学习曲线','fig:ch4-matched-development','ablation',ABLATIONS,'learning_curves.csv'),
 ('4.7','不同交互预算内的平均开发得分','fig:ch4-budget-quality','prefix_auc',MAIN,'budget_prefixes.csv'),
 ('4.8','TRoC近期旧决策的归一化有效样本量与策略散度','fig:ch4-history-diagnostics','history',('f1',),'f1_internal.csv'),
 ('4.9','TRoC与固定学习率变体的批后变化量及Actor学习率','fig:ch4-tv-learning-rate','control',('f1','fixed_actor_lr'),'f1_internal.csv')]

def figure_dir(number,caption):
    chapter,index=map(int,number.split('.'))
    return f'figure{chapter:02d}-{index:02d}_'+caption
def figure(number):
    item=next(x for x in FIGURES if x[0]==number);_,caption,label,*_=item
    return '\n'.join([r'\begin{figure}[H]',r' \centering',
       r' \includegraphics{chapters/chapter04_trajectory_reuse/figures/'+figure_dir(number,caption)+r'/figure.pdf}',
       r' \caption{'+caption+'}',r' \label{'+label+'}',r'\end{figure}'])
def f(x,places=2):
    x=float(x);assert math.isfinite(x);return f'{x:.{places}f}'
def pm(metric,scale=1):
    return '--' if not metric or not metric.get('n') else r'\('+f(metric['mean']*scale)+r'\pm'+f(metric['sample_sd']*scale)+r'\)'
def difference(x,unit='分'):
    if abs(x)<.005:return '差值不足0.01'+unit
    return ('高出' if x>0 else '低于')+f(abs(x))+unit
def unit_for(domain):return '分'if domain=='radio'else'个百分点'

def tabular(cols,head,rows,caption,label):
    return '\n'.join([r'\begin{table}[H]',r' \centering',r' \caption{'+caption+'}',r' \label{'+label+'}',
       r' \begingroup\small',r' \setlength{\tabcolsep}{3.5pt}',r' \begin{tabular}{'+cols+'}',r' \toprule',
       ' & '.join(head)+r' \\',r' \midrule',*[' & '.join(str(x)for x in row)+r' \\'for row in rows],
       r' \bottomrule',r' \end{tabular}',r' \endgroup',r'\end{table}'])

def main_table(s):
    rows=[]
    for d in DOMAINS:
        for m in MAIN:
            v=s['methods'][d][m];q=v['nq']['q95']
            rows.append([DOMAINS[d],NAMES[m],str(q['reached_n'])+'/5',pm({'n':5,'mean':q['restricted_mean_steps'],'sample_sd':q['sample_sd']},.001),pm(v['auc'],100),pm(v['endpoint'],100)])
        v=s['methods'][d]['A_inherited'];rows.append([DOMAINS[d],NAMES['A_inherited'],'--','--','--',pm(v['endpoint'],100)])
    return tabular('llrrrr',['环境','方法','确认次数','受限成本/千步','平均开发得分','终点测试得分'],rows,'主比较的交互成本与配置成效','tab:ch4-formal-results')

def main_text(s):
    parts=[]
    for d,cn in DOMAINS.items():
        g=s['methods'][d];a=g['f1'];q=a['nq']['q95']
        ext=min(MAIN[1:],key=lambda m:g[m]['nq']['q95']['restricted_mean_steps'])
        b=g[ext]['nq']['q95'];delta=(b['restricted_mean_steps']-q['restricted_mean_steps'])/1000
        cost=('少用'if delta>=0 else '多用')+f(abs(delta))+'千步'
        best=max(MAIN[1:],key=lambda m:g[m]['endpoint']['mean'])
        parts.append(cn+'环境中，TRoC有'+str(q['reached_n'])+'次独立运行确认达到主要质量要求，受限交互成本均值为'+f(q['restricted_mean_steps']/1000)+'千步。外部方法中，'+NAMES[ext]+'的受限成本最低，其确认次数为'+str(b['reached_n'])+'次。TRoC与该方法相比'+cost+'。终点测试得分最高的外部方法为'+NAMES[best]+'，TRoC的终点均值相对该方法'+difference(100*(a['endpoint']['mean']-g[best]['endpoint']['mean']),unit_for(d))+'。')
    parts.append('受限成本将未确认运行截断于训练上限，其均值同时受到到达速度和未确认次数影响。因此，成本数值应与确认次数共同解释，不能将删失上限读作实际达到质量要求的时刻。')
    return '\n\n'.join(parts)

def bridge_text(s):
    return '为衔接前章，表\\ref{tab:ch4-formal-results}仅列入第三章双随机联合策略在同一组独立终点情景上的测试结果。'+''.join(DOMAINS[d]+'环境中，TRoC相对该参照的终点均值'+difference(100*(s['methods'][d]['f1']['endpoint']['mean']-s['methods'][d]['A_inherited']['endpoint']['mean']),unit_for(d))+'。'for d in DOMAINS)+'该参照没有与本章一致的32情景开发曲线，其每512步评价8个情景的原始开发记录不并入本章每768步评价32个情景的交互成本或曲线面积。'

def ablation_table(s):
    rows=[]
    for d,cn in DOMAINS.items():
        for m in ABLATIONS:
            g=s['methods'][d][m];q=g['nq']['q95']
            rows.append([cn,NAMES[m],str(q['reached_n'])+'/5',f(q['restricted_mean_steps']/1000),pm(g['auc'],100),pm(g['endpoint'],100)])
    return tabular('llrrrr',['环境','变体','确认次数','受限成本/千步','平均开发得分','终点测试得分'],rows,'完整方法与三种消融变体','tab:ch4-reuse-ablations')

def ablation_text(s):
    out=[]
    for m in ABLATIONS[1:]:
        fragments=[]
        for d,cn in DOMAINS.items():
            a=s['methods'][d]['f1'];b=s['methods'][d][m]
            dc=(b['nq']['q95']['restricted_mean_steps']-a['nq']['q95']['restricted_mean_steps'])/1000
            da=100*(a['auc']['mean']-b['auc']['mean'])
            fragments.append(cn+'环境中，完整方法的受限成本比该变体'+('减少'if dc>=0 else '增加')+f(abs(dc))+'千步，平均开发得分'+difference(da,unit_for(d)))
        out.append(NAMES[m]+'的比较结果为'+ '，'.join(fragments)+'。')
    out.append('各项消融只用于检验对应设计在当前任务和训练配置下的作用。某项差值较小或方向随环境改变时，不将其解释为该组件普遍必要，也不把旧记录重复读取本身当作学习收益。')
    return '\n\n'.join(out)

def threshold_table(s):
    rows=[]
    for d,cn in DOMAINS.items():
        for m in MAIN:
            g=s['methods'][d][m];rows.append([cn,NAMES[m],*[str(g['nq'][q]['reached_n'])+'/5'for q in ('q90','q95','q100')]])
    return tabular('llrrr',['环境','方法','较低目标','主要目标','较高目标'],rows,'不同质量要求下的确认次数','tab:ch4-threshold-sensitivity')

def cost_table(s):
    rows=[]
    for d,cn in DOMAINS.items():
        for m in ABLATIONS:
            c=s['methods'][d][m].get('costs',{})
            rows.append([cn,NAMES[m],pm(c.get('actor_updates')),pm(c.get('critic_updates')),pm(c.get('collection_seconds')),pm(c.get('update_seconds'))])
    return tabular('llrrrr',['环境','变体','Actor更新次数','Critic更新次数','采集时间/s','更新时间/s'],rows,'轨迹复用的实际更新次数与计算开销','tab:ch4-update-costs')

def internal_text(rows):
    by=defaultdict(list)
    for r in rows:
        if r.get('method')!='f1':continue
        for k in ('history_only_ess_fraction','old_policy_kl','tv'):
            try:v=float(r.get(k,''))
            except (TypeError,ValueError):continue
            if math.isfinite(v):by[(r['domain'],int(r['seed']),k)].append(v)
    out=[]
    for d,cn in DOMAINS.items():
        vals={k:[statistics.fmean(by[(d,i,k)])for i in range(5)if by[(d,i,k)]]for k in ('history_only_ess_fraction','old_policy_kl','tv')}
        if all(len(v)==5 for v in vals.values()):
            out.append(cn+'环境中，先在每次独立运行内对实际更新记录取均值，再在五次运行间平均，旧决策归一化有效样本量为'+f(statistics.fmean(vals['history_only_ess_fraction']))+'，采集策略与更新前策略的平均KL散度为'+f(statistics.fmean(vals['old_policy_kl']),4)+'，批后样本策略变化量为'+f(statistics.fmean(vals['tv']),4)+'。')
        else:out.append(cn+'环境的内部指标仅在实际保存且有定义的更新上汇总，缺失项不以零填补。')
    return '\n\n'.join(out)

def threshold_description(s):
    out=[]
    for d,cn in DOMAINS.items():
        q=s['thresholds'][d]
        out.append(cn+'任务的较低目标与主要目标及较高目标依次为'+f(100*q['0.9'])+'和'+f(100*q['0.95'])+'及'+f(100*q['1.0'])+('分'if d=='radio'else'\\%'))
    return '质量要求变化时，仍采用训练前登记的三档绝对目标。'+'。'.join(out)+'。对应确认次数见表\\ref{tab:ch4-threshold-sensitivity}，未达到要求的运行保留删失身份。'

def prefix_text(rows):
    by=defaultdict(list)
    for r in rows or []:
        by[(r['domain'],r['method'],int(r['training_budget']))].append(float(r['normalized_auc']))
    out=[]
    for d,cn in DOMAINS.items():
        if not by[(d,'f1',6144)]:continue
        parts=[]
        for budget,name in [(6144,'6144步'),(100608,'完整预算')]:
            a=statistics.fmean(by[(d,'f1',budget)])
            best=max(MAIN[1:],key=lambda m:statistics.fmean(by[(d,m,budget)]))
            b=statistics.fmean(by[(d,best,budget)])
            parts.append(name+'内，TRoC的平均开发得分为'+f(100*a)+'，相对同预算得分最高的外部方法'+NAMES[best]+difference(100*(a-b),unit_for(d)))
        out.append(cn+'任务中，'+'。'.join(parts)+'。')
    return '\n\n'.join(out)

def make_section(s,internal,cites,prefixes=None):
    qr=100*s['thresholds']['radio']['0.95'];qv=100*s['thresholds']['v5']['0.95']
    lead=r'''% !TeX root = ../../main.tex
\Needspace{12\baselineskip}
\section{实验结果与分析}
\label{sec:ch5-experiments}
实验以达到给定配置质量所需的新训练交互数评价TRoC的样本效率，并结合固定预算内的配置成效分析学习结果。轨迹来源与更新规则的消融用于区分近期记录复用及配套校正的作用，质量要求与训练预算的变化进一步给出收益范围。旧记录的概率权重与实际更新开销用于解释复用的状态和代价。

\subsection{实验设置}
\label{subsec:ch5-experiments-1}
\subsubsection{实验环境}
实验采用图像下行传输与应急资源配置两个任务。两者均在期初取得有限总预算，依据逐期公开状态决定当期投入总量与对象间分配，只在任务终点返回配置成效奖励。本章沿用第三章的期末全额分配条件，未投入资源在期间结转，末期分配全部余额。各方法采用相同的物理转移和资源执行边界。

图像下行传输任务中，通信站在32轮内向100个终端传送图像，源图像取自EuroSAT RGB数据集，来源见第\ref{subsec:ch3-experiments-1}节。初始可用发射能量为480微焦耳，期间没有补充。通信站结合剩余能量，依据信道条件和接收进度，联合决定本轮总能量与各终端功率，单终端功率不超过2毫瓦。已接收数据保留用于重建，任务结束时以各终端重建图像的平均质量评价配置。

应急资源配置任务中，配置中心在24个决策期内为120个区域提供服务，初始等效柴油服务额度为20160升，期间没有补充。中心依据各区域的未完成需求与已出现总需求，结合期数和剩余额度决定本期投入及区域份额。未完成需求继续参与后续配置，未来新增需求和区域服务效率不向策略公开。任务结束时以各区域的平均需求完成比例评价配置。

\subsubsection{对比方法}
实验采用五种对比方法，比较不同采集与记录利用方式下的新交互需求。'''
    lead+=(
      'PROPS调整行为策略，增加相对目标策略采样不足动作的采集概率，并用裁剪目标与KL约束控制行为偏离\\cite{'+cites['props']+'}。'
      'SAC+ReDo在SAC更新中监测神经元活动，重新初始化低活动神经元的输入权重，并将输出权重置零，以保留后续学习所需的表示能力\\cite{'+cites['redo']+'}。'
      'MR.Q借助奖励与转移预测训练状态表示，再用该表示学习策略和价值函数，决策时不进行模型规划\\cite{'+cites['mrq']+'}。'
      'BRO通过正则化扩展价值网络，并结合乐观探索提高连续控制中的数据利用，本实验采用其快速配置BRO(Fast)\\cite{'+cites['bro']+'}。'
      'MAD-TD在高更新数据比训练中，将少量学习模型生成的当前策略转移与真实历史转移混合，用于减轻价值函数对当前策略动作的外推误差\\cite{'+cites['mad_td']+'}。'
      '各方法保留自身的采集与更新规则，并通过同一公开状态和可行配置映射执行资源投入。')
    lead+=r'''

\subsubsection{训练过程与算法参数设置}
各方法在每个任务中采用种子0至4进行五次独立训练，每次采集100608步新训练交互。通信与应急任务分别对应3144和4192个完整回合。所有预热采样均计入预算，重复读取记录与模型生成数据不计作新环境交互。每次训练只保留固定预算终点的模型，开发成绩不用于选择中间模型。

TRoC的对象编码与组合系数网络采用64和32单元隐藏层，通信与应急的组合系数维数分别为18和12。总量网络采用32单元隐藏层，两个输出头随机初始化。同一种子下，完整方法与三个消融变体采用相同初始参数。PROPS使用相同来源的策略均值结构，其余外部方法保留各自网络结构与随机初始化。

TRoC前32个完整回合按每批8条新轨迹预热，此后每次采集4条新完整轨迹，与最近保留的4条旧完整轨迹组成学习批次。策略与价值网络分别采用Adam优化器，初始学习率为\(3\times10^{-4}\)和\(10^{-3}\)。主要更新参数见表\ref{tab:ch4-reuse-training-parameters}。
'''
    lead+='\n'+tabular('ll',['参数','取值'],[
      ['新旧完整轨迹数','预热时8条新轨迹，此后4条新轨迹与4条旧轨迹'],
      ['每批最多更新轮数与小批量','4轮，64个时点'],
      ['裁剪宽度',r'预热0.2，此后\(0.2/1.5\)'],
      ['折扣与迹参数','均为1'],
      ['组合系数标准差',r'\(1/\sqrt K\)'],
      ['总量变量标准差',r'初始0.8，限制于\([0.1,1.5]\)'],
      ['V-trace两类截断上界','均为1'],
      ['梯度范数上限与KL早停阈值','0.5与0.12'],
      ['Actor学习率范围与调整倍数',r'\([10^{-5},10^{-3}]\)，1.03']],
      'TRoC的主要更新参数','tab:ch4-reuse-training-parameters')
    lead+=r'''
外部方法的主要设置见表\ref{tab:ch4-external-parameters}。PROPS的目标策略与行为策略初始学习率分别为\(5\times10^{-5}\)和\(10^{-4}\)。目标学习率取前期比较后固定的半学习率设置，前期仅在应急任务比较了\(5\times10^{-5}\)与\(10^{-4}\)及\(2\times10^{-4}\)三种目标初始学习率，行为初始学习率固定为\(10^{-4}\)。每组采用种子10460和10461各训练24576步，每768步在32个开发情景上随机执行评价。该试验的预登记筛选保留\(10^{-4}\)，本节所报半学习率不是其优选配置，前期随机评价结果不并入本节确定性评价统计。因此，对PROPS的比较结论限定于所报告的固定配置。SAC+ReDo采用论文回收规则的PyTorch SAC适配，MR.Q与BRO(Fast)及MAD-TD采用作者原生学习实现。
'''
    lead+='\n'+tabular('lrp{9.0cm}',['方法','折扣因子','主要更新设置'],[
      ['PROPS','1','目标与行为策略最多更新10轮和16轮，缓冲区容量8条完整轨迹'],
      ['SAC+ReDo','0.99','预热10000步，小批量256，每步8次回放更新，回收周期200000次优化更新'],
      ['MR.Q','0.99','小批量256，表示预测跨度5，价值跨度3，目标网络周期250'],
      ['BRO(Fast)','0.99','2500步后更新，小批量128，每步2次回放更新'],
      ['MAD-TD','1','预热5000步，小批量512，每步8次策略与价值更新，批中486条真实转移与26条模型生成转移']],
      '对比方法的主要训练设置','tab:ch4-external-parameters')
    lead+=r'''
MAD-TD采用512维潜在表示与151个价值分箱，策略与价值及模型初始学习率均为\(3\times10^{-4}\)，编码器初始学习率为\(10^{-4}\)。价值范围依据本任务公开奖励界固定为\([-1,1]\)。真实终点转移参与价值和奖励模型训练，只有终点的后继价值置零。预热结束后，模型与编码器每步新交互更新一次，策略与价值网络保持原生的高更新数据比。目标网络按作者代码采用保留系数\(\tau=0.99\)。新真实转移按0.05概率分入不参与参数更新的独立诊断缓冲区，发生的交互仍计入训练预算，该分流与开发和测试情景分开。训练终点不保存优化器或磁盘回放数据，各方法均以相同真实交互预算比较。

\subsubsection{评价指标与统计方法}
通信任务以各终端最终重建图像的平均SSIM评价配置，并保留负值。应急任务先计算各区域的累计需求完成比例，再作等权平均，无需求区域视为已完成。训练前及每768步新训练交互，在32个固定开发情景上进行确定性评价，每次训练共132个开发点。开发评价不参与网络训练，也不改变训练随机数状态。图中的SSIM乘100显示，应急完成比例以百分比显示，阴影为五次独立训练之间的样本标准差。

固定终点模型在256个共同测试情景上确定性执行。测试面板与训练和开发情景分别生成，已有模型沿用同一已登记面板的评价，不将该面板称为全新未见测试。高斯联合策略取控制分布均值后执行相同配置映射，其余方法采用原生部署动作。先对每个模型的情景结果取均值，再以五个训练种子为统计单位计算均值与样本标准差。开发与测试的实际环境交互单独计量。
'''
    lead+='\n样本效率按预先登记的绝对质量要求评价，主要要求在通信中为'+f(qr)+'分，在应急中为'+f(qv)+'\\%。目标在本轮训练前固定，未按本轮结果或新版第三章参照重新计算。对第\\(s\\)次训练，记第\\(k\\)个开发点的新交互数为\\(n_k=768k\\)，开发得分为\\(J_s(n_k)\\)，其中\\(k=0,\\ldots,131\\)。记主要质量要求为\\(q\\)，首次连续两个开发点达到要求时，以第二个开发点确认达标，即\n'
    lead+=r'''\begin{equation}
 N_{q,s}=\min\{n_k\mid 1\leq k\leq131,\ J_s(n_{k-1})\geq q,\ J_s(n_k)\geq q\}.
 \label{eq:ch4-attainment-cost}
\end{equation}
未确认达标的运行记为右删失，并以\(N_{q,s}=+\infty\)参与受限成本计算。训练预算记为\(N_{\max}=100608\)，受限交互成本为
\begin{equation}
 C_{q,s}(N_{\max})=\min\{N_{q,s},N_{\max}\}.
 \label{eq:ch4-restricted-cost}
\end{equation}
受限成本越低且确认次数越多，表示达到给定要求的新交互需求越低。预算封顶值表示尚未确认，不能读作实际达标时刻。首次连续确认也不表示后续每次评价均达标。

固定预算内的平均开发成效采用原始开发点的梯形积分。对止于第\(K\)个开发点的预算\(N_{\max}=n_K\)，定义
\begin{equation}
 \mathcal A_s(N_{\max})=\frac{1}{N_{\max}}\sum_{k=1}^{K}
 \frac{J_s(n_{k-1})+J_s(n_k)}{2}(n_k-n_{k-1}).
 \label{eq:ch4-normalized-learning-area}
\end{equation}
完整预算下\(K=131\)，该指标与开发得分同单位，越大表示给定学习窗口内的平均配置成效越高。达标成本回答新交互需求，平均开发成效与终点测试成效分别描述学习期间和固定终点的配置结果。

\subsection{样本效率与轨迹复用分析}
\label{subsec:ch5-experiments-2}
\subsubsection{新交互需求与配置成效比较}
同等质量下的新交互需求由固定开发情景的学习曲线及主要质量要求共同确定，如图\ref{fig:ch4-formal-main}所示。全部方法采用相同预算与评价间隔，曲线保留未平滑的开发点，不按最好成绩累计或强制单调化。
'''
    out=[lead,figure('4.4'),main_text(s),
      r'受限交互成本与确认次数同时用于区分较早确认和预算内未确认的情况，对比结果见图\ref{fig:ch4-completed-development}。平均开发成效和独立终点测试进一步给出相同预算下的配置质量，结果见表\ref{tab:ch4-formal-results}。',
      figure('4.5'),main_table(s),bridge_text(s),
      r'''\subsubsection{轨迹来源与更新规则的消融}
三种消融分别检验近期旧轨迹及离策略校正组合和学习率控制的作用。不使用旧轨迹的变体每批仍采集4条新完整轨迹，将当前4条重复组成8条批次，保持新交互节奏与批次条目数。该变体与第三章每批采集8条新轨迹的训练方式不同。不作离策略训练校正的变体保留新旧轨迹，以完整回报替代V-trace目标，并以冻结更新前策略密度作为策略比值的分母，裁剪中心仍为1，用于检验价值目标与策略校正的组合。固定Actor学习率变体采用\(3\times10^{-4}\)，保留轨迹复用及校正，其余裁剪与停止规则保持相同。

完整方法与三个变体各有五次100608步独立训练，同一种子的初始参数一致。最多优化轮数与批量相同，实际优化次数由停止规则决定并单独统计。开发轨迹与主要要求下的比较分别见图\ref{fig:ch4-matched-development}和表\ref{tab:ch4-reuse-ablations}。''',
      figure('4.6'),ablation_table(s),ablation_text(s),
      r'''\subsection{训练预算与策略更新分析}
\label{subsec:ch5-experiments-3}
\subsubsection{训练预算与质量要求变化}
不同训练预算下的平均开发成效用于判断收益出现的阶段，比较见图\ref{fig:ch4-budget-quality}。五个预算分别为6144步和12288步及24576步，再扩展至49152步与100608步。各预算均取同一次完整训练的前缀，不追加模型训练或按窗口重新调参。''',
      figure('4.7'),prefix_text(prefixes),
      threshold_description(s),
      threshold_table(s),
      r'''\subsubsection{旧记录概率权重与更新控制}
旧决策的权重集中程度与采集策略偏离用于描述近期记录在更新时的状态，结果见图\ref{fig:ch4-history-diagnostics}。对一次更新中有定义的\(M_{\mathrm{old}}>0\)个可更新旧决策，记第\(j\)个更新前策略密度与采集策略密度之比为\(w_j\)。权重按原始对数密度差计算，归一化有效样本量定义为
\begin{equation}
 e_{\mathrm{old}}=
 \frac{\left(\sum_{j=1}^{M_{\mathrm{old}}}w_j\right)^2}
 {M_{\mathrm{old}}\sum_{j=1}^{M_{\mathrm{old}}}w_j^2}.
 \label{eq:ch4-history-effective-weight}
\end{equation}
该比例衡量权重是否集中于少数记录，不等同于策略梯度方差。策略差异取采集策略到更新前策略的联合高斯KL散度，在同一组可更新旧决策上平均。预热阶段没有旧记录，不以零补入这两项指标。''',
      figure('4.8'),internal_text(internal),
      r'批后样本策略变化量与下一批Actor学习率的对应关系见图\ref{fig:ch4-tv-learning-rate}，固定学习率变体提供保持其余轨迹规则的对照。内部指标先在每次训练中按3072步新交互划分不重叠区间，再跨五次训练汇总。更新控制的实际收益仍由消融中的交互成本和配置成效判断，不能仅凭学习率随变化量调整认定优势。',
      figure('4.9'),
      r'''\subsubsection{实际优化次数与计算开销}
完整方法与三个消融的实际Actor和Critic优化次数，以及采集与参数更新的执行段耗时见表\ref{tab:ch4-update-costs}。提前停止后未发生的优化步不计入次数，开发与终点评价不计入这两类执行段耗时。运行使用AMD EPYC 7542与AMD EPYC 7B12处理器及NVIDIA RTX 3090显卡，各运行的实际设备和并发负载单独记录。阶段耗时是本次并行执行的观测值，包含相应执行段的等待，不作为独占设备下的算法速度排序。''',
      cost_table(s),
      '旧轨迹复用增加了同一批新交互能够参与的学习数据，也带来记录重算与参数优化开销。样本效率是否提高由达到相同质量的新交互需求检验，计算开销单独报告，不将交互节省等同于墙钟时间缩短。']
    return '\n\n'.join(x for x in out if x)+'\n'

def make_summary(s):
    text=r'''% !TeX root = ../../main.tex
\section{本章小结}
\label{sec:ch5-summary}
本章以达到给定配置质量所需的新交互数描述多轮资源配置策略学习的样本效率。TRoC将新完整轨迹与最近保留的完整轨迹组成学习批次，依据真实采集概率构造有限回合价值目标与优势，并结合裁剪目标和策略变化量控制更新。配置执行满足期末全额分配条件，反馈仍来自真实任务终点。

'''
    text+=''.join(DOMAINS[d]+'任务中，五次训练有'+str(s['methods'][d]['f1']['nq']['q95']['reached_n'])+'次确认达到预设主要质量要求，独立终点测试得分为'+pm(s['methods'][d]['f1']['endpoint'],100)+'。'for d in DOMAINS)
    gains=[]
    for d in DOMAINS:
        g=s['methods'][d];a=g['f1']['nq']['q95'];best=min(MAIN[1:],key=lambda m:g[m]['nq']['q95']['restricted_mean_steps']);b=g[best]['nq']['q95']
        gains.append(a['restricted_mean_steps']<b['restricted_mean_steps']and a['reached_n']>=b['reached_n'])
    if all(gains):text+='在主要质量要求下，TRoC在两个任务的受限交互成本均低于成本最低的外部对照，且确认次数不低于该对照。'
    elif any(gains):text+='主要质量要求下的交互成本收益随任务和对照方法改变，两个任务没有形成相对全部外部对照一致的优势。'
    else:text+='本次结果未形成相对全部外部对照一致的交互成本优势，收益应按各任务的确认次数与成本分别判断。'
    text+='平均开发成效与终点测试分别给出固定学习窗口和终点的配置结果，消融比较进一步检验近期旧轨迹及配套校正和更新控制的作用。旧记录权重仅描述分布失配，未据此声称策略梯度方差下降。结论限定于所报任务适配与训练配置，样本效率改进不直接代表计算效率改进。\n'
    return text
