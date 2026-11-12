"""Result-conditioned C4 text. Never supplies substitute experimental values."""
from __future__ import annotations
import math, statistics
from collections import defaultdict

MAIN=('f1','props','pgr','redo','mrq','bro')
ABLATIONS=('f1','no_history','no_offpolicy_correction','fixed_actor_lr')
NAMES={'f1':'TRoC','props':'PROPS','pgr':'PGR','redo':'SAC+ReDo','mrq':'MR.Q','bro':'BRO(Fast)',
 'no_history':'不使用旧轨迹','no_offpolicy_correction':'不作离策略训练校正','fixed_actor_lr':'固定Actor学习率','A_inherited':'第三章联合策略'}
DOMAINS={'radio':'通信','v5':'应急'}
FIGURES=[
 ('4.4','TRoC与外部方法的开发学习曲线','fig:ch4-formal-main','learning',MAIN,'learning_curves.csv'),
 ('4.5','TRoC与外部方法的受限交互成本及确认次数','fig:ch4-completed-development','cost',MAIN,None),
 ('4.6','TRoC与三种消融变体的开发学习曲线','fig:ch4-matched-development','ablation',ABLATIONS,'learning_curves.csv'),
 ('4.7','不同交互预算内的平均开发得分','fig:ch4-budget-quality','prefix_auc',MAIN,'budget_prefixes.csv'),
 ('4.8','TRoC近期旧决策的权重集中程度与策略散度','fig:ch4-history-diagnostics','history',('f1',),'f1_internal.csv'),
 ('4.9','TRoC与固定学习率变体的更新变化和Actor学习率','fig:ch4-tv-learning-rate','control',('f1','fixed_actor_lr'),'f1_internal.csv')]

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
        parts.append(cn+'环境中，TRoC有'+str(q['reached_n'])+'次独立运行确认达到主要质量要求，受限交互成本均值为'+f(q['restricted_mean_steps']/1000)+'千步。外部方法中，'+NAMES[ext]+'的受限成本最低，其确认次数为'+str(b['reached_n'])+'次。TRoC与该方法相比'+cost+'。终点测试得分最高的外部方法为'+NAMES[best]+'，TRoC的终点均值相对该方法'+difference(100*(a['endpoint']['mean']-g[best]['endpoint']['mean']))+'。')
    parts.append('受限成本将未确认运行截断于训练上限，其均值同时受到到达速度和未确认次数影响。因此，成本数值应与确认次数共同解释，不能将删失上限读作实际达到质量要求的时刻。')
    return '\n\n'.join(parts)

def bridge_text(s):
    return '为衔接前章，表\\ref{tab:ch4-formal-results}仅列入第三章联合策略在同一组独立终点情景上的测试结果。'+''.join(DOMAINS[d]+'环境中，TRoC相对该参照的终点均值'+difference(100*(s['methods'][d]['f1']['endpoint']['mean']-s['methods'][d]['A_inherited']['endpoint']['mean']))+'。'for d in DOMAINS)+'该参照没有与本章一致的32情景开发曲线，其历史8情景记录不并入主比较的交互成本或曲线面积。'

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
            fragments.append(cn+'环境中，完整方法的受限成本比该变体'+('减少'if dc>=0 else '增加')+f(abs(dc))+'千步，平均开发得分'+difference(da))
        out.append(NAMES[m]+'的比较结果为'+ '，'.join(fragments)+'。')
    out.append('各项消融只用于检验对应设计在当前任务和训练配置下的作用。某项差值较小或方向随环境改变时，不将其解释为该组件普遍必要，也不把旧记录重复读取本身当作学习收益。')
    return '\n\n'.join(out)

def threshold_table(s):
    rows=[]
    for d,cn in DOMAINS.items():
        for m in MAIN:
            g=s['methods'][d][m];rows.append([cn,NAMES[m],*[str(g['nq'][q]['reached_n'])+'/5'for q in ('q90','q95','q100')]])
    return tabular('llrrr',['环境','方法','90\\%要求','95\\%要求','100\\%要求'],rows,'不同质量要求下的确认次数','tab:ch4-threshold-sensitivity')

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
            out.append(cn+'环境中，先在每次独立运行内对实际更新记录取均值，再在五次运行间平均，旧记录有效权重比例为'+f(statistics.fmean(vals['history_only_ess_fraction']))+'，采集策略与更新前策略的平均KL散度为'+f(statistics.fmean(vals['old_policy_kl']),4)+'，采样策略变化量为'+f(statistics.fmean(vals['tv']),4)+'。')
        else:out.append(cn+'环境的内部指标仅在实际保存且有定义的更新上汇总，缺失项不以零填补。')
    return '\n\n'.join(out)

def make_section(s,internal,cites):
    q=s['thresholds'];qr=100*q['radio']['0.95'];qv=100*q['v5']['0.95']
    lead=r'''% !TeX root = ../../main.tex
\section{实验与分析}
\label{sec:ch5-experiments}
\subsection{实验设置}
\label{subsec:ch5-experiments-1}
\subsubsection{实验环境}
本章沿用前章的通信与应急资源配置环境。两个环境均在期初取得总预算，各期根据公开状态选择当期总投入与对象分配，并在任务终点返回实际配置成效。可行配置遵循式\eqref{eq:ch5-action}，期末资源余额为零。各方法使用同一物理转移与资源约束，训练和评价均检查实际扣减后的期末余额，不增加过程奖励。

\textbf{通信环境。}配置中心在32期内为100个通信对象安排功率投入。公开输入包含当前信道与待传任务等信息，不读取未来信道及环境内部评价。实际配置满足对象功率上限，策略输出经统一总量映射和空间解码后执行。对象共享编码的局部维数为3，原始联合控制维数为19。

\textbf{应急环境。}配置中心在24期内为120个区域安排应急资源，依据当前公开需求与资源余额逐期投入。未投入资源可结转至后期，末期分配全部余额。对象共享编码的局部维数为2，原始联合控制维数为13。完整回合的区域服务结果用于计算终点成效。

\subsubsection{对比方法}
主比较包含TRoC与五种外部方法。各方法通过同一公开状态和实际配置接口执行，方法差异保留在其数据利用及策略学习机制中。学习曲线比较的是新训练交互数，并非优化器更新次数。
'''
    lead+='\n\n'.join([
      'PROPS通过调整行为策略改善有限采样对目标策略分布的覆盖，并以采样记录更新目标策略\\cite{'+cites['props']+'}。本章使用目标策略与行为策略分别更新的实现，目标策略初始学习率为\\(5\\times10^{-5}\\)，行为策略学习率为\\(10^{-4}\\)。',
      'PGR使用条件生成模型构造补充转移，并利用相关性条件调节生成数据\\cite{'+cites['pgr']+'}。实际环境奖励保持不变，生成数据按已知终奖结构将非末期奖励置零，末期预测限制在公开奖励范围内。生成优先度不作为实际环境的附加奖励。',
      'SAC+ReDo在SAC学习中监测神经元活动，重置低活动神经元的输入连接并清零相应输出连接，缓解训练中的神经元休眠\\cite{'+cites['redo']+'}。其原始实现使用TensorFlow，本章将回收规则适配至PyTorch SAC，并保留按优化更新次数计量的回收周期。',
      'MR.Q通过奖励与转移等辅助预测学习状态表示，再在表示上训练策略和价值函数\\cite{'+cites['mrq']+'}。所用原生实现保留表示学习与多步价值更新，不用规划器替代实际策略决策。',
      'BRO(Fast)使用规模较大的正则化价值网络及乐观探索策略，提高连续控制中的数据利用能力\\cite{'+cites['bro']+'}。本章保留其快速配置的回放更新比，开发与终点评价使用部署策略，训练时的探索策略不混入评价。'])
    lead+=r'''

\subsubsection{训练过程与算法参数设置}
各方法在每个环境中分别从五个独立种子初始化，每次运行采集100608次新训练交互。通信与应急环境分别对应3144和4192个完整训练回合。训练任务种子与开发情景种子分开使用，开发评价前后隔离训练随机数状态。重复读取旧轨迹或生成补充数据不增加新训练交互数，环境中执行实际动作才计入该数值。

TRoC前32个完整回合按每批8条新轨迹预热，此后每批采集4条新完整轨迹并保留最近4条旧完整轨迹。折扣因子与迹参数均为1，V-trace的两个截断上界均为1。每批优化4轮，小批量为64，Actor初始学习率为\(3\times10^{-4}\)，Critic学习率为\(10^{-3}\)。Actor学习率在\([10^{-5},10^{-3}]\)内按第\ref{subsec:ch5-method-3}节调整，梯度范数上限为0.5，KL停止阈值为0.12。训练仅保存一个固定终点模型，评价结果不用于挑选训练中最好的一次模型。

外部方法保留各自的价值折扣与更新结构，主要参数见表\ref{tab:ch4-external-parameters}。PROPS的半学习率配置由前期实验后确定，在当前正式比较前固定。此前应急环境的有限敏感性测试中，目标学习率\(10^{-4}\)的结果优于\(5\times10^{-5}\)，因此本章不将所用配置称为原文默认配置或验证集最优配置。两组原始敏感性记录均保留，本章结论限于表中明确列出的配置。
'''
    lead+='\n'+tabular('lrp{9.3cm}',['方法','折扣因子','主要更新设置'],[
      ['PROPS','1','目标更新10轮，行为更新16轮，缓存8条完整轨迹'],
      ['PGR','1','回放更新比20，条件生成补充数据'],
      ['SAC+ReDo','0.99','小批量256，回放更新比8，10000次交互预热'],
      ['MR.Q','0.99','小批量256，表示跨度5，价值跨度3，目标周期250'],
      ['BRO(Fast)','0.99','小批量128，回放更新比2，2500次交互后更新']],
      '外部方法的主要训练配置','tab:ch4-external-parameters')
    lead+=r'''
PROPS目标策略的小批量在通信与应急环境中分别为8和6，行为策略的小批量分别为14和10。SAC+ReDo的回收周期为200000次优化更新，不能将其解释为200000次环境交互。各项设置在本轮训练前固定，不以终点测试结果调整。

开发与独立终点评价均使用确定性部署动作。高斯联合策略取原始控制变量的均值，再通过相同的可行配置映射执行。其他策略采用各自原生的确定性部署规则。不同方法可在CPU或GPU上运行，设备只影响工程执行，不改变训练步数和评价情景。异设备并行运行的耗时受设备与争用影响，不作为算法速度的公平排序。

\subsubsection{评价指标与统计方法}
每768次新训练交互，在32个预先固定的开发情景上评价一次，并在训练前与固定终点评价，共得到132个开发点。开发情景描述不同初始条件与外部扰动路径，并非同时设置32种环境模型。开发回合不参与学习，其物理交互数单独计量。横坐标只累计新训练交互数，纵坐标为开发情景的平均终点成效乘以100后的得分。均值曲线不作强制单调化，阴影为五次独立训练之间的样本标准差。
'''
    lead+='\n主要配置质量要求在正式训练前固定为第三章联合策略在共同32个开发情景上平均成效的95\\%，通信与应急分别为'+f(qr)+'分和'+f(qv)+'分。90\\%与100\\%要求用于检验结论对质量要求的敏感性，不根据本轮结果重新选择主要要求。'
    lead+=r'''
记第\(j\)个开发点对应的新训练交互数为\(n_j\)，开发平均得分为\(Y_j\)，预设质量要求为\(y_\star\)。主要指标取连续两次达到要求时的第二次确认步数，即
\begin{equation}
 \widehat N_{y_\star}=\min\{n_j:j\geq1,\ Y_{j-1}\geq y_\star,\ Y_j\geq y_\star\}.
 \label{eq:ch4-attainment-cost}
\end{equation}
没有连续确认时记\(\widehat N_{y_\star}=+\infty\)。未在训练上限\(B=100608\)内确认的运行按右删失报告，同时给出五次独立运行中实际确认的次数。为完整展示有限训练预算下的成本，定义受限成本
\begin{equation}
 C_{y_\star}(B)=\min(\widehat N_{y_\star},B).
 \label{eq:ch4-restricted-cost}
\end{equation}
删失时该值表示观测上限，不表示已经达到要求。交互成本越小且确认次数越多，说明在预设要求下的新交互需求越低。

训练期间整体质量以预算归一化的曲线面积表示，采用原始开发点的梯形积分
\begin{equation}
 \operatorname{AUC}(B)=\frac{1}{B}\sum_{j=1}^{M}
 \frac{Y_{j-1}+Y_j}{2}(n_j-n_{j-1}),\qquad n_0=0,\ n_M=B.
 \label{eq:ch4-normalized-learning-area}
\end{equation}
该值与得分同单位，表示给定新交互预算内的平均开发质量。终点模型另在256个未参与开发的共同独立情景上测试。先对每个模型的情景结果取均值，再以五个训练种子为统计单位计算均值和样本标准差，不将256个测试情景当作256个独立训练。终点得分用于检验减少交互后保留的配置成效，不以其单独替代样本效率指标。

\subsection{样本效率比较与组件作用}
\label{subsec:ch5-experiments-2}
\subsubsection{与外部方法的比较}
图\ref{fig:ch4-formal-main}展示相同新交互预算下的开发学习曲线，图中的水平线为训练前固定的主要质量要求。曲线提前跨过要求且连续确认，表示以较少的新环境交互学到满足要求的策略。曲线在阈值上方的高度反映成效，不直接等同于达到要求所需的交互数。
'''
    out=[lead,figure('4.4'),main_text(s),r'图\ref{fig:ch4-completed-development}将主要要求下的受限交互成本与实际确认次数同时列出。表\ref{tab:ch4-formal-results}进一步给出整个训练预算内的平均开发得分与独立终点测试得分，使新交互需求和配置质量可以分别判断。',figure('4.5'),main_table(s),bridge_text(s)]
    out+=[r'''\subsubsection{组件作用的消融比较}
三种消融变体分别针对旧轨迹利用与离策略校正及学习率控制。第一种不使用旧轨迹，每批仍采集4条新轨迹，并将当前记录重复组成8条的优化批次，使新交互节奏和优化批次大小与完整方法一致。该变体不等同于第三章方法。第二种保留新旧完整轨迹及真实采集概率，但训练时不使用离策略校正，采用完整蒙特卡洛回报构造价值目标与优势，策略比值以冻结更新前策略为中心。第三种保留轨迹复用及校正，将Actor学习率固定为\(3\times10^{-4}\)，其余裁剪与KL停止及梯度限制保持。

图\ref{fig:ch4-matched-development}比较完整方法与消融变体的开发轨迹，表\ref{tab:ch4-reuse-ablations}按相同主要要求汇总交互成本和配置成效。''',figure('4.6'),ablation_table(s),ablation_text(s)]
    out+=[r'''\subsection{预算条件与内部更新分析}
\label{subsec:ch5-experiments-3}
\subsubsection{不同交互预算与质量要求}
图\ref{fig:ch4-budget-quality}比较不同预算内的平均开发得分。五档新交互预算依次为6144步与12288步及24576步，再扩展至49152步和100608步。各预算取自同一次完整训练的前缀，不为每个预算另行训练或选择超参数。该比较用于判断收益集中在早期还是持续到较长训练过程，不能把较长预算下的终点优势倒推为所有早期预算下的优势。
''',figure('4.7'),r'表\ref{tab:ch4-threshold-sensitivity}在90\%与95\%及100\%三档要求下报告实际确认次数。较高要求下未确认的运行仍保持删失身份，不通过降低要求补记为成功。',threshold_table(s),r'''\subsubsection{旧记录权重与更新控制}
图\ref{fig:ch4-history-diagnostics}展示旧记录在更新前策略下的概率权重与采集策略差异。对一个实际更新批次中有定义的旧决策记录，记当前与采集密度比为\(w_j\)，旧决策数为\(m>0\)。权重集中程度用有效样本比例描述
\begin{equation}
 \operatorname{ESS}_{\mathrm{old}}/m=
 \frac{(\sum_{j=1}^{m}w_j)^2}{m\sum_{j=1}^{m}w_j^2}.
 \label{eq:ch4-history-effective-weight}
\end{equation}
该比例描述权重是否集中在少数记录上，不等于策略梯度方差，也不直接证明旧记录提高了学习成效。策略KL散度反映同一状态下两种策略分布的偏离。两项均只在实际有旧记录且指标有定义的更新上统计，预热阶段不以零填补。
''',figure('4.8'),internal_text(internal),r'''图\ref{fig:ch4-tv-learning-rate}将第\ref{subsec:ch5-method-3}节的采样策略变化量与实际Actor学习率并列，固定学习率变体提供相同轨迹规则下的对照。为便于观察，内部指标先按3072次新训练交互划为互不重叠的区间，在每次独立运行内平均后，再展示五次运行的均值与样本标准差。该汇总保留原始更新记录，不改变开发曲线与交互成本的计算。变化量触发学习率调整表示控制规则实际执行，其收益仍由消融比较中的交互成本与成效判断。
''',figure('4.9'),r'''\subsubsection{实际更新与计算开销}
表\ref{tab:ch4-update-costs}列出完整方法及三种消融的实际Actor与Critic优化次数，以及采集和参数更新的耗时。次数按实际发生的优化步骤累计，KL停止造成的提前结束不计作已执行更新。旧轨迹再次参与优化会增加可使用的数据，同时需要额外计算。减少新环境交互并不意味着减少优化次数或缩短墙钟时间。耗时用于说明本轮执行开销，不对异设备并行的外部方法作速度排序。''',cost_table(s)]
    return '\n\n'.join(out)+'\n'

def make_summary(s):
    text=r'''% !TeX root = ../../main.tex
\section{本章小结}
\label{sec:ch5-summary}
本章研究多轮资源配置策略学习中的新交互需求。TRoC将新完整轨迹与最近保留的完整轨迹共同用于训练，在原记录上依据真实采集概率构造有限回合V-trace目标与优势，并通过广义裁剪及策略变化控制更新。配置执行仍满足期末全额分配约束，学习反馈仍来自实际任务终点。

'''
    text+=''.join(DOMAINS[d]+'环境中，五次独立运行有'+str(s['methods'][d]['f1']['nq']['q95']['reached_n'])+'次确认达到预设主要质量要求，独立终点测试得分为'+pm(s['methods'][d]['f1']['endpoint'],100)+'。'for d in DOMAINS)
    gains=[]
    for d in DOMAINS:
        g=s['methods'][d];a=g['f1']['nq']['q95'];best=min(MAIN[1:],key=lambda m:g[m]['nq']['q95']['restricted_mean_steps']);b=g[best]['nq']['q95']
        gains.append(a['restricted_mean_steps']<b['restricted_mean_steps']and a['reached_n']>=b['reached_n'])
    if all(gains):text+='在本轮主要质量要求下，TRoC在两个环境中的受限交互成本均低于成本最低的外部对照，且确认次数不低于该对照。'
    elif any(gains):text+='主要质量要求下的交互成本收益随环境和对照方法改变，本轮结果不支持两个环境均优于全部外部对照。'
    else:text+='本轮未形成相对全部外部对照一致的交互成本优势，方法收益应按各环境的实际确认次数与成本分别判断。'
    text+='主比较以达到同等配置要求所需的新训练交互为主要判断，结合完整预算内的平均开发得分与独立终点成效。消融及旧记录权重诊断区分了轨迹复用和校正及更新控制的作用。本章结论限于明确列出的任务与配置，未将全部旧记录视为等效新样本，也未将样本效率改进等同于计算效率改进。\n'
    return text
