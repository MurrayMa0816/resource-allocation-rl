"""Minimal authorized full-spend consistency edits, prepared without live mutation."""
from __future__ import annotations
import hashlib

def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def replace_once(text,old,new,name,receipt):
    if text.count(old)!=1:raise ValueError('Guarded C4 span changed: '+name)
    receipt.append({'name':name,'old':old,'new':new,'old_sha256':sha(old),'new_sha256':sha(new)})
    return text.replace(old,new,1)

def prepare(chapter,problem,method):
    receipt=[]
    chapter=replace_once(chapter,'本章在允许终末余额的总预算上限条件下研究上述问题，仍以真实终点成效评价配置。',
      '本章沿用期末全额分配的资源约束研究上述问题，仍以真实终点成效评价配置。','chapter contract',receipt)
    problem=replace_once(problem,'当期总投入不超过资源余额，任务结束时允许有余额。',
      '当期总投入满足公开的可执行上下界，任务结束时分配全部资源。','problem narrative contract',receipt)
    old=r'''逐期可行动作集合为
\begin{equation}
 \mathcal A_B(s_t)=\left\{\boldsymbol a\in\mathbb R_+^N:
 a_i\leq c_{i,t},\quad \boldsymbol1^\top\boldsymbol a\leq R_t\right\},
 \qquad \boldsymbol a_t\in\mathcal A_B(s_t).
 \label{eq:ch5-action}
\end{equation}
这些约束允许中心选择\(0\leq q_t\leq U_t\)，对象也允许获得零投入。没有独立对象上限时可取\(c_{i,t}=R_t\)。\(U_t=0\)时只能执行零配置，总预算不要求在任务结束前全部用完。'''
    new=r'''逐期配置还需保留在期限内分配全部资源的可行性。当期公开总量下界记为\(L_t\)，按式\eqref{eq:ch4-full-spend-lower}由当前状态和公开容量确定，不读取未来扰动的实际值。逐期可行动作集合为
\begin{equation}
 \mathcal A_B(s_t)=\left\{\boldsymbol a\in\mathbb R_+^N:
 a_i\leq c_{i,t},\quad L_t\leq\boldsymbol1^\top\boldsymbol a\leq U_t\right\},
 \qquad \boldsymbol a_t\in\mathcal A_B(s_t).
 \label{eq:ch5-action}
\end{equation}
这些约束允许中心在\([L_t,U_t]\)内选择总投入，对象也允许获得零投入。没有独立对象上限时可取\(c_{i,t}=R_t\)。\(U_t=0\)时只能执行零配置。末期上下界均为剩余资源，因而期末满足\(R_T=0\)。'''
    if old not in problem:
        # A parallel modeling-language revision replaced the notation, not the
        # old cap contract. Preserve its other edits and explicit sum notation.
        old=r'''各对象投入不得超过各自上限，合计不得超过资源余额。逐期可行动作集合为
\begin{equation}
 \mathcal A_B(s_t)=\left\{\boldsymbol a\in\mathbb R_+^N:
 a_i\leq c_{i,t},\quad\sum_{i=1}^{N}a_i\leq R_t\right\},
 \qquad \boldsymbol a_t\in\mathcal A_B(s_t).
 \label{eq:ch5-action}
\end{equation}
因此中心可以选择\(0\leq q_t\leq U_t\)，各对象也可以获得零投入。没有独立对象上限时，取\(c_{i,t}=R_t\)表示仅受余额限制。\(U_t=0\)时只能执行零配置。全期累计投入不得超过初始资源，任务结束时允许有余额。'''
        new=r'''各对象投入不得超过各自上限，逐期配置还需保留在期限内分配全部资源的可行性。记当期公开总量下界为\(L_t\)，按式\eqref{eq:ch4-full-spend-lower}由当前状态与公开容量确定，不读取未来扰动的实际值。逐期可行动作集合为
\begin{equation}
 \mathcal A_B(s_t)=\left\{\boldsymbol a\in\mathbb R_+^N:
 a_i\leq c_{i,t},\quad L_t\leq\sum_{i=1}^{N}a_i\leq U_t\right\},
 \qquad \boldsymbol a_t\in\mathcal A_B(s_t).
 \label{eq:ch5-action}
\end{equation}
因此中心可以在\([L_t,U_t]\)内选择总投入，各对象也可以获得零投入。没有独立对象上限时，取\(c_{i,t}=R_t\)表示仅受余额限制。\(U_t=0\)时只能执行零配置。末期上下界均为剩余资源，因而期末满足\(R_T=0\)。'''
    problem=replace_once(problem,old,new,'feasible action set',receipt)
    old=r'由资源扣减可得\(\sum_{t=0}^{T-1}q_t=B_0-R_T\leq B_0\)。'
    if old not in problem:old=r'全期累计投入满足\(\sum_{t=0}^{T-1}q_t=B_0-R_T\leq B_0\)。'
    problem=replace_once(problem,old,r'由资源扣减与期末全额分配约束可得\(\sum_{t=0}^{T-1}q_t=B_0\)及\(R_T=0\)。','budget equality',receipt)
    old='终点成效由任务根据完整过程计算，终末余额不另计奖励或惩罚。'
    new='终点成效由任务根据完整过程计算，资源全额分配由可行动作约束保证，不另增加余额奖惩。'
    if old not in problem:
        old=r'终末余额不另计奖励或惩罚，\(J(\pi)\)作为配置质量的评价基准。'
        new=r'期末全额分配由动作约束保证，不另增加余额奖惩，\(J(\pi)\)作为配置质量的评价基准。'
    problem=replace_once(problem,old,new,'reward contract',receipt)
    start='固定配置映射将联合控制转换为满足预算上限的实际投入。'
    stop='任务末期仍使用式\\eqref{eq:ch4-adaptive-quota}，终点允许保留余额。'
    if method.count(start)!=1 or method.count(stop)!=1:raise ValueError('Quota section boundaries changed')
    a=method.index(start);b=method.index(stop,a)+len(stop);old=method[a:b]
    new=r'''固定配置映射将联合控制转换为满足期末全额分配约束的实际投入。总量上界\(U_t\)由式\eqref{eq:ch5-quota-bound}给出，下界\(L_t\)沿用式\eqref{eq:ch4-full-spend-lower}。通信环境依据当前待传任务与剩余期数，计算任务最快合法完成情况下仍可保证的后续公开容量，应急环境在非末期取零下界，在末期取全部余额。下界不使用未来信道或未来新增需求的实际值。

令当期可调范围为\(W_t=U_t-L_t\)。以剩余资源与剩余期数之比形成参照总量，并限制在公开区间内，即
\begin{equation}
 \bar q_t=\operatorname{clip}\!\left(\frac{R_t}{T-t},L_t,U_t\right),\qquad
 f_t=\frac{\bar q_t-L_t}{U_t-L_t}\quad (W_t>0).
 \label{eq:ch4-budget-reference}
\end{equation}
当\(W_t=0\)时直接取\(q_t=L_t\)。当区间非退化时，总量采用式\eqref{eq:ch4-full-spend-quota}的零调整保持映射。以\(\sigma\)表示逻辑函数，总量为
\begin{equation}
 q_t=\begin{cases}
 L_t,&W_t=0,\\
 L_t+W_t\max\{\tanh\eta_t,0\},&W_t>0,\ f_t=0,\\
 L_t+W_t\sigma\!\left(\operatorname{logit}(f_t)+\eta_t\right),&W_t>0,\ 0<f_t<1,\\
 L_t+W_t[1+\min\{\tanh\eta_t,0\}],&W_t>0,\ f_t=1.
 \end{cases}
 \label{eq:ch4-adaptive-quota}
\end{equation}
各分支在\(\eta_t=0\)时恢复参照总量。空间映射\(\mathcal D\)采用第\ref{subsec:ch3-method-2}节的固定基函数与有界单纯形投影，将空间系数和总量合为实际配置\(\boldsymbol a_t=\mathcal D(s_t,\boldsymbol w_t;q_t)\)。末期\(L_t=U_t=R_t\)，从而分配全部余额。'''
    method=replace_once(method,old,new,'quota geometry only',receipt)
    return {'chapter.tex':chapter,'section51_problem.tex':problem,'section52_method.tex':method},receipt
