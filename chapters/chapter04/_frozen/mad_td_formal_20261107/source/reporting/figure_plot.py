#!/usr/bin/env python3
"""Reproduce one accepted C4 data figure from adjacent CSV and configuration.

This file is copied unchanged into each published figure directory. It never
loads policies or invokes environments/training. Main curves are unsmoothed.
Internal measurements use per-seed nonoverlapping3072-step bin means only.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import sys

HERE=Path(__file__).resolve().parent
CACHE=Path('/hy-tmp/resource_allocation_results/chapter4/mad_td_formal_20261107/manuscript_preparation/runtime/plots')
for key,leaf in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
    p=CACHE/leaf;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

DOMAINS=('radio','v5')
NAMES={'radio':'图像下行传输','v5':'应急资源配置'}
LABELS={'f1':'TRoC','props':'PROPS','mad_td':'MAD-TD','redo':'SAC+ReDo','mrq':'MR.Q','bro':'BRO (Fast)',
        'no_history':'不使用旧轨迹','no_offpolicy_correction':'不作离策略训练校正','fixed_actor_lr':'固定Actor学习率'}
COLORS={'f1':'#D55E00','props':'#0072B2','mad_td':'#009E73','redo':'#7B5EA7','mrq':'#CC79A7','bro':'#666666',
        'no_history':'#0072B2','no_offpolicy_correction':'#009E73','fixed_actor_lr':'#7B5EA7'}
STYLES=('-','--','-.',':','--','-.')
MARKERS=('o','s','^','D','v','P')
STEPS=100608


def setup():
    roots=[]
    if os.environ.get('THESIS_FONT_DIR'):roots.append(Path(os.environ['THESIS_FONT_DIR']))
    roots += [p/'assets/fonts/windows'for p in HERE.parents]
    roots.append(Path('/root/gamewar/latex_thesis_resource_allocation/assets/fonts/windows'))
    files=None
    for root in roots:
        if root.is_dir():
            candidate={p.name.lower():p for p in root.iterdir()if p.is_file()}
            if all(x in candidate for x in ('times.ttf','simsun.ttc','simhei.ttf')):
                files=candidate;break
    if files is None:raise FileNotFoundError('Verified SimSun/Times New Roman fonts unavailable')
    for name in ('times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf','simsun.ttc','simhei.ttf'):
        if name in files:font_manager.fontManager.addfont(str(files[name]))
    plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':10.5,
        'axes.labelsize':10.5,'axes.titlesize':10.5,'legend.fontsize':9.5,
        'xtick.labelsize':9.5,'ytick.labelsize':9.5,'axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'cm',
        'svg.hashsalt':'c4-formal-100k-accepted-figure-v1','savefig.dpi':240})
    return files


def number(row,key):
    value=row.get(key,'')
    if value in ('','None','null','nan','NaN',None):return None
    value=float(value)
    return value if np.isfinite(value)else None


def base_axes(ax,domain,ylabel):
    ax.set_title(NAMES[domain],fontfamily=['Times New Roman','SimHei'],fontsize=10.5)
    ax.set_xlabel('新训练交互量（千步）');ax.set_ylabel(ylabel)
    ax.set_xlim(0,STEPS/1000);ax.set_xticks([0,25,50,75,100.608]);ax.set_xticklabels(['0','25','50','75','100.6'])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def score_label(domain):
    return '平均SSIM乘100'if domain=='radio'else'区域完成比例（%）'


def groups(rows,domain,method,xfield='training_step',yfield='mean_return'):
    result={}
    for seed in range(5):
        rr=[r for r in rows if r['domain']==domain and r['method']==method and int(r['seed'])==seed]
        rr=sorted(rr,key=lambda r:float(r[xfield]))
        if not rr:raise ValueError(('Missing accepted seed',domain,method,seed))
        result[seed]=rr
    x=np.array([float(r[xfield])for r in result[0]],dtype=float)
    for seed in range(1,5):
        if [float(r[xfield])for r in result[seed]]!=x.tolist():raise ValueError('Seed evaluation grids differ')
    y=np.array([[float(r[yfield])for r in result[s]]for s in range(5)])
    return x,y


def learning(rows,config,prefix=False):
    fig,axes=plt.subplots(1,2,figsize=(15.6/2.54,9.5/2.54));plotted=[]
    for k,domain in enumerate(DOMAINS):
        ax=axes[k]
        for i,method in enumerate(config['methods']):
            if prefix:x,y=groups(rows,domain,method,'training_budget','normalized_auc')
            else:x,y=groups(rows,domain,method)
            mean=y.mean(0)*100;sd=y.std(0,ddof=1)*100
            ax.plot(x/1000,mean,label=LABELS[method],color=COLORS[method],linestyle=STYLES[i%6],
                lw=1.2,marker=MARKERS[i%6],markersize=2.5,markevery=1 if prefix else 16)
            ax.fill_between(x/1000,mean-sd,mean+sd,color=COLORS[method],alpha=.10)
            plotted.extend(dict(domain=domain,method=method,training_step=int(a),mean_x100=float(b),sample_sd_x100=float(c),seed_count=5)for a,b,c in zip(x,mean,sd))
        base_axes(ax,domain,score_label(domain)if not prefix else'预算内平均开发得分')
        if not prefix and config.get('thresholds'):
            ax.axhline(config['thresholds'][domain],color='#888888',ls=':',lw=.7)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3 if len(handles)>4 else 2,frameon=False,bbox_to_anchor=(.5,.0))
    fig.tight_layout(rect=(0,.16,1,1))
    return fig,plotted


def costs(rows,config):
    fig,axes=plt.subplots(1,2,figsize=(15.6/2.54,9.5/2.54));plotted=[]
    for k,domain in enumerate(DOMAINS):
        ax=axes[k]
        for i,method in enumerate(config['methods']):
            rr=[r for r in rows if r['domain']==domain and r['method']==method]
            if len(rr)!=5:raise ValueError('Five seed costs required')
            values=np.array([float(r['restricted_steps'])for r in rr])/1000
            reached=sum(str(r['censored']).lower()not in ('true','1')for r in rr)
            bar=ax.bar(i,values.mean(),yerr=values.std(ddof=1),width=.70,color=COLORS[method],alpha=.78,
                error_kw=dict(elinewidth=.8,capsize=2,capthick=.8),hatch='//'if reached<5 else'')
            ax.text(i,values.mean()+values.std(ddof=1)+2,f'{reached}/5',ha='center',va='bottom',fontsize=9.5)
            plotted.append(dict(domain=domain,method=method,restricted_mean_steps=float(values.mean()*1000),sample_sd_steps=float(values.std(ddof=1)*1000),reached_n=reached,seed_count=5))
        ax.set_title(NAMES[domain],fontfamily=['Times New Roman','SimHei']);ax.set_ylabel('受限交互成本（千步）')
        ax.set_xticks(range(len(config['methods'])));ax.set_xticklabels([LABELS[m]for m in config['methods']],rotation=35,ha='right')
        ax.set_ylim(0,max(112,max(p['restricted_mean_steps']/1000+p['sample_sd_steps']/1000+8 for p in plotted if p['domain']==domain)))
    fig.tight_layout(rect=(0,.05,1,1));return fig,plotted


def binned(rows,domain,method,metric):
    records={}
    for seed in range(5):
        bins={}
        for r in rows:
            if r.get('domain')!=domain or r.get('method')!=method or int(r['seed'])!=seed:continue
            value=number(r,metric);step=number(r,'training_step')
            if value is None or step is None or step<=0:continue
            key=int((step-1)//3072);bins.setdefault(key,[]).append(value)
        for key,values in bins.items():records.setdefault(key,[]).append(float(np.mean(values)))
    output=[]
    for key,values in sorted(records.items()):
        output.append((min((key+1)*3072,STEPS),float(np.mean(values)),float(np.std(values,ddof=1))if len(values)>1 else None,len(values)))
    return output


def internal(rows,config,control=False):
    fig,axes=plt.subplots(2,2,figsize=(15.6/2.54,13.8/2.54));plotted=[]
    metrics=[('tv','更新后TV'),('actor_lr_before','Actor学习率')]if control else[
        ('history_only_ess_fraction','旧决策归一化有效样本量'),('old_policy_kl','采集策略到更新前策略KL')]
    for col,domain in enumerate(DOMAINS):
        for row,(metric,label)in enumerate(metrics):
            ax=axes[row,col]
            for i,method in enumerate(config['methods']):
                data=binned(rows,domain,method,metric)
                if not data:raise ValueError(('No real internal measurements',domain,method,metric))
                x=np.array([v[0]for v in data])/1000;y=np.array([v[1]for v in data]);sd=np.array([v[2]if v[2]is not None else np.nan for v in data])
                ax.plot(x,y,color=COLORS[method],linestyle=STYLES[i],label=LABELS[method],lw=1.15)
                valid=np.isfinite(sd)
                if valid.any():ax.fill_between(x[valid],(y-sd)[valid],(y+sd)[valid],color=COLORS[method],alpha=.12)
                plotted.extend(dict(domain=domain,method=method,metric=metric,training_step=d[0],mean=d[1],sample_sd=d[2],seed_count=d[3])for d in data)
            base_axes(ax,domain,label)
            if row==0 and control:
                ax.axhline(.2/1.5/2,color='#777777',ls=':',lw=.7);ax.axhline(.2/1.5/4,color='#999999',ls='--',lw=.7)
            if row==1 and control:ax.set_yscale('log')
            if row==0 and not control:ax.set_ylim(0,1.025)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=len(handles),frameon=False,bbox_to_anchor=(.5,.0))
    fig.tight_layout(rect=(0,.07,1,1));return fig,plotted


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=HERE/'regenerated');a=p.parse_args()
    config=json.loads((HERE/'figure_config.json').read_text());setup()
    with(HERE/'data.csv').open(newline='',encoding='utf-8')as f:data=list(csv.DictReader(f))
    if config['kind']in ('learning','ablation'):fig,points=learning(data,config)
    elif config['kind']=='prefix_auc':fig,points=learning(data,config,prefix=True)
    elif config['kind']=='cost':fig,points=costs(data,config)
    elif config['kind']=='history':fig,points=internal(data,config)
    elif config['kind']=='control':fig,points=internal(data,config,control=True)
    else:raise ValueError('Unknown figure role')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for suffix in ('pdf','png','svg'):
        fig.savefig(a.output_dir/('figure.'+suffix),dpi=200,metadata={'Creator':'accepted C4 local plot.py'}if suffix=='pdf'else None)
    plt.close(fig)
    fields=list(dict.fromkeys(k for row in points for k in row))
    with(a.output_dir/'plotted_points.csv').open('w',newline='',encoding='utf-8')as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(points)


if __name__=='__main__':main()
