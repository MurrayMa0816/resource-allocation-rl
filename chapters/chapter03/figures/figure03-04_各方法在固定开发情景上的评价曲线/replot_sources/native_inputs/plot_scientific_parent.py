#!/usr/bin/env python3
"""Present sealed Chapter 3 observations; never load models or environments.

Prepare all six assets: python render_thesis_figures.py --root ROOT
Regenerate one asset: python plot.py --render-asset . --output-dir regenerated
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, os, shutil, sys
from collections import defaultdict
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np

FULL='A_FULL'; PACE='A_PACE'; NOTASK='A_BUDGET_ONLY'; DIRECT='A_DIRECT_TOTAL'
CORE=(FULL,PACE,NOTASK,DIRECT)
MAIN=(FULL,'c_ppo','pfo','gradient_ppo','reflective_ppo','ppo_plus')
DOMAINS=('radio','v5'); SEEDS=tuple(range(5)); H={'radio':32,'v5':24}
STRONG={'radio':'c_ppo','v5':'gradient_ppo'}
NAMES={'radio':'通信功率分配','v5':'应急柴油分配'}
LABEL={FULL:'ReJoP（本文）',PACE:'规则总量',NOTASK:'去除任务摘要',DIRECT:'直接总量参数化',
       'c_ppo':'C-PPO','pfo':'PFO','gradient_ppo':'Gradient PPO','reflective_ppo':'Reflective PPO','ppo_plus':'PPO+'}
COLORS={FULL:'#0072B2',PACE:'#D55E00',NOTASK:'#009E73',DIRECT:'#7B5EA7',
        'c_ppo':'#D55E00','pfo':'#CC79A7','gradient_ppo':'#009E73','reflective_ppo':'#CC3311','ppo_plus':'#7B5EA7'}
FOLDERS={
 'main_development':'figure03-04_各方法在固定开发情景上的评价曲线',
 'main_training':'figure03-05_各方法的训练回报及滑动均值',
 'core_development':'figure03-06_ReJoP与总量决策变体的开发学习曲线',
 'core_training':'figure03-07_ReJoP与总量决策变体的训练回报及滑动均值',
 'conditions':'figure03-08_资源与时序条件变化下的配置成效',
 'configuration':'figure03-09_不同方法在固定测试情景中的跨期资源配置'}
TITLES={k:v.split('_',1)[1] for k,v in FOLDERS.items()}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(path):
 with (gzip.open if str(path).endswith('.gz') else open)(path,'rt',newline='') as f:return list(csv.DictReader(f))
def csvout(path,data):
 data=list(data);assert data
 with gzip.open(path,'wt',newline='') as f:
  w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)

def setup(cache,thesis):
 cache=cache.resolve()
 for key,name in [('MPLCONFIGDIR','matplotlib'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
  p=cache/name;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
 import matplotlib;matplotlib.use('Agg')
 from matplotlib import font_manager,rcParams
 fs=thesis/'assets/fonts/windows'; files={p.name.lower():p for p in fs.iterdir() if p.is_file()}
 for name in ('times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf','simsun.ttc','simhei.ttf'):
  if name in files:font_manager.fontManager.addfont(str(files[name]))
 assert all(name in files for name in ('times.ttf','simsun.ttc','simhei.ttf'))
 rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':10.5,
 'axes.labelsize':10.5,'axes.titlesize':10.5,'legend.fontsize':10.5,
 'xtick.labelsize':10.5,'ytick.labelsize':10.5,'axes.unicode_minus':False,
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'cm',
 'savefig.bbox':None})
 return {name:sha(files[name]) for name in ('times.ttf','simsun.ttc','simhei.ttf')}

def format_learning(ax,domain):
 from matplotlib.ticker import FuncFormatter,MaxNLocator
 ax.set_title(NAMES[domain],fontfamily=['Times New Roman','SimSun'])
 ax.set_xlabel('训练步数（×10³）');ax.set_ylabel('平均图像质量得分' if domain=='radio' else '平均需求完成比例（%）')
 ax.set_xlim(0,100608);ax.set_xticks([0,25000,50000,75000,100000]);ax.xaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value/1000:g}'))
 ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

def render(folder,dest):
 import matplotlib.pyplot as plt
 from matplotlib.ticker import MaxNLocator
 meta=read(folder/'source.json');role=meta['role'];data=rows(folder/'data.csv.gz');dest.mkdir(parents=True,exist_ok=True)
 aggregate=[]
 if role.endswith('development') or role.endswith('training'):
  arms=MAIN if role.startswith('main') else CORE
  isdev=role.endswith('development')
  fig,axes=plt.subplots(1,2,figsize=(15.6/2.54,9.0/2.54))
  for ax,d in zip(axes,DOMAINS):
   for i,m in enumerate(arms):
    series=[];grid=None
    for seed in SEEDS:
     rr=sorted([r for r in data if r['domain']==d and r['method']==m and int(r['seed'])==seed],key=lambda r:int(r['training_step']))
     xx=np.array([int(r['training_step']) for r in rr]);yy=np.array([float(r['score']) for r in rr]);assert len(rr) and (grid is None or np.array_equal(xx,grid));grid=xx;series.append(yy)
    y=np.stack(series);raw=y.mean(0)
    if isdev:
     assert y.shape==(5,198) and grid[0]==0 and grid[-1]==100608
     line=raw;sd=y.std(0,ddof=1);ax.fill_between(grid,line-sd,line+sd,color=COLORS[m],alpha=.10)
    else:
     assert y.shape==(5,100608//H[d]) and grid[-1]==100608
     ax.plot(grid,raw,color=COLORS[m],alpha=.16,lw=.45)
     ix=np.arange(y.shape[1]);start=np.maximum(0,ix-15);cs=np.pad(np.cumsum(y,axis=1),((0,0),(1,0)))
     smooth=(cs[:,ix+1]-cs[:,start])/(ix-start+1);line=smooth.mean(0);sd=np.full(line.shape,np.nan)
    ax.plot(grid,line,label=LABEL[m],color=COLORS[m],lw=1.2,linestyle='-',marker=('o','s','^','D','v','P')[i] if isdev else None,markersize=3.1,markevery=18,markerfacecolor=COLORS[m],markeredgecolor=COLORS[m])
    aggregate.extend(dict(domain=d,method=m,training_step=int(x),display_mean=float(v),raw_seed_mean=float(a),sample_sd=float(s) if isdev else '') for x,v,a,s in zip(grid,line,raw,sd))
   format_learning(ax,d)
  hs,ls=axes[0].get_legend_handles_labels();fig.legend(hs,ls,loc='lower center',ncol=3 if len(arms)==6 else 2,frameon=False,bbox_to_anchor=(.5,0))
  fig.tight_layout(rect=(0,.16,1,1))
 elif role=='conditions':
  fig,axes=plt.subplots(2,2,figsize=(15.6/2.54,14.5/2.54));idx=defaultdict(list)
  for r in data:idx[r['method'],r['domain'],r['condition']].append(float(r['score_mean']))
  for ri,d in enumerate(DOMAINS):
   for ci in (0,1):
    ax=axes[ri,ci]
    if ci==0:conds=['budget08','default','budget12'];x=[.8,1.,1.2];ms=[FULL,PACE,DIRECT,STRONG[d]];ticks=['0.8','1.0','1.2'];xlabel='初始预算相对值'
    elif d=='radio':conds=['persistence02','default','persistence095'];x=[0,1,2];ms=[FULL,PACE,NOTASK,STRONG[d]];ticks=['0.20','0.80','0.95'];xlabel='干扰状态保持概率'
    else:conds=['default','arrivals_front','arrivals_middle'];x=[0,1,2];ms=[FULL,PACE,NOTASK,STRONG[d]];ticks=['原时序','前移','中段移动'];xlabel='新增需求时序'
    for i,m in enumerate(ms):
     vals=[np.array(idx[m,d,c]) for c in conds];assert all(len(v)==5 for v in vals)
     ax.errorbar(x,[v.mean() for v in vals],yerr=[v.std(ddof=1) for v in vals],label=LABEL[m] if m!=STRONG[d] else 'C-PPO（通信）/Gradient PPO（柴油）',color='#555555' if m==STRONG[d] else COLORS[m],linestyle='-',marker=('o','s','^','D')[i],markersize=3.4,markerfacecolor=('#555555' if m==STRONG[d] else COLORS[m]),markeredgecolor=('#555555' if m==STRONG[d] else COLORS[m]),capsize=2,lw=1.1)
     aggregate.extend(dict(domain=d,method=m,condition=c,display_mean=float(v.mean()),sample_sd=float(v.std(ddof=1))) for c,v in zip(conds,vals))
    ax.set_xticks(x,ticks);ax.set_xlabel(xlabel);ax.set_ylabel('平均图像质量得分' if d=='radio' else '平均需求完成比例（%）');ax.set_title(('通信' if d=='radio' else '柴油')+('预算变化' if ci==0 else '时序条件变化'),fontfamily=['Times New Roman','SimSun']);ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
  hs={}
  for ax in axes.flat:
   for h,l in zip(*ax.get_legend_handles_labels()):hs.setdefault(l,h)
  fig.legend(hs.values(),hs.keys(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,0),handlelength=1.4,columnspacing=1.0,handletextpad=.4);fig.tight_layout(rect=(0,.12,1,1))
 elif role=='configuration':
  heat=rows(folder/'heatmap_data.csv.gz');fig,axes=plt.subplots(4,2,figsize=(15.6/2.54,16.5/2.54),sharex='col',gridspec_kw={'height_ratios':[1,1,1,1.25]})
  for ci,d in enumerate(DOMAINS):
   methods=(FULL,PACE,DIRECT,STRONG[d]);full=[]
   for i,m in enumerate(methods):
    rr=sorted([r for r in data if r['domain']==d and r['method']==m],key=lambda r:int(r['step']));assert len(rr)==H[d]
    b=float(rr[0]['R_before']);x=np.arange(1,H[d]+1);style='-';marker=('o','s','^','D')[i];color='#555555' if m==STRONG[d] else COLORS[m];label=LABEL[m] if m!=STRONG[d] else 'C-PPO（通信）/Gradient PPO（柴油）'
    if d=='radio':
     if m==FULL:axes[0,ci].plot(x,[float(r['interference_multiplier']) for r in rr],color='black',lw=1.1)
    else:axes[0,ci].plot(x,[float(r['sum_current_backlog'])/float(rr[0]['sum_current_backlog']) for r in rr],color=color,linestyle=style,lw=1.1,marker=marker,markersize=2.8,markevery=4,markerfacecolor=color,markeredgecolor=color)
    axes[1,ci].plot(x,[float(r['q'])/b for r in rr],label=label,color=color,linestyle=style,lw=1.2,marker=marker,markersize=2.8,markevery=4,markerfacecolor=color,markeredgecolor=color)
    axes[2,ci].plot(x,[float(r['R_after'])/b for r in rr],color=color,linestyle=style,lw=1.2,marker=marker,markersize=2.8,markevery=4,markerfacecolor=color,markeredgecolor=color)
    if m==FULL:full=rr
   b=float(full[0]['R_before']);x=np.arange(1,H[d]+1);forced=[i for i,r in enumerate(full) if int(r['interval_forced'])]
   axes[1,ci].plot(x,[float(r['q_reference_pace'])/b for r in full],color='#555555',linestyle=(0,(1,1)),lw=1.,label='ReJoP参照')
   if forced:axes[1,ci].scatter(x[forced],[float(full[i]['q'])/b for i in forced],marker='x',s=25,color='black',lw=1.1,label='ReJoP强制期',zorder=6)
   axes[0,ci].set_title(NAMES[d],fontfamily=['Times New Roman','SimSun']);axes[0,ci].set_ylabel('公共干扰倍数' if d=='radio' else '当前未完成需求\n/ 初始需求')
   axes[1,ci].set_ylabel('本期投入\n/ 初始总资源');axes[2,ci].set_ylabel('执行后余额\n/ 初始总资源')
   n=100 if d=='radio' else 120;matrix=np.full((n,H[d]),np.nan)
   for r in heat:
    if r['domain']==d:matrix[int(r['object_index'])-1,int(r['period'])]=float(r['allocation_percent_initial'])
   assert np.isfinite(matrix).all() and abs(float(matrix.sum())-100)<1e-7
   assert np.allclose(matrix.sum(0)/100,[float(r['q'])/b for r in full],rtol=0,atol=1e-9)
   image=axes[3,ci].imshow(matrix,origin='lower',aspect='auto',extent=(.5,H[d]+.5,.5,n+.5),vmin=0,cmap='viridis',interpolation='nearest')
   axes[3,ci].set_ylabel('ReJoP对象编号');axes[3,ci].set_xlabel('决策期');axes[3,ci].set_yticks([1,n//2,n]);axes[3,ci].set_xticks([1,H[d]//2,H[d]])
   bar=fig.colorbar(image,ax=axes[3,ci],fraction=.045,pad=.025);bar.set_label('对象投入份额（%）');bar.locator=MaxNLocator(nbins=3);bar.update_ticks()
   for ri in range(3):axes[ri,ci].yaxis.set_major_locator(MaxNLocator(nbins=3))
   for ax in axes[:,ci]:ax.set_xlim(.5,H[d]+.5)
  hs={}
  for ax in axes[1,:]:
   for h,l in zip(*ax.get_legend_handles_labels()):hs.setdefault(l,h)
  fig.legend(hs.values(),hs.keys(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,0),handlelength=1.4,columnspacing=1.0,handletextpad=.4);fig.tight_layout(rect=(0,.09,1,1))
  # Color bars narrow the bottom panels. Align all periods vertically so the
  # same period occupies the same horizontal position in all four rows.
  for ci in range(2):
   bottom=axes[3,ci].get_position()
   for ri in range(3):
    top=axes[ri,ci].get_position();axes[ri,ci].set_position([bottom.x0,top.y0,bottom.width,top.height])
 else:raise ValueError(role)
 # Keep the accepted axes geometry when renamed CJK titles have different ink heights.
 reference=folder/'layout_reference.json'
 if reference.exists():
  layout=json.loads(reference.read_text())
  assert len(fig.axes)==len(layout['axes'])
  for ax,record in zip(fig.axes,layout['axes']):ax.set_position(record['position'])
 from presentation import finish
 finish(fig,role)
 for ext in ('pdf','svg','png'):fig.savefig(dest/f'figure.{ext}',dpi=240)
 if aggregate:csvout(dest/'display_aggregates.csv.gz',aggregate)
 plt.close(fig)

