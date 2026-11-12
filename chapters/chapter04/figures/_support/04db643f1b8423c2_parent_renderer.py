#!/usr/bin/env python3
"""Presentation-only saved-log figures. No policies, environment or learning."""
from pathlib import Path
import argparse,csv,json,os,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for k,v in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
 p=ROOT/'runtime'/HERE.name/v;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
COLORS={'f1':'#D55E00','fixed_actor_lr':'#0072B2','radio':'#0072B2','v5':'#D55E00'}
LABELS={'f1':'TRoC（本文）','fixed_actor_lr':'固定Actor学习率','radio':'通信功率分配','v5':'应急柴油分配'}
DOMAINS=['radio','v5'];METHODS=['f1','fixed_actor_lr']

def setup():
 f=Path('/root/gamewar/latex_thesis_resource_allocation/assets/fonts/windows')
 for n in ['simsun.ttc','simhei.ttf','TIMES.TTF','TIMESBD.TTF','TIMESI.TTF','TIMESBI.TTF']:font_manager.fontManager.addfont(str(f/n))
 plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':10.5,'axes.labelsize':10.5,'axes.titlesize':10.5,'xtick.labelsize':10.5,'ytick.labelsize':10.5,'legend.fontsize':10.5,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,'font.weight':'normal','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'cm','svg.hashsalt':'chapter04-readable-mechanism-v1','savefig.dpi':200})

def read(n):
 with(HERE/n).open(newline='',encoding='utf-8')as f:return list(csv.DictReader(f))

def finite(x):
 try:return np.isfinite(float(x))
 except(ValueError,TypeError):return False

def valid(rows,domain,method,metric):
 return[r for r in rows if r['domain']==domain and r['method']==method and r['warmup']=='False'and r['replay_enabled']=='True'and float(r['active_events'])>0 and finite(r.get(metric))]

def writecsv(path,records):
 fields=list(dict.fromkeys(k for r in records for k in r))
 with path.open('w',newline='',encoding='utf-8')as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)

def timing_axes(ax,title,ylabel,letter):
 ax.set_title(f'({letter}) {title}',loc='center',pad=9,fontweight='normal')
 ax.set_xlim(0,100.608);ax.set_xticks([0,25,50,75,100]);ax.set_xticklabels(['0','25','50','75','100']);ax.get_xticklabels()[-1].set_ha('right')
 ax.set_xlabel('训练步数（×10³）');ax.set_ylabel(ylabel)

def trajectory(rows,domain,method,metric):
 raw={};bins={}
 for seed in range(5):
  rr=valid(rows,domain,method,metric);seedbins={}
  for r in rr:
   if int(r['seed'])!=seed:continue
   step=int(r['training_step']);v=float(r[metric]);raw.setdefault(step,[]).append(v);seedbins.setdefault((step-1)//3072,[]).append(v)
  for key,vals in seedbins.items():bins.setdefault(min((key+1)*3072,100608),[]).append(float(np.mean(vals)))
 def summarize(d):return [(step,float(np.mean(v)),float(np.std(v,ddof=1)),len(v))for step,v in sorted(d.items())]
 return summarize(raw),summarize(bins)

def ecdf_figure():
 data=read('ecdf_points.csv');fig,axes=plt.subplots(1,2,figsize=(15.6/2.54,8.25/2.54));fig.subplots_adjust(left=.12,right=.98,bottom=.31,top=.86,wspace=.43)
 specs=[('history_only_ess_fraction','旧记录权重有效样本比例（%）','旧记录的权重分布'),('old_policy_kl','旧策略与更新前策略的差异（KL）','新旧策略的差异')]
 for i,(metric,xlabel,title)in enumerate(specs):
  ax=axes[i]
  for domain in DOMAINS:
   z=[r for r in data if r['domain']==domain and r['metric']==metric];x=np.array([float(r['raw_value'])*(100 if metric=='history_only_ess_fraction'else 1)for r in z]);y=np.array([float(r['cumulative_batches_pct_equal_seed_mean'])for r in z]);sd=np.array([float(r['between_seed_sample_sd_percentage_points'])for r in z]);color=COLORS[domain]
   ax.step(x,y,where='post',color=color,lw=1.25,label=LABELS[domain]);ax.fill_between(x,np.maximum(0,y-sd),np.minimum(100,y+sd),step='post',color=color,alpha=.12)
  ax.set_title(f'({chr(97+i)}) {title}',loc='center',pad=9);ax.set_xlabel(xlabel);ax.set_ylabel('训练批次累计占比（%）');ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100])
  if i==0:ax.set_xlim(93,100);ax.set_xticks([93,95,97,99,100]);ax.get_xticklabels()[-1].set_ha('right')
  else:ax.set_xlim(0,max(float(r['raw_value'])for r in data if r['metric']==metric)*1.035);ax.set_xticks([0,.01,.02,.03]);ax.set_xticklabels(['0','0.01','0.02','0.03'])
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False,handlelength=1.9,columnspacing=1.5)
 return fig,data

def control_figure(rows):
 fig,axes=plt.subplots(2,2,figsize=(15.6/2.54,14.6/2.54));fig.subplots_adjust(left=.145,right=.98,bottom=.18,top=.925,wspace=.46,hspace=.85);points=[];bars=read('exceedance_summary.csv');seedrows=read('exceedance_per_seed.csv')
 for col,domain in enumerate(DOMAINS):
  ax=axes[0,col];ax.axhspan(1/30,1/15,color='#777777',alpha=.13,zorder=0);ax.axhline(1/30,color='#777777',lw=.7,ls='--');ax.axhline(1/15,color='#555555',lw=.7,ls=':')
  for method in METHODS:
   raw,binned=trajectory(rows,domain,method,'tv');xr=np.array([v[0]/1000 for v in raw]);yr=np.array([v[1]for v in raw]);x=np.array([v[0]/1000 for v in binned]);y=np.array([v[1]for v in binned]);sd=np.array([v[2]for v in binned]);color=COLORS[method]
   ax.plot(xr,yr,color=color,lw=.45,alpha=.19);ax.plot(x,y,color=color,lw=1.2,marker='o'if method=='f1'else's',markersize=2.8,markevery=4,label=LABELS[method]);ax.fill_between(x,y-sd,y+sd,color=color,alpha=.10)
   for kind,values in [('raw_batch_mean',raw),('nonoverlapping3072_step_bin',binned)]:
    points.extend({'panel':chr(97+col),'domain':domain,'method':method,'metric':'tv','kind':kind,'training_step':v[0],'mean':v[1],'sample_sd':v[2],'seed_count':v[3]}for v in values)
  timing_axes(ax,LABELS[domain],'单批策略更新幅度\n（样本估计）',chr(97+col));ax.set_ylim(0,.13);ax.set_yticks([0,.03,.06,.09,.12])
  ax=axes[1,col]
  for i,method in enumerate(METHODS):
   r=next(r for r in bars if r['domain']==domain and r['method']==method);mean=float(r['above_upper_pct_mean_equal_seed']);sd=float(r['above_upper_pct_sample_sd']);seedvals=[float(q['above_upper_pct'])for q in seedrows if q['domain']==domain and q['method']==method];color=COLORS[method];ax.bar(i,mean,width=.55,color=color,alpha=.77,edgecolor=color,linewidth=.6,yerr=sd,error_kw={'elinewidth':.8,'capsize':3,'capthick':.8})
   ax.text(i,mean+sd+4,f'{mean:.2f}%',ha='center',va='bottom',fontsize=10.5);points.append({'panel':chr(99+col),'domain':domain,'method':method,'metric':'fraction_D_gt_1over15_pct','kind':'equal_seed_mean','training_step':100608,'mean':mean,'sample_sd':sd,'seed_count':5})
  ax.set_title(f'({chr(99+col)}) {LABELS[domain]}',loc='center',pad=9);ax.set_ylabel('超过上参考值的\n批次比例（%）');ax.set_ylim(0,110);ax.set_yticks([0,25,50,75,100]);ax.set_xlim(-.65,1.65);ax.set_xticks([0,1]);ax.set_xticklabels(['TRoC（本文）','固定Actor\n学习率']);ax.tick_params(axis='x',length=0)
 handles=[Line2D([0],[0],color=COLORS[m],lw=1.3,marker='o'if m=='f1'else's',markersize=3,label=LABELS[m])for m in METHODS];handles.append(Patch(facecolor='#777777',alpha=.13,label='学习率调节参考区间'));fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.035),ncol=3,frameon=False,handlelength=1.8,columnspacing=1.25)
 return fig,points

def lr_figure(rows):
 fig,axes=plt.subplots(1,2,figsize=(15.6/2.54,8.3/2.54));fig.subplots_adjust(left=.14,right=.98,bottom=.31,top=.86,wspace=.44);points=[]
 for col,domain in enumerate(DOMAINS):
  ax=axes[col]
  for method in METHODS:
   _,z=trajectory(rows,domain,method,'actor_lr_before');x=np.array([v[0]/1000 for v in z]);y=np.array([v[1]for v in z]);sd=np.array([v[2]for v in z]);ax.plot(x,y,color=COLORS[method],lw=1.2,label=LABELS[method]);ax.fill_between(x,np.maximum(y-sd,1e-8),y+sd,color=COLORS[method],alpha=.12);points.extend({'domain':domain,'method':method,'metric':'actor_lr_before','kind':'nonoverlapping3072_step_bin','training_step':v[0],'mean':v[1],'sample_sd':v[2],'seed_count':v[3]}for v in z)
  timing_axes(ax,LABELS[domain],'本批实际采用的\nActor学习率',chr(97+col));ax.set_yscale('log');ax.set_ylim(4e-5,4e-4)
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False)
 return fig,points

def geometry(fig,path):
 fig.canvas.draw();ren=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height();rec=[]
 for i,ax in enumerate(fig.axes):
  ts=[('title',ax.title),('xlabel',ax.xaxis.label),('ylabel',ax.yaxis.label)]+[('xtick',t.label1)for t in ax.xaxis._update_ticks()]+[('ytick',t.label1)for t in ax.yaxis._update_ticks()]+[('annotation',t)for t in ax.texts]
  for k,t in ts:
   if not t.get_visible()or not t.get_text().strip():continue
   b=t.get_window_extent(ren);rec.append({'axis':i,'kind':k,'text':t.get_text(),'font_size':t.get_fontsize(),'font_family':t.get_fontfamily(),'font_weight':t.get_fontweight(),'font_style':t.get_fontstyle(),'bbox_xyxy':[b.x0,b.y0,b.x1,b.y1]})
 for lg in fig.legends:
  for t in lg.get_texts():
   b=t.get_window_extent(ren);rec.append({'axis':None,'kind':'legend','text':t.get_text(),'font_size':t.get_fontsize(),'font_family':t.get_fontfamily(),'font_weight':t.get_fontweight(),'font_style':t.get_fontstyle(),'bbox_xyxy':[b.x0,b.y0,b.x1,b.y1]})
 outside=[r for r in rec if r['bbox_xyxy'][0]< -1 or r['bbox_xyxy'][1]< -1 or r['bbox_xyxy'][2]>width+1 or r['bbox_xyxy'][3]>height+1]
 if outside:raise ValueError('Text outside canvas: '+json.dumps(outside,ensure_ascii=False))
 path.write_text(json.dumps({'schema':'chapter04-mechanism-presentation-geometry-v1','figure_inches':fig.get_size_inches().tolist(),'canvas_pixels':[width,height],'texts':rec,'text_outside_canvas':outside,'no_old_pixel_invariance_claim':True},ensure_ascii=False,indent=2)+'\n')

def export(fig,points,out,name):
 fig.set_dpi(200);geometry(fig,out/(name+'_geometry.json'))
 for suffix in ['pdf','png','svg']:fig.savefig(out/(name+'.'+suffix),dpi=200,metadata={'Creator':'chapter04 mechanism readability','CreationDate':None,'ModDate':None}if suffix=='pdf'else({'Date':None}if suffix=='svg'else None))
 plt.close(fig);writecsv(out/('plotted_points.csv'if name=='figure'else name+'_points.csv'),points)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=HERE);args=p.parse_args();out=args.output_dir.resolve()
 if not out.is_relative_to(ROOT):raise ValueError('All presentation outputs remain inside independent HY root')
 out.mkdir(parents=True,exist_ok=True);setup();config=json.loads((HERE/'figure_config.json').read_text());rows=read('data.csv')
 if config['kind']=='old_weight_distribution':fig,points=ecdf_figure();export(fig,points,out,'figure')
 elif config['kind']=='controlled_updates':
  fig,points=control_figure(rows);export(fig,points,out,'figure');fig,points=lr_figure(rows);export(fig,points,out,'learning_rate_diagnostic')
 else:raise ValueError('Unknown presentation kind')
if __name__=='__main__':main()
