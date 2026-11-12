#!/usr/bin/env python3
"""Independent readability renderer: immutable adjacent data, new geometry/provenance."""
from pathlib import Path
import os,json,csv,hashlib,argparse
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for k,n in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
 p=ROOT/'runtime'/HERE.name/n;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
METHODS=['f1','props','mad_td','redo','mrq','bro'];DOMAINS=['radio','v5'];BUDGETS=[6144,12288,24576,49152,100608]
LABELS={'f1':'TRoC（本文）','props':'PROPS','mad_td':'MAD-TD','redo':'SAC+ReDo','mrq':'MR.Q','bro':'BRO (Fast)'}
COLORS={'f1':'#D55E00','props':'#0072B2','mad_td':'#009E73','redo':'#7B5EA7','mrq':'#CC79A7','bro':'#666666'}
MARKERS=['o','s','^','D','v','P'];NAMES=['通信功率分配','应急柴油分配']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def setup():
 root=Path('/root/gamewar/latex_thesis_resource_allocation/assets/fonts/windows')
 for n in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf','simsun.ttc','simhei.ttf']:
  p=root/n
  if p.exists():font_manager.fontManager.addfont(str(p))
 plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':10.5,'axes.labelsize':10.5,'axes.titlesize':10.5,'legend.fontsize':10.5,'xtick.labelsize':10.5,'ytick.labelsize':10.5,'axes.titleweight':'normal','axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'c4-efficiency-readability-20261108','mathtext.fontset':'cm'})
def cost(rows,cfg):
 height_cm=8.4-15*2.54/72  # Remove one 15-bp header row; keep axes at the same absolute position.
 fig,axs=plt.subplots(1,2,sharey=True,figsize=(15.6/2.54,height_cm/2.54));fig.subplots_adjust(left=.17,right=.88,bottom=1.26/height_cm,top=(1.26+5.5)/height_cm,wspace=.38);points=[]
 for k,(ax,dom)in enumerate(zip(axs,DOMAINS)):
  for i,m in enumerate(METHODS):
   rr=[r for r in rows if r['domain']==dom and r['method']==m];assert len(rr)==5
   values=np.asarray([float(r['restricted_steps'])for r in rr])/1000;reached=sum(r['censored'].lower()not in ('true','1')for r in rr);mean=values.mean();sd=values.std(ddof=1)
   ax.barh(i,mean,xerr=sd,height=.64,color=COLORS[m],alpha=.78,hatch='///'if reached<5 else'',edgecolor='#666666'if reached<5 else COLORS[m],linewidth=.35,error_kw=dict(elinewidth=.8,capsize=2,capthick=.8))
   ax.text(1.11,i,f'{reached}/5',transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=10.5)
   points.append(dict(domain=dom,method=m,restricted_mean_steps=float(mean*1000),sample_sd_steps=float(sd*1000),reached_n=reached,seed_count=5))
  ax.set_title(f'({chr(97+k)}) {NAMES[k]}',loc='center',pad=7,fontsize=10.5);ax.set_xlim(0,112);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('训练步数（×10³）');ax.set_yticks(range(6));ax.set_yticklabels([LABELS[m]for m in METHODS]);ax.tick_params(axis='y',length=0,labelleft=k==0);ax.set_ylim(5.65,-.65)
  ax.text(1.11,1.03,'达到次数/5',transform=ax.transAxes,ha='center',va='bottom',fontsize=10.5)
 return fig,points

def prefix(rows,cfg):
 fig,axs=plt.subplots(1,2,figsize=(15.6/2.54,10.5/2.54));fig.subplots_adjust(left=.13,right=.98,bottom=.34,top=.84,wspace=.45);points=[]
 for k,(ax,dom)in enumerate(zip(axs,DOMAINS)):
  for i,m in enumerate(METHODS):
   rr=[[r for r in rows if r['domain']==dom and r['method']==m and int(r['seed'])==seed]for seed in range(5)]
   rr=[sorted(v,key=lambda r:int(r['training_budget']))for v in rr];assert all([int(r['training_budget'])for r in v]==BUDGETS for v in rr)
   y=np.asarray([[float(r['normalized_auc'])for r in v]for v in rr]);mean=y.mean(0)*100;sd=y.std(0,ddof=1)*100;x=np.arange(5)
   ax.plot(x,mean,color=COLORS[m],lw=1.2,marker=MARKERS[i],markersize=3.5,label=LABELS[m],zorder=3)
   ax.fill_between(x,mean-sd,mean+sd,color=COLORS[m],alpha=.10,zorder=0)
   points.extend(dict(domain=dom,method=m,training_step=n,mean_x100=float(a),sample_sd_x100=float(b),seed_count=5)for n,a,b in zip(BUDGETS,mean,sd))
  ax.set_title(f'({chr(97+k)}) {NAMES[k]}',loc='center',pad=8,fontsize=10.5);ax.set_xlim(-.18,4.18);ax.set_xticks(range(5));ax.set_xticklabels(['6.1','12.3','24.6','49.2','100.6']);ax.set_xlabel('统计至第几千步');ax.set_ylabel('训练期间平均\n图像质量得分'if dom=='radio'else'训练期间平均\n需求完成比例（%）')
 handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.55,.10),handlelength=1.8,columnspacing=1.1,handletextpad=.4,fontsize=10.5)
 fig.text(.5,.035,'同一次长训练从0步至所示步数的平均值',ha='center',va='bottom',fontsize=10.5)
 return fig,points

def geometry(fig):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height();records=[]
 def rec(t,role):
  b=t.get_window_extent(renderer);records.append(dict(role=role,text=t.get_text(),font_size=t.get_fontsize(),font_family=t.get_fontfamily(),bbox_display=[b.x0,b.y0,b.x1,b.y1]))
 for ax in fig.axes:
  for t in [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels(),*ax.texts]:
   if t.get_visible()and t.get_text():rec(t,'axis')
 for leg in fig.legends:
  for t in leg.get_texts():rec(t,'legend')
 for t in fig.texts:rec(t,'figure_note')
 assert all(t['font_size']==10.5 for t in records)
 return dict(schema='c4-readability-new-geometry-proof-v1',canvas_px=[w,h],figure_inches=list(fig.get_size_inches()),axes=[dict(position=list(a.get_position().bounds),xlim=list(a.get_xlim()),ylim=list(a.get_ylim()),xticks=[float(v)for v in a.get_xticks()],xticklabels=[t.get_text()for t in a.get_xticklabels()],title=a.get_title(),xlabel=a.get_xlabel(),ylabel=a.get_ylabel())for a in fig.axes],texts=records,all_text_10_5pt=True,geometry_changed_intentionally=True,old_label_only_proof_reused=False)
def main():
 a=argparse.ArgumentParser();a.add_argument('--output-dir',type=Path,default=HERE);args=a.parse_args();out=args.output_dir.resolve();assert out.is_relative_to(ROOT);out.mkdir(parents=True,exist_ok=True);setup();cfg=json.loads((HERE/'figure_config.json').read_text());rows=list(csv.DictReader((HERE/'data.csv').open()))
 fig,points=cost(rows,cfg)if cfg['kind']=='cost'else prefix(rows,cfg);fig.set_dpi(200);g=geometry(fig)
 (out/'render_geometry.json').write_text(json.dumps(g,ensure_ascii=False,indent=2)+'\n')
 for ext in ['pdf','png','svg']:
  p=out/('figure.'+ext);assert not p.is_symlink();fig.savefig(p,dpi=200,metadata={'Creator':'C4 independent readability renderer','CreationDate':None,'ModDate':None}if ext=='pdf'else({'Date':None}if ext=='svg'else None))
 plt.close(fig)
 with(out/'plotted_points.csv').open('w',newline='')as f:
  wr=csv.DictWriter(f,fieldnames=list(points[0]));wr.writeheader();wr.writerows(points)
 provenance=json.loads((HERE/'provenance/parent_binding.json').read_text());source=dict(schema='c4-comparison-scores-in-body-one-header-row-v1',status='RENDERED_PLAIN_ONE_HEADER_ROW_ALL_VALUES_PRESERVED',role=cfg['kind'],parent_binding=provenance,immediate_parent_directory='/hy-tmp/resource_allocation_results/chapter04/figures/figure45_spacing_scores_20261109_142748/assets/figure4-5',presentation_change='Remove the two comparison-score text lines; align subplot titles and reached-count headings in one header row. Reduce canvas height by 15 bp while preserving absolute axes positions, 5.5 cm axes height, bar dimensions and bottom label spacing.',data_sha256=sha(HERE/'data.csv'),figure_config_sha256=sha(HERE/'figure_config.json'),plot_sha256=sha(HERE/'plot.py'),render_geometry_sha256=sha(out/'render_geometry.json'),plotted_points_sha256=sha(out/'plotted_points.csv'),outputs_sha256={e:sha(out/('figure.'+e))for e in ['pdf','png','svg']},width_cm=15.6,height_cm=float(fig.get_size_inches()[1]*2.54),all_text_pt=10.5,fonts=['Times New Roman','SimSun'],new_render_and_geometry=True,old_label_only_proof_reused=False,formal_data_modified=False,derived_data_amendment=json.loads((HERE/'props_b_derivation.json').read_text()),model_loads=0,environment_steps=0,optimizer_updates=0,replot_command='python -B plot.py --output-dir NEW_HY_DIRECTORY_UNDER_REFINEMENT_ROOT')
 (out/'source.json').write_text(json.dumps(source,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
