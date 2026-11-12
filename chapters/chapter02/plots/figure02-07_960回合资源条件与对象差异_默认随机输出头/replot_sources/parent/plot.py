#!/usr/bin/env python3
"""Registered scale/latency and unchanged960 condition plots from local CSV."""
from pathlib import Path
from presentation_layout import balance_legend_spacing
import argparse,csv,hashlib,json,os
os.environ.setdefault('MPLCONFIGDIR','/hy-tmp/resource_allocation_results/chapter2/scale_structure_150k_20261106/cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
STYLES={'A_FULL':('BLoAR（本文）','#0072B2','o'),'A_DIRECT':('逐对象评分','#7B5EA7','D'),'PASPO':('PASPO','#E69F00','s')}
def setup(folder):
    names=['simsun.ttc','simhei.ttf','TIMES.TTF','TIMESBD.TTF','TIMESI.TTF','TIMESBI.TTF'];folder=Path(folder)
    for name in names:font_manager.fontManager.addfont(str(folder/name))
    plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':10.5,'axes.labelsize':10.5,'axes.titlesize':10.5,'xtick.labelsize':10.5,'ytick.labelsize':10.5,'legend.fontsize':10.5,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.75,'xtick.major.width':.65,'ytick.major.width':.65,'axes.grid':False,'mathtext.fontset':'cm','pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','path.simplify':False})
    return {str(folder/name):digest(folder/name) for name in names}
def readcsv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def decorate(ax,title,xlabel,ylabel):
    ax.set_title(title,loc='center',pad=9,fontfamily=['Times New Roman','SimSun'],fontsize=10.5);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.tick_params(direction='out',length=3)
def points(ax,x,y,sd,arm):
    label,color,marker=STYLES[arm]
    ax.errorbar(x,y,yerr=sd,color=color,marker=marker,markersize=4,markerfacecolor=color,markeredgecolor=color,linestyle='-',linewidth=1.2,elinewidth=.8,capsize=2,label=label)
def legend(fig,ax,ncol=2,bottom=.015):
    h,l=ax.get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.53,bottom),ncol=ncol,frameon=False,columnspacing=1.8,handlelength=2.)
def conditions(rows):
    expected={(d,a,c) for d in ['radio','v5'] for a in ['A_FULL','PASPO'] for c in ['default','resource_low','resource_high','heterogeneity_low','heterogeneity_high']}
    keyed={(r['domain'],r['arm_id'],r['condition_id']):r for r in rows}
    if len(rows)!=20 or set(keyed)!=expected:raise ValueError('Conditions must be exactly unchanged resource/heterogeneity20, without oldscale rows')
    fig,axes=plt.subplots(2,2,figsize=(15.6/2.54,4.95));fig.subplots_adjust(left=.105,right=.985,bottom=.16,top=.94,wspace=.38,hspace=.62);evidence={}
    for row,family in enumerate(['resource','heterogeneity']):
        x=[.75,1,1.25] if family=='resource' else [.5,1,1.5];conditions_=[family+'_low','default',family+'_high']
        for col,domain in enumerate(['radio','v5']):
            ax=axes[row,col]
            for arm in ['A_FULL','PASPO']:
                r=[keyed[domain,arm,c] for c in conditions_];values=[float(v['mean']) for v in r];sd=[float(v['sample_sd']) for v in r]
                if not np.isfinite(values+sd).all():raise ValueError('Nonfinite conditions')
                points(ax,x,values,sd,arm);evidence[family+'/'+domain+'/'+arm]={'conditions':conditions_,'mean':values,'sample_sd':sd,'training_episodes':960}
            ax.set_xticks(x);ax.set_xlim(min(x)-.09*(max(x)-min(x)),max(x)+.09*(max(x)-min(x)))
            decorate(ax,f"({'abcd'[2*row+col]}) "+('通信功率分配' if domain=='radio' else '应急柴油分配'),('发射能量相对值' if domain=='radio' else '柴油供给相对值') if family=='resource' else ('信道差异程度' if domain=='radio' else '需求差异程度'),'平均图像质量得分' if domain=='radio' else '平均需求完成比例（%）')
    for ax in axes[:,0]:ax.set_ylim(0,45);ax.set_yticks([0,10,20,30,40])
    for ax in axes[:,1]:ax.set_ylim(0,80);ax.set_yticks([0,20,40,60,80])
    legend(fig,axes[0,0]);return fig,evidence
def scale(rows):
    keyed={(int(r['regions']),r['arm_id']):r for r in rows}
    if len(rows)!=6 or set(keyed)!={(n,a) for n in [60,120,240] for a in ['A_FULL','A_DIRECT']}:raise ValueError('Scale must have30 models summarized into6 conditions')
    fig,axes=plt.subplots(1,2,figsize=(15.6/2.54,3.35));fig.subplots_adjust(left=.105,right=.985,bottom=.25,top=.87,wspace=.40);evidence={}
    for arm in ['A_FULL','A_DIRECT']:
        ordered=[keyed[n,arm] for n in [60,120,240]]
        for key,sdkey,ax in [('quality_mean','quality_sd',axes[0]),('latency_mean_ms','latency_sd_ms',axes[1])]:
            values=[float(r[key]) for r in ordered];sd=[float(r[sdkey]) for r in ordered]
            if not np.isfinite(values+sd).all():raise ValueError('Nonfinite actual result')
            points(ax,[60,120,240],values,sd,arm);evidence[arm+'/'+key]={'regions':[60,120,240],'mean':values,'sample_sd':sd,'independent_training_seeds':5,'interpolated_conditions':0}
    decorate(axes[0],'(a) 终点配置成效','区域数','平均需求完成比例（%）');decorate(axes[1],'(b) 单次决策耗时','区域数','决策耗时（ms）')
    for ax in axes:ax.set_xticks([60,120,240]);ax.set_xlim(45,255)
    legend(fig,axes[0]);return fig,evidence
def support(rows,target):
    evidence={};outputs={}
    with PdfPages(target/'密集开发曲线_支撑.pdf') as pdf:
        for n in [60,120,240]:
            fig,ax=plt.subplots(figsize=(15.6/2.54,3.3));fig.subplots_adjust(left=.13,right=.97,bottom=.25,top=.86)
            for arm in ['A_FULL','A_DIRECT']:
                data={(int(r['seed']),int(r['episode'])):float(r['score_x100']) for r in rows if int(r['regions'])==n and r['arm_id']==arm};ep=np.arange(0,6257,16)
                values=np.array([[data[s,int(e)] for e in ep] for s in range(5)])
                if len(data)!=1960:raise ValueError('Incomplete native dense5seed curve')
                mean=values.mean(0);sd=values.std(0,ddof=1);label,color,marker=STYLES[arm];x=ep*24/1000
                ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.1,linewidth=0);ax.plot(x,mean,color=color,marker=marker,markevery=28,markersize=3.5,markerfacecolor=color,markeredgecolor=color,linestyle='-',linewidth=1.2,label=label)
                evidence[str(n)+'/'+arm]={'observed_points_per_seed':392,'training_seeds':5,'smoothing':False,'mean':mean.tolist(),'sample_sd':sd.tolist()}
            decorate(ax,f'N={n}：真实开发评估','训练环境交互步数（×10³）','平均需求完成比例（%）');ax.set_xticks([0,30,60,90,120,150]);ax.set_xlim(-2,153);legend(fig,ax);pdf.savefig(fig);plt.close(fig)
    outputs['密集开发曲线_支撑.pdf']=digest(target/'密集开发曲线_支撑.pdf');return outputs,evidence
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['conditions','scale','support'],required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--font-dir',type=Path,default=Path('/root/gamewar/latex_thesis_resource_allocation/assets/fonts/windows'));args=p.parse_args()
    fonts=setup(args.font_dir);rows=readcsv(args.data);args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.kind=='support':outputs,evidence=support(rows,args.output_dir)
    else:
        fig,evidence=(conditions(rows) if args.kind=='conditions' else scale(rows));outputs={}
        balance_legend_spacing(fig)
        for ext in ['pdf','png','svg']:
            path=args.output_dir/('figure.'+ext);fig.savefig(path,dpi=240);outputs[path.name]=digest(path)
        plt.close(fig)
    manifest={'schema':'chapter2-scale-structure-registered-render-v1','status':'pass','kind':args.kind,'data_sha256':digest(args.data),'plot_sha256':digest(__file__),'fonts_sha256':fonts,'outputs_sha256':outputs,'evidence':evidence,'markers_filled':True,'new_training_or_policy_evaluation_performed':False}
    (args.output_dir/'render_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'pass','kind':args.kind,'outputs':outputs}))
if __name__=='__main__':main()
