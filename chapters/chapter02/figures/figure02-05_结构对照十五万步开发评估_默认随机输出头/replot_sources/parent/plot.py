#!/usr/bin/env python3
"""Plot registered40 dense development observations from adjacent data.csv only."""
from pathlib import Path
from presentation_layout import balance_legend_spacing
import argparse,csv,hashlib,json,os
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR','/hy-tmp/resource_allocation_results/chapter2/ablations_150k_20261105/cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
import numpy as np

ARMS=('A_FULL','A_RANDOM_MATCH','A_SOFTMAX','A_DIRECT')
DOMAINS=('radio','v5');SEEDS=tuple(range(5));HORIZONS={'radio':32,'v5':24};EPISODES={'radio':4688,'v5':6256}
STYLES={'A_FULL':('BLoAR（本文）','#0072B2','o'),'A_RANDOM_MATCH':('随机基函数','#E69F00','s'),'A_SOFTMAX':('比例映射','#009E73','^'),'A_DIRECT':('逐对象评分','#7B5EA7','D')}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def setup(font_dir):
    directory=Path(font_dir)
    for name in ['simsun.ttc','simhei.ttf','TIMES.TTF','TIMESBD.TTF','TIMESI.TTF','TIMESBI.TTF']:
        p=directory/name
        if not p.is_file():raise FileNotFoundError(p)
        font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':10.5,'axes.labelsize':10.5,'axes.titlesize':10.5,'xtick.labelsize':10.5,'ytick.labelsize':10.5,'legend.fontsize':10.5,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.75,'xtick.major.width':.65,'ytick.major.width':.65,'axes.grid':False,'mathtext.fontset':'cm','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','path.simplify':False})
    return {name:digest(directory/name) for name in ['simsun.ttc','simhei.ttf','TIMES.TTF','TIMESBD.TTF','TIMESI.TTF','TIMESBI.TTF']}

def load(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    actual={};curves={}
    for r in rows:
        d,arm,seed,e=r['domain'],r['arm_id'],int(r['seed']),int(r['episode'])
        if d not in DOMAINS or arm not in ARMS or seed not in SEEDS or e%16 or not 0<=e<=EPISODES[d]:raise ValueError('Unexpected curve identity or checkpoint')
        key=d,arm,seed,e
        if key in actual:raise ValueError('Duplicate curve record')
        x=int(r['environment_steps']);v=float(r['score_x100'])
        if x!=e*HORIZONS[d] or not np.isfinite(v):raise ValueError('Recorded step/score differs')
        actual[key]=v
    for d in DOMAINS:
        ep=np.arange(0,EPISODES[d]+1,16)
        for arm in ARMS:
            values=np.asarray([[actual[d,arm,seed,int(e)] for e in ep] for seed in SEEDS])
            curves[d,arm]=(ep,values)
    wanted=sum(len(range(0,EPISODES[d]+1,16))*len(ARMS)*len(SEEDS) for d in DOMAINS)
    if len(actual)!=wanted:raise ValueError('Dense40 curve cardinality differs')
    return curves

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,default=HERE/'data.csv');parser.add_argument('--output-dir',type=Path,default=HERE/'regenerated');parser.add_argument('--font-dir',type=Path,default=Path('/root/gamewar/latex_thesis_resource_allocation/assets/fonts/windows'));args=parser.parse_args()
    fonts=setup(args.font_dir);curves=load(args.data);args.output_dir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(15.6/2.54,3.25));fig.subplots_adjust(left=.105,right=.985,bottom=.25,top=.87,wspace=.38);evidence={}
    for d,ax,letter,title in zip(DOMAINS,axes,'ab',['通信功率分配','应急柴油分配']):
        for arm in ARMS:
            ep,values=curves[d,arm];x=ep*HORIZONS[d]/1000;mean=values.mean(0);sd=values.std(0,ddof=1);name,color,marker=STYLES[arm]
            ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.10,linewidth=0,zorder=1)
            marker_indices=list(range(0,len(x),max(1,len(x)//14)))
            if marker_indices[-1]!=len(x)-1:marker_indices.append(len(x)-1)
            ax.plot(x,mean,color=color,linestyle='-',linewidth=1.25,marker=marker,markevery=marker_indices,markersize=3.5,markerfacecolor=color,markeredgecolor=color,markeredgewidth=.75,label=name,zorder=3)
            evidence[d+'/'+arm]={'episodes':ep.tolist(),'environment_steps':(ep*HORIZONS[d]).tolist(),'mean':mean.tolist(),'sample_sd':sd.tolist(),'training_seeds':list(SEEDS),'observed_points_per_seed':len(ep),'interpolated_points':0}
        endpoint=EPISODES[d]*HORIZONS[d]/1000;ax.set_xlim(0,120);ax.set_xticks([0,30,60,90,120]);ax.get_xticklabels()[-1].set_horizontalalignment('right');ax.set_xlabel('训练步数（×10³）');ax.set_ylabel('平均图像质量得分' if d=='radio' else '平均需求完成比例（%）');ax.set_title(f'({letter}) {title}',loc='center',pad=9,fontfamily=['Times New Roman','SimSun'],fontsize=10.5);ax.tick_params(direction='out',length=3)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.012),ncol=4,frameon=False,handlelength=2.,columnspacing=1.3,handletextpad=.5)
    balance_legend_spacing(fig)
    outputs={}
    for ext in ['pdf','png','svg']:
        p=args.output_dir/('figure.'+ext);fig.savefig(p,dpi=240);outputs[p.name]=digest(p)
    plt.close(fig)
    manifest={'schema':'chapter2-registered40-dense150k-development-render-v1','status':'pass','data_sha256':digest(args.data),'plot_sha256':digest(__file__),'outputs_sha256':outputs,'fonts_sha256':fonts,'evidence':evidence,'training_or_policy_evaluation_performed':False,'smoothing':False,'markers_filled':True}
    (args.output_dir/'render_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'pass','outputs_sha256':outputs}))
if __name__=='__main__':main()
