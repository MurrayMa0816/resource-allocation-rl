"""C6 paper learning figure from nine sealed26-point DEV records only.

No model/environment/scientific runtime import. No TeX/index/build/publication
pointer operation. All generated data and figure assets remain on HY.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,os
from pathlib import Path
import numpy as np

from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
THESIS=next(p for p in HERE.parents if (p/'assets/fonts').is_dir())
REVISION=HERE/'generated'
SEEDS=(6400,6401,6402)
METHODS=('SeDVal','PASTA','COLA')
COLOURS={'SeDVal':'#154f80','PASTA':'#498f9d','COLA':'#9368a6'}
MARKERS={'SeDVal':'o','PASTA':'P','COLA':'D'}
WIDTH,HEIGHT=156.,81.


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_csv(path,rows):
    rows=list(rows)
    with Path(path).open('w',newline='',encoding='utf-8')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def generate(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    steps=np.arange(0,259201,10368)
    with (HERE/'learning_all3_seed_DEV.csv').open(newline='',encoding='utf-8') as stream:
        raw=list(csv.DictReader(stream))
    arrays={}
    for method in METHODS:
        points=[]
        for seed in SEEDS:
            records=[r for r in raw if r['method']==method and int(r['training_seed'])==seed]
            points.append([[float(next(r['weather_mean_percent'] for r in records if int(r['training_steps'])==step and r['metric']==metric)) for metric in ('C','JA')]for step in steps])
        arrays[method]=np.asarray(points)
    means={m:values.mean(0)for m,values in arrays.items()};sds={m:values.std(0,ddof=1)for m,values in arrays.items()}
    import shutil
    for name in ('learning_all3_seed_DEV.csv','learning_mean_sample_SD.csv','all7488_DEV_cases.csv'):shutil.copyfile(HERE/name,out/name)
    def bind(path):return Path(path)
    os.environ['MPLCONFIGDIR']=str(REVISION/'runtime_cache/paper_learning_matplotlib')
    os.environ['XDG_CACHE_HOME']=str(REVISION/'runtime_cache/paper_learning_xdg')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.text import Text
    fontroot=THESIS/'assets/fonts/windows'
    for name in('simsun.ttc','TIMES.TTF'):
        font_manager.fontManager.addfont(str(bind(fontroot/name)))
    plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':10.5,'axes.labelsize':10.5,'axes.titlesize':10.5,'xtick.labelsize':10.5,'ytick.labelsize':10.5,'legend.fontsize':10.5,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'c6-three-learning-comparison-v1','path.simplify':False,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(WIDTH/25.4,HEIGHT/25.4),dpi=200)
    axes=[fig.add_axes([17/WIDTH,22/HEIGHT,59.2/WIDTH,51/HEIGHT]),fig.add_axes([94/WIDTH,22/HEIGHT,59.2/WIDTH,51/HEIGHT])]
    styles=[]
    for k,(ax,title,ylabel)in enumerate(zip(axes,['(a) 灌溉区域产量达标率','(b) 按面积加权的平均相对产量'],['灌溉区域产量达标率\n（%）','按面积加权的平均相对产量\n（%）'])):
        for i,method in enumerate(METHODS):
            mean=means[method][:,k];sd=sds[method][:,k]
            assert np.min(mean-sd)>=0 and np.max(mean+sd)<=100
            ax.fill_between(steps/1000,mean-sd,mean+sd,color=COLOURS[method],alpha=.18,linewidth=0)
            markers=sorted(set(np.linspace(i%3,25,12,dtype=int).tolist()))
            line=ax.plot(steps/1000,mean,color=COLOURS[method],label=method,lw=1.25,ls='-',marker=MARKERS[method],ms=4.,mec=COLOURS[method],mfc=COLOURS[method],mew=.6,markevery=markers,clip_on=False)[0]
            styles.append(dict(axis=k,method=method,color=COLOURS[method],linestyle=line.get_linestyle(),linewidth_bp=line.get_linewidth(),marker=line.get_marker(),marker_size_bp=line.get_markersize(),marked_indices=markers,true_mean_vertices=len(line.get_xdata()),full_SD_band=True))
        ax.set(xlim=(0,259.2),ylim=(0,100));ax.set_xticks([0,50,100,150,200,250]);ax.set_yticks([0,20,40,60,80,100])
        ax.set_xlabel('训练步数（×10³）',labelpad=4.)
        ax.set_ylabel(ylabel,labelpad=4.)
        ax.set_title(title,loc='center',fontsize=10.5,fontweight='normal',pad=6.)
        ax.tick_params(axis='both',which='major',length=3.,width=.7,pad=3.,labelsize=10.5,direction='out')
        ax.grid(axis='y',alpha=.18,linewidth=.6)
    handles,labels=axes[0].get_legend_handles_labels()
    legend=fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,2/HEIGHT),ncol=3,frameon=False,handlelength=1.8,handletextpad=.5,columnspacing=1.25,labelspacing=.5,borderpad=0,borderaxespad=0)
    fig.canvas.draw();ren=fig.canvas.get_renderer();factor=72/fig.dpi
    text=[]
    for t in fig.findobj(Text):
        if not t.get_visible()or not t.get_text():continue
        bb=t.get_window_extent(ren);assert abs(t.get_fontsize()-10.5)<1e-8
        assert bb.x0>=-.5 and bb.y0>=-.5 and bb.x1<=fig.bbox.x1+.5 and bb.y1<=fig.bbox.y1+.5
        text.append(dict(text=t.get_text(),size_bp=t.get_fontsize(),bbox_bp=(np.array(bb.bounds)*factor).tolist()))
    title_checks=[]
    for ax in axes:
        box=ax.title.get_window_extent(ren);error=abs((box.x0+box.x1)/2-(ax.bbox.x0+ax.bbox.x1)/2);assert error<.5
        title_checks.append(dict(title=ax.get_title(),centre_error_pixels=float(error)))
    lb=legend.get_window_extent(ren);xlabel_bottom=min(ax.xaxis.label.get_window_extent(ren).y0 for ax in axes);gap=(xlabel_bottom-lb.y1)*factor
    assert gap>=3.
    geometry=dict(status='PASS',figure_mm=[WIDTH,HEIGHT],axes_mm=[59.2,51.],axes_positions_mm=[(np.asarray(ax.get_position().bounds)*[WIDTH,HEIGHT,WIDTH,HEIGHT]).tolist()for ax in axes],ordinary_text_bp=10.5,text=text,title_checks=title_checks,title_pad_bp=6.,axis_label_pad_bp=4.,xlabel_to_legend_bp=float(gap),legend_bottom_bp=float(lb.y0*factor),legend_labels=labels,legend_columns=3,legend_rows=1,styles=styles,all_data_lines_solid=True,all_data_linewidths_bp=[1.25],all26_real_vertices_preserved=True,no_smoothing=True,SD_full_unclipped=True)
    for ext in('pdf','svg','png'):
        kwargs={'dpi':260}if ext=='png'else{}
        if ext=='pdf':kwargs['metadata']=dict(CreationDate=None,ModDate=None,Creator='C6 sealed-DEV learning comparison plot')
        elif ext=='svg':kwargs['metadata']={'Date':None}
        fig.savefig(out/f'figure.{ext}',**kwargs)
    plt.close(fig)
    dump(out/'render_geometry.json',geometry)
    return geometry


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=REVISION/'figures/learning_comparison');args=parser.parse_args()
    generate(args.output_dir)


if __name__=='__main__':main()
