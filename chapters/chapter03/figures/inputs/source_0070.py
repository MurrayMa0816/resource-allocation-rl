"""Chapter 3 offline ablation style: saved observations only."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,os,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent.parent
THESIS=next(p for p in HERE.parents if (p/'assets/fonts/windows').is_dir())
ROOT=HERE
PARENT=HERE/'replot_sources/native_inputs'
ARMS=('A_FULL','A_PACE','A_RANDOM_OBJECT_ALLOCATION')
LABELS=('ReJoP（本文）','去总量学习','随机对象分配')
COLORS=('#0072B2','#A85A00','#7A3E87')
MARKERS=('o','s','^')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);assert not (out/'figure.pdf').exists()
 baseline={p.name:sha(p) for p in PARENT.iterdir() if p.is_file()}
 (out/'data.csv.gz').write_bytes((PARENT/'data.csv.gz').read_bytes())
 for key,leaf in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
  q=out/'runtime'/leaf;q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
 import numpy as np
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib import font_manager
 from matplotlib.text import Text
 fonts=THESIS/'assets/fonts/windows'
 files={p.name.lower():p for p in fonts.iterdir()if p.is_file()}
 for name in ('times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf','simsun.ttc'):
  if name in files:font_manager.fontManager.addfont(str(files[name]))
 plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':10.5,'axes.labelsize':10.5,'axes.titlesize':10.5,'xtick.labelsize':10.5,'ytick.labelsize':10.5,'legend.fontsize':10.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.bbox':None,'path.simplify':False})
 with gzip.open(out/'data.csv.gz','rt',newline='')as f:raw=list(csv.DictReader(f))
 assert len(raw)==5940
 parent=json.loads((PARENT/'artists.json').read_text());fig=plt.figure(figsize=(156/25.4,88/25.4),dpi=200);panels=[];axes=[];grid=np.r_[np.arange(0,100608,512),100608].astype(int)
 for i,d in enumerate(('radio','v5')):
  ax=fig.add_axes([(17 if i==0 else 94)/156,29/88,59.2/156,51/88]);axes.append(ax);series=[]
  for j,a in enumerate(ARMS):
   matrix=[]
   for seed in range(5):
    values=sorted((int(r['training_step']),float(r['score']))for r in raw if r['domain']==d and r['arm']==a and int(r['seed'])==seed);assert [t for t,v in values]==grid.tolist();matrix.append([v for t,v in values])
   m=np.asarray(matrix);mean=m.mean(0);sd=m.std(0,ddof=1);old=parent['panels'][i]['series'][j]
   assert old['arm']==a;np.testing.assert_array_equal(old['mean'],mean);np.testing.assert_array_equal(old['sample_SD'],sd);np.testing.assert_array_equal(old['plotted_x'],grid/1000)
   ax.fill_between(grid/1000,mean-sd,mean+sd,color=COLORS[j],alpha=.10,edgecolor='none',linewidth=0)
   line,=ax.plot(grid/1000,mean,color=COLORS[j],linestyle='-',linewidth=1.25,label=LABELS[j],marker=MARKERS[j],markersize=4,markeredgewidth=.6,markerfacecolor=COLORS[j],markeredgecolor=COLORS[j],markevery=sorted(set(np.linspace(j%3,len(grid)-1,12,dtype=int).tolist())))
   series.append(dict(arm=a,label=LABELS[j],x=line.get_xdata().tolist(),mean=mean.tolist(),sample_SD=sd.tolist(),lower=(mean-sd).tolist(),upper=(mean+sd).tolist(),linewidth_bp=1.25,linestyle='-',marker=MARKERS[j],markersize_bp=4,markevery=line.get_markevery()))
  ax.set_xlim(parent['panels'][i]['xlim']);ax.set_ylim(parent['panels'][i]['ylim']);ax.set_xticks([0,25,50,75,100],['0','25','50','75','100']);ax.set_yticks([20,30,40,50]if i==0 else[50,60,70,80]);ax.set_xlabel('训练步数（×10³）',labelpad=4);ax.set_ylabel('平均图像质量得分'if i==0 else'平均需求完成比例（%）',labelpad=4);ax.set_title(parent['panels'][i]['title'],loc='center',pad=6,fontweight='normal');ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='both',which='major',length=3,width=.7,pad=3,labelsize=10.5,direction='out');ax.set_axisbelow(True)
  panels.append(dict(domain=d,axis_mm=[17 if i==0 else 94,29,59.2,51],xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),xticks=ax.get_xticks().tolist(),yticks=ax.get_yticks().tolist(),series=series))
 leg=fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,2/88),ncol=3,frameon=False,handlelength=1.8,handletextpad=.5,columnspacing=1.25,labelspacing=.5,borderpad=0,borderaxespad=0)
 fig.canvas.draw();ren=fig.canvas.get_renderer();k=72/fig.dpi;texts=[]
 for t in fig.findobj(Text):
  if not t.get_visible()or not t.get_text():continue
  b=t.get_window_extent(ren);assert b.x0>=-.5 and b.y0>=-.5 and b.x1<=fig.bbox.x1+.5 and b.y1<=fig.bbox.y1+.5,t.get_text();assert t.get_fontsize()==10.5;texts.append(dict(text=t.get_text(),font_bp=t.get_fontsize(),bbox_bp=[v*k for v in b.bounds]))
 lb=leg.get_window_extent(ren);gap=min(a.xaxis.label.get_window_extent(ren).y0 for a in axes)-lb.y1;assert gap*k>=3
 geometry=dict(status='PASS',size_mm=[156,88],axes_mm=[59.2,51],panels=panels,font_bp=10.5,legend_columns=3,legend_rows=1,legend_bbox_bp=[v*k for v in lb.bounds],xlabel_to_legend_bp=gap*k,legend_to_canvas_bottom_bp=lb.y0*k,title_pad_bp=6,label_pad_bp=4,tick_length_bp=3,tick_width_bp=.7,tick_pad_bp=3,texts=texts)
 dump(out/'artists.json',geometry)
 for ext in ('pdf','png','svg'):fig.savefig(out/('figure.'+ext),dpi=600 if ext=='png'else 300,metadata={'CreationDate':None,'ModDate':None}if ext=='pdf'else({'Date':None}if ext=='svg'else None))
 plt.close(fig)
 dump(out/'source.json',dict(schema='chapter3-unified-ablation-style-v1',parent_directory=str(PARENT),input_sha256=baseline,renderer=str(Path(__file__).resolve()),renderer_sha256=sha(__file__),data_sha256=sha(out/'data.csv.gz'),all_5940_seed_observations_preserved=True,means_sample_sd_and_ranges_exact=True,native_mm=[156,88],axes_mm=[59.2,51],line_width_bp=1.25,solid_lines=True,markers=list(MARKERS),legend_rows=1,new_environment_steps=0,model_loads=0,training_runs=0,outputs={f'figure.{ext}':sha(out/f'figure.{ext}')for ext in ('pdf','png','svg')}))
 (out/'plot.py').write_text('import runpy\nrunpy.run_path('+repr(str(Path(__file__).resolve()))+',run_name="__main__")\n')
 print(out/'figure.pdf')
if __name__=='__main__':main()
