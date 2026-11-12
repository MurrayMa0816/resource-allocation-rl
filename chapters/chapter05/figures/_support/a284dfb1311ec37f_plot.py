from pathlib import Path
import os, json, csv, hashlib

ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT/'cache/matplotlib')
os.environ['XDG_CACHE_HOME'] = str(ROOT/'cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager, cbook
from matplotlib.patches import Patch

source=json.loads((ROOT/'source_binding.json').read_text())
csv_path=Path(source['csv'])
assert hashlib.sha256(csv_path.read_bytes()).hexdigest()==source['csv_sha256']
rows=list(csv.DictReader(csv_path.open()))
summary=json.loads(Path(source['summary']).read_text())['means']
fonts=Path('/root/gamewar/latex_thesis_resource_allocation/assets/fonts/windows')
font_paths={p.name.lower():p for p in fonts.iterdir() if p.is_file()}
for name in ['times.ttf','timesi.ttf','simsun.ttc']:
    font_manager.fontManager.addfont(str(font_paths[name]))
plt.rcParams.update({
    'font.family':['Times New Roman','SimSun'], 'font.size':10.5,
    'mathtext.fontset':'stix','axes.unicode_minus':False,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.8,
    'xtick.direction':'out','ytick.direction':'out',
})
width,height=156.,85.
fig=plt.figure(figsize=(width/25.4,height/25.4))
styles=[('full','#1f5a8d',-.17),('scalar_value','#df8539',.17)]
data=[]
for panel,(domain,title) in enumerate([('radio','(a) 通信功率分配'),('v5','(b) 应急柴油分配')]):
    ax=fig.add_axes([(18+75*panel)/width,23/height,59.2/width,51/height])
    ax.set_yscale('log')
    ax.set_ylim(.001,2)
    ax.set_xlim(-.55,2.55)
    ax.set_yticks([.001,.01,.1,1])
    ax.minorticks_off()
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#dddddd',linewidth=.5)
    ax.set_axisbelow(True)
    ax.set_title(title,pad=5)
    for arm,color,offset in styles:
        selected=sorted([r for r in rows if r['domain']==domain and r['arm']==arm],key=lambda r:int(r['seed']))
        assert [int(r['seed']) for r in selected]==list(range(5))
        for i,metric in enumerate(['C','J','F']):
            field='all_active_steps__mse_'+metric
            values=np.array([float(r[field]) for r in selected])
            mean,sd=float(values.mean()),float(values.std(ddof=1))
            assert (values>.001).all() and (values<2).all()
            assert abs(mean-summary[domain][arm][field]['mean'])<1e-14
            assert abs(sd-summary[domain][arm][field]['sd'])<1e-14
            # Quartiles and Tukey fences are computed in the original MSE scale.
            stats=cbook.boxplot_stats(values,whis=1.5)[0]
            pos=i+offset
            ax.bxp([stats],positions=[pos],widths=.26,showfliers=False,patch_artist=True,manage_ticks=False,
                   boxprops={'facecolor':color+'55','edgecolor':color,'linewidth':1.1},
                   medianprops={'color':color,'linewidth':1.8},
                   whiskerprops={'color':color,'linewidth':.9},
                   capprops={'color':color,'linewidth':.9})
            # All five observations, including Tukey fliers, remain visible as small solid ticks.
            x=pos+np.linspace(-.045,.045,5)
            ax.plot(x,values,linestyle='none',marker='_',markersize=3.2,markeredgewidth=.85,color='#222222',alpha=.85,zorder=5)
            data.append({'domain':domain,'arm':arm,'metric':metric,'seeds':list(range(5)),'values':values.tolist(),
                         'arithmetic_mean_not_box_center':mean,'sample_SD_not_plotted':sd,
                         'q1':float(stats['q1']),'median':float(stats['med']),'q3':float(stats['q3']),
                         'IQR':float(stats['iqr']),'whisker_low':float(stats['whislo']),'whisker_high':float(stats['whishi']),
                         'fliers_all_shown_as_raw_ticks':stats['fliers'].tolist(),
                         'raw_tick_x':x.tolist(),'quartiles_calculated_before_axis_log_transform':True})
    ax.set_xticks([0,1,2],['覆盖预测\n$C$','成效预测\n$J$','加权预测\n$F_{0.8}$'],fontfamily='SimSun')
fig.text(1.5/width,48.5/height,'预测均方误差（越低越好）',rotation=90,va='center',ha='left')
fig.legend(handles=[Patch(facecolor=c+'55',edgecolor=c,label=l) for (_,c,_),l in zip(styles,['SeDVal（分项监督）','标量监督'])],loc='lower center',bbox_to_anchor=(.5,1.8/height),ncol=2,frameon=False,handlelength=1.25,columnspacing=1.8,borderaxespad=0)
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
for text in fig.findobj(matplotlib.text.Text):
    if text.get_visible() and text.get_text():
        box=text.get_window_extent(renderer)
        assert box.x0>=-.8 and box.y0>=-.8 and box.x1<=fig.bbox.x1+.8 and box.y1<=fig.bbox.y1+.8,(text.get_text(),box.bounds)
for ext in ['pdf','png','svg']:
    fig.savefig(ROOT/('value_supervision_boxplot.'+ext),dpi=400,metadata={'Creator':'Offline boxplot of existing Chapter5 DEV metrics'} if ext=='pdf' else None)
(ROOT/'plotted_values.json').write_text(json.dumps({'status':'PASS','native_mm':[width,height],'axes_mm':[59.2,51],'yscale':'log','same_y_limits':[.001,2],
    'boxes':12,'raw_seed_ticks':60,'box_statistic':'Original-scale Q1/median/Q3; Tukey1.5IQR whiskers',
    'all_original_observations_including_fliers_retained':True,'new_scientific_execution':0,'data':data},ensure_ascii=False,indent=2)+'\n')
print(ROOT/'value_supervision_boxplot.pdf')
