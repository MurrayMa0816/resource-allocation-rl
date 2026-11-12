"""Offline unified presentation of four accepted learning figures.
Only artist styles and positions change. No models, environments or science run.
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,importlib.util,json,os,shutil,sys
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent
T=Path('/root/gamewar/latex_thesis_resource_allocation')
PRINT=Path('/hy-tmp/resource_allocation_results/thesis_print_format_implementation_20261109_160917/figures')
MARKERS=('o','s','^','D','v','P')
WIDTH=156.; HEIGHT={"2.4":88.,"3.4":88.,"4.4":88.,"5.4":164.}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load_module(p,name):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def clean(v):
 import numpy as np
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,(tuple,list)):return[clean(x)for x in v]
 if isinstance(v,dict):return{k:clean(x)for k,x in v.items()}
 if isinstance(v,(int,float,str,bool))or v is None:return v
 return str(v)
def digest(v):return hashlib.sha256(json.dumps(clean(v),sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def numeric(fig):
 return [dict(xlim=clean(a.get_xlim()),ylim=clean(a.get_ylim()),xscale=a.get_xscale(),yscale=a.get_yscale(),xticks=clean(a.get_xticks()),yticks=clean(a.get_yticks()),
  lines=[dict(x=clean(l.get_xdata()),y=clean(l.get_ydata()),color=clean(l.get_color()),alpha=l.get_alpha())for l in a.lines],
  collections=[dict(paths=[dict(vertices=clean(p.vertices),codes=clean(p.codes))for p in c.get_paths()],offsets=clean(c.get_offsets()),array=clean(c.get_array()),fc=clean(c.get_facecolors()),ec=clean(c.get_edgecolors()),alpha=c.get_alpha())for c in a.collections])for a in fig.axes]

def copy_inputs(number,out,parent):
 inp=out/'inputs';inp.mkdir(exist_ok=True)
 mapping={}
 if number=='2.4':
  for f in parent.iterdir():
   if f.is_file() and f.name in ('plot.py','data.csv','presentation_layout.py','source.json'):
    (inp/f.name).write_bytes(f.read_bytes());mapping[f.name]=sha(f)
 else:
  for f in (parent/'inputs').rglob('*'):
   if f.is_file():
    rel=f.relative_to(parent/'inputs');q=inp/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(f.read_bytes());mapping[str(rel)]=sha(f)
  if number=='4.4':
   q=inp/'parent_plot.py';s=q.read_text();old="PRESENTATION_ROOT=Path('/hy-tmp/resource_allocation_results/thesis_print_format_implementation_20261109_160917/figures')";new=f"PRESENTATION_ROOT=Path({str(R)!r})";assert s.count(old)==1;q.write_text(s.replace(old,new))
   dump(out/'input_cache_root_adaptation.json',dict(original_sha256=mapping['parent_plot.py'],modified_sha256=sha(q),old=old,new=new,reverse_source_sha256=hashlib.sha256(q.read_text().replace(new,old).encode()).hexdigest(),scientific_equations_changed=False))
 dump(out/'input_original_sha256.json',mapping)
 return inp,mapping

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--number',required=True,choices=list(HEIGHT));parser.add_argument('--output-dir',type=Path);args=parser.parse_args();n=args.number
 records=json.loads((R/'before/active_figures.json').read_text());record=next(x for x in records if x['图号']==n)
 active=T/record['图目录'];frozen_before=json.loads((R/'before/manifest.json').read_text());parent=Path(frozen_before[str(active/'figure.pdf')]['resolved']).parent;out=(args.output_dir or R/'assets'/('figure'+n.replace('.','-'))).resolve();assert out.is_relative_to(R);out.mkdir(parents=True,exist_ok=True)
 if (out/'inputs').exists():
  inp=out/'inputs';mapping=json.loads((out/'input_original_sha256.json').read_text())
 else:inp,mapping=copy_inputs(n,out,parent)
 for filename,expected in mapping.items():
  target=inp/filename
  if n=='4.4'and filename=='parent_plot.py':
   adaptation=json.loads((out/'input_cache_root_adaptation.json').read_text());restored=target.read_text().replace(adaptation['new'],adaptation['old']);assert hashlib.sha256(restored.encode()).hexdigest()==expected
  else:assert sha(target)==expected,filename
 for key,leaf in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
  q=R/'cache'/n/leaf;q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.figure import Figure
 from matplotlib.lines import Line2D
 from matplotlib.text import Text
 from matplotlib.transforms import Bbox
 from matplotlib import font_manager
 fs=T/'assets/fonts/windows';files={p.name.lower():p for p in fs.iterdir()if p.is_file()}
 for key in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf','simsun.ttc']:
  if key in files:font_manager.fontManager.addfont(str(files[key]))
 plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':10.5,'axes.labelsize':10.5,'axes.titlesize':10.5,'xtick.labelsize':10.5,'ytick.labelsize':10.5,'legend.fontsize':10.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.bbox':None,'path.simplify':False})
 metadata={}
 if n!='2.4':metadata=json.loads((parent/'spec.json').read_text())['source_metadata']
 oldsave=Figure.savefig;styled=set();report={};geom={};native={}
 def style(fig):
  style.last_fig=fig
  for t in fig.findobj(Text):
   if t.get_text()in metadata.get('mapping',{}):t.set_text(metadata['mapping'][t.get_text()])
  fig.canvas.draw();before=numeric(fig);native.update(before=before)
  axes=list(fig.axes);assert len(axes)==(4 if n=='5.4'else 2)
  handles,labels=axes[0].get_legend_handles_labels();assert len(handles)==6,(n,labels)
  colors=[matplotlib.colors.to_hex(h.get_color()).lower()for h in handles]
  for l in list(fig.legends):l.remove()
  fig.set_layout_engine(None);fig.set_size_inches(WIDTH/25.4,HEIGHT[n]/25.4,forward=True);fig.set_dpi(200)
  line_styles=[]
  for i,ax in enumerate(axes):
   column=i%2;row=i//2
   ax.set_xticks(before[i]['xticks']);ax.set_yticks(before[i]['yticks']);ax.set_xlim(before[i]['xlim']);ax.set_ylim(before[i]['ylim'])
   x0=(17.0 if column==0 else 94.0);y0=(105.0 if row==0 and len(axes)==4 else 29.0)
   ax.set_position([x0/WIDTH,y0/HEIGHT[n],59.2/WIDTH,51.0/HEIGHT[n]])
   title=ax.get_title();ax.set_title('');ax.set_title(title,loc='center',pad=6.0,fontfamily=['Times New Roman','SimSun'],fontsize=10.5,fontweight='normal')
   ax.set_xlabel('训练步数（×10³）',labelpad=4.0);ax.yaxis.labelpad=4.0
   ax.tick_params(axis='both',which='major',length=3.,width=.7,pad=3.,labelsize=10.5,direction='out')
   for l in ax.lines:
    color=matplotlib.colors.to_hex(l.get_color()).lower()
    if len(np.atleast_1d(l.get_xdata()))<2:continue
    index=colors.index(color)if color in colors else None;l.set_linestyle('-');l.set_linewidth(1.25)
    ismain=index is not None and l.get_label()in labels and (l.get_alpha()is None or l.get_alpha()>=.3)
    if ismain:
     l.set_marker(MARKERS[index]);l.set_markersize(4.0);l.set_markeredgewidth(.6);l.set_markerfacecolor(l.get_color());l.set_markeredgecolor(l.get_color())
     length=len(l.get_xdata());start=index%3;indices=np.linspace(start,max(start,length-1),12,dtype=int).tolist();l.set_markevery(sorted(set(indices)))
    else:
     l.set_marker('None')
    line_styles.append(dict(axis=i,label=l.get_label(),color=color,linewidth=l.get_linewidth(),linestyle=l.get_linestyle(),marker=l.get_marker(),markersize=l.get_markersize(),markevery=clean(l.get_markevery()),alpha=l.get_alpha(),main_curve=ismain))
  # Build legend from styled primary methods; column-major ordering makes rows
  # follow the native method order rather than shuffle methods across figures.
  primary={l.get_label():l for l in axes[0].lines if l.get_label()in labels}
  hs=[primary[l]for l in labels];order=[0,3,1,4,2,5]
  leg=fig.legend([hs[i]for i in order],[labels[i]for i in order],loc='lower center',bbox_to_anchor=(.5,2.0/HEIGHT[n]),ncol=3,frameon=False,handlelength=1.8,handletextpad=.5,columnspacing=1.25,labelspacing=.5,borderpad=0,borderaxespad=0)
  for t in fig.findobj(Text):t.set_fontsize(10.5);t.set_fontfamily(['Times New Roman','SimSun']);t.set_fontweight('normal')
  fig.canvas.draw();after=numeric(fig);assert digest(before)==digest(after),(n,'numeric artists modified')
  native.update(after=after,before_sha256=digest(before),after_sha256=digest(after))
  ren=fig.canvas.get_renderer();k=72/fig.dpi
  def bb(o):return clean(np.asarray(o.get_window_extent(ren).bounds)*k)
  # Native locators may create invisible tick labels outside view interval.
  hidden=set()
  for a in axes:
   for axis,lim in [(a.xaxis,a.get_xlim()),(a.yaxis,a.get_ylim())]:
    for tick in axis.get_major_ticks():
     if not min(lim)-1e-8<=tick.get_loc()<=max(lim)+1e-8:hidden.update((id(tick.label1),id(tick.label2)))
  texts=[]
  for t in fig.findobj(Text):
   if t.get_visible()and t.get_text()and id(t)not in hidden:
    b=bb(t);texts.append(dict(text=t.get_text(),font_bp=t.get_fontsize(),bbox_bp=b))
    assert b[0]>=-.5 and b[1]>=-.5 and b[0]+b[2]<=WIDTH*72/25.4+.5 and b[1]+b[3]<=HEIGHT[n]*72/25.4+.5,(n,'text outside canvas',texts[-1])
  legendbbox=leg.get_window_extent(ren);xlabelbottom=min(ax.xaxis.label.get_window_extent(ren).y0 for ax in axes[-2:]);legendbottom=legendbbox.y0*k;legendtop=legendbbox.y1*k
  gap=(xlabelbottom-legendbbox.y1)*k
  assert gap>=3,(n,'legend overlaps xlabels',gap)
  geom.update(number=n,status='PASS',figure_width_mm=WIDTH,columns=2,font_size_bp=10.5,all_data_lines_solid=all(x['linestyle']=='-'for x in line_styles),all_data_linewidths_bp=sorted({x['linewidth']for x in line_styles}),all_text_inside_canvas=True,x_limits_unchanged=all(a['xlim']==b['xlim']for a,b in zip(before,after)),y_limits_unchanged=all(a['ylim']==b['ylim']for a,b in zip(before,after)),x_limits=[a['xlim']for a in after],figure_mm=[WIDTH,HEIGHT[n]],axes=[dict(position_mm=clean(np.array(ax.get_position().bounds)*np.array([WIDTH,HEIGHT[n],WIDTH,HEIGHT[n]])),title=ax.get_title(),xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),title_bbox_bp=bb(ax.title),xlabel_bbox_bp=bb(ax.xaxis.label),ylabel_bbox_bp=bb(ax.yaxis.label),xlim=clean(ax.get_xlim()),ylim=clean(ax.get_ylim()))for ax in axes],lines=line_styles,texts=texts,legend_bbox_bp=bb(leg),legend_labels=labels,legend_columns=3,legend_rows=2,xlabel_to_legend_bp=gap,legend_to_canvas_bottom_bp=legendbottom,title_pad_bp=6.0,axis_label_pad_bp=4.0,ordinary_font_bp=10.5)
  report.update(status='PASS',number=n,numeric_before_sha256=digest(before),numeric_after_sha256=digest(after),numeric_artists_unchanged=True,all_primary_linewidth_bp=1.25,all_primary_lines_solid=True,all_primary_markers_unique=len({x['marker']for x in line_styles if x['main_curve']and x['axis']==0})==6,ordinary_font_bp=10.5,figure_mm=[WIDTH,HEIGHT[n]],axes_mm=[59.2,51.0],xrange_changed=False,smoothing_changed=False,source_data_changed=False,environment_steps=0,model_loads=0,policy_inference=0,training_steps=0)
 def save(fig,path,*a,**kw):
  if id(fig)not in styled:style(fig);styled.add(id(fig))
  # Only the new asset is written, never any accepted historical asset.
  p=Path(path).resolve();assert p.is_relative_to(out)
  return oldsave(fig,path,*a,**kw)
 Figure.savefig=save
 sys.path.insert(0,str(inp));sys.modules.pop('presentation',None);sys.modules.pop('presentation_style',None)
 try:
  if n=='2.4':
   m=load_module(inp/'plot.py','native2_4');m.HERE=inp;sys.argv=['native','--output-dir',str(out)];m.main()
   # Native c2 emits PDF only. The same styled object was closed, but remains
   # referenced by style closure? Capture all formats via save wrapper below.
  elif n=='3.4':
   m=load_module(inp/'plot_source.py','native3_4');m.setup(R/'cache'/n,T);m.render(inp,out)
  elif n=='4.4':
   m=load_module(inp/'plot.py','native4_4');m.HERE=inp;m.PUBLICATION=R;m.ROOT=R
   par=m.load_parent();fig,pts,cfg=m.build_parent(par);x,val,points,expected=m.random_values(par);m.replace_only_props(fig,x,expected)
   m.save_fig(fig,out);par.plt.close(fig)
  else:
   m=load_module(inp/'parent_renderer.py','native5_4');m.ROOT=R;m.render('learning',inp/'data',out)
 finally:Figure.savefig=oldsave
 assert len(styled)==1,(n,'expected one offline figure')
 # c2 only requested PDF; regenerate remaining formats from the same saved fig.
 if n=='2.4':
  fig=style.last_fig
  for ext in ['png','svg']:oldsave(fig,out/('figure.'+ext),dpi=260,metadata={'Date':None}if ext=='svg'else None)
 if (out/'display_aggregates.csv.gz').exists():
  original=parent/'display_aggregates.csv.gz'
  assert gzip.decompress((out/'display_aggregates.csv.gz').read_bytes())==gzip.decompress(original.read_bytes())
  (out/'display_aggregates.csv.gz').write_bytes(original.read_bytes())
 dump(out/'numeric_artist_geometry.json',native);dump(out/'render_geometry.json',geom);dump(out/'style_proof.json',report)
 # Preserve original data paths with byte-identical independently staged inputs.
 for name in ['data.csv','data.csv.gz','current_display_points.csv','plotted_points.csv','figure_config.json']:
  if (inp/name).exists()and not(out/name).exists():(out/name).symlink_to(inp/name)
 for f in ['presentation_proof.json']:
  if(out/f).exists():(out/f).rename(out/('inherited_native_'+f))
 src=dict(schema='four-chapter-main-learning-style-v1',number=n,parent_directory=str(parent),parent_pdf_sha256=sha(parent/'figure.pdf'),input_original_sha256=mapping,renderer=str(Path(__file__).resolve()),renderer_sha256=sha(__file__),numeric_before_sha256=report['numeric_before_sha256'],numeric_after_sha256=report['numeric_after_sha256'],source_data_changed=False,changes=['equal 1.25bp solid method lines','distinct color and filled markers','shared physical panel size and legend spacing'],outputs={f'figure.{ext}':sha(out/f'figure.{ext}')for ext in ['pdf','png','svg']},environment_steps=0,model_loads=0,training_runs=0)
 dump(out/'source.json',src)
 (out/'plot.py').write_text('from pathlib import Path\nimport runpy,sys\nsys.argv=[str(Path(__file__)),"--number",'+repr(n)+']+sys.argv[1:]\nrunpy.run_path('+repr(str(R/'render_learning_figures.py'))+',run_name="__main__")\n')
 (out/'README.rst').write_text('四章学习曲线统一展示\n====================\n\n方法曲线统一1.25bp实线，颜色和六种实心标记区分。普通10.5bp宋体/Times New Roman；每行双子图、相同单panel物理大小，底部三列两行共享图例。\n原始曲线点、SD阴影、平滑口径、范围及全部CSV保持。numeric_artist_geometry.json逐项记录前后数值。\n复绘：python plot.py --output-dir 新HY子目录（位于本修订根下）。0环境/模型/训练。\n')
 print(json.dumps(dict(number=n,status='PASS',output=str(out),geometry=report),ensure_ascii=False))

if __name__=='__main__':main()
