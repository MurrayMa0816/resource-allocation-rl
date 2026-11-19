"""Narrow display derivative: retain native plotting code; replace metric text only.
No environment, policy, model, tensor, optimizer, or training imports.
"""
from pathlib import Path
import argparse, json, hashlib, importlib.util, os, sys
ROOT=Path(__file__).resolve().parents[2]
SPECS=Path(__file__).with_name('specifications.json')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def run(number):
 s=next(s for s in json.loads(SPECS.read_text()) if s['number']==number)
 out=Path(s['asset']);checks=out/'checks';checks.mkdir(exist_ok=True)
 for k,v in [('MPLCONFIGDIR','matplotlib'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
  q=ROOT/'runtime'/out.name/v;q.mkdir(parents=True,exist_ok=True);os.environ[k]=str(q)
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 from matplotlib.figure import Figure
 from matplotlib.text import Text
 from matplotlib.lines import Line2D
 from matplotlib.collections import Collection
 from matplotlib.patches import Patch
 from PIL import Image
 def clean(x):
  if isinstance(x,np.ndarray):return dict(dtype=str(x.dtype),shape=list(x.shape),values=clean(x.tolist()))
  if isinstance(x,np.generic):return x.item()
  if isinstance(x,(list,tuple)):return [clean(v) for v in x]
  if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
  if isinstance(x,(float,int,str,bool)) or x is None:return x
  return str(x)
 def digest(d):return hashlib.sha256(json.dumps(clean(d),sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
 def snapshot(fig):
  records=[]
  for ax in fig.axes:
   group=dict(position=list(ax.get_position().bounds),xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),aspect=str(ax.get_aspect()),lines=[],patches=[],collections=[],images=[])
   for p in ax.findobj(Line2D):
    group['lines'].append(dict(x=clean(p.get_xdata()),y=clean(p.get_ydata()),color=p.get_color(),lw=p.get_linewidth(),ls=p.get_linestyle(),marker=p.get_marker(),ms=p.get_markersize(),mfc=p.get_markerfacecolor(),mec=p.get_markeredgecolor(),alpha=p.get_alpha(),visible=p.get_visible(),transform=p.get_transform().get_matrix().tolist()))
   for p in ax.findobj(Patch):
    group['patches'].append(dict(vertices=p.get_path().vertices.tolist(),codes=clean(p.get_path().codes),transform=p.get_transform().get_matrix().tolist(),fc=clean(p.get_facecolor()),ec=clean(p.get_edgecolor()),lw=p.get_linewidth(),alpha=p.get_alpha(),visible=p.get_visible()))
   for p in ax.findobj(Collection):
    paths=[dict(vertices=v.vertices.tolist(),codes=clean(v.codes)) for v in p.get_paths()]
    group['collections'].append(dict(paths=paths,offsets=clean(p.get_offsets()),transform=p.get_transform().get_matrix().tolist(),fc=clean(p.get_facecolors()),ec=clean(p.get_edgecolors()),lw=clean(p.get_linewidths()),alpha=p.get_alpha(),visible=p.get_visible(),array=clean(p.get_array()),clim=clean(p.get_clim())))
   for p in ax.images:group['images'].append(dict(array=clean(p.get_array()),extent=clean(p.get_extent()),clim=clean(p.get_clim())))
   records.append(group)
  return records
 old_save=Figure.savefig;done=set();proof={}
 def save(fig,path,*args,**kwargs):
  if id(fig) not in done:
   done.add(id(fig));fig.canvas.draw();before=snapshot(fig);before_digest=digest(before)
   # Re-render the unchanged native figure once to compare to the active parent.
   pngkw={k:v for k,v in kwargs.items() if k not in ('metadata','format')}
   old_save(fig,checks/'native_parent_reproduction.png',format='png',**pngkw)
   a=np.asarray(Image.open(checks/'native_parent_reproduction.png').convert('RGBA'));b=np.asarray(Image.open(Path(s['parent'])/'figure.png').convert('RGBA'))
   same=a.shape==b.shape and np.array_equal(a,b)
   dump(checks/'native_parent_reproduction.json',dict(shape_generated=list(a.shape),shape_parent=list(b.shape),pixels_equal=bool(same),parent_png_sha256=sha(Path(s['parent'])/'figure.png'),generated_sha256=sha(checks/'native_parent_reproduction.png')))
   if not same: raise AssertionError('Native parent pixels differ: '+number)
   replacements=[]
   for t in fig.findobj(Text):
    label=t.get_text()
    if label in s['mapping']:
     replacements.append(dict(old=label,new=s['mapping'][label],font_bp=t.get_fontsize(),position_before=list(t.get_position())))
     t.set_text(s['mapping'][label])
   assert len(replacements)==1,(number,replacements)
   fig.canvas.draw();after=snapshot(fig)
   assert digest(after)==before_digest, 'Non-text native artists changed'
   renderer=fig.canvas.get_renderer();fw,fh=fig.get_size_inches();texts=[]
   for t in fig.findobj(Text):
    if not t.get_visible() or not t.get_text():continue
    # Matplotlib locators retain unrendered out-of-range tick labels; skip them.
    hidden=False
    for ax in fig.axes:
     for axis,limits in ((ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())):
      for tick in axis.get_major_ticks():
       if t in (tick.label1,tick.label2) and not min(limits)-1e-9<=tick.get_loc()<=max(limits)+1e-9:hidden=True
    if hidden:continue
    b=t.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted());bb=list(b.bounds)
    texts.append(dict(text=t.get_text(),font_bp=t.get_fontsize(),weight=t.get_fontweight(),style=t.get_fontstyle(),family=t.get_fontfamily(),position=clean(t.get_position()),bbox_inches=bb))
   changed=[t for t in texts if t['text'] in s['mapping'].values()]
   assert all(t['font_bp']==10.5 and t['weight']=='normal' for t in changed)
   assert all(t['bbox_inches'][0]>=-0.002 and t['bbox_inches'][1]>=-0.002 and t['bbox_inches'][0]+t['bbox_inches'][2]<=fw+.002 and t['bbox_inches'][1]+t['bbox_inches'][3]<=fh+.002 for t in changed)
   intersections=[]
   for t in changed:
    x,y,w,h=t['bbox_inches']
    for v in texts:
     if v is t:continue
     a0,b0,c0,d0=v['bbox_inches']
     overlap=max(0,min(x+w,a0+c0)-max(x,a0))*max(0,min(y+h,b0+d0)-max(y,b0))
     if overlap>1e-8:intersections.append(dict(metric=t['text'],other=v['text'],overlap_square_inches=overlap))
   assert not intersections,(number,intersections)
   centers=[]
   for ax in fig.axes:
    if ax.get_title():
     b=ax.title.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted());center=(ax.get_position().x0+ax.get_position().width/2)*fw
     centers.append(dict(title=ax.get_title(),error_inches=float(b.x0+b.width/2-center)))
   assert all(abs(x['error_inches'])<1e-6 for x in centers)
   proof.update(schema='thesis-metric-label-display-derivative-v1',number=number,parent_native_reproduction_pixels_equal=True,label_replacements=replacements,nontext_before_sha256=before_digest,nontext_after_sha256=digest(after),nontext_equal=True,text=texts,changed_label_intersections=intersections,title_centers=centers,figure_inches=[float(fw),float(fh)],text_size_policy='10.5bp ordinary metric labels; original ID exception preserved',scientific_execution=dict(environment_steps=0,model_loads=0,policy_evaluations=0,training_runs=0))
   dump(checks/'metric_display_proof.json',proof)
  return old_save(fig,path,*args,**kwargs)
 Figure.savefig=save
 # The source is an exact byte copy of the currently active renderer.
 assert sha(out/'parent_renderer.py')==s['parent_renderer_sha256']
 sp=importlib.util.spec_from_file_location('metric_parent_renderer',out/'parent_renderer.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 if s['kind']=='c4':
  sys.argv=['parent_renderer.py','--output-dir',str(out)];m.main()
 elif s['kind']=='c5shared':
  m.ROOT=ROOT;m.render(s['role'],Path(s['data']),out)
 elif s['kind']=='c56':m.main()
 elif s['kind']=='c59':
  m.OUT=out;ds=m.source_data();mapping=m.build(ds)
  dump(checks/'original_data_geometry_check.json',mapping)
 Figure.savefig=old_save
 # Original renderer proofs are retained with explicit inherited names, since
 # their geometry was measured immediately before the narrow save-stage map.
 for name in ['presentation_proof.json','font_geometry.json','figure_geometry.json','source.json']:
  if (out/name).exists():(out/name).rename(out/('parent_'+name))
 manifest=dict(**s,schema='thesis-chinese-metric-figure-labels-v1',native_parent_renderer_byte_identical=True,changes=[dict(old=k,new=v)for k,v in s['mapping'].items()],nontext_equal=True,parent_native_reproduction_pixels_equal=True,outputs={f:sha(out/f)for f in ['figure.pdf','figure.svg','figure.png','checks/metric_display_proof.json']},parent_retained=True,scientific_execution=dict(environment_steps=0,model_loads=0,policy_evaluations=0,training_runs=0))
 dump(out/'source.json',manifest)
 (out/'README.rst').write_text('指标名称中文统一展示派生\n========================\n\n仅替换source.json列明的指标文字。当前父绘图源逐字节保留为parent_renderer.py，原数据只读。\nchecks/metric_display_proof.json证明替换前后非文字artist、曲线、点、坐标范围、误差条及配色完全相同；checks/native_parent_reproduction.json证明原生重绘与父PNG逐像素相同。\n普通指标文字保持10.5bp宋体/Times New Roman，原对象编号例外保持。复绘执行本目录plot.py；所有输出位于HY。\n',encoding='utf-8')
 (out/'plot.py').write_text('from pathlib import Path\nimport subprocess,sys\nsubprocess.run([sys.executable,"-B",'+repr(str(Path(__file__)))+','+repr(number)+'],check=True)\n')
 print(json.dumps(dict(status='PASS',number=number,output=str(out)),ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('number');args=p.parse_args();run(args.number)
