"""Offline content-spacing derivatives for accepted Chapters 4/5 graphics.

Parents recreate saved scalar/vector artists and retain their own guards.  This
outer layer only moves existing layout objects, never a model or environment.
"""
from pathlib import Path
import argparse,hashlib,json,importlib.util,sys,os,shutil,io,copy
sys.dont_write_bytecode=True
ROOT=Path('/hy-tmp/resource_allocation_results/thesis_figure_spacing_unification_20261110_000617')
THESIS=Path('/root/gamewar/latex_thesis_resource_allocation')
MAINTAINED=Path('/root/gamewar/resource_allocation_rl/tools/paper_experiments/visualization/spacing_20261110/plot_chapter45.py')
MAIN=Path('/hy-tmp/resource_allocation_results/thesis_main_learning_style_20261109')
PRINT=Path('/hy-tmp/resource_allocation_results/thesis_print_format_implementation_20261109_160917/figures')
METRIC=Path('/hy-tmp/resource_allocation_results/thesis_metric_chinese_unification_20261109_131953')
NUMBERS=['4.4','4.5','4.6','4.7','4.8','4.9','5.4','5.5','5.6','5.7','5.8']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p,name,source=None):
 if source is None:
  s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 import types
 m=types.ModuleType(name);m.__file__=str(p);sys.modules[name]=m;exec(compile(source,str(p),'exec'),m.__dict__);return m

def main():
 ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--number',required=True,choices=NUMBERS);ap.add_argument('--output-dir',type=Path);a=ap.parse_args();n=a.number
 baseline=next(x for x in json.loads((ROOT/'audit/baseline_figures.json').read_text())if x['图号']==n);parent=Path(baseline['physical_directory']);out=(a.output_dir or ROOT/'assets'/('figure'+n.replace('.',''))).resolve();assert out.is_relative_to(ROOT)and not(out/'figure.pdf').exists();out.mkdir(parents=True,exist_ok=True)
 for key,leaf in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:q=out/'runtime'/leaf;q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
 # Staged copies are independent inputs; accepted parents remain immutable.
 inp=out/'accepted_inputs';inp.mkdir(exist_ok=True)
 for f in parent.iterdir():
  if f.is_file()and f.suffix in ('.py','.csv','.json','.gz'):
   shutil.copyfile(f,inp/f.name)
 input_manifest={str(f.relative_to(inp)):sha(f)for f in inp.rglob('*')if f.is_file()}
 import numpy as np
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.figure import Figure
 from matplotlib.text import Text
 from matplotlib import font_manager
 from PIL import Image
 import fitz
 fonts={p.name.lower():p for p in (THESIS/'assets/fonts/windows').iterdir()if p.is_file()}
 for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf','simsun.ttc','simhei.ttf']:font_manager.fontManager.addfont(str(fonts[name]))
 def plain(v):
  if isinstance(v,np.ndarray):return plain(v.tolist())
  if isinstance(v,np.generic):return v.item()
  if isinstance(v,(tuple,list)):return[plain(x)for x in v]
  if isinstance(v,dict):return{str(k):plain(x)for k,x in v.items()}
  if isinstance(v,(int,float,str,bool))or v is None:return v
  return str(v)
 def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
 def scientific(fig):
  result=[]
  for ax in fig.axes:
   patches=[]
   for p in ax.patches:
    d=dict(type=type(p).__name__,path=plain(p.get_path().vertices),codes=plain(p.get_path().codes),facecolor=plain(p.get_facecolor()),edgecolor=plain(p.get_edgecolor()),linewidth=p.get_linewidth(),alpha=p.get_alpha(),hatch=p.get_hatch())
    for key in ['xy','width','height','center','radius','angle']:
     fn=getattr(p,'get_'+key,None)
     if fn is not None:d[key]=plain(fn())
    patches.append(d)
   result.append(dict(xlim=plain(ax.get_xlim()),ylim=plain(ax.get_ylim()),xscale=ax.get_xscale(),yscale=ax.get_yscale(),xticks=plain(ax.get_xticks()),yticks=plain(ax.get_yticks()),xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),title=ax.get_title(),xticklabels=[t.get_text()for t in ax.get_xticklabels()],yticklabels=[t.get_text()for t in ax.get_yticklabels()],lines=[dict(x=plain(l.get_xdata()),y=plain(l.get_ydata()),color=plain(l.get_color()),linewidth=l.get_linewidth(),linestyle=l.get_linestyle(),marker=l.get_marker(),markersize=l.get_markersize(),mfc=plain(l.get_markerfacecolor()),mec=plain(l.get_markeredgecolor()),alpha=l.get_alpha(),markevery=plain(l.get_markevery()),zorder=l.get_zorder())for l in ax.lines],collections=[dict(paths=[dict(vertices=plain(p.vertices),codes=plain(p.codes))for p in c.get_paths()],offsets=plain(c.get_offsets()),array=plain(c.get_array()),clim=plain(c.get_clim()),fc=plain(c.get_facecolors()),ec=plain(c.get_edgecolors()),lw=plain(c.get_linewidths()),alpha=c.get_alpha())for c in ax.collections],patches=patches,images=[dict(array=plain(i.get_array()),extent=plain(i.get_extent()),clim=plain(i.get_clim()))for i in ax.images],texts=[dict(text=t.get_text(),position=plain(t.get_position()),font_size=t.get_fontsize(),weight=t.get_fontweight())for t in ax.texts]))
  return result
 original_save=Figure.savefig;done=set();proof={};values={};captured=[]
 def valid_text(fig):
  hidden=set()
  for ax in fig.axes:
   if not ax.axison:
    hidden.update(id(t)for t in [ax.title,ax.xaxis.label,ax.yaxis.label,ax.xaxis.offsetText,ax.yaxis.offsetText,*ax.get_xticklabels(),*ax.get_yticklabels()])
   for axis,lim in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
    for tick in axis.get_major_ticks():
     if not min(lim)-1e-9<=tick.get_loc()<=max(lim)+1e-9:hidden.update((id(tick.label1),id(tick.label2)))
  return[t for t in fig.findobj(Text)if t.get_visible()and t.get_text().strip()and id(t)not in hidden]
 def style(fig,save_kwargs):
  fig.canvas.draw();oldw,oldh=fig.get_size_inches()*25.4;before=scientific(fig);values['before']=before;before_positions=[np.asarray(ax.get_position().bounds)*np.asarray([oldw,oldh,oldw,oldh])for ax in fig.axes]
  # Independently reproduce the accepted PNG before touching visible layout.
  target=Image.open(parent/'figure.png').convert('RGBA');targetarr=np.asarray(target);candidates=[]
  first=save_kwargs.get('dpi',matplotlib.rcParams['savefig.dpi']);first=fig.dpi if first=='figure'else first
  for dpi in [first,fig.dpi,200,220,240,260,300,600]:
   if dpi not in candidates:candidates.append(dpi)
  reproduced=False
  for dpi in candidates:
   if 'bbox_inches'not in save_kwargs and(abs(int(fig.get_figwidth()*dpi)-target.width)>1 or abs(int(fig.get_figheight()*dpi)-target.height)>1):continue
   reproduction_kwargs={k:v for k,v in save_kwargs.items()if k not in ['dpi','format','metadata']}
   buf=io.BytesIO();original_save(fig,buf,format='png',dpi=dpi,**reproduction_kwargs);candidate=Image.open(buf).convert('RGBA');ca=np.asarray(candidate)
   if ca.shape==targetarr.shape and np.array_equal(ca,targetarr):
    (out/'accepted_parent_reproduction.png').write_bytes(buf.getvalue());reproduced=True;break
  assert reproduced,(n,'Accepted parent PNG differs',target.size,candidates)
  # Authorized 2026-11-11 physical restoration, after byte-exact parent reproduction.
  authorized_axes_before_mm=plain(before_positions)
  target_height=164. if len(fig.axes)==4 else 88.
  assert len(fig.axes)in[2,4]
  fig.set_size_inches(156./25.4,target_height/25.4,forward=True)
  for i,ax in enumerate(fig.axes):
   ax.set_xticks(before[i]['xticks']);ax.set_yticks(before[i]['yticks'])
   ax.set_xlim(before[i]['xlim']);ax.set_ylim(before[i]['ylim'])
   row=i//2;column=i%2
   x0=17. if column==0 else 94.
   y0=105. if row==0 and len(fig.axes)==4 else 29.
   ax.set_position([x0/156.,y0/target_height,59.2/156.,51./target_height])
  oldw,oldh=156.,target_height
  before_positions=[np.asarray(ax.get_position().bounds)*np.asarray([oldw,oldh,oldw,oldh])for ax in fig.axes]
  proof['authorized_axes_before_mm']=authorized_axes_before_mm
  proof['authorized_axes_target_mm']=[59.2,51.]
  proof['axes_resize_authorized']=True
  fig.set_layout_engine(None)
  for ax in fig.axes:
   if ax.get_title():ax.set_title(ax.get_title(),loc='center',pad=6)
   ax.xaxis.labelpad=4;ax.yaxis.labelpad=4;ax.tick_params(axis='both',which='major',pad=3)
  fig.canvas.draw()
  # Only existing shared legends are moved. In-axis legends retain their roles.
  legends=list(fig.legends);assert len(legends)<=1,(n,'multiple figure legends require review');legend=legends[0]if legends else None
  # Preserve fixed case-map geometry: only the whole canvas is cropped.
  case=n=='5.8'
  def norm(s):return ''.join(s.split()).replace('−','-')
  def pdf_measure():
   fig.canvas.draw()
   buf=io.BytesIO();original_save(fig,buf,format='pdf',metadata={'CreationDate':None,'ModDate':None});doc=fitz.open(stream=buf.getvalue(),filetype='pdf');page=doc[0];fig.canvas.draw();lines=[]
   for block in page.get_text('rawdict')['blocks']:
    for line in block.get('lines',[]):
     chars=[c for span in line['spans']for c in span.get('chars',[])if not c['c'].isspace()];text=''.join(c['c']for c in chars).replace('−','-');lines.append((text,chars))
   H=page.rect.height
   def box_for_artist(t):
    b=t.get_window_extent(fig.canvas.get_renderer());expected=fitz.Rect(b.x0*72/fig.dpi,H-b.y1*72/fig.dpi,b.x1*72/fig.dpi,H-b.y0*72/fig.dpi);parts=[norm(v)for v in t.get_text().split('\n')if norm(v)];found=[]
    for part in parts:
     candidates=[]
     for text,chars in lines:
      start=0
      while part and(start:=text.find(part,start))>=0:
       bb=fitz.Rect(chars[start]['bbox'])
       for c in chars[start:start+len(part)]:bb|=fitz.Rect(c['bbox'])
       dist=(bb.x0+bb.x1-expected.x0-expected.x1)**2+(bb.y0+bb.y1-expected.y0-expected.y1)**2;candidates.append((dist,bb));start+=len(part)
     if candidates:found.append(min(candidates,key=lambda x:x[0])[1])
    if not found:return expected,False
    b=fitz.Rect(found[0])
    for v in found[1:]:b|=v
    return b,True
   texts=valid_text(fig);boxes=[(t,*box_for_artist(t))for t in texts];allboxes=[b for t,b,exact in boxes];content=fitz.Rect(allboxes[0])
   for b in allboxes[1:]:content|=b
   # Include plot frames even when the corresponding side has no labels.
   for ax in fig.axes:
    if not ax.axison:
     for patch in ax.patches:
      if patch.get_visible():
       b=patch.get_window_extent();content|=fitz.Rect(b.x0*72/fig.dpi,H-b.y1*72/fig.dpi,b.x1*72/fig.dpi,H-b.y0*72/fig.dpi)
     continue
    b=ax.get_window_extent();content|=fitz.Rect(b.x0*72/fig.dpi,H-b.y1*72/fig.dpi,b.x1*72/fig.dpi,H-b.y0*72/fig.dpi)
   positions=[np.asarray(ax.get_position().bounds)*np.asarray([oldw,fig.get_figheight()*25.4,oldw,fig.get_figheight()*25.4])for ax in fig.axes]
   rows=[]
   for i,p in sorted(enumerate(positions),key=lambda x:-x[1][1]):
    if case:break
    match=next((v for v in rows if abs(v['y']-p[1])<2),None)
    if match is None:match=dict(y=p[1],indices=[]);rows.append(match)
    match['indices'].append(i)
   def coordinate_artists(ids):
    ts=[]
    for i in ids:
     ax=fig.axes[i]
     ts.extend([ax.xaxis.label]if ax.get_xlabel()else[t for t in ax.get_xticklabels()if t.get_visible()and t.get_text()])
    return ts
   row_metrics=[]
   for upper,lower in zip(rows,rows[1:]):
    ut=coordinate_artists(upper['indices']);lt=[fig.axes[i].title for i in lower['indices']if fig.axes[i].get_title()]
    if ut and lt:row_metrics.append(dict(upper=upper['indices'],lower=lower['indices'],gap=min(box_for_artist(t)[0].y0 for t in lt)-max(box_for_artist(t)[0].y1 for t in ut)))
   label_ids=rows[-1]['indices']if rows else range(len(fig.axes));coords=coordinate_artists(label_ids);coordinate_bottom=max(box_for_artist(t)[0].y1 for t in coords)if coords else None
   legboxes=[box_for_artist(t)[0]for t in legend.get_texts()]if legend else[];legbox=None
   if legboxes:
    legbox=fitz.Rect(legboxes[0])
    for b in legboxes[1:]:legbox|=b
   titleboxes=[box_for_artist(ax.title)[0]for ax in fig.axes if ax.get_title()]
   # Figure-level notes are preserved, including any below the shared legend.
   notes=[(t,box_for_artist(t)[0])for t in fig.texts if t.get_visible()and t.get_text()];footer=bool(legbox and any(b.y1>legbox.y1+.1 for t,b in notes))
   result=dict(width_bp=page.rect.width,height_bp=H,content_top_bp=content.y0,content_bottom_bp=content.y1,legend_box_bp=list(legbox)if legbox else None,xlabel_to_legend_bp=(legbox.y0-coordinate_bottom)if legbox and coordinate_bottom is not None else None,bottom_margin_mm=(H-content.y1)*25.4/72,top_margin_mm=content.y0*25.4/72,row_metrics=row_metrics,all_pdf_text_exact=all(exact for t,b,exact in boxes),unmatched_texts=[t.get_text()for t,b,e in boxes if not e],footer_below_legend=footer)
   doc.close();return result
  frames_initial=pdf_measure();height=oldh
  # Iterate physical translations; axes extents never change.
  for iteration in range(5):
   f=pdf_measure();positions=[np.asarray(ax.get_position().bounds)*np.asarray([oldw,height,oldw,height])for ax in fig.axes];row_shifts={i:0. for i in range(len(fig.axes))}
   if not case:
    for row in reversed(f['row_metrics']):
     delta=15-row['gap']
     # Every row above this boundary travels together.
     boundary=min(positions[i][1]for i in row['upper'])
     for i,p in enumerate(positions):
      if p[1]>=boundary-1e-7:row_shifts[i]+=delta
   legend_down=(15-f['xlabel_to_legend_bp'])if legend and f['xlabel_to_legend_bp']is not None and not case else 0.
   # Move rows and legend before recomputing the true crop boundaries.
   figtexts=[(t,np.asarray(t.get_position())*np.asarray([oldw,height]))for t in fig.texts]
   for i,(ax,p)in enumerate(zip(fig.axes,positions)):
    if row_shifts[i]:ax.set_position([p[0]/oldw,(p[1]+row_shifts[i]*25.4/72)/height,p[2]/oldw,p[3]/height])
   if legend and legend_down:
    an=legend.get_bbox_to_anchor();anchor=(an.x0*72/fig.dpi,an.y0*72/fig.dpi);legend.set_bbox_to_anchor((anchor[0]/(oldw*72/25.4),(anchor[1]-legend_down)/(height*72/25.4)),transform=fig.transFigure)
   fig.canvas.draw();f=pdf_measure();topcrop=f['content_top_bp']-1.5*72/25.4;bottomcrop=f['height_bp']-f['content_bottom_bp']-2*72/25.4
   if max(abs(topcrop),abs(bottomcrop),abs(legend_down),*map(abs,row_shifts.values()))<.001:break
   positions=[np.asarray(ax.get_position().bounds)*np.asarray([oldw,height,oldw,height])for ax in fig.axes];newh=(height*72/25.4-topcrop-bottomcrop)*25.4/72
   anchor=None
   if legend:
    an=legend.get_bbox_to_anchor();anchor=(an.x0*72/fig.dpi,an.y0*72/fig.dpi)
   figtextpos=[(t,np.asarray(t.get_position())*np.asarray([oldw,height]))for t in fig.texts]
   fig.set_size_inches(oldw/25.4,newh/25.4,forward=True)
   for ax,p in zip(fig.axes,positions):ax.set_position([p[0]/oldw,(p[1]-bottomcrop*25.4/72)/newh,p[2]/oldw,p[3]/newh])
   if legend:legend.set_bbox_to_anchor((anchor[0]/(oldw*72/25.4),(anchor[1]-bottomcrop)/(newh*72/25.4)),transform=fig.transFigure)
   for t,p in figtextpos:t.set_position((p[0]/oldw,(p[1]-bottomcrop*25.4/72)/newh))
   height=newh;fig.canvas.draw()
  frames_final=pdf_measure();assert abs(frames_final['top_margin_mm']-1.5)<.003,(n,frames_final);assert abs(frames_final['bottom_margin_mm']-2)<.003,(n,frames_final)
  if legend and not case:assert abs(frames_final['xlabel_to_legend_bp']-15)<.003,(n,frames_final)
  if not case:assert all(abs(x['gap']-15)<.003 for x in frames_final['row_metrics']),(n,frames_final)
  finalpositions=[np.asarray(ax.get_position().bounds)*np.asarray([oldw,height,oldw,height])for ax in fig.axes];assert all(np.allclose(p[2:],q[2:],rtol=0,atol=1e-7)for p,q in zip(before_positions,finalpositions)),(n,'Axes physical dimensions changed',before_positions,finalpositions)
  after=scientific(fig);assert before==after,(n,'Scientific artists or original line styles changed');values.update(after=after,before_sha256=digest(before),after_sha256=digest(after))
  ren=fig.canvas.get_renderer();texts=[]
  for t in valid_text(fig):
   b=t.get_window_extent(ren);assert b.x0>=-.7 and b.y0>=-.7 and b.x1<=fig.bbox.x1+.7 and b.y1<=fig.bbox.y1+.7,(n,t.get_text(),b.bounds,fig.bbox.bounds);texts.append(dict(text=t.get_text(),font_bp=t.get_fontsize(),bbox_bp=(np.asarray(b.bounds)*72/fig.dpi).tolist()))
  proof.update(status='PASS',schema='cross-chapter-content-spacing-v1',number=n,native_mm=[oldw,height],axes_positions_before_mm=plain(before_positions),axes_positions_after_mm=plain(finalpositions),axes_physical_dimensions_exact=True,scientific_artist_values_and_styles_exact=True,numeric_before_sha256=values['before_sha256'],numeric_after_sha256=values['after_sha256'],accepted_parent_PNG_pixel_exact=True,pdf_frames_initial=frames_initial,pdf_frames_final=frames_final,measurement_basis='native PDF character/text frames with nearest-artist matching; not raster ink',shared_legend=bool(legend),legend_labels=[t.get_text()for t in legend.get_texts()]if legend else[],xlabel_to_legend_bp=frames_final['xlabel_to_legend_bp'],legend_to_file_bottom_mm=(frames_final['height_bp']-frames_final['legend_box_bp'][3])*25.4/72 if legend else None,content_bottom_to_file_bottom_mm=frames_final['bottom_margin_mm'],top_content_to_file_top_mm=frames_final['top_margin_mm'],row_gaps_bp=[x['gap']for x in frames_final['row_metrics']],title_pad_bp=6,axis_label_pad_bp=4,major_tick_pad_bp=3,texts=texts,case_map_internal_geometry_retained=case,environment_steps=0,model_loads=0,policy_inference=0,training_runs=0)
  captured.append(fig)
 def save(fig,path,*args,**kwargs):
  final=not isinstance(path,(io.BytesIO,bytes))and Path(path).name in ['figure.pdf','figure.png','figure.svg']
  if final and id(fig)not in done:style(fig,kwargs);done.add(id(fig))
  if final:
   kwargs.pop('bbox_inches',None);kwargs.pop('pad_inches',None)
   e=Path(path).suffix
   if e=='.pdf':kwargs['metadata']={'CreationDate':None,'ModDate':None}
   elif e=='.svg':kwargs['metadata']={'Date':None}
   elif e=='.png':kwargs['dpi']=300
  return original_save(fig,path,*args,**kwargs)
 # Parent invocation adapters change only private cache/output locations.
 oldargv=sys.argv[:];Figure.savefig=save
 try:
  if n in ['4.4','5.4']:
   (ROOT/'before').mkdir(exist_ok=True)
   for name in ['active_figures.json','manifest.json']:
    f=ROOT/'before'/name
    if f.exists():assert f.read_bytes()==(MAIN/'before'/name).read_bytes()
    else:shutil.copyfile(MAIN/'before'/name,f)
   m=load(MAIN/'render_learning_figures.py','accepted_main_'+n.replace('.',''));m.R=ROOT;sys.argv=['parent','--number',n,'--output-dir',str(out)];m.main()
  elif n in ['4.7','5.6','5.7','5.8']:
   manifest=json.loads((PRINT/'manifest.json').read_text());oldnum={'5.6':'5.7','5.7':'5.8','5.8':'5.9'}.get(n,n);p=ROOT/'execution/print_manifest_ch45.json'
   if p.exists():assert json.loads(p.read_text())==manifest
   else:dump(p,manifest)
   # ROOT/manifest.json is shared only by the print dispatcher, copied exactly.
   mp=ROOT/'manifest.json'
   if mp.exists():assert json.loads(mp.read_text())==manifest
   else:dump(mp,manifest)
   m=load(PRINT/'render_print_figures.py','accepted_print_'+n.replace('.',''));m.ROOT=ROOT;sys.argv=['parent','--number',oldnum,'--output-dir',str(out)];m.main()
  elif n in ['4.8','5.5']:
   oldnum='5.4'if n=='5.5'else n;specs=json.loads((METRIC/'agents/figures_c45/specifications.json').read_text());s=next(v for v in specs if v['number']==oldnum);s=copy.deepcopy(s);s['asset']=str(out);sp=out/'accepted_metric_specification.json';dump(sp,[s])
   for f in parent.iterdir():
    if f.is_file()and f.suffix in ['.csv','.json','.py','.gz']:shutil.copyfile(f,out/f.name)
   m=load(METRIC/'agents/figures_c45/render_metric_labels.py','accepted_metric_'+n.replace('.',''));m.ROOT=ROOT;m.SPECS=sp;m.run(oldnum)
  else:
   src=(parent/'plot.py').read_text();adapt=src.replace('HERE=Path(__file__).resolve().parent',f'HERE=Path({str(parent)!r})').replace('ROOT=HERE.parents[1]',f'ROOT=Path({str(ROOT)!r})');assert adapt!=src
   adapted=out/'accepted_inputs/accepted_cache_adapter.py';adapted.write_text(adapt);sys.path.insert(0,str(parent));sys.modules.pop('presentation_style',None);m=load(adapted,'accepted_scalar_'+n.replace('.',''));sys.argv=['parent','--output-dir',str(out)];m.main()
 finally:Figure.savefig=original_save;sys.argv=oldargv
 assert len(captured)==1,(n,len(captured))
 for name in ['source.json','render_geometry.json','presentation_geometry.json','print_presentation_proof.json','render_receipt.json','style_proof.json']:
  f=out/name
  if f.exists():f.rename(out/('inherited_parent_'+name))
 scientific_files={}
 for f in parent.rglob('*'):
  if not f.is_file()or any(x in f.parts for x in ['runtime','__pycache__','cache','checks']):continue
  if f.suffix not in ['.csv','.gz','.npz','.npy']:continue
  rel=f.relative_to(parent);q=out/rel
  if q.exists()or q.is_symlink():
   if q.is_symlink():q.unlink()
   elif q.read_bytes()!=f.read_bytes()and q.suffix!='.gz':raise AssertionError((n,'Saved scientific output differs',rel))
  q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(f.read_bytes());scientific_files[str(rel)]=sha(q)
 external_data={};external_copies={};external_bindings=[]
 if n=='5.5':
  external_root=Path(json.loads((parent/'source.json').read_text())['data'])
  for f in external_root.rglob('*'):
   if not f.is_file()or f.suffix not in ['.csv','.gz','.npy','.npz','.json']:continue
   q=out/'retained_external_scientific_data'/f.relative_to(external_root);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(f.read_bytes());external_data[str(f)]=sha(f);external_copies[str(q.relative_to(out))]=sha(q);external_bindings.append(dict(parent_absolute=str(f),new_relative=str(q.relative_to(out)),sha256=sha(f)))
 dump(out/'render_geometry.json',proof);dump(out/'numeric_artist_values_before_after.json',values)
 (out/'plot.py').write_text('import runpy,sys\nsys.argv=[__file__,"--number",'+repr(n)+']+sys.argv[1:]\nrunpy.run_path('+repr(str(MAINTAINED))+',run_name="__main__")\n')
 dump(out/'source.json',dict(schema='chapters45-content-spacing-derivative-v1',number=n,parent_directory=str(parent),parent_pdf_sha256=sha(parent/'figure.pdf'),parent_source_sha256=sha(parent/'source.json'),renderer=str(MAINTAINED),renderer_sha256=sha(MAINTAINED),scientific_data_sha256=scientific_files,external_scientific_data_sha256=external_data,retained_external_scientific_copies=external_copies,external_data_files=external_bindings,input_copies_sha256=input_manifest,scientific_values_and_styles_preserved=True,axes_physical_resize_authorized=True,axes_target_mm=[59.2,51.],pdf_character_frame_spacing=proof['pdf_frames_final'],output_sha256={f'figure.{e}':sha(out/f'figure.{e}')for e in ['pdf','png','svg']},environment_steps=0,model_loads=0,training_runs=0))
 dump(out/'render_receipt.json',dict(status='PASS',number=n,outputs_sha256={name:sha(out/name)for name in ['figure.pdf','figure.png','figure.svg','render_geometry.json','numeric_artist_values_before_after.json','source.json','plot.py']},scientific_values_exact=True,axes_physical_dimensions_exact=True,environment_steps=0,model_loads=0,training_runs=0));print(json.dumps(dict(status='PASS',number=n,native_mm=proof['native_mm'],shared_legend=proof['shared_legend'],gap_bp=proof['xlabel_to_legend_bp']),ensure_ascii=False))
if __name__=='__main__':main()
