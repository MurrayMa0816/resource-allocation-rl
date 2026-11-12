"""Chapter 6 native spacing from accepted saved observations only.

Preserves all numerical artists, physical axes sizes, maps, dates and labels.
No policy/model/environment or training code is imported or executed.
"""
from pathlib import Path

import sys
_FIGURE_SHARED = next(p for p in Path(__file__).resolve().parents if p.name == "figures") / "shared"
if str(_FIGURE_SHARED) not in sys.path:
    sys.path.insert(0, str(_FIGURE_SHARED))
from figure_paths import configure_fonts, fonts_root
SHARED_DATA = _FIGURE_SHARED / "data"
import argparse,hashlib,importlib.util,io,json,os,shutil,sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
ROOT=HERE
MAINTAINED=HERE/'plot.py'
STYLE=HERE/'source/accepted_statistical.py'
LEARNING=HERE/'source/accepted_learning.py'
PARENTS={key:HERE for key in ['6.3']}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def module(p,name):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def main():
 ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--number',choices=PARENTS,required=True);ap.add_argument('--output-dir',type=Path);a=ap.parse_args();n=a.number;parent=PARENTS[n];out=(a.output_dir or ROOT/'assets'/('figure'+n.replace('.',''))).resolve();assert out.is_relative_to(ROOT)and not(out/'figure.pdf').exists();out.mkdir(parents=True,exist_ok=True)
 protected={str(p):sha(p)for p in parent.rglob('*')if p.is_file()and not any(x in p.parts for x in ['runtime_cache','__pycache__'])}
 work=out/'accepted_stage';work.mkdir();(work/'audit').mkdir()
 for key,leaf in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
  q=out/'runtime'/leaf;q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
 import numpy as np
 import matplotlib;matplotlib.use('Agg')
 from matplotlib.figure import Figure
 from matplotlib.text import Text
 import fitz
 original_save=Figure.savefig;styled=set();proof={};values={}
 def plain(v):
  if isinstance(v,np.ndarray):return plain(v.tolist())
  if isinstance(v,np.generic):return v.item()
  if isinstance(v,(list,tuple)):return[plain(x)for x in v]
  if isinstance(v,dict):return{k:plain(x)for k,x in v.items()}
  return v
 def numeric(fig):
  result=[]
  for ax in fig.axes:
   result.append(dict(xlim=plain(ax.get_xlim()),ylim=plain(ax.get_ylim()),xticks=plain(ax.get_xticks()),yticks=plain(ax.get_yticks()),xscale=ax.get_xscale(),yscale=ax.get_yscale(),lines=[dict(x=plain(l.get_xdata()),y=plain(l.get_ydata()),color=plain(l.get_color()),alpha=l.get_alpha(),linewidth=l.get_linewidth(),linestyle=l.get_linestyle(),marker=l.get_marker(),markersize=l.get_markersize(),markevery=plain(l.get_markevery()))for l in ax.lines],collections=[dict(paths=[dict(vertices=plain(p.vertices),codes=plain(p.codes))for p in c.get_paths()],offsets=plain(c.get_offsets()),array=plain(c.get_array()),facecolors=plain(c.get_facecolors()),edgecolors=plain(c.get_edgecolors()),linewidths=plain(c.get_linewidths()),alpha=c.get_alpha())for c in ax.collections],patches=[dict(vertices=plain(p.get_path().vertices),codes=plain(p.get_path().codes),facecolor=plain(p.get_facecolor()),edgecolor=plain(p.get_edgecolor()),linewidth=p.get_linewidth(),alpha=p.get_alpha())for p in ax.patches],annotations=[dict(text=t.get_text(),position=plain(t.get_position()),xy=plain(getattr(t,'xy',None)),font=t.get_fontsize(),rotation=t.get_rotation())for t in ax.texts]))
  return result
 def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
 def style(fig):
  fig.canvas.draw();before=numeric(fig);values['before']=before
  width,height=fig.get_size_inches()*25.4;old_width=float(width);old_sizes=[(np.asarray(ax.get_position().bounds)*[width,height,width,height])[2:].tolist()for ax in fig.axes]
  assert len(fig.axes)==(1 if n=='6.7'else 2)
  assert len(fig.legends)==({'6.3':0,'6.5':1,'6.7':0,'6.8':2}[n])
  for i,ax in enumerate(fig.axes):
   if ax.get_title():ax.set_title(ax.get_title(),pad=20 if n=='6.3'and i==0 else 6)
   ax.xaxis.labelpad=4;ax.yaxis.labelpad=4;ax.tick_params(axis='both',which='major',pad=3)
  fig.set_layout_engine(None);fig.canvas.draw()
  def normalize(s):return ''.join(s.split())
  def frames():
   buf=io.BytesIO();original_save(fig,buf,format='pdf',metadata={'CreationDate':None,'ModDate':None});doc=fitz.open(stream=buf.getvalue(),filetype='pdf');page=doc[0];rows=[]
   for b in page.get_text('dict')['blocks']:
    for l in b.get('lines',[]):rows.append((''.join(s['text']for s in l['spans']),fitz.Rect(l['bbox'])))
   def boxes(labels):
    wanted={normalize(x)for x in labels if x};found=[b for s,b in rows if normalize(s)in wanted];assert found,(n,wanted,[s for s,b in rows]);return found
   xlabel=boxes([ax.get_xlabel()for ax in fig.axes if ax.get_xlabel()])
   titles=boxes([ax.get_title()for ax in fig.axes if ax.get_title()])if any(ax.get_title()for ax in fig.axes)else []
   legends=[boxes([t.get_text()for t in leg.get_texts()])for leg in fig.legends]
   all_boxes=[b for s,b in rows if s.strip()]
   # Axes spine + its declared outward tick length is the visible lower edge
   # when the upper panel deliberately has neither xlabel nor tick labels.
   upper_visible_bottom=None
   if n in ['6.3','6.8']:
    ax=fig.axes[0];tick=max(t.tick1line.get_markersize()for t in ax.xaxis.get_major_ticks()if t.tick1line.get_visible());upper_visible_bottom=page.rect.height-ax.bbox.y0*72/fig.dpi+tick
   row_gap=None;top_legend_gap=None;legend_lower_to_title=None
   if n=='6.3':row_gap=min(b.y0 for b in boxes([fig.axes[1].get_title()]))-upper_visible_bottom
   if n=='6.8':
    top_legend_gap=min(b.y0 for b in legends[0])-upper_visible_bottom
    legend_lower_to_title=min(b.y0 for b in boxes([fig.axes[1].get_title()]))-max(b.y1 for b in legends[0])
   bottom_leg=legends[-1]if legends else None
   gap=min(b.y0 for b in bottom_leg)-max(b.y1 for b in xlabel)if bottom_leg else None
   # No-legend Figure6.7's top is defined by all visible labels plus axes box;
   # this retains its budget annotations and avoids adding a fake title.
   top=min(b.y0 for b in titles)if titles else min(min(b.y0 for b in all_boxes),page.rect.height-max(ax.bbox.y1 for ax in fig.axes)*72/fig.dpi)
   bottom=max(b.y1 for b in bottom_leg)if bottom_leg else max(b.y1 for b in xlabel)
   result=dict(native_mm=[page.rect.width*25.4/72,page.rect.height*25.4/72],xlabel_to_legend_bp=gap,bottom_content_margin_mm=(page.rect.height-bottom)*25.4/72,top_content_margin_mm=top*25.4/72,upper_axis_including_ticks_to_lower_title_bp=row_gap,upper_axis_including_ticks_to_legend_bp=top_legend_gap,upper_legend_to_lower_title_bp=legend_lower_to_title,xlabel_frames_bp=[list(b)for b in xlabel],legend_frames_bp=[[list(b)for b in boxes_]for boxes_ in legends],title_frames_bp=[list(b)for b in titles])
   doc.close();return result
  def positions():return[np.asarray(ax.get_position().bounds)*[width,height,width,height]for ax in fig.axes]
  def move_axes(ids,up_bp):
   pos=positions()
   for i in ids:
    p=pos[i];fig.axes[i].set_position([p[0]/width,(p[1]+up_bp*25.4/72)/height,p[2]/width,p[3]/height])
  def move_legend(index,down_bp):
   leg=fig.legends[index];anchor=leg.get_bbox_to_anchor();x=anchor.x0/fig.bbox.width;y=(anchor.y0-down_bp*fig.dpi/72)/fig.bbox.height;leg.set_bbox_to_anchor((x,y),transform=fig.transFigure)
  initial=frames()
  for iteration in range(6):
   f=frames()
   if n=='6.3':move_axes([0],12-f['upper_axis_including_ticks_to_lower_title_bp'])
   if n=='6.8':
    # Move top panel and its legend together to establish legend->next title,
    # then correct top axes separately to establish ticks->legend.
    delta=12-f['upper_legend_to_lower_title_bp'];move_axes([0],delta);move_legend(0,-delta);fig.canvas.draw();f=frames();move_axes([0],8-f['upper_axis_including_ticks_to_legend_bp'])
   fig.canvas.draw();f=frames()
   if fig.legends:move_legend(len(fig.legends)-1,8-f['xlabel_to_legend_bp'])
   fig.canvas.draw();f=frames();top_crop=(f['top_content_margin_mm']-1.5)*72/25.4;bottom_crop=(f['bottom_content_margin_mm']-2)*72/25.4
   pos=positions();anchors=[(leg.get_bbox_to_anchor().x0/fig.bbox.width,leg.get_bbox_to_anchor().y0*72/fig.dpi)for leg in fig.legends];new_height=height-(top_crop+bottom_crop)*25.4/72;fig.set_size_inches(width/25.4,new_height/25.4,forward=True)
   for ax,p in zip(fig.axes,pos):ax.set_position([p[0]/width,(p[1]-bottom_crop*25.4/72)/new_height,p[2]/width,p[3]/new_height])
   for leg,(x,y)in zip(fig.legends,anchors):leg.set_bbox_to_anchor((x,(y-bottom_crop)/(new_height*72/25.4)),transform=fig.transFigure)
   height=new_height;fig.canvas.draw();final=frames()
   residual=[abs(final['bottom_content_margin_mm']-2),abs(final['top_content_margin_mm']-1.5)]
   if fig.legends:residual.append(abs(final['xlabel_to_legend_bp']-8))
   if n=='6.3':residual.append(abs(final['upper_axis_including_ticks_to_lower_title_bp']-12))
   if n=='6.8':residual.extend([abs(final['upper_axis_including_ticks_to_legend_bp']-8),abs(final['upper_legend_to_lower_title_bp']-12)])
   if max(residual)<.002:break
  assert max(residual)<.002,(n,final,residual)
  after=numeric(fig);assert before==after,('Scientific artists changed',n);new_sizes=[(np.asarray(ax.get_position().bounds)*[width,height,width,height])[2:].tolist()for ax in fig.axes];assert np.allclose(old_sizes,new_sizes,rtol=0,atol=1e-8);assert abs(width-old_width)<1e-8
  values.update(after=after,before_sha256=digest(before),after_sha256=digest(after));fig.canvas.draw();renderer=fig.canvas.get_renderer();text=[]
  for t in fig.findobj(Text):
   if not t.get_visible()or not t.get_text():continue
   b=t.get_window_extent(renderer);assert abs(t.get_fontsize()-10.5)<1e-8,(n,t.get_text(),t.get_fontsize())
   # Visible locator tick labels beyond the viewport are intentionally skipped.
   if b.x0<-.6 or b.y0<-.6 or b.x1>fig.bbox.x1+.6 or b.y1>fig.bbox.y1+.6:raise AssertionError((n,t.get_text(),'outside new canvas',b.bounds,fig.bbox.bounds))
   text.append(dict(text=t.get_text(),font_bp=t.get_fontsize(),bbox_bp=plain(np.asarray(b.bounds)*72/fig.dpi)))
  proof.update(schema='chapter6-measured-content-spacing-v1',status='PASS',number=n,native_mm=[float(width),float(height)],axes_sizes_mm=new_sizes,axes_positions_mm=[plain(np.asarray(ax.get_position().bounds)*[width,height,width,height])for ax in fig.axes],ordinary_font_bp=10.5,measurement_basis='native PDF text frames; upper unlabeled axes use spine plus declared tick length, not pixel ink',pdf_frames_before=initial,pdf_frames_after=final,legend_count=len(fig.legends),xlabel_to_legend_bp=final['xlabel_to_legend_bp'],legend_to_file_bottom_mm=final['bottom_content_margin_mm']if fig.legends else None,bottom_content_margin_mm=final['bottom_content_margin_mm'],top_content_margin_mm=final['top_content_margin_mm'],upper_axis_including_ticks_to_lower_title_bp=final['upper_axis_including_ticks_to_lower_title_bp'],upper_axis_including_ticks_to_legend_bp=final['upper_axis_including_ticks_to_legend_bp'],upper_legend_to_lower_title_bp=final['upper_legend_to_lower_title_bp'],all_text_inside_canvas=True,all_scientific_artist_values_and_line_styles_exact=True,numeric_before_sha256=digest(before),numeric_after_sha256=digest(after),texts=text,exceptions=['6.3 top titlepad20bp clears the preserved growth-stage name row']if n=='6.3'else[],environment_steps=0,model_loads=0,policy_inference=0,training_runs=0)
 def save(fig,destination,*args,**kwargs):
  if isinstance(destination,(str,os.PathLike))and Path(destination).name in ['figure.pdf','figure.png','figure.svg']:
   if id(fig)not in styled:style(fig);styled.add(id(fig))
   kwargs.pop('bbox_inches',None);kwargs.pop('pad_inches',None)
   if str(destination).endswith('.pdf'):kwargs['metadata']={'CreationDate':None,'ModDate':None}
   if str(destination).endswith('.svg'):kwargs['metadata']={'Date':None}
  return original_save(fig,destination,*args,**kwargs)
 Figure.savefig=save
 try:
  if n in ['6.3','6.8']:
   m=module(STYLE,'accepted_c6_style');m.ROOT=work;plt,m.FONT=m.setup();record=m.overview(plt)if n=='6.3'else m.feedback(plt);stage=work/'figures'/('overview'if n=='6.3'else'feedback')
  elif n=='6.5':
   m=module(LEARNING,'accepted_c6_learning');m.REVISION=work;m.generate(work/'figure');stage=work/'figure'
  else:
   m=module(HERE/'source/accepted_compact.py','accepted_c6_compact_budget');m.HERE=work;m.main();stage=work
 finally:Figure.savefig=original_save
 assert len(styled)==1
 for ext in ['pdf','png','svg']:shutil.copyfile(stage/('figure.'+ext),out/('figure.'+ext))
 data={}
 for p in parent.iterdir():
  if p.is_file()and p.suffix=='.csv':
   q=out/p.name
   candidate=stage/p.name
   if candidate.exists():assert candidate.read_bytes()==p.read_bytes(),('Saved CSV byte mismatch',n,p.name)
   shutil.copyfile(p,q);data[p.name]=sha(q)
 for f in ['manifest.json','source.json','render_geometry.json']:
  if(stage/f).exists():shutil.copyfile(stage/f,out/('inherited_parent_'+f))
 dump(out/'numeric_artist_values_before_after.json',values);dump(out/'render_geometry.json',proof)
 (out/'plot.py').write_text('import runpy,sys\nsys.argv=[__file__]+sys.argv[1:]\nrunpy.run_path('+repr(str(MAINTAINED))+',run_name="__main__")\n')
 dump(out/'source.json',dict(schema='chapter6-spacing-derivative-v1',number=n,parent_directory=str(parent),parent_pdf_sha256=sha(parent/'figure.pdf') if (parent/'figure.pdf').is_file() else None,parent_pdf_reference_available=(parent/'figure.pdf').is_file(),renderer=str(MAINTAINED),renderer_sha256=sha(MAINTAINED),accepted_renderer=str(LEARNING if n=='6.5'else parent/'plot.py'if n=='6.7'else STYLE),source_csv_sha256=data,parent_files_sha256=protected,all_parent_files_unchanged=True,science_before_after_digest=values['before_sha256'],numeric_identity=True,output_sha256={f'figure.{e}':sha(out/f'figure.{e}')for e in ['pdf','png','svg']},environment_steps=0,model_loads=0,policy_inference=0,training_runs=0))
 assert protected=={p:sha(p)for p in protected}
 dump(out/'render_receipt.json',dict(status='PASS',number=n,numeric_identity=True,source_CSV_bytes_preserved=True,all_parent_files_unchanged=True,outputs_sha256={str(p.relative_to(out)):sha(p)for p in [out/'figure.pdf',out/'figure.png',out/'figure.svg',out/'source.json',out/'render_geometry.json',out/'numeric_artist_values_before_after.json',out/'plot.py']},scientific_execution=dict(environment_steps=0,model_loads=0,policy_inference=0,training_runs=0),thesis_builds=0))
 print(json.dumps(dict(status='PASS',number=n,native_mm=proof['native_mm'],axes_sizes_mm=proof['axes_sizes_mm'],spacing=proof['pdf_frames_after']),ensure_ascii=False))
if __name__=='__main__':main()
