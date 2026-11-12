"""Offline Chapter 3 statistics with measured content spacing.

Saved science is recreated by the accepted parents, then only the artist layout
and explicitly recorded categorical wording are changed. No model/environment.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, shutil, sys
sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent.parent
THESIS=next(p for p in HERE.parents if (p/'assets/fonts/windows').is_dir())
ROOT=HERE
MAINTAINED=Path(__file__)
PARENTS={'3.6':HERE/'replot_sources/parent'}
SOURCE_RENDERERS={'3.6':HERE/'replot_sources'/'conditions.py'}
WIDTH=156.0

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--number',required=True,choices=list(PARENTS));ap.add_argument('--output-dir',type=Path);a=ap.parse_args();n=a.number;parent=PARENTS[n];out=(a.output_dir or ROOT/'assets'/('figure'+n.replace('.',''))).resolve();assert out!=HERE and not(out/'figure.pdf').exists();out.mkdir(parents=True,exist_ok=True)
 for key,leaf in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
  q=out/'runtime'/leaf;q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
 import numpy as np
 import matplotlib;matplotlib.use('Agg')
 from matplotlib.figure import Figure
 from matplotlib.text import Text
 from matplotlib.transforms import Bbox
 original_save=Figure.savefig;styled=set();proof={};values={}
 def plain(v):
  if isinstance(v,np.ndarray):return plain(v.tolist())
  if isinstance(v,np.generic):return v.item()
  if isinstance(v,(tuple,list)):return[plain(x)for x in v]
  if isinstance(v,dict):return{k:plain(x)for k,x in v.items()}
  return v
 def numeric(fig):
  return [dict(xlim=plain(ax.get_xlim()),ylim=plain(ax.get_ylim()),xticks=plain(ax.get_xticks()),yticks=plain(ax.get_yticks()),xscale=ax.get_xscale(),yscale=ax.get_yscale(),lines=[dict(x=plain(l.get_xdata()),y=plain(l.get_ydata()),color=plain(l.get_color()),alpha=l.get_alpha(),linewidth=l.get_linewidth(),linestyle=l.get_linestyle(),marker=l.get_marker(),markersize=l.get_markersize(),markevery=plain(l.get_markevery()))for l in ax.lines],collections=[dict(paths=[dict(vertices=plain(p.vertices),codes=plain(p.codes))for p in c.get_paths()],offsets=plain(c.get_offsets()),array=plain(c.get_array()),facecolors=plain(c.get_facecolors()),edgecolors=plain(c.get_edgecolors()),linewidths=plain(c.get_linewidths()),alpha=c.get_alpha())for c in ax.collections])for ax in fig.axes]
 def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
 def legend_box(legend,ren):
  artists=list(legend.get_texts())+list(getattr(legend,'legend_handles',[]));boxes=[x.get_window_extent(ren)for x in artists if x.get_visible()];return Bbox.union([b for b in boxes if np.isfinite(b.extents).all()])
 def style(fig):
  fig.canvas.draw();before=numeric(fig);values['before']=before
  assert len(fig.axes)==(4 if n=='3.6'else 2);assert len(fig.legends)==1
  if n=='3.6':
   ax=fig.axes[3];ax.set_xticks(ax.get_xticks(),['原始顺序','事件逆序','后段移中']);ax.set_xlabel('新增需求时序',labelpad=4);ax.set_title('(d) 柴油需求时序变化',loc='center',pad=6,fontfamily=['Times New Roman','SimSun'],fontsize=10.5);[t.set_ha('center')for t in ax.get_xticklabels()];ax.get_xticklabels()[-1].set_ha('right')
  fig.canvas.draw();old_h=fig.get_figheight()*25.4
  mm_positions=[np.array(ax.get_position().bounds)*np.array([WIDTH,old_h,WIDTH,old_h])for ax in fig.axes]
  assert all(np.allclose(p[2:],[59.2,51],rtol=0,atol=1e-9)for p in mm_positions)
  if n=='3.6':
   ren=fig.canvas.get_renderer();k=72/fig.dpi
   top_label_bottom=min(ax.xaxis.label.get_window_extent(ren).y0 for ax in fig.axes[:2])*k
   lower_title_top=max(ax.title.get_window_extent(ren).y1 for ax in fig.axes[2:])*k
   shift=(12-(top_label_bottom-lower_title_top))*25.4/72
   for i,ax in enumerate(fig.axes[:2]):mm_positions[i][1]+=shift;ax.set_position(mm_positions[i]/np.array([WIDTH,old_h,WIDTH,old_h]))
   fig.canvas.draw()
  legend=fig.legends[0];ren=fig.canvas.get_renderer();k=72/fig.dpi
  xlabel_bottom=min(ax.xaxis.label.get_window_extent(ren).y0 for ax in fig.axes[-2:])*k
  lb=legend_box(legend,ren);delta=xlabel_bottom-8-lb.y1*k
  anchor=legend.get_bbox_to_anchor();anchor_y=anchor.y0*k+delta
  legend.set_bbox_to_anchor((.5,anchor_y/(old_h*72/25.4)),transform=fig.transFigure);fig.canvas.draw();ren=fig.canvas.get_renderer();lb=legend_box(legend,ren)
  top=max(ax.title.get_window_extent(ren).y1 for ax in fig.axes)*k
  crop_bottom=lb.y0*k-2*72/25.4;crop_top=top+1.5*72/25.4;height=(crop_top-crop_bottom)*25.4/72
  fig.set_layout_engine(None);fig.set_size_inches(WIDTH/25.4,height/25.4,forward=True)
  for ax,p in zip(fig.axes,mm_positions):ax.set_position([p[0]/WIDTH,(p[1]-crop_bottom*25.4/72)/height,p[2]/WIDTH,p[3]/height])
  legend.set_bbox_to_anchor((.5,(anchor_y-crop_bottom)/(height*72/25.4)),transform=fig.transFigure)
  fig.canvas.draw();ren=fig.canvas.get_renderer();k=72/fig.dpi;lb=legend_box(legend,ren)
  final_gap=(min(ax.xaxis.label.get_window_extent(ren).y0 for ax in fig.axes[-2:])-lb.y1)*k;legend_bottom=lb.y0*k*25.4/72;top_margin=height-max(ax.title.get_window_extent(ren).y1 for ax in fig.axes)*25.4/fig.dpi
  assert abs(final_gap-8)<1e-7,(n,final_gap);assert abs(legend_bottom-2)<1e-7,(n,legend_bottom);assert abs(top_margin-1.5)<1e-7,(n,top_margin)
  row_gap=None
  if n=='3.6':
   row_gap=(min(ax.xaxis.label.get_window_extent(ren).y0 for ax in fig.axes[:2])-max(ax.title.get_window_extent(ren).y1 for ax in fig.axes[2:]))*k;assert abs(row_gap-12)<1e-7,row_gap
  # Calibrate against the emitted PDF text frames. Matplotlib layout boxes
  # include different font ascent/descent allowances; they are not PDF glyph
  # frames and must not be presented as pixel ink.
  import io, fitz
  def pdf_frames():
   stream=io.BytesIO();original_save(fig,stream,format='pdf',metadata={'CreationDate':None,'ModDate':None});pdf=fitz.open(stream=stream.getvalue(),filetype='pdf');page=pdf[0];rows=[]
   for block in page.get_text('dict')['blocks']:
    for line in block.get('lines',[]):
     text=''.join(x['text']for x in line['spans']);rows.append((text,line['bbox']))
   def normalize(text):return ''.join(text.split())
   def boxes(labels):
    labels={normalize(t)for t in labels};found=[fitz.Rect(b)for text,b in rows if normalize(text)in labels]
    assert found,('PDF text frame not found',n,labels,[t for t,b in rows]);return found
   xb=boxes([ax.get_xlabel()for ax in fig.axes[-2:]]);leg=boxes([t.get_text()for t in legend.get_texts()]);title=boxes([ax.get_title()for ax in fig.axes]);row=None
   if n=='3.6':
    upper=boxes([ax.get_xlabel()for ax in fig.axes[:2]]);lower=boxes([ax.get_title()for ax in fig.axes[2:]]);row=min(b.y0 for b in lower)-max(b.y1 for b in upper)
   result=dict(xlabel_to_legend_bp=min(b.y0 for b in leg)-max(b.y1 for b in xb),legend_to_file_bottom_mm=(page.rect.height-max(b.y1 for b in leg))*25.4/72,subtitle_to_file_top_mm=min(b.y0 for b in title)*25.4/72,upper_xlabel_to_lower_subtitle_bp=row,native_mm=[page.rect.width*25.4/72,page.rect.height*25.4/72],xlabel_frames_bp=[list(b)for b in xb],legend_frames_bp=[list(b)for b in leg],subtitle_frames_bp=[list(b)for b in title]);pdf.close();return result
  pdf_initial=pdf_frames()
  for calibration in range(3):
   frame=pdf_frames();row_shift=(12-frame['upper_xlabel_to_lower_subtitle_bp'])if n=='3.6'else 0.;legend_down=8-frame['xlabel_to_legend_bp'];top_crop=frame['subtitle_to_file_top_mm']*72/25.4-row_shift-1.5*72/25.4;bottom_crop=frame['legend_to_file_bottom_mm']*72/25.4-legend_down-2*72/25.4
   if max(abs(row_shift),abs(legend_down),abs(top_crop),abs(bottom_crop))<.001:break
   positions=[np.array(ax.get_position().bounds)*np.array([WIDTH,height,WIDTH,height])for ax in fig.axes];anchor_pt=legend.get_bbox_to_anchor().y0*72/fig.dpi;new_height=(height*72/25.4-top_crop-bottom_crop)*25.4/72
   fig.set_size_inches(WIDTH/25.4,new_height/25.4,forward=True)
   for i,(ax,p)in enumerate(zip(fig.axes,positions)):
    y=p[1]+(row_shift*25.4/72 if n=='3.6'and i<2 else 0)-bottom_crop*25.4/72;ax.set_position([p[0]/WIDTH,y/new_height,p[2]/WIDTH,p[3]/new_height])
   legend.set_bbox_to_anchor((.5,(anchor_pt-legend_down-bottom_crop)/(new_height*72/25.4)),transform=fig.transFigure);height=new_height;fig.canvas.draw()
  pdf_final=pdf_frames();assert abs(pdf_final['xlabel_to_legend_bp']-8)<.002,pdf_final;assert abs(pdf_final['legend_to_file_bottom_mm']-2)<.001,pdf_final;assert abs(pdf_final['subtitle_to_file_top_mm']-1.5)<.001,pdf_final
  if n=='3.6':assert abs(pdf_final['upper_xlabel_to_lower_subtitle_bp']-12)<.002,pdf_final
  ren=fig.canvas.get_renderer();k=72/fig.dpi;lb=legend_box(legend,ren);final_gap=(min(ax.xaxis.label.get_window_extent(ren).y0 for ax in fig.axes[-2:])-lb.y1)*k;legend_bottom=lb.y0*k*25.4/72;top_margin=height-max(ax.title.get_window_extent(ren).y1 for ax in fig.axes)*25.4/fig.dpi
  if n=='3.6':row_gap=(min(ax.xaxis.label.get_window_extent(ren).y0 for ax in fig.axes[:2])-max(ax.title.get_window_extent(ren).y1 for ax in fig.axes[2:]))*k
  after=numeric(fig);assert before==after,'Scientific numeric/style artists changed';values.update(after=after,before_sha256=digest(before),after_sha256=digest(after))
  hidden=set()
  for ax in fig.axes:
   for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
    for tick in axis.get_major_ticks():
     if not min(limits)-1e-8<=tick.get_loc()<=max(limits)+1e-8:hidden.update((id(tick.label1),id(tick.label2)))
  texts=[]
  for t in fig.findobj(Text):
   if not t.get_visible()or not t.get_text()or id(t)in hidden:continue
   b=t.get_window_extent(ren);assert b.x0>=-.5 and b.y0>=-.5 and b.x1<=fig.bbox.x1+.5 and b.y1<=fig.bbox.y1+.5,(n,t.get_text(),b.bounds);assert t.get_fontsize()==10.5,(n,t.get_text(),t.get_fontsize());texts.append(dict(text=t.get_text(),font_bp=t.get_fontsize(),bbox_bp=plain(np.array(b.bounds)*k)))
  proof.update(schema='chapter3-content-spacing-v1',status='PASS',number=n,native_mm=[WIDTH,height],axes_mm=[59.2,51],axes_position_mm=[plain(np.array(ax.get_position().bounds)*np.array([WIDTH,height,WIDTH,height]))for ax in fig.axes],xlabel_to_legend_bp=pdf_final['xlabel_to_legend_bp'],legend_to_file_bottom_mm=pdf_final['legend_to_file_bottom_mm'],subtitle_to_file_top_mm=pdf_final['subtitle_to_file_top_mm'],upper_xlabel_to_lower_subtitle_bp=pdf_final['upper_xlabel_to_lower_subtitle_bp'],measurement_basis='native PDF text frames; not pixel ink',pdf_frames_initial=pdf_initial,pdf_frames_final=pdf_final,matplotlib_layout_bbox=dict(xlabel_to_legend_bp=final_gap,legend_to_file_bottom_mm=legend_bottom,subtitle_to_file_top_mm=top_margin,upper_xlabel_to_lower_subtitle_bp=row_gap),legend_rows=2 if n=='3.4'else 1,legend_labels=[t.get_text()for t in legend.get_texts()],all_text_inside_canvas=True,ordinary_font_bp=10.5,scientific_artist_values_and_line_styles_exact=True,numeric_before_sha256=digest(before),numeric_after_sha256=digest(after),texts=texts,category_wording_changes={'old':['原顺序','后段提前','后段移至\\n中段'],'new':['原始顺序','事件逆序','后段移中'],'xlabel':'新增需求时序','title':'(d) 柴油需求时序变化'}if n=='3.6'else None,environment_steps=0,model_loads=0,policy_inferences=0,training_runs=0,thesis_builds=0)
 def save(fig,destination,*args,**kwargs):
  if Path(destination).name in ('figure.pdf','figure.png','figure.svg')and id(fig)not in styled:style(fig);styled.add(id(fig))
  suffix=Path(destination).suffix
  if Path(destination).name in ('figure.pdf','figure.png','figure.svg'):
   if suffix=='.png':kwargs['dpi']=600
   if suffix=='.pdf':kwargs['metadata']={'CreationDate':None,'ModDate':None}
   if suffix=='.svg':kwargs['metadata']={'Date':None}
  return original_save(fig,destination,*args,**kwargs)
 # Invoke accepted renderers with only output/cache roots adapted. Their saved
 # scientific inputs and all existing source/CSV guards remain in force.
 Figure.savefig=save;oldargv=sys.argv[:]
 try:
  m=module(SOURCE_RENDERERS[n],'accepted_'+n.replace('.','_'))
  if n=='3.4':m.R=ROOT;sys.argv=['accepted','--number',n,'--output-dir',str(out)]
  else:m.ROOT=ROOT;sys.argv=['accepted','--output-dir',str(out)]
  m.main()
 finally:Figure.savefig=original_save;sys.argv=oldargv
 assert len(styled)==1
 # Retain the inherited-stage metadata as inherited, never as final geometry.
 for name in ['source.json','render_geometry.json','style_proof.json','artists.json','render_receipt.json','artist_values_before_after.json']:
  f=out/name
  if f.exists():f.rename(out/('inherited_parent_'+name))
 scientific={}
 for name in ['data.csv.gz','display_aggregates.csv.gz','display_aggregates.csv','display_seed_data.csv','full_minus_pace.csv','full_minus_pace.json']:
  p=parent/name
  if not p.exists():continue
  q=out/name
  if q.exists():
   import gzip
   if name.endswith('.gz'):assert gzip.decompress(q.read_bytes())==gzip.decompress(p.read_bytes())
   else:assert q.read_bytes()==p.read_bytes()
   if q.is_symlink():q.unlink()
  q.write_bytes(p.read_bytes());scientific[name]=sha(q)
 dump(out/'render_geometry.json',proof);dump(out/'numeric_artist_values_before_after.json',values)
 if n=='3.5':
  old=json.loads((parent/'artists.json').read_text());final_artists=dict(status='PASS',size_mm=proof['native_mm'],axes_mm=proof['axes_mm'],panels=[dict(v,axis_mm=proof['axes_position_mm'][i])for i,v in enumerate(old['panels'])],texts=proof['texts'],pdf_text_frames=proof['pdf_frames_final'],measurement_basis=proof['measurement_basis'],font_bp=10.5,legend_rows=1,legend_columns=3,xlabel_to_legend_bp=proof['xlabel_to_legend_bp'],legend_to_canvas_bottom_bp=proof['legend_to_file_bottom_mm']*72/25.4);dump(out/'artists.json',final_artists)
 if n=='3.6':shutil.copyfile(parent/'artist_values_before_after.json',out/'accepted_scientific_errorbars.json')
 if n=='3.4':shutil.copyfile(parent/'numeric_artist_geometry.json',out/'accepted_scientific_artists.json')
 (out/'plot.py').write_text('import runpy,sys\nsys.argv=[__file__,"--number",'+repr(n)+']+sys.argv[1:]\nrunpy.run_path('+repr(str(MAINTAINED))+',run_name="__main__")\n')
 dump(out/'source.json',dict(schema='chapter3-measured-spacing-derivative-v1',number=n,parent_directory=str(parent),parent_pdf_sha256=sha(parent/'figure.pdf'),parent_source_sha256=sha(parent/'source.json'),accepted_renderer=str(SOURCE_RENDERERS[n]),accepted_renderer_sha256=sha(SOURCE_RENDERERS[n]),renderer=str(MAINTAINED),renderer_sha256=sha(MAINTAINED),scientific_data_sha256=scientific,native_mm=proof['native_mm'],science_before_after_digest=values['before_sha256'],scientific_artist_values_and_line_styles_exact=True,changes=['PDF text-frame measured xlabel-to-legend8bp','legend-to-file-bottom2mm','top subtitle-to-file-top1.5mm','four-panel inter-row ink12bp'if n=='3.6'else'content-dependent total height'],category_wording=proof['category_wording_changes'],output_sha256={f'figure.{e}':sha(out/f'figure.{e}')for e in ['pdf','png','svg']},environment_steps=0,model_loads=0,policy_inferences=0,training_runs=0))
 dump(out/'render_receipt.json',dict(status='PASS',number=n,outputs_sha256={str(p.relative_to(out)):sha(p)for p in [out/'figure.pdf',out/'figure.png',out/'figure.svg',out/'render_geometry.json',out/'numeric_artist_values_before_after.json',out/'source.json',out/'plot.py']},new_environment_steps=0,model_loads=0,training_runs=0,thesis_builds=0));print(json.dumps(dict(status='PASS',number=n,native_mm=proof['native_mm'],gap_bp=proof['xlabel_to_legend_bp']),ensure_ascii=False))
if __name__=='__main__':main()
