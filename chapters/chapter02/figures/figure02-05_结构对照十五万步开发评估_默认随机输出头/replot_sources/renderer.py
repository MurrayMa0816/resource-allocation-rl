"""Offline Chapter 2 content-spacing derivatives of the accepted native figures.

Only distances and outer page whitespace change. The accepted parent renderers
and their complete saved-data checks run first; no models or environments run.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,io,json,os,shutil,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent.parent
THESIS=next(p for p in HERE.parents if (p/'assets/fonts/windows').is_dir())
ROOT=HERE
MAINTAINED=Path(__file__)
MAIN=HERE/'replot_sources'
PRINT=MAIN
RECORDS={'2.5':{'physical_directory':HERE/'replot_sources/parent'}}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def copy_checked(src,dst):
 dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert src.read_bytes()==dst.read_bytes(),str(dst)
 else:dst.write_bytes(src.read_bytes())

def main():
 ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--number',choices=list(RECORDS),required=True);ap.add_argument('--output-dir',type=Path);a=ap.parse_args();n=a.number;parent=Path(RECORDS[n]['physical_directory']);out=(a.output_dir or ROOT/'assets'/('figure'+n.replace('.',''))).resolve();assert not(out/'figure.pdf').exists() and out!=HERE;out.mkdir(parents=True,exist_ok=True)
 for key,leaf in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
  p=out/'runtime'/leaf;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
 import numpy as np
 import matplotlib;matplotlib.use('Agg')
 from matplotlib.figure import Figure
 from matplotlib.text import Text
 from matplotlib.transforms import Bbox
 from PIL import Image
 import fitz
 original_save=Figure.savefig;state={};proof={};science={};styled=set()
 def plain(v):
  if isinstance(v,np.ndarray):return plain(v.tolist())
  if isinstance(v,np.generic):return v.item()
  if isinstance(v,(tuple,list)):return[plain(x)for x in v]
  if isinstance(v,dict):return{k:plain(x)for k,x in v.items()}
  if isinstance(v,(int,float,str,bool))or v is None:return v
  return str(v)
 def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
 def numeric(fig):
  return [dict(xlim=plain(ax.get_xlim()),ylim=plain(ax.get_ylim()),xticks=plain(ax.get_xticks()),yticks=plain(ax.get_yticks()),xscale=ax.get_xscale(),yscale=ax.get_yscale(),lines=[dict(x=plain(l.get_xdata()),y=plain(l.get_ydata()),color=plain(l.get_color()),linewidth=l.get_linewidth(),alpha=l.get_alpha(),linestyle=l.get_linestyle(),marker=l.get_marker(),markersize=l.get_markersize(),markevery=plain(l.get_markevery()),mfc=plain(l.get_markerfacecolor()),mec=plain(l.get_markeredgecolor()))for l in ax.lines],collections=[dict(paths=[dict(vertices=plain(p.vertices),codes=plain(p.codes))for p in c.get_paths()],offsets=plain(c.get_offsets()),array=plain(c.get_array()),facecolors=plain(c.get_facecolors()),edgecolors=plain(c.get_edgecolors()),linewidths=plain(c.get_linewidths()),alpha=c.get_alpha(),clim=plain(c.get_clim()))for c in ax.collections],patches=[dict(vertices=plain(p.get_path().vertices),codes=plain(p.get_path().codes),patch_transform=plain(p.get_patch_transform().get_matrix()),facecolor=plain(p.get_facecolor()),edgecolor=plain(p.get_edgecolor()),linewidth=p.get_linewidth())for p in ax.patches],images=[dict(array=plain(i.get_array()),extent=plain(i.get_extent()),clim=plain(i.get_clim()))for i in ax.images])for ax in fig.axes]
 def fig_artists(fig):
  return [dict(kind=type(p).__name__,vertices=plain(p.get_path().vertices),patch_transform=plain(p.get_patch_transform().get_matrix()),facecolor=plain(p.get_facecolor()),edgecolor=plain(p.get_edgecolor()),linewidth=p.get_linewidth())for p in fig.artists if hasattr(p,'get_path')]
 def titles(ax):return [t for t in [ax.title,ax._left_title,ax._right_title]if t.get_text()and t.get_visible()]
 def text(t):return ''.join(t.split())
 def style(fig):
  fig.canvas.draw();before=numeric(fig);before_figure=fig_artists(fig);old_width,old_height=fig.get_size_inches()*25.4
  positions=[np.array(ax.get_position().bounds)*np.array([old_width,old_height,old_width,old_height])for ax in fig.axes]
  dpi=round(Image.open(parent/'figure.png').info.get('dpi',(240,240))[0]);repro=out/'accepted_parent_reproduction.png';original_save(fig,repro,format='png',dpi=dpi)
  aa=np.asarray(Image.open(repro).convert('RGBA'));bb=np.asarray(Image.open(parent/'figure.png').convert('RGBA'));assert aa.shape==bb.shape and np.array_equal(aa,bb),(n,'accepted parent pixels differ',aa.shape,bb.shape)
  regular=n!='2.8';mainaxes=[ax for ax in fig.axes if ax.get_xlabel()and titles(ax)]if regular else []
  rows=[]
  if regular:
   for ax in mainaxes:
    for t in titles(ax):ax.set_title(t.get_text(),loc='center'if t is ax.title else 'left'if t is ax._left_title else 'right',pad=6,fontfamily=t.get_fontfamily(),fontsize=t.get_fontsize(),fontweight=t.get_fontweight())
    ax.xaxis.labelpad=4;ax.yaxis.labelpad=4;ax.tick_params(axis='both',which='major',pad=3)
   for ax in sorted(mainaxes,key=lambda z:-z.get_position().y0):
    y=ax.get_position().y0
    target=next((r for r in rows if abs(r[0].get_position().y0-y)<1e-5),None)
    if target is None:rows.append([ax])
    else:target.append(ax)
   assert len(fig.legends)==1,(n,'one shared legend required');legend=fig.legends[0]
  else:legend=None
  crop=None
  def frame(current_crop):
   kwargs=dict(format='pdf',metadata={'CreationDate':None,'ModDate':None})
   if current_crop is not None:kwargs.update(bbox_inches=current_crop,pad_inches=0)
   stream=io.BytesIO();original_save(fig,stream,**kwargs);d=fitz.open(stream=stream.getvalue(),filetype='pdf');page=d[0];items=[]
   for block in page.get_text('dict')['blocks']:
    for line in block.get('lines',[]):items.append((''.join(s['text']for s in line['spans']),fitz.Rect(line['bbox'])))
   def boxes(labels):
    wanted={text(s)for s in labels};found=[b for s,b in items if text(s)in wanted];assert found,(n,'PDF labels not found',wanted);return found
   if regular:
    xb=boxes([ax.get_xlabel()for ax in rows[-1]]);lb=boxes([t.get_text()for t in legend.get_texts()]);tb=boxes([t.get_text()for ax in mainaxes for t in titles(ax)]);gaps=[]
    for upper,lower in zip(rows,rows[1:]):gaps.append(min(b.y0 for b in boxes([t.get_text()for ax in lower for t in titles(ax)]))-max(b.y1 for b in boxes([ax.get_xlabel()for ax in upper])))
    result=dict(xlabel_to_legend_bp=min(b.y0 for b in lb)-max(b.y1 for b in xb),legend_to_file_bottom_mm=(page.rect.height-max(b.y1 for b in lb))*25.4/72,subtitle_to_file_top_mm=min(b.y0 for b in tb)*25.4/72,row_gaps_bp=gaps,native_mm=[page.rect.width*25.4/72,page.rect.height*25.4/72],xlabel_frames_bp=[list(b)for b in xb],legend_frames_bp=[list(b)for b in lb],subtitle_frames_bp=[list(b)for b in tb])
   else:
    assert items;tb=[b for s,b in items if s.startswith(tuple('('+x+')'for x in 'abcdef'))];assert len(tb)==6
    result=dict(xlabel_to_legend_bp=None,legend_to_file_bottom_mm=None,subtitle_to_file_top_mm=min(b.y0 for b in tb)*25.4/72,row_gaps_bp=[],native_mm=[page.rect.width*25.4/72,page.rect.height*25.4/72],all_text_to_bottom_mm=(page.rect.height-max(b.y1 for s,b in items))*25.4/72,all_text_to_top_mm=min(b.y0 for s,b in items)*25.4/72,subtitle_frames_bp=[list(b)for b in tb])
   result['all_text_inside_canvas']=all(b.x0>=-.1 and b.y0>=-.1 and b.x1<=page.rect.width+.1 and b.y1<=page.rect.height+.1 for s,b in items);result['PDF_text_count']=len(items);d.close();return result
  initial=frame(None)
  for iteration in range(4):
   f=frame(crop);bottom_crop=0 if crop is None else crop.y0*72;top_crop=0 if crop is None else old_height*72/25.4-crop.y1*72
   if regular:
    for upper,delta in zip(rows[:-1],[12-g for g in f['row_gaps_bp']]):
     if abs(delta)>.0001:
      for ax in upper:
       p=np.array(ax.get_position().bounds);p[1]+=delta/(old_height*72/25.4);ax.set_position(p)
    delta=8-f['xlabel_to_legend_bp'];anchor=legend.get_bbox_to_anchor();anchor_y=anchor.y0/fig.dpi*72;legend.set_bbox_to_anchor((anchor.x0/fig.bbox.width,(anchor_y-delta)/(old_height*72/25.4)),transform=fig.transFigure)
    fig.canvas.draw();f=frame(crop);bottom_crop+=(f['legend_to_file_bottom_mm']-2)*72/25.4;top_crop+=(f['subtitle_to_file_top_mm']-1.5)*72/25.4
   else:
    bottom_crop+=(f['all_text_to_bottom_mm']-2)*72/25.4;top_crop+=(f['subtitle_to_file_top_mm']-1.5)*72/25.4
   crop=Bbox.from_bounds(0,bottom_crop/72,old_width/25.4,(old_height*72/25.4-top_crop-bottom_crop)/72);fig.canvas.draw()
   f=frame(crop)
   errors=[abs(f['subtitle_to_file_top_mm']-1.5),abs((f['legend_to_file_bottom_mm']if regular else f['all_text_to_bottom_mm'])-2)]
   if regular:errors += [abs(f['xlabel_to_legend_bp']-8)]+[abs(g-12)for g in f['row_gaps_bp']]
   if max(errors)<.002:break
  final=frame(crop);assert abs(final['subtitle_to_file_top_mm']-1.5)<.002
  if regular:assert abs(final['xlabel_to_legend_bp']-8)<.002 and abs(final['legend_to_file_bottom_mm']-2)<.002 and all(abs(g-12)<.002 for g in final['row_gaps_bp'])
  else:assert abs(final['all_text_to_bottom_mm']-2)<.002
  assert final['all_text_inside_canvas'],(n,'text outside PDF canvas',final)
  after=numeric(fig);after_figure=fig_artists(fig);assert before==after,(n,'scientific data/style changed');assert before_figure==after_figure,(n,'figure-level allocation circle geometry changed')
  current=[np.array(ax.get_position().bounds)*np.array([old_width,old_height,old_width,old_height])for ax in fig.axes];assert all(np.allclose(p[2:],q[2:],rtol=0,atol=1e-8)for p,q in zip(positions,current))
  science.update(before=before,after=after,before_sha256=digest(before),after_sha256=digest(after),figure_artists_before=before_figure,figure_artists_after=after_figure,figure_artists_exact=before_figure==after_figure)
  proof.update(schema='thesis-measured-content-spacing-v1',status='PASS',number=n,parent_native_png_pixel_exact=True,native_mm=final['native_mm'],original_native_mm=[old_width,old_height],axes_physical_mm_before=[plain(p)for p in positions],axes_physical_mm_after=[plain(p)for p in current],axes_physical_width_height_unchanged=True,ordinary_font_bp=10.5,object_ID_font_bp=9 if n=='2.8'else None,object_ID_count=660 if n=='2.8'else None,all_scientific_data_style_exact=True,PDF_text_frame_measurements=final,PDF_text_frame_initial=initial,measurement_basis='native PDF text-line frames; not pixel ink',crop_bbox_inches=plain(crop.bounds),figure_internal_axes_and_figure_artist_geometry_unchanged=n=='2.8',standard_target_exception='Composite map has state/area scales rather than shared curve legend; keep original map geometry and add outer text bounds1.5mm/2mm'if n=='2.8'else None,environment_steps=0,model_loads=0,policy_inferences=0,training_runs=0)
  state.update(crop=crop,figure=fig);return crop
 def save(fig,dest,*args,**kwargs):
  name=Path(dest).name if isinstance(dest,(str,Path))else ''
  if name in ('figure.pdf','figure.png','figure.svg'):
   if id(fig)not in styled:styled.add(id(fig));style(fig)
   kwargs.update(bbox_inches=state['crop'],pad_inches=0)
   if name.endswith('.pdf'):kwargs['metadata']={'CreationDate':None,'ModDate':None}
   if name.endswith('.svg'):kwargs['metadata']={'Date':None}
   if name.endswith('.png'):kwargs['dpi']=300
  return original_save(fig,dest,*args,**kwargs)
 Figure.savefig=save;argv=sys.argv[:]
 try:
  if n=='2.4':
   m=module(MAIN/'main_learning.py','accepted_c2_main');m.R=out;sys.argv=['native','--number',n,'--output-dir',str(out)];m.main()
  elif n=='2.8':
   m=module(PRINT/'print_stage.py','accepted_c2_map');sys.argv=['native','--number',n,'--output-dir',str(out)];m.main()
  else:
   inp=out/'inputs';inp.mkdir(exist_ok=True)
   for name in ['plot.py','presentation_layout.py','data.csv']:copy_checked(parent/name,inp/name)
   sys.path.insert(0,str(inp));sys.modules.pop('presentation_layout',None);m=module(inp/'plot.py','accepted_c2_'+n[-1]);sys.argv=['native','--data',str(inp/'data.csv'),'--output-dir',str(out),'--font-dir',str(THESIS/'assets/fonts/windows')]
   if n in ['2.6','2.7']:sys.argv+=['--kind','scale'if n=='2.6'else'conditions']
   m.main()
 finally:Figure.savefig=original_save;sys.argv=argv
 assert len(styled)==1
 for name in ['source.json','render_geometry.json','render_manifest.json','style_proof.json','print_presentation_proof.json']:
  p=out/name
  if p.exists():p.rename(out/('inherited_parent_'+name))
 scientific={}
 for p in parent.rglob('*'):
  if not p.is_file()or any(s in p.parts for s in ['runtime','cache','__pycache__']):continue
  if p.suffix not in ['.csv','.gz']and p.name not in ['provenance.json']:continue
  rel=p.relative_to(parent);q=out/rel;q.parent.mkdir(parents=True,exist_ok=True)
  if q.exists()and q.is_symlink():q.unlink()
  q.write_bytes(p.read_bytes());scientific[str(rel)]=sha(q)
 dump(out/'render_geometry.json',proof);dump(out/'numeric_artist_values_before_after.json',science)
 (out/'plot.py').write_text('import runpy,sys\nsys.argv=[__file__,"--number",'+repr(n)+']+sys.argv[1:]\nrunpy.run_path('+repr(str(MAINTAINED))+',run_name="__main__")\n')
 dump(out/'source.json',dict(schema='chapter2-measured-content-spacing-v1',number=n,parent_directory=str(parent),parent_pdf_sha256=sha(parent/'figure.pdf'),parent_source_sha256=sha(parent/'source.json'),renderer=str(MAINTAINED),renderer_sha256=sha(MAINTAINED),scientific_data_sha256=scientific,all_scientific_data_style_exact=True,geometry_proof_sha256=sha(out/'render_geometry.json'),native_mm=proof['native_mm'],outputs_sha256={f'figure.{e}':sha(out/f'figure.{e}')for e in ['pdf','png','svg']},environment_steps=0,model_loads=0,policy_inferences=0,training_runs=0))
 dump(out/'render_receipt.json',dict(status='PASS',number=n,output_sha256={name:sha(out/name)for name in ['figure.pdf','figure.png','figure.svg','render_geometry.json','numeric_artist_values_before_after.json','source.json','plot.py']},scientific_data_sha256=scientific,parent_png_pixel_exact=True,environment_steps=0,model_loads=0,policy_inferences=0,training_runs=0,thesis_builds=0))
 (out/'README.rst').write_text('第二章图内距离统一\n====================\n\n接受父图完整离线复绘后只调整文字距离与外白，全部科学值和图元样式保持。数值before/after及PDF文字框测量见相邻JSON；PDF文字框不是像素墨迹。复绘：python plot.py --output-dir 新HY目录。\n')
 print(json.dumps(dict(status='PASS',number=n,native_mm=proof['native_mm'],PDF=proof['PDF_text_frame_measurements']),ensure_ascii=False))
if __name__=='__main__':main()
