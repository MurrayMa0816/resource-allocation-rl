"""Offline native re-layout of an accepted Chapter 4 figure; saved observations only."""
from pathlib import Path
import argparse,hashlib,json,os,sys,runpy,io
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,default=HERE/'regenerated');ap.add_argument('--font-dir',type=Path)
a=ap.parse_args();out=a.output_dir.resolve()
if out==HERE or (out/'figure.pdf').exists():raise ValueError('Choose a new output directory; accepted outputs are protected')
out.mkdir(parents=True,exist_ok=True)
thesis=next((p for p in HERE.parents if (p/'assets/fonts/windows').is_dir()),Path('/root/gamewar/latex_thesis_resource_allocation'))
fonts=a.font_dir or thesis/'assets/fonts/windows'
assert fonts.is_dir()
for env,leaf in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
 p=out/'runtime'/leaf;p.mkdir(parents=True,exist_ok=True);os.environ[env]=str(p)
config=json.loads((HERE/'geometry_config.json').read_text())
for name,h in json.loads((HERE/'native_parent_sha256.json').read_text()).items():assert hashlib.sha256((HERE/'native_parent'/name).read_bytes()).hexdigest()==h,name
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib import font_manager
import fitz
fontmap={p.name.lower():p for p in fonts.iterdir()if p.is_file()}
for n in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf','simsun.ttc','simhei.ttf']:font_manager.fontManager.addfont(str(fontmap[n]))
original=Figure.savefig;captured=[]
def capture(fig,path,*args,**kwargs):
 if isinstance(path,(str,Path)) and Path(path).name in ['figure.pdf','figure.png','figure.svg']:
  if all(fig is not x for x in captured):captured.append(fig)
 return original(fig,path,*args,**kwargs)
parent=HERE/'native_parent';p=parent/'plot.py';source=p.read_text();needle="THESIS=next(p for p in HERE.parents if (p/'assets/fonts').is_dir())";assert needle in source
source=source.replace(needle,'THESIS=Path('+repr(str(thesis))+')')
# Historical rebasing must exempt the explicitly local asset and output roots
# when a portable package is itself staged below the historical HY prefix.
allow="[str(HERE),str(destination),str(work),"+repr(str(out))+"]"
source=source.replace("if isinstance(s,bytes):return s","if isinstance(s,bytes):return s\n        if any(s==q or s.startswith(q+'/') for q in "+allow+"):return s")
source=source.replace("p=os.fsdecode(args[0])","p=os.fsdecode(args[0])\n            if any(p==q or p.startswith(q+'/') for q in "+allow+"):return")
oldargv=sys.argv[:];Figure.savefig=capture
try:
 sys.argv=[str(p),'--output-dir',str(out/'accepted_parent_render')]
 ns={'__file__':str(p),'__name__':'__main__'};exec(compile(source,str(p),'exec'),ns)
finally:Figure.savefig=original;sys.argv=oldargv
assert len(captured)==1,(len(captured),config['number'])
fig=captured[0];fig.set_layout_engine(None)
def plain(x):
 if isinstance(x,np.ndarray):return plain(x.tolist())
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,(list,tuple)):return [plain(z)for z in x]
 if isinstance(x,dict):return {str(k):plain(v)for k,v in x.items()}
 if isinstance(x,(int,float,str,bool))or x is None:return x
 return str(x)
def science():
 result=[]
 for ax in fig.axes:
  result.append(dict(xlim=plain(ax.get_xlim()),ylim=plain(ax.get_ylim()),xscale=ax.get_xscale(),yscale=ax.get_yscale(),xticks=plain(ax.get_xticks()),yticks=plain(ax.get_yticks()),xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),title=ax.get_title(),xticklabels=[t.get_text()for t in ax.get_xticklabels()],yticklabels=[t.get_text()for t in ax.get_yticklabels()],lines=[dict(x=plain(l.get_xdata()),y=plain(l.get_ydata()),color=plain(l.get_color()),linewidth=l.get_linewidth(),linestyle=l.get_linestyle(),marker=l.get_marker(),markersize=l.get_markersize(),mfc=plain(l.get_markerfacecolor()),mec=plain(l.get_markeredgecolor()),alpha=l.get_alpha(),markevery=plain(l.get_markevery()),zorder=l.get_zorder())for l in ax.lines],collections=[dict(paths=[dict(vertices=plain(p.vertices),codes=plain(p.codes))for p in c.get_paths()],offsets=plain(c.get_offsets()),array=plain(c.get_array()),clim=plain(c.get_clim()),fc=plain(c.get_facecolors()),ec=plain(c.get_edgecolors()),lw=plain(c.get_linewidths()),alpha=c.get_alpha())for c in ax.collections],patches=[dict(type=type(p).__name__,path=plain(p.get_path().vertices),codes=plain(p.get_path().codes),xy=plain(p.get_xy())if hasattr(p,'get_xy')else None,width=p.get_width()if hasattr(p,'get_width')else None,height=p.get_height()if hasattr(p,'get_height')else None,fc=plain(p.get_facecolor()),ec=plain(p.get_edgecolor()),lw=p.get_linewidth(),alpha=p.get_alpha(),hatch=p.get_hatch())for p in ax.patches],texts=[dict(text=t.get_text(),position=plain(t.get_position()),font=t.get_fontsize(),weight=t.get_fontweight())for t in ax.texts]))
 return result
# Approved wording changes only; retain all saved observations, artists and geometry.
fig.canvas.draw()
academic_before = science()
academic_font_before = [(t.get_text(),t.get_fontsize(),t.get_fontfamily(),t.get_fontweight()) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
academic_label_map = {'图像质量得分': '区间平均图像质量得分', '需求完成比例（%）': '区间平均需求完成比例（%）'}
academic_changes = []
for ax in fig.axes:
 for t in [ax.xaxis.label, ax.yaxis.label, *ax.texts]:
  if t.get_text() in academic_label_map:
   academic_changes.append(dict(old=t.get_text(),new=academic_label_map[t.get_text()],font_bp=t.get_fontsize(),rotation=t.get_rotation(),position=list(t.get_position())))
   t.set_text(academic_label_map[t.get_text()])
fig.canvas.draw()
academic_after = science()
def academic_expected(v):
 if isinstance(v,str): return academic_label_map.get(v,v)
 if isinstance(v,list): return [academic_expected(z) for z in v]
 if isinstance(v,dict): return {k:academic_expected(z) for k,z in v.items()}
 return v
assert academic_after == academic_expected(academic_before), 'Unapproved scientific artist change'
academic_font_after = [(t.get_text(),t.get_fontsize(),t.get_fontfamily(),t.get_fontweight()) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
assert json.loads(json.dumps(academic_font_after)) == academic_expected(json.loads(json.dumps(academic_font_before)))
assert len(academic_changes)==2, academic_changes
(out/'academic_label_changes.json').write_text(json.dumps(dict(status='PASS',label_changes=academic_changes,all_other_scientific_artists_exact=True,font_sizes_families_weights_preserved=True,scientific_execution=0),ensure_ascii=False,indent=2)+'\n')
fig.canvas.draw();before=science();font_before=[(t.get_text(),t.get_fontsize(),t.get_fontfamily(),t.get_fontweight())for t in fig.findobj(Text)if t.get_visible()and t.get_text().strip()]
W,H=fig.get_size_inches()*25.4
positions=lambda:[np.array(ax.get_position().bounds)*np.array([W,H,W,H])for ax in fig.axes]
beforepos=positions();assert len(fig.axes)==2
legends=list(fig.legends);assert len(legends)<=1;legend=legends[0]if legends else None
if config.get('reflow_ylabel_to_single_vertical_line'):
 for ax in fig.axes:ax.set_ylabel(ax.get_ylabel().replace('\n',''))
if config['target_axes_mm']:
 for i,(ax,p)in enumerate(zip(fig.axes,beforepos)):
  nw,nh=config['target_axes_mm'];assert nw>=p[2]-1e-7 and nh>=p[3]-1e-7
  ax.set_position([config['target_x_mm'][i]/W,p[1]/H,nw/W,nh/H])

def text_artists():
 hidden=set()
 for ax in fig.axes:
  for axis,lim in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
   for tick in axis.get_major_ticks():
    if not min(lim)-1e-9<=tick.get_loc()<=max(lim)+1e-9:hidden.update((id(tick.label1),id(tick.label2)))
 return[t for t in fig.findobj(Text)if t.get_visible()and t.get_text().strip()and id(t)not in hidden]
def norm(s):return ''.join(s.split()).replace('−','-')
def measure():
 fig.canvas.draw();buf=io.BytesIO();original(fig,buf,format='pdf',metadata={'CreationDate':None,'ModDate':None});doc=fitz.open(stream=buf.getvalue(),filetype='pdf');page=doc[0];height=page.rect.height;lines=[]
 for block in page.get_text('rawdict')['blocks']:
  for line in block.get('lines',[]):
   chars=[c for span in line['spans']for c in span.get('chars',[])if not c['c'].isspace()];lines.append((''.join(c['c']for c in chars).replace('−','-'),chars))
 def box(t):
  b=t.get_window_extent(fig.canvas.get_renderer());expected=fitz.Rect(b.x0*72/fig.dpi,height-b.y1*72/fig.dpi,b.x1*72/fig.dpi,height-b.y0*72/fig.dpi);found=[]
  for part in [norm(z)for z in t.get_text().split('\n')if norm(z)]:
   cand=[]
   for text,chars in lines:
    start=0
    while part and(start:=text.find(part,start))>=0:
     bb=fitz.Rect(chars[start]['bbox'])
     for c in chars[start:start+len(part)]:bb|=fitz.Rect(c['bbox'])
     dist=(bb.x0+bb.x1-expected.x0-expected.x1)**2+(bb.y0+bb.y1-expected.y0-expected.y1)**2;cand.append((dist,bb));start+=len(part)
   if cand:found.append(min(cand,key=lambda x:x[0])[1])
  if not found:return expected,False
  b=fitz.Rect(found[0])
  for q in found[1:]:b|=q
  return b,True
 boxes=[(t,*box(t))for t in text_artists()];content=fitz.Rect(boxes[0][1])
 for t,b,exact in boxes[1:]:content|=b
 for ax in fig.axes:
  b=ax.get_window_extent();content|=fitz.Rect(b.x0*72/fig.dpi,height-b.y1*72/fig.dpi,b.x1*72/fig.dpi,height-b.y0*72/fig.dpi)
 coords=[ax.xaxis.label if ax.get_xlabel()else None for ax in fig.axes];coords=[t for t in coords if t]
 if not coords:coords=[t for ax in fig.axes for t in ax.get_xticklabels()if t.get_visible()and t.get_text()]
 coordinate_bottom=max(box(t)[0].y1 for t in coords)
 lb=None
 if legend:
  lbs=[box(t)[0]for t in legend.get_texts()];lb=fitz.Rect(lbs[0])
  for b in lbs[1:]:lb|=b
 result=dict(width_bp=page.rect.width,height_bp=height,content_bbox_bp=list(content),content_top_bp=content.y0,content_bottom_bp=content.y1,legend_box_bp=list(lb)if lb else None,xlabel_bottom_bp=coordinate_bottom,xlabel_to_legend_bp=lb.y0-coordinate_bottom if lb else None,top_margin_mm=content.y0*25.4/72,bottom_margin_mm=(height-content.y1)*25.4/72,all_text_exact=all(e for t,b,e in boxes),unmatched=[t.get_text()for t,b,e in boxes if not e])
 doc.close();return result
initial=measure()
for iteration in range(8):
 f=measure()
 if legend:
  delta=config['xlabel_to_legend_bp']-f['xlabel_to_legend_bp'];an=legend.get_bbox_to_anchor();legend.set_bbox_to_anchor((an.x0/fig.dpi*25.4/W,(an.y0/fig.dpi*25.4-delta*25.4/72)/H),transform=fig.transFigure)
 f=measure();topcrop=f['content_top_bp']-config['top_mm']*72/25.4;bottomcrop=f['height_bp']-f['content_bottom_bp']-config['bottom_mm']*72/25.4
 if max(abs(topcrop),abs(bottomcrop),abs(delta)if legend else 0)<.001:break
 ps=positions();newh=(H*72/25.4-topcrop-bottomcrop)*25.4/72;an=legend.get_bbox_to_anchor()if legend else None;anchor=np.array([an.x0,an.y0])/fig.dpi*25.4 if legend else None;textpos=[(t,np.array(t.get_position())*np.array([W,H]))for t in fig.texts]
 fig.set_size_inches(W/25.4,newh/25.4,forward=True)
 for ax,p in zip(fig.axes,ps):ax.set_position([p[0]/W,(p[1]-bottomcrop*25.4/72)/newh,p[2]/W,p[3]/newh])
 if legend:legend.set_bbox_to_anchor((anchor[0]/W,(anchor[1]-bottomcrop*25.4/72)/newh),transform=fig.transFigure)
 for t,p in textpos:t.set_position((p[0]/W,(p[1]-bottomcrop*25.4/72)/newh))
 H=newh
final=measure();after=science()
def comparable(v):
 v=json.loads(json.dumps(v,ensure_ascii=False))
 if config.get('reflow_ylabel_to_single_vertical_line'):
  for ax in v:ax['ylabel']=ax['ylabel'].replace('\n','')
 return v
assert comparable(before)==comparable(after),'Scientific data or styles changed'
font_after=[(t.get_text(),t.get_fontsize(),t.get_fontfamily(),t.get_fontweight())for t in fig.findobj(Text)if t.get_visible()and t.get_text().strip()];assert [(t.replace('\n',''),*v)for t,*v in font_before]==[(t.replace('\n',''),*v)for t,*v in font_after],'Text or fonts changed'
finalpos=positions()
assert abs(final['top_margin_mm']-config['top_mm'])<.003 and abs(final['bottom_margin_mm']-config['bottom_mm'])<.003,final
if legend:assert abs(final['xlabel_to_legend_bp']-15)<.003,final
for p,q in zip(beforepos,finalpos):assert q[2]>=p[2]-1e-7 and q[3]>=p[3]-1e-7
assert np.allclose(finalpos[0][2:],finalpos[1][2:],rtol=0,atol=1e-6)
assert final['content_bbox_bp'][0]>=-.7 and final['content_bbox_bp'][2]<=W*72/25.4+.7,final
for ext in ['pdf','svg','png']:
 kwargs={'dpi':300}if ext=='png'else {'metadata':{'CreationDate':None,'ModDate':None}}if ext=='pdf'else {'metadata':{'Date':None}}
 original(fig,out/f'figure.{ext}',**kwargs)
def save(name,v):(out/name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
scientifichash=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
save('scientific_artist_values_before_after.json',dict(before=before,after=after,before_sha256=scientifichash(comparable(before)),after_sha256=scientifichash(comparable(after)),exact_except_explicit_ylabel_line_wrap=True))
save('render_geometry.json',dict(status='PASS',number=config['number'],native_mm=[W,H],axes_positions_before_mm=plain(beforepos),axes_positions_after_mm=plain(finalpos),all_axes_same_dimensions=True,no_axis_dimension_reduced=True,scientific_values_and_styles_exact=True,fonts_exact=True,fonts_before=font_before,fonts_after=font_after,pdf_frames_initial=initial,pdf_frames_final=final,history_basis=config['history_basis'],scientific_execution=0))
for q in parent.rglob('*.csv'):
 dst=out/'scientific_inputs'/q.relative_to(parent);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(q.read_bytes())
save('render_receipt.json',dict(status='PASS',number=config['number'],native_parent_render_verified=True,outputs_sha256={f'figure.{e}':hashlib.sha256((out/f'figure.{e}').read_bytes()).hexdigest()for e in ['pdf','svg','png']},scientific_values_and_styles_exact=True,fonts_exact=True,axes_positions_after_mm=plain(finalpos),scientific_execution=0))
print(json.dumps(dict(number=config['number'],status='PASS',native_mm=[W,H],axes_mm=plain(finalpos),spacing=final),ensure_ascii=False))
