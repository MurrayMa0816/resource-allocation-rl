"""Offline two-panel derivative of accepted Chapter4 update diagnostics."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,io,json,os,sys,types
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for key,leaf in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
 q=ROOT/'runtime'/leaf;q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def plain(v):
  if isinstance(v,np.ndarray):return plain(v.tolist())
  if isinstance(v,np.generic):return v.item()
  if isinstance(v,(tuple,list)):return[plain(x)for x in v]
  if isinstance(v,dict):return{str(k):plain(x)for k,x in v.items()}
  if isinstance(v,(int,float,str,bool))or v is None:return v
  return str(v)

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

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=HERE);args=ap.parse_args();out=args.output_dir.resolve()
 assert out.is_relative_to(ROOT);out.mkdir(parents=True,exist_ok=True)
 inp=HERE/'accepted_inputs';source=(inp/'parent_renderer.py').read_text().replace('HERE=Path(__file__).resolve().parent',f'HERE=Path({str(HERE)!r})').replace('ROOT=HERE.parents[1]',f'ROOT=Path({str(ROOT)!r})')
 parent=types.ModuleType('accepted_control_parent');parent.__file__=str(inp/'parent_renderer.py');exec(compile(source,parent.__file__,'exec'),parent.__dict__)
 parent.setup();fig,points=parent.control_figure(parent.read('data.csv'))
 expected=json.loads((inp/'numeric_artist_values_before_after.json').read_text())['after'][:2]
 before=scientific(fig)[:2];assert before==expected,'Parent A/B scientific artists differ'
 for ax in list(fig.axes)[2:]:fig.delaxes(ax)
 original=json.loads((inp/'render_geometry.json').read_text());positions=original['axes_positions_after_mm'][:2];width=156.;height=80.
 fig.set_size_inches(width/25.4,height/25.4,forward=True)
 for ax,pos in zip(fig.axes,positions):
  ax.set_title(ax.get_title(),loc='center',pad=6);ax.xaxis.labelpad=4;ax.yaxis.labelpad=4;ax.tick_params(axis='both',which='major',pad=3)
  ax.set_position([pos[0]/width,24/height,pos[2]/width,pos[3]/height])
 legend=fig.legends[0];legend.set_bbox_to_anchor((.5,2/height),transform=fig.transFigure)
 def norm(s):return ''.join(s.split()).replace('−','-')
 def measure():
  fig.canvas.draw();buf=io.BytesIO();fig.savefig(buf,format='pdf',metadata={'CreationDate':None,'ModDate':None});doc=fitz.open(stream=buf.getvalue(),filetype='pdf');page=doc[0];lines=[]
  for block in page.get_text('rawdict')['blocks']:
   for line in block.get('lines',[]):
    chars=[c for span in line['spans'] for c in span.get('chars',[]) if not c['c'].isspace()];lines.append((norm(''.join(c['c'] for c in chars)),chars))
  def boxes(label):
   label=norm(label);result=[]
   for text,chars in lines:
    start=0
    while (start:=text.find(label,start))>=0:
     rect=fitz.Rect(chars[start]['bbox'])
     for c in chars[start:start+len(label)]:rect|=fitz.Rect(c['bbox'])
     result.append(rect);start+=len(label)
   assert result,label
   return result
  xboxes=boxes('训练步数（×10³）');lboxes=[b for t in legend.get_texts() for b in boxes(t.get_text())];titles=[b for ax in fig.axes for b in boxes(ax.get_title())]
  result=dict(width_bp=page.rect.width,height_bp=page.rect.height,top_margin_mm=min(b.y0 for b in titles)*25.4/72,bottom_margin_mm=(page.rect.height-max(b.y1 for b in lboxes))*25.4/72,xlabel_to_legend_bp=min(b.y0 for b in lboxes)-max(b.y1 for b in xboxes))
  doc.close();return result
 for iteration in range(4):
  m=measure();delta=8-m['xlabel_to_legend_bp'];anchor=legend.get_bbox_to_anchor();anchor_y=anchor.y0/fig.dpi*25.4-delta*25.4/72
  legend.set_bbox_to_anchor((.5,anchor_y/height),transform=fig.transFigure);m=measure()
  topcrop=m['top_margin_mm']-1.5;bottomcrop=m['bottom_margin_mm']-2;newheight=height-topcrop-bottomcrop
  pos=[np.asarray(ax.get_position().bounds)*[width,height,width,height] for ax in fig.axes];anchor=legend.get_bbox_to_anchor();anchor_y=anchor.y0/fig.dpi*25.4
  fig.set_size_inches(width/25.4,newheight/25.4,forward=True)
  for ax,b in zip(fig.axes,pos):ax.set_position([b[0]/width,(b[1]-bottomcrop)/newheight,b[2]/width,b[3]/newheight])
  legend.set_bbox_to_anchor((.5,(anchor_y-bottomcrop)/newheight),transform=fig.transFigure);height=newheight
 final=measure();assert abs(final['top_margin_mm']-1.5)<.003 and abs(final['bottom_margin_mm']-2)<.003 and abs(final['xlabel_to_legend_bp']-8)<.003,final
 after=scientific(fig);assert before==after,'Kept A/B scientific values/styles changed'
 for suffix in ['pdf','png','svg']:
  kwargs={'dpi':300}
  if suffix=='pdf':kwargs['metadata']={'CreationDate':None,'ModDate':None}
  if suffix=='svg':kwargs['metadata']={'Date':None}
  fig.savefig(out/('figure.'+suffix),**kwargs)
 with(inp/'plotted_points.csv').open(newline='')as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=[row for row in reader if row['panel'] in ['a','b']]
 with(out/'plotted_points.csv').open('w',newline='')as f:writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 digest=lambda v:hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
 dump(out/'numeric_artist_values_before_after.json',dict(before=before,after=after,before_sha256=digest(before),after_sha256=digest(after),parent_first_two_axes_exact=True))
 dims=[(np.asarray(ax.get_position().bounds)*[width,height,width,height]).tolist() for ax in fig.axes]
 for b,c in zip(dims,positions):assert np.allclose(b[2:],c[2:],rtol=0,atol=1e-8)
 dump(out/'render_geometry.json',dict(status='PASS',number='4.8',native_mm=[width,height],axes_positions_after_mm=dims,axes_physical_dimensions_exact=True,scientific_artist_values_and_styles_exact=True,pdf_frames_final=final,top_content_to_file_top_mm=final['top_margin_mm'],content_bottom_to_file_bottom_mm=final['bottom_margin_mm'],xlabel_to_legend_bp=final['xlabel_to_legend_bp'],row_gaps_bp=[],legend_labels=[t.get_text() for t in legend.get_texts()],font_bp=10.5,environment_steps=0,model_loads=0,training_runs=0))
 dump(out/'source.json',dict(schema='chapter4-update-diagnostic-two-panel-v1',number='4.8',historical_number='4.9',kept_panels=['a','b'],removed_panels=['c','d'],parent_directory=str(json.loads((ROOT/'paths.json').read_text())['parent']),parent_first_two_axes_exact=True,scientific_data_sha256={name:sha(HERE/name)for name in ['data.csv','exceedance_per_seed.csv','exceedance_summary.csv']},output_sha256={name:sha(out/name)for name in ['figure.pdf','figure.png','figure.svg','plotted_points.csv']},environment_steps=0,model_loads=0,training_runs=0))
 print(json.dumps(dict(status='PASS',native_mm=[width,height],rows=len(rows),frames=final),ensure_ascii=False))
if __name__=='__main__':main()
