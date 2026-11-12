"""Offline print presentation derivative of the currently accepted native figures.

Only method dash patterns and numeric object-label font size/centering change.
Scientific arrays, geometry, axes, bands, patches, colors and marker identities stay.
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,importlib.util,json,os,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
HERE=Path(__file__).resolve().parent.parent
THESIS=next(p for p in HERE.parents if (p/'assets/fonts/windows').is_dir())

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load_module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--number',required=True);parser.add_argument('--output-dir',type=Path);args=parser.parse_args()
 spec=json.loads((ROOT/'print_spec.json').read_text())
 n=spec['number'];folder=HERE;inp=ROOT/'native_inputs';out=args.output_dir.resolve();out.mkdir(exist_ok=True,parents=True)
 cache=out/'runtime'/n
 for k,d in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
  q=cache/d;q.mkdir(exist_ok=True,parents=True);os.environ[k]=str(q)
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 from matplotlib.figure import Figure
 from matplotlib.lines import Line2D
 from matplotlib.text import Text
 from matplotlib.patches import Polygon
 from PIL import Image
 def clean(a):
  if isinstance(a,np.ndarray):return clean(a.tolist())
  if isinstance(a,np.generic):return a.item()
  if isinstance(a,(list,tuple)):return[clean(x)for x in a]
  if isinstance(a,dict):return{k:clean(v)for k,v in a.items()}
  if isinstance(a,(float,int,str,bool))or a is None:return a
  return str(a)
 def digest(a):return hashlib.sha256(json.dumps(clean(a),sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
 def science(fig):
  return [dict(position=list(ax.get_position().bounds),xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),xscale=ax.get_xscale(),yscale=ax.get_yscale(),xticks=clean(ax.get_xticks()),yticks=clean(ax.get_yticks()),
    lines=[dict(x=clean(l.get_xdata()),y=clean(l.get_ydata()),color=clean(l.get_color()),lw=l.get_linewidth(),alpha=l.get_alpha(),marker=l.get_marker(),ms=l.get_markersize(),mfc=clean(l.get_markerfacecolor()),mec=clean(l.get_markeredgecolor()),markevery=clean(l.get_markevery()))for l in ax.lines],
    collections=[dict(paths=[dict(vertices=clean(p.vertices),codes=clean(p.codes))for p in c.get_paths()],offsets=clean(c.get_offsets()),fc=clean(c.get_facecolors()),ec=clean(c.get_edgecolors()),lw=clean(c.get_linewidths()),alpha=c.get_alpha(),array=clean(c.get_array()),clim=clean(c.get_clim()))for c in ax.collections],
    patches=[dict(vertices=clean(p.get_path().vertices),codes=clean(p.get_path().codes),fc=clean(p.get_facecolor()),ec=clean(p.get_edgecolor()),lw=p.get_linewidth())for p in ax.patches],
    images=[dict(array=clean(i.get_array()),extent=clean(i.get_extent()),clim=clean(i.get_clim()))for i in ax.images])for ax in fig.axes]
 def style(line):return dict(linestyle=line.get_linestyle(),dash_pattern=clean(getattr(line,'_unscaled_dash_pattern',None)),marker=line.get_marker())
 c3={'#0072b2':'-','#d55e00':'--','#009e73':'-.','#cc79a7':(0,(5,1,1,1)),'#cc3311':':','#7b5ea7':(0,(3,1,1,1,1,1)),'#555555':'--'}
 c4={'#d55e00':'-','#0072b2':'--','#009e73':'-.','#7b5ea7':':','#cc79a7':(0,(5,1,1,1)),'#666666':(0,(3,1,1,1,1,1))}
 c5={'#154f80':'-','#dd873a':'--','#319664':'-.','#9368a6':':','#c9494d':(0,(5,1,1,1)),'#498f9d':(0,(3,1,1,1,1,1))}
 palette=c3 if n.startswith('3.')else c4 if n.startswith('4.')else c5
 save_original=Figure.savefig;done=set();proof={};fig_record=[]
 def save(fig,path,*a,**kw):
  if id(fig)not in done:
   done.add(id(fig))
   # Preserve the current metric terminology before comparing to its accepted PNG.
   for t in fig.findobj(Text):
    if t.get_text()in spec['source_metadata'].get('mapping',{}):t.set_text(spec['source_metadata']['mapping'][t.get_text()])
   fig.canvas.draw();before=science(fig);before_sha=digest(before)
   parent_png=ROOT/'native_parent.png';base_png=out/'native_parent_reproduction.png'
   pngkw={k:v for k,v in kw.items()if k not in ('metadata','format')};save_original(fig,base_png,format='png',**pngkw)
   aa=np.asarray(Image.open(base_png).convert('RGBA'));bb=np.asarray(Image.open(parent_png).convert('RGBA'))
   equal=aa.shape==bb.shape and np.array_equal(aa,bb)
   assert equal,(n,'native parent PNG differs',aa.shape,bb.shape,int(np.count_nonzero(aa!=bb))if aa.shape==bb.shape else None)
   fig.canvas.draw();renderer=fig.canvas.get_renderer();changed_lines=[];changed_ids=[]
   if spec['method_line_patterns']:
    for line in fig.findobj(Line2D):
     color=matplotlib.colors.to_hex(line.get_color()).lower()
     if color not in palette or line.get_linestyle().lower()in('none','')or (line.get_alpha()is not None and line.get_alpha()<.3):continue
     # Tick marks and point-only errorbar caps must keep their original geometry.
     if len(np.atleast_1d(line.get_xdata()))<2:continue
     old=style(line);line.set_linestyle(palette[color]);changed_lines.append(dict(label=line.get_label(),color=color,before=old,after=style(line)))
   if spec['resize_IDs']:
    for ax in fig.axes:
     for t in ax.texts:
      if t.get_text().isdigit()and abs(t.get_fontsize()-7.5)<1e-7:
       oldbb=t.get_window_extent(renderer);oldposition=list(t.get_position());oldmid=np.array([(oldbb.x0+oldbb.x1)/2,(oldbb.y0+oldbb.y1)/2]);t.set_fontsize(9.0)
       newbb=t.get_window_extent(renderer);newmid=np.array([(newbb.x0+newbb.x1)/2,(newbb.y0+newbb.y1)/2]);transform=t.get_transform();pos=np.asarray(t.get_position());t.set_position(transform.inverted().transform(transform.transform(pos)+oldmid-newmid))
       changed_ids.append(dict(axis=fig.axes.index(ax),text=t.get_text(),old_font_bp=7.5,new_font_bp=9.0,old_position=oldposition,new_position=list(t.get_position()),artist=t))
   fig.canvas.draw();after=science(fig);assert digest(after)==before_sha,(n,'scientific numeric artists changed')
   renderer=fig.canvas.get_renderer();bbox_records=[];collisions=[];min_gap=float('inf');hex_containment=[]
   for row in changed_ids:
    t=row.pop('artist');b=t.get_window_extent(renderer);row['bbox_bp']=[v*72/fig.dpi for v in (b.x0,b.y0,b.x1,b.y1)];bbox_records.append((row['axis'],t.get_text(),b))
    assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.x1 and b.y1<=fig.bbox.y1,(n,'ID outside canvas',row)
    # Test against the nearest complete hexagonal cell in display coordinates.
    ax=fig.axes[row['axis']];center=np.array([(b.x0+b.x1)/2,(b.y0+b.y1)/2]);cells=[]
    for patch in ax.patches:
     if isinstance(patch,Polygon)and len(patch.get_path().vertices)==7:
      hexpath=patch.get_path().transformed(patch.get_transform());v=hexpath.vertices[:-1];mid=v.mean(0);cells.append((np.linalg.norm(mid-center),hexpath))
    assert cells,(n,'no hex cell for ID',row['text'])
    cell=min(cells,key=lambda z:z[0])[1];corners=np.array([[b.x0,b.y0],[b.x0,b.y1],[b.x1,b.y0],[b.x1,b.y1]])
    inside=bool(cell.contains_points(corners,radius=1e-7).all());hex_containment.append(dict(axis=row['axis'],ID=row['text'],layout_bbox_inside_hex=inside))
    assert inside,(n,'enlarged label outside complete hex',row['text'])
   for i,(axid,txt,b)in enumerate(bbox_records):
    for ax2,txt2,c in bbox_records[i+1:]:
     if axid!=ax2:continue
     ox=max(0,min(b.x1,c.x1)-max(b.x0,c.x0));oy=max(0,min(b.y1,c.y1)-max(b.y0,c.y0))
     if ox>1e-7 and oy>1e-7:collisions.append([txt,txt2,ox*oy])
     gap=max(max(b.x0-c.x1,c.x0-b.x1),max(b.y0-c.y1,c.y0-b.y1))*72/fig.dpi;min_gap=min(min_gap,gap)
   assert not collisions,(n,'object ID collisions',collisions[:10])
   expected={'2.8':660,'3.7':400,'3.8':480,'5.9':440}.get(n,0);assert len(changed_ids)==expected,(n,len(changed_ids),expected)
   proof.update(schema='thesis-print-figure-presentation-v1',number=n,parent_native_png_pixels_equal=True,scientific_arrays_geometry_before_sha256=before_sha,scientific_arrays_geometry_after_sha256=digest(after),scientific_arrays_geometry_equal=True,axes_positions_ranges_ticks_unchanged=True,patches_and_bands_and_colors_and_markers_unchanged=True,figure_inches=list(fig.get_size_inches()),method_line_patterns=changed_lines,object_ID_font_bp=9.0 if expected else None,object_ID_count=len(changed_ids),object_ID_changes=changed_ids,ID_bbox_collisions=collisions,all_ID_layout_bboxes_inside_complete_hex=all(x['layout_bbox_inside_hex']for x in hex_containment),hex_containment=hex_containment,minimum_ID_bbox_gap_bp=None if not expected else min_gap,new_environment_steps=0,model_loads=0,policy_inferences=0,learning_steps=0)
   dump(out/'print_presentation_proof.json',proof);fig_record.append(fig)
  return save_original(fig,path,*a,**kw)
 Figure.savefig=save
 try:
  sys.path.insert(0,str(inp));sys.modules.pop('presentation',None);sys.modules.pop('presentation_style',None)
  m=load_module(inp/spec['native_source'],'native_'+n.replace('.','_'));kind=spec['kind']
  if kind=='c2case':sys.argv=['native','--output-dir',str(out)];m.main()
  elif kind=='c3shared':m.setup(cache,THESIS);m.render(inp,out)
  elif kind=='c3case':m.render(m.DEFAULT_DOMAIN,out)
  elif kind=='c4random':
   m.HERE=inp;m.PUBLICATION=ROOT;m.ROOT=ROOT
   parent=m.load_parent();fig,pts,cfg=m.build_parent(parent);x,values,points,expected=m.random_values(parent);m.replace_only_props(fig,x,expected);m.save_fig(fig,out);parent.plt.close(fig)
  elif kind=='c4prefix':
   m.HERE=inp;m.ROOT=ROOT;m.setup();cfg=json.loads((inp/'figure_config.json').read_text());data=list(csv.DictReader((inp/'data.csv').open()));fig,points=m.prefix(data,cfg);fig.set_dpi(200)
   for ext in ['pdf','png','svg']:fig.savefig(out/('figure.'+ext),dpi=200,metadata={'CreationDate':None,'ModDate':None}if ext=='pdf'else({'Date':None}if ext=='svg'else None))
   m.plt.close(fig)
  elif kind=='c5shared':m.ROOT=ROOT;m.render(spec['source_metadata']['role'],inp/'data',out)
  elif kind=='c5case':m.OUT=out;m.DATA=inp/'data';m.build(m.source_data())
  else:raise ValueError(kind)
 finally:Figure.savefig=save_original
 assert len(fig_record)==1,(n,len(fig_record))
 # Native renderers' pre-derivative checks are retained as inherited evidence.
 for name in ['checks.json','source.json','presentation_proof.json']:
  p=out/name
  if p.exists():p.rename(out/('inherited_native_'+name))
 if (out/'display_aggregates.csv.gz').exists():
  parent=Path(spec['parent_directory'])/'display_aggregates.csv.gz'
  assert gzip.decompress((out/'display_aggregates.csv.gz').read_bytes())==gzip.decompress(parent.read_bytes());(out/'display_aggregates.csv.gz').write_bytes(parent.read_bytes())
 for name in ['data.csv','data.csv.gz','period_totals.csv','current_display_points.csv','plotted_points.csv','figure_config.json','provenance.json']:
  p=inp/name
  if p.exists()and not(out/name).exists():(out/name).symlink_to(p)
 with Image.open(out/'figure.png')as im:im.convert('L').save(out/'figure_grayscale.png')
 outputs={f'figure.{e}':sha(out/f'figure.{e}')for e in ['pdf','svg','png']}
 dump(out/'source.json',dict(schema='thesis-print-readability-derivative-v1',number=n,scientific_values_changed=False,parent_directory=spec['parent_directory'],parent_pdf_sha256=spec['parent_files_sha256']['figure.pdf'],editable_source='plot.py',render_engine=str(ROOT/'render_print_figures.py'),native_source='inputs/'+spec['native_source'],input_files_sha256=spec['input_files_sha256'],changes=['distinct dash patterns preserving markers/colors'if spec['method_line_patterns']else None,'object IDs 7.5 -> 9bp, centered, no collisions'if spec['resize_IDs']else None],proof_sha256=sha(out/'print_presentation_proof.json'),outputs=outputs,new_environment_steps=0,model_loads=0,policy_inferences=0,training_runs=0))
 (out/'README.rst').write_text('打印可读性呈现派生\n==================\n\n本图由inputs内保存的当前原生绘图源与CSV重绘，先逐像素复现当前父PNG，再只修改方法线型或对象编号字号。科学数组、阴影、图元几何、坐标、颜色和marker保持；详细证据见print_presentation_proof.json。复绘：python plot.py --output-dir NEW_HY_SUBDIR。无环境、模型、推理或训练。\n')
 print(json.dumps({'number':n,'status':'PASS','output':str(out),'changed_IDs':proof['object_ID_count'],'changed_lines':len(proof['method_line_patterns']),'scientific_equal':True},ensure_ascii=False))
if __name__=='__main__':main()
