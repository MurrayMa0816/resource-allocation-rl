"""Offline Figure4.4: remove only its two gray horizontal comparison lines.

Replays the frozen accepted renderer with private cache/output locations, checks
pixel-exact reconstruction of the accepted spaced parent, and removes its two
identified Line2D artists. No model or environment modules are loaded.
"""
from pathlib import Path
import argparse,copy,hashlib,io,json,os,shutil,sys,types
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
TASK=HERE.parents[1]
PARENT=Path('/hy-tmp/resource_allocation_results/thesis_figure_spacing_unification_20261110_000617/assets/figure44')
FROZEN=HERE/'accepted_inputs'
Y_VALUES=[45.271407152182945,75.59877076662723]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def digest(d):return hashlib.sha256(json.dumps(d,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()

def main():
 ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--output-dir',type=Path,default=HERE);args=ap.parse_args();out=args.output_dir.resolve();assert out.is_relative_to(TASK) and not(out/'figure.pdf').exists(),out;out.mkdir(parents=True,exist_ok=True)
 # All writable replay/cache paths are private descendants of the new output.
 rr=out/'runtime/replay';(rr/'audit').mkdir(parents=True,exist_ok=True);shutil.copyfile(FROZEN/'baseline_figures.json',rr/'audit/baseline_figures.json')
 src=(FROZEN/'spacing_renderer.py').read_text();changes={
 "ROOT=Path('/hy-tmp/resource_allocation_results/thesis_figure_spacing_unification_20261110_000617')":f'ROOT=Path({str(rr)!r})',
 "MAIN=Path('/hy-tmp/resource_allocation_results/thesis_main_learning_style_20261109')":f'MAIN=Path({str(FROZEN/"main_chain")!r})',
 '  captured.append(fig)':'  captured.append(fig); EXPORTED_FIGURES.append(fig); EXPORTED_SCIENTIFIC.append(scientific)',
 }
 for old,new in changes.items():assert src.count(old)==1,old;src=src.replace(old,new)
 mod=types.ModuleType('frozen_spacing_44');mod.__file__=str(FROZEN/'spacing_renderer.py');mod.EXPORTED_FIGURES=[];mod.EXPORTED_SCIENTIFIC=[];sys.modules[mod.__name__]=mod;exec(compile(src,mod.__file__,'exec'),mod.__dict__)
 replay=rr/'assets/figure44';argv=sys.argv[:]
 try:sys.argv=['frozen-spacing','--number','4.4','--output-dir',str(replay)];mod.main()
 finally:sys.argv=argv
 assert len(mod.EXPORTED_FIGURES)==len(mod.EXPORTED_SCIENTIFIC)==1
 import numpy as np
 from PIL import Image
 import matplotlib
 import fitz
 fig=mod.EXPORTED_FIGURES[0];scientific=mod.EXPORTED_SCIENTIFIC[0]
 accepted=np.asarray(Image.open(PARENT/'figure.png').convert('RGBA'));reproduced=np.asarray(Image.open(replay/'figure.png').convert('RGBA'));assert accepted.shape==reproduced.shape and np.array_equal(accepted,reproduced),'Accepted parent pixels differ'
 before=scientific(fig);sealed=json.loads((FROZEN/'parent_numeric_artist_values_before_after.json').read_text())['after'];assert before==sealed,'Accepted final artist values differ'
 before_positions=[list(np.array(ax.get_position().bounds)*np.array([*fig.get_size_inches(),*fig.get_size_inches()])*25.4)for ax in fig.axes]
 removed=[];line_rows=[]
 for i,ax in enumerate(fig.axes):
  assert len(ax.lines)==13
  line=ax.lines[-1];record=before[i]['lines'][-1]
  assert record['x']==[0,1] and record['y']==[Y_VALUES[i],Y_VALUES[i]] and record['color']=='#888888' and record['linewidth']==1.25
  assert line.get_transform()==ax.get_yaxis_transform()
  y=fig.get_figheight()*300-ax.transData.transform((0,Y_VALUES[i]))[1]*300/fig.dpi
  bounds=ax.get_window_extent();x0=bounds.x0*300/fig.dpi;x1=bounds.x1*300/fig.dpi;line_rows.append((y,x0,x1));removed.append(record);line.remove()
 after=scientific(fig);expected=copy.deepcopy(before)
 for d in expected:d['lines']=d['lines'][:-1]
 assert after==expected,'A scientific value or style other than the reference lines changed'
 after_positions=[list(np.array(ax.get_position().bounds)*np.array([*fig.get_size_inches(),*fig.get_size_inches()])*25.4)for ax in fig.axes];assert before_positions==after_positions
 matplotlib.rcParams['svg.hashsalt']='ch4-figure44-reference-removal-20261110'
 for ext in ['pdf','png','svg']:
  kw={'dpi':300}
  if ext=='pdf':kw['metadata']={'CreationDate':None,'ModDate':None}
  if ext=='svg':kw['metadata']={'Date':None}
  fig.savefig(out/('figure.'+ext),**kw)
 new=np.asarray(Image.open(out/'figure.png').convert('RGBA'));assert new.shape==accepted.shape
 diff=np.any(new!=accepted,axis=2);allow=np.zeros(diff.shape,dtype=bool)
 for y,x0,x1 in line_rows:allow[max(0,int(y)-6):min(diff.shape[0],int(y)+7),max(0,int(x0)-2):min(diff.shape[1],int(x1)+3)]=True
 assert not np.any(diff & ~allow),'Raster changes outside the two reference-line bands'
 olddoc=fitz.open(PARENT/'figure.pdf');newdoc=fitz.open(out/'figure.pdf');assert olddoc[0].rect==newdoc[0].rect
 oldtext=olddoc[0].get_text('rawdict');newtext=newdoc[0].get_text('rawdict');assert oldtext==newtext,'PDF text, fonts or frames changed'
 geometry=json.loads((FROZEN/'parent_render_geometry.json').read_text());geometry.update(schema='chapter4-reference-line-removal-v1',numeric_before_sha256=digest(before),numeric_after_sha256=digest(after),only_reference_lines_removed=True,removed_reference_lines=removed,scientific_artist_values_and_styles_exact_except_removed_lines=True,parent_PNG_reconstruction_pixel_exact=True,PDF_text_dictionary_byte_equal=True,changed_pixel_count=int(diff.sum()),changed_pixels_confined_to_reference_line_bands=True,environment_steps=0,model_loads=0,policy_inference=0,training_runs=0)
 # The old all-artists equality flag described a layout-only derivative; the
 # new claim identifies the exact two authorized line removals instead.
 geometry.pop('scientific_artist_values_and_styles_exact',None)
 proof=dict(before=before,after=after,before_sha256=digest(before),after_sha256=digest(after),only_removed_lines=removed,all_other_artist_values_styles_and_axes_exact=True)
 dump(out/'numeric_artist_values_before_after.json',proof);dump(out/'scientific_before_after.json',proof);dump(out/'render_geometry.json',geometry)
 manifest=json.loads((FROZEN/'scientific_input_manifest.json').read_text())
 for name,value in manifest.items():
  srcfile=HERE/name;assert sha(srcfile)==value
  dst=out/name
  if dst!=srcfile:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(srcfile,dst)
 inputs={str(p.relative_to(HERE)):sha(p)for p in FROZEN.rglob('*')if p.is_file()}
 dump(out/'source.json',dict(schema='chapter4-reference-line-removal-v1',number='4.4',parent_directory=str(PARENT),parent_pdf_sha256=sha(PARENT/'figure.pdf'),renderer=str(HERE/'plot.py'),renderer_sha256=sha(HERE/'plot.py'),frozen_inputs_sha256=inputs,scientific_files_sha256=manifest,removed_reference_lines=removed,all_other_artists_exact=True,layout_and_PDF_text_exact=True,outputs_sha256={f'figure.{e}':sha(out/f'figure.{e}')for e in ['pdf','png','svg']},environment_steps=0,model_loads=0,policy_inference=0,training_runs=0))
 (out/'README.rst').write_text('图4.4水平比较线移除\n====================\n\n只删除通信与柴油两轴的灰色水平参考线。原曲线（包括柴油PROPS已接受完整曲线）、SD、数值、样式、图例、坐标范围与156毫米布局全部保持；删除前父PNG逐像素重现，删后差异仅限两条线带，PDF文字/字体/字框逐项相同。\n复绘：python '+str(HERE/'plot.py')+' --output-dir 本修订根下的新目录。冻结输入位于accepted_inputs；全部可写缓存和重现输出只在指定目录runtime内。0环境、模型、推理、训练。\n')
 print(json.dumps(dict(status='PASS',output=str(out),removed_lines=len(removed),changed_pixels=int(diff.sum()),pdf_sha256=sha(out/'figure.pdf'),png_sha256=sha(out/'figure.png'),svg_sha256=sha(out/'figure.svg')),ensure_ascii=False))
if __name__=='__main__':main()
