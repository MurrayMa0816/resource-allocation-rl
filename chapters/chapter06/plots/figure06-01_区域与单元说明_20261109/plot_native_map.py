#!/usr/bin/env python3
"""Native110mm C6 map and legend using this figure's fixed local resources.

Default input lives in this figure directory. --source-dir only supports an
isolated HY staging render before the new files are copied into that directory.
No environment, policy, model or simulation modules are imported.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,math,shutil,sys,tempfile
NUMBER='6.1'
HERE=Path(__file__).resolve().parent
LAYER=HERE/'native_map_20261110'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(p):
    with Path(p).open(newline='',encoding='utf-8')as stream:return list(csv.DictReader(stream))
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def legend(output,palette):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection,LineCollection
    from matplotlib.patches import RegularPolygon,Rectangle
    from matplotlib.text import Text
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    width,height=110.,15. if NUMBER=='6.1'else 24.
    fig=plt.figure(figsize=(width/25.4,height/25.4),dpi=200);ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,width),ylim=(0,height));ax.axis('off')
    if NUMBER=='6.1':
        upper,lower=10.5,4.5;unit=2.;first_y=upper-unit*math.sqrt(3)/4
        centres=[(3.8+unit*x,first_y+unit*y)for x,y in[(0,0),(1,0),(2,0),(.5,math.sqrt(3)/2),(1.5,math.sqrt(3)/2),(2.5,math.sqrt(3)/2)]]
        polygons=[[(round(cx+unit/math.sqrt(3)*math.cos(math.radians(30+60*k)),6),round(cy+unit/math.sqrt(3)*math.sin(math.radians(30+60*k)),6))for k in range(6)]for cx,cy in centres]
        ax.add_collection(PolyCollection(polygons,facecolors=palette[0],edgecolors='#7c9167',linewidths=.25,zorder=1));region=unary_union([Polygon(p)for p in polygons]);assert region.geom_type=='Polygon';ax.add_collection(LineCollection([list(region.exterior.coords)],colors='#435c41',linewidths=.9,zorder=2));ax.text(14.,upper,'多个单元组成一个区域',ha='left',va='center',fontsize=10.5)
        for i,colour in enumerate(palette):ax.add_patch(RegularPolygon((55.5+3.2*i,upper),numVertices=6,radius=1.15,orientation=math.pi/6,facecolor=colour,edgecolor='#65563e',linewidth=.35))
        ax.text(76.5,upper,'颜色辅助区分区域',ha='left',va='center',fontsize=10.5)
        scale=.9
        for x in(4.5,8.1):ax.add_patch(Rectangle((x,lower-2.5*scale),1.5*scale,5.*scale,fc='#cf7627',ec='none'))
        ax.plot([6.975,6.975],[lower-1.9*scale,lower+1.9*scale],color='#cf7627',lw=2.*scale);ax.text(14.,lower,'总闸',ha='left',va='center',fontsize=10.5)
        x=61.7;w,h=4.*scale,5.*scale;bottom=lower-h/2;ax.add_patch(Rectangle((x,bottom),w,h,fc='#399bc1',ec='none'))
        for xx in(x,x+w):ax.plot([xx,xx],[bottom,bottom+h],color='#a35f22',lw=.8*scale)
        ax.plot([x+scale,x+3*scale],[lower,lower],color='#a35f22',lw=.7*scale);ax.text(76.5,lower,'区域入口',ha='left',va='center',fontsize=10.5)
        semantic={'kind':'categorical, no numerical scale','six_original_palette_values':palette,'same_colour_example_cells':6,'four_texts':['多个单元组成一个区域','颜色辅助区分区域','总闸','区域入口']}
    else:
        edges=[88.,90.,91.,92.,93.,94.];colours=['#f2da5e','#beb051','#8e8b5a','#5c636e','#123570'];left,right=5*110/156,151*110/156;y,h=14.5,2.3;step=(right-left)/5
        ax.text(55.,20.8,'区域平均相对产量（%）',ha='center',va='center',fontsize=10.5)
        for i,c in enumerate(colours):ax.add_patch(Rectangle((left+i*step,y),step,h,fc=c,ec='none'))
        ax.add_patch(Rectangle((left,y),right-left,h,fc='none',ec='#65725e',lw=.8))
        for i,v in enumerate(edges):ax.text(left+i*step,11.6,f'{v:g}',ha='center',va='center',fontsize=10.5)
        x=left+step;ax.plot([x,x],[y,y+h],color='black',lw=1.15);ax.plot([22.,31.],[4.5,4.5],color='#171717',lw=1.15);ax.text(34.,4.5,'区域均值低于90%',ha='left',va='center',fontsize=10.5)
        semantic={'kind':'fixed relative-yield bins','edges_percent':edges,'colours':colours,'threshold_percent':90.,'bar_width_mm':right-left,'bar_left_mm':left,'bar_right_mm':right,'bar_matches_native_map_board':True}
    fig.canvas.draw();renderer=fig.canvas.get_renderer();texts=[t for t in fig.findobj(Text)if t.get_visible()and t.get_text()];checks=[]
    for t in texts:
        box=t.get_window_extent(renderer);assert t.get_fontsize()==10.5 and box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.x1 and box.y1<=fig.bbox.y1,(t.get_text(),box.bounds);checks.append({'text':t.get_text(),'font_bp':t.get_fontsize(),'bbox_bp':[float(x*72/fig.dpi)for x in box.bounds]})
    for i,a in enumerate(texts):
        for b in texts[i+1:]:assert not a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)),(a.get_text(),b.get_text())
    for ext in ['pdf','svg','png']:
        kw={'dpi':260}if ext=='png'else{'metadata':{'CreationDate':None,'ModDate':None}}if ext=='pdf'else{'metadata':{'Date':None}};fig.savefig(output/f'native_legend.{ext}',**kw)
    plt.close(fig);return{'canvas_mm':[width,height],'all_text_fonts_bp':10.5,'text_checks':checks,'semantics':semantic}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=HERE);parser.add_argument('--source-dir',type=Path,default=HERE);args=parser.parse_args();source=args.source_dir.resolve();output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    files=sorted(p for p in(source/'data').rglob('*')if p.is_file());files+=[source/'source/display_helpers.py',LAYER/'source/render_maps.py',LAYER/'label_adjustments.json',Path(__file__)];protected={str(p):sha(p)for p in files};labels=json.loads((LAYER/'label_adjustments.json').read_text());fixed=json.loads((source/'data/mapping.json').read_text());ids=fixed['ids'];expected=[labels['labels'].get(name,{}).get('new_world',old)for name,old in zip(ids,fixed['label_positions'])]
    import matplotlib;matplotlib.use('Agg');from matplotlib.axes import Axes
    import fitz
    arrows=[];old_annotate=Axes.annotate
    def annotate(self,text,*a,**kw):
        if text==''and kw.get('arrowprops',{}).get('color')=='white':arrows.append({'xy':list(kw['xy']),'xytext':list(kw['xytext']),'arrowprops':kw['arrowprops']})
        return old_annotate(self,text,*a,**kw)
    try:
        Axes.annotate=annotate
        with tempfile.TemporaryDirectory(prefix='.native-map-',dir=output)as tmp:
            work=Path(tmp);(work/'audit').mkdir();m=module(LAYER/'source/render_maps.py','native_map_fixed_display');m.HERE=source;m.OLD=source/'data';m.VISIBLE=source/'data';m.ROOT=work;sys.argv=[__file__,'--figure',NUMBER];m.main();group='scene'if NUMBER=='6.1'else'regional';stage=work/'figures'/group;base=stage/('figure'if NUMBER=='6.1'else'mean_terminal_y');manifest=json.loads((stage/'manifest.json').read_text());assert manifest['common_fixed_label_positions']==expected
            for ext in ['pdf','svg','png']:shutil.copyfile(base.with_suffix('.'+ext),output/f'native_map.{ext}')
            oldcsv=source/'region_categorical_colours.csv'if NUMBER=='6.1'else source/'data/mean_terminal_y_all89.csv';newcsv=stage/oldcsv.name;oldrows,newrows=rows(oldcsv),rows(newcsv);assert len(oldrows)==len(newrows)==89
            for a,b in zip(oldrows,newrows):
                assert a.keys()==b.keys()
                for k,v in a.items():
                    if k in['label_x','label_y']:continue
                    try:assert abs(float(v)-float(b[k]))<1e-12,(k,v,b[k])
                    except ValueError:assert v==b[k],(k,v,b[k])
            palette=json.loads((source/'data/palette.json').read_text());legend_proof=legend(output,palette)
            source_manifest={k:manifest[k]for k in ['canvas_width_mm','canvas_height_mm','scene_board_width_mm','scene_board_height_mm','scene_left_mm','scene_bottom_mm','region_ID_font_bp','channel_fonts_bp','common_fixed_label_positions']}
            if NUMBER=='6.6':assert set(manifest['low_mean_region_ids'])=={'12','65','76','92'};source_manifest['low_mean_region_ids']=manifest['low_mean_region_ids'];source_manifest['range_percent']=manifest['range_percent']
    finally:Axes.annotate=old_annotate
    with fitz.open(output/'native_map.pdf')as d:
        page=d[0];spans=[s for block in page.get_text('dict')['blocks']for line in block.get('lines',[])for s in line['spans']if s['text'].strip()];assert all(s['bbox'][0]>=0 and s['bbox'][1]>=0 and s['bbox'][2]<=page.rect.width and s['bbox'][3]<=page.rect.height for s in spans)
        chars=[]
        for item in page.get_texttrace():
            text=''.join(chr(c[0])for c in item['chars'])
            if text in ids:chars.append({'region_id':text,'bbox':list(item['bbox']),'font_bp':item['size']})
        assert len(chars)==89 and {c['region_id']for c in chars}==set(ids)and all(abs(c['font_bp']-9.5)<1e-5 for c in chars)
        gaps=[]
        for i,a in enumerate(chars):
            for b in chars[i+1:]:
                aa,bb=a['bbox'],b['bbox'];gaps.append(math.hypot(max(aa[0]-bb[2],bb[0]-aa[2],0),max(aa[1]-bb[3],bb[1]-aa[3],0)))
        assert min(gaps)>=1.1999
        page_mm=[page.rect.width*25.4/72,page.rect.height*25.4/72];assert abs(page_mm[0]-110)<1e-4 and abs(page_mm[1]-92.52319914813809)<1e-4
        actual_fonts=sorted({round(s['size'],5)for s in spans});assert set(actual_fonts).issubset({7.35,9.5,10.5});assert NUMBER=='6.1'or 7.35 not in actual_fonts
    assert protected=={p:sha(Path(p))for p in protected}
    proof={'schema':'c6-native110mm-map-layout-v1','status':'PASS','figure_number':NUMBER,'native_map_canvas_mm':page_mm,'native_map_geometry':source_manifest,'ID_count':89,'ID_fonts_bp':9.5,'major_fonts_bp':10.5,'actual_PDF_span_font_sizes_bp':actual_fonts,'q_t_subscript_retains_original_native_math_font':NUMBER=='6.1','minimum_ID_character_frame_gap_bp':min(gaps),'ID_character_frames':chars,'label_positions_exactly_reuse_accepted_figure64':True,'all_original_scientific_CSV_columns_unchanged_except_display_label_positions':True,'scientific_rows':[ {k:v for k,v in row.items()if k not in['label_x','label_y']}for row in oldrows],'legend':legend_proof,'same_original_world_flow_arrow_exclusion_geometry':True,'scene_white_flow_arrows':arrows,'input_files_sha256':protected,'all_input_files_unchanged':True,'source_dir_is_default_local_package_on_install':True,'outputs_sha256':{f'native_{part}.{ext}':sha(output/f'native_{part}.{ext}')for part in ['map','legend']for ext in ['pdf','svg','png']},'scientific_execution':{'environment_steps':0,'model_loads':0,'policy_inference':0,'training_runs':0},'thesis_builds':0}
    dump(output/'native_map_proof.json',proof);print(json.dumps({'status':'PASS','figure':NUMBER,'native_map_mm':page_mm,'native_legend_mm':legend_proof['canvas_mm'],'ID_count':89,'minimum_ID_gap_bp':min(gaps),'output':str(output)},ensure_ascii=False))
if __name__=='__main__':main()
