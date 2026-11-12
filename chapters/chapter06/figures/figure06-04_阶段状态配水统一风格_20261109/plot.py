#!/usr/bin/env python3
"""Retain the156mm map family with side channel names and15bp legend rows.

Default inputs are this figure's local data/source and project fonts. The
optional source-dir is only for isolated staging. No scientific runtime loads.
"""
from pathlib import Path

import sys
_FIGURE_SHARED = next(p for p in Path(__file__).resolve().parents if p.name == "figures") / "shared"
if str(_FIGURE_SHARED) not in sys.path:
    sys.path.insert(0, str(_FIGURE_SHARED))
from figure_paths import configure_fonts, fonts_root, validate_output_dir
SHARED_DATA = _FIGURE_SHARED / "data"
import argparse,csv,hashlib,importlib.util,json,math,sys,tempfile
NUMBER='6.4'
HERE=Path(__file__).resolve().parent
LAYER=HERE/'source'
MM=72/25.4
TITLES={0:'(a) 5月30日（返青首日）',46:'(b) 7月15日（孕穗首日）',62:'(c) 7月31日（抽穗首日）',72:'(d) 8月10日（乳熟首日）'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def module(p,name):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def csvrows(p):
    with Path(p).open(newline='',encoding='utf-8')as f:return list(csv.DictReader(f))
def ink(page,region=None):
    """Complete text frames and drawing strokes, including white circle halos."""
    boxes=[]
    for q in page.get_drawings():
        r=fitz.Rect(q['rect']);fill=q.get('fill');stroke=q.get('color')
        if fill is not None and all(v>.999 for v in fill)and stroke is None:continue
        vf=fill is not None and q.get('fill_opacity',1)>0;vs=stroke is not None and q.get('stroke_opacity',1)>0
        if not(vf or vs):continue
        w=float(q.get('width')or 0)if vs else 0.;r+=(-w/2,-w/2,w/2,w/2);boxes.append(r)
    for b in page.get_text('dict')['blocks']:
        for l in b.get('lines',[]):
            for s in l['spans']:
                if s['text'].strip():boxes.append(fitz.Rect(s['bbox']))
    if region is not None:
        selected=[box for box in boxes if box.intersects(region)]
        assert all((region+(-.03,-.03,.03,.03)).contains(box) for box in selected), 'Legend content crosses row boundary'
        boxes=selected
    assert boxes
    return fitz.Rect(min(b.x0 for b in boxes),min(b.y0 for b in boxes),max(b.x1 for b in boxes),max(b.y1 for b in boxes))
def id_frames(page,ids,region=None):
    values=[]
    for item in page.get_texttrace():
        text=''.join(chr(c[0])for c in item['chars']);b=fitz.Rect(item['bbox'])
        if text in ids and(region is None or region.contains(b)):values.append({'region_id':text,'bbox_bp':list(b),'font_bp':item['size']})
    assert len(values)==89 and {v['region_id']for v in values}==set(ids)and all(abs(v['font_bp']-9.5)<1e-4 for v in values)
    gaps=[]
    for i,a in enumerate(values):
        for b in values[i+1:]:
            aa,bb=a['bbox_bp'],b['bbox_bp'];gaps.append(math.hypot(max(aa[0]-bb[2],bb[0]-aa[2],0),max(aa[1]-bb[3],bb[1]-aa[3],0)))
    assert min(gaps)>1.19
    return values,min(gaps)
def compose(mapfile,legendfile,headerfile,output,ids):
    md,ld=fitz.open(mapfile),fitz.open(legendfile);mp,lp=md[0],ld[0];mb,lb=ink(mp),ink(lp);width=mp.rect.width;assert abs(width-lp.rect.width)<1e-4
    pieces=[]
    if headerfile is not None:
        hd=fitz.open(headerfile);hb=ink(hd[0]);pieces.append(('header',hd,fitz.Rect(0,hb.y0,width,hb.y1)))
    split=lp.rect.height/2
    first=ink(lp,fitz.Rect(0,0,width,split));second=ink(lp,fitz.Rect(0,split,width,lp.rect.height))
    pieces.extend([('map',md,fitz.Rect(0,mb.y0,width,mb.y1)),('legend_first',ld,fitz.Rect(0,first.y0,width,first.y1)),('legend_second',ld,fitz.Rect(0,second.y0,width,second.y1))])
    height=3.5*MM+sum(c.height for _,_,c in pieces)+15*(len(pieces)-1)
    doc=fitz.open();page=doc.new_page(width=width,height=height);y=1.5*MM;placements={}
    for name,src,clip in pieces:
        target=fitz.Rect(0,y,width,y+clip.height)
        # Measure each native component before composition. A reused PDF form
        # exposes its clipped-away paths through get_drawings(), so the raw
        # post-composition path union is not a visible-content measurement.
        source_ink=ink(src[0],clip)
        visible=source_ink+fitz.Rect(0,y-clip.y0,0,y-clip.y0)
        page.show_pdf_page(target,src,0,clip=clip,keep_proportion=False)
        placements[name]={'target_bbox_bp':list(target),'source_clip_bp':list(clip),'complete_visible_bounds_bp':list(visible),'physical_scale_xy':[1.,1.]}
        y+=clip.height+15
    doc.set_metadata({'creator':'C6 uncompressed native156mm maps;15bp visible legend composition','producer':'PyMuPDF vector composition;no data or font scaling'});doc.save(output.with_suffix('.pdf'),garbage=4,deflate=True,no_new_id=True);doc.close()
    with fitz.open(output.with_suffix('.pdf'))as final:
        p=final[0]
        visible=[fitz.Rect(v['complete_visible_bounds_bp'])for v in placements.values()]
        bounds=fitz.Rect(min(b.x0 for b in visible),min(b.y0 for b in visible),max(b.x1 for b in visible),max(b.y1 for b in visible))
        assert abs(bounds.y0-1.5*MM)<.03 and abs(p.rect.height-bounds.y1-2*MM)<.03
        ordered=list(placements.values())
        assert all(abs(ordered[i+1]['complete_visible_bounds_bp'][1]-ordered[i]['complete_visible_bounds_bp'][3]-15)<.03 for i in range(len(ordered)-1))
        frames,min_gap=id_frames(p,ids,fitz.Rect(placements['map']['target_bbox_bp']));fonts=sorted({round(s['size'],4)for b in p.get_text('dict')['blocks']for l in b.get('lines',[])for s in l['spans']if s['text'].strip()});assert set(fonts).issubset({7.35,9.5,10.5})
        assert bounds.x0>=0 and bounds.x1<=p.rect.width+.03
        p.get_pixmap(matrix=fitz.Matrix(3,3),alpha=False).save(output.with_suffix('.png'));output.with_suffix('.svg').write_text(p.get_svg_image(text_as_path=False))
        record={'native_canvas_mm':[p.rect.width/MM,p.rect.height/MM],'map_original_canvas_mm':[mp.rect.width/MM,mp.rect.height/MM],'actual_map_axes_mm':[146.,109.21471879190491],'legend_original_canvas_mm':[lp.rect.width/MM,lp.rect.height/MM],'complete_visible_bounds_bp':list(bounds),'top_visible_margin_mm':bounds.y0/MM,'bottom_visible_margin_mm':(p.rect.height-bounds.y1)/MM,'map_to_first_complete_legend_visible_gap_bp':15.,'legend_rows_visible_gap_bp':15.,'channel_names_in_existing_side_gutters':True,'header_to_map_visible_gap_bp':15. if headerfile is not None else None,'ID_count':89,'ID_font_bp':9.5,'major_font_bp':10.5,'actual_PDF_font_sizes_bp':fonts,'minimum_ID_character_frame_gap_bp':min_gap,'ID_character_frames':frames,'placements':placements,'measurement':'Complete text frames and vector strokes, including white circle halos, measured on native source components and translated at physical scale 1:1; legend row clips are checked to contain every selected whole frame/stroke; union excludes clipped-away duplicate PDF-form paths','outputs_sha256':{output.with_suffix('.'+e).name:sha(output.with_suffix('.'+e))for e in ['pdf','svg','png']}}
    md.close();ld.close()
    if headerfile is not None:hd.close()
    return record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--source-dir',type=Path,default=HERE);parser.add_argument('--font-dir',type=Path,help='Original font root containing windows/simsun.ttc and windows/TIMES.TTF');args=parser.parse_args()
    source=args.source_dir.resolve()
    try:
        output=validate_output_dir(__file__,args.output_dir,sources=(source,),font_directory=args.font_dir)
    except ValueError as exc:
        parser.error(str(exc))
    configure_fonts(__file__,args.font_dir,('windows/simsun.ttc','windows/TIMES.TTF'))
    output.mkdir(parents=True,exist_ok=True)
    global fitz
    import fitz
    protected={str(p):sha(p)for p in source.rglob('*')if p.is_file()and '__pycache__'not in p.parts and not any(x.startswith('.render')or x=='runtime_cache'for x in p.parts)}
    fixed=json.loads((SHARED_DATA/'mapping.json').read_text());ids=fixed['ids'];adjustment=json.loads((SHARED_DATA/'label_adjustments.json').read_text());expected=[adjustment['labels'].get(i,{}).get('new_world',old)for i,old in zip(ids,fixed['label_positions'])]
    import matplotlib;matplotlib.use('Agg')
    from matplotlib.patches import Circle
    circles=[];original_circle=Circle.__init__
    def circle(self,*a,**kw):
        original_circle(self,*a,**kw)
        if kw.get('ec')=='#A80058':circles.append(self)
    Circle.__init__=circle
    try:
        with tempfile.TemporaryDirectory(prefix='.fullsize15-',dir=output)as tmp:
            work=Path(tmp);(work/'audit').mkdir();renderer=module(LAYER/'render_maps.py','original156_common_maps');renderer.HERE=source;renderer.OLD=SHARED_DATA;renderer.VISIBLE=SHARED_DATA;renderer.ROOT=work;sys.argv=[__file__,'--figure',NUMBER];renderer.main();group={'6.1':'scene','6.4':'stage_overlay','6.6':'regional'}[NUMBER];stage=work/'figures'/group;manifest=json.loads((stage/'manifest.json').read_text());assert manifest['common_fixed_label_positions']==expected;assert abs(manifest['scene_board_width_mm']-146)<1e-9 and abs(manifest['canvas_width_mm']-156)<1e-9
            if NUMBER=='6.1':
                clarity=module(source/'source/scene_clarity.py','fullsize_clarity');clarity.PARENT=stage;clarity.REVISION=work;sys.argv=[__file__,'--output-dir',str(work/'clarity')];clarity.main();stage=work/'clarity'
            leg=stage/({'6.1':'legend','6.4':'legend','6.6':'mean_terminal_y_legend'}[NUMBER]+'.pdf');specs=[(0,stage/'figure.pdf','fullsize15')]if NUMBER=='6.1'else[(0,stage/'mean_terminal_y.pdf','fullsize15')]if NUMBER=='6.6'else[(day,stage/f'day{day:02}.pdf',f'fullsize15_day{day:02}')for day in [0,46,62,72]]
            geometry={};helper=module(source/'source/display_helpers.py','fullsize_header_helper');plt=helper.plotting(work)
            for day,mapfile,name in specs:
                header=None
                if NUMBER=='6.4':
                    header=work/f'header{day:02}.pdf';fig=plt.figure(figsize=(156/25.4,12/25.4));fig.text(.5,.5,TITLES[day],ha='center',va='center',fontsize=10.5);fig.savefig(header,metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
                geometry[name]=compose(mapfile,leg,header,output/name,ids)
            science={'all_original_input_files_unchanged':True,'input_files_sha256':protected,'fixed89ID_world_positions_reuse_accepted_optimized_layout':True,'fixed_ID_world_positions':expected,'geographic_data_and_colours_source_files_unchanged':True,'original_uncompressed_map_basis_mm':[156.,131.21471879190491],'actual_axes_mm':[146.,109.21471879190491],'raw_manifest':manifest,'restored_display_only_no_new_scientific_execution':True}
            if NUMBER in ['6.1','6.6']:
                oldcsv=source/'region_categorical_colours.csv'if NUMBER=='6.1'else SHARED_DATA/'mean_terminal_y_all89.csv';oldrows,newrows=csvrows(oldcsv),csvrows(stage/oldcsv.name);assert len(oldrows)==len(newrows)==89
                for a,b in zip(oldrows,newrows):
                    assert a.keys()==b.keys()
                    for k,v in a.items():
                        if k in ['label_x','label_y']:continue
                        try:assert abs(float(v)-float(b[k]))<1e-12
                        except ValueError:assert v==b[k]
                science.update(scientific_CSV_columns_unchanged=True,scientific_rows=[{k:v for k,v in a.items()if k not in['label_x','label_y']}for a in oldrows])
                if NUMBER=='6.6':assert set(manifest['low_mean_region_ids'])=={'12','65','76','92'};science['low_mean_region_ids']=manifest['low_mean_region_ids']
            else:
                records=[];index=0;mm_per=fixed['mm_per_geometry_unit']
                for day in [0,46,62,72]:
                    oldrows=[r for r in fixed['rows']if r['day']==day]
                    for row in oldrows:
                        rec={k:row[k]for k in ['day','region_id','allocation_m3','beginning_storage_mm','circle_x','circle_y','radius_geometry_units','drawn']}
                        if row['drawn']:
                            c=circles[index];index+=1;assert max(abs(float(c.center[j])-row[['circle_x','circle_y'][j]])for j in [0,1])<1e-10;assert abs(c.radius-row['radius_geometry_units'])<1e-10;assert c.get_linewidth()==1.5;assert c.get_path_effects()[0]._gc['linewidth']==3.;rmm=c.radius*mm_per;assert abs(rmm-1.6*math.sqrt(row['allocation_m3']/1000))<1e-10;rec.update(actual_radius_mm=rmm,actual_enclosed_area_mm2=math.pi*rmm*rmm)
                        records.append(rec)
                examples=[]
                for v,c in zip([125.,1000.,4000.],circles[index:]):
                    assert abs(c.radius-1.6*math.sqrt(v/1000))<1e-12 and c.get_linewidth()==1.5;examples.append({'value_m3':v,'radius_mm':c.radius,'enclosed_area_mm2':math.pi*c.radius*c.radius})
                assert len(circles)-index==3 and len(records)==356
                science.update(circle_records=records,circle_legend_examples=examples,circle_world_centres_and_radii_exact_original=True,physical_radius1000_mm=1.6,foreground_line_bp=1.5,white_halo_bp=3.,area_to_q_ratio_mm2_per_m3=math.pi*1.6**2/1000,original_mapping_function_unchanged=True,radius_factor_relative_to_filling110_version=156/110,area_factor_relative_to_filling110_version=(156/110)**2,note='Restore original uncompressed map and paper circle marker scale together; no q or area-ratio mapping function change. All fourdates each retain89records and complete own legend.')
    finally:Circle.__init__=original_circle
    assert protected=={p:sha(p)for p in protected}
    source_files=[Path(__file__),LAYER/'render_maps.py',SHARED_DATA/'label_adjustments.json'];bundle={'schema':'c6-original-map-side-labels-template15-v2','status':'PASS','figure_number':NUMBER,'outputs':geometry,'source_files_sha256':{str(p.relative_to(HERE.parent)):sha(p)for p in source_files},'scientific_execution':0,'thesis_builds':0,'active_edits':0};dump(output/'fullsize15_geometry.json',bundle);dump(output/'fullsize15_science.json',science);print(json.dumps({'status':'PASS','figure':NUMBER,'output':str(output),'native_canvases_mm':{k:v['native_canvas_mm']for k,v in geometry.items()}},ensure_ascii=False))
if __name__=='__main__':main()
