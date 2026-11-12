#!/usr/bin/env python3
"""Render the native110mm four-date maps as two readable shared-legend pairs.

Read only this figure package's local fixed data/source and the thesis fonts.
No training, model, environment, inference or evaluation is imported.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,re,shutil,sys,tempfile
import fitz
HERE=Path(__file__).resolve().parent
LAYER=HERE/'layout_pairs_20261110'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name):
    p=LAYER/'source'/f'{name}.py';spec=importlib.util.spec_from_file_location('pairs_'+name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=HERE);args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    parent_inputs=sorted(p for p in (HERE/'data').rglob('*')if p.is_file())+[HERE/'source/display_helpers.py']
    before={str(p.relative_to(HERE)):sha(p)for p in parent_inputs}
    fixed=json.loads((HERE/'data/mapping.json').read_text());labels=json.loads((LAYER/'label_adjustments.json').read_text());ids=fixed['ids'];scale=110/156
    assert len(ids)==89 and labels['shared_across_all_dates']and labels['unchanged_ID_font_bp']==9.5
    with tempfile.TemporaryDirectory(prefix='.render-pairs-',dir=args.output_dir)as tmp:
        work=Path(tmp);(work/'audit').mkdir();renderer=module('render_maps');renderer.ROOT=work;sys.argv=[str(LAYER/'source/render_maps.py'),'--figure','6.4'];renderer.main();maps=work/'figures/stage_overlay';composer=module('compose_pairs');composer.compose(work,maps,HERE)
        for key in ['ab','cd']:
            for ext in ['pdf','svg','png']:shutil.copyfile(work/f'pair_{key}.{ext}',args.output_dir/f'pair_{key}.{ext}')
        views=[]
        for day in [0,46,62,72]:
            with fitz.open(maps/f'day{day:02}.pdf')as d:
                spans=[]
                for item in d[0].get_texttrace():
                    text=''.join(chr(c[0])for c in item['chars'])
                    if text in ids:spans.append({'region_id':text,'bbox':list(item['bbox']),'font_bp':item['size']})
                assert len(spans)==89 and {v['region_id']for v in spans}==set(ids)and all(abs(v['font_bp']-9.5)<1e-5 for v in spans)
                gaps=[]
                for i,a in enumerate(spans):
                    for b in spans[i+1:]:
                        aa,bb=a['bbox'],b['bbox'];dx=max(aa[0]-bb[2],bb[0]-aa[2],0);dy=max(aa[1]-bb[3],bb[1]-aa[3],0);gaps.append(math.hypot(dx,dy))
                assert min(gaps)>=1.1999
                views.append({'day':day,'native_width_mm':d[0].rect.width*25.4/72,'native_height_mm':d[0].rect.height*25.4/72,'ID_count':89,'ID_font_bp':9.5,'minimum_ID_character_frame_gap_bp':min(gaps),'ID_character_frames':spans})
        rendered_mapping=json.loads((maps/'mapping.json').read_text());old_rows={(r['day'],r['region_id']):r for r in fixed['rows']};new_rows={(r['day'],r['region_id']):r for r in rendered_mapping['rows']};leaders=[];circles=[]
        for key,old in old_rows.items():
            new=new_rows[key]
            for field in ['allocation_m3','beginning_storage_mm','circle_x','circle_y','radius_geometry_units','drawn']:assert new[field]==old[field],field
            if old['label_leader']is not None:
                assert new['label_leader'][-1]==old['label_leader'][-1],('region_anchor_changed',key)
                if old['region_id']in labels['labels']:
                    shift=labels['labels'][old['region_id']];delta=[b-a for a,b in zip(shift['old_world'],shift['new_world'])]
                    assert all(abs(new['label_leader'][0][j]-old['label_leader'][0][j]-delta[j])<1e-10 for j in [0,1])
                leaders.append({'day':key[0],'region_id':key[1],'parent_label_end_world':old['label_leader'][0],'new_label_end_world':new['label_leader'][0],'region_anchor_world':new['label_leader'][-1],'region_anchor_exact':True})
            r=old['radius_geometry_units']*fixed['mm_per_geometry_unit']*scale
            circles.append({'day':key[0],'region_id':key[1],'allocation_m3':old['allocation_m3'],'drawn':old['drawn'],'center_world':[old['circle_x'],old['circle_y']],'radius_world':old['radius_geometry_units'],'actual_paper_radius_mm':r,'actual_paper_enclosed_area_mm2':math.pi*r*r})
        proof={'schema':'chapter6-two-date-pairs-native-layout-v1','status':'PASS','native_pair_canvas_mm':[156,225],'native_map_canvas_mm':[110,views[0]['native_height_mm']],'shared_legend_canvas_mm':[110,21.5],'major_font_bp':10.5,'ID_font_bp':9.5,'dates':['5/30','7/15','7/31','8/10'],'map_views':views,'circle_geometry_world_preserved':True,'all89_scientific_rows_each_day_exact':True,'physical_scale_relative_to_parent':scale,'physical_radius_mm_is_parent_times_scale':True,'physical_enclosed_area_mm2_is_parent_times_scale_squared':True,'physical_circle_records':circles,'circle_legend_examples':[{'value_m3':v,'radius_mm':1.6*scale*math.sqrt(v/1000)}for v in[125,1000,4000]],'label_adjustments':labels,'label_leader_endpoints':leaders,'new_internal_label_leader_anchors':labels.get('new_internal_region_anchors',{}),'all_label_badges_clear_four_date_maximum_circle_discs':True,'source_files_sha256':{str(p.relative_to(HERE)):sha(p)for p in [Path(__file__),LAYER/'source/render_maps.py',LAYER/'source/compose_pairs.py',LAYER/'label_adjustments.json']},'parent_inputs_sha256':before,'parent_inputs_unchanged':before=={str(p.relative_to(HERE)):sha(p)for p in parent_inputs},'scientific_execution':0,'outputs':{f'pair_{k}.{e}':sha(args.output_dir/f'pair_{k}.{e}')for k in ['ab','cd']for e in ['pdf','svg','png']},'note':'Original156mm world geometry is projected into a110mm native canvas; text fonts remain9.5/10.5bp. Legacy radius_mm columns belong to parent display. New physical millimetres and areas are explicitly reported here.'}
        assert proof['parent_inputs_unchanged'];(args.output_dir/'pairs_layout_proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','output':str(args.output_dir),'pair_canvas_mm':[156,225],'map_font_bp':9.5,'major_font_bp':10.5},ensure_ascii=False))
if __name__=='__main__':main()
