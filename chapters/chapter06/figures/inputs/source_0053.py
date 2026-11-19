"""Legend-only explanation of the existing89-region C6 scene.

The scene body is inherited byte-for-byte from its accepted active vector
asset. Only this explanatory legend is newly plotted. No science is imported.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os
from pathlib import Path

from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
THESIS=next(p for p in HERE.parents if (p/'assets/fonts').is_dir())
REVISION=HERE/'.scene-output'
PARENT=HERE/'.map-output/figures/scene'
GEOMETRY=HERE/'data/geometry/layout.json'
FONTROOT=THESIS/'assets/fonts/windows'
EXPECTED_PARENT={}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=REVISION/'figures/scene_clarity');args=parser.parse_args()
    out=args.output_dir.resolve()
    if not out.is_relative_to(REVISION):
        raise ValueError('All outputs belong to the new HY display revision')
    out.mkdir(parents=True,exist_ok=True)
    for name,digest in EXPECTED_PARENT.items():
        if sha(PARENT/name)!=digest:raise ValueError('Accepted parent scene changed: '+name)
    EXPECTED_PARENT.update({name:sha(PARENT/name)for name in ('figure.pdf','figure.svg','figure.png','region_categorical_colours.csv','manifest.json')})
    original=json.loads((PARENT/'manifest.json').read_text());geometry=json.loads(GEOMETRY.read_text())
    rows=list(csv.DictReader((PARENT/'region_categorical_colours.csv').open()))
    assert len(rows)==len(geometry['regions'])==89
    palette=original['original_palette']
    for index,(region,row)in enumerate(zip(geometry['regions'],rows)):
        assert region['id']==row['region_id']and int(row['owner'])==region['owner']
        assert row['colour']==palette[region['owner']%len(palette)]
    for name in('figure.pdf','figure.svg','figure.png','region_categorical_colours.csv'):
        target=out/name
        if target.exists()and target.read_bytes()!=(PARENT/name).read_bytes():
            raise ValueError('Refuse to overwrite a different inherited scene body')
        if not target.exists():target.write_bytes((PARENT/name).read_bytes())
    os.environ['MPLCONFIGDIR']=str(REVISION/'runtime_cache/scene_clarity_matplotlib')
    os.environ['XDG_CACHE_HOME']=str(REVISION/'runtime_cache/scene_clarity_xdg')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.collections import PolyCollection,LineCollection
    from matplotlib.patches import RegularPolygon,Rectangle
    from matplotlib.text import Text
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    fonts=[FONTROOT/name for name in('simsun.ttc','TIMES.TTF')]
    for font in fonts:font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':10.5,'svg.fonttype':'none','svg.hashsalt':'c6-scene-region-legend-clarity-v1','pdf.fonttype':42,'savefig.facecolor':'white','axes.unicode_minus':False})
    width,height=156.,28.
    fig=plt.figure(figsize=(width/25.4,height/25.4),dpi=200)
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,width),ylim=(0,height));ax.axis('off')
    # A generic6-cell group demonstrates containment, not a new mapped region.
    unit=2.8;centres=[(7.8+unit*x,19.6+unit*y)for x,y in[(0,0),(1,0),(2,0),(.5,math.sqrt(3)/2),(1.5,math.sqrt(3)/2),(2.5,math.sqrt(3)/2)]]
    polygons=[[(round(cx+unit/math.sqrt(3)*math.cos(math.radians(30+60*k)),6),round(cy+unit/math.sqrt(3)*math.sin(math.radians(30+60*k)),6))for k in range(6)]for cx,cy in centres]
    ax.add_collection(PolyCollection(polygons,facecolors=palette[0],edgecolors='#7c9167',linewidths=.25,zorder=1))
    group=unary_union([Polygon(p)for p in polygons])
    assert group.geom_type=='Polygon'
    boundary=list(group.exterior.coords)
    ax.add_collection(LineCollection([boundary],colors='#435c41',linewidths=.9,zorder=2))
    ax.text(22.,20.8,'多个六角形示意一个灌溉区域',ha='left',va='center',fontsize=10.5)
    # Separate categorical swatches have no value axis and no monotonic scale.
    for i,color in enumerate(palette):
        ax.add_patch(RegularPolygon((85.+3.2*i,20.8),numVertices=6,radius=1.15,orientation=math.pi/6,facecolor=color,edgecolor='#65563e',linewidth=.35))
    ax.text(105.,20.8,'颜色区分灌溉区域',ha='left',va='center',fontsize=10.5)
    # Retain the accepted gate/inlet glyph geometry and colour at native size.
    for x in(34.,38.):ax.add_patch(Rectangle((x,4.),1.5,5.,fc='#cf7627',ec='none'))
    ax.plot([36.75,36.75],[4.6,8.4],color='#cf7627',lw=2.)
    ax.text(43.,6.5,'总闸',ha='left',va='center',fontsize=10.5)
    ax.add_patch(Rectangle((104.,4.),4.,5.,fc='#399bc1',ec='none'))
    for x in(104.,108.):ax.plot([x,x],[4.,9.],color='#a35f22',lw=.8)
    ax.plot([105.,107.],[6.5,6.5],color='#a35f22',lw=.7)
    ax.text(112.,6.5,'灌溉区域入口',ha='left',va='center',fontsize=10.5)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();text_checks=[]
    for text in fig.findobj(Text):
        if not text.get_visible()or not text.get_text():continue
        bb=text.get_window_extent(renderer)
        assert text.get_fontsize()==10.5
        assert bb.x0>=0 and bb.y0>=0 and bb.x1<=fig.bbox.width and bb.y1<=fig.bbox.height
        text_checks.append(dict(text=text.get_text(),font_bp=10.5,bbox_pixels=list(bb.bounds)))
    for extension in('pdf','svg','png'):
        kwargs={'dpi':260}if extension=='png'else{}
        if extension=='pdf':kwargs['metadata']=dict(CreationDate=None,ModDate=None,Creator='C6 inherited scene explanatory legend')
        elif extension=='svg':kwargs['metadata']={'Date':None}
        fig.savefig(out/f'legend.{extension}',**kwargs)
    plt.close(fig)
    for name,digest in EXPECTED_PARENT.items():assert sha(PARENT/name)==digest
    for name in('figure.pdf','figure.svg','figure.png','region_categorical_colours.csv'):assert sha(out/name)==EXPECTED_PARENT[name]
    semantic=dict(region_count=89,service_hex_count=1257,minimum_cells_per_region=min(r['count']for r in geometry['regions']),maximum_cells_per_region=max(r['count']for r in geometry['regions']),fill_semantics='Six categorical colours cycle by source owner modulo6 to assist region recognition. Region boundaries and IDs define ownership; one hexagon is not one region.',colour_assignment='palette[region.owner % 6]',palette=palette,colour_counts=[sum(r['owner']%6==i for r in geometry['regions'])for i in range(6)],colour_reuse=True,adjacent_same_colour_region_pairs=[['7','26'],['64','76']],nominal_display_area_mu_per_service_cell=10,display_area_mu=12570,original_actual_area_mu=12618,area_calculations_use_original_unrounded_regions=True,mini_group=dict(cell_count=6,purpose='Generic containment legend only; not a new measured region or a new allocation observation',same_fill=True,deep_outer_boundary=True))
    snapshot=REVISION/'source/scene_clarity.py';snapshot.parent.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve()
    if source!=snapshot:
        if snapshot.exists()and snapshot.read_bytes()!=source.read_bytes():raise ValueError('Existing source snapshot differs')
        if not snapshot.exists():snapshot.write_bytes(source.read_bytes())
    inputs={str(PARENT/name):digest for name,digest in EXPECTED_PARENT.items()}
    inputs[str(GEOMETRY)]=sha(GEOMETRY);inputs.update({str(font):sha(font)for font in fonts})
    record=dict(status='CANDIDATE_FOR_NATIVE_PDF_REVIEW',source=str(source),source_sha256=sha(source),snapshot_source=str(snapshot),snapshot_sha256=sha(snapshot),input_files_sha256=inputs,inputs_unchanged=True,figure_body_inherited_exact_bytes=True,inherited_body_source=original['source'],inherited_body_source_sha256=original['source_sha256'],inherited_map_canvas_mm=[original['canvas_width_mm'],original['canvas_height_mm']],inherited_scene_board_width_mm=original['scene_board_width_mm'],legend_canvas_mm=[width,height],ordinary_legend_font_bp=10.5,legend_text_checks=text_checks,semantics=semantic,outputs_sha256={name:sha(out/name)for name in('figure.pdf','figure.svg','figure.png','legend.pdf','legend.svg','legend.png','region_categorical_colours.csv')},scientific_execution=dict(environment_steps=0,model_loads=0,inference_calls=0,training_runs=0,new_TEST=0),TeX_edits=0,index_edits=0,paper_builds=0,current_asset_or_pointer_modified=False)
    dump(out/'manifest.json',record)
    (out/'plot.py').write_text('from pathlib import Path\nimport runpy,sys\nscript=Path('+repr(str(snapshot))+')\nsys.argv=[str(script),"--output-dir",str(Path(__file__).resolve().parent)]+sys.argv[1:]\nrunpy.run_path(str(script),run_name="__main__")\n')
    dump(REVISION/'audit/scene_colour_group_semantics.json',semantic)
    print(json.dumps(dict(status='CANDIDATE_READY',output=str(out),map_body_exact=True,legend_mm=[width,height],region_count=89),ensure_ascii=False))


if __name__=='__main__':main()
