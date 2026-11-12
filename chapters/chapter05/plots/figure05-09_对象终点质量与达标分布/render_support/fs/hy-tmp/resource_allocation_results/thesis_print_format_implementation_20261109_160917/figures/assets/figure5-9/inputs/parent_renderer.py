"""C5 saved terminal outcomes: four independent bright hex maps.

This offline figure reads byte-identical local CSVs and frozen provenance only.
No policy, model, environment, inference or training code is imported or run.
"""
from pathlib import Path
import os,json,csv,hashlib,math,argparse
BASE=Path(__file__).resolve().parent
DATA=BASE/'data'
OUT=BASE
os.environ.setdefault('MPLCONFIGDIR','/hy-tmp/resource_allocation_results/experiments_consistency_five_rounds_20261108_205458/round1/chapter5/cache/matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,colors,cm
from matplotlib.patches import Polygon,Rectangle
from matplotlib.transforms import Bbox
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from shapely.geometry import Polygon as GeoPolygon
from shapely.ops import unary_union

FONT=Path('/root/gamewar/latex_thesis_resource_allocation/assets/fonts/windows')
for name in ['simsun.ttc','simhei.ttf','TIMES.TTF','TIMESBD.TTF']:
    font_manager.fontManager.addfont(str(FONT/name))
plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':10.5,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','axes.unicode_minus':False,
    'savefig.facecolor':'white','svg.hashsalt':'formal-hex-c5','axes.linewidth':.60,'path.simplify':False})
INK='#1F2B33';EDGE='#788992';RED='#8F1D2C'
WIDTH,HEIGHT=15.6,16.0
R=.335
ID_FONT=7.5
PT_CM=2.54/72
COLORS=['#F8FCFF','#DDF1FB','#B0DFF3','#78C5E7']
CMAP=colors.LinearSegmentedColormap.from_list('bright_terminal_quality',COLORS)
NORM=colors.Normalize(0,1,clip=False)
SQRT3=math.sqrt(3)
ID_PROP_REGULAR=FontProperties(fname=str(FONT/'TIMES.TTF'))
ID_PROP_BOLD=FontProperties(fname=str(FONT/'TIMESBD.TTF'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def text(ax,x,y,label,size=10.5,ha='left',bold=False,color=INK,**kw):
    return ax.text(x,y,label,fontsize=size,ha=ha,va='center',color=color,
        fontfamily=['Times New Roman','SimHei' if bold else 'SimSun'],**kw)


def source_data():
    manifest=json.loads((DATA/'c5_terminal_data_sources.json').read_text())
    datasets=[]
    for spec in manifest['figures']:
        domain=spec['domain'];file=DATA/f'c5_{domain}_seed0_case0_terminal_quality.csv'
        rows=list(csv.DictReader(file.open()))
        assert sha(file)==spec['csv_sha256']
        ids=np.array([int(row['object_id']) for row in rows]);assert np.array_equal(ids,np.arange(1,spec['N']+1))
        left=np.array([float(row['full_quality']) for row in rows]);right=np.array([float(row['only_j_quality']) for row in rows])
        threshold=np.array([float(row['threshold']) for row in rows]);assert np.all(threshold==spec['tau'])
        assert all(np.isfinite(v).all() and np.all(v>=0) and np.all(v<=1) for v in [left,right])
        saved_l=np.array([int(row['full_achieved']) for row in rows],dtype=bool)
        saved_r=np.array([int(row['only_j_achieved']) for row in rows],dtype=bool)
        assert np.array_equal(saved_l,left>=threshold) and np.array_equal(saved_r,right>=threshold)
        raw=[]
        for method,values,saved in [('full',left,saved_l),('only_j',right,saved_r)]:
            item=next(s for s in spec['sources'] if s['method']==method)
            count=int(np.count_nonzero(values>=spec['tau']));mean=float(values.mean())
            assert count==item['K']
            assert math.isclose(mean,item['J'],rel_tol=0,abs_tol=1e-14)
            raw.append(dict(method=method,path=item['path'],sha256=item['sha256'],K=count,J=mean,
                            local_values_SHA_bound_to_previously_verified_saved_quality=True))
        datasets.append(dict(domain=domain,N=len(rows),ids=ids,left=left,right=right,
            threshold=float(spec['tau']),left_ok=saved_l,right_ok=saved_r,
            csv=str(file),csv_sha256=sha(file),sources=raw,paired_identity=spec['paired_identity']))
    return datasets


def polygon(cx,cy,r=R):
    angles=np.pi/6+np.arange(6)*np.pi/3
    return np.column_stack([cx+r*np.cos(angles),cy+r*np.sin(angles)])


def id_label(ax,cx,cy,label,achieved):
    # Center each real regular/bold font's own digit ink, avoiding faux bold.
    prop=ID_PROP_BOLD if achieved else ID_PROP_REGULAR
    font_file=FONT/('TIMESBD.TTF' if achieved else 'TIMES.TTF')
    color=RED if achieved else INK
    bounds=TextPath((0,0),str(label),size=ID_FONT,prop=prop).get_extents()
    ox=cx-(bounds.x0+bounds.x1)/2*PT_CM
    oy=cy-(bounds.y0+bounds.y1)/2*PT_CM
    artist=ax.text(ox,oy,str(label),fontsize=ID_FONT,fontproperties=prop,
        ha='left',va='baseline',color=color,zorder=6)
    corners=np.array([[bounds.x0,bounds.y0],[bounds.x1,bounds.y0],
                      [bounds.x1,bounds.y1],[bounds.x0,bounds.y1]])*PT_CM+np.array([ox,oy])
    return artist,GeoPolygon(corners),corners,font_file,color


def build(datasets):
    fig=plt.figure(figsize=(WIDTH/2.54,HEIGHT/2.54))
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,WIDTH),ylim=(0,HEIGHT));ax.set_aspect('equal');ax.axis('off')
    display=[];label_checks=[]
    for row,ds in enumerate(datasets):
        domain=ds['domain'];row_top=[14.75,7.89][row]
        title_y=[15.13,8.27][row]
        cb_y=[9.15,1.25][row]
        task='通信' if domain=='radio' else '柴油'
        xy=np.array([[SQRT3*R*(i%10+.5*((i//10)%2)),-1.5*R*(i//10)] for i in range(ds['N'])])
        local_vertices=np.concatenate([polygon(*p) for p in xy],axis=0)
        xmin,ymin=local_vertices.min(axis=0);xmax,ymax=local_vertices.max(axis=0)
        method_maps=[]
        for col,method in enumerate(['full','only_j']):
            center_x=[3.86,11.74][col]
            method_name='SeDVal' if method=='full' else '仅平均成效目标'
            letter='abcd'[2*row+col]
            text(ax,center_x,title_y,f'({letter}) {task}·{method_name}',ha='center',bold=False)
            shift=np.array([center_x-(xmin+xmax)/2,row_top-ymax])
            positions=xy+shift
            values=ds['left'] if method=='full' else ds['right']
            flags=ds['left_ok'] if method=='full' else ds['right_ok']
            records=[];geoms=[]
            for k,point in enumerate(positions):
                cx,cy=map(float,point);vertices=polygon(cx,cy)
                geom=GeoPolygon(vertices);geoms.append(geom)
                ax.add_patch(Polygon(vertices,closed=True,facecolor=CMAP(NORM(values[k])),edgecolor=EDGE,lw=.37,zorder=1))
                achieved=bool(flags[k])
                number,label_geom,label_corners,font_file,color=id_label(ax,cx,cy,int(ds['ids'][k]),achieved)
                assert geom.covers(label_geom)
                clearance=geom.boundary.distance(label_geom)/PT_CM
                label_checks.append(dict(domain=domain,method=method,object_id=int(ds['ids'][k]),
                    achieved=achieved,font_file=str(font_file),ID_color=color,ink_to_hex_boundary_pt=clearance))
                records.append(dict(object_id=int(ds['ids'][k]),position_cm=[cx,cy],terminal_quality_raw=float(values[k]),
                    achieved=achieved,fill_rgba=list(CMAP(NORM(values[k]))),full_hex_polygon_cm=vertices.tolist(),
                    ID_ink_bbox_corners_cm=label_corners.tolist(),ID_font_file=str(font_file),
                    ID_font_weight='true bold' if achieved else 'regular',ID_color=color))
            union=unary_union(geoms)
            overlap=sum(g.area for g in geoms)-union.area;assert abs(overlap)<1e-8
            bx0,by0,bx1,by1=union.bounds
            ax.add_patch(Rectangle((bx0,by0),bx1-bx0,by1-by0,facecolor='none',edgecolor='#000000',lw=.60,zorder=4))
            source=next(s for s in ds['sources'] if s['method']==method)
            method_maps.append(dict(method=method,method_label=method_name,N=ds['N'],K=source['K'],J=source['J'],
                map_bounds_cm=[bx0,by0,bx1,by1],hex_overlap_area_cm2=overlap,
                records=records))
        assert np.allclose(np.array([r['position_cm'] for r in method_maps[0]['records']])-
                           np.array([r['position_cm'] for r in method_maps[1]['records']]),[-7.88,0],rtol=0,atol=1e-12)
        # One common terminal-quality scale for the two maps in this row.
        scale=1.0 if domain=='radio' else 100.0
        cax=fig.add_axes([4.10/WIDTH,cb_y/HEIGHT,10.75/WIDTH,.13/HEIGHT])
        ticks=[0,.2,.4,.6,.8,1] if scale==1 else [0,20,40,60,80,100]
        cb=fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0,scale),cmap=CMAP),cax=cax,orientation='horizontal',ticks=ticks)
        cb.outline.set_linewidth(.45);cb.outline.set_edgecolor('#000000');cb.ax.tick_params(labelsize=10.5,length=2,pad=2)
        cb.ax.set_xticklabels(['0','0.2','0.4','0.6','0.8','1'] if scale==1 else ['0','20','40','60','80','100'])
        text(ax,.72,cb_y+.065,'SSIM' if scale==1 else '需求完成比例 / %')
        display.append(dict(domain=domain,N=ds['N'],threshold_raw=ds['threshold'],
            raw_quality_bounds=[0,1],colorbar_display_bounds=[0,scale],maps=method_maps,
            original_ID_order='original object_id1..N in row-major10-column grid; no spatial/quality sorting'))
    fig.canvas.draw()
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:
        metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None
        fig.savefig(OUT/f'figure.{ext}',dpi=240,metadata=metadata,
                    bbox_inches=Bbox.from_extents(0,47/240,WIDTH/2.54,HEIGHT/2.54),pad_inches=0)
    plt.close(fig)
    red_bold=[c for c in label_checks if c['achieved']]
    black_regular=[c for c in label_checks if not c['achieved']]
    assert len(red_bold)==312 and len(black_regular)==128
    assert all(Path(c['font_file']).name=='TIMESBD.TTF' and c['ID_color']==RED for c in red_bold)
    assert all(Path(c['font_file']).name=='TIMES.TTF' and c['ID_color']==INK for c in black_regular)
    report=dict(status='PASS_FOR_VALUES_IDS_THRESHOLDS_GEOMETRY_AND_TRUE_FONT_STATUS',
        unique_original_objects=220,full_hex_cells=440,independent_maps=4,figure_size_cm=[WIDTH,HEIGHT-47/240*2.54],original_canvas_before_bottom_trim_cm=[WIDTH,HEIGHT],
        bottom_blank_trim_cm=47/240*2.54,
        complete_regular_shared_edge_hexagons=True,hex_radius_cm=R,
        achieved_red_true_bold_ID_count=len(red_bold),unmet_black_regular_ID_count=len(black_regular),
        achieved_rule='terminal quality >= task threshold',
        achieved_status='deep red original ID set with real TIMESBD.TTF',
        unmet_status='black original ID set with real TIMES.TTF',
        all_actual_ID_ink_boxes_within_whole_hex=True,
        minimum_actual_ID_ink_to_hex_boundary_clearance_pt=min(c['ink_to_hex_boundary_pt'] for c in label_checks),
        font_sources={name:dict(path=str(FONT/name),sha256=sha(FONT/name)) for name in ['TIMES.TTF','TIMESBD.TTF']},
        ID_font_pt=ID_FONT,main_font_pt=10.5,ID_order_unchanged=True,no_quality_or_spatial_sorting=True,
        palette=COLORS,red=RED,color_meaning='Terminal quality, with one common fixed color scale per task.',display=display)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=BASE)
    args=parser.parse_args()
    OUT=args.output_dir.resolve()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'checks').mkdir(parents=True,exist_ok=True)
    datasets=source_data();mapping=build(datasets)
    # Arrays remain in their immutable CSVs; emit only a small mapping summary.
    compact_maps=[dict(domain=d['domain'],N=d['N'],threshold_raw=d['threshold_raw'],raw_quality_bounds=d['raw_quality_bounds'],colorbar_display_bounds=d['colorbar_display_bounds'],maps=[{k:v for k,v in view.items() if k!='records'} for view in d['maps']]) for d in mapping['display']]
    data_check=dict(status='PASS_FOR_ORIGINAL_SAVED_VALUES_IDS_THRESHOLDS_KJ',selection='unchanged formal seed0/case0/standard budget',
        tasks=[dict(domain=d['domain'],N=d['N'],csv=d['csv'],csv_sha256=d['csv_sha256'],sources=d['sources'],paired_identity=d['paired_identity']) for d in datasets],
        local_CSV_bytes_bound_to_previously_raw_verified_quality=True,
        achieved_flags_recomputed_independently_from_raw_values=True,data_files_after_sha={d['domain']:sha(d['csv']) for d in datasets},
        model_loads=0,environment_steps=0,new_inference_calls=0,learning_updates=0)
    (OUT/'checks/data_geometry_check.json').write_text(json.dumps(dict(data=data_check,geometry={k:v for k,v in mapping.items() if k!='display'},maps=compact_maps),ensure_ascii=False,indent=2)+'\n')
    caption=dict(title='两种学习目标下的对象终点质量与达标分布',
        caption='上排为通信终点，下排为柴油终点；左列为SeDVal，右列为仅平均成效目标。每个完整六角对应原编号对象，同一任务两图的编号及位置一致，填色共用相同终点质量色标。红色粗体编号表示达标，黑色常规编号表示未达标。通信按SSIM≥0.20、柴油按需求完成比例≥70%计为达标。',
        selection='既有正式实验seed0/case0、标准资源条件下的配对终测案例，原选例、计划与条件保持。单案例不代表总体优劣；图形邻接与距离无地理含义。',
        metrics=[dict(domain=d['domain'],N=d['N'],threshold=d['threshold'],methods=[dict(method=s['method'],K=s['K'],J=s['J']) for s in d['sources']]) for d in datasets],
        body_suggestions=['填色表示单对象终点质量，红色粗体编号仅标示是否达到本任务门槛。','通信SeDVal和仅平均成效目标分别达标80个与67个，平均SSIM分别为0.5485和0.5481。','柴油SeDVal和仅平均成效目标分别达标89个与76个，平均需求完成比例分别为70.30%和72.85%。','图仅补充展示既有配对案例的对象分布，统计结论仍以原多种子实验为依据。'])
    (OUT/'caption.json').write_text(json.dumps(caption,ensure_ascii=False,indent=2)+'\n')
    (OUT/'source.json').write_text(json.dumps(dict(status='RENDERED_PENDING_FIVE_ROUND_REVIEW',
        editable_source=str(Path(__file__).resolve()),data_source='Byte-identical local terminal CSV, previously verified against frozen raw evaluation records; only local CSV/provenance needed for redraw',
        scope='Formal C5 figure asset from retained paired seed0/case0 values; no thesis writes',files={n:sha(OUT/n) for n in ['figure.pdf','figure.svg','figure.png','checks/data_geometry_check.json','caption.json']},
        model_loads=0,environment_steps=0,new_inference_calls=0,learning_updates=0),ensure_ascii=False,indent=2)+'\n')
    from PIL import Image
    Image.open(OUT/'figure.png').convert('L').save(OUT/'checks/figure_grayscale.png')
    print(json.dumps(dict(status='RENDERED_PENDING_FIVE_ROUND_REVIEW',outputs=str(OUT),geometry={k:v for k,v in mapping.items() if k!='display'}),ensure_ascii=False))
