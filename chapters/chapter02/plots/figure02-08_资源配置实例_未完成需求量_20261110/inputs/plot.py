"""Six retained decision snapshots; common within-task state/area scales."""
from pathlib import Path
import argparse,csv,hashlib,json,math,os,sys
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(BASE.parents[1]/'cache'/'c2_multitime_mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager,colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon,Rectangle,Circle,Ellipse

FONT=Path('/root/gamewar/latex_thesis_resource_allocation/assets/fonts/windows')
for name in ['simsun.ttc','TIMES.TTF']:font_manager.fontManager.addfont(str(FONT/name))
plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':10.5,
    'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path',
    'svg.hashsalt':'thesis-c2-shared-scale-three-timepoints','savefig.facecolor':'white'})

W,H=15.6,20.0
MAP_HEIGHT=5.3
MAP_TOPS=[19.1,12.9,6.7]
PERIODS={'radio':[1,16,32],'v5':[1,12,24]}
MAX_R=.30
CIRCLE_DY=-.59
ID_BASELINE_DY=-.085
PALETTES={'radio':{'fill_points':['#F2F8FC','#BDD9EA','#6E9FBC'],'circle_color':'#8B0000'},
          'v5':{'fill_points':['#FFF8EE','#F0D2A8','#C9955A'],'circle_color':'#003399'}}
STATE={d:LinearSegmentedColormap.from_list('state_'+d,p['fill_points'])for d,p in PALETTES.items()}
SCALES={'radio':{'state_vmin':0.,'state_vmax':300.,'allocation_reference':.8,'allocation_unit':'mW','state_ticks':[0,100,200,300],'circle_refs':[.2,.4,.8]},
        'v5':{'state_vmin':0.,'state_vmax':1000.,'allocation_reference':50.,'allocation_unit':'L','state_ticks':[0,500,1000],'circle_refs':[10,25,50]}}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rect(x,y,w,h):return[x/W,y/H,w/W,h/H]
def text(fig,x,y,s,size=10.5,ha='center'):
    return fig.text(x/W,y/H,s,ha=ha,va='baseline',fontsize=size,color='#111111')
def geometry(n,cols):
    ids=np.arange(n);xy=np.column_stack([np.sqrt(3)*(ids%cols+.5*((ids//cols)%2)),-1.5*(ids//cols)])
    angles=np.pi/6+np.arange(6)*np.pi/3;unit=np.column_stack([np.cos(angles),np.sin(angles)])
    polygons=xy[:,None,:]+unit[None,:,:];allv=polygons.reshape(-1,2)
    bounds=[float(allv[:,0].min()),float(allv[:,0].max()),float(allv[:,1].min()),float(allv[:,1].max())]
    return xy,polygons,bounds

def map_width(domain):
    _,_,bounds=geometry(100 if domain=='radio'else 120,10 if domain=='radio'else 12)
    return MAP_HEIGHT*(bounds[1]-bounds[0])/(bounds[3]-bounds[2])

def map_x(domain):
    return .45 if domain=='radio'else W-.45-map_width(domain)

def panel(fig,domain,period,row,rows):
    radio=domain=='radio';n=100 if radio else 120;s=SCALES[domain]
    assert len(rows)==n and[int(r['object_id'])for r in rows]==list(range(1,n+1))
    assert all((int(r['training_seed']),int(r['test_path_index']),int(r['period']))==(0,0,period)for r in rows)
    state=np.array([float(r['channel_power']if radio else r['remaining_demand'])for r in rows])
    native=np.array([float(r['action_native'])for r in rows]);allocation=native*(1000 if radio else 1)
    assert np.isfinite(state).all()and np.isfinite(allocation).all()and(state>=0).all()and(allocation>=0).all()
    assert state.max()<=s['state_vmax'] and allocation.max()<=s['allocation_reference']
    assert abs(allocation.sum()-(15 if radio else 840))<1e-10
    cols=10 if radio else 12
    xy,polygons,bounds=geometry(n,cols);xmin,xmax,ymin,ymax=bounds
    width=map_width(domain);height=MAP_HEIGHT;top=MAP_TOPS[row];bottom=top-height;x=map_x(domain)
    ax=fig.add_axes(rect(x,bottom,width,height));norm=colors.Normalize(s['state_vmin'],s['state_vmax'])
    ids=[];radius=MAX_R*np.sqrt(allocation/s['allocation_reference']);circle_rows=[]
    for i,(pt,poly,value,amount,radius_i)in enumerate(zip(xy,polygons,state,allocation,radius),1):
        ax.add_patch(Polygon(poly,closed=True,facecolor=STATE[domain](norm(value)),edgecolor='#77838A',lw=.30,clip_on=False))
        ids.append(ax.text(pt[0],pt[1]+ID_BASELINE_DY,str(i),ha='center',va='baseline',fontsize=7.5,color='#111111',zorder=4))
        center=(float(pt[0]),float(pt[1]+CIRCLE_DY))
        if amount>0:ax.add_patch(Circle(center,float(radius_i),facecolor='none',edgecolor=PALETTES[domain]['circle_color'],lw=.85,clip_on=False,zorder=3))
        circle_rows.append({'object_id':i,'state':float(value),'allocation_display':float(amount),'center':list(center),'radius_hex_units':float(radius_i),
                            'enclosed_area_hex_units':float(np.pi*radius_i**2),'zero_allocation':bool(amount==0),'allocation_glyph_drawn':bool(amount>0)})
    ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax);ax.set_aspect('equal');ax.axis('off')
    ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,facecolor='none',edgecolor='#000000',lw=.55,clip_on=False,zorder=2))
    letter=('abc'if radio else'def')[row]
    title_string=f'({letter}) '+('通信功率分配'if radio else'应急柴油分配')+f' 第{period}'+('轮'if radio else'期')
    title=text(fig,x+width/2,top+.20,title_string)
    return{'domain':domain,'period_one_based':period,'stage':['early','middle','late'][row],'N':n,'object_ids':list(range(1,n+1)),
      'positions':xy.tolist(),'cell_polygons':polygons.tolist(),'bounds':bounds,'frame_cm':[x,bottom,width,height],
      'grid_columns':cols,'grid_rows':10,
      'panel_label':letter,'title':title_string,'domain_palette':PALETTES[domain],
      'state_facecolors_rgba':[list(STATE[domain](norm(value)))for value in state],
      'state':state.tolist(),'allocation_native':native.tolist(),'allocation_display':allocation.tolist(),'allocation_unit':s['allocation_unit'],
      'state_norm':[s['state_vmin'],s['state_vmax']],'allocation_reference':s['allocation_reference'],
      'radii_hex':radius.tolist(),'circle_centers':[r['center']for r in circle_rows],
      'area_per_unit_hex':float(np.pi*MAX_R**2/s['allocation_reference']),
      'allocation_total':float(allocation.sum()),'positive_count':int(np.count_nonzero(allocation>0)),'zero_count':int(np.count_nonzero(allocation==0)),
      'state_nonzero_but_zero_action':int(np.count_nonzero((state>0)&(allocation==0))),'max_radius_hex_units':MAX_R,'radius_formula':'r_i = 0.30 sqrt(a_i / task_reference); shared task_reference for all three periods; no positive floor',
      'circle_rows':circle_rows,'circle_center_offset':[0,CIRCLE_DY],'ID_baseline_offset':[0,ID_BASELINE_DY],
      '_ax':ax,'_ids':ids,'_title':title}

def shared_legend(fig,domain):
    s=SCALES[domain];x=map_x(domain);width=map_width(domain);norm=colors.Normalize(s['state_vmin'],s['state_vmax'])
    bar_width=width*.46;circle_left=x+width*.53;circle_width=width*.47
    text(fig,x+bar_width/2,.90,'信道模平方'if domain=='radio'else'未完成任务量')
    cax=fig.add_axes(rect(x,.57,bar_width,.16));cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=STATE[domain]),cax=cax,orientation='horizontal',ticks=s['state_ticks'])
    cb.ax.tick_params(labelsize=10.5,length=2,pad=1.5);cb.outline.set_linewidth(.45);cb.outline.set_edgecolor('#000000')
    text(fig,circle_left+circle_width/2,.90,'功率（mW）'if domain=='radio'else'柴油投入（L）')
    xy,polygons,bounds=geometry(100 if domain=='radio'else 120,10 if domain=='radio'else 12);unit_cm=width/(bounds[1]-bounds[0]);records=[]
    for cx,value in zip([circle_left+circle_width*f for f in [.12,.50,.88]],s['circle_refs']):
        radius=MAX_R*math.sqrt(value/s['allocation_reference'])
        fig.add_artist(Ellipse((cx/W,.62/H),2*radius*unit_cm/W,2*radius*unit_cm/H,transform=fig.transFigure,facecolor='none',edgecolor=PALETTES[domain]['circle_color'],lw=.85))
        text(fig,cx,.10,f'{value:g}')
        records.append({'value':value,'radius_hex_units':radius,'radius_cm':radius*unit_cm,'reference_uses_same_scale_as_maps':True})
    return{'domain':domain,'state_vmin':s['state_vmin'],'state_vmax':s['state_vmax'],'allocation_reference':s['allocation_reference'],
           'allocation_unit':s['allocation_unit'],'MAX_R':MAX_R,'domain_palette':PALETTES[domain],'map_width_cm':width,'common_map_height_cm':MAP_HEIGHT,'map_unit_cm':unit_cm,'area_per_unit_cm2':float(np.pi*MAX_R**2/s['allocation_reference']*unit_cm**2),'area_per_unit_hex':float(np.pi*MAX_R**2/s['allocation_reference']),'circle_legend':records}

def check_geometry(fig,record):
    ax=record.pop('_ax');texts=record.pop('_ids');title=record.pop('_title');renderer=fig.canvas.get_renderer()
    normals=np.column_stack([np.cos(np.arange(6)*np.pi/3),np.sin(np.arange(6)*np.pi/3)])
    apothem=np.sqrt(3)/2;minimum_gap=float('inf');minimum_cell_margin=float('inf');minimum_circle_margin=float('inf');proof=[]
    unit_pixels=float(np.linalg.norm(ax.transData.transform([1,0])-ax.transData.transform([0,0])))
    unit_pixels_y=float(np.linalg.norm(ax.transData.transform([0,1])-ax.transData.transform([0,0])))
    assert abs(unit_pixels-unit_pixels_y)<1e-9,'Hex units must be isotropic'
    p0=np.array(record['cell_polygons'][0]);pp=ax.transData.transform(p0)
    edges=np.linalg.norm(np.roll(pp,-1,axis=0)-pp,axis=1)*2.54/fig.dpi
    assert np.ptp(edges)<1e-12,'Physical hex edges differ'
    actual=ax.get_position().bounds;actual_cm=[actual[0]*W,actual[1]*H,actual[2]*W,actual[3]*H]
    assert np.max(np.abs(np.array(actual_cm)-record['frame_cm']))<1e-9
    record['frame_actual_cm']=actual_cm
    x,y,w,h=actual_cm;record['frame_native_pdf_rect_pt']=[x*72/2.54,(H-y-h)*72/2.54,(x+w)*72/2.54,(H-y)*72/2.54]
    record['regular_hex']={'isotropic_axis_units':True,'unit_scale_x_cm':unit_pixels*2.54/fig.dpi,'unit_scale_y_cm':unit_pixels_y*2.54/fig.dpi,'physical_edge_lengths_cm':edges.tolist(),'edge_length_range_cm':float(np.ptp(edges))}
    stroke_halfwidth_units=(.85/2)*fig.dpi/72/unit_pixels
    for pt,t,row in zip(np.array(record['positions']),texts,record['circle_rows']):
        box=t.get_window_extent(renderer=renderer);bbox=ax.transData.inverted().transform([[box.x0,box.y0],[box.x1,box.y1]])
        corners=np.array([[bbox[0,0],bbox[0,1]],[bbox[0,0],bbox[1,1]],[bbox[1,0],bbox[0,1]],[bbox[1,0],bbox[1,1]]])
        margin=float(np.min(apothem-(corners-pt)@normals.T));assert margin>=-1e-8,('ID outside hex',record['domain'],record['period_one_based'],row['object_id'],margin)
        center=np.array(row['center']);radius=row['radius_hex_units'];r=radius+stroke_halfwidth_units if row['allocation_glyph_drawn']else 0
        circle_margin=float(np.min(apothem-((center-pt)@normals.T+r)))
        if row['allocation_glyph_drawn']:
            assert circle_margin>=-1e-8,('circle outside hex',row['object_id'])
            minimum_circle_margin=min(minimum_circle_margin,circle_margin)
        gap=float(bbox[0,1]-(center[1]+r));assert gap>0,('ID/circle overlap',record['domain'],record['period_one_based'],row['object_id'],gap)
        minimum_gap=min(minimum_gap,gap);minimum_cell_margin=min(minimum_cell_margin,margin)
        proof.append({'object_id':row['object_id'],'ID_bbox_data':bbox.tolist(),'ID_inside_hex':True,
                      'allocation_glyph_drawn':row['allocation_glyph_drawn'],'circle_including_stroke_inside_hex':True if row['allocation_glyph_drawn']else None,
                      'ID_circle_gap_hex_units':gap if row['allocation_glyph_drawn']else None})
    amounts=np.array(record['allocation_display']);radii=np.array(record['radii_hex']);positive=amounts>0
    ratios=np.pi*radii[positive]**2/amounts[positive];assert np.max(np.abs(ratios-record['area_per_unit_hex']))<1e-12
    assert(radii[~positive]==0).all()
    record['verification']={'all_original_IDs_in_order':True,'all_IDs_inside_own_hex':True,'all_circles_including_stroke_inside_own_hex':True,
      'all_IDs_disjoint_from_circle_including_stroke':True,'minimum_ID_circle_gap_hex_units':minimum_gap,'minimum_ID_hex_margin':minimum_cell_margin,
      'minimum_circle_hex_margin':minimum_circle_margin,'stroke_halfwidth_hex_units':stroke_halfwidth_units,'area_proportionality_max_error':float(np.max(np.abs(ratios-record['area_per_unit_hex']))),
      'positive_radius_floor_applied':False,'object_geometry_checks':proof}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=BASE);args=parser.parse_args()
    provenance=json.loads((BASE/'provenance.json').read_text());assert sha(BASE/'data.csv')==provenance['data_sha256']
    with (BASE/'data.csv').open(encoding='utf-8-sig',newline='')as f:rows=list(csv.DictReader(f))
    assert len(rows)==660
    fig=plt.figure(figsize=(W/2.54,H/2.54))
    panels=[]
    for row in range(3):
        for domain in ['radio','v5']:
            period=PERIODS[domain][row];data=sorted([r for r in rows if r['domain']==domain and int(r['period'])==period],key=lambda r:int(r['object_id']))
            panels.append(panel(fig,domain,period,row,data))
    shared=[shared_legend(fig,domain)for domain in ['radio','v5']]
    fig.canvas.draw()
    for record in panels:check_geometry(fig,record)
    unit_values=[p['regular_hex']['unit_scale_x_cm']for p in panels]+[p['regular_hex']['unit_scale_y_cm']for p in panels]
    assert np.ptp(unit_values)<1e-12,'Both tasks must use the same physical hex unit'
    gutter=map_x('v5')-map_x('radio')-map_width('radio');assert gutter>=1.0
    renderer=fig.canvas.get_renderer();all_text=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible()and t.get_text():
            box=t.get_window_extent(renderer);bounds=list(box.bounds)
            assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height,('text outside canvas',t.get_text(),bounds)
            assert t.get_fontsize()in[7.5,10.5]and t.get_fontweight()=='normal'
            all_text.append({'text':t.get_text(),'fontsize_bp':t.get_fontsize(),'bbox_pixels':list(box.bounds)})
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:
        kwargs={'metadata':{'CreationDate':None,'ModDate':None}}if ext=='pdf'else{'metadata':{'Date':None}}if ext=='svg'else{}
        fig.savefig(out/f'figure.{ext}',dpi=300,**kwargs)
    plt.close(fig)
    checks={'schema':'chapter2-three-timepoint-hex-shared-scale-v1','status':'PASS','source_csv':str(BASE/'data.csv'),'data_sha256':sha(BASE/'data.csv'),
      'width_cm':W,'height_cm':H,'common_height_cm':MAP_HEIGHT,'same_physical_hex_across_tasks':True,'common_hex_unit_cm':float(np.mean(unit_values)),'gutter_cm':gutter,'domain_palettes':PALETTES,'font_sizes_bp':{'main':10.5,'ticks':10.5,'object_ID':7.5},'shared_scales':shared,'panels':panels,
      'all_actual_visible_text_inside_canvas':True,'text_records':all_text,'model_loads':0,'environment_steps':0,'new_inference_calls':0,'training_updates':0}
    (out/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    source={'schema':'chapter2-six-retained-snapshots-presentation-v1','status':'RENDERED_PENDING_INDEPENDENT_REVIEW','editable_source':'plot.py','editable_source_sha256':sha(__file__),
      'input_csv':'data.csv','input_csv_sha256':sha(BASE/'data.csv'),'provenance':'provenance.json','provenance_sha256':sha(BASE/'provenance.json'),
      'required_runtime_data':['data.csv','provenance.json'],'periods':PERIODS,'parent_midpoint_records_unchanged':True,'source_values_unchanged':True,
      'figure_size_cm':[W,H],'scientific_values_unchanged':True,'same_task_three_panels_share_state_and_area_scales':True,'same_map_height_cm':MAP_HEIGHT,'domain_palettes':PALETTES,'column_major_panel_labels':True,'grid_columns':{'radio':10,'v5':12},'same_physical_hex_across_tasks':True,'common_hex_unit_cm':float(np.mean(unit_values)),
      'outputs':{ext:sha(out/f'figure.{ext}')for ext in ['pdf','svg','png']},'checks_sha256':sha(out/'checks.json'),
      'model_loads':0,'new_environment_steps':0,'new_inference_calls':0,'training_updates':0}
    (out/'source.json').write_text(json.dumps(source,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','figure':str(out/'figure.pdf'),'size_cm':[W,H],'panels':[{k:p[k]for k in ['domain','period_one_based','positive_count','zero_count','allocation_total']}for p in panels]},ensure_ascii=False))

if __name__=='__main__':main()
