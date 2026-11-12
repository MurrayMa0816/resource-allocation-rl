"""Common native layout for C6 scene, state/action and mean-yield maps.

Pure plotting of fixed geometry and sealed arrays. No environment/model imports.
"""
from pathlib import Path
import argparse,ast,csv,hashlib,importlib.util,json,math
import numpy as np

from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
THESIS=next(p for p in HERE.parents if (p/'assets/fonts').is_dir())
ROOT=Path(__file__).resolve().parents[1]/'.render-output'
OLD=HERE/'data'
VISIBLE=HERE/'data'
WIDTH,BOARD,XM,YM=156.,146.,5.,4.
TOP_CALLOUT_MM=6.
BOTTOM_CALLOUT_MM=8.
CHANNEL_LABEL_BP=10.5
GRID_LINE_BP=.25
DAYS=(0,46,62,72)
B_EDGES=np.array([80.,100.,120.,140.,160.,180.])
B_COLORS=('#fff9df','#ffedc5','#ffdc9c','#ffc775','#f6ac52')
Y_EDGES=np.array([88.,90.,91.,92.,93.,94.])
# Explicit cividis-inspired colors with strictly decreasing gray luminance.
Y_COLORS=('#f2da5e','#beb051','#8e8b5a','#5c636e','#123570')
MAGENTA='#A80058'
R1000,LINE,HALO=1.6,1.5,3.


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_csv(p,rows):
    rows=list(rows)
    with Path(p).open('w',newline='',encoding='utf-8')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--figure",choices=["6.1","6.4","6.6","all"],default="all");args=parser.parse_args()
    selected={"6.1":"scene","6.4":"stage_overlay","6.6":"regional"}
    selected_groups=set(selected.values()) if args.figure=="all" else {selected[args.figure]}
    from matplotlib.patches import Circle,Rectangle,FancyBboxPatch,Polygon as MplPolygon
    from matplotlib.collections import LineCollection
    import matplotlib.patheffects as pe
    from shapely.geometry import shape,LineString,box,Polygon
    from shapely.ops import unary_union
    spec=importlib.util.spec_from_file_location('common_sealed_map_display',HERE/'source/display_helpers.py')
    display=importlib.util.module_from_spec(spec);spec.loader.exec_module(display)
    plt=display.plotting(ROOT);plt.rcParams.update({'xtick.labelsize':10.5,'ytick.labelsize':10.5,'legend.fontsize':10.5,'svg.hashsalt':'chapter6-two-dates-native-110mm-20261110'})
    geom_files=sorted((OLD/'geometry').glob('*.json'))
    geo={p.stem:json.loads(p.read_text())for p in geom_files}
    style=json.loads((OLD/'channel_style.json').read_text())
    fixed_mapping_path=HERE/'data/mapping.json'
    fixed=json.loads(fixed_mapping_path.read_text())
    original_manifest_path=HERE/'data/visible_manifest.json'
    original_manifest=json.loads(original_manifest_path.read_text())
    static=OLD/'static_scene';static_source=HERE/'data/palette.json'
    palette=json.loads(static_source.read_text())
    static_gates=json.loads((static/'data/compact_gate_symbols.json').read_text())
    overrides=json.loads((static/'style_overrides.json').read_text())
    inputs=geom_files+[HERE/'source/display_helpers.py',OLD/'channel_style.json',fixed_mapping_path,original_manifest_path,
                      static_source,static/'data/layout.json',static/'data/compact_gate_symbols.json',static/'style_overrides.json',HERE/'data/mean_terminal_y_all89.csv']
    inputs +=[display.FONT_ROOT/n for n in display.FONT_NAMES]
    before={str(p):sha(p)for p in inputs}
    assert geo['layout']['regions']==json.loads((static/'data/layout.json').read_text())['regions']
    ids=[r['id']for r in geo['layout']['regions']];assert ids==fixed['ids']and len(ids)==89
    assert fixed['radius1000_mm']==R1000 and fixed['foreground_line_bp']==LINE and fixed['white_halo_line_bp']==HALO
    labels=np.asarray(fixed['label_positions']);centres=np.asarray(fixed['centres'])
    adjustment=json.loads((Path(__file__).resolve().parents[1]/'label_adjustments.json').read_text())['labels']
    for i,name in enumerate(ids):
        if name in adjustment:labels[i]=adjustment[name]['new_world']
    a=np.asarray([[r['allocation_m3']for r in fixed['rows']if r['day']==day]for day in DAYS])
    B=np.asarray([[r['beginning_storage_mm']for r in fixed['rows']if r['day']==day]for day in DAYS])
    dates=['2022-05-30','2022-07-15','2022-07-31','2022-08-10']
    with (HERE/'data/mean_terminal_y_all89.csv').open(newline='',encoding='utf-8')as stream:regional=list(csv.DictReader(stream))
    assert [r['region_id']for r in regional]==ids
    per_seed=np.asarray([[float(r[f'seed{seed}_weather128_mean_percent'])for r in regional]for seed in (6400,6401,6402)])
    mean_y=per_seed.mean(axis=0);sd_y=per_seed.std(axis=0,ddof=1)
    low=np.flatnonzero(mean_y<90)
    assert{ids[i]for i in low}=={'12','65','76','92'}
    frame=geo['layout']['board_rectangle'];mm_per=BOARD/(frame['x_max']-frame['x_min'])
    map_h=BOARD*(frame['y_max']-frame['y_min'])/(frame['x_max']-frame['x_min'])
    height=map_h+2*YM+TOP_CALLOUT_MM+BOTTOM_CALLOUT_MM
    board_bottom=YM+BOTTOM_CALLOUT_MM
    assert abs(mm_per-fixed['mm_per_geometry_unit'])<1e-12
    regions=[unary_union([Polygon(display.hexagon(c))for c in geo['layout']['cells']
                          if c['kind']=='service'and c['owner']==r['owner']]).buffer(0)for r in geo['layout']['regions']]
    gray_leaders={r['region_id']:r['label_leader']for r in fixed['rows']if r['day']==0 and r['label_leader']is not None}
    for name,line in gray_leaders.items():
        if name in adjustment:
            change=np.asarray(adjustment[name]['new_world'])-np.asarray(adjustment[name]['old_world'])
            line[0]=(np.asarray(line[0])+change).tolist()
    for name,anchor in json.loads((Path(__file__).resolve().parents[1]/'label_adjustments.json').read_text()).get('new_internal_region_anchors',{}).items():
        gray_leaders[name]=[labels[ids.index(name)].tolist(),anchor]
    for row in fixed['rows']:
        name=row['region_id']
        if name in gray_leaders:row['label_leader']=[list(v)for v in gray_leaders[name]]
        if name in adjustment:
            row['label_x'],row['label_y']=adjustment[name]['new_world']
    fixed['label_positions']=labels.tolist()
    effects=[pe.Stroke(linewidth=HALO,foreground='white'),pe.Normal()]
    records={}
    def shades(values,edges,colours):
        values=np.asarray(values);assert values.min()>=edges[0]and values.max()<=edges[-1]
        band=np.minimum(np.searchsorted(edges,values,side='right')-1,len(colours)-1)
        return[colours[int(i)]for i in band]
    def common_map(colours):
        fig=plt.figure(figsize=(110./25.4,height*(110./156.)/25.4))
        ax=fig.add_axes([XM/WIDTH,board_bottom/height,BOARD/WIDTH,map_h/height])
        original=display.allocation_colors;display.allocation_colors=lambda unused:colours
        try:display.draw_map(ax,np.arange(89),geo)
        finally:display.allocation_colors=original
        channel_texts=[t for t in ax.texts if'渠'in t.get_text()]
        for text,label in zip(channel_texts,style):
            text.set_position(label['position']);text.set_fontsize(label['font_size_pt'])
        for t in ax.texts:
            if t.get_text()in ids:
                t.set_position(labels[ids.index(t.get_text())]);t.set_clip_on(False)
        for leader in gray_leaders.values():
            ax.plot(*zip(*leader),color='#444444',lw=.55,zorder=5.6,clip_on=False,solid_capstyle='butt')
        for p in ax.patches:
            if isinstance(p,Rectangle):p.set_zorder(4.5)
        # Preserve the existing geometry and flow-arrow exclusion masks before
        # removing the repeated microscopic channel-name glyphs.
        fig.canvas.draw(); renderer=fig.canvas.get_renderer()
        legacy_channel_boxes=[]
        for t in channel_texts:
            bb=t.get_window_extent(renderer)
            (x0,y0),(x1,y1)=ax.transData.inverted().transform([[bb.x0,bb.y0],[bb.x1,bb.y1]])
            legacy_channel_boxes.append(box(x0,y0,x1,y1).buffer(.35/mm_per))
            t.remove()
        ax._legacy_channel_boxes=legacy_channel_boxes
        for collection in ax.collections:
            widths=collection.get_linewidths()
            if len(widths) and np.allclose(widths,.16):collection.set_linewidths(GRID_LINE_BP)
        # The five old 洪庙支渠 strings label consecutive reaches of the same
        # serpentine channel. One external label identifies that same channel;
        # no invented branch IDs/names are added.
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyArrowPatch
        def world_mm(x,y):return XM+(x-frame['x_min'])*mm_per,board_bottom+(y-frame['y_min'])*mm_per
        inlet=world_mm(*style[0]['position'])
        outlet=world_mm(48.35, -31.17691453623979)
        fig.text(8./WIDTH,(height-2.1)/height,'三分干渠',fontsize=CHANNEL_LABEL_BP,ha='left',va='top')
        fig.text(120./WIDTH,3.3/height,'洪庙支渠',fontsize=CHANNEL_LABEL_BP,ha='left',va='center')
        def leader(points):
            points=np.asarray(points,dtype=float)/[WIDTH,height]
            fig.add_artist(Line2D(points[:-1,0],points[:-1,1],transform=fig.transFigure,color='#444444',lw=.6,solid_capstyle='butt'))
            fig.add_artist(FancyArrowPatch(points[-2],points[-1],transform=fig.transFigure,arrowstyle='->',mutation_scale=5.,lw=.6,color='#444444',shrinkA=0,shrinkB=0))
        leader([(8.,height-6.0),(2.5,height-6.0),(2.5,inlet[1]),inlet])
        leader([(137.,6.3),(153.5,6.3),(153.5,outlet[1]),outlet])
        return fig,ax
    def save(fig,base,extra=None):
        base.parent.mkdir(parents=True,exist_ok=True);fig.canvas.draw()
        from matplotlib.text import Text
        renderer=fig.canvas.get_renderer();outside=[]
        for item in fig.findobj(match=Text):
            if not item.get_visible() or not item.get_text():continue
            bb=item.get_window_extent(renderer)
            if bb.x0<0 or bb.y0<0 or bb.x1>fig.bbox.x1 or bb.y1>fig.bbox.y1:outside.append(item.get_text())
        assert not outside,(str(base),outside)
        for ext in('pdf','svg','png'):
            options={'dpi':240}if ext=='png'else({'metadata':{'CreationDate':None,'ModDate':None}}if ext=='pdf'else{'metadata':{'Date':None}})
            fig.savefig(base.with_suffix('.'+ext),**options)
        result=dict(width_mm=float(fig.get_size_inches()[0]*25.4),height_mm=float(fig.get_size_inches()[1]*25.4),
            assets={ext:dict(path=str(base.with_suffix('.'+ext)),sha256=sha(base.with_suffix('.'+ext)))for ext in('pdf','svg','png')})
        if extra:result.update(extra)
        plt.close(fig);return result
    def legend_canvas(height_mm=28.):
        fig=plt.figure(figsize=(WIDTH/25.4,height_mm/25.4));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,WIDTH),ylim=(0,height_mm));ax.axis('off');return fig,ax
    def bar(ax,edges,colours,title,threshold=None):
        ax.text(WIDTH/2,25.,title,ha='center',va='center',fontsize=10.5)
        left,right,y,h=XM,WIDTH-XM,20.,2.3;step=(right-left)/len(colours)
        for i,c in enumerate(colours):ax.add_patch(Rectangle((left+i*step,y),step,h,fc=c,ec='none'))
        ax.add_patch(Rectangle((left,y),right-left,h,fc='none',ec='#65725e',lw=.8))
        for i,v in enumerate(edges):ax.text(left+i*step,17.2,f'{v:g}',ha='center',va='center',fontsize=10.5)
        if threshold is not None:
            x=left+int(np.flatnonzero(edges==threshold)[0])*step;ax.plot([x,x],[y,y+h],color='black',lw=1.15)
        return dict(left_mm=left,right_mm=right,length_mm=right-left,equals_scene_board=True)
    if 'scene' in selected_groups:
        # 6.1 keeps categorical colours and its conceptual flow/total-decision labels.
        scene_colors=[palette[r['owner']%len(palette)]for r in geo['layout']['regions']]
        fig,ax=common_map(scene_colors);fig.canvas.draw();renderer=fig.canvas.get_renderer()
        channel_boxes=ax._legacy_channel_boxes
        def arrow(first,last,width):
            if any(LineString([first,last]).buffer(.6/mm_per).intersects(b)for b in channel_boxes):return
            ax.annotate('',xy=last,xytext=first,arrowprops=dict(arrowstyle='-|>',lw=width,color='white'),zorder=5)
        feeder=static_gates['upstream_feeder'];left=-1.5-feeder['width_world']
        for index,(first,last)in enumerate(feeder['flow_arrows']):
            first,last=list(first),list(last)
            if index==0:first[0]=last[0]=left+overrides['upstream_flow_arrow']['x_fraction_from_left_bank']*feeder['width_world']
            arrow(first,last,1.1)
        for line in feeder['continuation_breaks']:ax.plot(*zip(*line),color='white',lw=1.3,zorder=5,solid_capstyle='butt')
        route=geo['layout']['canal']
        for i in range(8,len(route)-3,19):
            first,last=route[i],route[min(i+2,len(route)-1)];arrow([first['x'],first['y']],[last['x'],last['y']],.8)
        badge=dict(static_gates['total_decision_badge']);badge.update(overrides['q_t_badge']);cx,cy=badge['center'];w,h=badge['width'],badge['height']
        ax.plot(*zip(*badge['leader']),color=badge['stroke'],lw=1.,zorder=8.4)
        ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle='round,pad=0,rounding_size=.09',fc=badge['fill'],ec=badge['stroke'],lw=1.,zorder=8.5))
        dx,dy=overrides['q_t_badge']['text_offset_world'];ax.text(cx+dx,cy+dy,r'$q_t$',ha='center',va='center',fontsize=10.5,color=badge['text_color'],zorder=8.6)
        records['scene']=save(fig,ROOT/'figures/scene/figure',dict(fill_semantics='Original categorical region differentiation only',palette=palette))
        fig,ax=legend_canvas(12.)
        # Six categorical chips are not a scalar bar and have no numerical axis.
        for i,c in enumerate(palette):ax.add_patch(Rectangle((12.+i*2.,4.5),1.7,4.,fc=c,ec='#65563e',lw=.35))
        ax.text(27.,6.5,'灌溉区域',ha='left',va='center',fontsize=10.5)
        for x in(72.,76.):ax.add_patch(Rectangle((x,4.),1.5,5.,fc=display.ORANGE,ec='none'))
        ax.plot([74.75,74.75],[4.6,8.4],color=display.ORANGE,lw=2.)
        ax.text(81.,6.5,'总闸',ha='left',va='center',fontsize=10.5)
        ax.add_patch(Rectangle((112.,4.),4.,5.,fc='#399bc1',ec='none'))
        for x in(112.,116.):ax.plot([x,x],[4.,9.],color='#a35f22',lw=.8)
        ax.plot([113.,115.],[6.5,6.5],color='#a35f22',lw=.7)
        ax.text(120.,6.5,'区域入口',ha='left',va='center',fontsize=10.5)
        records['scene_legend']=save(fig,ROOT/'figures/scene/legend')
    if 'stage_overlay' in selected_groups:
        # 6.4 reproduces the adopted big circles, positions and data exactly.
        for k,day in enumerate(DAYS):
            fig,ax=common_map(shades(B[k],B_EDGES,B_COLORS))
            old_rows=[r for r in fixed['rows']if r['day']==day]
            for i,r in enumerate(old_rows):
                if not r['drawn']:continue
                if r['circle_leader']is not None:
                    line=ax.plot(*zip(*r['circle_leader']),color=MAGENTA,lw=.8,zorder=6.1,clip_on=False,solid_capstyle='butt')[0]
                    line.set_path_effects([pe.Stroke(linewidth=1.8,foreground='white'),pe.Normal()])
                c=Circle(centres[i],R1000*math.sqrt(a[k,i]/1000)/mm_per,fc='none',ec=MAGENTA,lw=LINE,zorder=6.4,clip_on=False)
                c.set_path_effects(effects);ax.add_patch(c)
            records[f'day{day:02}']=save(fig,ROOT/f'figures/stage_overlay/day{day:02}')
        fig,ax=legend_canvas();alignment=bar(ax,B_EDGES,B_COLORS,'日初储水量（mm）')
        ax.text(XM,7.3,'当日配水量\n（m³）',ha='left',va='center',fontsize=10.5,linespacing=1.05)
        examples=[]
        # The complete legend row starts at the scene left edge and its last value
        # ends exactly at the scene right edge. Physical circle scale is unchanged.
        for x,v in((45.,125.),(90.,1000.),(138.1916666667,4000.)):
            radius=R1000*math.sqrt(v/1000);c=Circle((x,7.3),radius,fc='none',ec=MAGENTA,lw=LINE);c.set_path_effects(effects);ax.add_patch(c)
            if v==4000:ax.text(WIDTH-XM,7.3,f'{v:g}',ha='right',va='center',fontsize=10.5)
            else:ax.text(x+radius+2.2,7.3,f'{v:g}',ha='left',va='center',fontsize=10.5)
            examples.append(dict(value_m3=v,radius_mm=radius))
        records['allocation_legend']=save(fig,ROOT/'figures/stage_overlay/parent_legend_not_for_pair',dict(storage_bar=alignment,circle_row_left_mm=5.,circle_row_right_mm=151.,circle_examples=examples))
    if 'regional' in selected_groups:
        # 6.6 is the same three-policy/128-weather statistic, never the exemplar.
        fig,ax=common_map(shades(mean_y,Y_EDGES,Y_COLORS))
        owners={geo['layout']['regions'][i]['owner']for i in low};edge_owners={}
        for c in geo['layout']['cells']:
            if c['kind']!='service':continue
            polygon=display.hexagon(c)
            for first,last in zip(polygon,polygon[1:]+polygon[:1]):edge_owners.setdefault(tuple(sorted((first,last))),[]).append(c['owner'])
        outline=[edge for edge,who in edge_owners.items()if any(o in owners for o in who)and(len(who)!=2 or who[0]!=who[1])]
        ax.add_collection(LineCollection(outline,colors='#171717',linewidths=1.15,zorder=2.6))
        records['regional']=save(fig,ROOT/'figures/regional/mean_terminal_y')
        fig,ax=legend_canvas();yalign=bar(ax,Y_EDGES,Y_COLORS,'区域平均相对产量（%）',threshold=90.)
        ax.plot([48.,57.],[7.3,7.3],color='#171717',lw=1.15);ax.text(60.,7.3,'区域均值低于90%',ha='left',va='center',fontsize=10.5)
        records['regional_legend']=save(fig,ROOT/'figures/regional/mean_terminal_y_legend',dict(yield_bar=yalign))
    if 'scene' in selected_groups:
        write_csv(ROOT/'figures/scene/region_categorical_colours.csv',
            (dict(region_id=ids[i],owner=r['owner'],categorical_colour_index=r['owner']%len(palette),colour=scene_colors[i],label_x=labels[i,0],label_y=labels[i,1])for i,r in enumerate(geo['layout']['regions'])))
    if 'stage_overlay' in selected_groups:
        write_csv(ROOT/'figures/stage_overlay/all89_four_dates_state_action.csv',
            ({k:v for k,v in row.items()if k not in('circle_leader','label_leader')}for row in fixed['rows']))
    if 'regional' in selected_groups:
        write_csv(ROOT/'figures/regional/mean_terminal_y_all89.csv',
            (dict(region_id=name,mean_relative_yield_percent=float(mean_y[i]),training_sample_sd_percent=float(sd_y[i]),
                  **{f'seed{seed}_weather128_mean_percent':float(per_seed[k,i])for k,seed in enumerate((6400,6401,6402))},
                  mean_below_90=bool(mean_y[i]<90),label_x=float(labels[i,0]),label_y=float(labels[i,1]))for i,name in enumerate(ids)))
    if 'stage_overlay' in selected_groups:dump(ROOT/'figures/stage_overlay/mapping.json',fixed)
    assert{str(p):sha(p)for p in inputs}==before
    common=dict(source=str(Path(__file__).resolve()),source_sha256=sha(__file__),read_only_inputs_sha256=before,
        inputs_unchanged=True,virtual_parent_canvas_width_mm=WIDTH,canvas_width_mm=110.,canvas_height_mm=height*110/156,scene_board_width_mm=BOARD*110/156,
        scene_board_height_mm=map_h*110/156,scene_left_mm=XM*110/156,scene_right_mm=(WIDTH-XM)*110/156,scene_bottom_mm=board_bottom*110/156,
        common_fixed_label_positions=labels.tolist(),original_big_circle_mapping_sha256=sha(fixed_mapping_path),
        region_ID_font_bp=9.5,channel_fonts_bp=[CHANNEL_LABEL_BP],channel_name_callouts=True,
        channel_semantics='One upstream 三分干渠 and one continuous serpentine 洪庙支渠; five old repeated tiny labels did not denote five separate channels',
        service_hex_grid_line_bp=GRID_LINE_BP,original_service_hex_grid_line_bp=.16,
        yield_palette=Y_COLORS,yield_palette_gray_luminance=[float(sum(c*w for c,w in zip(tuple(int(h[i:i+2],16)/255 for i in (1,3,5)),(.2126,.7152,.0722)))) for h in Y_COLORS],
        ordinary_legend_font_bp=10.5,
        quantitative_bar_left_mm=5.,quantitative_bar_right_mm=151.,quantitative_bar_length_mm=146.,
        scientific_execution=dict(environment_steps=0,model_loads=0,inference_calls=0,training_steps=0))
    for group,keys in [('scene',['scene','scene_legend']),('stage_overlay',[f'day{d:02}'for d in DAYS]+['allocation_legend']),('regional',['regional','regional_legend'])]:
        if group not in selected_groups:continue
        data=dict(common,group=group,assets={k:records[k]for k in keys})
        if group=='scene':data.update(fill_semantics='Original categorical differentiation only; no state/yield statistic',original_palette=palette)
        elif group=='stage_overlay':data.update(days=list(DAYS),dates=dates,seed=6400,weather_id=125,radius1000_mm=R1000,foreground_bp=LINE,white_halo_bp=HALO,zero_no_circle=True,positive_floor=False,mapping_unchanged=True)
        else:data.update(statistic='Within each policy128-weather mean, then mean over3 independent training seeds',low_mean_region_ids=[ids[i]for i in low],range_percent=[float(mean_y.min()),float(mean_y.max())],threshold=90.)
        dump(ROOT/f'figures/{group}/manifest.json',data)
    dump(ROOT/f'audit/map_family_generation_{args.figure}.json',dict(status='GENERATED_FOR_REVIEW',**common,assets=records))
    print(json.dumps(dict(status='MAP_FAMILY_GENERATED',canvas_mm=[WIDTH,height],scene_width_mm=BOARD,quantitative_bar_mm=[5.,151.,146.],groups=['scene','stage_overlay','regional']),ensure_ascii=False))


if __name__=='__main__':main()
