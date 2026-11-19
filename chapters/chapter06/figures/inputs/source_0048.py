"""Water-connected inlet stubs on median frontage hexes of unchanged89-region map.

Read-only display geometry. No environment/model/trajectory imports.
"""
from pathlib import Path
from collections import deque
import hashlib, json, math, os

ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT.parent / 'compact_canal_scene_20261106'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import PathPatch, Rectangle, Patch, Polygon as MplPolygon, Circle, FancyBboxPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.path import Path as MplPath
from shapely.geometry import shape, Polygon, LineString, box
from shapely.ops import unary_union

FONTS = Path('/root/gamewar/latex_thesis_resource_allocation/assets/fonts/windows')
for filename in ['simsun.ttc', 'TIMES.TTF']:
    font_manager.fontManager.addfont(FONTS / filename)
plt.rcParams.update({'font.family': ['Times New Roman', 'SimSun'], 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'svg.fonttype': 'path', 'mathtext.fontset': 'cm'})
DIRS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
PALETTE = ['#d2dfad', '#bacd90', '#e0e9c9', '#c5d9bb', '#e1dfad', '#add0ba']
WATER = '#399bc1'
ACCENT = '#cf6a24'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def polygon(c):
    return [(round(c['x'] + math.cos(math.radians(30 + 60*k)) / math.sqrt(3), 6), round(c['y'] + math.sin(math.radians(30 + 60*k)) / math.sqrt(3), 6)) for k in range(6)]


def mpl_path(geom):
    parts = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
    verts, codes = [], []
    for part in parts:
        for ring in [part.exterior] + list(part.interiors):
            points = list(ring.coords)
            verts.extend(points)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(points)-2) + [MplPath.CLOSEPOLY])
    return MplPath(verts, codes)


def neighbors(p):
    return [(p[0]+dq, p[1]+dr) for dq, dr in DIRS]


def bfs(start, goal, water):
    todo, parent = deque([start]), {start: None}
    while todo:
        here = todo.popleft()
        if here == goal:
            result = []
            while here is not None:
                result.append(here)
                here = parent[here]
            return result[::-1]
        for nxt in neighbors(here):
            if nxt in water and nxt not in parent:
                parent[nxt] = here
                todo.append(nxt)
    raise ValueError('Unreachable registered mouth')


def draw_cells(ax, cells):
    for role in ['background', 'service', 'canal', 'upstream_canal', 'canal_connector']:
        cc = [c for c in cells if c['kind'] == role]
        colors = [PALETTE[c['owner'] % 6] if role == 'service' else '#edf3f0' if role == 'canal' else '#f5f2e9' for c in cc]
        ax.add_collection(PolyCollection([polygon(c) for c in cc], facecolors=colors, edgecolors='#7c9167' if role == 'service' else '#9aadb1' if role == 'canal' else '#deded1', linewidths=.15, zorder=1))


class LargeGateLegend:
    def __init__(self,outer,inner,open_ends=False):
        self.outer,self.inner,self.open_ends=outer,inner,open_ends


class LargeGateHandler(HandlerBase):
    def create_artists(self,legend,handle,xd,yd,width,height,fontsize,trans):
        h=height*.95;w=h*.5;cx=xd+width/2;cy=yd+height/2
        if handle.open_ends=='branch_head':
            return [Rectangle((cx-w/2,cy+h*.2),w,h*.3,fc=handle.outer,ec='none',transform=trans),
                    Rectangle((cx-w/2,cy-h*.5),w,h*.3,fc=handle.outer,ec='none',transform=trans),
                    Line2D([cx,cx],[cy-h*.23,cy+h*.23],color=handle.outer,lw=1.5,transform=trans)]
        if handle.open_ends:
            return [Rectangle((cx-w/2,cy-h/2),w,h,fc=WATER,ec='none',transform=trans),
                    Line2D([cx-w/2,cx-w/2],[cy-h/2,cy+h/2],color=handle.outer,lw=1.1,solid_capstyle='butt',transform=trans),
                    Line2D([cx+w/2,cx+w/2],[cy-h/2,cy+h/2],color=handle.outer,lw=1.1,solid_capstyle='butt',transform=trans),
                    Line2D([cx-w*.25,cx+w*.25],[cy,cy],color=handle.outer,lw=1,marker='o',markersize=1.4,transform=trans)]
        return [Rectangle((cx-w/2,cy-h/2),w,h,fc=handle.outer,ec=handle.outer,lw=1.1,transform=trans),
                Rectangle((cx-w*.32,cy-h*.38),w*.64,h*.76,fc=handle.inner,ec='none',transform=trans)]


def draw_symbol(ax,glyph,outer,inner,lw):
    if glyph.get('glyph_kind')=='feeder_branch_sluice':
        for poly in glyph['pier_polygons']:
            ax.add_patch(MplPolygon(poly,closed=True,fc=outer,ec='none',zorder=8))
        for poly in glyph['pier_inner_polygons']:
            ax.add_patch(MplPolygon(poly,closed=True,fc=inner,ec='none',zorder=8.1))
        ax.plot(*zip(*glyph['gate_crossbar']),color=outer,lw=2.4,solid_capstyle='butt',zorder=8.2)
        for post in glyph['gate_posts']:
            ax.add_patch(Circle((post['x'],post['y']),post['r'],fc=outer,ec='none',zorder=8.3))
        return
    if glyph.get('glyph_kind')=='open_canal_attached_inlet':
        # The global opaque water fill already covers this inlet; keep hex edges visible.
        for rail in glyph['side_rails']:
            ax.plot(*zip(*rail),color=outer,lw=lw,solid_capstyle='butt',zorder=8.1)
        ax.plot(*zip(*glyph['gate_crossbar']),color=outer,lw=.8,solid_capstyle='butt',zorder=8.2)
        for post in glyph['gate_posts']:
            ax.add_patch(Circle((post['x'],post['y']),post['r'],fc=outer,ec='none',zorder=8.3))
        return
    ax.add_patch(MplPolygon(glyph['outer_polygon'],closed=True,fc=outer,ec=outer,lw=lw,zorder=8))
    ax.add_patch(MplPolygon(glyph['inner_polygon'],closed=True,fc=inner,ec='none',zorder=8.1))


def draw_bold_name(ax,label,font,scale):
    """CJK700 outlines avoid the CFF-to-Type42 font conversion issue."""
    x,y=label['position'];size=label['font_size_pt']
    parts=list(label['text']) if label.get('stacked') else [label['text']]
    spacing=size*label.get('line_spacing',1.12)/scale
    for i,text in enumerate(parts):
        path=TextPath((0,0),text,size=size,prop=font)
        bbox=path.get_extents();cy=y+((len(parts)-1)/2-i)*spacing
        transform=Affine2D().translate(-(bbox.x0+bbox.x1)/2,-(bbox.y0+bbox.y1)/2).scale(1/scale).translate(x,cy)
        ax.add_patch(PathPatch(path,transform=transform+ax.transData,fc=label['color'],ec='none',zorder=6))


def main():
    layout_path=ROOT/'data/layout.json';band_path=ROOT/'data/water_band.json'
    symbol_path=ROOT/'data/compact_gate_symbols.json'
    d=json.loads(layout_path.read_text());gates=json.loads(symbol_path.read_text())
    gates['channel_font']['path'] = str(ROOT/'assets/NotoSansCJKSC-Bold-subset.otf')
    assert sha(layout_path)==gates['layout_sha256']
    overrides=json.loads((ROOT/'style_overrides.json').read_text())
    badge_override=overrides['q_t_badge']
    gates['total_decision_badge'].update({key:badge_override[key] for key in ['center','width','height','font_size_pt','leader']})
    feeder_override=gates['upstream_feeder']
    feeder_shape=shape(feeder_override['water_geometry'])
    left=min(x for poly in ([feeder_shape] if feeder_shape.geom_type=='Polygon' else list(feeder_shape.geoms)) for x,y in poly.exterior.coords if x>-3)
    # Existing feeder width is explicit; use the registered vertical bank positions.
    original_left=-1.5-feeder_override['width_world']
    shifted_x=original_left+overrides['upstream_flow_arrow']['x_fraction_from_left_bank']*feeder_override['width_world']
    for point in feeder_override['flow_arrows'][0]:point[0]=shifted_x
    water=json.loads(band_path.read_text());band=shape(water['water_geometry'])
    assert water['geometry_sha256']==sha(layout_path)
    edges={}
    for c in d['cells']:
        if c['kind']!='service':continue
        pp=polygon(c)
        for a,b in zip(pp,pp[1:]+pp[:1]):edges.setdefault(tuple(sorted((a,b))),[]).append(c['owner'])
    boundaries=[e for e,owners in edges.items() if len(owners)!=2 or owners[0]!=owners[1]]
    internal=[e for owner in json.loads((ROOT/'data/internal_hex_edges.json').read_text())['owners'] for e in owner['edges']]
    frame=d['board_rectangle'];aspect=(frame['x_max']-frame['x_min'])/(frame['y_max']-frame['y_min'])
    width=156/25.4;mapwidth=width*.96;mapheight=mapwidth/aspect;height=mapheight+.60
    fig=plt.figure(figsize=(width,height));ax=fig.add_axes([.02,.105,.96,mapheight/height]);ax.axis('off');ax.set_aspect('equal')
    boldfont=font_manager.FontProperties(fname=gates['channel_font']['path'])
    mapscale=mapwidth*72/(frame['x_max']-frame['x_min'])
    name_bounds=[]
    for label in gates.get('channel_labels',[]):
        parts=list(label['text']) if label.get('stacked') else [label['text']]
        spacing=label['font_size_pt']*label.get('line_spacing',1.12)/mapscale
        for i,part in enumerate(parts):
            bbox=TextPath((0,0),part,size=label['font_size_pt'],prop=boldfont).get_extents()
            cx,cy=label['position'];cy+=((len(parts)-1)/2-i)*spacing
            halfw=bbox.width/(2*mapscale);halfh=bbox.height/(2*mapscale)
            clear=overrides['arrows']['extra_clearance_points']/mapscale
            name_bounds.append(box(cx-halfw-clear,cy-halfh-clear,cx+halfw+clear,cy+halfh+clear))
    skipped_arrows=[]
    def arrow_clear(a,b,kind,index):
        # A conservative 1.8pt halo includes the arrowhead beside the shaft.
        line=LineString([a,b]).buffer(1.8/mapscale)
        if any(line.intersects(bounds) for bounds in name_bounds):
            skipped_arrows.append(dict(kind=kind,index=index,start=a,end=b))
            return False
        return True
    draw_cells(ax,d['cells'])
    ax.add_collection(LineCollection(internal,colors='#7c9167',linewidths=.22,zorder=2))
    ax.add_collection(LineCollection(boundaries,colors='#435c41',linewidths=.65,zorder=2.5))
    # Fill connected water once to avoid an antialias hairline at the open mouths.
    combined=shape(json.loads((ROOT/'data/combined_water_geometry.json').read_text()))
    ax.add_patch(PathPatch(mpl_path(combined),fc=WATER,ec='none',zorder=3))
    grid=json.loads((ROOT/'data/water_hex_edges.json').read_text())
    ax.add_collection(LineCollection(grid['edges'],colors=grid['color'],linewidths=grid['linewidth_pt'],zorder=3.05))
    banks=shape(gates['main_bank_lines_geometry'])
    bank_parts=list(banks.geoms) if hasattr(banks,'geoms') else [banks]
    ax.add_collection(LineCollection([list(x.coords) for x in bank_parts],colors='#287ba0',linewidths=.4,capstyle='butt',zorder=3.1))
    feeder=gates['upstream_feeder']
    for arrow_index,(a,b) in enumerate(feeder['flow_arrows']):
        if not arrow_clear(a,b,'upstream',arrow_index):continue
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',lw=1.1,color='white'),zorder=5)
    for line in feeder['continuation_breaks']:
        ax.plot(*zip(*line),color='white',lw=1.3,solid_capstyle='butt',zorder=5)
    boldfont=font_manager.FontProperties(fname=gates['channel_font']['path'])
    mapscale=mapwidth*72/(frame['x_max']-frame['x_min'])
    for label in gates.get('channel_labels',[]):
        draw_bold_name(ax,label,boldfont,mapscale)
    # Old individual bank connectors are not displayed; the new stubs themselves bridge the blue band and frontage hex.
    route=d['canal']
    for i in range(8,len(route)-3,19):
        a,b=route[i],route[min(i+2,len(route)-1)]
        if not arrow_clear([a['x'],a['y']],[b['x'],b['y']],'branch',i):continue
        ax.annotate('',xy=(b['x'],b['y']),xytext=(a['x'],a['y']),arrowprops=dict(arrowstyle='-|>',lw=.8,color='white'),zorder=5)
    colors=gates['colors']
    for g in gates['regional_gates']:
        draw_symbol(ax,g,colors['regional'],colors['regional_inner'],1.05)
    draw_symbol(ax,gates['total_gate'],colors['total'],colors['total_inner'],1.5)
    badge=gates['total_decision_badge'];cx,cy=badge['center'];bw,bh=badge['width'],badge['height']
    ax.plot(*zip(*badge['leader']),color=badge['stroke'],lw=1.0,zorder=8.4)
    ax.add_patch(FancyBboxPatch((cx-bw/2,cy-bh/2),bw,bh,boxstyle='round,pad=0,rounding_size=.09',
                              fc=badge['fill'],ec=badge['stroke'],lw=1,zorder=8.5))
    ax.text(cx+badge_override['text_offset_world'][0],cy+badge_override['text_offset_world'][1],r'$q_t$',ha='center',va='center',fontsize=badge['font_size_pt'],color=badge['text_color'],zorder=8.6)
    for label in gates['region_labels']:
        ax.text(*label['position'],label['id'],ha='center',va='center',fontsize=9,color='#334c37',
                bbox=dict(fc='#fffffa',ec='none',pad=.35,alpha=.85),zorder=10)
    ax.set_xlim(frame['x_min'],frame['x_max']);ax.set_ylim(frame['y_min'],frame['y_max'])
    ax.add_patch(Rectangle((frame['x_min'],frame['y_min']),frame['x_max']-frame['x_min'],frame['y_max']-frame['y_min'],fc='none',ec='#65725e',lw=.9,zorder=20,clip_on=False))
    handles=[LargeGateLegend(colors['total'],colors['total_inner'],'branch_head'),LargeGateLegend(colors['regional'],colors['regional_inner'],True)]
    fig.legend(handles=handles,labels=['总闸：当期总量决策','区域入口'],handler_map={LargeGateLegend:LargeGateHandler()},
               loc='center',bbox_to_anchor=(.5,.047),ncol=2,fontsize=7.5,handleheight=1.6,frameon=False,columnspacing=3)
    output=ROOT/'figure.pdf'
    fig.savefig(output);fig.savefig(output.with_suffix('.svg'));fig.savefig(output.with_suffix('.png'),dpi=220);plt.close(fig)
    import fitz
    doc=fitz.open(output);page=doc[0]
    receipt=dict(status='RENDERED',pdf=str(output),pdf_sha256=sha(output),layout_sha256=sha(layout_path),
                 symbols_sha256=sha(symbol_path),water_band_sha256=sha(band_path),
                 final_pdf_size_points=[page.rect.width,page.rect.height],final_pdf_width_mm=page.rect.width/72*25.4,
                 regional_label_font='Times New Roman',regional_label_actual_size_pt=9,
                 branch_name_actual_size_pt=7.5,upstream_name_actual_size_pt=8,q_t_actual_size_pt=9.5,
                 source_regions=89,service_cells=1257,actual_area_mu=12618,full_rectangular_hex_matrix=d['offset_matrix'],
                 long_reaches=5,broad_turns=4,display_aspect_ratio=aspect,
                 source_chainages_and_display_positions_distinct=True,old93_models_actions_yields_loaded=False,
                 true_black_700_channel_name_outlines=True,all_entries_on_own_median_frontage_hex=True,
                 all_source_registered_gate_indices_monotone=True,physical_environment_or_model_changed=False)
    receipt.update(style_overrides_sha256=sha(ROOT/'style_overrides.json'),upstream_arrow_shifted_x=shifted_x,q_t_badge_world=gates['total_decision_badge'],skipped_arrows=skipped_arrows)
    (ROOT/'figure_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
