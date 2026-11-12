"""Readable one-step public render: allocation accounts and one field result.

This module reads only its arguments. A delayed report is never labelled as
the result of today's action, and unavailable step context remains unknown.
RGB and SVG share one drawing description so their content stays consistent.
"""
from __future__ import annotations

import html
import math
from numbers import Integral
import unicodedata

import numpy as np


_INK = "#173548"
_MUTED = "#526a7b"
_BORDER = "#ccd7e1"


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _at(value, index):
    if value is None:
        return None
    try:
        return value[index]
    except (IndexError, TypeError):
        return None


def _format(value, digits=1, suffix=""):
    number = _number(value)
    if number is None:
        return "—"
    text = f"{number:,.{digits}f}".rstrip("0").rstrip(".") if digits else f"{number:,.0f}"
    return text + suffix


def _short_water(value):
    if value >= 10000:
        return _format(value / 10000, 1, "万m³")
    if value >= 1000:
        return _format(value / 1000, 1, "千m³")
    return _format(value, 0, "m³")


def _hongmiao_drawing(obs, ids, *, chapter, action, view, ctx, selected, layout):
    """Approved two-sided schematic with live, public step values only."""
    from .direct_visualization import _capacity, _view, _blend, STAGE_NAMES, STAGE_COLORS, VIEW_LABELS
    ops = []
    def rect(x, y, w, h, fill, stroke=None, sw=1, radius=0, **attrs):
        ops.append(('rect', dict(x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,sw=sw,radius=radius,attrs=attrs)))
    def text(x,y,value,size=12,color=_INK,bold=False,align='left',max_width=None,**attrs):
        value=str(value)
        if max_width:
            units=sum(1. if unicodedata.east_asian_width(c) in 'WF' else .57 for c in value)
            size=min(size,max_width/max(units,1.))
        ops.append(('text',dict(x=x,y=y,value=value,size=size,fill=color,bold=bold,align=align,outline=False,attrs=attrs)))
    def line(x1,y1,x2,y2,color=_BORDER,width=1,**attrs):
        ops.append(('line',dict(x1=x1,y1=y1,x2=x2,y2=y2,fill=color,width=width,attrs=attrs)))
    def path(points,color,width,**attrs):
        ops.append(('polyline',dict(points=points,fill=color,width=width,attrs=attrs)))
    # Keep the public image size; devote the full width to all 93 areas.
    mx,my,mw,mh=28.,136.,1224.,574.
    sx,sy=mw/layout['width'],mh/layout['height']
    scale=min(sx,sy)
    def pt(p): return mx+p[0]*sx,my+p[1]*sy
    def box(b): return mx+b[0]*sx,my+b[1]*sy,b[2]*sx,b[3]*sy
    day=int(ctx.get('decision_step_index',obs['day_index']))
    has_step=bool(ctx.get('has_step',day>0))
    complete=bool(ctx.get('episode_complete',day>=int(obs['horizon_days'])))
    if action is None and not has_step: action=np.zeros(len(ids))
    total=float(math.fsum(action.tolist())) if action is not None else _number(ctx.get('allocated_m3'))
    orange,green,bank= '#C45A18','#EEF4E8',layout['channel_bank_color']
    water=layout['channel_color']
    rect(0,0,1280,900,'#f4f7f9')
    text(28,34,('开始配置前' if not has_step else f'第{day}次配置后')+' · 洪庙支渠资源配置',24,bold=True)
    date_text=(f"{ctx.get('executed_date','—')} → {ctx.get('state_date','—')}" if has_step else ctx.get('state_date','日期 —'))
    text(1252,32,f'{date_text}  |  第{chapter}章  |  决策 {day}/{int(obs["horizon_days"])}',13,align='right',max_width=525,data_role='allocation-date')
    text(28,55,'洪庙93个水稻灌溉区 · 19,908亩 · 同桩号分水口在图上两侧独立引出',12,color=_MUTED)
    irrigated = None if action is None else int(np.count_nonzero(action > 0))
    allocation_summary = ('尚未执行配置' if not has_step else '本步灌溉区域数未知' if irrigated is None
                          else f'本步灌溉 {irrigated} 区 · 未灌溉 {len(ids)-irrigated} 区')
    text(1252,55,allocation_summary,12,color=_MUTED,align='right',max_width=415,
         data_role='allocation-counts',data_irrigated_count='' if not has_step or irrigated is None else irrigated,
         data_non_irrigated_count='' if not has_step or irrigated is None else len(ids)-irrigated)
    carryable=bool(ctx.get('carryable',obs.get('budget_is_carryable',False)))
    before=ctx.get('available_before_m3',obs.get('available_water_m3') if not has_step else None)
    unused=ctx.get('unused_allowance_m3') if has_step else None
    fourth=('回合未用总预算' if complete else '剩余跨期预算') if carryable else ('回合已结束' if complete else '下一天额度')
    fourth_value=ctx.get('remaining_after_m3',obs.get('remaining_budget_m3')) if carryable else ctx.get('next_available_m3',obs.get('available_water_m3'))
    cards=[('本步可配置',before),('本步已配置',total),('本步未用' if has_step else '尚未执行',unused),(fourth,fourth_value)]
    for j,(label,value) in enumerate(cards):
        x=28+j*310
        rect(x,68,294,58,'white',_BORDER,radius=6)
        text(x+12,88,label,12,color=_MUTED)
        text(x+12,114,_format(value,1,' m³'),21,color=orange if j==1 else _INK,bold=True)
    rect(20,132,1240,594,'white',_BORDER,radius=7)
    # Legend leaves the approved inlet/total node geometry uncluttered.
    rect(315,145,13,9,green,'#7C8D79')
    text(334,154,'种植区域',10,color=_MUTED)
    rect(425,146,34,6,'#DAE7E5');rect(425,146,23,6,bank)
    text(466,154,'田内储水 / 容量',10,color=_MUTED)
    rect(600,145,13,9,green,orange,1.6);text(619,154,'本次获得配置',10,color=_MUTED)
    rect(751,145,13,9,None,'#173548',1.4);text(770,154,'当前查看区域',10,color=_MUTED)
    text(1239,154,'矩形面积 ∝ 中稻面积；面积单位：亩',10,color=_MUTED,align='right')
    for guide in layout['row_guides']:
        if guide['row']:
            text(*pt([guide['x'],guide['y']]),guide['text'],8,color=_MUTED)
    for points in [layout['inlet']['path'],layout['channel_path']]:
        path([pt(p) for p in points],bank,layout['channel_bank_width']*scale,data_role='main-canal')
        path([pt(p) for p in points],water,layout['channel_width']*scale)
    for points in layout['channel_arrows']:
        path([pt(p) for p in points],'#286A92',max(.8,1.6*scale),data_role='main-direction')
    inlet=layout['inlet']
    def gate(center,half_w,half_h,role,**attrs):
        x,y=pt(center);dx,dy=half_w*sx,half_h*sy
        path([(x-dx,y-dy),(x-dx,y+dy)],'#395D70',1.1,data_role=role,**attrs)
        path([(x+dx,y-dy),(x+dx,y+dy)],'#395D70',1.1)
        line(x-dx,y,x+dx,y,'#395D70',1.1)
    gate(inlet['gate_center'],inlet['gate_half_width'],inlet['gate_half_height'],'inlet-gate')
    text(*pt(inlet['label_position']),inlet['name']+' · '+inlet['gate_name'],inlet['label_font_size']*scale,bold=True)
    node=layout['total_allocation_node']
    path([pt(p) for p in node['diamond']+[node['diamond'][0]]],orange,1.25,data_role='total-allocation-node')
    for key in ('label','symbol','equation'):
        item=node[key];text(*pt(item['position']),item['text'],item['font_size']*scale,color=orange if key=='symbol' else _INK,align='center')
    item=node['value']
    text(*pt(item['position']),_format(total,1,' m³'),item['font_size']*scale,color=orange,bold=True,align='center',data_role='total-allocation-value',data_total_m3=total if total is not None else '',data_allocation_m3=total if total is not None else '')
    connector=[pt(p) for p in node['connector']]
    path(connector,orange,1.3)
    ax,ay=connector[-1];path([(ax-5,ay-3),(ax,ay),(ax-5,ay+3)],orange,1.3)
    _,colors,maximum=_view(obs,action,view)
    ratios=np.clip(np.asarray(obs['reported_field_storage_mm'])/_capacity(obs),0.,1.)
    for r in layout['regions']:
        i=r['index'];g=r['outlet'];attrs={'data_region_id':r['region_id'],'data_index':i}
        start,end=pt(g['connector_start']),pt(g['service_path'][-1])
        half_width=g['connector_width']*sx/2
        top,bottom=sorted((start[1],end[1]))
        rect(start[0]-half_width,top,half_width*2,bottom-top,water,data_role='water-connector',**attrs)
        for cx in (start[0]-half_width,start[0]+half_width):
            line(cx,top,cx,bottom,bank,max(.5,1.2*layout['width']/2240*scale))
        if g['kind']=='culvert_gate':gate(g['center'],g['half_width'],g['half_height'],'division-gate',**attrs)
        path([pt(p) for p in g['supply_arrow_shaft']],'#286A92',.8,data_role='supply-direction',**attrs)
        path([pt(p) for p in g['supply_arrow']],'#286A92',.8)
        text(*pt(g['label_position']),r['label'],g['label_font_size']*scale,bold=True,data_role='outlet-id',**attrs)
        amount=None if action is None else float(action[i])
        positive=amount is not None and amount>0
        b=box(r['rect'])
        rect(*b,green if view=='water' else colors[i],orange if positive else '#7C8D79',1.45 if positive else .65,data_role='rice-area',data_allocation_m3='' if amount is None else amount,**attrs)
        bx,by,bw,bh=box(r['storage_bar']['rect'])
        rect(bx,by,bw,bh,'#DAE7E5',radius=.5,data_role='storage-background',data_storage_ratio=float(ratios[i]),**attrs)
        rect(bx,by,bw*float(ratios[i]),bh,bank,radius=.5,data_role='storage-water',data_storage_ratio=float(ratios[i]),**attrs)
        label=r['area_label']
        text(*pt(label['position']),f"{r['area_mu']:g}亩",label['font_size']*scale,color='#52655B',align='center',data_role='area-label',**attrs)
    # Selection is independent of the orange applied-allocation marker.
    x,y,w,h=box(layout['regions'][selected]['rect'])
    rect(x-2,y-2,w+4,h+4,None,'#173548',1.,data_role='selection-outline',data_region_id=ids[selected])
    if view != 'water':
        text(40,711,'区域底色：'+VIEW_LABELS[view],9,color=_MUTED,max_width=280)
        if view == 'stage':
            for j,(label,color) in enumerate(zip(STAGE_NAMES,STAGE_COLORS)):
                x=340+j*147
                rect(x,700,14,10,color,data_role='color-legend')
                text(x+20,711,label,9,color=_MUTED)
        else:
            for j in range(48):
                rect(390+j*9,701,9.2,8,_blend(j/47),data_role='color-legend')
            text(380,711,'0',9,color=_MUTED,align='right')
            text(833,711,_format(maximum,2),9,color=_MUTED)
            if view == 'yield':
                rect(920,700,14,10,'#e5e9ed')
                text(940,711,'尚未评价',9,color=_MUTED)
            text(1239,711,'蓝条仍为储水比例',9,color=_MUTED,align='right')
    # A compact detail strip leaves the map at full width.
    r=layout['regions'][selected];amount=None if action is None else float(action[selected])
    for j in range(4):rect(28+j*310,737,294,107,'white',_BORDER,radius=6)
    text(40,756,r['outlet_name']+'对应灌溉区',14,bold=True,max_width=268)
    text(40,776,f"同断面 {r['section_code']} · {r['area_mu']:g}亩",11,color=_MUTED)
    text(40,801,('本步未灌溉：' if has_step and amount == 0 else '本步配置 ')+_format(amount,1,' m³'),16,bold=True,max_width=268)
    depth=None if amount is None else amount/(10.*float(obs['rice_area_ha'][selected]))
    pattern=ctx.get('allocation_pattern') or {}
    dry_days=_number(_at(pattern.get('consecutive_non_irrigation_days'),selected))
    depth_note='折合入田水深 '+_format(depth,2,' mm')
    if has_step and dry_days is not None:
        depth_note+=' · 连续未灌溉 '+_format(dry_days,0,'天')
    text(40,826,depth_note,12,color=_MUTED,max_width=270)
    previous=_number(_at(ctx.get('previous_reported_storage_mm'),selected));current=float(obs['reported_field_storage_mm'][selected])
    transition=has_step and bool(_at(ctx.get('transition_is_current'),selected))
    text(350,756,'田内水分：执行前 → 执行后' if transition else '初始田内水分' if not has_step else '收到的田内水分',13,bold=True)
    text(350,781,f'{_format(previous,2)} → {_format(current,2)} mm' if transition else _format(current,2,' mm'),18,bold=True)
    text(350,804,f'水分变化 {current-previous:+.2f} mm' if transition and previous is not None else '尚未执行灌溉' if not has_step else '缺少执行前对照',12)
    net=_at(ctx.get('public_net_other_mm'),selected) if transition else None
    text(350,825,'降雨−蒸散−渗漏−溢流 '+_format(net,2,' mm'),10,color=_MUTED,max_width=270)
    settled=bool(obs['settled_mask'][selected])
    text(660,756,'已完成最终评价' if settled else '当前阶段 '+STAGE_NAMES[int(obs['stage_index'][selected])],13,bold=True)
    params=np.asarray(obs['field_params'],dtype=float)[selected]
    available_capacity=params[0]*(params[2]-params[3])
    text(660,779,'根区可利用储水容量 '+_format(available_capacity,2,' mm'),12,color=_MUTED,max_width=270)
    text(660,802,'模拟相对产量 '+(_format(obs['settled_relative_yield'][selected],4) if settled else '—'),15,bold=True,max_width=270)
    guarantee=ctx.get('regional_guarantee');feedback=ctx.get('constraint_feedback',{})
    if guarantee:
        target=guarantee['guarantee_target'][selected];limit=guarantee['shortfall_tolerance'][selected]
        cost=feedback['constraint_costs'][selected] if feedback.get('constraints_evaluated') else None
        text(660,825,f'参考{100*target:g}% · 允许平均不足{100*limit:g}个百分点',10,color=_MUTED,max_width=270)
    else:text(660,825,'配置期末模型评价，不表示实际成熟。',10,color=_MUTED)
    text(970,756,'本步奖励 / 累计回报',13,bold=True)
    text(970,781,_format(ctx.get('reward',0. if not has_step else None),5)+' / '+_format(ctx.get('cumulative_return',0. if not has_step else None),5),18,bold=True,max_width=268)
    if guarantee:
        text(970,804,'本回合不足 '+(_format(100*cost,2,' 个百分点') if cost is not None else '期末评价'),12,max_width=268)
        text(970,825,'单回合不足不能判定长期保障达标。',10,color=_MUTED)
    else:
        text(970,805,f"本步新评价 {_format(ctx.get('newly_settled_count',0 if not has_step else None),0)} 个区域",12)
        text(970,825,f"已评价区域 {int(np.count_nonzero(obs['settled_mask']))} / 93",12,color=_MUTED)
    text(28,865,'同桩号分水口各自供水；两侧布置为示意，不代表实际岸别。渠形、间距和渠宽均为示意，田块不是实测边界。',11,color=_MUTED)
    text(28,888,'公开观测 · 面积与桩号：CN119514977A 表2 · 蓝条为储水比例，橙框为本步获得配置；未灌溉区域仍逐日计算水分变化。',10,color=_MUTED)
    return ops


def _drawing(observation, region_ids, *, chapter, last_action, view, context, selected_region):
    # Local import permits direct_visualization wrappers to delegate here.
    from .direct_visualization import select_layout, _view, _capacity, _blend, STAGE_NAMES, STAGE_COLORS, VIEW_LABELS

    obs, ctx = observation, {} if context is None else context
    ids = list(region_ids)
    areas = np.asarray(obs["rice_area_ha"], dtype=float)
    layout = select_layout(ids, areas, context=ctx, width=1208., height=640.)
    hongmiao = "row_guides" in layout
    n = len(ids)
    if last_action is None:
        last_action = ctx.get("executed_action_m3")
    action = None if last_action is None else np.asarray(last_action, dtype=float)
    if action is not None and (action.shape != (n,) or np.any(~np.isfinite(action)) or np.any(action < 0)):
        raise ValueError("last_action must be a finite nonnegative vector with one value per region")
    if selected_region is None:
        selected = int(np.argmax(action)) if action is not None else 0
    elif isinstance(selected_region, (bool, np.bool_)) or not isinstance(selected_region, Integral) or not 0 <= selected_region < n:
        raise ValueError("selected_region must be a zero-based region index")
    else:
        selected = int(selected_region)
    day = int(ctx.get("day", obs["day_index"]))
    has_step = bool(ctx.get("has_step", day > 0))
    decision_step = int(ctx.get("decision_step_index", day))
    complete = bool(ctx.get("episode_complete", day >= int(obs["horizon_days"])))
    _, colors, maximum = _view(obs, action, view)
    if hongmiao:
        return _hongmiao_drawing(obs, ids, chapter=chapter, action=action, view=view,
                                ctx=ctx, selected=selected, layout=layout)
    ops = []

    def rect(x, y, w, h, fill, stroke=None, sw=1, radius=0):
        ops.append(("rect", dict(x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, sw=sw, radius=radius)))

    def text(x, y, value, size=15, color=_INK, bold=False, align="left", max_width=None, outline=False):
        value = str(value)
        if max_width:
            units = sum(1. if unicodedata.east_asian_width(c) in "WF" else .57 for c in value)
            size = min(size, max_width / max(units, 1.))
        ops.append(("text", dict(x=x, y=y, value=value, size=size, fill=color,
                                 bold=bold, align=align, outline=outline)))

    def line(x1, y1, x2, y2, color=_BORDER, width=1):
        ops.append(("line", dict(x1=x1, y1=y1, x2=x2, y2=y2, fill=color, width=width)))

    def polyline(points, color, width):
        ops.append(("polyline", dict(points=points, fill=color, width=width)))

    rect(0, 0, 1280, 900, "#f1f5f9")
    title = f"第{decision_step}次配置后" if has_step else "开始配置前"
    text(28, 39, f"{title} · 直接灌田", 27, bold=True)
    execution_date = ctx.get("executed_date") if has_step else ctx.get("state_date")
    interval = f"{execution_date} → {ctx.get('state_date')}" if has_step else (execution_date or "日期 —")
    right_header = f"{interval}  |  第{chapter}章  |  决策{day}/{int(obs['horizon_days'])}"
    text(1252, 37, right_header, 15, align="right", max_width=530)
    text(28, 68, "洪庙93个水稻灌溉区 · 涵闸编号在渠旁，区域内标面积。" if hongmiao else "每块是一组统一配水的农田；大小按模拟面积，位置为示意。", 15, color=_MUTED)

    allocated = ctx.get("allocated_m3", float(action.sum()) if action is not None else (0. if not has_step else None))
    carryable = bool(ctx.get("carryable", obs.get("budget_is_carryable", False)))
    after_label = ctx.get("budget_after_label") or ("配置后剩余总预算" if carryable else "后续每日额度合计")
    cards = [
        ("本步可配置", ctx.get("available_before_m3", obs.get("available_water_m3") if not has_step else None), "执行前允许使用的水量", "#2c5574"),
        ("本步已配置", allocated, "已经全部灌入各区域农田", "#216f80"),
        ("本步未用", ctx.get("unused_allowance_m3"), "回合结束，不再配置" if complete else "仍在跨期预算内" if carryable else "固定日总量应足额配置", "#956932"),
        (after_label, ctx.get("remaining_after_m3", obs.get("remaining_budget_m3")), "回合结束，无后续配置" if complete else "可供后续安排" if carryable else ("固定日总量应足额配置" if "remaining_after_m3" in ctx else "按后续日期执行固定总量"), "#416577"),
    ]
    for j, (label, value, note, accent) in enumerate(cards):
        x = 28 + j * 310
        rect(x, 91, 294, 103, "white", _BORDER, radius=8)
        rect(x, 91, 5, 103, accent, radius=2)
        text(x+16, 116, label, 14, color=_MUTED, max_width=263)
        text(x+16, 151, _format(value, 1, " m³"), 24, bold=True, max_width=262)
        text(x+16, 176, note, 12, color=_MUTED, max_width=262)

    next_available = ctx.get("next_available_m3", obs.get("available_water_m3"))
    text(28, 223, (f"93 个水稻灌溉区  ·  面积 {_format(layout['total_area_mu'],0)} 亩 / {_format(float(areas.sum()),1)} ha" if hongmiao else f"{n} 个独立配置区域  ·  种植面积 {_format(float(areas.sum()), 0)} ha"), 15, bold=True)
    text(1252, 223, "回合已结束" if complete else f"下次可配置 {_format(next_available, 1, ' m³')}", 15, align="right")

    rect(28, 239, 854, 490, "white", _BORDER, radius=8)
    text(43, 265, f"颜色：{VIEW_LABELS[view]}    " + ("渠旁：涵闸编号 · 格内：面积（亩） · 同断面共用接点" if hongmiao else "格内：编号 / 本步配置量"), 13, color=_MUTED, max_width=817)
    mx, my, mw, mh = 43., 279., 824., 400.
    for guide in layout.get("row_guides", []):
        text(mx+guide["x"]/layout["width"]*mw, my+guide["y"]/layout["height"]*mh,
             guide["text"], 9, color=_MUTED, max_width=mw)
    def map_point(point):
        return mx+point[0]/layout["width"]*mw, my+point[1]/layout["height"]*mh

    if hongmiao:
        blue, bank = layout["channel_color"], layout["channel_bank_color"]
        canal_width = layout["channel_width"]/layout["width"]*mw
        for path in [layout["inlet"]["path"], layout["channel_path"]]:
            points = [map_point(p) for p in path]
            polyline(points, bank, canal_width+3.)
            polyline(points, blue, canal_width)
            polyline(points, layout["channel_highlight_color"], max(1.,canal_width*.17))
        for path in layout["channel_arrows"]:
            polyline([map_point(p) for p in path], "#ffffff", 1.6)
        for path in layout["channel_ripples"]:
            polyline([map_point(p) for p in path], "#b4e7fb", 1.)
        inlet = layout["inlet"]
        ix, iy = map_point(inlet["gate_center"])
        size = inlet["gate_size"]/layout["width"]*mw
        rect(ix-size/2, iy-size/2, size, size, "#e9f5fa", bank, 1.5)
        line(ix-size*.55, iy, ix+size*.55, iy, color=bank, width=2)
        tx, ty = map_point([inlet["label_x"], inlet["label_y"]])
        text(tx,ty,inlet["name"],10,color=bank,bold=True)
        tx, ty = map_point([inlet["label_x"], inlet["gate_label_y"]])
        text(tx,ty,inlet["gate_name"],8,color=_MUTED)
        tx, ty = map_point([layout["width"]*.27, inlet["label_y"]])
        text(tx, ty, "来水  " + " → ".join(layout["supply_chain"]), 10, color=bank,
             max_width=mw*.70)
        example = layout["service_example"]
        tx, ty = map_point([layout["width"]*.27, inlet["gate_label_y"]])
        text(tx, ty, "供水示例  " + " → ".join(example[k] for k in ("gate", "channel", "service")),
             9, color=_MUTED, max_width=mw*.70)
        for region in layout["regions"]:
            gate = region["outlet"]
            points = [map_point(p) for p in gate["service_path"]]
            polyline(points, bank, 3.)
            polyline(points, "#78c7e9", 1.5)
            gx, gy = map_point(gate["center"])
            size = gate["size"]/layout["width"]*mw
            if gate["kind"] == "culvert_gate":
                rect(gx-size*.7, gy-size/2, size*1.4, size, "#e9f5fa", bank, 1.5)
                line(gx-size*.8, gy, gx+size*.8, gy, color=bank, width=1)
            else:
                line(gx-size/2, gy-size, gx-size/2, gy+size, color=bank, width=2)
                line(gx+size/2, gy-size, gx+size/2, gy+size, color=bank, width=2)
            lx, ly = map_point(gate["label_position"])
            label_size = gate["label_font_size"]/layout["width"]*mw
            text(lx, ly+label_size*.35, region["label"], label_size, color=bank,
                 bold=True, align="center")
    map_rects = []
    for region, color in zip(layout["regions"], colors):
        i = region["index"]
        x, y, w, h = region["rect"]
        x, y, w, h = mx+x/layout["width"]*mw, my+y/layout["height"]*mh, w/layout["width"]*mw, h/layout["height"]*mh
        map_rects.append((x,y,w,h))
        rect(x, y, w, h, color, "#24384c", 1.4)
        if hongmiao:
            area_label = _format(region["area_mu"], 0, "亩")
            text(x+w/2, y+h/2+3, area_label, min(11.,max(5.,h-2)), align="center",
                 max_width=w-3, outline=True)
        else:
            display_action = action is not None and has_step and w >= 38 and h >= 31
            if w >= 17 and h >= 15:
                text(x+w/2, y+h/2+(0 if display_action else 3), str(i+1),
                     min(11.,max(5.,h-2)), align="center", max_width=w-2, outline=True)
            if display_action:
                text(x+w/2, y+h/2+12, _short_water(action[i]), 9, align="center", max_width=w-4, outline=True)
    for tie in layout.get("same_section_groups", []):
        x1, x2, ty = mx+tie["x1"]/layout["width"]*mw, mx+tie["x2"]/layout["width"]*mw, my+tie["y"]/layout["height"]*mh
        line(x1,ty,x2,ty,color="#526a7b")
        line(x1,ty-3,x1,ty,color="#526a7b")
        line(x2,ty-3,x2,ty,color="#526a7b")
    sx, sy, sw, sh = map_rects[selected]
    rect(sx+1.2, sy+1.2, max(0,sw-2.4), max(0,sh-2.4), None, "#cc542e", 3)
    ly = 708
    if view == "stage":
        for j, (name, color) in enumerate(zip(STAGE_NAMES, STAGE_COLORS)):
            x = 43 + j*137
            rect(x, ly-14, 14, 14, color, "#24384c")
            text(x+20, ly-1, name, 12)
    else:
        text(43, ly, "0", 12)
        for j in range(90):
            rect(69+j*2, ly-14, 2.1, 14, _blend(j/89))
        text(260, ly, "100%" if view in ("water", "yield") else _format(maximum, 0), 12)
        if view == "yield":
            rect(340, ly-14, 14, 14, "#e5e9ed", "#24384c")
            text(360, ly, "尚未评价", 12)
        text(867, ly, "橙框：右侧显示的区域", 12, color=_MUTED, align="right")

    # The side panel describes the applied action, then its observable effect.
    px, inner = 900, 918
    rect(px, 239, 352, 490, "white", _BORDER, radius=8)
    selected_item = layout["regions"][selected]
    text(inner, 267, (selected_item["outlet_name"]+"对应灌溉区" if hongmiao else f"区域 {selected+1} · {ids[selected]}"), 19, bold=True, max_width=316)
    text(inner, 291, ("断面编码 "+selected_item["section_code"] + (" · 同断面并列" if len(selected_item['same_section_ids']) > 1 else "") if hongmiao else "一组接受统一配水的农田"), 13, color=_MUTED)
    text(inner, 318, (f"面积 {_format(selected_item['area_mu'],0)} 亩 / {_format(areas[selected],2)} ha" if hongmiao else f"种植面积  {_format(areas[selected], 1, ' ha')}"), 15)
    amount = float(action[selected]) if action is not None else (0. if not has_step else None)
    irrigation_depth = amount/(10*areas[selected]) if amount is not None else None
    text(inner, 346, "本步配置 → 直接灌田", 14, bold=True)
    text(inner, 373, f"{_format(amount, 1, ' m³')}  →  {_format(irrigation_depth, 2, ' mm')}", 19, bold=True, max_width=316)
    line(inner, 390, 1233, 390)
    before = _number(_at(ctx.get("previous_reported_storage_mm"), selected))
    after = float(obs["reported_field_storage_mm"][selected])
    transition_flag = _at(ctx.get("transition_is_current"), selected)
    transition = bool(transition_flag) and has_step
    if not has_step:
        heading, pair = "初始田内水分", _format(after, 2, " mm")
    elif transition:
        heading = "田内水分：执行前 → 执行后"
        pair = f"{_format(before, 2)} → {_format(after, 2)} mm"
    elif transition_flag is not None:
        heading = "收到的水分报告：上次 → 本次"
        pair = f"{_format(before, 2)} → {_format(after, 2)} mm"
    else:
        heading, pair = "已收到的田内水分", _format(after, 2, " mm")
    text(inner, 415, heading, 13, color=_MUTED, max_width=316)
    text(inner, 443, pair, 20, bold=True, max_width=316)
    if transition and before is not None:
        text(inner, 468, f"本步水分变化  {after-before:+.2f} mm", 14)
    elif has_step and transition_flag is not None:
        age = _at(ctx.get("report_age_days", obs.get("report_age_days")), selected)
        prev_day = _at(ctx.get("previous_report_day_index"), selected)
        current_day = _at(ctx.get("current_report_day_index", obs.get("report_day_index")), selected)
        text(inner, 467, f"报告对应日 {_format(prev_day,0)} → {_format(current_day,0)}；报告年龄 {_format(age,0)} 天", 12, color=_MUTED, max_width=316)
    elif not has_step:
        text(inner, 468, "尚未执行灌溉", 14, color=_MUTED)
    else:
        text(inner, 468, "本步水分变化  —", 14, color=_MUTED)
    net = _number(_at(ctx.get("public_net_other_mm"), selected)) if transition else None
    text(inner, 494, "降雨－蒸散－渗漏－溢流（合计）", 13, color=_MUTED, max_width=316)
    text(inner, 520, (f"{net:+.2f} mm" if net is not None else "—"), 18, bold=True)
    explanation = ("水分变化 = 灌溉 + 上述合计" if net is not None else
                   "缺少本步执行前对照信息" if has_step and transition_flag is None else
                   "历史报告不代表本步执行前后水分" if has_step else "执行后可查看水量变化")
    text(inner, 543, explanation, 12, color=_MUTED, max_width=316)
    line(inner, 558, 1233, 558)
    stage = int(obs["stage_index"][selected])
    text(inner, 583, f"当前阶段  {STAGE_NAMES[stage]}", 14)
    settled = bool(obs["settled_mask"][selected])
    text(inner, 609, "产量评价  已完成最终评价" if settled else "产量评价  配置期结束时评价", 14)
    yield_label = _format(obs["settled_relative_yield"][selected], 4) if settled else "尚未评价"
    text(inner, 637, f"模拟相对产量  {yield_label}", 15, bold=True, max_width=316)
    guarantee = ctx.get("regional_guarantee")
    if guarantee:
        target = _at(guarantee.get("guarantee_target"), selected)
        tolerance = _at(guarantee.get("shortfall_tolerance"), selected)
        feedback = ctx.get("constraint_feedback", {})
        cost = _at(feedback.get("constraint_costs"), selected) if feedback.get("constraints_evaluated") else None
        text(inner, 667, f"参考水平 {_format(100*target,1)}% · 允许平均不足 {_format(100*tolerance,2)} 个百分点", 11, color=_MUTED, max_width=316)
        text(inner, 690, f"本回合不足  {_format(100*cost,2,' 个百分点') if cost is not None else '期末评价'}", 13, bold=True, max_width=316)
        text(inner, 715, "单回合不足不代表长期保障是否达标。", 11, color=_MUTED, max_width=316)
    else:
        text(inner, 670, "模型预估产量，不表示当天实际成熟。", 12, color=_MUTED)
        text(inner, 694, "格内为面积，配置水量见上方。" if hongmiao else "格内水量对应本步已执行的配置。", 11, color=_MUTED, max_width=316)

    rect(28, 746, 1224, 95, "#e5eef4", _BORDER, radius=8)
    newly = ctx.get("newly_settled_count", 0 if not has_step else None)
    total_settled = ctx.get("settled_count", int(np.asarray(obs["settled_mask"]).sum()))
    footer = [("本步奖励", _format(ctx.get("reward", 0 if not has_step else None), 5)),
              ("累计回报", _format(ctx.get("cumulative_return", 0 if not has_step else None), 5)),
              ("本步新评价", _format(newly, 0, " 个区域")),
              ("已评价区域", f"{_format(total_settled,0)} / {n}")]
    for j, (label, value) in enumerate(footer):
        x = 48+j*306
        text(x, 773, label, 14, color=_MUTED)
        text(x, 807, value, 24, bold=True, max_width=275)
    text(28, 865, "公开观测视图：按报告时刻显示水分；期末评价模拟相对产量，不表示当天实际成熟。", 12, color=_MUTED)
    text(28, 887, ("面积与断面来自 CN119514977A 表2；折返仅为排版，不表示真实走向、左右岸或田块边界。" if hongmiao else "位置不表示真实方位、距离或相邻关系。田内水分包括土壤和田面中的水，不是尚未使用的配置额度。"), 12, color=_MUTED)
    return ops


def _validate_size(width, height):
    if (isinstance(width, bool) or isinstance(height, bool) or
            not isinstance(width, Integral) or not isinstance(height, Integral) or width < 640 or height < 520):
        raise ValueError("render dimensions must be integers at least 640 x 520")


def render_step_svg(observation, region_ids, *, chapter, last_action=None, view="water",
                    width=1280, height=900, context=None, selected_region=None):
    """Return one standalone SVG of public step context and received reports."""
    _validate_size(width, height)
    ops = _drawing(observation, region_ids, chapter=chapter, last_action=last_action,
                   view=view, context=context, selected_region=selected_region)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 1280 900" role="img" aria-label="单步灌溉配置与状态变化">',
             '<g font-family="Noto Sans CJK SC,Microsoft YaHei,sans-serif">']
    for kind, item in ops:
        attrs = "".join(f' {key.replace("_", "-")}="{html.escape(str(value), quote=True)}"'
                        for key, value in item.get("attrs", {}).items())
        if kind == "rect":
            parts.append(f'<rect x="{item["x"]:.5f}" y="{item["y"]:.5f}" width="{item["w"]:.5f}" height="{item["h"]:.5f}" rx="{item["radius"]}" fill="{item["fill"] or "none"}" stroke="{item["stroke"] or "none"}" stroke-width="{item["sw"]}"{attrs}/>')
        elif kind == "polyline":
            points = " ".join(f"{x},{y}" for x,y in item["points"])
            parts.append(f'<polyline points="{points}" fill="none" stroke="{item["fill"]}" stroke-width="{item["width"]}" stroke-linecap="round" stroke-linejoin="round"{attrs}/>')
        elif kind == "line":
            parts.append(f'<line x1="{item["x1"]}" y1="{item["y1"]}" x2="{item["x2"]}" y2="{item["y2"]}" stroke="{item["fill"]}" stroke-width="{item["width"]}"{attrs}/>')
        else:
            anchor = {"left":"start", "right":"end", "center":"middle"}[item["align"]]
            outline = ' stroke="white" stroke-width="1.5" paint-order="stroke" stroke-linejoin="round"' if item["outline"] else ""
            parts.append(f'<text x="{item["x"]:.5f}" y="{item["y"]:.5f}" font-size="{item["size"]:.3f}" font-weight="{700 if item["bold"] else 400}" text-anchor="{anchor}" fill="{item["fill"]}"{outline}{attrs}>{html.escape(item["value"])}</text>')
    parts.append('</g></svg>')
    return "".join(parts)


def render_step_rgb(observation, region_ids, *, chapter, last_action=None, view="water",
                    width=1280, height=900, context=None, selected_region=None):
    """Return uint8 RGB; does not open a window or read environment truth."""
    _validate_size(width, height)
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError("RGB rendering requires irrigation-allocation-rl[visualization]") from exc
    ops = _drawing(observation, region_ids, chapter=chapter, last_action=last_action,
                   view=view, context=context, selected_region=selected_region)
    # Draw fine channel branches and labels above output resolution, then
    # downsample once. Large requested images retain native text detail.
    raster_scale = max(2., width / 1280., height / 900.)
    image = Image.new("RGB", (round(1280*raster_scale), round(900*raster_scale)), "#f1f5f9")
    draw = ImageDraw.Draw(image)
    fonts = {}

    def font(size, bold, value):
        # Noto CJK does not include every mathematical subscript. Pillow
        # has no browser-style glyph fallback, so use the Latin math font.
        math_text = any('\u2080' <= c <= '\u209f' for c in value)
        key = (max(1, round(size*raster_scale)), bold, math_text)
        if key not in fonts:
            suffix = "Bold" if bold else "Regular"
            candidates = [f"/usr/share/fonts/opentype/noto/NotoSansCJK-{suffix}.ttc",
                          "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
            if math_text:
                candidates.reverse()
            for filename in candidates:
                try:
                    fonts[key] = ImageFont.truetype(filename, key[0])
                    break
                except OSError:
                    pass
            else:
                fonts[key] = ImageFont.load_default()
        return fonts[key]

    for kind, item in ops:
        if kind == "rect":
            xy = tuple(v*raster_scale for v in (item["x"], item["y"], item["x"]+item["w"], item["y"]+item["h"]))
            # A zero-width live water bar must remain empty in RGB as in SVG.
            if item["w"] > 0 and item["h"] > 0:
                radius = min(item["radius"], item["w"]/2, item["h"]/2)*raster_scale
                style = dict(fill=item["fill"], outline=item["stroke"], width=max(1,round(item["sw"]*raster_scale)))
                if radius < 2:
                    draw.rectangle(xy, **style)
                else:
                    draw.rounded_rectangle(xy, radius=radius, **style)
        elif kind == "polyline":
            linewidth = max(1, round(item["width"]*raster_scale))
            points = [(x*raster_scale,y*raster_scale) for x,y in item["points"]]
            draw.line(points, fill=item["fill"], width=linewidth, joint="curve")
            radius = linewidth/2
            for x,y in points:
                draw.ellipse((x-radius,y-radius,x+radius,y+radius), fill=item["fill"])
        elif kind == "line":
            draw.line(tuple(item[k]*raster_scale for k in ("x1","y1","x2","y2")),fill=item["fill"],width=max(1,round(item["width"]*raster_scale)))
        else:
            # 's' uses the baseline, matching SVG text coordinates.
            anchor = {"left":"ls", "right":"rs", "center":"ms"}[item["align"]]
            draw.text((item["x"]*raster_scale,item["y"]*raster_scale),item["value"],font=font(item["size"],item["bold"],item["value"]),
                      fill=item["fill"],anchor=anchor,stroke_width=round(raster_scale) if item["outline"] else 0,
                      stroke_fill="white" if item["outline"] else None)
    if image.size != (width,height):
        image=image.resize((int(width),int(height)),Image.Resampling.LANCZOS)
    return np.asarray(image).copy()
