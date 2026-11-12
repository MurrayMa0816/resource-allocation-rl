"""Area-proportional views of direct irrigation, without inferred geography.

Public renders consume only the supplied public observation. Offline replay
can show evaluator truth only when capture_frame explicitly included it.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np

from .direct_recording import _jsonable


STAGE_NAMES = ("返青/分蘖", "孕穗", "抽穗", "乳熟", "黄熟", "日历成熟")
STAGE_COLORS = ("#c8dda3", "#80bcb1", "#ffd077", "#dca574", "#c4b9db", "#a5afba")
VIEW_LABELS = {"water": "田内水分 / 容量", "allocation": "本次已执行的灌溉量（m³）",
               "stage": "当前生长阶段", "yield": "已评价的模拟相对产量", "age": "报告年龄（天）"}


def build_layout(region_ids, areas_ha, *, width=1208., height=640.):
    """Partition the drawing by area; coordinates carry no distance meaning."""
    ids = list(region_ids)
    areas = np.asarray(areas_ha, dtype=float)
    if (not ids or len(set(ids)) != len(ids) or
            any(not isinstance(i, str) or not i for i in ids)):
        raise ValueError("region_ids must be nonempty unique strings")
    if areas.shape != (len(ids),) or not np.all(np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError("areas_ha must have one finite positive value per region")
    if any(not math.isfinite(v) or v <= 0 for v in (width, height)):
        raise ValueError("drawing dimensions must be positive")
    regions = [None] * len(ids)

    def partition(indices, x, y, w, h):
        if len(indices) == 1:
            i = indices[0]
            regions[i] = {"index": i, "region_id": ids[i], "area_ha": float(areas[i]),
                          "rect": [x, y, w, h]}
            return
        weights = areas[indices]
        total = math.fsum(weights)
        k = int(np.argmin(np.abs(np.cumsum(weights)[:-1] - total / 2))) + 1
        fraction = math.fsum(weights[:k]) / total
        if w >= h:
            partition(indices[:k], x, y, w * fraction, h)
            partition(indices[k:], x + w * fraction, y, w * (1 - fraction), h)
        else:
            partition(indices[:k], x, y, w, h * fraction)
            partition(indices[k:], x, y + h * fraction, w, h * (1 - fraction))

    partition(list(range(len(ids))), 0., 0., float(width), float(height))
    return {"identity": "generated_area_proportional_schematic_not_geographic",
            "width": float(width), "height": float(height),
            "total_area_ha": float(areas.sum()), "regions": regions}


def _capacity(obs):
    if "field_storage_capacity_mm" in obs:
        return np.asarray(obs["field_storage_capacity_mm"], dtype=float)
    params = np.asarray(obs["field_params"], dtype=float)
    return params[:, 0] * params[:, 1] + params[:, 5]


def select_layout(region_ids, areas_ha, *, context=None, width=1208., height=640.):
    """Use published Hongmiao identity only when IDs and areas actually match."""
    from .hongmiao_layout import LAYOUT_ID, build_hongmiao_layout, matches_hongmiao
    ids = list(region_ids)
    explicit = (context or {}).get("layout_kind") == LAYOUT_ID
    if explicit or matches_hongmiao(ids, areas_ha):
        return build_hongmiao_layout(ids, areas_ha, width=width, height=height)
    return build_layout(ids, areas_ha, width=width, height=height)


def _blend(z, low=(255, 235, 181), high=(41, 111, 164)):
    z = float(np.clip(z, 0, 1))
    return "#" + "".join(f"{round(a+(b-a)*z):02x}" for a, b in zip(low, high))


def _view(obs, allocation, view):
    if view not in VIEW_LABELS:
        raise ValueError(f"unknown view: {view}")
    n = len(obs["rice_area_ha"])
    settled = np.asarray(obs["settled_mask"], dtype=bool)
    if view == "stage":
        values = np.asarray(obs["stage_index"], dtype=int)
        colors = [STAGE_COLORS[int(v)] for v in values]
        return values, colors, 5.
    if view == "water":
        values = np.asarray(obs["reported_field_storage_mm"], dtype=float) / _capacity(obs)
        maximum = 1.
    elif view == "allocation":
        values = np.zeros(n) if allocation is None else np.asarray(allocation, dtype=float)
        maximum = max(float(values.max()), 1.)
    elif view == "yield":
        values = np.asarray(obs["settled_relative_yield"], dtype=float)
        maximum = 1.
    else:
        values = np.asarray(obs["report_age_days"], dtype=float)
        maximum = max(float(values.max()), 1.)
    colors = [_blend(v / maximum) for v in values]
    if view == "yield":
        colors = [color if done else "#e5e9ed" for color, done in zip(colors, settled)]
    return values, colors, maximum


def _dimensions(width, height):
    if width < 640 or height < 520:
        raise ValueError("render size must be at least 640 x 520")
    return 36., 158., width - 72., height - 260.


def render_svg(observation, region_ids, *, chapter, last_action=None,
               view="water", width=1280, height=900, context=None, selected_region=None):
    """Render a public single-step account, field map and regional changes."""
    from .direct_step_view import render_step_svg
    return render_step_svg(observation, region_ids, chapter=chapter, last_action=last_action,
                           view=view, width=width, height=height, context=context,
                           selected_region=selected_region)


def render_observation(observation, region_ids, *, chapter, last_action=None,
                       mode="rgb_array", view="water", width=1280, height=900,
                       context=None, selected_region=None):
    """Render public reports and public step bookkeeping without hidden truth."""
    if mode == "svg":
        return render_svg(observation, region_ids, chapter=chapter, last_action=last_action,
                          view=view, width=width, height=height, context=context,
                          selected_region=selected_region)
    if mode != "rgb_array":
        raise ValueError("mode must be 'rgb_array' or 'svg'")
    from .direct_step_view import render_step_rgb
    return render_step_rgb(observation, region_ids, chapter=chapter, last_action=last_action,
                          view=view, width=width, height=height, context=context,
                          selected_region=selected_region)


def write_rollout_html(frames, path, *, metadata=None):
    """Write a self-contained step replay; no CDN, maps, or network requests."""
    frames = list(frames)
    if not frames:
        raise ValueError("a replay needs at least one frame")
    metadata = {} if metadata is None else dict(metadata)
    metadata["access"] = ("privileged_evaluation_not_policy_input" if any("truth" in f for f in frames)
                          else "public_observation_only")
    n = len(frames[0]["observation"]["rice_area_ha"])
    context = frames[0].get("step_context", {})
    ids = metadata.get("region_ids", context.get("region_ids", [f"R{i+1:03d}" for i in range(n)]))
    layout = select_layout(ids, frames[0]["observation"]["rice_area_ha"], context=context)
    metadata["geometry_identity"] = layout["identity"]
    if "layout_variant" in layout:
        metadata["layout_variant"] = layout["layout_variant"]
    if "area_identity" in layout:
        metadata["area_identity"] = layout["area_identity"]
    payload = {"metadata": metadata, "frames": frames, "layout": layout,
               "stage_names": STAGE_NAMES, "stage_colors": STAGE_COLORS}
    data = json.dumps(_jsonable(payload), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    document = _HTML.replace("__REPLAY_DATA__", data)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


_HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>直接灌田 · 每步配置与变化</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#edf2f7;color:#173548;font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif}main{max-width:1640px;margin:auto;padding:22px}h1{font-size:27px;margin:0 0 8px}h3{margin:0 0 7px}p{line-height:1.6;margin:6px 0}.muted{color:#526778;font-size:13px}.panel{background:#fff;border:1px solid #ccd6df;border-radius:11px;padding:16px;margin-top:14px}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}button,select{font:inherit;font-size:14px;padding:7px 10px;border:1px solid #a6b7c5;border-radius:6px;background:white;color:#173548}button{cursor:pointer}button.primary{background:#173f59;color:white}input[type=range]{flex:1;min-width:160px}#status{font-size:14px;font-variant-numeric:tabular-nums}.grid{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:20px}.accounts,.rewards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.accounts{margin-top:14px}.account{background:white;border:1px solid #ccd6df;border-left:5px solid #477588;border-radius:8px;padding:12px 15px}.account:nth-child(2){border-left-color:#238075}.account:nth-child(3){border-left-color:#ac7b38}.account label,.rewards label{font-size:13px;color:#526778;display:block}.account strong,.rewards strong{display:block;font-size:24px;margin:4px 0;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.account small{font-size:12px;color:#526778}.rewards{background:#e5eef4}.legend{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:0 0 10px;font-size:12px}.swatch{display:inline-block;width:15px;height:15px;border:1px solid #24384c;vertical-align:middle;margin-right:5px}.bar{display:inline-block;width:150px;height:14px;background:linear-gradient(90deg,#ffebb5,#296fa4)}.map-viewport{overflow:auto}#map{width:100%;display:block}#map g.region{cursor:pointer}body:not(.hongmiao) #map g.region:focus rect,body:not(.hongmiao) #map g.region:hover rect{stroke:#071c2b;stroke-width:4}#detail{font-size:13px;border-left:1px solid #d5dfe7;padding-left:18px}#detail .amount{font-size:23px;font-weight:700;color:#216f80;margin:5px 0}#detail .transition{font-size:22px;font-weight:700;margin:6px 0}#detail .block{border-top:1px solid #d5dfe7;margin-top:13px;padding-top:12px}#detail .row{display:flex;justify-content:space-between;gap:12px;margin:6px 0}#detail .row span{color:#526778}#detail .row b{text-align:right}.notice{padding:8px 12px;background:#fff1ce;border-left:4px solid #a27516;margin-top:12px;font-size:13px}.selected{stroke:#cc542e!important;stroke-width:4!important}details{margin-top:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}a{color:#173f59}@media(max-width:1050px){.grid{grid-template-columns:minmax(0,1fr) 285px}.account strong{font-size:20px}}@media(max-width:800px){main{padding:12px}.grid{grid-template-columns:1fr}.accounts,.rewards{grid-template-columns:repeat(2,minmax(0,1fr))}#detail{border-left:0;border-top:1px solid #d5dfe7;padding:15px 0 0}.bar{width:110px}}

body.hongmiao main{max-width:2240px}.hongmiao .grid{grid-template-columns:minmax(0,1fr)}.hongmiao #detail{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px 24px;border-left:0;border-top:1px solid #d5dfe7;padding:16px 0 0}.hongmiao #detail>h3,.hongmiao #detail>p{grid-column:1/-1}.hongmiao #detail .block{margin-top:0}.hongmiao #map g.region:hover .rice-area,.hongmiao #map g.region:focus .rice-area{filter:brightness(.97)}.hongmiao .selection-key{border:2px dashed #173548;background:white}.hongmiao .allocation-key{border:2px solid #c45a18;background:#eef4e8}.hongmiao .storage-key{width:42px;height:9px;background:linear-gradient(90deg,#3478a5 70%,#dae7e5 70%);border:0}.hongmiao #map{min-width:920px}.hongmiao .map-viewport{background:white}.hongmiao #layout-note{max-width:1250px}@media(max-width:800px){.hongmiao #detail{grid-template-columns:minmax(0,1fr)}.hongmiao #map{min-width:920px}}
</style>
<main><h1 id="heading">直接灌田 · 每步配置与变化</h1>
<p class="muted" id="layout-note">每块是一组统一配水的农田；大小按模拟种植面积，位置为示意。格内显示编号和本步配置量，点击查看变化。</p>
<section class="panel"><div class="controls"><button id="previous" aria-label="上一帧">◀</button><button id="play" class="primary">播放</button><button id="next" aria-label="下一帧">▶</button><input id="slider" aria-label="回放日边界" type="range" min="0" value="0"><span id="status"></span></div>
<div class="controls" style="margin-top:10px"><label>颜色 <select id="view"><option value="water">田内水分 / 容量</option><option value="allocation">本步已配置水量</option><option value="stage">当前生长阶段</option><option value="yield">已评价的相对产量</option><option value="age">报告年龄</option></select></label><label>信息 <select id="access"><option value="reported">配置中心收到的报告</option><option value="truth">评估者当前真值</option></select></label><label>缩放 <select id="zoom"><option value="100">100%</option><option value="150">150%</option><option value="200">200%</option><option value="300">300%</option></select></label><label>区域 <select id="region-select" aria-label="选择区域"></select></label><button id="export">导出当前 SVG</button><span class="muted" id="next-allowance"></span></div>
<div id="privileged" class="notice" hidden>评估者视图：完整逐项通量用于离线核查；默认策略只读取正式状态接口。</div></section>
<div class="accounts" id="stats"></div>
<p class="muted" id="allocation-status"></p>
<section class="panel grid"><div><div class="legend" id="legend"></div><div class="map-viewport"><svg id="map" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="按模拟种植面积绘制的配置区域"></svg></div><p class="muted" id="timing"></p><p class="muted" id="map-source" hidden></p></div><aside id="detail"></aside></section>
<section class="panel rewards" id="rewards"></section>
<p class="muted">配置期内显示“尚未评价”；配置期结束时统一评价最终模拟相对产量。田内水分是土壤和田面中的水，不是尚未使用的配置额度。</p>
<details class="panel"><summary>场景与运行记录</summary><pre id="metadata"></pre></details></main>
<script id="replay-data" type="application/json">__REPLAY_DATA__</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById('replay-data').textContent), F=D.frames, L=D.layout;
const isHongmiao=Boolean(L.row_guides);
if(isHongmiao)document.body.classList.add('hongmiao');
const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
let frameIndex=0, selected=0, timer=null;
const labels={water:'田内水分占容量比例',allocation:'本步实际灌入田地的水（m³）',stage:'当前生长阶段',yield:'配置期结束时评价的模拟相对产量',age:'收到的报告年龄（天）'};
const known=x=>x!==null&&x!==undefined&&Number.isFinite(Number(x));
const fmt=(x,d=1)=>known(x)?Number(x).toLocaleString('zh-CN',{maximumFractionDigits:d}):'—';
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const shortWater=x=>!known(x)?'—':x>=10000?fmt(x/10000,1)+'万m³':x>=1000?fmt(x/1000,1)+'千m³':fmt(x,0)+'m³';
const item=(a,i)=>Array.isArray(a)?a[i]:null;
function capacity(o,i){return o.field_storage_capacity_mm?o.field_storage_capacity_mm[i]:o.field_params[i][0]*o.field_params[i][1]+o.field_params[i][5]}
function color(z){z=Math.min(1,Math.max(0,z));return 'rgb('+[255,235,181].map((v,i)=>Math.round(v+([41,111,164][i]-v)*z)).join(',')+')'}
function node(tag,attrs={},text){let el=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(text!==undefined)el.textContent=text;return el}
function previousFrame(k){return k>0&&F[k-1].day===F[k].day-1?F[k-1]:null}
function precedingDate(date){if(!/^\d{4}-\d{2}-\d{2}$/.test(date||''))return null;let t=Date.parse(date+'T00:00:00Z');return Number.isFinite(t)?new Date(t-86400000).toISOString().slice(0,10):null}
function context(k){
  const f=F[k],o=f.observation,p=previousFrame(k),po=p?p.observation:null,has=Number(f.day)>0,carry=Boolean(o.budget_is_carryable);
  const a=f.allocation_m3,used=Array.isArray(a)&&a.every(known)?a.reduce((s,x)=>s+Number(x),0):null;
  const irrigated=has&&Array.isArray(a)&&a.length===L.regions.length&&a.every(x=>known(x)&&Number(x)>=0)?a.filter(x=>Number(x)>0).length:null;
  const before=has?(po?po.available_water_m3:null):o.available_water_m3;
  const unused=known(before)&&known(used)?Math.max(0,before-used):null;
  const prevIndex=po?po.report_day_index:null,current=o.report_day_index;
  const currentTransition=L.regions.map((r,i)=>Boolean(has&&po&&item(prevIndex,i)===f.day-1&&item(current,i)===f.day));
  const net=L.regions.map((r,i)=>currentTransition[i]&&known(item(a,i))?o.reported_field_storage_mm[i]-po.reported_field_storage_mm[i]-a[i]/(10*o.rice_area_ha[i]):null);
  const completePrefix=F[0].day===0&&F.slice(0,k+1).every((r,j)=>j===0||r.day===F[j-1].day+1);
  const cumulative=known(f.return)?f.return:completePrefix&&F.slice(0,k+1).every(r=>known(r.reward))?F.slice(0,k+1).reduce((s,r)=>s+r.reward,0):null;
  const c={day:f.day,has_step:has,executed_date:has?(f.decision_date||precedingDate(f.date)):null,state_date:f.date,
    carryable:carry,available_before_m3:before,allocated_m3:used,unused_allowance_m3:unused,
    remaining_after_m3:carry?o.remaining_budget_m3:unused,next_available_m3:o.available_water_m3,
    reward:f.reward,cumulative_return:cumulative,settled_count:o.settled_mask.filter(Boolean).length,
    newly_settled_count:has?(po?o.settled_mask.filter((v,i)=>v&&!po.settled_mask[i]).length:null):0,
    previous_reported_storage_mm:po?po.reported_field_storage_mm:null,previous_report_day_index:prevIndex,
    current_report_day_index:current,report_age_days:o.report_age_days,
    irrigated_count:irrigated,non_irrigated_count:irrigated===null?null:L.regions.length-irrigated,
    transition_is_current:currentTransition,public_net_other_mm:net};
  return Object.assign(c,f.step_context||{});
}
function water(f,i){return $('access').value==='truth'&&f.truth?f.truth.field_storage_mm[i]:f.observation.reported_field_storage_mm[i]}
function values(f){let o=f.observation,v=$('view').value;return L.regions.map((r,i)=>v==='water'?water(f,i)/capacity(o,i):v==='allocation'?(item(f.allocation_m3,i)||0):v==='stage'?o.stage_index[i]:v==='yield'?o.settled_relative_yield[i]:o.report_age_days[i])}
function title(c){let step=c.decision_step_index??c.day;return c.has_step?'第'+step+'次配置后 · 直接灌田':'开始配置前 · 直接灌田'}
function allocationSummary(c){return !c.has_step?'尚未执行配置':!known(c.irrigated_count)?'本步灌溉区域数未知':'本步灌溉 '+fmt(c.irrigated_count,0)+' 区 · 未灌溉 '+fmt(c.non_irrigated_count,0)+' 区'}
function accountRows(c){return [
  ['本步可配置',c.available_before_m3,'执行前允许使用的水量'],
  ['本步已配置',c.allocated_m3,allocationSummary(c)],
  ['本步未用',c.unused_allowance_m3,c.episode_complete?'回合结束，不再配置':c.carryable?'仍保留在跨期预算中':'固定日总量应足额配置'],
  c.carryable?[c.episode_complete?'回合未用总预算':'配置后剩余总预算',c.remaining_after_m3,c.episode_complete?'回合结束，无后续配置':'可以留给后续日期']:['下一天额度',c.next_available_m3,'下一天固定配置总量']
]}
function rewardRows(c){return [['本步奖励',fmt(c.reward,5)],['累计回报',fmt(c.cumulative_return,5)],['本步新评价',fmt(c.newly_settled_count,0)+' 个区域'],['已评价区域',fmt(c.settled_count,0)+' / '+L.regions.length]]}
function detail(f,c){
  let o=f.observation,i=selected,r=L.regions[i],done=Boolean(o.settled_mask[i]),truth=$('access').value==='truth'&&f.truth;
  let amount=item(f.allocation_m3,i),depth=known(amount)?amount/(10*r.area_ha):null;
  let allocationNote=!c.has_step?'尚未执行配置。':known(amount)&&Number(amount)===0?'本步未灌溉；田间水分仍继续变化。':'折算水深便于比较不同面积的区域。';
  const dryDays=item(c.allocation_pattern?.consecutive_non_irrigation_days,i);
  if(c.has_step&&known(dryDays))allocationNote+=' 连续未灌溉 '+fmt(dryDays,0)+' 天。';
  let before=item(c.previous_reported_storage_mm,i),after=o.reported_field_storage_mm[i],current=Boolean(item(c.transition_is_current,i));
  if(truth){after=item(f.truth.field_storage_mm,i);before=item(f.truth.before_field_storage_mm,i);let p=previousFrame(frameIndex);if(!known(before)&&p&&p.truth)before=item(p.truth.field_storage_mm,i);current=c.has_step&&known(before)}
  let heading=!c.has_step?'初始田内水分':truth?'田内水分：执行前 → 执行后（真值）':current?'田内水分：执行前 → 执行后':'收到的水分报告：上次 → 本次';
  let pair=c.has_step?fmt(before,2)+' → '+fmt(after,2)+' mm':fmt(after,2)+' mm';
  let note=!c.has_step?'尚未执行灌溉':current?'本步水分变化 '+(after-before>=0?'+':'')+fmt(after-before,2)+' mm':truth?'缺少本步执行前真值，不能计算变化':'报告对应日 '+fmt(item(c.previous_report_day_index,i),0)+' → '+fmt(item(c.current_report_day_index,i),0)+'；报告年龄 '+fmt(item(c.report_age_days,i),0)+' 天。历史报告不代表本步灌溉效果。';
  let flow='';
  const row=(k,v)=>'<div class="row"><span>'+esc(k)+'</span><b>'+esc(v)+'</b></div>';
  const g=c.regional_guarantee||D.metadata.run_configuration?.regional_guarantee;
  const feedback=f.constraint_feedback||c.constraint_feedback;
  let guarantee='';
  if(g){guarantee='<div class="block"><b>第六章 · 区域保障</b>'+row('参考产量水平',fmt(100*g.guarantee_target[i],1)+'%')+row('允许平均不足',fmt(100*g.shortfall_tolerance[i],2)+' 个百分点')+row('本回合不足',feedback?.constraints_evaluated?fmt(100*feedback.constraint_costs[i],2)+' 个百分点':'期末评价')+'<p class="muted">先计算每个回合的正不足，再跨回合取平均。单个回合不能判定长期保障达标；这里的阈值是显式研究配置。</p></div>'}
  if(truth&&c.has_step){flow='<div class="block"><b>本步田间水量变化（评估核查）</b>'+[['daily_rain_mm','降雨'],['daily_et_mm','蒸散消耗'],['daily_drainage_mm','有效渗漏'],['daily_overflow_mm','溢流']].map(([key,label])=>row(label,fmt(item(f.truth[key],i),2)+' mm')).join('')+'<p class="muted">执行后 = 执行前 + 灌溉 + 降雨 − 蒸散 − 渗漏 − 溢流。</p></div>'}
  else if(c.has_step){let net=current?item(c.public_net_other_mm,i):null;flow='<div class="block">'+row('降雨 − 蒸散 − 渗漏 − 溢流',known(net)?(net>=0?'+':'')+fmt(net,2)+' mm':'—')+'<p class="muted">'+(known(net)?'只显示公开前后水分可推得的合计，不分拆未观测通量。':'无法从当前收到的报告确认本步水量变化。')+'</p></div>'}
  $('detail').innerHTML='<h3>'+esc(isHongmiao?r.outlet_name+'对应灌溉区':'区域 '+(i+1)+' · '+r.region_id)+'</h3><p class="muted">'+(isHongmiao?'断面编码 '+esc(r.section_code)+' · 面积 '+fmt(r.area_mu,0)+' 亩 / '+fmt(r.area_ha,2)+' ha'+(r.same_section_ids.length>1?'<br>同断面口门 '+esc(r.same_section_ids.join('、'))+'，不区分左右岸。':''):'一组统一接受配水的农田 · '+fmt(r.area_ha)+' ha')+'</p><div class="block"><b>本步配置 → 直接灌田</b><div class="amount">'+fmt(amount)+' m³ → '+fmt(depth,2)+' mm</div><p class="muted">'+esc(allocationNote)+'</p></div><div class="block"><b>'+esc(heading)+'</b><div class="transition">'+pair+'</div><p class="muted">'+esc(note)+'</p></div>'+flow+'<div class="block">'+row('当前生长阶段',D.stage_names[o.stage_index[i]])+row('产量评价',done?'已完成最终评价':'配置期结束时评价')+row('模拟相对产量',done?fmt(o.settled_relative_yield[i],4):'尚未评价')+'<p class="muted">'+'最终评价是模型预估，不表示当天已实际成熟。'+(isHongmiao?'格内为面积，配置水量见上方。':'格内显示本步已经执行的水量。')+'</p></div>';
  const fp=o.field_params?.[i];
  if(fp){$('detail').innerHTML+='<div class="block"><b>区域储水属性</b>'+row('根区可利用储水容量',fmt(fp[0]*(fp[2]-fp[3]),2)+' mm')+row('含田面水的总储水上限',fmt(capacity(o,i),2)+' mm')+'<p class="muted">可利用容量是田间持水量与萎蔫储水量之差，不是当前水量；区域取值为研究设定。</p></div>'}
  $('detail').innerHTML+=guarantee;
}

const hongmiaoPalette={area:'#EEF4E8',areaBorder:'#7C8D79',water:'#B4D8E9',bank:'#3478A5',arrow:'#286A92',allocation:'#C45A18',storageBackground:'#DAE7E5',selection:'#173548'};
const hongmiaoScale=()=>Math.min(L.width/2240,L.height/1302);
const pointString=path=>path.map(p=>p.join(',')).join(' ');
function drawGate(parent,center,halfWidth,halfHeight,role,index){
  const [x,y]=center;
  parent.appendChild(node('path',{'data-role':role,'data-index':index===undefined?'inlet':index,class:role==='inlet-gate'?'main-inlet-gate':'gate-symbol',d:`M${x-halfWidth} ${y-halfHeight}V${y+halfHeight}M${x+halfWidth} ${y-halfHeight}V${y+halfHeight}M${x-halfWidth} ${y}H${x+halfWidth}`,fill:'none',stroke:'#395D70','stroke-width':2.1*hongmiaoScale()}));
}
function hongmiaoText(parent,position,value,size,attrs={}){
  parent.appendChild(node('text',{x:position[0],y:position[1],'font-family':'Noto Sans CJK SC,Microsoft YaHei,sans-serif','font-size':size,fill:'#344B59','text-anchor':'middle',...attrs},value));
}
function hongmiaoLegendHtml(view,max){
  let common='<span><i class="swatch allocation-key"></i>橙边：本步获得配置</span><span><i class="swatch selection-key"></i>深色虚线：当前选中</span><span><i class="swatch storage-key"></i>蓝条：公开田内储水 / 容量</span>';
  if(view==='water')return '<span><i class="swatch" style="background:#EEF4E8;border-color:#7C8D79"></i>水稻灌溉区 · 大小按面积</span>'+common;
  let overlay=view==='stage'?D.stage_names.map((label,i)=>'<span><i class="swatch" style="background:'+D.stage_colors[i]+'"></i>'+label+'</span>').join(''):'<span>0</span><i class="bar"></i><span>'+(view==='yield'?'100%':fmt(max))+'</span>'+(view==='yield'?'<span><i class="swatch" style="background:#e5e9ed"></i>尚未评价</span>':'');
  return '<b>区域底色覆盖：'+labels[view]+'</b>'+overlay+common;
}
function drawHongmiao(svg,f,c,v,vals,max){
  const p=hongmiaoPalette,scale=hongmiaoScale(),sx=L.width/2240,sy=L.height/1302,o=f.observation;
  svg.setAttribute('data-layout-variant',L.layout_variant);
  svg.setAttribute('aria-label','洪庙93个水稻灌溉区及同桩号两侧示意，蓝条为公开储水，橙边为本步配置');
  svg.appendChild(node('rect',{x:0,y:0,width:L.width,height:L.height,fill:'white'}));
  hongmiaoText(svg,[36*sx,46*sy],'洪庙支渠资源配置',28*scale,{'text-anchor':'start','font-weight':700,fill:'#22333F'});
  hongmiaoText(svg,[2204*sx,44*sy],(c.has_step?'配置 '+(c.executed_date||'日期未知'):'初始 '+(c.state_date||'日期未知'))+' · t = '+(c.decision_step_index??c.day),18*scale,{'data-role':'allocation-date','text-anchor':'end',fill:'#485E6D'});
  svg.appendChild(node('path',{d:`M${36*sx} ${62*sy}H${2204*sx}`,stroke:'#C7D0D6','stroke-width':scale}));
  hongmiaoText(svg,[2204*sx,91*sy],'状态 '+(c.state_date||'日期未知')+' · '+(c.settled_count===L.regions.length?'已完成最终评价':'尚未评价'),16*scale,{'data-role':'state-date','text-anchor':'end',fill:'#485E6D'});
  const channel=node('g',{class:'channel','aria-label':'一条洪庙支渠，折返四行连续展开'});
  function waterline(path,main){
    const attrs={points:pointString(path),fill:'none','stroke-linejoin':'round','stroke-linecap':'round','data-role':'main-canal'};
    channel.appendChild(node('polyline',{...attrs,stroke:L.channel_bank_color||p.bank,'stroke-width':L.channel_bank_width}));
    channel.appendChild(node('polyline',{...attrs,class:main?'main-channel':'inlet-water',stroke:L.channel_color||p.water,'stroke-width':L.channel_width}));
  }
  waterline(L.inlet.path,false);waterline(L.channel_path,true);
  for(const arrow of L.channel_arrows)channel.appendChild(node('polyline',{points:pointString(arrow),fill:'none',stroke:p.arrow,'stroke-width':2*scale,'stroke-linecap':'round','stroke-linejoin':'round'}));
  drawGate(channel,L.inlet.gate_center,L.inlet.gate_half_width,L.inlet.gate_half_height,'inlet-gate');
  hongmiaoText(channel,L.inlet.label_position,L.inlet.name+' · '+L.inlet.gate_name,L.inlet.label_font_size,{'text-anchor':'start','font-weight':600});
  for(const guide of L.row_guides||[])if(guide.row>0)hongmiaoText(channel,[guide.x,guide.y],guide.text,14*scale,{'text-anchor':'start'});
  svg.appendChild(channel);
  const totalNode=L.total_allocation_node,total=c.has_step?(f.allocation_m3||[]).reduce((sum,x)=>sum+Number(x),0):0;
  const totalGroup=node('g',{'data-role':'total-allocation-node','data-total-m3':total});
  totalGroup.appendChild(node('polygon',{points:pointString(totalNode.diamond),fill:'#FFF0DB',stroke:p.allocation,'stroke-width':1.9*scale}));
  hongmiaoText(totalGroup,totalNode.label.position,totalNode.label.text,totalNode.label.font_size);
  hongmiaoText(totalGroup,totalNode.symbol.position,totalNode.symbol.text,totalNode.symbol.font_size,{fill:p.allocation,'font-family':'DejaVu Serif,serif','font-style':'italic'});
  hongmiaoText(totalGroup,totalNode.value.position,fmt(total,0)+' m³',totalNode.value.font_size,{'data-role':'total-allocation-value','data-total-m3':total,'data-allocation-m3':total,fill:p.allocation,'font-weight':600});
  hongmiaoText(totalGroup,totalNode.equation.position,totalNode.equation.text,totalNode.equation.font_size,{'font-family':'DejaVu Serif,serif'});
  const connector=totalNode.connector,end=connector[connector.length-1];
  totalGroup.appendChild(node('polyline',{points:pointString(connector),fill:'none',stroke:p.allocation,'stroke-width':2.4*scale}));
  totalGroup.appendChild(node('polygon',{points:pointString([[end[0],end[1]],[end[0]-10*sx,end[1]-5*sy],[end[0]-10*sx,end[1]+5*sy]]),fill:p.allocation}));
  svg.appendChild(totalGroup);
  for(const r of L.regions){
    const i=r.index,[x,y,w,h]=r.rect,outlet=r.outlet,amount=c.has_step?Number(item(f.allocation_m3,i)||0):0;
    const ratio=Math.max(0,Math.min(1,Number(o.reported_field_storage_mm[i])/capacity(o,i)));
    const g=node('g',{class:'region service-unit',tabindex:0,'data-index':i,'data-region-id':r.region_id,'data-allocation-m3':amount,'data-chainage':r.section_code,'data-side':r.side,'aria-label':r.outlet_name});
    g.appendChild(node('title',{},r.outlet_name+' · 桩号 '+r.section_code+' · '+fmt(r.area_mu,0)+'亩\n本步配置 '+fmt(amount)+' m³ · 公开储水 '+fmt(o.reported_field_storage_mm[i],2)+' mm / '+fmt(capacity(o,i),2)+' mm\n图上两侧布置为示意，不代表实际岸别。'));
    const cx=outlet.connector_start[0],y1=outlet.connector_start[1],last=outlet.service_path[outlet.service_path.length-1],top=Math.min(y1,last[1]),bottom=Math.max(y1,last[1]),half=outlet.connector_width/2;
    g.appendChild(node('rect',{'data-role':'water-connector','data-index':i,x:cx-half,y:top,width:outlet.connector_width,height:bottom-top,fill:p.water,stroke:'none'}));
    g.appendChild(node('path',{d:`M${cx-half} ${top}V${bottom}M${cx+half} ${top}V${bottom}`,fill:'none',stroke:p.bank,'stroke-width':1.2*scale}));
    if(outlet.kind==='culvert_gate')drawGate(g,outlet.center,outlet.half_width,outlet.half_height,'division-gate',i);
    for(const [role,path] of [['supply-direction-shaft',outlet.supply_arrow_shaft],['supply-direction',outlet.supply_arrow]])g.appendChild(node('polyline',{'data-role':role,'data-index':i,points:pointString(path),fill:'none',stroke:p.arrow,'stroke-width':1.35*scale,'stroke-linecap':'round','stroke-linejoin':'round'}));
    hongmiaoText(g,outlet.label_position,r.label,outlet.label_font_size,{'data-role':'outlet-id',class:'gate-label','text-anchor':outlet.label_align||'start','font-weight':600,fill:'#24323B'});
    const fill=v==='water'?p.area:v==='stage'?D.stage_colors[vals[i]]:v==='yield'&&!o.settled_mask[i]?'#E5E9ED':color(vals[i]/max);
    g.appendChild(node('rect',{'data-role':'rice-area',class:'rice-area','data-index':i,'data-region-id':r.region_id,'data-allocation-m3':amount,x,y,width:w,height:h,fill,stroke:amount>0?p.allocation:p.areaBorder,'stroke-width':(amount>0?2.5:1.3)*scale}));
    const [bx,by,bw,bh]=r.storage_bar.rect,barAttrs={x:bx,y:by,height:bh,rx:r.storage_bar.corner_radius,'data-index':i,'data-storage-ratio':ratio,stroke:'none'};
    g.appendChild(node('rect',{...barAttrs,'data-role':'storage-background',width:bw,fill:p.storageBackground}));
    g.appendChild(node('rect',{...barAttrs,'data-role':'storage-water',width:bw*ratio,fill:p.bank}));
    hongmiaoText(g,r.area_label.position,fmt(r.area_mu,0)+'亩',r.area_label.font_size,{'data-role':'area-label',class:'area-label','data-position':r.area_label.outside?'outside':'inside',fill:'#52655B','pointer-events':'none'});
    if(i===selected)g.appendChild(node('rect',{'data-role':'selection-outline',x:x-4*sx,y:y-4*sy,width:w+8*sx,height:h+8*sy,fill:'none',stroke:p.selection,'stroke-width':2*scale,'stroke-dasharray':`${5*scale} ${3*scale}`,'pointer-events':'none'}));
    const pick=()=>{selected=i;draw()};g.onclick=pick;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick()}};svg.appendChild(g);
  }
  svg.appendChild(node('path',{d:`M${36*sx} ${1239*sy}H${2204*sx}`,stroke:'#C7D0D6','stroke-width':scale}));
  hongmiaoText(svg,[36*sx,1263*sy],'同桩号的两个分水口在同一沿渠位置向两侧引出；两侧布置为示意，不代表实际岸别。蓝箭头为供水方向，橙边为本步获得配置。',16*scale,{'text-anchor':'start',fill:'#526571'});
  hongmiaoText(svg,[36*sx,1287*sy],'一条洪庙支渠折返展示；93个单元及面积对应公开资料，渠形、间距和渠宽为示意。总量按入田水量计，蓝条表示公开田内储水比例。',16*scale,{'text-anchor':'start',fill:'#526571'});
}

function draw(){
  let f=F[frameIndex],o=f.observation,c=context(frameIndex),v=$('view').value,vals=values(f),max=['water','yield'].includes(v)?1:Math.max(1,...vals);
  $('slider').value=frameIndex;$('region-select').value=String(selected);$('heading').textContent=title(c);$('status').textContent=(c.has_step?(c.executed_date||'执行日期未知'):(c.state_date||'初始日期未知'))+' · 决策 '+(c.decision_step_index??c.day)+'/'+(c.decision_horizon_steps??o.horizon_days)+' 次 · 状态日期 '+c.state_date;
  $('next-allowance').textContent=c.episode_complete?'回合已结束':'下次可配置 '+fmt(c.next_available_m3)+' m³';$('privileged').hidden=$('access').value!=='truth';
  $('allocation-status').textContent=allocationSummary(c)+'。各区域作为整体配置，未灌溉区域仍逐日计算水分变化。';
  $('stats').innerHTML=accountRows(c).map(([label,value,note])=>'<div class="account"><label>'+esc(label)+'</label><strong>'+fmt(value)+' <span style="font-size:14px">m³</span></strong><small>'+esc(note)+'</small></div>').join('');
  $('rewards').innerHTML=rewardRows(c).map(([label,value])=>'<div><label>'+esc(label)+'</label><strong>'+esc(value)+'</strong></div>').join('');
  let svg=$('map');svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${L.width} ${L.height}`);svg.appendChild(node('desc',{},(L.note||'矩形大小与模拟种植面积成比例；位置不表示真实地理。')+($('access').value==='truth'?'评估者真值':'配置中心收到的报告')));
  if(isHongmiao){drawHongmiao(svg,f,c,v,vals,max)}else{
  for(let r of L.regions){
    let i=r.index,[x,y,w,h]=r.rect,fill=v==='stage'?D.stage_colors[vals[i]]:v==='yield'&&!o.settled_mask[i]?'#e5e9ed':color(vals[i]/max);
    let label=r.label??String(i+1),g=node('g',{class:'region',tabindex:0,'data-region-id':r.region_id,'aria-label':isHongmiao?r.outlet_name:r.region_id});
    g.appendChild(node('title',{},(isHongmiao?r.outlet_name+' · 断面 '+r.section_code+' · '+fmt(r.area_mu,0)+' 亩':r.region_id+' · '+fmt(r.area_ha)+' ha')+' · 本步 '+fmt(item(f.allocation_m3,i))+' m³'));
    g.appendChild(node('rect',{x,y,width:w,height:h,fill,stroke:'#24384c','stroke-width':1.7,class:i===selected?'selected':''}));
    let attrs={'text-anchor':'middle','font-family':'sans-serif',fill:'#112e40',stroke:'white','stroke-width':1,'paint-order':'stroke','stroke-linejoin':'round','pointer-events':'none'};
    if(isHongmiao){
      let area=fmt(r.area_mu,0)+'亩',units=String(Math.round(r.area_mu)).length*.6+1;
      g.appendChild(node('text',{...attrs,class:'area-label',x:x+w/2,y:y+h/2,'dominant-baseline':'central','font-size':Math.min(13,h-2,(w-3)/units)},area));
    }else{
      let line2=w>=37&&h>=32;
      if(w>=20&&h>=16)g.appendChild(node('text',{...attrs,x:x+w/2,y:y+h/2+(line2?-2:3),'font-size':Math.min(13,h-2,(w-2)/Math.max(1,label.length*.9))},label));
      if(line2)g.appendChild(node('text',{...attrs,x:x+w/2,y:y+h/2+12,'font-size':Math.min(11,(w-4)/Math.max(1,shortWater(item(f.allocation_m3,i)).length*.62))},shortWater(item(f.allocation_m3,i))));
    }
    let pick=()=>{selected=i;draw()};g.onclick=pick;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick()}};svg.appendChild(g)
  }
  }
  if(isHongmiao){$('layout-note').textContent='一条洪庙支渠折返四行；11对同桩号口门向图上两侧引出。区域大小按面积，蓝条表示公开储水，橙边表示本步获得配置。两侧、渠形、间距和渠宽均为示意。';svg.setAttribute('aria-label','洪庙93个水稻灌溉区面积与涵闸位置示意图')}
  $('legend').innerHTML=isHongmiao?hongmiaoLegendHtml(v,max):v==='stage'?D.stage_names.map((s,i)=>'<span><i class="swatch" style="background:'+D.stage_colors[i]+'"></i>'+s+'</span>').join(''):'<b>'+labels[v]+'</b><span>0</span><i class="bar"></i><span>'+(['water','yield'].includes(v)?'100%':fmt(max))+'</span>'+(v==='yield'?'<span><i class="swatch" style="background:#e5e9ed"></i>尚未评价</span>':'');
  $('timing').textContent=L.regions.length+(isHongmiao?' 个水稻灌溉区 · 面积 '+fmt(L.total_area_mu,0)+' 亩 / '+fmt(L.total_area_ha,1)+' ha。':' 个独立配置区域 · 总面积 '+fmt(L.total_area_ha,0)+' ha。')+(c.has_step?'配置量属于 '+(c.executed_date||'此前执行日')+'；日历和评价状态更新至 '+(c.state_date||'本步结束')+'。':'初始帧尚未执行配置。')+($('access').value==='reported'&&o.report_age_days.some(x=>x>0)?' 水分颜色显示已收到的历史报告。':' 水分颜色显示当前水分。')+(isHongmiao?' 面积以亩标注，小区域标签放在框外。橙边只表示本步获得配置；深色虚线表示下方详情的选中区域。':' 橙框为右侧区域。');detail(f,c);
}
if(isHongmiao){$('map-source').hidden=false;$('map-source').innerHTML='编号/面积：<a href="https://patents.google.com/patent/CN119514977A/zh" target="_blank" rel="noopener">表2</a> · 来水关系：<a href="https://patents.google.com/patent/CN118521431B/zh" target="_blank" rel="noopener">渠系资料</a> · 同桩号口门同一沿渠位置向两侧引出；两侧、渠形、间距和渠宽均为示意。'}
function pause(){if(timer!==null)clearInterval(timer);timer=null;$('play').textContent='播放'}
$('slider').max=F.length-1;$('play').disabled=F.length<2;$('slider').oninput=()=>{pause();frameIndex=+$('slider').value;draw()};$('previous').onclick=()=>{pause();frameIndex=Math.max(0,frameIndex-1);draw()};$('next').onclick=()=>{pause();frameIndex=Math.min(F.length-1,frameIndex+1);draw()};$('play').onclick=()=>{if(timer!==null){pause();return}if(frameIndex===F.length-1)frameIndex=0;$('play').textContent='暂停';draw();timer=setInterval(()=>{frameIndex++;draw();if(frameIndex===F.length-1)pause()},450)};$('view').onchange=draw;$('access').onchange=draw;
for(let i of L.display_order||L.regions.map(r=>r.index)){let r=L.regions[i],opt=document.createElement('option');opt.value=i;opt.textContent=isHongmiao?r.outlet_name+' · '+fmt(r.area_mu,0)+'亩':'区域 '+(i+1)+' · '+r.region_id;$('region-select').appendChild(opt)}
$('region-select').onchange=()=>{selected=+$('region-select').value;draw()};$('zoom').onchange=()=>{$('map').style.width=$('zoom').value+'%'};
const hasTruth=F.every(f=>f.truth&&Array.isArray(f.truth.field_storage_mm));if(!hasTruth){$('access').querySelector('[value="truth"]').remove();$('access').disabled=true}

function exportHongmiaoSvg(){
  const f=F[frameIndex],c=context(frameIndex),v=$('view').value,g=c.regional_guarantee||D.metadata.run_configuration?.regional_guarantee,feedback=f.constraint_feedback||c.constraint_feedback;
  const height=L.height+(g?440:405),pixelWidth=L.reference_canvas?.[0]||2240,out=node('svg',{xmlns:NS,width:pixelWidth,height:Math.ceil(height*pixelWidth/L.width),viewBox:`0 0 ${L.width} ${height}`}),attrs={'font-family':'Noto Sans CJK SC,Microsoft YaHei,sans-serif',fill:'#173548'};
  const text=(x,y,value,size=13,extra={})=>out.appendChild(node('text',{...attrs,x,y,'font-size':size,...extra},value));
  out.appendChild(node('rect',{width:'100%',height:'100%',fill:'white'}));
  out.appendChild(node('desc',{},'洪庙93个水稻灌溉区，同桩号11对口门向图上两侧引出；两侧、渠形、间距和渠宽均为示意。蓝色储水条来自当前帧公开储水/容量；橙边只表示本步动作大于0，深色虚线为当前选中区域。'));
  text(14,30,title(c)+' · '+(c.executed_date||c.state_date||'日期未知'),23,{'font-weight':700});
  text(14,55,'状态日期 '+(c.state_date||'未知')+' · '+(v==='water'?'淡绿水稻灌溉区，蓝条为公开储水 / 容量':'区域底色覆盖：'+labels[v])+' · 面积单位：亩',13);
  accountRows(c).forEach(([label,value,note],i)=>{const x=14+i*L.width/4;text(x,86,label);text(x,112,fmt(value)+' m³',21);text(x,133,note,11)});
  const map=node('g',{transform:'translate(0,153)'});for(const child of $('map').children)map.appendChild(child.cloneNode(true));out.appendChild(map);
  appendHongmiaoExportLegend(out,L.height+182);
  rewardRows(c).forEach(([label,value],i)=>{const x=14+i*L.width/4;text(x,L.height+257,label);text(x,L.height+282,value,21)});
  let y=L.height+310;
  if(g){const i=selected;text(14,y,L.regions[i].outlet_name+' · 第六章区域保障：参考水平 '+fmt(100*g.guarantee_target[i],1)+'%；允许平均不足 '+fmt(100*g.shortfall_tolerance[i],2)+' 个百分点；本回合不足 '+(feedback?.constraints_evaluated?fmt(100*feedback.constraint_costs[i],3)+' 个百分点':'期末评价'),12);y+=23;text(14,y,'保障阈值为显式研究配置；逐回合先算正不足再平均，单回合不足不能判定长期保障是否达标。',11);y+=23;}
  text(14,y,'两侧布置为示意，不代表实际岸别；渠形、间距和渠宽非真实。总量按入田水量计，通道不参与输水计算。',12);y+=23;
  text(14,y,'蓝条始终为公开田内储水 / 容量；橙边为本步获得配置，深色虚线为选中区域。',12);y+=23;
  text(14,y,'期末统一评价模拟相对产量，不表示当天实际成熟。',12);
  return out;
}
function appendHongmiaoExportLegend(out,y){
  const v=$('view').value,attrs={'font-family':'Noto Sans CJK SC,Microsoft YaHei,sans-serif','font-size':12,fill:'#173548'};
  const text=(x,yy,value)=>out.appendChild(node('text',{...attrs,x,y:yy},value));
  if(v==='water'){
    out.appendChild(node('rect',{x:14,y:y-12,width:20,height:16,fill:'#EEF4E8',stroke:'#7C8D79'}));text(42,y,'水稻灌溉区 · 大小按面积');
  }else if(v==='stage'){
    D.stage_names.forEach((label,i)=>{const x=14+i*L.width/6;out.appendChild(node('rect',{x,y:y-12,width:14,height:14,fill:D.stage_colors[i],stroke:'#24384c'}));text(x+21,y,label)});
  }else{
    text(14,y,'区域底色：'+labels[v]);text(365,y,'0');for(let i=0;i<100;i++)out.appendChild(node('rect',{x:390+i*1.5,y:y-12,width:1.6,height:14,fill:color(i/99)}));text(553,y,v==='yield'?'100%':fmt(Math.max(1,...values(F[frameIndex]))));
    if(v==='yield'){out.appendChild(node('rect',{x:655,y:y-12,width:14,height:14,fill:'#E5E9ED',stroke:'#24384c'}));text(678,y,'尚未评价')}
  }
  y+=27;
  out.appendChild(node('rect',{x:14,y:y-12,width:20,height:16,fill:'#EEF4E8',stroke:'#C45A18','stroke-width':2}));text(42,y,'橙边：本步获得配置');
  out.appendChild(node('rect',{x:300,y:y-12,width:20,height:16,fill:'none',stroke:'#173548','stroke-width':2,'stroke-dasharray':'5 3'}));text(330,y,'深色虚线：当前选中');
  out.appendChild(node('rect',{x:625,y:y-9,width:44,height:7,fill:'#DAE7E5'}));out.appendChild(node('rect',{x:625,y:y-9,width:31,height:7,fill:'#3478A5'}));text(681,y,'蓝条：公开田内储水 / 容量');
}

function exportSvg(){
  if(isHongmiao)return exportHongmiaoSvg();
  let f=F[frameIndex],c=context(frameIndex),height=L.height+350,out=node('svg',{xmlns:NS,width:L.width,height,viewBox:`0 0 ${L.width} ${height}`}),attrs={'font-family':'Noto Sans CJK SC,Microsoft YaHei,sans-serif',fill:'#173548'};
  out.appendChild(node('rect',{width:'100%',height:'100%',fill:'white'}));out.appendChild(node('text',{...attrs,x:14,y:32,'font-size':24},title(c)+' · '+(c.executed_date||c.state_date||'日期未知')));
  out.appendChild(node('text',{...attrs,x:14,y:60,'font-size':14},(isHongmiao?'洪庙93个水稻灌溉区；涵闸编号在渠旁，区域内标面积；':'面积比例示意，非真实地理；')+($('access').value==='truth'?'评估者当前真值':'配置中心收到的报告')+(isHongmiao?' · 格内面积（亩）':' · 格内编号 / 本步配置量')));
  accountRows(c).forEach(([label,value,note],i)=>{let x=14+i*L.width/4;out.appendChild(node('text',{...attrs,x,y:94,'font-size':13},label));out.appendChild(node('text',{...attrs,x,y:122,'font-size':21},fmt(value)+' m³'));out.appendChild(node('text',{...attrs,x,y:145,'font-size':12},note))});
  out.appendChild(node('text',{...attrs,x:14,y:174,'font-size':14},'颜色：'+labels[$('view').value]));let map=node('g',{transform:'translate(0,190)'});for(let child of $('map').children)map.appendChild(child.cloneNode(true));out.appendChild(map);
  appendExportLegend(out,L.height+217);rewardRows(c).forEach(([label,value],i)=>{let x=14+i*L.width/4;out.appendChild(node('text',{...attrs,x,y:L.height+265,'font-size':13},label));out.appendChild(node('text',{...attrs,x,y:L.height+294,'font-size':22},value))});
  out.appendChild(node('text',{...attrs,x:14,y:L.height+329,'font-size':12},$('access').value==='reported'&&f.observation.report_age_days.some(x=>x>0)?'当前水分颜色为历史报告，不能直接解释为本步灌溉效果。':'配置期内尚未评价；期末统一评价模拟相对产量，不表示当天实际成熟。'));return out;
}
$('export').onclick=()=>{let blob=new Blob([new XMLSerializer().serializeToString(exportSvg())],{type:'image/svg+xml;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='irrigation_day_'+F[frameIndex].day+'_'+$('view').value+'.svg';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
function appendExportLegend(out,y){let v=$('view').value,attrs={'font-family':'Noto Sans CJK SC,Microsoft YaHei,sans-serif','font-size':13,fill:'#173548'};if(v==='stage'){D.stage_names.forEach((name,i)=>{let x=14+i*L.width/6;out.appendChild(node('rect',{x,y:y-13,width:16,height:16,fill:D.stage_colors[i],stroke:'#24384c'}));out.appendChild(node('text',{...attrs,x:x+23,y},name))})}else{let max=['water','yield'].includes(v)?1:Math.max(1,...values(F[frameIndex]));out.appendChild(node('text',{...attrs,x:14,y},'0'));for(let i=0;i<160;i++)out.appendChild(node('rect',{x:40+i*2,y:y-14,width:2.1,height:17,fill:color(i/159)}));out.appendChild(node('text',{...attrs,x:380,y},['water','yield'].includes(v)?'100%':fmt(max)));if(v==='yield'){out.appendChild(node('rect',{x:480,y:y-13,width:16,height:16,fill:'#e5e9ed',stroke:'#24384c'}));out.appendChild(node('text',{...attrs,x:505,y},'尚未评价'))}}}
$('metadata').textContent=JSON.stringify(D.metadata,null,2);draw();
</script></html>'''
