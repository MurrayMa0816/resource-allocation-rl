#!/usr/bin/env python3
"""C6 native statistical figures from sealed arrays and existing train logs only.

No model/environment imports, no new execution, no mutation of former figures.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
THESIS=next(p for p in HERE.parents if (p/'assets/fonts').is_dir())
ROOT = HERE / '.render-stage'
DATA = HERE
BLUE, ORANGE, GREY = '#236d9e', '#cf7627', '#999999'
SEEDS = (6400, 6401, 6402)
STAGES = (('返青', 0, 9), ('分蘖', 9, 46), ('孕穗', 46, 62), ('抽穗', 62, 72), ('乳熟', 72, 81))
MAP_DAYS = (0, 46, 62, 63, 72)
INPUTS = {}
FONT = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path)
    INPUTS[str(path)] = sha(path)
    return path


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_read(path):
    with bind(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    rows = list(rows)
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def setup():
    os.environ['MPLCONFIGDIR'] = str(ROOT / 'runtime_cache/statistical_style_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = THESIS/'assets/fonts/figure/ThesisFigureText-Regular.otf'
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({
        'font.family': family, 'font.size': 10.5, 'axes.labelsize': 10.5,
        'axes.titlesize': 10.5, 'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
        'legend.fontsize': 10.5, 'axes.unicode_minus': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'lines.linewidth': 1.15,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'c6-style-unification-statistical-v1', 'savefig.facecolor': 'white',
    })
    return plt, dict(path=str(font), sha256=sha(font), family=family,
                     ordinary_font_size_bp=10.5,
                     origin='Approved composite SimSun Chinese / Times New Roman Latin figure font')


def save(fig, group, plt, details):
    from matplotlib.text import Text
    folder = ROOT / 'figures' / group; folder.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    text_checks, outside, title_checks = [], [], []
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text():
            continue
        text_checks.append(dict(text=item.get_text(), size_bp=float(item.get_fontsize())))
        assert abs(item.get_fontsize() - 10.5) < 1e-8, text_checks[-1]
        box = item.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.x1 + .5 or box.y1 > fig.bbox.y1 + .5:
            outside.append(item.get_text())
    assert not outside, ('Text outside canvas', outside)
    for ax in fig.axes:
        if ax.get_title():
            box = ax.title.get_window_extent(renderer)
            error = abs((box.x0 + box.x1) / 2 - (ax.bbox.x0 + ax.bbox.x1) / 2)
            assert error < .5, (ax.get_title(), error)
            title_checks.append(dict(title=ax.get_title(), centre_error_pixels=float(error)))
    for legend in fig.legends:
        box = legend.get_window_extent(renderer)
        for ax in fig.axes:
            assert not box.overlaps(ax.bbox), 'Legend overlaps axes'
            for item in (ax.xaxis.label, ax.title):
                if item.get_text():
                    assert not box.overlaps(item.get_window_extent(renderer)), ('Legend intersects axis text', item.get_text())
    # Calibrate a common 2mm external bottom margin from actual PDF text.
    # Only empty vertical space is trimmed; full native width, axes and fonts
    # retain their physical size in all three exports.
    import io,fitz
    from matplotlib.transforms import Bbox
    metadata=dict(CreationDate=None,ModDate=None,Creator='C6 existing-array native statistical style renderer')
    buffer=io.BytesIO();fig.savefig(buffer,format='pdf',metadata=metadata)
    with fitz.open(stream=buffer.getvalue(),filetype='pdf') as measured:
        page=measured[0]
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans'] if s['text'].strip()]
        margin_mm=(page.rect.height-max(s['bbox'][3] for s in spans))*25.4/72
    trim_bottom_mm=max(0.,margin_mm-2.)
    export_box=Bbox.from_extents(0.,trim_bottom_mm/25.4,fig.get_figwidth(),fig.get_figheight())
    for extension in ('pdf', 'svg', 'png'):
        kwargs = dict(dpi=200) if extension == 'png' else {}
        if extension == 'pdf':kwargs['metadata']=metadata
        fig.savefig(folder / f'figure.{extension}',bbox_inches=export_box,pad_inches=0,**kwargs)
    with fitz.open(folder/'figure.pdf') as measured:
        actual_height_mm=measured[0].rect.height*25.4/72
    record = dict(status='NATIVE_CANDIDATE_REQUIRES_VISUAL_REVIEW', group=group,
                  width_mm=float(fig.get_size_inches()[0] * 25.4),
                  height_mm=actual_height_mm, source_canvas_height_mm=float(fig.get_size_inches()[1]*25.4),
                  external_bottom_margin_mm=2., trimmed_empty_bottom_mm=trim_bottom_mm,
                  font=FONT, all_visible_text_bp=10.5, visible_text_checks=text_checks,
                  text_bounds_pass=True, title_anchor_checks=title_checks,
                  source=str(Path(__file__).resolve()), source_sha256=sha(__file__),
                  read_only_inputs_sha256=dict(INPUTS),
                  assets={ext:dict(path=str(folder/f'figure.{ext}'), sha256=sha(folder/f'figure.{ext}')) for ext in ('pdf','svg','png')},
                  scientific_execution=dict(environment_steps=0, model_loads=0, inference=0, training=0, new_TEST=0),
                  TeX_edits=0, paper_builds=0, old_outputs_modified=False, **details)
    dump(folder / 'manifest.json', record); plt.close(fig)
    return record


def representative():
    qrows=csv_read(DATA/'daily_allocation81.csv')
    brows=csv_read(DATA/'beginning_balance82.csv')
    return dict(q=np.asarray([float(r['allocation_m3']) for r in qrows]),
                balance=np.asarray([float(r['beginning_balance_m3']) for r in brows]),
                dates=np.asarray([r['date'] for r in qrows]))


def overview(plt):
    case = representative(); q, balance = case['q'] / 10000, case['balance'] / 10000
    assert q.shape == (81,) and balance.shape == (82,)
    np.testing.assert_allclose(case['balance'][:-1] - case['q'], case['balance'][1:], atol=1e-7, rtol=0)
    fig = plt.figure(figsize=(144 / 25.4, 80 / 25.4))
    top = fig.add_axes([.16, .5375, .825, .3125])
    bottom = fig.add_axes([.16, .1625, .825, .25], sharex=top)
    top.plot(np.arange(81), q, color=BLUE)
    bottom.plot(np.arange(82), balance, color=BLUE)
    top.set(ylabel='当日总量（万m³）', ylim=(0, q.max() * 1.3))
    top.set_yticks([0, 5, 10])
    bottom.set(ylabel='日初余额（万m³）', xlabel='灌溉日期', ylim=(-8, 432))
    bottom.set_yticks([0, 200, 400])
    ticks = (0, 17, 32, 46, 62, 72, 81)
    for ax, values in ((top, q), (bottom, balance)):
        ax.scatter(MAP_DAYS, values[list(MAP_DAYS)], color=ORANGE, s=17,
                   zorder=5, edgecolors='white', linewidths=.35)
        ax.axvspan(61.6, 63.4, color=ORANGE, alpha=.12, zorder=0)
        for _, start, _ in STAGES[1:]:
            ax.axvline(start, color='#b1b1b1', ls=':', lw=.6, zorder=0)
        ax.set_xlim(-.8, 81.8); ax.set_xticks(ticks)
        ax.set_xticklabels([(date(2022,5,30)+timedelta(days=int(d))).strftime('%-m/%-d') for d in ticks])
        ax.grid(axis='y', alpha=.18, linewidth=.6)
    top.tick_params(labelbottom=False)
    top.set_title('(a) 每日配置总量', pad=20)
    bottom.set_title('(b) 日初供水余额', pad=8)
    phase_texts = []
    for name, start, stop in STAGES:
        phase_texts.append(top.text((start+stop)/2, 1.035, name, ha='center', va='bottom',
                                    transform=top.get_xaxis_transform()))
    for day, label, target in ((0,'5/30',8),(46,'7/15',40),(62,'7/31—8/1',61),(72,'8/10',75)):
        height = .80 if day in (0,46) else .94 if day == 62 else .56
        top.annotate(label, (day,q[day]), xytext=(target,height), textcoords=top.get_xaxis_transform(),
                     ha='center', va='center',
                     arrowprops=dict(arrowstyle='-',color='#646464',lw=.65,shrinkA=3,shrinkB=4))
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    for text in phase_texts:
        assert not text.get_window_extent(renderer).overlaps(top.title.get_window_extent(renderer))
    folder=ROOT/'figures/overview';folder.mkdir(parents=True,exist_ok=True)
    csv_write(folder/'daily_allocation81.csv', [dict(day=i,date=str(case['dates'][i]),allocation_m3=float(case['q'][i])) for i in range(81)])
    csv_write(folder/'beginning_balance82.csv', [dict(boundary=i,date=str(date(2022,5,30)+timedelta(days=i)),beginning_balance_m3=float(case['balance'][i])) for i in range(82)])
    return save(fig,'overview',plt,dict(figure_number='6.3',curve_node_counts=[81,82],
        representative=dict(training_seed=6400,weather_id=125),stages=[list(x) for x in STAGES],
        marker_days=list(MAP_DAYS),date_ticks=list(ticks),same_original_curve_marker_date_and_stage_values=True,
        final_balance_boundary='8/19 is the state boundary after81dailyactions, not an additional irrigation action'))


def feedback(plt):
    from matplotlib.path import Path as MplPath
    saved=csv_read(DATA/'seven_day_actual_weather_and_q.csv')
    days=np.asarray([int(r['day_zero_based']) for r in saved])
    original=dict(q=np.zeros(81),weather=np.zeros((81,2)),dates=np.empty(81,dtype='U10'))
    modes={key:dict(q=np.zeros(81),weather=np.zeros((81,2))) for key in ('feedback','replay')}
    for row,day in zip(saved,days):
        original['q'][day]=float(row['original_weather_plan_q_m3'])
        original['weather'][day,1]=float(row['original_ET0_mm_per_day'])
        original['dates'][day]=row['date']
        modes['replay']['q'][day]=float(row['same_changed_weather_replay_q_m3'])
        modes['feedback']['q'][day]=float(row['same_changed_weather_feedback_q_m3'])
        for mode in modes:modes[mode]['weather'][day,1]=float(row['changed_ET0_mm_per_day'])
    delta=modes['feedback']['q']-modes['replay']['q']
    old=modes['replay']['q'][days]/10000;new=modes['feedback']['q'][days]/10000
    fig=plt.figure(figsize=(144/25.4,132/25.4));top=fig.add_axes([.16,.755,.825,.18]);main_ax=fig.add_axes([.16,.195,.825,.40],sharex=top)
    first=top.plot(days,original['weather'][days,1],color=GREY,ls='--',marker='o',ms=3.8,mfc='white',label='原天气',zorder=3)[0]
    second=top.plot(days,modes['feedback']['weather'][days,1],color=ORANGE,marker='s',ms=3.8,label='变化后天气',zorder=4)[0]
    top.axvspan(61.5,64.5,color='#d5d8d9',alpha=.35,zorder=0)
    top.text(63,1.05,'7/31—8/2提高15%',ha='center',va='center')
    top.set(ylim=(0,8.8),ylabel='参考蒸散率（mm/d）');top.set_yticks([0,4,8]);top.tick_params(axis='x',labelbottom=False)
    top.set_title('(a) 参考蒸散率变化',pad=6);top.grid(axis='y',alpha=.18,linewidth=.6)
    main_ax.axvspan(61.5,64.5,color='#d5d8d9',alpha=.12,zorder=0)
    line_old=main_ax.plot(days,old,color=GREY,ls='--',marker='o',ms=4,mfc='white',label='继续执行原配水安排',zorder=3)[0]
    line_new=main_ax.plot(days,new,color=BLUE,marker='s',ms=3,label='依据新观测重新配水',zorder=4)[0]
    connections=[]
    for day,a,b in zip(days,old,new):
        if abs(a-b)>1e-12:
            main_ax.vlines(day,min(a,b),max(a,b),color='#666666',lw=.65,zorder=2.5);connections.append([(float(day),float(a)),(float(day),float(b))])
    main_ax.set(ylim=(4,13),ylabel='每日分配灌溉总水量\n（万m³）',xlabel='日期');main_ax.set_yticks([4,6,8,10,12])
    main_ax.set_xticks(days);main_ax.set_xticklabels(['7/30','7/31','8/1\n首次调整','8/2','8/3','8/4','8/5']);main_ax.set_xlim(60.5,67.5)
    main_ax.set_title('(b) 相同变化天气下的每日配水',pad=6);main_ax.grid(axis='y',alpha=.18,linewidth=.6)
    note=main_ax.text(.985,.975,'标注为同日较原安排增减水量（m³）',transform=main_ax.transAxes,ha='right',va='top')
    labels=[];rows=[]
    for i,day in enumerate(days):
        difference=float(delta[day]);csv_label='未调整' if abs(difference)<1e-8 else f"{'多配' if difference>0 else '少配'}{int(round(abs(difference)))}"
        label=csv_label.replace('未调整','水量未变').replace('多配','增加').replace('少配','减少')
        dy,va=(-17,'top') if day in {61,63,66} else (8,'bottom')
        labels.append(main_ax.annotate(label,(day,max(old[i],new[i])),xytext=(0,dy),textcoords='offset points',ha='center',va=va))
        rows.append(dict(day_zero_based=int(day),date=str(original['dates'][day]),original_ET0_mm_per_day=float(original['weather'][day,1]),
                         changed_ET0_mm_per_day=float(modes['feedback']['weather'][day,1]),ET0_multiplier=1.15 if 62<=day<=64 else 1.,
                         original_weather_plan_q_m3=float(original['q'][day]),same_changed_weather_replay_q_m3=float(modes['replay']['q'][day]),
                         same_changed_weather_feedback_q_m3=float(modes['feedback']['q'][day]),same_day_feedback_minus_replay_m3=difference,display_annotation_m3=csv_label))
    fig.legend([first,second],['原天气','变化后天气'],ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.5725,.714),handlelength=1.8,columnspacing=1.6,handletextpad=.5)
    fig.legend([line_old,line_new],['继续执行原配水安排','依据新观测重新配水'],ncol=2,frameon=False,loc='lower center',bbox_to_anchor=(.5725,.005),handlelength=1.8,columnspacing=1.2,handletextpad=.5)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();boxes=[x.get_window_extent(renderer) for x in labels];note_box=note.get_window_extent(renderer)
    paths=[x.get_path().transformed(main_ax.transData) for x in (line_old,line_new)]+[MplPath(main_ax.transData.transform(x)) for x in connections]
    for i,box in enumerate(boxes):
        assert not box.overlaps(note_box)
        for path in paths:assert not path.intersects_bbox(box,filled=False),('Annotation crosses curve',labels[i].get_text())
        for other in boxes[i+1:]:assert not box.overlaps(other)
    folder=ROOT/'figures/feedback';folder.mkdir(parents=True,exist_ok=True);csv_write(folder/'seven_day_actual_weather_and_q.csv',rows)
    return save(fig,'feedback',plt,dict(figure_number='6.8',actual_nodes_per_curve=7,no_smoothing=True,no_dual_y_axis=True,
        same_day_connection_count=len(connections),comparison='samechangedweather, same-date updatedstatefeedback minus originalactionreplay, not todayminusyesterday',
        first_action_response_zero_based=63,q_axis_limits_wan_m3=[4,13],input_output_layer_heights_mm=dict(weather=23.76,allocation=52.8),
        same_numeric_weather_q_and_annotations_as_previous_156mm_version=True))


def main():
    global FONT
    parser=argparse.ArgumentParser();parser.add_argument('--figures',nargs='+',choices=['6.3','6.5','6.7','6.8','all'],default=['all']);args=parser.parse_args()
    plt,FONT=setup();functions={'6.3':overview,'6.5':learning,'6.7':budgets,'6.8':feedback};selected=list(functions) if 'all' in args.figures else args.figures
    records=[]
    for key in selected:
        INPUTS.clear();records.append(functions[key](plt))
    index=ROOT/'figures/statistical_style_manifest.json';existing=json.loads(index.read_text()) if index.exists() else dict(figures={})
    for record in records:existing['figures'][record['figure_number']]=record
    existing.update(status='RENDERED_FROM_EXISTING_ARRAYS',source=str(Path(__file__).resolve()),source_sha256=sha(__file__),scientific_execution=dict(environment_steps=0,model_loads=0,inference=0,training=0,new_TEST=0))
    dump(index,existing)
    print(json.dumps([dict(figure=r['figure_number'],native_size_mm=[r['width_mm'],r['height_mm']],pdf=r['assets']['pdf']) for r in records],ensure_ascii=False))


if __name__=='__main__':main()
