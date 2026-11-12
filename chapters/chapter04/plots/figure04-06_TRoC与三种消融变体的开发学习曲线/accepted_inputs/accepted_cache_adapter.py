#!/usr/bin/env python3
"""Existing ablation chart with only fixed-learning-rate bars removed.

Only adjacent CSV/configuration and fixed fonts are read. No policies, environments,
training rewards or smoothed learning-curve crossing points are used.
"""
from pathlib import Path
import argparse,csv,json,os,hashlib

HERE=Path('/hy-tmp/resource_allocation_results/chapter04/figures/figure46_remove_fixed_lr_20261109_204148/assets/figure4-6')
ROOT=Path('/hy-tmp/resource_allocation_results/thesis_figure_spacing_unification_20261110_000617')
# Keep every new cache/output inside this figure owner's independent HY copy.
for key,leaf in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
    p=HERE/'runtime'/leaf;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from presentation_style import export_geometry

DOMAINS=('radio','v5')
NAMES={'radio':'通信功率分配','v5':'应急柴油分配'}
METHODS=('f1','no_history','no_offpolicy_correction')
DISPLAY_METHODS=METHODS
COLORS={'f1':'#D55E00','no_history':'#0072B2','no_offpolicy_correction':'#009E73','fixed_actor_lr':'#7B5EA7'}
TICKS=['TRoC','仅用\n新轨迹','旧轨迹\n不校正']

def setup():
    roots=[]
    if os.environ.get('THESIS_FONT_DIR'):roots.append(Path(os.environ['THESIS_FONT_DIR']))
    roots += [p/'assets/fonts/windows'for p in HERE.parents]
    roots.append(Path('/root/gamewar/latex_thesis_resource_allocation/assets/fonts/windows'))
    files=None
    for root in roots:
        if root.is_dir():
            candidate={p.name.lower():p for p in root.iterdir()if p.is_file()}
            if all(x in candidate for x in ('times.ttf','simsun.ttc')):
                files=candidate;break
    if files is None:raise FileNotFoundError('Verified SimSun/Times New Roman fonts unavailable')
    for name in ('times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf','simsun.ttc'):
        if name in files:font_manager.fontManager.addfont(str(files[name]))
    plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':10.5,
        'axes.labelsize':10.5,'axes.titlesize':10.5,'legend.fontsize':10.5,
        'xtick.labelsize':10.5,'ytick.labelsize':10.5,'font.weight':'normal',
        'axes.labelweight':'normal','axes.titleweight':'normal','axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'svg.hashsalt':'c4-two-panel-ablation-confirmation-interactions-20261109',
        'savefig.dpi':240})


def read_csv(p):
    with p.open(newline='')as f:return list(csv.DictReader(f))


def plot(config,rows):
    assert config['kind']=='ablation_confirmation_interactions'
    assert tuple(config['methods'])==METHODS and len(rows)==config['expected_confirmation_rows']==24
    assert tuple(config['display_methods'])==DISPLAY_METHODS
    fig,axes=plt.subplots(1,2,figsize=(15.6/2.54,7.7/2.54));bars=[]
    parent_points={(x['domain'],x['method']):x for x in read_csv(HERE/'parent_confirmation_bar_points.csv')}
    for col,domain in enumerate(DOMAINS):
        ax=axes[col]
        for i,method in enumerate(DISPLAY_METHODS):
            rr=sorted([r for r in rows if r['domain']==domain and r['method']==method],key=lambda r:int(r['seed']))
            meta=config['method_metadata'][method]
            assert len(rr)==len(meta['seeds']) and [int(r['seed'])for r in rr]==meta['seeds']
            assert all(int(r['training_steps'])==meta['training_steps'] for r in rr)
            assert all(r['nq_q95_censored']=='False'and int(r['nq_q95'])==int(r['nq_q95_restricted'])for r in rr)
            values=np.array([int(r['nq_q95_restricted'])for r in rr],dtype=float)
            mean_steps,sd_steps=float(values.mean()),float(values.std(ddof=1))
            mean,sd=mean_steps/1000,sd_steps/1000
            if not (domain=='v5' and method=='f1'):
                old=parent_points[(domain,method)]
                assert old['values_steps']==';'.join(r['nq_q95_restricted']for r in rr)
                assert np.isclose(mean_steps,float(old['mean_steps']),rtol=1e-14,atol=1e-10)
                assert np.isclose(sd_steps,float(old['sample_sd_steps']),rtol=1e-14,atol=1e-10)
                mean_steps,sd_steps=float(old['mean_steps']),float(old['sample_sd_steps'])
                mean,sd=float(old['mean_ksteps']),float(old['sample_sd_ksteps'])
            ax.bar(i,mean,yerr=sd,width=.64,color=COLORS[method],alpha=.78,
                error_kw=dict(elinewidth=1.,capsize=3,capthick=1.))
            ax.annotate(f'{mean:.2f}',xy=(i,mean+sd),xytext=(0,4),
                textcoords='offset points',ha='center',va='bottom',
                fontsize=10.5,fontfamily='Times New Roman',color='black',
                annotation_clip=False)
            # Raw step-unit statistics precede the display-unit conversion.
            # n/budget are explicit per method; this is not a matched ablation estimate.
            bars.append(dict(domain=domain,method=method,mean_steps=mean_steps,
                sample_sd_steps=sd_steps,mean_ksteps=mean,sample_sd_ksteps=sd,
                confirmed_runs=len(rr),seed_count=len(rr),right_censored_runs=0,
                target_score_x100=config['thresholds'][domain],
                values_steps=';'.join(r['nq_q95_restricted']for r in rr)))
            if not (domain=='v5' and method=='f1'):bars[-1]=dict(parent_points[(domain,method)])
        ax.set_title(f'({chr(97+col)}) {NAMES[domain]}',loc='center',pad=10,
            fontfamily=['Times New Roman','SimSun'],fontsize=10.5,fontweight='normal')
        ax.set_ylabel(('达到45.27分所需训练步数'if domain=='radio'else'达到75.60%所需训练步数')+'\n（×10³）',linespacing=1.25,labelpad=8)
        ax.set_xticks(range(3));ax.set_xticklabels(TICKS,linespacing=1.12)
        ax.tick_params(axis='x',pad=7)
        ax.set_ylim(0,80 if domain=='radio'else 22)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    fig.subplots_adjust(left=.125,right=.99,bottom=16.52/77,top=1-10.08/77,wspace=.46)
    bars.sort(key=lambda row:(DOMAINS.index(row['domain']),METHODS.index(row['method'])))
    return fig,bars


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=HERE)
    args=parser.parse_args();dest=args.output_dir.resolve()
    if not dest.is_relative_to(ROOT):raise ValueError('Outputs must stay inside this independent HY presentation root')
    dest.mkdir(parents=True,exist_ok=True)
    setup();config=json.loads((HERE/'figure_config.json').read_text())
    rows=read_csv(HERE/'confirmation_steps_per_seed.csv')
    fig,bars=plot(config,rows);fig.set_dpi(220)
    export_geometry(fig,dest/'render_geometry.json')
    for suffix in ['pdf','png','svg']:
        metadata={'Creator':'C4 descriptive full-method candidate and retained original ablation bars','CreationDate':None,'ModDate':None}if suffix=='pdf'else({'Date':None}if suffix=='svg'else None)
        fig.savefig(dest/('figure.'+suffix),dpi=220,metadata=metadata)
    plt.close(fig)
    for name in ['plotted_points.csv','confirmation_bar_points.csv']:
        with(dest/name).open('w',newline='')as f:
            w=csv.DictWriter(f,fieldnames=list(bars[0]));w.writeheader();w.writerows(bars)
    geometry=json.loads((dest/'render_geometry.json').read_text())
    (dest/'presentation_geometry.json').write_text(json.dumps({
        'schema':'chapter04-two-panel-readable-labels-geometry-v1',
        'figure_inches':geometry['figure_inches'],'axes_positions':geometry['axes_positions'],
        'axes_count':2,'all_six_remaining_parent_bar_values_preserved':True,
        'physical_axes_mm_unchanged':True,
        'display_method_order':list(DISPLAY_METHODS),
        'mean_labels_decimals':2,'mean_labels_position':'4pt above upper sample-SD cap',
        'matched_protocol_ablation_claimed':False,
        'all_learning_curve_panels_removed':True,'shared_legend':False,
        'old_pixel_or_geometry_invariance_claimed':False},ensure_ascii=False,indent=2)+'\n')
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    experiment=Path('/hy-tmp/resource_allocation_results/chapter04/experiments/lr_feedback_development_20261109_155109')
    parent=Path('/hy-tmp/resource_allocation_results/chapter04/figures/figure46_adjacent_values_20261109_195925/assets/figure4-6')
    source=dict(schema='chapter04-figure46-only-fixed-bars-removed-v1',
        status='RENDERED_ONLY_FIXED_LR_BARS_REMOVED',
        parent_figure=str(parent),experiment=str(experiment),
        completion_receipt_sha256=digest(experiment/'completion_receipt.json'),
        source_full_method={'radio':'relaxed_feedback','v5':'original_feedback'},fixed_source_method='fixed_3e4',
        retained_full_seeds=[7100,7101],retained_full_training_steps=49152,
        retained_original_reference_methods=['no_history','no_offpolicy_correction'],
        retained_original_reference_seeds=list(range(5)),retained_original_reference_training_steps=100608,
        parent_raw_DEV_sha256=digest(HERE/'parent_data.csv'),diesel_full_original_DEV_sha256=digest(HERE/'diesel_full_original_DEV.csv'),
        mixed_raw_DEV_rows=2900,confirmation_rows=24,all_four_matched_ablation_claimed=False,
        fixed_learning_rate_bars_removed_only=True,
        presentation_change='Remove only fixed-learning-rate bars; retain the original six bar statistics, internal source identities, method definition, colors, axes, font and mean annotations.',
        data_sha256=digest(HERE/'confirmation_steps_per_seed.csv'),
        figure_config_sha256=digest(HERE/'figure_config.json'),plot_sha256=digest(HERE/'plot.py'),
        presentation_style_sha256=digest(HERE/'presentation_style.py'),
        outputs_sha256={name:digest(dest/name)for name in ['figure.pdf','figure.svg','figure.png','plotted_points.csv','confirmation_bar_points.csv','render_geometry.json']},
        width_cm=15.6,height_cm=7.7,all_text_font_bp=10.5,model_loads=0,environment_steps=0,
        original_scientific_data_modified=False,new_scientific_execution=False,raw_confirmation_smoothing=False,
        model_or_protocol_retrained=False,full_fixed_independent_TEST_cases=0)
    (dest/'source.json').write_text(json.dumps(source,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
