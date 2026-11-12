#!/usr/bin/env python3
"""Render this figure from the adjacent recorded observations only.
Usage: python plot.py [--output-dir regenerated]
Requires Python 3, numpy and matplotlib; Chinese=SimSun, Latin=Times New Roman.
No environment, policy, model, training, evaluation or publication is invoked.
The thesis PDF copied beside this script remains the accepted original.
"""
from pathlib import Path
from presentation_layout import balance_legend_spacing
import argparse, csv, json, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# BEGIN THESIS FONT POLICY (presentation only)
def _register_thesis_plot_fonts():
    import os as _font_policy_os
    _requested = _font_policy_os.environ.get('THESIS_FONT_DIR')
    _roots = ([Path(_requested)] if _requested else []) + [
        parent / 'assets/fonts/windows' for parent in Path(__file__).resolve().parents]
    _roots += [Path('/root/gamewar/latex_thesis_resource_allocation/assets/fonts/windows'),
               Path('/root/.local/share/fonts/microsoft')]
    for _root in _roots:
        if not _root.is_dir():
            continue
        _files = {p.name.lower(): p for p in _root.iterdir() if p.is_file()}
        if all(name in _files for name in ('times.ttf', 'simsun.ttc', 'simhei.ttf')):
            for _name in ('times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf', 'simsun.ttc', 'simhei.ttf'):
                if _name in _files:
                    font_manager.fontManager.addfont(str(_files[_name]))
            return _root, _files
    raise FileNotFoundError('Set THESIS_FONT_DIR to a folder containing Times New Roman, SimSun and SimHei.')
_THESIS_FONT_POLICY_ROOT, _THESIS_FONT_POLICY_FILES = _register_thesis_plot_fonts()
# END THESIS FONT POLICY
from matplotlib.ticker import StrMethodFormatter, MaxNLocator
HERE=Path(__file__).resolve().parent

def rows(name='data.csv'):
    with (HERE/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def destination():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--output',type=Path,help='Full output filename (compatibility option)')
    group.add_argument('--output-dir',type=Path,default=HERE/'regenerated',help='Output folder; default: adjacent regenerated/')
    args=parser.parse_args()
    p=args.output if args.output is not None else args.output_dir/'figure.pdf'
    p.parent.mkdir(parents=True,exist_ok=True)
    return p

COLORS = {'A': '#0072B2', 'paspo': '#D55E00', 'rpo': '#009E73',
          'td7': '#7B5EA7', 'simba_v2': '#56A6B8', 'dime': '#A6761D'}
MARKERS = dict(A='o', paspo='s', rpo='^', td7='D', simba_v2='v', dime='P')
DOMAINS = ('radio', 'v5')
NAMES = dict(radio='通信功率分配', v5='应急柴油分配')

METHODS=('A','paspo','rpo','td7','simba_v2','dime')
def require(condition,message):
    if not condition: raise ValueError(message)

def setup():
    fonts = []
    for path in (_THESIS_FONT_POLICY_FILES['simsun.ttc'],
                 _THESIS_FONT_POLICY_FILES['times.ttf']):
        require(path.is_file(), 'Required thesis font is missing: ' + str(path))
        font_manager.fontManager.addfont(path)
        fonts.append(path)
    plt.rcParams.update({'font.family': ['Times New Roman', 'SimSun'],
        'font.size': 10.5, 'axes.labelsize': 10.5, 'axes.titlesize': 10.5,
        'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5, 'legend.fontsize': 10.5,
        'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .7, 'axes.grid': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'path.simplify': False, 'mathtext.fontset': 'cm'})
    return fonts

def main():
    setup()
    plt.rcParams.update({'font.size':10.5,'axes.labelsize':10.5,'axes.titlesize':10.5,'xtick.labelsize':10.5,'ytick.labelsize':10.5,'legend.fontsize':10.5, 'mathtext.fontset': 'cm'})
    data=rows();curves={d:{} for d in DOMAINS};LABELS={}
    for d in DOMAINS:
        for m in METHODS:
            group=[r for r in data if r['domain']==d and r['method']==m]
            group.sort(key=lambda r:int(r['environment_steps']))
            curves[d][m]=dict(x=np.array([int(r['environment_steps'])/1000 for r in group]),mean=np.array([float(r['mean_score_x100']) for r in group]),sd=np.array([float(r['between_seed_sample_sd']) for r in group]))
            LABELS[m]=group[0]['method_label'] + ('（本文）' if m=='A' else '')
    fig,axes=plt.subplots(1,2,figsize=(15.6/2.54,3.4))
    fig.subplots_adjust(left=.105,right=.985,bottom=.29,top=.87,wspace=.38)
    for letter,domain,ax in zip('ab',DOMAINS,axes):
     for method in METHODS:
      group=curves[domain][method];x=group['x']
      mean=group['mean'];sd=group['sd']
      color,marker=COLORS[method],MARKERS[method]
      ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.09,linewidth=0)
      indices=list(range(METHODS.index(method)%3,len(x),max(1,len(x)//13)))
      ax.plot(x,mean,color=color,linestyle='-',linewidth=1.3 if method=='A' else 1.,marker=marker,markevery=indices,
       markersize=3.2,markerfacecolor=color,markeredgecolor=color,markeredgewidth=.6,label=LABELS[method])
     ax.set_title(f'({letter}) {NAMES[domain]}',loc='center',pad=9, fontfamily=['Times New Roman', 'SimSun'], fontsize=10.5)
     ax.set_xlabel('训练步数（×10³）');ax.set_ylabel('平均图像质量得分' if domain=='radio' else '平均需求完成比例（%）')
     ax.set_xlim(0,120);ax.set_xticks([0,30,60,90,120]);ax.get_xticklabels()[-1].set_horizontalalignment('right');ax.tick_params(direction='out',length=3,width=.65)
    handles,labels=axes[0].get_legend_handles_labels();order=[0,3,1,4,2,5]
    fig.legend([handles[i] for i in order],[labels[i] for i in order],loc='lower center',bbox_to_anchor=(.53,.015),ncol=3,frameon=False,
     handlelength=2.,columnspacing=1.3,handletextpad=.5,labelspacing=.7)

    balance_legend_spacing(fig)
    fig.savefig(destination(),dpi=260)
    plt.close(fig)

if __name__=='__main__': main()
