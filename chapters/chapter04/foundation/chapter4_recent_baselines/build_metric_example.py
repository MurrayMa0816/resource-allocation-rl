"""Reuse existing V2 evaluations; consumes zero new environment interactions."""

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .common import RESULT_ROOT, json_write
from .metrics import sustained_attainment, attainment_curve, normalized_auc


def main():
    source=Path(str(_resolve_path('/hy-tmp/resource_allocation_results/chapter4/sample_efficiency_v2_20261101/analysis/curves.csv')))
    rows=list(csv.DictReader(source.open()))
    out=RESULT_ROOT/'metrics'
    out.mkdir(parents=True,exist_ok=True)
    font=Path('/root/.local/share/fonts/microsoft/simsun.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'pdf.fonttype':42,'font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    all_rows=[]
    points=[]
    for ax,domain,title,target,h in zip(axes,('radio','v5'),('通信任务','应急任务'),(30.,65.),(32,24)):
        for method,label,color in (('b0','第三章 A','#555555'),('f1','第四章 F1','#1677b8')):
            outcomes=[]
            for seed in range(60,65):
                curve=sorted([r for r in rows if r['domain']==domain and r['method']==method and int(r['seed'])==seed],
                             key=lambda r:int(r['training_steps']))
                x=[int(r['training_steps']) for r in curve]
                y=[float(r['score_x100']) for r in curve]
                result=sustained_attainment(x,y,target)
                outcomes.append(result)
                all_rows.append({'domain':domain,'method':label,'seed':seed,
                                 **result,**normalized_auc(x,y)})
            grid=np.arange(0,481,48)*h
            data=attainment_curve(outcomes,grid)
            ax.step(grid/1000,np.asarray(data['attained_fraction'])*100,where='post',label=label,color=color,lw=2.3)
            for x,p in zip(grid,data['attained_fraction']):
                points.append({'domain':domain,'method':label,'new_real_steps':int(x),
                               'confirmed_seeds':int(round(p*5)),'total_seeds':5,'fraction':p})
        ax.set(title=f'{title}：目标值 {target:g}',xlabel='新增真实训练交互步数（千步）',ylabel='训练种子达标比例（%）',
               xlim=(0,480*h/1000),ylim=(-3,105),yticks=[0,20,40,60,80,100])
        ax.grid(alpha=.2)
        ax.legend(loc='upper left')
    fig.suptitle('同一目标下，有多少个训练种子已经达标？',fontsize=15)
    fig.text(.5,.015,'每个方法 5 个训练种子；连续两次开发评估达标才确认。未达标种子保留。\n'
             '此图复用已有 V2 数据，不包含本轮五个新 baseline 的正式训练结果。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.1,1,.93))
    for extension in ('png','pdf','svg'):
        fig.savefig(out/f'第三章A与F1_训练种子达标比例.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    for name,data in (('per_seed_metrics.csv',all_rows),('attainment_curve.csv',points)):
        with (out/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]))
            writer.writeheader();writer.writerows(data)
    json_write(out/'provenance.json',dict(source=str(source),cohort='V2 seeds60..64,480episodes',
        new_environment_steps=0,new_training=0,thresholds={'radio':30.,'v5':65.},
        task_success_rate=False,metric='proportion of independent training seeds with confirmed attainment',
        caveat='Five training seeds only. This is descriptive; it neither proves uniform improvement nor a terminal-reward-specific causal benefit.'))


if __name__=='__main__':main()
