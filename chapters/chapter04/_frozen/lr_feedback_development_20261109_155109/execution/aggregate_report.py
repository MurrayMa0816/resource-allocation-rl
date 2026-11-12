"""Read-only aggregation of the fixed four-arm learning-rate DEV experiment.

No environment, agent, checkpoint or scientific implementation is imported.
The script requires every registered scientific job to have completed.
"""
from pathlib import Path
import argparse, csv, gzip, hashlib, itertools, json, math, os, statistics

SCOPE = Path(__file__).resolve().parents[1]
for key in ('MPLCONFIGDIR', 'XDG_CACHE_HOME'):
    os.environ[key] = str(SCOPE/'runtime'/'report_cache'/key.lower())
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
METHODS = ('original_feedback', 'fixed_3e4', 'fixed_1e4', 'relaxed_feedback')
NAMES = dict(original_feedback='原学习率反馈', fixed_3e4='固定学习率3e-4',
             fixed_1e4='固定学习率1e-4', relaxed_feedback='放宽反馈区间')
SHORT = dict(original_feedback='原反馈', fixed_3e4='固定3e-4',
             fixed_1e4='固定1e-4', relaxed_feedback='放宽反馈')
DOMAIN_NAMES = dict(radio='通信功率分配', v5='应急柴油分配')
COLORS = dict(original_feedback='#D97922', fixed_3e4='#258AC0',
              fixed_1e4='#398B64', relaxed_feedback='#8061A5')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def dump(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    temp.replace(path)

def csv_read(path):
    fn=gzip.open if str(path).endswith('.gz') else open
    with fn(path,'rt',newline='') as stream:
        return list(csv.DictReader(stream))

def csv_write(path, rows):
    rows=list(rows); assert rows
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def close(a,b,eps=2e-10):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert abs(float(a)-float(b)) <= eps*max(1,abs(float(a)),abs(float(b))), (a,b)

def mean_sd(values):
    values=list(values)
    return dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values),values=values)

def first_confirmation(steps,scores,threshold):
    return next((steps[i] for i in range(1,len(steps)) if scores[i-1]>=threshold and scores[i]>=threshold),None)

def analyze_job(job,plan):
    out=Path(job['output']);complete=read(out/'completed.json');protocol=read(out/'protocol.json')
    assert complete.get('success') is True and complete.get('status')=='completed'
    for key in ['domain','method','seed']:
        assert str(complete[key])==str(job[key]),(job['id'],key)
    steps_total=plan['fixed_training']['steps']; interval=plan['fixed_training']['DEV_every']
    cases=plan['fixed_training']['DEV_cases']; points=plan['fixed_training']['DEV_points_including_zero']
    assert int(complete['training_steps'])==steps_total
    assert int(complete['development_points'])==points
    assert int(complete['development_cases'])==cases
    assert complete['endpoint_reload_pass'] is True and complete['source_unchanged'] is True
    assert complete.get('final_holdout_opened') is False
    assert not complete.get('engineering_gate',False)
    # Every artifact recorded by the scientific worker is checked without loading the model.
    for name,digest in complete.get('artifacts_sha256',{}).items():
        assert sha(out/name)==digest,(job['id'],name)
    curve=csv_read(out/'evaluation_curve.csv.gz');raw=csv_read(out/'evaluation_paths.csv.gz')
    assert len(curve)==points and len(raw)==points*cases
    steps=[int(x['training_step']) for x in curve]
    assert steps==list(range(0,steps_total+1,interval))
    by_step={n:[] for n in steps}
    for row in raw:
        assert row['phase']=='dev';by_step[int(row['training_step'])].append(row)
    scores=[]
    for row,n in zip(curve,steps):
        assert row['phase']=='dev' and int(row['paths'])==cases
        paths=by_step[n];assert len(paths)==cases
        assert sorted(int(x['path_index']) for x in paths)==list(range(cases))
        native=[float(x['native_return']) for x in paths]
        for x in paths:close(float(x['score_x100']),100*float(x['native_return']))
        score=statistics.mean(native)*100;scores.append(score)
        close(score,row['score_x100_mean']);close(score/100,row['native_mean'])
        close(statistics.stdev(native),row['native_std'])
    auc=sum(.5*(scores[i-1]+scores[i])*(steps[i]-steps[i-1]) for i in range(1,len(steps)))/steps_total
    result=dict(id=job['id'],domain=job['domain'],method=job['method'],seed=int(job['seed']),
        training_steps=steps_total,development_points=points,development_cases=cases,
        average_DEV_score=auc,fixed_endpoint_DEV_score=scores[-1],initial_DEV_score=scores[0],
        actual_device=complete['actual_device'],actor_updates=int(complete['actor_updates']),critic_updates=int(complete['critic_updates']),
        learning_environment_steps=int(complete['training_steps']),DEV_environment_steps=int(complete['development_env_steps']),
        total_physical_environment_steps=int(complete['total_physical_environment_steps']),
        initial_actor_sha256=complete['initial_actor_sha256'],initial_critic_sha256=complete['initial_critic_sha256'],
        endpoint_reload_pass=complete['endpoint_reload_pass'],elapsed_seconds=float(complete['elapsed_seconds']))
    thresholds={}
    for key,threshold in plan['thresholds'][job['domain']].items():
        n=first_confirmation(steps,scores,threshold)
        thresholds[key]=dict(threshold=threshold,first_confirmation_steps=n,right_censored=n is None,
            restricted_confirmation_steps=n if n is not None else steps_total)
        result[key+'_threshold']=threshold;result[key+'_first_confirmation_steps']=n
        result[key+'_right_censored']=n is None;result[key+'_restricted_confirmation_steps']=n if n is not None else steps_total
    with gzip.open(out/'updates.jsonl.gz','rt') as stream:
        updates=[json.loads(line) for line in stream if line.strip()]
    assert updates
    post=[x for x in updates if not x['warmup']]
    assert post
    for x in updates:
        for key in ['actor_lr_before','actor_lr_after']:
            assert 1e-5<=float(x[key])<=1e-3
    before=[float(x['actor_lr_before']) for x in post]
    tv=[float(x['tv']) for x in post]
    result.update(postwarmup_batches=len(post),postwarmup_mean_actual_actor_lr=statistics.mean(before),
        postwarmup_mean_policy_change=statistics.mean(tv),postwarmup_exceed_original_upper_fraction=statistics.mean(x>1/15 for x in tv),
        final_actor_lr=float(updates[-1]['actor_lr_after']))
    # These are read-only sources for provenance, not duplicated scientific arrays.
    inputs={str(out/name):sha(out/name) for name in ['protocol.json','completed.json','evaluation_curve.csv.gz','evaluation_paths.csv.gz','updates.jsonl.gz']}
    return result,thresholds,dict(steps=steps,scores=scores,updates=updates,post=post),inputs

def aggregate(registry_path):
    plan=read(SCOPE/'protocol/plan.json');registry=read(registry_path);jobs=registry['jobs']
    assert len(jobs)==16
    assert {(j['domain'],j['method'],int(j['seed'])) for j in jobs}==set(itertools.product(['radio','v5'],METHODS,[7100,7101]))
    rows=[];curves={};provenance={};thresholds_by_job={}
    for job in jobs:
        row,thresholds,curve,inputs=analyze_job(job,plan)
        rows.append(row);curves[job['id']]=curve;provenance.update(inputs);thresholds_by_job[job['id']]=thresholds
    for domain,seed in itertools.product(['radio','v5'],[7100,7101]):
        group=[x for x in rows if x['domain']==domain and x['seed']==seed]
        assert len({x['initial_actor_sha256'] for x in group})==1
        assert len({x['initial_critic_sha256'] for x in group})==1
        assert len({x['actual_device'] for x in group})==1
        for x in group:close(x['initial_DEV_score'],group[0]['initial_DEV_score'])
    summaries=[]
    for domain,method in itertools.product(['radio','v5'],METHODS):
        group=sorted([x for x in rows if x['domain']==domain and x['method']==method],key=lambda x:x['seed'])
        summary=dict(domain=domain,method=method,seeds=[x['seed'] for x in group],n=2)
        for field in ['average_DEV_score','fixed_endpoint_DEV_score','postwarmup_mean_actual_actor_lr','postwarmup_mean_policy_change','postwarmup_exceed_original_upper_fraction']:
            summary[field]=mean_sd(x[field] for x in group)
        for key in ['q90','q95','q100']:
            summary[key]=dict(threshold=group[0][key+'_threshold'],
                restricted_confirmation_steps=mean_sd(x[key+'_restricted_confirmation_steps'] for x in group),
                reached=sum(not x[key+'_right_censored'] for x in group),total=2,
                first_confirmation_steps=[x[key+'_first_confirmation_steps'] for x in group])
        summaries.append(summary)
    paired=[]
    # All six unordered comparisons are retained; the report does not select a winning pair afterward.
    for domain,(a,b),seed in itertools.product(['radio','v5'],itertools.combinations(METHODS,2),[7100,7101]):
        aa=next(x for x in rows if x['domain']==domain and x['method']==a and x['seed']==seed)
        bb=next(x for x in rows if x['domain']==domain and x['method']==b and x['seed']==seed)
        item=dict(domain=domain,method_a=a,method_b=b,seed=seed,difference_direction='A minus B')
        for field in ['q90_restricted_confirmation_steps','q95_restricted_confirmation_steps','q100_restricted_confirmation_steps','average_DEV_score','fixed_endpoint_DEV_score']:
            item[field+'_difference']=aa[field]-bb[field]
        paired.append(item)
    ledger=[]
    for stage,path in [('development',registry_path),('engineering',SCOPE/'protocol/engineering_registry.json')]:
        for j in read(path)['jobs']:
            receipt=Path(j['output'])/'completed.json';c=read(receipt)
            assert c['success'] is True
            assert int(c['total_physical_environment_steps'])==int(c['training_steps'])+int(c['development_env_steps'])
            ledger.append(dict(stage=stage,id=j['id'],domain=j['domain'],method=j['method'],seed=j['seed'],
                training_steps=int(c.get('training_steps',0)),development_steps=int(c.get('development_env_steps',0)),
                endpoint_reload_steps=0,model_loads=int(c.get('model_loads',1 if c.get('endpoint_reload_pass') else 0)),
                physical_steps=int(c.get('total_physical_environment_steps',0)),actual_device=c.get('actual_device'),
                elapsed_seconds=c.get('elapsed_seconds'),endpoint_reload_pass=c.get('endpoint_reload_pass'),receipt=str(receipt),receipt_sha256=sha(receipt)))
    reload_path=SCOPE/'audit/independent/reload_results.json'
    reload=read(reload_path)
    assert reload['status']=='PASS' and len(reload['results'])==16 and reload['total_environment_steps']==14336
    assert {x['id'] for x in reload['results']}=={x['id'] for x in jobs}
    for c in reload['results']:
        assert c['casewise_comparison_pass'] is True and int(c['model_loads'])==1
        ledger.append(dict(stage='endpoint_reload',id=c['id'],domain=c['domain'],method=c['method'],seed=c['seed'],
            training_steps=0,development_steps=0,endpoint_reload_steps=int(c['environment_steps']),model_loads=1,
            physical_steps=int(c['environment_steps']),actual_device='cpu',elapsed_seconds=c.get('elapsed_seconds'),
            endpoint_reload_pass=True,receipt=str(reload_path),receipt_sha256=sha(reload_path)))
    fields=['training_steps','development_steps','endpoint_reload_steps','physical_steps','model_loads']
    totals={stage:{key:sum(x[key] for x in ledger if x['stage']==stage) for key in fields} for stage in ['development','engineering','endpoint_reload']}
    totals['all']={key:sum(x[key] for x in ledger) for key in fields}
    assert totals['development']['training_steps']==16*49152
    assert totals['development']['development_steps']==8*65*32*(32+24)
    assert totals['engineering']['physical_steps']==42240
    assert totals['endpoint_reload']['physical_steps']==14336
    assert totals['all']['physical_steps']==1774848
    provenance[str(reload_path)]=sha(reload_path)
    summary=dict(schema='chapter04-lr-feedback-development-results-v1',status='COMPLETE',
        experiment_scope=str(SCOPE),registry=str(registry_path),registry_sha256=sha(registry_path),plan_sha256=sha(SCOPE/'protocol/plan.json'),
        methods=list(METHODS),domains=['radio','v5'],training_seeds=[7100,7101],training_steps_per_case=49152,
        scientific_runs=16,DEV_points=16*65,DEV_cases=16*65*32,
        primary_comparison='Original q95; first two consecutive raw DEV points, second point confirms; capped at49152 with censor count.',
        statistics='mean and sample SD across two training seeds; descriptive development screen',
        scope_limitations=['No independent TEST has been opened.','No formal model/figure/paper replacement is authorized by this development result.',
            'Two seeds and49152 steps do not establish significance, convergence or general superiority.',
            'Restricted confirmation costs include runs not confirmed by the fixed budget; reached counts must be read with the means.',
            'fixed1e-4 uses the lower rate from the first warmup batch; it tests the entire constant-rate setting.',
            'The fixed1e-4 arm does not isolate lower fixed learning rate after matched warmup; worse results cannot exclude that explanation.',
            'All plots use unsmoothed raw DEV values; no threshold, seed, endpoint or successful run is selected after inspection.'],
        thresholds=plan['thresholds'],per_seed=rows,summary=summaries,paired_differences=paired,
        computational_ledger=dict(totals=totals,entries=ledger),provenance=provenance,
        scientific_execution_by_aggregation=dict(environment_steps=0,model_loads=0,inference=0,training=0,new_TEST=0))
    return summary,curves

def render_report(summary,curves):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import FuncFormatter,NullFormatter,LogLocator
    font_path=Path('/root/gamewar/latex_thesis_resource_allocation/assets/fonts/figure/ThesisFigureText-Regular.otf')
    font_manager.fontManager.addfont(str(font_path));font_name=font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({'font.family':font_name,'font.size':10.5,'axes.unicode_minus':False,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    report=SCOPE/'report';report.mkdir(exist_ok=True)
    pdf_path=report/'第四章_学习率反馈四组对照_开发测试结果.pdf'
    result_lookup={(x['domain'],x['method']):x for x in summary['summary']}
    def figure(page,title,subtitle=True):
        fig=plt.figure(figsize=(210/25.4,297/25.4))
        fig.text(.075,.958,title,fontsize=14,va='top')
        if subtitle:
            fig.text(.075,.928,'四组 × 两个任务 × 两个配对种子；每项从零训练49152步',fontsize=10.5,va='top')
        fig.text(.075,.025,'开发筛查：无独立TEST，正式论文与方法配置保持原身份。',fontsize=9)
        fig.text(.925,.025,str(page),fontsize=9,ha='right')
        return fig
    def text(fig,y,lines,size=10.5):
        for line in lines:
            fig.text(.075,y,line,fontsize=size,va='top');y-=.025
        return y
    def table(fig,position,headers,rows,widths=None,size=10.5):
        ax=fig.add_axes(position);ax.axis('off')
        t=ax.table(cellText=rows,colLabels=headers,cellLoc='center',loc='upper center',colWidths=widths)
        t.auto_set_font_size(False);t.set_fontsize(size)
        for (row,col),cell in t.get_celld().items():
            cell.set_height(.16 if len(rows)<=4 else .115)
            cell.set_linewidth(.45);cell.set_edgecolor('#BBBBBB')
            cell.set_facecolor('#EEEEEE' if row==0 else 'white')
        return t
    def display_sd(stat,div=1):return f"{stat['mean']/div:.2f}±{stat['sample_sd']/div:.2f}"
    with PdfPages(pdf_path,metadata={'Title':'第四章 学习率反馈四组对照 开发测试结果','Author':'Resource allocation research','CreationDate':None,'ModDate':None}) as pdf:
        fig=figure(1,'第四章：学习率反馈四组开发对照')
        text(fig,.884,[
            '本轮检查原反馈是否过于保守，以及较小固定学习率本身的作用。',
            '四组保持轨迹复用、校正、裁剪宽度、网络与评价设置一致。',
            '原反馈区间为[0.0333,0.0667]；放宽组为[0.05,0.10]。',
            '两反馈组初始学习率均为3e-4，调整倍数均为1.03。',
            '固定组分别全程采用3e-4或1e-4，含预热阶段。',
            '通信统一使用CPU，柴油统一使用CUDA；同任务四组后端相同。'])
        for domain,y in [('radio',.704),('v5',.466)]:
            fig.text(.075,y,DOMAIN_NAMES[domain]+'：原比较得分下的结果',fontsize=11.5,va='top')
            rows=[]
            for m in METHODS:
                x=result_lookup[(domain,m)]
                rows.append([SHORT[m],display_sd(x['q95']['restricted_confirmation_steps'],1000),f"{x['q95']['reached']}/2",
                    display_sd(x['average_DEV_score']),display_sd(x['fixed_endpoint_DEV_score'])])
            table(fig,[.07,y-.195,.86,.17],['设置','计入步数/千步','确认次数','平均开发成效','终点开发成效'],rows,[.20,.23,.13,.22,.22])
        q95=lambda d,m:result_lookup[(d,m)]['q95']['restricted_confirmation_steps']['mean']/1000
        delta={r:{d:q95(d,'relaxed_feedback')-q95(d,r) for d in ['radio','v5']} for r in ['original_feedback','fixed_3e4']}
        text(fig,.267,[
            f"放宽反馈相对原反馈：通信平均计入步数{delta['original_feedback']['radio']:+.2f}千步，柴油{delta['original_feedback']['v5']:+.2f}千步。",
            f"相对固定3e-4：通信{delta['fixed_3e4']['radio']:+.2f}千步，柴油{delta['fixed_3e4']['v5']:+.2f}千步（负值为较早）。"])
        text(fig,.195,[
            '步数为首次连续两个开发点达到比较得分的第二点；评价每768步一次。',
            '未确认运行按49152步计入平均，同时保留确认次数。',
            '平均开发成效为原始曲线梯形积分除以49152；终点为固定预算末点。',
            '通信成效单位为分，柴油为%；表中误差为两个种子的样本标准差。',
            '这轮结果用于比较候选设置，不据此宣称显著差异或正式采用。'])
        pdf.savefig(fig);fig.savefig(report/'page_1.png',dpi=110);plt.close(fig)

        fig=figure(2,'完整开发曲线：全部四组、两个任务',subtitle=False)
        for index,domain in enumerate(['radio','v5']):
            ax=fig.add_axes([.14,.565-index*.385,.79,.285])
            for m in METHODS:
                group=[x for x in summary['per_seed'] if x['domain']==domain and x['method']==m]
                yy=np.asarray([curves[x['id']]['scores'] for x in group]);xx=np.asarray(curves[group[0]['id']]['steps'])/1000
                means=yy.mean(axis=0);sd=yy.std(axis=0,ddof=1)
                ax.plot(xx,means,color=COLORS[m],lw=1.5,label=SHORT[m])
                ax.fill_between(xx,means-sd,means+sd,color=COLORS[m],alpha=.12,linewidth=0)
            q=summary['thresholds'][domain]['q95']
            ax.axhline(q,color='#777777',lw=.8,ls='--')
            ax.set_title(('(a) ' if index==0 else '(b) ')+DOMAIN_NAMES[domain],pad=9)
            ax.set_xlabel('新训练交互步数（×10³）')
            ax.set_ylabel('平均图像质量得分/分' if domain=='radio' else '平均需求完成比例/%')
            ax.set_xlim(0,49.152);ax.grid(alpha=.15)
            if index==0:ax.legend(loc='lower right',fontsize=9.5,frameon=False,ncol=2)
        text(fig,.125,['实线为两种子原始DEV均值，阴影为样本标准差；曲线未平滑。',
                       '灰色虚线为原主要比较得分；未调整比较值或选取中间模型。'])
        pdf.savefig(fig);fig.savefig(report/'page_2.png',dpi=110);plt.close(fig)

        fig=figure(3,'比较得分敏感性与配对差值',subtitle=False)
        text(fig,.884,['三档比较得分均沿用原登记值；所有组使用相同确认规则。',
                       '单元格为平均计入千步数，括号内为两次训练中的确认次数。'])
        rows=[]
        for d in ['radio','v5']:
            for q in ['q90','q95','q100']:
                vals=[]
                for m in METHODS:
                    x=result_lookup[(d,m)][q]
                    vals.append(f"{x['restricted_confirmation_steps']['mean']/1000:.2f} ({x['reached']}/2)")
                rows.append(['通信' if d=='radio' else '柴油',f"{summary['thresholds'][d][q]:.2f}",*vals])
        table(fig,[.065,.555,.87,.255],['任务','比较得分','原反馈','固定3e-4','固定1e-4','放宽反馈'],rows,[.10,.14,.19,.19,.19,.19],size=10)
        fig.text(.075,.505,'放宽反馈相对两项参照的配对变化（放宽组减参照）',fontsize=11.5,va='top')
        rows=[]
        for d,reference in itertools.product(['radio','v5'],['original_feedback','fixed_3e4']):
            delta=[]
            for seed in [7100,7101]:
                a=next(x for x in summary['per_seed'] if x['domain']==d and x['method']=='relaxed_feedback' and x['seed']==seed)
                b=next(x for x in summary['per_seed'] if x['domain']==d and x['method']==reference and x['seed']==seed)
                delta.append((a['q95_restricted_confirmation_steps']-b['q95_restricted_confirmation_steps'],a['average_DEV_score']-b['average_DEV_score'],a['fixed_endpoint_DEV_score']-b['fixed_endpoint_DEV_score']))
            rows.append(['通信' if d=='radio' else '柴油',SHORT[reference],
                         f"{delta[0][0]/1000:+.2f}/{delta[1][0]/1000:+.2f}",
                         f"{statistics.mean(x[1] for x in delta):+.2f}",f"{statistics.mean(x[2] for x in delta):+.2f}"])
        table(fig,[.065,.245,.87,.205],['任务','参照','两seed步数差/千步','平均成效差','终点成效差'],rows,[.10,.21,.29,.20,.20],size=10)
        text(fig,.190,['步数差为负表示放宽组较早确认；成效差为正表示放宽组较高。',
                       '两种子的逐项差值与全部六组两两配对均保存在CSV，不筛选比较。',
                       '受限步数相同可能包含两组均未确认，应同时读取确认次数。'])
        pdf.savefig(fig);fig.savefig(report/'page_3.png',dpi=110);plt.close(fig)

        fig=figure(4,'学习率、更新幅度与实际计算量')
        for column,d in enumerate(['radio','v5']):
            for rr,key in enumerate(['actor_lr_before','tv']):
                ax=fig.add_axes([.125+column*.455,.625-rr*.255,.345,.18])
                for m in METHODS:
                    group=[x for x in summary['per_seed'] if x['domain']==d and x['method']==m]
                    uu=[curves[x['id']]['post'] for x in group]
                    grid=[int(x['training_step']) for x in uu[0]]
                    assert all([int(x['training_step']) for x in row]==grid for row in uu)
                    yy=np.asarray([[float(x[key]) for x in u] for u in uu])
                    ax.plot(np.asarray(grid)/1000,yy.mean(axis=0),color=COLORS[m],lw=1.1,label=SHORT[m])
                ax.set_xlim(0,49.152);ax.set_xlabel('新训练交互步数（×10³）',fontsize=9)
                ax.tick_params(labelsize=9);ax.grid(alpha=.15)
                if rr==0:
                    ax.set_title(DOMAIN_NAMES[d],fontsize=10.5,pad=7);ax.set_ylabel('实际Actor学习率',fontsize=9);ax.set_yscale('log')
                    ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,3,5)))
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:.0e}'.replace('e-0','e-').replace('e+0','e+')))
                    ax.yaxis.set_minor_formatter(NullFormatter())
                else:ax.set_ylabel('单批策略变化幅度',fontsize=9)
        handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='center',bbox_to_anchor=(.52,.312),ncol=4,frameon=False,fontsize=9)
        t=summary['computational_ledger']['totals']
        text(fig,.265,[
            f"开发学习：{t['development']['training_steps']:,}步；DEV评价：{t['development']['development_steps']:,}步。",
            f"短工程10项：{t['engineering']['physical_steps']:,}步；独立重载：{t['endpoint_reload']['physical_steps']:,}步。",
            f"本轮累计：{t['all']['physical_steps']:,}物理步；独立TEST为0。",
            '过程图排除预热批次，显示逐批两种子均值；无平滑或抽取好看区间。',
            '更新幅度与学习率用于解释过程，其大小不直接等于配置成效。',
            '固定1e-4从预热首批即采用低率，与原反馈组的预热更新不同。',
            '相同预热后再采用较低固定率的作用，尚未由这项对照隔离。',
            '49152步的两种子结果不证明收敛、显著性或任意任务上的优越性。',
            '正式方法、论文图件与原实验结果均未由本开发报告替换。'])
        pdf.savefig(fig);fig.savefig(report/'page_4.png',dpi=110);plt.close(fig)
    return pdf_path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--allow-report-rebuild',action='store_true')
    args=parser.parse_args()
    results=SCOPE/'results';results.mkdir(exist_ok=True)
    summary_path=results/'summary.json'
    if summary_path.exists() and not args.allow_report_rebuild:
        raise FileExistsError('Final results already exist; explicit layout-only rebuild required')
    summary,curves=aggregate(args.registry)
    if summary_path.exists():
        previous=read(summary_path)
        assert previous==summary,'Report rebuild cannot change any numerical input or result'
    # No data products are emitted until every scientific job and engineering receipt has passed.
    dump(summary_path,summary)
    csv_write(results/'per_seed.csv',summary['per_seed'])
    csv_write(results/'paired_differences.csv',summary['paired_differences'])
    csv_write(results/'computational_ledger.csv',summary['computational_ledger']['entries'])
    flat=[]
    for x in summary['summary']:
        row=dict(domain=x['domain'],method=x['method'],n=x['n'])
        for key in ['average_DEV_score','fixed_endpoint_DEV_score']:
            row[key+'_mean']=x[key]['mean'];row[key+'_sample_sd']=x[key]['sample_sd']
        for q in ['q90','q95','q100']:
            row[q+'_mean_restricted_steps']=x[q]['restricted_confirmation_steps']['mean']
            row[q+'_sample_sd_steps']=x[q]['restricted_confirmation_steps']['sample_sd'];row[q+'_reached']=x[q]['reached']
        flat.append(row)
    csv_write(results/'method_domain_summary.csv',flat)
    pdf=render_report(summary,curves)
    dump(SCOPE/'report/report_receipt.json',dict(status='GENERATED_AWAITING_VISUAL_REVIEW',pdf=str(pdf),pdf_sha256=sha(pdf),pages=4,
        results=str(summary_path),results_sha256=sha(summary_path),aggregation_script=str(Path(__file__)),aggregation_script_sha256=sha(__file__),
        numerical_checks='Every registered endpoint receipt/artifact and all65×32 DEV path means/SD and three confirmation rules recomputed.',
        scientific_execution=dict(environment_steps=0,model_loads=0,inference=0,training=0,new_TEST=0)))
    print(json.dumps(dict(status='GENERATED',pdf=str(pdf),summary=str(summary_path),runs=16),ensure_ascii=False))

if __name__=='__main__':main()
