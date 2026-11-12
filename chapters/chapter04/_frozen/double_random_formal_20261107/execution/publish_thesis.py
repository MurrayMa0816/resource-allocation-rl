#!/usr/bin/env python3
"""Publish only the complete, independently accepted registered C4 cohort.

Default prepares a reproducible HY staging tree after all scientific gates.
--publish additionally integrates only C4 assets and runs the sole whole-thesis
build entry five actual forced passes. No training/environment calls occur.
"""
from __future__ import annotations
import argparse,ast,csv,datetime,fcntl,hashlib,json,math,os,re,shutil,statistics,subprocess,sys,time,traceback
from collections import Counter,defaultdict
from contextlib import contextmanager
from pathlib import Path

SCOPE=Path(__file__).resolve().parents[1]
THESIS=Path('/root/gamewar/latex_thesis_resource_allocation')
C4=Path('chapters/chapter04_trajectory_reuse')
sys.path.insert(0,str(SCOPE/'source/reporting'))
import thesis_content as content
import contract_patches
import citations

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.new');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)
def csvrows(path):
    with Path(path).open(newline='',encoding='utf-8-sig')as f:return list(csv.DictReader(f))
def savecsv(path,rows):
    assert rows,'No empty publication data'
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8')as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-11)
def require(condition,message):
    if not condition:raise ValueError(message)
def source_hashes():
    paths=[Path(__file__),*[SCOPE/'source/reporting'/n for n in ('thesis_content.py','contract_patches.py','citations.py','figure_plot.py')]]
    return {str(p):digest(p)for p in paths}

def gate():
    """Recheck accepted identity and aggregate CSVs, not optimistic progress."""
    ap=SCOPE/'independent/acceptance.json';sp=SCOPE/'reports/summary.json'
    require(ap.exists()and sp.exists(),'Full independent acceptance and aggregate summary are required; thesis unchanged')
    a=read(ap);s=read(sp);p=read(SCOPE/'plan.json');r=read(SCOPE/'reports/aggregation_receipt.json')
    require(a.get('status')=='PASS'and a.get('verified_jobs')==190,'Independent190-job acceptance not PASS')
    require(s.get('schema')=='chapter4-accepted-formal-summary-v1'and s.get('status')=='PASS','Unsupported or failed scientific summary')
    require(s.get('scope_root')==str(SCOPE),'Wrong scientific scope')
    for key,path in [('plan_sha256',SCOPE/'plan.json'),('quality_targets_sha256',SCOPE/'protocol/quality_targets.json')]:
        require(s[key]==a[key]==digest(path),'Bound '+key+' changed')
    require(s['acceptance_sha256']==digest(ap)==r['acceptance_sha256'],'Accepted cohort binding changed')
    require(r.get('status')=='PASS'and r['summary_sha256']==digest(sp),'Aggregation receipt mismatch')
    require((r['unique_training_runs'],r['inherited_models'],r['endpoint_panels'])==(90,10,100),'Incomplete result roster')
    require(len(p['jobs'])==190 and len({j['id']for j in p['jobs']})==190,'Wrong registered job roster')
    require((p['new_training_jobs'],p['reused_external_trainings'],p['unique_training_runs'],p['reused_A_models'])==(50,40,90,10),'Wrong mixed-origin cohort design')
    require((s['new_training_jobs'],s['reused_external_trainings'],s['unique_training_runs'],s['reused_A_models'])==(50,40,90,10),'Mixed-origin scientific summary is incomplete')
    require(s['initialization']==p['initialization']=='both_random','New F1-family initialization is not both_random')
    require(set(s['initialization_arms'])==set(p['initialization_arms'])=={'f1','props','no_history','no_offpolicy_correction','fixed_actor_lr'},'Wrong initialized method arms')
    require(s.get('A_reference_curve_excluded')is True,'Do not mix the new C3 original512x8 curve into C4 common768x32')
    require((r['new_training_jobs'],r['reused_external_trainings'])==(50,40),'Aggregation does not disclose retained training origins')
    require(s['physical_costs']['unique_new_training_steps']==5030400 and s['physical_costs']['retained_external_training_steps']==4024320,'New and retained interaction costs were confused')
    accepted={x['id']:x for x in a['jobs']};require(set(accepted)=={j['id']for j in p['jobs']},'Acceptance roster mismatch')
    for j in p['jobs']:
        for rel,h in accepted[j['id']]['input_sha256'].items():
            require(digest(Path(j['output'])/rel)==h,'Accepted raw input changed: '+j['id']+'/'+rel)
    require(s['training_steps']==100608 and s['evaluation_interval']==768 and s['development_cases']==32 and s['development_points']==132 and s['final_cases']==256,'Wrong budget or evaluation grid')
    require(s['seeds']==[0,1,2,3,4]and tuple(s['main_methods'])==content.MAIN and tuple(s['ablation_methods'])==content.ABLATIONS,'Wrong method/seed definition')
    require(s['A_reference_method']=='A_inherited','A baseline identity changed')
    validate_configurations(s)
    data={}
    for name,h in s['csv_sha256'].items():
        require(Path(name).name==name and name.endswith('.csv'),'Unsafe aggregate path')
        path=SCOPE/'reports'/name;require(digest(path)==h,'Aggregate CSV changed: '+name);data[name]=csvrows(path)
    numeric_audit(s,data)
    return s,data,dict(plan_sha256=digest(SCOPE/'plan.json'),acceptance_sha256=digest(ap),summary_sha256=digest(sp),csv_sha256=s['csv_sha256'])

def mrq_source_path():
    return Path(read(SCOPE/'plan.json')['old_scope'])/'source/offpolicy/offpolicy_agents/mrq/vendor/MRQ.py'

def validate_configurations(s):
    """Ensure prose parameters still match the accepted actual configurations."""
    mrqsource=mrq_source_path()
    tree=ast.parse(mrqsource.read_text());hp=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Hyperparameters')
    defaults={n.target.id:ast.literal_eval(n.value)for n in hp.body if isinstance(n,ast.AnnAssign)and isinstance(n.target,ast.Name)}
    require((defaults['batch_size'],defaults['target_update_freq'],defaults['enc_horizon'],defaults['Q_horizon'])==(256,250,5,3),'MR.Q author defaults differ from prose')
    for domain,target_batch,behavior_batch in [('radio',8,14),('v5',6,10)]:
        configs=s['method_configs']
        for method in content.MAIN:
            cfg=configs[method+'__'+domain]
            require(close(cfg['gamma'],1 if method in ('f1','props','pgr')else .99),'Configuration gamma changed')
            require(digest(cfg['protocol_path'])==cfg['protocol_sha256'],'Accepted actual protocol changed')
        p=configs['props__'+domain]['configuration'];e=p['effective_args']
        require(close(p['learning_rate'],5e-5)and close(p['props_learning_rate'],1e-4),'PROPS documented user-selected learning rates changed')
        require(e['minibatch_size']==target_batch and e['props_minibatch_size']==behavior_batch and p['update_epochs']==10 and p['props_update_epochs']==16 and p['buffer_episodes']==8,'PROPS actual batching differs from prose')
        p=configs['pgr__'+domain]['configuration']['pgr'];require(p['utd_ratio']==20,'PGR documented UTD changed')
        p=configs['redo__'+domain]['configuration']['algorithm'];require((p['warmup_steps'],p['replay_ratio'],p['batch_size'],p['recycle_period'])==(10000,8,256,200000),'ReDo actual parameters differ from prose')
        p=configs['bro__'+domain]['configuration']['algorithm'];require((p['start_training'],p['updates_per_step'],p['batch_size'])==(2500,2,128),'BRO actual parameters differ from prose')

def numeric_audit(s,data):
    """Independent display-level aggregation and censoring checks."""
    curves=defaultdict(list);endpoints=defaultdict(list)
    methods=set(content.MAIN)|set(content.ABLATIONS)
    for row in data['learning_curves.csv']:
        key=(row['domain'],row['method'],int(row['seed']));curves[key].append(row)
        require(close(row['score_x100'],100*float(row['mean_return'])),'Wrong reward display scale')
        require(int(row['development_cases'])==32,'Mixed development panels')
    require(len(curves)==90,'Main/ablation curve set incomplete')
    for row in data['endpoint_cases.csv']:
        key=(row['domain'],row['method'],int(row['seed']));endpoints[key].append(row)
        require(close(row['score_x100'],100*float(row['native_return'])),'Wrong endpoint display scale')
    require(len(endpoints)==100,'Endpoint cohort incomplete')
    for d in ('radio','v5'):
        require(set(s['methods'][d])==methods|{'A_inherited'},'Unexpected scientific result method')
        for m in methods|{'A_inherited'}:
            g=s['methods'][d][m];require(g['n']==5 and g['seeds']==list(range(5)),'Wrong statistical unit')
            means=[];aucs=[];hits=defaultdict(list)
            for seed in range(5):
                case=endpoints[(d,m,seed)]
                require(len(case)==256 and {int(x['case_index'])for x in case}==set(range(256)),'Incomplete independent final cases')
                require(all(abs(float(x['resource_fraction'])-1)<1e-5 for x in case),'Endpoint full-spend violation')
                means.append(statistics.fmean(float(x['native_return'])for x in case))
                if m=='A_inherited':continue
                rr=sorted(curves[(d,m,seed)],key=lambda x:int(x['training_step']))
                require([int(x['training_step'])for x in rr]==list(range(0,100608+1,768)),'Unexpected development grid')
                y=[float(x['mean_return'])for x in rr]
                aucs.append(sum((y[i-1]+y[i])*.5*768 for i in range(1,len(y)))/100608)
                for q,lab in [('0.9','q90'),('0.95','q95'),('1.0','q100')]:
                    threshold=s['thresholds'][d][q]
                    hit=next((768*i for i in range(1,len(y))if y[i-1]>=threshold and y[i]>=threshold),None)
                    hits[lab].append(hit)
            require(close(statistics.fmean(means),g['endpoint']['mean'])and close(statistics.stdev(means),g['endpoint']['sample_sd']),'Final mean/sample SD mismatch')
            if m=='A_inherited':require('auc'not in g and 'nq'not in g,'Historical8-case A was mixed into main grid');continue
            require(close(statistics.fmean(aucs),g['auc']['mean'])and close(statistics.stdev(aucs),g['auc']['sample_sd']),'AUC statistic mismatch')
            for lab,values in hits.items():
                q=g['nq'][lab];restricted=[x if x is not None else 100608 for x in values]
                require(q['reached_steps']==values and q['censor_flags']==[x is None for x in values],'Censor identity changed')
                require(q['reached_n']==sum(x is not None for x in values)and close(q['restricted_mean_steps'],statistics.fmean(restricted))and close(q['sample_sd'],statistics.stdev(restricted)),'Nq restricted cost mismatch')
    require(all(str(x['same_long_run_prefix']).lower()=='true'for x in data['budget_prefixes.csv']),'Unexpected independently tuned budget runs')

def readonly_figures_audit(thesis):
    """Refuse unnoticed cap labels rather than redraw unrelated diagrams."""
    result=[]
    for p in (thesis/C4/'figures').glob('figure04-0[123]_*/source.drawio'):
        text=p.read_text();bad=[x for x in ('终点允许保留余额','终末余额允许','终点可有余额','q=U sigmoid','预算上限映射')if x in text]
        require(not bad,'Method figure contains a stale cap label needing review: '+str(p))
        result.append({'path':str(p),'sha256':digest(p),'unchanged':True,'stale_cap_literals':bad})
    require(len(result)==3,'Three current C4 source.drawio files expected')
    return result

def prepare_stage(s,data,bindings,thesis):
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    stage=SCOPE/'publication'/('staging_'+stamp);stage.mkdir(parents=True)
    c4=thesis/C4;required=['chapter.tex','section51_problem.tex','section52_method.tex','section53_experiments.tex','section54_summary.tex','references.bib']
    original={n:(c4/n).read_text()for n in required};oldsha={str(c4/n):digest(c4/n)for n in required}
    mapping,newbib=citations.resolve(thesis)
    texts,patches=contract_patches.prepare(original['chapter.tex'],original['section51_problem.tex'],original['section52_method.tex'])
    texts['section53_experiments.tex']=content.make_section(s,data['f1_internal.csv'],mapping)
    texts['section54_summary.tex']=content.make_summary(s)
    texts['references.bib']=original['references.bib']+('\n\n% Registered C4 formal comparison primary sources\n'+newbib+'\n'if newbib else'')
    staged=stage/C4;staged.mkdir(parents=True)
    for name,text in texts.items():(staged/name).write_text(text)
    localfigs=[]
    for number,caption,label,kind,methods,csvname in content.FIGURES:
        folder=staged/'figures'/content.figure_dir(number,caption);folder.mkdir(parents=True)
        if kind=='cost':
            rows=[dict(domain=r['domain'],method=r['method'],seed=r['seed'],restricted_steps=r['nq_q95_restricted'],censored=r['nq_q95_censored'])for r in data['run_metrics.csv']if r['method']in methods]
        else:rows=[r for r in data[csvname]if r['method']in methods]
        savecsv(folder/'data.csv',rows)
        shutil.copyfile(SCOPE/'source/reporting/figure_plot.py',folder/'plot.py')
        write(folder/'figure_config.json',dict(kind=kind,methods=list(methods),thresholds={d:100*s['thresholds'][d]['0.95']for d in ('radio','v5')},display_scale=100,native_data=True,training_seeds=5,sample_sd=True,internal_nonoverlapping_bin_steps=3072,main_smoothing='none'))
        source=dict(**bindings,scientific_data=str(SCOPE/'reports'/(csvname or'run_metrics.csv')),local_data_sha256=digest(folder/'data.csv'),plot_sha256=digest(folder/'plot.py'),plot_source_sha256=digest(SCOPE/'source/reporting/figure_plot.py'),reproduction='python3 plot.py --output-dir .',native_width_cm=15.6,main_font_bp=10.5,tick_legend_font_bp=9.5,source_or_actual_measurements_only=True)
        write(folder/'source.json',source)
        (folder/'README.rst').write_text(caption+'\n'+'='*len(caption)+'\n\n数据来自已独立验收的本轮正式实验。相邻data.csv为实际原始开发或更新记录的筛选，source.json绑定完整汇总与验收。运行 python3 plot.py --output-dir . 可重新生成相同角色的图。主学习曲线未平滑，阴影为五次独立训练的样本标准差。内部指标的3072步不重叠汇总不参与达标及曲线面积计算。\n')
        log=stage/('plot_'+number+'.log')
        with log.open('w')as f:
            proc=subprocess.run([sys.executable,str(folder/'plot.py'),'--output-dir',str(folder)],stdout=f,stderr=subprocess.STDOUT,env=runtime_env(),timeout=300)
        require(proc.returncode==0,'Accepted figure failed: '+str(log))
        localfigs.append(dict(number=number,caption=caption,path=str(folder.relative_to(stage)),pdf_sha256=digest(folder/'figure.pdf')))
    audit=dict(status='PREPARED_NOT_PUBLISHED',**bindings,source_sha256=source_hashes(),thesis_root=str(thesis),old_owned_sha256=oldsha,contract_patch_spans=patches,figures=localfigs,unchanged_method_figures=readonly_figures_audit(thesis),citations=mapping,primary_source_reading=citations.SOURCES,actual_method_configurations=s['method_configs'],MRQ_native_hyperparameter_source_sha256=digest(mrq_source_path()),extra_scientific_environment_steps=0)
    write(stage/'preparation_receipt.json',audit)
    content_audit(stage)
    return stage,audit

def runtime_env():
    env=os.environ.copy();cache=SCOPE/'publication/runtime';cache.mkdir(parents=True,exist_ok=True)
    for key,leaf in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp'),('TEXMFVAR','texmfvar'),('TEXMFCACHE','texmfcache'),('PYTHONPYCACHEPREFIX','pycache')]:
        p=cache/leaf;p.mkdir(exist_ok=True);env[key]=str(p)
    env['THESIS_FONT_DIR']=str(THESIS/'assets/fonts/windows')
    return env

def content_audit(stage):
    path=stage/C4/'section53_experiments.tex';text=path.read_text()
    require(text.count('\\subsection{')==3,'4.3 must have exactly three subsections')
    require('960'not in text,'Stale training length')
    require(all(x not in text for x in ('B1','FoG','ADDQ','TD-MPC2','显著优于','已证明梯度方差下降')),'Stale methods or unsupported claims')
    require(text.count('256个')>=1 and '132个'in text and '100608'in text,'Missing actual protocol')
    require(all(p not in text for p in ('、','：','；','“','”','！')),'Narrative violates current punctuation rule D01')
    labels=re.findall(r'\\label\{([^}]+)\}',text);require(len(labels)==len(set(labels)),'Duplicate C4 labels')
    for number,caption,*_ in content.FIGURES:
        folder=stage/C4/'figures'/content.figure_dir(number,caption)
        require((folder/'figure.pdf').stat().st_size>1000 and (folder/'figure.png').exists(),'Missing rendered data figure')
    write(stage/'automated_content_checks.json',dict(status='PASS',subsections=3,figures=6,forbidden_punctuation_absent=True,wrong_old_roster_absent=True,manual_visual_review='pending'))

@contextmanager
def lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a')as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def revised_index(rows):
    retained=[r for r in rows if not(r['章节目录']=='chapter04_trajectory_reuse'and int(r['图号'].split('.')[1])>=4)]
    for number,caption,*_ in content.FIGURES:
        retained.append(dict(zip(['图号','图题','章节目录','图目录','正文PDF','可编辑图源','绘图程序','绘图数据'],
            [number,caption,'chapter04_trajectory_reuse',str(C4/'figures'/content.figure_dir(number,caption)),'figure.pdf','','plot.py','data.csv'])))
    return sorted(retained,key=lambda r:tuple(map(int,r['图号'].split('.'))))

def integrate(stage,audit,thesis):
    backup=stage/'backup';backup.mkdir();copied=[]
    with lock(thesis/'build/.whole_thesis_build.lock'):
        for p,h in audit['old_owned_sha256'].items():require(digest(p)==h,'C4 changed during staging; prepare against latest manuscript instead')
        gate() # Raw/summary source may never drift between staging and integration.
        shutil.copytree(thesis/C4,backup/C4)
        index=thesis/'chapters/FIGURE_INDEX.csv';shutil.copyfile(index,backup/'FIGURE_INDEX.csv')
        if(thesis/'build/main.pdf').exists():shutil.copyfile(thesis/'build/main.pdf',backup/'main.pdf')
        # Only own C4 sources and new experiment figure directories are integrated.
        for src in (stage/C4).iterdir():
            dst=thesis/C4/src.name
            if src.is_file():shutil.copyfile(src,dst);copied.append(str(dst))
        for item in audit['figures']:
            src=stage/item['path'];dst=thesis/item['path']
            require(not dst.exists(),'New formal figure directory already exists; avoid overwriting another publication')
            shutil.copytree(src,dst);copied.extend(str(p)for p in dst.rglob('*')if p.is_file())
        rows=revised_index(csvrows(index));savecsv(index,rows)
        state=dict(status='INTEGRATED_BUILD_PENDING',stage=str(stage),own_file_sha256={p:digest(p)for p in copied},C4_index_rows=[r for r in rows if r['章节目录']=='chapter04_trajectory_reuse'],backup=str(backup),other_chapters_not_overwritten=True)
        write(stage/'integration_receipt.json',state)
    return state

def actual_builds(stage,thesis):
    rc=stage/'force_latexmk.rc';rc.write_text('$go_mode = 1;\n1;\n')
    env=runtime_env();env['LATEXMKRCSYS']=str(rc)
    logs=[]
    for i in range(1,6):
        log=stage/('whole_build_round'+str(i)+'.log');started=time.time()
        with log.open('w')as f:
            proc=subprocess.run(['bash','scripts/build.sh'],cwd=thesis,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=1200)
        text=log.read_text(errors='replace')
        require(proc.returncode==0,'Whole thesis build failed: '+str(log))
        require('This is XeTeX'in text or re.search(r"Run number .*xelatex",text),'Build did not perform a real XeLaTeX pass: '+str(log))
        require((thesis/'build/main.pdf').stat().st_mtime>=started-2,'PDF is stale after actual build')
        # latexmk may correctly resolve warnings from an earlier pass. Check
        # the final TeX pass, not all historical text in this build's stdout.
        with lock(thesis/'build/.whole_thesis_build.lock'):
            final_log=(thesis/'build/main.log').read_text(errors='replace')
            require(not re.search(r'undefined references|Citation .* undefined|Reference .* undefined|There were undefined citations',final_log,re.I),'Unresolved citation/reference in final whole-thesis TeX pass')
            (stage/('whole_build_round'+str(i)+'.final_tex.log')).write_text(final_log)
        review={1:'three-subsection structure and result-dependent narrative',2:'raw accepted cohort and statistics rebound',3:'actual rendered pages and embedded figure fonts',4:'actual PDF numeric content against accepted summary',5:'final build and no unresolved cross references'}[i]
        if i==1:content_audit(stage)
        elif i==2:gate()
        elif i in (3,4):
            with lock(thesis/'build/.whole_thesis_build.lock'):inspect_pdf(stage,thesis,'round'+str(i))
        logs.append(dict(round=i,log=str(log),log_sha256=digest(log),pdf_sha256=digest(thesis/'build/main.pdf'),actual_xelatex=True,automated_review=review,manual_visual_review='pending'))
    write(stage/'five_actual_whole_builds.json',dict(status='PASS',build_entry='bash scripts/build.sh',force_mechanism='LATEXMKRCSYS $go_mode=1',passes=logs))
    return logs

def inspect_pdf(stage,thesis,role='final'):
    pdf=thesis/'build/main.pdf';out=stage/('rendered_'+role);out.mkdir()
    # PyMuPDF identifies real reader pages from text, not a guessed printed offset.
    import fitz
    doc=fitz.open(pdf);texts=[p.get_text()for p in doc];compact_pages=[re.sub(r'\s+','',t)for t in texts]
    start=next((i for i,t in enumerate(compact_pages)if '本章沿用前章的通信与应急资源配置环境'in t),None)
    require(start is not None,'Could not locate updated C4 experimental pages')
    end=next((i for i in range(start,len(texts))if '本章小结'in texts[i] and ('4.4'in texts[i]or'TRoC'in texts[i])),None)
    require(end is not None and end>start,'Could not locate updated C4 summary')
    joined='\n'.join(texts[start:end+1]);compact=re.sub(r'\s+','',joined)
    require('100608'in compact and '132'in compact and '256'in compact and 'MR.Q'in compact and 'PROPS'in compact,'Built PDF lacks new protocol or methods')
    require(all(('4.'+str(i))in compact for i in range(4,10)),'Built PDF missing an experimental figure number')
    summary=read(SCOPE/'reports/summary.json')
    expected=[content.f(100*summary['thresholds'][d]['0.95'])for d in ('radio','v5')]
    expected += [content.f(100*summary['methods'][d][m][metric]['mean'])for d in ('radio','v5')for m in set(content.MAIN)|set(content.ABLATIONS)|{'A_inherited'}for metric in (['endpoint']if m=='A_inherited'else['endpoint','auc'])]
    require(all(value in compact for value in expected),'PDF displayed statistics do not match accepted summary')
    pages=range(max(0,start-1),min(len(doc),end+2));render=[]
    for i in pages:
        page=doc[i];p=out/('reader_page_'+str(i+1)+'.png');page.get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False).save(p)
        render.append(dict(reader_page=i+1,path=str(p),sha256=digest(p)))
    (stage/('chapter4_built_text_'+role+'.txt')).write_text(joined)
    # Local figure dimensions/fonts also checked from their actual exported PDFs.
    figurechecks=[]
    for item in content.FIGURES:
        fp=thesis/C4/'figures'/content.figure_dir(item[0],item[1])/'figure.pdf';fd=fitz.open(fp);page=fd[0]
        width=page.rect.width*2.54/72;require(abs(width-15.6)<.02,'Unexpected exported figure natural width')
        fonts=page.get_fonts(full=True);require(fonts,'Exported figure has no embedded text fonts')
        embedded=[]
        for font in fonts:
            extracted=fd.extract_font(font[0]);embedded.append(bool(extracted[3]))
        require(all(embedded),'An experimental figure contains an unembedded font')
        fontnames=' '.join(x[3]for x in fonts).lower();require('timesnewroman'in fontnames and 'simsun'in fontnames,'Required English/Chinese figure fonts missing')
        figurechecks.append(dict(number=item[0],width_cm=width,font_names=[x[3]for x in fonts],all_fonts_embedded=True,pdf_sha256=digest(fp)))
    result=dict(status='PASS',experimental_reader_pages=[start+1,end+1],rendered_neighbor_pages=render,figure_checks=figurechecks,displayed_numeric_checks=len(expected),manual_visual_review='pending',automated_render_and_text_check_only=True)
    write(stage/('pdf_checks.json'if role=='final'else'pdf_checks_'+role+'.json'),result);return result

def publish(stage,audit,thesis):
    state=None
    try:
        state=integrate(stage,audit,thesis)
        logs=actual_builds(stage,thesis)
        # Confirm another writer did not overwrite our owned chapter after the build.
        require(all(digest(p)==h for p,h in state['own_file_sha256'].items()),'Owned C4 source changed during build')
        gate()
        pdf=thesis/'build/main.pdf';final=SCOPE/'publication/final';final.mkdir(exist_ok=True)
        with lock(thesis/'build/.whole_thesis_build.lock'):
            # No nested build call while this lock is held. Other chapters may
            # validly publish before or after this snapshot is copied.
            pdfcheck=inspect_pdf(stage,thesis)
            shutil.copyfile(pdf,final/'main.pdf');require(digest(pdf)==digest(final/'main.pdf'),'Concurrent PDF changed while copying')
        receipt=dict(status='PUBLISHED_AUTOMATED_CHECKS_PASS',published=True,**{k:audit[k]for k in ('plan_sha256','acceptance_sha256','summary_sha256')},
            pdf_path=str(final/'main.pdf'),pdf_sha256=digest(final/'main.pdf'),live_main_pdf=str(pdf),whole_thesis_build_passes=5,
            manual_visual_review='pending',manual_visual_review_claimed=False,stage=str(stage),build_logs=logs,
            scientific_environment_steps=0,source_sha256=source_hashes(),owned_published_sha256=state['own_file_sha256'],published_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pdf_checks=str(stage/'pdf_checks.json'))
        doc=thesis/'docs/chapter4_double_random_formal_20261107';doc.mkdir(exist_ok=True)
        write(doc/'publication_receipt.json',receipt)
        write(SCOPE/'publication/FINAL_THESIS_PUBLICATION.json',receipt)
        return receipt
    except BaseException:
        original_error=traceback.format_exc()
        # Observe rather than roll back. A copy may have failed halfway, and
        # independent writers may validly be updating other chapters.
        changed=[];inspection_errors=[]
        for path,oldsha in audit['old_owned_sha256'].items():
            try:
                p=Path(path)
                if not p.exists()or digest(p)!=oldsha:changed.append(str(p))
            except Exception as exc:inspection_errors.append(dict(path=str(path),error=repr(exc)))
        created=[]
        for item in audit.get('figures',[]):
            p=thesis/item['path']
            if p.exists():created.append(str(p))
        write(SCOPE/'publication/PUBLICATION_FAILED.json',dict(status='FAILED_NOT_PUBLISHED',stage=str(stage),error=original_error,published=False,
            new_sources_integrated=bool(changed or created),integration_completed=state is not None,
            observed_changed_C4_paths=changed,observed_new_figure_directories=created,inspection_errors=inspection_errors,
            observed_changes_do_not_assert_exclusive_authorship=True,old_assets_preserved=str(stage/'backup'),manual_visual_review='pending',
            recover_instruction='Inspect this stage and failure before re-running; preserve backups and do not roll back concurrent chapters.'))
        raise

def verify_existing_receipt(receipt,bindings):
    """An existing file is never a substitute for current scientific gates."""
    require(receipt.get('status')=='PUBLISHED_AUTOMATED_CHECKS_PASS'and receipt.get('published')is True,'Existing publication receipt is partial, stale, or failed')
    for key in ('plan_sha256','acceptance_sha256','summary_sha256'):
        require(receipt.get(key)==bindings[key],'Existing publication receipt '+key+' no longer matches current accepted cohort')
    immutable=SCOPE/'publication/final/main.pdf'
    require(receipt.get('pdf_path')==str(immutable),'Existing publication receipt must bind the registered immutable PDF')
    require(immutable.is_file()and digest(immutable)==receipt.get('pdf_sha256'),'Existing immutable PDF does not match publication receipt')
    require(receipt.get('whole_thesis_build_passes')==5,'Existing publication receipt lacks five actual build passes')

def self_test():
    # These are private synthetic arithmetic tests, never thesis or formal results.
    import tempfile
    p=SCOPE/'preflight/publisher_tests';p.mkdir(exist_ok=True)
    checks=[]
    require(content.difference(-.4)=='低于0.40分','Negative evidence phrasing failed');checks.append('negative results stay negative')
    require(content.difference(0)=='差值不足0.01分','Tie wording failed');checks.append('tie not called improvement')
    require(content.pm(dict(n=5,mean=.5,sample_sd=.1),100)==r'\(50.00\pm10.00\)','Display100 scale test');checks.append('native scale exactly once')
    mapping,newbib=citations.resolve(THESIS);require(len(mapping)==5,'Primary citation resolution');checks.append('canonical literature keys resolve')
    c=THESIS/C4;_,patches=contract_patches.prepare(*[(c/n).read_text()for n in ('chapter.tex','section51_problem.tex','section52_method.tex')]);require(len(patches)in(0,6),'Full-spend spans or verified already-synchronized contract');checks.append('all six guarded contract spans locate exactly once')
    try:contract_patches.replace_once('different','old','new','unit test',[])
    except ValueError:checks.append('drifting manuscript guard rejects')
    else:raise ValueError('Guard allowed changed input')
    readonly_figures_audit(THESIS);checks.append('current three drawio figures lack obsolete cap labels')
    for f in source_hashes():ast.parse(Path(f).read_text())
    checks.append('all publisher modules parse')
    if not(SCOPE/'independent/acceptance.json').exists():
        try:gate()
        except ValueError:checks.append('partial cohort rejected before staging/live writes')
        else:raise ValueError('Partial scientific cohort accepted')
    write(p/'selftest.json',dict(status='PASS',synthetic_only=True,new_environment_steps=0,live_thesis_modified=False,checks=checks,source_sha256=source_hashes()))
    print('Publisher self-test PASS:',p/'selftest.json')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scope',type=Path,default=SCOPE);parser.add_argument('--thesis-root',type=Path,default=THESIS);parser.add_argument('--prepare',action='store_true');parser.add_argument('--publish',action='store_true');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    require(args.scope.resolve()==SCOPE,'Publisher bound to registered scope')
    require(args.thesis_root.resolve()==THESIS,'Publisher restricted to authorized current thesis')
    if args.self_test:self_test();return 0
    with lock(SCOPE/'publication/.publisher.lock'):
        final=SCOPE/'publication/FINAL_THESIS_PUBLICATION.json'
        if final.exists():
            _,_,bindings=gate()
            verify_existing_receipt(read(final),bindings)
            print('Already published',final);return 0
        s,data,binding=gate();stage,audit=prepare_stage(s,data,binding,THESIS)
        if args.publish:
            result=publish(stage,audit,THESIS);print(result['status'],SCOPE/'publication/FINAL_THESIS_PUBLICATION.json')
        else:print('PREPARED_NOT_PUBLISHED',stage)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print('REFUSED_OR_FAILED:',str(exc),file=sys.stderr);raise
