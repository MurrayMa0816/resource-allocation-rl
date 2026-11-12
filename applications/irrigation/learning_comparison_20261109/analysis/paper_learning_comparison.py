"""C6 paper learning figure from nine sealed26-point DEV records only.

No model/environment/scientific runtime import. No TeX/index/build/publication
pointer operation. All generated data and figure assets remain on HY.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,os
from pathlib import Path
import numpy as np

STUDY=Path('/hy-tmp/resource_allocation_results/chapter06/experiments/reasonable_allocation89_20261109')
EXTERNAL=Path('/hy-tmp/resource_allocation_results/chapter06/experiments/external_learning_comparison_20261109')
REVISION=Path('/hy-tmp/resource_allocation_results/chapter06/paper_revisions/learning_comparison_publication_20261109_230111')
THESIS=Path('/root/gamewar/latex_thesis_resource_allocation')
STYLE=THESIS/'docs/main_learning_style_20261109/PRESENTATION.json'
STYLE_GEOMETRY=Path('/hy-tmp/resource_allocation_results/thesis_main_learning_style_20261109/assets/figure5-4/render_geometry.json')
SEEDS=(6400,6401,6402)
METHODS=('SeDVal','PASTA','COLA')
COLOURS={'SeDVal':'#154f80','PASTA':'#498f9d','COLA':'#9368a6'}
MARKERS={'SeDVal':'o','PASTA':'P','COLA':'D'}
WIDTH,HEIGHT=156.,81.


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_csv(path,rows):
    rows=list(rows)
    with Path(path).open('w',newline='',encoding='utf-8')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def generate(out):
    out=Path(out).resolve()
    if not out.is_relative_to(REVISION)or not out.is_relative_to('/hy-tmp'):
        raise ValueError('Only the new HY paper revision may receive output')
    out.mkdir(parents=True,exist_ok=True)
    inputs={}
    def bind(p):
        p=Path(p);inputs[str(p)]=sha(p);return p
    bind(STYLE);style=json.loads(bind(STYLE_GEOMETRY).read_text())
    assert style['figure_width_mm']==156 and style['all_data_linewidths_bp']==[1.25]
    expected_palette={x['label']:x for x in style['lines']if x['axis']==0 and x['main_curve']}
    for method,label in [('SeDVal','SeDVal（本文）'),('PASTA','PASTA'),('COLA','COLA')]:
        assert expected_palette[label]['color']==COLOURS[method]
        assert expected_palette[label]['marker']==MARKERS[method]
    audit_paths=[EXTERNAL/'audit/final_training_audit.json',EXTERNAL/'additional_cola_20261109/audit/final_training_audit.json']
    for p in audit_paths:
        audit=json.loads(bind(p).read_text())
        if audit['status']!='PASS_FIXED_TRAINING_COMPLETION_AND_STORED_DEV_NUMERICS':
            raise ValueError('Accepted stored-DEV numerical audit is required')
    fixed=json.loads(bind(STUDY/'formal/fixed_inputs89.json').read_text())
    area=np.asarray(fixed['area_ha']);ids=np.asarray(fixed['region_ids'])
    steps=np.arange(0,259201,10368);assert len(steps)==26
    cube={method:[]for method in METHODS};raw=[];cases=[];sources=[];weather_reference=None;forcing_reference=None
    max_error={'C':0.,'JA':0.}
    for method in METHODS:
        for seed in SEEDS:
            if method=='SeDVal':folder=STUDY/f'formal/training/budget_4000000__seed{seed}/attempt0'
            elif method=='PASTA':folder=EXTERNAL/f'formal/pasta__seed{seed}/attempt0'
            else:folder=EXTERNAL/f'additional_cola_20261109/formal/cola__seed{seed}/attempt0'
            path=bind(folder/'development.npz')
            with np.load(path,allow_pickle=False)as z:
                data={k:z[k].copy()for k in z.files}
            x=data['episode']*81 if method=='SeDVal'else data['learning_steps']
            np.testing.assert_array_equal(np.unique(x),steps)
            assert data['quality'].shape==(832,89)
            if method=='SeDVal':
                native=list(data['metric_names']);C=data['metrics'][:,native.index('C')];JA=data['metrics'][:,native.index('JA')]
                records={}
                for step in steps:
                    ep=int(step//81);p=bind(folder/f'development_{ep:04}.json');r=json.loads(p.read_text())
                    assert r['counts']['train']['physical_steps']==int(step)and r['episode']==ep and r['paths']==32
                    assert r['learner_unchanged']is True and r['model_selection']is False
                    records[int(step)]=r['means']
                source_role='existing original three SeDVal training jobs; no retraining'
            else:
                C,JA=data['C'],data['JA'];receipt=json.loads(bind(folder/'completed.json').read_text())
                assert receipt['status']=='COMPLETE'and receipt['learning_steps']==259200 and receipt['episodes']==3200
                records={}
                with gzip.open(bind(folder/'development.jsonl.gz'),'rt')as f:
                    for line in f:
                        r=json.loads(line);assert r['rng_restored']is True and r['learning_state_unchanged']is True
                        assert r['counts']['train']['physical_steps']==r['learning_steps']
                        records[int(r['learning_steps'])]={'C':r['mean_C'],'JA':r['mean_JA']}
                assert sorted(records)==steps.tolist()
                source_role='completed fixed endpoint new C6 training; DEV only'
            expected_C=(data['quality']>=.9).mean(1)
            expected_JA=np.einsum('ij,j->i',data['quality'],area)/area.sum()
            for metric,v,e in [('C',C,expected_C),('JA',JA,expected_JA)]:
                err=float(np.max(abs(v-e)));max_error[metric]=max(max_error[metric],err)
                np.testing.assert_allclose(v,e,rtol=0,atol=1e-12)
            if method!='SeDVal':
                np.testing.assert_array_equal(data['area_ha'],area);np.testing.assert_array_equal(data['region_ids'],ids)
            points=[]
            for step in steps:
                mask=x==step;assert mask.sum()==32
                order=np.argsort(data['weather_seed'][mask]);weather=data['weather_seed'][mask][order]
                forcing=data['weather_trace'][mask][order]
                if weather_reference is None:weather_reference=weather;forcing_reference=forcing
                np.testing.assert_array_equal(weather,weather_reference)
                np.testing.assert_array_equal(forcing,forcing_reference)
                means=np.array([float(C[mask].mean()),float(JA[mask].mean())])
                for metric,value in zip(('C','JA'),means):
                    np.testing.assert_allclose(value,records[int(step)][metric],rtol=0,atol=1e-12)
                    raw.append(dict(method=method,training_seed=seed,training_steps=int(step),completed_episodes=int(step//81),metric=metric,weather_count=32,weather_mean_percent=float(value*100)))
                points.append(means*100)
                for index in np.flatnonzero(mask):
                    cases.append(dict(method=method,training_seed=seed,training_steps=int(step),weather_seed=int(data['weather_seed'][index]),C_fraction=float(C[index]),JA_fraction=float(JA[index])))
            cube[method].append(points)
            sources.append(dict(method=method,training_seed=seed,source=str(path),sha256=sha(path),role=source_role,DEV_points=26,DEV_cases=832))
    arrays={m:np.asarray(v)for m,v in cube.items()}
    means={m:values.mean(0)for m,values in arrays.items()};sds={m:values.std(0,ddof=1)for m,values in arrays.items()}
    summaries=[]
    for m in METHODS:
        assert arrays[m].shape==(3,26,2)
        for j,step in enumerate(steps):
            for k,metric in enumerate(('C','JA')):
                summaries.append(dict(method=m,training_steps=int(step),x_thousand_steps=float(step/1000),metric=metric,mean_percent=float(means[m][j,k]),sample_SD_percent=float(sds[m][j,k]),lower_percent=float(means[m][j,k]-sds[m][j,k]),upper_percent=float(means[m][j,k]+sds[m][j,k]),training_count=3,DEV_weather_per_training=32))
    write_csv(out/'learning_all3_seed_DEV.csv',raw)
    write_csv(out/'learning_mean_sample_SD.csv',summaries)
    write_csv(out/'all7488_DEV_cases.csv',cases)
    os.environ['MPLCONFIGDIR']=str(REVISION/'runtime_cache/paper_learning_matplotlib')
    os.environ['XDG_CACHE_HOME']=str(REVISION/'runtime_cache/paper_learning_xdg')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.text import Text
    fontroot=THESIS/'assets/fonts/windows'
    for name in('simsun.ttc','TIMES.TTF'):
        font_manager.fontManager.addfont(str(bind(fontroot/name)))
    plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':10.5,'axes.labelsize':10.5,'axes.titlesize':10.5,'xtick.labelsize':10.5,'ytick.labelsize':10.5,'legend.fontsize':10.5,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'c6-three-learning-comparison-v1','path.simplify':False,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(WIDTH/25.4,HEIGHT/25.4),dpi=200)
    axes=[fig.add_axes([17/WIDTH,22/HEIGHT,59.2/WIDTH,51/HEIGHT]),fig.add_axes([94/WIDTH,22/HEIGHT,59.2/WIDTH,51/HEIGHT])]
    styles=[]
    for k,(ax,title,ylabel)in enumerate(zip(axes,['(a) 区域产量达标率','(b) 平均相对产量'],['区域产量达标率（%）','平均相对产量（%）'])):
        for i,method in enumerate(METHODS):
            mean=means[method][:,k];sd=sds[method][:,k]
            assert np.min(mean-sd)>=0 and np.max(mean+sd)<=100
            ax.fill_between(steps/1000,mean-sd,mean+sd,color=COLOURS[method],alpha=.18,linewidth=0)
            markers=sorted(set(np.linspace(i%3,25,12,dtype=int).tolist()))
            line=ax.plot(steps/1000,mean,color=COLOURS[method],label=method,lw=1.25,ls='-',marker=MARKERS[method],ms=4.,mec=COLOURS[method],mfc=COLOURS[method],mew=.6,markevery=markers,clip_on=False)[0]
            styles.append(dict(axis=k,method=method,color=COLOURS[method],linestyle=line.get_linestyle(),linewidth_bp=line.get_linewidth(),marker=line.get_marker(),marker_size_bp=line.get_markersize(),marked_indices=markers,true_mean_vertices=len(line.get_xdata()),full_SD_band=True))
        ax.set(xlim=(0,259.2),ylim=(0,100));ax.set_xticks([0,50,100,150,200,250]);ax.set_yticks([0,20,40,60,80,100])
        ax.set_xlabel('训练步数（×10³）',labelpad=4.)
        ax.set_ylabel(ylabel,labelpad=4.)
        ax.set_title(title,loc='center',fontsize=10.5,fontweight='normal',pad=6.)
        ax.tick_params(axis='both',which='major',length=3.,width=.7,pad=3.,labelsize=10.5,direction='out')
        ax.grid(axis='y',alpha=.18,linewidth=.6)
    handles,labels=axes[0].get_legend_handles_labels()
    legend=fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,2/HEIGHT),ncol=3,frameon=False,handlelength=1.8,handletextpad=.5,columnspacing=1.25,labelspacing=.5,borderpad=0,borderaxespad=0)
    fig.canvas.draw();ren=fig.canvas.get_renderer();factor=72/fig.dpi
    text=[]
    for t in fig.findobj(Text):
        if not t.get_visible()or not t.get_text():continue
        bb=t.get_window_extent(ren);assert abs(t.get_fontsize()-10.5)<1e-8
        assert bb.x0>=-.5 and bb.y0>=-.5 and bb.x1<=fig.bbox.x1+.5 and bb.y1<=fig.bbox.y1+.5
        text.append(dict(text=t.get_text(),size_bp=t.get_fontsize(),bbox_bp=(np.array(bb.bounds)*factor).tolist()))
    title_checks=[]
    for ax in axes:
        box=ax.title.get_window_extent(ren);error=abs((box.x0+box.x1)/2-(ax.bbox.x0+ax.bbox.x1)/2);assert error<.5
        title_checks.append(dict(title=ax.get_title(),centre_error_pixels=float(error)))
    lb=legend.get_window_extent(ren);xlabel_bottom=min(ax.xaxis.label.get_window_extent(ren).y0 for ax in axes);gap=(xlabel_bottom-lb.y1)*factor
    assert gap>=3.
    geometry=dict(status='PASS',figure_mm=[WIDTH,HEIGHT],axes_mm=[59.2,51.],axes_positions_mm=[(np.asarray(ax.get_position().bounds)*[WIDTH,HEIGHT,WIDTH,HEIGHT]).tolist()for ax in axes],ordinary_text_bp=10.5,text=text,title_checks=title_checks,title_pad_bp=6.,axis_label_pad_bp=4.,xlabel_to_legend_bp=float(gap),legend_bottom_bp=float(lb.y0*factor),legend_labels=labels,legend_columns=3,legend_rows=1,styles=styles,all_data_lines_solid=True,all_data_linewidths_bp=[1.25],all26_real_vertices_preserved=True,no_smoothing=True,SD_full_unclipped=True)
    for ext in('pdf','svg','png'):
        kwargs={'dpi':260}if ext=='png'else{}
        if ext=='pdf':kwargs['metadata']=dict(CreationDate=None,ModDate=None,Creator='C6 sealed-DEV learning comparison plot')
        elif ext=='svg':kwargs['metadata']={'Date':None}
        fig.savefig(out/f'figure.{ext}',**kwargs)
    plt.close(fig)
    dump(out/'render_geometry.json',geometry)
    assert {p:sha(p)for p in inputs}==inputs
    source_path=Path(__file__).resolve()
    snapshot=REVISION/'source/paper_learning_comparison_final.py';snapshot.parent.mkdir(parents=True,exist_ok=True)
    if source_path!=snapshot:
        if snapshot.exists()and snapshot.read_bytes()!=source_path.read_bytes():raise ValueError('Existing immutable plot snapshot differs')
        if not snapshot.exists():snapshot.write_bytes(source_path.read_bytes())
    outputs={name:sha(out/name)for name in ['figure.pdf','figure.svg','figure.png','learning_all3_seed_DEV.csv','learning_mean_sample_SD.csv','all7488_DEV_cases.csv','render_geometry.json']}
    provenance=dict(status='CANDIDATE_REQUIRES_ACTUAL_PDF_VISUAL_REVIEW',number='6.5',maintenance_source=str(source_path),maintenance_source_sha256=sha(source_path),snapshot_source=str(snapshot),snapshot_source_sha256=sha(snapshot),input_files_sha256=inputs,inputs_unchanged=True,methods=list(METHODS),training_seeds=list(SEEDS),sources=sources,statistics='At each recorded point first mean32fixedDEVweather within each training, then mean±sampleSD(ddof1)of3independent training means',DEV_points_per_training=26,total_DEV_cases=7488,x_axis='Recorded cumulative training physical steps/1000; DEV and optimizer steps excluded',x_steps=steps.tolist(),no_smoothing=True,no_bestDEV_selection=True,full_SD_unclipped=True,original_SeDVal_retrained=False,max_independent_metric_error=max_error,same32_DEV_weather_and_forcing_all9jobs=True,C_is_region_fraction_ge_point9=True,JA_is_area_weighted=True,endpoints={method:{metric:dict(mean_percent=float(means[method][-1,k]),sample_SD_percent=float(sds[method][-1,k]))for k,metric in enumerate(('C','JA'))}for method in METHODS},style_reference=str(STYLE),geometry_reference=str(STYLE_GEOMETRY),outputs_sha256=outputs,scientific_execution=dict(environment_steps=0,model_loads=0,policy_inference=0,training_runs=0,new_TEST_paths=0),TeX_edits=0,whole_thesis_builds=0,current_figure_or_pointer_modified=False)
    dump(out/'source.json',provenance)
    (out/'plot.py').write_text('from pathlib import Path\nimport runpy,sys\nscript=Path('+repr(str(snapshot))+')\nsys.argv=[str(script),"--output-dir",str(Path(__file__).resolve().parent)]+sys.argv[1:]\nrunpy.run_path(str(script),run_name="__main__")\n')
    generation=dict(geometry)
    generation.update(status='GENERATED',source_json=str(out/'source.json'),source_json_sha256=sha(out/'source.json'),scientific_execution=provenance['scientific_execution'])
    dump(REVISION/'audit/learning_figure_generation.json',generation)
    print(json.dumps(dict(status='GENERATED',output=str(out),figure_mm=[WIDTH,HEIGHT],all26points=True,methods=METHODS),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=REVISION/'figures/learning_comparison');args=parser.parse_args()
    generate(args.output_dir)


if __name__=='__main__':main()
