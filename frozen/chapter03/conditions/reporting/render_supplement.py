"""Validate the registered C3 supplement and render real results only.

CLI: python render_supplement.py --root ROOT --output ROOT/report
No environment, agent, network, or learner is imported or executed.
"""
from __future__ import annotations
import argparse, csv, datetime, gzip, hashlib, json, math, os, sys
sys.dont_write_bytecode = True
from collections import defaultdict
from pathlib import Path
import numpy as np

DOMAINS=('radio','v5'); SEEDS=tuple(range(5)); HORIZON={'radio':32,'v5':24}
DIRECT='A_DIRECT_TOTAL'; CORE=('A_FULL','A_PACE',DIRECT,'A_BUDGET_ONLY')
STRONG={'radio':'c_ppo','v5':'gradient_ppo'}
CONDITIONS={'radio':('default','budget08','budget12','persistence02','persistence095'),
            'v5':('default','budget08','budget12','arrivals_front','arrivals_middle')}
LABEL={'A_FULL':'ReJoP','A_PACE':'无学习总量修正',DIRECT:'直接总量映射','A_BUDGET_ONLY':'无任务摘要','c_ppo':'C-PPO','gradient_ppo':'Gradient PPO'}
COLOR={'A_FULL':'#cc4b37','A_PACE':'#657fa9',DIRECT:'#be8c28','A_BUDGET_ONLY':'#6a9b68','c_ppo':'#855f9e','gradient_ppo':'#855f9e'}
TITLE={'radio':'图像通信','v5':'应急电力'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read_csv(path):
 opener=gzip.open if str(path).endswith('.gz') else open
 with opener(path,'rt',newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows,fields=None):
 rows=list(rows);fields=fields or list(rows[0]) if rows else fields or []
 opener=gzip.open if str(path).endswith('.gz') else open
 with opener(path,'wt',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def expected_eval_keys():
 keys=set()
 for d in DOMAINS:
  for s in SEEDS:
   for m in (*CORE,STRONG[d]):keys.add((m,d,s,'default'))
   for c in ('budget08','budget12'):
    for m in ('A_FULL','A_PACE',DIRECT,STRONG[d]):keys.add((m,d,s,c))
   for c in CONDITIONS[d][3:]:
    for m in ('A_FULL','A_PACE','A_BUDGET_ONLY',STRONG[d]):keys.add((m,d,s,c))
 assert len(keys)==210
 return keys

def input_manifest_path(root):
 p=root/'protocol/input_models_manifest.json'
 return p if p.is_file() else Path(__file__).parent/'input_models_manifest.json'

def resolve_output(root,job):
 p=Path(job.get('output',root/'jobs'/job.get('id',job.get('job_id',''))))
 return p if p.is_absolute() else root/p

def completion_audit(folder,job):
 p=folder/'completed.json';c=read_json(p)
 assert c.get('success') is True,(str(folder),'not successful')
 assert not c.get('engineering_gate',False),(str(folder),'engineering cannot be formal')
 assert c['method']==job['method'] and c['domain']==job['domain'] and int(c['seed'])==int(job['seed']),(str(folder),'identity')
 if (folder/'failed.json').exists():raise AssertionError((str(folder),'failed.json conflicts with completion'))
 artifacts=c.get('artifacts_sha256',{})
 assert artifacts,(str(folder),'missing artifact digests')
 verified={}
 for rel,digest in artifacts.items():
  x=(folder/rel).resolve();assert x.is_relative_to(folder.resolve()),(str(folder),rel)
  assert x.is_file() and sha(x)==digest,(str(folder),rel,'artifact digest')
  verified[rel]=digest
 if c.get('protocol_sha256'):assert sha(folder/'protocol.json')==c['protocol_sha256']
 pmeta=read_json(folder/'protocol.json')
 return c,pmeta,{'completion_path':str(p),'completion_sha256':sha(p),'protocol_path':str(folder/'protocol.json'),'protocol_sha256':sha(folder/'protocol.json'),'verified_artifacts':verified}

def validate(root,plan):
 jobs=plan['jobs']; assert len(jobs)==220,('220 registered jobs required',len(jobs))
 trainjobs=[j for j in jobs if j['kind']=='train_direct'];evaljobs=[j for j in jobs if j['kind']=='evaluate']
 assert len(trainjobs)==10 and len(evaljobs)==210,(len(trainjobs),len(evaljobs))
 assert {(j['method'],j['domain'],int(j['seed'])) for j in trainjobs}=={(DIRECT,d,s) for d in DOMAINS for s in SEEDS}
 keys={(j['method'],j['domain'],int(j['seed']),j['condition']) for j in evaljobs}
 assert len(keys)==len(evaljobs) and keys==expected_eval_keys(),('wrong evaluation design',sorted(expected_eval_keys()-keys), sorted(keys-expected_eval_keys()))
 inputs=read_json(input_manifest_path(root))
 old={(m['method'],m['domain'],m['seed']):m for m in inputs['models']}
 provenance=[];allpaths=[];seedrows=[];stepfiles={};trainfolders={};models_before={}
 for j in trainjobs:
  folder=resolve_output(root,j);c,pmeta,pv=completion_audit(folder,j)
  assert c['training_steps']==100608 and c['endpoint_reload_pass'] is True
  assert c['model_count']==1 and not c['optimizer_saved'] and not c['replay_saved'] and not c['periodic_models_saved']
  assert c['full_spend_required'] and c['maximum_resource_fraction_error']<=1e-9
  assert c.get('source_unchanged',True)
  assert pmeta['fresh_initialization'] and pmeta['arguments']['path_master']==202611043101
  assert pmeta['arguments']['final_master']==202611058199
  mp=folder/c.get('model','model/state.pt');assert sha(mp)==c['model_sha256']
  key=(DIRECT,j['domain'],int(j['seed']));models_before[key]=sha(mp);trainfolders[key]=folder
  pv.update(kind='train_direct',method=DIRECT,domain=j['domain'],seed=int(j['seed']),model_sha256=c['model_sha256']);provenance.append(pv)
 for j in evaljobs:
  folder=resolve_output(root,j);c,pmeta,pv=completion_audit(folder,j)
  method,domain,seed,condition=j['method'],j['domain'],int(j['seed']),j['condition'];h=HORIZON[domain]
  assert c['condition']==condition and c['paths']==128
  for gate in ('all_native_replays_exact','all_full_spend','policy_and_normalization_unchanged','source_unchanged'):
   assert c.get(gate) is True,(str(folder),gate)
  assert c['learning_environment_steps']==0
  assert c['evaluation_environment_steps']==128*h and c['native_replay_environment_steps']==128*h
  assert pmeta['path_master']==202611058199 and pmeta['paths']==128
  assert pmeta['weights_and_learned_moments_frozen'] and pmeta['no_future_policy_input']
  registered_run=Path(pmeta['run']);basekey=(method,domain,seed)
  assert str(registered_run)==j.get('model_run',str(registered_run)),(str(folder),'registered model run changed')
  run=registered_run
  if not (run/'protocol.json').is_file() and method!=DIRECT:
   run=Path(old[basekey]['local_run_dir'])
  source=read_json(run/'protocol.json');done=read_json(run/'completed.json')
  assert sha(run/'protocol.json')==pmeta['training_protocol_sha256'] and sha(run/'completed.json')==pmeta['training_completion_sha256']
  basekey=(method,domain,seed)
  if method==DIRECT:
   assert run.resolve()==trainfolders[basekey].resolve();mp=run/'model/state.pt';assert sha(mp)==models_before[basekey]
  else:
   reg=old[basekey];mp=run/'model'/Path(reg['remote_model_path']).name
   assert sha(mp)==reg['model_sha256'];models_before[basekey]=sha(mp)
  identity=source.get('source_sha256',source.get('source_identity',{}).get('package',{}))
  assert pmeta['training_source_identity']==identity,(str(folder),'source closure mismatch')
  ep=read_csv(folder/'evaluation_paths.csv.gz');steps=read_csv(folder/'evaluation_steps.csv.gz')
  assert len(ep)==128 and {int(r['path_index']) for r in ep}==set(range(128))
  assert len(steps)==128*h
  stepgroups=defaultdict(list)
  for r in steps:stepgroups[int(r['path_index'])].append(r)
  for r in ep:
   pi=int(r['path_index']);pathseed=int(r['path_seed']);nr=float(r['native_return']);fraction=float(r['resource_fraction']);replay=float(r['replay_error'])
   assert math.isfinite(nr) and abs(fraction-1)<=1e-9 and replay==0 and int(r['steps'])==h
   sr=stepgroups[pi];assert [int(x['step']) for x in sr]==list(range(h))
   B0=float(sr[0]['R_before']);tol=max(1e-12,B0*1e-9)
   assert B0>0
   for t,x in enumerate(sr):
    q,R,Rnext,L,U=(float(x[k]) for k in ('q','R_before','R_after','L','U'))
    assert int(x['path_seed'])==pathseed and int(x['feasible'])==1
    assert L-tol<=q<=U+tol and abs(R-q-Rnext)<=tol
    if t:assert abs(R-float(sr[t-1]['R_after']))<=tol
   assert abs(float(sr[-1]['R_after']))<=tol and abs(sum(float(x['q']) for x in sr)/B0-1)<=1e-9
   if condition in ('arrivals_front','arrivals_middle'):
    assert int(r['exact_event_multiset_preserved'])==1 and float(r['stream_total_max_error'])<=1e-8
   norm=dict(method=method,domain=domain,seed=seed,condition=condition,path_index=pi,path_seed=pathseed,native_return=nr,score_x100=100*nr,resource_fraction=fraction,replay_error=replay)
   for name in ('last_quarter_resource_fraction','forced_steps','lower_hits_nonforced','upper_hits_nonforced','mean_confirmed_packet_fraction','total_demand','total_completed','total_ineffective','ineffective_fraction'):
    norm[name]=float(r[name]) if r.get(name,'')!='' else ''
   allpaths.append(norm)
  values=np.array([float(x['native_return']) for x in ep]);assert np.isclose(values.mean(),c['native_mean'],atol=1e-13,rtol=0)
  seedrow=dict(method=method,domain=domain,seed=seed,condition=condition,paths=128,training_steps=100608,native_mean=float(values.mean()),score_mean=float(values.mean()*100),resource_error_max=max(abs(float(x['resource_fraction'])-1) for x in ep),replay_error_max=max(abs(float(x['replay_error'])) for x in ep))
  for name in ('last_quarter_resource_fraction','forced_steps','lower_hits_nonforced','upper_hits_nonforced','mean_confirmed_packet_fraction','ineffective_fraction'):
   vv=[float(x[name]) for x in ep if x.get(name,'')!=''];seedrow[name]=float(np.mean(vv)) if vv else ''
  seedrows.append(seedrow)
  stepfiles[(method,domain,seed,condition)]=folder/'evaluation_steps.csv.gz'
  pv.update(kind='evaluate',method=method,domain=domain,seed=seed,condition=condition,model_sha256=sha(mp));provenance.append(pv)
 # Public final task/image paths are identical across policy and training seed.
 pairing=defaultdict(set)
 for r in allpaths:pairing[(r['domain'],r['condition'],r['path_index'])].add(r['path_seed'])
 assert all(len(v)==1 for v in pairing.values()),'Common final scenario pairing failed'
 # Parameter changes retain the same base scenario seeds; do not infer equality of transformed outcomes.
 basepair=defaultdict(set)
 for r in allpaths:basepair[(r['domain'],r['path_index'])].add(r['path_seed'])
 assert all(len(v)==1 for v in basepair.values()),'Condition base scenario pairing failed'
 assert len(allpaths)==26880 and len(seedrows)==210
 return seedrows,allpaths,provenance,stepfiles,trainfolders

def summaries(seedrows):
 grouped=defaultdict(list)
 for r in seedrows:grouped[(r['method'],r['domain'],r['condition'])].append(r)
 out=[]
 for (m,d,c),rr in sorted(grouped.items()):
  assert sorted(x['seed'] for x in rr)==list(SEEDS)
  scores=np.array([x['score_mean'] for x in rr]);row=dict(method=m,domain=d,condition=c,training_seeds=5,paths_per_model=128,mean_score=float(scores.mean()),sample_sd=float(scores.std(ddof=1)),min_seed=float(scores.min()),max_seed=float(scores.max()))
  for metric in ('last_quarter_resource_fraction','forced_steps','lower_hits_nonforced','upper_hits_nonforced','mean_confirmed_packet_fraction','ineffective_fraction'):
   vals=[r[metric] for r in rr if r[metric]!=''];row[metric]=float(np.mean(vals)) if vals else ''
  out.append(row)
 return out

def differences(seedrows):
 bykey={(x['method'],x['domain'],x['seed'],x['condition']):x for x in seedrows};out=[]
 for d in DOMAINS:
  for c in CONDITIONS[d]:
   ms=[m for m in (*CORE,STRONG[d]) if (m,d,0,c) in bykey]
   for m in ms:
    if m=='A_FULL':continue
    delta=[bykey[('A_FULL',d,s,c)]['score_mean']-bykey[(m,d,s,c)]['score_mean'] for s in SEEDS]
    out.append(dict(domain=d,condition=c,contrast='A_FULL minus '+m,mean_difference=float(np.mean(delta)),sample_sd_difference=float(np.std(delta,ddof=1)),**{f'seed{s}_difference':float(delta[s]) for s in SEEDS},statistical_unit='five training-seed model means on matched 128 final paths; descriptive, no significance claim'))
 return out

def font_setup(output):
 os.environ['MPLCONFIGDIR']=str(output/'runtime/matplotlib');(output/'runtime/matplotlib').mkdir(parents=True,exist_ok=True)
 import matplotlib;matplotlib.use('Agg')
 from matplotlib import font_manager,rcParams
 fonts=Path('/root/gamewar/latex_thesis_resource_allocation/assets/fonts/windows')
 chinese=fonts/'simsun.ttc'
 if chinese.exists():font_manager.fontManager.addfont(str(chinese));family=font_manager.FontProperties(fname=str(chinese)).get_name()
 else:
  candidates=[Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')]
  p=next((p for p in candidates if p.exists()),None)
  if p is None:raise FileNotFoundError('CJK font required')
  font_manager.fontManager.addfont(str(p));family=font_manager.FontProperties(fname=str(p)).get_name()
 rcParams.update({'font.family':[family,'DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,'pdf.fonttype':42,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})

def plot_results(root,output,seedrows,means,stepfiles,trainfolders,input_manifest):
 import matplotlib.pyplot as plt
 from matplotlib.backends.backend_pdf import PdfPages
 idx={(x['method'],x['domain'],x['condition']):x for x in means}
 seedidx={(x['method'],x['domain'],x['seed'],x['condition']):x for x in seedrows}
 figs=[]
 fig,axs=plt.subplots(1,2,figsize=(13,4.7),constrained_layout=True)
 for ax,d in zip(axs,DOMAINS):
  for k,m in enumerate(CORE):
   v=idx[(m,d,'default')];ax.errorbar(k,v['mean_score'],yerr=v['sample_sd'],fmt='o',markersize=7,capsize=5,color=COLOR[m],lw=1.5)
   vals=[seedidx[(m,d,s,'default')]['score_mean'] for s in SEEDS]
   ax.scatter(k+np.linspace(-.1,.1,5),vals,s=19,color=COLOR[m],alpha=.55,zorder=3)
  ax.set_xticks(range(4),[LABEL[m] for m in CORE]);ax.tick_params(axis='x',labelrotation=15);ax.set_title(TITLE[d]);ax.set_ylabel('终测分数（原始回报 × 100）')
 fig.suptitle('核心组件对照：5个独立训练种子，每模型128条配对情景\n误差线为训练种子均值的样本标准差',fontsize=13)
 fig.savefig(output/'core_ablations.pdf');fig.savefig(output/'core_ablations.png',dpi=200);figs.append(fig)
 # Raw fixed-development curves remain separate from independent final tests.
 old={(m['method'],m['domain'],m['seed']):(Path(m['local_run_dir']) if (Path(m['local_run_dir'])/'evaluation_curve.csv.gz').is_file() else Path(m['remote_run_dir'])) for m in input_manifest['models']}
 fig,axs=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
 devraw=[]
 for ax,d in zip(axs,DOMAINS):
  for m in CORE:
   curve=defaultdict(list)
   for s in SEEDS:
    folder=trainfolders[(m,d,s)] if m==DIRECT else old[(m,d,s)]
    rows=read_csv(folder/'evaluation_curve.csv.gz')
    used=set()
    for r in rows:
     if r.get('phase',r.get('evaluation_kind')) not in ('dev','development'):continue
     t=int(r['training_step'])
     if t in used:continue
     used.add(t);score=float(r['native_mean'])*100;curve[t].append(score);devraw.append(dict(method=m,domain=d,seed=s,training_step=t,mean_score=score,paths=int(r['paths'])))
   assert all(len(v)==5 for v in curve.values()),(m,d,'unpaired development schedule')
   ts=sorted(curve);mu=np.array([np.mean(curve[t]) for t in ts]);sd=np.array([np.std(curve[t],ddof=1) for t in ts]);ax.plot(ts,mu,label=LABEL[m],color=COLOR[m],lw=1.5);ax.fill_between(ts,mu-sd,mu+sd,color=COLOR[m],alpha=.11)
  ax.set_title(TITLE[d]);ax.set_xlabel('真实训练环境步');ax.set_ylabel('固定开发情景平均分数');ax.legend(fontsize=9)
 fig.suptitle('核心组件真实开发曲线：不平滑，阴影为5个训练种子的样本标准差',fontsize=13)
 fig.savefig(output/'core_development_curves.pdf');fig.savefig(output/'core_development_curves.png',dpi=200);figs.append(fig)
 write_csv(output/'core_development_raw.csv.gz',devraw)
 fig,axs=plt.subplots(2,2,figsize=(13,9.2),constrained_layout=True)
 for row,d in enumerate(DOMAINS):
  ax=axs[row,0];conds=['budget08','default','budget12'];x=np.array([.8,1.,1.2]);ms=['A_FULL','A_PACE',DIRECT,STRONG[d]]
  for m in ms:
   v=[idx[(m,d,c)] for c in conds];ax.errorbar(x,[r['mean_score'] for r in v],yerr=[r['sample_sd'] for r in v],label=LABEL[m],color=COLOR[m],marker='o',capsize=3,lw=1.5)
  ax.set_title(TITLE[d]+'：初始预算变化');ax.set_xlabel('相对默认预算');ax.set_xticks(x);ax.set_ylabel('终测分数');ax.legend(fontsize=9)
  ax=axs[row,1];conds=(['persistence02','default','persistence095'] if d=='radio' else ['default','arrivals_front','arrivals_middle']);x=([.2,.8,.95] if d=='radio' else [0,1,2]);ms=['A_FULL','A_PACE','A_BUDGET_ONLY',STRONG[d]]
  for m in ms:
   v=[idx[(m,d,c)] for c in conds];ax.errorbar(x,[r['mean_score'] for r in v],yerr=[r['sample_sd'] for r in v],label=LABEL[m],color=COLOR[m],marker='o',capsize=3,lw=1.5)
  ax.set_title(TITLE[d]+('：干扰状态持续性' if d=='radio' else '：新增需求时序变化'));ax.set_xlabel('干扰状态保持概率' if d=='radio' else '事件重排条件');ax.set_xticks(x,[str(v) for v in x] if d=='radio' else ['原时序','前移','中段移动']);ax.set_ylabel('终测分数');ax.legend(fontsize=9)
 fig.suptitle('冻结策略条件测试：每点5个训练种子 × 128条配对情景\n误差线为训练种子均值的样本标准差；预算和时序分别改变',fontsize=13)
 fig.savefig(output/'conditions_four_panels.pdf');fig.savefig(output/'conditions_four_panels.png',dpi=200);figs.append(fig)
 fig,axs=plt.subplots(2,2,figsize=(13,8.5),constrained_layout=True)
 configrows=[]
 for row,d in enumerate(DOMAINS):
  h=HORIZON[d]
  for m in ['A_FULL','A_PACE',DIRECT,STRONG[d]]:
   rows=[r for r in read_csv(stepfiles[(m,d,0,'default')]) if int(r['path_index'])==0];B=float(rows[0]['R_before']);t=np.arange(1,h+1)
   axs[row,0].plot(t,[float(r['q'])/B for r in rows],label=LABEL[m],color=COLOR[m],lw=1.5)
   axs[row,1].plot(np.arange(h+1),[1]+[float(r['R_after'])/B for r in rows],label=LABEL[m],color=COLOR[m],lw=1.5)
   for r in rows:configrows.append(dict(method=m,domain=d,seed=0,condition='default',**r))
  axs[row,0].set_title(TITLE[d]+'：各期投入');axs[row,0].set_ylabel('本期投入 / 初始总资源');axs[row,1].set_title(TITLE[d]+'：剩余资源');axs[row,1].set_ylabel('剩余资源 / 初始总资源')
  for ax in axs[row]:ax.set_xlabel('决策期');ax.legend(fontsize=9)
 fig.suptitle('预先固定案例：seed 0、情景 0、默认条件\n同一外生情景下各方法的闭环过程；并非分岔后的相同状态动作比较',fontsize=13)
 fig.savefig(output/'allocation_process.pdf');fig.savefig(output/'allocation_process.png',dpi=200);figs.append(fig)
 write_csv(output/'allocation_preselected_case.csv.gz',configrows)
 with PdfPages(output/'第三章_总量映射与条件变化_补充实验报告.pdf') as pdf:
  for fig in figs:pdf.savefig(fig,bbox_inches='tight')
  fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.94,'实验身份与统计口径',fontsize=18)
  lines=['10个Direct新模型：双场景各5个训练种子，每模型100608个真实学习环境步。','210个冻结策略评价单元：预算变化80，时序/持续性变化80，默认方法并集50。','这210项冻结评价共26880回合、752640个策略执行步；原生重放及训练内部开发/终测另计。','所有注册模型、日志和协议经过SHA核对；每回合资源全额使用，全部原生重放逐值相同。','终点分数 = 原始回报×100。均值和样本标准差以5个训练种子的128情景平均值为单位。','128条情景用于匹配环境随机性，不等于128个独立训练重复；不把640条路径作独立模型。','C-PPO（通信）与Gradient PPO（应急）在条件实验前选定，不依据本轮条件结果换方法。','不因曲线未上升或某个消融无差别而隐去真实结果；不宣称全部算法已经完全收敛。','预算/时序分别改变，不代表全因子交互实验；结果是冻结策略的条件敏感性证据。','配置过程使用预先固定seed0/path0，各闭环状态会分岔，不能作同状态因果断言。','原NoBudget与ConstantReference结果保留，本轮核心机制图呈现Full/PACE/Direct/无任务摘要。']
  fig.text(.06,.86,'\n\n'.join(lines),fontsize=11,va='top');pdf.savefig(fig);plt.close(fig)
  for d in DOMAINS:
   fig=plt.figure(figsize=(11.7,8.3));ax=fig.add_axes([.04,.08,.92,.8]);ax.axis('off');rows=[]
   for c in CONDITIONS[d]:
    for m in (*CORE,STRONG[d]):
     if (m,d,c) in idx:
      r=idx[(m,d,c)];rows.append([c,LABEL[m],f"{r['mean_score']:.3f}",f"{r['sample_sd']:.3f}"])
   table=ax.table(cellText=rows,colLabels=['条件','方法','5种子均值','样本标准差'],loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.35)
   fig.suptitle(TITLE[d]+'：完整终测结果',fontsize=16);pdf.savefig(fig);plt.close(fig)
 for fig in figs:plt.close(fig)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
 root=Path(a.root).resolve();out=Path(a.output).resolve()
 assert root.is_relative_to(Path('/hy-tmp')) and out.is_relative_to(root),'All new report artifacts must be physically on HY in this run root'
 out.mkdir(parents=True,exist_ok=True)
 planpath=root/'plan.json';plan=read_json(planpath);seedrows,paths,provenance,stepfiles,trainfolders=validate(root,plan)
 audit=dict(schema='chapter3-supplement-independent-results-audit-v1',status='PASS',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_path=str(planpath),plan_sha256=sha(planpath),reporter_sha256=sha(__file__),training_jobs=10,evaluation_jobs=210,paths_per_unit=128,total_frozen_policy_evaluation_episodes=26880,frozen_policy_executed_evaluation_steps=752640,frozen_policy_native_replay_steps=752640,direct_internal_monitoring_and_test_evaluation_steps=sum(read_json(resolve_output(root,j)/'completed.json')['evaluation_steps'] for j in plan['jobs'] if j['kind']=='train_direct'),direct_internal_native_replay_steps=sum(read_json(resolve_output(root,j)/'completed.json')['native_replay_environment_steps'] for j in plan['jobs'] if j['kind']=='train_direct'),training_steps_per_direct_model=100608,total_direct_learning_steps=1006080,five_training_seeds=list(SEEDS),all_registered_inputs_and_artifact_hashes_verified=True,all_common_scenarios_paired=True,all_terminal_budgets_fully_used=True,all_native_replays_exact=True,no_model_or_environment_execution_by_reporter=True,provenance=provenance)
 write_json(out/'independent_results_audit.json',audit)
 if a.validate_only:print(json.dumps({'status':'PASS','audit':str(out/'independent_results_audit.json')}));return
 means=summaries(seedrows);diff=differences(seedrows)
 write_csv(out/'evaluation_paths_raw.csv.gz',paths);write_csv(out/'evaluation_seed_means.csv',seedrows);write_csv(out/'evaluation_means_sample_sd.csv',means);write_csv(out/'paired_seed_differences.csv',diff)
 font_setup(out);plot_results(root,out,seedrows,means,stepfiles,trainfolders,read_json(input_manifest_path(root)))
 generated={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()}
 write_json(out/'report_completion.json',dict(schema='chapter3-supplement-report-completion-v1',status='complete',publication_ready=True,scientific_audit_sha256=sha(out/'independent_results_audit.json'),artifacts_sha256=generated,reporter_sha256=sha(__file__),paper_not_modified=True))
 print(json.dumps({'status':'complete','output':str(out),'training_jobs':10,'evaluation_jobs':210,'pdf':str(out/'第三章_总量映射与条件变化_补充实验报告.pdf')},ensure_ascii=False))
if __name__=='__main__':main()
