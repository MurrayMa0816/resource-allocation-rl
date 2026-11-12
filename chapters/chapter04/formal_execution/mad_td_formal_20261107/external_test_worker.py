"""New-scope evaluation dispatcher; retained learner/inference code stays byte-exact.

Only output roots and the locations of identically copied evaluation helpers are
rebound in memory. All100 endpoints must be sealed before final Task construction.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,sys,time,traceback
from pathlib import Path
ROOT=Path('/hy-tmp/resource_allocation_results/chapter4/mad_td_formal_20261107')
SOURCE=ROOT/'testing_source'
sys.path.insert(0,str(ROOT/'execution'))
from endpoint_release import sha,source_bindings,validate_release
METHODS=('f1','no_history','no_offpolicy_correction','fixed_actor_lr','props','A_inherited')

def dump(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Only current HY formal scope output is allowed')
    path.parent.mkdir(parents=True,exist_ok=True)
    q=path.with_suffix(path.suffix+'.writing');q.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(path)

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def test_bindings(*,engineering=False):
    seal=ROOT/'protocol/test_execution_seal.json'
    if seal.exists():
        files=json.loads(seal.read_text())['files_sha256']
        if not files or any(sha(p)!=h for p,h in files.items()):raise RuntimeError('Independent TEST execution source identity differs')
        return files
    if not engineering:raise RuntimeError('Independent TEST execution seal has not been prepared')
    files=source_bindings(required=False)
    for p in list(SOURCE.rglob('*'))+[Path(__file__).resolve(),ROOT/'source/f1_extension.py']:
        if p.is_file()and'__pycache__'not in p.parts:files[str(p)]=sha(p)
    return files

def released_protocol(args):
    if args.phase!='final' or args.eval_master!=2026110694199 or args.eval_cases!=256:raise ValueError('Final TEST uses the predeclared256-case final panel')
    files=test_bindings()
    gate_path=ROOT/'preflight/test_execution_gate.json';gate=json.loads(gate_path.read_text())
    if gate.get('status')!='PASS'or gate.get('test_execution_seal_sha256')!=sha(ROOT/'protocol/test_execution_seal.json'):raise RuntimeError('Independent TEST execution approval has not passed')
    release=validate_release(args.method,args.domain,args.seed,args.model)
    return files,release

def run_f1(args):
    legacy=load(SOURCE/'f1_worker.py','_madtd_formal_retained_f1_evaluator')
    legacy.SCOPE=ROOT
    argv=['--method',args.method,'--domain',args.domain,'--seed',str(args.seed),'--steps','100608','--output',str(args.output),'--device','cpu',
          '--evaluate-model',str(args.model),'--eval-master',str(args.eval_master),'--eval-cases',str(args.eval_cases),
          '--environment-root',str(args.environment_root),'--evaluation-phase',args.phase]
    parsed=legacy.parser().parse_args(argv)
    parsed.engineering_gate=args.engineering_gate
    return legacy.evaluate_model(parsed)

def run_props(args,bindings):
    here=SOURCE/'props';sys.path.insert(0,str(here))
    support=load(here/'formal_support.py','formal_support')
    support.SCOPE=ROOT;support.HERE=here;support.DEFAULT_ENV=args.environment_root
    support.bindings=lambda: test_bindings(engineering=args.engineering_gate)
    original_module=support.module
    def io_bound_module(path,name):
        result=original_module(path,name)
        if Path(path).name=='native_loop.py':result.ROOT=SOURCE
        return result
    support.module=io_bound_module
    legacy=load(SOURCE/'props_worker.py','_madtd_formal_retained_props_evaluator')
    argv=['--domain',args.domain,'--seed',str(args.seed),'--steps','100608','--output',str(args.output),'--device','cpu',
          '--evaluate-model',str(args.model),'--eval-master',str(args.eval_master),'--eval-cases',str(args.eval_cases),
          '--environment-root',str(args.environment_root),'--eval-phase',args.phase]
    parsed=support.parser('props').parse_args(argv);parsed.engineering_gate=args.engineering_gate
    return legacy.main(parsed)

def run_A(args):
    # The retained A evaluator performs and charges a native action replay after
    # each independent trajectory. Its extra physical work remains explicit.
    if args.output.exists():args.output.rmdir()  # only a verified empty attempt
    legacy=load(SOURCE/'a_reference_evaluate.py','_madtd_formal_retained_A_evaluator')
    previous=sys.argv
    sys.argv=[str(SOURCE/'a_reference_evaluate.py'),'--domain',args.domain,'--seed',str(args.seed),'--model',str(args.model),
              '--master',str(args.eval_master),'--phase',args.phase,'--cases',str(args.eval_cases),'--output',str(args.output),
              '--environment-root',str(args.environment_root)]
    try:legacy.main()
    finally:sys.argv=previous
    return json.loads((args.output/'completed.json').read_text())

def main(args):
    args.output=Path(args.output).resolve();args.model=Path(args.model).resolve();args.environment_root=Path(args.environment_root).resolve()
    if not args.output.is_relative_to(ROOT)or(args.output.exists()and any(args.output.iterdir())):raise ValueError('Use a fresh output attempt under current HY scope')
    if args.environment_root!=(ROOT/'source/environment').resolve():raise ValueError('Use the identical frozen public environment for this scope')
    if args.seed not in range(5):raise ValueError('Registered formal seed0..4 required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Retained independent evaluation is explicitly CPU-only')
    if args.engineering_gate:
        if args.phase!='dev'or args.eval_master!=202611043101 or args.eval_cases!=2:raise ValueError('Engineering interface check uses only2DEV cases; never TEST')
        before=test_bindings(engineering=True);release=None
    else:
        before,release=released_protocol(args)  # before backend import/Task/model deserialization
    started=time.perf_counter();modelsha=sha(args.model)
    try:
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
        os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
        if args.method=='props':native=run_props(args,before)
        elif args.method=='A_inherited':native=run_A(args)
        else:native=run_f1(args)
        if native.get('success')is not True:raise AssertionError('Original evaluation did not complete')
        if(native.get('method'),native.get('domain'),int(native.get('seed',-1)))!=(args.method,args.domain,args.seed):raise AssertionError('Original endpoint identity mismatch')
        if native.get('phase',args.phase)!=args.phase:raise AssertionError('Original final/dev phase mismatch')
        expected=args.eval_cases*(32 if args.domain=='radio' else 24)
        if native.get('evaluation_steps')!=expected:raise AssertionError('Actual evaluation physical step count differs')
        if sha(args.model)!=modelsha:raise AssertionError('Read-only fixed endpoint changed')
        if test_bindings(engineering=args.engineering_gate)!=before:raise AssertionError('Independent TEST source changed during execution')
        if release is not None and validate_release(args.method,args.domain,args.seed,args.model)!=release:raise AssertionError('All-model release changed during evaluation')
        replay=int(native.get('replay_environment_steps',0))
        summary=native.get('summary',native.get('native_receipt',native))
        mean=summary.get('native_mean',summary.get('mean_return',summary.get('mean')));std=summary.get('native_std',summary.get('std_return',summary.get('std')))
        if mean is None or not math.isfinite(float(mean)):raise AssertionError('Finite independent mean is required')
        import torch
        runtime=dict(actual_device='cpu',backend='torch',torch_version=torch.__version__,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),cpu_affinity=sorted(os.sched_getaffinity(0)))
        completed=dict(schema='chapter4-madtd-formal-retained-independent-test-v1',status='completed',success=True,
            method=args.method,domain=args.domain,seed=args.seed,phase=args.phase,engineering_only=args.engineering_gate,
            evaluation_only=True,formal_evaluation=not args.engineering_gate,formal_training_performed=False,
            training_steps=0,new_training_steps=0,new_optimizer_steps=0,evaluation_steps=expected,
            replay_environment_steps=replay,physical_environment_steps=expected+replay,
            holdout_cases=0 if args.engineering_gate else 256,cases=args.eval_cases,paths=args.eval_cases,
            evaluation_master=args.eval_master,test_master=args.eval_master,deterministic_eval=True,
            mean=float(mean),std=float(std)if std is not None else None,
            summary=summary,native_receipt=native,model=str(args.model),model_sha256=modelsha,
            source_binding=before,runtime_binding=runtime,actual_device='cpu',
            endpoint_release=release,endpoint_release_gate_sha256=release['gate_sha256']if release else None,
            test_execution_seal_sha256=sha(ROOT/'protocol/test_execution_seal.json')if(ROOT/'protocol/test_execution_seal.json').exists()else None,
            old_scope_descriptor_is_unchanged_historical_provenance=True,
            evaluation_math='Original retained native evaluator functions; only identical helper/output locations rebound in memory',
            optimizer_or_replay_saved=False,periodic_models_saved=False,new_models_saved=False,wall_seconds=time.perf_counter()-started)
        dump(args.output/'legacy_native_completed.json',native)
        dump(args.output/'runtime_binding.json',runtime)
        dump(args.output/'test_receipt.json',completed)
        dump(args.output/'completion.json',completed);dump(args.output/'completed.json',completed)
        return completed
    except BaseException:
        dump(args.output/'failed.json',dict(status='FAIL',traceback=traceback.format_exc(),engineering_only=args.engineering_gate,source_binding=before,model_sha256=modelsha))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--domain',choices=['radio','v5'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['final','dev'],default='final');p.add_argument('--eval-master',type=int,default=2026110694199)
    p.add_argument('--eval-cases',type=int,default=256);p.add_argument('--engineering-gate',action='store_true')
    p.add_argument('--environment-root',default=str(ROOT/'source/environment'))
    main(p.parse_args())
