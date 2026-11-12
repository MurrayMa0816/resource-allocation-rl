#!/usr/bin/env python3
"""Final Chapter 3: metadata first, explicit native/baseline/random-object work."""
from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'portable_paths.py').is_file())
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import portable_paths as paths
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
ARMS=('A_FULL','A_PACE','A_RANDOM_OBJECT_ALLOCATION')

def metadata():
    return json.loads((paths.final_data_root()/'INDEX.json').read_text())['chapters']['chapter03']

def option(args,key):
    values=[v.split('=',1)[1] for v in args if v.startswith(key+'=')]
    values += [args[i+1] for i,v in enumerate(args[:-1]) if v==key]
    if len(values)>1:raise ValueError('Repeated protected option: '+key)
    return values[0] if values else None

def environment():
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
        RESOURCE_ALLOCATION_PROJECT_ROOT=str(ROOT),
        RESOURCE_ALLOCATION_DATA_ROOT=str(paths.final_data_root()),
        FINAL_DATA_ROOT=str(paths.final_data_root()),
        ALLOCATION23_DATA_ROOT=str(paths.eurosat_root()))
    env['PYTHONPATH']=str(ROOT)
    return env

def guard_output(args):
    if any(v in ('-h','--help') for v in args):return
    output=option(args,'--output')
    if not output:raise ValueError('An explicit new --output directory is required')
    target=Path(output).expanduser().resolve()
    if target.exists():raise FileExistsError('Existing final results cannot be overwritten: '+str(target))
    protected=[ROOT/'chapters',ROOT/'allocation23',ROOT/'frozen',paths.final_data_root()/'datasets']
    if any(target.is_relative_to(p.resolve()) for p in protected):
        raise ValueError('Outputs must be separate from source and sealed records')

def run_source(script,args):
    guard_output(args)
    bootstrap='import sys; from frozen_adapter import run_file; run_file('+repr(str(script))+',sys.argv[1:])'
    return subprocess.run([sys.executable,'-c',bootstrap,*args],env=environment(),check=True).returncode

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv);cmd=args.pop(0) if args else 'inspect'
    if cmd in ('inspect','status','plan'):
        print(json.dumps(metadata(),ensure_ascii=False,indent=2));return 0
    if cmd in ('-h','--help'):
        p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''inspect|status|plan         Saved metadata only; no environment/model import
check                       Verify current frozen source bytes
config|ablations             Current final protocol and two adopted controls
train|evaluate ...           ReJoP or rule-total native worker (explicit only)
baseline inspect             Five adopted comparator configurations
baseline train|evaluate ...  Explicit C-PPO/PFO/Gradient PPO/RPO/PPO+ worker
random-object train ...      Current random-object DEV-only ablation
Historical dispatch/watchdog and retired controls are not public commands.''')
        p.print_help();return 0
    if cmd in ('config','ablations'):
        print((HERE/'configs'/('ablations.json' if cmd=='ablations' else 'current.json')).read_text());return 0
    if cmd=='check':
        # Publication release source check uses its actual exported inventory.
        if (ROOT / 'release_manifest.json').is_file():
            return subprocess.run([sys.executable, '-B', str(ROOT / 'release_tools.py'),
                '--root', str(ROOT), 'check'], check=False).returncode
        from formal_native import verified_capsule
        capsule=verified_capsule()
        print(json.dumps({'status':'PASS','chapter':3,'scientific_capsule':str(capsule),
            'current_arms':ARMS,'model_loads':0,'environment_execution':0},ensure_ascii=False,indent=2));return 0
    if cmd in ('baseline','baselines'):
        sub=args[0] if args else 'inspect'
        if sub in ('inspect','status','plan'):
            print((HERE/'configs/baselines.json').read_text());return 0
        if sub not in ('train','evaluate'):raise ValueError('Choose baseline inspect/train/evaluate')
        guard_output(args)
        if sub=='evaluate':
            # Existing endpoints retain their exact training source identity.
            # The immutable common evaluator checks that identity and content SHA.
            run=option(args,'--run')
            if not run and '--help' not in args and '-h' not in args:raise ValueError('Existing model evaluation requires --run')
            method='pfo'
            if run:
                protocol=json.loads((Path(run).expanduser()/'protocol.json').read_text())
                method=protocol.get('arguments',{}).get('method','')
            source=ROOT/'frozen/chapter03'/('pfo' if method=='pfo' else 'external')/'source'
            forwarded=list(args)
            defaults={'--source-root':str(source),'--registry':str(ROOT/'frozen/chapter03/native/conditions/registry.json'),
                      '--condition':'default','--data-root':str(paths.eurosat_root())}
            for key,value in defaults.items():
                if option(forwarded,key) is None:forwarded.extend([key,value])
            return run_source(ROOT/'frozen/chapter03/native/evaluate_worker.py',forwarded)
        if option(args,'--data-root') is None:args.extend(['--data-root',str(paths.eurosat_root())])
        bootstrap='import sys; from allocation23.chapter3_candidates.cli import main; main(sys.argv[1:])'
        return subprocess.run([sys.executable,'-c',bootstrap,*args],env=environment(),check=True).returncode
    if cmd=='random-object':
        if not args or args[0] in ('inspect','status','plan'):
            print(json.dumps({'arm':'A_RANDOM_OBJECT_ALLOCATION','scientific_scope':'DEV only; no independent new TEST',
                'source':str(HERE/'random_object_ablation_20261109/train.py')},ensure_ascii=False,indent=2));return 0
        if args[0]!='train':raise ValueError('Current random-object evidence is DEV-only; choose train explicitly')
        return run_source(HERE/'random_object_ablation_20261109/train.py',args[1:])
    if cmd in ('train','evaluate'):
        guard_output(args)
        arm=option(args,'--arm')
        if arm not in (None,'A_FULL','A_PACE'):
            raise ValueError('Current native arms are A_FULL/A_PACE; use random-object for its isolated worker')
        from formal_native import main as native
        return native([cmd,*args])
    raise ValueError('Unknown final Chapter3 command: '+cmd)

if __name__=='__main__':raise SystemExit(main())
