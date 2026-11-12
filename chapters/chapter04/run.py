"""Current C4 explicit source entry; inspect/validation never executes science."""
from pathlib import Path
import sys
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT.parents[1]))
sys.path.insert(0, str(_ROOT.parent))
import argparse, json
from portable_paths import project_root, resolve_path
METHODS = ('f1','props','redo','mrq','bro','mad_td','no_history','no_offpolicy_correction','fixed_actor_lr')

def source_entry(method):
    root=project_root()/'chapters/chapter04'
    if method in ('redo','mrq','bro'): return root/'source/offpolicy/execution/offpolicy_worker.py'
    if method=='mad_td': return root/'source/mad_td/execution/formal_worker.py'
    if method=='props': return root/'source/troc_props/execution/props_worker.py'
    return root/'source/troc_props/execution/f1_worker.py'

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--evaluation-only',action='store_true')
    args=parser.parse_args(argv)
    if not args.method or args.config is None:
        if args.execute:parser.error('Explicit --method and registered --config are required.')
        print(json.dumps({'chapter':4,'methods':METHODS,'formal_learning_steps':100608,
            'source_root':str(project_root()/'chapters/chapter04'),
            'default':'metadata inspect only','automatic_queue':False,
            'model_loading':0,'environment_steps':0},ensure_ascii=False,indent=2));return 0
    cfg=json.loads(args.config.read_text());steps=cfg.get('steps',cfg.get('training_steps'))
    if steps!=100608:parser.error('Current formal profiles require an explicit 100608-step registration. LR49152 diagnostics have a separate source entry.')
    if cfg.get('method',args.method)!=args.method:parser.error('Method differs from the registered config.')
    if type(cfg.get('seed'))is not int or cfg['seed']not in range(5):parser.error('An explicit current formal seed 0..4 is required.')
    if cfg.get('domain')not in ('radio','v5'):parser.error('Explicit radio/v5 domain required.')
    output=cfg.get('output',cfg.get('output_dir'))
    if not output:parser.error('A new explicit output directory is required.')
    dest=resolve_path(output).resolve();script=source_entry(args.method)
    summary={'source_entry':str(script),'method':args.method,'domain':cfg['domain'],
             'learning_steps':steps,'output':str(dest),'execute':args.execute,
             'evaluation_only':args.evaluation_only,'model_loading':0,'environment_steps':0}
    if not args.execute:print(json.dumps(summary,ensure_ascii=False,indent=2));return 0
    if dest.exists()and any(dest.iterdir()):parser.error('Refusing to replay an existing scientific output directory.')
    if args.evaluation_only and not cfg.get('evaluate_model'):parser.error('An explicit fixed endpoint is required for evaluation.')
    arguments=[]
    allowed=('domain','seed','steps','device','train_master','dev_master','eval_every','dev_cases','eval_master','eval_cases','environment_root')
    for key in allowed:
        if key in cfg:arguments += ['--'+key.replace('_','-'),str(cfg[key])]
    if args.method!='mad_td':arguments += ['--method',args.method]
    if 'eval_phase'in cfg:
        phase_flag='--evaluation-phase' if args.method in ('f1','no_history','no_offpolicy_correction','fixed_actor_lr') else '--eval-phase'
        if args.method!='mad_td':arguments += [phase_flag,str(cfg['eval_phase'])]
    if cfg.get('evaluate_model'):
        from chapters.chapter04._runtime import resolve_scientific_path
        arguments += ['--model' if args.method=='mad_td' else '--evaluate-model',str(resolve_scientific_path(cfg['evaluate_model']))]
    if args.method=='mad_td':arguments += ['--phase','test' if args.evaluation_only else 'train']
    arguments += ['--output',str(dest)]
    if 'environment_root'not in cfg:
        run='offpolicy' if args.method in ('redo','mrq','bro') else 'troc_props'
        arguments += ['--environment-root',str(project_root()/'chapters/chapter04/source'/run/'source/environment')]
    from chapters.chapter04._runtime import local_git_source_receipts, run_current_source
    with local_git_source_receipts():run_current_source(script,arguments)
    return 0
if __name__=='__main__':raise SystemExit(main())
