#!/usr/bin/env python3
"""第三章 ReJoP：联合决策、五种比较器与随机对象 DEV 消融。"""
import json,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
CHAPTERS=HERE.parent
def has(args,key):return any(x==key or x.startswith(key+'=') for x in args)
def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ('-h','--help'):
        print(__doc__)
        print('  config | check\n  train --domain radio|v5 --arm A_FULL|A_PACE --seed 0 --steps 100608 --output NEW_DIR\n  baseline train|evaluate [comparator options]\n  random-object train --domain radio|v5 --seed 0 --steps 100608 --output NEW_DIR\n  evaluate --run RUN --registry FILE --condition NAME --endpoint-seal FILE --output NEW_DIR\n  Shared environment source is owned by chapter02/source; it is not duplicated here.\n  Native evaluation retains the original sealed-40-endpoint protocol.')
        return 0
    cmd=args.pop(0)
    if cmd=='config':print((HERE/'configs/protocol.json').read_text());return 0
    sys.path[:0]=[str(CHAPTERS),str(CHAPTERS/'chapter02/source')]
    from source_runtime import verify,activate,execute_file
    if cmd=='check':
        print(json.dumps([verify(v) for v in ('chapter03.native','chapter03.comparators')],ensure_ascii=False,indent=2));return 0
    if cmd not in ('train','evaluate','baseline','random-object'):raise ValueError('Choose config/check/train/evaluate/baseline/random-object')
    if '--help' in args or '-h' in args:
        print('Native options: source/native/train_worker.py and evaluate_worker.py; comparator options: source/allocation23/chapter3_candidates/cli.py.');return 0
    if cmd=='baseline':
        if not args or args[0] not in ('train','evaluate'):raise ValueError('Choose baseline train/evaluate')
        activate('chapter03.comparators')
        from allocation23.chapter3_candidates.cli import main as baseline_main
        return baseline_main(args)
    activate('chapter03.native')
    from portable_paths import eurosat_root
    if not has(args,'--source-root'):args+=['--source-root',str(HERE/'source')]
    if not has(args,'--data-root'):args+=['--data-root',str(eurosat_root())]
    if cmd=='random-object':
        if not args or args.pop(0)!='train':raise ValueError('Only random-object train is exposed; its evidence is DEV only')
        if not has(args,'--arm'):args+=['--arm','A_RANDOM_OBJECT_ALLOCATION']
        if not has(args,'--final-master'):args+=['--final-master','202611093799']
        return execute_file(HERE/'source/random_object/train.py',args)
    if cmd=='train':
        arms=[args[i+1] for i,x in enumerate(args[:-1]) if x=='--arm']+[x.split('=',1)[1] for x in args if x.startswith('--arm=')]
        if any(x not in ('A_FULL','A_PACE') for x in arms):raise ValueError('Current native arms are A_FULL/A_PACE')
        return execute_file(HERE/'source/native/train_worker.py',args)
    if not has(args,'--direct-module'):args+=['--direct-module',str(HERE/'source/native/direct_policy.py')]
    return execute_file(HERE/'source/native/evaluate_worker.py',['evaluate',*args])
if __name__=='__main__':main()
