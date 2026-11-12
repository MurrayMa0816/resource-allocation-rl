#!/usr/bin/env python3
"""第二章 BLoAR：训练、评价与五种基线。帮助和配置仅使用标准库。"""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
CHAPTERS=HERE.parent
def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ('-h','--help'):
        print(__doc__)
        print('  config | check\n  train --domain v5|radio --seed 0 --episodes 960 --output NEW_DIR [native options]\n  evaluate --checkpoint FILE --paths 512 --output NEW_DIR\n  baseline train|evaluate [baseline options]\n  Add --legacy-zero-head before the command for the preserved older model revision.\n  Protocol: configs/protocol.json; scientific work runs only on explicit train/evaluate.')
        return 0
    legacy=args[0]=='--legacy-zero-head'
    if legacy:args.pop(0)
    cmd=args[0] if args else ''
    if cmd=='config':print((HERE/'configs/protocol.json').read_text());return 0
    sys.path[:0]=[str(CHAPTERS),str(HERE/'source')]
    from source_runtime import verify,activate
    view='chapter02.legacy' if legacy else 'chapter02.native'
    if cmd=='check':print(json.dumps(verify(view),ensure_ascii=False,indent=2));return 0
    if cmd not in ('train','evaluate','baseline'):raise ValueError('Choose config/check/train/evaluate/baseline')
    if cmd=='baseline' and (len(args)<2 or args[1] not in ('train','evaluate')):raise ValueError('Choose baseline train/evaluate')
    # Command help must not load a neural framework or construct an environment.
    if '--help' in args or '-h' in args:
        print('Native and baseline options are documented in source/allocation23/__main__.py and baselines/runner.py.');return 0
    for i,arg in enumerate(args):
        if arg=='--chapter' and (i+1==len(args) or args[i+1]!='2'):raise ValueError('This runner is chapter 2')
        if arg.startswith('--chapter=') and arg!='--chapter=2':raise ValueError('This runner is chapter 2')
        if arg=='--method' and i+1<len(args) and cmd!='baseline' and args[i+1]!='A':raise ValueError('BLoAR uses --method A; use baseline for comparisons')
    activate(view)
    if cmd=='baseline':
        from allocation23.baselines.runner import main as baseline_main
        forwarded=args[1:]
        if forwarded[0]=='train' and '--chapter' not in forwarded:forwarded+=['--chapter','2']
        return baseline_main(forwarded)
    if cmd=='train':
        if '--chapter' not in args:args+=['--chapter','2']
        if '--method' not in args:args+=['--method','A']
        if '--variant' not in args:args+=['--variant','A_FULL']
        if '--temporal' not in args:args+=['--temporal','fixed']
    from allocation23.__main__ import native_main
    return native_main(args)
if __name__=='__main__':main()
