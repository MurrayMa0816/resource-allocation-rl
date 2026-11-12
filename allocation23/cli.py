"""Current C2/C3 facade; metadata commands never construct environments."""
from __future__ import annotations
from pathlib import Path
import os,sys,json,subprocess

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    root=Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    import portable_paths as paths
    if not args or args[0] in ('-h','--help'):
        print('Current code: project chapter2|chapter3 inspect|check|train|evaluate|baseline. Scientific execution is explicit.');return 0
    chapter=None
    if args[0] in ('chapter2','chapter3'):chapter=int(args.pop(0)[-1])
    selected=[];clean=[];i=0
    while i<len(args):
        if args[i]=='--chapter':
            if i+1==len(args):raise ValueError('Missing --chapter value')
            selected.append(int(args[i+1]));i+=2
        elif args[i].startswith('--chapter='):selected.append(int(args[i].split('=',1)[1]));i+=1
        else:clean.append(args[i]);i+=1
    if selected:
        if len(set(selected))!=1 or chapter is not None and chapter!=selected[0]:raise ValueError('Conflicting chapter selectors')
        chapter=selected[0]
    args=clean or ['inspect']
    if args[0] in ('inspect','status','plan') and chapter is None:
        print(json.dumps(json.loads((paths.final_data_root()/'INDEX.json').read_text()),ensure_ascii=False,indent=2));return 0
    if args[0] in ('compare','comparison'):
        raise ValueError('Historical queues are frozen provenance; use current explicit chapter workers')
    if chapter not in (2,3):raise ValueError('Choose --chapter2/3 explicitly, or prefix chapter2/chapter3')
    entry=root/'chapters'/f'chapter{chapter:02d}'/'launcher.py'
    env=dict(os.environ,RESOURCE_ALLOCATION_PROJECT_ROOT=str(root),RESOURCE_ALLOCATION_DATA_ROOT=str(paths.final_data_root()),PYTHONDONTWRITEBYTECODE='1',ALLOCATION23_DATA_ROOT=str(paths.eurosat_root()))
    env['PYTHONPATH']=str(root)
    return subprocess.run([sys.executable,str(entry),*args],env=env,check=True).returncode
