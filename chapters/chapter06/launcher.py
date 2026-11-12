#!/usr/bin/env python3
"""Read-only Chapter6 entry for retained science, source identity and saved figures."""
from pathlib import Path
import argparse,json,os,subprocess,sys
PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT)not in sys.path:sys.path.insert(0,str(PROJECT))
from portable_paths import project_root,final_data_root,chapter_data_root

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('command',nargs='?',default='inspect',choices=['inspect','plan','status','check','verify','redraw','plot','path','compare'])
 args,rest=p.parse_known_args(argv)
 if args.command=='path':
  if rest:p.error('path takes no extra arguments')
  print(chapter_data_root(6));return 0
 if args.command=='compare':
  if rest not in ([],['inspect'],['status']):p.error('compare only reads current retained comparison; use compare inspect')
  target=project_root()/'applications/irrigation/learning_comparison_20261109/current_comparison.py'
  command=[sys.executable,'-B',str(target),'inspect']
 else:
  name={'plan':'inspect','status':'inspect','check':'verify','plot':'redraw'}.get(args.command,args.command)
  rest=[x for x in rest if x not in ['--json','--read-only']]
  command=[sys.executable,'-B',str(project_root()/'scripts/chapter06_portable.py'),name,'--data-root',str(final_data_root()),'--code-root',str(project_root()),'--figure-root',str((chapter_data_root(6)/'figures').relative_to(final_data_root())),*rest]
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',RESOURCE_ALLOCATION_DATA_ROOT=str(final_data_root()),FINAL_DATA_ROOT=str(final_data_root()))
 return subprocess.run(command,env=env,check=True).returncode
if __name__=='__main__':raise SystemExit(main())
