#!/usr/bin/env python3
"""Inspect the nine current C6 learning-comparison endpoints; no scientific imports."""
from pathlib import Path
import argparse,json,sys
_SHARED=Path(__file__).resolve().parents[1]
if str(_SHARED)not in sys.path:sys.path.insert(0,str(_SHARED))
import c6_paths

def current_state():
 document=json.loads((c6_paths.chapter_data_root(6)/'MODELS.json').read_text())
 jobs=[]
 for row in document['models']:
  value=row['model'];is_primary='reasonable_allocation89_'in value
  if is_primary and 'budget_4000000__'not in value:continue
  path=c6_paths.original_input(value)
  method='SeDVal'if is_primary else ('COLA'if '/cola__'in value else'PASTA')
  seed=int(path.parent.parent.name.rsplit('seed',1)[1])
  jobs.append(dict(method=method,seed=seed,model=str(path),sha256=row['sha256'],model_file_present=path.is_file(),training_episodes=3200,learning_steps=259200,evaluation_identity='common_DEV_only_for_external_comparison'))
 if len(jobs)!=9:raise ValueError('Expected exactly3 SeDVal400万+3 PASTA+3 COLA retained endpoints')
 return dict(chapter=6,methods=['SeDVal','PASTA','COLA'],models=jobs,new_joint_independent_TEST=False,scientific_imports=0,model_loads=0,environment_steps=0,training_started=False,retired_methods=['CAPQL'],historical_dispatch=False)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',nargs='?',default='inspect',choices=['inspect','status']);p.parse_args();print(json.dumps(current_state(),ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
