"""Frozen own-scope launch, with no implicit old queue restore."""
from pathlib import Path
import hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_PLAN_SHA='be34ccc89c0543dcd6fa099667c922d9f431272449642883e723028d42ba290c'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(ROOT/'plan.json')==EXPECTED_PLAN_SHA,'Registered plan changed'
plan=json.loads((ROOT/'plan.json').read_text())
assert plan['execution_scope']==str(ROOT) and len(plan['jobs'])==20 and plan['new_training_jobs']==10 and plan['evaluation_jobs']==10
assert {j['method'] for j in plan['jobs']}=={'pfo'}
for name,digest in plan['execution_sha256'].items():assert sha(ROOT/'execution'/name)==digest,name
for name,digest in plan['full_source_sha256'].items():assert sha(ROOT/'execution/source'/name)==digest,name
os.execv(sys.executable,[sys.executable,str(ROOT/'execution/watchdog.py')])
