"""Verify only this formal registered scope, then supervise it without old restores."""
from pathlib import Path
import hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_PLAN_SHA='359cca57ffd8a6cf41918b51c3ae7546478ca4ca6cdfcca83f4351b22ff2c035'
p=ROOT/'plan.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED_PLAN_SHA,'Registered plan changed'
plan=json.loads(p.read_text())
assert plan['execution_scope']==str(ROOT)
assert len(plan['jobs'])==130 and plan['new_training_jobs']==90 and not plan['crossq_included']
for name,digest in plan['execution_sha256'].items():
 assert hashlib.sha256((ROOT/'execution'/name).read_bytes()).hexdigest()==digest,name
for name,digest in plan['full_source_sha256'].items():
 assert hashlib.sha256((ROOT/'execution/source'/name).read_bytes()).hexdigest()==digest,name
os.execv(sys.executable,[sys.executable,str(ROOT/'execution/watchdog.py')])
