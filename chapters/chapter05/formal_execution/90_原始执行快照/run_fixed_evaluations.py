"""Run the sealed210 fixed-model panels with the same durable resource controller."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'execution'))
import dispatch


def main():
    path = R / 'fixed_evaluation_plan.json'
    plan = json.loads(path.read_text())
    assert len(plan['jobs']) == 210
    assert all(j['host'] == 'local' and j['device'] == 'cpu' and j['gpu_uuid'] is None for j in plan['jobs'])
    args = argparse.Namespace(phase='evaluation', host='local', plan_sha=hashlib.sha256(path.read_bytes()).hexdigest())
    return dispatch.controller(args)


if __name__ == '__main__':
    sys.exit(main())
