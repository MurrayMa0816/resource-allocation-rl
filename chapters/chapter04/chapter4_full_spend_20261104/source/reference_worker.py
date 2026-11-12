"""A/F1 native learning with the new shared full-spend physical contract.

No learning rule is patched here. This fresh-process entry binds the public
task before importing the historical runner and records independent ledgers.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

SCOPE = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(SCOPE / 'source/env'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = Path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'full_spend_completion.json').exists():
        raise FileExistsError('Completed attempts are immutable')
    from chapter4_full_spend_env.bootstrap import install
    binding = install(audit_dir=out/'resource_audit',
        context={'method':config['method'], 'domain':config['domain'],
                 'seed':config['seed'], 'scope':SCOPE.name})
    from chapter4_recent_baselines.runners import reference_runner
    frozen = Path(reference_runner.__file__)
    before = {'worker':sha(__file__), 'native_runner':sha(frozen),
              'config':sha(args.config)}
    write(out/'effective_full_spend_contract.json', {
        'resource_equality':'sum of actual physical allocations equals initial resource',
        'quota_interval':'current C3 public full-spend [L,U], including exact terminal endpoint',
        'inherited_quota_metadata':'Historical metadata string is not the executed transform; see bridge_identity.json',
        'learning_rule_unchanged':True, 'terminal_only_reward_unchanged':True,
        'holdout_opened':False, 'source_bindings':before,
        'binding_manifest':binding['copy_manifest_sha256']})
    result = reference_runner.run(config)
    ledgers = []
    for path in (out/'resource_audit').glob('resource_episodes_*.jsonl'):
        for line in path.open():
            ledgers.append(json.loads(line))
    train_phase = 'engineering' if config.get('smoke') else 'train'
    training = [x for x in ledgers if x['phase'] == train_phase]
    # Smoke evaluation also has engineering phase; identify exact train rows
    # using the native runner's independent training ledger instead.
    native_training = [json.loads(x) for x in (out/'training.jsonl').read_text().splitlines()]
    if len(native_training) != config['episodes']:
        raise AssertionError('Training episode count mismatch')
    max_relative = max(x['relative_allocation_error'] for x in ledgers)
    if max_relative > 1e-10 or any(not x['terminal_full_spend_assertion'] for x in ledgers):
        raise AssertionError('Actual full-spend execution audit failed')
    if not config.get('smoke') and len(training) != config['episodes']:
        raise AssertionError('Native rollout bypassed full-spend ledger')
    if before != {'worker':sha(__file__), 'native_runner':sha(frozen), 'config':sha(args.config)}:
        raise AssertionError('Execution source/config changed during training')
    write(out/'full_spend_completion.json', {'status':'PASS',
        'method':config['method'], 'domain':config['domain'], 'seed':config['seed'],
        'episodes':config['episodes'], 'all_executed_complete_episodes':len(ledgers),
        'training_episodes':len(native_training), 'max_relative_allocation_error':max_relative,
        'maximum_absolute_remaining':max(abs(x['remaining']) for x in ledgers),
        'nonterminal_reward_unchanged':True, 'source_unchanged':True,
        'holdout_opened':False, 'native_receipt':result})

if __name__ == '__main__':
    main()
