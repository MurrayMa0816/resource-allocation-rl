"""Collect actual readiness receipts. Never estimates or ranks effectiveness."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

from .common import RESULT_ROOT, ROOT, json_write
from .sources import FINAL_METHODS, verify


def main():
    rows=[]
    sources=[]
    for method in FINAL_METHODS:
        sources.append(verify(method))
        for domain in ('radio','v5'):
            relative={'props':f'smoke/props_{domain}/result.json',
                      'fog':f'smoke/fog/{domain}/receipt.json',
                      'addq':f'smoke/addq/{domain}/receipt.json',
                      'tdmpc2':f'smoke_loop_v2/tdmpc2/{domain}/receipt.json',
                      'pgr':f'smoke/pgr/{domain}/receipt.json'}[method]
            path=RESULT_ROOT/relative
            data=json.loads(path.read_text())
            assert data.get('smoke',data.get('smoke_only',False))
            if 'status' in data:assert data['status']=='smoke_passed'
            device=data.get('device',','.join(data.get('jax_devices',[])))
            assert 'cuda' in device, (method,domain,device)
            checks=data.get('mechanism_checks',{})
            actor_updates=int(data.get('actor_updates',checks.get('target_minibatches',0)))
            critic_updates=int(data.get('critic_updates',data.get('critic_minibatches',actor_updates)))
            assert actor_updates>0 and critic_updates>0
            if method=='props':assert checks['behavior_updates']>0 and checks['maximum_old_rows_in_target_update']>0
            if method=='fog':
                assert data['distinctive_mechanism']['growth_exercised'] and data['distinctive_mechanism']['weighted_replay_exercised']
                assert data['distinctive_mechanism']['bc_restore_exercised'] and data['bc_enabled_gradient_updates']>0
            if method=='addq':assert data['distinctive_mechanism']['quantiles_finite']
            if method=='tdmpc2':assert data['model_updates']>0 and data['environment_actions_from_native_MPPI']>0
            if method=='pgr':assert data['diffusion_training_steps']>0 and data['mixed_minibatches']>0
            rows.append(dict(method=method,domain=domain,readiness='PASS',device=device,
                observed_smoke_train_steps=int(data.get('real_steps',data.get('train_env_steps',0))),
                observed_actor_minibatch_updates=actor_updates,observed_critic_minibatch_updates=critic_updates,
                receipt=str(path),receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                performance_ranking_available=False))
    dry=json.loads((RESULT_ROOT/'dryrun/summary.json').read_text())
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in ROOT.rglob('*.py') if 'vendor' not in p.parts}
    summary=dict(status='five_methods_ready_for_training',final_methods=list(FINAL_METHODS),
        passed_domain_method_smokes=len(rows),expected_domain_method_smokes=10,
        official_core_sources_unchanged=True,all_five_smoked_on_gpu=True,
        formal_efficacy_training_launched=False,new_baseline_performance_ranking=None,
        smoke_reduced_configurations_are_not_formal_results=True,
        prior_A_F1_evidence_reused_without_new_training=True,
        formal_parameters_require_separate_performance_evaluation=True,
        dryrun_receipt=str(RESULT_ROOT/'dryrun/summary.json'),
        environment_and_metrics_tests=str(RESULT_ROOT/'tests_final_receipt.json'),
        sources=sources,method_domain_receipts=rows,implementation_sha256=hashes,
        reserve_not_counted=['crossq','bro'])
    json_write(RESULT_ROOT/'readiness_summary.json',summary)
    with (RESULT_ROOT/'readiness_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'status':summary['status'],'gpu_smokes':len(rows),'formal_training':False},ensure_ascii=False))


if __name__=='__main__':main()
