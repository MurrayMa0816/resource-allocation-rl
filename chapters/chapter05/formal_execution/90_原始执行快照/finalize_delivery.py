"""Record actual completed science and publication; never advances an incomplete run."""
from pathlib import Path
import datetime
import hashlib
import json
import os

R = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    training = read(R / 'training_audit.json')
    evaluation = read(R / 'final_evaluation_audit.json')
    registry = read(R / 'model_registry.json')
    publication = read(R / 'FINAL_THESIS_PUBLICATION.json')
    assert training['status'] == evaluation['status'] == 'PASS'
    assert registry['all_training_complete'] and len(registry['models']) == 90
    assert registry['training_audit_sha256'] == sha(R / 'training_audit.json')
    assert evaluation['training_models'] == 90 and evaluation['evaluation_jobs'] == 210
    assert evaluation['formal_test_episodes'] == 53760 and evaluation['formal_test_physical_steps'] == 1505280
    assert evaluation['new_learning_steps'] == evaluation['optimizer_steps'] == 0
    assert evaluation['report_bundle_sha256'] == sha(R / 'report_bundle.json')
    assert publication['status'] == 'PUBLISHED'
    assert publication['report_ready_sha256'] == sha(R / 'reports/final/publication_ready.json')
    assert sha(publication['immutable_pdf']) == publication['pdf_sha256']
    evidence = ['contract.json', 'plan.json', 'execution/source_manifest.json', 'deployment/gate_pass.json',
                'model_registry.json', 'training_audit.json', 'fixed_evaluation_plan.json',
                'final_evaluation_audit.json', 'report_bundle.json', 'reports/final/publication_ready.json',
                'FINAL_THESIS_PUBLICATION.json', 'audit/remote_mirror_verification.json']
    receipt = dict(schema='chapter5-formal-five-seed-completion-v1', status='complete',
                   utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), root=str(R),
                   science_seeds=list(range(5)), trained_models=90, training_steps_per_model=100608,
                   development_interval=512, development_paths=32, development_points=198,
                   formal_evaluations=210, evaluation_episodes=53760, evaluation_physical_steps=1505280,
                   training_model_reuse=0, automatic_more_seeds=False,
                   independent_pdf=publication['report_pdf'], thesis_pdf=publication['thesis_pdf'],
                   immutable_thesis_pdf=publication['immutable_pdf'], immutable_pdf_sha256=publication['pdf_sha256'],
                   evidence={p: sha(R / p) for p in evidence},
                   visual_review=publication.get('manual_or_agent_visual_review', 'See publication receipt'))
    write(R / 'completion_receipt.json', receipt)
    paths = [Path('/root/gamewar/resource_allocation_rl/chapters/chapter05/experiments/CURRENT_EXECUTION.json'),
             Path('/root/gamewar/resource_allocation_rl/chapters/chapter05/experiments/CURRENT_FORMAL_EXECUTION.json'),
             R.parent / 'LATEST_FORMAL_EXECUTION.json']
    for path in paths:
        if path.exists():
            current = read(path)
            if current.get('root') == str(R):
                current.update(status='complete', completion_receipt=str(R / 'completion_receipt.json'),
                               completion_receipt_sha256=sha(R / 'completion_receipt.json'),
                               updated_utc=receipt['utc'], immutable_pdf=publication['immutable_pdf'])
                write(path, current)
    return receipt


if __name__ == '__main__':
    main()
