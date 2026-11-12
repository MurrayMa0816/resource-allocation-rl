"""Standard-library-only audit of the formal baseline facade and its clock.

This is a static/clock check, never a training or environment experiment. It
neither imports native learning modules nor opens training/development/test
environments. Optional config validation checks source hashes without importing
Torch. Dynamic native update gates must be registered and run separately.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_worker():
    sys.dont_write_bytecode = True
    specification = importlib.util.spec_from_file_location(
        "chapter5_formal_baseline_static_subject", ROOT / "execution/baseline_worker.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def simulate_clock(worker, total, horizon, environments, internal=False, legacy=False):
    """Integer-only execution of native collection and PSA charging boundaries."""
    assert total % (horizon * environments) == 0 and 512 % environments == 0
    learning = next_eval = train = psa = psa_calls = last_psa = 0
    observed = []

    def evaluate():
        nonlocal next_eval
        assert learning == next_eval
        observed.append(learning)
        next_eval = (next_eval + 512 if legacy else
                     min(next_eval + 512, total) if next_eval < total else total + 512)

    evaluate()
    while learning < total:
        assert total - learning >= horizon * environments
        for tick in range(horizon):
            learning += environments
            train += environments
            if learning == next_eval and tick < horizon - 1:
                evaluate()
        if learning == next_eval:
            evaluate()
        packet = 3 * 5 * horizon
        if internal and total - learning >= packet and learning - last_psa >= 3072:
            for position in range(packet):
                learning += 1
                psa += 1
                if learning == next_eval and position != packet - 1:
                    evaluate()
            psa_calls += 1
            last_psa = learning
            if learning == next_eval:
                evaluate()
    assert learning == total and train + psa == total
    assert next_eval == total + 512
    assert observed == worker.evaluation_grid(total)
    assert len(observed) == len(set(observed))
    return dict(total=total, horizon=horizon, n_env=environments,
                charged_internal=internal, train_steps=train, internal_steps=psa,
                psa_packets=psa_calls, evaluation_points=len(observed),
                first=observed[0], last=observed[-1], last_interval=observed[-1]-observed[-2])


def run(native_backend, configs):
    worker = load_worker()
    original = Path(native_backend).read_text()
    proof = worker.schedule_adapter_proof(original)
    rows = []
    for total in (2688, 3072, 3840, 4224, 4352, 6144, 98304, 100608):
        for horizon in (24, 32):
            for environments in (1, 8):
                if total % (horizon * environments):
                    continue
                rows.append(simulate_clock(worker, total, horizon, environments))
                if environments == 1:
                    rows.append(simulate_clock(worker, total, horizon, environments, internal=True))
                if total % 512 == 0:
                    old = simulate_clock(worker, total, horizon, environments, legacy=True)
                    assert old == rows[-1] if not rows[-1]['charged_internal'] else old == rows[-2]
                    if environments == 1:
                        native = simulate_clock(worker, total, horizon, environments, internal=True, legacy=True)
                        assert native == rows[-1]
    assert len(worker.evaluation_grid(100608)) == 198
    assert worker.evaluation_grid(100608)[-2:] == [100352, 100608]
    for invalid in (0, -1, True, 1.5, "100608"):
        try:
            worker.evaluation_grid(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid grid input accepted")
    # The byte/source guard must reject even a harmless whitespace mutation;
    # that prevents silently adapting an unreviewed learner revision.
    try:
        worker.adapted_run_source(original + "\n")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Modified native source accepted")
    validations = [worker.validate_only(path) for path in configs]
    forbidden = [name for name in sys.modules
                 if name == "torch" or name.startswith("torch.")
                 or name.startswith("chapter5_full_spend_20261104")
                 or name == "numpy" or name.startswith("numpy.")]
    assert not forbidden, forbidden
    return {
        "status": "PASS_STATIC_AND_INTEGER_CLOCK_ONLY",
        "worker_sha256": hashlib.sha256((ROOT / "execution/baseline_worker.py").read_bytes()).hexdigest(),
        "adapter": proof, "clock_cases": rows, "configs_validated": len(validations),
        "validation_jobs": [row["job_id"] for row in validations],
        "new_environment_steps": 0, "new_neural_updates": 0,
        "holdout_steps": 0, "cuda_initialized": False, "model_written": False,
        "native_torch_numpy_environment_imported": False,
        "not_dynamic_native_or_performance_acceptance": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-backend", required=True)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run(args.native_backend, args.config)
    if args.output:
        path = Path(args.output).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("The static receipt must stay in the new formal HY scope")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
