"""Independently check the twelve registered native baseline engineering gates.

No holdout is opened and no model is imported here. Native run itself performs
the exact persisted-endpoint reload/action check; this audit binds that receipt
to disk hashes, full raw DEV grid, charged physical work, and real device proof.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("capql", "qpensieve", "cola", "morld", "pasta")
DOMAINS = {"radio": (32, 100), "v5": (24, 120)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def grid(total):
    values = list(range(0, total + 1, 512))
    return values if values[-1] == total else values + [total]


def expected_jobs():
    result = []
    for method in METHODS:
        for domain in DOMAINS:
            result.append((f"gate__{method}__{domain}__seed9907", method, domain,
                           4224 if method == "morld" else
                           4352 if method == "pasta" and domain == "radio" else
                           3840 if method == "pasta" and domain == "v5" else 3072))
    for domain in DOMAINS:
        result.append((f"gate__capql__{domain}__seed9907__tail", "capql", domain, 2688))
    return result


def audit_one(job, method, domain, total):
    output = ROOT / "engineering" / job
    completed = json.loads((output / "completed.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    cfg = protocol["config"]
    contract = ROOT / "contract.json"
    assert not (output / "failure.json").exists(), job
    assert completed["success"] is True and completed["engineering"] is True
    assert completed["formal_training_run"] is False
    assert cfg["engineering"] is True and cfg["seed"] == 9907
    assert cfg["id"] == job and cfg["method"] == method and cfg["domain"] == domain
    assert completed["method"] == method and completed["domain"] == domain
    assert cfg["learning_steps"] == completed["learning_steps"] == total
    assert cfg["holdout_paths"] == completed["holdout_paths"] == 0
    assert cfg["dev_paths"] == 8 and cfg["eval_every"] == 512
    assert completed["continuation_from_old_model"] is False
    assert completed["neural_parameters_changed"] is True and completed["updates"] > 0
    assert completed["endpoint_reload_exact"] is True
    assert completed["persisted_endpoint_checked"] is True
    assert completed["optimizer_backup"] is False and completed["replay_backup"] is False
    assert completed["source_unchanged"] is True and completed["holdout_not_opened"] is True
    assert completed["contract_sha256"] == cfg["contract_sha256"] == sha(contract)
    manifest = ROOT / "execution/source_manifest.json"
    assert completed["source_manifest_sha256"] == cfg["source_manifest_sha256"] == sha(manifest)
    assert completed["execution_worker_sha256"] == cfg["execution_worker_sha256"] == sha(ROOT / "execution/baseline_worker.py")
    counts = completed["counts"]
    assert counts["holdout"] == dict(physical_steps=0, episodes=0)
    assert counts["train"]["physical_steps"] + counts["internal"]["physical_steps"] == total
    expected_internal_steps = DOMAINS[domain][0] * 15 if method == "morld" else 0
    assert counts["internal"]["physical_steps"] == expected_internal_steps
    assert counts["internal"]["episodes"] == (15 if method == "morld" else 0)
    internal_records = rows(output / "internal.jsonl.gz")
    assert len(internal_records) == (1 if method == "morld" else 0)
    if method == "morld":
        assert len(internal_records[0]["rows"]) == 15
        assert internal_records[0]["stats"]["psa_applied"] is True
        assert internal_records[0]["stats"]["value_source"] == "charged_internal"
    assert completed["learning_budget_counts_training_plus_internal_psa"] is True
    assert protocol["evaluation_grid"] == completed["evaluation_grid"] == grid(total)
    adapter = completed["evaluation_schedule_adapter"]
    assert adapter["exact_substitutions"] == 3
    assert adapter["rollout_update_replay_psa_rng_and_endpoint_instructions_changed"] is False
    developments = rows(output / "development.jsonl.gz")
    curves = [row for row in developments if row["kind"] == "curve"]
    assert [row["learning_steps"] for row in curves] == grid(total)
    assert len(curves) == len(grid(total))
    h, n = DOMAINS[domain]
    for record in developments:
        assert record["rng_and_learning_state_unchanged"] is True
        assert len(record["rows"]) == 8
        for row in record["rows"]:
            assert row["domain"] == domain and row["N"] == n
            assert 0 <= row["K"] <= n
            spending = row["full_spend"]
            assert spending["full_spend_required"] is True
            assert abs(spending["terminal_remaining"]) <= spending["tolerance"]
            assert abs(spending["executed_total"] - spending["initial_budget"]) <= spending["tolerance"]
            assert all(math.isfinite(float(row[key])) for key in ("K", "C", "J", "native_return", "remaining"))
            assert len(row["quality"]) == n and len(row["period_totals"]) == h
    assert counts["development"]["episodes"] == 8 * len(developments)
    assert counts["development"]["physical_steps"] == h * 8 * len(developments)
    training = rows(output / "training.jsonl.gz")
    assert len(training) == counts["train"]["episodes"]
    for row in training:
        spending = row["full_spend"]
        assert spending["full_spend_required"] is True
        assert abs(spending["terminal_remaining"]) <= spending["tolerance"]
        assert abs(spending["executed_total"] - spending["initial_budget"]) <= spending["tolerance"]
    if method == "pasta":
        updates = rows(output / "updates.jsonl.gz")
        native_groups = [row for row in updates if row.get("native_update") is True]
        assert len(native_groups) == completed["updates"] == 3
        final = native_groups[-1]
        assert final["learning_steps"] == total
        assert final["optimized_real_training_steps"] == total
        assert final["collected_real_training_steps"] == total
        expected_tail = 256 if domain == "radio" else 768
        assert final["minibatch_steps_this_group"] == expected_tail // 64 * 10
        assert final["old_logp_recompute_max_error"] <= 2e-5
        terminal_adapter = completed["terminal_rollout_adapter"]
        assert terminal_adapter["exact_guard_or_sample_count_substitutions"] == 5
        assert terminal_adapter["exact_inverse_ast_equal"] is True
        assert terminal_adapter["ppo_gae_pcgrad_attention_and_decay_instructions_changed"] is False
        assert terminal_adapter["all_real_final_samples_optimized_without_padding"] is True
        assert completed["final_rollout_batch_geometry_modified"] is True
    else:
        assert completed["terminal_rollout_adapter"] is None
        assert completed["final_rollout_batch_geometry_modified"] is False
    endpoint = output / "endpoint.pt.gz"
    assert completed["endpoint_filename"] == endpoint.name
    assert sha(endpoint) == completed["endpoint_sha256"]
    assert not (output / "endpoint.pt.gz.pending").exists()
    binary = [path.name for path in output.iterdir()
              if path.suffix in (".pt", ".pth", ".npz", ".npy") or path.name.endswith(".pt.gz")]
    assert binary == ["endpoint.pt.gz"], binary
    unopened = json.loads((output / "holdout_not_opened.json").read_text())
    assert unopened["physical_steps"] == unopened["episodes"] == 0
    assert unopened["engineering"] is True and unopened["formal_training_run"] is False
    if method in ("capql", "pasta"):
        assert cfg["device"] == "cpu" and completed["gpu_proof"] is None
        proof = completed["cpu_proof"]
        assert proof["cpu_optimizer_tensors"] > 0 and proof["cuda_context_initialized"] is False
    else:
        assert cfg["device"] == "cuda:0" and completed["cpu_proof"] is None
        proof = completed["gpu_proof"]
        assert proof["cuda_optimizer_tensors"] > 0 and proof["cuda_update_event_ms"] > 0
        assert proof["gpu_uuid"] == proof["cuda_visible_devices"]
    assert proof["optimizer_state_tensors_on_registered_device"] > 0
    assert proof["device_fallback"] is False and proof["update_wall_seconds"] > 0
    return dict(job_id=job, method=method, domain=domain, learning_steps=total,
                curve_points=len(curves), reference_points=len(developments)-len(curves),
                updates=completed["updates"], extra_exchange_updates=completed["extra_exchange_updates"],
                training_physical_steps=counts["train"]["physical_steps"],
                internal_psa_physical_steps=counts["internal"]["physical_steps"],
                development_physical_steps=counts["development"]["physical_steps"],
                holdout_physical_steps=0, device=cfg["device"],
                peak_rss_bytes=proof["peak_rss_bytes"], endpoint_sha256=sha(endpoint),
                training_seconds=completed["training_seconds"])


def run():
    results = [audit_one(*job) for job in expected_jobs()]
    assert sum(row["learning_steps"] for row in results) == 40448
    return dict(status="PASS_BASELINE_NATIVE_ENGINEERING_GATES_ONLY",
                formal_training_completed=False, formal_results_claimed=False,
                jobs=len(results), total_learning_steps=40448, rows=results,
                primary_worker_update_calls_or_on_policy_groups=sum(row["updates"] for row in results),
                update_count_units="Native off-policy calls or PASTA rollout groups; not a cross-method Adam-step total",
                development_physical_steps=sum(row["development_physical_steps"] for row in results),
                holdout_physical_steps=0, psa_dynamic_gate_exercised=True,
                psa_charging_clock_static_audit="preflight/baseline_static.json",
                native_loss_equations_changed=False, pasta_terminal_rollout_geometry_modified=True,
                endpoint_reload_verified_by_native_worker=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Audit must remain in its exact new HY scope")
    result = run()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
