"""Run twelve disjoint baseline gates, at most four CPU and six CUDA jobs.

This launcher never discovers or starts formal configs. Each job is a fresh
native run. Existing successful gates are reused only as this same engineering
gate's receipts; a partial or failed output is refused for explicit review.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
REMOTE_PYTHON = Path("/hy-tmp/resource_allocation_results/chapter2/gpu0_five_seed_150k_20261104/runtime/python/bin/python")
GPUS = ("GPU-71946e17-6e0d-6a7b-1fa4-b3de51c38fe3", "GPU-c63f3af4-0bfd-7b90-9034-5e1728ae51de")


def write(path, payload):
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(host):
    if host not in ("local", "remote"):
        raise ValueError("Explicit local or remote host required")
    expected_methods = {"capql", "pasta"} if host == "local" else {"qpensieve", "cola", "morld"}
    expected_device = "cpu" if host == "local" else "cuda:0"
    concurrency = 4 if host == "local" else 6
    python = "/usr/local/bin/python3" if host == "local" else str(REMOTE_PYTHON)
    configs = []
    for path in sorted((ROOT / "engineering_configs").glob("*.json")):
        cfg = json.loads(path.read_text())
        if cfg.get("method") not in expected_methods:
            continue
        if cfg.get("engineering") is not True or cfg.get("seed") != 9907 or cfg.get("device") != expected_device:
            raise ValueError("Gate execution registration differs: " + str(path))
        configs.append((path, cfg))
    if len(configs) != 6:
        raise ValueError("Each host must receive its exact six baseline engineering gates")
    logs = ROOT / "preflight" / ("baseline_gate_logs_" + host)
    logs.mkdir(parents=True, exist_ok=True)
    status_path = ROOT / "preflight" / ("baseline_gate_status_" + host + ".json")
    queue = []
    finished = []
    for path, cfg in configs:
        output = Path(cfg["output"])
        if output.exists():
            completed_path = output / "completed.json"
            if not completed_path.exists() or (output / "failure.json").exists():
                raise FileExistsError("Review partial/failed gate explicitly before another attempt: " + str(output))
            done = json.loads(completed_path.read_text())
            if not done.get("success") or done.get("engineering") is not True or done.get("learning_steps") != cfg["learning_steps"]:
                raise RuntimeError("Existing gate receipt differs")
            finished.append(dict(job_id=cfg["id"], exit_code=0, existing_same_gate=True))
        else:
            queue.append((path, cfg))
    running = {}
    failure = False
    launched = 0
    while queue or running:
        while queue and len(running) < concurrency and not failure:
            path, cfg = queue.pop(0)
            environment = dict(os.environ)
            environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                               PYTHONPYCACHEPREFIX=str(ROOT / "runtime_cache/engineering_pycache"))
            if host == "local":
                environment["CUDA_VISIBLE_DEVICES"] = ""
                environment.pop("CH5_GPU_UUID", None)
                gpu = None
            else:
                usage = {gpu: sum(entry["gpu"] == gpu for entry in running.values()) for gpu in GPUS}
                gpu = min(GPUS, key=lambda item: usage[item])
                environment.update(CUDA_VISIBLE_DEVICES=gpu, CH5_GPU_UUID=gpu)
            log = (logs / (cfg["id"] + ".log")).open("ab")
            process = subprocess.Popen(
                [python, str(ROOT / "execution/baseline_worker.py"), "--config", str(path)],
                cwd=str(ROOT), env=environment, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            running[process.pid] = dict(process=process, log=log, config=cfg, gpu=gpu)
            launched += 1
        for pid, entry in list(running.items()):
            code = entry["process"].poll()
            if code is None:
                continue
            entry["log"].close()
            finished.append(dict(job_id=entry["config"]["id"], pid=pid, exit_code=code, gpu_uuid=entry["gpu"]))
            del running[pid]
            failure |= code != 0
        write(status_path, dict(status="FAILED_REVIEW_REQUIRED" if failure else "RUNNING" if queue or running else "COMPLETED",
                                host=host, scope="registered_baseline_engineering_only", controller_pid=os.getpid(),
                                expected_jobs=6, newly_launched=launched, completed=len(finished), queued=len(queue),
                                running=[dict(job_id=entry["config"]["id"], pid=pid, gpu_uuid=entry["gpu"])
                                         for pid, entry in running.items()], finished=finished,
                                formal_jobs_started=0, updated_epoch=time.time()))
        if failure and not running:
            break
        if queue or running:
            time.sleep(2)
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, choices=("local", "remote"))
    run(parser.parse_args().host)
