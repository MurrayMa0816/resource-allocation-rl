#!/usr/bin/env python3
"""Chapter 6 source inspection and explicit original-program dispatch."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CHAPTERS = HERE.parent
if str(CHAPTERS) not in sys.path:
    sys.path.insert(0, str(CHAPTERS))


def inspect():
    print(json.dumps({
        "chapter": 6,
        "region_count": 89,
        "horizon_days": 81,
        "methods": ["SeDVal", "PASTA", "COLA"],
        "experiment": "experiment",
        "comparison": "comparison",
        "shared_baselines": "../chapter05/source/baselines",
        "figures": "figures",
        "models_and_full_records_included": False,
    }, ensure_ascii=False, indent=2))
    return 0


def check():
    source = HERE / "experiment/source"
    manifest = json.loads((source / "hongmiao89/dependency_manifest.json").read_text())
    frozen = (source / "hongmiao89" / manifest["frozen_source_relative"]).resolve()
    errors = []
    for relative, expected in manifest["files_sha256"].items():
        path = frozen / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(relative)
    shared = CHAPTERS / "chapter05/source/baselines"
    adapter = ast.parse((HERE / "comparison/methods/pasta_adapter.py").read_text())
    expected_pasta = next(ast.literal_eval(node.value) for node in adapter.body
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "SOURCE_SHA256"
                                  for target in node.targets))
    for relative, expected in expected_pasta.items():
        path = shared / "pasta" / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append("../chapter05/source/baselines/pasta/" + relative)
    for relative in ["__init__.py", "cola/agent.py", "cola/configs.json", "_support/replay.py", "_support/runtime.py", "_support/common.py", "_support/update_clock.py"]:
        if not (shared / relative).is_file():
            errors.append("../chapter05/source/baselines/" + relative)
    print(json.dumps({
        "status": "PASS" if not errors else "FAIL",
        "bound_dependencies_checked": len(manifest["files_sha256"]),
        "shared_pasta_hashes_checked": len(expected_pasta),
        "shared_baseline_files_present": not any(value.startswith("../") for value in errors),
        "errors": errors,
    }, ensure_ascii=False, indent=2))
    return int(bool(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="Show chapter source locations")
    sub.add_parser("check", help="Check immutable dependency hashes without scientific imports")
    for name in ["train", "evaluate"]:
        item = sub.add_parser(name, help="Run the original program with an explicit job configuration")
        item.add_argument("--config", type=Path, required=True)
        item.add_argument("--execute", action="store_true", help="Execute the original program after configuration validation")
    comp = sub.add_parser("comparison", help="Run an original PASTA or COLA training worker")
    comp.add_argument("--method", choices=["pasta", "cola"], required=True)
    comp.add_argument("--config", type=Path, required=True)
    comp.add_argument("--execute", action="store_true", help="Execute the original worker after configuration validation")
    args, rest = parser.parse_known_args(argv)
    if args.command in ["inspect", "check"]:
        if rest:
            parser.error("Unexpected arguments: " + " ".join(rest))
        return inspect() if args.command == "inspect" else check()
    config = args.config.expanduser().resolve()
    if not config.is_file():
        parser.error("Configuration file does not exist: " + str(config))
    if args.command == "train":
        target = HERE / "experiment/execution/worker_train.py"
    elif args.command == "evaluate":
        target = HERE / "experiment/evaluation/evaluate.py"
    else:
        name = "worker.py" if args.method == "pasta" else "cola_worker.py"
        target = HERE / "comparison/execution" / name
    configuration = json.loads(config.read_text())
    if not isinstance(configuration, dict):
        parser.error("Configuration must be a JSON object")
    command = [sys.executable, "-B", str(target), "--config", str(config), *rest]
    if not args.execute:
        print(json.dumps({"status": "CONFIGURATION_READ", "command": command, "execute": False}, ensure_ascii=False, indent=2))
        return 0
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
