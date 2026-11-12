"""Chapter 5 command line; help and inspection use only the standard library."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT.parent))

def main(argv=None):
    parser = argparse.ArgumentParser(description="第五章 SeDVal：训练与固定策略评价")
    parser.add_argument("command", choices=("inspect", "train", "evaluate"), nargs="?", default="inspect")
    parser.add_argument("--config", type=Path, help="显式实验配置 JSON")
    parser.add_argument("--execute", action="store_true", help="执行配置指定的新实验")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    from chapters.chapter05 import _runtime
    if args.command == "inspect":
        import json
        print(json.dumps(_runtime.inspect(), ensure_ascii=False, indent=2))
        return 0
    if args.config is None:
        parser.error("train/evaluate requires --config")
    rest = ["--config", str(args.config)]
    if args.execute:
        rest.append("--execute")
    if args.validate_only:
        rest.append("--validate-only")
    return _runtime.main("train" if args.command == "train" else "evaluate", rest)

if __name__ == "__main__":
    raise SystemExit(main())
