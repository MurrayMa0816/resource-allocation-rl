"""Offline inspection and complete engineering demonstration, never training."""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import make_env
from ._util import jsonable


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Verify all packaged source and data hashes offline")
    verify = sub.add_parser("verify-episode", help="Independently verify a complete exported episode")
    verify.add_argument("directory")
    demo = sub.add_parser("demo", help="81-day fixture for environment/recording validation, not a learned policy")
    demo.add_argument("--output", required=True, help="New directory; refuses overwrite")
    demo.add_argument("--coverage-threshold", required=True, type=float)
    demo.add_argument("--coverage-weight", required=True, type=float)
    demo.add_argument("--reward-mode", choices=("joint", "mean_performance"), default="joint")
    demo.add_argument("--total-budget-m3", type=float, default=810000.)
    demo.add_argument("--weather-mode", choices=("persistent_drought", "historical", "markov"), default="persistent_drought")
    demo.add_argument("--historical-year", type=int, choices=(2022, 2024), default=2022)
    demo.add_argument("--rain-multiplier", type=float)
    demo.add_argument("--controller", choices=("area_daily", "zero", "random"), default="area_daily")
    demo.add_argument("--environment-seed", type=int, default=181)
    demo.add_argument("--controller-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "check":
        from .validation import verify_release
        result = verify_release()
    elif args.command == "verify-episode":
        from .recording import verify_episode_directory
        result = verify_episode_directory(args.directory)
    else:
        from .recording import run_episode, write_episode
        env = make_env(coverage_threshold=args.coverage_threshold,
            coverage_weight=args.coverage_weight, reward_mode=args.reward_mode,
            total_budget_m3=args.total_budget_m3, weather_mode=args.weather_mode,
            historical_year=args.historical_year, rain_multiplier=args.rain_multiplier)
        rng = np.random.default_rng(args.controller_seed)
        quota = env.initial_budget_m3 / env.T
        def controller(obs):
            area = obs["rice_area_ha"]
            available = float(obs["available_water_m3"])
            if args.controller == "zero":
                return np.zeros_like(area)
            if args.controller == "area_daily":
                return area / area.sum() * min(quota, available)
            weights = rng.uniform(size=len(area))
            return weights / weights.sum() * available * float(rng.uniform())
        try:
            record = run_episode(env, controller, environment_seed=args.environment_seed,
                controller_seed=args.controller_seed,
                controller_label=f"engineering_fixture_{args.controller}; not trained; no algorithm performance claim")
            output = write_episode(record, args.output)
            result = {"output": str(output), "steps": 81, "frames": 82,
                      "controller": args.controller, "formal_training_started": False}
        finally:
            env.close()
    print(json.dumps(jsonable(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

