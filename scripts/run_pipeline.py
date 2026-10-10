"""CLI entry point: run the full forecasting pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running as `python scripts/run_pipeline.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Athlete career forecasting pipeline")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--quick", action="store_true",
                        help="small dataset + few epochs (smoke run)")
    parser.add_argument("--event", choices=["100m", "800m"], default=None)
    parser.add_argument("--n-athletes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--root", default=None, help="project root (default: repo root)")
    args = parser.parse_args()

    overrides = {}
    if args.event:
        overrides["event"] = args.event
    if args.n_athletes:
        overrides.setdefault("data", {})["n_synthetic_athletes"] = args.n_athletes
    if args.epochs:
        overrides.setdefault("model", {})["epochs"] = args.epochs
    if args.quick:
        overrides.setdefault("data", {})["n_synthetic_athletes"] = 600
        overrides.setdefault("model", {})["epochs"] = 12
        overrides.setdefault("model", {})["early_stop_patience"] = 4

    run_pipeline(args.config if args.config != "config.yaml" else None, overrides, args.root)


if __name__ == "__main__":
    main()
