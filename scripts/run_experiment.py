#!/usr/bin/env python3
"""Run one fingerprint-validated continual-learning experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tafr_ids.training.config import load_experiment_config
from tafr_ids.training.reproducibility import configure_determinism
from tafr_ids.training.trainer import METHODS, run_experiment, validate_prepared_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    try:
        config = load_experiment_config(args.config.resolve())
        if args.method != config["method"]:
            raise ValueError("--method does not match the experiment configuration")
        prepared = validate_prepared_data(args.prepared_dir, config, repository)
        prepared_path, output_path = args.prepared_dir.resolve(), args.output_dir.resolve()
        if output_path == prepared_path or prepared_path in output_path.parents or output_path in prepared_path.parents:
            raise ValueError("Experiment output must be separate from prepared data")
        configure_determinism(config["seed"])
        report = run_experiment(config=config, prepared=prepared, output_dir=output_path, repository=repository, resume=args.resume)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Experiment blocked: {error}", file=sys.stderr)
        return 2
    metrics = report["continual_metrics"]
    print(f"Completed {args.method} seed-42; final average balanced accuracy={metrics['final_average_balanced_accuracy']:.6f}; artifacts={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
