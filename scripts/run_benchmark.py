#!/usr/bin/env python3
"""Run the frozen five-method benchmark, audit gate, and one-time final evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tafr_ids.evaluation.final import evaluate_final_checkpoint, load_logical_test_once
from tafr_ids.evaluation.reporting import write_json_atomic
from tafr_ids.training.config import load_experiment_config, validate_config_fairness
from tafr_ids.training.trainer import METHODS, canonical_fingerprint, validate_prepared_data


def _source_fingerprint(repository: Path) -> str:
    digest = hashlib.sha256()
    roots = ("src", "scripts", "configs", "dashboard", "tests")
    for relative in roots:
        for path in sorted((repository / relative).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".toml"}:
                digest.update(str(path.relative_to(repository)).encode() + b"\0")
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _strict_finite(value) -> bool:
    if isinstance(value, dict):
        return all(_strict_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_strict_finite(item) for item in value)
    if isinstance(value, float):
        return bool(np.isfinite(value))
    return True


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validation_row(report: dict) -> dict:
    metrics = report["continual_metrics"]
    return {
        "method": report["method"],
        "seed": report["seed"],
        "final_average_accuracy": metrics["final_average_task_accuracy"],
        "final_average_balanced_accuracy": metrics["final_average_balanced_accuracy"],
        "final_seen_class_macro_f1": metrics["final_seen_class_macro_f1"],
        "accuracy_forgetting": metrics["accuracy_average_forgetting"],
        "balanced_accuracy_forgetting": metrics["average_forgetting"],
        "forward_transfer": metrics["forward_transfer"],
        "runtime_seconds": report["resources"]["total_runtime_seconds"],
        "peak_cpu_memory_bytes": report["resources"]["peak_cpu_resident_memory_bytes"],
        "peak_cuda_allocated_bytes": report["resources"]["peak_cuda_allocated_bytes"],
    }


def _audit(
    repository: Path, prepared_dir: Path, configs: dict[str, dict], reports: dict[str, dict],
    source_before: str, access_log: dict, pytest_output: str,
) -> dict:
    failures = []
    fairness = validate_config_fairness(configs)
    if "passed" not in pytest_output:
        failures.append("complete test suite did not report a passing result")
    if set(reports) != set(METHODS) or any(report.get("status") != "completed" for report in reports.values()):
        failures.append("all five validation runs did not complete")
    if not all(_strict_finite(report) for report in reports.values()):
        failures.append("a validation artifact contains NaN or Infinity")
    prepared_fingerprints = {report.get("prepared_data_fingerprint") for report in reports.values()}
    protocol_fingerprints = {json.dumps(report.get("prepared_protocol_fingerprints"), sort_keys=True) for report in reports.values()}
    if len(prepared_fingerprints) != 1 or len(protocol_fingerprints) != 1:
        failures.append("prepared-data or protocol fingerprints differ")
    reference = reports["naive"]["history"]["evaluations"]
    for method in METHODS[1:]:
        actual = reports[method]["history"]["evaluations"]
        if actual["random_init"] != reference["random_init"] or actual["after_E1"] != reference["after_E1"]:
            failures.append(f"{method} E1 results differ from Naive")
    for method in METHODS[1:]:
        updates = reports[method]["history"].get("replay_updates", {})
        if set(updates) != set(("E1", "E2", "E3", "E4")):
            failures.append(f"{method} replay update history is incomplete")
            continue
        for number, experience in enumerate(("E1", "E2", "E3", "E4"), 1):
            update = updates[experience]
            if update["buffer_size"] != 2000 or not update["unique_ids"] or not update["eligible_only"]:
                failures.append(f"{method} {experience} violates capacity, uniqueness, or eligibility")
            if any(int(source.removeprefix("E")) > number for source in update["source_experience_counts"]):
                failures.append(f"{method} {experience} has future provenance")
            if sum(int(value) for value in update["actual_quotas"].values()) != update["buffer_size"]:
                failures.append(f"{method} {experience} quota total differs from occupancy")
    if access_log.get("events"):
        failures.append("project test-access log is not empty before the audit")
    if any("test" in name.lower() for name in reports["naive"].get("config", {}).get("data", {}) if name not in {"split_manifest"}):
        failures.append("unexpected logical-test data configuration")
    if _source_fingerprint(repository) != source_before:
        failures.append("source or configuration changed during validation benchmark")
    diff_check = subprocess.run(("git", "diff", "--check"), cwd=repository, capture_output=True, text=True)
    if diff_check.returncode:
        failures.append("git diff --check failed before test access")
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "tests": pytest_output.strip().splitlines()[-1],
        "five_runs_completed": len(reports) == 5,
        "strict_finite": all(_strict_finite(report) for report in reports.values()),
        "fingerprints_match": len(prepared_fingerprints) == len(protocol_fingerprints) == 1,
        "e1_strict_match": not any("E1 results differ" in failure for failure in failures),
        "fairness": fairness,
        "replay_capacity": 2000,
        "project_test_access_events_before_gate": access_log.get("events", []),
        "source_config_fingerprint": source_before,
        "source_config_frozen": _source_fingerprint(repository) == source_before,
        "git_diff_check": "passed" if diff_check.returncode == 0 else diff_check.stdout + diff_check.stderr,
        "rerun_policy": "fresh output directories; one invocation per method; no result-dependent changes or reruns",
        "prepared_directory": str(prepared_dir.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/benchmark_seed42"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    output = (repository / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    results_dir = (repository / args.results_dir).resolve() if not args.results_dir.is_absolute() else args.results_dir.resolve()
    try:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Benchmark output directory must be new and empty; result-dependent reruns are prohibited")
        output.mkdir(parents=True, exist_ok=True)
        access_path = output / "test_access_log.json"
        access_log = {"schema_version": 1, "events": []}
        write_json_atomic(access_path, access_log)
        configs = {method: load_experiment_config(repository / "configs" / "experiments" / f"{method}.toml") for method in METHODS}
        validate_config_fairness(configs)
        prepared = validate_prepared_data(args.prepared_dir.resolve(), configs["naive"], repository)
        source_before = _source_fingerprint(repository)
        test_run = subprocess.run((sys.executable, "-m", "pytest", "-q"), cwd=repository, check=True, capture_output=True, text=True)
        reports = {}
        for method in METHODS:
            destination = output / method
            command = (
                sys.executable, "scripts/run_experiment.py", "--method", method, "--config",
                f"configs/experiments/{method}.toml", "--prepared-dir", str(args.prepared_dir.resolve()),
                "--output-dir", str(destination),
            )
            completed = subprocess.run(command, cwd=repository, check=True, capture_output=True, text=True)
            print(completed.stdout.strip(), flush=True)
            reports[method] = json.loads((destination / "results.json").read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        validation_rows = [_validation_row(reports[method]) for method in METHODS]
        _write_csv(results_dir / "validation_benchmark_seed42.csv", validation_rows, list(validation_rows[0]))
        audit = _audit(repository, args.prepared_dir, configs, reports, source_before, access_log, test_run.stdout)
        write_json_atomic(output / "pre_test_audit.json", audit)
        if audit["status"] != "passed":
            raise ValueError("Scientific audit gate failed: " + "; ".join(audit["failures"]))

        ranked = sorted(
            validation_rows,
            key=lambda row: (-row["final_average_balanced_accuracy"], -row["final_seen_class_macro_f1"], row["balanced_accuracy_forgetting"], row["method"]),
        )
        selected = ranked[0]
        selection = {
            "method": selected["method"],
            "rule": ["highest final average validation balanced accuracy", "higher final seen-class validation Macro-F1", "lower balanced-accuracy forgetting", "lexical method ID"],
            "evidence": selected,
            "ranking": [row["method"] for row in ranked],
            "frozen_before_test_access": True,
        }
        write_json_atomic(output / "selected_method_pre_test.json", selection)

        access_log["events"].append({"event": "logical_test_load_started", "timestamp": datetime.now(timezone.utc).isoformat(), "selection_sha256": hashlib.sha256((output / "selected_method_pre_test.json").read_bytes()).hexdigest()})
        write_json_atomic(access_path, access_log)
        test_features, test_labels, test_identity = load_logical_test_once(
            args.data_dir, repository / configs["naive"]["data"]["split_manifest"],
            args.prepared_dir.resolve() / "preprocessor.joblib",
        )
        access_log["events"].append({"event": "logical_test_loaded_and_transformed", "timestamp": datetime.now(timezone.utc).isoformat(), **test_identity})
        write_json_atomic(access_path, access_log)
        test_metrics = {}
        for method in METHODS:
            access_log["events"].append({"event": "evaluation_started", "method": method, "timestamp": datetime.now(timezone.utc).isoformat()})
            write_json_atomic(access_path, access_log)
            report = reports[method]
            metrics = evaluate_final_checkpoint(
                output / method / "checkpoints" / "latest.pt", config=configs[method],
                config_fingerprint=report["config_fingerprint"], prepared_fingerprint=prepared.fingerprint,
                features=test_features, labels=test_labels,
            )
            if not _strict_finite(metrics):
                raise ValueError(f"Non-finite official-test result for {method}")
            test_metrics[method] = metrics
            write_json_atomic(output / method / "test_metrics.json", metrics)
            access_log["events"].append({"event": "evaluation_completed", "method": method, "timestamp": datetime.now(timezone.utc).isoformat()})
            write_json_atomic(access_path, access_log)
        test_rows = [{"method": method, "seed": 42, "accuracy": test_metrics[method]["accuracy"], "balanced_accuracy": test_metrics[method]["balanced_accuracy"], "macro_f1": test_metrics[method]["macro_f1"], "weighted_f1": test_metrics[method]["weighted_f1"]} for method in METHODS]
        _write_csv(results_dir / "final_test_seed42.csv", test_rows, list(test_rows[0]))
        comparison = []
        for validation in validation_rows:
            test = test_metrics[validation["method"]]
            comparison.append({
                "method": validation["method"],
                "validation_accuracy": validation["final_average_accuracy"],
                "validation_balanced_accuracy": validation["final_average_balanced_accuracy"],
                "validation_macro_f1": validation["final_seen_class_macro_f1"],
                "balanced_accuracy_forgetting": validation["balanced_accuracy_forgetting"],
                "forward_transfer": validation["forward_transfer"],
                "test_accuracy": test["accuracy"], "test_balanced_accuracy": test["balanced_accuracy"],
                "test_macro_f1": test["macro_f1"], "test_weighted_f1": test["weighted_f1"],
                "runtime_seconds": validation["runtime_seconds"], "peak_cpu_memory_bytes": validation["peak_cpu_memory_bytes"],
            })
        dashboard = {
            "schema_version": 1,
            "selection": {**selection, "explanation": f"{selection['method']} was frozen before test access by the declared validation ranking rule."},
            "comparison": comparison,
            "methods": {
                method: {
                    "validation": {
                        "matrices": reports[method]["matrices"], "matrix_axes": reports[method]["matrix_axes"],
                        "continual_metrics": reports[method]["continual_metrics"],
                        "history": {"evaluations": reports[method]["history"]["evaluations"], "replay_updates": reports[method]["history"]["replay_updates"]},
                    },
                    "test": test_metrics[method],
                }
                for method in METHODS
            },
            "test_identity": test_identity,
            "final_test_status": "consumed_and_finalized",
        }
        write_json_atomic(results_dir / "dashboard_data.json", dashboard)
        write_json_atomic(output / "benchmark_complete.json", {"status": "completed", "audit": audit, "selection": selection, "test_identity": test_identity, "test_access_log": access_log, "source_config_unchanged": _source_fingerprint(repository) == source_before})
        print(f"Benchmark complete; selected={selection['method']}; official test consumed and finalized", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, KeyError) as error:
        print(f"Benchmark blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
