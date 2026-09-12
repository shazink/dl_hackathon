"""Export safe, minimal runtime assets from finalized trusted local artifacts.

This utility reads existing results and checkpoints. It does not load any dataset,
fit a transformer, train a model, or evaluate a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import shutil

import joblib
import numpy as np
from safetensors.torch import load_file, save_file
import skops.io as skops_io
import torch

from tafr_ids.data.inspection import CATEGORICAL, FEATURES
from tafr_ids.data.loader import CLASSES
from tafr_ids.inference.bundle import ALLOWED_PREPROCESSOR_TYPES, METHODS, sha256


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))


def _canonical_fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _metric_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = {}
        for row in csv.DictReader(stream):
            values = {}
            for name, value in row.items():
                if name == "method":
                    continue
                values[name] = None if value == "" else int(value) if name == "seed" else float(value)
            rows[row["method"]] = values
    return rows


def _validate_scenarios(path: Path, input_dim: int) -> None:
    with np.load(path, allow_pickle=False) as archive:
        required = {"schema_version", "X", "y", "source_indices", "experiences", "class_names"}
        if set(archive.files) != required:
            raise ValueError("Simulation scenario archive differs from the expected inference-only schema")
        features, labels = archive["X"], archive["y"]
        if int(archive["schema_version"]) != 1 or features.dtype != np.float32:
            raise ValueError("Simulation scenario schema or dtype differs")
        if features.ndim != 2 or features.shape[1] != input_dim or not np.isfinite(features).all():
            raise ValueError("Simulation scenarios are not finite model inputs")
        if labels.dtype != np.int64 or not np.isin(labels, np.arange(len(CLASSES))).all():
            raise ValueError("Simulation scenario labels differ from the global mapping")
        if not np.array_equal(archive["class_names"], np.asarray([CLASSES[index] for index in labels])):
            raise ValueError("Simulation scenario names differ from the global mapping")
        if any(len(archive[name]) != len(labels) for name in ("source_indices", "experiences", "class_names")):
            raise ValueError("Simulation scenario row metadata is inconsistent")


def export(
    source_dir: Path,
    prepared_dir: Path,
    scenario_path: Path,
    output_dir: Path,
    results_dir: Path,
) -> None:
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    metadata_path = prepared_dir / "metadata.json"
    prepared = _json(metadata_path)
    input_dim = len(prepared["feature_names"])
    if input_dim != 156 or prepared["fit_scope"] != "E1 development only":
        raise ValueError("Prepared metadata is not the finalized E1-only 156-feature artifact")
    if prepared["files"]["preprocessor.joblib"] != sha256(prepared_dir / "preprocessor.joblib"):
        raise ValueError("Source preprocessor checksum differs from prepared metadata")
    processor = joblib.load(prepared_dir / "preprocessor.joblib")
    if tuple(processor.feature_names_) != tuple(prepared["feature_names"]):
        raise ValueError("Source preprocessor feature order differs from prepared metadata")
    _validate_scenarios(scenario_path, input_dim)
    validation_rows = _metric_rows(results_dir / "validation_benchmark_seed42.csv")
    test_rows = _metric_rows(results_dir / "final_test_seed42.csv")

    staging = output_dir.with_name(f".{output_dir.name}-export")
    if staging.exists():
        raise ValueError(f"Staging directory already exists: {staging}")
    shared = staging / "shared"
    shared.mkdir(parents=True)
    try:
        preprocessor_out = shared / "preprocessor.skops"
        skops_io.dump(processor, preprocessor_out)
        unknown = set(skops_io.get_untrusted_types(file=preprocessor_out))
        if unknown != ALLOWED_PREPROCESSOR_TYPES:
            raise ValueError(f"Exported preprocessor has unexpected types: {sorted(unknown)}")
        restored = skops_io.load(preprocessor_out, trusted=sorted(ALLOWED_PREPROCESSOR_TYPES))
        if tuple(restored.feature_names_) != tuple(prepared["feature_names"]):
            raise ValueError("Round-tripped preprocessor changed feature order")

        (shared / "feature_schema.json").write_text(json.dumps({
            "schema_version": 1,
            "raw_features": FEATURES,
            "raw_categorical_features": [name for name in FEATURES if name in CATEGORICAL],
            "raw_numeric_features": [name for name in FEATURES if name not in CATEGORICAL],
            "transformed_features": prepared["feature_names"],
            "input_dim": input_dim,
            "output_dtype": "float32",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (shared / "class_mapping.json").write_text(json.dumps({
            "schema_version": 1,
            "classes": list(CLASSES),
            "class_to_index": {name: index for index, name in enumerate(CLASSES)},
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.copyfile(scenario_path, shared / "simulation_scenarios.npz")

        shared_hashes = {
            name: sha256(shared / name)
            for name in ("preprocessor.skops", "feature_schema.json", "class_mapping.json", "simulation_scenarios.npz")
        }
        for method in METHODS:
            source = source_dir / method
            result = _json(source / "results.json")
            checkpoint_path = source / "checkpoints" / "latest.pt"
            checkpoint_sha = sha256(checkpoint_path)
            results_sha = sha256(source / "results.json")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if (
                result["status"] != "completed"
                or result["method"] != method
                or checkpoint.get("completed_experience") != 4
                or checkpoint.get("completed_epoch") != 20
                or checkpoint.get("config") != result["config"]
                or checkpoint.get("config_fingerprint") != result["config_fingerprint"]
                or checkpoint.get("prepared_data_fingerprint") != result["prepared_data_fingerprint"]
            ):
                raise ValueError(f"{method} source is not its finalized fingerprint-bound checkpoint")
            method_dir = staging / method
            method_dir.mkdir()
            tensor_state = {name: tensor.detach().cpu().contiguous() for name, tensor in checkpoint["model"].items()}
            save_file(tensor_state, method_dir / "model.safetensors")
            restored_state = load_file(method_dir / "model.safetensors")
            if tensor_state.keys() != restored_state.keys() or any(
                not torch.equal(tensor_state[name], restored_state[name]) for name in tensor_state
            ):
                raise ValueError(f"{method} weights changed during safetensors export")
            source_fingerprint_fields = {
                "checkpoint_sha256": checkpoint_sha,
                "results_sha256": results_sha,
                "configuration_fingerprint": result["config_fingerprint"],
                "prepared_data_fingerprint": result["prepared_data_fingerprint"],
            }
            bundle_metadata = {
                "schema_version": 1,
                "method_id": method,
                "seed": result["seed"],
                "input_dim": input_dim,
                "output_dim": len(CLASSES),
                "model_architecture": result["config"]["model"],
                "global_class_mapping": "../shared/class_mapping.json",
                "feature_schema": "../shared/feature_schema.json",
                "frozen_preprocessor": "../shared/preprocessor.skops",
                "source_experiment_fingerprint": _canonical_fingerprint(source_fingerprint_fields),
                "source_experiment": source_fingerprint_fields,
                "source_configuration_fingerprint": result["config_fingerprint"],
                "dataset_fingerprint": result["prepared_dataset_fingerprint"],
                "preparation_fingerprint": result["prepared_data_fingerprint"],
                "preparation_protocol_fingerprints": result["prepared_protocol_fingerprints"],
                "metric_references": {
                    "validation": {"path": "results/validation_benchmark_seed42.csv", "values": validation_rows[method]},
                    "final_test": {"path": "results/final_test_seed42.csv", "values": test_rows[method]},
                },
                "serialization": {
                    "model_format": "safetensors",
                    "model_format_schema_version": 1,
                    "safetensors_version": version("safetensors"),
                    "preprocessor_format": "skops",
                    "preprocessor_format_schema_version": 1,
                    "skops_version": version("skops"),
                    "source_pytorch_version": result["environment"]["pytorch"],
                    "export_pytorch_version": torch.__version__,
                },
                "sha256": {
                    "model_weights": sha256(method_dir / "model.safetensors"),
                    "preprocessor": shared_hashes["preprocessor.skops"],
                    "feature_schema": shared_hashes["feature_schema.json"],
                    "class_mapping": shared_hashes["class_mapping.json"],
                },
                "contents": "Final model tensors only; no optimizer, replay buffer, RNG state, samples, or training history.",
            }
            (method_dir / "metadata.json").write_text(
                json.dumps(bundle_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

        checksum_paths = [
            shared / "preprocessor.skops",
            shared / "feature_schema.json",
            shared / "class_mapping.json",
            shared / "simulation_scenarios.npz",
            *(staging / method / name for method in METHODS for name in ("model.safetensors", "metadata.json")),
        ]
        lines = [f"{sha256(path)}  {path.relative_to(staging).as_posix()}" for path in checksum_paths]
        (shared / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/experiments/benchmark_seed42"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("artifacts/data/prepared"))
    parser.add_argument("--scenarios", type=Path, default=Path("artifacts/simulation/scenarios.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    export(args.source_dir, args.prepared_dir, args.scenarios, args.output_dir, args.results_dir)
    print(f"Exported verified inference-only assets to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
