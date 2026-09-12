"""One-time logical-test loading and and finalized-checkpoint evaluation."""

from __future__ import annotations

import tomllib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from tafr_ids.data.inspection import HEADER, sha256
from tafr_ids.data.loader import CLASSES, CLASS_TO_INDEX, feature_frame, normalize_targets
from tafr_ids.models.mlp import TabularMLP
from tafr_ids.training.checkpoint import load_checkpoint
from tafr_ids.training.trainer import evaluate


def load_logical_test_once(data_dir: Path, split_manifest: Path, preprocessor_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Hash, load, schema-check, and transform the logical test exactly once per call."""
    with split_manifest.open("rb") as stream:
        manifest = tomllib.load(stream)
    expected = manifest["testing"]
    source = data_dir.resolve() / expected["physical_filename"]
    if not source.is_file() or sha256(source) != expected["sha256"]:
        raise ValueError("Logical-test source is missing or fails its pinned SHA-256")
    frame = pd.read_csv(source, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    if list(frame.columns) != HEADER or len(frame) != expected["expected_rows"]:
        raise ValueError("Logical-test schema or row count differs from the manifest")
    targets = normalize_targets(frame["attack_cat"])
    binary = frame["label"].str.strip().to_numpy()
    if not np.array_equal(binary, np.where(targets == "Normal", "0", "1")):
        raise ValueError("Logical-test binary label disagrees with its multiclass target")
    processor = joblib.load(preprocessor_path)
    transformed = processor.transform(feature_frame(frame))
    labels = np.asarray([CLASS_TO_INDEX[name] for name in targets], dtype=np.int64)
    if transformed.dtype != np.float32 or not np.isfinite(transformed).all() or sha256(source) != expected["sha256"]:
        raise ValueError("Logical-test transform is invalid or source changed during access")
    return transformed, labels, {"rows": len(labels), "features": transformed.shape[1], "sha256": expected["sha256"], "physical_filename": expected["physical_filename"]}


def evaluate_final_checkpoint(
    checkpoint_path: Path,
    *,
    config: dict,
    config_fingerprint: str,
    prepared_fingerprint: str,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict:
    state = load_checkpoint(
        checkpoint_path, config_fingerprint=config_fingerprint,
        prepared_data_fingerprint=prepared_fingerprint, map_location=torch.device("cpu"),
    )
    if state.get("completed_experience") != 4 or state.get("config") != config:
        raise ValueError("Final checkpoint is incomplete or has a configuration mismatch")
    model = TabularMLP(features.shape[1], len(CLASSES), config["model"]["hidden_dims"], config["model"]["dropout"])
    model.load_state_dict(state["model"])
    return evaluate(model, [(features, labels)], torch.device("cpu"), config["training"]["batch_size"])
