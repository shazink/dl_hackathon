"""Strict dashboard data, upload, and local inference helpers."""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from tafr_ids.data.inspection import CATEGORICAL, FEATURES, TARGETS
from tafr_ids.data.loader import CLASSES
from tafr_ids.models.mlp import TabularMLP

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_UPLOAD_ROWS = 500
SIMULATION_SCHEMA_VERSION = 1


def load_dashboard_data(path: str | Path = "results/dashboard_data.json") -> dict | None:
    source = Path(path)
    if not source.is_file():
        return None
    try:
        data = json.loads(source.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Dashboard results are corrupt: {error}") from error
    if data.get("schema_version") != 1 or not isinstance(data.get("methods"), dict):
        raise ValueError("Dashboard results have an unsupported schema")
    return data


def schema_template() -> bytes:
    return (",".join(FEATURES) + "\n").encode()


def load_simulation_scenarios(path: str | Path) -> dict | None:
    """Load prepared-development feature vectors without permitting pickle data."""
    source = Path(path)
    if not source.is_file():
        return None
    try:
        with np.load(source, allow_pickle=False) as archive:
            required = {"schema_version", "X", "y", "source_indices", "experiences", "class_names"}
            if not required.issubset(archive.files):
                raise ValueError("missing required arrays")
            scenarios = {name: archive[name].copy() for name in required}
    except (OSError, ValueError) as error:
        raise ValueError(f"Simulation scenarios are corrupt: {error}") from error
    if int(scenarios["schema_version"]) != SIMULATION_SCHEMA_VERSION:
        raise ValueError("Simulation scenarios have an unsupported schema")
    features = scenarios["X"]
    labels = scenarios["y"]
    row_count = len(labels)
    if features.ndim != 2 or features.shape[1] != 156 or features.dtype != np.float32:
        raise ValueError("Simulation scenarios must contain finite 156-column float32 features")
    if not np.isfinite(features).all():
        raise ValueError("Simulation scenarios contain non-finite features")
    if labels.dtype != np.int64 or any(len(scenarios[name]) != row_count for name in ("source_indices", "experiences", "class_names")):
        raise ValueError("Simulation scenario metadata is inconsistent")
    if row_count == 0 or not np.isin(labels, np.arange(len(CLASSES))).all():
        raise ValueError("Simulation scenario labels are invalid")
    expected_names = np.asarray([CLASSES[int(label)] for label in labels])
    if not np.array_equal(scenarios["class_names"], expected_names):
        raise ValueError("Simulation scenario class names do not match labels")
    return scenarios


def validate_upload(content: bytes) -> pd.DataFrame:
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload exceeds the 5 MB limit")
    try:
        frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    except Exception as error:
        raise ValueError("Malformed CSV upload") from error
    forbidden = (set(frame.columns) & (TARGETS | {"id"}))
    if forbidden:
        raise ValueError("Targets and IDs are prohibited: " + ", ".join(sorted(forbidden)))
    if list(frame.columns) != FEATURES:
        missing = sorted(set(FEATURES) - set(frame.columns))
        unexpected = sorted(set(frame.columns) - set(FEATURES))
        raise ValueError(f"Unexpected feature schema; missing={missing}, unexpected={unexpected}")
    if frame.empty or len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError("Upload must contain between 1 and 500 rows")
    result = frame.copy()
    for name in FEATURES:
        if (result[name].str.strip() == "").any():
            raise ValueError(f"Missing value in {name}")
        if name not in CATEGORICAL:
            result[name] = pd.to_numeric(result[name], errors="raise").astype(np.float64)
            if not np.isfinite(result[name].to_numpy()).all():
                raise ValueError(f"Non-finite value in {name}")
    return result


def _load_model(features: np.ndarray, checkpoint: str | Path) -> TabularMLP:
    if features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError("Model input must be finite float32 features")
    try:
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        model_config = state["config"]["model"]
        model = TabularMLP(features.shape[1], len(CLASSES), model_config["hidden_dims"], model_config["dropout"])
        model.load_state_dict(state["model"])
    except (OSError, KeyError, TypeError, RuntimeError, ValueError) as error:
        raise ValueError(f"Checkpoint is unavailable or incompatible: {error}") from error
    model.eval()
    return model


@torch.inference_mode()
def predict_features(features: np.ndarray, checkpoint: str | Path) -> pd.DataFrame:
    model = _load_model(features, checkpoint)
    probabilities = torch.softmax(model(torch.from_numpy(features)), dim=1).numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced non-finite probabilities")
    predictions = probabilities.argmax(axis=1)
    output = pd.DataFrame({"predicted_class": [CLASSES[index] for index in predictions], "confidence": probabilities.max(axis=1)})
    for index, name in enumerate(CLASSES):
        output[f"p_{name}"] = probabilities[:, index]
    return output


def predict_upload(
    frame: pd.DataFrame, checkpoint: str | Path, preprocessor_path: str | Path
) -> pd.DataFrame:
    try:
        processor = joblib.load(Path(preprocessor_path))
        features = processor.transform(frame)
    except (OSError, ValueError) as error:
        raise ValueError(f"Frozen preprocessor is unavailable or incompatible: {error}") from error
    if features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError("Frozen preprocessing produced invalid features")
    return predict_features(features, checkpoint)
