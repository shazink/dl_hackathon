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


@torch.inference_mode()
def predict_upload(
    frame: pd.DataFrame, checkpoint: str | Path, preprocessor_path: str | Path
) -> pd.DataFrame:
    processor = joblib.load(Path(preprocessor_path))
    features = processor.transform(frame)
    if features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError("Frozen preprocessing produced invalid features")
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    config = state.get("config", {})
    model = TabularMLP(features.shape[1], len(CLASSES), config["model"]["hidden_dims"], config["model"]["dropout"])
    model.load_state_dict(state["model"])
    model.eval()
    probabilities = torch.softmax(model(torch.from_numpy(features)), dim=1).numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced non-finite probabilities")
    predictions = probabilities.argmax(axis=1)
    output = pd.DataFrame({"predicted_class": [CLASSES[index] for index in predictions], "confidence": probabilities.max(axis=1)})
    for index, name in enumerate(CLASSES):
        output[f"p_{name}"] = probabilities[:, index]
    return output
