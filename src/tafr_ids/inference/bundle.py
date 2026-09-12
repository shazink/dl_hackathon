"""Checksum-verified loading for tracked inference-only assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import skops.io as skops_io
from safetensors.torch import load_file
import torch

from tafr_ids.data.inspection import FEATURES
from tafr_ids.data.loader import CLASSES
from tafr_ids.models.mlp import TabularMLP

METHODS = ("naive", "uniform", "tafr_f", "tafr_fu", "tafr")
ALLOWED_PREPROCESSOR_TYPES = frozenset(
    {"numpy.dtype", "tafr_ids.data.preprocessing.FrozenPreprocessor"}
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid inference metadata at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Inference metadata at {path} must be an object")
    return value


def _models_root(path: str | Path) -> Path:
    root = Path(path).resolve()
    if root.name in METHODS:
        root = root.parent
    if not (root / "shared" / "SHA256SUMS").is_file():
        raise ValueError(f"Inference asset manifest is unavailable under {root}")
    return root


def verify_asset_manifest(models_dir: str | Path) -> dict[str, str]:
    """Verify every tracked runtime asset declared in the checksum manifest."""
    root = _models_root(models_dir)
    manifest = root / "shared" / "SHA256SUMS"
    entries: dict[str, str] = {}
    try:
        lines = manifest.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Cannot read inference asset manifest: {error}") from error
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError("Inference asset manifest has an invalid line")
        expected, relative = parts
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError("Inference asset manifest escapes the models directory") from error
        if relative in entries or not candidate.is_file():
            raise ValueError(f"Inference asset is missing or duplicated: {relative}")
        actual = sha256(candidate)
        if actual != expected:
            raise ValueError(f"Inference asset checksum mismatch: {relative}")
        entries[relative] = actual
    expected_paths = {
        "shared/preprocessor.skops",
        "shared/feature_schema.json",
        "shared/class_mapping.json",
        "shared/simulation_scenarios.npz",
        *(f"{method}/model.safetensors" for method in METHODS),
        *(f"{method}/metadata.json" for method in METHODS),
    }
    if set(entries) != expected_paths:
        raise ValueError("Inference asset manifest has missing or unexpected entries")
    return entries


def load_preprocessor(path: str | Path):
    """Load the project preprocessor only when its skops type set is allowlisted."""
    source = Path(path)
    try:
        unknown = set(skops_io.get_untrusted_types(file=source))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
        raise ValueError(f"Frozen preprocessor is unavailable or incompatible: {error}") from error
    if unknown != ALLOWED_PREPROCESSOR_TYPES:
        raise ValueError(f"Frozen preprocessor contains unexpected types: {sorted(unknown)}")
    try:
        processor = skops_io.load(source, trusted=sorted(ALLOWED_PREPROCESSOR_TYPES))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
        raise ValueError(f"Frozen preprocessor is unavailable or incompatible: {error}") from error
    if tuple(getattr(processor, "feature_names_", ())) == ():
        raise ValueError("Frozen preprocessor has no transformed feature schema")
    return processor


def load_model_bundle(bundle_dir: str | Path) -> tuple[TabularMLP, dict[str, Any]]:
    """Verify and load one weights-only model bundle on CPU."""
    directory = Path(bundle_dir).resolve()
    method = directory.name
    if method not in METHODS:
        raise ValueError(f"Unknown inference method: {method}")
    root = _models_root(directory)
    checksums = verify_asset_manifest(root)
    metadata = _strict_json(directory / "metadata.json")
    class_mapping = _strict_json(root / "shared" / "class_mapping.json")
    feature_schema = _strict_json(root / "shared" / "feature_schema.json")
    try:
        if metadata["schema_version"] != 1 or metadata["method_id"] != method:
            raise ValueError("method metadata identity differs")
        if metadata["input_dim"] != feature_schema["input_dim"]:
            raise ValueError("input dimension differs from feature schema")
        if metadata["output_dim"] != len(CLASSES):
            raise ValueError("output dimension differs from class mapping")
        if feature_schema["raw_features"] != FEATURES:
            raise ValueError("raw feature order differs from package schema")
        if class_mapping["classes"] != list(CLASSES):
            raise ValueError("class order differs from package mapping")
        if class_mapping["class_to_index"] != {name: index for index, name in enumerate(CLASSES)}:
            raise ValueError("class mapping differs from package mapping")
        if metadata["sha256"]["model_weights"] != checksums[f"{method}/model.safetensors"]:
            raise ValueError("weight checksum differs from bundle metadata")
        expected_references = {
            "global_class_mapping": "../shared/class_mapping.json",
            "feature_schema": "../shared/feature_schema.json",
            "frozen_preprocessor": "../shared/preprocessor.skops",
        }
        if any(metadata[name] != expected for name, expected in expected_references.items()):
            raise ValueError("shared asset references differ from the bundle contract")
        model_config = metadata["model_architecture"]
        if model_config["normalization"] != "layer_norm" or model_config["activation"] != "relu":
            raise ValueError("unsupported model architecture")
        model = TabularMLP(
            metadata["input_dim"],
            len(CLASSES),
            model_config["hidden_dims"],
            model_config["dropout"],
        )
        state = load_file(directory / "model.safetensors", device="cpu")
        model.load_state_dict(state, strict=True)
    except (KeyError, TypeError, RuntimeError, ValueError, OSError) as error:
        raise ValueError(f"Inference bundle is unavailable or incompatible: {error}") from error
    model.eval()
    return model, metadata


@torch.inference_mode()
def predict_probabilities(features, bundle_dir: str | Path):
    model, metadata = load_model_bundle(bundle_dir)
    if (
        features.dtype.name != "float32"
        or features.ndim != 2
        or features.shape[1] != metadata["input_dim"]
    ):
        raise ValueError("Model input must be a two-dimensional float32 feature matrix")
    tensor = torch.from_numpy(features)
    if not torch.isfinite(tensor).all():
        raise ValueError("Model input must contain only finite values")
    probabilities = torch.softmax(model(tensor), dim=1).numpy()
    if not torch.isfinite(torch.from_numpy(probabilities)).all():
        raise ValueError("Model produced non-finite probabilities")
    return probabilities
