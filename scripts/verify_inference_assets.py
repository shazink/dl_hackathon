"""Verify tracked inference assets and smoke-test every finalized model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tafr_ids.inference.bundle import METHODS, load_model_bundle, load_preprocessor, predict_probabilities, verify_asset_manifest


def verify(models_dir: Path) -> None:
    checksums = verify_asset_manifest(models_dir)
    processor = load_preprocessor(models_dir / "shared" / "preprocessor.skops")
    if len(processor.feature_names_) != 156 or processor.fit_rows_ != 44210:
        raise ValueError("Frozen preprocessor scope or output dimension differs")
    with np.load(models_dir / "shared" / "simulation_scenarios.npz", allow_pickle=False) as archive:
        features = archive["X"]
    if features.dtype != np.float32 or features.shape != (80, 156) or not np.isfinite(features).all():
        raise ValueError("Tracked simulation scenarios differ from the verified export")
    for method in METHODS:
        model, metadata = load_model_bundle(models_dir / method)
        if model.parameter_count != 75146 or metadata["seed"] != 42:
            raise ValueError(f"{method} model architecture or seed differs")
        probabilities = predict_probabilities(features, models_dir / method)
        if probabilities.shape != (80, 10) or not np.isfinite(probabilities).all():
            raise ValueError(f"{method} inference output is invalid")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{method} probabilities do not sum to one")
    print(f"Verified {len(checksums)} checksums and finite inference for {len(METHODS)} models")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    verify(args.models_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
