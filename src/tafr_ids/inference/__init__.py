"""Portable inference-only bundle loading."""

from tafr_ids.inference.bundle import (
    METHODS,
    load_model_bundle,
    load_preprocessor,
    verify_asset_manifest,
)

__all__ = ["METHODS", "load_model_bundle", "load_preprocessor", "verify_asset_manifest"]
