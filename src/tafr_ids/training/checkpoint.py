"""Atomic, fingerprint-bound experiment checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def atomic_save(state: dict[str, Any], destination: str | Path) -> int:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(state, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path.stat().st_size


def load_checkpoint(
    path: str | Path,
    *,
    config_fingerprint: str,
    prepared_data_fingerprint: str,
    map_location: torch.device,
) -> dict[str, Any]:
    state = torch.load(Path(path), map_location=map_location, weights_only=False)
    if state.get("config_fingerprint") != config_fingerprint:
        raise ValueError("Checkpoint configuration fingerprint does not match")
    if state.get("prepared_data_fingerprint") != prepared_data_fingerprint:
        raise ValueError("Checkpoint prepared-data fingerprint does not match")
    return state
