#!/usr/bin/env python3
"""Prepare safe dashboard scenarios from development-only transformed arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tafr_ids.data.loader import CLASSES

SIMULATION_SCHEMA_VERSION = 1


def prepare(prepared_dir: Path, output: Path, samples_per_class: int = 8) -> int:
    if samples_per_class < 1:
        raise ValueError("--samples-per-class must be positive")
    candidates_by_class: dict[int, list[tuple[int, int, int, np.ndarray]]] = {}
    for experience in range(1, 5):
        prefix = prepared_dir / f"E{experience}_development"
        try:
            features = np.load(prefix.with_name(prefix.name + "_X.npy"), allow_pickle=False)
            labels = np.load(prefix.with_name(prefix.name + "_y.npy"), allow_pickle=False)
            indices = np.load(prefix.with_name(prefix.name + "_indices.npy"), allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"Prepared development artifacts are unavailable: {error}") from error
        if features.ndim != 2 or features.shape[1] != 156 or features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError(f"E{experience} development features are invalid")
        if labels.dtype != np.int64 or indices.dtype != np.int64 or len(features) != len(labels) or len(labels) != len(indices):
            raise ValueError(f"E{experience} development arrays are inconsistent")
        for label in np.unique(labels):
            positions = np.flatnonzero(labels == label)
            selected = positions[np.argsort(indices[positions], kind="stable")]
            candidates_by_class.setdefault(int(label), []).extend(
                (int(label), experience, int(indices[position]), features[position]) for position in selected
            )
    rows: list[tuple[int, int, int, np.ndarray]] = []
    for candidates in candidates_by_class.values():
        candidates.sort(key=lambda item: (item[2], item[1]))
        rows.extend(candidates[:samples_per_class])
    rows.sort(key=lambda item: (item[0], item[2], item[1]))
    present = {row[0] for row in rows}
    if present != set(range(len(CLASSES))):
        raise ValueError(f"Prepared development data does not cover every class: found {sorted(present)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(SIMULATION_SCHEMA_VERSION, dtype=np.int64),
        X=np.stack([row[3] for row in rows]).astype(np.float32, copy=False),
        y=np.asarray([row[0] for row in rows], dtype=np.int64),
        source_indices=np.asarray([row[2] for row in rows], dtype=np.int64),
        experiences=np.asarray([f"E{row[1]}" for row in rows]),
        class_names=np.asarray([CLASSES[row[0]] for row in rows]),
    )
    temporary.replace(output)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=8)
    args = parser.parse_args()
    count = prepare(args.prepared_dir, args.output, args.samples_per_class)
    print(f"Wrote {count} development-only scenarios to {args.output}")


if __name__ == "__main__":
    main()
