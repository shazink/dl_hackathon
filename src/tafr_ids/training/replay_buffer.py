"""Fixed-capacity replay memory with deterministic full replacement."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def selected_ids_fingerprint(sample_ids: np.ndarray) -> str:
    values = np.sort(np.asarray(sample_ids, dtype=np.int64))
    return hashlib.sha256(values.astype("<i8", copy=False).tobytes()).hexdigest()


def selection_seed(global_seed: int, experience_id: str, class_id: int, version: int) -> int:
    payload = f"{global_seed}|{experience_id}|{class_id}|{version}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass
class ReplayBuffer:
    features: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    source_experiences: np.ndarray

    @classmethod
    def empty(cls, input_dim: int) -> "ReplayBuffer":
        return cls(
            np.empty((0, input_dim), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int16),
        )

    def validate(self, capacity: int | None = None) -> None:
        size = len(self.sample_ids)
        if self.features.dtype != np.float32 or self.features.ndim != 2 or not np.isfinite(self.features).all():
            raise ValueError("Replay features must be a finite float32 matrix")
        if self.labels.dtype != np.int64 or self.sample_ids.dtype != np.int64:
            raise ValueError("Replay labels and IDs must be int64")
        if len(self.labels) != size or len(self.features) != size or len(self.source_experiences) != size:
            raise ValueError("Replay buffer arrays have inconsistent sizes")
        if len(np.unique(self.sample_ids)) != size:
            raise ValueError("Replay buffer contains duplicate stable sample IDs")
        if capacity is not None and size > capacity:
            raise ValueError("Replay buffer exceeds capacity")

    def checkpoint_state(self) -> dict:
        self.validate()
        return {
            "features": self.features,
            "labels": self.labels,
            "sample_ids": self.sample_ids,
            "source_experiences": self.source_experiences,
            "selected_ids_fingerprint": selected_ids_fingerprint(self.sample_ids),
        }

    @classmethod
    def from_checkpoint(cls, state: dict, capacity: int) -> "ReplayBuffer":
        result = cls(state["features"], state["labels"], state["sample_ids"], state["source_experiences"])
        result.validate(capacity)
        if selected_ids_fingerprint(result.sample_ids) != state.get("selected_ids_fingerprint"):
            raise ValueError("Replay-buffer selected-ID fingerprint mismatch")
        return result

    def candidates(
        self, current_features: np.ndarray, current_labels: np.ndarray, current_ids: np.ndarray, experience: int
    ) -> "ReplayBuffer":
        current_sources = np.full(len(current_ids), experience, dtype=np.int16)
        result = ReplayBuffer(
            np.concatenate((self.features, current_features)),
            np.concatenate((self.labels, current_labels)),
            np.concatenate((self.sample_ids, current_ids.astype(np.int64, copy=False))),
            np.concatenate((self.source_experiences, current_sources)),
        )
        result.validate()
        return result

    def select(self, quotas: dict[int, int], global_seed: int, experience_id: str, version: int) -> "ReplayBuffer":
        chosen: list[np.ndarray] = []
        for class_id in sorted(quotas):
            class_rows = np.flatnonzero(self.labels == class_id)
            class_rows = class_rows[np.argsort(self.sample_ids[class_rows], kind="stable")]
            generator = np.random.default_rng(selection_seed(global_seed, experience_id, class_id, version))
            permuted = class_rows[generator.permutation(len(class_rows))]
            chosen.append(permuted[: quotas[class_id]])
        rows = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
        result = ReplayBuffer(
            self.features[rows].copy(), self.labels[rows].copy(), self.sample_ids[rows].copy(),
            self.source_experiences[rows].copy(),
        )
        result.validate()
        return result
