"""Deterministic current/replay mini-batch composition."""

from __future__ import annotations

import numpy as np
import torch

from tafr_ids.training.replay_buffer import ReplayBuffer
from tafr_ids.training.strategies.naive import NaiveStrategy


class UniformReplayStrategy(NaiveStrategy):
    def mixed_batches(
        self,
        current_features: np.ndarray,
        current_labels: np.ndarray,
        buffer: ReplayBuffer,
        *,
        seed: int,
    ):
        if not len(buffer.sample_ids):
            raise ValueError("Replay mixing requires a nonempty fixed buffer")
        generator = np.random.default_rng(seed)
        order = generator.permutation(len(current_labels))
        current_exposure = replay_exposure = 0
        for start in range(0, len(order), 192):
            rows = order[start : start + 192]
            replay_count = round(len(rows) * 0.25 / 0.75)
            replay_rows = generator.integers(0, len(buffer.sample_ids), size=replay_count)
            features = np.concatenate((current_features[rows], buffer.features[replay_rows]))
            labels = np.concatenate((current_labels[rows], buffer.labels[replay_rows]))
            shuffled = generator.permutation(len(labels))
            current_exposure += len(rows)
            replay_exposure += replay_count
            yield torch.from_numpy(features[shuffled]), torch.from_numpy(labels[shuffled])
        self.last_exposure = {
            "current": current_exposure,
            "replay": replay_exposure,
            "stored_buffer": len(buffer.sample_ids),
        }
