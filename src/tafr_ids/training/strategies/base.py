"""Minimal strategy boundary shared by continual training methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
from torch import nn


class TrainingStrategy(ABC):
    @abstractmethod
    def training_batches(self, current_batches: Iterable[tuple[torch.Tensor, torch.Tensor]]):
        """Return batches allowed for the current experience."""

    @abstractmethod
    def observe(self, loss: torch.Tensor, model: nn.Module) -> None:
        """Perform one optimization update."""
