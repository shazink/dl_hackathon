"""Naive sequential fine-tuning without replay."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from tafr_ids.training.strategies.base import TrainingStrategy


class NaiveStrategy(TrainingStrategy):
    def __init__(self, optimizer: torch.optim.Optimizer, gradient_clip_norm: float) -> None:
        self.optimizer = optimizer
        self.gradient_clip_norm = gradient_clip_norm

    def training_batches(self, current_batches: Iterable[tuple[torch.Tensor, torch.Tensor]]):
        return current_batches

    def observe(self, loss: torch.Tensor, model: nn.Module) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
