"""Reusable multilayer perceptron for prepared tabular features."""

from __future__ import annotations

import torch
from torch import nn


class TabularMLP(nn.Module):
    """Single-head MLP with LayerNorm, ReLU, and dropout hidden blocks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 1 or not hidden_dims:
            raise ValueError("input_dim, output_dim, and hidden_dims must define a classifier")
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            if width <= 0:
                raise ValueError("Hidden dimensions must be positive")
            linear = nn.Linear(previous, width)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.LayerNorm(width), nn.ReLU(), nn.Dropout(dropout)))
            previous = width
        output = nn.Linear(previous, output_dim)
        nn.init.xavier_uniform_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
