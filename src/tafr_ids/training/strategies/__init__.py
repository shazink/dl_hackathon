"""Continual-learning training strategies."""

from tafr_ids.training.strategies.naive import NaiveStrategy

__all__ = ["NaiveStrategy"]
from tafr_ids.training.strategies.naive import NaiveStrategy
from tafr_ids.training.strategies.tafr import TAFRStrategy
from tafr_ids.training.strategies.uniform import UniformReplayStrategy

__all__ = ["NaiveStrategy", "TAFRStrategy", "UniformReplayStrategy"]
