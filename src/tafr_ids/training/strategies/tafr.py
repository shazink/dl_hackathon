"""TAFR variants share Uniform Replay's within-batch behavior."""

from tafr_ids.training.strategies.uniform import UniformReplayStrategy


class TAFRStrategy(UniformReplayStrategy):
    """Marker strategy: TAFR changes buffer class quotas, not batch composition."""
