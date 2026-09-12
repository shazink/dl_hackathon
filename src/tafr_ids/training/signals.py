"""Training-memory-only replay signals and fixed TAFR ablations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def normalize_signal(values: Mapping[int, float]) -> dict[int, float]:
    """Min-max normalize a finite class signal, with the locked zero-range rule."""
    result = {int(key): float(value) for key, value in values.items()}
    if not result or not np.isfinite(list(result.values())).all():
        raise ValueError("Replay signals must be finite and nonempty")
    low, high = min(result.values()), max(result.values())
    if high == low:
        return {key: 0.0 for key in result}
    return {key: (value - low) / (high - low) for key, value in result.items()}


def forgetting_scores(
    seen_classes: list[int], best_recall: Mapping[int, float], current_recall: Mapping[int, float]
) -> dict[int, float]:
    """Compute fixed-memory replay-proxy forgetting; absent buffer classes score zero."""
    scores = {}
    for class_id in seen_classes:
        if class_id not in current_recall:
            scores[class_id] = 0.0
        else:
            scores[class_id] = max(
                0.0, float(best_recall.get(class_id, current_recall[class_id])) - float(current_recall[class_id])
            )
    if not np.isfinite(list(scores.values())).all():
        raise ValueError("Forgetting scores must be finite")
    return scores


def rarity_scores(seen_classes: list[int], cumulative_counts: Mapping[int, int]) -> dict[int, float]:
    scores = {}
    for class_id in seen_classes:
        count = int(cumulative_counts.get(class_id, 0))
        if count <= 0:
            raise ValueError("Every seen class must have a positive cumulative count")
        scores[class_id] = 1.0 / np.sqrt(count)
    return scores


def priorities_for_method(
    method: str,
    normalized_forgetting: Mapping[int, float],
    normalized_uncertainty: Mapping[int, float],
    normalized_rarity: Mapping[int, float],
) -> tuple[dict[int, float], bool]:
    class_ids = sorted(normalized_forgetting)
    if set(class_ids) != set(normalized_uncertainty) or set(class_ids) != set(normalized_rarity):
        raise ValueError("Normalized replay signals must cover identical classes")
    if method == "uniform":
        priorities = {class_id: 1.0 for class_id in class_ids}
    elif method == "tafr_f":
        priorities = {class_id: normalized_forgetting[class_id] for class_id in class_ids}
    elif method == "tafr_fu":
        priorities = {
            class_id: 0.5 * normalized_forgetting[class_id] + 0.5 * normalized_uncertainty[class_id]
            for class_id in class_ids
        }
    elif method == "tafr":
        priorities = {
            class_id: (
                normalized_forgetting[class_id]
                + normalized_uncertainty[class_id]
                + normalized_rarity[class_id]
            ) / 3.0
            for class_id in class_ids
        }
    else:
        raise ValueError(f"Unknown replay method: {method}")
    if not np.isfinite(list(priorities.values())).all() or min(priorities.values(), default=0.0) < 0:
        raise ValueError("Replay priorities must be finite and nonnegative")
    fallback = bool(priorities) and max(priorities.values()) == 0.0
    if fallback:
        priorities = {class_id: 1.0 for class_id in class_ids}
    return priorities, fallback
