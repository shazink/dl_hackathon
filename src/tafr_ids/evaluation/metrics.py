"""Classification metrics with explicit global-class accounting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    mean_loss: float,
    class_names: Sequence[str],
) -> dict:
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.arange(len(class_names), dtype=np.int64)
    if targets.ndim != 1 or predictions.shape != targets.shape or targets.size == 0:
        raise ValueError("Targets and predictions must be equally sized, nonempty vectors")
    supports = np.bincount(targets, minlength=len(class_names))
    present = labels[supports > 0]
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        targets, predictions, labels=labels, average=None, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precisions[index]) if supports[index] else None,
            "recall": float(recalls[index]) if supports[index] else None,
            "f1": float(f1s[index]) if supports[index] else None,
            "support": int(supports[index]),
            "absent": bool(supports[index] == 0),
        }
        for index, name in enumerate(class_names)
    }
    return {
        "loss": float(mean_loss),
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(recalls[present].mean()),
        "macro_f1": float(f1_score(targets, predictions, labels=present, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, labels=labels, average="weighted", zero_division=0)),
        "per_class": per_class,
        "support": int(targets.size),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).astype(int).tolist(),
        "class_order": list(class_names),
        "macro_class_scope": [class_names[index] for index in present],
        "zero_division": 0,
    }
