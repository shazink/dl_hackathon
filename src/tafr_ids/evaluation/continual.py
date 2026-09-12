"""Continual-learning matrices and fixed aggregation formulas."""

from __future__ import annotations

import numpy as np


MATRIX_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def build_matrices(evaluations: dict, experiences: tuple[str, ...]) -> dict[str, list[list[float]]]:
    rows = ("random_init",) + tuple(f"after_{name}" for name in experiences)
    return {
        metric: [
            [float(evaluations[row][experience][metric]) for experience in experiences]
            for row in rows
        ]
        for metric in MATRIX_METRICS
    }


def summarize_matrix(matrix: list[list[float]]) -> dict:
    """Summarize R, where row 0 is random init and row i is after task i."""
    values = np.asarray(matrix, dtype=np.float64)
    task_count = values.shape[1]
    if values.shape != (task_count + 1, task_count):
        raise ValueError("Expected a random-reference row plus one row per task")
    final = values[-1]
    per_task_forgetting = []
    for task in range(task_count):
        learned_through_final = values[task + 1 :, task]
        per_task_forgetting.append(float(learned_through_final.max() - final[task]))
    forward_terms = [values[task, task] - values[0, task] for task in range(1, task_count)]
    return {
        "final_average": float(final.mean()),
        "per_task_forgetting": per_task_forgetting,
        "average_forgetting": float(np.mean(per_task_forgetting)),
        "forward_transfer": float(np.mean(forward_terms)) if forward_terms else 0.0,
    }


def continual_summary(matrices: dict[str, list[list[float]]]) -> dict:
    by_metric = {metric: summarize_matrix(matrix) for metric, matrix in matrices.items()}
    return {
        "by_metric": by_metric,
        "final_average_task_accuracy": by_metric["accuracy"]["final_average"],
        "final_average_balanced_accuracy": by_metric["balanced_accuracy"]["final_average"],
        "average_forgetting": by_metric["balanced_accuracy"]["average_forgetting"],
        "per_task_forgetting": by_metric["balanced_accuracy"]["per_task_forgetting"],
        "forward_transfer": by_metric["balanced_accuracy"]["forward_transfer"],
        "accuracy_average_forgetting": by_metric["accuracy"]["average_forgetting"],
        "accuracy_per_task_forgetting": by_metric["accuracy"]["per_task_forgetting"],
        "formulas": {
            "notation": "R[i,j] is validation performance on task j after training task i; R[0,j] is random initialization.",
            "final_average": "(1/T) * sum_j R[T,j]",
            "per_task_forgetting": "F_j = max_{i=j..T} R[i,j] - R[T,j], using one-based training rows",
            "average_forgetting": "(1/T) * sum_j F_j",
            "forward_transfer": "(1/(T-1)) * sum_{j=2..T} (R[j-1,j] - R[0,j])",
        },
    }
