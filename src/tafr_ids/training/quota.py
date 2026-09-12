"""Deterministic bounded largest-remainder replay quota allocation."""

from __future__ import annotations

import math
from collections.abc import Mapping


def allocate_quotas(
    candidate_counts: Mapping[int, int], priorities: Mapping[int, float], capacity: int
) -> tuple[dict[int, int], dict]:
    counts = {int(key): int(value) for key, value in candidate_counts.items() if int(value) > 0}
    if capacity <= 0 or not counts or set(counts) != set(priorities):
        raise ValueError("Quota allocation requires positive capacity and matching nonempty classes")
    if any(not math.isfinite(float(priorities[key])) or priorities[key] < 0 for key in counts):
        raise ValueError("Priorities must be finite and nonnegative")
    classes = sorted(counts)
    target = min(capacity, sum(counts.values()))
    uniform_quota = capacity / len(classes)
    minimum = math.floor(0.25 * uniform_quota)
    maximum = math.ceil(2.0 * uniform_quota)
    lower = {key: min(counts[key], minimum) for key in classes}
    upper = {key: min(counts[key], maximum) for key in classes}
    cap_relaxed = False
    relaxed: dict[int, int] = {}
    if sum(upper.values()) < target:
        cap_relaxed = True
        shortage = target - sum(upper.values())
        order = sorted(classes, key=lambda key: (-float(priorities[key]), key))
        while shortage:
            progressed = False
            for key in order:
                if upper[key] < counts[key]:
                    upper[key] += 1
                    relaxed[key] = relaxed.get(key, 0) + 1
                    shortage -= 1
                    progressed = True
                    if not shortage:
                        break
            if not progressed:
                break

    quota = dict(lower)
    remaining = target - sum(quota.values())
    rounds = []
    redistribution = []
    while remaining > 0:
        active = [key for key in classes if quota[key] < upper[key]]
        if not active:
            break
        weight_sum = sum(float(priorities[key]) for key in active)
        weights = {key: (float(priorities[key]) if weight_sum > 0 else 1.0) for key in active}
        effective_sum = sum(weights.values())
        ideals = {key: remaining * weights[key] / effective_sum for key in active}
        floors = {key: min(math.floor(ideals[key]), upper[key] - quota[key]) for key in active}
        allocated = sum(floors.values())
        for key, amount in floors.items():
            quota[key] += amount
        remaining -= allocated
        remainders = {key: ideals[key] - math.floor(ideals[key]) for key in active}
        rounds.append({
            "remaining_before": remaining + allocated,
            "ideal": ideals,
            "floor_allocated": floors,
            "fractional_remainder": remainders,
        })
        if remaining == 0:
            break
        ranked = sorted(
            (key for key in active if quota[key] < upper[key]),
            key=lambda key: (-remainders[key], -float(priorities[key]), key),
        )
        if not ranked:
            continue
        for key in ranked:
            quota[key] += 1
            remaining -= 1
            redistribution.append({"class_id": key, "reason": "largest_remainder"})
            if remaining == 0:
                break
    if sum(quota.values()) != target:
        raise ValueError("Replay quota allocator could not fill the eligible target")
    audit = {
        "target": target,
        "uniform_quota": uniform_quota,
        "minimum_quota": minimum,
        "maximum_quota": maximum,
        "candidate_counts": counts,
        "lower_bounds": lower,
        "upper_bounds": upper,
        "priorities": {key: float(priorities[key]) for key in classes},
        "requested_quotas": quota,
        "actual_quotas": dict(quota),
        "rounding": rounds,
        "redistribution": redistribution,
        "cap_relaxed": cap_relaxed,
        "cap_relaxed_classes": relaxed,
        "cap_relaxation_reason": "sum of bounded upper quotas was below target" if cap_relaxed else None,
    }
    return quota, audit
