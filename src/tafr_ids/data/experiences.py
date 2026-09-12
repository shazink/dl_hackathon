"""Development-count snake allocation and locked protocol validation."""

from collections import Counter
import json
import tomllib

import numpy as np

from tafr_ids.data.loader import CLASSES, CLASS_TO_INDEX
from tafr_ids.data.splits import SEED, assignment_fingerprint

SNAKE = (1, 2, 3, 4, 4, 3, 2, 1)


def construct_experiences(targets, partition, training_hash):
    if len(targets) != len(partition) or set(np.unique(partition)) != {0, 1} or set(targets) != set(CLASSES):
        raise ValueError("Invalid partition or canonical class coverage")
    counts = Counter(targets[partition == 0])
    attacks = sorted((name for name in counts if name != "Normal"), key=lambda name: (-counts[name], name))
    mapping = {name: SNAKE[index % len(SNAKE)] for index, name in enumerate(attacks)}
    assignments = np.zeros(len(targets), dtype=np.uint8)
    for name, experience in mapping.items():
        assignments[targets == name] = experience
    # Separate seeded streams for each partition; stable input row order.
    for part in (0, 1):
        normal = np.flatnonzero((targets == "Normal") & (partition == part))
        shuffled = np.random.default_rng(np.random.SeedSequence([SEED, part])).permutation(normal)
        for experience, rows in enumerate(np.array_split(shuffled, 4), 1):
            assignments[rows] = experience
    if not np.isin(assignments, [1, 2, 3, 4]).all():
        raise ValueError("Unassigned experience rows")
    config = {
        "protocol_version": "1.0",
        "seed": SEED,
        "target": "attack_cat",
        "classifier": "single_head_global_classes",
        "global_class_mapping": dict(CLASS_TO_INDEX),
        "attack_class_counts": {name: counts[name] for name in attacks},
        "allocation_algorithm": "Development attack counts descending, canonical name ascending tie-break; snake 1,2,3,4,4,3,2,1 repeated",
        "background_policy": "Normal recurs; each partition independently shuffled with NumPy PCG64 SeedSequence([42, partition_code]); array_split into four disjoint balanced subsets",
        "development_validation_policy": "sklearn train_test_split, test_size=0.2, stratify=canonical attack_cat, random_state=42; sorted zero-based indices; duplicate feature rows preserved",
        "target_normalization": "strip and casefold; canonical names plus backdoors -> Backdoor; reject unknown/missing",
        "preprocessing_policy": "Fit once on E1 development only; freeze; transform all later and validation subsets",
        "logical_training_sha256": training_hash,
        "split_assignment_fingerprint": assignment_fingerprint(partition, assignments),
        "experiences": {f"E{exp}": {"introduced_attack_classes": [name for name in attacks if mapping[name] == exp]} for exp in range(1, 5)},
    }
    return assignments, config


def validate_experience_config(path, expected):
    with open(path, "rb") as stream:
        actual = tomllib.load(stream)
    if actual != expected:
        raise ValueError("Locked experience config differs from recomputed training-only protocol")


def config_toml(config):
    """Serialize only this small protocol structure, with stable key order."""
    lines = []
    def section(values, path=()):
        if path:
            lines.append("[" + ".".join(json.dumps(key) for key in path) + "]")
        for key, value in values.items():
            if not isinstance(value, dict):
                lines.append(f"{json.dumps(key)} = {json.dumps(value)}")
        for key, value in values.items():
            if isinstance(value, dict):
                lines.append("")
                section(value, (*path, key))
    section(config)
    return "\n".join(lines) + "\n"
