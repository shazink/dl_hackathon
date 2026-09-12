"""Locked experiment configuration and cross-method fairness checks."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path

from tafr_ids.training.trainer import METHODS

LOCKED_COMMON = {
    "seed": 42,
    "device": "auto",
    "data": {"expected_input_dim": 156, "split_manifest": "configs/unsw_nb15_splits.toml", "experience_config": "configs/experiences.toml"},
    "model": {"hidden_dims": [256, 128], "normalization": "layer_norm", "activation": "relu", "dropout": 0.2, "initialization": "kaiming"},
    "training": {"loss": "cross_entropy", "optimizer": "adamw", "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 256, "epochs_per_experience": 20, "gradient_clip_norm": 5.0, "scheduler": "none", "early_stopping": False, "model_selection": "final_epoch", "optimizer_policy": "reset_each_experience", "num_workers": 0, "shuffle_training": True, "shuffle_validation": False},
    "evaluation": {"zero_division": 0, "primary_forgetting_metric": "balanced_accuracy", "evaluate_all_validation_experiences": True, "evaluate_seen_union": True, "logical_test_evaluation": False},
}

LOCKED_REPLAY = {
    "capacity": 2000, "fraction": 0.25, "current_batch_size": 192, "replay_batch_size": 64,
    "sample_with_replacement": True, "buffer_update_version": 1,
    "within_class_selection": "deterministic_uniform_sha256", "memory_replacement": "complete",
    "forgetting_source": "retained_training_memory_only",
}


def load_experiment_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    method = config.get("method")
    if method not in METHODS:
        raise ValueError("Unknown experiment method")
    if {key: config.get(key) for key in LOCKED_COMMON} != LOCKED_COMMON:
        raise ValueError("Experiment configuration differs from locked shared settings")
    expected_keys = {"method", *LOCKED_COMMON}
    if method == "naive":
        if set(config) != expected_keys:
            raise ValueError("Naive configuration has unexpected replay or extra fields")
    else:
        expected_keys.add("replay")
        if set(config) != expected_keys or config.get("replay") != LOCKED_REPLAY:
            raise ValueError("Replay configuration differs from the locked memory and mixing protocol")
    return config


def validate_config_fairness(configs: dict[str, dict]) -> dict:
    if set(configs) != set(METHODS):
        raise ValueError("Fairness validation requires exactly all five methods")
    differences = {}
    for method, config in configs.items():
        common = {key: copy.deepcopy(config[key]) for key in LOCKED_COMMON}
        if common != LOCKED_COMMON:
            differences[method] = "shared field mismatch"
        if config["method"] != method:
            differences[method] = "method ID mismatch"
        if method != "naive" and config.get("replay") != LOCKED_REPLAY:
            differences[method] = "replay protocol mismatch"
    if differences:
        raise ValueError(f"Cross-method fairness failed: {differences}")
    return {"passed": True, "methods": list(METHODS), "identical_locked_fields": list(LOCKED_COMMON), "allowed_difference": "replay behavior and fixed priority ablation only", "capacity": 2000, "fraction": 0.25}
