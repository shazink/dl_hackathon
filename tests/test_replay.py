import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tafr_ids.training.config import LOCKED_COMMON, LOCKED_REPLAY, validate_config_fairness
from tafr_ids.training.quota import allocate_quotas
from tafr_ids.training.replay_buffer import ReplayBuffer, selected_ids_fingerprint, selection_seed
from tafr_ids.training.reproducibility import configure_determinism
from tafr_ids.training.signals import forgetting_scores, normalize_signal, priorities_for_method, rarity_scores
from tafr_ids.training.strategies.uniform import UniformReplayStrategy
from tafr_ids.training.trainer import PreparedData, run_experiment


def test_signal_formulas_normalization_and_zero_fallback():
    assert forgetting_scores([0, 1, 2], {0: 0.9, 1: 0.2}, {0: 0.4, 1: 0.5}) == {0: 0.5, 1: 0.0, 2: 0.0}
    assert rarity_scores([0, 1], {0: 4, 1: 16}) == {0: 0.5, 1: 0.25}
    assert normalize_signal({0: 2.0, 1: 4.0, 2: 3.0}) == {0: 0.0, 1: 1.0, 2: 0.5}
    assert normalize_signal({0: 2.0, 1: 2.0}) == {0: 0.0, 1: 0.0}
    priorities, fallback = priorities_for_method("tafr_f", {0: 0.0, 1: 0.0}, {0: 1.0, 1: 0.0}, {0: 0.0, 1: 1.0})
    assert priorities == {0: 1.0, 1: 1.0} and fallback
    with pytest.raises(ValueError, match="finite"):
        normalize_signal({0: float("nan")})


def test_ablation_isolation():
    f, u, r = {0: 0.2, 1: 1.0}, {0: 0.8, 1: 0.0}, {0: 0.5, 1: 0.25}
    assert priorities_for_method("uniform", f, u, r)[0] == {0: 1.0, 1: 1.0}
    assert priorities_for_method("tafr_f", f, u, r)[0] == f
    assert priorities_for_method("tafr_fu", f, u, r)[0] == {0: 0.5, 1: 0.5}
    assert priorities_for_method("tafr", f, u, r)[0] == pytest.approx({0: 0.5, 1: 5 / 12})


def test_quota_rounding_caps_shortages_and_relaxation():
    quota, audit = allocate_quotas({0: 1000, 1: 1000, 2: 1000}, {0: 1.0, 1: 1.0, 2: 1.0}, 2000)
    assert quota == {0: 667, 1: 667, 2: 666}
    assert not audit["cap_relaxed"] and sum(quota.values()) == 2000
    quota, audit = allocate_quotas({0: 1, 1: 1, 2: 100}, {0: 0.0, 1: 0.0, 2: 1.0}, 100)
    assert quota == {0: 1, 1: 1, 2: 98}
    assert audit["cap_relaxed"] and audit["cap_relaxed_classes"] == {2: 31}


def test_memory_uniqueness_no_recovery_and_method_independent_selection():
    empty = ReplayBuffer.empty(2)
    features = np.arange(20, dtype=np.float32).reshape(10, 2)
    labels = np.array([0] * 5 + [1] * 5, dtype=np.int64)
    ids = np.arange(10, dtype=np.int64)
    candidates = empty.candidates(features, labels, ids, 1)
    first = candidates.select({0: 3, 1: 2}, 42, "E1", 1)
    second = candidates.select({0: 3, 1: 2}, 42, "E1", 1)
    assert np.array_equal(first.sample_ids, second.sample_ids)
    assert len(np.unique(first.sample_ids)) == 5
    assert selected_ids_fingerprint(first.sample_ids) == selected_ids_fingerprint(second.sample_ids)
    assert selection_seed(42, "E1", 0, 1) != selection_seed(42, "E1", 1, 1)
    with pytest.raises(ValueError, match="duplicate"):
        empty.candidates(np.ones((2, 2), np.float32), np.array([0, 0]), np.array([1, 1]), 1)


def test_replay_mixing_visits_current_once_and_uses_locked_ratio():
    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters())
    strategy = UniformReplayStrategy(optimizer, 5.0)
    current_x = np.arange(386 * 2, dtype=np.float32).reshape(386, 2)
    current_y = np.zeros(386, dtype=np.int64)
    memory = ReplayBuffer(np.ones((10, 2), np.float32), np.ones(10, np.int64), np.arange(100, 110, dtype=np.int64), np.ones(10, dtype=np.int16))
    batches = list(strategy.mixed_batches(current_x, current_y, memory, seed=42))
    assert [len(batch[1]) for batch in batches] == [256, 256, 3]
    assert strategy.last_exposure == {"current": 386, "replay": 129, "stored_buffer": 10}


def _tiny_config(method):
    config = {
        "method": method, "seed": 42, "device": "cpu",
        "data": {"expected_input_dim": 4, "split_manifest": "unused", "experience_config": "unused"},
        "model": {"hidden_dims": [8, 4], "normalization": "layer_norm", "activation": "relu", "dropout": 0.2, "initialization": "kaiming"},
        "training": {"loss": "cross_entropy", "optimizer": "adamw", "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 4, "epochs_per_experience": 1, "gradient_clip_norm": 5.0, "scheduler": "none", "early_stopping": False, "model_selection": "final_epoch", "optimizer_policy": "reset_each_experience", "num_workers": 0, "shuffle_training": True, "shuffle_validation": False},
        "evaluation": {"zero_division": 0, "primary_forgetting_metric": "balanced_accuracy", "evaluate_all_validation_experiences": True, "evaluate_seen_union": True, "logical_test_evaluation": False},
    }
    if method != "naive":
        config["replay"] = {**LOCKED_REPLAY, "capacity": 4}
    return config


def _tiny_prepared():
    arrays, index = {}, 0
    for number in range(1, 5):
        arrays[f"E{number}"] = {}
        for partition in ("development", "validation"):
            x = np.eye(4, dtype=np.float32).repeat(2, axis=0)
            y = np.array([0, number] * 4, dtype=np.int64)
            ids = np.arange(index, index + 8, dtype=np.int64)
            index += 8
            arrays[f"E{number}"][partition] = (x, y, ids)
    return PreparedData(arrays, {"protocol": {"logical_training_sha256": "fixture"}, "split_manifest_sha256": "split", "experience_config_sha256": "experience", "files": {"assignments.csv": "assignment"}}, "prepared", 4)


def test_e1_equivalence_checkpoint_resume_and_replay_audit(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    reports = {}
    for method in ("naive", "uniform"):
        configure_determinism(42)
        output = tmp_path / method
        reports[method] = run_experiment(config=_tiny_config(method), prepared=_tiny_prepared(), output_dir=output, repository=repository)
        resumed = run_experiment(config=_tiny_config(method), prepared=_tiny_prepared(), output_dir=output, repository=repository, resume=True)
        assert resumed["matrices"] == reports[method]["matrices"]
    assert reports["naive"]["history"]["evaluations"]["random_init"] == reports["uniform"]["history"]["evaluations"]["random_init"]
    assert reports["naive"]["history"]["evaluations"]["after_E1"] == reports["uniform"]["history"]["evaluations"]["after_E1"]
    for update in reports["uniform"]["history"]["replay_updates"].values():
        assert update["unique_ids"] and update["eligible_only"] and update["buffer_size"] == 4
    assert json.loads((tmp_path / "uniform" / "results.json").read_text())["protocol_notes"]["logical_test_evaluated"] is False


def test_config_fairness_accepts_only_strategy_difference():
    configs = {}
    for method in ("naive", "uniform", "tafr_f", "tafr_fu", "tafr"):
        configs[method] = {"method": method, **LOCKED_COMMON}
        if method != "naive":
            configs[method]["replay"] = LOCKED_REPLAY.copy()
    assert validate_config_fairness(configs)["passed"]
    configs["tafr"]["training"] = {**configs["tafr"]["training"], "learning_rate": 0.2}
    with pytest.raises(ValueError, match="fairness"):
        validate_config_fairness(configs)
