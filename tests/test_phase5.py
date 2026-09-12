import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tafr_ids.data.inspection import sha256
from tafr_ids.data.loader import CLASS_TO_INDEX
from tafr_ids.evaluation.continual import summarize_matrix
from tafr_ids.evaluation.metrics import classification_metrics
from tafr_ids.models.mlp import TabularMLP
from tafr_ids.training.checkpoint import atomic_save, load_checkpoint
from tafr_ids.training.reproducibility import configure_determinism
from tafr_ids.training.trainer import PreparedData, run_naive, validate_prepared_data


def tiny_config(input_dim=4):
    return {
        "method": "naive", "seed": 42, "device": "cpu",
        "data": {"expected_input_dim": input_dim, "split_manifest": "split.toml", "experience_config": "experiences.toml"},
        "model": {"hidden_dims": [8, 4], "normalization": "layer_norm", "activation": "relu", "dropout": 0.2, "initialization": "kaiming"},
        "training": {"loss": "cross_entropy", "optimizer": "adamw", "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 4, "epochs_per_experience": 1, "gradient_clip_norm": 5.0, "scheduler": "none", "early_stopping": False, "model_selection": "final_epoch", "optimizer_policy": "reset_each_experience", "num_workers": 0, "shuffle_training": True, "shuffle_validation": False},
        "evaluation": {"zero_division": 0, "primary_forgetting_metric": "balanced_accuracy", "evaluate_all_validation_experiences": True, "evaluate_seen_union": True, "logical_test_evaluation": False},
    }


def test_mlp_shape_initialization_and_parameter_count():
    configure_determinism(42)
    first = TabularMLP(4, 10, [8, 4], 0.2)
    configure_determinism(42)
    second = TabularMLP(4, 10, [8, 4], 0.2)
    assert torch.equal(next(first.parameters()), next(second.parameters()))
    assert first(torch.ones(3, 4)).shape == (3, 10)
    assert first.parameter_count == sum(parameter.numel() for parameter in first.parameters())


def test_metrics_record_absent_classes_and_fixed_confusion_matrix():
    result = classification_metrics(np.array([0, 0, 1]), np.array([0, 1, 1]), 0.5, ("a", "b", "c"))
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["balanced_accuracy"] == pytest.approx(0.75)
    assert result["per_class"]["c"] == {
        "precision": None, "recall": None, "f1": None, "support": 0, "absent": True
    }
    assert np.asarray(result["confusion_matrix"]).shape == (3, 3)


def test_continual_formulae():
    summary = summarize_matrix([
        [0.1, 0.2, 0.3],
        [0.8, 0.25, 0.35],
        [0.7, 0.7, 0.4],
        [0.6, 0.65, 0.8],
    ])
    assert summary["final_average"] == pytest.approx((0.6 + 0.65 + 0.8) / 3)
    assert summary["per_task_forgetting"] == pytest.approx([0.2, 0.05, 0.0])
    assert summary["forward_transfer"] == pytest.approx(((0.25 - 0.2) + (0.4 - 0.3)) / 2)


def test_checkpoint_fingerprint_guards(tmp_path):
    path = tmp_path / "latest.pt"
    atomic_save({"config_fingerprint": "a", "prepared_data_fingerprint": "b", "value": 3}, path)
    assert load_checkpoint(path, config_fingerprint="a", prepared_data_fingerprint="b", map_location=torch.device("cpu"))["value"] == 3
    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(path, config_fingerprint="wrong", prepared_data_fingerprint="b", map_location=torch.device("cpu"))


def test_prepared_validation_and_tamper_rejection(tmp_path):
    repository = tmp_path / "repository"
    prepared_dir = tmp_path / "prepared"
    repository.mkdir()
    prepared_dir.mkdir()
    split = repository / "split.toml"
    split.write_text('[training]\nsha256 = "dataset-hash"\n')
    mapping = "\n".join(f'{name} = {index}' for name, index in CLASS_TO_INDEX.items())
    experience_config = repository / "experiences.toml"
    assignments = prepared_dir / "assignments.csv"
    assignments.write_text("fixture\n")
    experience_config.write_text(f'split_assignment_fingerprint = "{sha256(assignments)}"\n[global_class_mapping]\n{mapping}\n')
    (prepared_dir / "preprocessor.joblib").write_bytes(b"fixture")
    np.save(prepared_dir / "preprocessor_fit_indices.npy", np.array([0], dtype=np.int64))
    subsets = {}
    next_index = 0
    for experience in range(1, 5):
        for partition in ("development", "validation"):
            prefix = f"E{experience}_{partition}"
            np.save(prepared_dir / f"{prefix}_X.npy", np.ones((2, 4), dtype=np.float32))
            np.save(prepared_dir / f"{prefix}_y.npy", np.array([0, experience], dtype=np.int64))
            np.save(prepared_dir / f"{prefix}_indices.npy", np.arange(next_index, next_index + 2, dtype=np.int64))
            next_index += 2
            subsets[prefix] = {"rows": 2, "features": 4}
    files = {path.name: sha256(path) for path in prepared_dir.iterdir()}
    metadata = {
        "files": files, "subsets": subsets, "feature_names": ["a", "b", "c", "d"],
        "split_manifest_sha256": sha256(split), "experience_config_sha256": sha256(experience_config),
        "logical_test": "Integrity verification only; no targets profiled or arrays prepared",
        "protocol": {"logical_training_sha256": "dataset-hash", "global_class_mapping": dict(CLASS_TO_INDEX)},
    }
    (prepared_dir / "metadata.json").write_text(json.dumps(metadata))
    validated = validate_prepared_data(prepared_dir, tiny_config(), repository)
    assert validated.input_dim == 4
    with (prepared_dir / "E1_development_X.npy").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_prepared_data(prepared_dir, tiny_config(), repository)


def test_tiny_naive_run_and_completed_resume(tmp_path):
    configure_determinism(42)
    arrays = {}
    index = 0
    for experience in range(1, 5):
        arrays[f"E{experience}"] = {}
        for partition in ("development", "validation"):
            features = np.eye(4, dtype=np.float32).repeat(2, axis=0)
            labels = np.array([0, experience] * 4, dtype=np.int64)
            indices = np.arange(index, index + 8, dtype=np.int64)
            index += 8
            arrays[f"E{experience}"][partition] = (features, labels, indices)
    prepared = PreparedData(
        arrays,
        {
            "protocol": {"logical_training_sha256": "fixture"},
            "split_manifest_sha256": "split",
            "experience_config_sha256": "experience",
            "files": {"assignments.csv": "assignments"},
        },
        "prepared",
        4,
    )
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "run"
    first = run_naive(config=tiny_config(), prepared=prepared, output_dir=output, repository=repository)
    assert first["status"] == "completed"
    assert len(first["history"]["epochs"]) == 4
    result_bytes = (output / "results.json").read_bytes()
    resumed = run_naive(config=tiny_config(), prepared=prepared, output_dir=output, repository=repository, resume=True)
    assert resumed["matrices"] == first["matrices"]
    assert resumed["resources"]["checkpoint_size_bytes"] > 0
    assert (output / "results.json").read_bytes() == result_bytes
