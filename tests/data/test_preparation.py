import copy
import csv
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from tafr_ids.data import artifacts, loader
from tafr_ids.data.experiences import config_toml, construct_experiences, validate_experience_config
from tafr_ids.data.inspection import CATEGORICAL, FEATURES, HEADER, sha256
from tafr_ids.data.loader import CLASSES, CLASS_TO_INDEX, normalize_targets
from tafr_ids.data.preprocessing import FrozenPreprocessor
from tafr_ids.data.splits import assignment_fingerprint, development_validation


def frame(size):
    return pd.DataFrame({name: (["tcp"] * size if name in CATEGORICAL else np.arange(size, dtype=float)) for name in FEATURES})


def test_target_normalization_and_immutable_mapping():
    assert normalize_targets([" normal ", "DOS", " Backdoors ", "reconnaissance"]).tolist() == ["Normal", "DoS", "Backdoor", "Reconnaissance"]
    for invalid in ("", None, np.nan, "unknown", "NormalAttack"):
        with pytest.raises(ValueError):
            normalize_targets([invalid])
    with pytest.raises(TypeError):
        CLASS_TO_INDEX["Normal"] = 100
    assert list(CLASS_TO_INDEX.values()) == list(range(10))


def test_loader_only_reads_logical_training_and_checks_binary_integrity(tmp_path, monkeypatch):
    path = tmp_path / "physical_testing.csv"
    def write(label):
        values = dict.fromkeys(HEADER, "1")
        values.update(attack_cat=" normal ", label=label, proto="tcp", service="-", state="FIN")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=HEADER)
            writer.writeheader()
            writer.writerow(values)
        manifest.write_text(f'[training]\nsha256 = "{sha256(path)}"\nexpected_rows = 1\n')
    manifest = tmp_path / "manifest.toml"
    monkeypatch.setattr(loader, "verified_split_paths", lambda *args: {"training": path, "testing": tmp_path / "MUST_NOT_BE_READ.csv"})
    write("0")
    features, targets, _ = loader.load_training(tmp_path, manifest)
    assert list(features.columns) == FEATURES and targets.tolist() == ["Normal"]
    assert features["service"].iloc[0] == "-"
    write("1")
    with pytest.raises(ValueError, match="Binary"):
        loader.load_training(tmp_path, manifest)
    manifest.write_text('[training]\nsha256 = "wrong"\nexpected_rows = 1\n')
    with pytest.raises(ValueError, match="changed before"):
        loader.load_training(tmp_path, manifest)


def test_deterministic_coverage_rare_classes_and_snake(tmp_path):
    targets = np.concatenate([np.repeat(name, 30 + index * 10) for index, name in enumerate(CLASSES)])
    partition = development_validation(targets)
    assert np.array_equal(partition, development_validation(targets))
    exp, config = construct_experiences(targets, partition, "f" * 64)
    again, config_again = construct_experiences(targets, partition, "f" * 64)
    assert np.array_equal(exp, again) and config == config_again
    joined = []
    for part in (0, 1):
        assert set(targets[partition == part]) == set(CLASSES)
        normals = []
        for experience in range(1, 5):
            indices = np.flatnonzero((partition == part) & (exp == experience))
            joined.extend(indices.tolist())
            normals.append(sum(targets[indices] == "Normal"))
        assert max(normals) - min(normals) <= 1
    assert sorted(joined) == list(range(len(targets))) and len(set(joined)) == len(targets)
    attacks = sorted(config["attack_class_counts"], key=lambda name: (-config["attack_class_counts"][name], name))
    assert [int(exp[np.flatnonzero(targets == name)[0]]) for name in attacks] == [1, 2, 3, 4, 4, 3, 2, 1, 1]
    path = tmp_path / "experiences.toml"
    path.write_text(config_toml(config))
    validate_experience_config(path, config)
    changed = copy.deepcopy(config)
    changed["seed"] = 43
    with pytest.raises(ValueError):
        validate_experience_config(path, changed)
    assert assignment_fingerprint(partition, exp) != assignment_fingerprint(1 - partition, exp)


def test_validation_labels_do_not_set_attack_order():
    targets = np.repeat(CLASSES, 40)
    part = development_validation(targets)
    _, first = construct_experiences(targets, part, "f" * 64)
    changed = targets.copy()
    changed[part == 1] = np.roll(changed[part == 1], 1)
    _, second = construct_experiences(changed, part, "f" * 64)
    assert first["attack_class_counts"] == second["attack_class_counts"]
    assert first["experiences"] == second["experiences"]


def test_fit_scope_freeze_missing_unknowns_and_roundtrip(tmp_path):
    e1 = frame(10)
    e1.loc[0, "dur"] = np.nan
    e1.loc[0, "proto"] = np.nan
    model = FrozenPreprocessor()
    with pytest.raises(ValueError):
        model.fit_e1(e1, partition="validation", experience=1)
    with pytest.raises(ValueError):
        model.fit_e1(e1, partition="development", experience=2)
    model.fit_e1(e1, partition="development", experience=1)
    before = joblib.hash(model)
    later = frame(3)
    later["proto"] = "never_seen_in_E1"
    later["dur"] = [np.nan, 10000.0, 20000.0]
    transformed = model.transform(later)
    assert transformed.dtype == np.float32 and np.isfinite(transformed).all()
    assert transformed.shape[1] == model.transform(e1).shape[1]
    assert joblib.hash(model) == before
    with pytest.raises(ValueError, match="frozen"):
        model.fit_e1(later, partition="development", experience=1)
    statistics = model._transformer.named_transformers_["numeric"].named_steps["impute"].statistics_
    assert statistics[0] == 5.0  # E1 dur median, unaffected by later data.
    path = tmp_path / "processor.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    assert np.array_equal(restored.transform(later), transformed)
    with pytest.raises(ValueError, match="frozen"):
        restored.fit_e1(e1, partition="development", experience=1)


@pytest.mark.parametrize("change", ["target", "reorder", "drop", "type", "infinite"])
def test_schema_and_leakage_rejection(change):
    features = frame(10)
    model = FrozenPreprocessor().fit_e1(features, partition="development", experience=1)
    if change == "target":
        features["attack_cat"] = "Normal"
    elif change == "reorder":
        features = features.iloc[:, ::-1]
    elif change == "drop":
        features = features.drop(columns="dur")
    elif change == "type":
        features["dur"] = "not_numeric"
    else:
        features.loc[0, "dur"] = np.inf
    with pytest.raises(ValueError):
        model.transform(features)


def test_artifact_pipeline_fit_indices_and_determinism(tmp_path, monkeypatch):
    targets = np.repeat(CLASSES, 40)
    features = frame(len(targets))
    monkeypatch.setattr(artifacts, "load_training", lambda *args: (features, targets, "f" * 64))
    partition = development_validation(targets)
    experience, config = construct_experiences(targets, partition, "f" * 64)
    config_path = tmp_path / "experiences.toml"
    config_path.write_text(config_toml(config))
    manifest_path = tmp_path / "splits.toml"
    manifest_path.write_text("fixture = true\n")
    first = artifacts.prepare(tmp_path / "archive", manifest_path, config_path, tmp_path / "first")
    second = artifacts.prepare(tmp_path / "archive", manifest_path, config_path, tmp_path / "second")
    assert first == second
    assert sum(entry["rows"] for entry in first["subsets"].values()) == len(targets)
    expected = np.flatnonzero((partition == 0) & (experience == 1))
    assert np.array_equal(np.load(tmp_path / "first" / "preprocessor_fit_indices.npy"), expected)
    assert not any("test" in name for name in first["files"])
    assert json.loads((tmp_path / "first" / "metadata.json").read_text())["fit_scope"] == "E1 development only"
    with pytest.raises(ValueError, match="already exists"):
        artifacts.prepare(tmp_path / "archive", manifest_path, config_path, tmp_path / "first")
