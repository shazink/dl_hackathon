import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tafr_ids.data import inspection as ins


def write_split(path, rows, header=None):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(ins.HEADER if header is None else header)
        writer.writerows(rows)


def row(**overrides):
    values = dict.fromkeys(ins.HEADER, "1")
    values.update(proto="tcp", service="-", state="FIN", attack_cat="Normal", label="0")
    values.update(overrides)
    return [values[name] for name in ins.HEADER]


def write_manifest(dataset, swapped=False):
    train, test = (ins.TEST, ins.TRAIN) if swapped else (ins.TRAIN, ins.TEST)
    manifest = dataset.parent / "splits.toml"
    text = (
        'dataset = "UNSW-NB15"\n'
        f'official_metadata_url = "{ins.SOURCE}"\n'
        f'mirror_url = "{ins.MIRROR}"\n'
        f'resolution_status = "{"swapped_filenames" if swapped else "canonical_filenames"}"\n'
        'provenance_status = "Synthetic test fixture"\n'
        'note = "Physical filenames are explicitly mapped."\n'
    )
    for logical, physical, canonical in (("training", train, ins.TRAIN), ("testing", test, ins.TEST)):
        text += (
            f'[{logical}]\nphysical_filename = "{physical}"\n'
            f'expected_rows = {ins.OFFICIAL_ROWS[canonical]}\n'
            f'sha256 = "{ins.sha256(dataset / physical)}"\n'
        )
    manifest.write_text(text)
    return manifest


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    root.mkdir()
    for name in ins.EXPECTED_FILES:
        (root / name).write_text("metadata\n", encoding="utf-8")
    write_split(root / ins.TRAIN, [row(), row(id="2")])
    write_split(root / ins.TEST, [row(attack_cat="TEST_ONLY_SECRET", label="1")])
    monkeypatch.setattr(ins, "OFFICIAL_ROWS", {ins.TRAIN: 2, ins.TEST: 1})
    return root


def inspect(dataset):
    return ins.inspect_dataset(dataset, write_manifest(dataset))


def test_profile_duplicates_hashes_and_repeatability(dataset):
    before = {p.name: p.read_bytes() for p in dataset.iterdir()}
    result = inspect(dataset)
    assert result == inspect(dataset)
    assert result["status"] == "complete"
    assert result["split_resolution"]["status"] == "canonical_filenames"
    assert result["training_profile"]["class_counts"] == {"Normal": 2}
    assert result["training_profile"]["duplicate_feature_rows_beyond_first"] == 1
    assert result["cross_split_duplicates"]["distinct_shared_feature_vectors"] == 1
    assert "TEST_ONLY_SECRET" not in json.dumps(result)
    assert set(result["splits"][ins.TEST]) == {
        "rows", "columns", "header", "schema_matches", "parseable", "malformed_rows",
        "missing_values", "invalid_values", "infinite_values", "id_integrity",
        "physical_filename", "logical_split",
    }
    assert result["splits"][ins.TRAIN]["id_integrity"] == {
        "present": True, "unique": True, "minimum": 1, "maximum": 2, "sequence_contiguous": True,
    }
    for name, content in before.items():
        assert (dataset / name).read_bytes() == content
        assert result["inventory"][name]["sha256"] == hashlib.sha256(content).hexdigest()


def test_swapped_roles_never_profile_logical_test(dataset):
    write_split(dataset / ins.TEST, [row(attack_cat="TrainAttack", label="1"), row(id="2")])
    write_split(dataset / ins.TRAIN, [row(attack_cat="LOGICAL_TEST_SECRET")])
    result = ins.inspect_dataset(dataset, write_manifest(dataset, swapped=True))
    assert result["status"] == "complete"
    assert result["split_resolution"]["status"] == "swapped_filenames"
    assert result["training_profile"]["physical_filename"] == ins.TEST
    assert result["training_profile"]["class_counts"] == {"Normal": 1, "TrainAttack": 1}
    assert "LOGICAL_TEST_SECRET" not in json.dumps(result)


def test_future_loader_paths_use_verified_roles_without_profiling(dataset, monkeypatch):
    write_split(dataset / ins.TEST, [row(), row(id="2")])
    write_split(dataset / ins.TRAIN, [row(attack_cat="LOGICAL_TEST_SECRET")])
    manifest = write_manifest(dataset, swapped=True)
    def forbidden(*args):
        pytest.fail("Path verification must not profile any targets")
    monkeypatch.setattr(ins, "_training_profile", forbidden)
    assert ins.verified_split_paths(dataset, manifest) == {
        "training": dataset / ins.TEST, "testing": dataset / ins.TRAIN,
    }
    write_split(dataset / ins.TEST, [row(dur="2"), row(id="2")])
    with pytest.raises(ValueError, match="SHA-256"):
        ins.verified_split_paths(dataset, manifest)


def test_test_targets_cannot_influence_training_profile(dataset):
    first = inspect(dataset)
    write_split(dataset / ins.TEST, [row(attack_cat="OTHER_SECRET", label="0")])
    # Explicit fixture re-fingerprinting; production never refreshes hashes automatically.
    second = inspect(dataset)
    assert first["training_profile"] == second["training_profile"]
    assert first["cross_split_duplicates"] == second["cross_split_duplicates"]


def test_no_manifest_never_profiles(dataset):
    result = ins.inspect_dataset(dataset)
    assert result["split_resolution"]["status"] == "canonical_filenames"
    assert result["training_profile"] is None
    assert result["status"] == "blocked"


def test_ambiguous_counts_never_profile(dataset, monkeypatch):
    monkeypatch.setattr(ins, "OFFICIAL_ROWS", {ins.TRAIN: 8, ins.TEST: 4})
    result = inspect(dataset)
    assert result["split_resolution"]["status"] == "ambiguous"
    assert result["training_profile"] is None


@pytest.mark.parametrize("change", ["hash", "count", "mapping", "resolution", "malformed", "missing_field"])
def test_manifest_drift_never_profiles(dataset, monkeypatch, change):
    manifest = write_manifest(dataset)
    content = manifest.read_text()
    if change == "hash":
        content = content.replace(ins.sha256(dataset / ins.TRAIN), "0" * 64)
    elif change == "count":
        content = content.replace("expected_rows = 2", "expected_rows = 200")
    elif change == "mapping":
        content = content.replace(ins.TRAIN, "../" + ins.TRAIN)
    elif change == "resolution":
        content = content.replace("canonical_filenames", "swapped_filenames")
    elif change == "malformed":
        content = "bad [ toml"
    else:
        content = content.replace('dataset = "UNSW-NB15"', "")
    manifest.write_text(content)
    def forbidden(*args):
        pytest.fail("Profile must not execute after failed validation")
    monkeypatch.setattr(ins, "_training_profile", forbidden)
    result = ins.inspect_dataset(dataset, manifest)
    assert result["status"] == "blocked"
    assert result["training_profile"] is None


@pytest.mark.parametrize("ids", [("1", "1"), ("1", "3"), ("0", "1"), ("1", "1.5"), ("1", "")])
def test_invalid_ids_fail_closed(dataset, ids):
    write_split(dataset / ins.TRAIN, [row(id=value) for value in ids])
    result = inspect(dataset)
    assert result["split_resolution"]["status"] == "invalid"
    assert result["training_profile"] is None


@pytest.mark.parametrize("bad", ["", "NaN", "inf", "-inf", "-1", "not-a-number"])
def test_invalid_numeric_integrity_blocks(dataset, bad):
    write_split(dataset / ins.TEST, [row(dur=bad)])
    result = inspect(dataset)
    assert result["status"] == "blocked"
    assert result["training_profile"] is None
    if "inf" in bad:
        assert result["splits"][ins.TEST]["infinite_values"] == {"dur": 1}


@pytest.mark.parametrize("mode", ["missing", "header", "ragged", "quote", "encoding"])
def test_file_and_schema_failures(dataset, mode):
    manifest = write_manifest(dataset)
    path = dataset / ins.TEST
    if mode == "missing":
        path.unlink()
    elif mode == "header":
        write_split(path, [row()], header=list(reversed(ins.HEADER)))
    elif mode == "ragged":
        write_split(path, [row()[:-1]])
    elif mode == "quote":
        path.write_text(",".join(ins.HEADER) + '\n"unterminated', encoding="utf-8")
    else:
        path.write_bytes(b"\xff\xfe")
    result = ins.inspect_dataset(dataset, manifest)
    assert result["split_resolution"]["status"] == "invalid"
    assert result["status"] == "blocked"
    assert result["training_profile"] is None


def test_training_target_conflicts(dataset):
    write_split(dataset / ins.TRAIN, [row(), row(id="2", attack_cat="Attack", label="1")])
    result = inspect(dataset)
    assert result["training_profile"]["conflicting_target_feature_groups"] == 1


def test_invalid_test_label_is_counted_only_as_integrity(dataset):
    write_split(dataset / ins.TEST, [row(label="SENSITIVE_INVALID_VALUE")])
    result = inspect(dataset)
    assert result["splits"][ins.TEST]["invalid_values"] == {"label": 1}
    assert "SENSITIVE_INVALID_VALUE" not in json.dumps(result)
    assert result["status"] == "blocked"


def cli(*args):
    script = Path(__file__).resolve().parents[2] / "scripts" / "inspect_unsw_nb15.py"
    return subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True)


def test_cli_output_blocked_json(dataset):
    output = dataset.parent / "artifacts" / "profile.json"
    run = cli("--data-dir", dataset, "--split-manifest", write_manifest(dataset), "--output", output)
    assert run.returncode == 2  # Subprocess uses real canonical counts.
    assert json.loads(output.read_text())["training_profile"] is None


def test_cli_protects_downloads(dataset):
    before = {p.name: p.read_bytes() for p in dataset.iterdir()}
    run = cli("--data-dir", dataset, "--output", dataset / "profile.json")
    assert run.returncode == 2
    assert "outside" in run.stderr
    assert before == {p.name: p.read_bytes() for p in dataset.iterdir()}


def test_cli_report_replacement_does_not_mutate_hardlinked_source(dataset):
    source = dataset / ins.TRAIN
    before = source.read_bytes()
    output = dataset.parent / "profile.json"
    output.hardlink_to(source)
    run = cli("--data-dir", dataset, "--output", output)
    assert run.returncode == 2
    assert source.read_bytes() == before
    assert json.loads(output.read_text())["training_profile"] is None


def test_cli_rejects_output_symlink_into_downloads(dataset):
    output = dataset.parent / "profile.json"
    source = dataset / ins.TRAIN
    before = source.read_bytes()
    output.symlink_to(source)
    run = cli("--data-dir", dataset, "--output", output)
    assert run.returncode == 2
    assert "outside" in run.stderr
    assert source.read_bytes() == before


def test_cli_missing_directory(tmp_path):
    run = cli("--data-dir", tmp_path / "absent")
    assert run.returncode == 2
    assert "Inspection failed" in run.stderr


def test_detects_mutation_before_profiling(dataset, monkeypatch):
    manifest = write_manifest(dataset)
    original = ins._integrity
    def changed(path):
        result = original(path)
        if path.name == ins.TEST:
            with path.open("a") as stream:
                stream.write("\n")
        return result
    monkeypatch.setattr(ins, "_integrity", changed)
    def forbidden(*args):
        pytest.fail("Profile must not run after source mutation")
    monkeypatch.setattr(ins, "_training_profile", forbidden)
    result = ins.inspect_dataset(dataset, manifest)
    assert result["status"] == "blocked"
    assert any("changed during inspection" in reason for reason in result["blockers"])
