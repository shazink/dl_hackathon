"""Read-only UNSW-NB15 inventory with a fail-closed training-profile gate.

No transformations, split construction, or learning take place here. Fingerprints
identify local bytes; matching row counts alone cannot authenticate a mirror.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tomllib
from collections import Counter
from pathlib import Path

TRAIN = "UNSW_NB15_training-set.csv"
TEST = "UNSW_NB15_testing-set.csv"
EXPECTED_FILES = (
    "NUSW-NB15_features.csv",
    *(f"UNSW-NB15_{i}.csv" for i in range(1, 5)),
    "UNSW-NB15_LIST_EVENTS.csv",
    TRAIN,
    TEST,
)
SOURCE = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
MIRROR = "https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/code"
OFFICIAL_ROWS = {TRAIN: 175341, TEST: 82332}
HEADER = (
    "id,dur,proto,service,state,spkts,dpkts,sbytes,dbytes,rate,sttl,dttl,"
    "sload,dload,sloss,dloss,sinpkt,dinpkt,sjit,djit,swin,stcpb,dtcpb,"
    "dwin,tcprtt,synack,ackdat,smean,dmean,trans_depth,response_body_len,"
    "ct_srv_src,ct_state_ttl,ct_dst_ltm,ct_src_dport_ltm,ct_dst_sport_ltm,"
    "ct_dst_src_ltm,is_ftp_login,ct_ftp_cmd,ct_flw_http_mthd,ct_src_ltm,"
    "ct_srv_dst,is_sm_ips_ports,attack_cat,label"
).split(",")
TARGETS = {"attack_cat", "label"}
CATEGORICAL = {"proto", "service", "state"}
FEATURES = [name for name in HEADER if name not in TARGETS | {"id"}]
MISSING = {"", "nan", "na", "null", "none"}


def sha256(path: Path) -> str:
    """Hash a file without changing it or loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_key(row: dict[str, str]) -> str:
    """Hash exact parsed feature strings; exclude id and both targets.

    This detects exact feature duplicates, not numeric-equivalent or near matches.
    JSON length/delimiter escaping avoids ambiguous concatenation.
    """
    return hashlib.sha256(
        json.dumps([row[name] for name in FEATURES], separators=(",", ":")).encode()
    ).hexdigest()


def _inventory(path: Path) -> dict:
    # Ancillary/raw files are inventoried by bytes only: their rows can contain
    # evaluation records and must not become an alternative profiling source.
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def _integrity(path: Path) -> tuple[dict, Counter]:
    """Only permitted integrity counts and feature fingerprints, for either split."""
    missing, invalid, infinite = Counter(), Counter(), Counter()
    ids = set()
    id_count = 0
    keys: Counter = Counter()
    report = {"rows": 0, "malformed_rows": 0, "parseable": True}
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, [])
            report.update(columns=len(header), header=header, schema_matches=header == HEADER)
            for values in reader:
                report["rows"] += 1
                if len(values) != len(header):
                    report["malformed_rows"] += 1
                    continue
                if not report["schema_matches"]:
                    continue
                row = dict(zip(header, values))
                token = row["id"].strip()
                if token.isascii() and token.isdecimal() and int(token) > 0:
                    ids.add(int(token))
                    id_count += 1
                for name, value in row.items():
                    stripped = value.strip()
                    if stripped.lower() in MISSING:
                        missing[name] += 1
                    elif name == "label":
                        # Integrity only: never count either valid target value.
                        if stripped not in {"0", "1"}:
                            invalid[name] += 1
                    elif name not in CATEGORICAL | {"attack_cat"}:
                        try:
                            number = float(stripped)
                            if math.isinf(number):
                                infinite[name] += 1
                            if not math.isfinite(number) or number < 0:
                                invalid[name] += 1
                            elif name == "id" and not (stripped.isascii() and stripped.isdecimal() and int(stripped) > 0):
                                invalid[name] += 1
                        except ValueError:
                            invalid[name] += 1
                keys[feature_key(row)] += 1
    except (csv.Error, UnicodeError):
        # Never echo raw failing rows, which could disclose test targets.
        report["parseable"] = False
    report["missing_values"] = dict(sorted(missing.items()))
    report["invalid_values"] = dict(sorted(invalid.items()))
    report["infinite_values"] = dict(sorted(infinite.items()))
    report["id_integrity"] = {
        "present": "id" in report.get("header", []),
        "unique": id_count == len(ids) == report["rows"],
        "minimum": min(ids) if ids else None,
        "maximum": max(ids) if ids else None,
        "sequence_contiguous": bool(ids) and len(ids) == report["rows"] and min(ids) == 1 and max(ids) == report["rows"],
    }
    return report, keys


def resolve_split_roles(splits: dict) -> dict:
    """Resolve only from official sizes and structural checks, never targets.

    This identifies roles under the documented protocol, not official byte identity.
    Consumers must additionally validate the manifest and hashes before loading.
    """
    for name in (TRAIN, TEST):
        split = splits.get(name, {})
        identity = split.get("id_integrity", {})
        if not (split.get("schema_matches") and split.get("parseable")
                and not split.get("malformed_rows") and identity.get("present")
                and identity.get("unique") and identity.get("sequence_contiguous")):
            return {"status": "invalid", "reason": "Required schema, parsing, or ID integrity failed.", "mapping": {}}
    counts = (splits[TRAIN]["rows"], splits[TEST]["rows"])
    if counts == (OFFICIAL_ROWS[TRAIN], OFFICIAL_ROWS[TEST]):
        status, training, testing = "canonical_filenames", TRAIN, TEST
    elif counts == (OFFICIAL_ROWS[TEST], OFFICIAL_ROWS[TRAIN]):
        status, training, testing = "swapped_filenames", TEST, TRAIN
    else:
        return {"status": "ambiguous", "reason": "Row counts do not uniquely match the official split sizes.", "mapping": {}}
    return {
        "status": status,
        "reason": "Official row counts and ordered schema, parseability, and unique contiguous IDs establish logical roles; labels are not used.",
        "mapping": {"training": training, "testing": testing},
    }


def validate_split_manifest(path: Path | str, resolution: dict, inventory: dict, splits: dict) -> dict:
    """Validate the portable, dataset-specific manifest; fail closed on any drift."""
    with Path(path).open("rb") as stream:
        manifest = tomllib.load(stream)
    try:
        if manifest["dataset"] != "UNSW-NB15" or manifest["official_metadata_url"] != SOURCE:
            raise ValueError("Manifest dataset or primary metadata authority differs")
        if manifest["resolution_status"] != resolution["status"]:
            raise ValueError("Manifest resolution status differs from current structural evidence")
        if resolution["status"] not in {"canonical_filenames", "swapped_filenames"}:
            raise ValueError("Split roles are unresolved")
        for logical, canonical in (("training", TRAIN), ("testing", TEST)):
            entry = manifest[logical]
            filename = resolution["mapping"][logical]
            if entry["physical_filename"] != filename:
                raise ValueError(f"Manifest {logical} mapping differs from structural evidence")
            if entry["expected_rows"] != OFFICIAL_ROWS[canonical] or splits[filename]["rows"] != entry["expected_rows"]:
                raise ValueError(f"Manifest {logical} row count differs")
            if entry["sha256"] != inventory[filename]["sha256"]:
                raise ValueError(f"Manifest {logical} SHA-256 differs")
        if not all(isinstance(manifest[key], str) and manifest[key] for key in ("mirror_url", "provenance_status", "note")):
            raise ValueError("Manifest provenance metadata is incomplete")
    except (KeyError, TypeError) as error:
        raise ValueError("Manifest is missing required fields or has invalid types") from error
    return manifest


def _training_profile(path: Path, keys: Counter) -> dict:
    """Called only after integrity and split-count checks pass."""
    classes, labels, categories = Counter(), Counter(), {name: Counter() for name in CATEGORICAL}
    numeric = {name: {"count": 0, "min": None, "max": None, "sum": 0.0, "integer_valued": True} for name in FEATURES if name not in CATEGORICAL}
    ids, conflicts = set(), {}
    duplicate_ids = 0
    inconsistent_targets = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            classes[row["attack_cat"]] += 1
            labels[row["label"]] += 1
            inconsistent_targets += (row["attack_cat"] == "Normal") != (row["label"] == "0")
            duplicate_ids += row["id"] in ids
            ids.add(row["id"])
            conflicts.setdefault(feature_key(row), set()).add((row["attack_cat"], row["label"]))
            for name, counts in categories.items():
                counts[row[name]] += 1
            for name, stats in numeric.items():
                value = float(row[name])
                stats["count"] += 1
                stats["min"] = value if stats["min"] is None else min(stats["min"], value)
                stats["max"] = value if stats["max"] is None else max(stats["max"], value)
                stats["sum"] += value
                stats["integer_valued"] &= value.is_integer()
    for stats in numeric.values():
        stats["mean"] = stats.pop("sum") / stats["count"]
    return {
        "class_counts": dict(sorted(classes.items())),
        "binary_label_counts": dict(sorted(labels.items())),
        "categorical_counts": {name: dict(sorted(counts.items())) for name, counts in sorted(categories.items())},
        "numeric_summary": dict(sorted(numeric.items())),
        "type_policy": "CSV strings; numeric features validated as finite nonnegative floats. integer_valued describes observed training values, not a fitted dtype conversion.",
        "duplicate_ids": duplicate_ids,
        "inconsistent_targets": inconsistent_targets,
        "duplicate_feature_rows_beyond_first": sum(count - 1 for count in keys.values()),
        "conflicting_target_feature_groups": sum(len(targets) > 1 for targets in conflicts.values()),
        "recommendations": [
            "Consider seed-42 stratified validation using training data only; check rare-class support before choosing a fraction.",
            "Assess grouping exact duplicate training feature vectors within a single partition to avoid validation leakage.",
            "Resolve conflicting training targets and rare-class constraints before proposing four experiences; no groups are finalized.",
        ],
    }


def inspect_dataset(
    data_dir: Path | str,
    split_manifest: Path | str | None = None,
    *,
    profile_training: bool = True,
) -> dict:
    """Inventory expected files and return deterministic JSON-serializable findings.

    Without a manifest this performs structural discovery only. Profiling requires
    a manifest whose hashes, counts and logical roles match current evidence.
    """
    root = Path(data_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("data-dir must be a directory")
    report = {
        "report_version": 2,
        "data_dir": str(root),
        "seed": 42,
        "provenance": {"user_supplied_mirror": MIRROR, "official_split_description": SOURCE,
                       "authentication": "Local hashes only; no trusted upstream hashes supplied."},
        "inventory": {}, "splits": {}, "blockers": [],
        "training_profile": None,
        "training_profile_requested": profile_training,
        "test_policy": "Integrity and feature-only cross-split duplicates only; no target distribution or feature summary.",
    }
    before = {}
    for name in EXPECTED_FILES:
        path = root / name
        if not path.is_file():
            report["blockers"].append(f"Missing expected file: {name}")
            continue
        before[name] = (path.stat().st_size, path.stat().st_mtime_ns)
        report["inventory"][name] = _inventory(path)
    keys = {}
    for name in (TRAIN, TEST):
        if name not in report["inventory"]:
            continue
        split, keys[name] = _integrity(root / name)
        report["splits"][name] = split
        if not split.get("schema_matches") or not split["parseable"] or split["malformed_rows"]:
            report["blockers"].append(f"Schema or parse failure: {name}")
        if split["missing_values"] or split["invalid_values"]:
            report["blockers"].append(f"Missing or invalid values: {name}")
        split["physical_filename"] = name
    resolution = resolve_split_roles(report["splits"])
    report["split_resolution"] = resolution
    if resolution["status"] in {"invalid", "ambiguous"}:
        report["blockers"].append(resolution["reason"])
    manifest = None
    if split_manifest is None:
        report["blockers"].append("A verified split manifest is required before training profiling.")
    else:
        try:
            manifest = validate_split_manifest(split_manifest, resolution, report["inventory"], report["splits"])
        except (OSError, ValueError) as error:
            report["blockers"].append(str(error))
    for logical, physical in resolution["mapping"].items():
        report["splits"][physical]["logical_split"] = logical
    report["manifest_verified"] = manifest is not None
    if manifest is not None:
        report["provenance"]["role_verification"] = manifest["provenance_status"]
        report["provenance"]["filename_note"] = manifest["note"]
    if all(name in keys and report["splits"][name].get("schema_matches") and report["splits"][name]["parseable"] and not report["splits"][name]["malformed_rows"] for name in (TRAIN, TEST)):
        overlap = keys[TRAIN].keys() & keys[TEST].keys()
        report["cross_split_duplicates"] = {
            "definition": "SHA-256 of exact parsed feature strings, excluding id, attack_cat, label; no near-duplicate audit.",
            "distinct_shared_feature_vectors": len(overlap),
            "training_named_rows_matching": sum(keys[TRAIN][key] for key in overlap),
            "testing_named_rows_matching": sum(keys[TEST][key] for key in overlap),
        }
        for logical, physical in resolution["mapping"].items():
            report["cross_split_duplicates"][f"logical_{logical}_rows_matching"] = sum(keys[physical][key] for key in overlap)
    # Recheck fingerprints before allowing the profiling pass as well as after it.
    for name, metadata in before.items():
        path = root / name
        if metadata != (path.stat().st_size, path.stat().st_mtime_ns) or sha256(path) != report["inventory"][name]["sha256"]:
            report["blockers"].append(f"File changed during inspection: {name}")
            report["training_profile"] = None
    if not report["blockers"] and profile_training:
        training = resolution["mapping"]["training"]
        report["training_profile"] = _training_profile(root / training, keys[training])
        report["training_profile"].update(logical_split="training", physical_filename=training)
        for name, metadata in before.items():
            path = root / name
            if metadata != (path.stat().st_size, path.stat().st_mtime_ns) or sha256(path) != report["inventory"][name]["sha256"]:
                report["blockers"].append(f"File changed during profiling: {name}")
                report["training_profile"] = None
    report["status"] = "blocked" if report["blockers"] else "complete"
    return report


def verified_split_paths(data_dir: Path | str, split_manifest: Path | str) -> dict[str, Path]:
    """Return logical paths for future loaders only after all checks pass.

    Does not profile targets. Call at load time; paths do not protect against
    subsequent external changes to files. Never derive roles from the basename.
    """
    report = inspect_dataset(data_dir, split_manifest, profile_training=False)
    if report["status"] != "complete":
        raise ValueError("Split validation blocked: " + "; ".join(report["blockers"]))
    root = Path(data_dir).expanduser().resolve()
    return {logical: root / physical for logical, physical in report["split_resolution"]["mapping"].items()}
