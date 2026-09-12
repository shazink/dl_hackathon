"""Reproducible training-only artifacts; no implicit protocol creation."""

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from tafr_ids.data.experiences import construct_experiences, validate_experience_config
from tafr_ids.data.inspection import sha256
from tafr_ids.data.loader import CLASS_TO_INDEX, load_training
from tafr_ids.data.preprocessing import FrozenPreprocessor
from tafr_ids.data.splits import assignment_bytes, development_validation


def prepare(data_dir, split_manifest, experience_config, output_dir):
    output = Path(output_dir).expanduser().resolve()
    source = Path(data_dir).expanduser().resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from downloaded data")
    if output.exists():
        raise ValueError("Output directory already exists; choose a new run directory")
    features, targets, training_hash = load_training(data_dir, split_manifest)
    partition = development_validation(targets)
    experience, protocol = construct_experiences(targets, partition, training_hash)
    validate_experience_config(experience_config, protocol)
    e1 = np.flatnonzero((partition == 0) & (experience == 1))
    processor = FrozenPreprocessor().fit_e1(features.iloc[e1], partition="development", experience=1)
    encoded = np.asarray([CLASS_TO_INDEX[name] for name in targets], dtype=np.int64)
    output.mkdir(parents=True)
    (output / "assignments.csv").write_bytes(assignment_bytes(partition, experience))
    np.save(output / "preprocessor_fit_indices.npy", e1, allow_pickle=False)
    joblib.dump(processor, output / "preprocessor.joblib", compress=0)
    summaries = {}
    for exp in range(1, 5):
        for part, name in ((0, "development"), (1, "validation")):
            indices = np.flatnonzero((partition == part) & (experience == exp))
            values = processor.transform(features.iloc[indices])
            prefix = f"E{exp}_{name}"
            np.save(output / f"{prefix}_X.npy", values, allow_pickle=False)
            np.save(output / f"{prefix}_y.npy", encoded[indices], allow_pickle=False)
            np.save(output / f"{prefix}_indices.npy", indices, allow_pickle=False)
            summaries[prefix] = {"rows": len(indices), "features": values.shape[1]}
    metadata = {
        "protocol": protocol,
        "feature_names": processor.feature_names_,
        "fit_scope": "E1 development only",
        "fit_rows": len(e1),
        "subsets": summaries,
        "logical_test": "Integrity verification only; no targets profiled or arrays prepared",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
        "split_manifest_sha256": sha256(Path(split_manifest)),
        "experience_config_sha256": sha256(Path(experience_config)),
        "files": {path.name: sha256(path) for path in sorted(output.iterdir())},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
