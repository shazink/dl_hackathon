"""Verified logical-training loading and strict multiclass target handling."""

from types import MappingProxyType
import tomllib

import numpy as np
import pandas as pd

from tafr_ids.data.inspection import CATEGORICAL, FEATURES, HEADER, MISSING, sha256, verified_split_paths

CLASSES = ("Normal", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms")
CLASS_TO_INDEX = MappingProxyType({name: index for index, name in enumerate(CLASSES)})
ALIASES = {name.casefold(): name for name in CLASSES} | {"backdoors": "Backdoor"}


def normalize_targets(values):
    result = []
    for value in values:
        if not isinstance(value, str) or value.strip().casefold() not in ALIASES:
            raise ValueError("Unknown or missing attack_cat value")
        result.append(ALIASES[value.strip().casefold()])
    return np.asarray(result)


def feature_frame(frame):
    if list(frame.columns) != HEADER:
        raise ValueError("Unexpected ordered training schema")
    features = frame.loc[:, FEATURES].copy()
    for name in FEATURES:
        features[name] = features[name].map(
            lambda value: np.nan if isinstance(value, str) and value.strip().lower() in MISSING else value
        )
        if name not in CATEGORICAL:
            features[name] = pd.to_numeric(features[name], errors="raise").astype(np.float64)
            if np.isinf(features[name].to_numpy()).any():
                raise ValueError("Infinite numeric feature")
    return features


def load_training(data_dir, split_manifest):
    paths = verified_split_paths(data_dir, split_manifest)
    with open(split_manifest, "rb") as stream:
        expected = tomllib.load(stream)["training"]
    path = paths["training"]
    if sha256(path) != expected["sha256"]:
        raise ValueError("Training source changed before load")
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    if sha256(path) != expected["sha256"] or len(frame) != expected["expected_rows"]:
        raise ValueError("Training source changed during load")
    targets = normalize_targets(frame["attack_cat"])
    binary = frame["label"].str.strip().to_numpy()
    if not np.array_equal(binary, np.where(targets == "Normal", "0", "1")):
        raise ValueError("Binary label disagrees with canonical multiclass target")
    return feature_frame(frame), targets, expected["sha256"]
