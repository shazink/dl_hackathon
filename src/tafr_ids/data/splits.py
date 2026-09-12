"""Seed-42 row-stratified assignments; indices are zero-based CSV data rows."""

import hashlib

import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42


def development_validation(targets):
    indices = np.arange(len(targets), dtype=np.int64)
    development, validation = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=targets)
    development, validation = np.sort(development), np.sort(validation)
    if np.intersect1d(development, validation).size or not np.array_equal(np.sort(np.concatenate([development, validation])), indices):
        raise ValueError("Split overlap or incomplete coverage")
    if set(targets[development]) != set(targets) or set(targets[validation]) != set(targets):
        raise ValueError("A rare class is absent from a partition")
    partition = np.zeros(len(targets), dtype=np.uint8)
    partition[validation] = 1
    return partition


def assignment_bytes(partition, experience):
    return ("row_index,partition,experience\n" + "".join(
        f"{index},{'development' if int(part) == 0 else 'validation'},{int(exp)}\n"
        for index, (part, exp) in enumerate(zip(partition, experience, strict=True))
    )).encode("utf-8")


def assignment_fingerprint(partition, experience):
    return hashlib.sha256(assignment_bytes(partition, experience)).hexdigest()
