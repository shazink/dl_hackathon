# Inference assets

The `models/` tree is the portable runtime boundary for the five finalized seed-42 methods. It exists so a clean clone can perform inference without complete training checkpoints, prepared arrays, raw data, or access to the original training machine.

## Inventory

| Path | Purpose |
| --- | --- |
| `models/{naive,uniform,tafr_f,tafr_fu,tafr}/model.safetensors` | Final model tensors only for one method. |
| `models/{method}/metadata.json` | Method ID, seed, architecture, dimensions, source/config/data/preparation fingerprints, metric references, serialization versions, and hashes. |
| `models/shared/preprocessor.skops` | Frozen E1-development transformer used for raw 42-feature CSV inference. |
| `models/shared/feature_schema.json` | Ordered raw columns, numeric/categorical split, ordered transformed columns, dimension, and dtype. |
| `models/shared/class_mapping.json` | Immutable ten-class output order and name-to-index mapping. |
| `models/shared/simulation_scenarios.npz` | 80 finite transformed development vectors, eight per global class, for offline Attack Simulation. |
| `models/shared/SHA256SUMS` | SHA-256 for all 14 runtime JSON/binary assets. |

The scenario archive is not raw network traffic, a validation set, or a logical-test sample. It contains previously prepared development-only vectors and their development scenario labels/source metadata.

## Why these formats

Model files use [safetensors](https://huggingface.co/docs/safetensors/index), a tensor-focused format that avoids pickle-based model loading. The scikit-learn preprocessing graph uses [skops persistence](https://skops.readthedocs.io/en/stable/persistence.html), which exposes non-default types for review instead of automatically loading arbitrary pickle content.

The loader does not blindly trust a bundled type list. `tafr_ids.inference.bundle` requires the file's reported non-default types to equal this hard-coded allowlist:

```text
numpy.dtype
tafr_ids.data.preprocessing.FrozenPreprocessor
```

The checksum manifest is verified before dashboard inference, scenario use, or model loading. Shared class/schema contracts are then compared with package constants, and safetensors state keys/shapes must strictly match the reconstructed MLP.

## Provenance

Each method's metadata records:

- seed 42 and method identity;
- the `156 → 256 → 128 → 10` architecture configuration;
- source experiment, checkpoint, results, and configuration fingerprints;
- logical-training dataset and preparation fingerprints;
- assignment, experience-config, and split-manifest fingerprints;
- links and exact values from the tracked validation/final-test CSV rows;
- source/export PyTorch, safetensors, skops, and serialization-schema versions;
- SHA-256 values for its weights and shared runtime assets.

At baseline commit `74281faa03accbca321dfc38d4515e25ae77543c`, the exported tensors were checked tensor-for-tensor against all five finalized local `latest.pt` states. Their logits were bit-identical on all 80 bundled development-only scenario vectors. The tracked scenario archive was byte-identical to the previously verified local development-only archive. No training, preprocessing fit, or metric evaluation was performed during export.

Complete source checkpoints remain ignored. Replay checkpoints can contain optimizer state, RNG state, metric history, and replay-buffer samples; none belongs in an inference-only distribution.

## Verification

From the repository root:

```bash
python scripts/verify_inference_assets.py
```

The command checks:

1. all 14 entries in `models/shared/SHA256SUMS` exist and match;
2. the manifest has no missing, duplicate, unexpected, or escaping paths;
3. the skops type allowlist, transformed feature count, and original E1 fit-row count;
4. all five model identities, seed values, strict state dictionaries, and 75,146-parameter architectures;
5. finite `(80, 10)` probability outputs whose rows sum to one.

This is an integrity and inference smoke test, not a benchmark rerun. It does not score scenario labels or load logical test.

## Export process

`scripts/export_inference_assets.py` is the auditable exporter used with trusted local finalized artifacts. Its default inputs are ignored checkpoint/result directories, the ignored fitted Joblib preprocessor, the verified local scenario archive, and tracked metric CSVs. It:

- rejects an existing output directory;
- validates finalized checkpoint/config/preparation fingerprints;
- round-trips every tensor through safetensors and requires equality;
- exports the existing fitted processor to skops and checks the exact type set and feature order;
- validates and copies the existing development-only scenario archive;
- writes metadata and the checksum manifest.

Exporting is not needed for normal use because `models/` is tracked. Do not overwrite or regenerate the finalized bundles merely to update documentation. Any legitimate future export must retain accurate provenance and be reviewed as a scientific artifact change.

See [model directory notes](../models/README.md), [dashboard usage](dashboard.md), and [reproducibility](reproducibility.md).
