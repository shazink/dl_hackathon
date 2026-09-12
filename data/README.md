# Local data

Place local UNSW-NB15 files under `data/raw/` or `data/downloads/`. Derived data may use `data/interim/` and `data/processed/`. Contents of those directories are ignored and must not be committed.

Keep the downloaded dataset in an external local directory. After installing the package, run:

```bash
python scripts/inspect_unsw_nb15.py --data-dir /path/to/archive \
  --split-manifest configs/unsw_nb15_splits.toml \
  --output artifacts/data/unsw_nb15_profile.json
```

The script hashes all eight expected files and checks both split CSVs for schema, parseability, ID integrity, missing/infinite values, invalid numeric values, binary-label validity, and feature-only overlap. Raw shards and ancillary CSVs are hashed only, not profiled or combined. It emits JSON to stdout or `--output` and returns 2 on validation blockers. Generated reports belong under an ignored directory such as `artifacts/`, never in the downloaded directory.

Structural verification confirmed swapped filenames. The manifest maps logical training to `UNSW_NB15_testing-set.csv` and logical testing to `UNSW_NB15_training-set.csv`, with pinned hashes and canonical counts. Files remain unchanged. Logical roles are verified by counts and structure; official byte identity remains unverified. Future loaders must call `tafr_ids.data.inspection.verified_split_paths(data_dir, split_manifest)` at load time and use its logical keys; it validates without profiling targets. Do not infer roles from basenames or silently refresh fingerprints.

Do not inspect or use the official test split to fit preprocessing, tune parameters, stop training, select features, design tasks, or finalize class experiences. Only the explicitly permitted Phase 2 integrity audit is allowed. See [the data card](../.ai/data_card.md).

Phases 3/4 preparation reads logical training only after verification. Use `scripts/prepare_unsw_nb15.py` with both tracked manifests and a fresh output directory under `artifacts/data/`. See [data preparation](../docs/data_preparation.md) for the command and artifact contract. Generated feature arrays, assignments, and serialized preprocessors must not be committed.
