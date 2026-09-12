# Locked data preparation protocol v1.0

The Phase 3/4 protocol is locked in `configs/experiences.toml` before any model results exist. Preparation uses only logical training; logical testing is read only by the existing permitted integrity verifier. Physical filenames never determine roles.

## Targets and row assignment

`attack_cat` is the multiclass target. Strip surrounding whitespace and compare case-insensitively to the ten canonical class names; additionally accept `Backdoors` as `Backdoor`. Other unknown or missing values fail. `label` is used only to confirm Normal = 0 and every attack = 1. The immutable global mapping is Normal=0, Analysis=1, Backdoor=2, DoS=3, Exploits=4, Fuzzers=5, Generic=6, Reconnaissance=7, Shellcode=8, Worms=9. Every future model uses one global single head.

Create the 80/20 row-stratified split with scikit-learn `train_test_split(test_size=0.2, stratify=canonical_targets, random_state=42)`. Store sorted zero-based data-row indices (header excluded). The real split has 140,272 development rows and 35,069 validation rows; each class is preserved in both partitions. No row is omitted or shared.

Sort attack classes by decreasing development count and canonical name for ties. Allocate along `1,2,3,4,4,3,2,1`, repeating as necessary. The locked introduction order is:

| Experience | Newly introduced attacks | Development rows including Normal | Validation rows including Normal |
| --- | --- | ---: | ---: |
| E1 | Generic, Shellcode, Worms | 44,210 | 11,053 |
| E2 | Exploits, Backdoor | 39,311 | 9,828 |
| E3 | Fuzzers, Analysis | 27,347 | 6,837 |
| E4 | DoS, Reconnaissance | 29,404 | 7,351 |

Normal recurs but its rows never recur. For each partition independently, shuffle Normal indices using NumPy PCG64 with `SeedSequence([42, partition_code])` (development=0, validation=1), then `array_split` into four subsets. Here each experience has 11,200 development and 2,800 validation Normal rows. Every attack row is assigned only to its introduction experience.

The assignment CSV SHA-256 is `f51d055d4661531ad80a39cbf6c749c4c4a6e3241ece6c132a3f4129d074ae08`. The preparation command recomputes the entire config and rejects any mismatch; it never regenerates the lock silently.

## E1-only fitting

After manifest verification, target validation, splitting, and experience construction, fit preprocessing only on E1 development's 44,210 feature rows. Remove `id`, `attack_cat`, and `label`. Schema order must exactly match the 42 allowed features. Numeric/categorical types are determined from E1 feature dtypes and checked against the declared numeric and nominal schema.

Use a scikit-learn ColumnTransformer: 39 numeric columns receive E1 median imputation and StandardScaler; `proto`, `service`, and `state` receive E1 most-frequent imputation and OneHotEncoder with unknown categories ignored. Numeric blocks appear first, then categorical blocks, each in source feature order. Unknown categories have all-zero encoding within that field and cannot expand the output. Missing-token handling matches inspection; `-` remains a categorical value. Infinite input, changed schema/types, target columns, and all-missing E1 columns fail explicitly. E1 has observations for every column in this run.

The wrapper permits fitting once with E1/development scope and rejects refitting, including after serialization. Future and validation subsets call transform only. Output is finite float32 with 156 columns in this run. Fitted transformers and feature order are serialized for later use. Load joblib files only from trusted local runs.

## Artifacts and reproducibility

Run from the repository root after installing the package:

```bash
python scripts/prepare_unsw_nb15.py \
  --data-dir ~/Downloads/archive \
  --split-manifest configs/unsw_nb15_splits.toml \
  --experience-config configs/experiences.toml \
  --output-dir artifacts/data/prepared
```

Choose a new output directory for each run; existing directories are not overwritten. A failed write can leave an incomplete directory; only a run with `metadata.json` and successful exit is complete. Artifacts are ignored by Git:

- `assignments.csv`: one row per logical-training row, partition, and experience.
- `preprocessor_fit_indices.npy`: exactly E1 development indices.
- `preprocessor.joblib`: fitted, frozen transformer.
- `E{1..4}_{development,validation}_{X,y,indices}.npy`: eight subsets, float32 features, int64 global target indices, and original row indices.
- `metadata.json`: protocol, ordered feature names, fit scope, shapes, package versions, configuration hashes, and SHA-256 of every other artifact.

Use Python 3.14.7 and `python -m pip install -r requirements-repro.txt -e '.[dev]'` to recreate the verified environment. General package compatibility remains Python 3.11+; other environments must still match the locked assignment fingerprint. No logical-test arrays are generated. No ML model is trained.

## Scientific limitations

The explicitly requested row-stratified split preserves all records, including duplicate feature vectors and conflicting labels. Disjoint row indices do not imply disjoint feature vectors: development/validation may share identical vectors, potentially making validation optimistic. The 1,772 conflicting-target feature groups observed in logical training are preserved without relabeling. This is a documented protocol limitation, not a claim of group-held-out validation. Any alternative grouped study must be separately specified and versioned; do not silently change this lock.

Official split roles are verified by counts and structure; equality to official hosted bytes remains unverified. Class order was selected by development counts only, without model results or logical-test statistics.
