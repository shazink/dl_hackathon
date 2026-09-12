# Data card

## Provenance and policy

- Intended dataset: UNSW-NB15. User supplied files at `/home/Ima/Downloads/archive` via [this Kaggle mirror](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/code).
- The [official UNSW description](https://research.unsw.edu.au/projects/unsw-nb15-dataset), checked 2026-09-12 and explicitly confirmed by the user, describes 175,341 training rows and 82,332 testing rows. Structural validation resolves the mirror as `swapped_filenames`. Official split roles are verified by documented counts and structural checks under the user-authorized rule. Byte-for-byte equality with an official UNSW-hosted copy remains unverified; Kaggle is a mirror, not the primary authority.
- Use the logical roles in `configs/unsw_nb15_splits.toml`, never physical filename suffixes. Validation must come only from logical training data, with seed 42. Logical test data must not guide fitting, tuning, early stopping, task design, feature selection, preprocessing, or experience construction. Fit preprocessing only on permitted training data.
- The user's Phase 2 exception allows test integrity, counts, schema/parseability, ID presence/uniqueness/range/continuity, missing/invalid/infinite-value checks, and feature-only cross-split duplicate auditing. The pipeline produced no logical-test target distribution or feature summary.
- Keep datasets external or in ignored `data/raw/`, `data/downloads/`, `data/interim/`, or `data/processed/`. Never commit raw or derived dataset contents. Downloaded files must remain unmodified.

## Verified local inventory — 2026-09-12

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `NUSW-NB15_features.csv` | 4,044 | `c55f19cceebb6360dc50f44f8a5f246ccefbcf8a6c604ac1ad46e643869cafce` |
| `UNSW-NB15_1.csv` | 168,979,718 | `7d851bbeabd27894ce39c8e78835c73341fc946652fb7743b9eff193b55eb511` |
| `UNSW-NB15_2.csv` | 165,221,021 | `6130ad02873cc6069ae695cf2844f2e8c2e9a9a1b7532dd82ab8f202757cacf8` |
| `UNSW-NB15_3.csv` | 154,588,103 | `ae990a96c3dfcd425ce2801aadb1727a34d5e0ae6d8215dbcdae60dedfaef640` |
| `UNSW-NB15_4.csv` | 97,588,754 | `cdf563692d51d405541dd659ddcdad9fa01f001f05fe9fc4b67f00ca12fbc96a` |
| `UNSW-NB15_LIST_EVENTS.csv` | 4,639 | `5b40f8128e2c87e691157c76debde04c328b0eb66bab97f1e7e17625f84497f2` |
| `UNSW_NB15_training-set.csv` | 15,380,800 | `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` |
| `UNSW_NB15_testing-set.csv` | 32,293,018 | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` |

All eight files were present. Before/after SHA-256, size, and modification-time checks agreed. Raw shards and ancillary files were not parsed by the pipeline or combined with the split files.

## Verified split integrity

| Local filename | Data rows | Columns | Ordered schema | Malformed rows |
| --- | ---: | ---: | --- | ---: |
| `UNSW_NB15_training-set.csv` | 82,332 | 45 | matches expected | 0 |
| `UNSW_NB15_testing-set.csv` | 175,341 | 45 | matches expected | 0 |

Both `id` columns are present and unique, with no gaps in `1..row_count`: the training-named file has minimum 1 and maximum 82,332; the testing-named file has minimum 1 and maximum 175,341. Continuity describes coverage of the ID range, not temporal ordering. No missing or infinite values were found. Ordered schemas and IDs were verified before any target profiling.

| Logical split | Physical filename | Canonical rows |
| --- | --- | ---: |
| training | `UNSW_NB15_testing-set.csv` | 175,341 |
| testing | `UNSW_NB15_training-set.csv` | 82,332 |

The portable manifest pins this mapping and both SHA-256 fingerprints. Current hashes, counts, ordered schemas, parsing, and ID integrity must pass before profiling or returning paths to future loaders. Any mapping or fingerprint drift blocks loading; no automatic manifest rewrite is permitted.

Both split CSVs parse using UTF-8 with optional BOM. Columns comprise `id`, 42 feature columns, and targets `attack_cat` and `label`. Nominal feature fields are `proto`, `service`, and `state`; the other 39 feature fields pass finite nonnegative numeric checks. These are schema/integrity observations, not fitted preprocessing choices.

The implemented checks found no missing or invalid values in either split. Missing means blank/whitespace or a case-insensitive `nan`, `na`, `null`, or `none` token. The `-` token is preserved, not declared missing. Numeric validity checks finite nonnegative values, positive integral IDs, and binary labels in `{0, 1}`. This is not exhaustive domain validation: categorical vocabularies, all integer-field constraints, and semantic feature ranges are not checked.

The permitted exact feature-only cross-split audit found **1,302 distinct shared feature vectors**, involving **8,541 training-named rows** and **10,279 testing-named rows**. Keys exclude `id`, `attack_cat`, and `label` and hash exact parsed feature strings. Numeric-equivalent strings and near duplicates are not detected. No rows were removed, no labels were compared across splits, and this result must not inform task or feature design.

## Training inspection status and unresolved questions

The earlier filename-only run was blocked and profiled neither file. After the user authorized count-plus-structure resolution and the manifest was verified, the logical training-only run completed with `status: complete`. The generated report is `artifacts/data/unsw_nb15_profile.json` (ignored by Git).

Logical training class counts, totaling 175,341:

| Class | Training rows |
| --- | ---: |
| Analysis | 2,000 |
| Backdoor | 1,746 |
| DoS | 12,264 |
| Exploits | 33,393 |
| Fuzzers | 18,184 |
| Generic | 40,000 |
| Normal | 56,000 |
| Reconnaissance | 10,491 |
| Shellcode | 1,133 |
| Worms | 130 |

Logical training binary labels: 56,000 normal (`0`) and 119,341 attack (`1`). No inconsistency was found between binary labels and the Normal category. There are 133 protocol, 13 service, and 9 state values. All 39 numeric features have nonconstant values; their min/max/mean and observed integer-valued status are in the artifact. These summaries are inspection output, not preprocessing decisions.

Logical training contains 74,301 repeated exact feature rows beyond the first occurrence, 1,772 exact feature groups with conflicting target tuples, and zero duplicate IDs. No data was cleaned or removed.

Training-only recommendations: assess seed-42 stratified validation with rare-class support explicitly checked (Worms has only 130 rows), and assess keeping identical feature vectors in one derived partition. Resolve the policy for conflicting-target feature groups before implementation. Do not deduplicate, relabel, choose a validation fraction, or assign experience groups automatically. These recommendations use only logical training evidence, not cross-split overlap or logical-test statistics.

- Phase 3/4 now follows the user's locked 80/20 row-stratified policy: duplicates and conflicting labels are preserved. This supersedes the earlier open split-policy recommendation; validation is row-disjoint, not guaranteed feature-disjoint.
- Four-experience grouping and global class mapping are locked in `configs/experiences.toml`; see `docs/data_preparation.md`.
- Dataset licensing and citation obligations remain to be verified; the project's MIT license does not license the dataset.

## Inspection checklist

- [x] Locate the eight supplied files without copying or modifying them.
- [x] Record byte sizes, SHA-256 fingerprints, split schema, row counts, and integrity checks.
- [x] Audit feature-only cross-split duplicates without reporting test target distributions.
- [x] Resolve official split roles from documented counts plus structural validation, with a pinned logical manifest. Official byte identity remains unverified.
- [x] Inspect logical training class counts, categorical values, numeric summaries, and duplicate groups.
- [x] Document training-derived validation recommendations for subsequent implementation.

## Phase 3/4 verified preparation

Logical training yields 140,272 development and 35,069 validation rows. Development attack counts in the lock are Generic 32,000; Exploits 26,714; Fuzzers 14,547; DoS 9,811; Reconnaissance 8,393; Analysis 1,600; Backdoor 1,397; Shellcode 906; Worms 104. Normal has 44,800 development rows split evenly across experiences. Validation is never used to order attacks or fit preprocessing.

The exact locked assignment fingerprint is `f51d055d4661531ad80a39cbf6c749c4c4a6e3241ece6c132a3f4129d074ae08`. E1 development supplies all 44,210 fitting rows; the frozen processor produces 156 finite float32 features for all eight subsets. Both targets and `id` are excluded. Generated artifacts are under `artifacts/data/prepared/`; an independent run under `prepared-repeat/` matched every file checksum. These are data preparation checks, not model experiments. No logical-test arrays or target distributions were generated.
