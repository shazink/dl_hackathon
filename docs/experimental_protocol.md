# Experimental protocol

## Fixed decisions

- Dataset: UNSW-NB15, using the official training and test split.
- The official test split remains evaluation-only. The explicit Phase 2 exception permits file integrity/SHA-256, row and column counts, header/schema compatibility, parseability, missing/invalid-value integrity checks, and feature-only cross-split duplicate auditing. It does not permit test target distributions or test feature summaries.
- Validation data is created only from official training data.
- Four experiences are locked in `configs/experiences.toml` using development-only attack counts and the prescribed snake algorithm. Normal recurs through disjoint subsets; each attack class is introduced exactly once.
- Global deterministic seed: 42. Seed Python, NumPy, PyTorch, CUDA where applicable, and DataLoader workers.
- Fit preprocessing once on E1 development only; freeze it for future experiences and validation.
- Core methods: naive sequential fine-tuning, uniform replay, and TAFR.
- Planned ablations: TAFR-F, TAFR-FU, and full TAFR, subject to the authoritative project specification. Include at least one ablation in the final study.
- The final pipeline must support a reproducible one-command run.
- The final report is limited to four pages.
- Never fabricate, hide, or silently replace experimental results.

## Prohibited leakage paths

The official test split must not influence model or preprocessing fitting, hyperparameter tuning, early stopping, feature selection, validation construction, task design, class order, experience grouping, or any other development decision. Validation must not cross its derived boundary into fitting. Preprocessing statistics and learned vocabularies must come only from data permitted for training at that stage.

Integrity findings may block dataset acceptance; they must not guide feature selection, experience construction, or validation design. If split provenance is ambiguous, withhold profiling of both files until identity is resolved. Do not silently remap filenames.

The user-authorized split-resolution rule uses documented canonical counts (175,341 training; 82,332 testing) plus ordered schema, required-column, parseability, and ID presence/uniqueness/minimum/maximum/continuity checks. Outcomes are exactly `canonical_filenames`, `swapped_filenames`, `ambiguous`, or `invalid`; the latter two block profiling. These structural checks are permitted on logical test data. Missing and infinite-value integrity checks also block acceptance when they fail.

This mirror resolves as `swapped_filenames`. `configs/unsw_nb15_splits.toml` explicitly maps logical training to the physical testing-named file, and logical testing to the physical training-named file. Verify pinned hashes, counts, schema, and mapping on every load. Official byte identity remains unverified; no labels or class frequencies were used to select logical roles. All references to training/testing in scientific rules mean logical roles.

## Fairness requirements

All methods must use the same data split, class order, architecture, optimizer, training budget, evaluation protocol, and replay-memory budget wherever applicable. Randomness must be controlled consistently. Any unavoidable method-specific exception must be explicitly justified and reported.

## Required metrics and artifacts

- Macro-F1
- Balanced accuracy
- Per-class recall
- Forgetting
- Forward transfer
- Runtime
- Memory usage
- Confusion matrices
- Continual-learning performance matrix

Metric definitions, aggregation rules, and measurement tooling must be fixed before results are produced.

## Unresolved decisions

- Authoritative definitions of TAFR-F, TAFR-FU, and full TAFR.
- Model architecture, optimizer, training budget, replay-memory budget, and metric implementation details.
- Which planned ablation(s) will appear in the final study.

## Locked Phase 3/4 extension

The target is canonical `attack_cat`, with `label` used only for binary consistency checks. Exclude both targets and `id` from features. A single immutable global class map is shared across methods and experiences. Use seed-42 stratified 80/20 development/validation row assignments from logical training only. The tracked experience config records all counts, assignments, policies, and fingerprints. See `docs/data_preparation.md` for the exact specification and verified sizes.

The user-required row split preserves duplicate feature vectors and conflicting labels. Zero row overlap is verified; feature-vector overlap across development/validation is possible and may inflate validation performance. This limitation must be reported in the final study. No deduplication, relabeling, or grouped split is silently substituted.
