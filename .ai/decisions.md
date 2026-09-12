# Decision log

Use this lightweight record for material scientific and architectural decisions.

## D-001 — Dataset and split boundary

- **Date:** 2026-09-12
- **Status:** accepted
- **Decision:** Use UNSW-NB15 with its official training and test split. Derive validation only from official training data, and leave the official test split untouched until evaluation.
- **Rationale:** Fixed project constraint and leakage boundary.
- **Consequences:** Test data cannot influence fitting, tuning, early stopping, task design, or feature selection.

## D-002 — Reproducibility seed

- **Date:** 2026-09-12
- **Status:** accepted
- **Decision:** Use global deterministic seed 42 for Python, NumPy, PyTorch, CUDA where applicable, and DataLoader workers.
- **Rationale:** Fixed project constraint.
- **Consequences:** Future pipelines must centralize and verify deterministic seeding.

## D-003 — Core comparisons and fairness

- **Date:** 2026-09-12
- **Status:** accepted
- **Decision:** Compare naive sequential fine-tuning, uniform replay, and TAFR under the same split, class order, architecture, optimizer, training budget, evaluation protocol, and replay-memory budget wherever applicable.
- **Rationale:** Fixed scientific comparison requirements.
- **Consequences:** Method-specific deviations must be justified and documented.

## D-004 — Four-experience class grouping

- **Date:** 2026-09-12
- **Status:** resolved by D-009 after logical-training inspection.
- **Decision:** Use four class-incremental experiences, but do not finalize their exact class grouping until class counts are inspected.
- **Rationale:** The experience sequence must be data-informed without consulting the test split for task design.
- **Consequences:** No exact class grouping may be treated as authoritative yet.

## D-005 — Phase 2 test integrity exception

- **Date:** 2026-09-12
- **Status:** accepted
- **Decision:** Per the user's Phase 2 instruction, permit only test file integrity/SHA-256, row/column counts, schema compatibility, parseability, missing/invalid-value checks, and feature-only cross-split duplicate auditing before evaluation.
- **Rationale:** Verify downloaded data without profiling test targets or using test features for scientific design.
- **Consequences:** This qualifies D-001's untouched wording but preserves its evaluation-only purpose. No test target distributions or descriptive feature statistics may be reported. Raw shards and ancillary files are hashed only.

## D-006 — Block profiling when split identity conflicts

- **Date:** 2026-09-12
- **Status:** superseded by D-008 for the explicitly authorized count-plus-structure resolution; ambiguous/invalid inputs remain blocked.
- **Decision:** Withhold training profiles when split sizes conflict with the [official UNSW description](https://research.unsw.edu.au/projects/unsw-nb15-dataset). Do not infer a filename reversal.
- **Rationale:** Local training-named and testing-named files have row counts opposite to the published split sizes. Counts alone do not authenticate provenance.
- **Consequences:** Inventory/integrity work has run, but class counts, training summaries, validation recommendations, and experience design remain blocked pending trusted split verification. No dataset is renamed or altered.

## D-007 — Minimal deterministic inspection implementation

- **Date:** 2026-09-12
- **Status:** accepted
- **Decision:** Use standard-library CSV streaming and SHA-256, stable JSON output, and exit code 2 for blocked validation. Exclude `id`, `attack_cat`, and `label` from feature duplicate keys.
- **Rationale:** This phase needs no additional runtime dependencies or transformations.
- **Consequences:** Duplicate detection covers exact parsed feature strings only, not numerically equivalent representations or near duplicates. Full file hashes are checked again after inspection. No random operations or split construction occur.

## D-008 — Verified logical split manifest for this mirror

- **Date:** 2026-09-12
- **Status:** accepted
- **Decision:** Under the user's explicit authority, resolve roles using official row counts plus ordered schema, parsing, required-column, and unique contiguous ID checks. The verified result is `swapped_filenames`: logical training is physical `UNSW_NB15_testing-set.csv` (175,341 rows), and logical testing is physical `UNSW_NB15_training-set.csv` (82,332 rows). Pin both hashes and roles in `configs/unsw_nb15_splits.toml`.
- **Rationale:** Structural evidence establishes roles without consulting target distributions. Kaggle is a mirror; official metadata is the authority for split sizes. Official byte equality is unverified.
- **Consequences:** No physical file is renamed or modified. The inspector and future loaders must validate the manifest, hashes, counts, schema, and IDs and fail closed on drift. Future loaders use `verified_split_paths`, never infer roles from filenames. Only logical training may be profiled. ID range/continuity checks extend the explicitly permitted test integrity audit. D-006's prior blanket block is superseded; ambiguous and invalid resolutions still block profiling.

## D-009 — Locked multiclass and four-experience protocol

- **Date:** 2026-09-12
- **Status:** accepted
- **Decision:** Follow the user's Phase 3 rule: canonical `attack_cat` target, global immutable class indices, seed-42 stratified 80/20 row split, and development-count snake allocation. Lock E1 Generic/Shellcode/Worms, E2 Exploits/Backdoor, E3 Fuzzers/Analysis, E4 DoS/Reconnaissance in `configs/experiences.toml`. Normal recurs via four disjoint subsets per partition.
- **Rationale:** Counts, tie-breaking, and allocation are deterministic and training-only; no model results or logical-test labels select the order.
- **Consequences:** Every development/validation row is assigned once. Existing duplicate feature vectors and conflicting labels are preserved as explicitly required by the row split; possible feature overlap can make validation optimistic and must be disclosed. Assignment fingerprint and source hash are locked before model results. Canonicalization accepts trimmed case-insensitive canonical names and Backdoors -> Backdoor; unknowns fail. `label` and `id` never enter model features.

## D-010 — Frozen E1-only preprocessing and reproducible arrays

- **Date:** 2026-09-12
- **Status:** accepted
- **Decision:** Use sklearn ColumnTransformer/Pipelines for E1-development median imputation and standardization, plus E1 most-frequent categorical imputation and one-hot encoding with unknown categories ignored. Fit once, freeze, and transform later/validation subsets into finite float32 arrays with stable ordering.
- **Rationale:** The user's explicit authority boundary prohibits future-experience and validation fitting. NumPy, pandas, scikit-learn, and joblib are needed for these components; PyTorch is not added.
- **Consequences:** All-missing E1 columns, unexpected schema/types, infinite inputs, or target columns fail. Persist assignments, fit indices, frozen processor, arrays, ordered features, hashes, and versions outside Git. Two real runs reproduced all artifacts byte-for-byte. The verified dependency environment is pinned in `requirements-repro.txt`; the standard-library inspector remains independent of these imports. No model has been trained.

## New entry template

### D-NNN — Title

- **Date:** YYYY-MM-DD
- **Status:** proposed | accepted | superseded | unresolved
- **Decision:**
- **Rationale:**
- **Consequences:**
