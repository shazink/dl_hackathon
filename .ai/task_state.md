# Task state

- **Current phase:** Phases 1–4 complete; ready for Phase 5.
- **Completed:** Scaffold; verified swapped logical split manifest; training-only inventory/profile; strict multiclass normalization and immutable global mapping; locked development-count snake experiences; seed-42 80/20 assignments; E1-only frozen preprocessing; reproducible data artifacts.
- **Logical mapping:** Training = physical `UNSW_NB15_testing-set.csv` (175,341 rows); testing = physical `UNSW_NB15_training-set.csv` (82,332 rows), both hash-pinned. No logical-test target distributions have been reported.
- **Locked protocol:** `configs/experiences.toml`, v1.0. E1 Generic/Shellcode/Worms; E2 Exploits/Backdoor; E3 Fuzzers/Analysis; E4 DoS/Reconnaissance. Normal recurs with disjoint rows. Assignments SHA-256: `f51d055d4661531ad80a39cbf6c749c4c4a6e3241ece6c132a3f4129d074ae08`.
- **Artifacts:** `artifacts/data/prepared/` and verification copy `artifacts/data/prepared-repeat/` (ignored). 140,272 development and 35,069 validation rows; E1 fit scope exactly 44,210 development rows; 156 finite float32 columns.
- **Next task:** Phase 5 naive sequential fine-tuning baseline, after fixing shared architecture, optimizer, training budget, and evaluation details.
- **Unresolved:** Authoritative TAFR components and later model/evaluation settings. Official byte identity is unverified. Row-stratified validation preserves duplicate/conflicting-label rows and may share identical feature vectors across partitions; disclose this limitation.
- **Last updated:** 2026-09-12.

## Verification

- Prerequisites: 37 existing tests passed; fresh manifest/count/schema/hash validation confirmed swapped roles without profiling targets.
- Editable install succeeded with NumPy, pandas, scikit-learn, and joblib; `pip check` passed. Verified Python 3.14.7 environment is pinned in `requirements-repro.txt`.
- `.venv/bin/python -m pytest -q`: 48 passed.
- `.venv/bin/python scripts/prepare_unsw_nb15.py --data-dir /home/Ima/Downloads/archive --split-manifest configs/unsw_nb15_splits.toml --experience-config configs/experiences.toml --output-dir artifacts/data/prepared`: passed; repeated with `prepared-repeat`.
- Independent artifact checks confirmed byte-identical output files, saved hashes, finite float32 arrays, global labels, complete/disjoint row coverage, and fit indices identical to E1 development.
- `git diff --check`: passed. Remote main was fetched and matched the local starting commit. The user explicitly authorized committing all valid project work and a safe non-force push to main; publication outcome is reported in the final handoff.

No model has been trained. No logical-test arrays have been prepared. Raw files remain external and unchanged. The experiment log still correctly records no ML experiments.
