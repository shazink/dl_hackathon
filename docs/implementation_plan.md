# Implementation plan

## 1. Repository initialization

**Acceptance:** requested tree exists; package installs in editable mode; import/version test passes; metadata parses; ignore rules protect secrets, datasets, and generated outputs; context documents agree. Completed.

## 2. Dataset acquisition and inspection

**Acceptance:** user-provided UNSW-NB15 files are located without committing them; official split identity, schema, training class counts, integrity checks, and fingerprints are recorded; test data does not inform design.

Completed under the documented count-plus-structure role-verification rule; official byte equality remains unverified. Phases 5–10 are also complete for the frozen seed-42 study.

## 3. Experience-sequence finalization

**Acceptance:** four class-incremental experiences and class order are justified from permitted training evidence, recorded as a decision, configured reproducibly, and tested for disjointness and coverage.

Completed: the prescribed development-count snake allocation and disjoint recurring Normal subsets are locked in `configs/experiences.toml`.

## 4. Leakage-safe preprocessing

**Acceptance:** transformations fit only on permitted training data; state can be serialized and reused; validation/test application cannot refit; leakage and determinism tests pass.

Completed for development/validation preparation: fit once on E1 development and freeze. Logical-test transformation remains deferred to evaluation. See `docs/data_preparation.md`.

## 5. Naive sequential fine-tuning baseline

**Acceptance:** shared model and training loop run sequentially without replay, obey seed and budgets, checkpoint reproducibly, and emit planned evaluation inputs.

## 6. Uniform replay baseline

**Acceptance:** uniform memory sampling works under the fixed budget; training differs from naive only where required; sampling and budget invariants are tested.

## 7. TAFR components

**Acceptance:** authoritative TAFR definitions are documented and implemented modularly; scoring, selection, and replay respect the common budget; unit tests cover invariants.

## 8. Ablations

**Acceptance:** TAFR-F, TAFR-FU, and full TAFR mappings are configuration-driven and verified; at least one ablation is selected for the final study.

## 9. Full evaluation and visualizations

**Acceptance:** every method uses the frozen protocol; required metrics, confusion matrices, and continual-learning matrix are produced with traceable artifacts, runtime, and memory measurements.

## 10. Reproducibility audit, report, and demo

**Acceptance:** a clean one-command run is documented and verified; artifacts link to fingerprints and configuration; claims match executed results; the final report stays within four pages; the demo is reproducible.
