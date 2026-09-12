# TAFR-IDS

TAFR-IDS (Threat-Aware Forgetting Replay for Continual Network Intrusion Detection) is a research project for studying replay strategies on the UNSW-NB15 dataset under a class-incremental continual-learning protocol.

**Status:** Phases 1–4 are complete: verified logical splits, locked experiences, and E1-only preprocessing with prepared development/validation artifacts. No model has been trained.

## Planned study

The core comparison will cover naive sequential fine-tuning, uniform replay, and TAFR. Planned TAFR ablations are TAFR-F, TAFR-FU, and full TAFR, subject to the authoritative project specification. At least one ablation will be included in the final study.

Scientific integrity takes priority: the official test split is evaluation-only, with the limited Phase 2 integrity-audit exception described in the protocol; validation comes only from official training data; preprocessing is fit only on permitted training data; comparisons use the same protocol and applicable budgets; and unexecuted work or results must never be presented as completed.

## Layout

- `src/tafr_ids/`: importable package and future implementation areas
- `configs/`: future reproducible experiment configurations
- `data/`: dataset placement guidance (dataset contents are ignored)
- `docs/`: architecture, protocol, and implementation plan
- `.ai/`: persistent project context, decisions, state, data card, and experiment log
- `tests/`: automated tests

## Local setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
python scripts/inspect_unsw_nb15.py --data-dir ~/Downloads/archive \
  --split-manifest configs/unsw_nb15_splits.toml \
  --output artifacts/data/unsw_nb15_profile.json
```

Inspection uses only the Python standard library and writes deterministic JSON to the specified artifact (or stdout when `--output` is omitted). Without a manifest, only structural discovery runs and profiling is blocked. Exit code 0 means inspection completed; 2 means an integrity, provenance, or manifest check failed. No dataset files are written. See [the data card](.ai/data_card.md) for findings and the unverified official-byte-identity limitation.

See [the experimental protocol](docs/experimental_protocol.md) and [implementation plan](docs/implementation_plan.md).

Prepare data using the locked protocol:

```bash
python scripts/prepare_unsw_nb15.py \
  --data-dir ~/Downloads/archive \
  --split-manifest configs/unsw_nb15_splits.toml \
  --experience-config configs/experiences.toml \
  --output-dir artifacts/data/prepared
```

Use a new output directory for each run. See [data preparation](docs/data_preparation.md) for artifacts, E1 fit boundaries, canonical targets, and the duplicate-row validation limitation. `requirements-repro.txt` records the verified Python 3.14.7 environment; install it alongside `-e '.[dev]'` for exact dependency versions.
