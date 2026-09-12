# TAFR-IDS

TAFR-IDS (Threat-Aware Forgetting Replay for Continual Network Intrusion Detection) compares Naive sequential fine-tuning, Uniform Replay, TAFR-F, TAFR-FU, and full TAFR on UNSW-NB15 under one frozen four-experience protocol.

The completed seed-42 study selected full TAFR before test access using validation balanced accuracy. Full TAFR reached 0.670742 validation balanced accuracy and 0.545146 on the one-time official test evaluation. Detailed generated artifacts and checkpoints stay local; concise honest results are tracked in `results/`.

## Protocol

The verified mirror has swapped physical filenames: logical training is `UNSW_NB15_testing-set.csv` (175,341 rows) and logical testing is `UNSW_NB15_training-set.csv` (82,332 rows). Place the downloaded UNSW-NB15 files together in a local directory. The manifest pins logical roles, counts, schemas, and SHA-256 values; official byte identity remains unverified.

Validation is a seed-42 stratified 80/20 row split of logical training. The 156-column preprocessor is fitted once on E1 development and frozen. Replay capacity is 2,000 unique samples; E2–E4 batches mix 192 current and 64 replay rows, with proportional replay for the last partial chunk. All methods share initialization, model, optimizer, budget, validation, checkpoint, and device policies.

## Setup and data preparation

Python 3.11 or newer is supported; `requirements-repro.txt` pins the verified Python 3.14.7 environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt -e '.[dev,dashboard]'

python scripts/inspect_unsw_nb15.py \
  --data-dir /path/to/archive \
  --split-manifest configs/unsw_nb15_splits.toml \
  --output artifacts/data/unsw_nb15_profile.json

python scripts/prepare_unsw_nb15.py \
  --data-dir /path/to/archive \
  --split-manifest configs/unsw_nb15_splits.toml \
  --experience-config configs/experiences.toml \
  --output-dir artifacts/data/prepared
```

Preparation writes only ignored local artifacts and never creates logical-test arrays. See [data preparation](docs/data_preparation.md) and the [experimental protocol](docs/experimental_protocol.md).

## Training and benchmark

Run one method with its matching config:

```bash
python scripts/run_experiment.py --method tafr \
  --config configs/experiments/tafr.toml \
  --prepared-dir artifacts/data/prepared \
  --output-dir artifacts/experiments/tafr_seed42
```

Run the controlled five-method benchmark, audit, frozen selection, and final one-time test evaluation only for a fresh protocol whose official test has not been consumed:

```bash
python scripts/run_benchmark.py \
  --prepared-dir artifacts/data/prepared \
  --data-dir /path/to/archive
```

Do not rerun the finalized seed-42 protocol based on its test outcome. Any later model or hyperparameter study requires a new versioned protocol and must not use these test results for tuning.

## Dashboard and tests

```bash
streamlit run dashboard/app.py
python -m pytest -q
python -m pip check
```

The dashboard distinguishes validation from final test, explores continual matrices and replay allocation, compares resources and forgetting, supports schema-validated local CSV inference, and provides a development-vector-only Attack Simulation page when ignored local artifacts are available. Prepare its safe offline scenarios with:

```bash
python scripts/prepare_simulation_scenarios.py \
  --prepared-dir artifacts/data/prepared \
  --output artifacts/simulation/scenarios.npz
```

See [dashboard documentation](docs/dashboard.md).

## Results and documentation

- [Validation benchmark](docs/benchmark_results.md)
- [Final evaluation](docs/final_evaluation.md)
- [TAFR method](docs/tafr_method.md)
- [Naive baseline](docs/naive_baseline.md)
- [Architecture](docs/architecture.md)
- `results/validation_benchmark_seed42.csv`
- `results/final_test_seed42.csv`
- `results/dashboard_data.json`

## Limitations

This is a single-seed result. The required row-stratified validation preserves duplicate and conflicting-label feature rows, so identical features can cross development and validation and make validation optimistic. Replay forgetting is a retained-memory proxy rather than an oracle over discarded rows. The official test split is consumed and finalized for protocol v1.0. Dataset split roles are structurally verified, but equality with official UNSW-hosted bytes is not established.
