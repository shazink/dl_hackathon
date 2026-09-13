# Reproducibility

There are two distinct workflows. Dashboard and inference reproduction use only tracked files. Dataset-dependent experiment reproduction is a separate research workflow and is not required to use the application.

## A. Clean-clone dashboard and inference

Python 3.11 or newer is supported. From a new directory:

```bash
git clone https://github.com/shazink/dl_hackathon.git
cd dl_hackathon
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,dashboard]'
python scripts/verify_inference_assets.py
python -m pytest -q
python -m pip check
python -m streamlit run dashboard/app.py
```

The final command starts a local server. The application should expose Overview, Continual Learning, TAFR Replay Intelligence, Method Comparison, Inference Demo, Attack Simulation, and Methodology and Limitations.

No UNSW-NB15 download is required for this workflow. `scripts/verify_inference_assets.py` verifies all 14 declared runtime checksums, the frozen preprocessor contract, every model architecture, and finite normalized probabilities from all five methods on the bundled development-only scenarios.

For raw CSV inference, use the schema-only template downloadable from Inference Demo. It contains the required ordered 42-feature header but no data. Uploaded rows are transformed in memory with the tracked frozen preprocessor.

### Exact finalized environment

`requirements-repro.txt` pins the Linux/Python 3.14.7 environment used for the finalized experiment and dashboard verification:

```bash
python -m pip install -r requirements-repro.txt -e '.[dev,dashboard]'
```

The project metadata remains less restrictive (`Python >=3.11` plus compatible dependency lower bounds) for ordinary use. Platform-specific PyTorch installation requirements may differ; follow the official PyTorch installation guidance when the exact Linux lock is unsuitable.

## B. Dataset-dependent validation experiments

This workflow is for research reproduction. It requires the UNSW-NB15 files matching `configs/unsw_nb15_splits.toml` and writes only ignored local artifacts.

1. Obtain UNSW-NB15 from the [official project page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) or the mirror identified in the split manifest. The project does not commit or redistribute raw data.
2. Keep all eight expected files together in an external directory or an ignored local data directory.
3. Verify the mirror before preparing any arrays:

```bash
python scripts/inspect_unsw_nb15.py \
  --data-dir /path/to/archive \
  --split-manifest configs/unsw_nb15_splits.toml \
  --output artifacts/data/unsw_nb15_profile.json
```

4. Create the fixed development/validation assignments and frozen E1 preprocessor:

```bash
python scripts/prepare_unsw_nb15.py \
  --data-dir /path/to/archive \
  --split-manifest configs/unsw_nb15_splits.toml \
  --experience-config configs/experiences.toml \
  --output-dir artifacts/data/prepared
```

5. Reproduce a validation experiment in a new output directory. Select the method and its matching config; for example:

```bash
python scripts/run_experiment.py \
  --method tafr \
  --config configs/experiments/tafr.toml \
  --prepared-dir artifacts/data/prepared \
  --output-dir artifacts/experiments/tafr_seed42
```

Valid method/config stems are `naive`, `uniform`, `tafr_f`, `tafr_fu`, and `tafr`. The runner validates data/config fingerprints, uses the frozen protocol, and does not load logical test.

### Final-test policy

Protocol v1.0's logical test has already been consumed and finalized. `scripts/run_benchmark.py` records the historical audit-gated workflow, including the one-time test step, but must not be rerun or used for tuning in response to the known outcome. Reproducing validation is allowed; a new final-test study requires a new versioned protocol whose development decisions do not use v1.0 test results.

## What is and is not reproducible from tracked files

| Capability | Clean clone only | Dataset required | Retraining required |
| --- | --- | --- | --- |
| Inspect finalized result summaries | Yes | No | No |
| Render all tracked dashboard charts | Yes | No | No |
| Verify/load all five finalized models | Yes | No | No |
| Run raw feature-only CSV inference | Yes | No; input rows are supplied by the user | No |
| Run Attack Simulation | Yes | No | No |
| Recreate prepared development/validation arrays | No | Yes | No |
| Reproduce a validation training run | No | Yes | Yes |
| Repeat protocol-v1.0 final-test evaluation | Prohibited by policy | — | — |

See [data preparation](data_preparation.md), [experimental protocol](experimental_protocol.md), and [inference assets](inference_assets.md).
