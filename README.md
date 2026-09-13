# TAFR-IDS — Threat-Aware Forgetting Replay for Continual Network Intrusion Detection

TAFR-IDS is an offline continual-learning intrusion-detection prototype built on the UNSW-NB15 dataset. It studies catastrophic forgetting as one multiclass neural classifier learns four sequential intrusion experiences. Every replay method uses the same fixed 2,000-example memory; TAFR changes how that memory is allocated across classes using forgetting, uncertainty, and rarity signals.

The repository includes the frozen seed-42 results and five inference-only model bundles. A clean clone can run the dashboard, CSV inference, and the safe Attack Simulation without the training dataset or the original training machine.

## Main result

Full TAFR was selected before final-test access using the declared validation ranking rule. Balanced accuracy was the primary selection metric.

| Method | Final average validation balanced accuracy |
| --- | ---: |
| Naive | 0.4517 |
| TAFR-F | 0.6485 |
| TAFR-FU | 0.6603 |
| Uniform Replay | 0.6616 |
| **Full TAFR** | **0.6707** |

The selected Full TAFR model achieved **0.5451 final-test balanced accuracy** in the one-time finalized evaluation. Validation/model-selection results and final-test results are intentionally distinguished; complete tracked summaries are in [validation_benchmark_seed42.csv](results/validation_benchmark_seed42.csv), [final_test_seed42.csv](results/final_test_seed42.csv), [the benchmark documentation](docs/benchmark_results.md), and [the final-evaluation record](docs/final_evaluation.md).

## Architecture

```text
UNSW-NB15
    ↓
split verification
    ↓
development / validation
    ↓
sequential experiences E1–E4
    ↓
frozen preprocessing
    ↓
MLP classifier
    ↓
continual-learning method
    ↓
evaluation
    ↓
dashboard
```

The classifier has 156 inputs, hidden widths 256 and 128, LayerNorm, ReLU, dropout 0.20, and one ten-class output head. Preprocessing is fitted once on E1 development data and then frozen. See [architecture](docs/architecture.md) and the [experimental protocol](docs/experimental_protocol.md).

## Continual-learning experiences

| Experience | Newly introduced attack classes |
| --- | --- |
| E1 | Generic, Shellcode, Worms |
| E2 | Exploits, Backdoor |
| E3 | Fuzzers, Analysis |
| E4 | DoS, Reconnaissance |

Normal traffic appears in every experience using different, mutually disjoint records. Each attack class is introduced exactly once. This ordering is a fixed, development-count-derived simulation of class-incremental learning, not a claim that the dataset is a chronological threat stream.

## Methods and TAFR

- **Naive:** sequential fine-tuning on only the current experience.
- **Uniform Replay:** fixed-memory replay with equal class priority.
- **TAFR-F:** replay allocation driven by normalized retained-memory forgetting.
- **TAFR-FU:** equal-weight normalized forgetting and uncertainty.
- **Full TAFR:** equal-weight normalized forgetting, uncertainty, and rarity.

For class `c`, the signals are:

- forgetting: the nonnegative drop from the best prior recall to current recall on retained training memory;
- uncertainty: mean `1 - max softmax probability` over eligible candidates;
- rarity: inverse square root of cumulative development support.

Each signal is independently min-max normalized. The resulting priority controls class quotas inside the same fixed 2,000-example replay capacity. TAFR does **not** change the neural-network architecture, optimizer, training budget, or evaluation schedule. Exact allocation, fallback, and deterministic selection rules are in [TAFR method](docs/tafr_method.md).

## Repository structure

```text
.
├── configs/       # frozen split, experience, and experiment manifests
├── dashboard/     # Streamlit application and display/data helpers
├── data/          # data-location policy; raw data is not tracked
├── docs/          # architecture, protocol, results, and usage guides
├── models/        # finalized inference-only assets and checksums
├── results/       # finalized public CSV/JSON result summaries
├── scripts/       # inspection, preparation, experiment, export, and verification CLIs
├── src/tafr_ids/  # installable data, training, evaluation, model, and inference code
└── tests/         # unit, integration, dashboard, and asset tests
```

The existing layout keeps public runtime assets separate from ignored training artifacts; no broad restructuring is required.

## Quick start

Python 3.11 or newer is supported.

```bash
git clone https://github.com/shazink/dl_hackathon.git
cd dl_hackathon
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,dashboard]'
python scripts/verify_inference_assets.py
python -m pytest -q
python -m streamlit run dashboard/app.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. `requirements-repro.txt` pins the complete environment used for the finalized run; use `python -m pip install -r requirements-repro.txt -e '.[dev,dashboard]'` when exact package-version reproduction is required.

## Dashboard

The Streamlit dashboard uses tracked files only and provides seven views:

- **Overview:** selected method and validation/final-test comparisons.
- **Continual Learning:** performance matrices, forgetting, forward transfer, recalls, and confusion matrices.
- **TAFR Replay Intelligence:** replay quotas and the forgetting, uncertainty, and rarity signals.
- **Method Comparison:** accuracy, Macro-F1, resource, and forgetting trade-offs.
- **Inference Demo:** schema-validated, in-memory prediction for an uploaded feature-only CSV.
- **Attack Simulation:** inference on deterministic development-only vectors with display-only response guidance.
- **Methodology and Limitations:** the scientific boundary and known caveats.

Uploaded inference data is limited to 5 MB and 500 rows, must follow the exact 42-feature schema, and is never persisted. See [dashboard usage](docs/dashboard.md).

## Inference assets and verification

The tracked [models](models/README.md) directory contains:

- five weights-only `model.safetensors` files;
- the frozen E1 `preprocessor.skops`;
- raw and transformed feature ordering;
- the immutable global class mapping;
- 80 development-only simulation vectors;
- per-method architecture, metric-reference, and provenance metadata;
- a SHA-256 manifest covering all 14 runtime JSON/binary assets.

These assets exist to make a clean clone usable without the original training machine. They contain no optimizer state, replay buffer, RNG state, training history, raw dataset, or logical-test rows.

```bash
python scripts/verify_inference_assets.py
```

The verifier checks every declared checksum, safely loads the frozen preprocessor, validates its 156-feature output contract and 44,210-row fit scope, loads all five models, checks the 75,146-parameter architecture and seed, and confirms finite ten-class probabilities that sum to one for all 80 bundled scenarios. See [inference asset provenance](docs/inference_assets.md).

## Reproducibility and dataset

The study uses seed 42, a fixed manifest-verified split, a seed-42 stratified 80/20 development/validation split of logical training, fixed E1–E4 experiences, and validation-only model selection. The selected method was frozen before the logical test was loaded. The official-test evaluation is consumed and finalized for protocol v1.0; it must not be used for later tuning or repeated in response to its result.

UNSW-NB15 is not committed. The verified local mirror has swapped physical filenames: logical training is the 175,341-row `UNSW_NB15_testing-set.csv`, and logical testing is the 82,332-row `UNSW_NB15_training-set.csv`. The project does not distribute a separate cleaned dataset. Preparation derives ignored local arrays from verified logical training only; the tracked simulation vectors are a small inference demonstration, not a dataset substitute.

Clean-clone dashboard/inference reproduction and dataset-dependent validation reproduction are documented separately in [reproducibility](docs/reproducibility.md). Dataset preparation details are in [data preparation](docs/data_preparation.md).

## Results policy

Frozen public results live under `results/`; detailed training checkpoints and generated research artifacts remain ignored under `artifacts/`. Without retraining, a clean clone can inspect every tracked result, render dashboard charts, verify all assets, run all five models, upload compatible feature CSVs, and run the offline simulator. Reproducing training or validation requires the hash-pinned dataset mirror and creates only local ignored artifacts.

Do not rerun the finalized protocol-v1.0 benchmark based on its known test outcome. Any new model, task order, feature choice, or hyperparameter study requires a newly versioned protocol that preserves test isolation.

## Limitations and safety scope

- The reported evidence uses a single seed: 42.
- The experience order is simulated rather than a true chronological threat stream.
- The required row-stratified split preserves duplicates and conflicting-label feature rows; identical feature vectors may cross development/validation boundaries and make validation optimistic.
- Logical split roles are structurally and hash verified, but byte identity with an official UNSW-hosted copy is not established.
- The forgetting signal uses retained training-memory recall as a proxy; discarded historical rows are not available to it.
- TAFR-IDS is an offline research prototype, not production network defense.

Attack Simulation is strictly offline. It sends no packets, scans no hosts, contacts no targets, changes no network or firewall configuration, and performs no enforcement. Its predictions and response text are advisory and displayed only in the current dashboard session.

## Documentation, acknowledgements, and references

- [UNSW-NB15 official project and dataset description](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [skops secure model persistence](https://skops.readthedocs.io/en/stable/persistence.html)
- [safetensors documentation](https://huggingface.co/docs/safetensors/index)
- [Streamlit documentation](https://docs.streamlit.io/)
- Project-specific continual-learning definitions: [experimental protocol](docs/experimental_protocol.md), [naive baseline](docs/naive_baseline.md), and [TAFR replay method](docs/tafr_method.md)

The source code is available under the [MIT License](LICENSE). The dataset is not covered by the repository's software license; consult the UNSW-NB15 source for its terms and citation guidance.
