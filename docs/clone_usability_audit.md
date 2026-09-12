# Fresh-clone usability audit

This audit records the repository state at commit `f59b6ff` before portable inference assets were added. It was performed without training, preprocessing fits, or logical-test access.

## What a clean checkout can do

| Capability | Pre-change result | Evidence |
| --- | --- | --- |
| Install the package | Yes, when build/runtime dependencies are available | An editable install from a clean local clone succeeded with the provisioned dependency environment. A fully isolated install could not contact PyPI in the sandbox; that was an environment network restriction, not a packaging error. |
| Import `tafr_ids` | Yes | Import succeeded from the clean checkout. |
| Start the Streamlit application | Code/test harness: yes; sandbox socket check: blocked | The tracked AppTest regression already exercised all views. A clean-checkout server launch reached socket creation and was denied by the sandbox (`PermissionError: [Errno 1] Operation not permitted`). |
| Load benchmark charts | Yes | `results/dashboard_data.json` is tracked, strict JSON, and contains all five methods. |
| Run model inference | No | The dashboard referenced ignored training checkpoints and an ignored Joblib preprocessor. None exists in a clean checkout. |
| Run Attack Simulation | No | The dashboard referenced the same ignored checkpoints plus an ignored scenario archive. None exists in a clean checkout. |

## Exact missing runtime assets

The pre-change dashboard required these paths, all excluded by the repository-wide `artifacts/` ignore rule:

- `artifacts/data/prepared/preprocessor.joblib`
- `artifacts/simulation/scenarios.npz`
- `artifacts/experiments/benchmark_seed42/naive/checkpoints/latest.pt`
- `artifacts/experiments/benchmark_seed42/uniform/checkpoints/latest.pt`
- `artifacts/experiments/benchmark_seed42/tafr_f/checkpoints/latest.pt`
- `artifacts/experiments/benchmark_seed42/tafr_fu/checkpoints/latest.pt`
- `artifacts/experiments/benchmark_seed42/tafr/checkpoints/latest.pt`

The five `latest.pt` files are complete training checkpoints, not suitable public runtime bundles: replay variants include optimizer state, RNG state, metric history, and replay-buffer data and are roughly 3.7 MB each. The fitted Joblib object also relies on pickle semantics. The clean clone therefore had no safe, minimal model or preprocessing asset from which inference could run.

## Path portability

Tracked dashboard runtime paths were repository-relative and resolved from `dashboard/app.py`; no tracked dashboard or loader path was tied to one machine. CLI examples intentionally accept `/path/to/archive`. Machine-specific paths occur only in ignored/local research records (for example the original dataset location and experiment environment metadata) and are not a viable runtime dependency.

## Required remediation

Publish weights-only model files, architecture and provenance metadata, the frozen E1 preprocessor in a non-pickle transport format, the raw/transformed feature schemas and class mapping, and development-only simulation vectors. Point inference and simulation at those tracked files, verify their checksums and schemas before use, and repeat installation/application/inference checks from a clean clone.
