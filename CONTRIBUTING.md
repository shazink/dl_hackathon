# Contributing

## Setup and checks

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,dashboard]'
python scripts/verify_inference_assets.py
python -m pytest -q
python -m pip check
git diff --check
```

Run the smallest relevant checks while developing, then run the complete sequence before submitting changes. Update tests and public documentation whenever behavior or paths change.

The repository does not currently configure a formatter or linter. Preserve the
existing style and run `git diff --check` before committing.

## Scientific invariants

- Never use logical test for fitting, tuning, early stopping, feature selection, task design, replay allocation, or validation.
- Never refit the frozen preprocessor or change finalized weights/results as a cleanup.
- Keep seed 42, the E1–E4 ordering, fixed 2,000-example replay capacity, shared model/training budget, and TAFR formulas unchanged unless a new scientific protocol is explicitly authorized.
- Keep raw data, prepared arrays, full checkpoints, logs, and secrets out of Git.
- Do not rerun protocol-v1.0 final-test evaluation; it is consumed and finalized.

Review `docs/experimental_protocol.md` before changing research code. New experiments must use fresh ignored output directories and must be reported honestly; never fabricate, hide, or silently replace results.
