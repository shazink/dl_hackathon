# Inference-only model bundles

These tracked assets make the finalized seed-42 TAFR-IDS models usable after a normal clone. They are exports of the completed protocol-v1.0 experiment; exporting them did not train, tune, refit, or evaluate a model.

Each method directory contains only final model tensors in `model.safetensors` and provenance/configuration in `metadata.json`. Full training checkpoints remain ignored because they also contain optimizer state, replay-buffer samples, RNG state, and training history.

`shared/` contains:

- `preprocessor.skops`: the frozen transformer fitted once on E1 development;
- `feature_schema.json`: ordered 42-column raw input schema and ordered 156-column transformed schema;
- `class_mapping.json`: immutable ten-class output mapping;
- `simulation_scenarios.npz`: 80 already-transformed development vectors, eight per class, for the safe offline dashboard simulation;
- `SHA256SUMS`: hashes for every runtime JSON/binary asset.

The scenario archive contains no logical-test records and is not raw traffic. It was copied byte-for-byte from the previously verified development-only artifact described in `docs/dashboard.md`. Model metadata links to the tracked validation and final-test summaries; those references document existing results and are not newly computed metrics.

Verify all files and run a finite-probability smoke test for all five models:

```bash
python scripts/verify_inference_assets.py
```

Launch the dashboard after installing the dashboard extra:

```bash
python -m pip install -e '.[dashboard]'
streamlit run dashboard/app.py
```

Raw CSV inference expects exactly the columns and order in `shared/feature_schema.json`. Do not edit a bundle without regenerating `shared/SHA256SUMS` and retaining accurate source provenance.
