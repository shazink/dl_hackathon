# Local dashboard

Install the optional dashboard dependencies and launch from the repository root:

```bash
python -m pip install -e '.[dashboard]'
streamlit run dashboard/app.py
```

The dashboard reads tracked `results/dashboard_data.json` and checksum-verified inference assets under `models/`. It does not invent missing metrics. Overview, continual matrices, replay signals/quotas, method tradeoffs, local inference, safe attack simulation, and methodology/limitations are separate views.

The seven sidebar views are Overview, Continual Learning, TAFR Replay Intelligence, Method Comparison, Inference Demo, Attack Simulation, and Methodology and Limitations. All paths are resolved from `dashboard/app.py`, so launch is independent of the caller's current directory.

Inference accepts a raw feature-only CSV up to 5 MB and 500 rows. Column order must exactly match the schema-only downloadable template. IDs, targets, extra/missing columns, empty values, malformed CSV, and non-finite numeric values are rejected. Uploaded bytes are processed in memory and never persisted. Official-test records are not bundled or loaded by the dashboard. Models are loaded from weights-only safetensors files and the preprocessor is loaded from skops with an exact project type allowlist. Every runtime JSON/binary file is SHA-256 verified before model loading.

## Safe attack simulation

The repository includes the previously prepared deterministic simulation inputs at `models/shared/simulation_scenarios.npz`. The export is byte-identical to the local archive produced by:

```bash
python scripts/prepare_simulation_scenarios.py \
  --prepared-dir artifacts/data/prepared \
  --output artifacts/simulation/scenarios.npz
```

The utility selects eight rows per class by original development-row index. It reads only `E1`–`E4` development arrays; it does not read validation or logical-test data, refit preprocessing, or modify experiment results. The tracked archive contains finite float32 model inputs plus their development class, experience, and source index.

The Attack Simulation page passes selected vectors to a frozen tracked inference bundle and displays the resulting class probabilities and advisory response. It has no packet, socket, scanning, exploit, target-connection, firewall, or network-configuration capability. Recommendations are display-only and are not benchmark metrics. A missing/corrupt scenario archive or bundle produces an explicit unavailable state.

Run `python scripts/verify_inference_assets.py` to validate all 14 declared checksums and smoke-test finite probabilities from every finalized method.

See [inference asset provenance](inference_assets.md) and [clean-clone reproduction](reproducibility.md).
