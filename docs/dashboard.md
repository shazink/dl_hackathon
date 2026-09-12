# Local dashboard

Install the optional dashboard dependencies and launch from the repository root:

```bash
python -m pip install -e '.[dashboard]'
streamlit run dashboard/app.py
```

The dashboard reads tracked `results/dashboard_data.json` plus ignored local artifacts when inference or simulation is requested. It does not invent missing metrics. Overview, continual matrices, replay signals/quotas, method tradeoffs, local inference, safe attack simulation, and methodology/limitations are separate views.

Inference accepts a raw feature-only CSV up to 5 MB and 500 rows. Column order must exactly match the schema-only downloadable template. IDs, targets, extra/missing columns, empty values, malformed CSV, and non-finite numeric values are rejected. Uploaded bytes are processed in memory and never persisted. Official-test records are not bundled or loaded by the dashboard. If ignored checkpoints or the frozen preprocessor are unavailable, the inference view remains usable as documentation and reports the missing local artifact.

## Safe attack simulation

Prepare deterministic simulation inputs from the already transformed development partitions:

```bash
python scripts/prepare_simulation_scenarios.py \
  --prepared-dir artifacts/data/prepared \
  --output artifacts/simulation/scenarios.npz
```

The utility selects eight rows per class by original development-row index. It reads only `E1`–`E4` development arrays; it does not read validation or logical-test data, refit preprocessing, or modify experiment results. The generated archive is ignored by Git and contains finite float32 model inputs plus their development class, experience, and source index.

The Attack Simulation page passes selected vectors to a frozen local checkpoint and displays the resulting class probabilities and advisory response. It has no packet, socket, scanning, exploit, target-connection, firewall, or network-configuration capability. Recommendations are display-only and are not benchmark metrics. A missing/corrupt scenario archive or checkpoint produces an explicit unavailable state.
