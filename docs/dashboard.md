# Local dashboard

Install the optional dashboard dependencies and launch from the repository root:

```bash
python -m pip install -e '.[dashboard]'
streamlit run dashboard/app.py
```

The dashboard reads only tracked `results/dashboard_data.json` plus local ignored checkpoints when inference is requested. It does not invent missing metrics. Overview, continual matrices, replay signals/quotas, method tradeoffs, local inference, and methodology/limitations are separate views.

Inference accepts a raw feature-only CSV up to 5 MB and 500 rows. Column order must exactly match the schema-only downloadable template. IDs, targets, extra/missing columns, empty values, malformed CSV, and non-finite numeric values are rejected. Uploaded bytes are processed in memory and never persisted. Official-test records are not bundled or loaded by the dashboard. If ignored checkpoints or the frozen preprocessor are unavailable, the inference view remains usable as documentation and reports the missing local artifact.
