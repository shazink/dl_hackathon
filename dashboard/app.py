"""TAFR-IDS local Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from dashboard.components import comparison_chart, matrix_heatmap, scatter_tradeoff
from dashboard.data import (
    load_dashboard_data,
    load_simulation_scenarios,
    predict_features,
    predict_upload,
    schema_template,
    validate_upload,
)

DATA_PATH = REPOSITORY / "results" / "dashboard_data.json"
MODELS_PATH = REPOSITORY / "models"
SCENARIO_PATH = MODELS_PATH / "shared" / "simulation_scenarios.npz"

st.set_page_config(page_title="TAFR-IDS", page_icon="🛡️", layout="wide")
st.title("TAFR-IDS")
st.caption("Threat-Aware Forgetting Replay · fixed seed-42 UNSW-NB15 study")

try:
    data = load_dashboard_data(DATA_PATH)
except ValueError as error:
    st.error(str(error))
    st.stop()
if data is None:
    st.warning("Tracked benchmark summaries are unavailable. Run scripts/run_benchmark.py first.")
    st.stop()

view = st.sidebar.radio(
    "View",
    ("Overview", "Continual Learning", "TAFR Replay Intelligence", "Method Comparison", "Inference Demo", "Attack Simulation", "Methodology and Limitations"),
)
comparison = data["comparison"]
methods = data["methods"]
selected = data["selection"]["method"]

if view == "Overview":
    st.header("Validated continual intrusion detection")
    st.write("Five independently initialized methods follow one frozen four-experience protocol. Validation selected the deployed method before the official test split was loaded.")
    row = next(item for item in comparison if item["method"] == selected)
    columns = st.columns(4)
    columns[0].metric("Selected method", selected)
    columns[1].metric("Validation balanced accuracy", f"{row['validation_balanced_accuracy']:.3f}")
    columns[2].metric("Final-test balanced accuracy", f"{row['test_balanced_accuracy']:.3f}")
    columns[3].metric("Final-test Macro-F1", f"{row['test_macro_f1']:.3f}")
    left, right = st.columns(2)
    left.plotly_chart(comparison_chart(comparison, "validation_balanced_accuracy", "Validation comparison"), width="stretch")
    right.plotly_chart(comparison_chart(comparison, "test_balanced_accuracy", "One-time final-test comparison"), width="stretch")
    st.info(data["selection"]["explanation"])

elif view == "Continual Learning":
    st.header("Continual learning dynamics")
    method = st.selectbox("Method", list(methods), index=list(methods).index(selected))
    record = methods[method]["validation"]
    st.write("E1: Generic, Shellcode, Worms · E2: Exploits, Backdoor · E3: Fuzzers, Analysis · E4: DoS, Reconnaissance. Normal recurs with disjoint rows.")
    tabs = st.tabs(["Accuracy", "Balanced accuracy", "Macro-F1"])
    for tab, metric in zip(tabs, ("accuracy", "balanced_accuracy", "macro_f1"), strict=True):
        with tab:
            st.plotly_chart(matrix_heatmap(record["matrices"][metric], record["matrix_axes"]["rows"], record["matrix_axes"]["columns"], metric.replace("_", " ").title()), width="stretch")
    metrics = record["continual_metrics"]
    c1, c2 = st.columns(2)
    c1.metric("Balanced-accuracy forgetting", f"{metrics['average_forgetting']:.3f}")
    c2.metric("Forward transfer", f"{metrics['forward_transfer']:.3f}")
    final_eval = record["history"]["evaluations"]["after_E4"]
    recalls = []
    for experience, evaluation in final_eval.items():
        for class_name, values in evaluation["per_class"].items():
            if values["recall"] is not None:
                recalls.append({"experience": experience, "class": class_name, "recall": values["recall"]})
    st.dataframe(pd.DataFrame(recalls), width="stretch", hide_index=True)
    st.subheader("Final official-test confusion matrix")
    test = methods[method]["test"]
    st.plotly_chart(matrix_heatmap(test["confusion_matrix"], test["class_order"], test["class_order"], "Final confusion matrix"), width="stretch")

elif view == "TAFR Replay Intelligence":
    st.header("TAFR replay intelligence")
    st.write("F = max(0, best retained-memory recall − current retained-memory recall); U = mean(1 − maximum softmax probability); R = 1/√(cumulative development count). Each signal is independently min-max normalized.")
    st.write("TAFR-F uses F; TAFR-FU averages F and U; full TAFR averages F, U, and R. Uniform Replay assigns priority 1. Within-class selection stays uniformly deterministic for every method.")
    method = st.selectbox("Replay method", [name for name in methods if name != "naive"])
    updates = methods[method]["validation"]["history"]["replay_updates"]
    allocation_rows = []
    signal_rows = []
    for experience, update in updates.items():
        for class_id, quota in update["actual_quotas"].items():
            allocation_rows.append({"experience": experience, "class_id": class_id, "quota": quota})
        for signal, values in update["raw_signals"].items():
            for class_id, value in values.items():
                signal_rows.append({"experience": experience, "class_id": class_id, "signal": signal, "raw": value, "normalized": update["normalized_signals"][signal][class_id]})
    st.subheader("Buffer allocation")
    st.dataframe(pd.DataFrame(allocation_rows), width="stretch", hide_index=True)
    st.subheader("Signals")
    st.dataframe(pd.DataFrame(signal_rows), width="stretch", hide_index=True)
    occupancy = [{"experience": key, "occupancy": value["buffer_size"], "capacity": value["buffer_capacity"]} for key, value in updates.items()]
    st.dataframe(pd.DataFrame(occupancy), width="stretch", hide_index=True)

elif view == "Method Comparison":
    st.header("Method comparison")
    st.dataframe(pd.DataFrame(comparison), width="stretch", hide_index=True)
    left, right = st.columns(2)
    left.plotly_chart(scatter_tradeoff(comparison), width="stretch")
    right.plotly_chart(comparison_chart(comparison, "validation_macro_f1", "Validation Macro-F1"), width="stretch")
    st.plotly_chart(comparison_chart(comparison, "runtime_seconds", "Training runtime (seconds)"), width="stretch")
    tafr = next(item for item in comparison if item["method"] == "tafr")
    naive = next(item for item in comparison if item["method"] == "naive")
    uniform = next(item for item in comparison if item["method"] == "uniform")
    st.info(f"On validation balanced accuracy, full TAFR {'beat' if tafr['validation_balanced_accuracy'] > naive['validation_balanced_accuracy'] else 'did not beat'} Naive and {'beat' if tafr['validation_balanced_accuracy'] > uniform['validation_balanced_accuracy'] else 'did not beat'} Uniform Replay.")

elif view == "Inference Demo":
    st.header("Local inference demo")
    method = st.selectbox("Checkpoint", list(methods), index=list(methods).index(selected))
    st.download_button("Download schema-only CSV template", schema_template(), "tafr_ids_schema.csv", "text/csv")
    upload = st.file_uploader("Raw UNSW-NB15 feature CSV (maximum 5 MB / 500 rows)", type="csv")
    bundle = MODELS_PATH / method
    preprocessor = MODELS_PATH / "shared" / "preprocessor.skops"
    if not (bundle / "model.safetensors").is_file() or not preprocessor.is_file():
        st.info("The selected tracked inference bundle is unavailable.")
    elif upload is not None:
        try:
            frame = validate_upload(upload.getvalue())
            predictions = predict_upload(frame, bundle, preprocessor)
            st.dataframe(predictions, width="stretch", hide_index=True)
        except (OSError, KeyError, ValueError) as error:
            st.error(str(error))
    st.caption("Uploads are processed in memory and are never persisted. Official test rows are neither bundled nor automatically loaded.")

elif view == "Attack Simulation":
    st.header("Attack Simulation")
    st.warning("Safe offline simulation only: no packets are sent, no hosts are scanned, no targets are contacted, and no network or firewall settings are changed.")
    st.write("This page runs frozen local models on a small, deterministic set of already-preprocessed development vectors. Outputs are live scenario inferences, not benchmark metrics or real-world attack results.")
    checkpoint_method = st.selectbox("Simulation checkpoint", list(methods), index=list(methods).index(selected))
    try:
        scenarios = load_simulation_scenarios(SCENARIO_PATH)
    except ValueError as error:
        st.error(str(error))
        scenarios = None
    bundle = MODELS_PATH / checkpoint_method
    if scenarios is None:
        st.info("The tracked development-only simulation scenarios are unavailable.")
    elif not (bundle / "model.safetensors").is_file():
        st.info("The selected tracked inference bundle is unavailable.")
    else:
        attack_names = sorted(set(scenarios["class_names"].tolist()) - {"Normal"})
        attack_name = st.selectbox("Development scenario class", attack_names)
        available = int((scenarios["class_names"] == attack_name).sum())
        flow_count = st.slider("Flow vectors", 1, available, min(4, available))
        response_threshold = st.slider("High-priority confidence threshold", 0.50, 0.99, 0.80, 0.01)
        if st.button("Run safe simulation", type="primary"):
            positions = np.flatnonzero(scenarios["class_names"] == attack_name)[:flow_count]
            try:
                predictions = predict_features(scenarios["X"][positions], bundle)
            except ValueError as error:
                st.error(str(error))
            else:
                result = predictions[["predicted_class", "confidence"]].copy()
                result.insert(0, "scenario_class", attack_name)
                result.insert(1, "source_experience", scenarios["experiences"][positions])
                result["simulated_response"] = [
                    "Analyst review: model did not flag the scenario" if name == "Normal"
                    else "High-priority alert: validate and review containment playbook" if confidence >= response_threshold
                    else "Triage alert: corroborate with additional telemetry"
                    for name, confidence in zip(result["predicted_class"], result["confidence"], strict=True)
                ]
                flagged = int((result["predicted_class"] != "Normal").sum())
                high_priority = int(((result["predicted_class"] != "Normal") & (result["confidence"] >= response_threshold)).sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Flow vectors assessed", len(result))
                c2.metric("Simulated alerts", flagged)
                c3.metric("High-priority reviews", high_priority)
                st.dataframe(result, width="stretch", hide_index=True)
                st.caption("Responses are recommendations displayed in this browser session only. No enforcement action is available or performed.")

else:
    st.header("Methodology and limitations")
    st.markdown("""
- The mirror's physical filenames are swapped: logical training is the 175,341-row `testing-set` file; logical testing is the 82,332-row `training-set` file. Hashes and schemas are pinned, but official byte identity remains unverified.
- Validation is an 80/20 seed-42 row-stratified split of logical training. The preprocessor was fitted once on E1 development and frozen.
- Replay capacity is 2,000 unique samples; E2–E4 batches mix 192 current with 64 replay samples, with proportional handling of the final current chunk.
- Forgetting allocation uses retained-memory proxy recall, not an oracle over discarded samples.
- Duplicate and conflicting-label feature rows are preserved; identical feature vectors may cross development and validation partitions and inflate validation performance.
- This is a single-seed study. The official test has been consumed exactly once for this finalized protocol; it cannot be used for subsequent tuning.
""")
    st.code("python scripts/run_benchmark.py --prepared-dir artifacts/data/prepared --data-dir /path/to/archive\nstreamlit run dashboard/app.py\npython -m pytest -q")
