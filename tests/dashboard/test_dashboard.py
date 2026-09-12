import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib
import torch

from dashboard.data import MAX_UPLOAD_BYTES, load_dashboard_data, predict_upload, schema_template, validate_upload
from tafr_ids.data.inspection import CATEGORICAL, FEATURES
from tafr_ids.data.preprocessing import FrozenPreprocessor
from tafr_ids.models.mlp import TabularMLP


def valid_frame():
    return pd.DataFrame({name: ["tcp" if name == "proto" else "-" if name == "service" else "FIN" if name == "state" else "1"] for name in FEATURES})


def csv_bytes(frame):
    return frame.to_csv(index=False).encode()


def test_missing_and_corrupt_dashboard_results(tmp_path):
    assert load_dashboard_data(tmp_path / "missing.json") is None
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text('{"schema_version": 1, "methods": {}, "bad": NaN}')
    with pytest.raises(ValueError, match="corrupt"):
        load_dashboard_data(corrupt)
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"schema_version": 1, "methods": {}}))
    assert load_dashboard_data(valid)["methods"] == {}


def test_schema_template_and_valid_upload():
    assert schema_template().decode().strip().split(",") == FEATURES
    result = validate_upload(csv_bytes(valid_frame()))
    assert list(result.columns) == FEATURES and len(result) == 1
    assert all(np.issubdtype(result[name].dtype, np.number) for name in FEATURES if name not in CATEGORICAL)


@pytest.mark.parametrize("forbidden", ["id", "attack_cat", "label"])
def test_upload_rejects_ids_and_targets(forbidden):
    frame = valid_frame()
    frame[forbidden] = "0"
    with pytest.raises(ValueError, match="prohibited"):
        validate_upload(csv_bytes(frame))


def test_upload_rejects_unexpected_missing_nonfinite_malformed_and_limits():
    frame = valid_frame()
    frame["unexpected"] = "1"
    with pytest.raises(ValueError, match="schema"):
        validate_upload(csv_bytes(frame))
    frame = valid_frame().drop(columns=[FEATURES[0]])
    with pytest.raises(ValueError, match="schema"):
        validate_upload(csv_bytes(frame))
    frame = valid_frame()
    numeric = next(name for name in FEATURES if name not in CATEGORICAL)
    frame[numeric] = "inf"
    with pytest.raises(ValueError, match="Non-finite"):
        validate_upload(csv_bytes(frame))
    with pytest.raises(ValueError, match="Malformed"):
        validate_upload(b'"unterminated')
    with pytest.raises(ValueError, match="5 MB"):
        validate_upload(b"x" * (MAX_UPLOAD_BYTES + 1))
    oversized = pd.concat([valid_frame()] * 501, ignore_index=True)
    with pytest.raises(ValueError, match="500"):
        validate_upload(csv_bytes(oversized))


def test_local_prediction_probabilities_are_finite(tmp_path):
    frame = validate_upload(csv_bytes(valid_frame()))
    processor = FrozenPreprocessor().fit_e1(frame, partition="development", experience=1)
    transformed = processor.transform(frame)
    model = TabularMLP(transformed.shape[1], 10, [8, 4], 0.0)
    processor_path = tmp_path / "processor.joblib"
    checkpoint_path = tmp_path / "checkpoint.pt"
    joblib.dump(processor, processor_path)
    torch.save({"config": {"model": {"hidden_dims": [8, 4], "dropout": 0.0}}, "model": model.state_dict()}, checkpoint_path)
    prediction = predict_upload(frame, checkpoint_path, processor_path)
    probability_columns = [name for name in prediction if name.startswith("p_")]
    assert np.isfinite(prediction[["confidence", *probability_columns]].to_numpy()).all()
    assert prediction[probability_columns].sum(axis=1).iloc[0] == pytest.approx(1.0)


def test_streamlit_app_starts_in_test_harness():
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
    result = AppTest.from_file(str(app)).run(timeout=20)
    assert not result.exception
