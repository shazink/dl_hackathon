import json

import pytest

from tafr_ids.evaluation.reporting import write_json_atomic


def test_strict_json_rejects_nan_and_leaves_no_destination(tmp_path):
    destination = tmp_path / "result.json"
    with pytest.raises(ValueError):
        write_json_atomic(destination, {"metric": float("nan")})
    assert not destination.exists()


def test_strict_json_round_trip(tmp_path):
    destination = tmp_path / "result.json"
    write_json_atomic(destination, {"metric": 0.5})
    assert json.loads(destination.read_text()) == {"metric": 0.5}
