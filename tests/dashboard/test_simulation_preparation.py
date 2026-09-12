from pathlib import Path

import numpy as np

from dashboard.data import load_simulation_scenarios
from scripts.prepare_simulation_scenarios import prepare


def test_prepare_uses_development_arrays_only(tmp_path: Path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for experience in range(1, 5):
        labels = np.asarray([0, experience * 2 - 1, experience * 2], dtype=np.int64)
        if experience == 1:
            labels = np.asarray([0, 6, 8, 9], dtype=np.int64)
        elif experience == 2:
            labels = np.asarray([0, 2, 4], dtype=np.int64)
        elif experience == 3:
            labels = np.asarray([0, 1, 5], dtype=np.int64)
        else:
            labels = np.asarray([0, 3, 7], dtype=np.int64)
        features = np.repeat(np.arange(len(labels), dtype=np.float32)[:, None], 156, axis=1)
        np.save(prepared / f"E{experience}_development_X.npy", features)
        np.save(prepared / f"E{experience}_development_y.npy", labels)
        np.save(prepared / f"E{experience}_development_indices.npy", np.arange(len(labels), dtype=np.int64) + experience * 100)
    output = tmp_path / "simulation" / "scenarios.npz"
    assert prepare(prepared, output, samples_per_class=1) == 10
    scenarios = load_simulation_scenarios(output)
    assert scenarios is not None
    assert set(scenarios["class_names"].tolist()) == {"Normal", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms"}
