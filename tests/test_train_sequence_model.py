"""Smoke : train_sequence_model tourne sur un mini-dataset synthétique."""
import importlib.util, sys, json, tempfile
from pathlib import Path
import numpy as np
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC / "02_data_science"))
_spec = importlib.util.spec_from_file_location(
    "train_sequence_model", _SRC / "02_data_science" / "train_sequence_model.py")
trm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trm)


def _mini(n=40, seed=0):
    rng = np.random.RandomState(seed)
    puuids = np.array([f"p{i // 4}" for i in range(n)], dtype=object)   # 4 games/joueur
    ranks = np.array(rng.choice(["diamond", "challenger"], n), dtype=object)
    y = (ranks == "challenger").astype(int)
    # signal : les challenger ont totalGold plus haut à la frame 10 (feature 2)
    seqs = rng.randn(n, 40, 20).astype(np.float32)
    seqs[y == 1, 10, 2] += 5.0
    return {
        "sequences": seqs, "mask": np.ones((n, 40), dtype=bool),
        "rank": ranks, "puuid": puuids,
        "match_id": np.array([f"m{i}" for i in range(n)], dtype=object),
        "champion": np.array(["Zeri"] * n, dtype=object),
    }


def test_train_returns_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(trm, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(trm.sd, "DATASET", tmp_path / "x.npz")
    np.savez(tmp_path / "x.npz", **_mini())
    metrics = trm.run(epochs=5, batch=8, seed=42, device_force="cpu")
    assert "tasks" in metrics and "high_elo" in metrics["tasks"]
    he = metrics["tasks"]["high_elo"]
    assert "auc_mean" in he and "auc_std" in he
    assert "baseline_tabular_auc" in he and "baseline_mlp_auc" in he
    assert (tmp_path / "sequence_supervised.pt").exists()
    assert (tmp_path / "sequence_metrics.json").exists()