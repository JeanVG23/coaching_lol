"""Smoke : pretrain SSL + finetune tourne sur mini-dataset synthétique."""
import importlib.util, sys
from pathlib import Path
import numpy as np
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC / "02_data_science"))
_spec = importlib.util.spec_from_file_location(
    "pretrain_sequence_model", _SRC / "02_data_science" / "pretrain_sequence_model.py")
ptm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ptm)


def _mini(n=40, seed=0):
    rng = np.random.RandomState(seed)
    puuids = np.array([f"p{i // 4}" for i in range(n)], dtype=object)
    ranks = np.array(rng.choice(["diamond", "challenger"], n), dtype=object)
    y = (ranks == "challenger").astype(int)
    seqs = rng.randn(n, 40, 27).astype(np.float32)   # 20 -> 27
    seqs[y == 1, 10, 2] += 5.0
    return {"sequences": seqs, "mask": np.ones((n, 40), dtype=bool), "rank": ranks,
            "puuid": puuids, "match_id": np.array([f"m{i}" for i in range(n)], dtype=object),
            "champion": np.array(["Zeri"] * n, dtype=object)}


def test_pretrain_returns_delta(tmp_path, monkeypatch):
    monkeypatch.setattr(ptm, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(ptm.sd, "DATASET", tmp_path / "x.npz")
    np.savez(tmp_path / "x.npz", **_mini())
    out = ptm.run(pretrain_epochs=5, finetune_epochs=5, batch=8, seed=42, device_force="cpu")
    assert "auc_supervised" in out and "auc_ssl" in out and "delta_ssl" in out
    assert (tmp_path / "sequence_encoder_pretrain.pt").exists()


def test_embed_game_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(ptm.sd, "DATASET", tmp_path / "x.npz")
    np.savez(tmp_path / "x.npz", **_mini())
    emb = ptm.embed_game(np.random.randn(40, 27).astype(np.float32),   # 20 -> 27
                         np.ones(40, dtype=bool), device_force="cpu")
    assert emb.shape == (64,)