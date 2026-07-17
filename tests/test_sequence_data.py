"""Tests sequence_data : labels, folds joueur-groupés, purge miroir, standardisation."""
import importlib.util, sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC / "02_data_science"))
import numpy as np
_spec = importlib.util.spec_from_file_location(
    "sequence_data", _SRC / "02_data_science" / "sequence_data.py")
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)


def _data(n=20, seed=0):
    rng = np.random.RandomState(seed)
    return {
        "sequences": rng.randn(n, 40, 20).astype(np.float32),
        "mask": np.ones((n, 40), dtype=bool),
        "label_highelo": rng.randint(0, 2, n),
        "rank": np.array(rng.choice(["diamond", "master", "grandmaster", "challenger"], n),
                         dtype=object),
        "puuid": np.array([f"p{i % 15}" for i in range(n)], dtype=object),
        "match_id": np.array([f"m{i // 2}" for i in range(n)], dtype=object),  # 2 rows/match
        "champion": np.array(["Zeri"] * n, dtype=object),
    }


def test_task_subset_highelo():
    d = _data()
    idx, y = sd.task_subset(d, "high_elo")
    assert len(idx) == len(d["rank"])
    assert set(y.tolist()) <= {0, 1}


def test_task_subset_dia_chall_filters():
    d = _data()
    idx, y = sd.task_subset(d, "dia_chall")
    # ne garde que diamond/challenger
    kept = d["rank"][idx]
    assert set(kept.tolist()) <= {"diamond", "challenger"}
    # label = 1 si challenger
    for i, yy in zip(idx, y):
        assert yy == (1 if d["rank"][i] == "challenger" else 0)


def test_player_folds_no_overlap():
    d = _data(n=20)
    idx, y = sd.task_subset(d, "high_elo")
    folds = sd.player_folds(d["puuid"][idx], y, n_splits=5, seed=42)
    seen = set()
    for tr, va in folds:
        trp = set(d["puuid"][idx][tr]); vap = set(d["puuid"][idx][va])
        assert trp.isdisjoint(vap)          # aucun joueur à la fois train et val
        assert vap.isdisjoint(seen)        # chaque joueur vu une seule fois en val
        seen |= vap


def test_mirror_purge_drops_opponent_of_val():
    # 2 rows par match ; si l'opponent puuid est en val, la row train du même match est purgée
    puuids = np.array(["a", "b", "a", "b"], dtype=object)
    match_ids = np.array(["m0", "m0", "m1", "m1"], dtype=object)
    train_idx = np.array([0, 2])           # row 0 (a,m0), row 2 (a,m1)
    val_puuids = {"b"}                      # b est en val -> row m0 (a) et m1 (a) sont miroir de b
    kept = sd.mirror_purge(train_idx, val_puuids, match_ids, puuids)
    assert len(kept) == 0                   # les 2 rows train partagent leur match avec b(val)


def test_standardize_fit_train_only():
    d = _data(n=20)
    train_idx = np.arange(15)
    mean, std = sd.standardize_fit(d["sequences"], d["mask"], train_idx)
    assert mean.shape == (20,) and std.shape == (20,)
    # sur le train, z-score donne mean≈0 std≈1 (frames valides toutes True ici)
    z = sd.standardize_apply(d["sequences"][train_idx], mean, std)
    assert abs(z.mean(axis=(0, 1))) .max() < 1e-5
    assert abs(z.std(axis=(0, 1)) - 1.0).max() < 1e-5


def test_standardize_uses_train_stats_on_val():
    d = _data(n=20, seed=1)
    train_idx = np.arange(15); val_idx = np.arange(15, 20)
    mean, std = sd.standardize_fit(d["sequences"], d["mask"], train_idx)
    z_val = sd.standardize_apply(d["sequences"][val_idx], mean, std)
    # val n'a PAS mean 0 (stats du train appliquées) -> on vérifie juste la shape + pas de NaN
    assert z_val.shape == (5, 40, 20)
    assert np.isfinite(z_val).all()