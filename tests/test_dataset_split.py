import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))                    # riotlib
sys.path.insert(0, str(ROOT / "src" / "core"))           # dataset_split
sys.path.insert(0, str(ROOT / "src" / "01_data_engineering"))  # build_split
import build_split as bs
import dataset_split as ds


def _ranks(n_per_rank: dict) -> dict:
    return {f"{rank}-{i}": rank for rank, n in n_per_rank.items() for i in range(n)}


def test_assign_deterministic():
    ranks = _ranks({"master": 100, "challenger": 40, "grandmaster": 12})
    assert bs.assign(ranks) == bs.assign(ranks)


def test_assign_disjoint_and_covers_all():
    ranks = _ranks({"master": 100, "challenger": 40, "grandmaster": 12})
    a = bs.assign(ranks)
    assert set(a) == set(ranks)                  # tout le monde est assigné
    assert set(a.values()) <= set(ds.BUCKETS)    # buckets valides seulement


def test_assign_stratified_proportions():
    a = bs.assign(_ranks({"master": 100}))
    counts = {b: sum(1 for v in a.values() if v == b) for b in ds.BUCKETS}
    assert counts == {"train": 70, "calibration": 15, "test": 15}


def test_load_split_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ds.load_split(tmp_path / "nope.json")


def test_puuids_in_rejects_unknown_bucket():
    with pytest.raises(ValueError):
        ds.puuids_in({"assignment": {}}, "validation")


def test_partition_filters_by_bucket():
    split = {"assignment": {"a": "train", "b": "test", "c": "train"}}
    df = pd.DataFrame({"puuid": ["a", "b", "c", "d"], "v": [1, 2, 3, 4]})
    assert set(ds.partition(df, split, "train")["puuid"]) == {"a", "c"}
    assert set(ds.partition(df, split, "test")["puuid"]) == {"b"}


def test_compute_dataset_identity_deterministic():
    puuids = ["p2", "p1", "p3"]
    features = ["feat_b", "feat_a"]
    split = {"p1": "train", "p2": "test", "p3": "calibration"}
    id1 = ds.compute_dataset_identity(puuids, features, split, min_player_games=15)
    id2 = ds.compute_dataset_identity(["p1", "p3", "p2"], ["feat_a", "feat_b"], split, min_player_games=15)
    assert id1 == id2
    assert id1["n_players"] == 3
    assert len(id1["fingerprint"]) == 16


def test_compute_dataset_identity_sensitive_to_changes():
    base = ds.compute_dataset_identity(["p1", "p2"], ["feat_a"], min_player_games=15)
    diff_puuids = ds.compute_dataset_identity(["p1", "p3"], ["feat_a"], min_player_games=15)
    diff_feats = ds.compute_dataset_identity(["p1", "p2"], ["feat_b"], min_player_games=15)
    diff_min_games = ds.compute_dataset_identity(["p1", "p2"], ["feat_a"], min_player_games=20)

    assert base["fingerprint"] != diff_puuids["fingerprint"]
    assert base["fingerprint"] != diff_feats["fingerprint"]
    assert base["fingerprint"] != diff_min_games["fingerprint"]

