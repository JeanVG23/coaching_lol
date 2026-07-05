"""Dimension temporelle du dataset : extract_game porte game_ts, build_dataset
l'expose (avec patch) en colonnes méta du parquet."""
import importlib.util
import sys
from pathlib import Path

import riotlib as rl

# build_dataset vit dans src/01_data_engineering/ (dossier non importable tel quel).
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
_spec = importlib.util.spec_from_file_location(
    "build_dataset", _SRC / "01_data_engineering" / "build_dataset.py")
build_dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_dataset)


def _match(**info_extra):
    def p(team, role, champ):
        return {"teamId": team, "teamPosition": role, "championName": champ, "win": True}
    parts = [
        p(100, "BOTTOM", "Zeri"), p(100, "UTILITY", "Lulu"), p(100, "JUNGLE", "Graves"),
        p(100, "MIDDLE", "Ahri"), p(100, "TOP", "Aatrox"),
        p(200, "BOTTOM", "Caitlyn"), p(200, "UTILITY", "Leona"),
        p(200, "JUNGLE", "JarvanIV"), p(200, "MIDDLE", "Syndra"), p(200, "TOP", "Sett"),
    ]
    info = {"mapId": 11, "queueId": 420, "gameVersion": "15.13.1.1",
            "participants": parts}
    info.update(info_extra)
    return {"metadata": {"matchId": "T1", "participants": [f"puuid{i}" for i in range(10)]},
            "info": info}


def test_extract_game_carries_game_ts():
    g = rl.extract_game(_match(gameStartTimestamp=1751700000000), {"info": {"frames": []}},
                        "puuid0", rank="test")
    assert g["game_ts"] == 1751700000000


def test_extract_game_game_ts_falls_back_to_game_creation():
    g = rl.extract_game(_match(gameCreation=1751600000000), {"info": {"frames": []}},
                        "puuid0", rank="test")
    assert g["game_ts"] == 1751600000000


def test_game_to_row_exposes_patch_and_game_ts():
    g = {"match_id": "EUW1_1", "puuid": "p1", "champion": "Zeri", "win": True,
         "patch": "15.13", "game_ts": 1751700000000,
         "lane": {}, "deaths": [], "kills": [], "assists": []}
    row = build_dataset.game_to_row(g, "challenger", "referentiel")
    assert row["patch"] == "15.13"
    assert row["game_ts"] == 1751700000000


def test_game_to_row_missing_temporal_yields_none():
    # vieux silver perso sans game_ts : la colonne existe mais reste vide (pas de KeyError)
    g = {"match_id": "EUW1_2", "puuid": "p1", "champion": "Zeri", "win": True,
         "lane": {}, "deaths": [], "kills": [], "assists": []}
    row = build_dataset.game_to_row(g, None, "personal:spadzze")
    assert row["patch"] is None
    assert row["game_ts"] is None
