import json

import pytest

import positioning as P
import payload as PL
import test_game_journal as TJ


def _dirs(tmp_path):
    """Silver perso (1 game ADC + 1 game jungle) + gold référentiel challenger/adc."""
    silver, gold = tmp_path / "silver", tmp_path / "gold"
    p = silver / "personal" / "spadzze"
    p.mkdir(parents=True)
    recs = [  # la game ADC n'est PAS la dernière ligne -> prouve le filtre de scope
        {"match_id": "EUW1_42", "puuid": TJ.ME, "role": "BOTTOM",
         "champion": "Zeri", "win": True, "queue": 420},
        {"match_id": "EUW1_43", "puuid": TJ.ME, "role": "JUNGLE",
         "champion": "Diana", "win": False, "queue": 420},
    ]
    (p / "games.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    facet = {"n_games": 400, "deaths_per_game": 4.2,
             "by_zone_phase": {"BOT|mid": 0.05},
             "death_gold_state": {"ahead": 0.2, "even": 0.3, "behind": 0.5}}
    d = gold / "referentiel" / "challenger" / "adc"
    d.mkdir(parents=True)
    (d / "aggregate.json").write_text(json.dumps(
        {"n_games": 1000, "winrate": 0.5,
         "overall": facet, "win": facet, "loss": facet}))
    return silver, gold


def _load_raw(base):
    if base.endswith("_match"):
        return TJ._match(win=True)
    return TJ._basic_timeline({4: [TJ._kill(270000, victim=1, killer=6)],
                               5: [TJ._buy(310000, 1)]})


def test_build_game_selects_latest_game_of_scope(tmp_path):
    silver, gold = _dirs(tmp_path)
    pl = PL.build_game("spadzze", scope="adc", target="challenger",
                       gold_dir=gold, silver_dir=silver, load_raw=_load_raw)
    assert pl["meta"]["match_id"] == "EUW1_42"       # la Diana jungle est ignorée
    assert pl["meta"]["kind"] == "game"
    assert pl["meta"]["champion"] == "Zeri" and pl["meta"]["opponent"] == "Jinx"
    assert len(pl["journal"]["deaths"]) == 1
    assert pl["journal"]["deaths"][0]["clock"] == "4:30"
    assert len(pl["journal"]["recalls"]) == 1


def test_build_game_benchmarks_at_same_outcome(tmp_path):
    silver, gold = _dirs(tmp_path)
    pl = PL.build_game("spadzze", scope="adc", target="challenger",
                       gold_dir=gold, silver_dir=silver, load_raw=_load_raw)
    b = pl["benchmarks"]
    assert b["outcome"] == "win"                     # game gagnée -> facette win
    assert b["deaths_per_game"] == 4.2
    assert b["death_gold_state"]["behind"] == 0.5
    assert b["n_games_ref"] == 1000


def test_build_game_by_match_id_and_not_found(tmp_path):
    silver, gold = _dirs(tmp_path)
    pl = PL.build_game("spadzze", match_id="EUW1_42", scope="adc",
                       gold_dir=gold, silver_dir=silver, load_raw=_load_raw)
    assert pl["meta"]["match_id"] == "EUW1_42"
    with pytest.raises(FileNotFoundError):
        PL.build_game("spadzze", match_id="EUW1_999", scope="adc",
                      gold_dir=gold, silver_dir=silver, load_raw=_load_raw)


def test_build_game_never_leaks_ml_only(tmp_path):
    silver, gold = _dirs(tmp_path)
    pl = PL.build_game("spadzze", scope="adc", target="challenger",
                       gold_dir=gold, silver_dir=silver, load_raw=_load_raw)
    blob = json.dumps(pl)
    assert all(k not in blob for k in P.ML_ONLY)
