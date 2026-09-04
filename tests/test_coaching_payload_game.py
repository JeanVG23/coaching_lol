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
         "champion": "Zeri", "win": True, "queue": 420,
         "comp": {"self_adc": "Zeri", "self_support": "Lulu",
                  "enemy_adc": "Jinx", "enemy_support": "Thresh",
                  "self_jungle": "Vi", "enemy_jungle": "LeeSin",
                  "enemy_mid": "Orianna"}},
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


# --- filter_scope (partagé _select_game / sélection batch) --------------------

def test_filter_scope_by_role_champion_and_all():
    records = [
        {"match_id": "m1", "role": "BOTTOM", "champion": "Zeri"},
        {"match_id": "m2", "role": "MIDDLE", "champion": "Ahri"},
        {"match_id": "m3", "role": "BOTTOM", "champion": "Jinx"},
    ]
    assert [r["match_id"] for r in PL.filter_scope(records, "adc")] == ["m1", "m3"]
    assert [r["match_id"] for r in PL.filter_scope(records, "zeri")] == ["m1"]
    assert [r["match_id"] for r in PL.filter_scope(records, "all")] == ["m1", "m2", "m3"]


# --- Items résolus et contexte de matchup ----------------------------------------

FAKE_CATALOG = {1055: {"name": "Doran's Blade", "cost": 450}}


def test_build_game_resolves_recall_items(tmp_path, monkeypatch):
    silver, gold = _dirs(tmp_path)
    monkeypatch.setattr(PL.cprof, "load_items", lambda: FAKE_CATALOG)
    pl = PL.build_game("spadzze", scope="adc", target="challenger",
                       gold_dir=gold, silver_dir=silver, load_raw=_load_raw)
    (r1,) = pl["journal"]["recalls"]
    assert r1["items"] == [{"name": "Doran's Blade", "cost": 450}]
    assert "item_ids" not in r1                      # ids bruts non exposés au LLM


def test_build_game_degrades_without_item_catalog(tmp_path, monkeypatch):
    silver, gold = _dirs(tmp_path)
    monkeypatch.setattr(PL.cprof, "load_items", lambda: {})
    pl = PL.build_game("spadzze", scope="adc", target="challenger",
                       gold_dir=gold, silver_dir=silver, load_raw=_load_raw)
    (r1,) = pl["journal"]["recalls"]
    assert "items" not in r1 and "item_ids" not in r1


def test_build_game_exposes_matchup_context(tmp_path, monkeypatch):
    silver, gold = _dirs(tmp_path)
    monkeypatch.setattr(PL.cprof, "load_items", lambda: {})
    monkeypatch.setattr(PL.cprof, "derive_context",
                        lambda comp: {"lane_pattern": "all_in",
                                      "gank_exposure": "high"})
    pl = PL.build_game("spadzze", scope="adc", target="challenger",
                       gold_dir=gold, silver_dir=silver, load_raw=_load_raw)
    ctx = pl["context"]
    assert ctx["comp"]["enemy_support"] == "Thresh"
    assert ctx["lane_pattern"] == "all_in" and ctx["gank_exposure"] == "high"


def test_build_game_exposes_lane_matchup_spells_runes_and_final_build(tmp_path, monkeypatch):
    silver, gold = _dirs(tmp_path)
    match = TJ._match()
    me, opponent = match["info"]["participants"][0], match["info"]["participants"][5]
    me.update({"summoner1Id": 4, "summoner2Id": 7, "item0": 1055,
               "perks": {"styles": [{"selections": [{"perk": 8005}]},
                                      {"selections": [{"perk": 9101}]}]}})
    opponent.update({"summoner1Id": 4, "summoner2Id": 21, "item0": 1055,
                     "perks": {"styles": [{"selections": [{"perk": 8021}]},
                                            {"selections": []}]}})

    def load(base):
        return match if base.endswith("_match") else TJ._basic_timeline()

    monkeypatch.setattr(PL.cprof, "load_items", lambda: FAKE_CATALOG)
    pl = PL.build_game("spadzze", scope="adc", target="challenger",
                       gold_dir=gold, silver_dir=silver, load_raw=load)
    matchup = pl["context"]["matchup"]
    assert matchup["lane_opponent"] == "Jinx"
    assert [spell["name"] for spell in matchup["player"]["summoner_spells"]] == [
        "Flash", "Heal"]
    assert matchup["player"]["keystone"]["name"] == "Press the Attack"
    assert matchup["player"]["secondary_runes"] == [{"id": 9101, "name": "Absorb Life"}]
    assert matchup["opponent"]["final_build"] == [FAKE_CATALOG[1055]]


def test_death_links_gold_to_actual_next_purchase(tmp_path, monkeypatch):
    """Régression feedback : 1 268 g avant BF 1 300 g n'est pas un mauvais reset."""
    silver, gold = _dirs(tmp_path)
    monkeypatch.setattr(PL.cprof, "load_items", lambda: {
        1038: {"name": "B. F. Sword", "cost": 1300},
    })

    def load(base):
        if base.endswith("_match"):
            return TJ._match()
        return TJ._basic_timeline({
            11: [TJ._kill(666000, victim=1, killer=6),
                 TJ._buy(677000, 1, item=1038)],
        })

    pl = PL.build_game("spadzze", scope="adc", target="challenger",
                       gold_dir=gold, silver_dir=silver, load_raw=load)
    death = pl["journal"]["deaths"][0]
    assert death["unspent_gold"] == 1234
    assert death["next_purchase"] == {
        "clock": "11:17",
        "items": [{"name": "B. F. Sword", "cost": 1300}],
        "cheapest_item_cost": 1300,
    }


def test_build_game_keeps_raw_matchup_without_silver_comp(tmp_path, monkeypatch):
    # Le matchup vient du match brut même si l'ancien silver n'a pas de comp.
    silver, gold = _dirs(tmp_path)
    monkeypatch.setattr(PL.cprof, "load_items", lambda: {})
    pl = PL.build_game("spadzze", match_id="EUW1_43", scope="adc",
                       gold_dir=gold, silver_dir=silver, load_raw=_load_raw)
    assert "comp" not in pl["context"]
    assert pl["context"]["matchup"]["lane_opponent"] == "Jinx"
