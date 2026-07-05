import json

import pytest

import coach as C
import schema as S


def _review_dict():
    ins = {"point": "p", "evidence": "e"}
    return {"strengths": [ins, ins, ins], "mistakes": [ins, ins, ins],
            "habits": ["h1", "h2"], "next_focus": "f", "confidence": 0.5}


def test_generate_review_validates(monkeypatch):
    monkeypatch.setattr(C.llm_client, "generate_json", lambda *a, **k: _review_dict())
    r = C.generate_review({"meta": {"player": "x", "scope": "adc", "target": "challenger",
                                    "outcome_focus": "loss", "n_games_me": 1}}, "m")
    assert isinstance(r, S.Review) and r.confidence == 0.5


def test_generate_review_retries_then_raises(monkeypatch):
    calls = {"n": 0}

    def bad(*a, **k):
        calls["n"] += 1
        return {"bogus": True}                       # invalide -> ValidationError

    monkeypatch.setattr(C.llm_client, "generate_json", bad)
    with pytest.raises(C.CoachValidationError):
        C.generate_review({"meta": {"player": "x", "scope": "adc", "target": "challenger",
                                    "outcome_focus": "loss", "n_games_me": 1}}, "m")
    assert calls["n"] == 2                            # 1 essai + 1 retry


def test_persist_appends_jsonl(tmp_path):
    pl = {"meta": {"scope": "adc", "target": "challenger", "outcome_focus": "loss"}}
    review = S.Review.model_validate(_review_dict())
    path = C.persist("spadzze", "deepseek-v4-pro", pl, review,
                     ts="2026-06-30T10:00:00", root=tmp_path)
    line = json.loads(path.read_text().splitlines()[-1])
    assert line["model"] == "deepseek-v4-pro"
    assert line["review"]["confidence"] == 0.5
    assert line["payload"]["meta"]["scope"] == "adc"
    assert line["ts"] == "2026-06-30T10:00:00"


def _game_review_dict():
    return {"strengths": [],
            "mistakes": [{"point": "m", "evidence": "mort à 17:05, drake dans 6 s"}],
            "next_focus": "f", "confidence": 0.4}


def _game_payload():
    return {"meta": {"player": "x", "scope": "adc", "target": "challenger",
                     "kind": "game", "match_id": "EUW1_42", "champion": "Zeri",
                     "opponent": "Jinx", "role": "BOTTOM", "win": False,
                     "duration_min": 30.0, "patch": "16.13",
                     "kda": {"kills": 5, "deaths": 3, "assists": 7}},
            "journal": {"deaths": [], "recalls": []},
            "benchmarks": {"outcome": "loss"}}


def test_generate_game_review_validates(monkeypatch):
    monkeypatch.setattr(C.llm_client, "generate_json",
                        lambda *a, **k: _game_review_dict())
    r = C.generate_game_review(_game_payload(), "m")
    assert isinstance(r, S.GameReview) and r.confidence == 0.4


def test_persist_game_records_kind_and_match_id(tmp_path):
    pl = _game_payload()
    review = S.GameReview.model_validate(_game_review_dict())
    path = C.persist("spadzze", "m", pl, review, ts="2026-07-05T10:00:00",
                     root=tmp_path)
    line = json.loads(path.read_text().splitlines()[-1])
    assert line["kind"] == "game" and line["match_id"] == "EUW1_42"
    assert line["review"]["mistakes"][0]["evidence"].startswith("mort à 17:05")


def test_render_game_text_has_sections():
    txt = C.render_game_text(S.GameReview.model_validate(_game_review_dict()))
    assert "Erreurs" in txt and "Focus" in txt


def test_render_text_is_french_and_has_sections():
    txt = C.render_text(S.Review.model_validate(_review_dict()))
    assert "Forces" in txt and "Erreurs" in txt and "Focus" in txt


def test_main_model_from_dotenv_when_no_flag(monkeypatch, tmp_path):
    """Sans --model ni OLLAMA_MODEL shell, le modèle vient du .env (load_env)."""
    monkeypatch.setattr(C.sys, "argv", ["coach.py", "--player", "x", "--scope", "adc"])
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setattr(C.rl, "load_env", lambda: {"OLLAMA_MODEL": "glm-5.2"})
    monkeypatch.setattr(C.payload_mod, "build",
                        lambda *a, **k: {"meta": {"scope": "adc", "target": "challenger",
                                                  "outcome_focus": "loss", "n_games_me": 1}})
    used = {}

    def fake_gen(model, system, user, sch):
        used["model"] = model
        return _review_dict()

    monkeypatch.setattr(C.llm_client, "generate_json", fake_gen)
    monkeypatch.setattr(C, "persist", lambda *a, **k: tmp_path / "reviews.jsonl")
    assert C.main() == 0
    assert used["model"] == "glm-5.2"             # .env honored, pas le défaut deepseek


# --- pending_game_matches (sélection batch, pure) -----------------------------

def test_pending_game_matches_dedups_scopes_and_sorts_recent_first():
    records = [
        {"match_id": "m1", "role": "BOTTOM", "game_ts": 100},
        {"match_id": "m2", "role": "BOTTOM", "game_ts": 300},   # déjà reviewée
        {"match_id": "m3", "role": "MIDDLE", "game_ts": 400},   # hors scope adc
        {"match_id": "m4", "role": "BOTTOM", "game_ts": 200},
    ]
    reviews = [{"kind": "game", "match_id": "m2", "model": "kimi-k2.6"},
               {"outcome_focus": "loss"}]          # agrégée : ne dédupe rien
    got = C.pending_game_matches(records, reviews, "adc", n=10)
    assert got == ["m4", "m1"]                     # ts décroissant, m2/m3 exclues


def test_pending_game_matches_limits_to_n():
    records = [{"match_id": f"m{i}", "role": "BOTTOM", "game_ts": i}
               for i in range(5)]
    got = C.pending_game_matches(records, [], "adc", n=2)
    assert got == ["m4", "m3"]


def test_pending_game_matches_falls_back_to_reversed_file_order():
    # silver perso antérieur au 2026-07-06 : pas de game_ts -> ordre inversé du fichier
    records = [{"match_id": f"m{i}", "role": "BOTTOM"} for i in range(3)]
    got = C.pending_game_matches(records, [], "adc", n=10)
    assert got == ["m2", "m1", "m0"]