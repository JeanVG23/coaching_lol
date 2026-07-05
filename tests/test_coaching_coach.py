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


# --- run_batch + --game-batch --------------------------------------------------

def _batch_env(tmp_path, n_games=3, reviewed=("EUW1_2",)):
    """Silver perso ADC + reviews.jsonl existant -> (root, silver_dir)."""
    silver = tmp_path / "silver"
    pdir = silver / "personal" / "spadzze"
    pdir.mkdir(parents=True)
    recs = [{"match_id": f"EUW1_{i}", "role": "BOTTOM", "puuid": "p",
             "champion": "Zeri", "game_ts": i} for i in range(n_games)]
    (pdir / "games.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n")
    root = tmp_path / "07_coaching"
    out = root / "spadzze"
    out.mkdir(parents=True)
    lines = [{"ts": f"t{m}", "model": "kimi-k2.6", "kind": "game",
              "match_id": m, "scope": "adc", "target": "challenger",
              "payload": {"meta": {}}, "review": _game_review_dict()}
             for m in reviewed]
    (out / "reviews.jsonl").write_text(
        "\n".join(json.dumps(l) for l in lines) + "\n")
    return root, silver


def test_run_batch_generates_dedups_and_continues_on_error(tmp_path, monkeypatch, capsys):
    root, silver = _batch_env(tmp_path)          # EUW1_2 déjà reviewée

    def fake_build_game(player, match_id=None, **kw):
        if match_id == "EUW1_0":
            raise FileNotFoundError(f"raw manquant pour {match_id}")
        pl = _game_payload()
        pl["meta"]["match_id"] = match_id
        return pl

    monkeypatch.setattr(C.payload_mod, "build_game", fake_build_game)
    monkeypatch.setattr(C.llm_client, "generate_json",
                        lambda *a, **k: _game_review_dict())
    rc = C.run_batch("spadzze", "adc", "challenger", "m", 10,
                     root=root, silver_dir=silver)
    assert rc == 0
    lines = [json.loads(l) for l in
             (root / "spadzze" / "reviews.jsonl").read_text().splitlines()]
    new_ids = {l["match_id"] for l in lines if l["ts"] != "tEUW1_2"}
    assert new_ids == {"EUW1_1"}                 # EUW1_0 échouée, EUW1_2 dédupliquée
    out = capsys.readouterr().out
    assert "1 générée" in out and "1 déjà reviewée" in out and "1 échouée" in out


def test_run_batch_returns_1_when_all_attempts_fail(tmp_path, monkeypatch, capsys):
    root, silver = _batch_env(tmp_path, reviewed=())

    def boom(*a, **k):
        raise C.llm_client.LLMError("api down")

    monkeypatch.setattr(C.payload_mod, "build_game",
                        lambda player, match_id=None, **kw: _game_payload())
    monkeypatch.setattr(C.llm_client, "generate_json", boom)
    rc = C.run_batch("spadzze", "adc", "challenger", "m", 2,
                     root=root, silver_dir=silver)
    assert rc == 1
    assert "échouée" in capsys.readouterr().out


def test_run_batch_nothing_to_do(tmp_path, capsys):
    root, silver = _batch_env(tmp_path, n_games=1, reviewed=("EUW1_0",))
    rc = C.run_batch("spadzze", "adc", "challenger", "m", 10,
                     root=root, silver_dir=silver)
    assert rc == 0
    assert "déjà reviewée" in capsys.readouterr().out


def test_main_game_and_game_batch_mutually_exclusive(monkeypatch, capsys):
    monkeypatch.setattr(C.sys, "argv",
                        ["coach.py", "--game", "--game-batch", "5"])
    with pytest.raises(SystemExit) as e:
        C.main()
    assert e.value.code == 2                     # erreur argparse


def test_run_batch_validation_error_saves_failed_under_root(tmp_path, monkeypatch):
    root, silver = _batch_env(tmp_path, reviewed=())
    real_failed = C.rl.DATA / "07_coaching"
    existed_before = (real_failed / "spadzze" / "failed").exists()

    monkeypatch.setattr(C.payload_mod, "build_game",
                        lambda player, match_id=None, **kw: _game_payload())
    monkeypatch.setattr(C.llm_client, "generate_json",
                        lambda *a, **k: {"bogus": True})     # invalide -> CoachValidationError après retry
    rc = C.run_batch("spadzze", "adc", "challenger", "m", 1,
                     root=root, silver_dir=silver)
    assert rc == 1
    failed_dir = root / "spadzze" / "failed"
    assert failed_dir.exists() and any(failed_dir.iterdir())
    assert (real_failed / "spadzze" / "failed").exists() == existed_before
