import json

import pytest

import coach as C
import llm_client as LC
import schema as S


def _gen(data, **usage):
    """Generation factice : sortie + telemetrie (cf. llm_client.generate)."""
    base = {"latency_ms": 1200, "attempts": 1, "prompt_tokens": 900,
            "completion_tokens": 300, "total_tokens": 1200,
            "server_duration_ms": 1100, "cost_usd": None}
    return LC.Generation(data, {**base, **usage})


def _review_dict():
    ins = {"point": "p", "evidence": "e"}
    return {"strengths": [ins, ins, ins], "mistakes": [ins, ins, ins],
            "habits": ["h1", "h2"], "next_focus": "f", "confidence": 0.5}


def test_generate_review_validates(monkeypatch):
    monkeypatch.setattr(C.llm_client, "generate", lambda *a, **k: _gen(_review_dict()))
    r, run = C.generate_review({"meta": {"player": "x", "scope": "adc", "target": "challenger",
                                    "outcome_focus": "loss", "n_games_me": 1}}, "m")
    assert isinstance(r, S.Review) and r.confidence == 0.5
    # la trace du run accompagne la review : version de prompt + coût de l'appel
    assert run["prompt_version"] == C.prompt_mod.PROMPT_VERSION
    assert run["total_tokens"] == 1200 and run["schema_retries"] == 0


def test_generate_review_retries_then_raises(monkeypatch):
    calls = {"n": 0}

    def bad(*a, **k):
        calls["n"] += 1
        return _gen({"bogus": True})                 # invalide -> ValidationError

    monkeypatch.setattr(C.llm_client, "generate", bad)
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
            "mistakes": [{"point": "m", "cause": "solo 1v1 sans flash",
                          "evidence": "mort à 17:05, drake dans 6 s"}],
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
    monkeypatch.setattr(C.llm_client, "generate",
                        lambda *a, **k: _gen(_game_review_dict()))
    r, run = C.generate_game_review(_game_payload(), "m")
    assert isinstance(r, S.GameReview) and r.confidence == 0.4
    assert run["prompt_version"] == C.prompt_mod.GAME_PROMPT_VERSION


def test_specialized_review_runs_two_axes_then_chief_without_rewriting(monkeypatch):
    death = {"strengths": [{"point": "bon placement", "cause": "reste derrière",
                             "evidence": "à 10:04, 300 dégâts"}],
             "mistakes": [{"point": "respecte le poke", "cause": "autos avant engage",
                            "evidence": "mort à 11:06, 600 dégâts"}],
             "next_focus": "placement", "confidence": 0.7}
    economy = {"strengths": [],
               "mistakes": [{"point": "reset avant objectif", "cause": "achat tardif",
                              "evidence": "recall à 12:00, 1 500 g"}],
               "next_focus": "reset", "confidence": 0.6}
    chief = {
        "summary_insight_id": "death_positioning:mistakes:0",
        "priority_mistake_ids": ["death_positioning:mistakes:0",
                                 "economy_build:mistakes:0"],
        "strength_insight_ids": ["death_positioning:strengths:0"],
        "next_focus_insight_id": "economy_build:mistakes:0",
        "confidence": 0.75,
    }

    def fake_generate(model, system, user, schema, **kwargs):
        if "agent chef" in system:
            return _gen(chief)
        if "MORTS & POSITIONNEMENT" in system:
            return _gen(death)
        if "ÉCONOMIE & BUILD" in system:
            return _gen(economy)
        raise AssertionError(system)

    monkeypatch.setattr(C.llm_client, "generate", fake_generate)
    review, run = C.generate_specialized_game_review(_game_payload(), "m")
    assert isinstance(review, S.SpecializedGameReview)
    assert review.summary == "respecte le poke"
    assert [item.point for item in review.mistakes] == [
        "respecte le poke", "reset avant objectif"]
    assert review.next_focus == "reset avant objectif"
    assert [axis.axis for axis in review.axes] == [
        "death_positioning", "economy_build"]
    assert len(run["stages"]) == 3


def test_persist_game_records_kind_and_match_id(tmp_path):
    pl = _game_payload()
    review = S.GameReview.model_validate(_game_review_dict())
    path = C.persist("spadzze", "m", pl, review, ts="2026-07-05T10:00:00",
                     root=tmp_path)
    line = json.loads(path.read_text().splitlines()[-1])
    assert line["kind"] == "game" and line["match_id"] == "EUW1_42"
    assert line["review"]["mistakes"][0]["evidence"].startswith("mort à 17:05")


def test_persist_records_the_run_trace(tmp_path):
    """Sans trace de run, une variation du taux d'utilite n'est attribuable ni au
    prompt ni au modele : la review persistee porte version de prompt + cout."""
    pl = _game_payload()
    review = S.GameReview.model_validate(_game_review_dict())
    run = {"prompt_version": "abc123", "latency_ms": 4200, "total_tokens": 1200,
           "cost_usd": None, "schema_retries": 1}
    path = C.persist("spadzze", "m", pl, review, ts="2026-07-05T10:00:00",
                     root=tmp_path, run=run)
    line = json.loads(path.read_text().splitlines()[-1])
    assert line["run"] == run


def test_generate_cumulates_usage_of_a_schema_retry(monkeypatch):
    """Une sortie rejetee par le schema a bien coute un appel : ses tokens
    comptent dans le run, sinon le cout par review est sous-estime."""
    seq = [_gen({"bogus": True}), _gen(_game_review_dict())]
    monkeypatch.setattr(C.llm_client, "generate", lambda *a, **k: seq.pop(0))
    _, run = C.generate_game_review(_game_payload(), "m")
    assert run["schema_retries"] == 1
    assert run["total_tokens"] == 2400 and run["latency_ms"] == 2400


def test_render_run_is_readable_without_price_table():
    txt = C.render_run({"prompt_version": "abc123", "latency_ms": 4200,
                        "total_tokens": 1200, "cost_usd": None})
    assert "abc123" in txt and "4.2 s" in txt and "1200 tokens" in txt
    assert "$" not in txt                        # pas de prix invente


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

    def fake_gen(model, system, user, sch, **kw):
        used["model"] = model
        return _gen(_review_dict())

    monkeypatch.setattr(C.llm_client, "generate", fake_gen)
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
    monkeypatch.setattr(C.llm_client, "generate",
                        lambda *a, **k: _gen(_game_review_dict()))
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
    monkeypatch.setattr(C.llm_client, "generate", boom)
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
    monkeypatch.setattr(C.llm_client, "generate",
                        lambda *a, **k: _gen({"bogus": True}))  # invalide -> CoachValidationError après retry
    rc = C.run_batch("spadzze", "adc", "challenger", "m", 1,
                     root=root, silver_dir=silver)
    assert rc == 1
    failed_dir = root / "spadzze" / "failed"
    assert failed_dir.exists() and any(failed_dir.iterdir())
    assert (real_failed / "spadzze" / "failed").exists() == existed_before
