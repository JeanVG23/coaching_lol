"""`make demo` : la chaîne complète sur les fixtures versionnées, hors réseau.

Ce test existe pour que la démo ne pourrisse pas. Une cible `make demo` non
testée casse au premier changement de features, et personne ne s'en aperçoit
avant le lecteur qui vient de cloner le dépôt.

Il vérifie aussi la propriété qui rend les fixtures publiables : aucun
identifiant Riot réel. La pseudonymisation est un invariant du dépôt, pas une
étape ponctuelle du script qui les a produites.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import champion_profiles as cp
import grounding
import mock_llm
import payload as payload_mod
import riotlib as rl
import schema as schema_mod

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "demo"
PLAYER, RANK = "spadzze", "challenger"


def test_fixtures_carry_no_real_identifier():
    """Les jetons démo sont reconnaissables ; tout le reste serait une fuite."""
    for path in sorted((FIXTURES / "01_raw").glob("*_match.json.zst")):
        match = rl._read_raw_at(path)
        assert match["metadata"]["matchId"].startswith("DEMO")
        for p in match["info"]["participants"]:
            assert p["puuid"].startswith("DEMO-PUUID-")
            assert re.fullmatch(r"Joueur\d{3}", p["riotIdGameName"])
            assert p["riotIdTagline"] == "DEMO"
            assert p["summonerId"].startswith("DEMO-SUMMONER-")


def test_the_pipeline_rebuilds_silver_and_gold(demo_data):
    """Le silver versionné n'est qu'une amorce : c'est le raw qui fait foi."""
    seed = json.loads((FIXTURES / "02_silver" / rl.KIND_PERSONAL / PLAYER
                       / "games.jsonl").read_text().splitlines()[0])
    assert set(seed) == {"match_id", "puuid", "rank"}   # amorce, pas des données

    games = rl.read_jsonl(rl.silver_games(rl.KIND_PERSONAL, PLAYER))
    assert len(games) == 24
    assert all({"champion", "role", "deaths", "position", "comp"} <= set(g)
               for g in games)
    agg = json.loads((rl.gold_base(rl.KIND_REF, RANK) / "adc"
                      / "aggregate.json").read_text())
    assert agg["n_games"] > 30 and agg["loss"]["by_zone_phase"]


def test_context_is_derived_offline(demo_data):
    """Le Data Dragon élagué doit suffire : sans lui tout retomberait en
    `unknown`, et la démo perdrait ce qu'elle a de plus parlant."""
    assert cp.load_items(), "catalogue d'items absent des fixtures"
    games = rl.read_jsonl(rl.silver_games(rl.KIND_REF, RANK))
    patterns = {cp.derive_context(g["comp"])["lane_pattern"]
                for g in games if g.get("comp")}
    assert patterns - {"unknown"}


def test_mock_llm_produces_a_schema_valid_grounded_review(demo_data, tmp_path):
    """Le bouchon passe par le vrai chemin : schéma Pydantic puis ancrage.

    S'il citait un chiffre absent du payload, `grounding` le verrait : c'est ce
    qui distingue un mock utile d'un décor.
    """
    pl = payload_mod.build_game(PLAYER, match_id=None, scope="adc", target=RANK)
    system, user = __import__("prompt").render_game(pl)
    gen = mock_llm.generate(mock_llm.MODEL_NAME, system, user,
                            schema_mod.game_review_json_schema())
    review = schema_mod.GameReview.model_validate(gen.data)
    check = grounding.check_review({"payload": pl, "review": review.model_dump()})
    score = grounding.score(check)
    assert score["grounded_rate"] == 1.0
    assert score["clock_rate"] == 1.0
    assert not check["asymmetry_violations"]


def test_make_demo_runs_end_to_end(tmp_path):
    """La cible elle-même, telle qu'un lecteur la lance après un clone.

    Non marquée « lente » à dessein : c'est la seule vérification que la commande
    mise en avant dans le README fonctionne encore. Un test qu'on saute par défaut
    ne protège rien."""
    proc = subprocess.run(["make", "demo", f"DEMO_DIR={tmp_path / 'demo'}"],
                          cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "ANCRAGE" in proc.stdout and "COACHING GAME" in proc.stdout
    assert sys.executable  # la démo tourne dans l'environnement Poetry
