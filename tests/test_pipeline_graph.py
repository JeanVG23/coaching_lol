"""Le Makefile declare le pipeline : on verifie que ce qu'il declare existe.

Un orchestrateur qui reference un script renomme echoue en production, au pire
moment (apres une collecte de plusieurs heures). Ces tests sont l'equivalent
d'un `--dry-run` de DAG : ils ne calculent rien, ils verifient que le graphe
tient debout et que l'ordre topologique est celui du pipeline medallion.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAKEFILE = ROOT / "Makefile"
pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make absent")


def _plan(tmp_path) -> list[str]:
    """Commandes que make lancerait pour `pipeline`, sur une racine de donnees
    vide : rien n'existe, donc tout le graphe se deroule."""
    out = subprocess.run(["make", "-n", "pipeline", f"DATA={tmp_path}"],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    return [ln.strip() for ln in out.stdout.splitlines() if "src/" in ln]


def test_every_script_referenced_by_the_makefile_exists():
    """Y compris les cibles reseau, que les tests ne lancent jamais : c'est
    justement celles-la qu'un renommage casserait en silence."""
    paths = set(re.findall(r"src/[\w./]+\.py", MAKEFILE.read_text()))
    assert paths, "aucun script reference : le Makefile a-t-il change de forme ?"
    missing = sorted(p for p in paths if not (ROOT / p).exists())
    assert not missing, f"scripts references mais absents : {missing}"


def test_the_plan_follows_the_medallion_order(tmp_path):
    plan = "\n".join(_plan(tmp_path))
    order = ["reextract_silver", "build_dataset", "build_player_dataset",
             "build_split", "train_player_ensemble", "calibrate_player_rank"]
    positions = [plan.find(step) for step in order]
    assert all(p >= 0 for p in positions), dict(zip(order, positions))
    assert positions == sorted(positions), f"ordre casse : {dict(zip(order, positions))}"


def test_the_offline_pipeline_never_calls_the_api_nor_publishes(tmp_path):
    """`make pipeline` doit rester hors-ligne : la collecte Riot et la
    publication Cloudflare sont des cibles explicites, jamais des dependances.
    Sans ce test, ajouter un `fetch_apex_lp` en prerequis passerait inapercu
    jusqu'a la premiere facture de quota."""
    forbidden = ("build_referential", "fetch_apex_lp", "densify_",
                 "sync_cloudflare", "aggregate_games")
    plan = "\n".join(_plan(tmp_path))
    assert not [name for name in forbidden if name in plan], plan
