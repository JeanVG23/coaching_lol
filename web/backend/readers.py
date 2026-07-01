# web/backend/readers.py
"""Lecteurs purs des données existantes (silver/gold/reviews/feedback/shap).

Aucune DB. Toutes les fonctions acceptent `data_root` (défaut rl.DATA) pour les tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import riotlib as rl

import settings


def _silver_games_path(slug: str, data_root: Path | None) -> Path:
    base = data_root if data_root is not None else rl.DATA
    return Path(base) / "02_silver" / "personal" / slug / "games.jsonl"


def _aggregate_path(slug: str, scope: str, data_root: Path | None) -> Path:
    base = data_root if data_root is not None else rl.DATA
    return Path(base) / "03_gold" / "personal" / slug / scope / "aggregate.json"


def _coaching_dir(slug: str, data_root: Path | None) -> Path:
    base = data_root if data_root is not None else rl.DATA
    return Path(base) / "07_coaching" / slug


def _shap_drivers_path(slug: str, data_root: Path | None) -> Path:
    base = data_root if data_root is not None else rl.DATA
    return Path(base) / "06_shap" / f"{slug}_drivers.json"


def _rank_path(slug: str, data_root: Path | None) -> Path:
    base = data_root if data_root is not None else rl.DATA
    return Path(base) / "02_silver" / "personal" / slug / "rank.json"


def _match_seq(match_id: str) -> int:
    """Numéro de séquence Riot (suffixe après '_'), croît avec le temps sur un
    même shard plateforme. `merge_jsonl` append en fin de fichier sans trier —
    on ne peut donc pas se fier à l'ordre du fichier pour le tri chronologique."""
    try:
        return int(match_id.rsplit("_", 1)[-1])
    except (ValueError, AttributeError):
        return 0


def read_games(slug: str, page: int = 1, size: int = 20,
               data_root: Path | None = None) -> dict:
    path = _silver_games_path(slug, data_root)
    rows: list[dict] = []
    if path.exists():
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = sorted(rows, key=lambda r: _match_seq(r.get("match_id", "")), reverse=True)
    total = len(rows)
    start = (page - 1) * size
    items = rows[start:start + size]
    return {"items": items, "page": page, "size": size, "total": total}


def read_aggregate(slug: str, scope: str = "all",
                   data_root: Path | None = None) -> dict | None:
    path = _aggregate_path(slug, scope, data_root)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_reviews(slug: str, data_root: Path | None = None) -> list[dict]:
    path = _coaching_dir(slug, data_root) / "reviews.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def read_feedback(slug: str, data_root: Path | None = None) -> list[dict]:
    path = _coaching_dir(slug, data_root) / "feedback.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def read_shap(slug: str, data_root: Path | None = None) -> dict:
    path = _shap_drivers_path(slug, data_root)
    if not path.exists():
        return {"available": False, "drivers": []}
    return {"available": True, "drivers": json.loads(path.read_text())}


def read_rank(slug: str, data_root: Path | None = None) -> dict | None:
    """Rang solo/duo mis en cache au dernier fetch. None si jamais fetché."""
    path = _rank_path(slug, data_root)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def account_summaries(data_root: Path | None = None) -> list[dict]:
    out = []
    for a in settings.load_accounts():
        games = read_games(a["slug"], page=1, size=1, data_root=data_root)
        revs = read_reviews(a["slug"], data_root=data_root)
        out.append({
            "slug": a["slug"],
            "riot_id": a["riot_id"],
            "region": a["region"],
            "games_count": games["total"],
            "last_review_ts": revs[-1]["ts"] if revs else None,
        })
    return out