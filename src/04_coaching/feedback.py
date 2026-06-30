#!/usr/bin/env python3
"""04_coaching — boucle d'évaluation : annotation + agrégation des reviews.

Sous-commandes :
  annotate  : choisir une review persistée, juger chaque insight (utile/faux + tag),
              persister dans data/07_coaching/<player>/feedback.jsonl.
  summary   : agréger les annotations (taux utile, par section, top tags, par
              modèle, tendance). Aucun appel réseau.

Usage :
  python3 src/04_coaching/feedback.py annotate --player spadzze [--ts <ts> | --last]
  python3 src/04_coaching/feedback.py summary  --player spadzze [--tag <t>] [--model <m>]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # accès src/
import riotlib as rl

import schema as schema_mod


def _reviews_path(player: str, root=None) -> Path:
    root = Path(root) if root is not None else rl.DATA / "07_coaching"
    return root / player / "reviews.jsonl"


def _feedback_path(player: str, root=None) -> Path:
    root = Path(root) if root is not None else rl.DATA / "07_coaching"
    return root / player / "feedback.jsonl"


def list_reviews(player: str, root=None) -> list[dict]:
    path = _reviews_path(player, root)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_review(player: str, ts: str, root=None) -> dict | None:
    for r in list_reviews(player, root):
        if r.get("ts") == ts:
            return r
    return None


def build_feedback(review: schema_mod.Review, ts: str, player: str, model: str,
                   rated_at: str,
                   responses: dict[tuple[str, int], tuple[bool, str | None, str | None]]
                   ) -> schema_mod.Feedback:
    """Construit un Feedback en n'incluant que les items présents dans responses
    (skip = item omis). Invariant tag-requis validé par FeedbackItem."""
    sections = [("strength", review.strengths), ("mistake", review.mistakes),
                ("habit", review.habits)]
    items = []
    for kind, section in sections:
        for i, _ in enumerate(section):
            key = (kind, i)
            if key not in responses:
                continue
            useful, tag, note = responses[key]
            items.append(schema_mod.FeedbackItem(kind=kind, index=i,
                                                 useful=useful, tag=tag, note=note))
    key = ("focus", 0)
    if key in responses:
        useful, tag, note = responses[key]
        items.append(schema_mod.FeedbackItem(kind="focus", index=0,
                                             useful=useful, tag=tag, note=note))
    return schema_mod.Feedback(ts=ts, player=player, rated_at=rated_at,
                               model=model, items=items)


def persist_feedback(player: str, fb: schema_mod.Feedback,
                     root=None) -> tuple[Path, bool]:
    """Append/écrase : un ts apparaît au plus une fois. Retourne (path, overwrote)."""
    path = _feedback_path(player, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    overwrote = False
    if path.exists():
        for l in path.read_text().splitlines():
            if not l.strip():
                continue
            if json.loads(l).get("ts") == fb.ts:
                overwrote = True
                continue
            kept.append(l)
    kept.append(json.dumps(fb.model_dump(), ensure_ascii=False))
    with path.open("w") as f:
        f.write("\n".join(kept) + "\n")
    return path, overwrote