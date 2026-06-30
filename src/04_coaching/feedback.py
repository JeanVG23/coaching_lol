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
from collections import Counter
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


# --- summarize / render ------------------------------------------------------

_KINDS = ("strength", "mistake", "habit", "focus")
_TREND_RECENT = 5
_LOW_SAMPLE = 10


def load_feedbacks(player: str, root=None) -> list[schema_mod.Feedback]:
    path = _feedback_path(player, root)
    if not path.exists():
        return []
    out = []
    for l in path.read_text().splitlines():
        if l.strip():
            out.append(schema_mod.Feedback.model_validate(json.loads(l)))
    return out


def summarize(fbs: list[schema_mod.Feedback]) -> dict:
    if not fbs:
        return {"n_reviews": 0}
    items = [it for fb in fbs for it in fb.items]
    n_items = len(items)
    n_useful = sum(1 for it in items if it.useful)
    by_kind: dict[str, dict] = {}
    for kind in _KINDS:
        ki = [it for it in items if it.kind == kind]
        u = sum(1 for it in ki if it.useful)
        by_kind[kind] = {"n": len(ki), "useful": u,
                         "rate": (u / len(ki)) if ki else None}
    tag_counts = Counter(it.tag for it in items if (not it.useful) and it.tag)
    by_model: dict[str, dict] = {}
    for fb in fbs:
        if not fb.items:
            continue
        m = fb.model
        bm = by_model.setdefault(m, {"n_reviews": 0, "n_items": 0, "useful": 0})
        bm["n_reviews"] += 1
        bm["n_items"] += len(fb.items)
        bm["useful"] += sum(1 for it in fb.items if it.useful)
    for m, bm in by_model.items():
        bm["rate"] = (bm["useful"] / bm["n_items"]) if bm["n_items"] else None

    ordered = sorted(fbs, key=lambda f: f.ts)
    low_sample = len(ordered) < _LOW_SAMPLE
    trend = None
    if not low_sample:
        def _rate(group):
            its = [it for fb in group for it in fb.items]
            return (sum(1 for it in its if it.useful) / len(its)) if its else None
        recent, prior = ordered[-_TREND_RECENT:], ordered[:-_TREND_RECENT]
        trend = {"recent": _rate(recent), "prior": _rate(prior)}

    return {"n_reviews": len(fbs), "n_items": n_items,
            "global_rate": (n_useful / n_items) if n_items else 0.0,
            "by_kind": by_kind, "top_tags": tag_counts.most_common(),
            "by_model": by_model, "low_sample": low_sample, "trend": trend}


def render_summary(stats: dict) -> str:
    if stats.get("n_reviews", 0) == 0:
        return "Aucune annotation — lance `feedback.py annotate` d'abord."
    lines = [f"Taux d'utilité global : {stats['global_rate']:.0%} "
             f"({stats['n_items']} items notés sur {stats['n_reviews']} reviews)",
             "\nPar section :"]
    label = {"strength": "Forces", "mistake": "Erreurs", "habit": "Habitudes",
             "focus": "Focus"}
    for kind in _KINDS:
        s = stats["by_kind"][kind]
        r = "—" if s["rate"] is None else f"{s['rate']:.0%}"
        lines.append(f"  {label[kind]:10} {r}  ({s['useful']}/{s['n']})")
    if stats["top_tags"]:
        lines.append("\nTop tags (conseils jugés faux) :")
        for tag, c in stats["top_tags"]:
            lines.append(f"  {tag:24} ×{c}")
    if stats["by_model"]:
        lines.append("\nPar modèle :")
        for m, bm in sorted(stats["by_model"].items()):
            r = "—" if bm["rate"] is None else f"{bm['rate']:.0%}"
            lines.append(f"  {m:18} {r}  ({bm['n_reviews']} reviews)")
    if stats["low_sample"]:
        lines.append(f"\nTendance : échantillon faible (<{_LOW_SAMPLE} reviews annotées), "
                     f"pas de tendance calculée.")
    elif stats["trend"]:
        t = stats["trend"]
        rp = lambda v: "—" if v is None else f"{v:.0%}"
        lines.append(f"\nTendance : 5 dernières {rp(t['recent'])} vs précédentes {rp(t['prior'])}")
    return "\n".join(lines)