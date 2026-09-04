#!/usr/bin/env python3
"""04_coaching — boucle d'évaluation : annotation + agrégation des reviews.

Sous-commandes :
  annotate  : choisir une review persistée, juger chaque insight (utile/faux + tag),
              persister dans data/07_coaching/<player>/feedback.jsonl.
  summary   : agréger les annotations (taux utile, par section, top tags, par
              modèle, tendance) — chaque top tag affiche jusqu'à 2 verbatims de
              notes libres associées, pour guider la correction du prompt/des
              features sans deviner ce qui cloche derrière un tag. Aucun appel réseau.

Usage :
  python3 src/04_coaching/feedback.py annotate --player spadzze [--ts <ts> | --last]
  python3 src/04_coaching/feedback.py summary  --player spadzze [--tag <t>] [--model <m>]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))  # accès src/core/
import riotlib as rl

import schema as schema_mod


def _reviews_path(player: str, root=None) -> Path:
    root = Path(root) if root is not None else rl.DATA / "07_coaching"
    return root / player / "reviews.jsonl"


def _feedback_path(player: str, root=None) -> Path:
    root = Path(root) if root is not None else rl.DATA / "07_coaching"
    return root / player / "feedback.jsonl"


def list_reviews(player: str, root=None) -> list[dict]:
    return rl.read_jsonl(_reviews_path(player, root))


def load_review(player: str, ts: str, root=None) -> dict | None:
    for r in list_reviews(player, root):
        if r.get("ts") == ts:
            return r
    return None


def build_feedback(review, ts: str, player: str, model: str,
                   rated_at: str,
                   responses: dict[tuple[str, int], tuple[bool, str | None, str | None]]
                   ) -> schema_mod.Feedback:
    """Construit un Feedback en n'incluant que les items présents dans responses
    (skip = item omis). Invariant tag-requis validé par FeedbackItem."""
    sections = [("strength", review.strengths), ("mistake", review.mistakes),
                ("habit", getattr(review, "habits", []))]   # GameReview : pas de habits
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
_NOTES_PER_TAG = 2
_OBJECTIVE_N = 10          # métrique projet : >=70 % de mistakes utiles
_OBJECTIVE_RATE = 0.70     # sur >=10 reviews par-game annotées


def load_feedbacks(player: str, root=None) -> list[schema_mod.Feedback]:
    return [schema_mod.Feedback.model_validate(row)
            for row in rl.read_jsonl(_feedback_path(player, root))]


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
    tag_notes: dict[str, list[str]] = {}
    for it in items:
        if (not it.useful) and it.tag and it.note:
            tag_notes.setdefault(it.tag, []).append(it.note)
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
            "tag_notes": tag_notes,
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
            for note in stats["tag_notes"].get(tag, [])[:_NOTES_PER_TAG]:
                lines.append(f"      » {note!r}")
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


def objective_stats(fbs: list[schema_mod.Feedback],
                    reviews: list[dict]) -> dict:
    """Métrique de succès par-game. Le feedback ne stocke pas le kind ->
    jointure ts avec reviews.jsonl ; seules les mistakes des reviews kind=game
    comptent (définition de la métrique)."""
    game_ts = {r.get("ts") for r in reviews if r.get("kind") == "game"}
    game_fbs = [f for f in fbs if f.ts in game_ts]
    mistakes = [it for f in game_fbs for it in f.items if it.kind == "mistake"]
    useful = sum(1 for it in mistakes if it.useful)
    return {"n_game_reviews_annotated": len(game_fbs),
            "target_n": _OBJECTIVE_N,
            "mistake_useful_rate": (useful / len(mistakes)) if mistakes else None,
            "target_rate": _OBJECTIVE_RATE}


def render_objective(obj: dict) -> str:
    rate = ("—" if obj["mistake_useful_rate"] is None
            else f"{obj['mistake_useful_rate']:.0%}")
    return (f"Objectif par-game : {obj['n_game_reviews_annotated']}"
            f"/{obj['target_n']} reviews annotées · mistakes utiles {rate} "
            f"(cible ≥{obj['target_rate']:.0%})")


def pending_reviews(reviews: list[dict],
                    fbs: list[schema_mod.Feedback]) -> list[dict]:
    """Reviews sans feedback (jointure par ts), plus anciennes d'abord.
    Une review entièrement skippée n'est jamais persistée -> reste pending."""
    done = {f.ts for f in fbs}
    return sorted((r for r in reviews if r.get("ts") not in done),
                  key=lambda r: r.get("ts") or "")


# --- annotate (flow interactif) + main() ------------------------------------

def _display_items(review) -> list[tuple[str, int, str]]:
    """Retourne [(kind, index, ligne_affichage)] pour les items (ordre fixe)."""
    out = []
    for i, ins in enumerate(review.strengths):
        cause = getattr(ins, "cause", None)
        cause_txt = f"  — pourquoi : {cause}" if cause else ""
        out.append(("strength", i, f"Force  {i}: {ins.point}{cause_txt}  ({ins.evidence})"))
    for i, ins in enumerate(review.mistakes):
        cause = getattr(ins, "cause", None)
        cause_txt = f"  — pourquoi : {cause}" if cause else ""
        out.append(("mistake", i, f"Erreur {i}: {ins.point}{cause_txt}  ({ins.evidence})"))
    for i, h in enumerate(getattr(review, "habits", [])):
        out.append(("habit", i, f"Habitude {i}: {h}"))
    out.append(("focus", 0, f"Focus : {review.next_focus}"))
    return out


def _prompt_useful(prompt, label_line) -> str | None:
    """Retourne 'y'/'n'/'s' (Entrée vide = skip = None)."""
    while True:
        ans = prompt(f"{label_line}\n  utile ? [y/n/s] : ").strip().lower()
        if ans in ("y", "n", "s", ""):
            return None if ans == "" else ans


def _annotate_one(chosen: dict, player: str, root, prompt) -> int:
    """Annote UNE review déjà sélectionnée ; persiste si >= 1 item noté."""
    cls = (schema_mod.GameReview if chosen.get("kind") == "game"
           else schema_mod.Review)
    review = cls.model_validate(chosen["review"])
    items = _display_items(review)
    responses: dict[tuple[str, int], tuple[bool, str | None, str | None]] = {}
    for kind, idx, line in items:
        ans = _prompt_useful(prompt, line)
        if ans is None or ans == "s":
            continue
        useful = (ans == "y")
        tag = None
        note = None
        if not useful:
            menu = "\n".join(f"  {j+1} | {t}" for j, t in enumerate(schema_mod.NEG_TAGS))
            while tag is None:
                ts_in = prompt(f"Pourquoi ? (numéro)\n{menu}\n  tag : ").strip()
                try:
                    tag = schema_mod.NEG_TAGS[int(ts_in) - 1]
                except (ValueError, IndexError):
                    pass
            note = prompt("Note (optionnel, Entrée = skip) : ").strip() or None
        responses[(kind, idx)] = (useful, tag, note)
    if not responses:
        print("Tout skippé — rien à persister.")
        return 0
    rated_at = datetime.now().isoformat(timespec="seconds")
    fb = build_feedback(review, ts=chosen["ts"], player=player,
                        model=chosen["model"], rated_at=rated_at, responses=responses)
    path, overwrote = persist_feedback(player, fb, root)
    n_useful = sum(1 for it in fb.items if it.useful)
    print(f"\n{len(fb.items)} items notés ({n_useful} utiles).")
    if overwrote:
        print(f"réannotation: écrase feedback précédent pour {fb.ts}")
    print(f"✓ feedback persisté dans {path}")
    return 0


def annotate(player: str, ts: str | None = None, last: bool = False,
             pending: bool = False, root=None, prompt=None) -> int:
    if prompt is None:
        prompt = input           # late binding : monkeypatch builtins.input pris en compte
    reviews = list_reviews(player, root)
    if not reviews:
        print("Aucune review pour ce joueur — génère-en via coach.py d'abord.")
        return 0
    if pending:
        pend = pending_reviews(reviews, load_feedbacks(player, root))
        if not pend:
            print("Aucune review en attente d'annotation.")
            return 0
        for i, r in enumerate(pend, 1):
            kind = "game" if r.get("kind") == "game" else "agrégée"
            what = r.get("match_id") or r.get("outcome_focus") or "?"
            ans = prompt(f"\n({i}/{len(pend)}) [{kind}] {r['ts']} | {r['model']} "
                         f"| {what}\n  [Entrée=annoter / n=passer / q=quitter] : "
                         ).strip().lower()
            if ans == "q":
                break
            if ans == "n":
                continue
            _annotate_one(r, player, root, prompt)
        return 0
    if ts is None:
        if last:
            chosen = reviews[-1]
        else:
            print("Reviews disponibles :")
            for i, r in enumerate(reviews, 1):
                what = r.get("outcome_focus") or r.get("match_id") or "?"
                print(f"  {i} | {r['ts']} | {r['model']} | {what}")
            sel = prompt("Choisis une review (numéro) : ").strip()
            try:
                chosen = reviews[int(sel) - 1]
            except (ValueError, IndexError):
                print("✗ sélection invalide")
                return 1
    else:
        chosen = next((r for r in reviews if r.get("ts") == ts), None)
        if chosen is None:
            print(f"✗ ts {ts} introuvable dans reviews.jsonl")
            return 1
    return _annotate_one(chosen, player, root, prompt)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="feedback.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("annotate", help="juger les insights d'une review")
    a.add_argument("--player", default="spadzze")
    sel = a.add_mutually_exclusive_group()
    sel.add_argument("--ts", default=None)
    sel.add_argument("--last", action="store_true")
    sel.add_argument("--pending", action="store_true",
                     help="annoter en série toutes les reviews sans feedback")

    s = sub.add_parser("summary", help="agrège les annotations")
    s.add_argument("--player", default="spadzze")
    s.add_argument("--tag", default=None, help="filtre par tag")
    s.add_argument("--model", default=None, help="filtre par modèle")

    args = ap.parse_args(argv)
    if args.cmd == "annotate":
        return annotate(args.player, ts=args.ts, last=args.last,
                        pending=args.pending)
    if args.cmd == "summary":
        fbs = load_feedbacks(args.player)
        objective = objective_stats(fbs, list_reviews(args.player))
        if args.model:
            fbs = [f for f in fbs if f.model == args.model]
        if args.tag:
            fbs = [f for f in fbs if any(it.tag == args.tag for it in f.items)]
        print(f"FEEDBACK — {args.player}")
        print(render_objective(objective))
        print(render_summary(summarize(fbs)))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
