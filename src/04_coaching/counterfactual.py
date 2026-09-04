#!/usr/bin/env python3
"""04_coaching — tests contrefactuels : le coach LIT-IL le payload ?

L'ancrage (`grounding.py`) vérifie que les chiffres cités existent. Il ne dit pas
si le modèle les EXPLOITE : un coach qui récite un pattern plausible (« tu meurs
en BOT early, zone de mort n°1 des ADC ») peut être parfaitement ancré et
totalement insensible à la game qu'on lui donne.

Le principe : on perturbe UNE dimension du payload, on régénère, et on vérifie
que la sortie suit mécaniquement. La review déjà persistée sert de référence,
donc une perturbation coûte UN appel, pas deux.

Chaque perturbation porte son attente vérifiable :

| perturbation        | attente                                                    |
|---------------------|------------------------------------------------------------|
| `no_deaths`         | journal pauvre -> `confidence` baisse (règle 7 du prompt)   |
| `zone_to_top`       | les morts citées basculent en TOP                          |
| `unspent_gold_zero` | le gold non dépensé cité s'effondre                         |

On mesure en plus l'ancrage de la sortie perturbée sur le payload perturbé : un
modèle qui récite les chiffres de la game d'origine voit son ancrage chuter, ce
qui distingue « il a lu » de « il a deviné ».

Usage :
  python3 src/04_coaching/counterfactual.py --player spadzze --n 3 [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))  # accès src/core/
import riotlib as rl

import coach as coach_mod
import feedback as feedback_mod
import grounding
import llm_client

ZONES = ("BOT", "MID", "TOP", "JUNGLE/RIVER")
GOLD_FLOOR = 150          # en dessous, un « gold non dépensé » n'est plus un signal


# --- perturbations (pures) ----------------------------------------------------

def perturb_no_deaths(payload: dict) -> dict:
    out = copy.deepcopy(payload)
    out["journal"]["deaths"] = []
    kda = (out.get("meta") or {}).get("kda")
    if isinstance(kda, dict):
        kda["deaths"] = 0
    return out


def perturb_zone_to_top(payload: dict) -> dict:
    """Toutes les morts basculent en TOP : le journal reste coherent, seule la
    zone change.

    TOP et pas MID : sur un payload ADC, `BOT` et `MID` apparaissent
    legitimement ailleurs (role, benchmarks de lane, consequences type
    « tour BOT perdue »). Une attente formulee sur eux n'est donc pas
    falsifiable : la premiere version du test echouait a 0/2 alors que le
    modele avait bien deplace ses huit morts, ancrage 1.00. TOP n'a aucune
    raison d'etre cite pour un ADC botlane, ce qui en fait un marqueur
    discriminant."""
    out = copy.deepcopy(payload)
    for death in out["journal"]["deaths"]:
        death["zone"] = "TOP"
    return out


def perturb_unspent_gold_zero(payload: dict) -> dict:
    out = copy.deepcopy(payload)
    for death in out["journal"]["deaths"]:
        if "unspent_gold" in death:
            death["unspent_gold"] = 0
    for recall in out["journal"].get("recalls", []):
        if "gold_before" in recall:
            recall["gold_before"] = 0
    return out


# --- observations sur une review (pures) --------------------------------------

def _texts(review: dict) -> list[str]:
    out = []
    for section in ("strengths", "mistakes"):
        for insight in review.get(section) or []:
            if isinstance(insight, dict):
                out += [insight[k] for k in ("point", "cause", "evidence")
                        if isinstance(insight.get(k), str)]
    if isinstance(review.get("next_focus"), str):
        out.append(review["next_focus"])
    return out


def zones_mentioned(review: dict) -> set[str]:
    joined = " ".join(_texts(review)).upper()
    return {zone for zone in ZONES if zone in joined}


def max_gold_cited(review: dict) -> float:
    values = [value for text in _texts(review)
              for _, value, unit in grounding.cited_numbers(text) if unit == "g"]
    return max(values) if values else 0.0


def confidence(review: dict) -> float:
    value = review.get("confidence")
    return float(value) if isinstance(value, (int, float)) else 0.0


# --- attentes -----------------------------------------------------------------

def check_no_deaths(base: dict, new: dict) -> dict:
    return {"expected": "confidence en baisse (journal pauvre)",
            "observed": f"{confidence(base):.2f} -> {confidence(new):.2f}",
            "passed": confidence(new) < confidence(base)}


def check_zone_to_top(base: dict, new: dict) -> dict:
    before, after = zones_mentioned(base), zones_mentioned(new)
    return {"expected": "les zones citées suivent le journal (morts en TOP)",
            "observed": f"{sorted(before) or '∅'} -> {sorted(after) or '∅'}",
            "passed": "TOP" in after}


def check_unspent_gold_zero(base: dict, new: dict) -> dict:
    before, after = max_gold_cited(base), max_gold_cited(new)
    return {"expected": f"plus de gold non dépensé cité au-dessus de {GOLD_FLOOR} g",
            "observed": f"max cité {before:.0f} g -> {after:.0f} g",
            "passed": after < max(GOLD_FLOOR, before)}


GEN_TIMEOUT_S = 600

PERTURBATIONS = {
    "no_deaths": (perturb_no_deaths, check_no_deaths,
                  "journal vidé de ses morts"),
    "zone_to_top": (perturb_zone_to_top, check_zone_to_top,
                    "toutes les morts déplacées en TOP"),
    "unspent_gold_zero": (perturb_unspent_gold_zero, check_unspent_gold_zero,
                          "gold non dépensé mis à zéro"),
}


# --- exécution ----------------------------------------------------------------

def baselines(player: str, n: int, root=None) -> list[dict]:
    """Reviews par-game déjà persistées, plus récentes d'abord : elles servent de
    référence, ce qui divise par deux le nombre d'appels."""
    records = [r for r in feedback_mod.list_reviews(player, root)
               if r.get("kind") == "game" and (r.get("payload") or {}).get("journal")]
    return sorted(records, key=lambda r: r.get("ts") or "", reverse=True)[:n]


def run_one(record: dict, name: str, model: str, generate=None) -> dict:
    """Applique une perturbation, régénère, confronte à l'attente. `generate` est
    injectable (tests : aucun appel réseau)."""
    apply_fn, check_fn, label = PERTURBATIONS[name]
    # Ollama Cloud rend en 60-150 s sur un journal complet : le defaut 180 s de
    # llm_client expirait sur les 6 runs (mesure vide, pas verdict). On laisse
    # largement respirer, une eval n'est pas interactive.
    generate = generate or (lambda pl, m: coach_mod.generate_game_review(
        pl, m, timeout=GEN_TIMEOUT_S))
    payload = apply_fn(record["payload"])
    review, run = generate(payload, model)
    new = review.model_dump() if hasattr(review, "model_dump") else review
    verdict = check_fn(record["review"], new)
    check = grounding.check_review({"payload": payload, "review": new})
    return {"match_id": record.get("match_id"), "baseline_ts": record.get("ts"),
            "perturbation": name, "label": label, "model": model,
            **verdict,
            "grounded_rate": grounding.score(check)["grounded_rate"],
            "run": run, "review": new}


def run(player: str, n: int = 3, model: str | None = None,
        names: list[str] | None = None, root=None, generate=None) -> dict:
    model = model or coach_mod.DEFAULT_MODEL
    names = names or list(PERTURBATIONS)
    rows: list[dict] = []
    for record in baselines(player, n, root):
        for name in names:
            try:
                rows.append(run_one(record, name, model, generate))
            except (llm_client.LLMError, coach_mod.CoachValidationError) as e:
                rows.append({"match_id": record.get("match_id"),
                             "baseline_ts": record.get("ts"),
                             "perturbation": name, "model": model,
                             "error": str(e), "passed": None})
    done = [r for r in rows if r.get("passed") is not None]
    grounded = [r["grounded_rate"] for r in done
                if r.get("grounded_rate") is not None]
    by_perturbation = {}
    for name in names:
        subset = [r for r in done if r["perturbation"] == name]
        by_perturbation[name] = {
            "n": len(subset),
            "pass_rate": (sum(1 for r in subset if r["passed"]) / len(subset))
            if subset else None,
        }
    return {"player": player, "model": model,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "n_runs": len(rows), "n_errors": len(rows) - len(done),
            "sensitivity": (sum(1 for r in done if r["passed"]) / len(done))
            if done else None,
            "grounded_rate": (sum(grounded) / len(grounded)) if grounded else None,
            "by_perturbation": by_perturbation, "runs": rows}


def out_path(player: str, root=None) -> Path:
    root = Path(root) if root is not None else rl.DATA / "07_coaching"
    return root / player / "eval" / "counterfactual.json"


def persist(player: str, report: dict, root=None) -> Path:
    """Écrit HORS de reviews.jsonl : une sortie contrefactuelle n'est pas une
    review du joueur, elle ne doit ni polluer le corpus d'annotation ni le site."""
    path = out_path(player, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return path


def render(report: dict) -> str:
    pct = lambda v: "—" if v is None else f"{v:.0%}"
    lines = [f"CONTREFACTUELS — {report['player']} ({report['model']})",
             f"  Sensibilité : {pct(report['sensitivity'])} "
             f"({report['n_runs']} runs, {report['n_errors']} en échec)",
             f"  Ancrage des sorties perturbées : {pct(report['grounded_rate'])}",
             ""]
    for name, stats in report["by_perturbation"].items():
        lines.append(f"  {name:20} {pct(stats['pass_rate'])}  (n={stats['n']})")
    lines.append("")
    for row in report["runs"]:
        if row.get("error"):
            lines.append(f"  ✗ {row['match_id']} | {row['perturbation']} : {row['error']}")
            continue
        mark = "✓" if row["passed"] else "✗"
        lines.append(f"  {mark} {row['match_id']} | {row['perturbation']} : "
                     f"{row['observed']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="counterfactual.py", description=__doc__)
    ap.add_argument("--player", default="spadzze")
    ap.add_argument("--n", type=int, default=3, help="games de référence")
    ap.add_argument("--model", default=None)
    ap.add_argument("--perturbation", action="append", choices=list(PERTURBATIONS),
                    help="restreint aux perturbations nommées (répétable)")
    ap.add_argument("--dry-run", action="store_true",
                    help="liste les runs prévus, aucun appel LLM")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    names = args.perturbation or list(PERTURBATIONS)
    if args.dry_run:
        records = baselines(args.player, args.n)
        print(f"{len(records) * len(names)} appels LLM prévus "
              f"({len(records)} games × {len(names)} perturbations) :")
        for record in records:
            for name in names:
                print(f"  {record['match_id']} | {name} — {PERTURBATIONS[name][2]}")
        return 0

    report = run(args.player, args.n, args.model, names)
    path = persist(args.player, report)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json
          else render(report))
    print(f"\n✓ rapport écrit dans {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
