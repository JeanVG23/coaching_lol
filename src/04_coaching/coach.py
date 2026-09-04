#!/usr/bin/env python3
"""04_coaching — compte-rendu de coaching agrégé, narré par Ollama Cloud.

Pipeline : gold (perso + référentiel) -> payload déterministe -> prompt ->
Ollama Cloud (structured output) -> Review validée (Pydantic) -> affichage FR +
persistance data/07_coaching/<player>/reviews.jsonl.

Usage : python3 src/04_coaching/coach.py --player spadzze --scope adc \
        [--outcome loss] [--target challenger] [--model kimi-k2.6]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))  # accès src/core/
import riotlib as rl
from pydantic import ValidationError

import payload as payload_mod
import prompt as prompt_mod
import schema as schema_mod
import llm_client
import mock_llm
import feedback as feedback_mod

# Modèle par défaut retenu après A/B (cf. README.md) : kimi-k2.6 respecte le plus
# fidèlement l'asymétrie (règle 3 : profondeur/overextension = observation neutre,
# pas une faute). Surclassable via --model ou OLLAMA_MODEL (.env).
DEFAULT_MODEL = "kimi-k2.6"


class CoachValidationError(RuntimeError):
    def __init__(self, raw):
        super().__init__("sortie LLM non conforme au schéma après retry")
        self.raw = raw


def _generate(system: str, user: str, sch: dict, cls, model: str,
              prompt_version: str, timeout: int = 180):
    """Retourne (review validée, run). `run` = trace d'exécution persistée avec
    la review : sans elle on ne peut ni rejouer une génération, ni attribuer une
    variation du taux d'utilité à un changement de prompt ou de modèle.
    Les tokens/latences des tentatives rejetées par le schéma sont cumulés :
    une sortie non conforme a bien coûté un appel."""
    last_raw = None
    total = {"latency_ms": 0, "prompt_tokens": 0, "completion_tokens": 0}
    usage: dict = {}
    for attempt in range(2):                 # 1 essai + 1 retry
        gen = llm_client.generate(model, system, user, sch, timeout=timeout)
        usage = dict(gen.usage)
        for k in total:
            value = usage.get(k)
            if isinstance(value, (int, float)):
                total[k] += value
        last_raw = gen.data
        try:
            review = cls.model_validate(last_raw)
        except ValidationError:
            continue
        usage.update(total)
        usage["total_tokens"] = total["prompt_tokens"] + total["completion_tokens"]
        usage["schema_retries"] = attempt
        usage["cost_usd"] = llm_client.estimate_cost(
            model, total["prompt_tokens"], total["completion_tokens"])
        return review, {"prompt_version": prompt_version, **usage}
    raise CoachValidationError(last_raw)


def generate_review(pl: dict, model: str, timeout: int = 180):
    system, user = prompt_mod.render(pl)
    return _generate(system, user, schema_mod.review_json_schema(),
                     schema_mod.Review, model, prompt_mod.PROMPT_VERSION, timeout)


def generate_game_review(pl: dict, model: str, timeout: int = 180):
    system, user = prompt_mod.render_game(pl)
    return _generate(system, user, schema_mod.game_review_json_schema(),
                     schema_mod.GameReview, model, prompt_mod.GAME_PROMPT_VERSION,
                     timeout)


def _indexed_axis(axis: str, review: schema_mod.GameReview) -> tuple[dict, dict]:
    """Ajoute des IDs hors LLM et retourne (vue chef, index id -> insight)."""
    dumped = review.model_dump()
    lookup = {}
    chief = {"axis": axis, "label": prompt_mod.AXIS_LABELS[axis],
             "strengths": [], "mistakes": []}
    for section in ("strengths", "mistakes"):
        for index, insight in enumerate(dumped[section]):
            insight_id = f"{axis}:{section}:{index}"
            chief[section].append({"id": insight_id, **insight})
            lookup[insight_id] = (section, insight)
    return chief, lookup


def _combined_run(runs: list[dict]) -> dict:
    additive = ("prompt_tokens", "completion_tokens", "total_tokens", "schema_retries")
    out = {key: sum(float(run.get(key) or 0) for run in runs) for key in additive}
    # Les deux spécialistes tournent en parallèle ; le chef les suit. La latence
    # murale estimée n'est donc pas la somme des trois appels.
    specialist_latency = max((float(run.get("latency_ms") or 0)
                              for run in runs if run.get("stage") != "chief"), default=0)
    chief_latency = sum(float(run.get("latency_ms") or 0)
                        for run in runs if run.get("stage") == "chief")
    out["latency_ms"] = specialist_latency + chief_latency
    costs = [float(run["cost_usd"]) for run in runs
             if isinstance(run.get("cost_usd"), (int, float))]
    out["cost_usd"] = sum(costs) if costs else None
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "schema_retries"):
        out[key] = int(out[key])
    out["prompt_version"] = prompt_mod.SPECIALIZED_PROMPT_VERSION
    out["stages"] = runs
    return out


def generate_specialized_game_review(pl: dict, model: str, timeout: int = 180):
    """Deux sous-agents en parallèle, puis un chef qui ne choisit que leurs IDs."""
    axes = tuple(prompt_mod.SPECIALIST_SYSTEMS)

    def generate_axis(axis: str):
        system, user = prompt_mod.render_specialist(pl, axis)
        review, run = _generate(system, user, schema_mod.game_review_json_schema(),
                                schema_mod.GameReview, model,
                                prompt_mod.version_of(system), timeout)
        return axis, review, run

    with ThreadPoolExecutor(max_workers=len(axes)) as pool:
        generated = list(pool.map(generate_axis, axes))

    chief_axes, lookup, axis_models, runs = [], {}, [], []
    for axis, review, run in generated:
        indexed, axis_lookup = _indexed_axis(axis, review)
        chief_axes.append(indexed)
        lookup.update(axis_lookup)
        axis_models.append(schema_mod.AxisReview(
            axis=axis, label=prompt_mod.AXIS_LABELS[axis], **review.model_dump()))
        runs.append({"stage": axis, **run})

    mistake_ids = [key for key, (kind, _) in lookup.items() if kind == "mistakes"]
    strength_ids = [key for key, (kind, _) in lookup.items() if kind == "strengths"]
    system, user = prompt_mod.render_chief(chief_axes)
    chief, chief_run = _generate(
        system, user, schema_mod.chief_selection_json_schema(mistake_ids, strength_ids),
        schema_mod.ChiefSelection, model, prompt_mod.version_of(system), timeout)
    runs.append({"stage": "chief", **chief_run})

    priority_ids = list(dict.fromkeys(chief.priority_mistake_ids))
    strength_selection = list(dict.fromkeys(chief.strength_insight_ids))
    valid = (chief.summary_insight_id in priority_ids
             and chief.next_focus_insight_id in priority_ids
             and all(key in mistake_ids for key in priority_ids)
             and all(key in strength_ids for key in strength_selection))
    if not valid:
        raise CoachValidationError(chief.model_dump())

    def insight(key: str) -> schema_mod.GameInsight:
        return schema_mod.GameInsight.model_validate(lookup[key][1])

    final = schema_mod.SpecializedGameReview(
        summary=insight(chief.summary_insight_id).point,
        strengths=[insight(key) for key in strength_selection],
        mistakes=[insight(key) for key in priority_ids],
        next_focus=insight(chief.next_focus_insight_id).point,
        confidence=chief.confidence,
        axes=axis_models,
    )
    return final, _combined_run(runs)


def pending_game_matches(records: list[dict], reviews: list[dict],
                         scope: str, n: int) -> list[str]:
    """Match_ids du scope sans review kind=game (dédup quel que soit le modèle),
    plus récents d'abord. Le silver perso antérieur au 2026-07-06 n'a pas de
    game_ts -> ces records (ts traité comme 0) retombent sur l'ordre
    d'apparition inversé du fichier, approximation de l'ordre de collecte."""
    reviewed = {r.get("match_id") for r in reviews if r.get("kind") == "game"}
    indexed = list(enumerate(payload_mod.filter_scope(records, scope)))
    indexed.sort(key=lambda p: (p[1].get("game_ts") or 0, p[0]), reverse=True)
    out: list[str] = []
    for _, rec in indexed:
        mid = rec.get("match_id")
        if not mid or mid in reviewed or mid in out:
            continue
        out.append(mid)
        if len(out) >= n:
            break
    return out


def run_batch(player: str, scope: str, target: str, model: str, n: int,
              root=None, silver_dir=None, specialized: bool = False) -> int:
    """Génère jusqu'à n reviews par-game sur les games du scope pas encore
    reviewées (kind=game). Continue sur échec d'une game ; bilan final."""
    silver = Path(silver_dir) if silver_dir is not None else rl.silver_dir()
    records = payload_mod._personal_records(player, silver)
    reviews = feedback_mod.list_reviews(player, root)
    already = len({r.get("match_id") for r in reviews if r.get("kind") == "game"}
                  & {r.get("match_id") for r in payload_mod.filter_scope(records, scope)})
    pending = pending_game_matches(records, reviews, scope, n)
    # Agrégat référentiel lu une fois pour tout le lot (identique à chaque game).
    # S'il manque, on laisse `build_game` échouer game par game (bilan inchangé).
    try:
        ref = payload_mod._load(rl.gold_dir(), rl.KIND_REF, target, scope) if pending else None
    except FileNotFoundError:
        ref = None
    done, failed = 0, 0
    seen_ts: set[str] = set()
    for mid in pending:
        ts = datetime.now().isoformat(timespec="seconds")
        if ts in seen_ts:                        # 2 games dans la même seconde
            ts = datetime.now().isoformat()      # (mocks/tests) -> microsecondes
        seen_ts.add(ts)
        try:
            pl = payload_mod.build_game(player, match_id=mid, scope=scope,
                                        target=target, silver_dir=silver,
                                        records=records, ref=ref)
            generate = generate_specialized_game_review if specialized else generate_game_review
            review, run = generate(pl, model)
        except FileNotFoundError as e:
            print(f"✗ {mid} : {e}", file=sys.stderr)
            failed += 1
            continue
        except llm_client.LLMError as e:
            print(f"✗ {mid} : appel LLM échoué : {e}", file=sys.stderr)
            failed += 1
            continue
        except CoachValidationError as e:
            p = _save_failed(player, ts, e.raw, root=root)
            print(f"✗ {mid} : {e} — brut sauvé dans {p}", file=sys.stderr)
            failed += 1
            continue
        persist(player, model, pl, review, ts, root=root, run=run)
        m = pl["meta"]
        issue = "victoire" if m.get("win") else "défaite"
        print(f"✓ {mid} : {m.get('champion')} ({issue}) reviewée")
        done += 1
    s = lambda k: "s" if k > 1 else ""
    print(f"\nBilan : {done} générée{s(done)} · {already} déjà reviewée{s(already)} "
          f"· {failed} échouée{s(failed)}")
    return 1 if (pending and done == 0) else 0


def persist(player: str, model: str, pl: dict, review, ts: str,
            root=None, run: dict | None = None) -> Path:
    root = Path(root) if root is not None else rl.DATA / "07_coaching"
    out = root / player
    out.mkdir(parents=True, exist_ok=True)
    meta = pl["meta"]
    record = {"ts": ts, "model": model,
              "scope": meta["scope"], "target": meta["target"],
              "run": run or {},          # version de prompt, latence, tokens, coût
              "payload": pl, "review": review.model_dump()}
    if meta.get("kind") == "game":           # review par-game (GameReview)
        record["kind"] = "game"
        record["match_id"] = meta["match_id"]
    else:                                    # review agrégée (Review)
        record["outcome_focus"] = meta["outcome_focus"]
    path = out / "reviews.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def render_game_text(review: schema_mod.GameReview) -> str:
    lines = [f"\n  Confiance : {review.confidence:.0%}"]
    if review.strengths:
        lines.append("\n  Forces :")
        lines += [f"    + {i.point}  — pourquoi : {i.cause}  ({i.evidence})"
                  for i in review.strengths]
    lines.append("\n  Erreurs prioritaires :")
    lines += [f"    - {i.point}  — pourquoi : {i.cause}  ({i.evidence})"
              for i in review.mistakes]
    lines.append(f"\n  Focus prochaine game : {review.next_focus}")
    return "\n".join(lines)


def render_text(review: schema_mod.Review) -> str:
    lines = [f"\n  Confiance : {review.confidence:.0%}", "\n  Forces :"]
    lines += [f"    + {i.point}  ({i.evidence})" for i in review.strengths]
    lines.append("\n  Erreurs prioritaires :")
    lines += [f"    - {i.point}  ({i.evidence})" for i in review.mistakes]
    lines.append("\n  Habitudes à corriger :")
    lines += [f"    • {h}" for h in review.habits]
    lines.append(f"\n  Focus prochaine game : {review.next_focus}")
    return "\n".join(lines)


def render_run(run: dict) -> str:
    """Ligne de trace affichée après la génération (mêmes champs que persistés)."""
    ms, tok = run.get("latency_ms"), run.get("total_tokens")
    cost = run.get("cost_usd")
    parts = [f"prompt {run.get('prompt_version', '?')}"]
    if ms is not None:
        parts.append(f"{ms / 1000:.1f} s")
    if tok is not None:
        parts.append(f"{tok} tokens")
    if run.get("schema_retries"):
        parts.append(f"{run['schema_retries']} retry schéma")
    if cost is not None:
        parts.append(f"${cost:.4f}")
    return "\n  Run : " + " · ".join(parts)


def _save_failed(player: str, ts: str, raw, root=None) -> Path:
    root = Path(root) if root is not None else rl.DATA / "07_coaching"
    out = root / player / "failed"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{ts.replace(':', '-')}.json"
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2))
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", default="spadzze")
    ap.add_argument("--scope", default="adc")
    ap.add_argument("--outcome", default="loss", choices=["overall", "win", "loss"])
    ap.add_argument("--target", default="challenger")
    ap.add_argument("--model", default=None)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--game", nargs="?", const="latest", default=None,
                     metavar="MATCH_ID",
                     help="review par-game : dernière game du scope, ou un match_id")
    grp.add_argument("--game-batch", type=int, nargs="?", const=10, default=None,
                     metavar="N",
                     help="génère les reviews par-game des N dernières games "
                          "du scope pas encore reviewées (défaut 10)")
    ap.add_argument("--mock-llm", action="store_true",
                    help="remplace l'appel Ollama par un générateur déterministe "
                         "(make demo : 0 réseau, 0 clé)")
    ap.add_argument("--specialized", action="store_true",
                    help="2 sous-agents (morts/positionnement + économie/build) "
                         "puis un chef ; 3 appels LLM par game")
    args = ap.parse_args()
    if args.mock_llm:
        # Substitution au point d'entrée réseau : tout le reste du chemin
        # (schéma, validation, télémétrie, persistance) reste celui de production.
        llm_client.generate = mock_llm.generate
        args.model = mock_llm.MODEL_NAME
        print("⚠ --mock-llm : sortie fabriquée à partir du payload, "
              "ce n'est pas du coaching.\n")
    # Résolution modèle : --model CLI > OLLAMA_MODEL (shell env) > .env > défaut.
    # load_env() ne peuple pas os.environ, donc on lit .env explicitement ici.
    if args.model is None:
        args.model = (os.environ.get("OLLAMA_MODEL")
                      or rl.load_env().get("OLLAMA_MODEL", DEFAULT_MODEL))

    if args.game_batch is not None:
        return run_batch(args.player, args.scope, args.target,
                         args.model, args.game_batch, specialized=args.specialized)

    ts = datetime.now().isoformat(timespec="seconds")
    per_game = args.game is not None
    try:
        if per_game:
            mid = None if args.game == "latest" else args.game
            pl = payload_mod.build_game(args.player, match_id=mid,
                                        scope=args.scope, target=args.target)
        else:
            pl = payload_mod.build(
                args.player, args.scope, args.target, args.outcome,
                game_reviews=feedback_mod.list_reviews(args.player),
            )
    except FileNotFoundError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    try:
        generate = (generate_specialized_game_review if per_game and args.specialized
                    else generate_game_review if per_game else generate_review)
        review, run = generate(pl, args.model)
    except llm_client.LLMError as e:
        print(f"✗ appel LLM échoué : {e}", file=sys.stderr)
        return 1
    except CoachValidationError as e:
        p = _save_failed(args.player, ts, e.raw)
        print(f"✗ {e} — brut sauvé dans {p}", file=sys.stderr)
        return 1

    path = persist(args.player, args.model, pl, review, ts, run=run)
    if per_game:
        m = pl["meta"]
        issue = "victoire" if m["win"] else "défaite"
        print(f"COACHING GAME — {m['match_id']} : {m['champion']} vs "
              f"{m.get('opponent') or '?'} ({issue}, {m['duration_min']} min, "
              f"vs {args.target}) [modèle {args.model}]")
        print(render_game_text(review))
    else:
        print(f"COACHING — {args.player} ({args.scope}, issue={args.outcome}, "
              f"vs {args.target}) [modèle {args.model}]")
        print(render_text(review))
    print(render_run(run))
    print(f"\n✓ review persistée dans {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
