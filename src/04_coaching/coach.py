#!/usr/bin/env python3
"""04_coaching — compte-rendu de coaching agrégé, narré par Ollama Cloud.

Pipeline : gold (perso + référentiel) -> payload déterministe -> prompt ->
Ollama Cloud (structured output) -> Review validée (Pydantic) -> affichage FR +
persistance data/07_coaching/<player>/reviews.jsonl.

Usage : python3 src/04_coaching/coach.py --player spadzze --scope adc \
        [--outcome loss] [--target challenger] [--model deepseek-v4-pro]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # accès src/
import riotlib as rl
from pydantic import ValidationError

import payload as payload_mod
import prompt as prompt_mod
import schema as schema_mod
import llm_client

DEFAULT_MODEL = "deepseek-v4-pro"


class CoachValidationError(RuntimeError):
    def __init__(self, raw):
        super().__init__("sortie LLM non conforme au schéma après retry")
        self.raw = raw


def generate_review(pl: dict, model: str) -> schema_mod.Review:
    system, user = prompt_mod.render(pl)
    sch = schema_mod.review_json_schema()
    last_raw = None
    for _ in range(2):                       # 1 essai + 1 retry
        last_raw = llm_client.generate_json(model, system, user, sch)
        try:
            return schema_mod.Review.model_validate(last_raw)
        except ValidationError:
            continue
    raise CoachValidationError(last_raw)


def persist(player: str, model: str, pl: dict, review: schema_mod.Review,
            ts: str, root=None) -> Path:
    root = Path(root) if root is not None else rl.DATA / "07_coaching"
    out = root / player
    out.mkdir(parents=True, exist_ok=True)
    record = {"ts": ts, "model": model,
              "scope": pl["meta"]["scope"], "target": pl["meta"]["target"],
              "outcome_focus": pl["meta"]["outcome_focus"],
              "payload": pl, "review": review.model_dump()}
    path = out / "reviews.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def render_text(review: schema_mod.Review) -> str:
    lines = [f"\n  Confiance : {review.confidence:.0%}", "\n  Forces :"]
    lines += [f"    + {i.point}  ({i.evidence})" for i in review.strengths]
    lines.append("\n  Erreurs prioritaires :")
    lines += [f"    - {i.point}  ({i.evidence})" for i in review.mistakes]
    lines.append("\n  Habitudes à corriger :")
    lines += [f"    • {h}" for h in review.habits]
    lines.append(f"\n  Focus prochaine game : {review.next_focus}")
    return "\n".join(lines)


def _save_failed(player: str, ts: str, raw) -> Path:
    out = rl.DATA / "07_coaching" / player / "failed"
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
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL))
    args = ap.parse_args()

    ts = datetime.now().isoformat(timespec="seconds")
    try:
        pl = payload_mod.build(args.player, args.scope, args.target, args.outcome)
    except FileNotFoundError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    try:
        review = generate_review(pl, args.model)
    except llm_client.LLMError as e:
        print(f"✗ appel LLM échoué : {e}", file=sys.stderr)
        return 1
    except CoachValidationError as e:
        p = _save_failed(args.player, ts, e.raw)
        print(f"✗ {e} — brut sauvé dans {p}", file=sys.stderr)
        return 1

    path = persist(args.player, args.model, pl, review, ts)
    print(f"COACHING — {args.player} ({args.scope}, issue={args.outcome}, "
          f"vs {args.target}) [modèle {args.model}]")
    print(render_text(review))
    print(f"\n✓ review persistée dans {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())