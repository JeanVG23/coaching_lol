#!/usr/bin/env python3
"""A/B automatique de modèles : ancrage, sensibilité, latence et retries.

Chaque modèle regénère sa propre baseline avant les perturbations. Comparer GLM à
une baseline Kimi fausserait notamment le test de confiance ``no_deaths``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import riotlib as rl

import coach
import counterfactual
import grounding
import llm_client

DEFAULT_MODELS = ("kimi-k2.6", "glm-5.3")


def _dump(review) -> dict:
    return review.model_dump() if hasattr(review, "model_dump") else dict(review)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _model_run(model: str, records: list[dict], names: list[str], generate) -> dict:
    rows, errors = [], []
    for source in records:
        try:
            base_review, base_run = generate(source["payload"], model)
        except (llm_client.LLMError, coach.CoachValidationError) as error:
            errors.append({"match_id": source.get("match_id"), "stage": "baseline",
                           "error": str(error)})
            continue
        base = _dump(base_review)
        base_score = grounding.score(grounding.check_review({
            "payload": source["payload"], "review": base,
        }))
        row = {"match_id": source.get("match_id"), "baseline": base_score,
               "runs": [{"stage": "baseline", **base_run}]}
        fresh = {**source, "review": base}
        for name in names:
            try:
                result = counterfactual.run_one(fresh, name, model, generate=generate)
                row["runs"].append({"stage": name, **result.get("run", {})})
                row.setdefault("counterfactuals", []).append({
                    key: result.get(key) for key in
                    ("perturbation", "passed", "grounded_rate", "observed")
                })
            except (llm_client.LLMError, coach.CoachValidationError) as error:
                errors.append({"match_id": source.get("match_id"), "stage": name,
                               "error": str(error)})
        rows.append(row)

    generated_runs = [run for row in rows for run in row["runs"]]
    counterfactuals = [item for row in rows for item in row.get("counterfactuals", [])]
    decided = [item for item in counterfactuals if item.get("passed") is not None]
    grounded_cf = [float(item["grounded_rate"]) for item in decided
                   if item.get("grounded_rate") is not None]
    grounded_base = [float(row["baseline"]["grounded_rate"]) for row in rows
                     if row["baseline"].get("grounded_rate") is not None]
    return {
        "model": model,
        "n_games": len(rows),
        "n_errors": len(errors),
        "baseline_grounded_rate": _mean(grounded_base),
        "sensitivity": (sum(1 for item in decided if item["passed"]) / len(decided)
                        if decided else None),
        "counterfactual_grounded_rate": _mean(grounded_cf),
        "latency_ms_total": sum(float(run.get("latency_ms") or 0) for run in generated_runs),
        "schema_retries": sum(int(run.get("schema_retries") or 0) for run in generated_runs),
        "total_tokens": sum(int(run.get("total_tokens") or 0) for run in generated_runs),
        "errors": errors,
        "games": rows,
    }


def _rank_key(result: dict) -> tuple:
    """Qualité d'abord ; latence ne départage qu'à qualité égale."""
    return (
        -(result.get("n_errors") or 0),
        result.get("sensitivity") if result.get("sensitivity") is not None else -1,
        result.get("baseline_grounded_rate")
        if result.get("baseline_grounded_rate") is not None else -1,
        result.get("counterfactual_grounded_rate")
        if result.get("counterfactual_grounded_rate") is not None else -1,
        -(result.get("schema_retries") or 0),
        -(result.get("latency_ms_total") or 0),
    )


def run(player: str, models: list[str] | None = None, n: int = 3,
        names: list[str] | None = None, root=None, generate=None) -> dict:
    models = models or list(DEFAULT_MODELS)
    names = names or list(counterfactual.PERTURBATIONS)
    records = counterfactual.baselines(player, n, root)
    generate = generate or (lambda payload, model: coach.generate_game_review(
        payload, model, timeout=counterfactual.GEN_TIMEOUT_S))
    results = [_model_run(model, records, names, generate) for model in models]
    ranked = sorted(results, key=_rank_key, reverse=True)
    return {
        "player": player,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {"n_source_games": len(records), "perturbations": names,
                     "calls_planned": len(records) * (1 + len(names)) * len(models)},
        "recommended_model": ranked[0]["model"] if ranked and records else None,
        "models": results,
    }


def out_path(player: str, root=None) -> Path:
    base = Path(root) if root is not None else rl.DATA / "07_coaching"
    return base / player / "eval" / "model_ab.json"


def persist(player: str, report: dict, root=None) -> Path:
    path = out_path(player, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return path


def render(report: dict) -> str:
    pct = lambda value: "—" if value is None else f"{value:.0%}"
    lines = [f"A/B MODÈLES — {report['player']}",
             f"  Recommandation automatique : {report['recommended_model'] or '—'}", ""]
    for result in report["models"]:
        lines.append(
            f"  {result['model']}: sensibilité {pct(result['sensitivity'])}, "
            f"ancrage {pct(result['baseline_grounded_rate'])}, "
            f"{result['n_errors']} erreur(s), {result['schema_retries']} retry(s), "
            f"{result['latency_ms_total'] / 1000:.1f} s cumulées")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model_ab.py", description=__doc__)
    parser.add_argument("--player", default="spadzze")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    models = args.models or list(DEFAULT_MODELS)
    records = counterfactual.baselines(args.player, args.n)
    calls = len(records) * (1 + len(counterfactual.PERTURBATIONS)) * len(models)
    if args.dry_run:
        print(f"{calls} appels LLM prévus : {len(records)} games × "
              f"{1 + len(counterfactual.PERTURBATIONS)} variantes × {len(models)} modèles")
        return 0
    report = run(args.player, models, args.n)
    path = persist(args.player, report)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render(report))
    print(f"\n✓ rapport écrit dans {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
