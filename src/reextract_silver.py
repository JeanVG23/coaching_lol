"""Ré-extrait le silver depuis le raw caché — 0 appel API.

À lancer après toute évolution de extract_game (ici : ajout de comp). Lit les
fichiers raw (data/01_raw/<matchId>_match.json + _timeline.json) pour chaque
(match_id, puuid) déjà présent dans le silver, rejoue extract_game, réécrit.
"""
from __future__ import annotations

import json
import sys

import riotlib as rl


def reextract_one(match: dict, timeline: dict, puuid: str, rank):
    return rl.extract_game(match, timeline, puuid, rank=rank)


def _load_raw(match_id: str):
    m = rl.RAW_DIR / f"{match_id}_match.json"
    t = rl.RAW_DIR / f"{match_id}_timeline.json"
    if not m.exists() or not t.exists():
        return None
    try:
        return json.loads(m.read_text()), json.loads(t.read_text())
    except json.JSONDecodeError:
        return None


def _reextract_dir(d, rank):
    games = rl.read_jsonl(d / "games.jsonl")
    if not games:
        return 0, 0
    out, miss = [], 0
    for g in games:
        raw = _load_raw(g["match_id"])
        if not raw:
            out.append(g)  # garde l'ancien record si raw absent
            miss += 1
            continue
        new = rl.extract_game(raw[0], raw[1], g["puuid"], rank=g.get("rank"))
        out.append(new if new else g)
    rl.write_jsonl(d / "games.jsonl", out)
    return len(out), miss


def main() -> int:
    total, missing = 0, 0
    for kind, root in (("referentiel", rl.SILVER_DIR / "referentiel"),
                       ("personal", rl.SILVER_DIR / "personal")):
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            n, miss = _reextract_dir(d, d.name)
            total += n
            missing += miss
            print(f"  {kind}/{d.name}: {n} games re-extraits ({miss} raw manquants)")
    print(f"\n{total} games re-extraits ({missing} sans raw -> record conserve).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
