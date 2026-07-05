#!/usr/bin/env python3
"""
densify_targets.py — sélectionne les joueurs à densifier "chirurgicalement" pour
atteindre le seuil MIN_PLAYER_GAMES du dataset ML per-player (cf.
src/01_data_engineering/build_player_dataset.py, seuil actuel 20 après le sweep
5/10/15/20/25/30 — AUC out-of-fold pic à 0.667-0.670 sur 20-25 games/joueur).

0 appel API : relit data/04_dataset/adc_dataset.parquet (référentiel, déjà densifié
double-ADC depuis le raw) pour compter, PAR JOUEUR, combien de games il a déjà dans
le dataset ML — pas depuis le silver (qui ne stocke qu'UNE perspective par game
collectée et sous-compte donc les joueurs vus surtout comme ADC ennemi).

Chirurgical = on ne cible QUE la bande [--min-games, --threshold[ :
- < --min-games : trop loin du seuil, chasser leur historique gaspille des appels
  API pour un gain incertain (peut-être qu'ils n'ont même pas rejoué depuis).
- >= --threshold : déjà qualifiés, rien à faire.
Le fichier de sortie trie les cibles par écart croissant (joueurs les plus proches
du seuil en premier) : {puuid: {"rank": ..., "gap": ...}}, directement consommable
par `densify_players.py --target-list`, qui traite le dict dans l'ordre d'insertion
(gains "faciles" en premier si le run est interrompu) ET s'arrête, PAR JOUEUR, dès
que `gap` games ADC neuves ont été trouvées — au lieu d'épuiser tout `--history`
pour chaque joueur (l'ancien comportement, ~3x plus lent : la collecte initiale
plafonnait à 5 games/joueur, donc quasi tout l'historique récent d'un joueur est
neuf, et sans arrêt anticipé un joueur avec un gap de 2 peut se faire fetcher 40+
games pour rien).

Sortie : data/04_dataset/densify_targets.json + résumé stderr (n cibles, games
totales à récupérer par rang, non lancé — ce script ne fait AUCUN appel réseau).

Usage :
    poetry run python3 src/collection/densify_targets.py --threshold 20 --min-games 8

Étape suivante (NON exécutée par ce script, à lancer séparément quand prêt) :
    poetry run python3 src/collection/densify_players.py \\
        --target-list data/04_dataset/densify_targets.json --history 60
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))  # ml_features, riotlib
import pandas as pd
import riotlib as rl
import ml_features as mf

DATASET_DIR = rl.DATA / "04_dataset"
DEFAULT_THRESHOLD = 20
DEFAULT_MIN_GAMES = 8
RANKS = ["diamond", "master", "grandmaster", "challenger"]


def arg(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def select_targets(df: pd.DataFrame, threshold: int, min_games: int) -> dict[str, dict]:
    """df : rows per-ADC référentiel (colonnes puuid, rank, match_id...).
    Retourne {puuid: {"rank": str, "n_games": int, "gap": int}}, trié par gap
    croissant (joueurs les plus proches du seuil en premier)."""
    targets: dict[str, dict] = {}
    for puuid, g in df.groupby("puuid"):
        n = len(g)
        if n < min_games or n >= threshold:
            continue
        targets[puuid] = {
            "rank": mf.resolve_rank(g),
            "n_games": n,
            "gap": threshold - n,
        }
    return dict(sorted(targets.items(), key=lambda kv: kv[1]["gap"]))


def main() -> int:
    threshold = int(arg("--threshold", DEFAULT_THRESHOLD))
    min_games = int(arg("--min-games", DEFAULT_MIN_GAMES))
    out_path = Path(arg("--out", str(DATASET_DIR / "densify_targets.json")))

    dataset_path = DATASET_DIR / "adc_dataset.parquet"
    if not dataset_path.exists():
        print(f"✗ {dataset_path} introuvable — lancer build_dataset.py d'abord.", file=sys.stderr)
        return 1

    df = pd.read_parquet(dataset_path)
    ref = df[df["source"] == "referentiel"].copy()
    print(f"  {len(ref)} games référentiel | {ref['puuid'].nunique()} joueurs uniques", file=sys.stderr)

    targets = select_targets(ref, threshold=threshold, min_games=min_games)

    print(f"\n  Cibles [{min_games}, {threshold}[ games : {len(targets)} joueurs", file=sys.stderr)
    total_gap = 0
    for rank in RANKS:
        rank_targets = {p: t for p, t in targets.items() if t["rank"] == rank}
        gap = sum(t["gap"] for t in rank_targets.values())
        total_gap += gap
        print(f"    {rank:<12} {len(rank_targets):>4} joueurs | {gap:>5} games manquantes (cumulé)",
              file=sys.stderr)
    print(f"    {'total':<12} {len(targets):>4} joueurs | {total_gap:>5} games manquantes (cumulé)",
          file=sys.stderr)

    if not targets:
        print("\n  ⚠ Aucune cible dans la bande — rien à écrire.", file=sys.stderr)
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {p: {"rank": t["rank"], "gap": t["gap"]} for p, t in targets.items()}, indent=2))
    print(f"\n✓ Cibles écrites dans {out_path}", file=sys.stderr)
    print(
        "\n  Étape suivante (à lancer séparément, consomme du quota API) :\n"
        f"    poetry run python3 src/collection/densify_players.py "
        f"--target-list {out_path} --history 60",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
