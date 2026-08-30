#!/usr/bin/env python3
"""
densify_targets.py — sélectionne les joueurs à densifier "chirurgicalement" pour
atteindre le sweet spot ~30 games/joueur du dataset ML per-player (cf.
src/02_data_science/analyze_auc_vs_ngames.py, Courbe 2 pool fixe : N=15→0.588,
N=30→0.635 peak, puis plateau/déclin ; prod qualify=15 cap=all = 0.635 ≈ le peak,
déjà au plateau — N n'est PAS le levier du plafond ~0.65 master/GM).

0 appel API : relit data/04_dataset/adc_dataset.parquet (référentiel, déjà densifié
double-ADC depuis le raw) pour compter, PAR JOUEUR, combien de games il a déjà dans
le dataset ML — pas depuis le silver (qui ne stocke qu'UNE perspective par game
collectée et sous-compte donc les joueurs vus surtout comme ADC ennemi).

Chirurgical = on ne cible QUE la bande [--min-games, --threshold[ :
- < --min-games : trop loin du seuil, chasser leur historique gaspille des appels
  API pour un gain incertain (peut-être qu'ils n'ont même pas rejoué depuis).
- >= --threshold : déjà qualifiés, rien à faire.
--exclude-ranks : rangs à exclure de la sélection (ex: "diamond"). La frontière
  réellement apprise = challenger vs master (high ≈ 81% challenger, low ≈ 73%
  master, cf. dataset_report.py) ; densifier diamond pousse la classe low vers
  diamond, plus loin du boundary master/GM = bruit au lieu de sharper la frontière.
  Cf. densify_sweet_spot.py qui bake-in ce filtre + le sweet spot 30.
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
    poetry run python3 src/collection/densify_targets.py --exclude-ranks diamond

Étape suivante (NON exécutée par ce script, à lancer séparément quand prêt) :
    poetry run python3 src/collection/densify_players.py \\
        --target-list data/04_dataset/densify_targets.json --history 60

Ou en une commande (sélection + scraping) via l'orchestrateur :
    poetry run python3 src/collection/densify_sweet_spot.py --run --history 60
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
DEFAULT_THRESHOLD = 30
DEFAULT_MIN_GAMES = 15
RANKS = ["diamond", "master", "grandmaster", "challenger"]


def arg(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def select_targets(df: pd.DataFrame, threshold: int, min_games: int,
                   exclude_ranks: set[str] | None = None) -> dict[str, dict]:
    """df : rows per-ADC référentiel (colonnes puuid, rank, match_id...).
    Retourne {puuid: {"rank": str, "n_games": int, "gap": int}}, trié par gap
    croissant (joueurs les plus proches du seuil en premier).
    exclude_ranks : rangs à exclure (ex: {"diamond"} — densifier diamond pousse la
    classe low loin du boundary master/GM = bruit, cf. analyze_auc_vs_ngames.py)."""
    targets: dict[str, dict] = {}
    exclude_ranks = exclude_ranks or set()
    for puuid, g in df.groupby("puuid"):
        n = len(g)
        if n < min_games or n >= threshold:
            continue
        rank = mf.resolve_rank(g)
        if rank in exclude_ranks:
            continue
        targets[puuid] = {"rank": rank, "n_games": n, "gap": threshold - n}
    return dict(sorted(targets.items(), key=lambda kv: kv[1]["gap"]))


def main() -> int:
    threshold = int(arg("--threshold", DEFAULT_THRESHOLD))
    min_games = int(arg("--min-games", DEFAULT_MIN_GAMES))
    exclude_raw = arg("--exclude-ranks", "")
    exclude_ranks = set(r.strip() for r in exclude_raw.split(",") if r.strip())
    out_path = Path(arg("--out", str(DATASET_DIR / "densify_targets.json")))

    dataset_path = DATASET_DIR / "adc_dataset.parquet"
    if not dataset_path.exists():
        print(f"✗ {dataset_path} introuvable — lancer build_dataset.py d'abord.", file=sys.stderr)
        return 1

    df = pd.read_parquet(dataset_path)
    ref = df[df["source"] == "referentiel"].copy()
    print(f"  {len(ref)} games référentiel | {ref['puuid'].nunique()} joueurs uniques", file=sys.stderr)
    if exclude_ranks:
        print(f"  exclut : {sorted(exclude_ranks)}", file=sys.stderr)

    targets = select_targets(ref, threshold=threshold, min_games=min_games,
                             exclude_ranks=exclude_ranks)

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
