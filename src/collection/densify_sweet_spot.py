#!/usr/bin/env python3
"""
densify_sweet_spot.py — orchestre la densification ciblée sur le sweet spot
~30 games/joueur identifié par src/02_data_science/analyze_auc_vs_ngames.py
(Courbe 2, pool fixe ≥50 : N=15→0.588, N=30→0.635 peak, puis plateau/déclin), en
excluant diamond.

Pourquoi ce script existe : l'analyse AUC-vs-N a tué l'hypothèse "plus de games par
joueur = plus d'AUC". N n'est PAS le levier du plafond ~0.65 master/GM. La
densification sert deux objectifs précis désormais :
  1. AMÉLIORER les features des joueurs de la bande [15,30[ : prod cappait à
     "all-available", donc un joueur à 20 games utilise 20 (sous sweet spot,
     features bruitées). Le pousser à 30 = +AUC direct (Courbe 2 : 15→30 = +0.047).
  2. RÉDUIRE la variance CV en grossissant le pool qualifié (Courbe 1 bruitée à
     haut N = bruit de petit pool, pas du signal).

Exclut diamond par défaut : la frontière réellement apprise est challenger vs
master (high ≈ 81% challenger, low ≈ 73% master, cf. dataset_report.py). Les 3
classes au bord sont master / GM / challenger. Densifier diamond pousse la classe
low VERS diamond, plus loin du boundary master/GM = bruit dans la classe low au
lieu de sharper la frontière.

Chaîne (deux étapes, A = 0 API, B = scraping) :
  A. densify_targets.select_targets(threshold=30, min_games=15, exclude=diamond)
     → data/04_dataset/densify_targets.json (trié par gap croissant : gains faciles
     en premier si le run est interrompu ; consommé par densify_players.py qui
     s'arrête PAR JOUEUR dès que `gap` games ADC neuves sont trouvées).
  B. densify_players.py --target-list ... --history N (scraping, quota API).

Usage (dry-run : sélection seulement, 0 appel API, montre le plan) :
    poetry run python3 src/collection/densify_sweet_spot.py

Usage (sélection + scraping) :
    poetry run python3 src/collection/densify_sweet_spot.py --run --history 60

Flags :
    --threshold N   sweet spot cible (défaut 30)
    --min-games N   ne pas chasser en dessous (défaut 15 = seuil qualif prod ;
                    abaisser pour grossir le pool via joueurs quasi-qualifiés)
    --exclude RANKS rangs à exclure, séparés par virgule (défaut "diamond" ;
                    passer --exclude "" pour ne rien exclure)
    --history N     profondeur d'historique par joueur (défaut 60, transmis à
                    densify_players.py)
    --days N         fenêtre temporelle en jours (défaut 28)
    --region X       plateforme (défaut : euw1 ou RIOT_REGION)
    --run            lance le scraping après la sélection (sinon dry-run)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                     # densify_targets
sys.path.insert(0, str(HERE.parent / "core"))      # riotlib
import pandas as pd
import riotlib as rl
from cli import arg, flag as has_flag
import densify_targets as dt

DATASET_DIR = rl.DATA / "04_dataset"
DEFAULT_THRESHOLD = 30
DEFAULT_MIN_GAMES = 15
DEFAULT_EXCLUDE = "diamond"
DEFAULT_HISTORY = 60
DEFAULT_DAYS = 28


def main() -> int:
    threshold = int(arg("--threshold", DEFAULT_THRESHOLD))
    min_games = int(arg("--min-games", DEFAULT_MIN_GAMES))
    exclude_raw = arg("--exclude", DEFAULT_EXCLUDE)
    exclude_ranks = set(r.strip() for r in exclude_raw.split(",") if r.strip()) if exclude_raw else set()
    history = int(arg("--history", DEFAULT_HISTORY))
    days = int(arg("--days", DEFAULT_DAYS))
    region = arg("--region")
    out_path = DATASET_DIR / "densify_targets.json"

    ref = dt.load_referential()
    if ref is None:
        return 1
    print(f"  plan : bande [{min_games}, {threshold}[ games"
          + (f", exclut {sorted(exclude_ranks)}" if exclude_ranks else "")
          + ", tri par gap croissant", file=sys.stderr)

    targets = dt.select_targets(ref, threshold=threshold, min_games=min_games,
                                 exclude_ranks=exclude_ranks)

    print(f"\n  Cibles : {len(targets)} joueurs", file=sys.stderr)
    dt.print_recap(targets, exclude_ranks)

    if not targets:
        print("\n  ⚠ Aucune cible dans la bande — rien à faire.", file=sys.stderr)
        return 0

    dt.write_targets(targets, out_path)
    print(f"\n✓ Cibles écrites : {out_path}", file=sys.stderr)

    cmd = [sys.executable, str(HERE / "densify_players.py"),
           "--target-list", str(out_path), "--history", str(history),
           "--days", str(days)]
    if region:
        cmd += ["--region", region]

    if has_flag("--run"):
        print(f"\n→ Lancement scraping : {' '.join(cmd)}\n", file=sys.stderr)
        return subprocess.call(cmd)
    else:
        print("\n  Dry-run (sélection seulement, 0 appel API). Pour scraper :", file=sys.stderr)
        print(f"    poetry run python3 src/collection/densify_sweet_spot.py --run --history {history}",
              file=sys.stderr)
        print("  ou directement :", file=sys.stderr)
        print(f"    poetry run python3 src/collection/densify_players.py "
              f"--target-list {out_path} --history {history}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())