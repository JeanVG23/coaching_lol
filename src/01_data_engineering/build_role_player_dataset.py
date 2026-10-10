#!/usr/bin/env python3
"""Construit les datasets joueur par rôle en fenêtres à profondeur fixe.

Une ligne représente ``(puuid, role, as_of)`` : les N dernières parties du rôle
antérieures au snapshot, agrégées puis labellisées avec le rang de ce snapshot.
La profondeur fixe empêche le volume d'historique collecté de devenir un proxy du
rang via les statistiques de dispersion.

Le rang du snapshot est attribué quelle que soit son ancienneté par rapport aux
parties agrégées : la borne de 14 jours qui existait ici venait du rythme des
patchs LoL, pas d'une propriété de la donnée, et la tenir imposait de recollecter
tout le corpus à chaque capture. L'écart n'est plus refusé, il est mesuré et
conservé dans ``label_age_days`` puis résumé dans le sidecar de provenance.

Usage : poetry run python3 src/01_data_engineering/build_role_player_dataset.py \
            --role JUNGLE [--n-window 20] [--snapshot-day AAAA-MM-JJ]
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "collection"))

import pandas as pd  # noqa: E402

import cli  # noqa: E402
import fetch_ladder  # noqa: E402
import ml_features as mf  # noqa: E402
import riotlib as rl  # noqa: E402
import role_features as rf  # noqa: E402

DATASET_DIR = rl.DATA / "04_dataset"
N_WINDOW = 20
MIN_PLAYER_GAMES = 15
_APEX_HIGH = {"GRANDMASTER", "CHALLENGER"}
_DAY_MS = 86_400_000


def window_match_ids(games: pd.DataFrame, puuid: str, *, n_window: int,
                     as_of_ms: int) -> list[str]:
    """IDs des N dernières parties de ``puuid`` au plus tard à ``as_of``."""
    sub = games[(games["puuid"] == puuid) & (games["game_ts"] <= as_of_ms)]
    sub = sub.sort_values("game_ts", ascending=False).head(n_window)
    return sub["match_id"].tolist()


def build_windows(games: pd.DataFrame, snapshot: dict, role: str, *,
                  n_window: int = N_WINDOW, as_of_ms: int) -> pd.DataFrame:
    """Agrège une fenêtre exactement longue de N pour chaque joueur éligible."""
    if n_window < MIN_PLAYER_GAMES:
        raise ValueError(
            f"n_window doit être >= MIN_PLAYER_GAMES ({MIN_PLAYER_GAMES})")
    role = rf.normalize_role(role)
    features = rf.ROLE_FEATURES[role]
    rows = []

    if "role" in games:
        normalized_roles = games["role"].map(rf.normalize_role)
        role_games = games[normalized_roles == role]
    else:
        role_games = games
    for puuid, group in role_games.groupby("puuid"):
        label = snapshot.get(puuid)
        if not label:
            continue

        sub = group[group["game_ts"] <= as_of_ms]
        sub = sub.sort_values("game_ts", ascending=False).head(n_window)
        if len(sub) < n_window:
            continue

        rec = {
            "puuid": puuid,
            "role": role,
            "as_of": as_of_ms,
            # Distance entre la dernière partie agrégée et la capture du rang.
            # Elle n'exclut plus la fenêtre, mais elle voyage avec elle : un rang
            # relevé des mois après les parties reste un rang daté, pas un rang
            # contemporain, et le lecteur doit pouvoir le voir.
            "label_age_days": (as_of_ms - int(sub["game_ts"].max())) // _DAY_MS,
            "game_ts_oldest": int(sub["game_ts"].min()),
            "game_ts_newest": int(sub["game_ts"].max()),
            "tier": label["tier"],
            "division": label.get("division"),
            "lp": label.get("lp"),
            "n_window": n_window,
            "high_elo": int(label["tier"] in _APEX_HIGH),
        }
        aggregates = mf.aggregate_player_features(sub, features)
        aggregates.pop("n_games", None)
        rec.update(aggregates)
        rows.append(rec)

    return pd.DataFrame(rows)


def provenance(out: pd.DataFrame, *, role: str, platform: str, day: str,
               as_of_ms: int, n_window: int) -> dict:
    """Ce que le parquet ne dit pas de lui-même.

    Il stocke `as_of` sans dire de quand datent les parties agrégées : rien ne
    permet donc de lire, depuis le dataset seul, que le rang a pu être relevé
    des mois après elles. Ce sidecar l'écrit noir sur blanc, ce qui remplace la
    borne d'âge retirée : le décalage n'est plus refusé, il est publié.
    """
    ages = out["label_age_days"].astype(int)
    return {
        "built_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "role": role,
        "platform": platform,
        "snapshot_day": day,
        "as_of_ms": int(as_of_ms),
        "n_window": int(n_window),
        "n_rows": int(len(out)),
        "n_by_tier": {str(tier): int(count)
                      for tier, count in out["tier"].value_counts().items()},
        "label_age_days": {"min": int(ages.min()),
                           "p50": int(ages.median()),
                           "max": int(ages.max())},
        "game_ts_oldest": int(out["game_ts_oldest"].min()),
        "game_ts_newest": int(out["game_ts_newest"].max()),
    }


def main() -> int:
    role = rf.normalize_role(cli.arg("--role", "BOTTOM"))
    if role not in rf.ROLES:
        print(f"rôle inconnu : {role} (attendus : {', '.join(rf.ROLES)})",
              file=sys.stderr)
        return 2

    n_window = cli.int_arg("--n-window", N_WINDOW)
    if n_window < MIN_PLAYER_GAMES:
        print(f"n-window doit être >= {MIN_PLAYER_GAMES}", file=sys.stderr)
        return 2
    day = cli.arg("--snapshot-day", dt.date.today().isoformat())
    platform = cli.arg("--platform", "euw1")

    games_path = DATASET_DIR / f"{role.lower()}_dataset.parquet"
    legacy_path = DATASET_DIR / "utility_dataset.parquet"
    if role == "SUPPORT" and not games_path.exists() and legacy_path.exists():
        games_path = legacy_path
    games = pd.read_parquet(games_path)
    snapshot = fetch_ladder.load_snapshot(platform, day)
    captured = {int(row["ts"]) for row in snapshot.values()
                if row.get("ts") is not None}
    if len(captured) > 1:
        raise ValueError(f"snapshot {platform}/{day} contient plusieurs timestamps")
    # Repli pour les anciennes fixtures/snapshots, avant l'ajout de `ts`.
    as_of_ms = (captured.pop() if captured else int(
        dt.datetime.fromisoformat(day).replace(tzinfo=dt.UTC).timestamp() * 1000))
    out = build_windows(games, snapshot, role, n_window=n_window,
                        as_of_ms=as_of_ms)

    path = DATASET_DIR / f"{role.lower()}_player_dataset.parquet"
    # Une table vide par-dessus une table pleine efface une collecte entière et
    # ne se voit qu'au ré-entraînement. Supprimer un dataset doit rester un geste
    # explicite, pas l'effet de bord d'un rebuild qui n'a rien trouvé.
    if out.empty and path.exists() and len(pd.read_parquet(path)):
        print(f"✗ 0 fenêtre {role} : {path.name} existant conservé "
              f"(supprimer le fichier pour le remplacer par une table vide)",
              file=sys.stderr)
        return 1
    out.to_parquet(path, index=False)
    print(f"✓ {len(out)} fenêtres {role} (N={n_window}) -> {path}")
    if not out.empty:
        meta = provenance(out, role=role, platform=platform, day=day,
                          as_of_ms=as_of_ms, n_window=n_window)
        rl.dump_json_atomic(path.with_suffix(".meta.json"), meta)
        print(f"  répartition : {meta['n_by_tier']}")
        print(f"  âge du label (jours) : {meta['label_age_days']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
