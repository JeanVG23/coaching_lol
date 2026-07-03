# src/pipeline_ops/validate_traits.py
"""Validation data-driven de champion_traits.json.

Croise les axes curés (lane_pattern, power_curve, playstyle, gank_threat, roam)
avec les statistiques réelles agrégées depuis data/02_silver/referentiel/.

Pour chaque champion × axe × label_curé, on calcule des stats de jeu réelles
(CS@10, ganks@15, roam_count, etc.) et on compare à la distribution du groupe
de label (ex. tous les junglers 'gank_threat=high').

Sortie :
  - Console : rapport texte lisible, verdicts par champion × axe
  - data/00_static/derived/champion_axis_validation.json : rapport structuré

0 appel API. Lecture seule.

Usage :
    python3 src/pipeline_ops/validate_traits.py
    python3 src/pipeline_ops/validate_traits.py --json out.json
    python3 src/pipeline_ops/validate_traits.py --min-games 30
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import champion_profiles as cp
import riotlib as rl

STATIC_DIR = rl.ROOT / "data" / "00_static"
DERIVED_DIR = STATIC_DIR / "derived"
SILVER_REF_DIR = rl.ROOT / "data" / "02_silver" / "referentiel"
DEFAULT_OUT = DERIVED_DIR / "champion_axis_validation.json"

# Axes curés + leur rôle "porteur" (où l'axe a du sens).
# Symétrique de _AXE_PORTEUSES dans audit_polyvalence, mais avec
# un rôle **attendu** unique (le rôle principal où l'axe s'applique).
_AXE_ROLE = {
    "playstyle": "JUNGLE",
    "gank_threat": "JUNGLE",
    "roam": "MIDDLE",      # mid roam OU support roam
    "lane_pattern": "BOTTOM",  # ADC + support de botlane
    "power_curve": "BOTTOM",   # ADC + mid + support
}

# Rôles additionnels acceptés par axe (pour le calcul de stats par rôle_effectif).
_AXE_ROLES = {
    "playstyle": ("JUNGLE",),
    "gank_threat": ("JUNGLE",),
    "roam": ("MIDDLE", "UTILITY"),
    "lane_pattern": ("BOTTOM", "UTILITY"),
    "power_curve": ("BOTTOM", "MIDDLE", "UTILITY"),
}

MIN_GAMES_DEFAULT = 20


# ---------- Chargement silver référentiel ----------

def load_silver_referentials(ranks: list[str] | None = None) -> list[dict]:
    """Charge toutes les games silver des dossiers référentiels.

    Args:
        ranks: liste de ranks à inclure (None = tous, ex. ['challenger']).

    Returns:
        Liste de games silver (dicts).
    """
    games = []
    for rank_dir in sorted(SILVER_REF_DIR.iterdir()):
        if not rank_dir.is_dir():
            continue
        if ranks and rank_dir.name not in ranks:
            continue
        games_path = rank_dir / "games.jsonl"
        if not games_path.exists():
            continue
        with open(games_path) as f:
            for line in f:
                if not line.strip():
                    continue
                games.append(json.loads(line))
    return games


def index_games_by_champ_role(games: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Indexe { (champion, role): [games] }.

    Utilise le **target player** (champion+role du record silver).
    On filtre role != '?' et role != 'TOP' (axe pas couvert par un toplaner seul).
    """
    idx: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for g in games:
        champ = g.get("champion")
        role = g.get("role")
        if not champ or not role or role == "?":
            continue
        idx[(champ, role)].append(g)
    return idx


def game_count_per_axis(idx: dict[tuple[str, str], list[dict]]) -> dict[str, int]:
    """Combien de games couvrent chaque axe, tous rôles confondus."""
    out: dict[str, int] = {}
    for axis, roles in _AXE_ROLES.items():
        n = 0
        for role in roles:
            for (champ, r), gs in idx.items():
                if r == role:
                    n += len(gs)
        out[axis] = n
    return out


# ---------- Stats par axe ----------

# Gank signal : on compose un score = présence offensive jungler.
# Sources (depuis silver, position features) :
#   - frac_enemy_half  : fraction de frames en terrain ennemi (river/invade/gank)
#   - frac_roam_mid    : présence hors jungle/river en mid-game (visites lanes)
#   - wards_killed     : agression vision (dénombre les wards ennemis)
# Pondérations calibrées pour spread [0, 1].
_GANK_W = {"enemy_half": 0.6, "roam_mid": 0.25, "wards_killed": 0.15}


def _gank_score_for_game(game: dict) -> float | None:
    """Score gank pour UNE game jungler. None si position features indisponibles."""
    pos = game.get("position") or {}
    eh = pos.get("frac_enemy_half")
    rm = pos.get("frac_roam_mid")
    wk = pos.get("wards_killed")
    if eh is None:
        return None
    # Normalisation empirique (max observés ~0.35, 0.55, 8.0).
    eh_n = min(eh / 0.35, 1.0)
    rm_n = min((rm or 0) / 0.55, 1.0)
    wk_n = min((wk or 0) / 8.0, 1.0)
    return (
        _GANK_W["enemy_half"] * eh_n
        + _GANK_W["roam_mid"] * rm_n
        + _GANK_W["wards_killed"] * wk_n
    )


# === Gank detector depuis la raw timeline ===
# Lecture des positions du jungler minute par minute (frames 0..15).
# On compte les "visites de lane" : frames où le jungler est dans TOP/MID/BOT
# (≠ JUNGLE/RIVER) ET qu'un champion ennemi est dans la même zone (proxy vision
# game — la vraie "présence" ennemie en lane).
#
# Pourquoi lire la raw : la silver ne stocke que les features agrégées (moyennes),
# pas la trajectoire frame-par-frame. C'est ~5ms par game (déjà mesuré).
_LANE_ZONES = ("TOP", "MID", "BOT")


def _detect_lane_visits(match: dict, timeline: dict, target_puuid: str,
                        target_role: str, max_minute: int = 15) -> dict:
    """Pour UN joueur, compte les visites de lane pendant les N premières minutes.

    Returns:
        {
          "lane_visits": int,        # nb de frames en zone lane (TOP/MID/BOT)
          "gank_frames": int,        # frames en lane avec ennemi dans la même zone
          "gank_kills": int,         # CHAMPION_KILL events pendant ces frames
          "early_deaths": int,       # morts du target en early game
        }
    """
    meta = match.get("metadata", {})
    puuid_to_idx = {p: i + 1 for i, p in enumerate(meta.get("participants", []))}
    if target_puuid not in puuid_to_idx:
        return {"lane_visits": 0, "gank_frames": 0, "gank_kills": 0, "early_deaths": 0}
    my_idx = puuid_to_idx[target_puuid]
    my_team = 100 if my_idx <= 5 else 200
    enemy_idxs = set(range(1, 11)) - set(range(1, 6) if my_team == 100 else range(6, 11))

    lane_visits = gank_frames = gank_kills = early_deaths = 0
    # Cache : pour chaque frame, calcule une fois la zone de chaque joueur
    for fr in timeline.get("info", {}).get("frames", []):
        minute = round(fr["timestamp"] / 60000)
        if minute > max_minute:
            break
        pf = fr.get("participantFrames", {})
        my_pf = pf.get(str(my_idx), {})
        my_pos = my_pf.get("position") or {}
        if not my_pos.get("x"):
            continue
        from riotlib import approx_zone
        my_zone = approx_zone(my_pos["x"], my_pos["y"])
        if my_zone not in _LANE_ZONES:
            continue
        lane_visits += 1
        # Y a-t-il un ennemi dans la même zone ?
        enemy_in_zone = False
        for eidx in enemy_idxs:
            epf = pf.get(str(eidx), {})
            epos = epf.get("position") or {}
            if not epos.get("x"):
                continue
            if approx_zone(epos["x"], epos["y"]) == my_zone:
                enemy_in_zone = True
                break
        if enemy_in_zone:
            gank_frames += 1
        # Kills pendant ce frame : un kill adjacent à un gank ?
        for ev in fr.get("events", []):
            if ev.get("type") != "CHAMPION_KILL":
                continue
            assisting = ev.get("assistingParticipantIds") or []
            if ev.get("killerId") == my_idx or my_idx in assisting:
                gank_kills += 1
            if ev.get("victimId") == my_idx and minute <= 14:
                early_deaths += 1
    return {
        "lane_visits": lane_visits,
        "gank_frames": gank_frames,
        "gank_kills": gank_kills,
        "early_deaths": early_deaths,
    }


def compute_gank_stats_from_raw(
    idx: dict[tuple[str, str], list[dict]],
    min_games: int = MIN_GAMES_DEFAULT,
    progress: bool = True,
) -> dict[str, dict]:
    """Lit les raw timelines et calcule les stats gank par jungler.

    Returns:
        {champion: {"n": int, "lane_visits_mean": float, "gank_frames_mean": float, "gank_kills_mean": float, "raw": [per-game dicts]}}
    """
    from riotlib import _read_raw
    per_game: dict[str, list[dict]] = defaultdict(list)
    jungler_games = [(champ, g) for (champ, role), gs in idx.items()
                     if role == "JUNGLE" for g in gs]
    n_total = len(jungler_games)
    n_done = 0
    if progress and n_total:
        print(f"  Lecture raw timelines pour {n_total} games jungler…", flush=True)
    for champ, game in jungler_games:
        match_id = game.get("match_id")
        if not match_id:
            continue
        # match_id format : "EUW1_7902443455" — déduplique le préfixe.
        if "_" in match_id:
            mid = match_id
        else:
            mid = f"EUW1_{match_id}"
        match = _read_raw(f"{mid}_match")
        timeline = _read_raw(f"{mid}_timeline")
        if not match or not timeline:
            continue
        info = _detect_lane_visits(match, timeline, game["puuid"], "JUNGLE")
        per_game[champ].append(info)
        n_done += 1
        if progress and n_total and n_done % 500 == 0:
            print(f"    {n_done}/{n_total}…", flush=True)

    out: dict[str, dict] = {}
    for champ, recs in per_game.items():
        if len(recs) < min_games:
            continue
        out[champ] = {
            "n": len(recs),
            "lane_visits_mean": fmean(r["lane_visits"] for r in recs),
            "gank_frames_mean": fmean(r["gank_frames"] for r in recs),
            "gank_kills_mean": fmean(r["gank_kills"] for r in recs),
            "raw": recs,
        }
    return out


# === Roam stats (mid + support) ===
# Source unique : silver position.frac_roam_mid (moyenne des frames mid-game
# passées hors de sa lane propre). Raw timeline trop sparse (1 frame/min) pour
# détecter les roams brefs — silver l'agrège déjà sur la phase mid.
def compute_roam_stats(
    idx: dict[tuple[str, str], list[dict]],
    min_games: int = MIN_GAMES_DEFAULT,
) -> dict[str, dict]:
    """Calcule frac_roam_mid moyen par champion mid/support.

    Returns:
        {champion: {"n": int, "roam_mean": float, "roam_values": list[float]}}
    """
    out: dict[str, dict] = {}
    for (champ, role), games in idx.items():
        if role not in ("MIDDLE", "UTILITY"):
            continue
        vals = []
        for g in games:
            pos = g.get("position") or {}
            r = pos.get("frac_roam_mid")
            if r is not None:
                vals.append(r)
        if len(vals) < min_games:
            continue
        out[champ] = {
            "n": len(vals),
            "roam_mean": fmean(vals),
            "roam_values": vals,
        }
    return out


def group_roam_stats_by_label(
    per_champ: dict[str, dict],
    traits: dict,
) -> dict[str, dict]:
    """Agrège roam_mean par label 'roam' curé (high/med/low).

    Returns: {label: {"n_champions", "n_games", "score_median", "score_p25", "score_p75"}}
    """
    by_label: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for champ, data in per_champ.items():
        label = traits.get(champ, {}).get("roam")
        if not label or label == "unknown":
            continue
        by_label[label].extend(data["roam_values"])
        counts[label] += 1
    out: dict[str, dict] = {}
    for label, scores in by_label.items():
        if not scores:
            continue
        out[label] = {
            "n_champions": counts[label],
            "n_games": len(scores),
            "score_median": median(scores),
            "score_mean": fmean(scores),
            "score_p25": _percentile(scores, 25),
            "score_p75": _percentile(scores, 75),
        }
    return out


# === Lane pattern stats (ADC + support) ===
# Pour un duo botlane, le `lane_pattern` se manifeste par :
#   - poke : CS@10 positif (push passif, farm safe), peu de kills en early
#   - all_in : forte kill participation en early (2v2 fights)
#   - sustain : milieu de tableau
#   - scaling : CS plus bas en early (focus farm sans push), KD neutre
#
# Source : silver 'lane.csd10' (CS diff vs opponent) + 'kills' early count.
def compute_lane_pattern_stats(
    idx: dict[tuple[str, str], list[dict]],
    min_games: int = MIN_GAMES_DEFAULT,
) -> dict[str, dict]:
    """Calcule csd10 et early_kp par champion (rôles BOTTOM/UTILITY).

    Returns:
        {champion: {"n", "csd10_mean", "early_kp_mean", "csd10_values", "early_kp_values"}}
    """
    out: dict[str, dict] = {}
    for (champ, role), games in idx.items():
        if role not in ("BOTTOM", "UTILITY"):
            continue
        csd10_vals, ekp_vals = [], []
        for g in games:
            lane = g.get("lane") or {}
            csd10 = lane.get("csd10")
            kills = g.get("kills") or []
            early_kp = sum(1 for k in kills if k.get("phase") == "early")
            if csd10 is not None:
                csd10_vals.append(csd10)
            ekp_vals.append(early_kp)
        if len(csd10_vals) < min_games:
            continue
        out[champ] = {
            "n": len(csd10_vals),
            "csd10_mean": fmean(csd10_vals),
            "early_kp_mean": fmean(ekp_vals) if ekp_vals else 0.0,
            "csd10_values": csd10_vals,
            "early_kp_values": ekp_vals,
        }
    return out


def group_lane_pattern_stats_by_label(
    per_champ: dict[str, dict],
    traits: dict,
) -> dict[str, dict]:
    """Agrège csd10_mean par label 'lane_pattern' curé.

    Returns: {label: {n_champions, n_games, score_median, score_p25, score_p75}}
    """
    by_label: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for champ, data in per_champ.items():
        label = traits.get(champ, {}).get("lane_pattern")
        if not label or label == "unknown":
            continue
        by_label[label].extend(data["csd10_values"])
        counts[label] += 1
    out: dict[str, dict] = {}
    for label, scores in by_label.items():
        if not scores:
            continue
        out[label] = {
            "n_champions": counts[label],
            "n_games": len(scores),
            "score_median": median(scores),
            "score_mean": fmean(scores),
            "score_p25": _percentile(scores, 25),
            "score_p75": _percentile(scores, 75),
        }
    return out


# === Power curve stats (ADC + mid + support) ===
# Winrate conditionnée par durée de game. Lit gameDuration depuis la raw.
# Hypothèses : early power → winrate_haut quand game < 25min ; late power →
# winrate_haut quand game >= 35min.
def compute_power_curve_stats(
    idx: dict[tuple[str, str], list[dict]],
    min_games: int = MIN_GAMES_DEFAULT,
    progress: bool = True,
) -> dict[str, dict]:
    """Winrate par champion par bucket de durée (raw match.gameDuration).

    Returns:
        {champion: {
          "n": int,
          "winrate_overall": float,
          "winrate_short": float,   # < 25min
          "winrate_mid": float,     # 25-34min
          "winrate_long": float,    # >= 35min
          "raw": [per-game dicts]
        }}
    """
    from riotlib import _read_raw
    per_game: dict[str, list[dict]] = defaultdict(list)
    pc_games = [(champ, g) for (champ, role), gs in idx.items()
                if role in ("BOTTOM", "MIDDLE", "UTILITY")
                for g in gs]
    n_total = len(pc_games)
    n_done = 0
    if progress and n_total:
        print(f"  Lecture raw pour {n_total} games power_curve…", flush=True)
    for champ, game in pc_games:
        match_id = game.get("match_id")
        if not match_id:
            continue
        mid = match_id if "_" in match_id else f"EUW1_{match_id}"
        match = _read_raw(f"{mid}_match")
        if not match:
            continue
        duration_min = match.get("info", {}).get("gameDuration", 0) / 60.0
        if duration_min <= 0:
            continue
        win = bool(game.get("win"))
        per_game[champ].append({
            "win": win,
            "duration": duration_min,
        })
        n_done += 1
        if progress and n_total and n_done % 1000 == 0:
            print(f"    {n_done}/{n_total}…", flush=True)

    out: dict[str, dict] = {}
    for champ, recs in per_game.items():
        if len(recs) < min_games:
            continue
        short_games = [r for r in recs if r["duration"] < 25]
        mid_games = [r for r in recs if 25 <= r["duration"] < 35]
        long_games = [r for r in recs if r["duration"] >= 35]
        n_short, n_mid, n_long = len(short_games), len(mid_games), len(long_games)
        wr_overall = fmean(1 if r["win"] else 0 for r in recs) if recs else 0
        wr_short = fmean(1 if r["win"] else 0 for r in short_games) if short_games else None
        wr_mid = fmean(1 if r["win"] else 0 for r in mid_games) if mid_games else None
        wr_long = fmean(1 if r["win"] else 0 for r in long_games) if long_games else None
        out[champ] = {
            "n": len(recs),
            "n_short": n_short,
            "n_mid": n_mid,
            "n_long": n_long,
            "winrate_overall": wr_overall,
            "winrate_short": wr_short,
            "winrate_mid": wr_mid,
            "winrate_long": wr_long,
            "raw": recs,
        }
    return out


def group_power_curve_stats_by_label(
    per_champ: dict[str, dict],
    traits: dict,
    bucket: str = "long",
) -> dict[str, dict]:
    """Agrège winrate par bucket de durée, par label 'power_curve' curé.

    Args:
        bucket: "short" / "mid" / "long" / "overall"
    """
    key = f"winrate_{bucket}"
    by_label: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for champ, data in per_champ.items():
        label = traits.get(champ, {}).get("power_curve")
        if not label or label == "unknown":
            continue
        v = data.get(key)
        if v is None:
            continue
        by_label[label].append(v)
        counts[label] += 1
    out: dict[str, dict] = {}
    for label, scores in by_label.items():
        if not scores:
            continue
        out[label] = {
            "n_champions": counts[label],
            "score_median": median(scores),
            "score_mean": fmean(scores),
            "score_p25": _percentile(scores, 25),
            "score_p75": _percentile(scores, 75),
        }
    return out


def compute_gank_stats(idx: dict[tuple[str, str], list[dict]]) -> dict[str, dict]:
    """Stats gank par champion (rôle JUNGLE) : score moyen + n_games.

    Returns:
        {champion: {"n": int, "score_mean": float, "scores": list[float]}}
    """
    out: dict[str, dict] = {}
    for (champ, role), games in idx.items():
        if role != "JUNGLE":
            continue
        scores = [s for g in games if (s := _gank_score_for_game(g)) is not None]
        if not scores:
            continue
        out[champ] = {
            "n": len(scores),
            "score_mean": fmean(scores),
            "score_median": median(scores),
            "scores": scores,  # brut, pour calcul de verdicts plus tard
        }
    return out


def group_gank_stats_by_label(
    per_champ: dict[str, dict],
    traits: dict,
) -> dict[str, dict]:
    """Agrège les stats gank par label curé (ganking/skirmish/farming, high/med/low).

    Returns:
        {label: {"n_champions": int, "n_games": int, "score_median": float, ...}}

    Utilise `gank_kills_mean` (par champion), pas le count par game.
    """
    by_label: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for champ, data in per_champ.items():
        trait = traits.get(champ, {})
        value = data.get("gank_kills_mean")
        if value is None:
            continue
        for axis in ("playstyle", "gank_threat"):
            label = trait.get(axis)
            if not label or label == "unknown":
                continue
            by_label[f"{axis}={label}"].append(value)
            counts[f"{axis}={label}"] += 1
    out: dict[str, dict] = {}
    for label, scores in by_label.items():
        if not scores:
            continue
        out[label] = {
            "n_champions": counts[label],
            "n_games": counts[label],  # placeholder, n_games == n_champions
            "score_median": median(scores),
            "score_mean": fmean(scores),
            "score_p25": _percentile(scores, 25),
            "score_p75": _percentile(scores, 75),
        }
    return out


def _percentile(values: list[float], p: int) -> float:
    """Percentile linéaire simple, sans dépendance numpy."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def validate_champion_axis(
    champ: str,
    axis: str,
    actual_value: float,
    group_dist: dict,
    n_champ_games: int,
) -> str:
    """Compare la valeur d'un champion à la distribution de son groupe curé.

    Args:
        champ: nom du champion (debug only)
        axis: nom de l'axe ('gank_threat', 'playstyle', ...)
        actual_value: valeur mesurée (ex. gank_kills_mean)
        group_dist: {"score_median": float, "score_p25": float, "score_p75": float, "score_mean": float}
        n_champ_games: nb de games du champion (pour signal de fiabilité)

    Returns:
        verdict: "validated" | "contradicted" | "neutral" | "insufficient_data"
    """
    if n_champ_games < 20:
        return "insufficient_data"
    med = group_dist.get("score_median", 0)
    p25 = group_dist.get("score_p25", 0)
    p75 = group_dist.get("score_p75", 0)
    if actual_value > p75:
        return "above_group"   # champion plus extrême que 75% du groupe
    if actual_value < p25:
        return "below_group"   # champion plus bas que 75% du groupe
    if abs(actual_value - med) < 0.3 * (p75 - p25 + 0.1):
        return "validated"
    return "neutral"


# ---------- Main ----------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validation data-driven de champion_traits.json (0 API)."
    )
    ap.add_argument("--ranks", nargs="+", default=None,
                    help="ranks à inclure (ex. challenger master). Défaut: tous.")
    ap.add_argument("--min-games", type=int, default=MIN_GAMES_DEFAULT,
                    help=f"seuil minimum de games (défaut: {MIN_GAMES_DEFAULT}).")
    ap.add_argument("--json", metavar="PATH", default=str(DEFAULT_OUT),
                    help=f"sortie JSON (défaut: {DEFAULT_OUT}).")
    args = ap.parse_args()

    print("=== validate_traits — Phase 1 (squelette + chargement) ===")
    print(f"Loading silver referentials from {SILVER_REF_DIR}…")
    games = load_silver_referentials(args.ranks)
    print(f"  → {len(games)} games chargées")

    if args.ranks:
        print(f"  → filtrées sur ranks: {args.ranks}")

    print("Indexing par (champion, role)…")
    idx = index_games_by_champ_role(games)
    print(f"  → {len(idx)} combos (champion, role) distincts")

    # Combos au-dessus du seuil
    n_above = sum(1 for gs in idx.values() if len(gs) >= args.min_games)
    print(f"  → {n_above} combos avec >= {args.min_games} games")

    # Couverture par axe
    coverage = game_count_per_axis(idx)
    print("\nCouverture par axe (games où l'axe a du sens) :")
    for axis, n in sorted(coverage.items()):
        roles = ", ".join(_AXE_ROLES[axis])
        print(f"  • {axis:<14} {n:>6} games   (rôles: {roles})")

    # Top combos par nombre de games (debug)
    top = sorted(idx.items(), key=lambda kv: -len(kv[1]))[:10]
    print(f"\nTop 10 combos (champion, role) par nombre de games :")
    for (champ, role), gs in top:
        print(f"  • {champ:<18} {role:<8} {len(gs)} games")

    # --- Phase 2 : gank stats (raw timelines) ---
    print("\n=== Phase 2 — Stats gank (junglers, raw timelines) ===")
    traits = cp.load_traits()
    per_champ_gank = compute_gank_stats_from_raw(idx, min_games=args.min_games)

    print(f"  {len(per_champ_gank)} junglers avec >= {args.min_games} games")
    print(f"\n  Top 10 par gank_frames moyen (proxy gank) :")
    top = sorted(per_champ_gank.items(), key=lambda kv: -kv[1]["gank_frames_mean"])[:10]
    for champ, st in top:
        print(f"  • {champ:<14} lane_visits={st['lane_visits_mean']:>4.1f} "
              f"gank_frames={st['gank_frames_mean']:>4.1f} "
              f"gank_kills={st['gank_kills_mean']:>4.1f} n={st['n']}")

    print(f"\n  Bottom 10 par gank_frames moyen :")
    bot = sorted(per_champ_gank.items(), key=lambda kv: kv[1]["gank_frames_mean"])[:10]
    for champ, st in bot:
        print(f"  • {champ:<14} lane_visits={st['lane_visits_mean']:>4.1f} "
              f"gank_frames={st['gank_frames_mean']:>4.1f} "
              f"gank_kills={st['gank_kills_mean']:>4.1f} n={st['n']}")

    # Distribution par label
    by_label_gank = group_gank_stats_by_label(per_champ_gank, traits)
    if by_label_gank:
        print(f"\n  Distribution par label (gank_frames moyen) :")
        print(f"  {'label':<24} {'n_champs':>9} {'n_games':>9} {'médian':>8} {'P25':>6} {'P75':>6}")
        for label, st in sorted(by_label_gank.items()):
            print(f"  {label:<24} {st['n_champions']:>9} {st['n_games']:>9} "
                  f"{st['score_median']:>8.3f} {st['score_p25']:>6.3f} {st['score_p75']:>6.3f}")

    # --- Phase 3 : roam stats (mid + support) ---
    print("\n=== Phase 3 — Stats roam (mid + support, silver position) ===")
    per_champ_roam = compute_roam_stats(idx, min_games=args.min_games)
    by_label_roam = group_roam_stats_by_label(per_champ_roam, traits)

    print(f"  {len(per_champ_roam)} champions mid/support avec score calculé")
    if by_label_roam:
        print(f"\n  Distribution par label 'roam' (frac_roam_mid) :")
        print(f"  {'label':<12} {'n_champs':>9} {'n_games':>9} {'médian':>8} {'P25':>6} {'P75':>6}")
        for label, st in sorted(by_label_roam.items()):
            print(f"  {label:<12} {st['n_champions']:>9} {st['n_games']:>9} "
                  f"{st['score_median']:>8.3f} {st['score_p25']:>6.3f} {st['score_p75']:>6.3f}")

    print(f"\n  Top 5 mid roamers :")
    mid_roamers = sorted(
        [(c, d) for c, d in per_champ_roam.items()
         if any(g.get("role") == "MIDDLE" for g in idx.get((c, "MIDDLE"), []))],
        key=lambda kv: -kv[1]["roam_mean"],
    )[:5]
    for champ, st in mid_roamers:
        print(f"  • {champ:<14} roam_mean={st['roam_mean']:.3f} n={st['n']}")

    # --- Phase 4 : lane_pattern stats (ADC + support) ---
    print("\n=== Phase 4 — Stats lane_pattern (ADC + support) ===")
    per_champ_lp = compute_lane_pattern_stats(idx, min_games=args.min_games)
    by_label_lp = group_lane_pattern_stats_by_label(per_champ_lp, traits)

    print(f"  {len(per_champ_lp)} champions ADC/support avec stats calculées")
    if by_label_lp:
        print(f"\n  Distribution par label 'lane_pattern' (csd10) :")
        print(f"  {'label':<12} {'n_champs':>9} {'n_games':>9} {'médian':>8} {'P25':>6} {'P75':>6}")
        for label, st in sorted(by_label_lp.items()):
            print(f"  {label:<12} {st['n_champions']:>9} {st['n_games']:>9} "
                  f"{st['score_median']:>8.2f} {st['score_p25']:>6.2f} {st['score_p75']:>6.2f}")

    print(f"\n  Top 5 'all_in' présumés (csd10 négatif + early_kp haut) :")
    all_in = sorted(
        [(c, d) for c, d in per_champ_lp.items()],
        key=lambda kv: (-kv[1]["early_kp_mean"], kv[1]["csd10_mean"]),
    )[:5]
    for champ, st in all_in:
        print(f"  • {champ:<14} csd10={st['csd10_mean']:>6.1f} early_kp={st['early_kp_mean']:>4.2f} n={st['n']}")

    print(f"\n  Top 5 'poke' présumés (csd10 haut + early_kp bas) :")
    poke = sorted(
        [(c, d) for c, d in per_champ_lp.items()],
        key=lambda kv: (kv[1]["early_kp_mean"], -kv[1]["csd10_mean"]),
    )[:5]
    for champ, st in poke:
        print(f"  • {champ:<14} csd10={st['csd10_mean']:>6.1f} early_kp={st['early_kp_mean']:>4.2f} n={st['n']}")

    # --- Phase 5 : power_curve stats (ADC + mid + support) ---
    print("\n=== Phase 5 — Stats power_curve (ADC + mid + support, raw gameDuration) ===")
    per_champ_pc = compute_power_curve_stats(idx, min_games=args.min_games)

    print(f"  {len(per_champ_pc)} champions avec winrates calculés")
    print(f"\n  Top 5 late-game scalers (winrate_long - winrate_short > 0.05) :")
    scalers = sorted(
        [(c, d) for c, d in per_champ_pc.items()
         if d.get("winrate_long") is not None and d.get("winrate_short") is not None],
        key=lambda kv: (kv[1]["winrate_long"] or 0) - (kv[1]["winrate_short"] or 0),
        reverse=True,
    )[:5]
    for champ, st in scalers:
        delta = (st["winrate_long"] or 0) - (st["winrate_short"] or 0)
        print(f"  • {champ:<14} WR_short={st['winrate_short']:>5.1%} "
              f"WR_long={st['winrate_long']:>5.1%} Δ={delta:+.1%} n={st['n']}")

    print(f"\n  Top 5 early-game bullies (winrate_short > winrate_long) :")
    bullies = sorted(
        [(c, d) for c, d in per_champ_pc.items()
         if d.get("winrate_long") is not None and d.get("winrate_short") is not None],
        key=lambda kv: (kv[1]["winrate_short"] or 0) - (kv[1]["winrate_long"] or 0),
        reverse=True,
    )[:5]
    for champ, st in bullies:
        delta = (st["winrate_short"] or 0) - (st["winrate_long"] or 0)
        print(f"  • {champ:<14} WR_short={st['winrate_short']:>5.1%} "
              f"WR_long={st['winrate_long']:>5.1%} Δ={delta:+.1%} n={st['n']}")

    by_label_pc = group_power_curve_stats_by_label(per_champ_pc, traits, bucket="long")
    if by_label_pc:
        print(f"\n  Distribution par label 'power_curve' (winrate_long) :")
        print(f"  {'label':<10} {'n_champs':>9} {'médian':>8} {'P25':>6} {'P75':>6}")
        for label, st in sorted(by_label_pc.items()):
            print(f"  {label:<10} {st['n_champions']:>9} "
                  f"{st['score_median']:>8.1%} {st['score_p25']:>6.1%} {st['score_p75']:>6.1%}")

    # --- Phase 6 : Validation + sortie JSON ---
    print("\n=== Phase 6 — Validation globale + sortie JSON ===")
    report = build_report(
        traits=traits,
        per_champ_gank=per_champ_gank,
        per_champ_roam=per_champ_roam,
        per_champ_lp=per_champ_lp,
        per_champ_pc=per_champ_pc,
        by_label_gank=by_label_gank,
        by_label_roam=by_label_roam,
        by_label_lp=by_label_lp,
        by_label_pc=by_label_pc,
        min_games=args.min_games,
        n_total_games=len(games),
    )

    # --- Phase 7 : Proposals pour champions sans axes ---
    print("\n=== Phase 7 — Proposals pour champions sans axes curés ===")
    proposals = build_proposals(
        traits=traits,
        per_champ_gank=per_champ_gank,
        per_champ_roam=per_champ_roam,
        per_champ_lp=per_champ_lp,
        per_champ_pc=per_champ_pc,
        by_label_gank=by_label_gank,
        by_label_roam=by_label_roam,
        by_label_lp=by_label_lp,
        by_label_pc=by_label_pc,
        min_games=args.min_games,
    )
    report["proposals"] = proposals
    print(f"  {len(proposals)} champions avec assez de données et SANS axes curés")
    if proposals:
        print(f"\n  Top 10 par nombre de games :")
        top = sorted(proposals, key=lambda p: -p["n_games"])[:10]
        for p in top:
            ax_str = ", ".join(f"{a}={v}" for a, v in p["proposed_axes"].items())
            print(f"  • {p['champion']:<18} ({p['role']:<8}, n={p['n_games']:>3}) → {ax_str}")

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  → JSON écrit : {out_path}  ({out_path.stat().st_size // 1024} Ko)")

    # Affiche le rapport texte
    print()
    print(render_text(report))

    print(f"\n✓ Toutes les phases OK (1+2+3+4+5+6+7).")
    return 0


# ---------- Phase 7 : proposals ----------

def _nearest_label(value: float, label_dists: dict) -> str | None:
    """Renvoie le label dont la médiane est la plus proche de `value`.

    label_dists: {label: {"score_median": float, ...}}
    """
    if not label_dists or value is None:
        return None
    best_label, best_dist = None, float("inf")
    for label, st in label_dists.items():
        med = st.get("score_median")
        if med is None:
            continue
        d = abs(value - med)
        if d < best_dist:
            best_dist = d
            best_label = label
    return best_label


def build_proposals(
    traits: dict,
    per_champ_gank: dict,
    per_champ_roam: dict,
    per_champ_lp: dict,
    per_champ_pc: dict,
    by_label_gank: dict,
    by_label_roam: dict,
    by_label_lp: dict,
    by_label_pc: dict,
    min_games: int,
) -> list[dict]:
    """Pour chaque champion avec assez de games mais SANS axes curés, propose
    un set d'axes basé sur les stats mesurées (nearest-group).

    Returns:
        [{
          "champion": "X",
          "role": "JUNGLE",
          "n_games": int,
          "proposed_axes": {"playstyle": "ganking", "gank_threat": "high"},
          "evidence": {axis: {"value": ..., "nearest_label": ..., "group_median": ...}}
        }, ...]
    """
    proposals: list[dict] = []
    # Set de tous les champions dans le dataset
    champs_in_data: set[str] = set()
    for d in (per_champ_gank, per_champ_roam, per_champ_lp, per_champ_pc):
        champs_in_data.update(d.keys())
    # Filtre : ceux sans axes curés (= pas dans traits OU traits[champ] vide)
    candidates = []
    for c in champs_in_data:
        t = traits.get(c, {})
        if t:  # a déjà des axes
            continue
        # Trouve le rôle principal (premier trouvé)
        role = None
        n_games = 0
        for r, per_champ_map in (("JUNGLE", per_champ_gank), ("MIDDLE", per_champ_roam),
                                  ("BOTTOM", per_champ_lp), ("UTILITY", per_champ_roam)):
            if c in per_champ_map and per_champ_map[c].get("n", 0) >= min_games:
                role = r
                n_games = per_champ_map[c]["n"]
                break
        if not role or n_games < min_games:
            continue
        candidates.append((c, role, n_games))

    for champ, role, n_games in candidates:
        proposed: dict = {}
        evidence: dict = {}
        if role == "JUNGLE":
            data = per_champ_gank.get(champ, {})
            v = data.get("gank_kills_mean")
            if v is not None:
                for axis in ("playstyle", "gank_threat"):
                    label_dists = {k.split("=")[1]: v for k, v in by_label_gank.items()
                                   if k.startswith(f"{axis}=")}
                    nearest = _nearest_label(v, label_dists)
                    if nearest:
                        proposed[axis] = nearest
                        evidence[axis] = {
                            "value": v,
                            "nearest_label": nearest,
                            "group_median": label_dists[nearest].get("score_median"),
                        }
        elif role in ("MIDDLE", "UTILITY"):
            data = per_champ_roam.get(champ, {})
            v = data.get("roam_mean")
            if v is not None:
                nearest = _nearest_label(v, by_label_roam)
                if nearest:
                    proposed["roam"] = nearest
                    evidence["roam"] = {
                        "value": v,
                        "nearest_label": nearest,
                        "group_median": by_label_roam[nearest].get("score_median"),
                    }
            # Et power_curve si ADC/mid
            if role in ("MIDDLE", "BOTTOM", "UTILITY"):
                pc_data = per_champ_pc.get(champ, {})
                v_pc = pc_data.get("winrate_long")
                if v_pc is not None:
                    nearest_pc = _nearest_label(v_pc, by_label_pc)
                    if nearest_pc:
                        proposed["power_curve"] = nearest_pc
                        evidence["power_curve"] = {
                            "value": v_pc,
                            "nearest_label": nearest_pc,
                            "group_median": by_label_pc[nearest_pc].get("score_median"),
                        }
        elif role == "BOTTOM":
            # ADC : lane_pattern + power_curve
            data_lp = per_champ_lp.get(champ, {})
            v = data_lp.get("early_kp_mean")
            if v is not None:
                nearest = _nearest_label(v, by_label_lp)
                if nearest:
                    proposed["lane_pattern"] = nearest
                    evidence["lane_pattern"] = {
                        "value": v,
                        "nearest_label": nearest,
                        "group_median": by_label_lp[nearest].get("score_median"),
                    }
            pc_data = per_champ_pc.get(champ, {})
            v_pc = pc_data.get("winrate_long")
            if v_pc is not None:
                nearest_pc = _nearest_label(v_pc, by_label_pc)
                if nearest_pc:
                    proposed["power_curve"] = nearest_pc
                    evidence["power_curve"] = {
                        "value": v_pc,
                        "nearest_label": nearest_pc,
                        "group_median": by_label_pc[nearest_pc].get("score_median"),
                    }

        if proposed:
            proposals.append({
                "champion": champ,
                "role": role,
                "n_games": n_games,
                "proposed_axes": proposed,
                "evidence": evidence,
            })
    return proposals


# ---------- Phase 6 : rapport structuré + validation globale ----------

# Axes et leur signal utilisé pour la validation.
# Chaque axe : (clé dans per_champ_*, group_dist_*, "above" ou "below" est bon)
_AXIS_SIGNAL = {
    "playstyle": ("per_champ_gank", "gank_kills_mean", "above"),  # ganking > farming
    "gank_threat": ("per_champ_gank", "gank_kills_mean", "above"),
    "roam": ("per_champ_roam", "roam_mean", "above"),
    "lane_pattern": ("per_champ_lp", "early_kp_mean", "above"),  # all_in > poke en KP
    "power_curve": ("per_champ_pc", "winrate_long", "above"),  # late > early en late
}


def _verdict_from_value(
    value: float | None,
    group_dist: dict | None,
    n: int,
    min_games: int,
    direction: str,
) -> str:
    """Verdict d'un champion vs distribution groupe pour un axe donné.

    direction: "above" = champion devrait être au-dessus du médian ;
               "below" = champion devrait être en-dessous.
    """
    if n < min_games or value is None or not group_dist:
        return "insufficient_data"
    med = group_dist.get("score_median")
    p25 = group_dist.get("score_p25")
    p75 = group_dist.get("score_p75")
    if med is None or p25 is None or p75 is None:
        return "insufficient_data"
    if direction == "above":
        if value > p75:
            return "above_group"
        if value < p25:
            return "below_group"
    else:  # "below"
        if value < p25:
            return "above_group"
        if value > p75:
            return "below_group"
    if abs(value - med) < 0.3 * (p75 - p25 + 0.01):
        return "validated"
    return "neutral"


def _champ_value(
    champ: str,
    axis: str,
    per_champ_gank: dict,
    per_champ_roam: dict,
    per_champ_lp: dict,
    per_champ_pc: dict,
) -> tuple[float | None, int, dict | None]:
    """Renvoie (value, n_games, group_dist) pour un champion × axe.

    group_dist est dérivée du label curé.
    """
    traits = {}  # hack : sera passé par l'appelant — géré dans build_report
    return None, 0, None


def _build_champions_block(
    traits: dict,
    per_champ_gank: dict,
    per_champ_roam: dict,
    per_champ_lp: dict,
    per_champ_pc: dict,
    by_label_gank: dict,
    by_label_roam: dict,
    by_label_lp: dict,
    by_label_pc: dict,
    min_games: int,
) -> dict:
    """Construit le bloc 'champions' du rapport : pour chaque champion avec axes
    curés, donne les stats mesurées et un verdict par axe.

    Se concentre sur les champions qui apparaissent dans la table curée ET qui
    ont assez de games. Pour les champions curés mais absents des données,
    le verdict sera 'insufficient_data'.
    """
    champions: dict = {}
    for champ, trait_axes in traits.items():
        # Récupère les stats disponibles par axe
        block: dict = {"axes": trait_axes, "verdicts": {}}
        any_stat = False

        for axis, (per_champ_key, signal, direction) in _AXIS_SIGNAL.items():
            per_champ_map = {
                "per_champ_gank": per_champ_gank,
                "per_champ_roam": per_champ_roam,
                "per_champ_lp": per_champ_lp,
                "per_champ_pc": per_champ_pc,
            }[per_champ_key]
            data = per_champ_map.get(champ)
            if not data:
                continue
            value = data.get(signal)
            n = data.get("n", 0)
            label = trait_axes.get(axis)
            if not label or label == "unknown" or value is None:
                continue
            any_stat = True
            # Récupère la distribution du groupe pour ce label
            group_dist = (
                by_label_gank.get(f"{axis}={label}")
                or by_label_roam.get(label)
                or by_label_lp.get(label)
                or by_label_pc.get(label)
            )
            verdict = _verdict_from_value(value, group_dist, n, min_games, direction)
            block["verdicts"][axis] = {
                "label": label,
                "value": value,
                "n": n,
                "verdict": verdict,
                "group_median": group_dist.get("score_median") if group_dist else None,
                "group_p25": group_dist.get("score_p25") if group_dist else None,
                "group_p75": group_dist.get("score_p75") if group_dist else None,
            }
        if any_stat:
            champions[champ] = block
    return champions


def _build_discrepancies(champions: dict) -> list[dict]:
    """Liste les verdicts 'below_group' (champion sous-performant vs son groupe).

    Un champion en dessous de P25 du groupe est suspect. Au-dessus de P75 aussi,
    mais c'est moins inquiétant (le champion est juste meilleur que la moyenne du
    label).
    """
    out: list[dict] = []
    for champ, block in champions.items():
        for axis, v in block.get("verdicts", {}).items():
            if v["verdict"] == "below_group":
                out.append({
                    "champion": champ,
                    "axis": axis,
                    "current_label": v["label"],
                    "actual_value": v["value"],
                    "group_median": v["group_median"],
                    "group_p25": v["group_p25"],
                    "evidence": (
                        f"value={v['value']:.3f} < P25 groupe={v['group_p25']:.3f} "
                        f"(médian={v['group_median']:.3f}, n_games={v['n']})"
                    ),
                })
    return out


def build_report(
    traits: dict,
    per_champ_gank: dict,
    per_champ_roam: dict,
    per_champ_lp: dict,
    per_champ_pc: dict,
    by_label_gank: dict,
    by_label_roam: dict,
    by_label_lp: dict,
    by_label_pc: dict,
    min_games: int,
    n_total_games: int,
) -> dict:
    """Construit le rapport complet (prêt à écrire en JSON).

    Returns:
        {
          "generated_at": "...",
          "config": {"min_games": int, "n_total_games": int},
          "axes": {
            "playstyle": {
              "by_label": {...},
              "n_validated": int,
              "n_below_group": int,
              "n_insufficient_data": int,
            },
            ...
          },
          "champions": {...},
          "discrepancies": [...],
        }
    """
    from datetime import datetime, timezone
    champions = _build_champions_block(
        traits, per_champ_gank, per_champ_roam, per_champ_lp, per_champ_pc,
        by_label_gank, by_label_roam, by_label_lp, by_label_pc, min_games,
    )
    discrepancies = _build_discrepancies(champions)

    # Stats par axe
    axes_block: dict = {}
    for axis, (per_champ_key, signal, _direction) in _AXIS_SIGNAL.items():
        by_label = (
            by_label_gank if per_champ_key == "per_champ_gank"
            else by_label_roam if per_champ_key == "per_champ_roam"
            else by_label_lp if per_champ_key == "per_champ_lp"
            else by_label_pc
        )
        # Compte les verdicts par axe
        n_validated = n_above = n_below = n_insuf = n_neutral = 0
        for champ, block in champions.items():
            v = block.get("verdicts", {}).get(axis)
            if not v:
                continue
            vstr = v["verdict"]
            if vstr == "validated": n_validated += 1
            elif vstr == "above_group": n_above += 1
            elif vstr == "below_group": n_below += 1
            elif vstr == "insufficient_data": n_insuf += 1
            else: n_neutral += 1
        axes_block[axis] = {
            "by_label": by_label,
            "n_champions": len(champions),
            "n_validated": n_validated,
            "n_above_group": n_above,
            "n_below_group": n_below,
            "n_neutral": n_neutral,
            "n_insufficient_data": n_insuf,
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "min_games": min_games,
            "n_total_games": n_total_games,
        },
        "axes": axes_block,
        "champions": champions,
        "discrepancies": discrepancies,
    }


def render_text(report: dict) -> str:
    """Rapport texte console, lisible."""
    s: list[str] = []
    cfg = report["config"]
    s.append(f"=== Validation data-driven champion_traits.json ===")
    s.append(f"  {cfg['n_total_games']} games analysées, "
             f"min_games={cfg['min_games']}")
    s.append("")

    for axis, block in report["axes"].items():
        s.append(f"--- {axis} ---")
        s.append(f"  validés    : {block['n_validated']}")
        s.append(f"  > P75      : {block['n_above_group']}  (champion meilleur que 75% du groupe)")
        s.append(f"  < P25      : {block['n_below_group']}  (suspect)")
        s.append(f"  neutres    : {block['n_neutral']}")
        s.append(f"  < min_games: {block['n_insufficient_data']}")
        s.append("")

    n_disc = len(report["discrepancies"])
    s.append(f"=== {n_disc} DISCREPANCIES (champions en-dessous de P25 du groupe) ===")
    if n_disc == 0:
        s.append("  ✓ Aucune discrepancy détectée.")
    else:
        for d in report["discrepancies"][:30]:  # cap à 30 pour la lisibilité
            s.append(f"  • {d['champion']:<18} {d['axis']:<14} "
                     f"label='{d['current_label']}'")
            s.append(f"      {d['evidence']}")
    if n_disc > 30:
        s.append(f"  ... et {n_disc - 30} autres (voir JSON).")
    return "\n".join(s)


if __name__ == "__main__":
    sys.exit(main())
