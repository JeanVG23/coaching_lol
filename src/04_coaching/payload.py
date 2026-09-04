"""Gold (perso + référentiel) -> payload de coaching compact et déterministe.

PUR (sauf lecture des aggregate.json). Les features ont déjà conclu : on
sélectionne ici les signaux saillants (flag `notable`) selon des seuils fixes.
N'expose QUE des métriques asymétrie-safe : positioning ⊂ COACHING_SAFE, et
les features de profondeur sont `descriptive_only` (jamais une erreur).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))  # accès src/core/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reporting"))  # accès compare
import riotlib as rl
import positioning
import game_journal as gj
import compare
import champion_profiles as cprof

LANE_SIGNALS = ["gd10", "gd14", "gd20", "csd10", "csd14"]
LANE_LABELS = {"gd10": "gold diff @10", "gd14": "gold diff @14", "gd20": "gold diff @20",
               "csd10": "cs diff @10", "csd14": "cs diff @14"}

# clé -> (label, unit, notable_threshold | None, descriptive_only)
# Profondeur (avg/max_map_depth, frac_overextended) : threshold=None + descriptive_only=True
# -> jamais `notable`, observable mais jamais prescrit.
POS_META = {
    "frac_own_lane_early": ("% lane (early)", "pct", 0.08, False),
    "frac_river_early":    ("% river (early)", "pct", 0.08, False),
    "frac_roam_mid":       ("% roam (mid)", "pct", 0.08, False),
    "frac_enemy_half":     ("% moitié ennemie", "pct", 0.08, False),
    "frac_base":           ("% en base", "pct", 0.08, False),
    "frac_overextended":   ("% over-extended", "pct", None, True),
    "avg_map_depth":       ("profondeur moy.", "u", None, True),
    "max_map_depth":       ("profondeur max", "u", None, True),
    "avg_dist_to_ally":    ("isolement (allié)", "u", 200.0, False),
    "gold_dead_time":      ("temps mort (s)", "s", 20.0, False),
    "wards_placed":        ("wards posées", "ward", 2.0, False),
    "wards_placed_early":  ("wards early", "ward", 1.0, False),
    "control_wards_placed": ("control wards", "ward", 1.0, False),
    "wards_killed":        ("wards détruites", "ward", 2.0, False),
}
# Garde-fou asymétrie : la table doit couvrir EXACTEMENT les features safe, ni plus ni moins.
assert set(POS_META) == positioning.COACHING_SAFE, \
    "POS_META doit refléter exactement positioning.COACHING_SAFE"

LOW_SAMPLE_THRESHOLD = 30

# Match-V5 expose les IDs, pas les libellés. Tables stables et volontairement
# locales : enrichir un payload ne doit jamais provoquer un appel Data Dragon.
SUMMONER_SPELLS = {
    1: "Cleanse", 3: "Exhaust", 4: "Flash", 6: "Ghost", 7: "Heal",
    11: "Smite", 12: "Teleport", 13: "Clarity", 14: "Ignite", 21: "Barrier",
    30: "To the King!", 31: "Poro Toss", 32: "Mark",
}
RUNES = {
    8005: "Press the Attack", 8008: "Lethal Tempo", 8010: "Conqueror",
    8021: "Fleet Footwork", 8112: "Electrocute", 8128: "Dark Harvest",
    9923: "Hail of Blades", 8214: "Summon Aery", 8229: "Arcane Comet",
    8230: "Phase Rush", 8351: "Glacial Augment", 8360: "Unsealed Spellbook",
    8369: "First Strike", 8437: "Grasp of the Undying", 8439: "Aftershock",
    8465: "Guardian",
    # Runes secondaires les plus fréquentes (les IDs inconnus restent exposés
    # sous forme ``rune_<id>`` plutôt que d'être silencieusement supprimés).
    8009: "Presence of Mind", 9101: "Absorb Life", 9111: "Triumph",
    9104: "Legend: Alacrity", 9105: "Legend: Haste", 9103: "Legend: Bloodline",
    8014: "Coup de Grace", 8017: "Cut Down", 8299: "Last Stand",
    8126: "Cheap Shot", 8139: "Taste of Blood", 8143: "Sudden Impact",
    8136: "Zombie Ward", 8120: "Ghost Poro", 8138: "Eyeball Collection",
    8135: "Treasure Hunter", 8134: "Relentless Hunter", 8105: "Ultimate Hunter",
    8224: "Nullifying Orb", 8226: "Manaflow Band", 8275: "Nimbus Cloak",
    8210: "Transcendence", 8234: "Celerity", 8233: "Absolute Focus",
    8237: "Scorch", 8232: "Waterwalking", 8236: "Gathering Storm",
    8446: "Demolish", 8463: "Font of Life", 8401: "Shield Bash",
    8429: "Conditioning", 8444: "Second Wind", 8473: "Bone Plating",
    8451: "Overgrowth", 8453: "Revitalize", 8242: "Unflinching",
    8306: "Hextech Flashtraption", 8304: "Magical Footwear", 8321: "Cash Back",
    8313: "Triple Tonic", 8345: "Biscuit Delivery", 8347: "Cosmic Insight",
    8410: "Approach Velocity", 8316: "Jack of All Trades",
}


def _lane_signals(mf: dict, rf: dict) -> list[dict]:
    out = []
    lane_me, lane_rf = mf.get("lane", {}), rf.get("lane", {})
    for key in LANE_SIGNALS:
        you, ref = lane_me.get(key), lane_rf.get(key)
        if you is None or ref is None:
            continue
        delta = you - ref
        unit = "cs" if key.startswith("cs") else "g"
        notable = abs(delta) >= 2 if unit == "cs" else abs(delta) > 150
        out.append({"group": "lane", "key": key, "label": LANE_LABELS[key],
                    "you": you, "ref": ref, "delta": delta, "unit": unit,
                    "notable": notable})
    return out


def _pos_signals(mf: dict, rf: dict) -> list[dict]:
    out = []
    pos_me, pos_rf = mf.get("positioning", {}), rf.get("positioning", {})
    for key in sorted(POS_META):
        you, ref = pos_me.get(key), pos_rf.get(key)
        if you is None or ref is None:
            continue
        label, unit, thr, descriptive = POS_META[key]
        delta = round(you - ref, 4)
        notable = (thr is not None) and abs(delta) >= thr
        sig = {"group": "positioning", "key": key, "label": label,
               "you": you, "ref": ref, "delta": delta, "unit": unit,
               "notable": notable}
        if descriptive:
            sig["descriptive_only"] = True
        out.append(sig)
    return out


def _zone_phase_signals(mf: dict, rf: dict, top: int = 5) -> list[dict]:
    me_zp, rf_zp = mf.get("by_zone_phase", {}), rf.get("by_zone_phase", {})
    rows = []
    # `sorted` et non l'ordre d'un set : à delta égal, l'ordre du set dépend du
    # hachage (donc du PYTHONHASHSEED, donc du run) et le TypeScript, lui, part de
    # clés triées. Deux payloads différents pour la même game, et une divergence de
    # parité que seul un ex aequo révèle.
    for key in sorted(set(me_zp) | set(rf_zp)):
        you, ref = me_zp.get(key, 0.0), rf_zp.get(key, 0.0)
        delta = round(you - ref, 4)
        rows.append({"group": "deaths_zone_phase", "key": key,
                     "label": f"morts {key}", "you": you, "ref": ref,
                     "delta": delta, "unit": "pct", "notable": delta >= 0.08})
    rows.sort(key=lambda s: s["delta"], reverse=True)   # où tu sur-meurs d'abord
    return rows[:top]


def _gold_state_signals(mf: dict, rf: dict) -> list[dict]:
    me_gs, rf_gs = mf.get("death_gold_state", {}), rf.get("death_gold_state", {})
    labels = {"ahead": "morts en avance", "even": "morts à égalité", "behind": "morts en retard"}
    out = []
    for key, label in labels.items():
        you, ref = me_gs.get(key), rf_gs.get(key)
        if you is None or ref is None:
            continue
        delta = round(you - ref, 4)
        out.append({"group": "death_gold_state", "key": key, "label": label,
                    "you": you, "ref": ref, "delta": delta, "unit": "pct",
                    "notable": abs(delta) >= 0.10})
    return out


def _load(gold_dir: Path, kind: str, name: str, scope: str) -> dict:
    path = gold_dir / kind / name / scope / "aggregate.json"
    if not path.exists():
        raise FileNotFoundError(f"gold manquant : {path}")
    return json.loads(path.read_text())


def _game_review_causes(reviews: list[dict], scope: str, limit: int = 20) -> list[dict]:
    """Map : sorties par-game -> causes qualitatives, sans preuves chiffrées.

    Les ``evidence`` et match_ids sont volontairement retirés. Ils viennent d'un
    LLM et ne doivent jamais devenir des chiffres citables dans la synthèse.
    """
    eligible = []
    for record in reviews:
        if record.get("kind") != "game":
            continue
        meta = (record.get("payload") or {}).get("meta") or {}
        if scope != "all" and record.get("scope", meta.get("scope")) != scope:
            continue
        review = record.get("review") or {}

        def qualitative(section: str) -> list[dict]:
            out = []
            for insight in review.get(section) or []:
                if not isinstance(insight, dict):
                    continue
                point, cause = insight.get("point"), insight.get("cause")
                if isinstance(point, str) and isinstance(cause, str) and cause.strip():
                    out.append({"point": point, "cause": cause})
            return out

        strengths, mistakes = qualitative("strengths"), qualitative("mistakes")
        if not strengths and not mistakes:
            continue
        eligible.append({
            "ts": record.get("ts") or "",
            "champion": meta.get("champion"),
            "outcome": "win" if meta.get("win") else "loss",
            "strengths": strengths,
            "mistakes": mistakes,
        })
    eligible.sort(key=lambda row: row["ts"], reverse=True)
    return [{k: value for k, value in row.items() if k != "ts"}
            for row in eligible[:limit]]


def build(player: str, scope: str = "adc", target: str = "challenger",
          outcome: str = "loss", gold_dir=None, game_reviews=None,
          review_limit: int = 20) -> dict:
    gold_dir = Path(gold_dir) if gold_dir is not None else rl.gold_dir()
    me = _load(gold_dir, rl.KIND_PERSONAL, player, scope)
    ref = _load(gold_dir, rl.KIND_REF, target, scope)
    mf, rf = me[outcome], ref[outcome]

    meta = {
        "player": player, "scope": scope, "target": target,
        "outcome_focus": outcome, "patch": me.get("patch", "?"),
        "n_games_me": me["n_games"], "n_games_ref": ref["n_games"],
        "winrate_me": me["winrate"],
        "low_sample": me["n_games"] < LOW_SAMPLE_THRESHOLD,
        "deaths_per_game": {oc: {"you": me[oc]["deaths_per_game"],
                                 "ref": ref[oc]["deaths_per_game"]}
                            for oc in ("overall", "win", "loss")},
    }
    signals = (_lane_signals(mf, rf) + _pos_signals(mf, rf)
               + _zone_phase_signals(mf, rf) + _gold_state_signals(mf, rf))

    context = {}
    for axis in ("lane_pattern", "gank_exposure"):
        cb = compare.context_benchmark(me, ref, axis, outcome)
        if cb:
            context[axis] = cb

    out = {"meta": meta, "signals": signals, "context": context}
    causes = _game_review_causes(game_reviews or [], scope, review_limit)
    if causes:
        out["game_review_causes"] = causes
        meta["n_game_reviews_used"] = len(causes)
    return out


# --- Payload par-game : journal ancré + repères référentiel -------------------

def _resolve_recall_items(recall: dict, catalog: dict) -> dict:
    """item_ids bruts -> items {nom, coût} ; ids bruts jamais exposés au LLM."""
    out = {k: v for k, v in recall.items() if k != "item_ids"}
    items = [catalog[i] for i in recall.get("item_ids", []) if i in catalog]
    if items:
        out["items"] = items
    return out


def _named_id(value, catalog: dict[int, str], prefix: str) -> dict | None:
    if not isinstance(value, int) or value <= 0:
        return None
    return {"id": value, "name": catalog.get(value, f"{prefix}_{value}")}


def _participant_matchup(participant: dict, item_catalog: dict) -> dict:
    """Ce que le scoreboard/champ select montrait sur un joueur de lane."""
    spells = [_named_id(participant.get(key), SUMMONER_SPELLS, "spell")
              for key in ("summoner1Id", "summoner2Id")]
    styles = ((participant.get("perks") or {}).get("styles") or [])
    primary = ((styles[0].get("selections") or []) if styles else [])
    secondary = ((styles[1].get("selections") or []) if len(styles) > 1 else [])
    keystone_id = primary[0].get("perk") if primary else None
    secondary_ids = [selection.get("perk") for selection in secondary[:2]]
    item_ids = [participant.get(f"item{i}") for i in range(6)]
    build = [item_catalog[item_id] for item_id in item_ids
             if isinstance(item_id, int) and item_id > 0 and item_id in item_catalog]
    out = {
        "champion": participant.get("championName"),
        "summoner_spells": [spell for spell in spells if spell],
        "keystone": _named_id(keystone_id, RUNES, "rune"),
        "secondary_runes": [rune for rune in
                            (_named_id(rid, RUNES, "rune") for rid in secondary_ids)
                            if rune],
        "final_build": build,
    }
    return out


def _matchup_context(match: dict, puuid: str, item_catalog: dict) -> dict | None:
    pid = rl.participant_id(match, puuid)
    if pid is None:
        return None
    participants = match.get("info", {}).get("participants") or []
    if pid > len(participants):
        return None
    me = participants[pid - 1]
    role = me.get("teamPosition") or ""
    enemy_pid = (rl.find_pid(match, team=rl.enemy_team_of(me.get("teamId")), role=role)
                 if role and me.get("teamId") in (100, 200) else None)
    if enemy_pid is None or enemy_pid > len(participants):
        return None
    opponent = participants[enemy_pid - 1]
    return {
        "lane_opponent": opponent.get("championName"),
        "player": _participant_matchup(me, item_catalog),
        "opponent": _participant_matchup(opponent, item_catalog),
    }


def _next_purchase(recall: dict) -> dict | None:
    items = recall.get("items") or []
    if not items:
        return None
    positive_costs = [item.get("cost") for item in items
                      if isinstance(item, dict)
                      and isinstance(item.get("cost"), (int, float))
                      and item["cost"] > 0]
    out = {"clock": recall.get("clock"), "items": items}
    if positive_costs:
        out["cheapest_item_cost"] = min(positive_costs)
    return out


def _attach_next_purchases(deaths: list[dict], recalls: list[dict]) -> list[dict]:
    """Lie chaque mort à la première visite de shop qui la suit.

    C'est le garde-fou mécanique du cas 1 268 g avant une B.F. Sword à
    1 300 g : le modèle n'a plus à reconstruire le lien entre deux listes.
    """
    out = []
    for death in deaths:
        enriched = dict(death)
        following = next((recall for recall in recalls
                          if recall.get("t_ms", -1) >= death.get("t_ms", 0)), None)
        purchase = _next_purchase(following) if following else None
        if purchase:
            enriched["next_purchase"] = purchase
        out.append(enriched)
    return out


def filter_scope(records: list[dict], scope: str) -> list[dict]:
    """Sous-liste des records du scope, ordre du fichier préservé.

    Délègue à `rl.filter_scope` (table `ROLE_SCOPES`, les 5 rôles) : la table locale
    `_SCOPE_ROLE = {"adc": "BOTTOM"}` gelait le coaching par-game sur l'ADC, et un
    `--scope mid` retombait silencieusement sur le filtre « nom de champion ».
    """
    return rl.filter_scope(records, scope)


def _personal_records(player: str, silver_dir: Path) -> list[dict]:
    path = silver_dir / rl.KIND_PERSONAL / player / "games.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"silver perso manquant : {path}")
    return rl.read_jsonl(path)


def _select_game(records: list[dict], scope: str, match_id: str | None) -> dict:
    if match_id is not None:
        rec = next((r for r in records if r.get("match_id") == match_id), None)
        if rec is None:
            raise FileNotFoundError(f"game {match_id} absente du silver perso")
        return rec
    records = filter_scope(records, scope)
    if not records:
        raise FileNotFoundError(f"aucune game du scope {scope} dans le silver perso")
    return records[-1]           # la plus récente du scope


def build_game(player: str, match_id: str | None = None, scope: str = "adc",
               target: str = "challenger", gold_dir=None, silver_dir=None,
               load_raw=None, records=None, ref=None) -> dict:
    """Journal d'UNE game + repères référentiel à issue égale -> payload par-game.

    `records` et `ref` sont injectables : en lot (`coach.py --game-batch N`) le silver
    perso et l'agrégat référentiel sont identiques pour toutes les games, les relire
    par game coûtait N parses redondants (même motif que `load_raw`).
    """
    gold_dir = Path(gold_dir) if gold_dir is not None else rl.gold_dir()
    silver_dir = Path(silver_dir) if silver_dir is not None else rl.silver_dir()
    load_raw = load_raw if load_raw is not None else rl._read_raw

    if records is None:
        records = _personal_records(player, silver_dir)
    rec = _select_game(records, scope, match_id)
    mid = rec["match_id"]
    match, timeline = load_raw(f"{mid}_match"), load_raw(f"{mid}_timeline")
    if match is None or timeline is None:
        raise FileNotFoundError(f"raw manquant pour {mid}")
    journal = gj.game_journal(match, timeline, rec["puuid"])
    if journal is None:
        raise FileNotFoundError(f"game {mid} hors Faille de l'invocateur")

    if ref is None:
        ref = _load(gold_dir, rl.KIND_REF, target, scope)
    outcome = "win" if journal["win"] else "loss"
    rf = ref[outcome]
    meta = {"player": player, "scope": scope, "target": target, "kind": "game",
            **{k: journal[k] for k in ("match_id", "champion", "opponent", "role",
                                       "win", "duration_min", "kda", "patch")}}
    benchmarks = {
        # Repères challenger à ISSUE ÉGALE (médianes agrégées) : contexte de
        # comparaison, jamais « tu aurais dû savoir ».
        "outcome": outcome,
        "n_games_ref": ref["n_games"],
        "deaths_per_game": rf.get("deaths_per_game"),
        "death_zone_phase": rf.get("by_zone_phase", {}),
        "death_gold_state": rf.get("death_gold_state", {}),
    }
    catalog = cprof.load_items()
    recalls = [_resolve_recall_items(r, catalog) for r in journal["recalls"]]
    deaths = _attach_next_purchases(journal["deaths"], recalls)
    out = {"meta": meta,
           "journal": {"deaths": deaths, "recalls": recalls},
           "benchmarks": benchmarks}
    comp = rec.get("comp")
    matchup = _matchup_context(match, rec["puuid"], catalog)
    if comp or matchup:
        # Champ select + scoreboard = informations visibles (asymétrie-safe).
        out["context"] = {}
        if comp:
            out["context"].update({"comp": comp, **cprof.derive_context(comp)})
        if matchup:
            out["context"]["matchup"] = matchup
    return out
