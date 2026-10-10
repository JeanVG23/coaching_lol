"""Gold (perso + référentiel) -> payload de coaching compact et déterministe.

PUR (sauf lecture des aggregate.json). Les features ont déjà conclu : on
sélectionne ici les signaux saillants (flag `notable`) selon des seuils fixes.
N'expose QUE des métriques asymétrie-safe : positioning ⊂ COACHING_SAFE, et
les features de profondeur sont `descriptive_only` (jamais une erreur).
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
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
    "frac_enemy_half":     ("% enemy half", "pct", 0.08, False),
    "frac_base":           ("% at base", "pct", 0.08, False),
    "frac_overextended":   ("% over-extended", "pct", None, True),
    "avg_map_depth":       ("average map depth", "u", None, True),
    "max_map_depth":       ("maximum map depth", "u", None, True),
    "avg_dist_to_ally":    ("ally isolation", "u", 200.0, False),
    "gold_dead_time":      ("dead time (s)", "s", 20.0, False),
    "wards_placed":        ("wards placed", "ward", 2.0, False),
    "wards_placed_early":  ("wards early", "ward", 1.0, False),
    "control_wards_placed": ("control wards", "ward", 1.0, False),
    "wards_killed":        ("wards cleared", "ward", 2.0, False),
}
# Garde-fou asymétrie : la table doit couvrir EXACTEMENT les features safe, ni plus ni moins.
assert set(POS_META) == positioning.COACHING_SAFE, \
    "POS_META doit refléter exactement positioning.COACHING_SAFE"

LOW_SAMPLE_THRESHOLD = 30
GAME_BUNDLE_SCHEMA_VERSION = 1
GAME_PAYLOAD_BUNDLE_MAX_BYTES = 20 * 1024 * 1024

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
                     "label": f"deaths {key}", "you": you, "ref": ref,
                     "delta": delta, "unit": "pct", "notable": delta >= 0.08})
    rows.sort(key=lambda s: s["delta"], reverse=True)   # où tu sur-meurs d'abord
    return rows[:top]


def _gold_state_signals(mf: dict, rf: dict) -> list[dict]:
    me_gs, rf_gs = mf.get("death_gold_state", {}), rf.get("death_gold_state", {})
    labels = {"ahead": "deaths while ahead", "even": "deaths while even", "behind": "deaths while behind"}
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
    data = rl.read_json(path)
    if data is None:
        raise FileNotFoundError(f"missing gold data: {path}")
    return data


def _review_matches_scope(record: dict, scope: str) -> bool:
    """Une review par-game appartient à un scope global ou de rôle.

    Les anciennes reviews n'ont pas toujours ``meta.role`` : leur ``record.scope``
    reste un fallback de compatibilité pour les scopes de rôle.
    """
    meta = (record.get("payload") or {}).get("meta") or {}
    wanted = scope.lower()
    if wanted not in rl.ROLE_SCOPES:
        return False
    if rl.filter_scope([{"role": meta.get("role")}], scope):
        return True
    return str(record.get("scope") or meta.get("scope") or "").lower() == wanted


def _game_review_sample(reviews: list[dict], scope: str,
                        max_per_outcome: int = 2) -> dict:
    """Sélection qualitative bornée avec compteurs disponibles et utilisés.

    Les ``evidence`` et match_ids sont volontairement retirés des ``causes``. Ils
    viennent d'un LLM et ne doivent jamais devenir des chiffres citables dans la
    synthèse.
    """
    if max_per_outcome < 1:
        raise ValueError("max_per_outcome doit être >= 1")

    eligible = []
    for record in reviews:
        if record.get("kind") != "game" or not _review_matches_scope(record, scope):
            continue
        meta = (record.get("payload") or {}).get("meta") or {}
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
        if strengths or mistakes:
            eligible.append({
                "ts": record.get("ts") or "",
                "match_id": record.get("match_id") or meta.get("match_id") or "",
                "champion": meta.get("champion"),
                "outcome": "win" if meta.get("win") else "loss",
                "strengths": strengths,
                "mistakes": mistakes,
            })

    eligible.sort(key=lambda row: (row["ts"], row["match_id"]), reverse=True)
    # Une régénération conserve l'ancien run, mais ne transforme pas une partie en
    # plusieurs observations qualitatives. La plus récente gagne.
    seen_matches: set[str] = set()
    deduped = []
    for row in eligible:
        identity = row["match_id"] or f"legacy:{row['ts']}"
        if identity not in seen_matches:
            seen_matches.add(identity)
            deduped.append(row)
    eligible = deduped
    wins = [row for row in eligible if row["outcome"] == "win"]
    losses = [row for row in eligible if row["outcome"] == "loss"]
    if not eligible:
        mode, selected = "none", []
    elif wins and losses:
        take_each = min(len(wins), len(losses), max_per_outcome)
        mode = "balanced"
        selected = wins[:take_each] + losses[:take_each]
        selected.sort(key=lambda row: (row["ts"], row["match_id"]), reverse=True)
    else:
        mode, selected = "unbalanced", eligible[:1]

    causes = [{k: value for k, value in row.items() if k not in ("ts", "match_id")}
              for row in selected]
    used_wins = sum(1 for row in causes if row["outcome"] == "win")
    used_losses = len(causes) - used_wins
    return {
        "mode": mode,
        "available": {"total": len(eligible), "wins": len(wins), "losses": len(losses)},
        "used": {"total": len(causes), "wins": used_wins, "losses": used_losses},
        "causes": causes,
    }


def build(player: str, scope: str = "adc", target: str = "challenger",
          outcome: str = "loss", gold_dir=None, game_reviews=None,
          max_reviews_per_outcome: int = 2) -> dict:
    scope = scope.lower()
    if scope not in rl.ROLE_SCOPES:
        raise ValueError(f"scope de benchmark inconnu : {scope}")
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
    sample = _game_review_sample(game_reviews or [], scope, max_reviews_per_outcome)
    causes = sample["causes"]
    available, used = sample["available"], sample["used"]
    meta.update({
        "qualitative_mode": sample["mode"],
        "n_game_reviews_available": available["total"],
        "n_game_reviews_available_wins": available["wins"],
        "n_game_reviews_available_losses": available["losses"],
        "n_game_reviews_used": used["total"],
        "n_game_reviews_used_wins": used["wins"],
        "n_game_reviews_used_losses": used["losses"],
        # Compatibilité UI/payload historiques ; supprimer après migration.
        "n_game_reviews_wins": available["wins"],
        "n_game_reviews_losses": available["losses"],
        "unbalanced_causes": sample["mode"] == "unbalanced",
    })
    if causes:
        out["game_review_causes"] = causes
    return out


# --- Payload par-game : journal ancré + repères référentiel -------------------

# Les motifs d'indisponibilité d'un payload par-game sont PUBLIÉS dans KV. Les
# dériver du type d'exception plutôt que du texte français du message : une
# reformulation de message reclasserait sinon silencieusement toutes les entrées.
class RawMissing(FileNotFoundError):
    """Le match ou sa timeline ne sont pas dans le cache raw local."""


class BenchmarkMissing(FileNotFoundError):
    """Aucun agrégat référentiel exploitable pour ce scope."""


class GameNotEligible(FileNotFoundError):
    """La game existe mais sort du périmètre coaché (hors Faille…)."""



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

    Délègue à `rl.filter_scope` (table `ROLE_SCOPES`, les 5 rôles) afin que la
    sélection CLI et les agrégats partagent exactement la même sémantique.
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
               load_raw=None, records=None, ref=None, item_catalog=None,
               record=None) -> dict:
    """Journal d'UNE game + repères référentiel à issue égale -> payload par-game.

    `records` et `ref` sont injectables : en lot (`coach.py --game-batch N`) le silver
    perso et l'agrégat référentiel sont identiques pour toutes les games, les relire
    par game coûtait N parses redondants (même motif que `load_raw`).

    `record` court-circuite la sélection quand l'appelant tient déjà la ligne silver
    (`build_game_bundle`) : la retrouver par `match_id` rebalayait tout l'historique
    à chaque game.
    """
    scope = scope.lower()
    if scope not in rl.ROLE_SCOPES:
        raise ValueError(f"scope de benchmark inconnu : {scope}")
    gold_dir = Path(gold_dir) if gold_dir is not None else rl.gold_dir()
    silver_dir = Path(silver_dir) if silver_dir is not None else rl.silver_dir()
    load_raw = load_raw if load_raw is not None else rl._read_raw

    if record is None:
        if records is None:
            records = _personal_records(player, silver_dir)
        record = _select_game(records, scope, match_id)
    mid = record["match_id"]
    match, timeline = load_raw(f"{mid}_match"), load_raw(f"{mid}_timeline")
    if match is None or timeline is None:
        raise RawMissing(f"raw manquant pour {mid}")
    # Le catalogue est resolu une fois : il sert au journal (notion d'objet
    # fini, donc les blocs de spike) ET a la resolution des noms d'items.
    catalog = cprof.load_items() if item_catalog is None else item_catalog
    journal = gj.game_journal(match, timeline, record["puuid"], items=catalog)
    if journal is None:
        raise GameNotEligible(f"game {mid} hors Faille de l'invocateur")

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
    recalls = [_resolve_recall_items(r, catalog) for r in journal["recalls"]]
    deaths = _attach_next_purchases(journal["deaths"], recalls)
    out = {"meta": meta,
           "journal": {"deaths": deaths, "recalls": recalls},
           "benchmarks": benchmarks}
    comp = record.get("comp")
    matchup = _matchup_context(match, record["puuid"], catalog)
    if comp or matchup:
        # Champ select + scoreboard = informations visibles (asymétrie-safe).
        out["context"] = {}
        if comp:
            out["context"].update({"comp": comp, **cprof.derive_context(comp)})
        if matchup:
            out["context"]["matchup"] = matchup
    return out


_ROLE_TO_SCOPE = {role: scope for scope, role in rl.ROLE_SCOPES.items() if role}


def _game_benchmark_scope(record: dict, target: str, gold_dir: Path,
                          cache: dict, load_ref=None) -> tuple[str, dict]:
    """Rôle si disponible, sinon global ; retourne l'agrégat lu.

    `cache` est le mémo des agrégats déjà lus pour CE bundle (`target` y est
    constant, il ne fait donc pas partie de la clé) : sans lui le même
    référentiel serait reparsé une fois par game.
    """
    role_scope = _ROLE_TO_SCOPE.get(record.get("role"), "all")
    for scope in dict.fromkeys((role_scope, "all")):
        if scope not in cache:
            try:
                cache[scope] = (load_ref(scope) if load_ref is not None else
                                _load(gold_dir, rl.KIND_REF, target, scope))
            except FileNotFoundError:
                cache[scope] = None
        ref = cache[scope]
        if ref and ref.get("n_games", 0) > 0:
            return scope, ref
    raise BenchmarkMissing(
        f"{target} benchmark missing for {role_scope}")


_REASONS = {RawMissing: "raw_missing",
            BenchmarkMissing: "benchmark_missing",
            GameNotEligible: "not_eligible"}


def build_game_bundle(player: str, records: list[dict] | None = None,
                      target: str = "challenger", max_games: int = 50,
                      gold_dir=None, silver_dir=None, load_raw=None,
                      item_catalog=None, now=None, load_ref=None,
                      existing: dict | None = None) -> dict:
    """Payloads unitaires nettoyés prêts à servir depuis KV.

    ``existing`` rend la reconstruction incrémentale : les entrées compatibles
    encore dans la fenêtre sont conservées, tandis que les absences et les
    indisponibilités sont retentées via ``load_raw`` / ``load_ref``. Le service
    d'ingestion peut ainsi réparer un bundle sans relire 50 timelines dans R2.
    """
    if max_games < 1:
        raise ValueError("max_games doit être >= 1")
    gold = Path(gold_dir) if gold_dir is not None else rl.gold_dir()
    silver = Path(silver_dir) if silver_dir is not None else rl.silver_dir()
    rows = records if records is not None else _personal_records(player, silver)
    def recency(record):
        try:
            sequence = int(str(record.get("match_id") or "").rsplit("_", 1)[-1])
        except ValueError:
            sequence = 0
        return record.get("game_ts") or 0, sequence, str(record.get("match_id") or "")

    selected = sorted(rows, key=recency, reverse=True)[:max_games]
    catalog = cprof.load_items() if item_catalog is None else item_catalog
    raw_loader = load_raw if load_raw is not None else rl._read_raw
    items, unavailable, ref_cache = {}, [], {}
    reusable = {}
    if (isinstance(existing, dict)
            and existing.get("target") == target
            and existing.get("schema_version", GAME_BUNDLE_SCHEMA_VERSION)
                == GAME_BUNDLE_SCHEMA_VERSION
            and isinstance(existing.get("items"), dict)):
        reusable = existing["items"]
    for record in selected:
        match_id = record.get("match_id")
        if not match_id:
            continue
        cached = reusable.get(match_id)
        if (isinstance(cached, dict) and isinstance(cached.get("payload"), dict)
                and cached.get("payload_hash") and cached.get("benchmark_scope")):
            items[match_id] = cached
            continue
        try:
            benchmark_scope, ref = _game_benchmark_scope(
                record, target, gold, ref_cache, load_ref=load_ref,
            )
            game_payload = build_game(
                player, match_id=match_id, scope=benchmark_scope, target=target,
                gold_dir=gold, silver_dir=silver, load_raw=raw_loader,
                records=rows, ref=ref, item_catalog=catalog, record=record,
            )
        except (RawMissing, BenchmarkMissing, GameNotEligible) as error:
            unavailable.append({"match_id": match_id, "reason": _REASONS[type(error)]})
            continue
        canonical = json.dumps(game_payload, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"))
        items[match_id] = {
            "payload_hash": hashlib.sha256(canonical.encode()).hexdigest()[:12],
            "benchmark_scope": benchmark_scope,
            "payload": game_payload,
        }
    timestamp = now() if now is not None else datetime.now(timezone.utc).isoformat()
    return {"schema_version": GAME_BUNDLE_SCHEMA_VERSION,
            "generated_at": timestamp, "target": target, "max_games": max_games,
            "items": items, "unavailable": unavailable}


def encode_game_bundle(bundle: dict,
                       max_bytes: int = GAME_PAYLOAD_BUNDLE_MAX_BYTES) -> str:
    """Sérialise un bundle et applique la même limite pour tous les producteurs."""
    encoded = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    size = len(encoded.encode("utf-8"))
    if size > max_bytes:
        raise RuntimeError(f"bundle coaching > {max_bytes} octets ({size} octets)")
    return encoded
