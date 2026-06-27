# Spec — Collecte de référentiels (benchmarks multi-rangs)

Date : 2026-06-27
Statut : approuvé

## Objectif

Constituer des jeux de données de référence (benchmarks) à partir de joueurs d'autres
rangs (Diamond, Master, Grandmaster, Challenger) pour situer les performances de Spadzze
(low Master, ~300 LP). Trois granularités par rang : tous rôles (`all`), ADC (`adc`),
champion (`zeri`).

But final : comparer `personal/spadzze/<scope>` vs `referentiel/<rank>/<scope>` et sortir
les écarts actionnables (« BOT-early 37% vs 22% en challenger → ton écart principal »).

## Architecture médaillon (raw → silver → gold)

```
data/
  raw/                          # JSON API brut, immuable, partagé par matchId
    <matchId>_match.json
    <matchId>_timeline.json
  silver/                       # nettoyé/normalisé : 1 ligne JSONL = 1 game
    referentiel/<rank>/games.jsonl
    referentiel/<rank>/sources.json     # puuids samplés, patch, date, compteurs
    personal/<player>/games.jsonl
  gold/                         # agrégats prêts conso (benchmarks comparables)
    referentiel/<rank>/<scope>/aggregate.json
    personal/<player>/<scope>/aggregate.json
```

- **raw** = source de vérité brute, immuable, cache partagé (idempotence/resume).
- **silver** = transformation : extraction des morts contextualisées, typage, filtrage
  SR (mapId 11) + patch courant, dédup par matchId.
- **gold** = agrégation par scope, schéma identique perso ↔ référentiel pour comparaison.

## Modules

- `riotlib.py` *(refactor partagé)* : `RiotClient` (+ méthodes ligues), `approx_zone`,
  `phase_of`, `patch_of`, `get_match_timeline` (cache raw), `extract_game` (silver),
  `aggregate` (gold), `filter_scope`, constantes de chemins. `phase1_pull.py` et
  `aggregate_games.py` sont refactorés pour l'importer (fin de la duplication).
- `build_referential.py` *(nouveau)* : collecte les référentiels par rang.
- `compare.py` *(nouveau)* : compare une slice perso à une slice référentiel.

## Flux build_referential

Pour chaque rang ∈ `[challenger, grandmaster, master, diamond]` :
1. ~25 puuids : apex via `league-v4 .../{apex}leagues/by-queue/RANKED_SOLO_5x5` ;
   diamond via `league-exp-v4 entries/RANKED_SOLO_5x5/DIAMOND/{I..IV}`.
2. Par joueur : `match-v5 ids` (queue=420, ~30 ids).
3. Par match (dédup cache `raw/` + set vu) : fetch, garder SR + patch courant, ~20/joueur.
4. silver : `games.jsonl` + `sources.json`.
5. gold : 3 slices (`all`, `adc`, `zeri`) → `aggregate.json`.

Patch courant : major.minor le plus récent observé (ex. `16.13`), surchargeable `--patch`.

## Schéma silver (games.jsonl, 1 ligne/game)

```json
{ "match_id": "EUW1_x", "rank": "challenger", "patch": "16.13",
  "champion": "Zeri", "role": "BOTTOM", "win": true, "queue": 420,
  "deaths": [ {"minute": 8, "phase": "early", "zone": "BOT",
               "killer_role": "BOTTOM", "killer_champ": "Caitlyn"} ] }
```

## Schéma gold (aggregate.json, comparable)

```json
{ "scope": "adc", "rank": "challenger", "patch": "16.13",
  "n_games": 98, "deaths_per_game": 3.1, "winrate": 0.55,
  "by_zone": {"BOT": 0.22, ...}, "by_phase": {"early": 0.38, ...},
  "by_killer_role": {...}, "by_zone_phase": {"BOT|early": 0.18, ...},
  "raw_counts": { ... } }
```

`filter_scope` : `all`=tout ; `adc`=role BOTTOM ; sinon nom de champion (Zeri).

## Robustesse / périmètre

- Idempotent & resumable via cache `raw/`.
- Skip propre : puuid absent des entrées ligue, timeline indispo, non-SR, hors patch.
- YAGNI : pas de DuckDB (JSONL/JSON), pas d'async, pas de fetch Zeri dédié (slice du pool).
- Limite assumée : slice Zeri fine (~10-40/rang) → `n_games` affiché + alerte si trop faible.
- Données dans `data/` (gitignoré) ; puuids tiers = données publiques, non versionnées.

## Hors scope (plus tard)

- DuckDB/Parquet sur le gold.
- Sampling Zeri ciblé pour densifier la slice champion.
- Features enrichies (gold/level diff à la mort, solo vs teamfight).
