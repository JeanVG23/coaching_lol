# Coaching LoL — Coach IA personnalisé pour League of Legends

> Projet en phase de conception (greenfield). Ce document fixe la vision, la stack
> et le schéma de données cibles. Rien n'est figé tant que l'implémentation n'a pas
> commencé.

## Vision

Construire un **coach IA centré sur le joueur** (et non sur l'équipe entière), capable
de produire un compte-rendu de fin de game qui dépasse le simple résumé de stats.

Le problème des outils existants (op.gg, u.gg, score OP, etc.) : ils s'appuient
essentiellement sur des **stats classiques** (KDA, nombre de tourelles, gold, etc.).
Résultat : des conseils pauvres et souvent faux du type « meurs moins » — alors que le
vrai conseil serait « place-toi ici plutôt que là ». On veut capturer le **positionnement
et les déplacements réels**, pas juste des agrégats.

### Principes directeurs

1. **Positionnement > stats brutes.** Une vraie carte des déplacements des 10 joueurs
   apporte beaucoup plus d'information qu'un KDA.
2. **Respect de l'asymétrie de l'information.** Le coach ne doit JAMAIS reprocher une
   décision basée sur une info que le joueur n'avait pas (ex. « tu n'aurais pas dû push,
   le jungle ennemi était botside » alors que le joueur n'avait aucune vision dessus).
   On ne raisonne, pour le coaching, que sur l'information réellement disponible au joueur.
3. **Centré sur soi.** L'objectif est de se cibler personnellement. Une frame/minute
   suffit pour un avis sur toute l'équipe ; pour soi on veut une granularité plus fine
   (cooldowns, sorts loupés ou non, etc.).
4. **Le LLM ne voit pas la vidéo brute.** Il reçoit un **journal structuré** d'événements
   et d'états déjà extraits proprement. L'extraction (API Riot + vision) fait le travail ;
   le LLM ne fait que raconter.
5. **D'abord le journal fiable, ensuite le coaching.** Premier objectif = transformer une
   game en journal structuré fiable. Le coaching vient après.
6. **Riot-first, CV-for-the-gaps.** (voir ci-dessous) La donnée structurée gratuite et
   exacte de l'API Riot est la source principale ; la computer vision ne sert qu'à
   combler ce que l'API ne fournit pas.

## Source de données : Riot-first, CV-for-the-gaps

C'est la décision d'architecture centrale. **La vision n'est PAS la source principale.**
L'API Riot fournit gratuitement, sans erreur d'OCR, l'essentiel des données :

### Ce que l'API Riot donne (colonne vertébrale)

- **Match-V5 + endpoint Timeline** (post-game) : positions (x/y) de **tous** les
  champions **toutes les 60 s**, gold/XP/items par joueur, et tous les events discrets
  (kills, objectifs, wards posées/détruites, level-ups, ordre de skill, achats).
- **Live Client Data API** (`https://127.0.0.1:2999`, pendant la game) : tes abilities,
  runes, items, gold, level, scoreboard, feed d'events — en temps réel.

### Accès API confirmé (clé production « Coach_LoL_LLMs », 39 méthodes)

Clé de **production** (limites généreuses, pas d'expiration 24 h) → pas besoin de rate
limiter agressif, juste un backoff poli sur 429. Bundle retenu :

| API | Endpoint clé | Rôle |
|-----|-------------|------|
| **account-v1** | `accounts/by-riot-id/{gameName}/{tagLine}` | Riot ID → `puuid` (porte d'entrée, routing **régional**) |
| **match-v5** | `matches/by-puuid/{puuid}/ids`, `matches/{id}`, **`matches/{id}/timeline`** | Cœur du MVP. La **timeline** (positions/60 s + events) est le joyau, routing **régional** |
| **match-v5** | `matches/by-puuid/{puuid}/replays` | Bonus Phase 2 : pont vers les replays .rofl (extraction CV à froid) |
| league-v4 | `entries/by-puuid/{puuid}` | Elo/LP pour contexte → **rend summoner-v4 inutile**, routing **plateforme** |
| lol-challenges-v1 | `player-data/{puuid}`, `challenges/percentiles` | Bonus benchmarking profil (percentiles sur centaines de métriques) |
| champion-mastery-v4 | `champion-masteries/by-puuid/{puuid}` | Profilage (main vs pick peu maîtrisé), Phase 2 |
| spectator-v5 | `active-games/by-summoner/{puuid}` | Live uniquement → Phase 2 (trigger capture) |

**Piège routing** : account-v1 + match-v5 = régional (`europe`/`americas`/`asia`) ;
league-v4 / mastery / spectator = plateforme (`euw1`…).

**À zapper** : summoner-v4 (contourné par `league-v4 by-puuid`), clash-v1, tournament-*,
lol-status-v4, champion-v3 — hors scope coach individuel.

### Ce que la CV doit combler (phase 2 seulement)

Uniquement ce que Riot ne donne pas :
- cooldowns exacts de tes sorts,
- skillshots loupés / touchés,
- micro-positionnement entre deux frames de 60 s,
- zone de caméra.

### Les deux niveaux d'information (CRITIQUE pour l'asymétrie)

Il faut distinguer formellement deux jeux de données et ne jamais les confondre :

- **« Ce que je savais »** — Live Client Data API : ne renvoie **que ce que le joueur
  voit** (fog of war respecté nativement). → C'est la **seule** base autorisée pour
  reprocher / juger une décision.
- **« Ce qui s'est réellement passé »** — Match Timeline post-game : contient l'info
  **complète** (positions ennemies même invisibles). → Sert uniquement à **labelliser
  a posteriori** (« en fait le jungle était botside »). Ne JAMAIS la présenter au LLM
  comme une connaissance qu'avait le joueur.

## Stack cible

Langage principal : **Python** (meilleur écosystème vision/ML que Node).

| Brique | Techno |
|--------|--------|
| Données de jeu (principal) | API Riot : Match-V5 + Timeline, Live Client Data API |
| Extraction frames / vidéo (phase 2) | FFmpeg |
| Traitement image / ROI / tracking (phase 2) | OpenCV, NumPy |
| OCR (phase 2) | Tesseract (robuste, gratuit) ou EasyOCR |
| Détection objets minimap (optionnel, tardif) | YOLO-like (Ultralytics/supervision) — sinon template matching + règles |
| Validation des schémas | Pydantic |
| Format de stockage | Parquet (compact, colonne) — JSONL acceptable en ingestion |
| Analytics local | DuckDB (sans serveur, idéal logs horodatés) |
| Extraction structurée locale | Ollama, **structured output JSON** (schéma imposé, température basse) |
| Synthèse stratégique finale | Petit modèle local OK pour l'extraction ; envisager un modèle plus gros (API) pour la narration nuancée |

### À éviter au début (sur-ingénierie)

- Toute computer vision tant que le MVP timeline n'est pas validé
- Scraping de vidéos YouTube — les timelines challenger de l'API Riot sont mieux et
  directement structurées
- Gros fine-tuning de LLM
- Modèle supervisé d'erreurs (pas de dataset labellisé) — heuristiques déterministes d'abord
- Base SQL lourde (Postgres) — DuckDB suffit
- « Full video understanding » de bout en bout

## Séquencement (par phases)

### Phase 1 — Coach 100 % API, zéro vision (MVP)

> Récupère tes N dernières games via Match-V5 → calcule des features macro/positionnement
> → Ollama → compte-rendu.

But : **valider l'hypothèse « le coach est-il utile ? »** avant d'investir dans le
pipeline CV (la partie dure et risquée). Si le coach basé timeline est déjà bon, on sait
où mettre la CV ensuite. S'il est mauvais, le problème vient des features, pas de la
vision — leçon apprise pour pas cher.

### Phase 2 — CV pour les trous

Ajout ciblé de la vision uniquement sur ce qui manque vraiment (cooldowns, skillshots,
micro-position, caméra). Piste maligne pour l'extraction fine sans impacter la game live :
rejouer la game depuis les **fichiers replay (.rofl)** en mode spectateur, où la Live
Client API et la caméra restent disponibles.

### Phase 3 — ML / spécialisation (si justifié)

Heuristiques → puis ML supervisé seulement une fois des labels accumulés. Pas de
fine-tuning avant d'en avoir prouvé le besoin.

## Le levier de qualité : les features, pas le LLM

La qualité du rendu final dépend à ~90 % de la **couche de features**, pas du modèle.
Le LLM ne fait que raconter ce que les features ont déjà conclu. Investir là.

- **Coaching relatif à un benchmark challenger, pas absolu.** « Tu recall à 1450 g en
  moyenne, les challengers de ton matchup à 1100 » est concret et **vérifiable** ; « recall
  plus tôt » est une opinion creuse. Les benchmarks viennent **directement des timelines
  high-elo de l'API Riot**.
- Features macro à fort signal : proximité aux objectifs avant leur spawn, timing de recall
  vs état de wave, **morts en fog vs morts en vision**, indice d'overextension (distance à
  la tour la plus proche × ennemis non visibles), gold dead time, diff de CS/XP par minute.

## Schéma de dataset

Principe : **pas « une ligne = une game »**. Au moins 4 tables/fichiers logiques.

### `games` — métadonnées globales
patch, champion, rôle, durée, résultat, side, elo approximatif.

### `states_timeline` — états échantillonnés dans le temps
`game_id`, `timestamp_ms`, `my_hp_pct`, `my_mana_pct`, `q_cd`, `w_cd`, `e_cd`, `r_cd`,
`flash_cd`, `tp_cd`, `visible_enemy_count`, `visible_on_minimap_top/mid/bot/river`,
`camera_zone`, `gold_unspent`, `wave_state_estimate`.
(Positions/gold/level viennent de la timeline Riot ; HP/mana/cooldowns/caméra = CV phase 2.)

### `events` — événements discrets
`game_id`, `timestamp_ms`, `event_type`, `actor` (self / ally / enemy_visible), `zone`,
`confidence`, `payload_json`.
Exemples d'event_type : mort, recall, objectif, spell cast, skillshot estimé raté/touché,
entrée en river, disparition de 2+ ennemis, ouverture du scoreboard.

### `reviews` — labels et résumés
erreurs détectées, bons moves, scores lane/macro/vision, résumé final généré.

## Pipeline de résumé

1. Récupération des données (Riot API en phase 1 ; + extraction CV en phase 2).
2. Conversion en événements et états structurés.
3. Agrégation en **features haut niveau** (benchmarkées challenger quand possible), par ex. :
   - « 3 pushes sans vision en side lane »
   - « 2 recalls tardifs avant drake (challengers : -350 g plus tôt) »
   - « forte discipline de reset après crash »
   - « sorts majeurs souvent lancés sans setup »
4. Envoi de ce résumé structuré au LLM.
5. Sortie LLM imposée par schéma JSON :
   - `strengths[]` — 3 points positifs
   - `mistakes[]` — 3 erreurs prioritaires
   - habitudes à corriger (2)
   - `next_focus[]` — 1 focus pour la prochaine game
   - `confidence`
   - `evidence[]`

## Pistes de coach envisagées

- **Coach champ select** — analyse des picks/bans et matchups.
- **Coach in-game / fin de game** — l'axe principal : journal agrégé, compte-rendu en fin
  de partie.
- **Overlay de lecture d'écran** — lit l'écran du joueur et convertit en données avant de
  les passer au LLM, garantissant le respect de l'asymétrie d'information. (Attention ToS
  Riot : overlay en lecture seule, aucune automatisation d'input.)

## Évaluation

Prévoir **dès le départ** une boucle de feedback (« ce conseil était-il juste / utile ? »)
— sinon impossible de savoir si le coach s'améliore. Le coaching benchmarké challenger est
intrinsèquement plus vérifiable que les opinions absolues du LLM.

## Architecture du code (médaillon, numérotée pour l'ordre du pipeline)

Code dans `src/`, données dans `data/` (couches numérotées). Lancer depuis la racine :
`python3 src/<script>.py` (Python met `src/` sur le path → `import riotlib` marche).

```
src/                                     # tout le code Python
data/
  00_static/                             # données statiques versionnées (hors data/ ignoré)
    champion_traits.json                 # table curée (power_curve/lane_pattern/playstyle/gank_threat/roam)
    ddragon/<version>/championFull.json  # cache Data Dragon (attackrange/tags), figé par version
  01_raw/                                # JSON API brut, immuable, cache partagé par matchId
  02_silver/{referentiel/<rank>,personal/<player>}/games.jsonl   # 1 ligne = 1 game nettoyée (+ comp)
  03_gold/{referentiel/<rank>,personal/<player>}/<scope>/aggregate.json   # agrégats benchmarks
```
⚠️ `data/` est gitignoré SAUF `data/00_static/champion_traits.json` (force-add : c'est de la
config source, pas de la donnée). Le cache DDragon sous `00_static/ddragon/` reste ignoré.
⚠️ Les chemins des couches sont définis dans `src/riotlib.py` (`RAW_DIR`/`SILVER_DIR`/
`GOLD_DIR`). Renommer un dossier data SANS mettre à jour le code → le code recrée l'ancien.

- **`src/riotlib.py`** — socle partagé : `RiotClient` (routing régional account/match vs
  plateforme league ; rate-limiter ~1.3s/appel), helpers (`approx_zone`, `phase_of`,
  `patch_of`), `get_match_timeline` (cache raw), `extract_game` (silver, + benchmark de lane
  + sous-objet `comp` des 6 champions botlane), `aggregate`/`write_gold` (gold, facettes
  win/loss + dimension `by_lane_context`), chemins médaillon. Importe `champion_profiles`.
- **`src/champion_profiles.py`** — identité champion : `champion_vector` (Data Dragon +
  table curée, résolution casse-insensible), `derive_context(comp)` → 2 axes coarse
  `lane_pattern` (poke/all_in/scaling/mixed/unknown, du duo ennemi) et `gank_exposure`
  (low/med/high/unknown, jungler+mid ennemis atténués par ton jungler). `fetch_ddragon`
  (one-shot, idempotent). Dégradation propre : champion inconnu → `unknown`, jamais d'erreur.
- **`src/reextract_silver.py`** — ré-extrait le silver depuis le raw caché (**0 appel API**) ;
  à relancer après toute évolution d'`extract_game` (ex. ajout du `comp`).
- **`src/list_unknown_champions.py`** — scanner : champions du silver absents de la table
  curée, triés par fréquence (pour compléter `champion_traits.json` au fil de l'eau).
- **`src/phase1_pull.py`** — spike : détail visuel d'UNE game (déplacements/minute + morts).
- **`src/aggregate_games.py`** — pipeline perso : N games → silver + gold (all/adc/zeri).
- **`src/build_referential.py`** — collecte les benchmarks par rang (league-v4/-exp-v4).
- **`src/rebuild_gold.py`** — régénère tout le gold depuis le silver, sans appel API.
- **`src/compare.py`** — livrable coaching : slice perso vs référentiels, à issue égale.
  Section **benchmark conditionné** par contexte de lane (`context_benchmark`, seuil de
  repli `MIN_CONTEXT_N=8` loggué, `unknown` exclu du bucket dominant).
- **Tests** : `tests/` (pytest), couvrent la dérivation déterministe + l'extraction comp +
  l'agrégation contextuelle. Lancer : `.venv/bin/python -m pytest tests/`.

Features clés : **facettes win/loss** (neutralise le biais d'issue), **benchmark de lane**
(gold/CS/XP diff @10/@14/@20 vs adversaire), **gold-state des morts** (avance/retard),
**contexte de matchup botlane** (lane_pattern + gank_exposure, benchmarkés à contexte égal).
Scopes : `all` · `adc` (BOTTOM) · `zeri` (champion). Filtre patch courant, SR (mapId 11),
ranked solo (queue 420). Spec : `docs/superpowers/specs/`.

Pipeline contexte (0 API) : `champion_profiles` (fetch DDragon one-shot) → `reextract_silver`
(silver + comp) → compléter `champion_traits.json` via `list_unknown_champions` → `rebuild_gold`
(+ `by_lane_context`) → `compare`. **Principe asymétrie** : le comp (info post-game complète)
sert UNIQUEMENT de contexte de benchmark (« en lane X, les challengers font Y »), jamais à
reprocher une décision sur une info cachée.

## État d'avancement

- **Phase 1 VALIDÉE** ✅ — positionnement reconstruit sans vision. Insight type :
  « 1 ennemi = 5/8 de tes morts » >> « meurs moins ».
- **Phase 1.5 — agrégation multi-games** ✅ — pattern récurrent confirmé sur 14-20 games :
  ~37% des morts ADC = BOT en early game ; l'ennemi ADC signe ~45% des morts.
- **Phase 1.6 — référentiels multi-rangs** 🚧 — **Challenger collecté** (341 games / 83 ADC,
  patch 16.13) avec features lane. Reste diamond/master/GM. `compare.py` opérationnel.
- **Premier verdict ADC** : laning = LE levier (≈ -10 à -16 CS @14 vs challenger, *toutes*
  issues) ; mauvaise gestion du retard (gold@20 en lose -1252 vs -322) ; morts = symptôme.
- Clé **dev** (throttle ~100 req/2min, attentes 429 si saturé) → rate-limiter intégré.
  `.env` (clé `RIOT_API_ID` ; pas de `RIOT_REGION` → passer `--region euw1`), `data/` ignoré.

### Prochaines étapes

1. Enrichir les **features** de morts (gold/level diff à la mort, solo vs teamfight).
2. Brancher **Ollama** (MCP `ask-ollama`, structured output) sur le diff perso↔référentiel.
3. **Benchmark Zeri** densifié (sampling champion ciblé) si la slice reste trop fine.
4. Industrialiser : DuckDB/Parquet sur le gold, modèles Pydantic.
5. Phase 2 (CV / Live Client) seulement si le coach démontre sa valeur.

## Notes de développement

- Scripts encore au stade prototype ; migration Pydantic + Parquet/DuckDB prévue à
  l'industrialisation (cf. prochaines étapes).
- Lancer la collecte : `python3 src/build_referential.py --region euw1 [--rank R] [--players N]`.
- Régénérer le gold après un changement de features : `python3 src/rebuild_gold.py`.
- Verdict : `python3 src/compare.py --scope adc --outcome {loss,win,overall}`.
- Garder la sortie LLM strictement typée (schéma JSON + validation Pydantic) pour éviter
  les résumés qui partent en vrille.
- Heuristiques déterministes (explicables, debuggables) avant tout ML supervisé.
