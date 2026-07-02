# Design — Capture Live Client Data API (v1 : collecte + matching, sans features)

> Date : 2026-07-02. Statut : validé en brainstorming, prêt pour le plan d'implémentation.

## Contexte & motivation

Le schéma `states_timeline` prévoit déjà des champs marqués « CV phase 2 » (HP/mana,
cooldowns Q/W/E/R/flash/TP) faute de mieux. Or la **Live Client Data API** de Riot
(`https://127.0.0.1:2999/liveclientdata/*`, servie localement par le client League pendant
une game) donne exactement cette donnée — cooldowns exacts, HP/mana, gold, events — en JSON
structuré, sans OCR ni erreur de vision, et en respectant **nativement** le brouillard de
guerre (elle ne renvoie que ce qui est visible sur ton écran).

Décision issue de la discussion de brainstorming : ni durcir le ML classique (plafond
d'information déjà atteint sur les features macro, AUC ~0.59 au raccord Master/GM) ni
investir sur le LLM (le goulot n'est pas la narration mais la donnée en entrée) — le
prochain levier accessible sans computer vision, c'est cette source. **Limite assumée** :
cette donnée n'est capturable qu'en live, sur les games que TU joues (ou tout joueur qui
ferait tourner ce script) — elle ne peut donc pas alimenter les modèles ML partagés
(entraînés sur un référentiel de milliers de joueurs), seulement enrichir le diagnostic
personnel côté heuristiques/coaching.

Objectif de ce spec, volontairement restreint : **collecte + stockage + rattachement au
matchId**. Aucune extraction de features, aucun branchement coaching — on regarde d'abord
la donnée brute captée sur plusieurs games avant de décider ce qu'on en tire (cohérent
avec la démarche « heuristiques d'abord, une fois la donnée vue » déjà appliquée pour le
positionnement, cf. `2026-06-30-macro-positioning-design.md`).

## Contrainte fondamentale : localhost uniquement, multi-PC

`127.0.0.1` n'est joignable que depuis la machine qui fait tourner le client League au
moment de la game — aucun contournement réseau possible. Le joueur jouant depuis plusieurs
PC (dont certains sans environnement de dev, ni VSCode, ni le repo), le script de capture
doit pouvoir tourner sur une machine où seul Python est installable :
- **Fichier unique, zéro dépendance externe** (stdlib uniquement) — pas de `pip install`
  au-delà de l'installation de Python lui-même.
- Copiable par clé USB / drive partagé, exécutable depuis une invite de commandes.
- Le rapatriement du résultat vers la machine principale se fait manuellement (copie de
  fichiers) — pas d'automatisation réseau en v1 (YAGNI : peu de PC, peu de games).

## Architecture — 3 pièces

### 1. `src/live_capture.py` — script de capture autonome

- Stdlib uniquement : `urllib.request` (contexte SSL non-vérifié pour le certificat
  auto-signé de l'API locale), `json`, `time`, `pathlib`, `signal` (Ctrl+C propre),
  `platform` (nom de machine, pour le meta).
- Boucle de poll sur `/liveclientdata/allgamedata`, intervalle configurable (défaut ~2-3 s).
- États : **attente** (endpoint ne répond pas encore — champ select) → **capture** (dès la
  première réponse, un snapshot brut par ligne écrit en JSONL) → **fin** dès N échecs
  consécutifs (game terminée) ou interruption manuelle (Ctrl+C) → fermeture propre du
  fichier dans tous les cas.
- Sorties, écrites à côté du script :
  - `<start_iso>_<champion>.jsonl` — un dump `allgamedata` complet par ligne, horodaté.
  - `<start_iso>_<champion>_meta.json` — heure de début/fin (wall clock), champion, nom de
    machine (`platform.node()`) — sert exclusivement au matching après coup.
- Usage : lancement **manuel** (`python live_capture.py`) juste avant/pendant le champ
  select ; pas de détection automatique de lancement de partie (décision de brainstorming :
  démon en arrière-plan jugé disproportionné pour le gain).
- Sur un PC secondaire : installer Python une fois, copier ce fichier unique, lancer depuis
  un terminal, puis ramener les deux fichiers de sortie sur la machine principale.

### 2. Nouvelle couche `data/01_raw_live/`

Parallèle à `01_raw` (API Riot) plutôt que mélangée dedans : source brute différente,
cycle de vie différent (dépôt manuel, pas un cache d'appel API).
- `pending/` — fichiers déposés (capturés sur n'importe quel PC), pas encore reliés à un
  matchId.
- `matched/` — une fois reliés : renommés `<matchId>_live.jsonl` +
  `<matchId>_live_meta.json`, cohérent avec la convention de cache par matchId de
  `riotlib.py` (`_read_raw`/`_write_raw`).
- Dépôt : copier manuellement les fichiers de sortie de `live_capture.py` dans
  `data/01_raw_live/pending/`.

### 3. Matching après coup

- Mode `--match` de `live_capture.py` (une fonction dédiée ; script séparé seulement si la
  logique s'avère assez volumineuse à l'implémentation pour justifier l'isolation).
- Pour chaque fichier de `pending/` :
  1. Lit le sidecar meta (champion, heure de début, heure de fin → durée).
  2. Récupère les games récentes du joueur via Match-V5 (`matches/by-puuid/{puuid}/ids` +
     détails ; réutilise le cache raw existant si déjà tiré, sinon appelle l'API).
  3. Cherche une game dont le champion + l'heure de début (tolérance ± quelques minutes) +
     la durée (tolérance ± ~1 minute) correspondent.
  4. Trouvé → déplace vers `matched/` sous `<matchId>_live.jsonl` (+ meta). Pas trouvé →
     laissé en `pending/`, relançable sans risque (idempotent — ne touche jamais un fichier
     déjà dans `matched/`).
- Tourne uniquement sur la machine principale (a besoin de la clé API + de `riotlib`) —
  jamais sur un PC secondaire.
- Ambiguïté attendue faible (peu de games/jour, champion+heure+durée déjà très
  discriminant) ; en cas d'ambiguïté avérée (plusieurs candidats plausibles), logguer un
  avertissement explicite plutôt que deviner silencieusement.

## Data flow

```
PC secondaire (ou principal)         Machine principale
────────────────────────────         ──────────────────────────────────────
live_capture.py (stdlib only)
  │ poll 127.0.0.1:2999
  ▼
<ts>_<champ>.jsonl + _meta.json
  │ copie manuelle (USB / drive)
  ▼                                   data/01_raw_live/pending/
                                        │
                                        │ live_capture.py --match (lit Match-V5)
                                        ▼
                                      data/01_raw_live/matched/<matchId>_live.jsonl
                                        (rien en aval pour l'instant — v1 s'arrête ici)
```

## Gestion des erreurs

- **Poll** : une erreur réseau isolée ne doit pas interrompre la boucle — seul un nombre de
  échecs consécutifs déclenche la fin de capture. L'avertissement SSL (certificat
  auto-signé) est supprimé une fois au démarrage, pas loggué à chaque requête.
- **Écriture** : flush après chaque snapshot, pour ne pas perdre toute la session si le PC
  plante ou si la fenêtre est fermée brutalement.
- **Interruption** (Ctrl+C) : fermeture propre du fichier + écriture de l'heure de fin dans
  le meta — jamais un fichier `.jsonl` sans meta associé.
- **Matching** : un fichier meta corrompu ou incomplet est sauté avec un log, sans jamais
  interrompre le traitement des autres fichiers en attente.

## Plan de tests

La capture elle-même dépend d'une vraie game Live Client en cours → pas de test
automatisé possible, vérification manuelle en conditions réelles (comme les autres scripts
d'ingestion du projet, ex. `phase1_pull.py`).

La logique de **matching**, elle, est une fonction pure testable :
`find_matching_game(capture_meta, candidate_games) -> match_id | None`.
`tests/test_live_capture_matching.py` (pytest), fixtures synthétiques :

| Test | Setup | Assertion |
|---|---|---|
| match exact | 1 candidat correspondant champion+heure+durée | `match_id` du candidat retourné |
| aucun candidat proche | tous les candidats hors tolérance | `None` (reste en pending) |
| ambiguïté | 2 candidats plausibles proches | comportement explicite défini (le plus proche en heure de début) + warning logué |
| bornes de tolérance | candidat juste dans / juste hors la fenêtre | bascule correcte aux limites |

## Séquencement

**Incrément unique (ce spec)** — capture + stockage + matching, rien d'autre :
1. `src/live_capture.py` : boucle de poll + écriture JSONL/meta (stdlib only).
2. `data/01_raw_live/{pending,matched}/` créés à la volée par le script.
3. Fonction de matching + tests (TDD : tests d'abord sur `find_matching_game`).
4. Vérification manuelle en conditions réelles : une game capturée sur la machine
   principale, une game capturée sur un PC secondaire puis copiée à la main — les deux
   doivent atterrir dans `matched/` après `--match`.

**Hors scope (différé, décisions explicites du brainstorming) :**
- Extraction de features depuis les snapshots (discipline de cooldowns, état HP/mana,
  etc.) — décidée après avoir regardé la donnée brute réelle sur plusieurs games.
- Branchement sur `states_timeline`/`events` (silver) ou sur le payload de coaching.
- Automatisation du transfert PC secondaire → machine principale (réseau, sync cloud) — la
  copie manuelle suffit à l'usage actuel (peu de PC, peu de games).
- Détection automatique de lancement de partie (démon en arrière-plan) — lancement manuel
  retenu en brainstorming.
- Alimentation des modèles ML partagés (XGBoost/RF/EBM) — structurellement hors de portée
  sans que le référentiel entier (des milliers de joueurs) capture aussi cette donnée ; ce
  n'est pas un objectif de ce projet.

## Critères de succès

- `live_capture.py` tourne sur un PC où seul Python est installé (aucun `pip install`),
  capture une game réelle du début à la fin sans intervention.
- Un fichier + meta copiés depuis un PC secondaire sont matchés correctement au bon
  `matchId` par `--match`.
- Tests de matching verts.
- Aucune modification du pipeline existant (silver/gold/dataset/coaching) — strictement
  additif.
