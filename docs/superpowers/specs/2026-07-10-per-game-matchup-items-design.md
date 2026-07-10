# Contexte matchup + items réels dans le payload par-game

**Date** : 2026-07-10
**Statut** : validé (brainstorm)
**Origine** : boucle de feedback (`feedback.py summary`) — objectif par-game à 50 % de
mistakes utiles (cible ≥70 %). Les verbatims des tags négatifs pointent deux trous de
features dans le payload par-game, pas des défauts de narration :

- *« je ne sais pas vraiment pourquoi je suis mort… rajouter plus de détails sur la game
  comme le match up »* → le journal par-game n'expose pas le contexte de matchup, alors
  que le silver porte déjà le `comp` et que `derive_context` existe.
- *« avec 1200 golds il faut prendre en compte que souvent les adcs doivent attendre
  1.3k golds pour avoir la BT »* → le coach reproche du gold non dépensé sans connaître
  le build réel du joueur.

Thèse projet appliquée : la qualité dépend des features, pas du LLM. On enrichit le
payload avec ce que le feedback réclame.

## Décisions produit

1. **Items réels achetés** (pas de table statique de breakpoints) : le journal expose,
   par recall, les items réellement achetés (noms + coûts via Data Dragon `item.json`).
   Grounded dans la game, asymétrie-safe (le joueur connaît ses achats).
2. **Comp + buckets** : le payload expose les noms des 7 champions (`comp`) ET les
   buckets déterministes (`lane_pattern`, `gank_exposure`). Le LLM est autorisé à
   mobiliser sa connaissance générale des champions pour expliquer un mécanisme,
   ancré sur les événements du journal. Risque assumé : erreur possible sur un
   champion récent.
3. **Architecture « journal pur + résolution payload »** : `game_journal` capture les
   ids bruts (0 dépendance Data Dragon), `champion_profiles` porte le catalogue,
   `payload.build_game` résout. Une responsabilité par couche.

## Design

### 1. `src/core/game_journal.py` — capture brute des items (module pur)

- `_recalls` capture aujourd'hui les timestamps d'`ITEM_PURCHASED` ; chaque visite
  gagne `item_ids: [int]` (ordre d'achat).
- Les événements `ITEM_UNDO` sont honorés **sur la liste chronologique des achats,
  avant clustering en visites** : chaque undo retire le dernier achat encore présent
  dont l'item correspond au `beforeId` (évite d'afficher un item annulé — cas fréquent
  achat/undo/rachat ailleurs).
- `ITEM_SOLD` ignoré en v1 (approximation documentée en tête de module).
- Aucune dépendance nouvelle ; le module reste pur et testable à froid.

### 2. `src/core/champion_profiles.py` — catalogue d'items Data Dragon

Même pattern que `championFull.json` :

- `fetch_ddragon_items(version=None)` — one-shot, idempotent, écrit
  `data/00_static/ddragon/<version>/item.json` (refresh = supprimer le fichier).
- `load_items()` — `lru_cache`, → `{item_id: {"name": str, "cost": int}}`
  (coût = `gold.total`). Fichier absent → `{}` (dégradation propre, comme
  `load_ddragon`).

### 3. `src/04_coaching/payload.py` — `build_game` enrichi

- **Items** : les `item_ids` de chaque recall du journal sont résolus en
  `items: [{"name", "cost"}]` dans le payload. Les ids bruts ne sont pas exposés au
  LLM. Id inconnu du catalogue → omis. Catalogue vide (item.json non fetché) →
  visite sans clé `items`, le reste du payload fonctionne.
- **Matchup** : le silver record (`rec`, déjà chargé par `_select_game`) porte `comp` →
  nouveau bloc payload :
  `"context": {"comp": {...}, "lane_pattern": <bucket>, "gank_exposure": <bucket>}`
  via `champion_profiles.derive_context(rec["comp"])`. Record sans `comp` (vieux
  silver) → bloc omis.

### 4. `src/04_coaching/prompt.py` — `SYSTEM_GAME` durci sur deux points

- **Règle matchup** (nouvelle) : le bloc `context` donne le champ select — information
  que le joueur avait (asymétrie-safe). Le LLM est **autorisé** à mobiliser sa
  connaissance générale des champions pour expliquer le mécanisme d'une mort
  (« Pyke = hook + engage ») dans la `cause`, toujours ancré sur un événement du
  journal — jamais inventer un événement. `lane_pattern`/`gank_exposure` restent les
  conclusions déterministes prioritaires.
- **Règle gold étendue** (actuelle règle 3 recalls) : le gold non dépensé se juge
  **relativement au prochain achat réel** — retenir du gold sous le coût d'un
  composant effectivement acheté au recall suivant (ex. 1 200 g avant une B.F. Sword
  à 1 300 g) est un choix légitime, pas une erreur.
- `render_game` inchangé (le payload JSON contient tout).

### 5. Tests (pytest, suites existantes étendues)

- Journal : capture `item_ids` par visite, honorage d'`ITEM_UNDO`.
- Catalogue : parsing `load_items` (nom + coût), fichier absent → `{}`.
- Payload : `build_game` expose `context` + items résolus ; dégradations (record sans
  `comp`, catalogue vide, id inconnu).
- Prompt : présence des nouvelles règles dans `SYSTEM_GAME`.

## Hors scope

- Le payload agrégé (`payload.build`) — inchangé.
- Le schéma Pydantic (`GameReview`/`GameInsight`) — inchangé, `cause`/`evidence`
  absorbent le nouveau contexte.
- `ITEM_SOLD`, table statique de breakpoints génériques (écartée en décision 1).
- Ré-extraction silver : aucun champ silver ne change (le `comp` existe déjà).

## Critère de succès

Après implémentation : regénérer des reviews par-game (`coach.py --game-batch`),
annoter (`feedback.py annotate --pending`) jusqu'à ≥10 reviews par-game, viser
≥70 % de mistakes utiles et la disparition des tags « je ne sais pas pourquoi »
et des faux positifs gold (« BT à 1300g »).
