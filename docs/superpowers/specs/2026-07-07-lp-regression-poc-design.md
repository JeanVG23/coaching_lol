# Design — POC régression LP (Master→GM→Challenger)

> Date : 2026-07-07. Statut : validé en brainstorming, prêt pour le plan d'implémentation.

## Contexte & motivation

Le modèle per-player actuel (`data/05_model/*_player_highelo.pkl`) classe en binaire
high (GM+Chall) vs low (Master+Diamond), **AUC_cv purgée 0.6354** sur 982 joueurs
(491/491). Ce dataset équilibré jette 533 joueurs low-elo qualifiés (≥15 games ADC)
via `n_min = min(len(pos), len(neg))` dans `build_player_dataset.py` — 1024 joueurs
low disponibles pour seulement 491 gardés. Le vrai facteur limitant est le pool
high-elo (chall+GM), lui-même plafonné par la taille réelle de ces ligues sur EUW
(~300 challenger + ~1000 GM, dont une fraction seulement joue ADC) : scraper plus
côté EUW a un rendement décroissant, et élargir à d'autres régions introduirait un
risque de confusion (dynamiques LP différentes, patch non synchronisé).

Piste alternative explorée : Master, GrandMaster et Challenger partagent la **même
échelle de LP continue, sans division ni reset** (contrairement à Diamond qui a des
divisions I-IV avec LP resetté à chaque division). Le LP réel est donc un proxy de
skill bien plus fin que le label de tier discret actuel — un Master à 190 LP est
proche de la GM, un Master à 0 LP vient de promouvoir, et le label binaire actuel les
traite identiquement. Si ce signal existe, il **résout aussi le problème de pool** :
plus besoin d'équilibrer des classes, donc plus de balance-cap, donc tous les joueurs
qualifiés Master/GM/Chall (1278, cf. `dataset_report.py`) deviennent utilisables sans
scraping supplémentaire.

Objectif de ce POC : **tester si le LP réel porte un signal prédictif au-delà de ce
que le modèle binaire capture déjà**, avant d'envisager une implémentation en
production. Travail confiné à `poc/`, n'affecte aucun artefact ni chemin de code prod
(`web/backend/ml_rank.py` continue de servir le modèle binaire existant, inchangé).

## Décisions actées en brainstorming

1. **Scope** : POC uniquement (`poc/script/` + `poc/output/`), pas d'implémentation
   prod dans ce spec. Le modèle binaire existant n'est ni remplacé ni modifié.
2. **Diamond exclu entièrement** — pas de tentative de normalisation LP par division
   (hypothèse de conversion non validée, irait à l'encontre de l'objectif du POC qui
   est d'isoler un signal LP propre). Diamond reste couvert par le modèle `dia_chall`
   existant, inchangé.
3. **Décalage temporel LP ignoré pour le POC** — le LP est récupéré *maintenant*,
   alors que les games datent d'il y a jusqu'à 13 jours (médiane 6 jours). Pas de
   correction de drift ; documenté comme limite connue dans la sortie du script. Le
   filtre "joueur toujours présent dans `apex_league` aujourd'hui" (cf. §Architecture)
   élimine déjà gratuitement les cas de changement de tier depuis la collecte.
4. **Métrique headline : Spearman** (corrélation de rang), pas R²/RMSE — robuste à
   l'échelle non linéaire du LP, comparable philosophiquement à l'AUC actuel. RMSE
   reporté en complément, sans effort supplémentaire notable.
5. **Un seul modèle (XGBoost regressor)**, pas l'ensemble à 3 (xgb/rf/ebm) du
   pipeline binaire — le POC répond à une question binaire ("le signal existe-t-il et
   dépasse-t-il le plafond ~0.65 actuel ?"), pas à produire un livrable final avec
   SHAP. L'ensemble complet sera fait dans une vraie implémentation si le POC est
   concluant.
6. **Purged CV réutilisée** de `train_player_ensemble.py` (adaptée : `KFold` au lieu
   de `StratifiedKFold`, `y` étant continu). Sans cette purge, ~37 % des games
   partagées entre les 2 ADC d'une même game (features en miroir) gonfleraient
   artificiellement le Spearman — invaliderait la conclusion même du POC.
7. **Couverture complète du pool (1278 joueurs)**, pas d'échantillon — rendu gratuit
   par la découverte d'architecture ci-dessous (3 appels API au lieu de 1278).

## Découverte d'architecture : `apex_league` est un fetch en masse

`RiotClient.apex_league(tier)` (déjà utilisé par `build_referential.py`) retourne la
**liste complète** des entrées d'un tier apex (`puuid` + `leaguePoints` pour chaque
joueur actuellement classé) en **un seul appel** par tier. Récupérer le LP courant de
nos 1278 joueurs qualifiés ne coûte donc que **3 appels API** (challenger, GM,
master), pas 1278 appels individuels via `entries_by_puuid` — élimine le compromis
temps/couverture initialement anticipé.

## Architecture

```
poc/script/fetch_apex_lp.py          (NOUVEAU, seul script qui touche l'API — 3 appels)
      apex_league("challenger"|"grandmaster"|"master") ──>
      {puuid: {tier, leaguePoints}} pour tous les joueurs actuellement classés
      ──> poc/output/apex_lp.json

poc/script/train_lp_regression.py    (NOUVEAU, 0 appel API, itérable librement)
      data/04_dataset/adc_player_dataset.parquet (existant, ≥15 games qualifiés)
      + poc/output/apex_lp.json
      ──filtre rank∈{master,gm,chall} ET puuid présent dans apex_lp.json──>
      ──purged CV (KFold) + XGBRegressor, y=leaguePoints──>
      ──Spearman pooled + Spearman within-tier (master/GM/chall séparément) + RMSE──>
      poc/output/lp_regression_metrics.json
```

### `poc/script/fetch_apex_lp.py`

- Instancie `RiotClient` (mêmes conventions que les autres scripts collection :
  `.env` pour la clé, `--region` si besoin).
- Appelle `apex_league("challenger")`, `apex_league("grandmaster")`,
  `apex_league("master")`.
- Construit `{puuid: {"tier": tier, "leaguePoints": int}}` — en cas de doublon
  improbable entre tiers, dernier tier itéré gagne (ne devrait pas arriver, un joueur
  n'apparaît que dans un seul tier apex à la fois).
- Écrit `poc/output/apex_lp.json`. Log : nombre d'entrées par tier, nombre de puuids
  au total.
- Script relançable indépendamment pour rafraîchir le LP sans retoucher au modèle.

### `poc/script/train_lp_regression.py`

- Charge `adc_player_dataset.parquet`, filtre `rank ∈ {master, grandmaster,
  challenger}`.
- Charge `apex_lp.json`, ne garde que les joueurs dont le `puuid` y est présent —
  filtre gratuit des joueurs ayant changé de tier depuis la collecte (cf. décision 3).
- Log : nombre de joueurs avant/après filtre LP, par tier.
- `y = leaguePoints` (depuis `apex_lp.json`), `X = ml_features.FEATURES` (mêmes 39
  features que le modèle binaire, réutilisées telles quelles).
- **Purged CV** : adapte `purged_train_features` de `train_player_ensemble.py` — même
  logique (recalcul des agrégats de train en excluant les matchs partagés avec la
  val), `KFold` (pas `StratifiedKFold`, `y` continu) sur les joueurs.
- Un `XGBRegressor` par fold (hyperparamètres repris du modèle binaire :
  `max_depth=3`, `n_estimators=300`, `min_child_weight=5`, à ajuster seulement si
  divergence flagrante).
- Prédictions out-of-fold poolées sur tous les folds.
- **Deux niveaux de métrique** :
  - **Pooled** : Spearman entre LP prédit et LP réel, tous tiers confondus —
    comparable à l'AUC actuel.
  - **Within-tier** : Spearman calculé séparément sur le sous-ensemble master, puis
    GM, puis challenger. C'est le test décisif : le LP varie mécaniquement par tier
    (un Challenger a par définition un LP plus haut qu'un Master), donc un bon score
    pooled pourrait juste redécouvrir la frontière de tier déjà connue. Le score
    within-tier isole si le modèle discrimine une vraie granularité de skill
    au-delà du tier.
  - RMSE pooled en complément (pas de RMSE within-tier, échantillons trop petits par
    tier pour être informatifs au-delà du Spearman).
- Écrit `poc/output/lp_regression_metrics.json` : `spearman_pooled`, `spearman_by_tier`
  (dict par tier + n), `rmse_pooled`, `n_players_total`, `n_players_by_tier`,
  `n_dropped_no_lp` (joueurs qualifiés mais absents de `apex_lp.json`).

## Ce qui ne change pas

- `web/backend/ml_rank.py`, les modèles `*_player_highelo.pkl`/`*_highelo.pkl`
  (binaire), `player_metrics.json`, `player_rank_calibration.json` — aucun n'est lu,
  écrit ou modifié par ce POC au-delà de la lecture de `adc_player_dataset.parquet`
  en entrée (lecture seule).
- `build_player_dataset.py`, `train_player_ensemble.py`, `calibrate_player_rank.py` —
  inchangés.

## Séquencement

1. `poc/script/fetch_apex_lp.py` → vérifier `poc/output/apex_lp.json` (nombre
   d'entrées par tier plausible : ~300 challenger, quelques centaines à ~1000 GM,
   plusieurs milliers master).
2. `poc/script/train_lp_regression.py` → inspecter `lp_regression_metrics.json`.
3. Lecture des résultats avant toute suite : si `spearman_by_tier` est proche de 0
   dans au moins 2 des 3 tiers, le signal LP n'apporte rien au-delà du tier connu —
   conclusion du POC = négatif, pas d'implémentation prod. Si `spearman_by_tier` est
   notablement > 0 dans au moins master (le tier le plus peuplé), signal prometteur →
   discussion d'une vraie implémentation (ensemble complet, gestion du drift temporel,
   câblage éventuel dans `ml_rank.py`).

## Critères de succès du POC

- Les deux scripts tournent de bout en bout et produisent `apex_lp.json` +
  `lp_regression_metrics.json`.
- `lp_regression_metrics.json` permet de trancher sans ambiguïté entre "signal
  within-tier réel" et "juste une redécouverte du tier" — c'est la question posée par
  ce POC, pas un score absolu à atteindre.

## Hors scope (différé)

- Toute implémentation prod (nouveau modèle en parallèle du binaire, câblage
  `ml_rank.py`) — dépend du résultat de ce POC.
- Correction du décalage temporel LP (snapshot LP à la collecte plutôt qu'au moment
  du POC).
- Diamond (normalisation LP par division).
- Ensemble complet (xgb+rf+ebm) et analyse SHAP sur la cible LP.
- Extension multi-région (évoquée en amont du brainstorming, écartée pour l'instant :
  risque de confusion, et ce POC teste d'abord si le levier LP suffit sans toucher au
  pool géographique).
