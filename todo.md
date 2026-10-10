# TODO — RiftSense

> État au 2026-09-21. Ce fichier ne conserve que les travaux encore actionnables.
> Les résultats historiques et fonctionnalités terminées sont documentés dans
> `CLAUDE.md`, `docs/MODEL_CARD.md` et `src/04_coaching/README.md`.

## ✅ Priorité 0 — incrément LLM enrichi validé (2026-09-07)

Lot frais de 10 reviews par-game généré sous la cohorte de prompt `350f7c404b5b`
(`kimi-k2.6`), évalué automatiquement AVANT toute annotation humaine :

- **Éval automatique** : ancrage des nombres 97,2 % (252 nombres, 92,5 % exacts),
  horodatages 99,3 % (134 horloges), 0 violation d'asymétrie
  (`grounding.py --player spadzze --kind game --prompt-version 350f7c404b5b`) ;
  contrefactuels 3 runs, sensibilité 1,00, ancrage 93,1 %
  (`data/07_coaching/spadzze/eval/counterfactual.json`).
- **Annotation humaine** : 10/10 reviews annotées, 100 % de `mistakes` utiles (40 items),
  contre 96,4 % pour la cohorte antérieure `none` (12 reviews). Critère produit
  (≥70 % sur ≥10 reviews) atteint.
- Les deux cohortes restent deux populations **observées** : games, payload et prompt ont
  changé ensemble, l'écart n'est pas attribuable au seul prompt.

Reste facultatif : `model_ab.py --player spadzze --n 3` (`kimi-k2.6` vs `glm-5.3`) si le
choix du modèle doit être rouvert. Le payload y est contrôlé, la mesure porte sur ancrage,
sensibilité, schéma et latence.

## ⚡ Priorité 0 bis — terminer la validation du coaching enrichi

Design validé, amendé et implémenté en TDD :
`docs/superpowers/specs/2026-09-05-coaching-recalls-tracking-categories-design.md`
(recalls jugés aux CS perdues et au spike adverse, jungle tracking, erreurs découpées et
titrées par catégorie). Le payload, le schéma strict, le prompt, l'évaluation par
catégorie, le contrefactuel `no_opponent_spike` et l'affichage web sont en place.

- [x] **Exécuter le plan en TDD** jusqu'à l'intégration frontend et aux contrôles
  automatiques. Première empreinte par-game : `e1f3a4cb13da`.
- [x] **Compléter le lot de 10 reviews** sous cette empreinte et le comparer à la cohorte
  `350f7c404b5b` comme troisième population observée (le seuil ≥70 % est déjà atteint : la
  question devient la précision des recalls et la lisibilité des erreurs, pas l'utilité).
  Le lot mélange 1 review `kimi-k2.6` et 9 `deepseek-v4.1-flash`, sous le même prompt.
  DeepSeek : 9/9 sans retry, latence médiane 94,6 s (772 s cumulées) ; Kimi : 1 retry,
  850,5 s. Les 294 horodatages sont exacts et l'asymétrie compte 0 violation, mais
  l'ancrage numérique strict n'est que de 81,9 % (484/591) : **cohorte rejetée** sous le
  seuil de 90 %. Les 107 rejets DeepSeek correspondent à des valeurs du payload citées
  sans unité explicite (`Stormrazor (3200)`, `Yasuo 550`, `19 vs 17`) ; ne pas rendre le
  contrôleur permissif en les rapprochant de n'importe quel nombre du journal.
- [x] **Exiger une unité ou un nom de champ pour chaque chiffre compact** et régénérer les
  mêmes 10 games avec `deepseek-v4.1-flash`. Cohorte homogène `07692894c681` : 10/10 sans
  retry, ancrage numérique strict **91,8 %** (669/729, dont 645 exacts), 308/308
  horodatages exacts et 0 violation d'asymétrie. Latence médiane 93,2 s, 890,9 s cumulées,
  359 128 tokens. Le parseur reconnaît désormais les séparateurs de milliers anglais
  (`3,200 gold`) sans élargir le rapprochement inter-unités ; les 60 citations restantes
  restent refusées.
- [x] **Mesurer les contrefactuels** sur 3 games × 4 perturbations avec DeepSeek : 12/12
  réponses exploitables après reprise ciblée, sensibilité **91,7 %** et grounding des
  sorties perturbées **95,7 %**. `no_deaths`, `unspent_gold_zero` et
  `no_opponent_spike` passent 3/3 ; `zone_to_top` passe 2/3. L'annotation humaine de cette
  cohorte est explicitement sautée : sa fidélité est mesurée, son utilité n'est donc pas
  nouvellement validée. **Ne pas présenter ce lot comme annoté.**
- [ ] **Décider si la cohorte peut être publiée sans nouvelle mesure
  d'utilité.** La cohorte Kimi précédente reste la seule preuve humaine (100 % de mistakes
  utiles sur 10 reviews) ; le lot DeepSeek ne valide que fidélité et sensibilité.
- [x] **Synchroniser la cohorte DeepSeek dans KV** en mode coaching-only : 10 reviews
  `07692894c681` présentes dans `riftsense:spadzze:reviews`, avec conservation d'une review
  web-only lors de la fusion. Aucun silver/gold/ML n'a été réécrit.
- [x] **Redéployer le Worker** pour activer le hash `07692894c681`. Version Cloudflare
  `b0f1bdf7-1c35-42c3-a24a-bfa3c8456c27` vérifiée via l'API publique : 10/10 reviews
  DeepSeek servies, 5 victoires + 5 défaites, 10 matchs `ready`. Les 21 anciennes analyses
  restent conservées et classées `stale`.
- [ ] **Décider du coût d'une mise à jour globale** côté site. Le verrou `CoachGate` et le
  coaching unitaire à la demande évitent la régénération de 30 games d'un clic ; il reste à
  fixer si un bouton « tout mettre à jour » existe, et sous quel plafond.

## 🔎 Priorité 1 — corriger le diagnostic ML avant de réentraîner

- [x] **Corriger le faux positif `dataset_report` 1 345 → 982.** Le split canonique contient
  l'union des datasets rang + LP (1 345 joueurs), alors que le classifieur de rang consomme
  `adc_player_dataset.parquet` (982 joueurs). `model_crosscheck` intersecte désormais
  la population du split avec les `puuid` réellement consommés par le modèle (687 train +
  148 calibration + 147 test = 982). L'alerte de dérive fallacieuse est supprimée.
- [x] **Ajouter une identité de dataset aux artefacts ML** (empreinte des `puuid`, features,
  seuil `MIN_PLAYER_GAMES`, split et fenêtre de données). Déclencher un réentraînement
  seulement si cette identité ou le protocole diffère réellement du modèle servi
  (`compute_dataset_identity` dans `dataset_split.py`, court-circuit d'idempotence
  avec flag `--force` dans `train_player_ensemble.py`).
- [ ] **Décider explicitement de la politique temporelle.** Les données actuelles viennent
  uniquement du patch 16.13 et ont environ 60–73 jours : soit le modèle est une étude figée
  sur ce patch et on le documente, soit il est servi comme produit courant et il faut définir
  un cycle collecte → rebuild → évaluation held-out → publication.

## 📊 Priorité 2 — données et objectif ML

- [ ] **Choisir une frontière de rang défendable avant de collecter davantage.**
  - Grandmaster est une population structurellement petite : ne pas créer de ticket
    « densifier GM » impossible à fermer.
  - Pour une frontière Master/GM, fusionner GM avec Master ou Challenger selon la question,
    ou collecter plusieurs régions si une estimation GM spécifique est indispensable.
  - La frontière `dia_chall` est la seule clairement séparable aujourd'hui (AUC ≈0,72 contre
    ≈0,59 pour `high_elo`) : densifier Diamond est pertinent uniquement si elle devient
    l'objectif principal.
- [ ] **Reformuler la robustesse SHAP.** SHAP explique actuellement le profil ML ; il ne doit
  pas devenir automatiquement une prescription. Mesurer la stabilité des directions entre
  folds, graines, modèles et patches, puis confronter uniquement les features
  `COACHING_SAFE` aux heuristiques et au feedback humain.

### Pas de densification à lancer actuellement

- **Équilibre binaire** : le dataset per-player est déjà équilibré à 491 high / 491 low.
  Le sujet restant est la définition de la frontière, pas un undersampling supplémentaire.

## 🔧 Priorité 3 — maintenance opportuniste

- [ ] **Mutualiser `compare.py` et `payload._*_signals`** seulement lors de la prochaine
  évolution des signaux/seuils. Le recouvrement existe, mais les consommateurs sont distincts
  et les tests de parité réduisent le risque immédiat.
- [ ] **Remplacer “industrialisation Pydantic + Parquet/DuckDB” par des tickets concrets**
  lorsqu'un défaut observé le justifie. Parquet, `04_dataset`, `05_model` et les schémas
  Pydantic du coaching sont déjà en place ; DuckDB n'est pas nécessaire au volume courant.
- [ ] **Nettoyer la documentation devenue incohérente** : volumes historiques obsolètes,
  critères d'évaluation désormais atteints et sens de `avg/max_map_depth` (valeur haute =
  marqueur plus Diamond/risqué dans le code, jamais une prescription positive).

## ✅ Acquis à ne plus remettre dans la file de travaux

- Reviews agrégées et par-game (`--game`, `--game-batch`), agents spécialisés optionnels.
- Payload enrichi : dégâts fatals, matchup, sorts/runes/build et prochain achat réel.
- Structured output, validation Pydantic, grounding, contrefactuels et feedback humain.
- Cloudflare Worker + KV + SPA, streaming SSE et chat coaching asymétrie-safe.
- Pipeline Parquet/ML/SHAP et manifeste `COACHING_SAFE` / `ML_ONLY`.
