# Enrichissement events du transformer séquentiel — Design (étape 2)

> Date : 2026-07-18. Branche : `research/sequence-events` (depuis `master`).
> Suite du spec `2026-07-18-sequence-transformer-design.md` (étape 1 livrée :
> dia_chall 0.645, high_elo 0.546, delta_ssl −0.0195). Ce spec réalise l'« étape 2
> d'enrichissement » que le spec d'origine explicitait comme différée (§5 et §Pièges).

## But

Élargir le state vector v1 `[40, 20]` → v2 `[40, 27]` (+7 canaux events binaires par
minute), ré-entraîner le transformer supervisé et le SSL sur le même protocole CV purgé,
et comparer les AUC à v1. **Question de recherche, falsifiable : les events discrets
ajoutent-ils du signal temporel que l'agrégat (mean/p10/p90) détruit ?**

Le spec d'origine actait que v1 teste « ce que les agrégats résument déjà » et que « le
signal à forte valeur (trades, all-ins, roams, timing de recall) vit dans les events
discrets, différés en étape 2 ». Ce spec réalise cette étape 2.

## Décisions de design

### D1 — Schéma d'encodage : canaux empilés (channel-stacking)

On ajoute des canaux binaires/occurrence par frame minute à côté des 20 continues,
plutôt que d'intercaler des tokens d'events ou un two-stream.

- Forme `sequences` : `[N, 40, 27]` = 20 continues (v1) + 7 binaires (v2).
- Le transformer est **inchangé sauf la projection d'entrée** `d_in=27` (déjà
  paramétrée sur `SequenceEncoder`/`SequenceClassifier`/`ReconstructHead`).
- **Comparatif apples-to-apples vs v1 conservé** : même architecture, même CV purgé,
  même seed, même budget d'epochs. Le delta AUC v2 − v1 mesure uniquement l'apport des
  canaux events. C'est la raison du choix : on garde un comparatif falsifiable et propre.

### D2 — 7 canaux events (scope K=7)

Tous **COACHING_SAFE** : information que le joueur avait (sa propre mort, l'annonce de
la mort adverse, les timers d'objectifs = HUD public). Aucun proxy fog, aucune info
inférée. L'asymétrie du spec d'origine (§Principes, §game_journal) est respectée.

| Canal | Définition | Source timeline |
|---|---|---|
| `self_death_m` | 0/1 — le ciblé meurt dans la minute m | `CHAMPION_KILL` avec `victimId==pid`, `t∈[m·60000,(m+1)·60000[` |
| `opp_death_m` | 0/1 — l'ADC adverse meurt dans la minute m (visible, annoncé) | `CHAMPION_KILL` avec `victimId==opp_pid` |
| `self_recall_m` | 0/1 — une visite de shop démarre dans la minute m | cluster d'`ITEM_PURCHASED` (réuse la logique de `game_journal._recalls`) |
| `drake_up` | 0/1 — drake respawné (up) à la fin de la minute m | réuse `game_journal.OBJECTIVES["DRAGON"]` + `_objective_kills` |
| `baron_up` | 0/1 — baron up à la fin de la minute m | `game_journal.OBJECTIVES["BARON_NASHOR"]` |
| `is_ganked` | 0/1 — mort du ciblé par gank jungle dans la minute m | death + `enemy_jungle_pid ∈ (killer ∪ assisters)` |
| `is_solo_death` | 0/1 — mort solo (0 assists) du ciblé dans la minute m | death + `len(assistingParticipantIds)==0` |

`is_ganked` et `is_solo_death` ne sont non-zéro que les minutes où `self_death_m=1`
(flags de contexte de mort). Une minute peut avoir `self_death_m=1` avec les deux flags
à 0 (mort en teamfight / 2v2 avec assists hors jungle) — c'est attendu.

Occurrences binaires (pas de comptes) : les morts/recalls sont rares (~6–8 morts/game
sur 40 minutes), le binaire suffit à v2. Si un joueur meurt plusieurs fois dans la
même minute (respawn > 1 min, quasi-impossible avant très late game), on écrase à 1.

### D3 — Standardisation : deux blocs

- **20 cols continues** : z-score train-only par fold (comme v1, non négociable — un
  null par non-standardisation serait un artefact d'optimisation, pas une réponse de
  recherche).
- **7 cols binaires** : laissées brutes `[0,1]`. Justification : les binaires n'ont pas
  le problème d'échelle (type `totalGold` ~15000 dans la même projection que position
  [0,1]) qui motivait la standardisation v1 ; z-scorer des binaires rares gonflerait
  les 1 rares en ~+6 sans bénéfice.

Implémentation : `sequence_data.standardize_fit` / `standardize_apply` gagnent un
paramètre `bin_cols` (indices des colonnes à laisser brutes). Défaut `None` =
comportement v1 actuel (toutes cols standardisées) → **v1 reproductible inchangé**.
v2 passe `bin_cols=range(20, 27)`. Backward-compatible : aucune régression sur les
tests v1 existants.

### D4 — Extraction (DRY, 0 API)

`build_sequence_dataset.py` gagne un helper :

```python
def _event_channels(match, timeline, pid, opp_pid, my_team, enemy_jungle_pid) -> np.ndarray:
    """-> [40, 7] bool/float32. Canaux events binaires par minute, COACHING_SAFE."""
```

- Réutilise `game_journal.OBJECTIVES` (timers drake/baron) et `_objective_kills` pour
  `drake_up`/`baron_up` (pas de redéfinition des timers).
- Réutilise la détection killer/assisters de `game_journal._deaths` (logique de
  `enemy_jungle_pid` et `assistingParticipantIds`) — mais en version **allégée** : on
  ne calcule que le bucket minute + les flags gank/solo, **pas** `gold_state`,
  `consequences`, `unspent_gold` (coût inutile sur 43–95k games × 2 ADC).
- `build_sequence` concatène `self_s + opp_s + _diffs(...) + event_channels[minute]`
  → frame 27-d. `mask` inchangé (validité des frames continue ; les canaux events
  héritent de la validité de la frame minute correspondante).

### D5 — Transformer / dataset

- `SequenceEncoder`/`SequenceClassifier`/`ReconstructHead` : déjà paramétrés en `d_in`
  → v2 passe `d_in=27`. **Aucune autre modif d'architecture** (4 couches, d_model=64,
  nhead=4, masked-mean-pool — inchangés ; l'ablation CLS/attention-pool reste différée).
- `adc_sequence_dataset.npz` : `sequences` devient `[N, 40, 27]` (overwrite du fichier).
- `train_sequence_model.py` et `pretrain_sequence_model.py` déduisent `d_in` de
  `data["sequences"].shape[-1]` (pas de hardcode 20) → résilients à v1/v2.
- `ReconstructHead` reconstruit désormais les 27-d (les 7 binaires inclus) → le prétexte
  SSL mask-and-reconstruct porte enfin sur du signal **non-lisse** (la reconstruction
  d'un `self_death_m=1` masqué est un vrai défi, pas une interpolation).

### D6 — Métriques : fichier séparé

v2 écrit dans `data/05_model/sequence_metrics_v2.json` (fichier distinct), **préserve**
le record v1 `sequence_metrics.json` pour le delta de comparaison. Même structure
(`tasks` + `ssl` + `params`). Le verdict se lit sur le delta v2 − v1 sur `dia_chall`.

### D7 — Branche parallèle, zéro perturbation

- Branche `research/sequence-events` depuis `master`.
- Fichiers touchés : `src/02_data_science/sequence_model.py`, `sequence_data.py`,
  `train_sequence_model.py`, `pretrain_sequence_model.py`,
  `src/01_data_engineering/build_sequence_dataset.py`, `tests/test_sequence*.py`,
  `tests/test_build_sequence_dataset.py`, et la mise à jour CLAUDE.md en fin de plan.
- **Aucun overlap** avec le pending work non-committé de l'auteur (web/shap/densify/
  pyproject) → pas d'entanglement. Pipeline tabulaire + coach web non touchés.
- Mac M4 CPU, `--device cpu` (MPS ne supporte pas `aten::_nested_tensor_from_mask_left_aligned`
  pour le `src_key_padding_mask` — inchangé vs étape 1).
- Pas de push vers `origin`.

## Tests (TDD)

- `tests/test_build_sequence_dataset.py` — timeline synthétique :
  - une mort gank à 4:42 (`victimId==pid`, `killerId==enemy_jungle_pid`, 0 assists) →
    `self_death_m[4]=1`, `is_ganked[4]=1`, `is_solo_death[4]=0` ;
  - drake tué à 7:00 → `drake_up[6]=1` (up avant le kill), `drake_up[9]=0`
    (down après kill), `drake_up[12]=1` (respawn 5 min après le kill) ;
  - shape `[40, 27]` ; les 20 continues sont identiques à la sortie v1 (régression).
- `tests/test_sequence_data.py` — `standardize_fit(bin_cols=range(20,27))` laisse les
  binaires bruts (std=1 / mean=0 appliqués sans effet) et standardise les continues.
- `tests/test_train_sequence_model.py`, `tests/test_pretrain_sequence_model.py`,
  `tests/test_sequence_model.py` — `_mini` passe à 27-d, `d_in` propagé ; le signal
  injecté migre sur un canal event (ex. `self_death_m` à 1 pour les challenger) pour
  que le smoke garde un AUC > 0.5 (sinon le signal injecté sur `frame 10 feature 2`
  reste valable aussi — au choix, l'important est qu'un canal porte le signal).

Convention environnementale inchangée : jamais `pytest tests/` complet (cohabitation
torch + xgboost → SIGSEGV libomp). La baseline tabulaire reste RF + EBM (xgb exclu),
comme en étape 1.

## Re-training & verdict

1. `poetry run python3 src/01_data_engineering/build_sequence_dataset.py` (regen 27-d).
2. `poetry run python3 src/02_data_science/train_sequence_model.py --device cpu --epochs 30`.
3. `poetry run python3 src/02_data_science/pretrain_sequence_model.py` (pretrain 15 /
   finetune 30, mêmes valeurs qu'étape 1 pour un delta apples-to-apples).
4. Lecture dans `sequence_metrics_v2.json` :
   - **dia_chall v2** : > 0.645 → thèse renforcée (events ajoutent du signal que
     l'agrégat rate). ≈ 0.645 → les canaux events à résolution frame n'aident pas
     (piste : predictive SSL étape 3, ou event-tokens intercalés).
   - **high_elo v2** : ≈ 0.546 attendu (bruit de label, non interprétable — cf. §Pièges
     du spec d'origine). Ne pas conclure.
   - **delta_ssl v2** : > 0 → le prétexte non-lisse fonctionne enfin (SSL justifié).
     ≈ 0 → le mask-and-reconstruct reste faible même avec events (prétexte prédictif
     = étape 3).
5. Mise à jour CLAUDE.md : bloc « Recherche — transformer séquentiel + SSL » enrichi
   (étape 2 livrée + verdict + chiffres v2).

## Critère de succès

- **dia_chall v2 > 0.645** : les events ajoutent du signal que l'agrégat rate → la
  thèse « l'ordre temporel porte du signal que les médianes tuent » est démontrée pour
  de bon, et le transformer se justifie au-delà du +0.012 de v1.
- **≈ 0.645** : les canaux events à résolution frame (60 s) ne suffisent pas → réorienter
  vers un prétexte SSL prédictif (étape 3) ou des tokens d'events intercalés (sub-minute).
- **delta_ssl v2 > 0** : le SSL mask-and-reconstruct décolle enfin avec du signal
  non-lisse — bonus, non requis pour valider l'étape 2.

## Hors scope (différé)

- Wards (visibilité, asymétrie à soigner : wards ennemies = fog), skill-order (faible
  pour le rang), opp_recall (inféré, non visible directement), building kills (contexte
  déjà capté indirectement) — non inclus en v2 pour garder le comparatif propre et
  COACHING_SAFE.
- Étape 3 : prétexte SSL prédictif (future-event), event-tokens intercalés, scaling
  3080 Ti, multi-tâche (rang + win + LP), embedding réutilisé pour le coaching.
- HF Transformers / modèles pré-entraînés génériques — écarté (trop lourd, moins
  pédagogique, hors thèse).