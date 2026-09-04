# Model card — Coaching LoL

> Documente les modèles **réellement servis** en production sur
> `https://coaching-lol.jeanvg.fr`, leur protocole d'évaluation et leurs limites.
> Chiffres extraits de `data/05_model/*.json` (artefacts de run, pas de recopie manuelle).
> Dernière mise à jour : 2026-09-04. Runs de référence : 2026-07-07 (LP) et 2026-07-18 (rang).

## 1. En un coup d'œil

| | Modèle de rang | Régression LP |
|---|---|---|
| Tâche | classification binaire high-elo | régression du LP courant |
| Unité | **1 joueur** (≥15 games ADC) | 1 joueur apex |
| Artefacts | `{xgb,rf,ebm}_player_highelo.pkl` | `{xgb,rf,ebm}_player_lp.pkl` |
| Headline (**test held-out**) | **AUC 0.677** (n=147) | **Spearman 0.5373** (n=170) |
| Sélection des hyperparamètres | k-fold purgée sur le train | idem |
| Servi ? | oui (`src/core/ml_rank.py`) | oui, si le rang placé est apex |

Deux familles de modèles existent dans le dépôt mais **ne sont pas servies** : les
modèles per-game (`{xgb,rf,ebm}_highelo.pkl`, dépréciés le 2026-07-18) et le
transformer séquentiel (branche de recherche). Voir §7.

## 2. Usage prévu, et usage exclu

**Prévu** : estimer le rang d'un joueur ADC à partir de la régularité de ses parties, et
alimenter l'onglet Historique du site avec une confiance affichée explicitement.

**Exclu, et cette exclusion est mécanique dans le code** : aucune feature classée `ML_ONLY`
dans `src/core/positioning.py` ne peut atteindre le coaching. Ce sont des proxys de vision
(ce que le joueur ne pouvait pas savoir sur le moment) : les utiliser pour reprocher une
décision violerait l'asymétrie d'information, principe fondateur du projet. `reporting/compare.py`
porte un `assert` au chargement qui fait **crasher le process** si une feature `ML_ONLY`
fuit vers la couche coaching. Sur les 17 features de positionnement, 14 sont `COACHING_SAFE`.

**Exclu également** : la profondeur de carte (`pos_avg_map_depth`, `pos_max_map_depth`) est
`descriptive_only`. Son sens est contre-intuitif (valeur haute → rang *inférieur*), donc elle
sert de marqueur de risque, jamais de prescription.

Ce modèle n'est pas un outil de classement officiel et ne doit pas servir à évaluer
quelqu'un d'autre que soi-même.

## 3. Données

| | |
|---|---|
| Source | API Riot : Match-V5 + Timeline (aucune donnée scrapée, aucune vision) |
| Région | EUW1, file ranked solo (queue 420), Summoner's Rift (mapId 11) |
| Patch | **16.13 exclusivement** (95 390 lignes sur 95 390 du référentiel) |
| Fenêtre | parties âgées de 60,0 à 72,4 jours (médiane 65,7) au 2026-09-04 |
| Volumétrie per-game | 47 701 parties, 95 416 lignes, 14 919 joueurs |
| Répartition par rang | master 40 174, challenger 25 675, diamond 17 325, grandmaster 12 216 |
| Dataset per-player | 982 joueurs qualifiés (≥15 games ADC), équilibré **491 / 491** |

**1 ligne = 1 ADC d'une partie**, pas 1 partie. Les deux ADC de chaque partie sont
ré-extraits depuis le raw local (0 appel API), ce qui densifie le dataset d'un facteur ~2.

Le jeu est **naturellement déséquilibré** avant équilibrage (37 891 high contre 57 499 low) ;
le dataset per-player est ramené à 491/491 par cap.

Les données personnelles (26 lignes, compte `spadzze`) sont exclues de l'entraînement : elles
ne servent qu'au coaching.

## 4. Label

`high_elo` oppose **GM+Challenger** à **Master+Diamond**. C'est délibérément la frontière la
plus difficile : deux tiers adjacents, quasi indistinguables sur des features macro. Une
seconde cible, `dia_chall` (diamond contre challenger), retire le milieu et sert de contrôle :
elle répond à « le signal existe-t-il tout court ? » indépendamment de la difficulté de la
frontière. Les deux définitions vivent dans `src/core/ranks.py`, source unique.

Le rang d'un joueur présent sur plusieurs rangs est résolu au **mode**, tie-break sur le rang
le plus bas.

## 5. Protocole d'évaluation

C'est le point sur lequel ce projet est le plus exigeant, et il mérite d'être lu avant les
chiffres.

**Split canonique unique** (`data/04_dataset/split.json`, `src/core/dataset_split.py`) : par
joueur, stratifié par rang, graine fixe, 70 / 15 / 15 (train / calibration / test). Le même
fichier sert à tous les modèles, donc aucun modèle n'a jamais vu le test d'un autre.

**Purged CV.** Environ 37 % des parties des joueurs qualifiés opposent **deux joueurs du
dataset** : leurs features sont en miroir (mon `gold diff` est l'opposé du sien). Un CV naïf
laisse donc fuir de l'information entre folds. À chaque fold, les agrégats de train sont
**recalculés en excluant les parties jouées contre un joueur de validation ou du holdout**
(`purged_train_features`). Le group-CV par composantes connexes était impossible : le graphe
des parties partagées forme une composante géante à 98,7 %.

**Fuite mesurée, pas supposée** : ≈ **+0,005 d'AUC**, pour 8,7 % des parties de train purgées
et 0 joueur perdu. `audit_leakage.py` est un script dédié à ce diagnostic.

**Sélection sur le train, headline sur le test.** Les hyperparamètres sont choisis en k-fold
**sur le train uniquement** ; le chiffre publié vient du **test held-out**, jamais d'un OOF.
Le bucket de calibration reste inutilisé à ce jour (réservé à une future couche conforme).

## 6. Métriques

### 6.1 Rang (`player_metrics.json`)

| | AUC | Accuracy | n | positifs / négatifs |
|---|---|---|---|---|
| CV sur le train (purgée) | 0.5912 | 0.5619 | 687 | 344 / 343 |
| **Test held-out** | **0.677** | 0.6259 | 147 | 73 / 74 |

Par modèle en CV train : EBM 0.6107, RF 0.6034, XGBoost 0.5712.

Split : 1 345 joueurs (train 942, calibration 202, test 201).

> Le test est **plus haut** que la CV train, ce qui est inhabituel et doit être lu comme du
> bruit d'échantillonnage à n=147, pas comme une qualité. L'intervalle de confiance à ce n est
> large ; le chiffre honnête à retenir est « entre 0,59 et 0,68 selon la coupe ».

### 6.2 LP (`player_lp_metrics.json`)

| | Spearman pooled | RMSE | n |
|---|---|---|---|
| CV sur le train (purgée) | 0.4931 | 535.4 LP | 805 |
| **Test held-out** | **0.5373** | 555.1 LP | 170 |

Par tier sur le test : challenger 0.6601 (n=55), grandmaster 0.7545 (**n=11, non
interprétable**), master 0.3634 (n=104). Baseline POC à battre : 0.5028.

130 joueurs écartés faute de LP courant. LP relevé le 2026-07-07.

### 6.3 Ce que le modèle regarde

**La dispersion, pas la moyenne.** Les statistiques de dispersion (`std`, `p10`, `p90`)
concentrent **65,3 %** du signal SHAP du modèle de rang (60,3 % pour l'EBM, calculé
indépendamment), contre 20,3 % pour la moyenne. Le nombre de parties agrégées ne pèse que
0,6 %, ce qui est rassurant : le modèle ne prédit pas le rang à partir du volume de jeu.

Top 5 SHAP : `deaths_late__std`, `pos_control_wards_placed__std`,
`pos_frac_overextended__p90`, `support_deaths_early__std`, `deaths_solo__std`.

C'est le résultat le plus actionnable du projet : **le rang tient au plancher, pas à la
moyenne**. Ce n'est pas la bonne partie qui distingue, c'est la mauvaise.

## 7. Modèles présents mais non servis

**Per-game, déprécié le 2026-07-18.** `train_ensemble.py` et `calibrate_rank.py` sont arrêtés.
Prédire le rang depuis **une** partie donnait 0.63 / 0.59 selon la frontière, trop instable
pour être affiché. Code et artefacts conservés pour l'historique.

**Transformer séquentiel (recherche).** Transformer 4 couches, `d_model=64`, sur les séquences
d'états par minute, CV purgée identique au baseline.

| tâche | séquence | tabulaire | MLP contrôle | n lignes |
|---|---|---|---|---|
| `dia_chall` | **0.6448** (±0.008) | 0.6326 | 0.5295 | 42 996 |
| `high_elo` | 0.5460 (±0.005) | 0.5536 | 0.5044 | 95 378 |

Lecture : sur la frontière séparable, la représentation séquentielle capte un signal que
l'agrégat manque (+0,012). Sur `high_elo`, séquence et tabulaire sont dans le bruit.

## 8. Résultats négatifs assumés

Ils sont documentés ici volontairement : les taire donnerait une image fausse du projet.

- **SSL sans effet.** Pré-entraînement mask-and-reconstruct : `delta_ssl = -0.0195`. Une v2
  ajoutant des canaux d'événements binaires (censés être moins lisses) fait **pire**
  (`delta_ssl = -0.0252`, `dia_chall` 0.6437 contre 0.6448). Le prétexte MSE est faible sur des
  signaux lisses (gold monotone, position continue → quasi-interpolation). Ce n'est pas un
  verdict sur le SSL en général : un prétexte prédictif reste à tester.
- **`high_elo` plafonne à ≈ 0.589.** Master contre GM n'est pas séparable sur features macro.
  La calibration le montre crûment : proba moyenne master **0.4888**, grandmaster **0.4868**.
  L'ordre est **inversé**, les deux tiers sont confondus. Seuls diamond (0.4219) et challenger
  (0.5263) se détachent.
- **Plus de parties par joueur n'aide pas.** Sweep du cap N : 15 → 0.588, 20 → 0.619, 25 →
  0.628, **30 → 0.635 (pic)**, 40 → 0.624, 50 → 0.599. Au-delà de ~30, du bruit. Les leviers
  sont le pool, les features ou la frontière, pas N.

## 9. Limites assumées

- **Transfert de rang aux deux ADC.** Le rang d'une partie est le rang de collecte du joueur
  ciblé, transféré **aux deux ADC** en supposant un MMR homogène dans le lobby. Vrai en solo
  queue high-elo, faux si on descend en elo. L'ADC adverse n'a donc pas son rang réel mesuré.
- **Grandmaster sous-représenté.** 106 joueurs GM dans le split (train 74, calibration 16,
  test 16), et seulement 11 dans le test LP. Toutes les métriques GM (dont le Spearman 0.7545
  ci-dessus) sont **du bruit** et ne doivent pas être citées seules.
- **Drift entre le modèle servi et le dataset courant.** Le modèle en production a été entraîné
  sur 1 345 joueurs ; le dataset actuel en compte 982 après densification. `dataset_report.py`
  signale l'écart (`drift: true`). Le ré-entraînement est dû.
- **Un seul patch, une seule région.** Tout vient du patch 16.13 sur EUW. Aucune garantie de
  transfert à un autre patch (les équilibrages changent l'économie du jeu) ni à une autre région.
- **Drift temporel du label LP.** Le LP a été relevé le 2026-07-07 alors que les parties
  s'étalent jusqu'à ~13 jours autour. Le label bouge pendant la fenêtre observée.
- **Un seul joueur en coaching.** La boucle d'évaluation LLM repose sur les annotations d'une
  seule personne (l'auteur), sur 17 analyses. C'est une mesure, pas une validation externe.
- **Aucune donnée démographique.** Le dataset ne contient que des identifiants de jeu
  pseudonymes (`puuid`) et des statistiques de partie. Aucune analyse d'équité par sous-groupe
  n'est donc possible, ni pertinente ici.

## 10. Évaluation de la couche LLM

Le coach n'est pas évalué comme le modèle ML. Trois familles indépendantes :

1. **Annotation humaine** (`feedback.py`) : critère du projet **≥70 % de mistakes utiles sur
   ≥10 analyses par-partie**. Atteint le 2026-09-04 : **96 % sur 12 analyses**. Utilité globale
   87 % (92 items) ; par section : erreurs 93 %, focus 94 %, habitudes 80 %, **forces 73 %**.
   Publié en lecture directe sur `GET /api/c/<slug>/eval`, recalculé côté serveur à chaque
   consultation (un blob précalculé serait périmé dès l'annotation suivante).
2. **Ancrage déterministe** (`grounding.py`) : chaque chiffre et horodatage cité existe-t-il
   dans le payload ? Le détecteur est **cloisonné par unité** (sans quoi n'importe quel nombre
   du journal ancre n'importe quelle statistique : 30 % de détection au lieu de 91 %) et
   **calibré par contrôle négatif** (`ROUNDED_REL=0.01`, mesuré dans `tests/test_grounding.py`).
   Un taux d'ancrage sans mesure de puissance ne veut rien dire.
3. **Contrefactuel** (`counterfactual.py`) : on perturbe une dimension du payload, on régénère,
   on vérifie que la sortie suit. Mesures du 2026-09-04, `kimi-k2.6`, **5 runs valides sur 5** :

   | perturbation | attente | observé |
   |---|---|---|
   | `no_deaths` | la confiance baisse | 0.85 → 0.30 et 0.90 → 0.60 (2/2) |
   | `unspent_gold_zero` | le gold cité s'effondre | 2 052 g → 0 g (1/1) |
   | `zone_to_top` | les morts citées basculent en TOP | TOP apparaît (2/2, ancrage 100 %) |

> Le 2026-09-04, ce harnais a détecté un défaut **dans son propre test**, pas dans le modèle.
> La perturbation de zone attendait la disparition du mot « BOT », que le payload d'un ADC
> botlane cite légitimement ailleurs (rôle, benchmarks de lane, « tour BOT perdue »). Elle
> affichait donc 0/2 alors que le modèle avait correctement déplacé ses huit morts, avec un
> ancrage de 1.00. Remplacée par un déplacement vers TOP, zone non ambiante pour un ADC : la
> mesure passe à 2/2. Un test d'éval qui n'est pas lui-même falsifiable ne mesure rien.

## 11. Reproduire

```bash
poetry install
python3 src/01_data_engineering/build_split.py            # split canonique, graine fixe
python3 src/01_data_engineering/build_player_dataset.py
python3 src/02_data_science/train_player_ensemble.py      # rang
python3 src/02_data_science/train_player_lp.py            # LP
python3 src/pipeline_ops/dataset_report.py --json         # etat des lieux + detection de drift
```

⚠️ Sur macOS, `torch` et `xgboost` ne cohabitent pas (double chargement de `libomp` →
SIGSEGV). Les runs séquentiels utilisent un baseline tabulaire RF+EBM sans XGBoost, sur
`--device cpu` (MPS n'implémente pas le nested-tensor de `src_key_padding_mask`).

`data/` est gitignoré : sans corpus local, ces commandes n'ont rien à lire.

## 12. Traçabilité

- Historique complet des runs : `data/05_model/player_metrics.json`,
  `player_lp_metrics.json`, `sequence_metrics{,_v2}.json`, `auc_vs_ngames.json`.
- Décisions de conception datées : `docs/superpowers/specs/`.
- Chaque analyse LLM persistée porte un bloc `run` : empreinte sha256 du prompt système
  (impossible d'oublier de la bumper), modèle, latence, tokens **retries de schéma inclus**,
  et `schema_retries`. Sans ce bloc, une variation du taux d'utilité ne serait attribuable ni
  au prompt ni au modèle.
