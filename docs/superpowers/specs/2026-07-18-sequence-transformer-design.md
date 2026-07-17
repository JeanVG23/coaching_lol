# Modélisation séquentielle + self-supervised des timelines

**Date** : 2026-07-18
**Statut** : validé (brainstorm)
**Origine** : choix d'orientation projet. Le pipeline ML actuel plafonne : l'ensemble
tabulaire per-game est à **AUC 0.589** sur la frontière master/GM (0.724 sur dia_chall),
et le modèle per-player (constance) à 0.635 en CV purgée. Le diagnostic projet est clair :
les leviers sont le pool, les features, ou la frontière de rang — *pas* tuner l'ensemble.

La cause du plafond : la timeline est une **séquence** (états par minute), et la couche de
features l'aplatis en scalars agrégés (mean/p10/p90). La dynamique temporelle est détruite
avant même d'atteindre le modèle.

Thèse de recherche : un **transformer sur la séquence brute par-minute** capte la structure
temporelle que l'agrégation jette, et pourrait battre le baseline tabulaire. Un **pretraining
self-supervised** (mask-and-reconstruct) au-dessus pourrait ajouter du signal — ou non,
à cette échelle de données (~8k games). Le résultat négatif est un résultat valable ; le
design est fait pour le mesurer proprement, pas pour le forcer.

**Objectif secondaire explicite** : apprentissage. L'auteur part de loin sur
transformers/embeddings. Le design intègre la pédagogie : petit transformer écrit à la main
en PyTorch (pas de HF Transformers), chaque brique lisible, progression incrémentale où
chaque étape est un résultat falsifiable.

## Décisions

1. **Périmètre per-game** : 1 ADC d'une game = 1 séquence = 1 label `high_elo`. Comparaison
   pomme-pomme avec l'ensemble tabulaire per-game (même unité). Le modèle per-player
   (constance, agrégats par joueur) est hors scope — autre formulation, autre question.
2. **State vector depuis `participantFrames` uniquement (v1)** : `[pos_x, pos_y,
   totalGold, currentGold, xp, level, minionsKilled, jungleMinionsKilled]` pour l'ADC ciblé
   **et** l'ADC adverse (le matchup, signal de lane central), + 4 diffs relatives (gold, cs,
   xp, level) = **20 dims/frame**. Aucun parsing d'events en v1.
3. **Standardisation per-feature dès la v1** (non négociable) : `totalGold` ~15000 dans la
   même projection linéaire que `position` ∈ [0,1] rend l'entraînement instable et produirait
   un null qui est un **artefact d'optimisation, pas une réponse à la question de recherche**
   — ça contaminerait directement l'interprétation du delta SSL. On standardise chaque
   feature (z-score : mean/std par feature sur le **train** du fold, stats figées et
   réappliquées à val/test) avant la projection `20→d_model`. La position reste [0,1].
4. **Frontières co-primaires, interprétation asymétrique** : dia_chall (0.724, marge
   réelle) et master/GM (0.589) sont **toutes deux rapportées comme primaires**, mais
   l'interprétation d'un null diffère. master/GM est probablement **bruit de label, pas
   perte de représentation** : la frontière est peu séparable, le MMR se chevauche, et le
   flaw de transfert de rang sur l'ADC adverse est hérité identiquement ici. Une meilleure
   représentation de la même timeline ne remonte pas une frontière limitée par le bruit.
   Donc un null sur master/GM est **non interprétable** (« la représentation n'aide pas » vs
   « la frontière est du bruit » sont indistinguables). dia_chall désambiguïse : c'est là
   que le signal existe, c'est là que le test est interprétable. Le verdict de la thèse
   séquence se lit sur dia_chall.
5. **Events discrets en étape 2** (kills, items, objectifs, skill order) : enrichissement
   ultérieur, pas dans le périmètre initial. Garder v1 petit et falsifiable — **mais** un
   null en v1 ne réfute pas la thèse séquence : v1 teste exactement ce que les agrégats
   résument déjà (les mean/p10/p90 sont des résumés de ces mêmes frames 60 s). Le signal
   temporel à forte valeur (trades, all-ins, roams) vit dans les events discrets, différés en
   étape 2. Cf. §« Pièges d'interprétation ».
6. **Transformer à la main en PyTorch** (pas de HF Transformers) : 4 couches, d_model=64,
   nhead=4, ~150-300k params. Tractable sur **MPS (Mac M4)** en quelques minutes pour
   8k games × 5 folds. La 3080 Ti est repli si on scale.
7. **CV purgé identique à l'existant** : StratifiedKFold sur les joueurs (l'identité du joueur
   porte le rang) + purge des games miroir (les deux ADC d'une game sont des lignes en miroir ;
   si l'adversaire d'une game de train est un joueur de val, on drop). Réutilise la discipline
   de `train_player_ensemble.py` — pas une invention. Permet la comparaison stricte au
   baseline.
8. **Self-supervised en delta mesurable (étape 2)** : pretrain mask-and-reconstruct sur TOUTES
   les games (les deux ADC, sans label), puis fine-tune sur `high_elo` avec le même CV. La
   question de recherche = AUC étape 2 vs AUC étape 1. On ne suppose pas que ça aide — **et**
   un ≈0 delta n'est pas un verdict sur le SSL en général : le prétexte MSE-reconstruct est
   faible sur des signaux lisses (gold monotone, position continue → quasi-interpolation),
   le modèle peut cartonner la reconstruction sans rien apprendre de pertinent au rang.
   Cf. §4 et §« Pièges d'interprétation ».
9. **Branche parallèle, zéro perturbation** : le pipeline tabulaire et le coach web ne sont pas
   touchés. `torch` déjà dispo (2.10.0, MPS OK) ; pas de dépendance lourde nouvelle.

## Design

### 1. `src/01_data_engineering/build_sequence_dataset.py` — raw → séquences (0 API)

- Réutilise `adc_puuids(match)` et `build_rank_map()` de `build_dataset.py`, et
  `rl._read_raw` pour relire le raw compressé. Zéro duplication de la logique métier.
- Pour chaque game référentielle, pour chaque ADC (`adc_puuids`) : itère
  `timeline["info"]["frames"]`, extrait par minute le `participantFrame` de l'ADC ciblé
  et de l'ADC adverse (résolu par `teamPosition == "BOTTOM"` opposé), construit le vecteur
  20-d, plus le `mask` (True aux minutes valides).
- **Cap T=40 minutes** ; games plus courtes → pad avec `mask=False`. Games > 40 min →
  tronquées (rare en high-elo, documenté).
- **Normalisation** : position divisée par la taille de map (~14800) → [0,1]. **Chaque
  feature standardisée en z-score** (mean/std par feature sur le **train** du fold, stats
  figées, réappliquées à val/test) avant la projection `20→d_model`. Non négociable :
  `totalGold` ~15000 dans la même projection que `position` ∈ [0,1] rend l'entraînement
  instable et un null serait un artefact d'optimisation, pas une réponse à la question de
  recherche.
- **Labels (deux configs de classification, miroir du baseline existant)** :
  - `high_elo` (frontière master/GM) : toutes les games, label = rang ∈ {GM, challenger}
    vs {diamond, master}. Baseline attendu 0.589.
  - `dia_chall` (frontière diamond/challenger) : games filtrées au rang ∈ {diamond,
    challenger}, label = challenger. Baseline attendu 0.724.
  - On stocke `rank` brut par row ; les deux labels se dérivent de `rank` au load, pas de
    colonnes dupliquées.
- **Sortie** : `data/04_dataset/adc_sequence_dataset.npz` —
  `sequences [N, 40, 20]` (float32), `mask [N, 40]` (bool), `labels [N]` (int),
  `puuids [N]` (str), `match_ids [N]` (str), `ranks [N]` (str), `champions [N]` (str).

### 2. `src/02_data_science/sequence_model.py` — encodeur + têtes (PyTorch pur)

- `SequenceEncoder(d_in=20, d_model=64, nhead=4, n_layers=4, ff=128, dropout=0.1,
  max_len=40)` : projection linéaire `20→64` + embedding positionnel appris `40→64`,
  stack de `TransformerEncoderLayer(batch_first=True)` avec `src_key_padding_mask=~mask`,
  méthode `forward(x, mask) → [B, T, 64]` (sortie par frame).
- `ClassifierHead` : masked-mean-pool sur les frames valides (`(x * mask).sum(1) /
  mask.sum(1)`) → linéaire `64→1`. Plus simple/pédagogique qu'un token CLS. **Tradeoff nommé
  explicitement** : le mean-pool est *lui-même une agrégation* — tension avec la thèse
  « l'ordre compte ». L'attention mixe temporellement en amont, donc c'est récupérable, mais
  si le modèle sous-performe, le goulot peut être le pooling, pas l'encodeur. On garde en
  tête un **CLS token / attention-pool en ablation** si le null tombe, pour isoler la cause.
- `ReconstructHead` (étape 2) : à partir des frames encodées, linéaire `64→20` pour
  reconstruire le state des frames **masquées uniquement**.
- Tout lisible dans un fichier ; pas de HF.

### 3. `src/02_data_science/train_sequence_model.py` — étape 1 supervisée

- Charge `adc_sequence_dataset.npz`, construit 5 folds player-groupés (StratifiedKFold
  sur `puuid`, stratifié sur `label`), purge les games miroir (drop de train toute game
  dont l'ADC adverse est un puuid de val — via `match_id` croisé avec les rows adverses).
- Par fold : entraîne `SequenceEncoder + ClassifierHead` sur MPS, `AdamW(lr=3e-4)`,
  schedule cosinus, dropout, BCE-with-logits, early-stop sur l'AUC de val (patience ~10),
  batch ~64-128, seed fixe.
- Évalue AUC val par fold, reporte mean ± std.
- **Baselines dans le même run** (comparaison propre) :
  - ensemble tabulaire per-game (relit `adc_dataset.parquet`, mêmes folds joueurs) :
    0.589 master/GM attendu, 0.724 dia_chall.
  - MLP sur les features agrégées existantes (sanité : le gain vient-il de la séquence ou
    juste d'un réseau ?).
- Cible : **dia_chall et master/GM co-primaires** (cf. décision 4) — deux configs de
  classification distinctes (cf. §1 labels), entraînées/évaluées séparément. Le verdict de la
  thèse séquence se lit sur **dia_chall** (0.724, marge réelle, test interprétable) ;
  master/GM (0.589) est rapporté mais son null est non interprétable (bruit de label probable).
- **Sortie** : `data/05_model/sequence_supervised.pt` (meilleur modèle) +
  `data/05_model/sequence_metrics.json` (AUC par fold, mean/std, baselines, params, seed,
  device). Même esprit que `player_metrics.json`.

### 4. `src/02_data_science/pretrain_sequence_model.py` — étape 2 self-supervised

- Pretrain **par-fold, train-only** (puriste) : pour chaque fold, on standardise train-only
  (mêmes stats qu'au finetune → transfert non saboté), on pretrain l'encodeur mask-and-reconstruct
  sur le **TRAIN du fold uniquement** (joueurs de val jamais vus → pas de fuite transductive),
  puis on finetune. Masque ~15 % des minutes au hasard (par séquence), reconstruit le state
  20-d des frames masquées. Perte : MSE sur les frames masquées uniquement. Choix délibéré
  vs pretrain sur toutes les données : on sacrifie du volume de pretrain par fold pour un
  `delta_ssl` propre (pas d'avantage transductif). `delta_ssl = AUC_ssl − AUC_supervisé`
  (étape 1, même CV purgé).
- **Faiblesse connue du prétexte** : MSE-reconstruct sur des signaux lisses (gold monotone,
  position continue) est de la **quasi-interpolation** — le modèle peut cartonner la
  reconstruction sans rien apprendre de pertinent au rang. Un prétexte plus dur (prédiction
  d'un événement futur : « la minute t+2 sera-t-elle un death ? un recall ? ») serait un
  vrai test du SSL, au prix de complexité — laissé en étape 3. À minima on acte qu'un
  ≈0 `delta_ssl` n'est pas un verdict sur le SSL en général, juste sur ce prétexte-là.
- `AdamW`, quelques epochs (early-stop sur la perte de val), seed fixe.
- **Sortie** : `data/05_model/sequence_encoder_pretrain.pt`.
- **Fine-tune** : charge l'encodeur pré-entraîné, ajoute `ClassifierHead` fraîche,
  ré-entraîne sur `high_elo` avec le **même** CV purgé qu'étape 1. Reporte AUC étape 2.
- **Delta** : `sequence_metrics.json` gagne `auc_supervised`, `auc_ssl`, `delta_ssl` et
  la réponse à la question de recherche.
- **Bonus** : fonction `embed_game(sequence, mask) → [64]` pour inspecter les embeddings
  (projection 2-D de games colorées par rang/champion — « voir » un embedding).

### 5. Tests (pytest)

- `tests/test_build_sequence_dataset.py` : décodage d'une game connue du raw → vecteur
  attendu à une minute donnée (positions/gold/cs), shapes `[N,40,20]` + `mask` cohérent
  (False au-delà de la durée réelle), `puuid`/`match_id` alignés, label `high_elo` cohérent
  avec `rank`.
- `tests/test_sequence_model.py` : forward sur tenseur factice `[4, 40, 20]` avec mask
  partiel → shapes attendues pour encoder/classifier/reconstruct ; le masked-mean-pool
  ignore bien les frames `mask=False` (vérification : deux séquences identiques sauf pad
  donnent le même pooled).

## Pièges d'interprétation (lire avant de conclure)

Trois nulls possibles, trois significations différentes — ne pas conclure trop large :

1. **Null sur master/GM (≈0.59)** : non interprétable. La frontière est probablement limitée
   par le bruit de label (MMR chevauché + flaw de transfert de rang sur l'ADC adverse,
   hérité ici à l'identique), pas par la représentation. Une meilleure timeline ne remonte
   pas un plafond de bruit. Le verdict se lit sur dia_chall.
2. **Null sur dia_chall en v1** : ne réfute pas la thèse séquence. v1 teste une séquence de
   `participantFrames` — exactement ce que les agrégats mean/p10/p90 résument déjà. Le
   signal temporel à forte valeur (trades, all-ins, roams, timing de recall) vit dans les
   **events discrets**, différés en étape 2. Un null v1 = « peu de signal sous l'agrégat »,
   pas « la séquence ne sert à rien ». La conclusion propre est conditionnelle à la portée
   des events.
3. **≈0 `delta_ssl`** : ne condamne pas le SSL. Le prétexte MSE-reconstruct est faible sur
   signaux lisses (interpolation). Répond à « ce prétexte aide-t-il ? », pas à « le SSL
   aide-t-il ? ». Un prétexte prédictif (étape 3) reste à tester.
   > Choix puriste résolvant la caveat transductive : pretrain **par-fold train-only**
   > (pas de fuite val) → un delta positif est un signal propre, pas un artefact transductif.
   > Coût : moins de données de pretrain par fold.

Chaque null est donc assorti de la raison pour laquelle il pourrait être un artefact. Le
design vise à mesurer proprement, pas à forcer un résultat positif — mais aussi à ne pas
tirer de conclusion trop large d'un null attendu.

## Hors scope

- Modèle per-player (constance) et régression LP — autres formulations, non touchés.
- Coach web, payload, prompt, boucle LLM — non touchés. Branche parallèle de recherche.
- Events discrets (kills, items, objectifs, skill order) — étape 2 d'enrichissement future.
- Scaling à la 3080 Ti / multi-tâche (rang + win + LP) / embedding réutilisé pour le
  coaching — étape 3 optionnelle, si passion confirmée, hors de ce design.
- HF Transformers / modèles pré-entraînés génériques — écarté : trop lourd et moins
  formateur qu'un transformer écrit à la main sur nos données.

## Critère de succès

- **Étape 1 livrée** : `sequence_metrics.json` rapporte AUC mean ± std sur **dia_chall et
  master/GM co-primaires**, comparé au baseline tabulaire (même CV purgé) et au MLP sanité.
  Le verdict « séquence > agrégat ? » se lit sur dia_chall (master/GM rapporté mais non
  interprétable si null, cf. §Pièges). Un modèle `sequence_supervised.pt` entraîné.
- **Étape 2 livrée** : `auc_supervised` / `auc_ssl` / `delta_ssl` dans le même fichier,
  accompagné du caveat « ce prétexte-là, pas le SSL en général ». Un encodeur pré-entraîné
  `.pt`.
- L'auteur comprend concrètement : embedding, attention, masking, pretrain/finetune.
- On peut s'arrêter à l'étape 1 avec un résultat appris ; l'étape 2 est un delta mesurable,
  pas un prérequis.