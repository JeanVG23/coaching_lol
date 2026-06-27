# Composition & matchup comme contexte du coaching post-game (Phase A)

> Statut : design validé (2026-06-27). Phase A uniquement. Le « coach de draft »
> (pré-game, win rates de matchup, suggestions de counter) est explicitement **hors
> scope** et fera l'objet d'un produit séparé plus tard.

## Problème

Le coaching actuel compare les features perso (gd10, csd14, morts…) à un référentiel
challenger **global**, toutes situations confondues. C'est injuste et peu vérifiable : un
`gd10 = -510` peut venir d'un vrai défaut de jeu **ou** d'un contexte défavorable (matchup
de lane perdant, fort potentiel de gank ennemi). Sans le contexte, on ne peut pas trancher
— et reprocher une lane perdue à cause du contexte viole le principe d'asymétrie
d'information du projet.

## Objectif

Capturer le **contexte de situation de botlane** de chaque game et l'utiliser pour :
1. **conditionner les benchmarks** (« en lane all-in + forte exposition gank, les
   challengers font gd10 -200 ; toi -510 ») — concret et vérifiable ;
2. **afficher le contexte** dans le rapport (les 6 champions pertinents + patterns dérivés).

Tout en restant **Riot-first** : aucune donnée de win rate scrapée. Les seules sources sont
les matchs qu'on pull déjà (10 champions présents dans chaque match) + Data Dragon (statique,
gratuit) + une fine table de connaissance curée.

## Périmètre : 6 champions sur 10

Le contexte de botlane ne se limite pas au 2v2. On modélise :

- **Botlane 2v2** : ton ADC, ton support, l'ADC ennemi, le support ennemi → *pression de
  lane directe*. Le support change l'identité de la lane (Caitlyn+Lux = poke/zone ;
  Caitlyn+Leona = all-in early).
- **Les 2 junglers** : le tien (potentiel d'aide / contre-gank) + l'ennemi (menace de
  gank/dive). Un Jarvan/Lee qui gank ≠ un Karthus qui farm.
- **Le mid ennemi** : potentiel de roam/gank vers bot (Ahri/Akshan roament ≫ Cassiopeia).

Les 4 autres champions (top des deux équipes, mid allié) sont hors périmètre Phase A : peu
d'impact direct sur ton laning ADC. La compo **équipe complète** (engage/poke/scaling) est
volontairement reportée (YAGNI) — la machinerie de vecteurs ci-dessous la permettra plus tard.

## Architecture des composants

### 1. Couche d'identité champion — `src/champion_profiles.py`

Module isolé, sans état réseau au runtime (cache disque). Responsabilité unique : donner le
vecteur d'identité d'un champion.

**Source A — Data Dragon (gratuit, officiel, mis en cache).**
- Téléchargement one-shot de `championFull.json` (ou par champion) → cache dans
  `data/00_static/ddragon/<version>/`.
- Extrait par champion : `attackrange` (proxy ranged/short & poke), `tags`
  (Marksman/Support/Mage/Assassin/Fighter/Tank), stats de base utiles.
- La version DDragon est figée (paramètre/const) pour la reproductibilité ; un refresh est
  une action explicite, pas automatique.

**Source B — table curée `data/00_static/champion_traits.json`.**
Ce que DDragon n'a pas (connaissance de jeu). ~80 champions des rôles ADC/support/jungle/mid.
Pré-rempli par l'assistant, **validé par l'utilisateur**. Champs par champion :

| Rôle ciblé | Axes curés |
|-----------|-----------|
| ADC, support | `power_curve` ∈ {early, mid, late} ; `lane_pattern` ∈ {poke, all_in, sustain, scaling} |
| jungle | `playstyle` ∈ {ganking, farming, skirmish} ; `gank_threat` ∈ {low, med, high} |
| mid | `roam` ∈ {low, med, high} |

Un champion peut porter plusieurs jeux d'axes (flex pick) ; on indexe par nom de champion,
les axes non pertinents pour la game sont ignorés à la dérivation.

**API du module :**
- `champion_vector(name) -> dict` — merge DDragon + traits curés (None/`unknown` si absent
  de la table ; jamais d'exception bloquante, on loggue le manque).
- `lane_profile(adc, support) -> dict` — profil de pression de lane d'un duo.
- `derive_context(bot_self, sup_self, bot_enemy, sup_enemy, jgl_self, jgl_enemy, mid_enemy)
  -> {lane_pattern, gank_exposure}` — voir §3.

**Gestion des trous** : un champion absent de la table curée → ses axes valent `unknown` ;
la dérivation dégrade proprement (bucket `unknown`) plutôt que de planter. Tout champion
`unknown` rencontré est loggué pour compléter la table au fil de l'eau.

### 2. Extraction du matchup — `extract_game` (dans `src/riotlib.py`)

`extract_game` dispose déjà de `pid_role` et `pid_champ` (rôle + champion de chaque
participant) et de `my_team`. On ajoute la résolution des 6 champions par (rôle, équipe) :

```
bot_self_support  = champ du teamPosition UTILITY, même équipe que moi
bot_enemy         = champ du teamPosition BOTTOM,  équipe adverse   (= lane["opponent"] déjà extrait)
bot_enemy_support = champ du teamPosition UTILITY, équipe adverse
jgl_self          = champ du teamPosition JUNGLE,  même équipe
jgl_enemy         = champ du teamPosition JUNGLE,  équipe adverse
mid_enemy         = champ du teamPosition MIDDLE,  équipe adverse
```

Ces 6 noms (+ mon champion déjà présent) sont stockés dans un sous-objet `comp` du record
silver. `extract_game` reste agnostique des traits : il stocke les **noms**, la dérivation
des buckets se fait ailleurs (séparation des responsabilités, et permet de re-dériver si la
table curée évolue sans re-toucher l'extraction).

Robustesse : si un rôle manque (parties sans `teamPosition` propre, remakes), le champ
correspondant vaut `None` et la dérivation gère l'absence.

### 3. Dérivation des deux axes de contexte

Entrée riche (6 champions, multi-axes) → **sortie coarse** (peu de buckets), sinon les
83 games ADC challenger sont trop éparses pour benchmarker. Deux axes dérivés :

1. **`lane_pattern`** ∈ {poke, all_in, scaling, mixed, unknown}
   Dérivé du 2v2 botlane (ton duo vs duo ennemi) à partir de `lane_pattern`/`power_curve`
   des 4 champions, via des règles déterministes simples (ex. présence d'un all-in fort côté
   ennemi → `all_in` ; deux scaling + poke → `scaling`/`poke`). Règles explicites, testables.

2. **`gank_exposure`** ∈ {low, med, high, unknown}
   Dérivé de : `gank_threat` du jungler ennemi + `roam` du mid ennemi, **atténué** par le
   `playstyle` de ton jungler (un jungler allié `ganking` réduit l'exposition nette via
   contre-présence). Barème déterministe, documenté dans le code.

Ces deux axes sont calculés par `champion_profiles.derive_context(...)`. Ils ne sont PAS
stockés en dur dans le silver (qui ne garde que les noms) : ils sont (re)calculés à la
construction du gold / dataset, pour rester re-dérivables si la table curée change.

### 4. Benchmark conditionné — gold + `compare.py`

- `aggregate` / `write_gold` gagnent une **dimension `by_lane_context`** : facettes par
  `lane_pattern` et par `gank_exposure` (en plus des facettes win/loss existantes).
- `compare.py` : pour la slice perso, identifie le contexte dominant et compare au même
  contexte côté référentiel. **Repli explicite sur le global** si le bucket a moins de N
  games (seuil constant, ex. N=8) — repli **loggué**, jamais silencieux (principe « no
  silent caps » : un manque de données ne doit pas se déguiser en couverture).

### 5. Restitution

Le rapport (`compare.py` aujourd'hui, LLM plus tard) affiche :
- le détail des 6 champions de la situation botlane,
- les deux patterns dérivés (`lane_pattern`, `gank_exposure`),
- le benchmark conditionné (ou la mention du repli global + raison).

## Flux de données & migration

```
DDragon (one-shot) ─┐
                    ├─► champion_profiles ─► derive_context ─┐
champion_traits ────┘                                        │
                                                             ▼
raw cache (5674 fichiers, 0 API) ─► reextract_silver ─► silver (+comp) ─► rebuild_gold (+by_lane_context) ─► compare.py
```

**Migration sans API** : le silver actuel n'a pas le sous-objet `comp`. Le raw étant en
cache, on ajoute `src/reextract_silver.py` (raw cache → silver, **0 appel API**) qui rejoue
`extract_game` sur tous les matchs cachés et réécrit le silver perso + référentiels. Puis
`rebuild_gold.py` (déjà existant, à étendre avec `by_lane_context`) régénère le gold.

Ordre d'exécution de la migration :
1. `champion_profiles.py` + `champion_traits.json` (pré-rempli, validé).
2. `reextract_silver.py` → silver enrichi du `comp`.
3. `rebuild_gold.py` étendu → gold avec `by_lane_context`.
4. `compare.py` étendu → restitution conditionnée.

## Décisions explicites (anti-ambiguïté)

- Le silver stocke les **noms de champions** (`comp`), pas les buckets dérivés. Les buckets
  sont recalculés en aval → re-dérivables si la table curée évolue, sans re-extraction.
- Les axes de contexte sortent en **petit nombre de buckets** (2 axes coarse), pas en combos
  exacts : choix dicté par la taille d'échantillon (~83 games ADC challenger).
- Tout champion absent de la table curée → `unknown`, dégradation propre + log. Jamais de
  crash sur un champion non référencé.
- Tout repli d'un benchmark conditionné sur le global est **loggué** avec la raison.
- DDragon est figé sur une version explicite ; refresh = action manuelle.

## Hors scope (Phase A)

- Win rates de matchup / counters / suggestions de pick → **coach de draft**, produit séparé
  ultérieur.
- Compo **équipe complète** (archétypes engage/poke/dive sur les 5) → plus tard, réutilisera
  `champion_vector`.
- Scraping lolalytics / u.gg → écarté (duplique la « stat classique », fragile, hors thèse).

## Critères de réussite

- Chaque game silver porte un `comp` complet (6 champions résolus quand les rôles existent).
- `compare.py` produit au moins un benchmark conditionné (ou un repli loggué) sur une slice
  perso réelle.
- La table curée couvre les champions effectivement rencontrés dans les games perso +
  référentiel ADC (les manques sont listés par le log pour complétion).
- Aucune régression : la chaîne `reextract_silver → rebuild_gold → compare` tourne sans appel
  API et sans casser les facettes win/loss existantes.
