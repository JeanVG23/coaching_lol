# Design — Features macro-positionnement (timeline API, 0 CV)

> Date : 2026-06-30. Statut : validé en brainstorming, prêt pour le plan d'implémentation.

## Contexte & motivation

Le pipeline ML (EBM-primary, cible `dia_chall` Diamond-vs-Challenger) a montré que les
features **macro/laning** (CS/XP/gold) portent un signal de rang réel mais modeste
(AUC ~0.66). La timeline Riot fournit la **position (x,y) des 10 joueurs toutes les 60 s**,
aujourd'hui quasi inexploitée (`extract_game` n'en tire que tes morts, ta base, ta distance
au drake). Objectif : **densifier la couche de features avec du positionnement macro**,
dérivé exclusivement de la timeline (aucune CV), pour répondre : *le positionnement
porte-t-il un signal que le laning ne capte pas ?* — et alimenter le coaching benchmarké.

Décision de phasage produit (cf. discussion) : le **positionnement macro** a déjà son
corpus de comparaison gratuit (timeline tous rangs). C'est distinct du **micro** (cooldowns,
skillshots, caméra) qui, lui, exigera la CV sur tes propres replays `.rofl` en Phase 2 et
n'aura pas de corpus challenger (jugé en absolu/self-baseline). Ce spec ne couvre que le macro.

## Consommateurs (les deux)

Une seule couche de features, deux contraintes :
- **ML/SHAP** : scalaires par game, injectés dans le classifieur EBM `dia_chall`.
- **Coaching** (`compare.py`) : métriques interprétables, benchmarkées à issue égale,
  **respect strict de l'asymétrie d'information** côté prescription.

## Architecture — module dédié `src/positioning.py`

Fonction pure (pas d'I/O, pas d'appel API) :

```python
def positioning_features(timeline: dict, participant_id: int, my_team: int,
                         my_fr: dict[int, dict], deaths: list[dict]) -> dict:
    """1 passe sur les frames -> dict plat de scalaires positionnels.
    Tout dérivé de la timeline (positions/60s + events). Aucune CV."""
```

- `extract_game` l'appelle après sa boucle existante et niche le retour sous
  `record["position"] = {...}`. On lui passe `my_fr` et `deaths` déjà calculés ; le module
  refait sa propre passe pour les positions des 10 joueurs (non extraites aujourd'hui).
- **Manifeste d'asymétrie**, garde-fou central exporté par le module :
  ```python
  COACHING_SAFE: set[str]   # features exactes/asymétrie-safe -> compare.py
  ML_ONLY: set[str]         # proxys vision approximés -> jamais en prescription
  ```
- **Contrat de sortie** : dict plat `{feature_name: float|int|None}`, `None` si non
  calculable (game trop courte) → laissé NaN (géré nativement par XGBoost/EBM).

### Pourquoi un module et pas inline dans `extract_game`
`extract_game` fait déjà ~180 lignes (lane + morts + comp). Le proxy vision et la math
d'over-extension ont une logique subtile qui exige des tests unitaires isolés, impossibles
proprement si noyés dans `extract_game`. Module = responsabilité unique, testable, garde
`extract_game` lisible. Coût : une 2e passe sur ~35 frames/game, négligeable vs la
décompression zstd qui domine.

### Data flow (0 appel API, tout depuis le raw caché)

```
positioning.py  ─┐
                 ├─> extract_game() ─> record["position"]
raw (match,tl) ──┘         │
                           ├─> build_dataset.py  : pioche les clés num. -> colonnes ML (EBM dia_chall)
                           ├─> reextract_silver  : réécrit le silver enrichi
                           └─> aggregate/write_gold ─> compare.py : benchmark à issue égale sur COACHING_SAFE
```

## Catalogue des features (17 : 14 🟢 exact/safe, 3 🔵 proxy ML-only)

Notation : 🟢 = exact, asymétrie-safe → ML + coaching · 🔵 = proxy approximé → ML-only
(jamais prescrit). Toutes par game, `None`→NaN si non calculable.

### Famille A — Présence carte & roam (ta seule position)
| Feature | Définition | Tag |
|---|---|---|
| `frac_own_lane_early` | frac. frames en BOT (lane ADC), phase early | 🟢 |
| `frac_river_early` | frac. frames JUNGLE/RIVER, early | 🟢 |
| `frac_roam_mid` | frac. frames hors BOT (MID/TOP) en mid — proxy roam | 🟢 |
| `frac_enemy_half` | frac. frames côté ennemi (signe anti-diagonale `x+y`) | 🟢 |
| `frac_base` | frac. frames près de ta fontaine (proxy temps base) | 🟢 |

### Famille B — Over-extension & safety
Over-extension = profondeur dans la moitié ennemie via distance signée à l'anti-diagonale
`x+y=MAP` (tower-free, robuste, cohérent avec `approx_zone`). Raffinement tour-précis = v2.
| Feature | Définition | Tag |
|---|---|---|
| `avg_map_depth` | profondeur moy. en terrain ennemi (0 si jamais) | 🟢 |
| `max_map_depth` | pic d'over-extension sur la game | 🟢 |
| `frac_overextended` | frac. frames au-delà du seuil de profondeur | 🟢 |
| `avg_dist_to_ally` | distance moy. au plus proche allié (isolement) | 🟢 |
| `gold_dead_time` | temps mort total estimé (table BRW level×temps, déterministe) | 🟢 |

### Famille C — Vision
**Exact (comptes, via `creatorId`/`killerId` == toi) :**
| Feature | Définition | Tag |
|---|---|---|
| `wards_placed` / `wards_placed_early` | wards posées par toi (total / early) | 🟢 |
| `control_wards_placed` | dont control wards | 🟢 |
| `wards_killed` | wards ennemies détruites par toi | 🟢 |

**Proxy proximité-allié (sight range `SIGHT=1350`) :**
| Feature | Définition | Tag |
|---|---|---|
| `frac_deaths_in_fog` | frac. de tes morts sans allié à portée de vue du lieu (pos. exacte au kill, alliés interpolés) | 🔵 |
| `avg_unaccounted_enemies` | moy./frame du #ennemis hors vue de toute ton équipe | 🔵 |
| `overext_x_unaccounted` | moy(profondeur × ennemis non-vus) — indice de risque | 🔵 |

## Math sensible

### Proxy de vision (`SIGHT = 1350`)
Un point est "en vision de mon équipe" au temps T si un allié (mon équipe, moi inclus) est
à ≤ `SIGHT`. Pas de wards (positions absentes de la timeline — vérifié : `WARD_PLACED`
n'a que `creatorId`/`wardType`/`timestamp`).
- `frac_deaths_in_fog` : pos. exacte de la mort (`CHAMPION_KILL.position`), positions
  alliées **interpolées linéairement** entre les deux frames 60 s encadrant le timestamp.
  En vision si un allié interpolé ≤ `SIGHT`. 0 mort → `None`.
- `avg_unaccounted_enemies` : par frame, ennemi "non vu" si aucun allié ≤ `SIGHT`. Moyenne
  sur les frames de jeu. Pas d'interpolation (déjà sur les snapshots).
- `overext_x_unaccounted` : `mean(map_depth_frame × unaccounted_frame)`.

### Over-extension (anti-diagonale, tower-free)
```
MAP  = (MAP_W + MAP_H) / 2
raw  = (x + y - MAP) / sqrt(2)
depth = raw if my_team == 100 else -raw    # >0 = terrain ennemi, <0 = chez moi
map_depth_frame = max(0, depth)
```
`frac_overextended` = frac. frames où `depth > SEUIL` (défaut ~2000u au-delà du milieu, à caler).

### `gold_dead_time`
Table BRW (base respawn wait) par niveau + facteur temps (formule Riot, patch-stable 16.x).
Niveau au moment de la mort lu depuis la frame la plus proche. `gold_dead_time = Σ respawn(level, t)`.
Documentée comme constante patch-versionnée (comme le cache DDragon).

### Garde-fou asymétrie (mécanique)
`positioning.py` exporte `COACHING_SAFE`/`ML_ONLY`. `compare.py` lit **exclusivement**
`COACHING_SAFE`. Un test assert que les 3 proxys ∈ `ML_ONLY` et que
`ML_ONLY ∩ COACHING_SAFE == ∅` → impossible de prescrire un proxy par accident.
Justification du tag 🔵 sur `avg_unaccounted_enemies`/`overext_x_unaccounted` : on compte
les ennemis **non vus** (info que le joueur AVAIT — il savait combien manquaient), jamais
leur position. Défendable, mais la granularité 60 s impose la prudence ML-only.

## Plan de tests (`tests/test_positioning.py`, pytest, timelines synthétiques)

| Test | Setup | Assertion |
|---|---|---|
| over-extension | base alliée vs deep enemy half | `map_depth=0` chez moi ; `>0` croissant côté ennemi ; symétrie team 100/200 |
| zones | frames en BOT/MID/RIVER connus | fractions attendues, somme ≈ 1 |
| dist allié | positions alliées connues | `avg_dist_to_ally` = min euclidien attendu |
| ennemis non-vus | 5 ennemis, 1 collé à un allié, 4 loin | `unaccounted = 4` ce frame |
| mort en fog | allié interpolé <1350 vs >1350 | in-vision vs fog corrects ; interpolation vérifiée sur 2 frames |
| comptes wards | events mix toi/autres | seuls les tiens comptés ; control wards isolées |
| dead time | morts à niveaux connus | = somme table BRW |
| **garde-fou asymétrie** | — | `ML_ONLY ∩ COACHING_SAFE == ∅` et 3 proxys ∈ `ML_ONLY` |
| intégration | vraie game du raw | `extract_game` renvoie `position` avec 17 clés, types corrects |

## Séquencement (2 sous-incréments)

**Incrément 1 — couche features (ce spec) :**
1. `positioning.py` + tests (TDD : tests d'abord).
2. Brancher dans `extract_game` (sous-objet `position`).
3. `reextract_silver` → silver enrichi (0 API). `build_dataset` pioche les 17 clés.
4. Relancer EBM `--target dia_chall` + `shap_analysis` → verdict : le positionnement
   apporte-t-il un signal au-delà du laning ? (delta d'AUC avec vs sans positionnement).

**Incrément 2 — coaching (spec séparé) :** câbler les 13 🟢 dans `aggregate`/`write_gold`
+ `compare.py` (benchmark à issue égale). Dépend du verdict de l'incrément 1 (on benchmarke
ce qui s'avère discriminant).

## Critères de succès (incrément 1)
- Tests verts ; `extract_game` produit `position` (17 clés) sur le raw réel.
- Dataset densifié relancé ; **AUC EBM avec vs sans positionnement comparée** = valeur
  ajoutée mesurée du positionnement.
- Garde-fou asymétrie testé.

## Hors scope (différé)
- Famille "proximité objectifs pré-spawn" (drake/herald/baron) — non retenue en v1.
- Over-extension tour-précise (coords de tours) — v2.
- Tout proxy vision basé wards (impossible : positions absentes).
- Micro (cooldowns, skillshots, caméra) — Phase 2 CV sur replays.
- Câblage coaching `compare.py` — incrément 2, spec séparé.
