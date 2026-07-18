# Chaîne causale intra-game : conséquences des morts dans le journal

**Date** : 2026-07-18
**Statut** : validé (brainstorming)
**Objectif** : le coaching par-game doit parler de causalité en chaîne — relier chaque
mort à ses conséquences concrètes (« mort à 26:04 → Baron perdu 56 s après → -1 840 g
d'écart »). Aujourd'hui chaque mort du journal est isolée : le LLM ne voit pas l'aval.

## Principe directeur

Fidèle au levier du projet (« le levier = les features, pas le LLM ») : la causalité
est **calculée mécaniquement** dans `game_journal.py` depuis la timeline Riot (0 API,
0 CV). Le LLM ne fait que raconter des liens déjà établis — déterministe, debuggable,
zéro hallucination causale. Approche « journal chronologique complet laissé à
l'inférence du LLM » explicitement écartée.

## 1. `src/core/game_journal.py` — bloc `consequences` par mort

Nouvelle fonction pure `_consequences(timeline, t_ms, my_team) -> dict`, appelée dans
`_deaths` pour attacher à chaque mort :

```json
"consequences": {
  "objectives_lost": [{"type": "BARON_NASHOR", "clock": "27:00", "delta_s": 56}],
  "buildings_lost":  [{"type": "TOWER", "lane": "MID", "clock": "26:40"}],
  "team_gold_swing_90s": -1840
}
```

- **`objectives_lost`** — `ELITE_MONSTER_KILL` pris par l'équipe ennemie
  (`killerTeamId != my_team`) dans les `CONSEQUENCE_WINDOW_S = 60` s suivant la mort.
  `delta_s` = secondes entre la mort et la prise.
- **`buildings_lost`** — `BUILDING_KILL` sur un bâtiment de MON équipe dans la même
  fenêtre. ⚠️ Sémantique timeline : `teamId` d'un `BUILDING_KILL` = l'équipe qui
  **perd** le bâtiment → filtre `teamId == my_team`, verrouillé par test. Champs :
  `type` (`buildingType`, ou `towerType` si plus précis), `lane` (`laneType`), `clock`.
- **`team_gold_swing_90s`** — écart de gold d'équipe (somme des `totalGold` des 5
  `participantFrames` par équipe, mon équipe − ennemie) entre la dernière frame
  ≤ t_mort et la première frame ≥ t_mort + `GOLD_SWING_WINDOW_S = 90` s. Valeur =
  swing (écart après − écart avant). `null` si frame post-mort absente (fin de game).
- Constantes en tête de module à côté d'`OBJECTIVES`/`IMMINENT_WINDOW_S`, documentées
  comme approximation v1 : la timeline ne donne pas le death timer réel, la fenêtre
  fixe est assumée.
- Clés vides omises ; si rien dans la fenêtre et swing null → pas de clé
  `consequences` du tout (pas de bruit dans le payload).

**Asymétrie** : conforme au manifeste du module — annonces d'objectif/tour à l'écran,
gold d'équipe visible au scoreboard. Aucun proxy de vision.

## 2. `src/04_coaching/prompt.py` — `SYSTEM_GAME`

Nouvelle règle (chaîne causale) :

- Quand une mort porte des `consequences`, la `cause`/`evidence` de l'erreur DOIT
  restituer la chaîne : « mort à 26:04 → Baron perdu 56 s après, -1 840 g d'écart en
  90 s ». C'est le coût réel de la mort, pas juste l'événement.
- Garde-fou : la fenêtre est une **corrélation temporelle forte, pas une preuve
  absolue** — formuler « pendant que tu étais mort / juste après ta mort, l'ennemi a
  pris X », jamais inventer un lien hors journal.

## 3. Inchangés

- `schema.py` — le champ `cause` obligatoire de `GameInsight` existe déjà (2026-07-08).
- `payload.py` — `build_game` embarque le journal tel quel.
- Aucun rebuild silver/gold : le journal se calcule à la volée depuis le raw.

## 4. Tests — `tests/test_game_journal_consequences.py`

Timeline synthétique (pattern des tests de dérivation déterministe existants) :

- Baron ennemi à mort+40 s → dans `objectives_lost` ; à mort+70 s → exclu (fenêtre 60 s).
- Objectif pris par MON équipe pendant la fenêtre → exclu.
- `BUILDING_KILL` `teamId == my_team` → dans `buildings_lost` ; `teamId` ennemi → exclu
  (verrouille la sémantique « teamId = équipe qui perd »).
- Gold swing : frames avant/après connues → valeur exacte ; pas de frame post-mort →
  `null` ; ni événement ni swing → pas de clé `consequences`.

## 5. Validation bout en bout

Régénérer une review (`coach.py --game`) sur une game connue (ex. `EUW1_7900379457`,
mort à 26:04 avant un Baron perdu) et vérifier que la chaîne apparaît dans `cause`/
`evidence` de la sortie LLM.
