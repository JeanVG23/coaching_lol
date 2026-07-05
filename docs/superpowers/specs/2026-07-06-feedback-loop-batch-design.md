# Boucle de feedback — génération batch + annotation en série

**Date** : 2026-07-06
**Statut** : validé (conversation), approche A retenue ; l'approche C
(génération automatique post-game) est notée comme suite, hors scope ici.

## Problème

La métrique de succès du compte-rendu par-game est « ≥70 % de mistakes utiles
sur ≥10 reviews par-game annotées ». État courant : 6 reviews persistées dont
**1 seule par-game**, 1 seule annotation (agrégée) → métrique à 0/10. Le goulot
n'est pas l'outillage d'annotation (CLI + web existants) mais le **volume** :
`coach.py --game` ne traite qu'une game à la fois, et `annotate` demande de
relancer la commande et choisir un numéro pour chaque review.

## Design (3 blocs)

### 1. Génération batch — `coach.py --game-batch N` (défaut 10)

- Nouveau flag, **incompatible avec `--game`** (erreur argparse si les deux).
- Sélection : games du scope (ADC) du silver perso, plus récentes d'abord
  (tri `game_ts` décroissant quand présent ; le silver perso antérieur au
  2026-07-06 n'a pas ce champ → repli sur l'ordre d'apparition inversé du
  fichier, approximation de l'ordre de collecte),
  MOINS celles ayant déjà une review `kind=game` dans `reviews.jsonl`
  (dédup par `match_id`, quel que soit le modèle — une review par game suffit
  pour la métrique).
- Boucle : `payload.build_game(match_id)` → `generate_game_review` → `persist`,
  séquentiel. **On continue sur échec** d'une game (LLMError → log ;
  CoachValidationError → brut sauvé dans `failed/`, comme le chemin actuel ;
  FileNotFoundError raw/silver → log). Bilan final :
  `générées / déjà reviewées / échouées`.
- La sélection/dédup est une fonction pure testable
  (`pending_game_matches(records, reviews, scope, n)`). La boucle LLM reste
  fine et non testée (réseau).

### 2. Annotation en série — `feedback.py annotate --pending`

- Itère sur les reviews **sans feedback** (jointure par `ts` avec
  `feedback.jsonl`), plus anciennes d'abord, kind affiché (`game`/agrégée).
- Avant chaque review : `[Entrée=annoter / n=passer / q=quitter]`.
- Chaque review est persistée dès qu'elle est finie (comportement
  `persist_feedback` actuel conservé → interruptible sans perte).
- Une review entièrement skippée reste pending (pas de marquage artificiel).
- Fonction pure testable : `pending_reviews(reviews, feedbacks)`.

### 3. Suivi de la métrique — bloc « Objectif » dans `summary`

- Le feedback ne stocke pas le kind → jointure `ts` ↔ `reviews.jsonl`.
- Nouveau bloc, calculé **uniquement sur les mistakes des reviews
  `kind=game`** (définition de la métrique) :
  `Objectif par-game : X/10 reviews annotées · mistakes utiles Y % (cible ≥70 %)`.
- Fonction pure testable : `objective_stats(feedbacks, reviews)` →
  `{n_game_reviews_annotated, target_n, mistake_useful_rate, target_rate}`.

## Hors scope

- Approche C : génération automatique après chaque game (dépend de
  l'automatisation de la collecte perso) — prochaine étape après A.
- Modification du prompt / schéma / modèle par défaut.
- Côté web (l'annotation ✓/✗ + note y existe déjà).

## Tests (TDD)

- `pending_game_matches` : dédup par match_id reviewé, tri récent d'abord,
  limite N, filtre scope.
- `pending_reviews` : jointure ts, ordre chronologique.
- `objective_stats` : ne compte que les mistakes des reviews `kind=game`,
  taux correct, robuste si 0 review par-game annotée.
- Flow `--pending` : testé avec `prompt` injecté (pattern annotate existant).
