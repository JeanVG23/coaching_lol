# Boucle d'évaluation — scoring d'utilité des reviews

> Spec de l'incrément « boucle de feedback » (todo.md, court terme). Objectif :
> pouvoir dire si le coach s'améliore, en s'appuyant sur la persistance
> payload+review déjà posée — sans re-générer les reviews.

## Contexte & motivation

`src/04_coaching/` génère et persiste des reviews dans
`data/07_coaching/<player>/reviews.jsonl` (1 ligne JSON par run : `ts`, `model`,
`scope`, `target`, `outcome_focus`, `payload`, `review`). Aujourd'hui rien ne dit si
ces reviews sont **justes / utiles**. Le coaching benchmarké challenger est
intrinsèquement plus vérifiable que des opinions absolues — mais il faut fermer la
boucle en capturant le jugement du joueur sur chaque conseil produit.

Le principe d'asymétrie est non-négociable : si un modèle prescrit la profondeur
comme une faute (règle 3), la boucle doit le capter comme un signal catégorisé, pas
juste un 👎 générique.

## Décisions (verrouillées au brainstorming)

- **Mode** : CLI interactive (`input()` sur stdin).
- **Granularité** : par-insight — les 9 items de la review (3 forces, 3 erreurs,
  2 habitudes, 1 focus).
- **Raison sur 👎** : tag fixe + note libre optionnelle.
- **Structure** : un script `feedback.py` avec sous-commandes `annotate` et `summary`,
  schéma Pydantic partagé dans `schema.py`.

## Architecture

```
src/04_coaching/
  feedback.py          # CLI: annotate + summary (nouveau)
  schema.py            # + Feedback / FeedbackItem / NEG_TAGS (existant + ajout)
data/07_coaching/<player>/
  reviews.jsonl        # existant — clé = ts
  feedback.jsonl       # NOUVEAU — 1 ligne = 1 review annotée
tests/
  test_coaching_feedback.py   # nouveau
```

`feedback.jsonl` est un output local (sous `data/`, gitignoré).

## Modèle de données (Pydantic v2, dans `schema.py`)

```python
from typing import Annotated, Literal

class FeedbackItem(BaseModel):
    kind: Literal["strength", "mistake", "habit", "focus"]
    index: int                          # position dans sa section (focus = 0)
    useful: bool                         # jugement du joueur
    tag: Literal["asymetrie", "stat-inventee", "profondeur-en-faute",
                 "trop-vague", "non-actionnable", "autre"] | None = None
    note: str | None = None              # libre, optionnel

class Feedback(BaseModel):
    ts: str                              # clé = ts de la review annotée
    player: str
    rated_at: str                        # ISO timestamp de l'annotation
    model: str                            # copié de la review (récap par modèle)
    overall_useful: bool | None = None    # verdict global optionnel (non collecté
                                         #   par le flow interactif ; laissé pour usage
                                         #   futur/scripté)
    items: list[FeedbackItem]            # items annotés (≤9 ; skips omis)

NEG_TAGS = ("asymetrie", "stat-inventee", "profondeur-en-faute",
            "trop-vague", "non-actionnable", "autre")
```

Invariant : `tag` obligatoire si `useful=False` (validé Pydantic via
`@model_validator(mode="after")`). Un `ts` apparaît au plus une fois dans
`feedback.jsonl` (réannotation = écrase).

## `feedback.py annotate` — flow interactif

1. Lit `reviews.jsonl` du joueur, liste les N dernières reviews
   (`# | ts | date | model | outcome`).
2. L'utilisateur choisit une review (numéro), ou `--ts <ts>` (sélection directe),
   ou `--last` (la plus récente).
3. Charge la review, défile les 9 items :
   - affiche `kind[i] point (evidence)` ; pour `habit`/`focus` : le texte seul.
   - prompt utile ? `[y/n/s]` — `s` = skip (item omis, pas noté).
   - si `n` : prompt tag (menu numéroté `NEG_TAGS`) + note libre (1 ligne, Entrée = skip).
4. Construit `Feedback`, persiste dans `feedback.jsonl`.
   - Si `ts` déjà présent : écrase la ligne existante + log
     `réannotation: écrase feedback précédent pour <ts>`.
5. Récap session : `3/9 utiles, 2× profondeur-en-faute`.

Aucun appel réseau. Entrées : `--player spadzze` (défaut), `--ts`, `--last`.

## `feedback.py summary`

Lit `feedback.jsonl`. Pas de jointure avec `reviews.jsonl` : `Feedback` porte
déjà `ts` (pour le tri temporel) et `model` (pour le récap par modèle). Imprime :

- **Taux d'utilité global** : `X% des items utiles (n items notés sur m reviews)`.
- **Par section** : forces / erreurs / habitudes / focus — taux chacun
  (détecte « les erreurs sont bonnes, les habitudes ratent »).
- **Top tags** : compte des tags sur les 👎 — **le signal actionnable** pour durcir
  le prompt (ex : `profondeur-en-faute: 4` → durcir la règle 3).
- **Par modèle** : taux d'utilité par `model` (complète l'A/B : qualité narration ≠
  qualité perçue).
- **Tendance** : taux d'utilité des 5 dernières reviews annotées vs les
  précédentes (le « coach s'améliore-t-il »). Seuil low_sample : `<10 reviews
  annotées au total` → avertissement au lieu de la tendance. Tri par `ts` (clé
  de la review, portée par `Feedback`).

Entrées : `--player spadzze`, `--tag <tag>` (filtre), `--model <model>`.

## Tag set

`asymetrie` · `stat-inventee` · `profondeur-en-faute` · `trop-vague` ·
`non-actionnable` · `autre`.

Cible les modes d'échec connus du projet : règle 1 (asymétrie), règle 2 (preuve
inventée), règle 3 (profondeur en faute) + deux génériques (vague, non-actionnable).
Évolutif : ajouter un tag = ajouter à la `Literal` et à `NEG_TAGS`.

## Cas limites & erreurs

- Aucune review dans `reviews.jsonl` → message clair, exit 0 (pas un échec).
- `ts` introuvable → erreur explicite, exit 1.
- Réannotation d'un `ts` → écrase + log (la boucle doit permettre de re-juger
  après amélioration du coach).
- Review avec <9 items (malformée) → on annote ce qui existe, pas de crash.
- `feedback.jsonl` absent au `summary` → « aucune annotation », exit 0.
- Skip massif (tous items skippés) → pas de persistance (feedback vide = pas de feedback).
- `tag` absent sur un `useful=False` → erreur de validation Pydantic (invariant).

## Tests (`tests/test_coaching_feedback.py`)

- `build_feedback` depuis une `Review` + dict de réponses
  `{(kind,index): (useful, tag?, note?)}` → `Feedback` valide (skips omis).
- `persist` écrase un `ts` existant (1 ligne finale, la bonne).
- `summary` sur un fixture `feedback.jsonl` : taux global, par section, top tags
  corrects.
- `summary` low_sample (<5) → avertissement, pas de tendance.
- `annotate` flou interactif : monkeypatch `input` avec une séquence de réponses →
  vérifie le `Feedback` résultant + la persistance.
- Asymétrie : un item taggé `profondeur-en-faute` et négatif → compté dans top tags
  (la boucle capte une régression de règle 3).

## Hors scope

- Compte-rendu par-game (autre incrément du todo).
- UI graphique (la CLI interactive suffit pour un usage solo).
- Scoring automatique (pas de juge LLM : le jugement du joueur est la ground truth).