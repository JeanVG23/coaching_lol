# derived/

Données **dérivées** (régénérées par script) à partir de
`../champion_traits.json` + `../../02_silver/referentiel/`.

## Contenu

- `champion_axis_validation.json` : rapport data-driven de validation de
  `champion_traits.json`. Régénéré par
  `python3 src/pipeline_ops/validate_traits.py`. 0 appel API.

## Régénération

```bash
python3 src/pipeline_ops/validate_traits.py
```

Le script écrit toujours dans ce dossier (chemin par défaut).

## Quand régénérer

- Après un changement de patch (les stats par champion peuvent dériver)
- Après une mise à jour majeure de `champion_traits.json` (nouveaux axes)
- Après un changement de méthodologie (modifier `validate_traits.py`)

## Source de vérité

Ce dossier est **dérivé** — il ne se modifie pas à la main. Les corrections
doivent aller dans `champion_traits.json` (en éditant le JSON source).
