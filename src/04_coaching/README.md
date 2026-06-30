# 04_coaching — Narration LLM du coaching

Compte-rendu de coaching agrégé, narré par un LLM (Ollama Cloud, structured output)
à partir d'un payload déterministe dérivé du diff perso ↔ référentiel.

## Pipeline

```
gold (perso + référentiel) → payload.build (déterministe) → prompt.render
   → llm_client.generate_json (Ollama Cloud, format=JSON-schema)
   → schema.Review (validation Pydantic) → render_text (FR) + persiste reviews.jsonl
```

## Modules

| Fichier | Rôle |
|---|---|
| `payload.py` | gold perso+réf → payload déterministe, **safe-only** (positioning ⊂ `COACHING_SAFE`, profondeur `descriptive_only`) |
| `prompt.py` | system (asymétrie + benchmark-relatif + règle format, FR) + user |
| `schema.py` | Pydantic `Review` : 3 forces / 3 erreurs / 2 habitudes / 1 focus / confidence, **preuve chiffrée par point** |
| `llm_client.py` | client `https://ollama.com/api/chat`, `OLLAMA_API_KEY`, `format`=JSON-schema, retries 429/5xx |
| `coach.py` | CLI : payload→prompt→client→validation→affiche+persiste. `DEFAULT_MODEL = "kimi-k2.6"` |

## Usage

```bash
python3 src/04_coaching/coach.py --player spadzze --scope adc [--outcome loss] \
                                  [--target challenger] [--model kimi-k2.6]
```

Résolution du modèle : `--model` (CLI) > `OLLAMA_MODEL` (shell env) > `OLLAMA_MODEL` (`.env`)
> `DEFAULT_MODEL`. Sortie persistée dans `data/07_coaching/<player>/reviews.jsonl`
(1 ligne JSON par run : `ts`, `model`, `scope`, `target`, `outcome_focus`, `payload`, `review`).

---

## A/B testing des modèles (2026-06-30)

**Protocole** : même payload (`spadzze`, scope `adc`, issue `loss`, vs challenger) rejoué sur
4 modèles d'Ollama Cloud. Comparaison sur 5 critères : conformité au schéma, respect de
l'asymétrie (règle 1), profondeur traitée comme observation neutre (règle 3), concret &
benchmark-relatif (règle 4), exploitation du benchmark contextuel de lane.

Catalogue récupéré via `GET https://ollama.com/api/tags` (35 modèles disponibles).
`kimi-k2.7` n'existe qu'en variante `-code` (orientée code) ; `kimi-k2.6` est l'équivalent
général retenu.

### Résultats

| Modèle | Conf. | Format | Règle 3 (profondeur neutre) | Benchmark contextuel | Verdict |
|---|---|---|---|---|---|
| `deepseek-v4-pro` | 0.60 | OK **après durcissement du prompt** | neutre (« écart typique de rang ») | non | baseline, plate |
| `glm-5.2` | 0.55 | OK natif | non mentionnée (sûr) | non | la plus concise, deltas bien mis en forme |
| `minimax-m3` | 0.50 | OK natif | non mentionnée (sûr) | **oui** (`all-in : -429 g @10 vs +24 g`) | narration la plus riche, exploite `context_benchmark` |
| `kimi-k2.6` | 0.60 | OK natif | **la meilleure** (« marqueurs descriptifs de ton rang, pas des fautes ») | non | respecte le plus fidèlement l'asymétrie/règle 3 |

Tous conformes au schéma (3 forces / 3 erreurs / 2 habitudes, clés anglaises exactes,
habits en chaînes). Aucun reproche fondé sur info cachée (asymétrie tenue bout-en-bout).

### Classement

```
kimi-k2.6  ≥  minimax-m3  >  glm-5.2  >  deepseek-v4-pro
```

### Enseignements

1. **Le durcissement du prompt est model-agnostic.** Les 4 modèles honorent le schéma
   après la règle 7 de `prompt.SYSTEM` (JSON strict + clés anglaises exactes + habits en
   chaînes). C'est cette règle qui porte la conformité, **pas** la contrainte `format`
   d'Ollama — `deepseek-v4-pro` l'ignorait pour les schémas complexes (d'où le durcissement,
   commit `5ced84b`). Confirme in vivo la thèse projet : la qualité vient du prompt + des
   features, pas du modèle. Le LLM ne fait que raconter.
2. **Discriminateur = la règle 3 (asymétrie/profondeur).** Seul `kimi-k2.6` cadre
   explicitement profondeur + overextension comme marqueurs descriptifs de rang, pas
   comme des fautes — exactement l'intent de la règle 3 durcie. `minimax-m3` se distingue
   par l'exploitation du `context_benchmark` (le seul à citer le pattern `all-in`).
3. **Choix du défaut** : `kimi-k2.6` retenu pour le respect maximal de l'asymétrie, principe
   non-négociable du projet. `minimax-m3` à privilégier si l'on veut exploiter le benchmark
   contextuel de lane. Surclassable à tout moment via `--model`.

### Reproduction

```bash
for m in deepseek-v4-pro glm-5.2 minimax-m3 kimi-k2.6; do
  python3 src/04_coaching/coach.py --player spadzze --scope adc --outcome loss --model $m
done
# comparer : chaque run append une ligne dans data/07_coaching/spadzze/reviews.jsonl
.venv/bin/python -c "import json,pathlib; [print(r['model'], r['review']['confidence']) \
  for r in (json.loads(l) for l in pathlib.Path('data/07_coaching/spadzze/reviews.jsonl').read_text().splitlines() if l.strip())]"
```