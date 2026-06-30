# Design — Narration LLM du coaching (Ollama Cloud, sortie typée)

> Date : 2026-06-30. Statut : validé en brainstorming, prêt pour le plan d'implémentation.
> Approche retenue : **A — appel unique, structured output, payload déterministe.**

## Contexte & motivation

La couche features est solide et validée (positionnement : AUC dia_chall 0.655 → 0.724).
`compare.py` produit déjà un diff riche perso vs référentiels, à issue égale, asymétrie-safe
— mais c'est un **tableau qu'un humain lit**. Le livrable « coach IA » du vision-doc
(`strengths[]` / `mistakes[]` / `next_focus[]`) n'existe pas encore. C'est le **goulot** :
tant qu'on ne ferme pas la boucle, on ne peut pas tester l'hypothèse fondatrice « le coach
est-il utile ? ».

Cet incrément ferme la boucle : **sérialiser le diff gold en payload → Ollama Cloud en
structured output → review typée + persistée.** Principe directeur du projet : *le LLM ne
fait que raconter ce que les features ont déjà conclu.*

## Décisions de cadrage (brainstorming)

| Décision | Choix |
|---|---|
| Runtime LLM | **Ollama Cloud** (clé `OLLAMA_API_KEY` dans `.env`), modèles hébergés raisonnement |
| Modèle par défaut | **`deepseek-v4-pro`** (configurable `--model`/`OLLAMA_MODEL`) ; A/B ultérieur (glm-5.2, minimax-m3, kimi-k2.7) |
| Granularité | **Agrégé** (N dernières games via gold/compare), pas par-game |
| Feedback loop | **Narration persistée** (payload+review horodatés) ; pas de scoring d'utilité dans cet incrément |
| Approche | **A** : 1 appel, payload déterministe, validation Pydantic |
| Client HTTP | `requests` brut (pas de lib `ollama`) — contrôle auth/retry/timeout, 0 dépendance en plus |
| Dépendance ajoutée | `pydantic>=2` |

## Architecture — `src/04_coaching/`

Cinq unités à responsabilité unique. Les deux premières sont **pures** (sans réseau),
donc entièrement testables hors-ligne.

```
src/04_coaching/
  payload.py      # gold aggregate.json (perso + ref cible) → payload coaching compact.
                  #   Sélection DÉTERMINISTE des signaux saillants. PUR.
                  #   N'expose QUE des métriques asymétrie-safe (positioning ⊂ COACHING_SAFE).
  prompt.py       # payload → (system, user). PUR. Encode asymétrie, phrasé benchmark-relatif,
                  #   FR, nuance profondeur, attentes de schéma.
  schema.py       # modèles Pydantic de la review + JSON-schema dérivé pour Ollama `format`.
  llm_client.py   # client Ollama Cloud (HTTP). generate_json(model, system, user, schema) → dict.
                  #   Lit OLLAMA_API_KEY via riotlib.load_env. Retries/backoff. Aucune logique métier.
  coach.py        # CLI orchestrateur : payload → prompt → client → validation Pydantic
                  #   → affiche + persiste (data/07_coaching/<player>/reviews.jsonl).
```

### Flux de données (0 nouvelle extraction — on relit le gold existant)

```
data/03_gold/personal/<player>/<scope>/aggregate.json   ─┐
data/03_gold/referentiel/<target>/<scope>/aggregate.json ─┼─► payload.build()
                                                          │       │ (dict compact, safe-only)
                                                          │       ▼
                                                    prompt.render() ─► (system, user)
                                                          │
                                                          ▼
                              llm_client.generate_json(model, …, schema) ──► Ollama Cloud
                                                          │
                                                          ▼
                              schema.Review.model_validate() ──► coach: print + append JSONL
```

### Choix de placement
- **`payload.py` relit le gold** plutôt que d'appeler `compare.py` (qui ne fait qu'imprimer).
  Petit recouvrement de logique « delta saillant » assumé : consommateurs distincts, pas de
  refactor de `compare` dans cet incrément.
- **Persistance sous `data/07_coaching/`** (pas dans le gold) : une review est une *sortie*,
  pas un agrégat. `data/` reste gitignoré.

## Contrat du payload

`payload.build(player, scope, target, outcome) -> dict`. Relit deux `aggregate.json`.
**Tout est soit une métrique vécue par le joueur, soit une valeur de référence challenger ;
jamais d'info cachée ennemie. `positioning` filtré sur `COACHING_SAFE` (3 proxys `ML_ONLY`
exclus à la source).**

```jsonc
{
  "meta": { "player": "spadzze", "scope": "adc", "target": "challenger",
            "outcome_focus": "loss", "patch": "16.13",
            "n_games_me": 15, "n_games_ref": 1026, "winrate_me": 0.67,
            "low_sample": true,                   // n_games_me < 30 → confidence abaissée
            "deaths_per_game": { "overall": {"you": 6.27, "ref": 5.68},
                                 "win":     {"you": 5.6,  "ref": 4.52},
                                 "loss":    {"you": 7.6,  "ref": 6.98} } },
  "signals": [
    { "group": "lane", "key": "csd14", "label": "CS diff @14",
      "you": 0, "ref": -5, "delta": 5, "unit": "cs", "notable": false },
    { "group": "positioning", "key": "frac_roam_mid", "label": "% roam (mid)",
      "you": 0.50, "ref": 0.70, "delta": -0.20, "unit": "pct", "notable": true },
    { "group": "positioning", "key": "max_map_depth", "label": "profondeur max",
      "you": 2728, "ref": 1633, "delta": 1095, "unit": "u",
      "notable": false, "descriptive_only": true },  // ⚠ profondeur ↑ = diamond, JAMAIS prescrit
    { "group": "deaths_zone_phase", "key": "BOT|mid", "label": "morts BOT en mid",
      "you": 0.29, "ref": 0.05, "delta": 0.24, "unit": "pct", "notable": true },
    { "group": "death_gold_state", "key": "behind", "label": "morts en retard",
      "you": 0.50, "ref": 0.54, "delta": -0.04, "unit": "pct", "notable": false }
  ],
  "context": {                                    // benchmark conditionné (cf. compare.py)
    "lane_pattern":  { "bucket": "all_in", "gd10_me": -429, "gd10_ref": 24,  "n_me": 7, "n_ref": 520 },
    "gank_exposure": { "bucket": "low",    "gd10_me": -379, "gd10_ref": 127, "n_me": 7, "n_ref": 396 }
  }
}
```

### Groupes de signaux
- **`lane`** : `gd10, gd14, gd20, csd10, csd14` à l'issue focus (perso vs réf, médianes).
- **`positioning`** : les 14 features `COACHING_SAFE` (médianes perso vs réf à l'issue focus).
- **`deaths_zone_phase`** : top écarts « où tu sur-meurs » (perso% − réf%), même calcul que `compare`.
- **`death_gold_state`** : parts ahead/even/behind des morts.

Les **morts/game** (overall+win+loss, contexte global) vivent dans `meta.deaths_per_game`
— pas dans `signals` (ce ne sont pas des écarts saillants tranchés mais un cadrage), donc
pas de flag `notable`.

### Salience déterministe (`notable`) — « les features concluent »
Seuils par métrique :
- lane : `notable` si `|Δgold| > 150` **ou** `|Δcs| ≥ 2`.
- positioning (fractions) : `notable` si `|Δ| ≥ 0.08`.
- positioning (profondeur : `avg_map_depth`, `max_map_depth`, `frac_overextended`) :
  **toujours `descriptive_only:true`, jamais `notable`**.
- deaths_zone_phase : `notable` si `Δ ≥ 0.08` (aligné sur `compare`).
- death_gold_state : `notable` si `|Δ| ≥ 0.10`.

### Défauts (configurables)
- `outcome_focus = loss` (le plus diagnostique). Morts/game fournies overall+win+loss ;
  le détail (lane, positioning, zone×phase, gold-state) sur l'issue focus.
- `target = challenger`.
- `low_sample = (n_games_me < 30)` → pilote la `confidence`.

## Client Ollama Cloud

```python
def generate_json(model: str, system: str, user: str, schema: dict,
                  temperature: float = 0.2, timeout: int = 180) -> dict:
    """POST https://ollama.com/api/chat (Bearer OLLAMA_API_KEY), format=<schema>,
    stream=false → JSON parsé du message. Erreur claire sinon."""
```
- Auth `Authorization: Bearer <OLLAMA_API_KEY>` via `riotlib.load_env()`. Erreur explicite si absente.
- Structured output natif : `format = schema` (JSON-schema dérivé de Pydantic).
- `temperature = 0.2` (narration stable/reproductible).
- Robustesse (calquée sur `RiotClient`) : retries + backoff sur 429/5xx/timeout ; `timeout=180s` ;
  401 → « vérifie `OLLAMA_API_KEY` ». Zéro logique métier.

## Schéma de sortie (Pydantic)

```python
class Insight(BaseModel):
    point: str       # affirmation FR
    evidence: str    # preuve chiffrée tirée du payload ("roam mid 50% vs 70% challenger")

class Review(BaseModel):
    strengths: list[Insight]   # exactement 3 (min=max=3)
    mistakes:  list[Insight]   # exactement 3 (priorisées)
    habits:    list[str]       # exactement 2 (habitudes à corriger)
    next_focus: str            # 1 focus pour la prochaine game
    confidence: float          # 0..1 (ge=0, le=1) — abaissée si meta.low_sample
```
- Longueurs fixes (3/3/2) → reflétées dans le JSON-schema (`minItems`/`maxItems`) passé à `format`.
  Ollama contraint + Pydantic re-valide ; déviation → 1 retry, puis échec propre (brut sauvé).
- **Évolution vs CLAUDE.md** : `evidence[]` global fusionné **dans chaque `Insight`** (plus
  vérifiable, pas de conseil sans preuve). CLAUDE.md sera mis à jour.
- `schema.py` expose `Review` **et** `Review.model_json_schema()`.

### Persistance — `data/07_coaching/<player>/reviews.jsonl` (append)
```jsonc
{ "ts": "2026-06-30T14:22:01", "model": "deepseek-v4-pro",
  "scope": "adc", "target": "challenger", "outcome_focus": "loss",
  "payload": { … },   // payload exact rejouable
  "review":  { … } }  // Review validée
```
Payload + review ensemble → éval future sans re-génération ; rejouable pour l'A/B modèles.

## Stratégie de prompt

`prompt.render(payload) -> (system, user)`, **pure**.

**System** (FR, structuré) : rôle = coach LoL personnel expert ; reçoit un JSON de signaux
**déjà calculés** (joueur vs benchmark challenger) ; rôle = **raconter & prioriser**, jamais
calculer ni inventer. Règles absolues :
1. **Asymétrie** — ne jamais reprocher une décision sur une info non disponible ; les `ref`
   sont des repères (« les challengers font Y »), jamais « tu aurais dû savoir X ».
2. **Preuve obligatoire** — chaque point cite sa stat du payload ; aucune stat hors payload.
3. **Priorité** — d'abord `notable:true` ; un `descriptive_only:true` = observation neutre,
   **jamais** une erreur (profondeur ↑ n'est PAS un défaut).
4. **Concret & benchmark-relatif** — « recall à 1450 g vs 1100 » ✅, « meurs moins » ❌.
5. `meta.low_sample:true` → abaisser `confidence` + signaler.
6. Français, tutoiement, concis. Respecter le schéma.

**User** : `"Signaux de tes {n_games_me} dernières games ({scope}, issue={outcome_focus},
vs {target}) :\n\n{payload_json}\n\nProduis la review."`

La **sélection** vit dans `payload.py` (déterministe) ; le prompt n'impose que le *cadre de
narration*. Itération qualité = prompt + seuils de salience, pas l'archi.

## Gestion d'erreurs

| Cas | Comportement |
|---|---|
| Gold perso ou réf. cible absent | message clair + exit 1 |
| `OLLAMA_API_KEY` manquante | erreur explicite avant tout appel |
| Réseau / timeout / 429 / 5xx | retries + backoff ; échec final → message + exit non-zéro |
| 401 | « vérifie `OLLAMA_API_KEY` » |
| Sortie LLM non conforme (malgré `format`) | 1 retry ; 2ᵉ échec → brut sauvé sous `data/07_coaching/<player>/failed/` + exit non-zéro |
| 0 signal `notable` | review produite quand même (forces = « au niveau sur X »), `confidence` ajustée |

## Plan de tests (`tests/`, pytest, zéro réseau par défaut)

| Module | Assertions |
|---|---|
| `payload` | seuils de salience (`notable`) ; **safe-only** (aucune clé `ML_ONLY` ne sort ; positioning ⊆ `COACHING_SAFE`) ; profondeur toujours `descriptive_only` ; flag `low_sample`. Gold synthétiques. |
| `prompt` | règle d'asymétrie + nuance profondeur présentes ; **aucun nom de feature `ML_ONLY`** dans le prompt rendu. |
| `schema` | `Review` accepte un bon objet ; rejette mauvaises longueurs (4 forces) et `confidence` hors [0,1]. |
| `llm_client` | `requests.post` monkeypatché → contenu canné ; parsing, header `Authorization`, passage de `format`. Chemins 401/timeout sur mock. |
| `coach` | `llm_client` monkeypatché → review cannée → écriture JSONL + affichage vérifiés. Test d'intégration réel marqué `skip` si pas de `OLLAMA_API_KEY`. |

## Critères de succès (incrément 1)

- `python3 src/04_coaching/coach.py --player spadzze --scope adc` produit une `Review` valide,
  l'affiche en FR, et l'append dans `data/07_coaching/spadzze/reviews.jsonl`.
- Tous les tests verts (réseau mocké) ; garde-fou asymétrie testé (aucun `ML_ONLY` dans payload/prompt).
- Au moins une review réelle générée via `deepseek-v4-pro`, relue manuellement : conseils
  concrets, benchmark-relatifs, avec preuve chiffrée, sans reproche basé sur info cachée.

## Hors scope (différé)

- Compte-rendu **par game** (fin de partie) — incrément ultérieur (payload par-game).
- **Scoring d'utilité** / boucle d'éval active — la persistance payload+review la rend
  possible plus tard, mais pas implémentée ici.
- A/B multi-modèles automatisé — manuel (changer `--model`) pour l'instant.
- Refactor de `compare.py` pour exposer une fonction de données partagée.
- Tout fine-tuning / RAG.
```
