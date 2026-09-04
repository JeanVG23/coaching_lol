"""04_coaching — générateur déterministe substituable au LLM (0 réseau).

Sert `make demo` : sans clé Ollama, la chaîne s'arrêterait juste avant l'étape la
plus visible du projet. Ce module n'imite PAS un modèle, il en occupe la place :
il relit le payload envoyé au modèle et en extrait mécaniquement quelques faits.

Deux propriétés le rendent utile plutôt que décoratif :

- il traverse le vrai chemin de production (`llm_client.generate` -> validation
  Pydantic -> persistance), donc la démo casse si le schéma ou le payload cassent ;
- il ne cite que des chiffres réellement présents dans le payload, donc
  `grounding.py` peut s'exécuter derrière et rendre un taux d'ancrage vrai.

Il ne raisonne pas : ses `point` sont des gabarits. Toute sortie produite ici est
marquée `"model": "mock"` et ne doit jamais être présentée comme du coaching.
"""
from __future__ import annotations

import json

import llm_client

MODEL_NAME = "mock"


def payload_from_user(user: str) -> dict:
    """Récupère le payload JSON encadré par le texte de `prompt.render*`.

    Le mock lit exactement ce que le modèle lit : si `prompt.py` cessait
    d'embarquer le payload, la démo échouerait au lieu d'inventer.
    """
    start, end = user.find("{"), user.rfind("}")
    if start < 0 or end <= start:
        raise llm_client.LLMError("payload introuvable dans le message utilisateur")
    return json.loads(user[start:end + 1])


def _fmt_gold(value) -> str:
    return f"{int(value)} g" if isinstance(value, (int, float)) else "0 g"


def _game_review(pl: dict) -> dict:
    deaths = (pl.get("journal") or {}).get("deaths") or []
    recalls = (pl.get("journal") or {}).get("recalls") or []
    worst = max(deaths, key=lambda d: d.get("unspent_gold") or 0, default=None)
    first = deaths[0] if deaths else None

    mistakes = []
    if worst:
        mistakes.append({
            "point": f"Tu meurs avec de l'or non dépensé en {worst.get('zone', '?')}.",
            "cause": (f"mort {'en solo' if worst.get('is_solo') else 'à plusieurs'} "
                      f"contre {worst.get('killer_champ') or 'un adversaire'}, "
                      f"phase {worst.get('phase', '?')}"),
            "evidence": (f"mort à {worst.get('clock', '0:00')} en "
                         f"{worst.get('zone', '?')}, "
                         f"{_fmt_gold(worst.get('unspent_gold'))} non dépensés"),
        })
    if first and first is not worst:
        mistakes.append({
            "point": f"Ta première mort arrive en {first.get('zone', '?')}.",
            "cause": f"tué par {first.get('killer_champ') or 'un adversaire'}",
            "evidence": (f"mort à {first.get('clock', '0:00')} en "
                         f"{first.get('zone', '?')}"),
        })
    if not mistakes:                       # journal sans mort : le schéma en exige une
        mistakes.append({
            "point": "Rien à reprocher sur les morts de cette game.",
            "cause": "aucune mort au journal",
            "evidence": "0 mort relevée entre 0:00 et la fin de partie",
        })

    strengths = []
    if recalls:
        r = recalls[0]
        strengths.append({
            "point": "Tu repasses à la base avec de quoi acheter.",
            "cause": "recall déclenché sur un seuil d'or, pas sur une mort",
            "evidence": (f"recall à {r.get('clock', '0:00')} avec "
                         f"{_fmt_gold(r.get('gold_before'))}"),
        })
    # La confiance suit la matière disponible : c'est la règle que le harnais
    # contrefactuel vérifie (journal vidé -> confiance en baisse).
    return {"strengths": strengths, "mistakes": mistakes[:3],
            "next_focus": "Dépenser avant de reprendre un duel.",
            "confidence": round(min(0.9, 0.2 + 0.1 * len(deaths)), 2)}


def _insight(point: str, evidence: str) -> dict:
    return {"point": point, "evidence": evidence}


def _fmt(signal: dict) -> str:
    unit = {"pct": " %", "g": " g", "cs": " cs", "u": " u"}.get(signal.get("unit"), "")
    scale = 100 if signal.get("unit") == "pct" else 1
    return (f"{signal['label']} : {round(signal['you'] * scale, 1)}{unit} "
            f"contre {round(signal['ref'] * scale, 1)}{unit} en référence")


def _review(pl: dict) -> dict:
    meta = pl.get("meta") or {}
    # `descriptive_only` écarté : la profondeur de carte se lit, elle ne se
    # reproche pas (manifeste d'asymétrie). Un mock qui l'ignorerait ferait
    # échouer le contrôle d'asymétrie de `grounding.py`, ce qui est le but.
    signals = [s for s in (pl.get("signals") or [])
               if s.get("notable") and not s.get("descriptive_only")
               and isinstance(s.get("you"), (int, float))
               and isinstance(s.get("ref"), (int, float))]
    signals.sort(key=lambda s: abs(s.get("delta") or 0), reverse=True)
    worst = signals[:3]
    deaths = (meta.get("deaths_per_game") or {}).get("loss") or {}

    mistakes = [_insight(f"Écart marqué sur {s['label']}.", _fmt(s)) for s in worst]
    while len(mistakes) < 3:                 # le schéma en exige exactement 3
        mistakes.append(_insight(
            "Tu meurs plus que la référence en défaite.",
            f"{deaths.get('you', 0)} morts/game contre {deaths.get('ref', 0)}"))
    return {
        "strengths": [_insight(
            "Le pool analysé est assez homogène pour être comparé.",
            f"{meta.get('n_games_me', 0)} games sur le scope, "
            f"contre {meta.get('n_games_ref', 0)} en référence")],
        "mistakes": mistakes[:3],
        "habits": ["Vérifier l'or non dépensé avant chaque duel.",
                   "Reculer d'un cran dès que la vision de rivière est perdue."],
        "next_focus": "Tenir l'avance acquise entre la 14e et la 20e minute.",
        "confidence": 0.3 if meta.get("low_sample") else 0.6,
    }


def generate(model: str, system: str, user: str, schema: dict,
             temperature: float = 0.2, timeout: int = 180):
    """Signature de `llm_client.generate` : substituable sans rien changer ailleurs."""
    pl = payload_from_user(user)
    data = _game_review(pl) if pl.get("journal") else _review(pl)
    return llm_client.Generation(data, {"latency_ms": 0, "prompt_tokens": 0,
                                        "completion_tokens": 0, "attempts": 1,
                                        "model": MODEL_NAME})
