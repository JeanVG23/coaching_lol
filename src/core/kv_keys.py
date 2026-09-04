"""src/core/kv_keys.py — schéma des clés Cloudflare KV (côté Python).

Le producteur (`sync_cloudflare.py`) construisait ces clés en f-strings dispersées
et le consommateur (`web/cf/src/readers.ts`) porte la même table en TypeScript :
contrat inter-langages sans point unique de vérité, où renommer une clé d'un côté
casse le site en silence, sans erreur de type ni de test.

Deux runtimes, donc deux tables (pas de runtime commun), mais une seule table par
langage et un test de parité qui échoue à la moindre divergence
(cf. tests/test_kv_keys_parity.py).
"""
from __future__ import annotations

# {nom logique: gabarit}. Les paramètres nommés correspondent 1:1 à ceux de
# `KEYS` dans web/cf/src/readers.ts.
TEMPLATES = {
    "games": "silver:{slug}:games",
    "rank": "silver:{slug}:rank",
    "gold": "gold:{slug}:{scope}",
    "ref": "ref:{rank}:{scope}",
    "pred": "pred:{slug}",
    "shap": "shap:{slug}:drivers",
    "reviews": "coaching:{slug}:reviews",
    "feedback": "coaching:{slug}:feedback",
    "chats": "coaching:{slug}:chats",
}


def key(name: str, **params: str) -> str:
    """Clé KV pour `name`. Lève KeyError sur un nom inconnu, et le formatage
    lève sur un paramètre manquant (plutôt que produire une clé silencieusement
    fausse comme une f-string mal recopiée)."""
    return TEMPLATES[name].format(**params)
