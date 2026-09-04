"""Parité du schéma de clés KV entre le producteur Python et le Worker TypeScript.

Le sync (`src/core/kv_keys.py`) écrit les clés, le Worker (`web/cf/src/readers.ts`)
les lit : deux runtimes, donc deux tables, mais renommer une clé d'un seul côté
cassait le site en silence — sans erreur de type ni test. Ce test échoue à la
moindre divergence de nom, de gabarit ou de paramètre.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "core"))

import kv_keys  # noqa: E402

READERS_TS = ROOT / "web" / "cf" / "src" / "readers.ts"
# games: (slug: string) => `silver:${slug}:games`,
_ENTRY = re.compile(
    r"^\s*(?P<name>\w+):\s*\((?P<args>[^)]*)\)\s*=>\s*`(?P<tpl>[^`]*)`",
    re.M,
)


def _ts_templates() -> dict[str, str]:
    """Table `KEYS` de readers.ts, convertie au format des gabarits Python."""
    src = READERS_TS.read_text()
    block = src[src.index("export const KEYS = {"):]
    block = block[:block.index("\n};")]
    out = {}
    for m in _ENTRY.finditer(block):
        # `silver:${slug}:games` -> "silver:{slug}:games"
        out[m.group("name")] = re.sub(r"\$\{(\w+)\}", r"{\1}", m.group("tpl"))
    return out


def test_kv_key_templates_match():
    ts = _ts_templates()
    assert ts, f"aucune clé extraite de {READERS_TS} — le format de KEYS a changé"
    assert ts == kv_keys.TEMPLATES, (
        "divergence de schéma KV entre src/core/kv_keys.py et web/cf/src/readers.ts"
    )


def test_key_rejects_unknown_name():
    try:
        kv_keys.key("inconnue", slug="x")
    except KeyError:
        return
    raise AssertionError("une clé inconnue doit lever plutôt que produire une clé fausse")
