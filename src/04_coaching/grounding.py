#!/usr/bin/env python3
"""04_coaching — vérifications d'ancrage des reviews persistées (0 réseau, 0 LLM).

Le prompt pose trois règles dures que rien ne vérifiait :

1. « n'invente aucune stat absente du payload » (règle 2) — aucun contrôle.
2. l'horodatage `mm:ss` : le schéma `AnchoredInsight` vérifie sa PRÉSENCE, pas sa
   VÉRACITÉ. Une evidence citant 17:05 alors que la mort est à 14:22 passe la
   validation Pydantic.
3. asymétrie : les features `descriptive_only` (profondeur, over-extension) sont
   des observations neutres, jamais des fautes.

Ces trois règles sont mécaniquement vérifiables sur les reviews DÉJÀ persistées :
elles donnent un score d'hallucination sans annoter quoi que ce soit, et elles
tournent en CI. Elles ne remplacent pas l'annotation humaine (qui juge l'UTILITÉ) :
elles mesurent la FIDÉLITÉ au payload, qui en est le préalable.

Usage :
  python3 src/04_coaching/grounding.py --player spadzze [--json] [--details]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))  # accès src/core/

import feedback as feedback_mod

# --- extraction des nombres cités --------------------------------------------

_CLOCK_RE = re.compile(r"\b(\d{1,2}):([0-5]\d)\b")
# 1 225 / 1 225 (espace fine) / 11,3 / 0.85 — jamais la partie d'un mm:ss (exclu
# en retirant les horloges du texte au préalable).
_NUM_RE = re.compile(r"\d{1,3}(?:[\s  ]\d{3})+|\d+(?:[.,]\d+)?")
_SPACES = str.maketrans({" ": "", " ": "", " ": ""})

# Tolérances de rapprochement d'un nombre cité à une valeur du payload.
EXACT_EPS = 0.05          # écart d'arrondi d'affichage
# Tolérance d'arrondi calibrée par contrôle négatif (cf. tests/test_grounding.py) :
# à 5 % le détecteur ne repérait que 56 % des chiffres falsifiés, à 1 % il en
# repère 93 % sans perdre les citations légitimes (95 % d'ancrage réel).
ROUNDED_REL = 0.01
CLOCK_NEAR_S = 30         # horodatage voisin d'un événement réel


def cited_numbers(text: str) -> list[tuple[str, float, str]]:
    """[(brut, valeur, unité)] du texte, horloges exclues."""
    stripped = _CLOCK_RE.sub(" ", text)
    out = []
    for m in _NUM_RE.finditer(stripped):
        raw = m.group(0)
        try:
            value = float(raw.translate(_SPACES).replace(",", "."))
        except ValueError:
            continue
        out.append((raw, value, unit_of_citation(stripped[m.end():m.end() + 8])))
    return out


def cited_clocks(text: str) -> list[str]:
    return [f"{int(m.group(1))}:{m.group(2)}" for m in _CLOCK_RE.finditer(text)]


# --- valeurs disponibles dans le payload -------------------------------------
#
# Rapprocher un chiffre cité de N'IMPORTE quelle valeur du payload ne prouve rien :
# un journal de game contient des centaines de nombres, et un contrôle négatif
# (falsifier les chiffres d'une review réelle) ne détectait que ~30 % des
# falsifications. On indexe donc les valeurs PAR UNITÉ, et un « 1 225 g » ne peut
# s'ancrer que sur un champ de gold, pas sur un timestamp qui passait par là.

UNITS = ("g", "cs", "pct", "s", "min", "u", "n", "morts")
ANY = "any"

# Unité déduite du nom de champ. Ordre significatif : le premier motif gagne.
_KEY_UNITS = (
    (("gold", "gd10", "gd14", "gd20", "cost", "price"), "g"),
    (("csd", "cs_", "creep"), "cs"),
    (("delta_s", "dead_time", "duration_s", "_seconds"), "s"),
    (("minute", "duration_min"), "min"),
    (("depth", "dist"), "u"),
    (("level", "kills", "deaths", "assists", "items_bought", "n_games",
      "wards", "count"), "n"),
)
# `"unit"` déclarée par les signaux du payload agrégé -> bloc de valeurs.
_DECLARED_UNITS = {"g": "g", "cs": "cs", "s": "s", "u": "u", "pct": "pct",
                   "min": "min", "ward": "n", "n": "n"}
# Champs internes jamais cités tels quels : les inclure gonflait l'espace des
# valeurs plausibles sans qu'un coach puisse légitimement les citer.
_IGNORED_KEYS = ("t_ms",)


def _unit_of(key: str) -> str | None:
    low = key.lower()
    for patterns, unit in _KEY_UNITS:
        if any(pattern in low for pattern in patterns):
            return unit
    return None


def _walk(node, key: str, out: dict[str, set[float]], inherited: str | None = None) -> None:
    """L'unité se transmet du parent à l'enfant : le payload agrégé range les
    valeurs sous `{"gd14": {"me": .., "ref": .., "delta": ..}}`, et lire l'unité
    sur la feuille (`me`) rangeait des gold parmi les dénombrements."""
    if isinstance(node, bool) or node is None:
        return
    own = _unit_of(key) if key else None
    if isinstance(node, (int, float)):
        if key in _IGNORED_KEYS:
            return
        value = abs(float(node))
        # Une valeur strictement fractionnaire est une part, quel que soit le nom
        # du champ : `death_gold_state.ahead = 0.2538` porte « gold » dans son
        # chemin sans être un montant d'or.
        unit = "pct" if 0.0 < value < 1.0 else (own or inherited or "n")
        out[unit].add(value)
        if unit == "pct":
            out["pct"].add(round(value * 100, 4))
        return
    if isinstance(node, dict):
        # Le payload agrégé DÉCLARE l'unité de chaque signal (`"unit": "g"`) :
        # une déclaration explicite prime sur l'heuristique de nom de champ.
        declared = _DECLARED_UNITS.get(node.get("unit")) if isinstance(
            node.get("unit"), str) else None
        for child_key, child in node.items():
            _walk(child, child_key, out, declared or own or inherited)
        return
    if isinstance(node, list):
        for child in node:
            _walk(child, key, out, own or inherited)


def _strings_and_keys(node) -> list[str]:
    """Textes du payload : les noms de métriques portent des nombres (`gd14`,
    « cs diff @10 ») qu'une evidence cite légitimement."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [k for k in node] + [t for child in node.values()
                                    for t in _strings_and_keys(child)]
    if isinstance(node, list):
        return [t for child in node for t in _strings_and_keys(child)]
    return []


def _add_derived(payload: dict, out: dict[str, set[float]]) -> None:
    """Dénombrements et parts dérivables du journal : « 4 morts en BOT »,
    « 60 % de tes morts », « avant la 12e minute » sont ancrés sans figurer
    tels quels dans le payload."""
    journal = payload.get("journal") or {}
    kda = ((payload.get("meta") or {}).get("kda") or {})
    if isinstance(kda.get("deaths"), (int, float)):
        out["morts"].add(float(kda["deaths"]))
    for name in ("deaths", "recalls"):
        rows = [r for r in (journal.get(name) or []) if isinstance(r, dict)]
        total = len(rows)
        out["n"].add(float(total))
        # « 4 morts en BOT » est la citation la plus fréquente d'un dénombrement :
        # la cloisonner évite qu'elle s'ancre sur un niveau ou un nombre de kills.
        counted = out["morts"] if name == "deaths" else out["n"]
        counted.add(float(total))
        groups = [Counter(r.get(field) for r in rows)
                  for field in ("zone", "phase", "gold_state",
                                "killer_champ", "killer_role")]
        groups.append(Counter((r.get("zone"), r.get("phase")) for r in rows))
        for counts in groups:
            for count in counts.values():
                out["n"].add(float(count))
                counted.add(float(count))
                if total:
                    out["pct"].add(round(100.0 * count / total, 4))
        for row in rows:
            clock = row.get("clock")
            if clock:
                minutes, seconds = clock.split(":")
                out["min"].add(float(int(minutes) + (1 if int(seconds) else 0)))


def payload_index(payload: dict) -> dict[str, set[float]]:
    """{unité: valeurs citables}. `ANY` reste le repli des citations sans unité
    explicite, et porte en plus les nombres des NOMS de métriques (`@14`)."""
    out: dict[str, set[float]] = {unit: set() for unit in UNITS}
    _walk(payload, "", out)
    _add_derived(payload, out)
    names: set[float] = set()
    for text in _strings_and_keys(payload):
        names.update(abs(value) for _, value, _ in cited_numbers(text))
    out[ANY] = names
    return out


def payload_clocks(payload: dict) -> set[str]:
    journal = payload.get("journal") or {}
    return {row["clock"] for name in ("deaths", "recalls")
            for row in (journal.get(name) or [])
            if isinstance(row, dict) and row.get("clock")}


def _clock_seconds(clock: str) -> int:
    minutes, seconds = clock.split(":")
    return int(minutes) * 60 + int(seconds)


# Unité portée par le texte qui suit immédiatement le nombre.
_EXACT_SUFFIXES = {"g": "g", "or": "g", "cs": "cs", "s": "s"}
_PREFIX_SUFFIXES = (("sec", "s"), ("min", "min"), ("unit", "u"),
                    ("mort", "morts"), ("déc", "morts"), ("dec", "morts"))


def unit_of_citation(tail: str) -> str:
    """Unité d'un « 1 225 g » / « 32,7 % » / « 18 s ». Sans unité explicite, la
    citation est un dénombrement ou une minute (« 3 morts », « la 12e minute »)."""
    token = tail.strip().lower()
    if token.startswith("%"):
        return "pct"
    word = re.match(r"[a-zà-ÿ]+", token)
    if word:
        found = word.group(0)
        if found in _EXACT_SUFFIXES:
            return _EXACT_SUFFIXES[found]
        for prefix, unit in _PREFIX_SUFFIXES:
            if found.startswith(prefix):
                return unit
    return ANY


# --- classement d'un nombre / d'une horloge ----------------------------------

def classify_number(value: float, index: dict[str, set[float]],
                    unit: str = ANY) -> str:
    """'exact' | 'arrondi' | 'non_ancre', dans l'unité citée."""
    value = abs(value)
    available = index.get(unit) or set()
    if unit == ANY:
        # Sans unité : un dénombrement, une minute, une distance, ou le nombre
        # porté par un nom de métrique (« gd14 », « @20 »). Pas un gold, toujours
        # cité avec son unité. Une fraction brute (« 0,29 des morts ») reste
        # rapprochable du bloc de pourcentages.
        buckets = ["n", "min", "u", ANY] + (["pct"] if value < 1 else [])
        available = set().union(*(index.get(b) or set() for b in buckets))
    elif unit == "min":
        # « gold diff @14 » : le repère de minute vient du NOM de la métrique.
        available = available | (index.get(ANY) or set())
    best = min((abs(value - c) for c in available), default=None)
    if best is None:
        return "non_ancre"
    if best <= EXACT_EPS:
        return "exact"
    if best <= max(EXACT_EPS, ROUNDED_REL * abs(value)):
        return "arrondi"
    # Une part dérivée s'énonce en entier (« 22 % de tes morts » pour 2/9 = 22,2).
    if value == int(value) and any(round(c) == value for c in available):
        return "arrondi"
    return "non_ancre"


def classify_clock(clock: str, available: set[str]) -> str:
    if clock in available:
        return "exact"
    if not available:
        return "non_ancre"
    ref = _clock_seconds(clock)
    if min(abs(ref - _clock_seconds(c)) for c in available) <= CLOCK_NEAR_S:
        return "voisin"
    return "non_ancre"


# --- asymétrie : une observation descriptive n'est jamais une faute ----------

# Les features `descriptive_only` du payload (profondeur de carte, over-extension)
# corrèlent au rang INFÉRIEUR : les prescrire inverserait le conseil.
DESCRIPTIVE_TERMS = ("profondeur", "over-extend", "overextens", "surextension",
                     "sur-extension")
PRESCRIPTIVE_SECTIONS = ("mistakes", "habits", "next_focus")


def asymmetry_violations(review: dict) -> list[str]:
    out = []
    for section in PRESCRIPTIVE_SECTIONS:
        for text in _texts(review.get(section)):
            low = text.lower()
            for term in DESCRIPTIVE_TERMS:
                if term in low:
                    out.append(f"{section}: « {text[:90]} »")
                    break
    return out


def _texts(node) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [v for k, v in node.items()
                if isinstance(v, str) and k in ("point", "cause", "evidence")]
    if isinstance(node, list):
        return [t for child in node for t in _texts(child)]
    return []


# --- rapport ------------------------------------------------------------------

def _insight_texts(review: dict) -> list[str]:
    """Textes porteurs de preuve. `point` est la leçon (formulation libre),
    `evidence` et `cause` sont les seuls tenus d'être ancrés."""
    out = []
    for section in ("strengths", "mistakes"):
        for insight in review.get(section) or []:
            if isinstance(insight, dict):
                out += [insight[k] for k in ("evidence", "cause")
                        if isinstance(insight.get(k), str)]
    return out


def check_review(record: dict) -> dict:
    payload, review = record.get("payload") or {}, record.get("review") or {}
    index, clocks = payload_index(payload), payload_clocks(payload)
    numbers, times = [], []
    for text in _insight_texts(review):
        for raw, value, unit in cited_numbers(text):
            numbers.append({"raw": f"{raw} {unit}".strip(), "unit": unit,
                            "status": classify_number(value, index, unit)})
        for clock in cited_clocks(text):
            times.append({"raw": clock, "status": classify_clock(clock, clocks)})
    return {
        "ts": record.get("ts"),
        "kind": record.get("kind", "aggregate"),
        "match_id": record.get("match_id"),
        "model": record.get("model"),
        "prompt_version": (record.get("run") or {}).get("prompt_version"),
        "numbers": numbers,
        "clocks": times,
        "asymmetry_violations": asymmetry_violations(review),
    }


def _rate(rows: list[dict], good: tuple[str, ...]) -> float | None:
    return (sum(1 for r in rows if r["status"] in good) / len(rows)) if rows else None


def score(check: dict) -> dict:
    return {
        "n_numbers": len(check["numbers"]),
        "grounded_rate": _rate(check["numbers"], ("exact", "arrondi")),
        "exact_rate": _rate(check["numbers"], ("exact",)),
        "n_clocks": len(check["clocks"]),
        "clock_rate": _rate(check["clocks"], ("exact", "voisin")),
        "clock_exact_rate": _rate(check["clocks"], ("exact",)),
        "n_asymmetry_violations": len(check["asymmetry_violations"]),
    }


def report(player: str, root=None, kind: str | None = None) -> dict:
    records = feedback_mod.list_reviews(player, root)
    if kind:
        records = [r for r in records
                   if (r.get("kind") or "aggregate") == kind]
    checks = [check_review(r) for r in records]
    numbers = [n for c in checks for n in c["numbers"]]
    clocks = [c for check in checks for c in check["clocks"]]
    return {
        "player": player,
        "n_reviews": len(checks),
        "numbers": {"n": len(numbers),
                    "grounded_rate": _rate(numbers, ("exact", "arrondi")),
                    "exact_rate": _rate(numbers, ("exact",)),
                    "by_status": dict(Counter(n["status"] for n in numbers))},
        "clocks": {"n": len(clocks),
                   "anchored_rate": _rate(clocks, ("exact", "voisin")),
                   "exact_rate": _rate(clocks, ("exact",)),
                   "by_status": dict(Counter(c["status"] for c in clocks))},
        "n_asymmetry_violations": sum(len(c["asymmetry_violations"]) for c in checks),
        "reviews": [{**{k: c[k] for k in ("ts", "kind", "match_id", "model",
                                          "prompt_version")},
                     **score(c)} for c in checks],
        "offenders": [{"ts": c["ts"], "match_id": c["match_id"],
                       "numbers": [n["raw"] for n in c["numbers"]
                                   if n["status"] == "non_ancre"],
                       "clocks": [t["raw"] for t in c["clocks"]
                                  if t["status"] == "non_ancre"],
                       "asymmetry": c["asymmetry_violations"]}
                      for c in checks
                      if any(n["status"] == "non_ancre" for n in c["numbers"])
                      or any(t["status"] == "non_ancre" for t in c["clocks"])
                      or c["asymmetry_violations"]],
    }


def render(rep: dict, details: bool = False) -> str:
    pct = lambda v: "—" if v is None else f"{v:.0%}"
    lines = [f"ANCRAGE — {rep['player']} ({rep['n_reviews']} reviews)",
             f"  Chiffres ancrés  {pct(rep['numbers']['grounded_rate'])} "
             f"({rep['numbers']['n']} cités, dont exacts "
             f"{pct(rep['numbers']['exact_rate'])})",
             f"  Horodatages      {pct(rep['clocks']['anchored_rate'])} "
             f"({rep['clocks']['n']} cités, dont exacts "
             f"{pct(rep['clocks']['exact_rate'])})",
             f"  Asymétrie        {rep['n_asymmetry_violations']} violation(s) "
             f"(observation descriptive présentée comme une faute)"]
    if details and rep["offenders"]:
        lines.append("\nÉléments non ancrés :")
        for off in rep["offenders"]:
            what = off["match_id"] or "agrégée"
            lines.append(f"  {off['ts']} | {what}")
            if off["numbers"]:
                lines.append(f"      chiffres : {', '.join(off['numbers'])}")
            if off["clocks"]:
                lines.append(f"      horaires : {', '.join(off['clocks'])}")
            for violation in off["asymmetry"]:
                lines.append(f"      asymétrie : {violation}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="grounding.py", description=__doc__)
    ap.add_argument("--player", default="spadzze")
    ap.add_argument("--kind", choices=["game", "aggregate"], default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--details", action="store_true",
                    help="liste les chiffres et horodatages non ancrés")
    args = ap.parse_args(argv)
    rep = report(args.player, kind=args.kind)
    print(json.dumps(rep, ensure_ascii=False, indent=2) if args.json
          else render(rep, args.details))
    return 0


if __name__ == "__main__":
    sys.exit(main())
