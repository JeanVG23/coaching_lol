#!/usr/bin/env python3
"""build_demo_fixtures — fabrique le jeu de données démo versionné (0 appel API).

`data/` est intégralement gitignoré : après un `git clone`, personne ne peut rien
lancer. Ce script prélève quelques dizaines de games dans le cache local, les
**pseudonymise**, et écrit une racine de données autonome dans `tests/fixtures/demo/`.
`make demo` la copie dans un répertoire jetable puis rejoue les scripts de
production dessus (`reextract_silver` -> `rebuild_gold` -> `compare` -> `coach`).

Pseudonymisation : aucun identifiant Riot réel ne doit survivre dans le dépôt.
Le remplacement ne se fie pas à une liste de clés (un champ oublié = une fuite) :
on collecte d'abord tous les identifiants réels des games retenues, puis on
réécrit **toute chaîne du JSON** qui en fait partie, quelle que soit sa place.
`--audit` relit ce qui a été écrit et échoue si une valeur réelle a survécu.

Usage :
    poetry run python3 src/pipeline_ops/build_demo_fixtures.py [--personal 20]
                       [--referential 30] [--rank challenger] [--player spadzze]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import champion_profiles as cp
import riotlib as rl

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "tests" / "fixtures" / "demo"
DEMO_PLATFORM = "DEMO1"  # `readers.matchSeq` trie sur la queue numérique de l'id


# --- sélection ----------------------------------------------------------------

def _has_raw(match_id: str) -> bool:
    return all(rl._raw_path(f"{match_id}_{kind}") is not None
               for kind in ("match", "timeline"))


def pick(games: list[dict], n_matches: int, *, patch: str,
         queue: int) -> list[dict]:
    """Toutes les lignes silver des `n_matches` parties les plus récentes éligibles.

    On sélectionne par MATCH, pas par ligne, puis on reprend toutes les lignes de
    ces matchs. Le coût en dépôt est celui du raw (~55 Ko par partie) : prendre les
    9 joueurs extraits d'une partie déjà téléchargée densifie les agrégats sans
    peser un octet de plus.
    """
    eligible = [g for g in games
                if g.get("patch") == patch and g.get("queue") == queue]
    keep, seen = [], set()
    for mid in sorted({g["match_id"] for g in eligible}, reverse=True):
        if not _has_raw(mid):
            continue
        seen.add(mid)
        if len(seen) >= n_matches:
            break
    keep = [g for g in eligible if g["match_id"] in seen]
    return sorted(keep, key=lambda g: (g["match_id"], g["puuid"]))


# --- pseudonymisation ---------------------------------------------------------

NAME_KEYS = ("riotIdGameName", "summonerName")  # seuls champs portant un pseudo


class Anonymizer:
    """Deux tables, parce que deux natures d'identifiants.

    Les identifiants **opaques** (`puuid`, `summonerId`, `matchId`) ne veulent rien
    dire ailleurs : on les remplace partout, par balayage récursif, ce qui rend
    impossible l'oubli d'un champ. Les **pseudonymes** sont remplacés uniquement aux
    clés qui les portent : un joueur nommé « Aatrox » existe, et un remplacement
    global réécrirait le `championName` de toutes les games, en silence.
    """

    def __init__(self) -> None:
        self.opaque: dict[str, str] = {}
        self.names: dict[str, str] = {}
        self._n = {"puuid": 0, "summoner": 0, "name": 0, "match": 0}

    def _add(self, table: dict, real: str, kind: str, template: str) -> str:
        if real in table:
            return table[real]
        self._n[kind] += 1
        table[real] = template.format(self._n[kind])
        return table[real]

    def puuid(self, real: str) -> str:
        return self._add(self.opaque, real, "puuid", "DEMO-PUUID-{:04d}")

    def summoner(self, real: str) -> str:
        return self._add(self.opaque, real, "summoner", "DEMO-SUMMONER-{:04d}")

    def game_name(self, real: str) -> str:
        return self._add(self.names, real, "name", "Joueur{:03d}")

    def match_id(self, real: str) -> str:
        return self._add(self.opaque, real, "match", DEMO_PLATFORM + "_{:07d}")

    def scrub(self, node):
        """Remplace récursivement tout identifiant opaque connu."""
        if isinstance(node, str):
            return self.opaque.get(node, node)
        if isinstance(node, list):
            return [self.scrub(x) for x in node]
        if isinstance(node, dict):
            return {k: self.scrub(v) for k, v in node.items()}
        return node


def register(anon: Anonymizer, match: dict) -> None:
    """Enregistre tous les identifiants d'une game AVANT toute réécriture."""
    anon.match_id(match["metadata"]["matchId"])
    for p in match["info"]["participants"]:
        anon.puuid(p["puuid"])
        if isinstance(p.get("summonerId"), str) and p["summonerId"]:
            anon.summoner(p["summonerId"])
        for key in NAME_KEYS:
            if p.get(key):
                anon.game_name(p[key])


def scrub_match(anon: Anonymizer, match: dict, game_id: int) -> dict:
    out = anon.scrub(match)
    out["info"]["gameId"] = game_id
    for p in out["info"]["participants"]:
        for key in NAME_KEYS:
            if p.get(key):
                p[key] = anon.names[p[key]]
        if "riotIdTagline" in p:
            p["riotIdTagline"] = "DEMO"
    return out


# --- catalogues statiques -----------------------------------------------------

def trim_champions(raw: dict) -> dict:
    """Ne garde que ce que `load_ddragon` lit : `attackrange` et `tags`.

    On garde TOUS les champions (17 Ko une fois élagué) plutôt que les seuls
    présents dans les fixtures : un champion manquant se dégrade silencieusement
    en `unknown` et viderait `derive_context` de son sens.
    """
    return {"data": {k: {"id": c["id"],
                         "stats": {"attackrange": c.get("stats", {}).get("attackrange")},
                         "tags": c.get("tags", [])}
                     for k, c in raw["data"].items()}}


def trim_items(raw: dict) -> dict:
    """Ne garde que ce que `_parse_items` lit : le nom et le coût total."""
    return {"data": {k: {"name": it.get("name", ""),
                         "gold": {"total": it.get("gold", {}).get("total")}}
                     for k, it in raw["data"].items()}}


# --- écriture -----------------------------------------------------------------

def write_static(out: Path) -> None:
    static = out / "00_static"
    static.mkdir(parents=True, exist_ok=True)
    shutil.copy(cp.TRAITS_PATH, static / "champion_traits.json")
    src = cp.STATIC_DIR / "ddragon" / cp.DDRAGON_VERSION
    dest = static / "ddragon" / cp.DDRAGON_VERSION
    dest.mkdir(parents=True, exist_ok=True)
    for name, trim in (("championFull.json", trim_champions), ("item.json", trim_items)):
        if not (src / name).exists():
            sys.exit(f"✗ catalogue absent : {src / name} (lancer fetch_ddragon d'abord)")
        (dest / name).write_text(json.dumps(trim(json.loads((src / name).read_text())),
                                            ensure_ascii=False, separators=(",", ":")))


def build(player: str, rank: str, n_personal: int, n_ref: int,
          patch: str, queue: int) -> dict:
    personal = pick(rl.read_jsonl(rl.silver_games(rl.KIND_PERSONAL, player)),
                    n_personal, patch=patch, queue=queue)
    referential = pick(rl.read_jsonl(rl.silver_games(rl.KIND_REF, rank)),
                       n_ref, patch=patch, queue=queue)
    if not personal or not referential:
        sys.exit("✗ pas assez de games avec raw disponible pour ce patch/queue.")

    if OUT.exists():
        shutil.rmtree(OUT)
    raw_dir = OUT / "01_raw"
    raw_dir.mkdir(parents=True)

    anon = Anonymizer()
    selected = personal + referential
    # Deux passes : tout enregistrer avant de réécrire, sinon un identifiant vu
    # dans la game 12 ne serait pas remplacé dans la game 3 déjà écrite.
    loaded = {}
    for i, g in enumerate(sorted(selected, key=lambda g: g["match_id"])):
        mid = g["match_id"]
        if mid in loaded:
            continue
        match = rl._read_raw(f"{mid}_match")
        timeline = rl._read_raw(f"{mid}_timeline")
        register(anon, match)
        loaded[mid] = (match, timeline, i)

    for mid, (match, timeline, i) in loaded.items():
        new_id = anon.opaque[mid]
        for kind, payload in (("match", scrub_match(anon, match, i + 1)),
                              ("timeline", anon.scrub(timeline))):
            rl._write_raw_at(raw_dir / f"{new_id}_{kind}.json.zst", payload)

    # Le silver versionné n'est qu'une AMORCE : `reextract_silver` réécrit chaque
    # record depuis le raw et ne lit que ces trois champs. Embarquer les records
    # complets (800 Ko) ferait doublon avec le raw, et surtout ferait vieillir une
    # copie figée à chaque évolution d'`extract_game`.
    silver = OUT / "02_silver"
    for kind, name, rows in ((rl.KIND_PERSONAL, player, personal),
                             (rl.KIND_REF, rank, referential)):
        path = silver / kind / name / "games.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rl.write_jsonl(path, [{"match_id": anon.opaque[r["match_id"]],
                               "puuid": anon.opaque[r["puuid"]],
                               "rank": r.get("rank")} for r in rows])

    write_static(OUT)
    manifest = {
        "player": player, "rank": rank, "patch": patch, "queue": queue,
        "n_rows_personal": len(personal), "n_rows_referential": len(referential),
        "n_matches": len(loaded), "n_identifiers_remapped": len(anon.opaque) + len(anon.names),
        "note": ("Games réelles pseudonymisées : puuid, summonerId, riot id et "
                 "match id remplacés par des jetons déterministes. Régénérer avec "
                 "src/pipeline_ops/build_demo_fixtures.py."),
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"manifest": manifest, "anon": anon}


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for x in node:
            yield from _strings(x)
    elif isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from _strings(v)


def audit(anon: Anonymizer) -> int:
    """Relit ce qui a été écrit et échoue si un identifiant réel a survécu.

    Deux contrôles, chacun adapté à sa nature : inclusion brute pour les opaques
    (ils ne peuvent pas apparaître légitimement), et vérification que TOUT champ de
    pseudonyme vaut bien un jeton démo. Chercher les pseudonymes partout donnerait
    des faux positifs à la pelle (le joueur « Aatrox », le perk « Mercenary »).
    """
    leaks = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file():
            continue
        if path.name.endswith(".json.zst"):
            doc = rl._read_raw_at(path)
        elif path.suffix == ".jsonl":
            doc = [json.loads(line) for line in path.read_text().splitlines() if line]
        elif path.suffix == ".json":
            doc = json.loads(path.read_text())
        else:
            continue
        blob = "\n".join(set(_strings(doc)))
        for real in anon.opaque:
            if real in blob:
                leaks.append((path.relative_to(OUT), real, "identifiant opaque"))
        for p in (doc.get("info", {}).get("participants", [])
                  if isinstance(doc, dict) else []):
            for key in NAME_KEYS:
                if p.get(key) and not p[key].startswith("Joueur"):
                    leaks.append((path.relative_to(OUT), p[key], f"pseudonyme ({key})"))
    for path, real, how in leaks[:10]:
        print(f"  ✗ {path} contient {real[:16]}… ({how})", file=sys.stderr)
    return len(leaks)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="build_demo_fixtures.py", description=__doc__)
    ap.add_argument("--player", default="spadzze")
    ap.add_argument("--rank", default="challenger")
    ap.add_argument("--personal", type=int, default=24,
                    help="nombre de MATCHS (toutes leurs lignes silver sont reprises)")
    ap.add_argument("--referential", type=int, default=25)
    ap.add_argument("--patch", default="16.13")
    ap.add_argument("--queue", type=int, default=rl.QUEUE_SOLO)
    args = ap.parse_args(argv)

    built = build(args.player, args.rank, args.personal, args.referential,
                  args.patch, args.queue)
    n_leaks = audit(built["anon"])
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    m = built["manifest"]
    print(f"  {m['n_matches']} matchs pseudonymisés -> "
          f"{m['n_rows_personal']} lignes silver perso "
          f"+ {m['n_rows_referential']} référentiel")
    print(f"  {m['n_identifiers_remapped']} identifiants remplacés")
    print(f"  {size / 1e6:.1f} Mo dans {OUT.relative_to(ROOT)}")
    if n_leaks:
        print(f"\n✗ AUDIT ÉCHOUÉ : {n_leaks} fuites d'identifiants.", file=sys.stderr)
        return 1
    print("\n✓ audit : aucun identifiant réel dans les fixtures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
