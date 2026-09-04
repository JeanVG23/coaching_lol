#!/usr/bin/env python3
"""compresse le cache raw (data/01_raw) en zstd — migration one-shot, 0 appel API.

Parcourt tous les `<matchId>_match.json` / `<matchId>_timeline.json` et les
recompresse en `.json.zst` (via riotlib._write_raw, même niveau ZSTD_LEVEL que le
pipeline). Vérifie le roundtrip (decompress == original) AVANT de supprimer
l'original — échec -> on garde l'original et on signale.

Idempotent : saute tout fichier `.json` dont le `.json.zst` existe déjà. Gère aussi
les éventuels `.json.gz` (décompresse gz puis recompresse en zst).

Usage:
    python3 src/pipeline_ops/compress_raw.py            # migration réelle
    python3 src/pipeline_ops/compress_raw.py --dry-run  # simule, n'écrit/supprime rien
    python3 src/pipeline_ops/compress_raw.py --jobs 8   # parallélisme (défaut: nb cores)
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import riotlib as rl

# Les workers héritent de ZSTD_LEVEL via rl. On importe zstandard ici aussi pour
# éviter un round-trip par fichier via le helper (plus direct, mais cohérent).
import zstandard as zstd

_CCTX = zstd.ZstdCompressor(level=rl.ZSTD_LEVEL)
_DCTX = zstd.ZstdDecompressor()


def _read_original(path):
    """Renvoie les bytes originaux d'un fichier raw (.json ou .json.gz)."""
    data = path.read_bytes()
    if path.name.endswith(".gz"):
        data = gzip.decompress(data)
    return data


def _compress_one(path_str: str) -> tuple[str, int, int, str]:
    """Compresse un fichier .json/.json.gz -> .json.zst avec vérification.
    Retourne (path, orig_bytes, zst_bytes, statut)."""
    path = rl.RAW_DIR / path_str
    name = path.name
    if name.endswith(".json"):
        zst_name = name + ".zst"
    elif name.endswith(".json.gz"):
        zst_name = name[:-3] + ".zst"  # foo.json.gz -> foo.json.zst
    else:
        return (path_str, 0, 0, "skip-not-json")
    zst_path = rl.RAW_DIR / zst_name
    if zst_path.exists():
        return (path_str, 0, 0, "already-zst")
    try:
        orig = _read_original(path)
        comp = _CCTX.compress(orig)
        zst_path.write_bytes(comp)
        # Vérification : décompresser et comparer AVANT de supprimer l'original.
        if _DCTX.decompress(comp) != orig:
            zst_path.unlink(missing_ok=True)
            return (path_str, len(orig), len(comp), "verify-failed")
        os.unlink(path)
        return (path_str, len(orig), len(comp), "ok")
    except Exception as e:  # ne supprime jamais l'original en cas d'erreur
        zst_path.unlink(missing_ok=True)
        return (path_str, 0, 0, f"error: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compresse le cache raw en zstd.")
    ap.add_argument("--dry-run", action="store_true", help="simule, n'écrit/supprime rien")
    ap.add_argument("--jobs", "-j", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help="nb workers (défaut: cores-1)")
    args = ap.parse_args()

    if not rl.RAW_DIR.exists():
        print(f"✗ {rl.RAW_DIR} introuvable.", file=sys.stderr)
        return 1

    # Cible : tous les .json et .json.gz (pas les .json.zst déjà faits).
    files = [p.name for p in rl.RAW_DIR.iterdir()
             if p.is_file() and (p.name.endswith(".json") or p.name.endswith(".json.gz"))]
    # Exclure ceux déjà migrés (le .json.zst correspondant existe).
    files = [n for n in files if not (rl.RAW_DIR / (n + ".zst" if n.endswith(".json")
                                  else n[:-3] + ".zst")).exists()]
    print(f"{len(files)} fichiers à compresser ({args.jobs} workers)"
          + (" [DRY-RUN]" if args.dry_run else ""))
    if not files:
        print("Rien à faire — cache raw déjà compressé.")
        return 0

    t0 = time.time()
    orig_total = zst_total = 0
    ok = skipped = failed = 0
    failures = []

    if args.dry_run:
        results = []
        for n in files:
            p = rl.RAW_DIR / n
            try:
                ob = len(_read_original(p))
                cb = len(_CCTX.compress(_read_original(p)))
                results.append((n, ob, cb, "dry-run"))
            except Exception as e:
                results.append((n, 0, 0, f"error: {e}"))
    else:
        with Pool(args.jobs) as pool:
            results = pool.map(_compress_one, files, chunksize=32)

    for name, ob, cb, status in results:
        if status in ("ok", "dry-run"):
            ok += 1
            orig_total += ob
            zst_total += cb
        elif status in ("already-zst", "skip-not-json"):
            skipped += 1
        else:
            failed += 1
            failures.append((name, status))

    dt = time.time() - t0
    print(f"\n{ok} compressés, {skipped} ignorés, {failed} échecs en {dt:.1f}s")
    if orig_total:
        ratio = orig_total / zst_total if zst_total else 0
        print(f"Taille raw concernée : {orig_total/1e6:.0f} Mo -> {zst_total/1e6:.0f} Mo "
              f"(×{ratio:.1f})")
    if failures:
        print(f"\n⚠ {len(failures)} échecs (originaux conservés) :", file=sys.stderr)
        for n, s in failures[:20]:
            print(f"  {n}: {s}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())