#!/usr/bin/env python3
"""
archive_patch — archive les données du patch courant avant de passer au suivant.

Déplace le contenu complet des dossiers raw, silver, gold et dataset vers
un sous-dossier d'archive spécifique au patch.
"""
import json
import shutil
import sys
from pathlib import Path

import riotlib as rl

def main():
    # Déterminer le patch actuel depuis sources.json
    sources_path = rl.SILVER_DIR / "referentiel" / "challenger" / "sources.json"
    if not sources_path.exists():
        # Essayer un autre rang si challenger n'existe pas
        for rank in ["grandmaster", "master", "diamond"]:
            alt = rl.SILVER_DIR / "referentiel" / rank / "sources.json"
            if alt.exists():
                sources_path = alt
                break
                
    if not sources_path.exists():
        print("✗ Aucune donnée silver trouvée (pas de sources.json). Rien à archiver.", file=sys.stderr)
        return 1
        
    try:
        sources = json.loads(sources_path.read_text())
        patch = sources.get("patch")
    except json.JSONDecodeError:
        patch = None
        
    if not patch:
        print("✗ Impossible de déterminer le patch à partir des données actuelles.", file=sys.stderr)
        return 1
        
    archive_dir = rl.DATA / "archive" / patch
    zip_path = rl.DATA / "archive" / f"{patch}.zip"
    
    if zip_path.exists() or archive_dir.exists():
        print(f"✗ L'archive pour le patch {patch} existe déjà ({zip_path}).", file=sys.stderr)
        print("Pour forcer une nouvelle archive, supprimez manuellement ce fichier.", file=sys.stderr)
        return 1
        
    archive_dir.mkdir(parents=True)
    
    # Déplacer les dossiers temporairement
    folders_to_archive = [
        rl.RAW_DIR,
        rl.SILVER_DIR,
        rl.GOLD_DIR,
        rl.DATA / "04_dataset"
    ]
    
    archived_count = 0
    for folder in folders_to_archive:
        if folder.exists() and folder.is_dir():
            target = archive_dir / folder.name
            shutil.move(str(folder), str(target))
            archived_count += 1
            print(f"  → Déplacé: {folder.name} vers archive/{patch}/{folder.name}")
            
    if archived_count > 0:
        # Créer le zip
        print(f"  → Compression en cours : {patch}.zip ...")
        shutil.make_archive(str(archive_dir), 'zip', str(archive_dir))
        # Nettoyer le dossier non zippé
        shutil.rmtree(str(archive_dir))
        print(f"\n✓ Succès: {archived_count} dossiers archivés et compressés dans {patch}.zip.")
        print("Le pipeline est maintenant vide et prêt pour le nouveau patch.")
    else:
        print("⚠ Aucun dossier à archiver.")
        archive_dir.rmdir()
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
