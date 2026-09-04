import os
import sys

# macOS (Apple Silicon) : torch et scikit-learn embarquent CHACUN leur libomp.dylib (xgboost
# en lie une 3e via Homebrew). Deux runtimes OpenMP initialisés dans le même processus →
# SIGSEGV ou deadlock dès la première opération parallélisée du second runtime. Reproduit :
# la suite plantait dans train_sequence_model._train_one_task juste après les fits
# sklearn/xgboost de test_train_player_ensemble. OMP_NUM_THREADS=1 supprime la collision,
# KMP_DUPLICATE_LIB_OK autorise la cohabitation (le flag SEUL ne suffit pas : vérifié, le
# processus se bloque avec des threads multiples). Ces lignes doivent précéder tout import
# qui charge libomp (il lit son environnement à son initialisation). setdefault : un export
# explicite de l'utilisateur gagne. Rien dans ~/.zshrc : c'est un défaut du dépôt, pas de la
# machine, et un export global plafonnerait les threads de tous les autres projets.
# Canary de non-régression : tests/test_openmp_coexistence.py.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "core"))
sys.path.insert(0, str(_SRC / "collection"))
sys.path.insert(0, str(_SRC / "pipeline_ops"))
sys.path.insert(0, str(_SRC / "reporting"))
sys.path.insert(0, str(_SRC / "04_coaching"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # fixtures partagées entre tests


import contextlib  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

import pytest  # noqa: E402

FIXTURES_DEMO = _Path(__file__).resolve().parent / "fixtures" / "demo"


@contextlib.contextmanager
def _pointing_at(root: _Path):
    """Fait pointer la pile médaillon vers `root`, puis restaure.

    On substitue les attributs de module plutôt que la variable d'environnement :
    `COACHING_DATA_DIR` est lue à l'import, or les modules sont déjà chargés.
    """
    import champion_profiles as cp
    import riotlib as rl

    saved = {name: getattr(rl, name)
             for name in ("DATA", "RAW_DIR", "SILVER_DIR", "GOLD_DIR")}
    saved_static = cp.STATIC_DIR
    rl.DATA = root
    rl.RAW_DIR = root / rl.LAYER_RAW
    rl.SILVER_DIR = root / rl.LAYER_SILVER
    rl.GOLD_DIR = root / rl.LAYER_GOLD
    cp.STATIC_DIR = root / "00_static"
    cp._invalidate_catalogs()
    cp.load_items.cache_clear()
    try:
        yield root
    finally:
        for name, value in saved.items():
            setattr(rl, name, value)
        cp.STATIC_DIR = saved_static
        cp._invalidate_catalogs()
        cp.load_items.cache_clear()


@pytest.fixture(scope="session")
def _demo_root(tmp_path_factory):
    """Construit la pile démo UNE fois : `reextract_silver` + `rebuild_gold` sur
    les 49 parties versionnées."""
    import rebuild_gold
    import reextract_silver

    root = tmp_path_factory.mktemp("demo-data")
    shutil.copytree(FIXTURES_DEMO, root, dirs_exist_ok=True)
    with _pointing_at(root):
        reextract_silver.main()
        rebuild_gold.main()
    return root


@pytest.fixture
def demo_data(_demo_root):
    """Pile médaillon complète, construite depuis les fixtures versionnées.

    Existe pour que la CI VÉRIFIE quelque chose : sans données dans le dépôt, les
    tests qui ont besoin d'agrégats se sautaient, donc passaient au vert sans rien
    contrôler.

    Portée fonction, et non session : un redirigeage qui survit à son test
    contamine ceux qui suivent (`test_positioning` lit le raw réel et échouait
    silencieusement à la place). La construction, elle, reste mutualisée.
    """
    with _pointing_at(_demo_root) as root:
        yield root
