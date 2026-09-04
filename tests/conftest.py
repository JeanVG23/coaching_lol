import sys
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
