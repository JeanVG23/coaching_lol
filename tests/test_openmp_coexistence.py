"""Non-régression du conflit libomp macOS (torch + sklearn + xgboost, même processus).

torch et scikit-learn embarquent chacun leur libomp.dylib (cf. le garde-fou en tête de
tests/conftest.py) ; deux runtimes OpenMP initialisés dans le même processus → SIGSEGV ou
deadlock. En isolation, la course est trop capricieuse pour servir de test : la même
séquence bloque ou passe selon le minutage exact (des simples prints entre les fits
basculent l'issue, vérifié). Le scénario réel, lui, est déterministe : la suite segfaultait
dans train_sequence_model._train_one_task après les fits sklearn/xgboost d'un test
antérieur. D'où les deux tests :

- `test_openmp_guard_armed_on_darwin` : le CANARY. Sans garde-fou, la suite crashe en
  silence ; ce tripwire transforme sa disparition en échec propre et nommée.
- `test_openmp_runtimes_coexist` : le SMOKE. Garde-fou armé, la cohabitation doit
  fonctionner de bout en bout (fits des 3 runtimes puis opération torch).
"""

import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="garde-fou libomp : macOS seulement")
def test_openmp_guard_armed_on_darwin():
    # Strict (== "1") et non « il suffit qu'elle soit posée » : un export shell type
    # OMP_NUM_THREADS=8 gagne sur le setdefault de conftest et la suite redevient
    # exposée à la course ; autant le dénoncer ici, nommément.
    assert os.environ.get("OMP_NUM_THREADS") == "1", (
        "OMP_NUM_THREADS != 1 : sans thread unique, les libomp de torch et scikit-learn "
        "entrent en collision dans la suite (SIGSEGV historique dans "
        "train_sequence_model._train_one_task). Le garde-fou en tête de tests/conftest.py "
        "a-t-il disparu, ou un export shell le contourne-t-il ?"
    )
    assert os.environ.get("KMP_DUPLICATE_LIB_OK", "").upper() == "TRUE", (
        "KMP_DUPLICATE_LIB_OK != TRUE : le second runtime OpenMP refusera de cohabiter "
        "avec celui déjà initialisé. Le garde-fou en tête de tests/conftest.py a-t-il "
        "disparu ?"
    )


def test_openmp_runtimes_coexist():
    pytest.importorskip("torch")  # groupe `deep` absent en CI : skip propre, cf. tests séquence
    pytest.importorskip("xgboost")

    _SCRIPT = """
import numpy as np
from sklearn.ensemble import RandomForestClassifier
X = np.random.RandomState(0).rand(200, 10)
y = (X[:, 0] > 0.5).astype(int)
RandomForestClassifier(n_estimators=10).fit(X, y)          # initialise le libomp de sklearn
import xgboost as xgb
xgb.XGBClassifier(n_estimators=5, max_depth=2).fit(X, y)     # ... puis celui d'xgboost
import torch
z = torch.randn(256, 256) @ torch.randn(256, 256)           # ... puis celui de torch
assert np.isfinite(z.numpy()).all()
print("coexistence-ok")
"""
    # Le sous-processus hérite de l'environnement posé par conftest.py : avec le garde-fou,
    # il rend la main en quelques secondes. Un blocage ici reste possible si le garde-fou
    # a disparu (la course est capricieuse, cf. docstring du module) : le timeout le
    # transforme en échec lisible, mais c'est le tripwire ci-dessus qui garantit la
    # détection ; ce test ne prouve que le chemin nominal.
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SCRIPT],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "Deadlock libomp : torch et sklearn/xgboost ne cohabitent plus dans un même "
            "processus. Le garde-fou OMP_NUM_THREADS=1 + KMP_DUPLICATE_LIB_OK=TRUE en tête "
            "de tests/conftest.py a-t-il disparu ?"
        )
    assert "coexistence-ok" in proc.stdout, proc.stderr
