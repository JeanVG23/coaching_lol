"""train_player_ensemble vit dans src/02_data_science/ (dossier non importable tel
quel, même pattern de chargement que tests/test_build_dataset_flatten.py). Seule la
fonction pure dispersion_share_analysis est testée ici — le reste (CV, fit des
modèles) est vérifié par exécution réelle (cf. plan, Task 4 Step 6)."""
import importlib.util
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC / "core"))
_spec = importlib.util.spec_from_file_location(
    "train_player_ensemble", _SRC / "02_data_science" / "train_player_ensemble.py")
train_player_ensemble = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_player_ensemble)


def test_dispersion_share_analysis_groups_by_stat_suffix():
    per_feature = {
        "csm10__mean": 1.0, "csm10__p50": 1.0,                        # central = 2.0
        "csm10__std": 3.0, "csm10__p10": 1.0, "csm10__p90": 0.0,      # dispersion = 4.0
        "n_games": 0.5,
    }
    result = train_player_ensemble.dispersion_share_analysis(per_feature)
    assert result["dispersion_share_of_signal"] == pytest.approx(4.0 / 6.0, abs=1e-4)
    assert result["share_by_stat"]["std"] == pytest.approx(3.0 / 6.5, abs=1e-4)
    assert result["share_by_stat"]["n_games"] == pytest.approx(0.5 / 6.5, abs=1e-4)


def test_dispersion_share_analysis_ignores_unknown_suffix():
    per_feature = {"weird__unknownstat": 5.0, "csm10__mean": 1.0}
    result = train_player_ensemble.dispersion_share_analysis(per_feature)
    # "unknownstat" n'est dans aucun bucket -> exclu du total (mean=1.0 seul compte)
    assert result["share_by_stat"]["mean"] == pytest.approx(1.0)
