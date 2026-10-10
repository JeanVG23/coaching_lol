"""Contrats de l'entraînement public par rôle."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import role_features as rf

SPEC = importlib.util.spec_from_file_location(
    "tre", Path("src/02_data_science/train_role_ensemble.py"))
tre = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tre)


def test_driver_stability_signe_constant_vaut_un():
    folds = [{"csm10__mean": 0.4}, {"csm10__mean": 0.3},
             {"csm10__mean": 0.5}]
    assert tre.driver_stability(folds)["csm10__mean"] == 1.0


def test_driver_stability_signe_alternant_descend():
    folds = [{"csm10__mean": 0.4}, {"csm10__mean": -0.3},
             {"csm10__mean": 0.5}, {"csm10__mean": -0.2}]
    assert tre.driver_stability(folds)["csm10__mean"] == 0.5


def test_driver_stability_ignore_les_features_absentes_d_un_fold():
    folds = [{"a": 1.0}, {"a": 1.0, "b": -1.0}]
    out = tre.driver_stability(folds)
    assert out["a"] == 1.0
    assert out["b"] == 1.0


def test_features_ebm_public_excluent_ml_only():
    for role in rf.ROLES:
        cols = tre.ebm_feature_columns(role)
        hidden_prefixes = tuple(f"{feature}__" for feature in rf.HIDDEN_ML_ONLY)
        assert not any(column.startswith(hidden_prefixes) for column in cols)


def test_features_ebm_public_sont_des_colonnes_agregees():
    """Hors `win_rate`, seuls des agrégats de fenêtre entrent dans le modèle.

    `n_games` en particulier reste dehors : c'est la profondeur d'historique
    collecté, donc un proxy direct du rang plutôt qu'une description du jeu.
    """
    cols = tre.ebm_feature_columns("JUNGLE")
    assert all("__" in column for column in cols if column != "win_rate")
    assert "n_games" not in cols


def test_features_ebm_mid_contiennent_les_diffs_de_lane_descriptifs():
    cols = set(tre.ebm_feature_columns("MIDDLE"))
    for feature in rf.MIDDLE_DESCRIPTIVE:
        assert {f"{feature}__{stat}" for stat in ("mean", "std", "p10", "p50", "p90")}.issubset(cols)


def test_split_holdout_est_disjoint_du_train():
    players = pd.DataFrame({"high_elo": [0, 1] * 50},
                           index=[f"p{i}" for i in range(100)])
    train, holdout = tre.split_holdout(players, frac=0.15)
    assert set(train.index).isdisjoint(holdout.index)
    assert len(holdout) == 15
    assert len(train) + len(holdout) == 100


def test_split_holdout_conserve_l_equilibre_des_classes():
    players = pd.DataFrame({"high_elo": [0, 1] * 50},
                           index=[f"p{i}" for i in range(100)])
    _, holdout = tre.split_holdout(players, frac=0.20)
    assert holdout["high_elo"].sum() == len(holdout) // 2


def test_purge_train_reconstruit_une_fenetre_fixe_sans_futur():
    players = pd.DataFrame(
        {"as_of": [100], "n_window": [2]}, index=["train"])
    games = pd.DataFrame([
        {"puuid": "train", "role": "JUNGLE", "match_id": "shared",
         "game_ts": 90, "win": 1, "csm10": 100.0},
        {"puuid": "train", "role": "JUNGLE", "match_id": "recent",
         "game_ts": 80, "win": 1, "csm10": 8.0},
        {"puuid": "train", "role": "JUNGLE", "match_id": "backfill",
         "game_ts": 70, "win": 0, "csm10": 6.0},
        {"puuid": "train", "role": "JUNGLE", "match_id": "future",
         "game_ts": 110, "win": 1, "csm10": 1000.0},
        {"puuid": "validation", "role": "JUNGLE", "match_id": "shared",
         "game_ts": 90, "win": 0, "csm10": 1.0},
    ])
    out, dropped = tre.purged_fixed_window_features(
        games, players, ["validation"], "JUNGLE")
    assert dropped == []
    assert out.iloc[0]["csm10__mean"] == 7.0
    assert "n_games" not in out.columns


def test_purge_droppe_un_joueur_sans_n_parties_eligibles():
    players = pd.DataFrame(
        {"as_of": [100], "n_window": [2]}, index=["train"])
    games = pd.DataFrame([
        {"puuid": "train", "role": "JUNGLE", "match_id": "only",
         "game_ts": 80, "win": 1, "csm10": 8.0},
    ])
    out, dropped = tre.purged_fixed_window_features(
        games, players, [], "JUNGLE")
    assert out.empty
    assert dropped == ["train"]


def test_purge_support_lit_les_anciens_rows_utility():
    players = pd.DataFrame(
        {"as_of": [100], "n_window": [2]}, index=["train"])
    games = pd.DataFrame([
        {"puuid": "train", "role": "UTILITY", "match_id": "one",
         "game_ts": 80, "win": 1, "csm10": 8.0},
        {"puuid": "train", "role": "UTILITY", "match_id": "two",
         "game_ts": 70, "win": 0, "csm10": 6.0},
    ])
    out, dropped = tre.purged_fixed_window_features(
        games, players, [], "SUPPORT")
    assert dropped == []
    assert out.iloc[0]["csm10__mean"] == 7.0


class FakeEbm:
    """Reproduit le contrat d'eval_terms : des contributions CENTRÉES par terme."""

    def __init__(self, contributions: dict[str, list[float]]):
        self.term_names_ = list(contributions)
        self._matrix = np.array(
            [contributions[name] for name in self.term_names_], dtype=float).T

    def eval_terms(self, X):
        return self._matrix


def test_direction_positive_quand_la_contribution_monte_avec_la_feature():
    values = list(range(10))
    centered = [float(v) - 4.5 for v in values]
    X = pd.DataFrame({"csm10__mean": values})
    assert tre._ebm_term_directions(FakeEbm({"csm10__mean": centered}), X) \
        ["csm10__mean"] > 0


def test_direction_negative_quand_la_contribution_descend():
    values = list(range(10))
    centered = [4.5 - float(v) for v in values]
    X = pd.DataFrame({"csm10__mean": values})
    assert tre._ebm_term_directions(FakeEbm({"csm10__mean": centered}), X) \
        ["csm10__mean"] < 0


def test_direction_nulle_pour_un_terme_plat():
    X = pd.DataFrame({"csm10__mean": [3.0] * 10})
    assert tre._ebm_term_directions(FakeEbm({"csm10__mean": [0.0] * 10}), X) \
        ["csm10__mean"] == 0.0


def test_direction_ebm_reelle_suit_le_sens_de_la_feature():
    """Le sens vient de la shape function, pas de la moyenne des contributions.

    eval_terms rend des contributions centrées : leur moyenne vaut zéro par
    construction, donc son signe est du bruit numérique.
    """
    rng = np.random.default_rng(0)
    signal = rng.normal(size=200)
    X = pd.DataFrame({"csm10__mean": signal, "bruit__mean": rng.normal(size=200)})
    model = tre.make_models()["ebm"]
    model.fit(X, (signal > 0).astype(int))

    terms = np.asarray(model.eval_terms(X), dtype=float)
    column = list(model.term_names_).index("csm10__mean")
    assert abs(terms[:, column].mean()) < 1e-6
    assert terms[:, column].std() > 0.1

    assert tre._ebm_term_directions(model, X)["csm10__mean"] > 0


def test_driver_stability_ne_recompense_pas_un_terme_toujours_plat():
    assert tre.driver_stability([{"a": 0.0}, {"a": 0.0}])["a"] == 0.0


def test_driver_stability_penalise_un_sens_qui_disparait():
    """Un terme plat ne vote pas : il ne peut pas devenir le sens majoritaire."""
    folds = [{"a": 0.5}, {"a": 0.0}, {"a": 0.0}]
    assert tre.driver_stability(folds)["a"] == pytest.approx(1 / 3)


def test_une_feature_absente_du_dataset_est_refusee():
    """Une feature manquante doit faire échouer, jamais devenir une colonne plate.

    `reindex` la fabriquerait en NaN et `fillna(0)` en zéros parfaitement
    plausibles : le modèle apprendrait alors sur une feature qui n'existe pas,
    et l'EBM publierait sa shape function comme s'il l'avait observée.
    """
    frame = pd.DataFrame({"csm10__mean": [1.0, 2.0]}, index=["a", "b"])
    with pytest.raises(KeyError, match="csm14__mean"):
        tre._feature_matrix(frame, ["csm10__mean", "csm14__mean"])


def test_la_valeur_manquante_reste_manquante():
    """`ml_features` pose l'invariant : NaN propagée, pas d'imputation.

    Imputer zéro sur un diff de plates ne dit pas « inconnu », ça dit « autant
    de plates que l'adversaire ». Les trois modèles gèrent le NaN nativement.
    """
    frame = pd.DataFrame({"csm10__mean": [1.0, np.nan]}, index=["a", "b"])
    out = tre._feature_matrix(frame, ["csm10__mean"])
    assert out["csm10__mean"].isna().tolist() == [False, True]


def test_un_joueur_sans_features_agregees_est_refuse():
    """Purger un joueur le retire du train ; il ne devient pas un joueur moyen."""
    frame = pd.DataFrame({"csm10__mean": [1.0]}, index=["a"])
    with pytest.raises(KeyError):
        tre._feature_matrix(frame, ["csm10__mean"], index=["a", "b"])


def test_driver_stability_rend_les_features_dans_un_ordre_stable():
    """Deux runs sur les mêmes données doivent écrire le même fichier.

    Les noms sortaient d'un `set` : leur ordre changeait d'un processus à
    l'autre, si bien qu'un diff entre deux métriques ne distinguait plus un
    vrai changement du remue-ménage de la table de hachage.
    """
    folds = [{f"f{i}": 1.0 for i in range(10)}]
    assert list(tre.driver_stability(folds)) == sorted(f"f{i}" for i in range(10))


def test_les_metriques_reprennent_la_provenance_du_dataset(tmp_path):
    """Les métriques sont lues seules, loin du parquet qui les a produites.

    Sans ce report, l'âge du label reste dans un sidecar que personne n'ouvre au
    moment de décider l'ouverture publique.
    """
    (tmp_path / "jungle_player_dataset.meta.json").write_text(
        '{"label_age_days": {"min": 73, "p50": 79, "max": 85}}')
    out = tre.dataset_provenance(tmp_path / "jungle_player_dataset.parquet")
    assert out["label_age_days"]["max"] == 85


def test_un_dataset_sans_sidecar_ne_fabrique_pas_de_provenance(tmp_path):
    """Mieux vaut un bloc vide qu'une provenance inventée par défaut."""
    assert tre.dataset_provenance(tmp_path / "absent_player_dataset.parquet") == {}


def test_le_taux_de_victoire_entre_dans_les_features_publiques():
    """Le win_rate est calculé et stocké dans chaque dataset par rôle, mais
    aucun modèle par rôle ne le voyait, alors que le modèle ADC servi l'utilise.

    Sur une fenêtre à N fixe il est strictement comparable d'un joueur à l'autre,
    ce qu'il n'est pas dans le dataset ADC où `n_games` varie. Sans lui, une série
    de défaites tire tous les agrégats de perf vers le profil low elo sans que le
    modèle puisse l'attribuer à la variance de résultat.
    """
    for role in rf.ROLES:
        columns = tre.ebm_feature_columns(role)
        assert "win_rate" in columns
        assert "n_games" not in columns


# --- frontière servie et held-out répété ------------------------------------

def test_la_frontiere_par_defaut_est_diamond():
    """MASTER est inséparable de DIAMOND dans cet espace de features.

    Sonde du 2026-09-18, cinq rôles, mêmes datasets : DIAMOND vs GM+CHALL rend
    0.76 à 0.92 d'AUC held-out, DIAMOND vs MASTER seul rend 0.53 à 0.61, soit le
    hasard. Servir la frontière MASTER vs GM+CHALL revenait à demander au modèle
    de trancher entre deux populations que la donnée ne distingue pas.
    """
    import inspect

    assert tre.DEFAULT_BOUNDARY == "diamond"
    assert inspect.signature(tre.train_role).parameters["boundary"].default \
        == tre.DEFAULT_BOUNDARY


def _players(n_high=6, n_low=30):
    tiers = ["CHALLENGER"] * n_high + ["DIAMOND"] * n_low
    return pd.DataFrame(
        {"tier": tiers,
         "high_elo": [1] * n_high + [0] * n_low},
        index=[f"p{i}" for i in range(n_high + n_low)])


def test_le_tirage_de_balance_suit_la_graine():
    """Quels Diamants servent de classe basse fait partie du tirage.

    La classe haute est limitante (93 à 198 joueurs par rôle contre 271 à 389
    Diamants) : à graine fixe, les mêmes Diamants étaient repris à chaque run et
    la variance de ce choix restait invisible.
    """
    une = tre._balance(_players(), {"DIAMOND"}, seed=1)
    autre = tre._balance(_players(), {"DIAMOND"}, seed=2)
    assert set(une.index) != set(autre.index)


def test_la_decoupe_du_heldout_suit_la_graine():
    une, _ = tre.split_holdout(_players(20, 20), frac=0.25, seed=1)
    autre, _ = tre.split_holdout(_players(20, 20), frac=0.25, seed=2)
    assert set(une.index) != set(autre.index)


def test_la_mediane_par_modele_ignore_la_colonne_de_graine():
    rows = [{"seed": 1, "ebm": 0.70}, {"seed": 2, "ebm": 0.90},
            {"seed": 3, "ebm": 0.80}]
    assert tre._median_by_model(rows) == {"ebm": 0.80}


class _RankingModel:
    """Modèle jouet : classe sur la première colonne, sans rien apprendre.

    L'intégration testée ici est le PROTOCOLE (un held-out par graine), pas la
    qualité de l'ajustement : entraîner trois vrais ensembles sur 24 joueurs
    rendrait le test lent sans rien vérifier de plus.
    """

    term_names_: list[str] = []

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        values = np.nan_to_num(X.iloc[:, 0].to_numpy(dtype=float))
        return np.column_stack([-values, values])

    def eval_terms(self, X):
        return np.zeros((len(X), 0))


def _synthetic_role_data(role="JUNGLE", n_high=12, n_low=12, n_window=4):
    """Fenêtres à profondeur fixe, deux classes séparables, zéro lecture disque."""
    import ml_features as mf

    rng = np.random.default_rng(0)
    features = rf.public_features(role)
    games = []
    for index in range(n_high + n_low):
        high = index < n_high
        for game in range(n_window):
            row = {"puuid": f"p{index}", "role": role,
                   "match_id": f"m{index}-{game}", "game_ts": 100 - game,
                   "win": int(rng.random() < (0.6 if high else 0.4))}
            row.update({name: float(rng.normal(1.0 if high else 0.0))
                        for name in features})
            games.append(row)
    games_df = pd.DataFrame(games)

    rows = []
    for index in range(n_high + n_low):
        high = index < n_high
        aggregates = mf.aggregate_player_features(
            games_df[games_df["puuid"] == f"p{index}"], features)
        aggregates.pop("n_games", None)
        rows.append({"puuid": f"p{index}",
                     "tier": "CHALLENGER" if high else "DIAMOND",
                     "high_elo": int(high), "n_window": n_window,
                     "as_of": 100, **aggregates})
    return pd.DataFrame(rows), games_df


def test_le_heldout_est_repete_sur_chaque_graine(monkeypatch):
    """Un held-out de 28 à 60 joueurs ne se lit pas sur un seul tirage.

    Mesuré le 2026-09-18 sur dix graines : le même modèle, sur les mêmes
    données, vaut 0.663 ou 0.918 en TOP selon les 28 personnes tirées. Publier
    un tirage revenait à publier ce tirage-là, pas le modèle.
    """
    monkeypatch.setattr(tre, "make_models",
                        lambda: {name: _RankingModel()
                                 for name in ("xgb", "rf", "ebm")})
    player_df, games_df = _synthetic_role_data()

    metrics = tre.train_role(player_df, games_df, "JUNGLE", n_splits=2,
                             holdout_seeds=(1, 2, 3))

    assert [row["seed"] for row in metrics["auc_heldout_seeds"]] == [1, 2, 3]
    assert metrics["auc_heldout_median"]["ebm"] == pytest.approx(
        float(np.median([row["ebm"] for row in metrics["auc_heldout_seeds"]])))
    assert metrics["boundary"] == "diamond"


def test_les_metriques_ne_publient_plus_un_tirage_unique(monkeypatch):
    """`auc_heldout` disparaît au lieu de changer de sens.

    Garder la clé en y écrivant la médiane servirait un nombre sous le nom d'un
    autre : les anciennes métriques doivent échouer bruyamment, pas se lire de
    travers.
    """
    monkeypatch.setattr(tre, "make_models",
                        lambda: {name: _RankingModel()
                                 for name in ("xgb", "rf", "ebm")})
    player_df, games_df = _synthetic_role_data()

    metrics = tre.train_role(player_df, games_df, "JUNGLE", n_splits=2,
                             holdout_seeds=(1, 2))

    assert "auc_heldout" not in metrics


def test_l_export_accompagne_le_modele_et_partage_son_identite(
        tmp_path, monkeypatch):
    """L'export et les métriques issus d'un run partagent leur identité."""
    import json as json_module

    monkeypatch.setattr(tre, "MODEL_DIR", tmp_path)
    player_df, games_df = _synthetic_role_data(role="TOP")
    metrics = tre.train_role(
        player_df, games_df, "TOP", n_splits=2, holdout_seeds=(42,))
    tre.save_role_artifacts(
        player_df, "TOP", metrics, boundary=tre.DEFAULT_BOUNDARY)

    export = json_module.loads((tmp_path / "top_ebm_export.json").read_text())
    saved = json_module.loads(
        (tmp_path / "top_player_metrics.json").read_text())
    assert export["schema_version"] == 1
    assert export["role"] == "TOP"
    assert export["boundary"] == "diamond"
    assert export["features"] == tre.ebm_feature_columns("TOP")
    assert export["model_id"] == saved["model_id"]
    assert set(export["shapes"]) == set(export["features"])


def test_l_export_est_ecrit_apres_les_metriques(tmp_path, monkeypatch):
    """La date de l'export empêche Make de relancer sans fin le modèle."""
    monkeypatch.setattr(tre, "MODEL_DIR", tmp_path)
    player_df, games_df = _synthetic_role_data(role="TOP")
    metrics = tre.train_role(
        player_df, games_df, "TOP", n_splits=2, holdout_seeds=(42,))
    tre.save_role_artifacts(
        player_df, "TOP", metrics, boundary=tre.DEFAULT_BOUNDARY)
    assert ((tmp_path / "top_ebm_export.json").stat().st_mtime_ns
            >= (tmp_path / "top_player_metrics.json").stat().st_mtime_ns)


def test_save_role_artifacts_records_dataset_identity(tmp_path, monkeypatch):
    import json as json_module

    monkeypatch.setattr(tre, "MODEL_DIR", tmp_path)
    player_df, games_df = _synthetic_role_data(role="TOP")
    metrics = tre.train_role(
        player_df, games_df, "TOP", n_splits=2, holdout_seeds=(42,))
    tre.save_role_artifacts(
        player_df, "TOP", metrics, boundary=tre.DEFAULT_BOUNDARY)
    saved = json_module.loads(
        (tmp_path / "top_player_metrics.json").read_text())
    assert "dataset_identity" in saved
    assert "fingerprint" in saved["dataset_identity"]
    assert saved["dataset_identity"]["role"] == "TOP"
    assert saved["dataset_identity"]["boundary"] == "diamond"
