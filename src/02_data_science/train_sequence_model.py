#!/usr/bin/env python3
"""
02_data_science — étape 1 : transformer séquentiel supervisé (high_elo + dia_chall),
CV purgé (joueurs + miroir), comparé aux baselines tabulaire (RF+EBM) et MLP sur les mêmes
folds. Écrit sequence_metrics.json + sequence_supervised.pt.

Verdict « séquence > agrégat ? » se lit sur dia_chall (master/GM null = bruit de label,
non interprétable — cf. spec 2026-07-18 §Pièges).

Usage : poetry run python3 src/02_data_science/train_sequence_model.py [--epochs N]
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

# MPS (Apple Silicon) ne supporte pas aten::_nested_tensor_from_mask_left_aligned,
# utilisé par nn.TransformerEncoder avec src_key_padding_mask. Fallback CPU officiel
# PyTorch pour les ops MPS manquantes (cf. github.com/pytorch/pytorch/issues/141287).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_CORE = Path(__file__).resolve().parent.parent / "core"
sys.path.insert(0, str(_CORE))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # sequence_model, sequence_data
import numpy as np
import pandas as pd
import torch
import riotlib as rl
import sequence_model as sm
import sequence_data as sd

MODEL_DIR = rl.DATA / "05_model"
TABULAR = rl.DATA / "04_dataset" / "adc_dataset.parquet"
SEED = 42


def _baseline_tabular(task, folds, idx, y, data, seed=SEED):
    """Ensemble RF+EBM sur adc_dataset.parquet (features agrégées), MÊME fold partition
    joueur + purge miroir que le transformer. Donne à l'agrégat son meilleur coup (pas un
    modèle sous-tuné qui offrirait une victoire facile au transformer). NB : les
    0.724/0.589 de CLAUDE.md viennent d'un autre protocole (ensemble 3-modèles, CV per-game
    non purgée) -> notre baseline est le comparatif propre, PAS une reproduction de ces
    chiffres.

    Note : EBM (interpret.glassbox) remplace xgboost ici — contrainte macOS libomp :
    torch + xgboost cohabitent mal dans le même process (double-load libomp.dylib ->
    SIGSEGV). EBM est un baseline glass-box GA²M légitime et coexiste proprement avec torch.
    """
    # EBM au lieu de xgboost (contrainte macOS libomp — cf. docstring).
    from interpret.glassbox import ExplainableBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    df = pd.read_parquet(TABULAR)
    feat_cols = [c for c in df.columns if c not in
                 ("match_id", "puuid", "source", "rank", "champion", "win",
                  "patch", "game_ts", "rank_ord", "high_elo")]
    oof_ebm = np.full(len(idx), np.nan)
    oof_rf = np.full(len(idx), np.nan)
    for tr_i, va_i in folds:
        tr_puuids = set(data["puuid"][idx][tr_i]); va_puuids = set(data["puuid"][idx][va_i])
        tr_df = df[df["puuid"].isin(tr_puuids)
                   & ~df["match_id"].isin(set(data["match_id"][idx][va_i]))]
        va_df = df[df["puuid"].isin(va_puuids)]
        if task == "dia_chall":
            tr_df = tr_df[tr_df["rank"].isin(sd.DIA_CHALL)]
            va_df = va_df[va_df["rank"].isin(sd.DIA_CHALL)]
        yt = tr_df["rank"].isin(sd.HIGH_ELO if task == "high_elo" else {"challenger"}).astype(int)
        yv = va_df["rank"].isin(sd.HIGH_ELO if task == "high_elo" else {"challenger"}).astype(int)
        if len(set(yv)) < 2:
            continue
        me = ExplainableBoostingClassifier(random_state=seed)
        mr = RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=4,
                                    max_features="sqrt", n_jobs=-1, random_state=seed)
        Xt = tr_df[feat_cols]; Xv = va_df[feat_cols]
        me.fit(Xt.fillna(0), yt)
        mr.fit(Xt.fillna(Xt.median(numeric_only=True)), yt)
        p_e = me.predict_proba(Xv.fillna(0))[:, 1]
        p_r = mr.predict_proba(Xv.fillna(Xv.median(numeric_only=True)))[:, 1]
        va_map = {(r["match_id"], r["puuid"]): (pe, pr)
                  for (_, r), pe, pr in zip(va_df.iterrows(), p_e, p_r)}
        for j in va_i:
            v = va_map.get((data["match_id"][idx[j]], data["puuid"][idx[j]]))
            if v is not None:
                oof_ebm[j], oof_rf[j] = v
    ens = np.where(np.isnan(oof_ebm) | np.isnan(oof_rf), np.nan, (oof_ebm + oof_rf) / 2.0)
    mask = ~np.isnan(ens)
    return float(roc_auc_score(y[mask], ens[mask])) if mask.sum() and len(set(y[mask])) > 1 else None


def _baseline_mlp(task, folds, idx, y, data, seed=SEED):
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    df = pd.read_parquet(TABULAR)
    feat_cols = [c for c in df.columns if c not in
                 ("match_id", "puuid", "source", "rank", "champion", "win",
                  "patch", "game_ts", "rank_ord", "high_elo")]
    oof = np.full(len(idx), np.nan)
    for tr_i, va_i in folds:
        tr_puuids = set(data["puuid"][idx][tr_i]); va_puuids = set(data["puuid"][idx][va_i])
        tr_df = df[df["puuid"].isin(tr_puuids)
                   & ~df["match_id"].isin(set(data["match_id"][idx][va_i]))]
        va_df = df[df["puuid"].isin(va_puuids)]
        if task == "dia_chall":
            tr_df = tr_df[tr_df["rank"].isin(sd.DIA_CHALL)]
            va_df = va_df[va_df["rank"].isin(sd.DIA_CHALL)]
        yt = tr_df["rank"].isin(sd.HIGH_ELO if task == "high_elo" else {"challenger"}).astype(int)
        yv = va_df["rank"].isin(sd.HIGH_ELO if task == "high_elo" else {"challenger"}).astype(int)
        if len(set(yv)) < 2:
            continue
        m = MLPClassifier(hidden_layer_sizes=(64,), max_iter=80, random_state=seed)
        Xt = tr_df[feat_cols].fillna(0).values; Xv = va_df[feat_cols].fillna(0).values
        m.fit(Xt, yt)
        proba = m.predict_proba(Xv)[:, 1]
        key = lambda i: (data["match_id"][idx[i]], data["puuid"][idx[i]])
        va_map = {(r["match_id"], r["puuid"]): p for (_, r), p in zip(va_df.iterrows(), proba)}
        for j in va_i:
            oof[j] = va_map.get(key(j), np.nan)
    mask = ~np.isnan(oof)
    return float(roc_auc_score(y[mask], oof[mask])) if mask.sum() and len(set(y[mask])) > 1 else None


def _train_one_task(task, data, device, epochs, batch, seed, d_in):
    from sklearn.metrics import roc_auc_score
    import copy
    idx, y = sd.task_subset(data, task)
    if len(set(y)) < 2 or len(idx) < 30:
        return ({"auc_mean": None, "auc_std": None, "n_rows": int(len(idx)),
                 "reason": "trop peu de rows ou 1 classe"}, None)
    bin_cols = range(20, d_in) if d_in > 20 else None
    folds = sd.player_folds(data["puuid"][idx], y, n_splits=5, seed=seed)
    oof = np.full(len(idx), np.nan)
    best_state_global, best_auc_global = None, -1.0
    for fi, (tr_i, va_i) in enumerate(folds):
        print(f"    fold {fi+1}/5  train={len(tr_i)} val={len(va_i)}", flush=True)
        val_puuids = set(data["puuid"][idx][va_i])
        tr_purged = sd.mirror_purge(tr_i, val_puuids, data["match_id"][idx], data["puuid"][idx])
        if len(tr_purged) == 0:
            continue
        mean, std = sd.standardize_fit(data["sequences"], data["mask"], idx[tr_purged], bin_cols)
        Xs = sd.standardize_apply(data["sequences"], mean, std)
        Xtr = torch.from_numpy(Xs[idx[tr_purged]]).to(device)
        Mtr = torch.from_numpy(data["mask"][idx[tr_purged]]).to(device)
        ytr = torch.from_numpy(y[tr_purged].astype(np.float32)).to(device)
        Xva = torch.from_numpy(Xs[idx[va_i]]).to(device)
        Mva = torch.from_numpy(data["mask"][idx[va_i]]).to(device)
        model = sm.SequenceClassifier(d_in=d_in).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_fold_auc, best_state, patience, bad = 0.0, None, 10, 0
        for ep in range(epochs):
            if ep % 10 == 0:
                print(f"      ep {ep}/{epochs}", flush=True)
            model.train()
            perm = torch.randperm(len(Xtr))
            for b in range(0, len(perm), batch):
                bi = perm[b:b + batch]
                logits = model(Xtr[bi], Mtr[bi])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ytr[bi])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(Xva, Mva)).cpu().numpy()
            if len(set(y[va_i])) < 2:
                continue
            auc = roc_auc_score(y[va_i], pv)
            if auc > best_fold_auc:
                best_fold_auc = auc
                bad = 0                                  # patience reset sur amélioration
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        # OOF au MEILLEUR état (restaure best_state, pas le dernier)
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            oof[va_i] = torch.sigmoid(model(Xva, Mva)).cpu().numpy()
        if best_fold_auc > best_auc_global:
            best_auc_global = best_fold_auc
            best_state_global = best_state
        print(f"    fold {fi+1} done  best_fold_auc={best_fold_auc:.4f}  (patience@{bad})", flush=True)
    mask = ~np.isnan(oof)
    auc = float(roc_auc_score(y[mask], oof[mask])) if mask.sum() and len(set(y[mask])) > 1 else None
    per_fold = [float(roc_auc_score(y[v], oof[v])) for (_, v) in folds
                if not np.isnan(oof[v]).any() and len(set(y[v])) > 1]
    return ({
        "auc_mean": auc,
        "auc_std": float(np.std(per_fold)) if per_fold else None,
        "n_rows": int(len(idx)), "n_val_folds": len(per_fold),
        "baseline_tabular_auc": _baseline_tabular(task, folds, idx, y, data, seed),
        "baseline_mlp_auc": _baseline_mlp(task, folds, idx, y, data, seed),
        "best_fold_auc": float(best_auc_global),
    }, best_state_global)


def run(epochs=60, batch=64, seed=SEED, device_force=None,
        metrics_name="sequence_metrics.json") -> dict:
    # sd.DATASET lu au call-time (pas via default arg de load_dataset, qui est bound au def-time)
    # -> permet au test de monkeypatcher sd.DATASET pour brancher un mini-dataset synthétique.
    data = sd.load_dataset(sd.DATASET)
    device = torch.device(device_force) if device_force else sm.get_device()
    d_in = int(data["sequences"].shape[-1])          # auto : 20 (v1) ou 27 (v2)
    print(f"  device={device} | {len(data['sequences'])} séquences | d_in={d_in}")
    metrics = {"tasks": {}, "params": {"epochs": epochs, "batch": batch, "seed": seed,
                "d_model": 64, "n_layers": 4, "nhead": 4, "d_in": d_in, "device": str(device)}}
    saved_state = None
    for task in sd.TASKS:
        print(f"\n=== tâche {task} ===")
        m, state = _train_one_task(task, data, device, epochs, batch, seed, d_in)
        metrics["tasks"][task] = m
        print(f"  séquence AUC={m.get('auc_mean')} (±{m.get('auc_std')})  "
              f"tabulaire={m.get('baseline_tabular_auc')}  mlp={m.get('baseline_mlp_auc')}")
        if task == "high_elo" and state is not None:
            saved_state = state                              # meilleur modèle high_elo (best fold)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if saved_state is not None:
        torch.save(saved_state, MODEL_DIR / "sequence_supervised.pt")
    (MODEL_DIR / metrics_name).write_text(json.dumps(metrics, indent=2))
    print(f"\n✓ {MODEL_DIR}/{metrics_name}")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", type=str, default=None,
                    help="force device (ex: 'cpu') ; défaut = auto (MPS/CUDA/CPU)")
    ap.add_argument("--metrics-name", type=str, default="sequence_metrics.json",
                    help="nom du fichier métriques (v2: sequence_metrics_v2.json)")
    args = ap.parse_args()
    run(epochs=args.epochs, batch=args.batch, device_force=args.device,
        metrics_name=args.metrics_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())