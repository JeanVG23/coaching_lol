#!/usr/bin/env python3
"""
02_data_science — étape 2 : pretrain self-supervised (mask-and-reconstruct) PURISTE
par-fold : pour chaque fold, on standardise train-only, on pretrain l'encodeur sur le
TRAIN du fold uniquement (joueurs de val jamais vus -> pas de fuite transductive), puis
on finetune le classifieur high_elo. Mêmes stats de standardisation au pretrain et au
finetune -> transfert non saboté. delta_ssl = AUC_ssl - AUC_supervisé (étape 1, même CV),
donc un delta propre (pas d'avantage transductif). NB : moins de données de pretrain par
fold (~train rows seulement) que si on préentraînait sur tout — choix délibéré pour la
propreté du comparatif.

⚠ Le prétexte MSE-reconstruct est FAIBLE sur signaux lisses (gold monotone, position
continue -> quasi-interpolation) : le modèle peut cartonner la reconstruction sans rien
apprendre de pertinent au rang. Un ≈0 delta_ssl n'est PAS un verdict sur le SSL en général,
juste sur ce prétexte (un prétexte prédictif = étape 3, cf. spec 2026-07-18 §Pièges).

Usage : poetry run python3 src/02_data_science/pretrain_sequence_model.py
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

# MPS (Apple Silicon) ne supporte pas aten::_nested_tensor_from_mask_left_aligned,
# utilisé par nn.TransformerEncoder avec src_key_padding_mask. Fallback CPU officiel
# PyTorch pour les ops MPS manquantes (cf. github.com/pytorch/pytorch/issues/141287).
# setdefault pour ne pas écraser un env utilisateur explicite.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_CORE = Path(__file__).resolve().parent.parent / "core"
sys.path.insert(0, str(_CORE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import riotlib as rl
import sequence_model as sm
import sequence_data as sd

MODEL_DIR = rl.DATA / "05_model"
SEED = 42
MASK_FRAC = 0.15


def _ssl_cv(data, device, pretrain_epochs, finetune_epochs, batch, seed):
    """SSL puriste par-fold : pour chaque fold, (1) standardise train-only, (2) pretrain
    mask-and-reconstruct sur le TRAIN du fold uniquement, (3) finetune classifieur avec
    l'encodeur pré-entraîné. Même échelle d'entrée au pretrain et au finetune (mêmes stats
    train-only) -> transfert non saboté. Aucune fuite : joueurs de val jamais vus au
    pretrain. delta_ssl est donc un signal propre (pas d'avantage transductif)."""
    from sklearn.metrics import roc_auc_score
    idx, y = sd.task_subset(data, "high_elo")
    folds = sd.player_folds(data["puuid"][idx], y, n_splits=5, seed=seed)
    oof = np.full(len(idx), np.nan)
    best_enc, best_auc = None, -1.0
    for tr_i, va_i in folds:
        val_puuids = set(data["puuid"][idx][va_i])
        tr_purged = sd.mirror_purge(tr_i, val_puuids, data["match_id"][idx], data["puuid"][idx])
        if len(tr_purged) == 0:
            continue
        mean, std = sd.standardize_fit(data["sequences"], data["mask"], idx[tr_purged])
        Xs = sd.standardize_apply(data["sequences"], mean, std)
        Xtr = torch.from_numpy(Xs[idx[tr_purged]]).to(device)
        Mtr = torch.from_numpy(data["mask"][idx[tr_purged]]).to(device)
        ytr = torch.from_numpy(y[tr_purged].astype(np.float32)).to(device)
        Xva = torch.from_numpy(Xs[idx[va_i]]).to(device)
        Mva = torch.from_numpy(data["mask"][idx[va_i]]).to(device)
        n = len(Xtr)
        # (1)+(2) pretrain SSL sur le TRAIN du fold
        enc = sm.SequenceEncoder().to(device)
        head = sm.ReconstructHead().to(device)
        opt_p = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                                  lr=3e-4, weight_decay=1e-2)
        for ep in range(pretrain_epochs):
            enc.train(); head.train()
            perm = torch.randperm(n)
            for b0 in range(0, n, batch):
                bi = perm[b0:b0 + batch]
                xb, mb = Xtr[bi], Mtr[bi]
                ssl_mask = (torch.rand_like(mb.float()) < MASK_FRAC) & mb   # 15% des valides masquées
                h = enc(xb, mb & ~ssl_mask)                                 # encode sans les masquées
                pred = head(h)
                loss = ((pred - xb) ** 2 * ssl_mask.unsqueeze(-1)).sum() / \
                       ssl_mask.float().sum().clamp(min=1.0) / 20.0
                opt_p.zero_grad(); loss.backward(); opt_p.step()
        # (3) finetune : encodeur pré-entraîné + tête fraîche
        clf = sm.SequenceClassifier().to(device)
        clf.encoder.load_state_dict(enc.state_dict())
        opt_f = torch.optim.AdamW(clf.parameters(), lr=3e-4, weight_decay=1e-2)
        for ep in range(finetune_epochs):
            clf.train()
            perm = torch.randperm(n)
            for b0 in range(0, n, batch):
                bi = perm[b0:b0 + batch]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    clf(Xtr[bi], Mtr[bi]), ytr[bi])
                opt_f.zero_grad(); loss.backward(); opt_f.step()
        clf.eval()
        with torch.no_grad():
            oof[va_i] = torch.sigmoid(clf(Xva, Mva)).cpu().numpy()
        if len(set(y[va_i])) > 1:
            a = roc_auc_score(y[va_i], oof[va_i])
            if a > best_auc:
                best_auc = a
                best_enc = {k: v.detach().cpu().clone()
                            for k, v in clf.encoder.state_dict().items()}
    mask = ~np.isnan(oof)
    auc = float(roc_auc_score(y[mask], oof[mask])) if mask.sum() and len(set(y[mask])) > 1 else None
    return auc, best_enc


def run(pretrain_epochs=30, finetune_epochs=40, batch=64, seed=SEED, device_force=None) -> dict:
    # sd.DATASET lu au call-time (pas via default arg de load_dataset, qui est bound au def-time)
    # -> permet au test de monkeypatcher sd.DATASET pour brancher un mini-dataset synthétique.
    data = sd.load_dataset(sd.DATASET)
    device = torch.device(device_force) if device_force else sm.get_device()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    auc_ssl, best_enc = _ssl_cv(data, device, pretrain_epochs, finetune_epochs, batch, seed)
    # lit l'AUC supervisé étape 1 (même protocole CV purgé, sans pretrain)
    metrics_path = MODEL_DIR / "sequence_metrics.json"
    prev = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    auc_sup = prev.get("tasks", {}).get("high_elo", {}).get("auc_mean")
    out = {"auc_supervised": auc_sup, "auc_ssl": auc_ssl,
           "delta_ssl": (auc_ssl - auc_sup) if (auc_ssl is not None and auc_sup is not None) else None,
           "pretext": "mask-and-reconstruct (MSE, 15% mask) — prétexte faible sur signaux lisses ; "
                      "≈0 delta n'est pas un verdict sur le SSL en général (cf. spec §Pièges). "
                      "Pretrain par-fold train-only : pas d'avantage transductif, delta propre.",
           "params": {"pretrain_epochs": pretrain_epochs, "finetune_epochs": finetune_epochs,
                      "seed": seed, "device": str(device)}}
    prev["ssl"] = out
    metrics_path.write_text(json.dumps(prev, indent=2))
    if best_enc is not None:
        torch.save(best_enc, MODEL_DIR / "sequence_encoder_pretrain.pt")
    print(f"  AUC supervisé={auc_sup}  AUC ssl={auc_ssl}  delta={out['delta_ssl']}")
    print(f"✓ {MODEL_DIR}/sequence_encoder_pretrain.pt + delta dans sequence_metrics.json")
    return out


def embed_game(seq: np.ndarray, mask: np.ndarray, device_force=None) -> np.ndarray:
    """Vecteur d'embedding 64-d d'une game (pour inspection : projection 2D colorée rang)."""
    device = torch.device(device_force) if device_force else sm.get_device()
    clf = sm.SequenceClassifier().to(device)
    clf.eval()
    with torch.no_grad():
        x = torch.from_numpy(seq[None].astype(np.float32)).to(device)
        m = torch.from_numpy(mask[None].astype(bool)).to(device)
        return clf.embed(x, m).cpu().numpy()[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--device", type=str, default=None,
                    help="force device (ex: 'cpu') ; défaut = auto (MPS/CUDA/CPU)")
    args = ap.parse_args()
    run(pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs,
        device_force=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())