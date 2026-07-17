#!/usr/bin/env python3
"""
02_data_science — modules PyTorch purs pour le transformer séquentiel (cf. spec
2026-07-18). Pas de HF Transformers : tout est écrit à la main, lisible, pédagogique.

- SequenceEncoder : projection 20->d_model + positional embedding appris + N couches
  TransformerEncoderLayer. src_key_padding_mask ignore les minutes paddées.
- ClassifierHead : masked-mean-pool sur les frames valides -> logit binaire.
  (Tradeoff : le pool est une agrégation -> ablation CLS/attention-pool si null, cf. spec.)
- ReconstructHead : projection d_model -> d_in pour le SSL mask-and-reconstruct.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SequenceEncoder(nn.Module):
    def __init__(self, d_in: int = 20, d_model: int = 64, nhead: int = 4,
                 n_layers: int = 4, ff: int = 128, dropout: float = 0.1,
                 max_len: int = 40):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            dropout=dropout, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.max_len = max_len

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x [B,T,d_in], mask [B,T] bool (True = frame valide)
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.proj(x) + self.pos(positions)
        h = self.enc(h, src_key_padding_mask=~mask)   # True at pad
        return h                                        # [B,T,d_model]


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Moyenne sur les frames valides. h [B,T,d], mask [B,T] bool."""
    m = mask.unsqueeze(-1).float()                     # [B,T,1]
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)      # [B,d]


class ClassifierHead(nn.Module):
    def __init__(self, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = masked_mean(h, mask)
        return self.fc(pooled).squeeze(-1)              # [B]


class SequenceClassifier(nn.Module):
    def __init__(self, d_in: int = 20, d_model: int = 64, nhead: int = 4,
                 n_layers: int = 4, ff: int = 128, dropout: float = 0.1, max_len: int = 40):
        super().__init__()
        self.encoder = SequenceEncoder(d_in, d_model, nhead, n_layers, ff, dropout, max_len)
        self.head = ClassifierHead(d_model, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, mask), mask)

    def embed(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_mean(self.encoder(x, mask), mask)  # [B,d_model]