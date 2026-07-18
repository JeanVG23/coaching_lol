"""Tests sequence_model : encoder, masked-mean-pool, reconstruct."""
import importlib.util, sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC / "02_data_science"))
_spec = importlib.util.spec_from_file_location(
    "sequence_model", _SRC / "02_data_science" / "sequence_model.py")
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)

import torch

def test_encoder_forward_shapes():
    torch.manual_seed(0)
    enc = sm.SequenceEncoder(d_in=20, d_model=64, max_len=40)
    x = torch.randn(4, 40, 20)
    mask = torch.ones(4, 40, dtype=torch.bool)
    mask[:, 30:] = False                      # 30 minutes valides
    h = enc(x, mask)
    assert h.shape == (4, 40, 64)


def test_masked_mean_ignores_pad():
    torch.manual_seed(0)
    h = torch.randn(2, 40, 64)
    m = torch.ones(2, 40, dtype=torch.bool); m[:, 30:] = False   # 30 frames valides
    # le pooled ne dépend que des frames valides : changer le contenu paddé ne change rien
    a = sm.masked_mean(h, m)
    h_pad_garbage = h.clone()
    h_pad_garbage[:, 30:] = torch.randn(2, 10, 64) * 1000.0
    b = sm.masked_mean(h_pad_garbage, m)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    # et le pooled == moyenne directe des 30 frames valides
    torch.testing.assert_close(a, h[:, :30].mean(dim=1), rtol=1e-5, atol=1e-6)


def test_classifier_head_shape():
    torch.manual_seed(0)
    clf = sm.SequenceClassifier(d_in=20, d_model=64, max_len=40)
    x = torch.randn(8, 40, 20)
    mask = torch.ones(8, 40, dtype=torch.bool); mask[:, 25:] = False
    logits = clf(x, mask)
    assert logits.shape == (8,)


def test_reconstruct_head_shape():
    rh = sm.ReconstructHead(d_model=64, d_in=20)
    h = torch.randn(4, 40, 64)
    out = rh(h)
    assert out.shape == (4, 40, 20)


def test_sequence_classifier_accepts_d_in_27():
    torch.manual_seed(0)
    clf = sm.SequenceClassifier(d_in=27, d_model=64, max_len=40)
    x = torch.randn(4, 40, 27)
    mask = torch.ones(4, 40, dtype=torch.bool)
    assert clf(x, mask).shape == (4,)
    assert clf.embed(x, mask).shape == (4, 64)