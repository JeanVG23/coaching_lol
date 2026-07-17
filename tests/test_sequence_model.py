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