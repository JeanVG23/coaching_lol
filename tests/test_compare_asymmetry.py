"""Incrément 2 — garde-fou asymétrie côté coaching : compare ne benchmarke QUE
des features exactes/safe (jamais de proxy vision ML_ONLY)."""
import compare
import positioning as P


def test_compare_pos_rows_are_coaching_safe():
    keys = {k for _, k, _ in compare.POS_ROWS}
    assert keys <= P.COACHING_SAFE
    assert keys.isdisjoint(P.ML_ONLY)


def test_compare_pos_rows_formats_valid():
    assert all(fmt in ("pct", "num") for _, _, fmt in compare.POS_ROWS)
