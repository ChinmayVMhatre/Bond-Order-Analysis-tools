"""Tests for parsing and decomposition."""

from pathlib import Path

import numpy as np
import pytest

from bonts import (load_triple, mbs_matrix, minpop_bond_orders, decompose,
                   decompose_from_ratio, decompose_many, bond_table,
                   antibonding)

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
R_LOG, TS_LOG, P_LOG = (DATA / f"sn2_{s}.log"
                        for s in ("reactant", "ts", "product"))


@pytest.fixture(scope="module")
def triple():
    return load_triple(R_LOG, TS_LOG, P_LOG)


def test_mbs_matrix_shape_and_symmetry():
    M = mbs_matrix(TS_LOG)
    assert M.shape == (6, 6)
    assert np.allclose(M, M.T)


def test_mbs_values_match_file():
    M = mbs_matrix(TS_LOG)
    assert M[0, 1] == pytest.approx(0.225)      # F-C shared population
    assert M[1, 5] == pytest.approx(0.240)      # C-Cl
    assert M[0, 5] == pytest.approx(-0.031)     # negative, non-bonded
    assert M[5, 5] == pytest.approx(17.106)     # diagonal, second column block


def test_bond_orders_doubled_and_diagonal_zeroed():
    B = minpop_bond_orders(TS_LOG)
    assert np.allclose(np.diag(B), 0.0)
    assert B[0, 1] == pytest.approx(0.450)
    assert np.allclose(B, B.T)


def test_missing_section_returns_none(tmp_path):
    f = tmp_path / "empty.log"
    f.write_text("Normal termination of Gaussian 16.\n")
    assert mbs_matrix(f) is None


def test_load_triple_rejects_mismatched_sizes(tmp_path):
    bad = tmp_path / "bad.log"
    bad.write_text("   MBS Condensed to atoms (all electrons):\n"
                   "             1             2\n"
                   "      1  C       1.000000\n"
                   "      2  H       0.500000      0.800000\n\n")
    with pytest.raises(ValueError, match="atom counts differ"):
        load_triple(R_LOG, bad, P_LOG)


def test_decompose_identity(triple):
    B_R, B_TS, B_P, _ = triple
    res = decompose(B_R, B_TS, B_P)
    assert res["contrib"].sum() == pytest.approx(res["n_ts"])
    assert res["weight"].sum() == pytest.approx(1.0)
    assert res["forming"] + res["breaking"] == pytest.approx(res["n_ts"])


def test_decompose_endpoints(triple):
    B_R, _, B_P, _ = triple
    assert decompose(B_R, B_R, B_P)["n_ts"] == pytest.approx(0.0)
    assert decompose(B_R, B_P, B_P)["n_ts"] == pytest.approx(1.0)


def test_decompose_linear_midpoint(triple):
    B_R, _, B_P, _ = triple
    mid = B_R + 0.5 * (B_P - B_R)
    assert decompose(B_R, mid, B_P)["n_ts"] == pytest.approx(0.5)


def test_identical_endpoints_raise(triple):
    B_R, B_TS, _, _ = triple
    with pytest.raises(ValueError, match="no reaction direction"):
        decompose(B_R, B_TS, B_R)


def test_from_ratio_matches_decompose(triple):
    B_R, B_TS, B_P, _ = triple
    res = decompose(B_R, B_TS, B_P)
    out = decompose_from_ratio(np.nan_to_num(res["ratio"], nan=0.0), res["dP"])
    assert out["n_ts"] == pytest.approx(res["n_ts"])


def test_from_ratio_rejects_heavy_nonfinite(triple):
    B_R, B_TS, B_P, _ = triple
    res = decompose(B_R, B_TS, B_P)
    ratio = np.nan_to_num(res["ratio"], nan=0.0)
    ratio[0, 1] = ratio[1, 0] = np.inf          # F-C carries real weight
    with pytest.raises(ValueError, match="non-finite"):
        decompose_from_ratio(ratio, res["dP"])


def test_bond_table_sums_to_n_ts(triple):
    B_R, B_TS, B_P, symbols = triple
    res = decompose(B_R, B_TS, B_P)
    df = bond_table(res, symbols, min_weight=0.0)
    assert df["contrib"].sum() == pytest.approx(res["n_ts"])
    assert df["weight"].sum() == pytest.approx(1.0)


def test_decompose_many_matches_single(triple):
    B_R, B_TS, B_P, _ = triple
    m = 5
    out = decompose_many(np.stack([B_R] * m), np.stack([B_TS] * m),
                         np.stack([B_P] * m))
    single = decompose(B_R, B_TS, B_P)["n_ts"]
    assert np.allclose(out["n_ts"], single)
    assert np.allclose(out["contrib"].sum(axis=(1, 2)), out["n_ts"])


def test_antibonding_zero_on_the_line(triple):
    """a_excess vanishes for any n_TS when the TS lies on the straight line."""
    B_R, _, B_P, _ = triple
    for n in (0.1, 0.25, 0.5, 0.75, 0.9):
        lin = B_R + n * (B_P - B_R)
        assert antibonding(B_R, lin, B_P)["a_excess"] == pytest.approx(0.0,
                                                                      abs=1e-12)


def test_antibonding_additivity(triple):
    B_R, B_TS, B_P, _ = triple
    res = antibonding(B_R, B_TS, B_P)
    assert res["excess"].sum() == pytest.approx(res["a_excess"])


def test_transient_antibonding_detected(triple):
    """F...Cl is near zero at both endpoints and antibonding at the TS."""
    B_R, B_TS, B_P, _ = triple
    res = antibonding(B_R, B_TS, B_P)
    assert (0, 5) in res["classes"]["transient_antibonding"]
