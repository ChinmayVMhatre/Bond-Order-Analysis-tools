"""Tests for bonts. Run with: python -m pytest tests/ -q"""

from pathlib import Path

import numpy as np
import pytest

import bonts

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
FILES = [DATA / f"sn2_{s}.log" for s in ("R", "TS", "P")]


@pytest.fixture(scope="module")
def result():
    return bonts.analyse_reaction(*FILES)


def test_parse_shape_and_electrons():
    mbs = bonts.parse_mbs_condensed(bonts.read_output(FILES[1]))
    assert mbs.shape == (7, 7)
    assert np.allclose(mbs, mbs.T)
    assert np.isclose(mbs.sum(), 28.0, atol=1e-4)


def test_bond_orders_diagonal_zero():
    mbs = bonts.parse_mbs_condensed(bonts.read_output(FILES[1]))
    bo = bonts.mbs_bond_orders(mbs)
    assert np.allclose(np.diag(bo), 0.0)
    assert np.isclose(bo[0, 2], 2 * mbs[0, 2])


def test_labels(result):
    assert result["labels"] == ["O1", "H2", "C3", "H4", "H5", "H6", "F7"]


def test_decompose_identities(result):
    d = result["bonding"]
    assert np.isclose(d["contrib"].sum(), d["n_ts"])
    assert np.isclose(d["weight"].sum(), 1.0)
    assert np.isclose(d["forming"] + d["breaking"], d["n_ts"])


def test_antibonding_identity(result):
    a = result["anti"]
    assert np.isclose(a["excess"].sum(), a["a_excess"])


def test_antibonding_independent_of_n_ts(result):
    """excess vanishes when the TS lies on the straight R->P line, for any n."""
    R, P = result["BO_R"], result["BO_P"]
    for n in (0.1, 0.3, 0.5, 0.7, 0.9):
        T = R + n * (P - R)
        a = bonts.antibonding(R, T, P)
        assert abs(a["a_excess"]) < 1e-12


def test_expected_classes(result):
    cls = result["anti"]["classes"]
    assert (0, 6) in cls["transient_antibonding"]     # O1...F7 only at the TS
    assert (1, 2) in cls["becomes_antibonding"]       # H2...C3 positive -> negative


def test_missing_section(tmp_path):
    p = tmp_path / "empty.log"
    p.write_text("nothing here\n")
    with pytest.raises(ValueError):
        bonts.load_bond_orders(p)
