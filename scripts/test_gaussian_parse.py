"""
Regression tests against a real Gaussian output.

These pin the numbers documented in docs/07_validated_numbers.md. They SKIP
unless a reference file is available: put dsgdb9nsd_000200.out.bz2 in
tests/data/ or set QM9_TEST_LOG=/path/to/file.out.bz2
"""
import os
from pathlib import Path

import numpy as np
import pytest

import gaussian_parse as gp

CANDIDATES = [
    os.environ.get("QM9_TEST_LOG"),
    Path(__file__).parent / "data" / "dsgdb9nsd_000200.out.bz2",
]
LOG = next((str(p) for p in CANDIDATES if p and Path(p).exists()), None)
needs_log = pytest.mark.skipif(LOG is None,
                               reason="reference Gaussian log not available")

Z = np.array([8, 6, 7, 6, 6, 7, 1, 1, 1, 1])


@pytest.fixture
def jobs():
    return gp.split_jobs(gp.read_log(LOG))


@needs_log
def test_two_jobs_found(jobs):
    """'Entering Gaussian System' appears once; the split must use
    'Initial command:' instead."""
    assert len(jobs) == 2
    assert "Opt" in gp.job_route(jobs[0])
    assert "ROHF" in gp.job_route(jobs[1])


@needs_log
def test_scf_energies(jobs):
    assert np.isclose(gp.scf_energy(jobs[0])[1], -301.539850194, atol=1e-8)
    assert np.isclose(gp.scf_energy(jobs[1])[1], -299.789162415, atol=1e-8)


@needs_log
def test_dipole_is_scoped_per_job(jobs):
    """Taking the last match in the whole file returns the ROHF value."""
    assert np.isclose(gp.dipole(jobs[0]), 2.1937)
    assert np.isclose(gp.dipole(jobs[1]), 2.2651)


@needs_log
def test_condensed_row_sum_gives_the_charges(jobs):
    """The DIAGONAL is the net population; the ROW SUM is the gross."""
    c = gp.condensed_to_atoms(jobs[1], kind="mbs")
    assert np.isclose(c["net_pop"][0], 7.982530, atol=1e-5)
    assert np.isclose(c["gross_pop"][0], 8.528085, atol=1e-5)
    charges = Z - c["gross_pop"]
    expected = np.array([-0.528084, 0.563379, -0.514855, 0.021484, -0.034297,
                         -0.430492, 0.379927, 0.114509, 0.121565, 0.306864])
    assert np.allclose(charges, expected, atol=1e-5)


@needs_log
def test_three_condensed_blocks_are_distinct(jobs):
    kinds = [gp.condensed_to_atoms(jobs[1], kind=k)
             for k in ("plain", "bonding", "mbs")]
    assert all(k is not None for k in kinds)
    diags = [k["net_pop"][0] for k in kinds]
    assert len(set(np.round(diags, 6))) == 3


@needs_log
def test_electron_count_from_orbital_populations(jobs):
    """The '6D 0' tokenisation bug made this 43.9586 instead of 44."""
    g = gp.gross_orbital_populations(jobs[1], kind="mbs")
    assert np.isclose(g["total"].sum(), 44.0, atol=1e-3)
    gf = gp.gross_orbital_populations(jobs[1], kind="plain")
    assert np.isclose(gf["total"].sum(), 44.0, atol=1e-3)
    assert any(" " in str(l) for l in gf["labels"])     # '6D 0' style present


@needs_log
def test_shell_sums_match_condensed_row_sums(jobs):
    """Two independent code paths must agree."""
    g = gp.gross_orbital_populations(jobs[1], kind="mbs")
    c = gp.condensed_to_atoms(jobs[1], kind="mbs")
    shells = g["s_pop"] + g["p_pop"] + g["d_pop"] + g["f_pop"]
    assert np.allclose(shells, c["gross_pop"], atol=1e-4)


@needs_log
def test_mayer_valency_and_bond_orders(jobs):
    m = gp.mayer(jobs[1])
    assert np.isclose(m["valency"][0], 1.9863, atol=1e-4)     # O, divalent
    assert np.isclose(m["valency"][5], 3.3031, atol=1e-4)     # pyrrole-type N
    assert np.allclose(np.diag(m["bond_orders"]), 0.0)
    assert np.isclose(m["bond_orders"][3, 4], 1.6923, atol=1e-4)   # C4-C5
    assert np.isclose(m["bond_orders"][0, 1], 1.0227, atol=1e-4)   # O1-C2, enol


@needs_log
def test_bonding_charges_are_not_charges(jobs):
    b = gp.bonding_populations(jobs[1])
    assert not np.isclose(b["bonding_population"].sum(), 0.0, atol=1e-3)


@needs_log
def test_density_matrix_shapes(jobs):
    P = gp.find_matrix(jobs[1], gp.H_DENSITY, occurrence=0)
    assert P.shape == (244, 244)
    assert np.allclose(P, P.T)


@needs_log
def test_geometry_and_gaussian_bond_list(jobs):
    atoms = gp.parse_geometry(jobs[0])
    assert len(atoms) == 10
    assert np.array_equal(np.asarray(atoms.numbers), Z)
    bonds = gp.parse_bond_list(jobs[0])
    assert len(bonds) == 10
    assert (0, 1) in bonds


@needs_log
def test_electron_count_from_mbs_matrix_sum(jobs):
    M = gp.condensed_to_atoms(jobs[1], kind="mbs")["matrix"]
    assert np.isclose(M.sum(), 44.0, atol=1e-4)


@needs_log
def test_load_all(jobs):
    d = gp.load_all(LOG)
    assert np.isclose(d["n_electrons"], 44.0, atol=1e-4)
    assert np.isclose(d["dipole"], 2.1937)      # job 1, the DFT value
    assert d["pauling"].shape == (10, 10)


# ---------------- printed matrices: no data file needed ----------------
WIBERG_IAO = """
 Wiberg bond-order matrix (orthogonalized IAOs, Knizia)
               C1        O2        C3        C4        C5
     C1    0.0000    0.9554    0.0045    0.0086    0.0032
     O2    0.9554    0.0000    0.9502    0.0083    0.0168
     C3    0.0045    0.9502    0.0000    1.0322    0.0165
     C4    0.0086    0.0083    1.0322    0.0000    2.9114
     C5    0.0032    0.0168    0.0165    2.9114    0.0000
"""

WIBERG_NAO = """
 Wiberg bond index matrix in the NAO basis:

     Atom    1       2       3       4       5
     ---- ------  ------  ------  ------  ------
   1.  C  0.0000  0.2257  0.0023  0.0016  0.0007
   2.  O  0.2257  0.0000  0.2261  0.0036  0.0041
   3.  C  0.0023  0.2261  0.0000  0.2604  0.0074
   4.  C  0.0016  0.0036  0.2604  0.0000  0.7268
   5.  C  0.0007  0.0041  0.0074  0.7268  0.0000
"""

PAULING_PRINT = """
Pauling bond-order matrix (diagonal = row-sum valence)
               C1        C2        H3        H4
     C1  1.951683  0.975724  0.000000  0.975959
     C2  0.975724  0.975724  0.000000  0.000000
     H3  0.000000  0.000000  0.000000  0.000000
     H4  0.975959  0.000000  0.000000  0.975959
"""


def test_wiberg_iao_parses():
    r = gp.parse_bo_matrix(WIBERG_IAO, "wiberg_iao")
    assert r["matrix"].shape == (5, 5)
    assert np.isclose(r["matrix"][3, 4], 2.9114)        # the triple bond
    assert r["diagonal_kind"] == "zero"
    assert list(r["symbols"]) == ["C", "O", "C", "C", "C"]


def test_wiberg_nao_parses():
    r = gp.parse_bo_matrix(WIBERG_NAO, "wiberg_nao")
    assert r["matrix"].shape == (5, 5)
    assert np.isclose(r["matrix"][3, 4], 0.7268)
    assert np.allclose(r["matrix"], r["matrix"].T)


def test_pauling_print_diagonal_residual_is_small():
    r = gp.parse_bo_matrix(PAULING_PRINT, "pauling")
    assert r["diagonal_kind"] == "row_sum_valence"
    assert r["diagonal_residual"] < 1e-5
    assert np.allclose(np.diag(r["bond_orders"]), 0.0)


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="kind must be one of"):
        gp.parse_bo_matrix(WIBERG_IAO, "nbo")


def test_missing_section_returns_none():
    assert gp.parse_bo_matrix("nothing here", "pauling") is None
