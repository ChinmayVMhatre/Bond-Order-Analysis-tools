"""Identities that must hold for the n_TS decomposition. No data files needed."""
import numpy as np
import pytest

import nts_decompose as D


def mat(n, bonds):
    M = np.zeros((n, n))
    for (i, j), v in bonds.items():
        M[i, j] = M[j, i] = v
    return M


@pytest.fixture
def h_transfer():
    """O1-H2...N3 -> O1...H2-N3, with N3-C4 a spectator."""
    def mk(oh, hn, nc):
        return mat(4, {(0, 1): oh, (1, 2): hn, (2, 3): nc, (0, 2): 0.02})
    return mk(0.90, 0.05, 1.00), mk(0.45, 0.50, 1.02), mk(0.05, 0.90, 1.00)


def test_contributions_sum_to_n_ts(h_transfer):
    r = D.decompose(*h_transfer)
    assert np.isclose(r["contrib"].sum(), r["n_ts"])
    assert np.isclose(r["n_ts"], 0.5294117647, atol=1e-9)


def test_weights_sum_to_one(h_transfer):
    r = D.decompose(*h_transfer)
    assert np.isclose(r["weight"].sum(), 1.0)


def test_channels_partition_n_ts(h_transfer):
    r = D.decompose(*h_transfer)
    assert np.isclose(r["forming"] + r["breaking"], r["n_ts"])


def test_spectator_contributes_zero(h_transfer):
    """N3-C4 moves at the TS but not over R->P, so it must contribute 0."""
    r = D.decompose(*h_transfer)
    assert r["dT"][2, 3] != 0.0            # it does move at the TS
    assert np.isclose(r["dP"][2, 3], 0.0)  # but not over the reaction
    assert np.isclose(r["contrib"][2, 3], 0.0)


def test_ratio_input_matches_matrix_input(h_transfer):
    R, T, P = h_transfer
    dT, dP = T - R, P - R
    for A in (dT, dP):
        np.fill_diagonal(A, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = dT / dP
    assert (~np.isfinite(ratio)).any()     # the inf/nan the docs warn about
    a = D.decompose(R, T, P)["n_ts"]
    b = D.decompose_from_ratio(ratio, dP)["n_ts"]
    assert np.isclose(a, b)


def test_nonfinite_ratio_with_weight_raises(h_transfer):
    R, T, P = h_transfer
    dP = P - R
    np.fill_diagonal(dP, 0.0)
    ratio = np.zeros_like(dP)
    ratio[0, 1] = ratio[1, 0] = np.inf     # heavy weight, non-finite
    with pytest.raises(ValueError, match="non-finite"):
        D.decompose_from_ratio(ratio, dP)


def test_bond_table_sums_to_n_ts(h_transfer):
    r = D.decompose(*h_transfer)
    df = D.bond_table(r, ["O1", "H2", "N3", "C4"])
    assert np.isclose(df["contrib"].sum(), r["n_ts"], atol=1e-9)


def test_per_atom_sums_to_n_ts(h_transfer):
    """Each bond's contribution is split between its two atoms."""
    r = D.decompose(*h_transfer)
    assert np.isclose(r["per_atom"].sum(), r["n_ts"])
    # the shared hydrogen carries twice what each terminal atom does
    assert np.isclose(r["per_atom"][1], 2 * r["per_atom"][0])


def test_identical_endpoints_raise(h_transfer):
    R, T, _ = h_transfer
    with pytest.raises(ValueError, match="no reaction direction"):
        D.decompose(R, T, R)


def test_decompose_many_matches_loop(h_transfer):
    R, T, P = h_transfer
    m = 7
    out = D.decompose_many(np.stack([R] * m), np.stack([T] * m),
                           np.stack([P] * m))
    single = D.decompose(R, T, P)["n_ts"]
    assert out["n_ts"].shape == (m,)
    assert np.allclose(out["n_ts"], single)
    assert np.allclose(out["forming"] + out["breaking"], out["n_ts"])


# ---------------- delocalisation diagnostics ----------------
def test_participation_counts_localised_vs_delocalised():
    R2 = mat(4, {(0, 1): 0.9, (1, 2): 0.05})
    T2 = mat(4, {(0, 1): 0.45, (1, 2): 0.50})
    assert np.isclose(D.participation(R2, T2)["n_eff"], 2.0, atol=1e-6)

    m = 6
    Rp = mat(m, {(0, 1): 1.8, (1, 2): 1.0, (2, 3): 1.8,
                 (3, 4): 0.1, (4, 5): 1.9, (5, 0): 0.1})
    Tp = mat(m, {(0, 1): 1.4, (1, 2): 1.4, (2, 3): 1.4,
                 (3, 4): 0.6, (4, 5): 1.4, (5, 0): 0.6})
    assert D.participation(Rp, Tp)["n_eff"] > 5.0


def test_transient_bond_invisible_to_n_ts_but_detected():
    k = 4
    R = mat(k, {(0, 1): 0.95, (2, 3): 0.95})
    T = mat(k, {(0, 1): 0.50, (2, 3): 0.50, (1, 2): 0.60})
    P = mat(k, {(0, 1): 0.05, (2, 3): 0.05})
    r = D.decompose(R, T, P)
    assert np.isclose(r["weight"][1, 2], 0.0)
    assert np.isclose(r["contrib"][1, 2], 0.0)
    tr = D.transient_bonds(R, T, P)
    assert (1, 2) in tr["pairs"]
    assert tr["transient_index"] > 0.3


def test_path_curvature_straight_is_one():
    k = 4
    R = mat(k, {(0, 1): 0.95, (2, 3): 0.95})
    P = mat(k, {(0, 1): 0.05, (2, 3): 0.05})
    straight = [R + (P - R) * t for t in np.linspace(0, 1, 7)]
    assert np.isclose(D.path_curvature(straight)["kappa"], 1.0, atol=1e-9)

    curved = []
    for t in np.linspace(0, 1, 7):
        M = (R + (P - R) * t).copy()
        M[1, 2] = M[2, 1] = 0.8 * np.sin(np.pi * t)
        curved.append(M)
    assert D.path_curvature(curved)["kappa"] > 1.5


# ---------------- antibonding ----------------
def test_antibonding_split_is_nonnegative():
    M = mat(3, {(0, 1): 0.5, (1, 2): -0.3})
    A = D.antibonding(M)
    assert (A >= 0).all()
    assert np.isclose(A[1, 2], 0.3)
    assert np.isclose(D.total_antibonding(M), 0.3)


def test_sign_crossing_classified_with_zero_position():
    sym = ["A1", "B2", "C3", "D4"]
    R = mat(4, {(0, 1): 0.90, (1, 2): 0.05, (0, 2): +0.30})
    T = mat(4, {(0, 1): 0.45, (1, 2): 0.50, (0, 2): +0.02})
    P = mat(4, {(0, 1): 0.05, (1, 2): 0.90, (0, 2): -0.25})
    tab = D.classify_sign_changes(R, T, P, tol=1e-3, symbols=sym)
    row = tab[tab["pair"] == "A1-C3"].iloc[0]
    assert row["kind"] == "bonding->antibonding"
    assert np.isclose(row["n_zero"], 0.30 / (0.30 + 0.25))
    assert row["ts_past_crossing"] is False or row["ts_past_crossing"] == False


def test_transient_antibonding_seen_only_by_excess():
    R = mat(4, {(0, 1): 0.90, (1, 2): 0.05, (0, 2): 0.00})
    T = mat(4, {(0, 1): 0.45, (1, 2): 0.50, (0, 2): -0.22})
    P = mat(4, {(0, 1): 0.05, (1, 2): 0.90, (0, 2): 0.00})
    prog = D.antibond_progress(R, T, P)
    assert not prog["reliable"]                      # no antibonding change R->P
    exc = D.antibond_excess(R, T, P)
    assert exc["a_hat"] > 0.1
    assert np.isclose(exc["excess_matrix"][0, 2], -(-0.22))
    tab = D.classify_sign_changes(R, T, P, tol=1e-3)
    assert (tab["kind"] == "transient antibonding").any()


def test_no_antibonding_gives_empty_table():
    R = mat(4, {(0, 1): 0.90, (1, 2): 0.05})
    T = mat(4, {(0, 1): 0.45, (1, 2): 0.50})
    P = mat(4, {(0, 1): 0.05, (1, 2): 0.90})
    assert len(D.classify_sign_changes(R, T, P)) == 0
    assert np.isclose(D.antibond_excess(R, T, P)["a_hat"], 0.0)
