"""bo_scalar, reacting_bonds, bo_change, pauling_bo, rho_from_dicts."""
import numpy as np
import pytest

import bo_scalar as S
import reacting_bonds as RB
import bo_change as BC
import pauling_bo as PB
import rho_from_dicts as RD


def mat(n, bonds):
    M = np.zeros((n, n))
    for (i, j), v in bonds.items():
        M[i, j] = M[j, i] = v
    return M


# ---------------- bo_scalar ----------------
@pytest.fixture
def triple():
    mk = lambda oh, hn, nc: mat(4, {(0, 1): oh, (1, 2): hn, (2, 3): nc})
    return mk(0.90, 0.05, 1.00), mk(0.45, 0.50, 1.00), mk(0.05, 0.90, 1.00)


def test_ts_progress_endpoints(triple):
    R, T, P = triple
    assert np.isclose(S.ts_progress(R, R, P), 0.0)
    assert np.isclose(S.ts_progress(R, P, P), 1.0)


def test_ts_progress_scale_invariant(triple):
    R, T, P = triple
    a = S.ts_progress(R, T, P)
    b = S.ts_progress(0.6 * R, 0.6 * T, 0.6 * P)
    assert np.isclose(a, b)


def test_change_magnitude_zero_for_no_change(triple):
    R, _, _ = triple
    assert np.isclose(S.change_magnitude(R, R), 0.0)


def test_asynchronicity_discriminates(triple):
    R, _, P = triple
    mk = lambda oh, hn: mat(4, {(0, 1): oh, (1, 2): hn, (2, 3): 1.00})
    sync = mk(0.50, 0.50)
    asyn = mk(0.85, 0.80)
    assert S.asynchronicity(R, sync, P) < S.asynchronicity(R, asyn, P)


# ---------------- reacting_bonds ----------------
def test_proton_transfer_breaking_and_forming():
    Z = [8, 1, 7]
    R = [[0, 0, 0], [0.98, 0, 0], [2.80, 0, 0]]
    P = [[0, 0, 0], [1.78, 0, 0], [2.80, 0, 0]]
    o = RB.find_reacting_bonds(Z, R, P, verbose=False)
    assert o["breaking"] == [(0, 1)]
    assert o["forming"] == [(1, 2)]


def test_pi_bond_change_invisible_to_xor():
    """C=C -> C-C: connectivity identical, must be caught as order-change."""
    Z = [6, 6, 1, 1]
    R = [[0, 0, 0], [1.33, 0, 0], [-0.55, 0.93, 0], [1.88, -0.93, 0]]
    P = [[0, 0, 0], [1.54, 0, 0], [-0.55, 0.93, 0], [2.09, -0.93, 0]]
    o = RB.find_reacting_bonds(Z, R, P, verbose=False)
    assert not (o["A_R"] != o["A_P"]).any()        # XOR finds nothing
    assert o["order_change"] == [(0, 1)]


# ---------------- bo_change ----------------
def test_calibrated_thresholds_recover_classes():
    """
    SN2-like: Nu(0) attacks C(1), LG(2) leaves; H3/H4/H5 spectator C-H bonds
    and a remote C6-C7 bond that changes order.

    NOTE the fixture size matters. `delta` is median + 6*MAD of |dBO| over
    bonds present in BOTH states, so it needs several such bonds to estimate a
    noise floor. With only ~4 bonds the MAD is dominated by the reacting ones
    and delta blows up (observed: 3.08), classifying everything as spectator.
    """
    rng = np.random.default_rng(0)
    n = 8

    def build(nu, lg, cc, noise=0.012):
        M = np.abs(rng.normal(0, noise, (n, n)))
        M = (M + M.T) / 2
        for (i, j), v in {(0, 1): nu, (1, 2): lg, (1, 3): 0.98, (1, 4): 0.98,
                          (1, 5): 0.98, (6, 7): cc, (0, 6): 0.97}.items():
            M[i, j] = M[j, i] = v
        np.fill_diagonal(M, 0.0)
        return M

    data = [(build(rng.uniform(0.02, 0.09), rng.uniform(0.88, 0.99),
                   rng.uniform(1.80, 1.92)),
             build(rng.uniform(0.88, 0.99), rng.uniform(0.02, 0.09),
                   rng.uniform(1.00, 1.12)))
            for _ in range(80)]
    tau, delta = BC.calibrate_thresholds(data, verbose=False)
    assert 0.0 < tau < 0.5
    r = BC.classify_pairs(*data[0], tau=tau, delta=delta, verbose=False)
    assert (0, 1) in r["forming"]
    assert (1, 2) in r["breaking"]
    assert (6, 7) in r["order_change"]


def test_log_ratio_separates_classes():
    n = 4
    R = mat(n, {(0, 1): 0.05, (1, 2): 0.93, (2, 3): 1.00})
    P = mat(n, {(0, 1): 0.93, (1, 2): 0.05, (2, 3): 1.00})
    lr = BC.log_ratio(R, P)
    assert lr[0, 1] > 1.5          # forming
    assert lr[1, 2] < -1.5         # breaking
    assert abs(lr[2, 3]) < 0.1     # spectator


# ---------------- pauling_bo ----------------
def test_pauling_diagonal_is_row_sum():
    symbols = ["C", "H", "H", "H", "H"]
    coords = [(0, 0, 0), (1.09, 0, 0), (-0.36, 1.03, 0),
              (-0.36, -0.51, 0.89), (-0.36, -0.51, -0.89)]
    M = PB.bond_order_matrix(symbols, coords, as_numpy=True)
    off = M.copy()
    np.fill_diagonal(off, 0.0)
    assert np.allclose(np.diag(M), off.sum(axis=1))
    assert 3.5 < M[0, 0] < 4.5     # carbon valence near 4


def test_pauling_rejects_unknown_element():
    with pytest.raises(ValueError, match="not in the radii table"):
        PB.bond_order_matrix(["C", "Xe"], [(0, 0, 0), (1.5, 0, 0)])


def test_pauling_split_valence():
    symbols = ["C", "H"]
    M = PB.bond_order_matrix(symbols, [(0, 0, 0), (1.09, 0, 0)], as_numpy=True)
    val, BO = PB.split_valence(M)
    assert np.allclose(np.diag(BO), 0.0)
    assert np.allclose(val, BO.sum(axis=1))


# ---------------- rho_from_dicts ----------------
def test_condensed_length_and_symmetry():
    M = mat(10, {(0, 1): 0.5, (2, 3): -0.2})
    v = RD.condensed(M)
    assert v.shape == (45,)        # n(n-1)/2


def test_spearman_is_factor_invariant():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 8)); A = np.abs(A + A.T)
    B = A * 0.7 + np.abs(rng.normal(0, 0.05, (8, 8)))
    B = (B + B.T) / 2
    a1, b1 = RD.condensed(A, 2.0), RD.condensed(B, 1.0)
    a2, b2 = RD.condensed(A, 1.0), RD.condensed(B, 4.0)
    assert spearmanr(a1, b1).statistic == spearmanr(a2, b2).statistic


def test_mask_from_is_required():
    d = {"a": {"m1": mat(4, {(0, 1): 0.5})},
         "b": {"m1": mat(4, {(0, 1): 0.4})}}
    with pytest.raises(ValueError, match="mask_from is required"):
        RD.build_rho(d, verbose=False)


def test_no_mask_is_explicit_and_works():
    rng = np.random.default_rng(1)
    def M():
        A = np.abs(rng.normal(0.5, 0.2, (6, 6)))
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0.0)
        return A
    ids = [f"m{i}" for i in range(12)]
    d = {"a": {i: M() for i in ids}, "b": {i: M() for i in ids}}
    _, methods, RHO = RD.build_rho(d, mask_from=False, verbose=False)
    assert RHO.shape == (12, 2, 2)
    assert RD.LAST_BUILD["use_mask"] is False
    assert RD.LAST_BUILD["n_used"][0] == 15      # all 6*5/2 elements


def test_group_map_forms_agree():
    methods = ["a", "b", "c", "d"]
    g1 = {"X": ["a", "b"], "Y": ["c", "d"]}
    g2 = {"a": "X", "b": "X", "c": "Y", "d": "Y"}
    g3 = lambda m: "X" if m in ("a", "b") else "Y"
    sets = [sorted(set(RD.pair_groups(methods, g)[0].values()))
            for g in (g1, g2, g3)]
    assert sets[0] == sets[1] == sets[2] == ["X-X", "X-Y", "Y-Y"]


def test_pair_string_rejected():
    with pytest.raises(TypeError, match="ambiguous"):
        RD.select_pairs(["a-1", "b"], pairs=["a-1-b"])
