"""
nts_decompose.py — Decompose the global Hammond progress n_TS into additive
per-element (per-bond) contributions.

THE IDENTITY
------------
With  dT = B_TS - B_R,  dP = B_P - B_R,  and the per-element ratio
n_ij = dT_ij / dP_ij, the projection form of n_TS is exactly a WEIGHTED MEAN of
that ratio matrix:

    n_TS = sum(dT * dP) / sum(dP * dP) = sum(n_ij * w_ij) / sum(w_ij),
    w_ij = dP_ij ** 2

so each element contributes additively

    c_ij = dT_ij * dP_ij / sum(dP ** 2),      sum(c_ij) = n_TS exactly.

c_ij is computed from dT*dP, NOT from n_ij*w_ij. The two are algebraically the
same but n_ij is inf/nan for spectator bonds (dP -> 0), and inf*0 = nan would
poison the sum. Computing from dT*dP is numerically safe and needs no masking.

Everything is elementwise numpy: O(N^2) per molecule, no Python loops.
"""

import numpy as np


def _off(M):
    M = np.asarray(M, dtype=float).copy()
    np.fill_diagonal(M, 0.0)
    return M


def decompose(BO_R, BO_TS, BO_P, eps=1e-12):
    """
    Full decomposition for one R/TS/P triple.

    Returns dict:
      n_ts        float       global progress (identical to bo_scalar.ts_progress)
      contrib     (N,N)       additive contributions; sum == n_ts
      weight      (N,N)       w_ij / sum(w); sum == 1
      ratio       (N,N)       per-element n_ij, NaN where w is negligible
      dT, dP      (N,N)       the two difference matrices
      per_atom    (N,)        row sums of contrib (each bond counted twice:
                              sum(per_atom) == 2 * n_ts)
      forming     float       contribution of bonds with dP > 0
      breaking    float       contribution of bonds with dP < 0
      w_forming / w_breaking  their shares of the total weight
    """
    dT, dP = _off(BO_TS) - _off(BO_R), _off(BO_P) - _off(BO_R)
    W = dP * dP
    tot = float(W.sum())
    if tot <= 0:
        raise ValueError("B_P == B_R: no reaction direction, n_TS undefined")

    contrib = (dT * dP) / tot
    weight = W / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(W > eps * tot, dT / dP, np.nan)

    fwd, rev = dP > 0, dP < 0
    return {"n_ts": float(contrib.sum()),
            "contrib": contrib, "weight": weight, "ratio": ratio,
            "dT": dT, "dP": dP,
            "per_atom": contrib.sum(axis=1),
            "forming": float(contrib[fwd].sum()),
            "breaking": float(contrib[rev].sum()),
            "w_forming": float(weight[fwd].sum()),
            "mat_forming": weight[fwd],
            "w_breaking": float(weight[rev].sum()),
           "mat_breaking": weight[rev]}


def decompose_from_ratio(ratio, dP, eps=1e-12):
    """
    Same decomposition when you ALREADY have the ratio matrix
    n_ij = (BO_TS - BO_R)/(BO_P - BO_R) and the denominator matrix dP.

    Non-finite entries of `ratio` are set to zero only where their weight is
    negligible; a non-finite ratio carrying real weight is an error and is
    reported rather than silently dropped.
    """
    ratio = np.asarray(ratio, dtype=float).copy()
    dP = _off(dP)
    W = dP * dP
    tot = float(W.sum())
    if tot <= 0:
        raise ValueError("dP is all zero: no reaction direction")

    bad = ~np.isfinite(ratio)
    heavy = bad & (W > eps * tot)
    if heavy.any():
        idx = np.argwhere(np.triu(heavy, 1))
        raise ValueError(
            f"non-finite ratio at element(s) with real weight: "
            f"{[tuple(t) for t in idx[:5]]} -- recompute those from dT/dP")
    ratio[bad] = 0.0

    contrib = ratio * W / tot
    fwd, rev = dP > 0, dP < 0
    return {"n_ts": float(contrib.sum()), "contrib": contrib,
            "weight": W / tot, "ratio": ratio, "dP": dP,
            "per_atom": contrib.sum(axis=1),
            "forming": float(contrib[fwd].sum()),
            "breaking": float(contrib[rev].sum()),
            "w_forming": float((W / tot)[fwd].sum()),
            "w_breaking": float((W / tot)[rev].sum())}


def bond_table(res, symbols=None, min_weight=1e-3):
    """
    Per-unique-bond table (needs pandas). Off-diagonal elements appear twice in
    the matrix, so values are DOUBLED here: the 'contrib' column sums to n_ts.
    """
    import pandas as pd
    C, W, Rt = res["contrib"], res["weight"], res["ratio"]
    n = C.shape[0]
    sym = symbols or [str(i + 1) for i in range(n)]
    iu = np.triu_indices(n, 1)
    df = pd.DataFrame({
        "i": iu[0], "j": iu[1],
        "bond": [f"{sym[a]}-{sym[b]}" for a, b in zip(*iu)],
        "dP": res["dP"][iu], "dT": res.get("dT", res["dP"] * np.nan)[iu],
        "ratio_n_ij": Rt[iu],
        "weight": 2 * W[iu],
        "contrib": 2 * C[iu],
    })
    df["kind"] = np.where(df["dP"] > 0, "forming",
                          np.where(df["dP"] < 0, "breaking", "spectator"))
    df = df[df["weight"] >= min_weight].copy()
    return df.sort_values("weight", ascending=False).reset_index(drop=True)


def decompose_many(BO_R, BO_TS, BO_P):
    """
    Vectorised over a stack of molecules with the SAME N.
    Inputs (m, N, N) -> returns n_ts (m,), contrib (m,N,N), weight (m,N,N),
    forming (m,), breaking (m,). No Python loop over molecules.
    """
    R3, T3, P3 = (np.asarray(x, dtype=float).copy() for x in (BO_R, BO_TS, BO_P))
    for A in (R3, T3, P3):                       # zero every diagonal at once
        A[:, np.arange(A.shape[1]), np.arange(A.shape[1])] = 0.0
    dT, dP = T3 - R3, P3 - R3
    W = dP * dP
    tot = W.sum(axis=(1, 2))
    if np.any(tot <= 0):
        raise ValueError(f"{int((tot <= 0).sum())} molecule(s) have B_P == B_R")
    contrib = (dT * dP) / tot[:, None, None]
    weight = W / tot[:, None, None]
    fwd, rev = dP > 0, dP < 0
    return {"n_ts": contrib.sum(axis=(1, 2)),
            "contrib": contrib, "weight": weight,
            "forming": np.where(fwd, contrib, 0).sum(axis=(1, 2)),
            "breaking": np.where(rev, contrib, 0).sum(axis=(1, 2)),
            "w_forming": np.where(fwd, weight, 0).sum(axis=(1, 2))}
