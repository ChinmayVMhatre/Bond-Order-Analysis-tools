"""Decompose the global progress n_TS into per-bond contributions."""

import numpy as np


def _off(M):
    M = np.asarray(M, dtype=float).copy()
    np.fill_diagonal(M, 0.0)
    return M


def decompose(B_R, B_TS, B_P, eps=1e-12):
    """Per-element decomposition of n_TS for one reactant/TS/product triple.

    Returns a dict with:
      n_ts                  global progress (0 = reactant-like, 1 = product-like)
      contrib   (N, N)      additive contributions; sum equals n_ts
      weight    (N, N)      normalised weights; sum equals 1
      ratio     (N, N)      per-element ratio, NaN where the weight is negligible
      dT, dP    (N, N)      B_TS - B_R and B_P - B_R
      per_atom  (N,)        row sums of contrib
      forming / breaking    contributions of elements with dP > 0 / dP < 0
      w_forming / w_breaking   their shares of the total weight
    """
    dT, dP = _off(B_TS) - _off(B_R), _off(B_P) - _off(B_R)
    W = dP * dP
    total = float(W.sum())
    if total <= 0:
        raise ValueError("B_P equals B_R: no reaction direction, n_TS undefined")

    contrib = (dT * dP) / total
    weight = W / total
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(W > eps * total, dT / dP, np.nan)

    fwd, rev = dP > 0, dP < 0
    return {"n_ts": float(contrib.sum()),
            "contrib": contrib, "weight": weight, "ratio": ratio,
            "dT": dT, "dP": dP,
            "per_atom": contrib.sum(axis=1),
            "forming": float(contrib[fwd].sum()),
            "breaking": float(contrib[rev].sum()),
            "w_forming": float(weight[fwd].sum()),
            "w_breaking": float(weight[rev].sum())}


def decompose_from_ratio(ratio, dP, eps=1e-12):
    """Decomposition from a precomputed ratio matrix and dP = B_P - B_R.

    Non-finite ratio entries are zeroed where their weight is negligible; a
    non-finite entry carrying real weight raises instead of being dropped.
    """
    ratio = np.asarray(ratio, dtype=float).copy()
    dP = _off(dP)
    W = dP * dP
    total = float(W.sum())
    if total <= 0:
        raise ValueError("dP is all zero: no reaction direction")

    bad = ~np.isfinite(ratio)
    heavy = bad & (W > eps * total)
    if heavy.any():
        idx = np.argwhere(np.triu(heavy, 1))
        raise ValueError("non-finite ratio at element(s) carrying real weight: "
                         f"{[tuple(t) for t in idx[:5]]}")
    ratio[bad] = 0.0

    contrib = ratio * W / total
    fwd, rev = dP > 0, dP < 0
    return {"n_ts": float(contrib.sum()), "contrib": contrib,
            "weight": W / total, "ratio": ratio, "dP": dP,
            "per_atom": contrib.sum(axis=1),
            "forming": float(contrib[fwd].sum()),
            "breaking": float(contrib[rev].sum()),
            "w_forming": float((W / total)[fwd].sum()),
            "w_breaking": float((W / total)[rev].sum())}


def bond_table(res, symbols=None, min_weight=1e-3):
    """Per-bond table as a pandas DataFrame.

    Off-diagonal elements appear twice in the matrices, so weight and contrib
    are doubled here: the contrib column sums to n_ts.
    """
    import pandas as pd

    C, W, R = res["contrib"], res["weight"], res["ratio"]
    n = C.shape[0]
    sym = symbols if symbols is not None else [str(i + 1) for i in range(n)]
    iu = np.triu_indices(n, 1)

    df = pd.DataFrame({
        "i": iu[0], "j": iu[1],
        "bond": [f"{sym[a]}{a + 1}-{sym[b]}{b + 1}" for a, b in zip(*iu)],
        "dP": res["dP"][iu], "dT": res["dT"][iu],
        "ratio": R[iu], "weight": 2 * W[iu], "contrib": 2 * C[iu],
    })
    df["kind"] = np.where(df["dP"] > 0, "forming",
                          np.where(df["dP"] < 0, "breaking", "spectator"))
    df = df[df["weight"] >= min_weight]
    return df.sort_values("weight", ascending=False).reset_index(drop=True)


def decompose_many(B_R, B_TS, B_P):
    """Vectorised decomposition over stacks of shape (m, N, N) with a common N.

    Returns n_ts (m,), contrib (m, N, N), weight (m, N, N), forming (m,),
    breaking (m,) and w_forming (m,).
    """
    R3, T3, P3 = (np.asarray(x, dtype=float).copy() for x in (B_R, B_TS, B_P))
    k = R3.shape[1]
    for A in (R3, T3, P3):
        A[:, np.arange(k), np.arange(k)] = 0.0

    dT, dP = T3 - R3, P3 - R3
    W = dP * dP
    total = W.sum(axis=(1, 2))
    if np.any(total <= 0):
        raise ValueError(f"{int((total <= 0).sum())} entries have B_P == B_R")

    contrib = (dT * dP) / total[:, None, None]
    weight = W / total[:, None, None]
    fwd, rev = dP > 0, dP < 0
    return {"n_ts": contrib.sum(axis=(1, 2)),
            "contrib": contrib, "weight": weight,
            "forming": np.where(fwd, contrib, 0).sum(axis=(1, 2)),
            "breaking": np.where(rev, contrib, 0).sum(axis=(1, 2)),
            "w_forming": np.where(fwd, weight, 0).sum(axis=(1, 2))}
