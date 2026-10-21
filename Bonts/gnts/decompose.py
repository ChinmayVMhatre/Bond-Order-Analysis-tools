"""n_TS decomposition into per-bond bonding and antibonding contributions."""

import numpy as np


def _off(M):
    M = np.asarray(M, dtype=float).copy()
    np.fill_diagonal(M, 0.0)
    return M


def _mneg(x):
    return np.maximum(-np.asarray(x, dtype=float), 0.0)


# --------------------------------------------------------------------------
# bonding channel
# --------------------------------------------------------------------------
def bonding(BO_R, BO_TS, BO_P, eps=1e-12):
    """
    Decompose the global progress n_TS into additive per-element contributions.

    Returns dict with:
      n_ts                  global progress
      contrib   (N,N)       additive contributions; sum == n_ts
      weight    (N,N)       weight shares; sum == 1
      ratio     (N,N)       per-element ratio, NaN where the weight is negligible
      dT, dP    (N,N)       TS and product difference matrices
      per_atom  (N,)        row sums of contrib
      forming / breaking    contributions of dP > 0 and dP < 0 elements
      w_forming / w_breaking  their weight shares
    """
    dT, dP = _off(BO_TS) - _off(BO_R), _off(BO_P) - _off(BO_R)
    W = dP * dP
    tot = float(W.sum())
    if tot <= 0:
        raise ValueError("B_P equals B_R: no reaction direction")

    contrib = (dT * dP) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(W > eps * tot, dT / dP, np.nan)

    fwd, rev = dP > 0, dP < 0
    return {"n_ts": float(contrib.sum()), "contrib": contrib, "weight": W / tot,
            "ratio": ratio, "dT": dT, "dP": dP,
            "per_atom": contrib.sum(axis=1),
            "forming": float(contrib[fwd].sum()),
            "breaking": float(contrib[rev].sum()),
            "w_forming": float((W / tot)[fwd].sum()),
            "w_breaking": float((W / tot)[rev].sum())}


def bond_table(res, labels=None,sorting_by = 'contrib', min_weight=1e-3):
    """
    Per-unique-pair bonding table (pandas DataFrame). Values are doubled so the
    'contrib' column sums to n_ts.
    """
    import pandas as pd
    C, W, Rt = res["contrib"], res["weight"], res["ratio"]
    n = C.shape[0]
    lab = labels or [str(i + 1) for i in range(n)]
    iu = np.triu_indices(n, 1)
    df = pd.DataFrame({
        "bond": [f"{lab[a]}-{lab[b]}" for a, b in zip(*iu)],
        "dP": res["dP"][iu], "dT": res["dT"][iu],
        "ratio": Rt[iu], "weight": 2 * W[iu], "contrib": 2 * C[iu],
    })
    df["kind"] = np.where(df["dP"] > 0, "forming",
                          np.where(df["dP"] < 0, "breaking", "spectator"))
    df = df[df["weight"] >= min_weight]
    if sorting_by is not None:
        df = df.sort_values(sorting_by, ascending=False).reset_index(drop=True)
        
    return df.sort_values(sorting_by, ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# antibonding channel
# --------------------------------------------------------------------------
def antibonding(BO_R, BO_TS, BO_P, n_ts=None, zero_tol=2e-2, min_anti=2e-2,
                w_frac=1e-2):
    """
    Quantify antibonding (negative) matrix elements across the triple.

    Returns dict with:
      a_excess              scalar index; > 0 means more antibonding character
                            at the TS than the straight R->P reference gives
      excess    (N,N)       additive contributions; sum == a_excess
      m_R/m_TS/m_P (N,N)    antibonding magnitudes per state
      a_R/a_TS/a_P          their totals, on the sum|dP| scale
      crossing  (N,N)       interpolated zero-crossing progress; NaN if no
                            sign change
      crossed_early (N,N)   antibonding at the TS and earlier than the reference
      classes               pair lists per pattern
      patterns              the same as boolean masks
      a_progress            secondary index inside the antibonding channel
      n_ts                  value used for the reference
    """
    R, T, P = _off(BO_R), _off(BO_TS), _off(BO_P)
    dT, dP = T - R, P - R
    denom_w = float((dP * dP).sum())
    if denom_w <= 0:
        raise ValueError("B_P equals B_R: no reaction direction")
    n = float((dT * dP).sum() / denom_w) if n_ts is None else float(n_ts)

    scale = float(np.abs(dP).sum())
    lin = R + n * dP
    mR, mT, mP, mL = _mneg(R), _mneg(T), _mneg(P), _mneg(lin)
    excess = (mT - mL) / scale

    with np.errstate(divide="ignore", invalid="ignore"):
        x0 = np.where(np.sign(R) * np.sign(P) < 0, -R / dP, np.nan)
    crossed_early = (mT > min_anti) & np.isfinite(x0) & (n > x0) & (mL <= mT)

    W = dP * dP
    quiet = W <= w_frac * max(W.max(), 1e-300)
    anti_R, anti_P, anti_T = mR > min_anti, mP > min_anti, mT > min_anti
    near0_R, near0_P = np.abs(R) <= zero_tol, np.abs(P) <= zero_tol

    patterns = {
        "becomes_antibonding": (~anti_R) & anti_P,
        "loses_antibonding": anti_R & (~anti_P),
        "persistent_antibonding": anti_R & anti_P,
        "transient_antibonding": near0_R & near0_P & anti_T & quiet,
        "transient_any": (~anti_R) & (~anti_P) & anti_T,
    }
    classes = {k: [tuple(int(x) for x in t) for t in np.argwhere(np.triu(v, 1))]
               for k, v in patterns.items()}

    dmP, dmT = mP - mR, mT - mR
    dm_denom = float((dmP * dmP).sum())
    a_prog = float((dmT * dmP).sum() / dm_denom) if dm_denom > 0 else np.nan

    return {"n_ts": n, "a_excess": float(excess.sum()), "excess": excess,
            "m_R": mR, "m_TS": mT, "m_P": mP,
            "a_R": float(mR.sum() / scale), "a_TS": float(mT.sum() / scale),
            "a_P": float(mP.sum() / scale),
            "crossing": x0, "crossed_early": crossed_early,
            "classes": classes, "patterns": patterns,
            "a_progress": a_prog, "scale": scale}


def antibonding_table(res, BO_R=None, BO_TS=None, BO_P=None, labels=None,
                      min_abs=1e-4):
    """
    Per-unique-pair antibonding table (pandas DataFrame). Values are doubled so
    the 'excess' column sums to a_excess.
    """
    import pandas as pd
    E = res["excess"]
    n = E.shape[0]
    lab = labels or [str(i + 1) for i in range(n)]
    iu = np.triu_indices(n, 1)

    label = np.full((n, n), "", dtype=object)
    for name in ("persistent_antibonding", "loses_antibonding",
                 "becomes_antibonding", "transient_any",
                 "transient_antibonding"):
        label[res["patterns"][name]] = name

    d = {"bond": [f"{lab[a]}-{lab[b]}" for a, b in zip(*iu)],
         "class": label[iu], "excess": 2 * E[iu],
         "m_R": res["m_R"][iu], "m_TS": res["m_TS"][iu], "m_P": res["m_P"][iu],
         "crossing": res["crossing"][iu],
         "crossed_early": res["crossed_early"][iu]}
    if BO_R is not None:
        d["BO_R"] = _off(BO_R)[iu]
        d["BO_TS"] = _off(BO_TS)[iu]
        d["BO_P"] = _off(BO_P)[iu]
    df = pd.DataFrame(d)
    df = df[(df["excess"].abs() >= min_abs) | (df["class"] != "")]
    return df.sort_values("excess", key=np.abs,
                          ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# convenience
# --------------------------------------------------------------------------
def analyse_reaction(path_R, path_TS, path_P, factor=2.0, labels=None, **kw):
    """
    Parse three Gaussian outputs and run both channels.

    Returns dict with keys: labels, BO_R, BO_TS, BO_P, bonding, anti.
    """
    from .gaussian import load_bond_orders

    BO_R, lab, _ = load_bond_orders(path_R, factor=factor)
    BO_TS, _, _ = load_bond_orders(path_TS, factor=factor)
    BO_P, _, _ = load_bond_orders(path_P, factor=factor)

    shapes = {BO_R.shape[0], BO_TS.shape[0], BO_P.shape[0]}
    if len(shapes) != 1:
        raise ValueError(f"atom-count mismatch across R/TS/P: {sorted(shapes)}")

    lab = labels or lab
    dec = decompose(BO_R, BO_TS, BO_P)
    anti = antibonding(BO_R, BO_TS, BO_P, n_ts=dec["n_ts"], **kw)
    return {"labels": lab, "BO_R": BO_R, "BO_TS": BO_TS, "BO_P": BO_P,
            "bonding": dec, "anti": anti}


def summary(res):
    """Print the scalar results of analyse_reaction()."""
    d, a = res["bonding"], res["anti"]
    print(f"atoms            : {len(res['labels'])}")
    print(f"n_TS             : {d['n_ts']:.4f}")
    print(f"  forming        : {d['forming']:.4f}  (weight {d['w_forming']:.3f})")
    print(f"  breaking       : {d['breaking']:.4f}  (weight {d['w_breaking']:.3f})")
    print(f"a_excess         : {a['a_excess']:+.4f}")
    print(f"  antibonding R  : {a['a_R']:.4f}")
    print(f"  antibonding TS : {a['a_TS']:.4f}")
    print(f"  antibonding P  : {a['a_P']:.4f}")
    hits = {k: v for k, v in a["classes"].items() if v}
    if hits:
        print("antibonding classes:")
        for k, v in hits.items():
            names = [f"{res['labels'][i]}-{res['labels'][j]}" for i, j in v]
            print(f"  {k:24s} {names}")
