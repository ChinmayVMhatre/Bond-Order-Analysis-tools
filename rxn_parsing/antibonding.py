"""Antibonding channel: negative bond-order matrix elements across R/TS/P."""

import numpy as np

from .decompose import _off


def _mneg(x):
    """Antibonding magnitude max(-x, 0); zero wherever the element is bonding."""
    return np.maximum(-np.asarray(x, dtype=float), 0.0)


def antibonding(B_R, B_TS, B_P, n_ts=None, zero_tol=2e-2, min_anti=2e-2,
                w_frac=1e-2):
    """Quantify antibonding change across a reactant/TS/product triple.

    Returns a dict with:
      a_excess               scalar index; positive means the TS carries more
                             antibonding character than the straight reference
      excess     (N, N)      additive contributions; sum equals a_excess
      m_R/m_TS/m_P (N, N)    antibonding magnitudes per state
      a_R/a_TS/a_P           total antibonding magnitude per state
      crossing   (N, N)      progress at which a sign-changing element passes
                             through zero; NaN otherwise
      crossed_early (N, N)   antibonding at the TS and earlier than the reference
      classes                pair lists per pattern
      patterns               the same patterns as boolean masks
      a_progress             secondary index computed within the antibonding
                             channel; not independent of n_ts
      n_ts                   the value used
    """
    R, T, P = _off(B_R), _off(B_TS), _off(B_P)
    dT, dP = T - R, P - R
    denom = float((dP * dP).sum())
    if denom <= 0:
        raise ValueError("B_P equals B_R: no reaction direction")
    n = float((dT * dP).sum() / denom) if n_ts is None else float(n_ts)

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
    classes = {k: [tuple(t) for t in np.argwhere(np.triu(v, 1))]
               for k, v in patterns.items()}

    dmP, dmT = mP - mR, mT - mR
    dm = float((dmP * dmP).sum())
    a_prog = float((dmT * dmP).sum() / dm) if dm > 0 else np.nan

    return {"n_ts": n, "a_excess": float(excess.sum()), "excess": excess,
            "m_R": mR, "m_TS": mT, "m_P": mP,
            "a_R": float(mR.sum() / scale), "a_TS": float(mT.sum() / scale),
            "a_P": float(mP.sum() / scale),
            "crossing": x0, "crossed_early": crossed_early,
            "classes": classes, "patterns": patterns,
            "a_progress": a_prog, "scale": scale}


def antibonding_table(res, B_R=None, B_TS=None, B_P=None, symbols=None,
                      min_abs=1e-4):
    """Per-bond antibonding table as a pandas DataFrame.

    Values are doubled so the excess column sums to a_excess.
    """
    import pandas as pd

    E = res["excess"]
    n = E.shape[0]
    sym = symbols if symbols is not None else [str(i + 1) for i in range(n)]
    iu = np.triu_indices(n, 1)

    label = np.full((n, n), "", dtype=object)
    for name in ("persistent_antibonding", "loses_antibonding",
                 "becomes_antibonding", "transient_any",
                 "transient_antibonding"):
        label[res["patterns"][name]] = name

    data = {"bond": [f"{sym[a]}{a + 1}-{sym[b]}{b + 1}" for a, b in zip(*iu)],
            "class": label[iu], "excess": 2 * E[iu],
            "m_R": res["m_R"][iu], "m_TS": res["m_TS"][iu],
            "m_P": res["m_P"][iu],
            "crossing": res["crossing"][iu],
            "crossed_early": res["crossed_early"][iu]}
    if B_R is not None:
        data["BO_R"] = _off(B_R)[iu]
        data["BO_TS"] = _off(B_TS)[iu]
        data["BO_P"] = _off(B_P)[iu]

    df = pd.DataFrame(data)
    df = df[(df["excess"].abs() >= min_abs) | (df["class"] != "")]
    return df.sort_values("excess", key=np.abs,
                          ascending=False).reset_index(drop=True)
