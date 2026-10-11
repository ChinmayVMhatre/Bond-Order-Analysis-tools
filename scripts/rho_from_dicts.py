#!/usr/bin/env python3
"""
rho_from_dicts.py — Per-molecule Spearman correlations between bond-order
methods held in dictionaries {mol_id: (n,n) matrix}, plotted as one figure.

PROCEDURE
  1. intersect the dictionary keys so every method contributes the same molecules
  2. symmetrise, zero the diagonal, apply the per-method scale factor
  3. squareform() -> condensed off-diagonal vector, length n(n-1)/2
  4. build a positivity mask from MinPop and apply it to every method
  5. one Spearman matrix per molecule (all pairs at once)
  6. store as a compact (m, k, k) float32 array and plot per-pair distributions

TWO THINGS WORTH KNOWING
------------------------
* The scale factors (x2 MinPop, x4 IAO/NBO) do NOT change Spearman. rho is
  rank-based and a uniform positive multiplier preserves ranks exactly, so
  these factors are no-ops here. They are applied anyway to keep the matrices
  consistent with the Wasserstein pipeline, where scale DOES matter.

* The mask is defined by MinPop, so every pair -- including pauling-reaxff --
  is evaluated only on the pairs MinPop considers positive. That is a real
  asymmetry in the design: it removes the flood of near-zero non-bonded
  entries that would otherwise dominate the ranks and inflate rho, but it also
  conditions the empirical-empirical comparison on a QM method's support.
  Worth stating in any write-up.

NOTHING IS HARDCODED PER METHOD
-------------------------------
There is no built-in list of method names, scale factors or "ab initio"
membership. Supply them at call time, or keep them in a JSON file:

    factors   : {method: multiplier}. Anything absent defaults to 1.0.
                (Irrelevant to rho -- see above -- but kept so the same config
                can drive the scale-sensitive Wasserstein pipeline.)
    group_map : how methods are grouped for colouring/filtering. Accepts
                  {method: label}          e.g. {"iao": "QM", "pauling": "emp"}
                  {label: [methods]}       e.g. {"QM": ["iao","nbo"], ...}
                  a callable method -> label
                None = every method in one group; plots in a single colour
                with no legend.
    mask_from : which method defines the positivity mask. REQUIRED -- it is a
                scientific choice, so there is no default.

Group labels for a PAIR are derived, not enumerated: the two method labels are
sorted and joined, so {"QM","emp"} yields "QM-QM", "QM-emp", "emp-emp"
automatically, and colours are assigned from a colormap.

JUPYTER USE
-----------
    import rho_from_dicts as R

    GROUPS  = {"QM": ["minpop-sto-3g", "minpop-minao", "iao", "nbo"],
               "emp": ["pauling", "reaxff"]}
    FACTORS = {"minpop-sto-3g": 2.0, "minpop-minao": 2.0,
               "iao": 4.0, "nbo": 4.0}

    ids, methods, RHO = R.build_rho(dicts, factors=FACTORS,
                                    mask_from="minpop-sto-3g")

    R.available_pairs(methods, group_map=GROUPS)
    R.plot_rho(methods, RHO, group_map=GROUPS)
    R.plot_rho(methods, RHO, group_map=GROUPS, groups=["QM-QM"])
    R.plot_rho(methods, RHO, pairs=[("iao", "nbo"), ("pauling", "reaxff")])

    # or keep the config in a file
    cfg = R.load_config("methods.json")     # {"factors": {...}, "groups": {...}}
    ids, methods, RHO = R.build_rho(dicts, **cfg["build"])

NOTE: pairs are given as TUPLES, not "a-b" strings, because method names may
legitimately contain hyphens ('minpop-sto-3g'), which makes a hyphen-delimited
string ambiguous.

The plotting backend is NOT forced. In a notebook the figure renders inline and
plot_rho returns (fig, ax) so it can be customised further. Pass save="x.png"
to write a file, or backend="Agg" for a headless script.
"""

import itertools
import json

import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

# No per-method constants live here on purpose: factors and grouping are
# supplied by the caller (or a JSON config). Only the palette is a default,
# and it is indexed by however many pair-groups actually turn up.
PALETTE = ["#2166ac", "#b2182b", "#7f7f7f", "#1b7837", "#762a83",
           "#e08214", "#4393c3", "#d6604d"]
UNGROUPED = "all"

# Session defaults. Empty by design -- set them ONCE in your notebook and every
# call picks them up, instead of passing group_map to each function:
#     R.DEFAULT_GROUP_MAP = {"QM": [...], "emp": [...]}
#     R.DEFAULT_COLORS    = {"QM-QM": "tab:blue", "QM-emp": "tab:red"}
# or in one go:  R.set_defaults(groups=..., factors=..., mask_from=..., colors=...)
DEFAULT_GROUP_MAP = None
DEFAULT_COLORS = None
DEFAULT_FACTORS = None
DEFAULT_MASK_FROM = None


def set_defaults(groups=None, factors=None, mask_from=None, colors=None,
                 verbose=True):
    """Set session defaults so they need not be passed to every call."""
    global DEFAULT_GROUP_MAP, DEFAULT_COLORS, DEFAULT_FACTORS, DEFAULT_MASK_FROM
    if groups is not None:
        DEFAULT_GROUP_MAP = groups
    if colors is not None:
        DEFAULT_COLORS = dict(colors)
    if factors is not None:
        DEFAULT_FACTORS = dict(factors)
    if mask_from is not None:
        DEFAULT_MASK_FROM = mask_from
    if verbose:
        print(f"defaults set: groups={'yes' if DEFAULT_GROUP_MAP else 'no'}, "
              f"colors={'yes' if DEFAULT_COLORS else 'no'}, "
              f"factors={'yes' if DEFAULT_FACTORS else 'no'}, "
              f"mask_from={DEFAULT_MASK_FROM!r}")


def load_config(path):
    """
    Read method configuration from JSON, e.g.

        {"factors": {"minpop-sto-3g": 2.0, "iao": 4.0, "nbo": 4.0},
         "groups":  {"QM": ["minpop-sto-3g", "iao", "nbo"],
                     "emp": ["pauling", "reaxff"]},
         "mask_from": "minpop-sto-3g"}

    Returns the parsed dict plus a ready-to-splat 'build' sub-dict for
    build_rho(**cfg["build"]).
    """
    with open(path) as f:
        cfg = json.load(f)
    cfg["build"] = {k: cfg[k] for k in ("factors", "mask_from") if k in cfg}
    cfg.setdefault("groups", None)
    return cfg


def resolve_groups(group_map, methods):
    """
    Normalise any accepted grouping spec into {method: label}.

    Accepts {method: label}, {label: [methods]}, a callable, or None
    (everything in one group). Methods not mentioned fall back to UNGROUPED.
    """
    if group_map is None:
        return {m: UNGROUPED for m in methods}
    if callable(group_map):
        return {m: str(group_map(m)) for m in methods}
    if not isinstance(group_map, dict):
        raise TypeError("group_map must be a dict, a callable, or None")

    # {label: [methods]} if any value is a list/tuple/set
    if any(isinstance(v, (list, tuple, set)) for v in group_map.values()):
        out = {}
        low = {m.lower(): m for m in methods}
        for label, members in group_map.items():
            for m in members:
                key = low.get(str(m).lower())
                if key is not None:
                    out[key] = str(label)
        unknown = {str(m) for members in group_map.values() for m in members
                   if str(m).lower() not in low}
        if unknown:
            print(f"  [note] group_map lists method(s) not present: "
                  f"{sorted(unknown)}")
    else:                                    # {method: label}
        low = {m.lower(): m for m in methods}
        out = {low[k.lower()]: str(v) for k, v in group_map.items()
               if k.lower() in low}

    missing = [m for m in methods if m not in out]
    if missing:
        print(f"  [note] no group given for {missing} -> '{UNGROUPED}'")
    return {m: out.get(m, UNGROUPED) for m in methods}


def pair_groups(methods, group_map=None, colors=None):
    """
    {pair: derived label} and {label: colour}, both built from the data.

    group_map / colors fall back to DEFAULT_GROUP_MAP / DEFAULT_COLORS when None,
    so a grouping set once with set_defaults() applies everywhere.
    `colors` may be keyed by a PAIR label ('QM-emp') or by a METHOD group
    ('QM'), in which case same-group pairs take that colour.
    """
    if group_map is None:
        group_map = DEFAULT_GROUP_MAP
    if colors is None:
        colors = DEFAULT_COLORS

    member = resolve_groups(group_map, methods)
    labels = {}
    for p in itertools.combinations(methods, 2):
        a, b = sorted((member[p[0]], member[p[1]]))
        labels[p] = a if a == b and group_map is None else f"{a}-{b}"

    uniq = sorted(set(labels.values()))
    if len(uniq) > len(PALETTE):
        print(f"  [note] {len(uniq)} group labels but only {len(PALETTE)} "
              f"palette entries -- colours will repeat. Pass colors={{...}}.")
    colours = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(uniq)}

    if colors:
        for key, c in colors.items():
            if key in colours:                       # pair label, e.g. 'QM-emp'
                colours[key] = c
            else:                                    # method group, e.g. 'QM'
                same = f"{key}-{key}"
                if same in colours:
                    colours[same] = c
        unused = [k for k in colors
                  if k not in uniq and f"{k}-{k}" not in uniq]
        if unused:
            print(f"  [note] colors key(s) {unused} match no group label; "
                  f"labels present: {uniq}")
    return labels, colours


def condensed(M, factor=1.0):
    """Symmetrise, zero the diagonal, scale, and flatten to a condensed vector."""
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)                 # squareform demands exact symmetry
    np.fill_diagonal(M, 0.0)            # and a zero diagonal
    return squareform(M, checks=False) * factor


def build_rho(dicts, mask_from=None, factors=None, min_pairs=5,
              verbose=True):
    """
    dicts     : {method: {mol_id: (n,n) array}}
    mask_from : method defining the positivity mask. REQUIRED -- it is a
                scientific choice and is deliberately not defaulted.
    factors   : {method: off-diagonal multiplier}; missing -> 1.0. None -> all 1.0.

    Returns (ids, methods, RHO) with RHO shape (m, k, k), float32, diagonal 1.
    """
    methods = list(dicts)
    if factors is None:
        factors = DEFAULT_FACTORS
    factors = dict(factors or {})
    if mask_from is None:
        mask_from = DEFAULT_MASK_FROM

    if mask_from is None:
        raise ValueError(
            "mask_from is required -- it selects the method whose positive "
            f"elements define the comparison set. Choose one of {methods}, "
            "or set it once with R.set_defaults(mask_from=...).")
    if mask_from not in dicts:
        raise ValueError(f"mask_from='{mask_from}' not among {methods}")

    no_factor = [m for m in methods if m not in factors]
    if no_factor and verbose:
        print(f"  [note] no factor given for {no_factor} -> using 1.0 "
              f"(harmless for rho; matters for the Wasserstein pipeline)")

    ids = sorted(set.intersection(*(set(d) for d in dicts.values())),
                 key=lambda s: (len(str(s)), str(s)))
    if verbose:
        for m in methods:
            print(f"  {m:14s} {len(dicts[m]):6d} molecules")
        print(f"  common keys: {len(ids)}")

    rho_rows, kept, skipped, n_used = [], [], [], []
    for mid in ids:
        mats = {m: dicts[m][mid] for m in methods}
        shapes = {m: np.shape(M)[0] for m, M in mats.items()}
        if len(set(shapes.values())) != 1:
            skipped.append((mid, f"size mismatch {shapes}"))
            continue

        vecs = {m: condensed(mats[m], factors.get(m, 1.0)) for m in methods}
        mask = vecs[mask_from] > 0                    # positivity mask from MinPop
        npair = int(mask.sum())
        if npair < min_pairs:
            skipped.append((mid, f"only {npair} positive pairs"))
            continue

        X = np.column_stack([vecs[m][mask] for m in methods])
        if (X.std(axis=0) == 0).any():                # constant column -> rho undefined
            skipped.append((mid, "constant column after masking"))
            continue

        R = spearmanr(X).statistic                    # (k, k) in one call
        R = np.atleast_2d(R)
        if R.shape != (len(methods), len(methods)):   # k == 2 returns a scalar
            r = float(np.ravel(R)[0])
            R = np.array([[1.0, r], [r, 1.0]])
        rho_rows.append(R.astype(np.float32))
        kept.append(mid)
        n_used.append(npair)

    if not rho_rows:
        raise RuntimeError("no molecules survived; check mask/min_pairs")
    RHO = np.stack(rho_rows)
    if verbose:
        nu = np.asarray(n_used)
        print(f"  rho array: {RHO.shape} {RHO.dtype} "
              f"({RHO.nbytes/1e3:.1f} KB)"
              + (f"; {len(skipped)} skipped" if skipped else ""))
        print(f"  masked pairs per molecule: min {nu.min()} "
              f"median {int(np.median(nu))} max {nu.max()}")
        for s in skipped[:5]:
            print(f"    skipped {s[0]}: {s[1]}")
    return kept, methods, RHO


def save_rho(path, ids, methods, RHO):
    np.savez_compressed(path, ids=np.array([str(i) for i in ids]),
                        methods=np.array(methods), rho=RHO)
    print(f"saved {path}")


def load_rho(path):
    z = np.load(path)
    return list(z["ids"]), list(z["methods"]), z["rho"]


# ==========================================================================
# pair selection
# ==========================================================================
def group_of(pair, group_map=None, methods=None):
    """Derived group label for one pair, e.g. 'QM-QM' or 'QM-emp'."""
    methods = list(methods or pair)
    labels, _ = pair_groups(methods, group_map)
    key = tuple(pair)
    return labels.get(key) or labels[tuple(reversed(key))]


def available_pairs(methods, group_map=None, colors=None, show=True):
    """List every plottable pair with its derived group label and colour."""
    pairs = list(itertools.combinations(methods, 2))
    labels, colours = pair_groups(methods, group_map, colors)
    if show:
        print(f"{len(pairs)} pairs from {len(methods)} methods:")
        for p in pairs:
            print(f"  {str(p):46s} {labels[p]}")
        print("group label -> colour:")
        for g, c in colours.items():
            print(f"  {g:24s} {c}")
    return pairs


def _canon(p):
    """Order-insensitive, case-insensitive key for a pair."""
    if isinstance(p, str):
        raise TypeError(
            f"pair {p!r} given as a string. Use a tuple, e.g. ('iao','nbo') -- "
            "method names may contain hyphens, so 'a-b' strings are ambiguous.")
    a, b = p
    return tuple(sorted((str(a).lower(), str(b).lower())))


def select_pairs(methods, pairs=None, involving=None, exclude=None,
                 groups=None, group_map=None, colors=None):
    """
    Choose which method pairs to plot. Filters combine (AND); all optional:

      pairs     : explicit list of (a, b) tuples; order and case do not matter
      involving : method name or list -- keep pairs touching any of them
      exclude   : list of (a, b) tuples to drop
      groups    : keep only these labels, e.g. ['QM-QM', 'QM-empirical']

    Returns pairs in `methods` order. Raises with the available list if an
    explicit pair is not present.
    """
    all_pairs = list(itertools.combinations(methods, 2))
    lookup = {_canon(p): p for p in all_pairs}
    out = all_pairs

    if pairs is not None:
        chosen, unknown = set(), []
        for p in pairs:
            key = _canon(p)
            if key in lookup:
                chosen.add(lookup[key])
            else:
                unknown.append(tuple(p))
        if unknown:
            raise ValueError(
                f"pair(s) not available: {unknown}\n"
                f"available: {[tuple(p) for p in all_pairs]}")
        out = [p for p in all_pairs if p in chosen]

    if involving is not None:
        want = {involving.lower()} if isinstance(involving, str) \
            else {str(x).lower() for x in involving}
        unknown = want - {m.lower() for m in methods}
        if unknown:
            raise ValueError(f"involving: unknown method(s) {sorted(unknown)}; "
                             f"methods are {methods}")
        out = [p for p in out if {p[0].lower(), p[1].lower()} & want]

    if groups is not None:
        labels, _ = pair_groups(methods, group_map)
        keep = {groups} if isinstance(groups, str) else set(groups)
        valid = sorted(set(labels.values()))
        bad = keep - set(valid)
        if bad:
            raise ValueError(f"unknown group(s) {sorted(bad)}; present: {valid}")
        out = [p for p in out if labels[p] in keep]

    if exclude:
        drop = {_canon(p) for p in exclude}
        out = [p for p in out if _canon(p) not in drop]

    if not out:
        raise ValueError("no pairs left after filtering")
    return out


# ==========================================================================
# plot
# ==========================================================================
def plot_rho(methods, RHO, pairs=None, involving=None, exclude=None,
             groups=None, group_map=None, colors=None, sort="median",
             save=None, ax=None, title=None, xlabel=None, show_stats=True,
             backend=None, seed=0):
    """
    One horizontal violin per method pair, sorted by median rho.

    Pair selection : pairs / involving / exclude / groups (see select_pairs)
    group_map      : method grouping for colour/legend/filtering; see
                     resolve_groups. None falls back to DEFAULT_GROUP_MAP, and
                     if that is unset every pair shares one colour.
    colors         : {label: colour} override, keyed by pair label ('QM-emp')
                     or method group ('QM'). Falls back to DEFAULT_COLORS.
    sort           : 'median' (default), 'iqr', 'name', or 'given'
    save           : path to write a PNG; None = do not write (notebook-friendly)
    ax             : plot into an existing Axes instead of creating a figure
    backend        : e.g. 'Agg' for headless scripts. NOT set by default, so
                     notebook inline rendering is preserved.

    Returns (fig, ax).
    """
    if backend:
        import matplotlib
        matplotlib.use(backend)
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx = {m: i for i, m in enumerate(methods)}
    chosen = select_pairs(methods, pairs, involving, exclude, groups, group_map)
    labels, colours = pair_groups(methods, group_map, colors)

    # The most common cause of "everything is one colour": no grouping reached
    # this call, so every pair landed in the single UNGROUPED bucket.
    if len(set(labels.values())) == 1 and (group_map or DEFAULT_GROUP_MAP) is None:
        print("  [note] all pairs share one colour because no grouping was "
              "given. Pass group_map={'QM': [...], 'emp': [...]} here, or set "
              "it once with R.set_defaults(groups=...).")

    data = {p: RHO[:, idx[p[0]], idx[p[1]]] for p in chosen}
    data = {p: v[np.isfinite(v)] for p, v in data.items()}
    data = {p: v for p, v in data.items() if v.size}
    if not data:
        raise ValueError("selected pairs contain no finite rho values")

    if sort == "median":
        order = sorted(data, key=lambda p: np.median(data[p]))
    elif sort == "iqr":
        order = sorted(data, key=lambda p: -np.subtract(
            *np.percentile(data[p], [75, 25])))
    elif sort == "name":
        order = sorted(data, key=lambda p: (p[0].lower(), p[1].lower()))
    elif sort == "given":
        order = [p for p in chosen if p in data]
    else:
        raise ValueError("sort must be 'median', 'iqr', 'name' or 'given'")

    if ax is None:
        fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(order) + 2.6))
    else:
        fig = ax.figure

    for y, p in enumerate(order, 1):
        v, c = data[p], colours[labels[p]]
        ax.scatter(v, y + rng.uniform(-.16, .16, len(v)), s=3, alpha=.18,
                   color=c, zorder=1)
        if v.size > 1 and v.std() > 0:              # violinplot needs spread
            for b in ax.violinplot([v], positions=[y], vert=False, widths=.8,
                                   showextrema=False)["bodies"]:
                b.set_facecolor(c); b.set_alpha(.45)
                b.set_edgecolor(c); b.set_linewidth(1.2)
        q1, md, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([q1, q3], [y, y], color="k", lw=2.2, zorder=3,
                solid_capstyle="butt")
        ax.plot([md], [y], "o", ms=5.5, color="w", mec="k", mew=1.2, zorder=4)
        ax.text(1.008, y, f"{md:.3f}", va="center", fontsize=8.5)

    ax.axvline(1.0, color="k", lw=.8, ls=":", alpha=.6)
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([f"{a} – {b}" for a, b in order], fontsize=9.5)
    ax.set_xlabel(xlabel or "Spearman rank correlation per molecule "
                            "(MinPop-positive pairs only)")
    lo = min(v.min() for v in data.values())
    ax.set_xlim(max(-1.02, lo - .06), 1.06)
    ax.set_ylim(.4, len(order) + .6)
    n_all = len(list(itertools.combinations(methods, 2)))
    # ax.set_title(title or f"Method agreement across m = {len(RHO)} molecules\n"
    #              f"{len(order)} of {n_all} pairs (median ● , IQR ▬)",
    #              fontsize=11)
    shown = sorted({labels[p] for p in order})
    if len(shown) > 1:
        for g in shown:
            ax.plot([], [], lw=6, color=colours[g], alpha=.55, label=g)
        ax.legend(fontsize=8.5, loc="best", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=400)
        print(f"saved {save}")

    if show_stats:
        print(f"\n{'pair':40s} {'median':>7s} {'IQR':>7s} {'min':>7s} {'n':>6s}")
        for p in reversed(order):
            v = data[p]
            q1, q3 = np.percentile(v, [25, 75])
            print(f"{p[0]+' - '+p[1]:40s} {np.median(v):7.3f} {q3-q1:7.3f} "
                  f"{v.min():7.3f} {len(v):6d}")
    return fig, ax


def pair_table(methods, RHO, pairs=None, involving=None, exclude=None,
               groups=None, group_map=None, colors=None):
    """Summary statistics for the selected pairs as a DataFrame (needs pandas)."""
    import pandas as pd
    idx = {m: i for i, m in enumerate(methods)}
    chosen = select_pairs(methods, pairs, involving, exclude, groups, group_map)
    labels, _ = pair_groups(methods, group_map, colors)
    rows = []
    for p in chosen:
        v = RHO[:, idx[p[0]], idx[p[1]]]
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        q1, q3 = np.percentile(v, [25, 75])
        rows.append({"method_a": p[0], "method_b": p[1],
                     "group": labels[p], "n": len(v),
                     "median": np.median(v), "iqr": q3 - q1,
                     "min": v.min(), "max": v.max(),
                     "frac_below_0.5": float((v < 0.5).mean())})
    return pd.DataFrame(rows).sort_values("median", ascending=False)


def long_table(ids, methods, RHO, pairs=None, involving=None, exclude=None,
               groups=None, group_map=None):
    """Per-molecule long form for the selected pairs (needs pandas)."""
    import pandas as pd
    idx = {m: i for i, m in enumerate(methods)}
    chosen = select_pairs(methods, pairs, involving, exclude, groups, group_map)
    labels, _ = pair_groups(methods, group_map, colors)
    rows = []
    for p in chosen:
        col = RHO[:, idx[p[0]], idx[p[1]]]
        for mid, r in zip(ids, col):
            rows.append({"mol_id": mid, "method_a": p[0], "method_b": p[1],
                         "group": labels[p], "rho": float(r)})
    return pd.DataFrame(rows)

def wasserstein_pairs(methods, RHO, reference="perfect", pairs=None,
                      involving=None, exclude=None, groups=None,
                      group_map=None, colors=None):
    """
    W1 distance from `reference` for each selected method pair.
    Returns a list of dicts (pair, label, w, median, n), sorted by w.
    """
    from scipy.stats import wasserstein_distance
 
    idx = {m: i for i, m in enumerate(methods)}
    chosen = select_pairs(methods, pairs, involving, exclude, groups, group_map)
    labels, _ = pair_groups(methods, group_map, colors)
 
    data = {}
    for p in chosen:
        v = RHO[:, idx[p[0]], idx[p[1]]]
        v = v[np.isfinite(v)]
        if v.size:
            data[p] = v
    if not data:
        raise ValueError("no finite rho values in the selected pairs")
 
    if isinstance(reference, str) and reference == "perfect":
        ref_vec, ref_name = np.array([1.0]), "perfect agreement (rho = 1)"
    elif isinstance(reference, str) and reference == "pooled":
        ref_vec = np.concatenate(list(data.values()))
        ref_name = "pooled distribution of the selected pairs"
    elif isinstance(reference, (int, float)):
        ref_vec, ref_name = np.array([float(reference)]), f"rho = {float(reference)}"
    else:                                              # a pair tuple
        key = _canon(reference)
        match = [p for p in itertools.combinations(methods, 2)
                 if _canon(p) == key]
        if not match:
            raise ValueError(f"reference pair {tuple(reference)} not available")
        rp = match[0]
        col = RHO[:, idx[rp[0]], idx[rp[1]]]
        ref_vec = col[np.isfinite(col)]
        ref_name = f"{rp[0]} – {rp[1]}"
        if not ref_vec.size:
            raise ValueError(f"reference pair {rp} has no finite values")
 
    rows = []
    for p, v in data.items():
        rows.append({"pair": p, "label": labels[p],
                     "w": float(wasserstein_distance(v, ref_vec)),
                     "median": float(np.median(v)), "n": int(v.size)})
    rows.sort(key=lambda r: r["w"])
    for r in rows:
        r["reference"] = ref_name
    return rows
 
 
def wasserstein_matrix(methods, RHO, pairs=None, involving=None, exclude=None,
                       groups=None, group_map=None):
    """
    Full P x P matrix of W1 distances BETWEEN the per-pair rho distributions.
    Returns (pair_list, W). Use when you want every comparison rather than
    distances to one reference; P grows as k(k-1)/2 so this gets large fast.
    """
    from scipy.stats import wasserstein_distance
 
    idx = {m: i for i, m in enumerate(methods)}
    chosen = select_pairs(methods, pairs, involving, exclude, groups, group_map)
    vecs = []
    keep = []
    for p in chosen:
        v = RHO[:, idx[p[0]], idx[p[1]]]
        v = v[np.isfinite(v)]
        if v.size:
            vecs.append(v)
            keep.append(p)
    P = len(keep)
    W = np.zeros((P, P))
    for i in range(P):
        for j in range(i + 1, P):
            W[i, j] = W[j, i] = wasserstein_distance(vecs[i], vecs[j])
    return keep, W
 
def plot_lollipop(methods, RHO, reference="perfect", pairs=None, involving=None,
                  exclude=None, groups=None, group_map=None, colors=None,
                  ascending=True, save=None, ax=None, title=None, xlabel=None,
                  show_stats=True, annotate=True, backend=None):
    """
    Lollipop plot of the per-pair Wasserstein distances (see wasserstein_pairs).
 
    Stems run from 0 to W, one row per method pair, coloured by group and
    sorted by distance. ascending=True puts the closest pair at the top.
 
    Returns (fig, ax).
    """
    if backend:
        import matplotlib
        matplotlib.use(backend)
    import matplotlib.pyplot as plt
 
    rows = wasserstein_pairs(methods, RHO, reference, pairs, involving,
                             exclude, groups, group_map, colors)
    _, colours = pair_groups(methods, group_map, colors)
    if not ascending:
        rows = rows[::-1]
    ref_name = rows[0]["reference"]
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.0, 0.46 * len(rows) + 2.4))
    else:
        fig = ax.figure
 
    ys = np.arange(len(rows), 0, -1)          # first row at the top
    for y, r in zip(ys, rows):
        c = colours[r["label"]]
        ax.hlines(y, 0, r["w"], color=c, lw=2.0, alpha=.75, zorder=2)
        ax.plot([r["w"]], [y], "o", ms=8, color=c, mec="k", mew=.8, zorder=3)
        if annotate:
            ax.text(r["w"], y, f"  {r['w']:.3f}", va="center", fontsize=8.5)
 
    ax.axvline(0, color="k", lw=.8, alpha=.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{p['pair'][0]} – {p['pair'][1]}" for p in rows],
                       fontsize=9.5)
    ax.set_xlabel(xlabel or f"Wasserstein distance of the rho distribution")
    hi = max(r["w"] for r in rows)
    ax.set_xlim(0, hi * 1.22 if hi > 0 else 1)
    ax.set_ylim(0.4, len(rows) + 0.6)
    # ax.set_title(title or f"Divergence of per-pair agreement, m = {len(RHO)} "
    #                       f"molecules\n{len(rows)} pairs, reference: {ref_name}",
    #              fontsize=11)
    shown = sorted({r["label"] for r in rows})
    if len(shown) > 1:
        for g in shown:
            ax.plot([], [], "o", ms=8, color=colours[g], mec="k", mew=.8, label=g)
        ax.legend(fontsize=8.5, loc="best", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.25, lw=.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
 
    if save:
        fig.savefig(save, dpi=180)
        print(f"saved {save}")
 
    if show_stats:
        print(f"reference: {ref_name}")
        print(f"\n{'pair':40s} {'W':>8s} {'median rho':>11s} {'group':>20s}")
        for r in rows:
            print(f"{r['pair'][0]+' - '+r['pair'][1]:40s} {r['w']:8.4f} "
                  f"{r['median']:11.3f} {r['label']:>20s}")
    return fig, ax
