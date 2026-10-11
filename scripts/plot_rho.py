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

# Values accepted by mask_from to mean "use every off-diagonal element".
NO_MASK = ("none", "all", "off", "false", "nomask", "no_mask")

# Filled in by build_rho so plot_rho can label the x-axis correctly.
LAST_BUILD = {}


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
    if mask_from is not None or mask_from is False:
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
              mask_min=0.0, verbose=True):
    """
    dicts     : {method: {mol_id: (n,n) array}}
    mask_from : how the comparison element set is chosen. REQUIRED -- it is a
                scientific choice and is deliberately not defaulted, but
                "no mask" is a valid, explicit answer:

                  "<method>"            keep elements where that method is
                                        > mask_min (the usual choice)
                  False / "none" /      keep EVERY off-diagonal element
                  "all" / "off"
                  None                  raises -- you must choose

    mask_min  : threshold for the mask (default 0.0 -> strictly positive).
                Ignored when no mask is used.
    factors   : {method: off-diagonal multiplier}; missing -> 1.0. None -> all 1.0.

    Returns (ids, methods, RHO) with RHO shape (m, k, k), float32, diagonal 1.

    WITHOUT A MASK the vectors keep every atom pair, including the many
    near-zero non-bonded entries and (for Mulliken-type methods) the negative
    ones. Those dominate the ranking and inflate rho -- see the note printed at
    the end of the build. Unmasked rho answers "do the methods agree about the
    whole matrix", masked rho answers "do they agree about the bonds".
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
            "pass mask_from=False (or 'none') to use every off-diagonal "
            "element, or set it once with R.set_defaults(mask_from=...).")

    use_mask = not (mask_from is False
                    or str(mask_from).strip().lower() in NO_MASK)
    if use_mask and mask_from not in dicts:
        raise ValueError(f"mask_from='{mask_from}' not among {methods}; "
                         f"or pass False / 'none' for no mask")

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
        if use_mask:
            mask = vecs[mask_from] > mask_min
        else:
            mask = np.ones_like(next(iter(vecs.values())), dtype=bool)
        npair = int(mask.sum())
        if npair < min_pairs:
            skipped.append((mid, f"only {npair} elements after masking"))
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
    nu = np.asarray(n_used)

    global LAST_BUILD
    LAST_BUILD = {"mask_from": (mask_from if use_mask else None),
                  "use_mask": use_mask, "mask_min": mask_min,
                  "n_used": nu, "m": len(kept), "methods": list(methods)}

    if verbose:
        print(f"  rho array: {RHO.shape} {RHO.dtype} "
              f"({RHO.nbytes/1e3:.1f} KB)"
              + (f"; {len(skipped)} skipped" if skipped else ""))
        if use_mask:
            print(f"  mask: {mask_from} > {mask_min}  -> elements per molecule: "
                  f"min {nu.min()} median {int(np.median(nu))} max {nu.max()}")
        else:
            print(f"  NO MASK: all {int(nu[0]) if nu.size else 0} off-diagonal "
                  f"elements used (min {nu.min()} median "
                  f"{int(np.median(nu))} max {nu.max()})")
            print("  [note] unmasked vectors include the near-zero non-bonded "
                  "entries and any negative ones; these dominate the ranks and "
                  "typically inflate rho. Compare against a masked build before "
                  "drawing conclusions.")
        for s in skipped[:5]:
            print(f"    skipped {s[0]}: {s[1]}")
    return kept, methods, RHO


def mask_label(build=None):
    """Human-readable description of the element set used by the last build."""
    b = build or LAST_BUILD
    if not b:
        return "elements as built"
    if not b.get("use_mask"):
        return "all off-diagonal elements (no mask)"
    thr = b.get("mask_min", 0.0)
    return (f"{b['mask_from']} > {thr:g} elements only" if thr
            else f"{b['mask_from']}-positive elements only")


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
    ax.set_xlabel(xlabel or f"Spearman rank correlation per molecule "
                            f"({mask_label()})")
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
        fig.savefig(save, dpi=180)
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


# ==========================================================================
# Wasserstein distances between the per-pair rho distributions
# ==========================================================================
# Each method pair contributes a DISTRIBUTION of rho over the m molecules, so a
# Wasserstein distance needs a stated reference. Three are supported:
#
#   reference="perfect"      1-D W1 to a point mass at rho = 1, i.e. distance
#                            from perfect agreement. Interpretable and on the
#                            same scale as rho.
#                            HONEST NOTE: for a point mass this reduces exactly
#                            to mean(1 - rho) -- it is a mean gap, not a shape
#                            comparison. Use it when you want "how far from
#                            agreeing", not "how differently shaped".
#   reference=("iao","nbo")  W1 to that pair's own rho distribution. A genuine
#                            EMD between two empirical distributions: it does
#                            compare shape and is not reducible to a mean.
#                            Use it to benchmark every pair against one you
#                            trust (e.g. your tightest QM-QM pair).
#   reference="pooled"       W1 to the pooled distribution of all selected
#                            pairs. Flags atypical pairs.

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
    ax.set_xlabel(xlabel or f"Wasserstein distance of the rho distribution\n"
                            f"from {ref_name}")
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


# ==========================================================================
# distances BETWEEN two rho distributions, + lollipop plot
# ==========================================================================
# Each method pair contributes one distribution of rho over the m molecules.
# This compares two such distributions, e.g. W(rho[a-b], rho[a1-b1]).
#
# Unlike the bond-order spectra case, there is NO scale problem here: rho is
# dimensionless and bounded in [-1, 1], so the distance is already in rho units
# and needs no normalisation.
#
# Wasserstein and KS both IGNORE the molecule-level pairing, even though every
# rho distribution is over the same molecules. metric='paired_mad' uses that
# pairing and answers a different question -- "how far apart are these two
# correlations on the SAME molecule, typically" -- which can be large even when
# the two distributions coincide. Report both if the pairing matters.

_METRICS = ("wasserstein", "ks", "median_diff", "paired_mad")


def pair_distance(methods, RHO, comparisons=None, pairs=None, involving=None,
                  exclude=None, groups=None, group_map=None,
                  metric="wasserstein", verbose=True):
    """
    Distance between the rho distributions of method pairs.

    comparisons : list of ((a, b), (a1, b1)) pair-of-pairs. If None, every
                  combination of the pairs left by the pairs/involving/
                  exclude/groups filters is used.
    metric      : 'wasserstein' (default) | 'ks' | 'median_diff' | 'paired_mad'

    Returns a DataFrame sorted by distance, with the two pair labels, their
    medians, the signed median difference, and n.
    """
    import pandas as pd
    from scipy.stats import wasserstein_distance, ks_2samp

    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}")

    idx = {m: i for i, m in enumerate(methods)}
    labels, _ = pair_groups(methods, group_map)
    pool = select_pairs(methods, pairs, involving, exclude, groups, group_map)

    def vec(p):
        return RHO[:, idx[p[0]], idx[p[1]]]

    if comparisons is None:
        comparisons = list(itertools.combinations(pool, 2))
    else:
        lookup = {_canon(p): p for p in itertools.combinations(methods, 2)}
        norm, unknown = [], []
        for c in comparisons:
            if len(c) != 2:
                raise ValueError(f"comparison {c!r} must be two pairs, "
                                 "e.g. (('a','b'), ('a1','b1'))")
            got = []
            for p in c:
                key = _canon(p)
                if key in lookup:
                    got.append(lookup[key])
                else:
                    unknown.append(tuple(p))
            if len(got) == 2:
                norm.append(tuple(got))
        if unknown:
            raise ValueError(f"pair(s) not available: {unknown}")
        comparisons = norm

    rows = []
    for pa, pb in comparisons:
        va, vb = vec(pa), vec(pb)
        finite = np.isfinite(va) & np.isfinite(vb)
        fa, fb = va[np.isfinite(va)], vb[np.isfinite(vb)]
        if not fa.size or not fb.size:
            continue
        if metric == "wasserstein":
            d = wasserstein_distance(fa, fb)
        elif metric == "ks":
            d = ks_2samp(fa, fb).statistic
        elif metric == "median_diff":
            d = abs(np.median(fa) - np.median(fb))
        else:                                    # paired_mad, uses the pairing
            if finite.sum() == 0:
                continue
            d = float(np.median(np.abs(va[finite] - vb[finite])))
        rows.append({
            "pair_a": f"{pa[0]}–{pa[1]}", "pair_b": f"{pb[0]}–{pb[1]}",
            "label": f"{pa[0]}–{pa[1]}  vs  {pb[0]}–{pb[1]}",
            "group_a": labels[pa], "group_b": labels[pb],
            "same_group": labels[pa] == labels[pb],
            "median_a": float(np.median(fa)), "median_b": float(np.median(fb)),
            "delta_median": float(np.median(fa) - np.median(fb)),
            "distance": float(d), "metric": metric,
            "n": int(min(fa.size, fb.size)),
        })

    if not rows:
        raise ValueError("no comparisons produced a distance")
    df = pd.DataFrame(rows).sort_values("distance", ascending=False)
    if verbose:
        print(f"  {len(df)} comparisons, metric='{metric}'  "
              f"(rho is bounded [-1,1], so no normalisation needed)")
        print(f"  distance: min {df['distance'].min():.4f}  "
              f"median {df['distance'].median():.4f}  "
              f"max {df['distance'].max():.4f}")
    return df.reset_index(drop=True)


def plot_lollipop(table, value="distance", top=None, ax=None, colors=None,
                  color_by="same_group", title=None, xlabel=None, save=None,
                  sort=True, annotate=True, backend=None):
    """
    Lollipop plot of the pair-of-pairs distances returned by pair_distance().

    value    : column to plot (default 'distance')
    top      : keep only the N largest (None = all)
    color_by : 'same_group' (within- vs across-group comparisons),
               'group_a', or None for a single colour
    annotate : print the value at the end of each stem

    Returns (fig, ax).
    """
    if backend:
        import matplotlib
        matplotlib.use(backend)
    import matplotlib.pyplot as plt
    import pandas as pd

    df = table.copy()
    if sort:
        df = df.sort_values(value, ascending=True)
    if top:
        df = df.tail(int(top))
    if df.empty:
        raise ValueError("nothing left to plot")

    if color_by == "same_group":
        keys = df["same_group"].map({True: "within group", False: "across groups"})
    elif color_by in df.columns:
        keys = df[color_by].astype(str)
    else:
        keys = pd.Series(["all"] * len(df), index=df.index)

    uniq = list(dict.fromkeys(keys))
    cmap = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(uniq)}
    if colors:
        cmap.update({k: v for k, v in colors.items() if k in cmap})

    if ax is None:
        fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(df) + 2.2))
    else:
        fig = ax.figure

    y = np.arange(1, len(df) + 1)
    for yi, (_, row), g in zip(y, df.iterrows(), keys):
        c = cmap[g]
        ax.plot([0, row[value]], [yi, yi], color=c, lw=1.8, alpha=.75, zorder=1)
        ax.plot([row[value]], [yi], "o", ms=8, color=c, mec="w", mew=1.1,
                zorder=3)
        if annotate:
            ax.text(row[value] + 0.004 * max(df[value].max(), 1e-6), yi,
                    f"{row[value]:.3f}", va="center", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.set_xlim(0, df[value].max() * 1.18)
    ax.set_ylim(.4, len(df) + .6)
    metric = df["metric"].iloc[0] if "metric" in df else value
    ax.set_xlabel(xlabel or f"{metric} distance between the two rho distributions")
    # ax.set_title(title or f"Distance between rho distributions "
    #                       f"({len(df)} comparisons, metric = {metric})",
    #              fontsize=11)
    if len(uniq) > 1:
        for g in uniq:
            ax.plot([], [], "o", color=cmap[g], ms=8, label=g)
        ax.legend(fontsize=8.5, loc="best", frameon=False)
    ax.grid(axis="x", alpha=.25, lw=.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=180)
        print(f"saved {save}")
    return fig, ax


# ==========================================================================
# Second-order comparison: distances BETWEEN the per-pair rho distributions
# ==========================================================================
# plot_rho shows one rho distribution per METHOD PAIR. The question here is how
# far apart two of those DISTRIBUTIONS are -- e.g. is the spread of
# (iao, nbo) distinguishable from that of (pauling, reaxff)? For P selected
# method pairs there are P(P-1)/2 combinations.
#
# Wasserstein is directly interpretable here: rho is dimensionless and bounded
# in [-1, 1], so a distance of 0.05 means "typically 0.05 rho units apart".
# That is unlike the eigenvalue-spectrum case, where raw Wasserstein was
# dominated by each method's scale and had to be normalised.
#
# CAVEAT WORTH STATING: every distribution is built from the SAME molecules, so
# the two samples are paired. Wasserstein and KS both ignore that pairing and
# treat them as independent samples. 'median_abs_diff' is reported alongside as
# the paired statistic -- the median |rho_A - rho_B| computed molecule by
# molecule. When the two disagree, the paired number is the more informative
# one.

def distribution_distances(methods, RHO, pairs=None, involving=None,
                           exclude=None, groups=None, group_map=None,
                           colors=None, include_ks=True):
    """
    Wasserstein distance between the rho distributions of every combination of
    two selected method pairs.

    Returns a DataFrame (needs pandas) with one row per combination:
      pair_a, pair_b        the two method pairs, as 'a|b' strings
      label_a, label_b      their derived group labels
      relation              'same group' | 'different group'
      wasserstein           W(rho_a, rho_b)
      ks_d, ks_p            two-sample KS (optional)
      median_abs_diff       PAIRED: median |rho_a - rho_b| over molecules
      median_a, median_b    for context
      n                     molecules used
    """
    import pandas as pd
    from scipy.stats import wasserstein_distance, ks_2samp

    idx = {m: i for i, m in enumerate(methods)}
    chosen = select_pairs(methods, pairs, involving, exclude, groups, group_map)
    labels, _ = pair_groups(methods, group_map, colors)

    series = {}
    for p in chosen:
        v = RHO[:, idx[p[0]], idx[p[1]]]
        series[p] = v

    rows = []
    for pa, pb in itertools.combinations(chosen, 2):
        a, b = series[pa], series[pb]
        both = np.isfinite(a) & np.isfinite(b)        # keep the pairing intact
        if both.sum() < 2:
            continue
        av, bv = a[both], b[both]
        row = {"pair_a": f"{pa[0]}|{pa[1]}", "pair_b": f"{pb[0]}|{pb[1]}",
               "label_a": labels[pa], "label_b": labels[pb],
               "relation": "same group" if labels[pa] == labels[pb]
                           else "different group",
               "wasserstein": float(wasserstein_distance(av, bv)),
               "median_abs_diff": float(np.median(np.abs(av - bv))),
               "median_a": float(np.median(av)),
               "median_b": float(np.median(bv)),
               "n": int(both.sum())}
        if include_ks:
            st = ks_2samp(av, bv)
            row["ks_d"] = float(st.statistic)
            row["ks_p"] = float(st.pvalue)
        rows.append(row)

    if not rows:
        raise ValueError("fewer than two selected pairs -- nothing to compare")
    df = pd.DataFrame(rows).sort_values("wasserstein", ascending=False)
    return df.reset_index(drop=True)


def plot_lollipop(df, value="wasserstein", top=None, sort="value",
                  relation_colors=None, save=None, ax=None, title=None,
                  xlabel=None, annotate=True, backend=None):
    """
    Lollipop plot of the distribution-to-distribution distances.

    df       : output of distribution_distances()
    value    : column to plot -- 'wasserstein' (default), 'median_abs_diff',
               or 'ks_d'
    top      : keep only the N largest (useful; P pairs give P(P-1)/2 rows)
    sort     : 'value' (default) or 'name'
    relation_colors : {'same group': c, 'different group': c} override

    Returns (fig, ax).
    """
    if backend:
        import matplotlib
        matplotlib.use(backend)
    import matplotlib.pyplot as plt

    if value not in df.columns:
        raise ValueError(f"value='{value}' not in {list(df.columns)}")
    d = df.copy()
    if sort == "value":
        d = d.sort_values(value, ascending=True)
    elif sort == "name":
        d = d.sort_values(["pair_a", "pair_b"], ascending=False)
    else:
        raise ValueError("sort must be 'value' or 'name'")
    if top:
        d = d.nlargest(top, value).sort_values(value, ascending=True)

    rc = {"same group": PALETTE[0], "different group": PALETTE[1]}
    if relation_colors:
        rc.update(relation_colors)

    lab = [f"{a.replace('|', '–')}   vs   {b.replace('|', '–')}"
           for a, b in zip(d["pair_a"], d["pair_b"])]
    y = np.arange(len(d)) + 1
    vals = d[value].to_numpy()
    cols = [rc.get(r, PALETTE[2]) for r in d["relation"]]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(d) + 2.4))
    else:
        fig = ax.figure

    ax.hlines(y, 0, vals, color=cols, lw=1.6, alpha=.75, zorder=1)
    ax.scatter(vals, y, s=46, color=cols, zorder=3,
               edgecolor="white", linewidth=.9)
    if annotate:
        span = vals.max() - min(0, vals.min())
        for yy, v in zip(y, vals):
            ax.text(v + 0.012 * max(span, 1e-9), yy, f"{v:.3f}",
                    va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=8.5)
    ax.set_xlabel(xlabel or {"wasserstein": "Wasserstein distance between the "
                                            "two rho distributions",
                             "median_abs_diff": "median |Δrho| per molecule "
                                                "(paired)",
                             "ks_d": "KS statistic D"}.get(value, value))
    ax.set_xlim(0, vals.max() * 1.16 if vals.max() > 0 else 1)
    ax.set_ylim(.3, len(d) + .7)
    # ax.set_title(title or f"Distance between rho distributions "
    #                       f"({len(d)} of {len(df)} combinations)", fontsize=11)
    for r in [x for x in ("same group", "different group")
              if x in set(d["relation"])]:
        ax.plot([], [], "o-", color=rc[r], label=r, ms=6)
    if len(set(d["relation"])) > 1:
        ax.legend(fontsize=8.5, loc="best", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.25, lw=.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=180)
        print(f"saved {save}")
    return fig, ax
