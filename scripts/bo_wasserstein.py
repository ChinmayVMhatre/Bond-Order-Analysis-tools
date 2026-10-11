#!/usr/bin/env python3
"""
bo_wasserstein.py — Per-molecule Wasserstein distances between the bond-order
eigenvalue spectra of MinPop / Pauling / ReaxFF.

PIPELINE
  1. For selected molecule IDs, read the atom-atom matrix from each method's
     HDF5 file and convert it to a bond-order matrix:
        off-diagonal *= offdiag_factor   (2.0 for MinPop, 1.0 for the others)
        diagonal      = 0
  2. Eigenvalues of each matrix (real part only).
  3. Wasserstein distance between the spectra for each METHOD PAIR, computed
     inside the same loop so no matrices are retained.
  4. Plot the three pairwise series over the m molecules.

TWO SPEC POINTS, RESOLVED HERE
------------------------------
* Wasserstein compares TWO distributions, so a single method has no distance of
  its own. Three methods give three PAIRS -> three series of length m
  (minpop-pauling, minpop-reaxff, pauling-reaxff).

* The *2 applies to MinPop ONLY. MBS condensed-to-atoms off-diagonals are shared
  POPULATIONS, so *2 turns them into bond orders. Pauling and ReaxFF are already
  bond orders; doubling them would inject a bogus factor of 2 which, because
  Wasserstein is scale-sensitive, would masquerade as a method difference. Set
  per source via `offdiag_factor`.

SCALE CAVEAT (this is weakness #2 in the analysis plan)
-------------------------------------------------------
Wasserstein distance is NOT scale-invariant. Methods report bond orders on
different scales, so a large raw distance can reflect nothing more than a
uniform rescaling -- which carries no information. The script therefore computes
BOTH:
    raw        : distances between spectra as-is
    normalised : each spectrum divided by its own RMS first, so a pure rescaling
                 gives distance ~0 and only SHAPE differences survive
Report both. Divergence between the two panels is the interesting result.

Zeroing the diagonal forces trace = 0, so every spectrum is automatically
centred -- which is exactly the A1 fix the plan asks for.
"""

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
from scipy.stats import wasserstein_distance, ks_2samp


# ==========================================================================
# CONFIG — edit paths / dataset hints here
# ==========================================================================
SOURCES = {
    "minpop":  {"path": "qm9gr.h5",     "offdiag_factor": 2.0, "hint": "mbs"},
    "pauling": {"path": "qm9_pauling.h5", "offdiag_factor": 1.0, "hint": "bond_order"},
    "reaxff":  {"path": "qm9_reaxff_bonds_merged_rev.h5",  "offdiag_factor": 1.0, "hint": "reaxff"},
}
PAIRS = [("minpop", "pauling"), ("minpop", "reaxff"), ("pauling", "reaxff")]
COLORS = {"minpop-pauling": "#1f77b4", "minpop-reaxff": "#d62728",
          "pauling-reaxff": "#2ca02c"}


# ==========================================================================
# ID normalisation
# ==========================================================================
def norm_id(v):
    """Trailing integer of any QM9 id form: 'dsgdb9nsd_000200', '000200', 200."""
    import re
    if isinstance(v, (int, np.integer)):
        return int(v)
    m = re.findall(r"(\d+)", str(v))
    return int(m[-1]) if m else None


# ==========================================================================
# readers — one class, three layouts auto-detected
# ==========================================================================
class MatrixSource:
    """
    Lazily reads atom-atom matrices from one HDF5 file. Supports:
      SPARSE FLAT   bond_i / bond_j / bond_order + bond_ptr (+ atom_ptr, n_atoms)
      DENSE FLAT    <matrix dataset> + <ptr dataset>        (full N*N or packed triangle)
      HIERARCHICAL  /molecules/<id>/.../<dataset matching hint>
    """

    def __init__(self, name, path, offdiag_factor=1.0, hint="",
                 layout=None, matrix_key=None, ptr_key=None, bo_key=None):
        self.name, self.path = name, path
        self.factor, self.hint = float(offdiag_factor), hint.lower()
        self.layout_override = layout
        self.matrix_key_override = matrix_key
        self.ptr_key_override = ptr_key
        self.bo_key_override = bo_key
        self._mat_key = self._ptr_key = None
        self._bi = self._bj = self._pairs = self._bo = None
        self.f = h5py.File(path, "r")
        self.layout = self._detect()
        if bo_key:
            self._bo = bo_key
        self.index = self._build_index()

    # ---------- layout ----------
    def _keys(self):
        return list(self.f.keys())

    def _len(self, k):
        try:
            return int(self.f[k].shape[0])
        except Exception:
            return -1

    def _pick(self, names):
        """First existing key from a list of candidate names (case-insensitive)."""
        low = {k.lower(): k for k in self.f.keys()}
        for n in names:
            if n.lower() in low:
                return low[n.lower()]
        return None

    def _detect(self):
        f = self.f
        if self.layout_override:
            return self.layout_override
        if "molecules" in f:
            return "hier"

        n_mol = self._len("mol_id") if "mol_id" in f else -1

        # ---- SPARSE: bond index arrays + values, names allowed to vary ----
        self._bi = self._pick(["bond_i", "bond_idx_i", "bond_index_i", "i", "atom_i"])
        self._bj = self._pick(["bond_j", "bond_idx_j", "bond_index_j", "j", "atom_j"])
        self._pairs = self._pick(["bond_idx_pairs", "bond_indices", "bond_pairs"])
        self._bo = self._pick(["bond_order", "bond_orders", "bo", "bond_bo",
                               "reaxff_bond_order", "reaxff_bond_orders",
                               "pauling_bond_order", "order", "orders"])
        if self._bo is not None and (self._pairs is not None or
                                     (self._bi is not None and self._bj is not None)):
            return "sparse"

        # ---- DENSE: a concatenated matrix dataset ----
        # Candidates must LOOK like matrix data: a 2-D dataset, or a 1-D dataset
        # far longer than the molecule count. This is what stops a per-molecule
        # scalar column such as 'reaxff_energy' (length == n_mol) being mistaken
        # for the matrix, which is the bug that produced
        #   "found matrix 'reaxff_energy' but no pointer dataset".
        if self.matrix_key_override:
            self._mat_key = self.matrix_key_override
        else:
            cands = []
            for k in f.keys():
                if k in ("mol_id", "n_atoms", "n_bonds") or "ptr" in k.lower():
                    continue
                if not (self.hint in k.lower() or "matrix" in k.lower()
                        or "bond_order" in k.lower()):
                    continue
                d = f[k]
                if getattr(d, "ndim", 1) >= 2:
                    cands.append(k)
                elif n_mol > 0 and self._len(k) > n_mol:
                    cands.append(k)
                elif n_mol <= 0:
                    cands.append(k)
            # prefer a key that actually says 'matrix'
            cands.sort(key=lambda k: ("matrix" not in k.lower(), len(k)))
            self._mat_key = cands[0] if cands else None

        if self._mat_key is not None:
            self._ptr_key = (self.ptr_key_override
                             or self._pick([f"{self._mat_key}_ptr",
                                            f"{self._mat_key}_offsets",
                                            "matrix_ptr", "matrix_offsets",
                                            "mat_ptr", "ptr"])
                             or next((k for k in f.keys()
                                      if ("ptr" in k.lower() or "offset" in k.lower())
                                      and not k.lower().startswith("atom")), None))
            if self._ptr_key is None and "n_atoms" in f:
                self._ptr_key = "__derive__"      # build offsets from n_atoms
            if self._ptr_key is not None:
                return "dense"

        self._explain_failure()

    def _explain_failure(self):
        lines = [f"{self.path}: could not identify the matrix layout.", "",
                 "root datasets:"]
        for k in self.f.keys():
            d = self.f[k]
            shape = getattr(d, "shape", "(group)")
            dt = getattr(d, "dtype", "")
            lines.append(f"  {k:32s} {str(shape):>16s}  {dt}")
        lines += ["", "The reader needs ONE of:",
                  "  sparse : a bond-order value array plus bond index arrays",
                  "           (bond_i/bond_j or bond_idx_pairs) and either a",
                  "           *_ptr dataset or n_bonds to build offsets from",
                  "  dense  : a concatenated matrix dataset plus a *_ptr dataset",
                  "           (or n_atoms, from which offsets are derived)",
                  "  hier   : a /molecules/<id>/ group tree", "",
                  "Set them explicitly in SOURCES, e.g.",
                  "  'reaxff': {'path': ..., 'offdiag_factor': 1.0,",
                  "             'layout': 'sparse', 'bo_key': 'bond_orders'}",
                  "or run:  python3 describe_h5.py <file>"]
        raise SystemExit("\n".join(lines))

    def _build_index(self):
        f = self.f
        if self.layout == "hier":
            return {norm_id(k): k for k in f["molecules"].keys()}
        ids = f["mol_id"].asstr()[:]
        return {norm_id(s): i for i, s in enumerate(ids)}

    def _offsets(self, count_key):
        """Cumulative offsets from a per-molecule count array."""
        c = np.asarray(self.f[count_key][:], dtype=np.int64)
        return np.concatenate([[0], np.cumsum(c)])

    # ---------- retrieval ----------
    def bond_order_matrix(self, mol_id):
        """Bond-order matrix: off-diagonal * factor, diagonal zeroed. None if absent."""
        key = self.index.get(norm_id(mol_id))
        if key is None:
            return None
        M = (self._from_hier(key) if self.layout == "hier"
             else self._from_sparse(key) if self.layout == "sparse"
             else self._from_dense(key))
        if M is None:
            return None
        M = np.asarray(M, dtype=float).copy()
        np.fill_diagonal(M, 0.0)
        M *= self.factor                      # off-diagonal only; diagonal is 0
        return 0.5 * (M + M.T)                # enforce exact symmetry

    def _from_sparse(self, row):
        f = self.f
        ptr_key = (self.ptr_key_override
                   or self._pick(["bond_ptr", "bonds_ptr", "bond_offsets",
                                  "bond_offset", "bonds_offsets",
                                  "bond_index_ptr", "bond_slices"]))
        if ptr_key is not None:
            ptr = np.asarray(f[ptr_key][:], dtype=np.int64)
            # some builders store m start offsets rather than m+1 pointers
            if len(ptr) == len(self.index):
                ptr = np.append(ptr, int(f[self._bo].shape[0]))
        elif self._pick(["n_bonds", "nbonds"]):
            ptr = self._offsets(self._pick(["n_bonds", "nbonds"]))
        else:
            raise SystemExit(f"{self.path}: sparse layout needs bond_ptr or n_bonds")
        b0, b1 = int(ptr[row]), int(ptr[row + 1])
        n_key = self._pick(["n_atoms", "natoms", "n_atom", "natom", "num_atoms"])
        if n_key is None:
            raise SystemExit(f"{self.path}: sparse layout needs an atom-count "
                             f"dataset (n_atoms). Root keys: {sorted(f.keys())}")
        n = int(f[n_key][row])

        if self._pairs is not None:                     # (M, 2) pair array
            pr = np.asarray(f[self._pairs][b0:b1], dtype=int)
            bi, bj = pr[:, 0], pr[:, 1]
        else:
            bi = np.asarray(f[self._bi][b0:b1], dtype=int)
            bj = np.asarray(f[self._bj][b0:b1], dtype=int)
        bo = np.asarray(f[self._bo][b0:b1], dtype=float)

        if bi.size and (bi.max() >= n or bj.max() >= n):  # 1-indexed file
            bi, bj = bi - 1, bj - 1
        M = np.zeros((n, n))
        M[bi, bj] = bo
        M[bj, bi] = bo
        return M

    def _from_dense(self, row):
        f = self.f
        if self._ptr_key == "__derive__":
            na = np.asarray(f["n_atoms"][:], dtype=np.int64)
            ptr = np.concatenate([[0], np.cumsum(na * na)])
        else:
            ptr = np.asarray(f[self._ptr_key][:], dtype=np.int64)
        p0, p1 = int(ptr[row]), int(ptr[row + 1])
        n = int(f["n_atoms"][row])
        flat = np.asarray(f[self._mat_key][p0:p1], dtype=float)
        return self._unpack(flat, n)

    def _from_hier(self, key):
        g = self.f["molecules"][key]
        d = self._find(g)
        if d is None:
            return None
        arr = np.asarray(d[...], dtype=float)
        if arr.ndim == 2:
            return arr
        n = int(round((-1 + np.sqrt(1 + 8 * arr.size)) / 2))
        return self._unpack(arr, n)

    def _find(self, group):
        stack = [("", group)]
        while stack:
            pre, g = stack.pop()
            for k, item in g.items():
                name = f"{pre}{k}"
                if isinstance(item, h5py.Dataset):
                    if self.hint in name.lower() and item.ndim in (1, 2):
                        return item
                else:
                    stack.append((name + "/", item))
        return None

    @staticmethod
    def _unpack(flat, n):
        if flat.size == n * n:
            return flat.reshape(n, n)
        tri = n * (n + 1) // 2
        if flat.size == tri:
            M = np.zeros((n, n))
            M[np.tril_indices(n)] = flat
            return M + M.T - np.diag(np.diag(M))
        raise ValueError(f"cannot reshape {flat.size} values for n={n}")

    def close(self):
        self.f.close()


# ==========================================================================
# spectra
# ==========================================================================
def spectrum(M):
    """
    Real eigenvalues of a bond-order matrix, ascending.
    The matrix is symmetric with a zeroed diagonal, so eigvalsh gives exactly
    real eigenvalues (no imaginary part to discard) and trace = 0 centres the
    spectrum. eigvals + .real is used only if symmetry somehow fails.
    """
    M = np.asarray(M, dtype=float)
    if np.allclose(M, M.T, atol=1e-10):
        return np.sort(np.linalg.eigvalsh(M))
    return np.sort(np.linalg.eigvals(M).real)


def rms_normalise(ev):
    r = float(np.sqrt((ev ** 2).mean()))
    return ev / r if r > 0 else ev


# ==========================================================================
# main
# ==========================================================================
def run(ids, sources=SOURCES, out_prefix="bo_wasserstein-all", verbose=True,
        keep_matrices=("pauling", "reaxff"), keep_ids=None, progress_every=None):
    """
    keep_matrices : method names whose bond-order matrices are returned, as
                    {method: {mol_id: (n,n) array}}. Set to () or None to keep
                    nothing (the original streaming behaviour -- matrices are
                    discarded each iteration).
    keep_ids      : optional subset of ids to retain matrices for; None = every
                    id that passes the checks. Use it when m is large, since
                    retention is the only part of this pipeline whose memory
                    grows with m.
    """
    keep_matrices = tuple(keep_matrices or ())
    # Retention is the only memory that scales with m. Running the full dataset
    # while keeping every matrix will exhaust RAM, so warn loudly rather than
    # dying halfway through a long job.
    if keep_matrices and keep_ids is None and len(ids) > 20000:
        print(f"  [warn] keep_matrices={keep_matrices} with {len(ids)} molecules "
              f"will hold every matrix in RAM.\n"
              f"         Pass keep_matrices=() (CLI: --keep none) or restrict "
              f"with keep_ids=[...].")
    if progress_every is None:
        progress_every = max(1, len(ids) // 20) if len(ids) > 5000 else 0
    unknown = [k for k in keep_matrices if k not in sources]
    if unknown:
        raise ValueError(f"keep_matrices names not in sources: {unknown}")
    keep_set = None if keep_ids is None else {norm_id(i) for i in keep_ids}
    matrices = {k: {} for k in keep_matrices}

    src = {}
    for name, cfg in sources.items():
        s = MatrixSource(name, cfg["path"], cfg["offdiag_factor"], cfg.get("hint", ""),
                         layout=cfg.get("layout"), matrix_key=cfg.get("matrix_key"),
                         ptr_key=cfg.get("ptr_key"), bo_key=cfg.get("bo_key"))
        src[name] = s
        if verbose:
            print(f"  {name:8s} {cfg['path']}  layout={s.layout}  "
                  f"molecules={len(s.index)}  offdiag_factor={s.factor}")

    names = list(sources)
    raw = {f"{a}-{b}": [] for a, b in PAIRS}
    nrm = {f"{a}-{b}": [] for a, b in PAIRS}
    kept_ids, n_atoms, skipped = [], [], []

    # ---- single loop: load, spectra, all pairwise distances, discard matrices ----
    for mid in ids:
        mats = {nm: src[nm].bond_order_matrix(mid) for nm in names}
        absent = [nm for nm, M in mats.items() if M is None]
        if absent:
            skipped.append((mid, f"absent in {','.join(absent)}"))
            continue
        shapes = {nm: M.shape[0] for nm, M in mats.items()}
        if len(set(shapes.values())) != 1:
            skipped.append((mid, f"size mismatch {shapes}"))
            continue

        if keep_matrices and (keep_set is None or norm_id(mid) in keep_set):
            for nm in keep_matrices:
                matrices[nm][mid] = mats[nm]     # already a fresh array

        ev = {nm: spectrum(M) for nm, M in mats.items()}
        evn = {nm: rms_normalise(e) for nm, e in ev.items()}
        for a, b in PAIRS:
            k = f"{a}-{b}"
            raw[k].append(wasserstein_distance(ev[a], ev[b]))
            nrm[k].append(wasserstein_distance(evn[a], evn[b]))
        kept_ids.append(mid)
        n_atoms.append(next(iter(shapes.values())))
        del mats, ev, evn
        if progress_every and len(kept_ids) % progress_every == 0:
            print(f"    {len(kept_ids)}/{len(ids)} processed", flush=True)

    for s in src.values():
        s.close()

    raw = {k: np.asarray(v) for k, v in raw.items()}
    nrm = {k: np.asarray(v) for k, v in nrm.items()}
    m = len(kept_ids)
    if verbose:
        print(f"\nm = {m} molecules with all three methods present"
              f"{f'; {len(skipped)} skipped' if skipped else ''}")
        for s in skipped[:5]:
            print(f"    skipped {s[0]}: {s[1]}")
        for nm in keep_matrices:
            nbytes = sum(M.nbytes for M in matrices[nm].values())
            print(f"  retained {len(matrices[nm]):5d} {nm} matrices "
                  f"({nbytes/1e6:.2f} MB)")
        _summary(raw, nrm)

    if m:
        _plot(kept_ids, np.asarray(n_atoms), raw, nrm, out_prefix)
    return {"ids": kept_ids, "n_atoms": np.asarray(n_atoms),
            "raw": raw, "normalised": nrm, "skipped": skipped,
            "matrices": matrices}


def save_matrices(matrices, path):
    """
    Write {method: {mol_id: (n,n)}} to a compressed .npz. Matrices have varying
    n, so each is stored under its own key '<method>/<mol_id>' rather than
    stacked into one ragged array.
    """
    flat = {f"{m}/{mid}": M for m, d in matrices.items() for mid, M in d.items()}
    if not flat:
        print("no matrices retained -- nothing written")
        return None
    np.savez_compressed(path, **flat)
    print(f"saved {len(flat)} matrices to {path}")
    return path


def load_matrices(path):
    """Inverse of save_matrices -> {method: {mol_id: (n,n)}}."""
    z = np.load(path)
    out = {}
    for k in z.files:
        m, mid = k.split("/", 1)
        out.setdefault(m, {})[mid] = z[k]
    return out


def _summary(raw, nrm):
    print(f"\n{'pair':18s} {'raw mean':>10s} {'raw med':>9s} "
          f"{'norm mean':>11s} {'norm med':>9s}")
    for k in raw:
        print(f"{k:18s} {raw[k].mean():10.4f} {np.median(raw[k]):9.4f} "
              f"{nrm[k].mean():11.4f} {np.median(nrm[k]):9.4f}")
    ks = list(raw)
    print("\npairwise KS tests between the raw distance distributions:")
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            st = ks_2samp(raw[ks[i]], raw[ks[j]])
            print(f"  {ks[i]} vs {ks[j]}: D={st.statistic:.3f}  p={st.pvalue:.2e}")


SCATTER_MAX = 5000          # above this, per-molecule panels switch to density


def _plot(ids, n_atoms, raw, nrm, prefix):
    m = len(ids)
    big = m > SCATTER_MAX
    rng = np.random.default_rng(0)
    # a plain scatter of ~10^5 points is an unreadable blob and a huge PNG, so
    # for large m the per-molecule panel shows a subsample and the size panel
    # switches to a 2-D density
    sub = rng.choice(m, SCATTER_MAX, replace=False) if big else np.arange(m)
    sub.sort()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    x = np.arange(m)

    for row, (data, tag) in enumerate(((raw, "raw"), (nrm, "RMS-normalised"))):
        ax = axes[row, 0]
        for k, v in data.items():
            ax.plot(x[sub], v[sub], ".", ms=2 if big else 3,
                    alpha=0.35 if big else 0.6, label=k, color=COLORS[k])
        ax.set_xlabel("molecule index")
        ax.set_ylabel(f"Wasserstein distance ({tag})")
        ax.set_title(f"per-molecule distance, m={m}"
                     + (f" ({len(sub)} shown)" if big else "") + f" ({tag})")
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        bins = np.histogram_bin_edges(np.concatenate(list(data.values())),
                                      bins=60 if big else 40)
        for k, v in data.items():
            ax.hist(v, bins=bins, histtype="step", lw=1.6, label=k,
                    color=COLORS[k], density=big)
        ax.set_xlabel(f"Wasserstein distance ({tag})")
        ax.set_ylabel("density" if big else "count")
        ax.set_title(f"distribution ({tag})")
        ax.legend(fontsize=8)

        ax = axes[row, 2]
        if big:
            allx = np.concatenate([n_atoms] * len(data))
            ally = np.concatenate(list(data.values()))
            hb = ax.hexbin(allx, ally, gridsize=40, cmap="viridis",
                           bins="log", mincnt=1)
            fig.colorbar(hb, ax=ax, label="molecules (log)")
            for k, v in data.items():                # median trend per pair
                order = np.argsort(n_atoms)
                uniq = np.unique(n_atoms)
                med = [np.median(v[n_atoms == u]) for u in uniq]
                ax.plot(uniq, med, "-", lw=1.6, color=COLORS[k], label=k)
        else:
            for k, v in data.items():
                ax.plot(n_atoms + np.random.uniform(-.2, .2, len(v)), v, ".",
                        ms=3, alpha=0.5, label=k, color=COLORS[k])
        ax.set_xlabel("atoms per molecule")
        ax.set_ylabel(f"Wasserstein distance ({tag})")
        ax.set_title(f"size dependence ({tag})")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{prefix}.png", dpi=180)
    print(f"\nsaved {prefix}.png")


def all_ids(sources=SOURCES, verbose=True):
    """
    Every molecule present in ALL sources, as the raw id strings of the first
    source. Opens each file, reads only its id index, and closes it -- no
    matrices are touched, so this is cheap even for the full dataset.
    """
    per, raw_of = {}, {}
    for name, cfg in sources.items():
        s = MatrixSource(name, cfg["path"], cfg.get("offdiag_factor", 1.0),
                         cfg.get("hint", ""), cfg.get("datasets"))
        per[name] = set(s.index)
        if not raw_of:
            if s.layout == "hier":
                raw_of = {norm_id(k): k for k in s.f["molecules"].keys()}
            else:
                ids = s.f[s._pick(["mol_id", "mol_ids", "molecule_id", "ids"])]
                raw_of = {norm_id(v): v for v in ids.asstr()[:]}
        s.close()
        if verbose:
            print(f"  {name:8s} {len(per[name]):7d} molecules")

    common = set.intersection(*per.values())
    if verbose:
        print(f"  common to all sources: {len(common)}")
        for name, ss in per.items():
            miss = len(ss - common)
            if miss:
                print(f"    {name}: {miss} not shared")
    return [raw_of[q] for q in sorted(common) if q in raw_of]


def load_ids(path):
    if path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        col = next((c for c in df.columns
                    if c.lower() in ("id", "mol_id", "molecule_id", "qm9_id")),
                   df.columns[0])
        return df[col].tolist()
    return [ln.strip() for ln in open(path) if ln.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ids", default=None,
                    help="txt (one id per line) or csv with an id column. "
                         "OMIT to process EVERY molecule common to all files.")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N ids (quick test)")
    ap.add_argument("--stride", type=int, default=None,
                    help="take every Nth id (uniform subsample of the full set)")
    ap.add_argument("--minpop", default=SOURCES["minpop"]["path"])
    ap.add_argument("--pauling", default=SOURCES["pauling"]["path"])
    ap.add_argument("--reaxff", default=SOURCES["reaxff"]["path"])
    ap.add_argument("--minpop-factor", type=float, default=2.0,
                    help="off-diagonal multiplier for MinPop (default 2.0)")
    ap.add_argument("--out", default="bo_wasserstein")
    ap.add_argument("--keep", default="pauling,reaxff",
                    help="comma-separated methods whose BO matrices to return "
                         "(default 'pauling,reaxff'; use 'none' to keep none)")
    ap.add_argument("--save-matrices", default=None,
                    help="also write the retained matrices to this .npz path")
    args = ap.parse_args()

    cfg = {k: dict(v) for k, v in SOURCES.items()}
    cfg["minpop"]["path"] = args.minpop
    cfg["pauling"]["path"] = args.pauling
    cfg["reaxff"]["path"] = args.reaxff
    cfg["minpop"]["offdiag_factor"] = args.minpop_factor

    keep = () if args.keep.strip().lower() in ("none", "") \
        else tuple(s.strip() for s in args.keep.split(","))

    if args.ids:
        ids = load_ids(args.ids)
        print(f"{len(ids)} ids requested from {args.ids}")
    else:
        print("no --ids given: using every molecule common to all sources")
        ids = all_ids(cfg)
    if args.stride:
        ids = ids[::args.stride]
        print(f"  stride {args.stride} -> {len(ids)} ids")
    if args.limit:
        ids = ids[:args.limit]
        print(f"  limit -> {len(ids)} ids")
    res = run(ids, cfg, out_prefix=args.out, keep_matrices=keep)
    if args.save_matrices:
        save_matrices(res["matrices"], args.save_matrices)


if __name__ == "__main__":
    main()
