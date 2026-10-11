#!/usr/bin/env python3
"""
pauling_bo.py — Empirical (Pauling) bond-order matrix.

Bond order:  n = exp[ (D1 - D) / b ]        (Pauling 1947; Nature of the Chemical Bond)
  D  : observed interatomic distance (Angstrom)
  D1 : reference single-bond length = r1(A) + r1(B), from covalent radii
  b  : decay constant, derived per element pair from multiple-bond radii anchors:
         b = (D1 - D2)/ln(2)   and/or   b = (D1 - D3)/ln(3)
       (averaged when both exist; fallback 0.30 A, Pauling's original value)

Matrix convention (matches the manuscript plan AND Gaussian's Mayer block):
  - off-diagonal (i,j): Pauling bond order for BONDED pairs, 0.0 otherwise
  - diagonal (i,i): row sum = empirical valence of atom i
    (the bond-valence-sum analog; Brown & Altermatt 1985)

USE AS A LIBRARY (added for pipeline integration; CLI behaviour unchanged):
  from pauling_bo import bond_order_matrix, from_atoms
  M = bond_order_matrix(symbols, coords, bonds=bonds)      # list-of-lists or ndarray
  M = from_atoms(ase_atoms, bonds=bonds, as_numpy=True)    # ASE Atoms -> ndarray

Usage (CLI):
  python3 pauling_bo.py molecule.xyz                 # print + write molecule_pauling_bo.csv
  python3 pauling_bo.py molecule.xyz --scale 1.3
  python3 pauling_bo.py molecule.xyz --bonds bonds.txt

!! VERIFY BEFORE MANUSCRIPT USE !!
The radii below are entered from memory of the cited papers. A student must
check every value against:
  r1: Pyykko & Atsumi, Chem. Eur. J. 2009, 15, 186   (single-bond radii)
  r2: Pyykko & Atsumi, Chem. Eur. J. 2009, 15, 12770 (double-bond radii)
  r3: Pyykko, Riedel, Patzschke, Chem. Eur. J. 2005, 11, 3511 (triple-bond radii)
and correct as needed. The SI table for the paper is built from the same values.
"""

import argparse
import math
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Covalent radii (Angstrom).  VERIFY against the papers cited in the header.
# r2/r3 = None means no multiple-bond radius exists (e.g., hydrogen).
# --------------------------------------------------------------------------
RADII = {
    #        r1     r2     r3
    "H": (0.32, None, None),
    "C": (0.75, 0.67, 0.60),
    "N": (0.71, 0.60, 0.54),
    "O": (0.63, 0.57, 0.53),
    "F": (0.64, 0.59, 0.53),
}

B_FALLBACK = 0.30  # Angstrom; Pauling 1947, used when no multiple-bond anchor exists


def read_xyz(path):
    """Return (symbols, coords) from a standard XYZ file."""
    lines = Path(path).read_text().strip().splitlines()
    try:
        natoms = int(lines[0].split()[0])
    except (ValueError, IndexError):
        sys.exit(f"ERROR: first line of {path} must be the atom count.")
    atom_lines = lines[2 : 2 + natoms]
    if len(atom_lines) < natoms:
        sys.exit(f"ERROR: {path} declares {natoms} atoms but has {len(atom_lines)}.")
    symbols, coords = [], []
    for ln in atom_lines:
        parts = ln.split()
        sym = parts[0].capitalize()
        if sym not in RADII:
            sys.exit(f"ERROR: element '{sym}' not in parameter table (CHNOF only).")
        symbols.append(sym)
        coords.append(tuple(float(x) for x in parts[1:4]))
    return symbols, coords


def distance(a, b):
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))


def pair_parameters(sym_a, sym_b):
    """Return (D1, b) for an element pair, deriving b from available anchors."""
    r1a, r2a, r3a = RADII[sym_a]
    r1b, r2b, r3b = RADII[sym_b]
    d1 = r1a + r1b
    b_estimates = []
    if r2a is not None and r2b is not None:
        d2 = r2a + r2b
        b_estimates.append((d1 - d2) / math.log(2.0))
    if r3a is not None and r3b is not None:
        d3 = r3a + r3b
        b_estimates.append((d1 - d3) / math.log(3.0))
    b = sum(b_estimates) / len(b_estimates) if b_estimates else B_FALLBACK
    return d1, b


def detect_bonds(symbols, coords, scale):
    """Distance-cutoff connectivity: bonded if D < scale*(r1_A + r1_B)."""
    bonds = []
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):
            d = distance(coords[i], coords[j])
            cutoff = scale * (RADII[symbols[i]][0] + RADII[symbols[j]][0])
            if d < cutoff:
                bonds.append((i, j))
    return bonds


def read_bonds_file(path, natoms):
    """Read explicit connectivity: one '<i> <j>' pair per line, 1-indexed."""
    bonds = []
    for ln in Path(path).read_text().strip().splitlines():
        if not ln.strip() or ln.strip().startswith("#"):
            continue
        i, j = (int(x) for x in ln.split()[:2])
        if not (1 <= i <= natoms and 1 <= j <= natoms):
            sys.exit(f"ERROR: bond '{ln.strip()}' out of range 1..{natoms}.")
        bonds.append((min(i, j) - 1, max(i, j) - 1))
    return sorted(set(bonds))


def build_matrix(symbols, coords, bonds):
    """N x N Pauling bond-order matrix; diagonal = row sums (empirical valence)."""
    n = len(symbols)
    bo = [[0.0] * n for _ in range(n)]
    for i, j in bonds:
        d = distance(coords[i], coords[j])
        d1, b = pair_parameters(symbols[i], symbols[j])
        order = math.exp((d1 - d) / b)
        bo[i][j] = bo[j][i] = order
    for i in range(n):
        bo[i][i] = sum(bo[i][j] for j in range(n) if j != i)
    return bo


# ==========================================================================
# Library API (added for pipeline integration)
# ==========================================================================
def bond_order_matrix(symbols, coords, bonds=None, scale=1.25, full=False,
                      as_numpy=False):
    """
    Build the Pauling bond-order matrix from symbols + coordinates.

    bonds : list of 0-indexed (i, j) pairs. If None, connectivity is chosen by
            `full` (all pairs) or a distance cutoff at `scale`.
    Returns list-of-lists, or an ndarray if as_numpy=True.
    """
    symbols = [str(s).capitalize() for s in symbols]
    unknown = sorted({s for s in symbols if s not in RADII})
    if unknown:
        raise ValueError(f"element(s) {unknown} not in the radii table (CHNOF only)")
    coords = [tuple(float(x) for x in c) for c in coords]

    n = len(symbols)
    if bonds is None:
        bonds = ([(i, j) for i in range(n) for j in range(i + 1, n)] if full
                 else detect_bonds(symbols, coords, scale))
    bonds = sorted({(min(i, j), max(i, j)) for i, j in bonds})

    M = build_matrix(symbols, coords, bonds)
    if as_numpy:
        import numpy as np
        return np.asarray(M, dtype=float)
    return M


def from_atoms(atoms, bonds=None, scale=1.25, full=False, as_numpy=True):
    """
    Pauling bond-order matrix straight from an ASE Atoms object.

    Atom ORDER is preserved, so the result is index-aligned with any other
    atom x atom matrix parsed from the same Gaussian job (MBS, Mayer).
    """
    symbols = atoms.get_chemical_symbols()
    coords = atoms.get_positions()
    return bond_order_matrix(symbols, coords, bonds=bonds, scale=scale,
                             full=full, as_numpy=as_numpy)


def split_valence(M):
    """
    Separate the two quantities packed into the matrix, mirroring the treatment
    of Gaussian's Mayer block: diagonal = empirical valence, off-diagonal = bond
    order. Returns (valence (N,), bond_orders (N,N) with zeroed diagonal).
    """
    import numpy as np
    M = np.asarray(M, dtype=float)
    val = np.diag(M).copy()
    BO = M.copy()
    np.fill_diagonal(BO, 0.0)
    return val, BO


def main():
    ap = argparse.ArgumentParser(description="Pauling bond-order matrix from XYZ.")
    ap.add_argument("xyz", help="input .xyz file (CHNOF elements)")
    ap.add_argument("--scale", type=float, default=1.25,
                    help="bond-detection cutoff factor (default 1.25)")
    ap.add_argument("--bonds", default=None,
                    help="optional connectivity file: '<i> <j>' per line, 1-indexed "
                         "(preferred for production; overrides distance detection)")
    ap.add_argument("--full", action="store_true",
                    help="compute the exponential for ALL atom pairs (full-matrix null), "
                         "not just detected/declared bonds; diagonal remains the row sum")
    ap.add_argument("--out", default=None, help="output CSV path (default: <stem>_pauling_bo.csv)")
    args = ap.parse_args()

    symbols, coords = read_xyz(args.xyz)
    if args.full:
        n = len(symbols)
        bonds = [(i, j) for i in range(n) for j in range(i + 1, n)]
        print("[--full: exponential evaluated for all pairs — nonbonded elements are "
              "the geometry-null baseline for delocalization fingerprints]", file=sys.stderr)
    elif args.bonds:
        bonds = read_bonds_file(args.bonds, len(symbols))
    else:
        bonds = detect_bonds(symbols, coords, args.scale)
        print(f"[distance-based connectivity, scale={args.scale} — "
              f"prefer --bonds from SMILES for production]", file=sys.stderr)

    bo = build_matrix(symbols, coords, bonds)

    # ---- report ----
    labels = [f"{s}{k+1}" for k, s in enumerate(symbols)]
    print(f"\n{Path(args.xyz).name}: {len(symbols)} atoms, {len(bonds)} bonds\n")
    print(f"{'bond':>10} {'D (A)':>8} {'D1 (A)':>8} {'b (A)':>7} {'order':>7}")
    for i, j in bonds:
        d = distance(coords[i], coords[j])
        d1, b = pair_parameters(symbols[i], symbols[j])
        n_ord = math.exp((d1 - d) / b)
        flag = "  <-- check" if (n_ord < 0.3 or n_ord > 3.5) else ""
        print(f"{labels[i]+'-'+labels[j]:>10} {d:8.3f} {d1:8.3f} {b:7.3f} {n_ord:7.3f}{flag}")
    print(f"\n{'atom':>6} {'valence (row sum)':>18}")
    for i, lab in enumerate(labels):
        print(f"{lab:>6} {bo[i][i]:18.3f}")

    # ---- CSV output ----
    out = args.out or (Path(args.xyz).stem + "_pauling_bo.csv")
    with open(out, "w") as fh:
        fh.write("," + ",".join(labels) + "\n")
        for i, lab in enumerate(labels):
            fh.write(lab + "," + ",".join(f"{bo[i][j]:.6f}" for j in range(len(labels))) + "\n")
    print(f"\nmatrix written to {out}")


if __name__ == "__main__":
    main()
