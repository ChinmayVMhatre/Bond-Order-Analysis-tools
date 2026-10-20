#!/usr/bin/env python3
"""Write the model-reaction Gaussian output files in examples/data/.

The three files are SYNTHETIC: they are hand-built to contain a correctly
formatted geometry block and MBS Condensed to atoms block, so the parser and the
decomposition can be exercised without shipping a large real calculation. They
are not the output of an actual Gaussian run.

Model reaction (Walden inversion):

    F(-) + CH3-Cl  ->  TS  ->  CH3-F + Cl(-)

Atom order in all three files: F1 C2 H3 H4 H5 Cl6
"""

from pathlib import Path

NUMBERS = [9, 6, 1, 1, 1, 17]
HERE = Path(__file__).resolve().parent / "data"

# Cartesian coordinates, Angstrom, collinear F...C...Cl along z
GEOM = {
    "reactant": [(0.000, 0.000, -2.950), (0.000, 0.000, 0.000),
                 (1.027, 0.000, -0.350), (-0.513, 0.889, -0.350),
                 (-0.513, -0.889, -0.350), (0.000, 0.000, 1.785)],
    "ts":       [(0.000, 0.000, -2.010), (0.000, 0.000, 0.000),
                 (1.058, 0.000, 0.000), (-0.529, 0.916, 0.000),
                 (-0.529, -0.916, 0.000), (0.000, 0.000, 2.140)],
    "product":  [(0.000, 0.000, -1.395), (0.000, 0.000, 0.000),
                 (1.027, 0.000, 0.350), (-0.513, 0.889, 0.350),
                 (-0.513, -0.889, 0.350), (0.000, 0.000, 3.060)],
}

# MBS condensed-to-atoms matrices: diagonal = net population, off-diagonal =
# shared population. Bond order = 2 x off-diagonal, so a shared population of
# 0.485 corresponds to a bond order of 0.97.
#
# Non-bonded pairs carry small values, some negative (antibonding), which is
# normal for a Mulliken-type partition.
def matrix(fc, ccl, ch, f_h, cl_h, hh, f_cl, diag):
    """Assemble the 6x6 matrix from the distinct shared populations."""
    M = [[0.0] * 6 for _ in range(6)]
    pairs = {(0, 1): fc, (1, 5): ccl, (0, 5): f_cl,
             (1, 2): ch, (1, 3): ch, (1, 4): ch,
             (0, 2): f_h, (0, 3): f_h, (0, 4): f_h,
             (2, 5): cl_h, (3, 5): cl_h, (4, 5): cl_h,
             (2, 3): hh, (2, 4): hh, (3, 4): hh}
    for (i, j), v in pairs.items():
        M[i][j] = M[j][i] = v
    for i, d in enumerate(diag):
        M[i][i] = d
    return M


MBS = {
    # F-C bond order 0.10 -> shared 0.050 ; C-Cl 0.90 -> 0.450
    "reactant": matrix(fc=0.050, ccl=0.450, ch=0.487, f_h=-0.004,
                       cl_h=-0.021, hh=-0.016, f_cl=0.002,
                       diag=[9.412, 4.905, 0.682, 0.682, 0.682, 16.904]),
    # F-C 0.45 -> 0.225 ; C-Cl 0.48 -> 0.240
    "ts":       matrix(fc=0.225, ccl=0.240, ch=0.476, f_h=-0.019,
                       cl_h=-0.018, hh=-0.022, f_cl=-0.031,
                       diag=[9.118, 4.846, 0.701, 0.701, 0.701, 17.106]),
    # F-C 0.92 -> 0.460 ; C-Cl 0.08 -> 0.040
    "product":  matrix(fc=0.460, ccl=0.040, ch=0.485, f_h=-0.020,
                       cl_h=-0.003, hh=-0.018, f_cl=0.001,
                       diag=[8.886, 4.902, 0.690, 0.690, 0.690, 17.368]),
}

ROUTE = {"reactant": "#P B3LYP/6-31+G(d,p) Opt Pop=(Full) IOp(6/27=122)",
         "ts":       "#P B3LYP/6-31+G(d,p) Opt=TS Pop=(Full) IOp(6/27=122)",
         "product":  "#P B3LYP/6-31+G(d,p) Opt Pop=(Full) IOp(6/27=122)"}

TITLE = {"reactant": "F(-) + CH3Cl, separated reactants",
         "ts":       "F...CH3...Cl Walden inversion transition state",
         "product":  "CH3F + Cl(-), separated products"}


def blocked(matrix_rows, symbols, per_block=5):
    """Render a symmetric matrix in Gaussian's blocked lower-triangular style."""
    n = len(matrix_rows)
    out = []
    for start in range(0, n, per_block):
        cols = list(range(start, min(start + per_block, n)))
        out.append("".join(f"{c + 1:14d}" for c in cols))
        for r in range(start, n):
            cells = "".join(f"{matrix_rows[r][c]:14.6f}"
                            for c in cols if c <= r)
            out.append(f"{r + 1:7d}  {symbols[r]:<2s}{cells}")
    return out


def write(state, path):
    sym = {9: "F", 6: "C", 1: "H", 17: "Cl"}
    symbols = [sym[z] for z in NUMBERS]
    L = ["Entering Gaussian System, Link 0=g16",
         " Initial command:",
         " Entering Link 1 = /opt/g16/l1.exe",
         f" {ROUTE[state]}",
         " ----------------------------------------------------",
         f" {TITLE[state]}",
         " ----------------------------------------------------",
         "",
         "                         Standard orientation:",
         " ---------------------------------------------------------------------",
         " Center     Atomic      Atomic             Coordinates (Angstroms)",
         " Number     Number       Type             X           Y           Z",
         " ---------------------------------------------------------------------"]
    for k, (z, xyz) in enumerate(zip(NUMBERS, GEOM[state]), start=1):
        L.append(f"{k:7d}{z:11d}{0:12d}    "
                 f"{xyz[0]:12.6f}{xyz[1]:12.6f}{xyz[2]:12.6f}")
    L += [" ---------------------------------------------------------------------",
          "",
          " SCF Done:  E(RB3LYP) =  -599.123456789     A.U. after   9 cycles",
          "",
          "   MBS Condensed to atoms (all electrons):"]
    L += blocked(MBS[state], symbols)
    L += ["",
          " Normal termination of Gaussian 16."]
    path.write_text("\n".join(L) + "\n")
    return path


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    for state, name in (("reactant", "sn2_reactant.log"),
                        ("ts", "sn2_ts.log"),
                        ("product", "sn2_product.log")):
        print("wrote", write(state, HERE / name))


if __name__ == "__main__":
    main()
