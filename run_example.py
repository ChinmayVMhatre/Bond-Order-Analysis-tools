#!/usr/bin/env python3
"""Worked example: parse the model SN2 reaction and decompose n_TS.

    F(-) + CH3-Cl  ->  TS  ->  CH3-F + Cl(-)

Run from the repository root:

    python examples/run_example.py
"""

from pathlib import Path

import numpy as np

from bonts import (load_triple, decompose, bond_table,
                   antibonding, antibonding_table)

DATA = Path(__file__).resolve().parent / "data"


def main():
    # ---- 1. parse the MinPop (MBS) bond-order matrices -------------------
    B_R, B_TS, B_P, symbols = load_triple(DATA / "sn2_reactant.log",
                                          DATA / "sn2_ts.log",
                                          DATA / "sn2_product.log")
    labels = [f"{s}{i + 1}" for i, s in enumerate(symbols)]
    print(f"atoms: {' '.join(labels)}   matrices: {B_R.shape}\n")

    np.set_printoptions(precision=3, suppress=True)
    print("MinPop bond orders, reactant (diagonal zeroed):")
    print(B_R, "\n")

    # ---- 2. decompose n_TS ----------------------------------------------
    res = decompose(B_R, B_TS, B_P)

    print(f"n_TS                     {res['n_ts']:.4f}")
    print(f"  forming channel        {res['forming']:.4f}"
          f"   (weight share {res['w_forming']:.3f})")
    print(f"  breaking channel       {res['breaking']:.4f}"
          f"   (weight share {res['w_breaking']:.3f})")
    print(f"  sum of contributions   {res['contrib'].sum():.4f}"
          f"   (equals n_TS: {np.isclose(res['contrib'].sum(), res['n_ts'])})")
    print(f"  sum of weights         {res['weight'].sum():.4f}\n")

    print("per-bond contributions:")
    print(bond_table(res, symbols).to_string(index=False), "\n")

    # ---- 3. antibonding channel -----------------------------------------
    anti = antibonding(B_R, B_TS, B_P, n_ts=res["n_ts"])
    print(f"a_excess                 {anti['a_excess']:+.4f}")
    print(f"  antibonding magnitude  R {anti['a_R']:.4f}"
          f"   TS {anti['a_TS']:.4f}   P {anti['a_P']:.4f}\n")

    table = antibonding_table(anti, B_R, B_TS, B_P, symbols)
    if len(table):
        print("antibonding elements:")
        print(table.to_string(index=False))
    else:
        print("no antibonding elements above threshold")


if __name__ == "__main__":
    main()
