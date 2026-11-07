#!/usr/bin/env python3
"""
Example: SN2 model reaction   OH- + CH3F  ->  CH3OH + F-

Atom order in all three files: O1 H2 C3 H4 H5 H6 F7

Run from the repository root:
    python examples/run_example.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Bonts

DATA = Path(__file__).resolve().parent / "data"


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    # ---- 1. parse the three Gaussian outputs ----------------------------
    res = Bonts.analyse_reaction(DATA / "sn2_R.log",
                                DATA / "sn2_TS.log",
                                DATA / "sn2_P.log")
    labels = res["labels"]
    print("labels:", labels, "\n")

    # ---- 2. scalar summary ----------------------------------------------
    Bonts.summary(res)

    # ---- 3. bonding channel, per bond -----------------------------------
    print("\n--- bonding contributions -------------------------------------")
    print(Bonts.bond_table(res["bonding"], labels).to_string(index=False))

    # ---- 4. antibonding channel, per bond -------------------------------
    print("\n--- antibonding contributions ---------------------------------")
    print(Bonts.antibonding_table(res["anti"], res["BO_R"], res["BO_TS"],
                                 res["BO_P"], labels).to_string(index=False))

    # ---- 5. identity checks ---------------------------------------------
    d, a = res["bonding"], res["anti"]
    print("\n--- checks ----------------------------------------------------")
    print(f"sum(contrib) == n_ts        : "
          f"{np.isclose(d['contrib'].sum(), d['n_ts'])}")
    print(f"forming + breaking == n_ts  : "
          f"{np.isclose(d['forming'] + d['breaking'], d['n_ts'])}")
    print(f"sum(weight) == 1            : "
          f"{np.isclose(d['weight'].sum(), 1.0)}")
    print(f"sum(excess) == a_excess     : "
          f"{np.isclose(a['excess'].sum(), a['a_excess'])}")

    # ---- 6. parsing the pieces directly ---------------------------------
    print("\n--- low-level access ------------------------------------------")
    text = Bonts.read_output(DATA / "sn2_TS.log")
    mbs = Bonts.parse_mbs_condensed(text)
    print(f"raw MBS matrix        : {mbs.shape}")
    print(f"total electrons       : {mbs.sum():.4f}")
    print(f"gross populations     : {mbs.sum(axis=1)}")
    print(f"bond orders (x2, diag 0) O1-C3 = "
          f"{Bonts.mbs_bond_orders(mbs)[0, 2]:.4f}")


if __name__ == "__main__":
    main()
