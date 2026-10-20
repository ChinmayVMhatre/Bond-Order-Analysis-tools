"""Command-line interface: bonts-nts R.log TS.log P.log"""

import argparse

import numpy as np

from .gaussian import load_triple
from .decompose import decompose, bond_table
from .antibonding import antibonding, antibonding_table


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="bonts-nts",
        description="Parse MinPop MBS bond orders from three Gaussian output "
                    "files and decompose n_TS.")
    ap.add_argument("reactant")
    ap.add_argument("ts")
    ap.add_argument("product")
    ap.add_argument("--factor", type=float, default=2.0,
                    help="off-diagonal multiplier applied to the MBS matrix "
                         "(default 2.0)")
    ap.add_argument("--min-weight", type=float, default=1e-3,
                    help="drop bonds below this weight from the table")
    ap.add_argument("--antibonding", action="store_true",
                    help="also report the antibonding channel")
    ap.add_argument("--csv", default=None,
                    help="write the per-bond table to this path")
    args = ap.parse_args(argv)

    B_R, B_TS, B_P, symbols = load_triple(args.reactant, args.ts, args.product,
                                          factor=args.factor)
    res = decompose(B_R, B_TS, B_P)

    print(f"atoms      {len(B_R)}")
    print(f"n_TS       {res['n_ts']:.4f}")
    print(f"forming    {res['forming']:.4f}  (weight {res['w_forming']:.3f})")
    print(f"breaking   {res['breaking']:.4f}  (weight {res['w_breaking']:.3f})")
    print()

    table = bond_table(res, symbols, min_weight=args.min_weight)
    print(table.to_string(index=False))

    if args.antibonding:
        anti = antibonding(B_R, B_TS, B_P, n_ts=res["n_ts"])
        print(f"\na_excess   {anti['a_excess']:+.4f}")
        at = antibonding_table(anti, B_R, B_TS, B_P, symbols)
        if len(at):
            print(at.to_string(index=False))

    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
