"""gnts -- Gaussian MinPop/MBS parsing and n_TS bonding/antibonding analysis."""

from .gaussian import (
    read_output,
    split_jobs,
    parse_mbs_condensed,
    mbs_bond_orders,
    parse_geometry,
    load_bond_orders,
)
from .decompose import (
    bonding,
    bond_table,
    antibonding,
    antibonding_table,
    analyse_reaction,
    summary,
)

__version__ = "0.1.0"

__all__ = [
    "read_output", "split_jobs", "parse_mbs_condensed", "mbs_bond_orders",
    "parse_geometry", "load_bond_orders",
    "decompose", "bond_table", "antibonding", "antibonding_table",
    "analyse_reaction", "summary",
]
