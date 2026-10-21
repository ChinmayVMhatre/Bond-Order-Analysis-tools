"""Parse MinPop (MBS condensed-to-atoms) matrices from Gaussian output files."""

import bz2
import re

import numpy as np

SYMBOL = {1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
          15: "P", 16: "S", 17: "Cl"}

H_MBS = re.compile(r"^\s*MBS Condensed to atoms \(all electrons\):")
RE_VALUE = re.compile(r"^[+-]?\d*\.\d+(?:[DEde][+-]?\d+)?$")


def read_output(path):
    """Read a Gaussian output file. Plain text by default; .bz2 also accepted."""
    opener = bz2.open if str(path).endswith(".bz2") else open
    with opener(path, "rt", errors="replace") as fh:
        return fh.read()


def split_jobs(text):
    """Split a Link1 multi-job output into per-job texts."""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.strip() == "Initial command:"]
    if not starts:
        return [text]
    starts[0] = 0
    bounds = starts + [len(lines)]
    return ["\n".join(lines[bounds[k]:bounds[k + 1]]) for k in range(len(starts))]


def _f(tok):
    return float(tok.replace("D", "E").replace("d", "e"))


def _is_value(tok):
    """A token is a value only if it contains a decimal point."""
    return bool(RE_VALUE.match(tok))


def parse_mbs_condensed(text, occurrence=-1):
    """
    Parse the MBS condensed-to-atoms matrix from Gaussian's blocked
    lower-triangular print. Returns a full symmetric (N, N) array, or None.
    """
    lines = text.splitlines()
    hits = [i for i, ln in enumerate(lines) if H_MBS.search(ln)]
    if not hits:
        return None

    entries, cur_cols, started = {}, [], False
    for ln in lines[hits[occurrence] + 1:]:
        toks = ln.split()
        if not toks:
            if started:
                break
            continue
        if all(t.lstrip("+-").isdigit() for t in toks):
            cur_cols = [int(t) - 1 for t in toks]
            continue
        if toks[0].lstrip("+-").isdigit() and cur_cols:
            row = int(toks[0]) - 1
            vals = [_f(t) for t in toks[1:] if _is_value(t)]
            for k, v in enumerate(vals):
                if k >= len(cur_cols):
                    break
                entries[(row, cur_cols[k])] = v
                entries[(cur_cols[k], row)] = v
            started = True
            continue
        if started:
            break

    if not entries:
        return None
    n = max(max(i, j) for i, j in entries) + 1
    M = np.zeros((n, n))
    for (i, j), v in entries.items():
        M[i, j] = v
    return M


def mbs_bond_orders(mbs, factor=2.0):
    """
    MinPop bond-order matrix from an MBS condensed-to-atoms matrix:
    off-diagonals scaled by `factor`, diagonal set to zero.
    """
    M = np.asarray(mbs, dtype=float).copy()
    np.fill_diagonal(M, 0.0)
    return 0.5 * (M + M.T) * float(factor)


def parse_geometry(text):
    """Return (atomic_numbers, symbols, coordinates) from the last orientation block."""
    lines = text.splitlines()
    std = [i for i, ln in enumerate(lines) if "Standard orientation:" in ln]
    inp = [i for i, ln in enumerate(lines) if "Input orientation:" in ln]
    starts = std or inp
    if not starts:
        return None, None, None

    numbers, coords, started = [], [], False
    for ln in lines[starts[-1] + 1:]:
        toks = ln.split()
        if len(toks) == 6 and _is_value(toks[3]):
            numbers.append(int(float(toks[1])))
            coords.append([float(toks[3]), float(toks[4]), float(toks[5])])
            started = True
            continue
        if started:
            break
    if not numbers:
        return None, None, None
    numbers = np.array(numbers, dtype=int)
    return numbers, [SYMBOL.get(int(z), f"Z{z}") for z in numbers], np.array(coords)


def load_bond_orders(path, factor=2.0, job=-1, occurrence=-1):
    """
    One call: read a Gaussian output and return
    (bond_order_matrix, labels, atomic_numbers).
    """
    text = read_output(path)
    jobs = split_jobs(text)
    mbs = parse_mbs_condensed(jobs[job], occurrence=occurrence)
    if mbs is None:
        raise ValueError(f"{path}: no 'MBS Condensed to atoms' section found")
    bo = mbs_bond_orders(mbs, factor=factor)

    numbers, symbols, _ = parse_geometry(jobs[0])
    if symbols is not None and len(symbols) == bo.shape[0]:
        labels = [f"{s}{i + 1}" for i, s in enumerate(symbols)]
    else:
        labels = [str(i + 1) for i in range(bo.shape[0])]
        numbers = None
    return bo, labels, numbers
