"""Parse MinPop (MBS) bond orders out of Gaussian output files."""

import bz2
import re

import numpy as np

MBS_HEADER_RE = re.compile(
    r"^\s*MBS Condensed to atoms \(all electrons\):", re.IGNORECASE)

_VALUE_RE = re.compile(r"^[+-]?\d*\.\d+(?:[DEde][+-]?\d+)?$")
SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl",
          35: "Br", 53: "I"}


def _is_value(tok):
    """A token is numeric data only if it carries a decimal point.

    Gaussian always prints values with one, while row indices and the magnetic
    components of AO labels ('6D 0', '8F+1') do not.
    """
    return bool(_VALUE_RE.match(tok))


def _to_float(tok):
    return float(tok.replace("D", "E").replace("d", "e"))


def read_log(path):
    """Read a Gaussian output file, transparently handling .bz2."""
    opener = bz2.open if str(path).endswith(".bz2") else open
    with opener(path, "rt", errors="replace") as fh:
        return fh.read()


def split_jobs(text):
    """Split a Link1 multi-job log into per-job texts."""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.strip() == "Initial command:"]
    if not starts:
        starts = [i for i, ln in enumerate(lines)
                  if ln.startswith(" Entering Link 1 =")]
    if not starts:
        return [text]
    starts[0] = 0
    bounds = starts + [len(lines)]
    return ["\n".join(lines[bounds[k]:bounds[k + 1]])
            for k in range(len(starts))]


def parse_atom_matrix(text, header_re, occurrence=-1):
    """Read a symmetric atom x atom matrix from Gaussian's blocked format.

    Gaussian prints five columns at a time, lower triangle only. Both triangles
    are filled on return.
    """
    lines = text.splitlines()
    hits = [i for i, ln in enumerate(lines) if header_re.search(ln)]
    if not hits:
        return None

    entries, cols, started = {}, [], False
    for ln in lines[hits[occurrence] + 1:]:
        toks = ln.split()
        if not toks:
            if started:
                break
            continue
        if all(t.lstrip("+-").isdigit() for t in toks):
            cols = [int(t) - 1 for t in toks]
            continue
        if toks[0].lstrip("+-").isdigit() and cols:
            row = int(toks[0]) - 1
            vals = [_to_float(t) for t in toks[1:] if _is_value(t)]
            for k, v in enumerate(vals):
                if k >= len(cols):
                    break
                entries[(row, cols[k])] = v
                entries[(cols[k], row)] = v
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


def mbs_matrix(path, occurrence=-1):
    """MBS condensed-to-atoms matrix as printed, from a file path.

    Returns None if the file has no MBS section.
    """
    jobs = split_jobs(read_log(path))
    for job in reversed(jobs):
        M = parse_atom_matrix(job, MBS_HEADER_RE, occurrence)
        if M is not None:
            return M
    return None


def minpop_bond_orders(path, factor=2.0, occurrence=-1):
    """MinPop bond-order matrix: off-diagonal x factor, diagonal zeroed.

    The MBS off-diagonals are shared populations; factor=2.0 converts them to
    bond orders. The diagonal (net population) is set to zero.
    """
    M = mbs_matrix(path, occurrence)
    if M is None:
        return None
    B = np.array(M, dtype=float)
    np.fill_diagonal(B, 0.0)
    B = 0.5 * (B + B.T) * float(factor)
    return B


def parse_geometry(text):
    """Atomic numbers and coordinates from the last orientation block.

    Returns (numbers, coords) or (None, None).
    """
    lines = text.splitlines()
    std = [i for i, ln in enumerate(lines) if "Standard orientation:" in ln]
    inp = [i for i, ln in enumerate(lines) if "Input orientation:" in ln]
    starts = std or inp
    if not starts:
        return None, None

    numbers, coords, started = [], [], False
    for ln in lines[starts[-1] + 1:]:
        toks = ln.split()
        if len(toks) == 6 and _is_value(toks[3]):
            numbers.append(int(float(toks[1])))
            coords.append([float(t) for t in toks[3:6]])
            started = True
            continue
        if started:
            break
    if not numbers:
        return None, None
    return np.array(numbers, dtype=int), np.array(coords, dtype=float)


def symbols_from(path):
    """Element symbols for the atoms in a Gaussian output file."""
    numbers, _ = parse_geometry(read_log(path))
    if numbers is None:
        return None
    return [SYMBOL.get(int(z), f"Z{int(z)}") for z in numbers]


def load_triple(path_R, path_TS, path_P, factor=2.0):
    """MinPop bond-order matrices for a reactant/TS/product triple.

    Returns (B_R, B_TS, B_P, symbols). Raises if a file lacks an MBS section or
    the three matrices disagree in size.
    """
    mats, labels = [], ("reactant", "TS", "product")
    for path, label in zip((path_R, path_TS, path_P), labels):
        B = minpop_bond_orders(path, factor=factor)
        if B is None:
            raise ValueError(f"no MBS section found in the {label} file: {path}")
        mats.append(B)

    shapes = {label: M.shape[0] for label, M in zip(labels, mats)}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"atom counts differ between files: {shapes}")

    return (*mats, symbols_from(path_R))
