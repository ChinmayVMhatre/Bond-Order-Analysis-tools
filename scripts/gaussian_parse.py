"""
Gaussian 16 log parser for BEBOP-style two-job files (Opt + ROHF Pop=Full).

Handles the traps that break naive parsing:
  1. Multi-job (Link1) files -> sections are selected per job, not globally.
  2. Three different "Condensed to atoms" blocks (plain / Bonding / MBS)
     -> matched with anchored, mutually exclusive headers.
  3. Condensed-to-atoms DIAGONAL is the net (intra-atomic) population;
     the ROW SUM is the gross atomic population. charge = Z - row_sum.
  4. "Mayer Atomic Bond Orders" matrix mixes two quantities: the diagonal is
     atomic valency, the off-diagonal is bond order -> returned separately.
  5. "Bonding Mulliken charges" are NOT charges (they sum to ~39 here);
     they are bonding populations. Labelled as such.
  6. Gross orbital populations: atom tag appears only on an atom's first row
     -> forward-filled to give per-orbital atom indices.
"""

import re
import bz2
import numpy as np

# ---- section headers: anchored so MBS/Bonding/plain never collide ----
H_COND_PLAIN   = re.compile(r"^\s{2,}Condensed to atoms \(all electrons\):")
H_COND_BONDING = re.compile(r"^\s{2,}Bonding Condensed to atoms \(all electrons\):")
H_COND_MBS     = re.compile(r"^\s{2,}MBS Condensed to atoms \(all electrons\):")
H_MAYER        = re.compile(r"^\s*(?:MBS )?Atomic Valencies and Mayer Atomic Bond Orders:")
H_GOP_PLAIN    = re.compile(r"^\s{2,}Gross orbital populations:")
H_GOP_MBS      = re.compile(r"^\s{2,}MBS Gross orbital populations:")
H_DENSITY      = re.compile(r"^\s{2,}(?:Alpha |Beta )?(?:MBS |Bonding )?Density Matrix:")

RE_DIPOLE_TOT  = re.compile(r"Tot=\s*(-?\d+\.\d+)")
RE_SCF         = re.compile(r"SCF Done:\s+E\((\S+)\)\s*=\s*(-?\d+\.\d+)")


# --------------------------------------------------------------------------
# job splitting
# --------------------------------------------------------------------------
def read_log(path):
    opener = bz2.open if str(path).endswith(".bz2") else open
    with opener(path, "rt", errors="replace") as f:
        return f.read()


def split_jobs(text):
    """
    Split a Link1 multi-job log into per-job texts.

    The reliable boundary is the 'Initial command:' line that Link 0 writes at
    the start of every job ('Entering Gaussian System' appears only once, at the
    very top of the file, so it cannot be used to find later jobs).
    """
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.strip() == "Initial command:"]
    if not starts:
        starts = [i for i, ln in enumerate(lines) if ln.startswith(" Entering Link 1 =")]
    if not starts:
        return [text]
    starts[0] = 0                      # keep the file preamble with job 1
    bounds = starts + [len(lines)]
    return ["\n".join(lines[bounds[k]:bounds[k + 1]]) for k in range(len(starts))]


def job_route(job_text):
    """Return the route line(s) of a job, joined."""
    lines = job_text.splitlines()
    out = []
    for i, ln in enumerate(lines):
        if ln.startswith(" #"):
            out.append(ln.strip())
            for nxt in lines[i + 1:i + 3]:            # route can wrap
                if nxt.startswith(" ---"):
                    break
                out.append(nxt.strip())
            break
    return " ".join(out)


# --------------------------------------------------------------------------
# generic blocked-matrix reader
# --------------------------------------------------------------------------
RE_VALUE = re.compile(r"^[+-]?\d*\.\d+(?:[DEde][+-]?\d+)?$")


def _f(tok):
    return float(tok.replace("D", "E").replace("d", "e"))


def _is_value(tok):
    """
    True only for real numeric values. Gaussian always prints them with a
    decimal point, whereas AO labels contain bare integers for the magnetic
    component ('6D 0', '8F+1') and rows start with an integer index -- so
    requiring a '.' cleanly separates data from labels.
    """
    return bool(RE_VALUE.match(tok))


def _parse_blocked(lines, start, symmetric=True):
    """Read Gaussian's 5-column blocked (lower-triangular) matrix print."""
    entries, cur_cols, have = {}, [], False
    for ln in lines[start:]:
        toks = ln.split()
        if not toks:
            if have:
                break
            continue
        if all(t.lstrip("+-").isdigit() for t in toks):        # column header
            cur_cols = [int(t) - 1 for t in toks]
            continue
        if toks[0].lstrip("+-").isdigit() and cur_cols:
            row = int(toks[0]) - 1
            vals = [_f(t) for t in toks[1:] if _is_value(t)]
            for k, v in enumerate(vals):
                if k >= len(cur_cols):
                    break
                c = cur_cols[k]
                entries[(row, c)] = v
                if symmetric:
                    entries[(c, row)] = v
            have = True
            continue
        if have:
            break
    if not entries:
        return None
    n = max(max(i, j) for i, j in entries) + 1
    M = np.zeros((n, n))
    for (i, j), v in entries.items():
        M[i, j] = v
    return M


def find_matrix(job_text, header_re, occurrence=-1, symmetric=True):
    """Parse the matrix under the given header. occurrence: 0=first, -1=last."""
    lines = job_text.splitlines()
    hits = [i for i, ln in enumerate(lines) if header_re.search(ln)]
    if not hits:
        return None
    return _parse_blocked(lines, hits[occurrence] + 1, symmetric=symmetric)


# --------------------------------------------------------------------------
# trap-aware extractors
# --------------------------------------------------------------------------
def condensed_to_atoms(job_text, kind="mbs", occurrence=-1):
    """
    Return dict with the atom-atom matrix and CORRECT derived populations.

      matrix        (N,N)  as printed (symmetric, full -- both triangles filled)
      net_pop       (N,)   diagonal: density NOT shared with neighbours
      gross_pop     (N,)   ROW SUM: total electrons assigned to the atom
      shared_pop    (N,)   gross - net: density shared with neighbours
    """
    hdr = {"plain": H_COND_PLAIN, "bonding": H_COND_BONDING, "mbs": H_COND_MBS}[kind]
    M = find_matrix(job_text, hdr, occurrence=occurrence)
    if M is None:
        return None
    net, gross = np.diag(M).copy(), M.sum(axis=1)
    return {"matrix": M, "net_pop": net, "gross_pop": gross,
            "shared_pop": gross - net, "kind": kind}


def mayer(job_text, occurrence=-1):
    """
    Split the Mayer block into its two physically distinct parts.

      valency      (N,)   diagonal -- atomic valency
      bond_orders  (N,N)  off-diagonal only, diagonal zeroed
    """
    M = find_matrix(job_text, H_MAYER, occurrence=occurrence)
    if M is None:
        return None
    val = np.diag(M).copy()
    BO = M.copy()
    np.fill_diagonal(BO, 0.0)
    return {"valency": val, "bond_orders": BO, "raw": M}


def gross_orbital_populations(job_text, kind="mbs"):
    """
    Parse 'Gross orbital populations' -> per-AO electron counts.

    Returns dict with:
      labels        (K,) str    '1S','2PX','6D 0','8F+1', ...
      atom_index    (K,) int    0-based atom each AO belongs to (forward-filled)
      atom_symbol   (K,) str
      total/alpha/beta/spin (K,) float
      s_pop,p_pop,d_pop,f_pop (N,) float   per-atom shell sums
    """
    hdr = H_GOP_MBS if kind == "mbs" else H_GOP_PLAIN
    lines = job_text.splitlines()
    hits = [i for i, ln in enumerate(lines) if hdr.search(ln)]
    if not hits:
        return None

    labels, atom_idx, atom_sym, cols = [], [], [], []
    cur_atom, cur_sym = None, None
    for ln in lines[hits[-1] + 1:]:
        if not ln.strip():
            break
        toks = ln.split()
        if toks[0] == "Total":                      # column header row
            continue
        if not toks[0].lstrip("+-").isdigit():
            break
        # forms: "<k> <atom#> <Sym> <label...> <nums>"  or  "<k> <label...> <nums>"
        rest = toks[1:]
        if len(rest) >= 2 and rest[0].isdigit() and rest[1].isalpha():
            cur_atom, cur_sym = int(rest[0]) - 1, rest[1]
            rest = rest[2:]
        nums = [_f(t) for t in rest if _is_value(t)]
        lab = [t for t in rest if not _is_value(t)]
        if not nums:
            break
        # 'D 0' / 'F+1' labels split across tokens -> rejoin with a space
        labels.append(" ".join(lab))
        atom_idx.append(cur_atom)
        atom_sym.append(cur_sym)
        cols.append(nums[:4] + [np.nan] * (4 - len(nums[:4])))

    if not labels:
        return None
    cols = np.array(cols, dtype=float)
    atom_idx = np.array(atom_idx, dtype=int)
    labels = np.array(labels)

    natoms = atom_idx.max() + 1
    shells = {}
    for letter in "SPDF":
        # angular momentum letter is the first alphabetic char after the shell number
        mask = np.array([re.match(r"\d+([A-Z])", lb) and
                         re.match(r"\d+([A-Z])", lb).group(1) == letter
                         for lb in labels])
        acc = np.zeros(natoms)
        for a in range(natoms):
            acc[a] = cols[mask & (atom_idx == a), 0].sum()
        shells[letter.lower() + "_pop"] = acc

    return {"labels": labels, "atom_index": atom_idx,
            "atom_symbol": np.array(atom_sym), "total": cols[:, 0],
            "alpha": cols[:, 1], "beta": cols[:, 2], "spin": cols[:, 3],
            **shells}


def dipole(job_text):
    """Total dipole (Debye) from the LAST 'Dipole moment' block of THIS job."""
    lines = job_text.splitlines()
    tot = np.nan
    for i, ln in enumerate(lines):
        if "Dipole moment" in ln and "Debye" in ln:
            for probe in lines[i + 1:i + 3]:
                m = RE_DIPOLE_TOT.search(probe)
                if m:
                    tot = float(m.group(1))
                    break
    return tot


def scf_energy(job_text):
    """Return (method, energy) from the last 'SCF Done' line of this job."""
    hits = RE_SCF.findall(job_text)
    return (hits[-1][0], float(hits[-1][1])) if hits else (None, np.nan)


def bonding_populations(job_text):
    """
    'Bonding Mulliken charges' are NOT charges -- they are bonding populations
    and do not sum to zero. Returned under an honest name.
    """
    lines = job_text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Bonding Mulliken charges"):
            vals = []
            for probe in lines[i + 2:]:
                toks = probe.split()
                if len(toks) >= 3 and toks[0].isdigit():
                    try:
                        vals.append(float(toks[2]))
                    except ValueError:
                        break
                elif vals:
                    break
            return {"bonding_population": np.array(vals)}
    return None
