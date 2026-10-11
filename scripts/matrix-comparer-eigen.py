#!/usr/bin/env python3
"""
Gaussian Output Matrix Parser & Comparator — with Eigenvalue Analysis
======================================================================
Parses Gaussian .log/.out files and extracts:
  1. Mulliken condensed-to-atoms (MBS) matrix
  2. Mayer bond order matrix (pair-list and/or condensed)
  3. Mayer valence / free valence (per-atom vectors)
  4. Wiberg bond index matrix (from NBO)
  5. Overlap matrix (*** Overlap ***) condensed to atoms
  6. Lowdin condensed-to-atoms matrix
  7. Alpha/Beta condensed matrices (open-shell)
  8. Bonding population condensed matrix (pop=bonding)

Computes scalar magnitude metrics AND full eigenvalue analysis for each
matrix, then produces UMAP-style color comparison plots.

Usage:
    python gaussian_matrix_compare.py file1.log file2.log file3.log ...
    python gaussian_matrix_compare.py *.log
"""

import re
import sys
import os
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not found. Install with: pip install matplotlib")
    print("         Text-only output will be used.\n")

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ============================================================================
# PARSING ENGINE
# ============================================================================

def parse_condensed_matrix(lines, start_idx, n_atoms):
    """
    Parse a Gaussian NxN matrix printed in column blocks of 5-6 columns.
    Works for condensed-to-atoms and full basis-function matrices.

    Gaussian format:
                 1          2          3          4          5
      1  O  8.240928   0.272300   0.273918  -0.025955   0.001285
      2  H  0.272300   0.338679  -0.021509  -0.013372  -0.000111
    """
    matrix = np.zeros((n_atoms, n_atoms))
    idx = start_idx
    cols_parsed = 0

    while cols_parsed < n_atoms and idx < len(lines):
        line = lines[idx].strip()

        col_header_match = re.match(r'^\s*(\d+(?:\s+\d+)*)\s*$', line)
        if col_header_match:
            col_indices = [int(x) - 1 for x in col_header_match.group().split()]
            idx += 1

            for row in range(n_atoms):
                if idx >= len(lines):
                    break
                data_line = lines[idx].strip()
                if not data_line or re.match(r'^[A-Za-z]', data_line):
                    break

                parts = re.split(r'\s+', data_line)
                values = []
                row_idx = None
                for p in parts:
                    try:
                        val = float(p)
                        if row_idx is None and val == int(val) and 1 <= int(val) <= n_atoms:
                            row_idx = int(val) - 1
                        else:
                            values.append(val)
                    except ValueError:
                        continue

                if row_idx is not None and values:
                    for ci, val in zip(col_indices, values):
                        if ci < n_atoms and row_idx < n_atoms:
                            matrix[row_idx, ci] = val
                idx += 1

            cols_parsed = max(cols_parsed,
                              max(col_indices) + 1 if col_indices else cols_parsed)
        else:
            # Stop if we hit a section that clearly isn't matrix data
            if line and not re.match(r'^\s*\d', line) and cols_parsed > 0:
                break
            idx += 1

    return matrix


def parse_overlap_matrix(lines, start_idx, n_basis):
    """
    Parse the full basis-function overlap matrix (*** Overlap ***).
    Returns the n_basis x n_basis matrix.
    """
    return parse_condensed_matrix(lines, start_idx, n_basis)


def condense_to_atoms(basis_matrix, basis_to_atom, n_atoms):
    """
    Condense a basis-function matrix (Nbasis x Nbasis) to an
    atom-condensed matrix (Natom x Natom) by summing blocks.
    """
    condensed = np.zeros((n_atoms, n_atoms))
    for i in range(basis_matrix.shape[0]):
        for j in range(basis_matrix.shape[1]):
            ai = basis_to_atom[i]
            aj = basis_to_atom[j]
            condensed[ai, aj] += basis_matrix[i, j]
    return condensed


def parse_mayer_bond_orders_pairlist(lines, start_idx, n_atoms):
    """
    Parse Mayer bond orders printed in pair-list format:
      1-  2  O - H   0.9123
      1-  3  O - H   0.9087
      2-  3  H - H   0.0012
    
    Also handles the format:
      Mayer atomic bond orders:
                          1          2          3
        1  O      ...     0.272     0.273
        2  H      0.272   ...      -0.021
    (which is the condensed block format -- delegate to parse_condensed_matrix)
    """
    matrix = np.zeros((n_atoms, n_atoms))
    idx = start_idx
    found_any = False

    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            if found_any:
                break
            idx += 1
            continue

        # Match pair format: "1-  2  O - H   0.9123" or "1  2  0.9123"
        m = re.match(
            r'\s*(\d+)\s*-?\s*(\d+)\s+[A-Za-z]*\s*-?\s*[A-Za-z]*\s+'
            r'(-?[\d.]+(?:[eEdD][+-]?\d+)?)',
            line
        )
        if m:
            i = int(m.group(1)) - 1
            j = int(m.group(2)) - 1
            val = float(m.group(3).replace('D', 'E').replace('d', 'e'))
            if 0 <= i < n_atoms and 0 <= j < n_atoms:
                matrix[i, j] = val
                matrix[j, i] = val
                found_any = True
            idx += 1
            continue

        # Simpler pair format: "1  2  0.9123"
        m2 = re.match(r'\s*(\d+)\s+(\d+)\s+(-?[\d.]+(?:[eEdD][+-]?\d+)?)', line)
        if m2:
            i = int(m2.group(1)) - 1
            j = int(m2.group(2)) - 1
            val = float(m2.group(3).replace('D', 'E').replace('d', 'e'))
            if 0 <= i < n_atoms and 0 <= j < n_atoms:
                matrix[i, j] = val
                matrix[j, i] = val
                found_any = True
            idx += 1
            continue

        # If we hit a column-header line, it's a condensed block format
        if re.match(r'^\s*\d+(\s+\d+)+\s*$', line):
            return parse_condensed_matrix(lines, idx, n_atoms)

        # Non-matching line: stop
        if found_any:
            break
        idx += 1

    return matrix if found_any else None


def parse_wiberg_nbo_matrix(lines, start_idx, n_atoms):
    """
    Parse Wiberg bond index matrix from NBO output.
    Format:
      Wiberg bond index matrix in the NAO basis:
      
          Atom    1       2       3       4
          ----  ------  ------  ------  ------
        1.  O   0.0000  0.8241  0.8230  0.0010
        2.  H   0.8241  0.0000  0.0056  0.0001
    """
    matrix = np.zeros((n_atoms, n_atoms))
    idx = start_idx
    cols_parsed = 0

    while cols_parsed < n_atoms and idx < len(lines):
        line = lines[idx].strip()

        # Skip "Atom" header and dashes
        if re.match(r'^\s*Atom\s+', line) or re.match(r'^\s*-+', line):
            idx += 1
            continue

        # Column header: just numbers
        col_match = re.match(r'^\s*(\d+(?:\s+\d+)*)\s*$', line)
        if col_match:
            col_indices = [int(x) - 1 for x in col_match.group().split()]
            idx += 1

            for row in range(n_atoms):
                if idx >= len(lines):
                    break
                data_line = lines[idx].strip()
                if not data_line or re.match(r'^\s*Atom\s+', data_line):
                    break
                if re.match(r'^\s*-+', data_line):
                    idx += 1
                    continue

                # Format: "1.  O   0.0000  0.8241 ..." or "1  O  0.0000 0.8241"
                nums = re.findall(r'-?[\d.]+(?:[eE][+-]?\d+)?', data_line)
                if len(nums) >= 2:
                    row_idx = int(float(nums[0])) - 1
                    values = [float(x) for x in nums[1:]]
                    for ci, val in zip(col_indices, values):
                        if 0 <= ci < n_atoms and 0 <= row_idx < n_atoms:
                            matrix[row_idx, ci] = val
                idx += 1

            cols_parsed = max(cols_parsed,
                              max(col_indices) + 1 if col_indices else cols_parsed)
        else:
            if cols_parsed > 0:
                break
            idx += 1

    return matrix


def find_atom_count(lines):
    """Extract number of atoms from the Gaussian output."""
    for line in lines:
        m = re.search(r'NAtoms=\s*(\d+)', line)
        if m:
            return int(m.group(1))
    in_orient = False
    count = 0
    for line in lines:
        if 'Standard orientation' in line or 'Input orientation' in line:
            in_orient = True
            count = 0
            continue
        if in_orient and '-----' in line:
            continue
        if in_orient and re.match(r'\s*\d+\s+\d+\s+\d+', line):
            count += 1
        elif in_orient and count > 0 and not line.strip():
            return count
    return count


def find_n_basis(lines):
    """Extract number of basis functions."""
    for line in lines:
        m = re.search(r'(\d+)\s+basis functions', line)
        if m:
            return int(m.group(1))
        m2 = re.search(r'NBasis=\s*(\d+)', line)
        if m2:
            return int(m2.group(1))
    return 0


def find_basis_to_atom_map(lines, n_atoms, n_basis):
    """
    Build a mapping from basis function index -> atom index.
    Uses the gross orbital populations section or shell info.
    Falls back to equal distribution if not found.
    """
    # Try to find from "Gross orbital populations" or shell-to-atom map
    for i, line in enumerate(lines):
        if 'Shell to atom map' in line or 'Gross orbital populations' in line:
            # Read the shell-to-atom map
            mapping = []
            j = i + 1
            while j < len(lines) and len(mapping) < n_basis:
                nums = re.findall(r'\d+', lines[j])
                if nums:
                    mapping.extend([int(x) - 1 for x in nums])
                else:
                    break
                j += 1
            if len(mapping) >= n_basis:
                return mapping[:n_basis]

    # Fallback: distribute basis functions equally across atoms
    per_atom = max(1, n_basis // n_atoms)
    mapping = []
    for a in range(n_atoms):
        count = per_atom if a < n_atoms - 1 else (n_basis - len(mapping))
        mapping.extend([a] * count)
    return mapping[:n_basis]


ELEMENT_MAP = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
    23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
    30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
    37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
    44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba',
    72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
    79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi',
}


def find_atom_labels(lines):
    """Extract atom labels from standard/input orientation."""
    labels = []
    in_orient = False
    dash_count = 0
    for line in lines:
        if 'Standard orientation' in line or 'Input orientation' in line:
            in_orient = True
            dash_count = 0
            labels = []
            continue
        if in_orient:
            if '-----' in line:
                dash_count += 1
                if dash_count >= 3:
                    in_orient = False
                continue
            m = re.match(r'\s*(\d+)\s+(\d+)\s+\d+', line)
            if m:
                atomic_num = int(m.group(2))
                labels.append(ELEMENT_MAP.get(atomic_num, f'X{atomic_num}'))
    return labels


def parse_mayer_valence(lines, n_atoms):
    """
    Parse Mayer total valences and free valences (per-atom vectors).
    Format:
      Mayer total valence:
        1  O   1.8234
        2  H   0.9123
      Mayer free valence:
        1  O   0.0012
        2  H   0.0005
    """
    total_valence = None
    free_valence = None

    for i, line in enumerate(lines):
        if re.search(r'total\s+(atomic\s+)?valence', line, re.IGNORECASE):
            vals = np.zeros(n_atoms)
            j = i + 1
            count = 0
            while j < len(lines) and count < n_atoms:
                m = re.match(r'\s*(\d+)\s+[A-Za-z]+\s+(-?[\d.]+)', lines[j])
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < n_atoms:
                        vals[idx] = float(m.group(2))
                        count += 1
                elif count > 0:
                    break
                j += 1
            if count > 0:
                total_valence = vals

        if re.search(r'free\s+valence', line, re.IGNORECASE):
            vals = np.zeros(n_atoms)
            j = i + 1
            count = 0
            while j < len(lines) and count < n_atoms:
                m = re.match(r'\s*(\d+)\s+[A-Za-z]+\s+(-?[\d.]+)', lines[j])
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < n_atoms:
                        vals[idx] = float(m.group(2))
                        count += 1
                elif count > 0:
                    break
                j += 1
            if count > 0:
                free_valence = vals

    return total_valence, free_valence


def parse_gaussian_file(filepath):
    """Parse a single Gaussian output file for all relevant matrices."""
    with open(filepath, 'r', errors='replace') as f:
        lines = f.readlines()

    n_atoms = find_atom_count(lines)
    n_basis = find_n_basis(lines)
    atom_labels = find_atom_labels(lines)

    if n_atoms == 0:
        print(f"  Warning: Could not determine atom count in {filepath}")
        return None

    result = {
        'filename': os.path.basename(filepath),
        'filepath': str(filepath),
        'n_atoms': n_atoms,
        'n_basis': n_basis,
        'atom_labels': atom_labels if atom_labels else [f'A{i+1}' for i in range(n_atoms)],
        'matrices': {},
        'vectors': {},
    }

    # --- 1. Mulliken condensed-to-atoms (MBS) ---
    for i, line in enumerate(lines):
        if 'Condensed to atoms (all electrons)' in line:
            mat = parse_condensed_matrix(lines, i + 1, n_atoms)
            if np.any(mat != 0):
                result['matrices']['MBS_Mulliken'] = mat

    # --- 2. Mayer bond orders (multiple possible formats) ---
    for i, line in enumerate(lines):
        # Format A: "Mayer atomic bond orders:" followed by pair list or block
        if re.search(r'Mayer\s+atomic\s+bond\s+order', line, re.IGNORECASE):
            mat = parse_mayer_bond_orders_pairlist(lines, i + 1, n_atoms)
            if mat is not None and np.any(mat != 0):
                result['matrices']['Mayer_bond_order'] = mat

        # Format B: "Mayer bond order matrix" as condensed block
        elif re.search(r'Mayer\s+bond\s+order\s+matrix', line, re.IGNORECASE):
            for j in range(i + 1, min(i + 5, len(lines))):
                if re.match(r'\s*\d+(\s+\d+)*\s*$', lines[j].strip()):
                    mat = parse_condensed_matrix(lines, j, n_atoms)
                    if np.any(mat != 0):
                        result['matrices']['Mayer_bond_order'] = mat
                    break

    # --- 3. Mayer valence / free valence vectors ---
    total_val, free_val = parse_mayer_valence(lines, n_atoms)
    if total_val is not None:
        result['vectors']['Mayer_total_valence'] = total_val
    if free_val is not None:
        result['vectors']['Mayer_free_valence'] = free_val

    # --- 4. Lowdin condensed-to-atoms ---
    for i, line in enumerate(lines):
        if re.search(r'Lowdin\s+.*condensed\s+to\s+atoms', line, re.IGNORECASE):
            mat = parse_condensed_matrix(lines, i + 1, n_atoms)
            if np.any(mat != 0):
                result['matrices']['Lowdin_condensed'] = mat

    # --- 5. Wiberg bond index matrix (NBO) ---
    for i, line in enumerate(lines):
        if 'Wiberg bond index matrix in the NAO basis' in line:
            mat = parse_wiberg_nbo_matrix(lines, i + 1, n_atoms)
            if np.any(mat != 0):
                result['matrices']['Wiberg_bond_index'] = mat

    # --- 6. Wiberg bond index (Lowdin orthogonalized AO basis) ---
    for i, line in enumerate(lines):
        if re.search(r'Wiberg bond index.*Lowdin', line, re.IGNORECASE):
            mat = parse_wiberg_nbo_matrix(lines, i + 1, n_atoms)
            if np.any(mat != 0):
                result['matrices']['Wiberg_Lowdin'] = mat

    # --- 7. Overlap matrix (*** Overlap ***) -> condensed to atoms ---
    if n_basis > 0:
        for i, line in enumerate(lines):
            if '*** Overlap ***' in line:
                basis_map = find_basis_to_atom_map(lines, n_atoms, n_basis)
                overlap_basis = parse_overlap_matrix(lines, i + 1, n_basis)
                if np.any(overlap_basis != 0):
                    overlap_condensed = condense_to_atoms(
                        overlap_basis, basis_map, n_atoms)
                    result['matrices']['Overlap_condensed'] = overlap_condensed
                break  # only parse the first occurrence

    # --- 8. Alpha/Beta condensed (open-shell) ---
    for i, line in enumerate(lines):
        if re.search(r'Alpha.*condensed\s+to\s+atoms', line, re.IGNORECASE):
            mat = parse_condensed_matrix(lines, i + 1, n_atoms)
            if np.any(mat != 0):
                result['matrices']['Alpha_condensed'] = mat
        if re.search(r'Beta.*condensed\s+to\s+atoms', line, re.IGNORECASE):
            mat = parse_condensed_matrix(lines, i + 1, n_atoms)
            if np.any(mat != 0):
                result['matrices']['Beta_condensed'] = mat

    # --- 9. Bonding population condensed (pop=bonding) ---
    for i, line in enumerate(lines):
        if re.search(r'bonding.*condensed\s+to\s+atoms', line, re.IGNORECASE):
            mat = parse_condensed_matrix(lines, i + 1, n_atoms)
            if np.any(mat != 0):
                result['matrices']['Bonding_pop_condensed'] = mat

    # --- 10. Mayer orbital populations (if present as matrix) ---
    for i, line in enumerate(lines):
        if re.search(r'Mayer.*orbital.*population', line, re.IGNORECASE):
            for j in range(i, min(i + 5, len(lines))):
                if re.match(r'\s*\d+(\s+\d+)*\s*$', lines[j].strip()):
                    mat = parse_condensed_matrix(lines, j, n_atoms)
                    if np.any(mat != 0):
                        result['matrices']['Mayer_orbital_pop'] = mat
                    break

    if not result['matrices']:
        print(f"  Warning: No condensed matrices found in {filepath}")
        print(f"           Keywords needed: pop=full, pop=bonding, pop=nboread")

    return result


# ============================================================================
# EIGENVALUE ANALYSIS ENGINE
# ============================================================================

def compute_eigenvalues(matrix):
    """Compute eigenvalues, sorted descending."""
    if np.allclose(matrix, matrix.T, atol=1e-8):
        eigs = np.linalg.eigvalsh(matrix)
    else:
        eigs = np.real(np.linalg.eigvals(matrix))
    return np.sort(eigs)[::-1]


def compute_eigen_metrics(eigenvalues):
    """Derive physically meaningful metrics from the eigenvalue spectrum."""
    eigs = eigenvalues
    abs_eigs = np.abs(eigs)
    n = len(eigs)

    sum2 = np.sum(eigs ** 2)
    sum4 = np.sum(eigs ** 4)
    participation_ratio = (sum2 ** 2) / sum4 if sum4 > 1e-30 else n

    prob = abs_eigs ** 2
    total = np.sum(prob)
    if total > 1e-30:
        prob = prob / total
        prob = prob[prob > 1e-30]
        spectral_entropy = -np.sum(prob * np.log(prob))
        max_entropy = np.log(n)
        normalized_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0
    else:
        spectral_entropy = 0.0
        normalized_entropy = 0.0

    spectral_gap = abs(eigs[0] - eigs[1]) if n >= 2 else 0.0
    spectral_width = eigs[0] - eigs[-1]

    threshold = 0.05 * abs_eigs[0] if abs_eigs[0] > 1e-30 else 1e-10
    n_significant = int(np.sum(abs_eigs > threshold))
    effective_rank = np.sum(abs_eigs) / abs_eigs[0] if abs_eigs[0] > 1e-30 else 0

    sorted_abs = np.sort(abs_eigs)
    cumulative = np.cumsum(sorted_abs)
    total_sum = cumulative[-1]
    gini = (1 - 2 * np.sum(cumulative) / (n * total_sum) + 1 / n) if total_sum > 1e-30 else 0.0

    n_positive = int(np.sum(eigs > 1e-10))
    n_negative = int(np.sum(eigs < -1e-10))
    pos_neg_ratio = n_positive / max(n_negative, 1)
    sum_positive = np.sum(eigs[eigs > 0])
    sum_negative = np.sum(eigs[eigs < 0])

    return {
        'lambda_max': eigs[0],
        'lambda_min': eigs[-1],
        'lambda_2nd': eigs[1] if n >= 2 else 0.0,
        'Spectral gap': spectral_gap,
        'Spectral width': spectral_width,
        'Participation ratio': participation_ratio,
        'Spectral entropy': spectral_entropy,
        'Normalized entropy': normalized_entropy,
        'N significant eig': n_significant,
        'Effective rank': effective_rank,
        'Gini coefficient': gini,
        'N positive eig': n_positive,
        'N negative eig': n_negative,
        'Pos/Neg ratio': pos_neg_ratio,
        'Sum positive eig': sum_positive,
        'Sum negative eig': sum_negative,
        'Mean |eig|': np.mean(abs_eigs),
        'Std |eig|': np.std(abs_eigs),
    }


# ============================================================================
# SCALAR METRICS ENGINE
# ============================================================================

def compute_metrics(matrix):
    """Compute scalar metrics for a single NxN matrix."""
    n = matrix.shape[0]
    flat = matrix.flatten()
    mask = ~np.eye(n, dtype=bool)
    off_diag = matrix[mask]
    diag = np.diag(matrix)

    return {
        'Frobenius norm': np.linalg.norm(matrix, 'fro'),
        'Mean |value|': np.mean(np.abs(flat)),
        'Max |value|': np.max(np.abs(flat)),
        'Trace': np.trace(matrix),
        'RMS': np.sqrt(np.mean(flat ** 2)),
        'Spectral radius': np.max(np.abs(compute_eigenvalues(matrix))),
        'Off-diag mean': np.mean(np.abs(off_diag)),
        'Off-diag max': np.max(np.abs(off_diag)),
        'Diag mean': np.mean(diag),
        'Diag std': np.std(diag),
        'Sum (total)': np.sum(matrix),
        'Condition number': np.linalg.cond(matrix)
            if abs(np.linalg.det(matrix)) > 1e-30 else np.inf,
    }


def compute_feature_vector(matrix):
    """Build feature vector combining scalar + eigenvalue features."""
    m = compute_metrics(matrix)
    eigs = compute_eigenvalues(matrix)
    em = compute_eigen_metrics(eigs)

    return np.array([
        m['Frobenius norm'], m['Mean |value|'], m['Max |value|'],
        m['Trace'], m['RMS'], m['Spectral radius'],
        m['Off-diag mean'], m['Off-diag max'],
        m['Diag mean'], m['Sum (total)'],
        em['lambda_max'], em['lambda_min'], em['Spectral gap'],
        em['Spectral width'], em['Participation ratio'],
        em['Normalized entropy'], em['Effective rank'],
        em['Gini coefficient'], em['Pos/Neg ratio'],
        em['Sum positive eig'], em['Sum negative eig'],
    ])


# ============================================================================
# TEXT OUTPUT
# ============================================================================

def fmt_val(val):
    """Format a numeric value for table display."""
    if isinstance(val, int):
        return f"{val:>14d}"
    if abs(val) > 1e6 or (abs(val) < 1e-3 and val != 0):
        return f"{val:>14.4e}"
    return f"{val:>14.4f}"


def print_comparison_table(all_data, matrix_type):
    entries = [(d['filename'], d['matrices'][matrix_type])
               for d in all_data if matrix_type in d['matrices']]
    if not entries:
        return

    print(f"\n{'='*80}")
    print(f"  SCALAR METRICS: {matrix_type}")
    print(f"{'='*80}")

    all_metrics = [(fn, compute_metrics(mat)) for fn, mat in entries]
    header = f"{'Metric':<22}" + ''.join(f"  {fn[:14]:>14}" for fn, _ in all_metrics)
    print(header)
    print("-" * len(header))

    for mn in all_metrics[0][1]:
        row = f"{mn:<22}" + ''.join(f"  {fmt_val(met[mn])}" for _, met in all_metrics)
        print(row)


def print_eigenvalue_table(all_data, matrix_type):
    entries = [(d['filename'], d['matrices'][matrix_type])
               for d in all_data if matrix_type in d['matrices']]
    if not entries:
        return

    print(f"\n{'='*80}")
    print(f"  EIGENVALUE ANALYSIS: {matrix_type}")
    print(f"{'='*80}")

    all_em = []
    all_eigs = []
    for fn, mat in entries:
        eigs = compute_eigenvalues(mat)
        all_em.append((fn, compute_eigen_metrics(eigs)))
        all_eigs.append((fn, eigs))

    header = f"{'Eigen-Metric':<22}" + ''.join(f"  {fn[:14]:>14}" for fn, _ in all_em)
    print(header)
    print("-" * len(header))

    for mn in all_em[0][1]:
        row = f"{mn:<22}" + ''.join(f"  {fmt_val(em[mn])}" for _, em in all_em)
        print(row)

    max_show = min(10, min(len(e) for _, e in all_eigs))
    print(f"\n  Top {max_show} eigenvalues:")
    header2 = f"{'#':<5}" + ''.join(f"  {fn[:14]:>14}" for fn, _ in all_eigs)
    print(header2)
    print("-" * len(header2))
    for k in range(max_show):
        row = f"{k+1:<5}" + ''.join(f"  {eigs[k]:>14.6f}" for _, eigs in all_eigs)
        print(row)
    print()


def print_vector_table(all_data, vec_name):
    entries = [(d['filename'], d['vectors'][vec_name], d['atom_labels'])
               for d in all_data if vec_name in d.get('vectors', {})]
    if not entries:
        return

    print(f"\n{'='*80}")
    print(f"  VECTOR: {vec_name}")
    print(f"{'='*80}")

    header = f"{'Atom':<8}" + ''.join(f"  {fn[:14]:>14}" for fn, _, _ in entries)
    print(header)
    print("-" * len(header))

    n_atoms = max(len(v) for _, v, _ in entries)
    for k in range(n_atoms):
        labels = entries[0][2]
        label = labels[k] if k < len(labels) else f'A{k+1}'
        row = f"{k+1:>3} {label:<4}"
        for _, vec, _ in entries:
            row += f"  {vec[k]:>14.6f}" if k < len(vec) else f"  {'N/A':>14}"
        print(row)

    # Summary stats
    print(f"\n  Summary:")
    print(f"  {'Statistic':<14}" + ''.join(f"  {fn[:14]:>14}" for fn, _, _ in entries))
    for stat_name, stat_fn in [('Mean', np.mean), ('Std', np.std),
                                ('Min', np.min), ('Max', np.max), ('Sum', np.sum)]:
        row = f"  {stat_name:<14}"
        for _, vec, _ in entries:
            row += f"  {stat_fn(vec):>14.6f}"
        print(row)
    print()


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison(all_data):
    """Generate full comparison plots."""
    if not HAS_MPL:
        return

    all_types = sorted(set(mt for d in all_data for mt in d['matrices']))
    if not all_types:
        print("No matrices found to plot.")
        return

    filenames = [d['filename'] for d in all_data]
    n_files = len(filenames)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_files, 3)))

    # Global style maps used across ALL figures
    file_markers = ['o', 's', 'D', '^', 'v', 'P', 'X', 'h', '*', 'p']
    file_colors_global = plt.cm.Set1(np.linspace(0, 0.9, max(n_files, 2)))
    type_cmap = plt.cm.tab10
    type_color = {mt: type_cmap(i / max(len(all_types)-1, 1))
                  for i, mt in enumerate(all_types)}
    file_marker = {fn: file_markers[i % len(file_markers)]
                   for i, fn in enumerate(filenames)}
    file_color = {fn: file_colors_global[i] for i, fn in enumerate(filenames)}

    # ---- FIGURE 0: Standalone Legend Reference Sheet ----
    fig_leg = plt.figure(figsize=(8, max(3, (n_files + len(all_types)) * 0.45 + 2)))
    ax_l = fig_leg.add_subplot(111)
    ax_l.axis('off')
    ax_l.set_title('Legend Reference', fontsize=16, fontweight='bold', pad=20)

    handles = []

    # Files section
    handles.append(plt.Line2D([], [], linestyle='None', marker='None',
                               label='FILES  (identified by marker shape + line color)'))
    for i, fn in enumerate(filenames):
        mk = file_markers[i % len(file_markers)]
        handles.append(plt.Line2D(
            [0], [0], marker=mk, color=colors[i % len(colors)],
            markeredgecolor='black', markersize=10, linewidth=2,
            linestyle='-', label=f'  {fn}'))

    handles.append(plt.Line2D([], [], linestyle='None', marker='None', label=' '))

    # Matrix types section
    handles.append(plt.Line2D([], [], linestyle='None', marker='None',
                               label='MATRIX TYPES  (identified by color in dashboard)'))
    for mt in all_types:
        handles.append(plt.Line2D(
            [0], [0], marker='s', color=type_color[mt],
            markeredgecolor='black', markersize=12,
            linestyle='None', label=f'  {mt}'))

    handles.append(plt.Line2D([], [], linestyle='None', marker='None', label=' '))

    # Plot index
    handles.append(plt.Line2D([], [], linestyle='None', marker='None',
                               label='FIGURES'))
    handles.append(plt.Line2D([], [], linestyle='None', marker='None',
                               label='  Fig 1: Eigenvalue spectra + cumulative'))
    handles.append(plt.Line2D([], [], linestyle='None', marker='None',
                               label='  Fig 2: Eigenvalue-derived metrics (bars)'))
    handles.append(plt.Line2D([], [], linestyle='None', marker='None',
                               label='  heatmap_<type>.png: Per-matrix heatmaps'))
    handles.append(plt.Line2D([], [], linestyle='None', marker='None',
                               label='  Fig 4: PCA/UMAP embedding'))
    handles.append(plt.Line2D([], [], linestyle='None', marker='None',
                               label='  Fig 5: Physical interpretation dashboard'))
    handles.append(plt.Line2D([], [], linestyle='None', marker='None',
                               label='  Fig 6: Per-atom valence comparison'))

    ax_l.legend(handles=handles, loc='center', fontsize=11,
                frameon=True, framealpha=1.0, edgecolor='#999999',
                fancybox=True, borderpad=1.5, labelspacing=0.9,
                handletextpad=1.0, handlelength=2.5)

    fig_leg.savefig('legend_reference.png', dpi=150, bbox_inches='tight',
                     facecolor='white')
    print("  Saved: legend_reference.png")

    # ---- FIGURE 1: Eigenvalue spectra ----
    n_types = len(all_types)
    fig1, axes1 = plt.subplots(2, n_types, figsize=(6 * n_types, 10), squeeze=False)
    fig1.suptitle('Eigenvalue Analysis Across Gaussian Files',
                  fontsize=14, fontweight='bold')

    for ti, mtype in enumerate(all_types):
        ax_spec = axes1[0, ti]
        ax_cum = axes1[1, ti]

        entries = [(d['filename'], d['matrices'][mtype])
                   for d in all_data if mtype in d['matrices']]
        if not entries:
            ax_spec.text(0.5, 0.5, 'No data', transform=ax_spec.transAxes,
                         ha='center', color='gray')
            continue

        for fi, (fname, mat) in enumerate(entries):
            eigs = compute_eigenvalues(mat)
            c = colors[fi % len(colors)]
            ax_spec.plot(range(1, len(eigs)+1), eigs, 'o-',
                         color=c, label=fname, markersize=5, linewidth=1.5)
            ax_spec.axhline(y=0, color='gray', linestyle='--', alpha=0.4)

            abs_eigs = np.abs(eigs)
            total = np.sum(abs_eigs)
            if total > 0:
                cum = np.cumsum(abs_eigs) / total * 100
                ax_cum.plot(range(1, len(eigs)+1), cum, 's-',
                            color=c, label=fname, markersize=4, linewidth=1.5)

        ax_spec.set_title(f'{mtype}\nEigenvalue Spectrum', fontsize=10, fontweight='bold')
        ax_spec.set_xlabel('Eigenvalue index')
        ax_spec.set_ylabel('Eigenvalue')
        ax_spec.grid(alpha=0.3)

        ax_cum.set_title(f'{mtype}\nCumulative |eig| Distribution',
                         fontsize=10, fontweight='bold')
        ax_cum.set_xlabel('Number of eigenvalues')
        ax_cum.set_ylabel('Cumulative % of total |eig|')
        ax_cum.axhline(y=90, color='red', linestyle=':', alpha=0.5)
        ax_cum.grid(alpha=0.3)
        ax_cum.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig('eigenvalue_spectra.png', dpi=300, bbox_inches='tight')
    print("  Saved: eigenvalue_spectra.png")

    # ---- FIGURE 2: Eigenvalue-derived metrics bars ----
    eigen_keys = ['Participation ratio', 'Normalized entropy',
                  'Gini coefficient', 'Effective rank', 'Spectral gap']

    fig2, axes2 = plt.subplots(n_types, 1, figsize=(12, 4*n_types), squeeze=False)
    fig2.suptitle('Eigenvalue-Derived Metrics', fontsize=14, fontweight='bold', y=1.01)

    for ti, mtype in enumerate(all_types):
        ax = axes2[ti, 0]
        entries = [(d['filename'], d['matrices'][mtype])
                   for d in all_data if mtype in d['matrices']]
        if not entries:
            ax.set_title(mtype)
            continue

        x = np.arange(len(eigen_keys))
        width = 0.8 / len(entries)
        for fi, (fname, mat) in enumerate(entries):
            em = compute_eigen_metrics(compute_eigenvalues(mat))
            vals = [em[k] for k in eigen_keys]
            offset = (fi - len(entries)/2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=fname,
                   color=colors[fi % len(colors)], edgecolor='white', linewidth=0.5)

        ax.set_title(mtype, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(eigen_keys, rotation=25, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('eigenvalue_metrics_bars.png', dpi=300, bbox_inches='tight')
    print("  Saved: eigenvalue_metrics_bars.png")

    # ---- CSV EXPORT: Per-matrix-type metric tables ----
    scalar_csv_keys = ['Frobenius norm', 'Mean |value|', 'Off-diag mean',
                       'Trace', 'RMS']
    eigen_csv_keys = ['lambda_max', 'Spectral gap', 'Participation ratio',
                      'Normalized entropy', 'Gini coefficient',
                      'Effective rank', 'Sum positive eig', 'Sum negative eig']

    for mtype in all_types:
        entries = [(d['filename'], d['matrices'][mtype])
                   for d in all_data if mtype in d['matrices']]
        if not entries:
            continue

        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', mtype)
        csv_path = f'metrics_{safe_name}.csv'

        header_cols = (['File'] +
                       [f'[S] {k}' for k in scalar_csv_keys] +
                       [f'[E] {k}' for k in eigen_csv_keys])

        rows = []
        for fn, mat in entries:
            sm = compute_metrics(mat)
            em = compute_eigen_metrics(compute_eigenvalues(mat))
            row = [fn]
            for k in scalar_csv_keys:
                row.append(f'{sm[k]:.8f}')
            for k in eigen_csv_keys:
                row.append(f'{em[k]:.8f}')
            rows.append(row)

        with open(csv_path, 'w') as f:
            f.write(','.join(header_cols) + '\n')
            for row in rows:
                f.write(','.join(row) + '\n')

        print(f"  Saved: {csv_path}")

    # ---- FIGURE 3: Per-matrix-type heatmaps (one PNG each) ----
    #
    # Each heatmap shows:
    #   Rows    = Gaussian output files
    #   Columns = [S] scalar + [E] eigenvalue metrics for ONE matrix type
    #   Color   = column-normalized (blue=min across files, red=max)

    scalar_keys = ['Frobenius norm', 'Mean |value|', 'Off-diag mean', 'Trace', 'RMS']
    eigen_hm_keys = ['lambda_max', 'Spectral gap', 'Participation ratio',
                     'Normalized entropy', 'Gini coefficient',
                     'Effective rank', 'Sum positive eig', 'Sum negative eig']

    scalar_short = ['Frob. norm', 'Mean |val|', 'Off-diag\nmean', 'Trace', 'RMS']
    eigen_short = ['eig_max', 'Spec. gap', 'Partic.\nratio', 'Norm.\nentropy',
                   'Gini\ncoeff.', 'Eff. rank', 'Sum(+eig)', 'Sum(-eig)']

    all_short = scalar_short + eigen_short
    all_keys_s = scalar_keys
    all_keys_e = eigen_hm_keys
    n_s = len(scalar_keys)
    n_e = len(eigen_hm_keys)
    n_metrics = n_s + n_e

    for mtype in all_types:
        entries = [(d['filename'], d['matrices'][mtype])
                   for d in all_data if mtype in d['matrices']]
        if len(entries) < 2:
            continue

        fnames = [fn for fn, _ in entries]
        nr = len(fnames)

        # Build data matrix: rows=files, cols=metrics
        dm = np.zeros((nr, n_metrics))
        for ri, (fn, mat) in enumerate(entries):
            sm = compute_metrics(mat)
            em = compute_eigen_metrics(compute_eigenvalues(mat))
            for ci, km in enumerate(all_keys_s):
                dm[ri, ci] = sm[km]
            for ci, km in enumerate(all_keys_e):
                dm[ri, n_s + ci] = em[km]

        # Column-normalize
        cmin = dm.min(axis=0)
        cmax = dm.max(axis=0)
        crange = cmax - cmin
        crange[crange == 0] = 1
        normed_dm = (dm - cmin) / crange

        # Sizing: fixed cell dimensions
        cell_w = 1.0
        cell_h = 0.6
        margin_l = 2.2
        margin_b = 1.8
        margin_t = 1.6
        margin_r = 1.6

        fw = margin_l + n_metrics * cell_w + margin_r
        fh = margin_t + nr * cell_h + margin_b
        fw = max(10, min(20, fw))
        fh = max(4, min(12, fh))

        fig_hm, ax_hm = plt.subplots(figsize=(fw, fh))
        fig_hm.subplots_adjust(
            left=margin_l / fw,
            bottom=margin_b / fh,
            right=1.0 - margin_r / fw,
            top=1.0 - margin_t / fh
        )

        im = ax_hm.imshow(normed_dm, cmap='RdYlBu_r', aspect='auto',
                           vmin=0, vmax=1)

        # Column labels
        col_tags = [f'[S] {s}' for s in scalar_short] + \
                   [f'[E] {s}' for s in eigen_short]
        ax_hm.set_xticks(range(n_metrics))
        ax_hm.set_xticklabels(col_tags, rotation=45, ha='right', fontsize=8)

        # Row labels
        ax_hm.set_yticks(range(nr))
        ax_hm.set_yticklabels(fnames, fontsize=9)

        # Cell values
        cell_fs = max(6, min(9, int(70 / max(n_metrics, 1))))
        for i in range(nr):
            for j in range(n_metrics):
                val = dm[i, j]
                txt = f'{val:.3f}' if abs(val) < 100 else f'{val:.2e}'
                ax_hm.text(j, i, txt, ha='center', va='center',
                           fontsize=cell_fs,
                           color='white' if normed_dm[i, j] > 0.6 else 'black')

        # Separator between scalar and eigenvalue groups
        ax_hm.axvline(x=n_s - 0.5, color='black', linewidth=1.5, alpha=0.8)

        # Group labels at the top
        ax_hm.text((n_s - 1) / 2.0, -0.7,
                   'SCALAR METRICS', ha='center', va='bottom',
                   fontsize=9, fontweight='bold', color='#444444',
                   transform=ax_hm.get_xaxis_transform())
        ax_hm.text(n_s + (n_e - 1) / 2.0, -0.7,
                   'EIGENVALUE METRICS', ha='center', va='bottom',
                   fontsize=9, fontweight='bold', color='#444444',
                   transform=ax_hm.get_xaxis_transform())

        # Colorbar
        cbar = plt.colorbar(im, ax=ax_hm, shrink=0.8, pad=0.02, aspect=25)
        cbar.set_label('Column-normalized\n(0=min, 1=max)', fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        # Title
        ax_hm.set_title(
            f'{mtype}\n'
            f'Blue = lowest value across files  |  Red = highest',
            fontsize=11, fontweight='bold', pad=25
        )

        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', mtype)
        fname_out = f'heatmap_{safe_name}.png'
        plt.savefig(fname_out, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig_hm)
        print(f"  Saved: {fname_out}")

    # ---- FIGURE 4: PCA/UMAP embedding ----
    if n_files >= 3:
        feature_vectors = {}
        for mtype in all_types:
            vecs, labs = [], []
            for d in all_data:
                if mtype in d['matrices']:
                    vecs.append(compute_feature_vector(d['matrices'][mtype]))
                    labs.append(d['filename'])
            if len(vecs) >= 3:
                feature_vectors[mtype] = (np.array(vecs), labs)

        if feature_vectors:
            n_embed = len(feature_vectors)
            fig4, axes4 = plt.subplots(1, n_embed, figsize=(7*n_embed, 6), squeeze=False)
            fig4.suptitle('Dimensionality Reduction (Scalar + Eigenvalue Features)',
                          fontsize=13, fontweight='bold')

            for ei, (mtype, (vecs, labs)) in enumerate(feature_vectors.items()):
                ax = axes4[0, ei]
                if HAS_SKLEARN:
                    vecs_s = StandardScaler().fit_transform(vecs)
                else:
                    vecs_s = (vecs - vecs.mean(0)) / (vecs.std(0) + 1e-10)

                if HAS_UMAP and len(vecs) >= 4:
                    coords = umap.UMAP(n_components=2,
                                        n_neighbors=min(3, len(vecs)-1),
                                        random_state=42).fit_transform(vecs_s)
                    method = 'UMAP'
                elif HAS_SKLEARN:
                    coords = PCA(n_components=2).fit_transform(vecs_s)
                    method = 'PCA'
                else:
                    coords = vecs_s[:, :2]
                    method = '2D Proj'

                markers = ['o', 's', 'D', '^', 'v', 'P', 'X', 'h', '*', 'p']
                lcols = plt.cm.Set1(np.linspace(0, 0.9, len(labs)))
                for k, lab in enumerate(labs):
                    ax.scatter(coords[k, 0], coords[k, 1],
                               c=[lcols[k]], s=80,alpha=0.35,marker=markers[k % len(markers)],
                               edgecolors='black', linewidth=1.2, zorder=5, label=lab)

                ax.set_title(f'{mtype}\n({method})', fontsize=10)
                ax.set_xlabel(f'{method} dim 1')
                ax.set_ylabel(f'{method} dim 2')

            plt.tight_layout()
            plt.savefig('embedding_with_eigenvalues.png', dpi=300, bbox_inches='tight')
            print("  Saved: embedding_with_eigenvalues.png")

    # ---- FIGURE 5: Physical interpretation dashboard (no legends) ----
    fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
    fig5.suptitle('Physical Interpretation Dashboard  (see legend_reference.png)',
                  fontsize=14, fontweight='bold')

    ax_a = axes5[0, 0]
    ax_b = axes5[0, 1]
    ax_c = axes5[1, 0]
    ax_d = axes5[1, 1]

    # Style mapping for dashboard scatter
    seen_fn = set()
    unique_fn = []
    for d in all_data:
        if d['filename'] not in seen_fn:
            unique_fn.append(d['filename'])
            seen_fn.add(d['filename'])

    fmarker = {fn: file_markers[i % len(file_markers)]
               for i, fn in enumerate(unique_fn)}

    def scat(ax, xs, ys, labs, mtype):
        col = type_color[mtype]
        for k, lab in enumerate(labs):
            mk = fmarker[lab]
            ax.scatter(xs[k], ys[k], s=80, alpha=0.35, marker=mk, c=[col],
                       edgecolors='black', linewidth=0.8, zorder=5)

    # (a) Delocalization
    for mt in all_types:
        prs, ents, labs = [], [], []
        for d in all_data:
            if mt in d['matrices']:
                em = compute_eigen_metrics(compute_eigenvalues(d['matrices'][mt]))
                prs.append(em['Participation ratio'])
                ents.append(em['Normalized entropy'])
                labs.append(d['filename'])
        if prs:
            scat(ax_a, prs, ents, labs, mt)
    ax_a.set_xlabel('Participation Ratio  ->  more modes', fontsize=9)
    ax_a.set_ylabel('Normalized Entropy  ->  delocalized', fontsize=9)
    ax_a.set_title('(a) Bonding Delocalization', fontsize=11, fontweight='bold')
    ax_a.grid(alpha=0.25)

    # (b) Concentration
    for mt in all_types:
        ginis, ranks, labs = [], [], []
        for d in all_data:
            if mt in d['matrices']:
                em = compute_eigen_metrics(compute_eigenvalues(d['matrices'][mt]))
                ginis.append(em['Gini coefficient'])
                ranks.append(em['Effective rank'])
                labs.append(d['filename'])
        if ginis:
            scat(ax_b, ginis, ranks, labs, mt)
    ax_b.set_xlabel('Gini Coefficient  ->  concentrated', fontsize=9)
    ax_b.set_ylabel('Effective Rank  ->  more dimensions', fontsize=9)
    ax_b.set_title('(b) Bonding Concentration', fontsize=11, fontweight='bold')
    ax_b.grid(alpha=0.25)

    # (c) Bonding vs Antibonding
    for mt in all_types:
        ps, ns, labs = [], [], []
        for d in all_data:
            if mt in d['matrices']:
                em = compute_eigen_metrics(compute_eigenvalues(d['matrices'][mt]))
                ps.append(em['Sum positive eig'])
                ns.append(abs(em['Sum negative eig']))
                labs.append(d['filename'])
        if ps:
            scat(ax_c, ps, ns, labs, mt)
    xl = ax_c.get_xlim()[1]
    yl = ax_c.get_ylim()[1]
    lm = max(xl, yl) if xl > 0 and yl > 0 else 1.0
    ax_c.plot([0, lm], [0, lm], 'k--', alpha=0.25, linewidth=1)
    ax_c.set_xlabel('Sum(positive eig)  --  Bonding', fontsize=9)
    ax_c.set_ylabel('|Sum(negative eig)|  --  Antibonding', fontsize=9)
    ax_c.set_title('(c) Bonding vs Antibonding', fontsize=11, fontweight='bold')
    ax_c.grid(alpha=0.25)

    # (d) Dominant mode
    for mt in all_types:
        gaps, lms, labs = [], [], []
        for d in all_data:
            if mt in d['matrices']:
                em = compute_eigen_metrics(compute_eigenvalues(d['matrices'][mt]))
                gaps.append(em['Spectral gap'])
                lms.append(em['lambda_max'])
                labs.append(d['filename'])
        if gaps:
            scat(ax_d, lms, gaps, labs, mt)
    ax_d.set_xlabel('eig_max  (dominant interaction)', fontsize=9)
    ax_d.set_ylabel('Spectral Gap  (mode separation)', fontsize=9)
    ax_d.set_title('(d) Dominant Mode Analysis', fontsize=11, fontweight='bold')
    ax_d.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig('physical_interpretation.png', dpi=150, bbox_inches='tight')
    print("  Saved: physical_interpretation.png")

    # ---- FIGURE 6: Valence vectors comparison (if available) ----
    all_vec_names = sorted(set(vn for d in all_data for vn in d.get('vectors', {})))
    if all_vec_names and n_files >= 2:
        fig6, axes6 = plt.subplots(1, len(all_vec_names),
                                    figsize=(7*len(all_vec_names), 5), squeeze=False)
        fig6.suptitle('Per-Atom Valence Comparison', fontsize=13, fontweight='bold')

        for vi, vname in enumerate(all_vec_names):
            ax = axes6[0, vi]
            for fi, d in enumerate(all_data):
                if vname in d.get('vectors', {}):
                    vec = d['vectors'][vname]
                    labels = d['atom_labels']
                    x = np.arange(len(vec))
                    ax.bar(x + fi * 0.8 / n_files, vec, 0.8 / n_files,
                           label=d['filename'], color=colors[fi % len(colors)],
                           edgecolor='white', linewidth=0.3)

            ax.set_title(vname, fontsize=11, fontweight='bold')
            ax.set_xlabel('Atom')
            ax.set_ylabel('Value')
            if all_data[0]['atom_labels']:
                ax.set_xticks(np.arange(len(all_data[0]['atom_labels'])))
                ax.set_xticklabels(
                    [f"{i+1}{lb}" for i, lb in enumerate(all_data[0]['atom_labels'])],
                    rotation=45, ha='right', fontsize=7)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('valence_comparison.png', dpi=150, bbox_inches='tight')
        print("  Saved: valence_comparison.png")

    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Example:")
        print("  python gaussian_matrix_compare.py calc1.log calc2.log calc3.log")
        print("  python gaussian_matrix_compare.py *.out")
        print("\nGaussian keywords needed:")
        print("  pop=full          -> MBS Mulliken, Mayer bond orders, Lowdin")
        print("  pop=bonding       -> Bonding population matrix")
        print("  pop=nboread       -> Wiberg bond indices (add $nbo bndidx $end)")
        print("  iop(3/33=1)       -> Print overlap matrix")
        print("\nDependencies:")
        print("  Required: numpy")
        print("  Plotting: matplotlib")
        print("  Optional: scikit-learn, umap-learn")
        sys.exit(1)

    filepaths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_file():
            filepaths.append(p)
        else:
            print(f"  Skipping: {arg} (not found)")

    if not filepaths:
        print("No valid files found.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Gaussian Matrix Parser & Comparator")
    print(f"  (with Eigenvalue & Overlap Analysis)")
    print(f"  Processing {len(filepaths)} file(s)")
    print(f"{'='*60}\n")

    all_data = []
    for fp in filepaths:
        print(f"  Parsing: {fp.name} ...", end=" ")
        result = parse_gaussian_file(fp)
        if result:
            nm = len(result['matrices'])
            nv = len(result.get('vectors', {}))
            print(f"OK -- {result['n_atoms']} atoms, {result['n_basis']} basis fns, "
                  f"{nm} matrices, {nv} vectors")
            for mtype, mat in result['matrices'].items():
                print(f"    -> {mtype}: {mat.shape[0]}x{mat.shape[1]}")
            for vname, vec in result.get('vectors', {}).items():
                print(f"    -> {vname}: [{len(vec)} atoms]")
            all_data.append(result)
        else:
            print("FAILED")

    if not all_data:
        print("\nNo data extracted. Check your Gaussian output files.")
        sys.exit(1)

    all_types = sorted(set(mt for d in all_data for mt in d['matrices']))

    for mtype in all_types:
        print_comparison_table(all_data, mtype)

    for mtype in all_types:
        print_eigenvalue_table(all_data, mtype)

    all_vec_names = sorted(set(vn for d in all_data for vn in d.get('vectors', {})))
    for vname in all_vec_names:
        print_vector_table(all_data, vname)

    if HAS_MPL:
        print("\nGenerating plots...")
        plot_comparison(all_data)
    else:
        print("\nInstall matplotlib for visual output: pip install matplotlib")

    print("\nDone.")


if __name__ == '__main__':
    main()
