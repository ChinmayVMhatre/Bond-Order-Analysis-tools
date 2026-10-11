#!/usr/bin/env python3
"""
Gaussian Output Matrix Parser & Comparator
===========================================
Parses Gaussian .log/.out files and extracts:
  1. Mulliken Bond Sum (MBS) condensed-to-atoms matrix
  2. Mayer bond order matrix (Lowdin / Mayer)
  3. Wiberg bond index matrix (from NBO)
  4. Overlap matrix (condensed)
  5. Any additional condensed-to-atoms matrices

Computes scalar magnitude metrics for each matrix and produces
a UMAP-style color comparison plot.

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
    import matplotlib.colors as mcolors
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
    Parse a Gaussian condensed-to-atoms NxN matrix printed in column blocks.
    
    Gaussian prints large matrices in blocks of 5-6 columns at a time:
        1          2          3          4          5
    1  O  8.240928   0.272300   0.273918  -0.025955   0.001285
    2  H  0.272300   0.338679  -0.021509  -0.013372  -0.000111
    ...
    Then continues with columns 6, 7, ... in the next block.
    """
    matrix = np.zeros((n_atoms, n_atoms))
    idx = start_idx
    cols_parsed = 0

    while cols_parsed < n_atoms and idx < len(lines):
        line = lines[idx].strip()

        # Detect column header line (e.g., "1          2          3")
        col_header_match = re.match(r'^\s*(\d+(?:\s+\d+)*)\s*$', line)
        if col_header_match:
            col_indices = [int(x) - 1 for x in col_header_match.group().split()]
            idx += 1

            # Read data rows for this block
            for row in range(n_atoms):
                if idx >= len(lines):
                    break
                data_line = lines[idx].strip()
                if not data_line:
                    idx += 1
                    continue

                # Match: "1  O  8.240928  0.272300 ..." or "1  8.240928  0.272300 ..."
                # The atom label may or may not be present
                parts = re.split(r'\s+', data_line)
                
                # Find where the numeric values start
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
                        continue  # skip atom labels like "O", "H", etc.

                if row_idx is not None and values:
                    for ci, val in zip(col_indices, values):
                        if ci < n_atoms and row_idx < n_atoms:
                            matrix[row_idx, ci] = val

                idx += 1

            cols_parsed = max(cols_parsed, max(col_indices) + 1 if col_indices else cols_parsed)
        else:
            idx += 1

    return matrix


def find_atom_count(lines):
    """Extract number of atoms from the Gaussian output."""
    for line in lines:
        m = re.search(r'NAtoms=\s*(\d+)', line)
        if m:
            return int(m.group(1))
    # Fallback: count atoms from standard orientation
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


def find_atom_labels(lines):
    """Extract atom labels from standard/input orientation."""
    ELEMENT_MAP = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
        9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
        16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
        23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
        30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
        44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 77: 'Ir', 78: 'Pt',
        79: 'Au', 80: 'Hg',
    }
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


def parse_gaussian_file(filepath):
    """
    Parse a single Gaussian output file for all relevant condensed matrices.
    
    Returns dict: {
        'filename': str,
        'n_atoms': int,
        'atom_labels': list,
        'matrices': {
            'MBS_condensed': np.array or None,
            'Mayer_bond_order': np.array or None,
            'Wiberg_bond_index': np.array or None,
            ...
        }
    }
    """
    with open(filepath, 'r', errors='replace') as f:
        lines = f.readlines()

    n_atoms = find_atom_count(lines)
    atom_labels = find_atom_labels(lines)
    
    if n_atoms == 0:
        print(f"  Warning: Could not determine atom count in {filepath}")
        return None

    result = {
        'filename': os.path.basename(filepath),
        'filepath': str(filepath),
        'n_atoms': n_atoms,
        'atom_labels': atom_labels if atom_labels else [f'A{i+1}' for i in range(n_atoms)],
        'matrices': {}
    }

    # --- 1. Mulliken condensed-to-atoms (MBS) ---
    # Look for "Condensed to atoms (all electrons):"
    for i, line in enumerate(lines):
        if 'Condensed to atoms (all electrons)' in line:
            mat = parse_condensed_matrix(lines, i + 1, n_atoms)
            if np.any(mat != 0):
                result['matrices']['MBS_condensed_Mulliken'] = mat

    # --- 2. Mayer bond order matrix ---
    # "Mayer bond order" or "Lowdin bond orders" printed condensed to atoms
    for i, line in enumerate(lines):
        if re.search(r'Mayer\s+(atomic|bond)\s+order', line, re.IGNORECASE):
            # Search forward for the matrix block
            for j in range(i, min(i + 5, len(lines))):
                if re.match(r'\s*\d+(\s+\d+)*\s*$', lines[j].strip()):
                    mat = parse_condensed_matrix(lines, j, n_atoms)
                    if np.any(mat != 0):
                        result['matrices']['Mayer_bond_order'] = mat
                    break

    # --- 3. Lowdin condensed-to-atoms ---
    for i, line in enumerate(lines):
        if re.search(r'Lowdin\s+.*condensed\s+to\s+atoms', line, re.IGNORECASE):
            mat = parse_condensed_matrix(lines, i + 1, n_atoms)
            if np.any(mat != 0):
                result['matrices']['Lowdin_condensed'] = mat

    # --- 4. Wiberg bond index matrix (NBO) ---
    for i, line in enumerate(lines):
        if 'Wiberg bond index matrix in the NAO basis' in line:
            # Skip header lines until we hit the column numbers
            for j in range(i + 1, min(i + 10, len(lines))):
                if re.match(r'\s*Atom\s+', lines[j]):
                    continue
                if re.match(r'\s*----', lines[j]):
                    continue
                if re.match(r'\s*\d+', lines[j]):
                    mat = parse_condensed_matrix(lines, j - 1 if re.match(r'\s*\d+\s+\d+\s+\d+', lines[j-1]) else j, n_atoms)
                    if np.any(mat != 0):
                        result['matrices']['Wiberg_bond_index'] = mat
                    break

    # --- 5. Condensed to atoms (alpha/beta separately for open-shell) ---
    alpha_found = False
    for i, line in enumerate(lines):
        if 'Condensed to atoms (all electrons)' in line and not alpha_found:
            continue  # already parsed above
        if re.search(r'Alpha.*condensed to atoms', line, re.IGNORECASE):
            mat = parse_condensed_matrix(lines, i + 1, n_atoms)
            if np.any(mat != 0):
                result['matrices']['Alpha_condensed'] = mat
                alpha_found = True
        if re.search(r'Beta.*condensed to atoms', line, re.IGNORECASE):
            mat = parse_condensed_matrix(lines, i + 1, n_atoms)
            if np.any(mat != 0):
                result['matrices']['Beta_condensed'] = mat

    # --- 6. Mayer orbital populations ---
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
        print(f"           Make sure you used pop=full or equivalent keywords.")

    return result


# ============================================================================
# METRICS ENGINE
# ============================================================================

def compute_metrics(matrix):
    """Compute a suite of scalar metrics for a single NxN matrix."""
    n = matrix.shape[0]
    flat = matrix.flatten()
    
    # Off-diagonal elements (bonding interactions)
    mask = ~np.eye(n, dtype=bool)
    off_diag = matrix[mask]
    diag = np.diag(matrix)

    metrics = {
        'Frobenius norm': np.linalg.norm(matrix, 'fro'),
        'Mean |value|': np.mean(np.abs(flat)),
        'Max |value|': np.max(np.abs(flat)),
        'Trace': np.trace(matrix),
        'RMS': np.sqrt(np.mean(flat ** 2)),
        'Spectral radius': np.max(np.abs(np.linalg.eigvalsh(matrix))) if np.allclose(matrix, matrix.T) else np.max(np.abs(np.linalg.eigvals(matrix))),
        'Off-diag mean': np.mean(np.abs(off_diag)),
        'Off-diag max': np.max(np.abs(off_diag)),
        'Diag mean': np.mean(diag),
        'Diag std': np.std(diag),
        'Sum (total)': np.sum(matrix),
        'Condition number': np.linalg.cond(matrix) if np.linalg.det(matrix) != 0 else np.inf,
    }
    return metrics


def compute_feature_vector(matrix):
    """Create a feature vector from a matrix for embedding/comparison."""
    m = compute_metrics(matrix)
    return np.array([m['Frobenius norm'], m['Mean |value|'], m['Max |value|'],
                     m['Trace'], m['RMS'], m['Spectral radius'],
                     m['Off-diag mean'], m['Off-diag max'],
                     m['Diag mean'], m['Sum (total)']])


# ============================================================================
# VISUALIZATION
# ============================================================================

def print_comparison_table(all_data, matrix_type):
    """Print a text-based comparison table."""
    entries = [(d['filename'], d['matrices'][matrix_type])
               for d in all_data if matrix_type in d['matrices']]
    
    if not entries:
        print(f"\n  No files contain '{matrix_type}' matrix.\n")
        return

    print(f"\n{'='*80}")
    print(f"  COMPARISON: {matrix_type}")
    print(f"{'='*80}")

    all_metrics = []
    for fname, mat in entries:
        all_metrics.append((fname, compute_metrics(mat)))

    metric_names = list(all_metrics[0][1].keys())
    
    # Header
    header = f"{'Metric':<22}"
    for fname, _ in all_metrics:
        header += f"  {fname[:14]:>14}"
    print(header)
    print("-" * len(header))

    for mn in metric_names:
        row = f"{mn:<22}"
        for _, metrics in all_metrics:
            val = metrics[mn]
            if abs(val) > 1e6 or (abs(val) < 1e-3 and val != 0):
                row += f"  {val:>14.4e}"
            else:
                row += f"  {val:>14.4f}"
        print(row[:5])
    print()


def plot_comparison(all_data):
    """Generate matplotlib comparison plots."""
    if not HAS_MPL:
        return

    # Collect all matrix types across files
    all_types = set()
    for d in all_data:
        all_types.update(d['matrices'].keys())
    all_types = sorted(all_types)

    if not all_types:
        print("No matrices found to plot.")
        return

    n_types = len(all_types)
    filenames = [d['filename'] for d in all_data]
    n_files = len(filenames)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_files, 3)))

    # --- Figure 1: Bar chart comparison across metrics ---
    fig, axes = plt.subplots(n_types, 1, figsize=(12, 4 * n_types), squeeze=False)
    fig.suptitle('Matrix Magnitude Comparison Across Gaussian Files',
                 fontsize=14, fontweight='bold', y=1.01)

    key_metrics = ['Frobenius norm', 'Mean |value|', 'Off-diag mean', 'Trace', 'RMS']

    for ti, mtype in enumerate(all_types):
        ax = axes[ti, 0]
        entries = [(d['filename'], d['matrices'][mtype])
                   for d in all_data if mtype in d['matrices']]
        
        if not entries:
            ax.text(0.5, 0.5, f'No data for {mtype}', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_title(mtype)
            continue

        x = np.arange(len(key_metrics))
        width = 0.8 / len(entries)

        for fi, (fname, mat) in enumerate(entries):
            metrics = compute_metrics(mat)
            vals = [metrics[k] for k in key_metrics]
            offset = (fi - len(entries) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=fname,
                          color=colors[fi % len(colors)], edgecolor='white', linewidth=0.5)

        ax.set_title(mtype, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(key_metrics, rotation=25, ha='right', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('matrix_comparison_bars.png', dpi=300, bbox_inches='tight')
    print("  Saved: matrix_comparison_bars.png")

    # --- Figure 2: UMAP-style color map (heatmap of metrics) ---
    fig2, ax2 = plt.subplots(figsize=(10, max(3, n_files * 0.8 + 2)))

    # Build a matrix: rows = files, cols = (metric x matrix_type)
    col_labels = []
    data_matrix = []

    for d in all_data:
        row = []
        for mtype in all_types:
            if mtype in d['matrices']:
                m = compute_metrics(d['matrices'][mtype])
                for km in key_metrics:
                    row.append(m[km])
                    if len(data_matrix) == 0:
                        col_labels.append(f"{mtype[:12]}\n{km[:12]}")
            else:
                for km in key_metrics:
                    row.append(np.nan)
                    if len(data_matrix) == 0:
                        col_labels.append(f"{mtype[:12]}\n{km[:12]}")
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)
    
    # Normalize columns for color mapping
    with np.errstate(invalid='ignore'):
        col_min = np.nanmin(data_matrix, axis=0)
        col_max = np.nanmax(data_matrix, axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1
        normed = (data_matrix - col_min) / col_range

    im = ax2.imshow(normed, cmap='RdBu', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(col_labels)))
    ax2.set_xticklabels(col_labels, rotation=60, ha='right', fontsize=7)
    ax2.set_yticks(range(n_files))
    ax2.set_yticklabels(filenames, fontsize=7)
    ax2.set_title('UMAP-Style Color Comparison\n(Normalized metrics across files)',
                   fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Relative magnitude (0=low, 1=high)', shrink=0.8)

    # Add text annotations
    for i in range(n_files):
        for j in range(len(col_labels)):
            val = data_matrix[i, j]
            if not np.isnan(val):
                txt = f'{val:.2f}' if abs(val) < 100 else f'{val:.1e}'
                ax2.text(j, i, txt, ha='center', va='center', fontsize=6,
                         color='white' if normed[i, j] > 0.6 else 'black')

    plt.tight_layout()
    plt.savefig('matrix_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    print("  Saved: matrix_comparison_heatmap.png")

    # --- Figure 3: PCA / UMAP embedding if multiple files ---
    if n_files >= 3:
        feature_vectors = {}
        for mtype in all_types:
            vecs = []
            labels = []
            for d in all_data:
                if mtype in d['matrices']:
                    vecs.append(compute_feature_vector(d['matrices'][mtype]))
                    labels.append(d['filename'])
            if len(vecs) >= 3:
                feature_vectors[mtype] = (np.array(vecs), labels)

        if feature_vectors:
            n_embed = len(feature_vectors)
            fig3, axes3 = plt.subplots(1, n_embed, figsize=(6 * n_embed, 5), squeeze=False)
            fig3.suptitle('Dimensionality Reduction: Files in Feature Space',
                          fontsize=13, fontweight='bold')

            for ei, (mtype, (vecs, labs)) in enumerate(feature_vectors.items()):
                ax = axes3[0, ei]
                
                # Standardize
                scaler = StandardScaler() if HAS_SKLEARN else None
                if scaler:
                    vecs_scaled = scaler.fit_transform(vecs)
                else:
                    vecs_scaled = (vecs - vecs.mean(0)) / (vecs.std(0) + 1e-10)

                # Try UMAP, fall back to PCA
                if HAS_UMAP and len(vecs) >= 4:
                    reducer = umap.UMAP(n_components=2, n_neighbors=min(3, len(vecs)-1),
                                         random_state=42)
                    coords = reducer.fit_transform(vecs_scaled)
                    method = 'UMAP'
                elif HAS_SKLEARN:
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(vecs_scaled)
                    method = 'PCA'
                else:
                    # Manual 2D: just use first two features
                    coords = vecs_scaled[:, :2]
                    method = '2D Projection'

                # Color by Frobenius norm
                frob_vals = vecs[:, 0]
                sc = ax.scatter(coords[:, 0], coords[:, 1], c=frob_vals,
                                cmap='coolwarm',alpha=0.4, s=50, edgecolors='black', linewidth=1.5,
                                zorder=5)
                for k, lab in enumerate(labs):
                    ax.annotate(lab[:15], (coords[k, 0], coords[k, 1]),
                                textcoords="offset points", xytext=(8, 8),
                                fontsize=8, fontweight='bold')

                ax.set_title(f'{mtype}\n({method}, colored by Frobenius norm)', fontsize=10)
                ax.set_xlabel(f'{method} dim 1')
                ax.set_ylabel(f'{method} dim 2')
                plt.colorbar(sc, ax=ax, label='Frobenius norm', shrink=0.8)

            plt.tight_layout()
            plt.savefig('matrix_comparison_embedding.png', dpi=300, bbox_inches='tight')
            print("  Saved: matrix_comparison_embedding.png")

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
    print(f"  Processing {len(filepaths)} file(s)")
    print(f"{'='*60}\n")

    all_data = []
    for fp in filepaths:
        print(f"  Parsing: {fp.name} ...", end=" ")
        result = parse_gaussian_file(fp)
        if result:
            n_mat = len(result['matrices'])
            print(f"OK — {result['n_atoms']} atoms, {n_mat} matrix type(s) found")
            for mtype in result['matrices']:
                shape = result['matrices'][mtype].shape
                print(f"    -> {mtype}: {shape[0]}x{shape[1]}")
            all_data.append(result)
        else:
            print("FAILED")

    if not all_data:
        print("\nNo data extracted. Check your Gaussian output files.")
        sys.exit(1)

    # Print text comparison tables
    all_types = set()
    for d in all_data:
        all_types.update(d['matrices'].keys())

    for mtype in sorted(all_types):
        print_comparison_table(all_data, mtype)

    # Generate plots
    if HAS_MPL:
        print("\nGenerating plots...")
        plot_comparison(all_data)
    else:
        print("\nInstall matplotlib for visual output: pip install matplotlib")
        print("Optional: pip install scikit-learn umap-learn")

    print("\nDone.")


if __name__ == '__main__':
    main()
