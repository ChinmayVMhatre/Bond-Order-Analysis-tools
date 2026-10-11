"""
Molecular Graph Embedding Pipeline
====================================
Reads the unified QM9+BEBOP HDF5 file and generates molecular graph
embeddings for visualization in low-dimensional space.

Approach: Hybrid multi-channel descriptor
------------------------------------------
Each molecule is converted into a fixed-length numerical vector (~30 dims)
by combining three information channels:

  1. EDGE CHANNEL (BEBOP spectral):
     - Build weighted graph Laplacians using BEBOP net sigma and pi energies
     - Extract top-k eigenvalues from each Laplacian
     - Sigma eigenvalues encode the sigma bonding network topology
     - Pi eigenvalues separate aromatic from saturated compounds

  2. NODE CHANNEL (Gaussian orbital):
     - Aggregate per-atom features into fixed-length molecular summaries
     - Mean/std of S, P, D, F orbital populations
     - Mean/std/range of Mulliken charges
     - Captures hybridization character and charge distribution

  3. GLOBAL CHANNEL:
     - SCF energy, BEBOP atomization energy
     - Atom count, bond count
     - Molecular composition (counts of C, H, N, O, F)

The concatenated descriptor is z-normalized, then projected to 2D/3D
using UMAP (default), t-SNE, or PCA.

Usage:
------
  # Generate embeddings and plot (default UMAP, color by atomization energy)
  python mol_embedding.py --hdf5 qm9_data.h5 --output embeddings/

  # Use t-SNE instead, color by element composition
  python mol_embedding.py --hdf5 qm9_data.h5 --method tsne --color-by composition

  # Limit to first 5000 molecules for quick test
  python mol_embedding.py --hdf5 qm9_data.h5 --limit 5000 --output test_embed/

  # 3D projection
  python mol_embedding.py --hdf5 qm9_data.h5 --dims 3

  # Export descriptors as CSV (no plotting)
  python mol_embedding.py --hdf5 qm9_data.h5 --export-csv descriptors.csv

  # Inspect descriptor composition for a single molecule
  python mol_embedding.py --hdf5 qm9_data.h5 --inspect-mol dsgdb9nsd_000200

Requirements:
-------------
  pip install h5py numpy scipy scikit-learn matplotlib umap-learn hdbscan

Graph representation:
---------------------
  Nodes = atoms
    Features: Z, Mulliken charge, S/P/D/F orbital populations, self-energy

  Edges = covalent bonds
    Features: bond length, net sigma/pi/total energy, gross sigma/pi energy

  Global: SCF energy, atomization energy, composition

HDF5 input schema (from qm9_to_hdf5.py):
-----------------------------------------
  /molecules/<mol_id>/geometry/coordinates       (N,3) float64
  /molecules/<mol_id>/geometry/atomic_numbers    (N,)  int32
  /molecules/<mol_id>/geometry/bond_indices      (M,2) int32
  /molecules/<mol_id>/geometry/bond_lengths      (M,)  float64
  /molecules/<mol_id>/geometry/.attrs[scf_energy] float64
  /molecules/<mol_id>/mulliken/charges           (N,)  float64
  /molecules/<mol_id>/mulliken/orbital_populations/s_population (N,) float64
  /molecules/<mol_id>/mulliken/orbital_populations/p_population (N,) float64
  /molecules/<mol_id>/mulliken/orbital_populations/d_population (N,) float64
  /molecules/<mol_id>/mulliken/orbital_populations/f_population (N,) float64
  /molecules/<mol_id>/bond_energies/matrices/net_sigma     (N,N) float64
  /molecules/<mol_id>/bond_energies/matrices/net_pi        (N,N) float64
  /molecules/<mol_id>/bond_energies/matrices/net_total     (N,N) float64
  /molecules/<mol_id>/bond_energies/matrices/gross_sigma   (N,N) float64
  /molecules/<mol_id>/bond_energies/matrices/gross_pi      (N,N) float64
  /molecules/<mol_id>/bond_energies/matrices/composite_table (N,N) float64
  /molecules/<mol_id>/bond_energies/.attrs[atomization_energy] float64

Output:
-------
  embeddings/
    descriptors.npy          -- (N_mol, ~30) raw descriptor matrix
    embeddings_2d.npy        -- (N_mol, 2) projected coordinates
    mol_ids.txt              -- molecule IDs in order
    metadata.npz             -- all color-by arrays for plotting
    scatter_atomization.png  -- 2D scatter colored by atomization energy
    scatter_composition.png  -- 2D scatter colored by element composition
    scatter_pi_fraction.png  -- 2D scatter colored by pi bonding fraction
    scatter_clusters.png     -- 2D scatter colored by HDBSCAN clusters
    README.txt               -- description of all output files
"""

import os
import sys
import h5py
import numpy as np
import argparse
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ATOMIC_NUMBERS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
ELEMENT_INDEX = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_laplacian_eigenvalues(matrix, k=8):
    """
    Compute top-k smallest non-trivial eigenvalues of the graph Laplacian
    built from an NxN energy matrix.

    The Laplacian is: L = D - W
    where W[i,j] = |energy[i,j]| (absolute value, since bond energies are negative)
    and D is the degree matrix.

    Returns k eigenvalues, zero-padded if the molecule has fewer than k+1 atoms.
    """
    n = matrix.shape[0]
    # Build symmetric weight matrix from lower triangle
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            val = matrix[i, j] if matrix[i, j] != 0 else matrix[j, i]
            w = abs(val)
            W[i, j] = w
            W[j, i] = w

    # If all weights are zero, return zeros
    if np.all(W == 0):
        return np.zeros(k)

    D = np.diag(W.sum(axis=1))
    L = D - W

    try:
        eigenvalues = np.linalg.eigvalsh(L)
    except np.linalg.LinAlgError:
        return np.zeros(k)

    # Sort and take smallest non-trivial (skip first which is ~0)
    eigenvalues = np.sort(np.abs(eigenvalues))
    # Skip the first eigenvalue (always ~0 for connected graphs)
    nontrivial = eigenvalues[1:]

    # Zero-pad to length k
    result = np.zeros(k)
    actual = min(len(nontrivial), k)
    result[:actual] = nontrivial[:actual]
    return result


def extract_edge_features(mol_group, k=8):
    """
    Extract spectral edge features from BEBOP matrices.

    Returns:
        sigma_eigs: (k,) eigenvalues from sigma Laplacian
        pi_eigs:    (k,) eigenvalues from pi Laplacian
        edge_stats: (6,) summary stats of bond energies
                    [mean_net_total, std_net_total, min_net_total,
                     mean_gross_total, pi_fraction, sigma_fraction]
    """
    sigma_eigs = np.zeros(k)
    pi_eigs = np.zeros(k)
    edge_stats = np.zeros(6)

    if "bond_energies/matrices" not in mol_group:
        return sigma_eigs, pi_eigs, edge_stats

    matrices = mol_group["bond_energies/matrices"]

    # Sigma Laplacian
    if "net_sigma" in matrices:
        net_sigma = matrices["net_sigma"][:]
        sigma_eigs = compute_laplacian_eigenvalues(net_sigma, k)

    # Pi Laplacian
    if "net_pi" in matrices:
        net_pi = matrices["net_pi"][:]
        pi_eigs = compute_laplacian_eigenvalues(net_pi, k)

    # Edge summary statistics
    if "net_total" in matrices:
        net_total = matrices["net_total"][:]
        n = net_total.shape[0]
        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                v = net_total[i, j] if net_total[i, j] != 0 else net_total[j, i]
                if v != 0:
                    vals.append(v)
        if vals:
            va = np.array(vals)
            edge_stats[0] = va.mean()
            edge_stats[1] = va.std()
            edge_stats[2] = va.min()

    if "gross_total" in matrices:
        gross_total = matrices["gross_total"][:]
        n = gross_total.shape[0]
        gvals = []
        for i in range(n):
            for j in range(i + 1, n):
                v = gross_total[i, j] if gross_total[i, j] != 0 else gross_total[j, i]
                if v != 0:
                    gvals.append(v)
        if gvals:
            edge_stats[3] = np.mean(gvals)

    # Pi fraction: what fraction of total bonding is pi?
    if "net_sigma" in matrices and "net_pi" in matrices:
        ns = matrices["net_sigma"][:]
        np2 = matrices["net_pi"][:]
        sig_sum = abs(ns[ns != 0].sum()) if np.any(ns != 0) else 0
        pi_sum = abs(np2[np2 != 0].sum()) if np.any(np2 != 0) else 0
        total = sig_sum + pi_sum
        if total > 0:
            edge_stats[4] = pi_sum / total   # pi fraction
            edge_stats[5] = sig_sum / total   # sigma fraction

    return sigma_eigs, pi_eigs, edge_stats


def extract_node_features(mol_group):
    """
    Extract aggregated node features from Gaussian orbital populations.

    Returns:
        node_stats: (12,) array
            [mean_s, std_s, mean_p, std_p, mean_d, std_d, mean_f, std_f,
             mean_q, std_q, min_q, max_q]
    """
    node_stats = np.zeros(12)

    # Orbital populations
    if "mulliken/orbital_populations" in mol_group:
        op = mol_group["mulliken/orbital_populations"]
        for i, key in enumerate(["s_population", "p_population", "d_population", "f_population"]):
            if key in op:
                arr = op[key][:]
                node_stats[i * 2] = arr.mean()
                node_stats[i * 2 + 1] = arr.std()

    # Mulliken charges
    if "mulliken/charges" in mol_group:
        q = mol_group["mulliken/charges"][:]
        node_stats[8] = q.mean()
        node_stats[9] = q.std()
        node_stats[10] = q.min()
        node_stats[11] = q.max()

    return node_stats


def extract_global_features(mol_group):
    """
    Extract global molecular features.

    Returns:
        global_feats: (11,) array
            [scf_energy, atomization_energy, n_atoms, n_bonds,
             n_C, n_H, n_N, n_O, n_F, mean_bond_length, std_bond_length]
    """
    gf = np.zeros(11)

    # SCF energy
    if "geometry" in mol_group:
        gf[0] = mol_group["geometry"].attrs.get("scf_energy", 0.0)

    # Atomization energy
    if "bond_energies" in mol_group:
        gf[1] = mol_group["bond_energies"].attrs.get("atomization_energy", 0.0)

    # Atom counts
    if "geometry/atomic_numbers" in mol_group:
        atoms = mol_group["geometry/atomic_numbers"][:]
        gf[2] = len(atoms)
        counts = Counter(int(z) for z in atoms)
        gf[4] = counts.get(6, 0)   # C
        gf[5] = counts.get(1, 0)   # H
        gf[6] = counts.get(7, 0)   # N
        gf[7] = counts.get(8, 0)   # O
        gf[8] = counts.get(9, 0)   # F

    # Bond stats
    if "geometry/bond_lengths" in mol_group:
        bl = mol_group["geometry/bond_lengths"][:]
        gf[3] = len(bl)
        if len(bl) > 0:
            gf[9] = bl.mean()
            gf[10] = bl.std()

    return gf


def extract_descriptor(mol_group, k_eigs=8):
    """
    Extract the full molecular descriptor by concatenating all channels.

    Returns:
        descriptor: (k_eigs*2 + 6 + 12 + 11,) = ~41 dim vector
    """
    sigma_eigs, pi_eigs, edge_stats = extract_edge_features(mol_group, k=k_eigs)
    node_stats = extract_node_features(mol_group)
    global_feats = extract_global_features(mol_group)
    return np.concatenate([sigma_eigs, pi_eigs, edge_stats, node_stats, global_feats])


def get_descriptor_labels(k_eigs=8):
    """Return human-readable labels for each descriptor dimension."""
    labels = []
    for i in range(k_eigs):
        labels.append("sigma_eig_%d" % (i + 1))
    for i in range(k_eigs):
        labels.append("pi_eig_%d" % (i + 1))
    labels.extend(["mean_net_total", "std_net_total", "min_net_total",
                    "mean_gross_total", "pi_fraction", "sigma_fraction"])
    labels.extend(["mean_s_pop", "std_s_pop", "mean_p_pop", "std_p_pop",
                    "mean_d_pop", "std_d_pop", "mean_f_pop", "std_f_pop",
                    "mean_charge", "std_charge", "min_charge", "max_charge"])
    labels.extend(["scf_energy", "atomization_energy", "n_atoms", "n_bonds",
                    "n_C", "n_H", "n_N", "n_O", "n_F",
                    "mean_bond_length", "std_bond_length"])
    return labels


# ---------------------------------------------------------------------------
# Embedding and visualization
# ---------------------------------------------------------------------------

def reduce_dimensions(descriptors, method="umap", n_dims=2, random_state=42):
    """
    Project descriptors to low-dimensional space.

    Args:
        descriptors: (N, D) array, z-normalized
        method: 'umap', 'tsne', or 'pca'
        n_dims: 2 or 3
    Returns:
        embedding: (N, n_dims) array
    """
    if method == "umap":
        try:
            import umap
        except ImportError:
            log.error("umap-learn not installed. Install with: pip install umap-learn")
            sys.exit(1)
        reducer = umap.UMAP(
            n_components=n_dims,
            n_neighbors=30,
            min_dist=0.3,
            metric="euclidean",
            random_state=random_state,
        )
        return reducer.fit_transform(descriptors)

    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=n_dims,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=random_state,
        )
        return reducer.fit_transform(descriptors)

    elif method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_dims, random_state=random_state)
        return reducer.fit_transform(descriptors)

    else:
        raise ValueError("Unknown method: %s. Use 'umap', 'tsne', or 'pca'" % method)


def cluster_embeddings(embedding, min_cluster_size=50):
    """
    Cluster the 2D/3D embedding using HDBSCAN.
    Returns integer cluster labels (-1 = noise).
    """
    try:
        import hdbscan
    except ImportError:
        log.warning("hdbscan not installed, using KMeans instead. pip install hdbscan")
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=10, random_state=42, n_init=10)
        return km.fit_predict(embedding)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10)
    return clusterer.fit_predict(embedding)


def compute_coloring_arrays(mol_ids, hdf5_path):
    """
    Compute arrays for coloring the scatter plot by different properties.

    Returns dict of {name: (values_array, colormap_name, label_string)}
    """
    colors = {}
    n = len(mol_ids)

    atomization = np.zeros(n)
    pi_frac = np.zeros(n)
    n_atoms = np.zeros(n)
    n_heavy = np.zeros(n)
    max_element = []  # dominant non-H element

    with h5py.File(hdf5_path, "r") as f:
        for i, mid in enumerate(mol_ids):
            mol = f["molecules/" + mid]

            if "bond_energies" in mol:
                atomization[i] = mol["bond_energies"].attrs.get("atomization_energy", 0.0)

            if "bond_energies/matrices/net_sigma" in mol and "bond_energies/matrices/net_pi" in mol:
                ns = mol["bond_energies/matrices/net_sigma"][:]
                np2 = mol["bond_energies/matrices/net_pi"][:]
                ss = abs(ns[ns != 0].sum()) if np.any(ns != 0) else 0
                ps = abs(np2[np2 != 0].sum()) if np.any(np2 != 0) else 0
                total = ss + ps
                pi_frac[i] = ps / total if total > 0 else 0

            if "geometry/atomic_numbers" in mol:
                atoms = mol["geometry/atomic_numbers"][:]
                n_atoms[i] = len(atoms)
                counts = Counter(int(z) for z in atoms)
                n_heavy[i] = sum(v for k, v in counts.items() if k != 1)
                # Dominant heavy element
                heavy = {k: v for k, v in counts.items() if k != 1}
                if heavy:
                    dom = max(heavy, key=heavy.get)
                    max_element.append(ATOMIC_NUMBERS.get(dom, "X"))
                else:
                    max_element.append("H")

    # Map element to integer for coloring
    elem_set = sorted(set(max_element))
    elem_map = {e: i for i, e in enumerate(elem_set)}
    elem_int = np.array([elem_map[e] for e in max_element])

    colors["atomization"] = (atomization, "RdYlBu_r", "Atomization energy (kcal/mol)")
    colors["pi_fraction"] = (pi_frac, "magma", "Pi bonding fraction")
    colors["n_atoms"] = (n_atoms, "viridis", "Number of atoms")
    colors["n_heavy"] = (n_heavy, "plasma", "Number of heavy atoms")
    colors["composition"] = (elem_int, "tab10", "Dominant heavy element")
    colors["_elem_labels"] = (max_element, elem_set, elem_map)

    return colors


def plot_2d_scatter(embedding, values, cmap, label, title, filepath,
                    elem_labels=None, elem_set=None):
    """Generate a publication-quality 2D scatter plot."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if elem_labels is not None and elem_set is not None:
        # Categorical coloring
        for i, elem in enumerate(elem_set):
            mask = np.array([e == elem for e in elem_labels])
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       s=2, alpha=0.4, label=elem)
        ax.legend(markerscale=5, fontsize=10, framealpha=0.8)
    else:
        sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=values, cmap=cmap, s=2, alpha=0.4)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(label, fontsize=11)

    ax.set_xlabel("Dimension 1", fontsize=11)
    ax.set_ylabel("Dimension 2", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(filepath, dpi=500, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", filepath)


# ---------------------------------------------------------------------------
# Single molecule inspector
# ---------------------------------------------------------------------------

def inspect_molecule(hdf5_path, mol_id, k_eigs=8):
    """Print the full descriptor breakdown for a single molecule."""
    with h5py.File(hdf5_path, "r") as f:
        if mol_id not in f["molecules"]:
            print("Molecule not found: " + mol_id)
            return
        mol = f["molecules/" + mol_id]
        desc = extract_descriptor(mol, k_eigs=k_eigs)
        labels = get_descriptor_labels(k_eigs)

    print("=" * 60)
    print("  MOLECULAR DESCRIPTOR: " + mol_id)
    print("=" * 60)
    print()
    print("  %-25s  %12s" % ("Feature", "Value"))
    print("  " + "-" * 40)

    section = ""
    for lbl, val in zip(labels, desc):
        # Section headers
        if lbl.startswith("sigma_eig") and section != "edge_sigma":
            section = "edge_sigma"
            print()
            print("  EDGE CHANNEL: Sigma Laplacian eigenvalues")
            print("  " + "-" * 40)
        elif lbl.startswith("pi_eig") and section != "edge_pi":
            section = "edge_pi"
            print()
            print("  EDGE CHANNEL: Pi Laplacian eigenvalues")
            print("  " + "-" * 40)
        elif lbl == "mean_net_total" and section != "edge_stats":
            section = "edge_stats"
            print()
            print("  EDGE CHANNEL: Bond energy statistics")
            print("  " + "-" * 40)
        elif lbl == "mean_s_pop" and section != "node":
            section = "node"
            print()
            print("  NODE CHANNEL: Orbital population stats")
            print("  " + "-" * 40)
        elif lbl == "scf_energy" and section != "global":
            section = "global"
            print()
            print("  GLOBAL CHANNEL: Molecular properties")
            print("  " + "-" * 40)

        print("  %-25s  %12.6f" % (lbl, val))

    print()
    print("  Total descriptor length: %d" % len(desc))
    print("=" * 60)


# ---------------------------------------------------------------------------
# Write output README
# ---------------------------------------------------------------------------

def write_output_readme(output_dir, method, n_dims, n_mols, n_features):
    """Write a README.txt describing all output files."""
    lines = []
    lines.append("=" * 60)
    lines.append("Molecular Graph Embedding Output")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Generated by: mol_embedding.py")
    lines.append("Reduction method: %s" % method)
    lines.append("Dimensions: %d" % n_dims)
    lines.append("Molecules: %d" % n_mols)
    lines.append("Descriptor dimensions: %d" % n_features)
    lines.append("")
    lines.append("Files:")
    lines.append("-" * 40)
    lines.append("")
    lines.append("descriptors.npy")
    lines.append("  Raw molecular descriptors before dimensionality reduction.")
    lines.append("  Shape: (n_molecules, %d)" % n_features)
    lines.append("  Each row is one molecule. Columns are z-normalized features.")
    lines.append("  See descriptor_labels.txt for column names.")
    lines.append("")
    lines.append("descriptor_labels.txt")
    lines.append("  One label per line, matching columns of descriptors.npy.")
    lines.append("  Groups: sigma_eig_*, pi_eig_*, bond stats, orbital stats, globals.")
    lines.append("")
    lines.append("embeddings_%dd.npy" % n_dims)
    lines.append("  Low-dimensional projection via %s." % method.upper())
    lines.append("  Shape: (n_molecules, %d)" % n_dims)
    lines.append("")
    lines.append("mol_ids.txt")
    lines.append("  Molecule IDs in the same row order as descriptors/embeddings.")
    lines.append("  One ID per line (e.g. dsgdb9nsd_000200).")
    lines.append("")
    lines.append("metadata.npz")
    lines.append("  Numpy archive with arrays for coloring scatter plots:")
    lines.append("    atomization    -- atomization energy (kcal/mol)")
    lines.append("    pi_fraction    -- fraction of bonding from pi orbitals")
    lines.append("    n_atoms        -- number of atoms per molecule")
    lines.append("    n_heavy        -- number of heavy (non-H) atoms")
    lines.append("    composition    -- integer label for dominant element")
    lines.append("    cluster_labels -- HDBSCAN cluster assignments (-1=noise)")
    lines.append("")
    lines.append("scatter_*.png")
    lines.append("  Publication-quality scatter plots (500 dpi) colored by:")
    lines.append("    atomization  -- atomization energy gradient")
    lines.append("    composition  -- dominant heavy element (categorical)")
    lines.append("    pi_fraction  -- pi bonding fraction gradient")
    lines.append("    clusters     -- HDBSCAN cluster labels")
    lines.append("")
    lines.append("How to reload:")
    lines.append("-" * 40)
    lines.append("")
    lines.append("  import numpy as np")
    lines.append("  desc = np.load('descriptors.npy')")
    lines.append("  emb = np.load('embeddings_2d.npy')")
    lines.append("  meta = np.load('metadata.npz')")
    lines.append("  mol_ids = open('mol_ids.txt').read().strip().split('\\n')")
    lines.append("")
    lines.append("  # Custom scatter:")
    lines.append("  import matplotlib.pyplot as plt")
    lines.append("  plt.scatter(emb[:,0], emb[:,1], c=meta['atomization'], s=1)")
    lines.append("  plt.colorbar(label='Atomization E (kcal/mol)')")
    lines.append("  plt.savefig('my_plot.png', dpi=500)")
    lines.append("")
    lines.append("=" * 60)

    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, "w") as fh:
        fh.write("\n".join(lines))
    log.info("Saved: %s", readme_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_embedding_pipeline(hdf5_path, output_dir, method="umap", n_dims=2,
                           k_eigs=8, limit=None, export_csv=None):
    """
    Main pipeline:
    1. Read HDF5 and extract descriptors for all molecules
    2. Z-normalize
    3. Reduce dimensions
    4. Cluster
    5. Generate plots
    6. Save outputs
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Extract descriptors
    log.info("Reading HDF5: %s", hdf5_path)
    mol_ids = []
    descriptors = []

    with h5py.File(hdf5_path, "r") as f:
        all_mols = sorted(f["molecules"].keys())
        if limit is not None:
            all_mols = all_mols[:limit]
        log.info("Processing %d molecules...", len(all_mols))

        try:
            from tqdm import tqdm
            iterator = tqdm(all_mols, desc="Extracting features")
        except ImportError:
            iterator = all_mols

        for mid in iterator:
            mol = f["molecules/" + mid]
            try:
                desc = extract_descriptor(mol, k_eigs=k_eigs)
                descriptors.append(desc)
                mol_ids.append(mid)
            except Exception as e:
                log.warning("Skipping %s: %s", mid, e)

    descriptors = np.array(descriptors, dtype=np.float64)
    n_mols, n_features = descriptors.shape
    log.info("Extracted descriptors: %d molecules x %d features", n_mols, n_features)

    # Export raw CSV if requested
    if export_csv:
        labels = get_descriptor_labels(k_eigs)
        header = ",".join(["mol_id"] + labels)
        rows = []
        for mid, desc in zip(mol_ids, descriptors):
            row = mid + "," + ",".join("%.8f" % v for v in desc)
            rows.append(row)
        with open(export_csv, "w") as fh:
            fh.write(header + "\n")
            fh.write("\n".join(rows))
        log.info("Exported CSV: %s (%d rows)", export_csv, len(rows))
        if not output_dir:
            return

    # Step 2: Z-normalize (handle zero-variance columns)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    desc_scaled = scaler.fit_transform(descriptors)
    # Replace NaN from zero-variance columns with 0
    desc_scaled = np.nan_to_num(desc_scaled, nan=0.0)

    # Step 3: Reduce dimensions
    log.info("Reducing dimensions with %s (-> %dD)...", method.upper(), n_dims)
    embedding = reduce_dimensions(desc_scaled, method=method, n_dims=n_dims)
    log.info("Embedding shape: %s", embedding.shape)

    # Step 4: Cluster
    log.info("Clustering with HDBSCAN...")
    cluster_labels = cluster_embeddings(embedding)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    log.info("Found %d clusters, %d noise points", n_clusters, n_noise)

    # Step 5: Compute coloring arrays
    log.info("Computing coloring arrays...")
    colors = compute_coloring_arrays(mol_ids, hdf5_path)

    # Step 6: Save outputs
    np.save(os.path.join(output_dir, "descriptors.npy"), descriptors)
    np.save(os.path.join(output_dir, "embeddings_%dd.npy" % n_dims), embedding)

    with open(os.path.join(output_dir, "mol_ids.txt"), "w") as fh:
        fh.write("\n".join(mol_ids))

    labels_list = get_descriptor_labels(k_eigs)
    with open(os.path.join(output_dir, "descriptor_labels.txt"), "w") as fh:
        fh.write("\n".join(labels_list))

    np.savez(
        os.path.join(output_dir, "metadata.npz"),
        atomization=colors["atomization"][0],
        pi_fraction=colors["pi_fraction"][0],
        n_atoms=colors["n_atoms"][0],
        n_heavy=colors["n_heavy"][0],
        composition=colors["composition"][0],
        cluster_labels=cluster_labels,
    )

    write_output_readme(output_dir, method, n_dims, n_mols, n_features)

    # Step 7: Generate plots
    if n_dims == 2:
        log.info("Generating scatter plots...")

        plot_2d_scatter(
            embedding, colors["atomization"][0], "RdYlBu_r",
            "Atomization energy (kcal/mol)",
            "QM9+BEBOP: colored by atomization energy",
            os.path.join(output_dir, "scatter_atomization.png"),
        )

        el_labels, el_set, el_map = colors["_elem_labels"]
        plot_2d_scatter(
            embedding, None, "tab10",
            "Dominant element",
            "QM9+BEBOP: colored by dominant heavy element",
            os.path.join(output_dir, "scatter_composition.png"),
            elem_labels=el_labels, elem_set=el_set,
        )

        plot_2d_scatter(
            embedding, colors["pi_fraction"][0], "magma",
            "Pi bonding fraction",
            "QM9+BEBOP: colored by pi bonding fraction",
            os.path.join(output_dir, "scatter_pi_fraction.png"),
        )

        plot_2d_scatter(
            embedding, cluster_labels.astype(float), "tab20",
            "Cluster ID",
            "QM9+BEBOP: HDBSCAN clusters (%d found)" % n_clusters,
            os.path.join(output_dir, "scatter_clusters.png"),
        )

    log.info("Done. Output saved to: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate molecular graph embeddings from QM9+BEBOP HDF5"
    )
    parser.add_argument("--hdf5", required=True, help="Input HDF5 file from qm9_to_hdf5.py")
    parser.add_argument("--output", default="embeddings", help="Output directory (default: embeddings/)")
    parser.add_argument("--method", default="umap", choices=["umap", "tsne", "pca"],
                        help="Dimensionality reduction method (default: umap)")
    parser.add_argument("--dims", type=int, default=2, choices=[2, 3],
                        help="Output dimensions (default: 2)")
    parser.add_argument("--k-eigs", type=int, default=8,
                        help="Number of Laplacian eigenvalues per channel (default: 8)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N molecules (for testing)")
    parser.add_argument("--export-csv", default=None, metavar="FILE",
                        help="Export raw descriptors as CSV")
    parser.add_argument("--inspect-mol", default=None, metavar="MOL_ID",
                        help="Print descriptor breakdown for a single molecule")

    args = parser.parse_args()

    if args.inspect_mol:
        inspect_molecule(args.hdf5, args.inspect_mol, k_eigs=args.k_eigs)
    else:
        run_embedding_pipeline(
            hdf5_path=args.hdf5,
            output_dir=args.output,
            method=args.method,
            n_dims=args.dims,
            k_eigs=args.k_eigs,
            limit=args.limit,
            export_csv=args.export_csv,
        )


if __name__ == "__main__":
    main()
