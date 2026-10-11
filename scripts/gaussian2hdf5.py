#!/usr/bin/env python3
"""
Gaussian Output -> Molecular Graph -> HDF5 Pipeline
=====================================================
Parses Gaussian output files (plain or bz2-compressed) and optionally
BEBOP JSON bond energy files from QM9-style datasets into a unified
HDF5 file with molecular graph features.

Node features : MBS diagonal (gross atomic population)
Edge features : MBS off-diagonal (bond orders)
Graph descriptors : topology, spectral, node/edge stats, WL kernel

Usage:
    python gaussian_to_hdf5.py --single-gaussian dsgdb9nsd_000006.out.bz2
    python gaussian_to_hdf5.py --single-json dsgdb9nsd_000200_bopout.json.bz2
    python gaussian_to_hdf5.py --gaussian-dir ./data/ --output test.h5 --limit 10
    python gaussian_to_hdf5.py --gaussian-dir ./data/ --json-dir ./data/ --output qm9.h5 --n-workers 8 --recursive
    python gaussian_to_hdf5.py --inspect qm9.h5
    python gaussian_to_hdf5.py --inspect qm9.h5 --inspect-mol dsgdb9nsd_000200

Dependencies:
    pip install numpy h5py networkx
    molecular_graph_viz.py must be in the same directory or on PYTHONPATH
"""

import re
import sys
import os
import bz2
import json
import argparse
import numpy as np
import h5py
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import networkx as nx
except ImportError:
    print("ERROR: pip install networkx")
    sys.exit(1)

try:
    from molgraph import (
        parse_condensed_matrix,
        matrix_to_molecular_graph,
        compute_all_descriptors,
        find_atom_count,
        find_atom_labels,
        find_coordinates,
        PAT_COL_HEADER,
        ELEMENT_MAP,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("WARNING: molecular_graph_viz.py not found.")
    print("         Graph descriptors will not be computed.\n")

# Safe regex patterns — built via concatenation to prevent string splitting
_END = "$"
_COL_HDR = r"^\d+(\s+\d+)*" + _END
_ATOM_LINE = r"\s*\d+\s+\d+\s+\d+"
_ALPHA_START = r"^[A-Za-z]"
_DIGIT_START = r"^\s*\d"

PAT_COL = re.compile(_COL_HDR)
PAT_SCF = re.compile(r"SCF Done.*=\s*(-?[\d.]+)")
PAT_ZPE = re.compile(r"Zero-point correction=\s*(-?[\d.]+)")
PAT_ENTH = re.compile(r"Sum of electronic and thermal Enthalpies=\s*(-?[\d.]+)")
PAT_FREE = re.compile(r"Sum of electronic and thermal Free Energies=\s*(-?[\d.]+)")
PAT_NATOMS = re.compile(r"NAtoms=\s*(\d+)")
PAT_MAYER = re.compile(r"Mayer\s+(atomic\s+)?bond\s+order", re.IGNORECASE)
PAT_LOWDIN = re.compile(r"Lowdin\s+.*condensed\s+to\s+atoms", re.IGNORECASE)
PAT_BONDING = re.compile(r"bonding.*condensed\s+to\s+atoms", re.IGNORECASE)
PAT_ORIENT = re.compile(r"Standard orientation|Input orientation")
PAT_ATOM_LABEL = re.compile(r"\s*(\d+)\s+(\d+)\s+\d+")
PAT_MULLIKEN = re.compile(r"\s*\d+\s+[A-Za-z]+\s+(-?[\d.]+)")

ELEMENT_MAP_FALLBACK = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 26: "Fe", 27: "Co",
    28: "Ni", 29: "Cu", 30: "Zn", 35: "Br", 44: "Ru", 46: "Pd", 78: "Pt",
    79: "Au",
}

if HAS_VIZ:
    EMAP = ELEMENT_MAP
    SYMBOL_TO_Z = {v: k for k, v in ELEMENT_MAP.items()}
else:
    EMAP = ELEMENT_MAP_FALLBACK
    SYMBOL_TO_Z = {v: k for k, v in ELEMENT_MAP_FALLBACK.items()}

GAUSS_EXTS = {".out", ".log", ".g09", ".g16", ".gau"}
JSON_EXTS = {".json"}

FAMILY_KEYS = {
    "topology": [
        "n_nodes", "n_edges", "density", "degree_mean", "degree_std",
        "degree_max", "diameter", "avg_path_len", "clustering_mean",
        "clustering_std", "n_components",
    ],
    "spectral": (
        ["adj_eig_" + str(i) for i in range(10)]
        + ["spectral_radius", "spectral_gap", "adj_participation_ratio", "adj_entropy"]
        + ["lap_eig_" + str(i) for i in range(10)]
        + ["fiedler", "laplacian_max", "laplacian_mean", "laplacian_std"]
    ),
    "node_feat": [
        "pop_mean", "pop_std", "pop_min", "pop_max", "pop_range",
        "charge_dev_mean", "charge_dev_std", "charge_dev_absmax",
    ],
    "edge_feat": [
        "bo_mean", "bo_std", "bo_max", "bo_min", "bo_total",
        "bo_positive_sum", "bo_negative_sum", "n_strong", "n_medium", "n_weak",
    ],
    "wl_kernel": [
        "wl" + str(i) + "_" + s
        for i in range(4) for s in ["n_unique", "max_count", "entropy"]
    ],
}


# ============================================================================
# FILE I/O
# ============================================================================

def read_file_lines(filepath):
    fp = Path(filepath)
    if fp.suffix == ".bz2" or str(fp).endswith(".bz2"):
        with bz2.open(fp, "rt", errors="replace") as f:
            return f.readlines()
    else:
        with open(fp, "r", errors="replace") as f:
            return f.readlines()


def read_json_file(filepath):
    fp = Path(filepath)
    if fp.suffix == ".bz2" or str(fp).endswith(".bz2"):
        with bz2.open(fp, "rt", errors="replace") as f:
            return json.load(f)
    else:
        with open(fp, "r") as f:
            return json.load(f)


def extract_mol_id(filepath):
    name = Path(filepath).name
    if name.endswith(".bz2"):
        name = name[:-4]
    for ext in [".out", ".log", ".g09", ".g16", ".gau", ".json"]:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    name = re.sub(r"_bopout", "", name)
    return name


def find_files(directory, extensions, recursive=False):
    directory = Path(directory)
    files = {}
    glob_fn = directory.rglob if recursive else directory.glob
    for ext in extensions:
        for pattern in ["*" + ext, "*" + ext + ".bz2"]:
            for f in sorted(glob_fn(pattern)):
                if f.is_file():
                    mid = extract_mol_id(f)
                    files[mid] = f
    return files


# ============================================================================
# GAUSSIAN PARSER
# ============================================================================

def _fb_find_atom_count(lines):
    for line in lines:
        m = PAT_NATOMS.search(line)
        if m:
            return int(m.group(1))
    return 0


def _fb_find_atom_labels(lines):
    labels = []
    in_o = False
    dc = 0
    for line in lines:
        if PAT_ORIENT.search(line):
            in_o = True
            dc = 0
            labels = []
            continue
        if in_o:
            if "-----" in line:
                dc += 1
                if dc >= 3:
                    in_o = False
                continue
            m = PAT_ATOM_LABEL.match(line)
            if m:
                labels.append(EMAP.get(int(m.group(2)), "X"))
    return labels


def _fb_find_coordinates(lines):
    coords = []
    in_o = False
    dc = 0
    for line in lines:
        if PAT_ORIENT.search(line):
            in_o = True
            dc = 0
            coords = []
            continue
        if in_o:
            if "-----" in line:
                dc += 1
                if dc >= 3:
                    in_o = False
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    coords.append([float(parts[3]), float(parts[4]), float(parts[5])])
                except ValueError:
                    pass
    return np.array(coords) if coords else None


def _fb_parse_condensed(lines, start_idx, n):
    matrix = np.zeros((n, n))
    idx = start_idx
    cols_done = 0
    while cols_done < n and idx < len(lines):
        s = lines[idx].strip()
        if PAT_COL.match(s):
            ci = [int(x) - 1 for x in s.split()]
            idx += 1
            for _ in range(n):
                if idx >= len(lines):
                    break
                dl = lines[idx].strip()
                if not dl or re.match(_ALPHA_START, dl):
                    break
                parts = re.split(r"\s+", dl)
                vals = []
                ri = None
                for p in parts:
                    try:
                        v = float(p)
                        if ri is None and v == int(v) and 1 <= int(v) <= n:
                            ri = int(v) - 1
                        else:
                            vals.append(v)
                    except ValueError:
                        continue
                if ri is not None and vals:
                    for c, v in zip(ci, vals):
                        if c < n and ri < n:
                            matrix[ri, c] = v
                idx += 1
            cols_done = max(cols_done, max(ci) + 1 if ci else cols_done)
        else:
            if s and not re.match(_DIGIT_START, s) and cols_done > 0:
                break
            idx += 1
    return matrix


def _get_parse_fn():
    if HAS_VIZ:
        return parse_condensed_matrix
    return _fb_parse_condensed


def _get_atom_count(lines):
    if HAS_VIZ:
        return find_atom_count(lines)
    return _fb_find_atom_count(lines)


def _get_atom_labels(lines):
    if HAS_VIZ:
        return find_atom_labels(lines)
    return _fb_find_atom_labels(lines)


def _get_coordinates(lines):
    if HAS_VIZ:
        return find_coordinates(lines)
    return _fb_find_coordinates(lines)


def parse_gaussian_lines(lines, filepath=None):
    na = _get_atom_count(lines)
    if na == 0:
        return None

    labels = _get_atom_labels(lines)
    coords = _get_coordinates(lines)
    pfn = _get_parse_fn()

    result = {
        "filename": Path(filepath).name if filepath else "unknown",
        "filepath": str(filepath) if filepath else "",
        "n_atoms": na,
        "atom_labels": labels or ["A" + str(i + 1) for i in range(na)],
        "coordinates": coords,
        "matrices": {},
        "energies": {},
    }

    # Energies
    for line in lines:
        m = PAT_SCF.search(line)
        if m:
            result["energies"]["scf_energy"] = float(m.group(1))
    for line in lines:
        m = PAT_ZPE.search(line)
        if m:
            result["energies"]["zpe"] = float(m.group(1))
    for line in lines:
        m = PAT_ENTH.search(line)
        if m:
            result["energies"]["enthalpy"] = float(m.group(1))
    for line in lines:
        m = PAT_FREE.search(line)
        if m:
            result["energies"]["free_energy"] = float(m.group(1))

    # Dipole
    for i, line in enumerate(lines):
        if "Dipole moment" in line and i + 1 < len(lines):
            parts = lines[i + 1].split()
            try:
                result["energies"]["dipole_x"] = float(parts[1])
                result["energies"]["dipole_y"] = float(parts[3])
                result["energies"]["dipole_z"] = float(parts[5])
                result["energies"]["dipole_tot"] = float(parts[7])
            except (IndexError, ValueError):
                pass

    # Mulliken charges (last)
    mulliken = None
    for i, line in enumerate(lines):
        if "Mulliken charges:" in line or "Mulliken atomic charges:" in line:
            charges = []
            j = i + 2
            while j < len(lines) and len(charges) < na:
                m = PAT_MULLIKEN.match(lines[j])
                if m:
                    charges.append(float(m.group(1)))
                elif charges:
                    break
                j += 1
            if len(charges) == na:
                mulliken = np.array(charges)
    if mulliken is not None:
        result["mulliken_charges"] = mulliken

    # MBS Mulliken (last)
    mbs = None
    for i, line in enumerate(lines):
        if "Condensed to atoms (all electrons)" in line:
            mat = pfn(lines, i + 1, na)
            if np.any(mat != 0):
                mbs = mat
    if mbs is not None:
        result["matrices"]["MBS_Mulliken"] = mbs

    # Mayer bond order (last)
    mayer = None
    for i, line in enumerate(lines):
        if PAT_MAYER.search(line):
            for j in range(i + 1, min(i + 5, len(lines))):
                if PAT_COL.match(lines[j].strip()):
                    mat = pfn(lines, j, na)
                    if np.any(mat != 0):
                        mayer = mat
                    break
    if mayer is not None:
        result["matrices"]["Mayer_bond_order"] = mayer

    # Lowdin (last)
    lowdin = None
    for i, line in enumerate(lines):
        if PAT_LOWDIN.search(line):
            mat = pfn(lines, i + 1, na)
            if np.any(mat != 0):
                lowdin = mat
    if lowdin is not None:
        result["matrices"]["Lowdin_condensed"] = lowdin

    # Bonding pop (last)
    bonding = None
    for i, line in enumerate(lines):
        if PAT_BONDING.search(line):
            mat = pfn(lines, i + 1, na)
            if np.any(mat != 0):
                bonding = mat
    if bonding is not None:
        result["matrices"]["Bonding_pop_condensed"] = bonding

    # Convergence check
    result["converged"] = any("Normal termination" in l for l in lines[-20:])

    return result


# ============================================================================
# BEBOP JSON PARSER
# ============================================================================

def parse_bebop_json(filepath):
    data = read_json_file(filepath)
    result = {
        "filename": Path(filepath).name,
        "atomization_energy": data.get("atomization_energy", None),
        "bebop_method": data.get("method", "unknown"),
        "bebop_version": data.get("version", "unknown"),
    }
    if "bonds" in data:
        bonds = data["bonds"]
        result["net_energies"] = np.array(
            [b.get("net_energy", 0) for b in bonds], dtype=np.float64
        )
        result["bond_labels"] = [b.get("label", "") for b in bonds]
    if "net_sigma" in data:
        result["net_sigma"] = np.array(data["net_sigma"], dtype=np.float64)
    if "composite_table" in data:
        result["composite_table"] = np.array(data["composite_table"], dtype=np.float64)
    if "net_energies" in data and "net_energies" not in result:
        result["net_energies"] = np.array(data["net_energies"], dtype=np.float64)
    if "bond_labels" in data and "bond_labels" not in result:
        result["bond_labels"] = data["bond_labels"]
    return result


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_graph_data(gauss_data, threshold=0.05):
    if not HAS_VIZ:
        return None, None
    if "MBS_Mulliken" not in gauss_data["matrices"]:
        return None, None
    mat = gauss_data["matrices"]["MBS_Mulliken"]
    G = matrix_to_molecular_graph(mat, gauss_data["atom_labels"], threshold)
    desc = compute_all_descriptors(G, mat)
    return G, desc


# ============================================================================
# HDF5 WRITER
# ============================================================================

def extract_formula(atom_labels):
    counts = {}
    for lb in atom_labels:
        counts[lb] = counts.get(lb, 0) + 1
    parts = []
    for el in ["C", "H"]:
        if el in counts:
            parts.append(el + (str(counts[el]) if counts[el] > 1 else ""))
            del counts[el]
    for el in sorted(counts):
        parts.append(el + (str(counts[el]) if counts[el] > 1 else ""))
    return "".join(parts)


def write_molecule(h5file, mol_id, gauss_data, bebop_data, G, descriptors, threshold):
    dt_str = h5py.string_dtype()
    na = gauss_data["n_atoms"]
    labels = gauss_data["atom_labels"]
    coords = gauss_data["coordinates"]
    atomic_nums = np.array([SYMBOL_TO_Z.get(lb, 0) for lb in labels], dtype=np.int32)

    grp = h5file.create_group("molecules/" + mol_id)
    grp.attrs["formula"] = extract_formula(labels)
    grp.attrs["n_atoms"] = na
    grp.attrs["n_bonds"] = len(G.edges) if G else 0
    grp.attrs["filename"] = gauss_data["filename"]
    grp.attrs["converged"] = gauss_data.get("converged", False)

    # /geometry
    geo = grp.create_group("geometry")
    if coords is not None and len(coords) == na:
        geo.create_dataset("coordinates", data=coords.astype(np.float64))
    else:
        geo.create_dataset("coordinates", data=np.zeros((na, 3), dtype=np.float64))
    geo.create_dataset("atomic_numbers", data=atomic_nums)
    geo.create_dataset("atom_labels", data=np.array(labels, dtype=object), dtype=dt_str)
    for key in ["scf_energy", "zpe", "enthalpy", "free_energy",
                "dipole_x", "dipole_y", "dipole_z", "dipole_tot"]:
        if key in gauss_data.get("energies", {}):
            geo.attrs[key] = gauss_data["energies"][key]
    if "mulliken_charges" in gauss_data:
        geo.create_dataset("mulliken_charges", data=gauss_data["mulliken_charges"])

    # /mbs_population
    if "MBS_Mulliken" in gauss_data["matrices"]:
        mbs = gauss_data["matrices"]["MBS_Mulliken"]
        mg = grp.create_group("mbs_population")
        mg.create_dataset("condensed_matrix", data=mbs.astype(np.float64))
        mg.create_dataset("node_populations", data=np.diag(mbs))
        cdev = np.array([mbs[i, i] - SYMBOL_TO_Z.get(labels[i], 0)
                         for i in range(na)], dtype=np.float64)
        mg.create_dataset("node_charge_dev", data=cdev)
        mg.attrs["trace"] = float(np.trace(mbs))

    # /bond_orders
    if "MBS_Mulliken" in gauss_data["matrices"]:
        mbs = gauss_data["matrices"]["MBS_Mulliken"]
        bg = grp.create_group("bond_orders")
        bo_mat = mbs.copy()
        np.fill_diagonal(bo_mat, 0.0)
        bg.create_dataset("bond_order_matrix", data=bo_mat.astype(np.float64))

        if G and len(G.edges) > 0:
            edges = list(G.edges)
            ne = len(edges)
            ea = np.array(edges, dtype=np.int32)
            ebo = np.array([G[u][v]["bond_order"] for u, v in edges], dtype=np.float64)
            eab = np.array([G[u][v]["abs_bo"] for u, v in edges], dtype=np.float64)
            etp = np.array([G[u][v]["bond_type"] for u, v in edges], dtype=object)
        else:
            ne = 0
            ea = np.zeros((0, 2), dtype=np.int32)
            ebo = np.zeros(0, dtype=np.float64)
            eab = np.zeros(0, dtype=np.float64)
            etp = np.array([], dtype=object)

        bg.create_dataset("edge_list", data=ea)
        bg.create_dataset("edge_bond_orders", data=ebo)
        bg.create_dataset("edge_abs_bond_orders", data=eab)
        bg.create_dataset("edge_bond_types", data=etp, dtype=dt_str)
        bg.attrs["n_edges"] = ne
        bg.attrs["threshold"] = threshold
        bg.attrs["total_bond_order"] = float(ebo.sum()) if ne > 0 else 0.0
        bg.attrs["avg_bond_order"] = float(ebo.mean()) if ne > 0 else 0.0
        bg.attrs["max_bond_order"] = float(eab.max()) if ne > 0 else 0.0

    # /bond_energies (BEBOP)
    if bebop_data:
        be = grp.create_group("bond_energies")
        if bebop_data.get("atomization_energy") is not None:
            be.attrs["atomization_energy"] = bebop_data["atomization_energy"]
        if bebop_data.get("bebop_method"):
            be.attrs["bebop_method"] = bebop_data["bebop_method"]
        if bebop_data.get("bebop_version"):
            be.attrs["bebop_version"] = bebop_data["bebop_version"]
        if "net_energies" in bebop_data:
            be.create_dataset("net_energies", data=bebop_data["net_energies"])
        if "bond_labels" in bebop_data:
            bl = bebop_data["bond_labels"]
            be.create_dataset("bond_labels", data=np.array(bl, dtype=object), dtype=dt_str)
        if "net_sigma" in bebop_data or "composite_table" in bebop_data:
            mat_g = be.create_group("matrices")
            if "net_sigma" in bebop_data:
                mat_g.create_dataset("net_sigma", data=bebop_data["net_sigma"])
            if "composite_table" in bebop_data:
                mat_g.create_dataset("composite_table", data=bebop_data["composite_table"])

    # /graph_descriptors
    if descriptors:
        gd = grp.create_group("graph_descriptors")
        all_keys = sorted(descriptors.keys())
        all_vals = np.array([float(descriptors[k]) for k in all_keys], dtype=np.float64)
        gd.create_dataset("feature_names", data=np.array(all_keys, dtype=object), dtype=dt_str)
        gd.create_dataset("feature_values", data=all_vals)
        fg = gd.create_group("families")
        for fname, fkeys in FAMILY_KEYS.items():
            fv = np.array([float(descriptors.get(k, 0)) for k in fkeys], dtype=np.float64)
            ds = fg.create_dataset(fname, data=fv)
            ds.attrs["keys"] = ",".join(fkeys)

    # /eigenvalues
    if G and na >= 2:
        eg = grp.create_group("eigenvalues")
        A = nx.adjacency_matrix(G, weight="weight").toarray()
        adj_e = np.sort(np.linalg.eigvalsh(A))[::-1]
        L = nx.laplacian_matrix(G, weight="weight").toarray().astype(float)
        lap_e = np.sort(np.linalg.eigvalsh(L))
        eg.create_dataset("adjacency_eigs", data=adj_e)
        eg.create_dataset("laplacian_eigs", data=lap_e)
        eg.attrs["fiedler_value"] = float(lap_e[1]) if len(lap_e) >= 2 else 0.0
        eg.attrs["spectral_gap"] = float(abs(adj_e[0] - adj_e[1])) if len(adj_e) >= 2 else 0.0
        ae = np.abs(adj_e)
        s2 = np.sum(ae ** 2)
        s4 = np.sum(ae ** 4)
        eg.attrs["participation_ratio"] = float((s2 ** 2) / s4) if s4 > 1e-30 else float(na)

    # /other_matrices
    others = [k for k in gauss_data["matrices"] if k != "MBS_Mulliken"]
    if others:
        om = grp.create_group("other_matrices")
        for mt in others:
            safe = re.sub(r"[^a-zA-Z0-9_]", "_", mt)
            om.create_dataset(safe, data=gauss_data["matrices"][mt].astype(np.float64))


def write_global_metadata(h5file, mol_ids, all_descriptors, threshold):
    for key in ["metadata", "descriptor_matrix"]:
        if key in h5file:
            del h5file[key]
    dt_str = h5py.string_dtype()
    meta = h5file.create_group("metadata")
    meta.attrs["n_molecules"] = len(mol_ids)
    meta.attrs["threshold"] = threshold
    meta.attrs["source"] = "gaussian_to_hdf5.py"
    meta.create_dataset("molecule_ids", data=np.array(mol_ids, dtype=object), dtype=dt_str)
    if all_descriptors:
        keys = sorted(all_descriptors[0].keys())
        X = np.array([[float(d.get(k, 0)) for k in keys] for d in all_descriptors],
                      dtype=np.float64)
        dg = h5file.create_group("descriptor_matrix")
        dg.create_dataset("values", data=X)
        dg.create_dataset("feature_names", data=np.array(keys, dtype=object), dtype=dt_str)
        dg.create_dataset("molecule_ids", data=np.array(mol_ids, dtype=object), dtype=dt_str)
        dg.attrs["shape"] = str(X.shape[0]) + " x " + str(X.shape[1])


# ============================================================================
# INSPECTION MODES
# ============================================================================

def inspect_single_gaussian(filepath):
    print("\n  Parsing: " + str(filepath))
    lines = read_file_lines(filepath)
    data = parse_gaussian_lines(lines, filepath)
    if not data:
        print("  FAILED")
        return
    print("  Molecule ID   : " + extract_mol_id(filepath))
    print("  Atoms         : " + str(data["n_atoms"]))
    print("  Labels        : " + " ".join(data["atom_labels"]))
    print("  Formula       : " + extract_formula(data["atom_labels"]))
    print("  Converged     : " + str(data.get("converged", "?")))
    if data["energies"]:
        print("  Energies:")
        for k, v in data["energies"].items():
            print("    " + k.ljust(20) + " = " + str(round(v, 8)))
    print("  Matrices: " + str(list(data["matrices"].keys())))
    for mt, mat in data["matrices"].items():
        tr = round(float(np.trace(mat)), 4)
        fr = round(float(np.linalg.norm(mat, "fro")), 4)
        print("    " + mt + ": " + str(mat.shape) + " trace=" + str(tr) + " Frob=" + str(fr))
    if data["coordinates"] is not None:
        print("  Coordinates   : " + str(data["coordinates"].shape))
    G, desc = build_graph_data(data)
    if G:
        print("\n  Graph: " + str(len(G.nodes)) + " nodes, " + str(len(G.edges)) + " edges")
        if desc:
            print("  Descriptors: " + str(len(desc)) + " features")


def inspect_single_json(filepath):
    print("\n  Parsing: " + str(filepath))
    data = parse_bebop_json(filepath)
    print("  Molecule ID       : " + extract_mol_id(filepath))
    print("  BEBOP method      : " + str(data.get("bebop_method", "?")))
    print("  Atomization energy: " + str(data.get("atomization_energy", "N/A")))
    if "net_energies" in data:
        ne = data["net_energies"]
        print("  Net energies      : " + str(len(ne)) + " bonds, sum=" + str(round(ne.sum(), 6)))
    if "bond_labels" in data:
        print("  Bond labels       : " + str(data["bond_labels"]))
    if "composite_table" in data:
        ct = data["composite_table"]
        print("  Composite table   : " + str(ct.shape))
        print("    Self-energies   : " + str(np.diag(ct)))


def inspect_hdf5(filepath, mol_id=None):
    print("\n  Inspecting: " + str(filepath))
    with h5py.File(filepath, "r") as h5:
        if "metadata" in h5:
            meta = h5["metadata"]
            print("  Molecules  : " + str(meta.attrs.get("n_molecules", "?")))
            print("  Threshold  : " + str(meta.attrs.get("threshold", "?")))
        if "descriptor_matrix" in h5:
            print("  Descriptors: " + str(h5["descriptor_matrix/values"].shape))
        if "molecules" not in h5:
            print("  No /molecules group.")
            return
        mids = sorted(h5["molecules"].keys())
        print("  Molecule IDs: " + str(len(mids)))
        if mol_id:
            if mol_id not in h5["molecules"]:
                print("  Not found: " + mol_id)
                return
            _print_tree(h5["molecules/" + mol_id], "    ")
        else:
            for mid in mids[:5]:
                g = h5["molecules/" + mid]
                formula = g.attrs.get("formula", "?")
                na = g.attrs.get("n_atoms", "?")
                nb = g.attrs.get("n_bonds", "?")
                print("    " + mid.ljust(30) + "  " + str(formula).ljust(12)
                      + "  atoms=" + str(na) + "  bonds=" + str(nb))
            if len(mids) > 5:
                print("    ... and " + str(len(mids) - 5) + " more")
            print("\n  Full tree of first molecule:")
            _print_tree(h5["molecules/" + mids[0]], "    ")


def _print_tree(grp, indent=""):
    for key in grp:
        item = grp[key]
        if isinstance(item, h5py.Group):
            attr_str = ""
            if item.attrs:
                attr_str = "  [@" + ", @".join(item.attrs.keys()) + "]"
            print(indent + "/" + key + "/" + attr_str)
            _print_tree(item, indent + "    ")
        else:
            print(indent + key + ": " + str(item.shape) + " " + str(item.dtype))
    for k, v in grp.attrs.items():
        val = v.decode() if isinstance(v, bytes) else v
        if isinstance(val, (float, np.floating)):
            val = str(round(float(val), 6))
        print(indent + "@" + str(k) + " = " + str(val))


# ============================================================================
# BATCH WORKER
# ============================================================================

def _patch_one_molecule(args_tuple):
    """Worker: parse one Gaussian file and return MBS data."""
    mol_id, gauss_path, threshold = args_tuple
    try:
        lines = read_file_lines(gauss_path)
        na = find_atom_count(lines)
        if na == 0:
            return (mol_id, None, "no atoms")

        labels = find_atom_labels(lines) or ["X"] * na
        mbs = find_mbs_condensed(lines, na)
        if mbs is None:
            return (mol_id, None, "no MBS found")

        # Build graph if possible
        G = None
        desc = None
        if HAS_VIZ and HAS_NX:
            G = matrix_to_molecular_graph(mbs, labels, threshold)
            desc = compute_all_descriptors(G, mbs)

        return (mol_id, {
            "mbs": mbs,
            "labels": labels,
            "n_atoms": na,
            "graph": G,
            "descriptors": desc,
        }, "ok")
    except Exception as e:
        return (mol_id, None, str(e))

def patch_mbs_into_hdf5(h5path, gaussian_dir, threshold=0.05,
                         recursive=False, recompute_graph=False,
                         overwrite=False, limit=None, n_workers=1):
    """
    Patch MBS condensed matrices into an existing HDF5 file.

    For each molecule in the HDF5:
      1. Find the matching Gaussian file in gaussian_dir
      2. Re-parse to extract MBS condensed matrix
      3. Add /mbs_population group (node features)
      4. Update /bond_orders group (edge features)
      5. Optionally recompute /graph_descriptors and /eigenvalues

    Usage:
        python gaussian_to_hdf5.py --patch-mbs existing.h5 --gaussian-dir ./data/
        python gaussian_to_hdf5.py --patch-mbs existing.h5 --gaussian-dir ./data/ --patch-recompute-graph --patch-overwrite
    """
    h5path = Path(h5path)
    if not h5path.is_file():
        print("  ERROR: HDF5 not found: " + str(h5path))
        sys.exit(1)

    if not gaussian_dir:
        print("  ERROR: --gaussian-dir required for --patch-mbs")
        sys.exit(1)

    print("")
    print("=" * 60)
    print("  MBS Patch Utility")
    print("  HDF5 file    : " + str(h5path))
    print("  Gaussian dir : " + str(gaussian_dir))
    print("  Threshold    : " + str(threshold))
    print("  Overwrite    : " + str(overwrite))
    print("  Recompute    : " + str(recompute_graph))
    print("=" * 60)
    print("")

    # Discover Gaussian files
    gauss_files = find_files(gaussian_dir, GAUSS_EXTS, recursive)
    print("  Found " + str(len(gauss_files)) + " Gaussian files")

    # Read existing molecule IDs from HDF5
    with h5py.File(h5path, "r") as h5:
        if "molecules" not in h5:
            print("  ERROR: No /molecules group in HDF5.")
            sys.exit(1)
        existing_mids = sorted(h5["molecules"].keys())

    print("  Existing molecules in HDF5: " + str(len(existing_mids)))

    # Match molecules to Gaussian files
    work = []
    n_skip_has_mbs = 0
    n_skip_no_gauss = 0

    with h5py.File(h5path, "r") as h5:
        for mid in existing_mids:
            if limit and len(work) >= limit:
                break

            # Check if MBS already exists
            mol_path = "molecules/" + mid
            has_mbs = "mbs_population" in h5[mol_path]

            if has_mbs and not overwrite:
                n_skip_has_mbs += 1
                continue

            # Find matching Gaussian file
            if mid in gauss_files:
                work.append((mid, gauss_files[mid], threshold))
            else:
                # Try matching by filename stored in attrs
                fname = h5[mol_path].attrs.get("filename", "")
                if isinstance(fname, bytes):
                    fname = fname.decode()
                fname_id = extract_mol_id(fname)
                if fname_id in gauss_files:
                    work.append((fname_id, gauss_files[fname_id], threshold))
                else:
                    n_skip_no_gauss += 1

    print("  To patch     : " + str(len(work)))
    print("  Skip (has MBS): " + str(n_skip_has_mbs))
    print("  Skip (no file): " + str(n_skip_no_gauss))

    if not work:
        print("\n  Nothing to patch.")
        return

    # Parse Gaussian files
    print("\n  Parsing Gaussian files...")
    results = []
    n_ok = 0
    n_fail = 0
    n_no_mbs = 0

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_patch_one_molecule, w): w[0] for w in work}
            for future in as_completed(futures):
                mid = futures[future]
                try:
                    res = future.result()
                    if res[1] is not None:
                        results.append(res)
                        n_ok += 1
                    else:
                        n_fail += 1
                        if res[2] == "no MBS found":
                            n_no_mbs += 1
                except Exception:
                    n_fail += 1
                total = n_ok + n_fail
                if total % 100 == 0:
                    print("    Progress: " + str(total) + "/" + str(len(work)))
    else:
        for i, w in enumerate(work):
            res = _patch_one_molecule(w)
            if res[1] is not None:
                results.append(res)
                n_ok += 1
            else:
                n_fail += 1
                if res[2] == "no MBS found":
                    n_no_mbs += 1
            done = i + 1
            if done % 50 == 0 or done == len(work):
                print("    Progress: " + str(done) + "/" + str(len(work))
                      + " (" + str(n_ok) + " ok, " + str(n_no_mbs) + " no-MBS)")

    print("\n  Parsed: " + str(n_ok) + " with MBS, "
          + str(n_no_mbs) + " no MBS, " + str(n_fail - n_no_mbs) + " errors")

    if not results:
        print("  No MBS data to patch.")
        return

    # Write patches to HDF5
    print("\n  Patching HDF5...")
    dt_str = h5py.string_dtype()
    n_patched = 0
    n_graph_updated = 0

    with h5py.File(h5path, "a") as h5:
        for mol_id, patch_data, status in results:
            mol_path = "molecules/" + mol_id
            if mol_path not in h5:
                # Try to find by scanning existing mids
                found = False
                for emid in existing_mids:
                    if mol_id in emid or emid in mol_id:
                        mol_path = "molecules/" + emid
                        found = True
                        break
                if not found:
                    continue

            mol = h5[mol_path]
            mbs = patch_data["mbs"]
            na = patch_data["n_atoms"]
            labels = patch_data["labels"]
            G = patch_data["graph"]
            desc = patch_data["descriptors"]

            # --- Patch /mbs_population ---
            if "mbs_population" in mol:
                del mol["mbs_population"]

            mg = mol.create_group("mbs_population")
            mg.create_dataset("condensed_matrix", data=mbs.astype(np.float64))
            mg.create_dataset("node_populations", data=np.diag(mbs))
            cdev = np.array([mbs[i, i] - SYMBOL_TO_Z.get(labels[i], 0)
                             for i in range(na)], dtype=np.float64)
            mg.create_dataset("node_charge_dev", data=cdev)
            mg.attrs["trace"] = float(np.trace(mbs))

            # --- Patch /bond_orders ---
            if "bond_orders" in mol:
                del mol["bond_orders"]

            bg = mol.create_group("bond_orders")
            bo_mat = mbs.copy()
            np.fill_diagonal(bo_mat, 0.0)
            bg.create_dataset("bond_order_matrix", data=bo_mat.astype(np.float64))

            if G and len(G.edges) > 0:
                edges = list(G.edges)
                ne = len(edges)
                ea = np.array(edges, dtype=np.int32)
                ebo = np.array([G[u][v]["bond_order"] for u, v in edges], dtype=np.float64)
                eab = np.array([G[u][v]["abs_bo"] for u, v in edges], dtype=np.float64)
            else:
                ne = 0
                ea = np.zeros((0, 2), dtype=np.int32)
                ebo = np.zeros(0, dtype=np.float64)
                eab = np.zeros(0, dtype=np.float64)

            bg.create_dataset("edge_list", data=ea)
            bg.create_dataset("edge_bond_orders", data=ebo)
            bg.create_dataset("edge_abs_bond_orders", data=eab)
            bg.attrs["n_edges"] = ne
            bg.attrs["threshold"] = threshold
            bg.attrs["total_bond_order"] = float(ebo.sum()) if ne > 0 else 0.0
            bg.attrs["avg_bond_order"] = float(ebo.mean()) if ne > 0 else 0.0
            bg.attrs["max_bond_order"] = float(eab.max()) if ne > 0 else 0.0

            # Update root n_bonds attr
            mol.attrs["n_bonds"] = ne

            # --- Optionally recompute graph descriptors + eigenvalues ---
            if recompute_graph and G and desc:
                if "graph_descriptors" in mol:
                    del mol["graph_descriptors"]
                gd = mol.create_group("graph_descriptors")
                all_keys = sorted(desc.keys())
                all_vals = np.array([float(desc[k]) for k in all_keys], dtype=np.float64)
                gd.create_dataset("feature_names",
                                  data=np.array(all_keys, dtype=object), dtype=dt_str)
                gd.create_dataset("feature_values", data=all_vals)
                fg = gd.create_group("families")
                for fname, fkeys in FAMILY_KEYS.items():
                    fv = np.array([float(desc.get(k, 0)) for k in fkeys], dtype=np.float64)
                    ds = fg.create_dataset(fname, data=fv)
                    ds.attrs["keys"] = ",".join(fkeys)

                if "eigenvalues" in mol:
                    del mol["eigenvalues"]
                if na >= 2 and HAS_NX:
                    eg = mol.create_group("eigenvalues")
                    A = nx.adjacency_matrix(G, weight="weight").toarray()
                    adj_e = np.sort(np.linalg.eigvalsh(A))[::-1]
                    L = nx.laplacian_matrix(G, weight="weight").toarray().astype(float)
                    lap_e = np.sort(np.linalg.eigvalsh(L))
                    eg.create_dataset("adjacency_eigs", data=adj_e)
                    eg.create_dataset("laplacian_eigs", data=lap_e)
                    eg.attrs["fiedler_value"] = float(lap_e[1]) if len(lap_e) >= 2 else 0.0
                    eg.attrs["spectral_gap"] = (
                        float(abs(adj_e[0] - adj_e[1])) if len(adj_e) >= 2 else 0.0)
                    ae = np.abs(adj_e)
                    s2 = np.sum(ae ** 2)
                    s4 = np.sum(ae ** 4)
                    eg.attrs["participation_ratio"] = (
                        float((s2 ** 2) / s4) if s4 > 1e-30 else float(na))

                n_graph_updated += 1

            n_patched += 1

        # Rebuild global descriptor matrix if graphs were recomputed
        if recompute_graph and n_graph_updated > 0:
            print("  Rebuilding descriptor matrix...")
            all_mids = sorted(h5["molecules"].keys())
            all_descs = []
            valid_mids = []
            for mid in all_mids:
                gd_path = "molecules/" + mid + "/graph_descriptors"
                if gd_path in h5:
                    gd = h5[gd_path]
                    if "feature_names" in gd and "feature_values" in gd:
                        all_descs.append(gd["feature_values"][:])
                        valid_mids.append(mid)
            if all_descs:
                names_raw = h5["molecules/" + valid_mids[0] + "/graph_descriptors/feature_names"][:]
                names = [n.decode() if isinstance(n, bytes) else n for n in names_raw]
                X = np.array(all_descs, dtype=np.float64)

                if "descriptor_matrix" in h5:
                    del h5["descriptor_matrix"]
                dg = h5.create_group("descriptor_matrix")
                dg.create_dataset("values", data=X)
                dg.create_dataset("feature_names",
                                  data=np.array(names, dtype=object), dtype=dt_str)
                dg.create_dataset("molecule_ids",
                                  data=np.array(valid_mids, dtype=object), dtype=dt_str)
                dg.attrs["shape"] = str(X.shape[0]) + " x " + str(X.shape[1])
                print("  Descriptor matrix: " + str(X.shape[0]) + " x " + str(X.shape[1]))

    # Summary
    print("")
    print("  Patch complete:")
    print("    MBS added/updated : " + str(n_patched))
    print("    Graphs recomputed : " + str(n_graph_updated))
    print("")

    inspect_hdf5(str(h5path))
    print("\nDone.")

# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="QM9 Dataset -> HDF5 Pipeline")
    ap.add_argument("--single-gaussian", type=str, default=None)
    ap.add_argument("--single-json", type=str, default=None)
    ap.add_argument("--gaussian-dir", type=str, default=None)
    ap.add_argument("--json-dir", type=str, default=None)
    ap.add_argument("--output", "-o", type=str, default="molecular_graphs.h5")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--inspect", type=str, default=None)
    ap.add_argument("--inspect-mol", type=str, default=None)

    # Patch utilities
    ap.add_argument("--patch-mbs", type=str, default=None,
                    help="Patch MBS matrices into existing HDF5. "
                         "Pass the HDF5 file path. Requires --gaussian-dir.")
    ap.add_argument("--patch-recompute-graph", action="store_true",
                    help="Also recompute graph descriptors and eigenvalues during patch")
    ap.add_argument("--patch-overwrite", action="store_true",
                    help="Overwrite existing mbs_population groups during patch")

    args = ap.parse_args()

    if args.patch_mbs:
        patch_mbs_into_hdf5(
            h5path=args.patch_mbs,
            gaussian_dir=args.gaussian_dir,
            threshold=args.threshold,
            recursive=args.recursive,
            recompute_graph=args.patch_recompute_graph,
            overwrite=args.patch_overwrite,
            limit=args.limit,
            n_workers=args.n_workers,
        )
        return

    if args.single_gaussian:
        inspect_single_gaussian(args.single_gaussian)
        return
    if args.single_json:
        inspect_single_json(args.single_json)
        return
    if args.inspect:
        inspect_hdf5(args.inspect, args.inspect_mol)
        return
    if not args.gaussian_dir:
        ap.print_help()
        sys.exit(1)

    print("")
    print("=" * 60)
    print("  QM9 Dataset -> HDF5 Pipeline")
    print("  Gaussian dir : " + str(args.gaussian_dir))
    print("  JSON dir     : " + str(args.json_dir or "None"))
    print("  Output       : " + str(args.output))
    print("  Threshold    : " + str(args.threshold))
    print("  Workers      : " + str(args.n_workers))
    print("=" * 60)
    print("")

    gauss_files = find_files(args.gaussian_dir, GAUSS_EXTS, args.recursive)
    print("  Found " + str(len(gauss_files)) + " Gaussian files")

    json_files = {}
    if args.json_dir:
        json_files = find_files(args.json_dir, JSON_EXTS, args.recursive)
        print("  Found " + str(len(json_files)) + " JSON files")

    mol_ids = sorted(gauss_files.keys())
    if args.limit:
        mol_ids = mol_ids[: args.limit]
    print("  Processing " + str(len(mol_ids)) + " molecules\n")

    work = [(mid, gauss_files[mid], json_files.get(mid, None), args.threshold)
            for mid in mol_ids]

    results = []
    n_ok = 0
    n_fail = 0
    n_no_mbs = 0

    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {pool.submit(process_one_molecule, w): w[0] for w in work}
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                        n_ok += 1
                        if not res[1].get("_debug_mbs_found", False):
                            n_no_mbs += 1
                    else:
                        n_fail += 1
                except Exception:
                    n_fail += 1
                total = n_ok + n_fail
                if total % 100 == 0:
                    print("    Progress: " + str(total) + "/" + str(len(work)))
    else:
        for i, w in enumerate(work):
            res = process_one_molecule(w)
            if res:
                results.append(res)
                n_ok += 1
                if not res[1].get("_debug_mbs_found", False):
                    n_no_mbs += 1
            else:
                n_fail += 1
            done = i + 1
            if done % 50 == 0 or done == len(work):
                print("    Progress: " + str(done) + "/" + str(len(work))
                      + " (" + str(n_ok) + " ok, " + str(n_fail) + " fail, "
                      + str(n_no_mbs) + " no-MBS)")

    print("\n  Results: " + str(n_ok) + " ok, " + str(n_fail) + " failed, "
          + str(n_no_mbs) + " missing MBS matrix")

    if n_no_mbs > 0 and n_no_mbs == n_ok:
        print("\n  WARNING: No MBS matrices found in ANY file.")
        print("  This usually means the Gaussian runs did not use pop=full.")
        print("  Run --single-gaussian on one file to debug.")

    if not results:
        sys.exit(1)

    results.sort(key=lambda x: x[0])

    mode = "a" if args.append else "w"
    print("\n  Writing HDF5: " + args.output)

    all_mol_ids = []
    all_descriptors = []

    with h5py.File(args.output, mode) as h5:
        for mol_id, gauss_data, bebop_data, G, desc in results:
            if args.append and ("molecules/" + mol_id) in h5:
                continue
            write_molecule(h5, mol_id, gauss_data, bebop_data, G, desc, args.threshold)
            all_mol_ids.append(mol_id)
            if desc:
                all_descriptors.append(desc)
        write_global_metadata(h5, all_mol_ids, all_descriptors, args.threshold)

    print("  Wrote " + str(len(all_mol_ids)) + " molecules")

    # CSV
    if all_descriptors:
        csv_path = args.output.rsplit(".", 1)[0] + "_descriptors.csv"
        keys = sorted(all_descriptors[0].keys())
        with open(csv_path, "w") as f:
            f.write("molecule," + ",".join(keys) + "\n")
            for mid, desc in zip(all_mol_ids, all_descriptors):
                vals = []
                for k in keys:
                    v = desc.get(k, 0)
                    vals.append(str(round(v, 8)) if isinstance(v, float) else str(v))
                f.write(mid + "," + ",".join(vals) + "\n")
        print("  Saved: " + csv_path)

    inspect_hdf5(args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()

