"""
QM9 Dataset -> HDF5 Pipeline
============================
Parses bz2-compressed Gaussian output files and BEBOP JSON bond energy files
from the QM9 dataset (B3LYP-D3BJ/CBSB7) into a unified HDF5 file.

Usage:
    python qm9_to_hdf5.py --single-gaussian dsgdb9nsd_000006.out.bz2
    python qm9_to_hdf5.py --single-json dsgdb9nsd_000200_bopout.json.bz2
    python qm9_to_hdf5.py --gaussian-dir ./data/ --json-dir ./data/ --output test.h5 --limit 10
    python qm9_to_hdf5.py --gaussian-dir ./data/ --json-dir ./data/ --output qm9.h5 --n-workers 8 --recursive
    python qm9_to_hdf5.py --inspect qm9.h5

Requirements:
    pip install h5py numpy tqdm scipy
"""

import bz2
import re
import json
import h5py
import numpy as np
import argparse
import logging
import os
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional
from dataclasses import dataclass
from collections import Counter
from scipy.spatial.distance import pdist, squareform

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ATOMIC_NUMBERS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
ATOMIC_SYMBOLS = {v: k for k, v in ATOMIC_NUMBERS.items()}
COVALENT_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57}
BOND_TOLERANCE = 0.4


@dataclass
class GaussianData:
    mol_id: str
    atomic_numbers: np.ndarray
    coordinates: np.ndarray
    scf_energy: float
    mulliken_charges: np.ndarray
    converged: bool = True
    orbital_populations: dict = None
    bond_indices: np.ndarray = None
    bond_lengths: np.ndarray = None
    bond_elements: list = None

    def compute_bonds(self):
        n = len(self.atomic_numbers)
        if n < 2:
            self.bond_indices = np.empty((0, 2), dtype=int)
            self.bond_lengths = np.empty(0)
            self.bond_elements = []
            return
        symbols = [ATOMIC_NUMBERS.get(z, "X") for z in self.atomic_numbers]
        dist_matrix = squareform(pdist(self.coordinates))
        bi, bj, bl = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                rc = COVALENT_RADII.get(symbols[i], 0.77) + COVALENT_RADII.get(symbols[j], 0.77)
                if dist_matrix[i, j] < rc + BOND_TOLERANCE:
                    bi.append(i)
                    bj.append(j)
                    bl.append(dist_matrix[i, j])
        self.bond_indices = np.column_stack([bi, bj]) if bi else np.empty((0, 2), dtype=int)
        self.bond_lengths = np.array(bl, dtype=np.float64)
        self.bond_elements = [tuple(sorted((symbols[i], symbols[j]))) for i, j in zip(bi, bj)]


@dataclass
class BondEnergyData:
    mol_id: str
    bond_labels: list
    bond_energies_net: list
    bond_idx_pairs: list
    bond_elem_pairs: list
    atomization_energy: float
    gross_sigma: np.ndarray
    gross_pi: np.ndarray
    gross_total: np.ndarray
    net_sigma: np.ndarray
    net_pi: np.ndarray
    net_total: np.ndarray
    composite_table: np.ndarray
    bebop_method: str = ""
    bebop_version: str = ""


def extract_mol_id(filename):
    stem = Path(filename).name
    while stem.endswith((".bz2", ".out", ".log", ".json")):
        stem = stem.rsplit(".", 1)[0]
    stem = re.sub(r"_bopout$", "", stem)
    return stem


def _compute_formula(atomic_numbers):
    counts = Counter(ATOMIC_NUMBERS.get(z, "X") for z in atomic_numbers)
    parts = []
    for elem in ["C", "H"]:
        if elem in counts:
            parts.append(elem + (str(counts.pop(elem)) if counts[elem] > 1 else ""))
    for elem in sorted(counts):
        parts.append(elem + (str(counts[elem]) if counts[elem] > 1 else ""))
    return "".join(parts)


def _write_str_dataset(group, name, str_list):
    dt = h5py.special_dtype(vlen=str)
    ds = group.create_dataset(name, shape=(len(str_list),), dtype=dt, chunks=True, compression="gzip")
    for i, s in enumerate(str_list):
        ds[i] = s


def parse_gaussian_output(text, mol_id):
    # 1. Coordinates (last orientation block)
    orient_pat = re.compile(
        r"(?:Standard|Input) orientation:.*?-{60,}\s*\n"
        r"\s*Center\s+Atomic\s+Atomic\s+Coordinates \(Angstroms\)\s*\n"
        r"\s*Number\s+Number\s+Type\s+X\s+Y\s+Z\s*\n"
        r"\s*-{60,}\s*\n(.*?)\s*-{60,}",
        re.DOTALL,
    )
    orient_matches = list(orient_pat.finditer(text))
    if not orient_matches:
        raise ValueError(f"[{mol_id}] No orientation block found")
    coord_block = orient_matches[-1].group(1)
    atomic_numbers, coords = [], []
    for line in coord_block.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 6:
            try:
                atomic_numbers.append(int(parts[1]))
                coords.append([float(parts[3]), float(parts[4]), float(parts[5])])
            except (ValueError, IndexError):
                continue
    if not coords:
        raise ValueError(f"[{mol_id}] No coordinates parsed")
    atomic_numbers = np.array(atomic_numbers, dtype=int)
    coordinates = np.array(coords, dtype=np.float64)

    # 2. SCF energy (last)
    scf_matches = re.findall(r"SCF Done:.*?=\s*([-\d.]+)", text)
    if not scf_matches:
        raise ValueError(f"[{mol_id}] No SCF energy found")
    scf_energy = float(scf_matches[-1])

    # 3. Find post-optimization section
    converged = "Stationary point found" in text or "Optimization completed" in text
    post_opt_start = 0
    for marker in ["Stationary point found", "Optimization completed", "-- Stationary point found"]:
        pos = text.rfind(marker)
        if pos > post_opt_start:
            post_opt_start = pos
    post_opt_text = text[post_opt_start:] if post_opt_start > 0 else text

    # 4. Mulliken charges (from post-opt only)
    mull_pat = re.compile(r"Mulliken charges:\s*\n\s*\d+\s*\n(.*?)(?:Sum of Mulliken|^\s*$)", re.DOTALL | re.MULTILINE)
    mull_matches = list(mull_pat.finditer(post_opt_text))
    if not mull_matches:
        mull_matches = list(mull_pat.finditer(text))
    if not mull_matches:
        raise ValueError(f"[{mol_id}] No Mulliken charges found")
    charge_block = mull_matches[0].group(1)
    charges = []
    for line in charge_block.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 3:
            try:
                charges.append(float(parts[2]))
            except (ValueError, IndexError):
                continue
    if len(charges) != len(atomic_numbers):
        raise ValueError(f"[{mol_id}] Mulliken count mismatch: {len(charges)} vs {len(atomic_numbers)}")
    mulliken_charges = np.array(charges, dtype=np.float64)

    # 5. Orbital populations (from post-opt only)
    orbital_populations = None
    orbpop_pat = re.compile(r"Gross orbital populations:(.*?)(?:Condensed to atoms|Mulliken charges|Electronic spatial extent|\*\*\*\*)", re.DOTALL)
    orbpop_match = orbpop_pat.search(post_opt_text)
    if not orbpop_match:
        orbpop_all = list(orbpop_pat.finditer(text))
        orbpop_match = orbpop_all[-1] if orbpop_all else None

    if orbpop_match:
        orbpop_block = orbpop_match.group(1)
        orbital_populations = {}
        current_atom_idx = -1
        current_element = None

        for line in orbpop_block.strip().split("\n"):
            ls = line.strip()
            if not ls:
                continue
            if "Total" in ls and "Alpha" in ls:
                continue

            new_atom_pat = r"\s*(\d+)\s+(\d+)\s+([A-Z][a-z]?)\s+(.*)"
            m = re.match(new_atom_pat, ls)
            if m:
                current_atom_idx = int(m.group(2)) - 1
                current_element = m.group(3)
                remainder = m.group(4).strip()
            else:
                cont_m = re.match(r"\s*(\d+)\s+(.*)", ls)
                if not cont_m:
                    continue
                remainder = cont_m.group(2).strip()

            if current_atom_idx < 0:
                continue

            val_m = re.search(r"(-?\d+\.\d+)", remainder)
            if not val_m:
                continue

            orb_label = remainder[: val_m.start()].strip()
            values = remainder[val_m.start() :].split()

            try:
                total = float(values[0]) if len(values) > 0 else 0.0
                alpha = float(values[1]) if len(values) > 1 else 0.0
                beta = float(values[2]) if len(values) > 2 else 0.0
                spin = float(values[3]) if len(values) > 3 else 0.0
            except (ValueError, IndexError):
                continue

            if current_atom_idx not in orbital_populations:
                orbital_populations[current_atom_idx] = {
                    "element": current_element,
                    "orbitals": [],
                    "s_total": 0.0,
                    "p_total": 0.0,
                    "d_total": 0.0,
                    "f_total": 0.0,
                }

            orbital_populations[current_atom_idx]["orbitals"].append(
                {"label": orb_label, "total": total, "alpha": alpha, "beta": beta, "spin": spin}
            )

            shell_m = re.match(r"\d+([SPDF])", orb_label.upper())
            if shell_m:
                sh = shell_m.group(1)
                if sh == "S":
                    orbital_populations[current_atom_idx]["s_total"] += total
                elif sh == "P":
                    orbital_populations[current_atom_idx]["p_total"] += total
                elif sh == "D":
                    orbital_populations[current_atom_idx]["d_total"] += total
                elif sh == "F":
                    orbital_populations[current_atom_idx]["f_total"] += total

    data = GaussianData(
        mol_id=mol_id,
        atomic_numbers=atomic_numbers,
        coordinates=coordinates,
        scf_energy=scf_energy,
        mulliken_charges=mulliken_charges,
        converged=converged,
        orbital_populations=orbital_populations,
    )
    data.compute_bonds()
    return data


def read_and_parse_gaussian(filepath):
    mol_id = extract_mol_id(filepath)
    try:
        if filepath.endswith(".bz2"):
            with bz2.open(filepath, "rt", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        else:
            with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        return parse_gaussian_output(text, mol_id)
    except Exception as e:
        log.warning("Failed to parse %s: %s", mol_id, e)
        return None


def _parse_bebop_bond_label(label):
    parts = label.split("-")
    if len(parts) != 2:
        raise ValueError("Cannot parse bond label: " + label)
    def _split(s):
        m = re.match(r"([A-Za-z]+)(\d+)", s)
        if not m:
            raise ValueError("Cannot parse atom label: " + s)
        return m.group(1), int(m.group(2)) - 1
    e1, i1 = _split(parts[0])
    e2, i2 = _split(parts[1])
    return (i1, i2), (e1, e2)


def parse_bond_energy_json(text, mol_id):
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("[%s] JSON parse error: %s", mol_id, e)
        return None
    if "bond energies" not in data or "sorted net bond energies" not in data:
        log.warning("[%s] JSON missing expected BEBOP keys", mol_id)
        return None
    try:
        ss = data["sorted net bond energies"]
        bond_labels = ss["sorted bonds"]
        bond_energies_net = ss["bond energies"]
        bond_idx_pairs, bond_elem_pairs = [], []
        for label in bond_labels:
            ip, ep = _parse_bebop_bond_label(label)
            bond_idx_pairs.append(ip)
            bond_elem_pairs.append(ep)
        be = data["bond energies"]
        return BondEnergyData(
            mol_id=mol_id,
            bond_labels=bond_labels,
            bond_energies_net=bond_energies_net,
            bond_idx_pairs=bond_idx_pairs,
            bond_elem_pairs=bond_elem_pairs,
            atomization_energy=float(data.get("atomization energy", 0.0)),
            gross_sigma=np.array(be["gross"]["sigma"], dtype=np.float64),
            gross_pi=np.array(be["gross"]["pi"], dtype=np.float64),
            gross_total=np.array(be["gross"]["total"], dtype=np.float64),
            net_sigma=np.array(be["net"]["sigma"], dtype=np.float64),
            net_pi=np.array(be["net"]["pi"], dtype=np.float64),
            net_total=np.array(be["net"]["total"], dtype=np.float64),
            composite_table=np.array(be["composite table"], dtype=np.float64),
            bebop_method=data.get("method", ""),
            bebop_version=data.get("version", ""),
        )
    except (KeyError, ValueError, IndexError) as e:
        log.warning("[%s] Error parsing BEBOP JSON: %s", mol_id, e)
        return None


def read_and_parse_json(filepath):
    mol_id = extract_mol_id(filepath)
    try:
        if filepath.endswith(".bz2"):
            with bz2.open(filepath, "rt", encoding="utf-8") as fh:
                text = fh.read()
        else:
            with open(filepath, "r", encoding="utf-8") as fh:
                text = fh.read()
        return parse_bond_energy_json(text, mol_id)
    except Exception as e:
        log.warning("Failed to parse JSON %s: %s", mol_id, e)
        return None


class HDF5Writer:
    def __init__(self, output_path):
        self.output_path = output_path
        self._bond_lengths = []
        self._bond_mol_ids = []
        self._bond_elem_pairs = []
        self._charges = []
        self._charge_mol_ids = []
        self._charge_elements = []
        self._bond_energies_flat = []
        self._be_mol_ids = []
        self._be_pairs = []
        self._be_labels = []
        self._atomization_energies = []
        self._ae_mol_ids = []
        self._scf_energies = []
        self._scf_mol_ids = []
        self._s_pops = []
        self._p_pops = []
        self._d_pops = []
        self._f_pops = []
        self._pop_mol_ids = []
        self._pop_elements = []
        self._mol_count = 0
        self._failed = 0

    def open(self):
        self.h5 = h5py.File(self.output_path, "w")
        self.h5.attrs["dataset"] = "QM9 (re-computed)"
        self.h5.attrs["method"] = "B3LYP-D3BJ/CBSB7"
        self.h5.attrs["route"] = "#Opt EmpiricalDispersion=GD3BJ B3LYP/CBSB7 SCF=TIGHT"
        self.h5.create_group("molecules")
        self.h5.create_group("lookup")
        return self

    def write_molecule(self, gauss, bond_e=None):
        mol_id = gauss.mol_id
        mg = self.h5["molecules"].create_group(mol_id)
        symbols = [ATOMIC_NUMBERS.get(z, "X") for z in gauss.atomic_numbers]
        mg.attrs["formula"] = _compute_formula(gauss.atomic_numbers)
        mg.attrs["n_atoms"] = len(gauss.atomic_numbers)
        mg.attrs["n_bonds"] = len(gauss.bond_lengths)
        mg.attrs["converged"] = gauss.converged

        gg = mg.create_group("geometry")
        gg.create_dataset("coordinates", data=gauss.coordinates, dtype="float64")
        gg.create_dataset("atomic_numbers", data=gauss.atomic_numbers, dtype="int32")
        gg.attrs["scf_energy"] = gauss.scf_energy
        if len(gauss.bond_indices) > 0:
            gg.create_dataset("bond_indices", data=gauss.bond_indices, dtype="int32")
            gg.create_dataset("bond_lengths", data=gauss.bond_lengths, dtype="float64")

        mk = mg.create_group("mulliken")
        mk.create_dataset("charges", data=gauss.mulliken_charges, dtype="float64")

        if gauss.orbital_populations:
            op = mk.create_group("orbital_populations")
            na = len(gauss.atomic_numbers)
            s_p = np.zeros(na, dtype=np.float64)
            p_p = np.zeros(na, dtype=np.float64)
            d_p = np.zeros(na, dtype=np.float64)
            f_p = np.zeros(na, dtype=np.float64)
            for ai, ad in gauss.orbital_populations.items():
                if ai < na:
                    s_p[ai] = ad["s_total"]
                    p_p[ai] = ad["p_total"]
                    d_p[ai] = ad["d_total"]
                    f_p[ai] = ad.get("f_total", 0.0)
            op.create_dataset("s_population", data=s_p, dtype="float64")
            op.create_dataset("p_population", data=p_p, dtype="float64")
            op.create_dataset("d_population", data=d_p, dtype="float64")
            op.create_dataset("f_population", data=f_p, dtype="float64")
            dt = h5py.special_dtype(vlen=str)
            all_labels, all_totals, all_ai = [], [], []
            for ai2 in sorted(gauss.orbital_populations.keys()):
                for orb in gauss.orbital_populations[ai2]["orbitals"]:
                    all_labels.append(orb["label"])
                    all_totals.append(orb["total"])
                    all_ai.append(ai2)
            if all_labels:
                op.create_dataset("orbital_totals", data=np.array(all_totals, dtype="float64"))
                op.create_dataset("orbital_atom_indices", data=np.array(all_ai, dtype="int32"))
                lds = op.create_dataset("orbital_labels", shape=(len(all_labels),), dtype=dt)
                for i, s in enumerate(all_labels):
                    lds[i] = s

        if bond_e and bond_e.bond_energies_net:
            be = mg.create_group("bond_energies")
            be.attrs["atomization_energy"] = bond_e.atomization_energy
            be.attrs["bebop_method"] = bond_e.bebop_method
            be.attrs["bebop_version"] = bond_e.bebop_version
            be.create_dataset("net_energies", data=np.array(bond_e.bond_energies_net, dtype="float64"))
            dt = h5py.special_dtype(vlen=str)
            lds = be.create_dataset("bond_labels", shape=(len(bond_e.bond_labels),), dtype=dt)
            for i, s in enumerate(bond_e.bond_labels):
                lds[i] = s
            be.create_dataset("bond_idx_pairs", data=np.array(bond_e.bond_idx_pairs, dtype="int32"))
            mx = be.create_group("matrices")
            for k, v in [("gross_sigma", bond_e.gross_sigma), ("gross_pi", bond_e.gross_pi),
                         ("gross_total", bond_e.gross_total), ("net_sigma", bond_e.net_sigma),
                         ("net_pi", bond_e.net_pi), ("net_total", bond_e.net_total),
                         ("composite_table", bond_e.composite_table)]:
                mx.create_dataset(k, data=v, dtype="float64", compression="gzip")

        for length, ep in zip(gauss.bond_lengths, gauss.bond_elements):
            self._bond_lengths.append(length)
            self._bond_mol_ids.append(mol_id)
            self._bond_elem_pairs.append(ep[0] + "-" + ep[1])
        for charge, z in zip(gauss.mulliken_charges, gauss.atomic_numbers):
            self._charges.append(charge)
            self._charge_mol_ids.append(mol_id)
            self._charge_elements.append(ATOMIC_NUMBERS.get(z, "X"))
        self._scf_energies.append(gauss.scf_energy)
        self._scf_mol_ids.append(mol_id)
        if gauss.orbital_populations:
            for ai3 in sorted(gauss.orbital_populations.keys()):
                ad3 = gauss.orbital_populations[ai3]
                z3 = gauss.atomic_numbers[ai3] if ai3 < len(gauss.atomic_numbers) else 0
                self._s_pops.append(ad3["s_total"])
                self._p_pops.append(ad3["p_total"])
                self._d_pops.append(ad3["d_total"])
                self._f_pops.append(ad3.get("f_total", 0.0))
                self._pop_mol_ids.append(mol_id)
                self._pop_elements.append(ATOMIC_NUMBERS.get(z3, "X"))
        if bond_e and bond_e.bond_energies_net:
            for energy, label, ep2 in zip(bond_e.bond_energies_net, bond_e.bond_labels, bond_e.bond_elem_pairs):
                self._bond_energies_flat.append(energy)
                self._be_mol_ids.append(mol_id)
                self._be_pairs.append(ep2[0] + "-" + ep2[1])
                self._be_labels.append(label)
            self._atomization_energies.append(bond_e.atomization_energy)
            self._ae_mol_ids.append(mol_id)
        self._mol_count += 1

    def _write_readme(self):
        lines = []
        lines.append("=============================================================================")
        lines.append(" QM9 + BEBOP Unified Dataset -- HDF5 Directory Guide")
        lines.append("=============================================================================")
        lines.append("")
        lines.append(" Created by: qm9_to_hdf5.py")
        lines.append(" Gaussian method: #Opt EmpiricalDispersion=GD3BJ B3LYP/CBSB7 SCF=TIGHT")
        lines.append(" Bond energies: BEBOP-1 (Bond Energy from Bond Order Population)")
        lines.append(" Units: Coordinates (Angstrom), SCF energy (Hartree),")
        lines.append("        Bond/atomization energies (kcal/mol), Mulliken charges (e),")
        lines.append("        Orbital populations (electrons)")
        lines.append("")
        lines.append("=============================================================================")
        lines.append(" /molecules/<mol_id>/ -- Per-molecule hierarchical data")
        lines.append("=============================================================================")
        lines.append("  Attrs: formula, n_atoms, n_bonds, converged")
        lines.append("")
        lines.append("  /geometry/                       -- FROM GAUSSIAN (post-optimization)")
        lines.append("    coordinates       (N,3) float64   Cartesian coords (Angstrom)")
        lines.append("    atomic_numbers    (N,)  int32     Atomic numbers")
        lines.append("    bond_indices      (M,2) int32     Atom index pairs for each bond")
        lines.append("    bond_lengths      (M,)  float64   Bond lengths (Angstrom)")
        lines.append("    @scf_energy       float64         SCF total energy (Hartree)")
        lines.append("")
        lines.append("  /mulliken/                       -- FROM GAUSSIAN (post-optimization)")
        lines.append("    charges           (N,)  float64   Mulliken atomic charges (e)")
        lines.append("")
        lines.append("    /orbital_populations/          -- FROM GAUSSIAN (pop=full, post-opt)")
        lines.append("      s_population    (N,)  float64   Total S orbital pop per atom")
        lines.append("      p_population    (N,)  float64   Total P orbital pop per atom")
        lines.append("      d_population    (N,)  float64   Total D orbital pop per atom")
        lines.append("      f_population    (N,)  float64   Total F orbital pop per atom")
        lines.append("      orbital_totals  (K,)  float64   Individual orbital populations")
        lines.append("      orbital_labels  (K,)  string    Labels (1S, 2PX, 6D 0, 8F+1...)")
        lines.append("      orbital_atom_indices (K,) int32 Which atom each orbital belongs to")
        lines.append("")
        lines.append("  /bond_energies/                  -- FROM BEBOP JSON")
        lines.append("    net_energies      (B,)  float64   Sorted net bond energies (kcal/mol)")
        lines.append("    bond_labels       (B,)  string    Bond labels (C4-C5, O1-H7)")
        lines.append("    bond_idx_pairs    (B,2) int32     0-based atom index pairs")
        lines.append("    @atomization_energy, @bebop_method, @bebop_version")
        lines.append("")
        lines.append("    /matrices/                     -- Full NxN interaction matrices")
        lines.append("      gross_sigma, gross_pi, gross_total  (N,N) float64")
        lines.append("      net_sigma, net_pi, net_total        (N,N) float64")
        lines.append("      composite_table                     (N,N) float64 (diag=self-E)")
        lines.append("")
        lines.append("=============================================================================")
        lines.append(" /lookup/ -- Flat arrays for fast cross-molecule queries")
        lines.append("=============================================================================")
        lines.append("  all_bond_lengths, bond_mol_ids, bond_element_pairs")
        lines.append("  all_charges, charge_mol_ids, charge_elements")
        lines.append("  all_s/p/d/f_populations, pop_mol_ids, pop_elements")
        lines.append("  all_scf_energies, scf_mol_ids")
        lines.append("  all_bond_energies, be_mol_ids, be_element_pairs, be_bond_labels")
        lines.append("  all_atomization_energies, ae_mol_ids")
        lines.append("")
        lines.append("=============================================================================")
        lines.append(" EXAMPLE: Accessing Gaussian output data")
        lines.append("=============================================================================")
        lines.append("")
        lines.append("  import h5py, numpy as np")
        lines.append("")
        lines.append('  with h5py.File("qm9_data.h5", "r") as f:')
        lines.append('      mol = f["molecules/dsgdb9nsd_000200"]')
        lines.append('      coords = mol["geometry/coordinates"][:]')
        lines.append('      atoms  = mol["geometry/atomic_numbers"][:]')
        lines.append('      scf_e  = mol["geometry"].attrs["scf_energy"]')
        lines.append('      charges = mol["mulliken/charges"][:]')
        lines.append('      s_pop = mol["mulliken/orbital_populations/s_population"][:]')
        lines.append('      p_pop = mol["mulliken/orbital_populations/p_population"][:]')
        lines.append('      d_pop = mol["mulliken/orbital_populations/d_population"][:]')
        lines.append('      f_pop = mol["mulliken/orbital_populations/f_population"][:]')
        lines.append("")
        lines.append("=============================================================================")
        lines.append(" EXAMPLE: Cross-molecule queries (using lookup tables)")
        lines.append("=============================================================================")
        lines.append("")
        lines.append('  with h5py.File("qm9_data.h5", "r") as f:')
        lines.append('      lengths = f["lookup/all_bond_lengths"][:]')
        lines.append('      pairs   = [s.decode() for s in f["lookup/bond_element_pairs"][:]]')
        lines.append('      mask_ch = np.array([p == "C-H" for p in pairs])')
        lines.append("      ch_lengths = lengths[mask_ch]")
        lines.append("")
        lines.append('      s_pops = f["lookup/all_s_populations"][:]')
        lines.append('      pop_el = [s.decode() for s in f["lookup/pop_elements"][:]]')
        lines.append('      mask_c = np.array([e == "C" for e in pop_el])')
        lines.append("      c_s_pops = s_pops[mask_c]")
        lines.append("")
        lines.append("=============================================================================")
        lines.append(" EXAMPLE: Accessing BEBOP data")
        lines.append("=============================================================================")
        lines.append("")
        lines.append('  with h5py.File("qm9_data.h5", "r") as f:')
        lines.append('      mol = f["molecules/dsgdb9nsd_000200"]')
        lines.append('      energies = mol["bond_energies/net_energies"][:]')
        lines.append('      labels = [s.decode() for s in mol["bond_energies/bond_labels"][:]]')
        lines.append('      net_sigma = mol["bond_energies/matrices/net_sigma"][:]')
        lines.append('      comp = mol["bond_energies/matrices/composite_table"][:]')
        lines.append("      self_energies = np.diag(comp)")
        lines.append("")
        lines.append("=============================================================================")
        lines.append(" EXAMPLE: Correlating orbital pops with BEBOP energies")
        lines.append("=============================================================================")
        lines.append("")
        lines.append('  with h5py.File("qm9_data.h5", "r") as f:')
        lines.append("      import matplotlib.pyplot as plt")
        lines.append("      p_all, sig_all = [], []")
        lines.append('      for mid in list(f["molecules"].keys())[:1000]:')
        lines.append('          mol = f["molecules/" + mid]')
        lines.append('          if "bond_energies/matrices" not in mol: continue')
        lines.append('          if "mulliken/orbital_populations" not in mol: continue')
        lines.append('          pp = mol["mulliken/orbital_populations/p_population"][:]')
        lines.append('          ns = mol["bond_energies/matrices/net_sigma"][:]')
        lines.append('          bi = mol["bond_energies/bond_idx_pairs"][:]')
        lines.append("          for k in range(len(bi)):")
        lines.append("              i, j = bi[k]")
        lines.append("              se = ns[i,j] if ns[i,j] != 0 else ns[j,i]")
        lines.append("              if se != 0:")
        lines.append("                  p_all.append((pp[i]+pp[j])/2)")
        lines.append("                  sig_all.append(se)")
        lines.append("      plt.scatter(p_all, sig_all, s=1, alpha=0.3)")
        lines.append('      plt.savefig("p_pop_vs_sigma.png", dpi=300)')
        lines.append("")
        lines.append("=============================================================================")
        readme_text = "\n".join(lines)
        self.h5.create_dataset("README", data=readme_text.encode("utf-8"))

    def finalize(self):
        lg = self.h5["lookup"]
        if self._bond_lengths:
            lg.create_dataset("all_bond_lengths", data=np.array(self._bond_lengths), dtype="float64", chunks=True, compression="gzip")
            _write_str_dataset(lg, "bond_mol_ids", self._bond_mol_ids)
            _write_str_dataset(lg, "bond_element_pairs", self._bond_elem_pairs)
        if self._charges:
            lg.create_dataset("all_charges", data=np.array(self._charges), dtype="float64", chunks=True, compression="gzip")
            _write_str_dataset(lg, "charge_mol_ids", self._charge_mol_ids)
            _write_str_dataset(lg, "charge_elements", self._charge_elements)
        if self._scf_energies:
            lg.create_dataset("all_scf_energies", data=np.array(self._scf_energies), dtype="float64", chunks=True, compression="gzip")
            _write_str_dataset(lg, "scf_mol_ids", self._scf_mol_ids)
        if self._s_pops:
            lg.create_dataset("all_s_populations", data=np.array(self._s_pops), dtype="float64", chunks=True, compression="gzip")
            lg.create_dataset("all_p_populations", data=np.array(self._p_pops), dtype="float64", chunks=True, compression="gzip")
            lg.create_dataset("all_d_populations", data=np.array(self._d_pops), dtype="float64", chunks=True, compression="gzip")
            lg.create_dataset("all_f_populations", data=np.array(self._f_pops), dtype="float64", chunks=True, compression="gzip")
            _write_str_dataset(lg, "pop_mol_ids", self._pop_mol_ids)
            _write_str_dataset(lg, "pop_elements", self._pop_elements)
        if self._bond_energies_flat:
            lg.create_dataset("all_bond_energies", data=np.array(self._bond_energies_flat), dtype="float64", chunks=True, compression="gzip")
            _write_str_dataset(lg, "be_mol_ids", self._be_mol_ids)
            _write_str_dataset(lg, "be_element_pairs", self._be_pairs)
            _write_str_dataset(lg, "be_bond_labels", self._be_labels)
        if self._atomization_energies:
            lg.create_dataset("all_atomization_energies", data=np.array(self._atomization_energies), dtype="float64", chunks=True, compression="gzip")
            _write_str_dataset(lg, "ae_mol_ids", self._ae_mol_ids)
        self._write_readme()
        self.h5.attrs["n_molecules"] = self._mol_count
        self.h5.attrs["n_failed"] = self._failed
        self.h5.attrs["total_bonds"] = len(self._bond_lengths)
        self.h5.attrs["total_atoms"] = len(self._charges)
        self.h5.close()
        log.info("HDF5 written: %s", self.output_path)
        log.info("  Molecules: %d (%d failed)", self._mol_count, self._failed)
        log.info("  Total bonds: %d", len(self._bond_lengths))
        log.info("  Total atoms: %d", len(self._charges))
        log.info("  Total bond energies: %d", len(self._bond_energies_flat))


def match_files(gaussian_dir, json_dir, recursive=False):
    gauss_files, json_files = {}, {}
    if recursive:
        for root, dirs, files in os.walk(gaussian_dir):
            for fname in files:
                fp = os.path.join(root, fname)
                if fname.endswith((".out.bz2", ".log.bz2", ".out", ".log")):
                    gauss_files[extract_mol_id(fname)] = fp
        jroot = json_dir if json_dir != gaussian_dir else gaussian_dir
        for root, dirs, files in os.walk(jroot):
            for fname in files:
                fp = os.path.join(root, fname)
                if fname.endswith((".json.bz2", ".json")):
                    json_files[extract_mol_id(fname)] = fp
    else:
        for ext in ["*.out.bz2", "*.log.bz2", "*.out", "*.log"]:
            for fp in glob.glob(os.path.join(gaussian_dir, ext)):
                gauss_files[extract_mol_id(fp)] = fp
        for ext in ["*.json.bz2", "*.json"]:
            for fp in glob.glob(os.path.join(json_dir, ext)):
                json_files[extract_mol_id(fp)] = fp
    pairs, matched = [], 0
    for mid, gp in sorted(gauss_files.items()):
        jp = json_files.get(mid, None)
        pairs.append((gp, jp))
        if jp:
            matched += 1
    log.info("Found %d Gaussian, %d JSON files. Matched: %d", len(gauss_files), len(json_files), matched)
    return pairs


def process_one_molecule(args):
    gp, jp = args
    gd = read_and_parse_gaussian(gp)
    bd = read_and_parse_json(jp) if jp else None
    return gd, bd


def run_pipeline(gaussian_dir, json_dir, output, n_workers=4, limit=None, recursive=False):
    file_pairs = match_files(gaussian_dir, json_dir, recursive=recursive)
    if not file_pairs:
        log.error("No files found.")
        return
    if limit is not None:
        file_pairs = file_pairs[:limit]
        log.info("Limited to first %d molecule(s)", limit)
    writer = HDF5Writer(output)
    writer.open()
    try:
        from tqdm import tqdm
        progress = tqdm(total=len(file_pairs), desc="Processing")
    except ImportError:
        progress = None
    if n_workers <= 1:
        for pair in file_pairs:
            gd, bd = process_one_molecule(pair)
            if gd:
                writer.write_molecule(gd, bd)
            else:
                writer._failed += 1
            if progress:
                progress.update(1)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futs = {exe.submit(process_one_molecule, p): p for p in file_pairs}
            for fut in as_completed(futs):
                try:
                    gd, bd = fut.result()
                    if gd:
                        writer.write_molecule(gd, bd)
                    else:
                        writer._failed += 1
                except Exception as e:
                    writer._failed += 1
                    log.warning("Processing error: %s", e)
                if progress:
                    progress.update(1)
    if progress:
        progress.close()
    writer.finalize()


def test_single_gaussian(filepath):
    print("Parsing: " + filepath)
    print("=" * 60)
    data = read_and_parse_gaussian(filepath)
    if data is None:
        print("FAILED: Could not parse file")
        return
    symbols = [ATOMIC_NUMBERS.get(z, "X") for z in data.atomic_numbers]
    formula = _compute_formula(data.atomic_numbers)
    print("  Molecule ID   : " + data.mol_id)
    print("  Formula       : " + formula)
    print("  Atoms         : %d" % len(data.atomic_numbers))
    print("  Converged     : %s" % data.converged)
    print("  SCF Energy    : %.10f Ha" % data.scf_energy)
    print()
    print("  Coordinates (Angstrom):")
    print("  %4s  %4s  %12s  %12s  %12s  %10s" % ("#", "Elem", "X", "Y", "Z", "Charge"))
    print("  " + "-" * 56)
    for i in range(len(data.atomic_numbers)):
        sym = ATOMIC_NUMBERS.get(data.atomic_numbers[i], "X")
        c = data.coordinates[i]
        q = data.mulliken_charges[i]
        print("  %4d  %4s  %12.6f  %12.6f  %12.6f  %+10.6f" % (i + 1, sym, c[0], c[1], c[2], q))
    print()
    print("  Detected bonds: %d" % len(data.bond_lengths))
    if len(data.bond_lengths) > 0:
        print("  %4s  %6s  %6s  %6s  %10s" % ("#", "Atom1", "Atom2", "Pair", "Length"))
        print("  " + "-" * 40)
        for i in range(len(data.bond_lengths)):
            ip = data.bond_indices[i]
            a1 = symbols[ip[0]] + str(ip[0] + 1)
            a2 = symbols[ip[1]] + str(ip[1] + 1)
            ps = data.bond_elements[i][0] + "-" + data.bond_elements[i][1]
            print("  %4d  %6s  %6s  %6s  %10.4f A" % (i + 1, a1, a2, ps, data.bond_lengths[i]))
    print()
    print("  Mulliken charge summary:")
    for elem in sorted(set(symbols)):
        mask = [s == elem for s in symbols]
        ec = data.mulliken_charges[mask]
        print("    %s: mean=%+.4f, range=[%+.4f, %+.4f]" % (elem, ec.mean(), ec.min(), ec.max()))
    if data.orbital_populations:
        print()
        print("  Orbital populations (pop=full): %d atoms" % len(data.orbital_populations))
        print("  %4s  %4s  %10s  %10s  %10s  %10s  %8s" % ("#", "Elem", "S pop", "P pop", "D pop", "F pop", "Orbitals"))
        print("  " + "-" * 62)
        for ai in sorted(data.orbital_populations.keys()):
            ad = data.orbital_populations[ai]
            norb = len(ad["orbitals"])
            print("  %4d  %4s  %10.4f  %10.4f  %10.4f  %10.4f  %8d" % (
                ai + 1, ad["element"], ad["s_total"], ad["p_total"], ad["d_total"], ad.get("f_total", 0.0), norb))
        first_idx = sorted(data.orbital_populations.keys())[0]
        first = data.orbital_populations[first_idx]
        print()
        print("  Orbital detail (atom %d -- %s, %d orbitals):" % (first_idx + 1, first["element"], len(first["orbitals"])))
        print("    %8s  %10s  %10s  %10s  %6s" % ("Label", "Total", "Alpha", "Beta", "Shell"))
        print("    " + "-" * 50)
        for orb in first["orbitals"]:
            sm = re.match(r"\d+([SPDF])", orb["label"].upper())
            sh = sm.group(1) if sm else "?"
            print("    %8s  %10.5f  %10.5f  %10.5f  %6s" % (orb["label"], orb["total"], orb["alpha"], orb["beta"], sh))
    else:
        print()
        print("  Orbital populations: not found (needs pop=full in route)")
    print()
    print("=" * 60)
    print("  Parser OK")
    print("=" * 60)


def test_single_json(filepath):
    print("Parsing: " + filepath)
    print("=" * 60)
    data = read_and_parse_json(filepath)
    if data is None:
        print("FAILED: Could not parse file")
        return
    print("  Molecule ID       : " + data.mol_id)
    print("  BEBOP method      : " + data.bebop_method)
    print("  BEBOP version     : " + data.bebop_version)
    print("  Atomization energy: %.4f kcal/mol" % data.atomization_energy)
    print("  Matrix size       : %dx%d" % (data.net_total.shape[0], data.net_total.shape[1]))
    print()
    print("  Sorted net bond energies:")
    print("  %4s  %10s  %12s  %10s  %14s" % ("#", "Bond", "Atoms", "Elements", "Energy"))
    print("  " + "-" * 56)
    for i in range(len(data.bond_labels)):
        ip = data.bond_idx_pairs[i]
        ep = data.bond_elem_pairs[i]
        print("  %4d  %10s  (%4d,%4d)  %10s  %14.4f kcal/mol" % (
            i + 1, data.bond_labels[i], ip[0], ip[1], ep[0] + "-" + ep[1], data.bond_energies_net[i]))
    print()
    print("  Composite table diagonal (atomic self-energies):")
    diag = np.diag(data.composite_table)
    for i, val in enumerate(diag):
        if val != 0.0:
            print("    Atom %d: %12.4f kcal/mol" % (i + 1, val))
    print()
    print("  Matrix summary:")
    for name, mat in [("Gross sigma", data.gross_sigma), ("Gross pi", data.gross_pi),
                      ("Net sigma", data.net_sigma), ("Net pi", data.net_pi)]:
        nz = np.count_nonzero(mat)
        if nz > 0:
            vals = mat[mat != 0]
            print("    %12s: %3d nonzero, range=[%.2f, %.2f]" % (name, nz, vals.min(), vals.max()))
        else:
            print("    %12s: all zeros" % name)
    print()
    print("=" * 60)
    print("  Parser OK")
    print("=" * 60)


def _describe_by_group(values, labels, unit="", title=""):
    unique = sorted(set(labels))
    print("    %s (by type):" % title)
    print("    %8s  %8s  %10s  %10s  %10s  %10s  %10s" % ("type", "count", "mean", "std", "min", "50%", "max"))
    print("    %s  %s  %s  %s  %s  %s  %s" % ("-" * 8, "-" * 8, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10))
    for g in unique:
        mask = np.array([x == g for x in labels])
        sub = values[mask]
        med = np.median(sub)
        print("    %8s  %8d  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f" % (g, len(sub), sub.mean(), sub.std(), sub.min(), med, sub.max()))
    print()


def inspect_hdf5(filepath):
    import sys
    if not os.path.exists(filepath):
        print("Error: File not found: " + filepath)
        sys.exit(1)
    with h5py.File(filepath, "r") as f:
        file_size = os.path.getsize(filepath)
        print()
        print("=" * 72)
        print("  HDF5 SUMMARY: " + filepath)
        print("  File size: %.1f MB" % (file_size / (1024 ** 2)))
        print("=" * 72)
        print()
        print("  ROOT ATTRIBUTES")
        print("  " + "-" * 40)
        for k, v in f.attrs.items():
            print("    %-30s %s" % (k, v))
        print()
        print("  STRUCTURE OVERVIEW")
        print("  " + "-" * 40)
        counts = {"groups": 0, "datasets": 0}
        def _cnt(name, obj):
            if isinstance(obj, h5py.Group):
                counts["groups"] += 1
            else:
                counts["datasets"] += 1
        f.visititems(_cnt)
        print("    Total groups .......... %d" % counts["groups"])
        print("    Total datasets ........ %d" % counts["datasets"])
        mol_ids = []
        if "molecules" in f:
            mol_ids = list(f["molecules"].keys())
            print("    Molecules ............. %d" % len(mol_ids))
            if mol_ids:
                sid = mol_ids[0]
                sample = f["molecules/" + sid]
                print()
                print("  SCHEMA (from %s):" % sid)
                print("  " + "-" * 40)
                def _schema(name, obj):
                    depth = name.count("/") + 1
                    pfx = "    " + "  " * depth
                    if isinstance(obj, h5py.Group):
                        print("%s%s/" % (pfx, name))
                        for ak, av in obj.attrs.items():
                            print("%s  @%s = %s" % (pfx, ak, av))
                    elif isinstance(obj, h5py.Dataset):
                        print("%s%s  %s %s" % (pfx, name, obj.shape, obj.dtype))
                        for ak, av in obj.attrs.items():
                            print("%s  @%s = %s" % (pfx, ak, av))
                sample.visititems(_schema)
        if "README" in f:
            print()
            print("=" * 72)
            print("  DIRECTORY GUIDE + EXAMPLES")
            print("=" * 72)
            print(f["README"][()].decode())
        if "lookup" in f:
            lg = f["lookup"]
            print("=" * 72)
            print("  DATASET STATISTICS")
            print("=" * 72)
            if "all_bond_lengths" in lg:
                bl = lg["all_bond_lengths"][:]
                pairs = [s.decode() for s in lg["bond_element_pairs"][:]]
                print()
                print("  BOND LENGTHS (from geometry) -- %d total" % len(bl))
                print("  " + "-" * 40)
                _describe_by_group(bl, pairs, title="Bond lengths")
            if "all_charges" in lg:
                ch = lg["all_charges"][:]
                elems = [s.decode() for s in lg["charge_elements"][:]]
                print("  MULLIKEN CHARGES -- %d total" % len(ch))
                print("  " + "-" * 40)
                _describe_by_group(ch, elems, title="Mulliken charges")
            if "all_s_populations" in lg:
                sp = lg["all_s_populations"][:]
                pp = lg["all_p_populations"][:]
                dp = lg["all_d_populations"][:]
                fp = lg["all_f_populations"][:] if "all_f_populations" in lg else np.zeros_like(sp)
                pe = [s.decode() for s in lg["pop_elements"][:]]
                print("  ORBITAL POPULATIONS (Mulliken, pop=full) -- %d atoms" % len(sp))
                print("  " + "-" * 40)
                for pn, pa in [("S population", sp), ("P population", pp), ("D population", dp), ("F population", fp)]:
                    if np.all(pa == 0):
                        print("    %s: all zeros" % pn)
                        print()
                        continue
                    _describe_by_group(pa, pe, title=pn)
            if "all_scf_energies" in lg:
                scf = lg["all_scf_energies"][:]
                pcts = np.percentile(scf, [25, 50, 75])
                print("  SCF ENERGIES -- %d molecules" % len(scf))
                print("  " + "-" * 40)
                print("    mean=%.6f  std=%.6f  min=%.6f  50%%=%.6f  max=%.6f Ha" % (scf.mean(), scf.std(), scf.min(), pcts[1], scf.max()))
                print()
            if "all_bond_energies" in lg:
                be = lg["all_bond_energies"][:]
                bep = [s.decode() for s in lg["be_element_pairs"][:]]
                print("  BEBOP NET BOND ENERGIES -- %d total" % len(be))
                print("  " + "-" * 40)
                _describe_by_group(be, bep, title="Net bond energies")
            if "all_atomization_energies" in lg:
                ae = lg["all_atomization_energies"][:]
                pcts = np.percentile(ae, [25, 50, 75])
                print("  ATOMIZATION ENERGIES -- %d molecules" % len(ae))
                print("  " + "-" * 40)
                print("    mean=%.2f  std=%.2f  min=%.2f  50%%=%.2f  max=%.2f kcal/mol" % (ae.mean(), ae.std(), ae.min(), pcts[1], ae.max()))
                print()
        if "molecules" in f and mol_ids:
            has_mat = any("bond_energies/matrices" in f["molecules/" + m] for m in mol_ids[:10])
            if has_mat:
                print("=" * 72)
                print("  BEBOP MATRIX DECOMPOSITION STATISTICS")
                print("=" * 72)
                mnames = [("gross_sigma", "Gross sigma"), ("gross_pi", "Gross pi"), ("gross_total", "Gross total"),
                          ("net_sigma", "Net sigma"), ("net_pi", "Net pi"), ("net_total", "Net total")]
                decomp = {}
                self_e = {}
                log.info("Scanning molecules for matrix statistics...")
                for mid in mol_ids:
                    mol = f["molecules/" + mid]
                    if "bond_energies/matrices" not in mol or "geometry/atomic_numbers" not in mol:
                        continue
                    atoms = mol["geometry/atomic_numbers"][:]
                    syms = [ATOMIC_NUMBERS.get(z, "X") for z in atoms]
                    n = len(atoms)
                    mats = mol["bond_energies/matrices"]
                    if "composite_table" in mats:
                        dg = np.diag(mats["composite_table"][:])
                        for ii, val in enumerate(dg):
                            el = syms[ii]
                            self_e.setdefault(el, []).append(val)
                    for mk2, ml in mnames:
                        if mk2 not in mats:
                            continue
                        mt = mats[mk2][:]
                        for ii in range(n):
                            for jj in range(ii + 1, n):
                                val = mt[ii, jj]
                                if val == 0.0:
                                    val = mt[jj, ii]
                                if val == 0.0:
                                    continue
                                ep3 = "-".join(sorted((syms[ii], syms[jj])))
                                decomp.setdefault((mk2, ep3), []).append(val)
                if self_e:
                    print()
                    print("  ATOMIC SELF-ENERGIES (composite table diagonal)")
                    print("  " + "-" * 40)
                    print("    %6s  %8s  %10s  %10s  %10s  %10s  %10s" % ("elem", "count", "mean", "std", "min", "50%", "max"))
                    print("    %s  %s  %s  %s  %s  %s  %s" % ("-" * 6, "-" * 8, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10))
                    for el2 in sorted(self_e.keys()):
                        arr = np.array(self_e[el2])
                        print("    %6s  %8d  %10.2f  %10.2f  %10.2f  %10.2f  %10.2f" % (el2, len(arr), arr.mean(), arr.std(), arr.min(), np.median(arr), arr.max()))
                    print()
                for mk3, ml3 in mnames:
                    eps = sorted(set(e for (m, e) in decomp.keys() if m == mk3))
                    if not eps:
                        continue
                    allv = []
                    for e in eps:
                        allv.extend(decomp.get((mk3, e), []))
                    aa = np.array(allv)
                    if len(aa) == 0 or np.all(aa == 0):
                        print("  %s -- all zeros" % ml3.upper())
                        print()
                        continue
                    print("  %s -- %d pairwise values" % (ml3.upper(), len(aa)))
                    print("  " + "-" * 40)
                    print("    %8s  %8s  %10s  %10s  %10s  %10s  %10s" % ("type", "count", "mean", "std", "min", "50%", "max"))
                    print("    %s  %s  %s  %s  %s  %s  %s" % ("-" * 8, "-" * 8, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10))
                    for e2 in eps:
                        arr = np.array(decomp[(mk3, e2)])
                        print("    %8s  %8d  %10.2f  %10.2f  %10.2f  %10.2f  %10.2f" % (e2, len(arr), arr.mean(), arr.std(), arr.min(), np.median(arr), arr.max()))
                    print()
        print("=" * 72)
        print("  DATA QUALITY")
        print("=" * 72)
        nm = f.attrs.get("n_molecules", 0)
        nf = f.attrs.get("n_failed", 0)
        tot = nm + nf
        pct = (nm / tot * 100) if tot > 0 else 0
        print("    Processed ............. %d / %d (%.1f%%)" % (nm, tot, pct))
        print("    Failed ................ %d" % nf)
        if mol_ids:
            nwb = sum(1 for m in mol_ids if "bond_energies" in f["molecules/" + m])
            nwo = len(mol_ids) - nwb
            print("    With BEBOP data ....... %d" % nwb)
            print("    Without BEBOP data .... %d" % nwo)
        print()
        print("=" * 72)
        print("  Inspect complete.")
        print("=" * 72)
        print()


def main():
    parser = argparse.ArgumentParser(description="QM9 Gaussian + BEBOP JSON to HDF5")
    parser.add_argument("--gaussian-dir", default=None)
    parser.add_argument("--json-dir", default=None)
    parser.add_argument("--output", default="qm9_data.h5")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--inspect", default=None, metavar="HDF5_FILE")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--single-gaussian", default=None, metavar="FILE")
    parser.add_argument("--single-json", default=None, metavar="FILE")
    parser.add_argument("--recursive", action="store_true", default=False)
    args = parser.parse_args()
    if args.inspect:
        inspect_hdf5(args.inspect)
    elif args.single_gaussian:
        test_single_gaussian(args.single_gaussian)
    elif args.single_json:
        test_single_json(args.single_json)
    else:
        if not args.gaussian_dir or not args.json_dir:
            parser.error("--gaussian-dir and --json-dir required")
        run_pipeline(args.gaussian_dir, args.json_dir, args.output, args.n_workers, args.limit, args.recursive)


if __name__ == "__main__":
    main()
