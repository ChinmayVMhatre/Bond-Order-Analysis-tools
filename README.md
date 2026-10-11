# qm9-bond-orders

Parsing, storage and cross-method comparison of **bond-order matrices** for QM9
and Transition1x.

The project compares *ab initio* bond orders (MinPop/Mulliken, Mayer, Wiberg over
IAOs and NAOs) against *empirical* ones (Pauling, ReaxFF), and quantifies
transition-state character from bond-order matrices alone.

Every numeric claim in `docs/` was produced by running this code on a real
Gaussian output and, where possible, cross-checked against values printed in the
file itself. `tests/` pins those numbers as regression tests.

---

## Install

```bash
git clone https://github.com/<you>/qm9-bond-orders.git
cd qm9-bond-orders
pip install -e .                 # core
pip install -e ".[structures,embedding,cheminfo,dev]"   # everything
```

Modules sit flat at the repo root, so `import gaussian_parse` works both from a
notebook opened here and after `pip install -e .`.

---

## The four workflows

### 1. Gaussian output parsing for bond orders

`gaussian_parse.py`, `pauling_bo.py`

```python
import gaussian_parse as gp

jobs = gp.split_jobs(gp.read_log("mol.out.bz2"))   # Link1-aware
c    = gp.condensed_to_atoms(jobs[1], kind="mbs")  # plain | bonding | mbs
m    = gp.mayer(jobs[1])                           # valency + bond_orders split
g    = gp.gross_orbital_populations(jobs[1], "mbs")
d    = gp.load_all("mol.out.bz2")                  # everything, aligned
```

Also reads bond-order matrices **printed by other programs**:

```python
gp.bo_matrix(path, "wiberg_iao")   # Knizia / IboView
gp.bo_matrix(path, "wiberg_nao")   # NBO, pop=NBO
gp.bo_matrix(path, "pauling")      # diagonal = row-sum valence
gp.bo_matrix(path, "reaxff")       # diagonal = ABO
```

**Six parsing traps are documented and handled** — each one silently produces
wrong numbers rather than raising. See `docs/02_parsing_guide.md`. The two most
costly:

- The condensed-to-atoms **diagonal is not the population**; the **row sum** is.
- AO labels like `6D 0` break naive tokenisation (the bare `0` parses as a
  float), which turned 44 electrons into 43.9586.

### 2. HDF5 storage of QM9 bond-order data

`build_pauling_h5.py`, `browse_pauling_h5.py`, `inspect_mbs.py`,
`verify_subset.py`, `merge_flat_h5.py`, `describe_h5.py`, `browse_hdf5.py`

```bash
python build_pauling_h5.py --base-dir . --out qm9_pauling.h5 --nproc 8
python browse_pauling_h5.py qm9_pauling.h5 --mol dsgdb9nsd_000200 --matrix
python verify_subset.py qm9_pauling.h5 subset.csv
```

All of QM9 (133,885 molecules) fits in **~27 MB uncompressed, ~14 MB gzipped**,
via sparse `(i, j, order)` triples plus flat ragged arrays with offset pointers.
Dense `float64` matrices would be ~347 MB, and a `/molecules/<id>/` group
hierarchy adds ~200 MB of metadata before any data. File discovery is recursive.
See `docs/03_hdf5_storage_design.md`.

### 3. Transition-state analysis and n_TS decomposition

`nts_decompose.py`, `bo_scalar.py`, `reacting_bonds.py`, `bo_change.py`

```python
import nts_decompose as D

r = D.decompose(BO_R, BO_TS, BO_P)      # per-bond additive contributions
D.bond_table(r, symbols)                # contrib column sums to n_TS
D.decompose_many(R3, T3, P3)            # vectorised: 2000 molecules in ~2 ms
```

The global Hammond progress is a **weighted mean of the per-element ratio
matrix**, with weights $(\Delta B^{RP}_{ij})^2$, so it decomposes additively and
splits into **forming vs breaking channels**. Also included:

| Quantity | What it adds beyond n_TS |
|---|---|
| `off_path` → f_⊥ | delocalisation orthogonal to the R→P direction |
| `transient_bonds` | bonds present only at the TS (weight exactly 0 in n_TS) |
| `participation` → N_eff | effective number of bonds carrying the change |
| `path_curvature` → κ | non-monotonic paths (needs NEB images) |
| `antibond_report` | negative matrix elements: sign crossings and transient antibonding |

See `docs/05_reaction_ts_analysis.md` and `docs/08_nts_decomposition.md`.

### 4. Spearman rank-correlation violin plots

`rho_from_dicts.py`, `rho_plot.py`, `rho_pipeline.py`, `bo_wasserstein.py`

```python
import rho_from_dicts as R

R.set_defaults(groups={"QM": ["minpop", "iao", "nbo"],
                       "emp": ["pauling", "reaxff"]},
               factors={"minpop": 2.0, "iao": 4.0, "nbo": 4.0},
               mask_from="minpop")

ids, methods, RHO = R.build_rho(dicts)          # (m, k, k) float32
fig, ax = R.plot_rho(methods, RHO)              # one violin per method pair
fig, ax = R.plot_rho(methods, RHO, pairs=[("iao", "nbo")])
d = R.distribution_distances(methods, RHO)      # distances BETWEEN distributions
fig, ax = R.plot_lollipop(d, top=15)
```

Replaces 600 per-molecule heatmaps with **one distribution per method pair**,
sorted by median, coloured by a user-supplied grouping. Nothing is hardcoded:
factors, grouping and colours come from the caller or `examples/methods.json`.
See `docs/04_bond_order_comparison.md`.

---

## Other scripts

`mbs_umap.py`, `mbs_condensed_umap.py`, `qm9_soap_umap.py`,
`chemiscope_mbs.py`, `pcovc_analysis.py` — embedding and visualisation
(SOAP/UMAP, variable-N handling, interactive chemiscope maps, PCovC).
See `docs/06_representation_and_embedding.md`.

---

## Documentation

| File | Contents |
|---|---|
| `docs/01_gaussian_output_anatomy.md` | Section-by-section read of a two-job BEBOP output and the physics of each derived property |
| `docs/02_parsing_guide.md` | API, four printed-matrix formats, the six traps |
| `docs/03_hdf5_storage_design.md` | Flat sparse layout, size arithmetic, dataset-name aliases, subset verification |
| `docs/04_bond_order_comparison.md` | Eigenvalue spectra, Wasserstein, Spearman, plotting, representation choice |
| `docs/05_reaction_ts_analysis.md` | Reacting-bond identification, TS scalars, Excel formulas |
| `docs/06_representation_and_embedding.md` | SOAP maths and the orbital connection, chemiscope, PCovC |
| `docs/07_validated_numbers.md` | Every verified value, for regression testing |
| `docs/08_nts_decomposition.md` | Per-bond decomposition of n_TS; forming vs breaking |

---

## Tests

```bash
pytest                                  # 39 tests
QM9_TEST_LOG=/path/to/mol.out.bz2 pytest   # enables the 13 data-backed ones
```

`tests/test_gaussian_parse.py` pins the documented numbers against a real
Gaussian output; it skips unless the file is present (see `tests/data/README.md`),
since datasets are not committed. The other two modules are data-free and always
run.

---

## Known limitations

- The Pauling covalent radii in `pauling_bo.py` are **not yet verified** against
  the Pyykkö papers. The HDF5 builder records `radii_verified = False` in the
  file attributes so the data and the SI table cannot silently diverge.
- Antibonding analysis applies only to methods whose off-diagonals can go
  negative (MinPop, Mayer, Wiberg). Pauling and ReaxFF are non-negative by
  construction, so every antibonding quantity is identically zero for them —
  a structural incapacity, not a null result.
- `n_anti` has a fragile denominator; use `a_hat` unless `rel_change` is healthy.
- Raw Wasserstein distance between eigenvalue spectra is **scale-dominated**;
  always read the RMS-normalised panel alongside it.

## Licence

MIT — see `LICENSE`.
