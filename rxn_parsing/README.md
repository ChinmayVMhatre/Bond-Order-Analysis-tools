# bonts

Parse MinPop (MBS) bond orders from Gaussian output files and decompose the
transition-state progress `n_TS` into per-bond contributions.

## Install

```bash
git clone https://github.com/<user>/bonts.git
cd bonts
pip install -e .
```

Requires Python 3.9+, numpy and pandas. Tests need pytest.

## Quick start

```bash
python examples/run_example.py
```

```bash
bonts-nts examples/data/sn2_reactant.log \
          examples/data/sn2_ts.log \
          examples/data/sn2_product.log --antibonding
```

```python
from bonts import load_triple, decompose, bond_table

B_R, B_TS, B_P, symbols = load_triple("R.log", "TS.log", "P.log")
res = decompose(B_R, B_TS, B_P)
print(res["n_ts"], res["forming"], res["breaking"])
print(bond_table(res, symbols))
```

## Layout

```
bonts/
  gaussian.py       parsing Gaussian output
  decompose.py      n_TS decomposition
  antibonding.py    negative matrix elements
  cli.py            command-line entry point
examples/
  run_example.py        worked example on the model reaction
  make_model_data.py    regenerates the example data files
  data/                 model reaction: F(-) + CH3Cl -> CH3F + Cl(-)
tests/
  test_bonts.py     test suite
```

## Functions

### `bonts.gaussian`

| Function | Does |
|---|---|
| `read_log(path)` | Read a Gaussian output file, handling `.bz2` |
| `split_jobs(text)` | Split a Link1 multi-job log into per-job texts |
| `parse_atom_matrix(text, header_re, occurrence=-1)` | Read a symmetric atom × atom matrix from Gaussian's blocked lower-triangular format |
| `mbs_matrix(path, occurrence=-1)` | MBS condensed-to-atoms matrix as printed; `None` if absent |
| `minpop_bond_orders(path, factor=2.0, occurrence=-1)` | MinPop bond-order matrix: off-diagonal × `factor`, diagonal zeroed |
| `parse_geometry(text)` | Atomic numbers and coordinates from the last orientation block |
| `symbols_from(path)` | Element symbols for the atoms in a file |
| `load_triple(path_R, path_TS, path_P, factor=2.0)` | Bond-order matrices for a reactant/TS/product triple, plus symbols |

### `bonts.decompose`

| Function | Does |
|---|---|
| `decompose(B_R, B_TS, B_P)` | Per-element decomposition of `n_TS`; returns `n_ts`, `contrib`, `weight`, `ratio`, `dT`, `dP`, `per_atom`, `forming`, `breaking`, `w_forming`, `w_breaking` |
| `decompose_from_ratio(ratio, dP)` | Same, starting from a precomputed ratio matrix and `dP` |
| `bond_table(res, symbols=None, min_weight=1e-3)` | Per-bond DataFrame; `contrib` column sums to `n_ts` |
| `decompose_many(B_R, B_TS, B_P)` | Vectorised decomposition over `(m, N, N)` stacks |

### `bonts.antibonding`

| Function | Does |
|---|---|
| `antibonding(B_R, B_TS, B_P, n_ts=None)` | Antibonding index and per-element excess; returns `a_excess`, `excess`, `m_R`/`m_TS`/`m_P`, `a_R`/`a_TS`/`a_P`, `crossing`, `crossed_early`, `classes`, `patterns`, `a_progress` |
| `antibonding_table(res, B_R, B_TS, B_P, symbols=None)` | Per-bond DataFrame of antibonding elements with their class |

### `bonts.cli`

| Entry point | Does |
|---|---|
| `bonts-nts R.log TS.log P.log` | Parse the three files, print `n_TS` with the forming/breaking split and the per-bond table |
| `--factor`, `--min-weight`, `--antibonding`, `--csv` | Off-diagonal multiplier, table cutoff, antibonding report, CSV output |

### `examples`

| Script | Does |
|---|---|
| `run_example.py` | Parses the model reaction, prints `n_TS`, the per-bond table and the antibonding table |
| `make_model_data.py` | Regenerates the three files in `examples/data/` |

## Example data

`examples/data/` holds three Gaussian-format files for
F⁻ + CH₃–Cl → CH₃–F + Cl⁻ (atom order F1 C2 H3 H4 H5 Cl6). They are
**synthetic**: hand-built to carry a valid geometry block and MBS
condensed-to-atoms block so the code can be exercised without shipping a large
calculation. They are not the output of an actual Gaussian run.

Expected output:

```
n_TS       0.4701
forming    0.2128  (weight 0.500)
breaking   0.2573  (weight 0.500)

 i  j   bond    dP    dT    ratio   weight  contrib     kind
 0  1  F1-C2  0.82  0.35 0.426829 0.497389  0.21230  forming
 1  5 C2-Cl6 -0.82 -0.42 0.512195 0.497389  0.25476 breaking
```

## Tests

```bash
pytest tests/
```

16 tests covering matrix parsing against known file values, the wrapped second
column block, missing sections, size mismatches, the `n_TS` identities
(`sum(contrib) == n_ts`, endpoints 0 and 1, linear midpoint 0.5), the ratio
round-trip, batch/single agreement, and antibonding additivity and detection.
