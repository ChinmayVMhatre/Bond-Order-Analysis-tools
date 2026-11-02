# BOnts

Parses MinPop (MBS condensed-to-atoms) matrices from Gaussian output files and
decompose the reaction progress as a scalar term `n_TS` into per-bond bonding and
antibonding contributions.

## Install

```bash
git clone https://github.com/<user>/Bond-Order-Analysis-tools.git
cd BOnts
pip install -e .
```

Requires `numpy` and `pandas`; `pytest` for the tests.

## Layout

```
bonts/
  gaussian.py     reading Gaussian output, MBS matrix extraction
  decompose.py    n_TS decomposition, bonding and antibonding channels
examples/
  run_example.py  worked example
  data/           three Gaussian outputs for the model reaction
tests/
  test_gnts.py
```

## Functions

Abbreviations:
R = Reactant
P = Product
TS = Transition State

### `BOnts.gaussian`

| Function | Does |
|---|---|
| `read_output(path)` | Read a Gaussian output file. Plain text; `.bz2` also accepted. |
| `split_jobs(text)` | Split a Link1 multi-job output into per-job texts. |
| `parse_mbs_condensed(text, occurrence=-1)` | Extract the `MBS Condensed to atoms` matrix as a full symmetric `(N, N)` array. Returns `None` if the section is absent. |
| `mbs_bond_orders(mbs, factor=2.0)` | Convert an MBS matrix to bond orders: off-diagonals scaled by `factor`, diagonal set to zero. |
| `parse_geometry(text)` | Return `(atomic_numbers, symbols, coordinates)` from the last orientation block. |
| `load_bond_orders(path, factor=2.0, job=-1, occurrence=-1)` | One call: file path to `(bond_order_matrix, labels, atomic_numbers)`. |

### `BOnts.decompose`

| Function | Does |
|---|---|
| `decompose(BO_R, BO_TS, BO_P)` | Bonding channel. Returns `n_ts`, per-element `contrib` (sums to `n_ts`), `weight` (sums to 1), `ratio`, `dT`, `dP`, `per_atom`, and the `forming` / `breaking` split with their weight shares. |
| `bond_table(res, labels, min_weight=1e-3)` | Per-bond bonding table as a DataFrame; the `contrib` column sums to `n_ts`. |
| `antibonding(BO_R, BO_TS, BO_P, n_ts=None, zero_tol=2e-2, min_anti=2e-2, w_frac=1e-2)` | Antibonding channel. Returns `a_excess`, per-element `excess` (sums to `a_excess`), antibonding magnitudes `m_R`/`m_TS`/`m_P` and totals `a_R`/`a_TS`/`a_P`, the zero-`crossing` progress, a `crossed_early` mask, pattern `classes` and `patterns`, and `a_progress`. |
| `antibonding_table(res, BO_R, BO_TS, BO_P, labels, min_abs=1e-4)` | Per-bond antibonding table as a DataFrame; the `excess` column sums to `a_excess`. |
| `analyse_reaction(path_R, path_TS, path_P, factor=2.0, labels=None)` | Parse three output files and run both channels. Returns `labels`, `BO_R`, `BO_TS`, `BO_P`, `bonding`, `anti`. Raises if the atom counts disagree. |
| `summary(res)` | Print the scalar results of `analyse_reaction`. |

### Antibonding classes

`antibonding` sorts every atom pair into:

| Class                    | Condition                                                             |
| ------------------------ | --------------------------------------------------------------------- |
| `becomes_antibonding`    | bonding or zero in R, antibonding in P                                |
| `loses_antibonding`      | antibonding in R, not in P                                            |
| `persistent_antibonding` | antibonding in both endpoints                                         |
| `transient_antibonding`  | near-zero MinPOP in R and P, antibonding at the TS, negligible weight |
| `transient_any`          | neither endpoint antibonding, antibonding at the TS                   |

## Usage

```python
import BOnts

res = BOnts.analyse_reaction("sn2_R.log", "sn2_TS.log", "sn2_P.log")
BOnts.summary(res)

print(BOnts.bond_table(res["bonding"], res["labels"]))
print(BOnts.antibonding_table(res["anti"], res["BO_R"], res["BO_TS"],
                             res["BO_P"], res["labels"]))
```

Individual steps:

```python
text = BOnts.read_output("sn2_TS.log")
mbs  = BOnts.parse_mbs_condensed(text)      # (N, N) MBS matrix
bo   = BOnts.mbs_bond_orders(mbs)           # off-diagonals x2, diagonal 0

bo_R, labels, Z = BOnts.load_bond_orders("sn2_R.log")
```

## Example

```bash
python examples/run_example.py
```

Model reaction: `OH- + CH3F -> CH3OH + F-`, atom order `O1 H2 C3 H4 H5 H6 F7`.
The three files in `examples/data/` are `sn2_R.log`, `sn2_TS.log`, `sn2_P.log`.

## Tests

```bash
python -m pytest tests/ -q
```
