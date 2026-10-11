# Pushing qm9-bond-orders to GitHub

The repo in `qm9-bond-orders/` is already initialised, on branch `main`, with
one commit (43 tracked files). Only the remote step is left — this sandbox has
no network access.

## 1. Create the empty remote

Either on github.com (**New repository**, name `qm9-bond-orders`, **do not**
initialise with a README/licence/gitignore — they already exist), or with the
GitHub CLI:

```bash
cd qm9-bond-orders
gh repo create qm9-bond-orders --public --source=. --remote=origin --push
```

That single `gh` command does steps 2 and 3 as well.

## 2. Set your identity on the commit

The commit was made with a placeholder author. Fix it before pushing:

```bash
cd qm9-bond-orders
git config user.name  "Chinmay Mhatre"
git config user.email "your-github-email@example.com"
git commit --amend --reset-author --no-edit
```

## 3. Add the remote and push

```bash
git remote add origin https://github.com/<your-username>/qm9-bond-orders.git
git push -u origin main
```

## 4. Verify

```bash
git log --oneline
git ls-files | wc -l        # 43
pytest                       # 39 tests; 13 skip without the reference log
```

---

## Before you make it public

- **Author email.** Step 2 above; the placeholder is `chinmay@example.com`.
- **Licence.** `LICENSE` is MIT with your name. If your group or funders
  (NRL / DTRA / Lubrizol) require a different licence or an acknowledgement,
  change it now rather than after the first clone.
- **Pauling radii.** `pauling_bo.py` still carries the `!! VERIFY BEFORE
  MANUSCRIPT USE !!` note. Keep it until the values are checked against the
  Pyykkö papers.
- **No data is committed.** `.gitignore` excludes `*.h5`, `*.bz2`, `*.npz`,
  `*.out`, `*.log`, `data/`. Confirm with `git ls-files | grep -E '\.(h5|bz2)$'`
  (should be empty). If you want the tests to run in CI you will need to host a
  single reference output somewhere, or commit it deliberately with
  `git add -f`.
- **Unpublished results.** `docs/` describes methodology and validation numbers,
  not dataset results, but read `docs/07_validated_numbers.md` once to confirm
  nothing there pre-empts a paper.

## Optional next steps

- **CI.** A minimal `.github/workflows/test.yml` running `pip install -e .[dev]`
  then `pytest` would catch regressions on every push. The data-free tests (26
  of 39) run without any dataset.
- **Zenodo DOI.** Linking the repo to Zenodo and tagging `v0.1.0` gives a
  citable DOI for the methods paper.
- **`CITATION.cff`.** Worth adding once the paper has a reference.
