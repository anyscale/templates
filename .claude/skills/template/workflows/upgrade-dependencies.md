# Upgrade template dependencies (recompile depsets for a new Ray version)

Regenerate the templates' locked Python deps (`templates/<name>/python_depset.lock`) against a new
Ray version by adding a `build_arg_set` for that version and recompiling with `raydepsets`.

**Base locks and image freezes are mostly automated.** The `ray-base-locks` GitHub Action
(`.github/workflows/ray-base-locks.yaml` + `scripts/ray-bump/prepare-base-locks.py`) recompiles the *base* locks
for a new Ray version and refreshes the image freezes, opening a PR as soon as Ray publishes that version's
`deplocks/` and copying the current `(py, cuda)` matrix forward. Run this manual procedure when that job reports
**needs human** — Ray changed the matrix (a py/cuda added, dropped, or moved), so the copy-forward can't apply —
or for a whole-repo dependency change. Either way that's only half the job: repointing each template's
`python_depset.lock` is the per-template fanout's work (`bump-ray-version.md`).

**Pairs with the image bump.** A template's image Ray version and the Ray version its
`python_depset.lock` was compiled against must match — run this alongside
`bump-ray-version.md`, not instead of it. System details: `../references/dependencies.md`.

Inputs: target Ray version `<NEW>` (e.g. `2.56.0`). Derive tokens: bundle prefix `ray2560`,
`PYTHON_SHORT` `311`/`312`, `RAY_VERSION` `2.56.0`.

## 1. Pre-check upstream availability
Two independent upstreams must have shipped, and both lag a Ray release by days — Ray's `deplocks/` (the
base-compile pre-hooks fetch them) and the published base **images** (the freezes are fetched from them):

```bash
curl -fsI "https://raw.githubusercontent.com/ray-project/ray/ray-<NEW>/python/deplocks/ray_img/ray_img_py311.lock" >/dev/null \
  && echo "deplocks published" || echo "deplocks NOT published yet — wait"
curl -fsSL https://docs.anyscale.com/base-images/index.json \
  | python3 -c 'import json,sys; print(sum(e["rayVersion"]=="<NEW>" for e in json.load(sys.stdin)), "images")'
```
Either missing → stop and wait. Without deplocks the fetch fallbacks may not produce equivalent locks;
without the images every template would silently compile against the previous version's freeze.

## 2. Add the new build_arg_sets
In `dependencies/template.depsets.yaml`, add bundles for `<NEW>` mirroring the existing py311/py312
entries with `RAY_VERSION: "<NEW>"`. **Default: replace** the old `ray<OLD>_*` bundles (we ship one
Ray version). Keep both only if a transitional dual-version ship is explicitly wanted.

## 3. Refresh the image freezes
Template locks are seeded from `dependencies/images/<image>.freeze.txt`, not from the base lock
(`../references/dependencies.md` "Image freezes"). Fetch `<NEW>`'s:

```bash
bash dependencies/scripts/refresh-image-freezes.sh <NEW>
```

It **warns and continues** on an image that isn't published yet, so read its output: a skipped image
means every template on it would compile against `<OLD>`'s freeze. Wait for the image and re-run.

## 4. Repoint every active entry
Change each active depset's `build_arg_sets:` list from `ray<OLD>_*` → `ray<NEW>_*`. This cascades:
the base `compile` entry emits a new version-stamped lock, and each template's `seed-image-freeze.sh`
pre_hook (which interpolates `${RAY_VERSION}`) picks up the new freeze. A repo-wide `ray<OLD>` →
`ray<NEW>` swap in the build_arg_set references is usually the whole edit — but where `<NEW>`'s image
moved py/CUDA, also fix the literal parts of that entry's freeze filename to match.

## 5. Recompile (batch if needed)
```bash
./update_deps.sh                       # everything
./update_deps.sh --name <depset-name>  # one entry while iterating
```
Runs natively on Linux or macOS — see `../references/dependencies.md` "Running it".

**Batched rollout (recommended for a full bump).** `--check` and a full `./update_deps.sh` build the
entire matrix and are slow. Split into grouped PRs the way the initial rollout did (see `git log`
PRs #730–#738): comment out every entry except the batch you're recompiling on this branch, so the
branch rebuilds only that batch. Uncomment in later PRs.

## 6. Drop stale base locks
Per-template `python_depset.lock` files and image freezes are overwritten in place, but base locks are
version-stamped. Delete `dependencies/depsets/ray_<OLD>_*` once nothing references the old version.
(Keep N-1 only if a rollback path is wanted — decide explicitly.)

## 7. Validate
```bash
./update_deps.sh --check    # must be clean (recompiles all entries + diffs vs committed)
```
Then sanity-check a representative template runs (`rayapp test <name>` /
`references/run-tests-locally-with-rayapp.md`). For per-template test dispatch and recovery, reuse
`../references/testing-template.md`.

## Common failures
See `../references/dependencies.md` "Gotchas".
