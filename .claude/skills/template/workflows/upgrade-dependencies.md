# Upgrade template dependencies (recompile depsets for a new Ray version)

Regenerate the templates' locked Python deps (`templates/<name>/python_depset.lock`) against a new
Ray version by adding a `build_arg_set` for that version and recompiling with `raydepsets`.

**Pairs with the image bump.** A template's image Ray version and the Ray version its
`python_depset.lock` was compiled against must match — run this alongside
`bump-ray-version.md`, not instead of it. System details: `../references/dependencies.md`.

**Audience — a human upgrading the whole repo at once.** This is the batch path: add the new
bundle, repoint *every* active entry, recompile the matrix, drop stale base locks. The automated
per-template bump does **not** follow this — it keeps the old bundle and repoints only its own
entry, because a single-template branch that replaces bundles or repoints all entries breaks
`check-depsets` for every other template. That incremental recipe lives in `bump-ray-version.md`
→ "Recompile the dependency lock." Use *this* file only when deliberately recompiling many
templates together.

Inputs: target Ray version `<NEW>` (e.g. `2.56.0`). Derive tokens: bundle prefix `ray2560`,
`PYTHON_SHORT` `311`/`312`, `RAY_VERSION` `2.56.0`.

## 1. Pre-check upstream availability
The base-compile pre-hooks fetch from `ray-project/ray/ray-<NEW>/python/deplocks/...`. Confirm those
exist before starting — Ray's deplocks lag the release by a few days:

```bash
curl -fsI "https://raw.githubusercontent.com/ray-project/ray/ray-<NEW>/python/deplocks/ray_img/ray_img_py311.lock" >/dev/null \
  && echo "deplocks published" || echo "NOT published yet — wait"
```
Not published → stop and wait (the fetch fallbacks may not produce equivalent locks).

## 2. Add the new build_arg_sets
In `dependencies/template.depsets.yaml`, add bundles for `<NEW>` mirroring the existing py311/py312
entries with `RAY_VERSION: "<NEW>"`. **Default: replace** the old `ray<OLD>_*` bundles (we ship one
Ray version). Keep both only if a transitional dual-version ship is explicitly wanted.

## 3. Repoint every active entry
Change each active depset's `build_arg_sets:` list from `ray<OLD>_*` → `ray<NEW>_*`. This cascades:
base `compile` entries emit new version-stamped locks, and each `expand` entry's `source_depset`
(which interpolates `${RAY_VERSION}`) now pulls from the new base lock. A repo-wide
`ray<OLD>` → `ray<NEW>` swap in the build_arg_set references is usually the whole edit.

## 4. Recompile (batch if needed)
```bash
./update_deps.sh                       # everything
./update_deps.sh --name <depset-name>  # one entry while iterating
```
Runs on **linux-x86_64 only**; on macOS use Docker — see `../references/dependencies.md` "Running it".

**Batched rollout (recommended for a full bump).** `--check` and a full `./update_deps.sh` build the
entire matrix and are slow. Split into grouped PRs the way the initial rollout did (see `git log`
PRs #730–#738): comment out every entry except the batch you're recompiling on this branch, so
`check-depsets` only gates the locks you changed. Uncomment in later PRs.

## 5. Drop stale base locks
Per-template `python_depset.lock` files are overwritten in place, but base locks are version-stamped.
Delete `dependencies/depsets/ray_<OLD>_*` / `rayllm_<OLD>_*` once nothing references the old version.
(Keep N-1 only if a rollback path is wanted — decide explicitly.)

## 6. Validate (the CI gate)
```bash
./update_deps.sh --check    # must be clean — this is exactly the check-depsets job
```
Then sanity-check a representative lock installs and the template runs (`rayapp test <name>` /
`references/run-tests-locally-with-rayapp.md`). For per-template test dispatch and recovery, reuse
`../references/testing-template.md`.

## Common failures
See `../references/dependencies.md` "Gotchas".
