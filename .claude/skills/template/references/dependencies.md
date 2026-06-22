# Template dependencies (depsets)

How locked Python dependencies work for templates. The upgrade procedure lives in
`../workflows/upgrade-dependencies.md`; this file is the system reference it leans on.

## What ships with a template

Most templates carry a fully-pinned, hashed **`templates/<name>/python_depset.lock`** alongside
their `requirements.txt`. The template installs it into the driver and into Ray workers:

```python
!uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
ray.init(runtime_env={"pip": os.path.join(DEMO_ROOT, "python_depset.lock"), ...})
```

So the lock must be self-consistent **and** consistent with the Ray version preinstalled in the
template's image. A Ray-version bump that changes the image without recompiling the lock leaves the
two out of sync — see `../workflows/bump-ray-version.md`.

## The tool: `raydepsets`

A standalone binary fetched by repo-root **`update_deps.sh`** from
`github.com/ray-project/raydepsets/releases` (pinned `v0.0.1`), cached at `/tmp/raydepsets`. Run it
through the wrapper, never directly:

```bash
./update_deps.sh                       # build every depset
./update_deps.sh --name <depset-name>  # build one depset (+ its dependencies)
./update_deps.sh --check               # recompile to a temp dir, diff vs committed locks, fail on drift
```

`update_deps.sh` expands to `raydepsets build dependencies/template.depsets.yaml --workspace-dir <repo-root> "$@"`.
It compiles via `uv pip compile --generate-hashes`.

> The tool's upstream source is `~/repos/ray/ci/raydepsets/`, but that HEAD has drifted from the
> pinned `v0.0.1` binary (HEAD's `expand` takes a `depsets:` list; v0.0.1 takes `source_depset:`).
> **`dependencies/template.depsets.yaml` is the source of truth for the schema that actually runs.**

## The config: `dependencies/template.depsets.yaml`

Two top-level keys.

**`build_arg_sets`** — named `${VAR}` bundles. Today:

```yaml
build_arg_sets:
  ray2551_py311_cu128: {RAY_VERSION: "2.55.1", PYTHON_VERSION: "3.11", PYTHON_SHORT: "311", CUDA_VARIANT: "cu128"}
  ray2551_py312_cu128: {RAY_VERSION: "2.55.1", PYTHON_VERSION: "3.12", PYTHON_SHORT: "312", CUDA_VARIANT: "cu128"}
```

**`depsets`** — a list of entries. Each entry's `build_arg_sets:` field lists which bundle(s) it
expands over; the tool emits one concrete depset per bundle, substituting `${VAR}` into `name`,
`output`, `requirements`, `source_depset`, `append_flags`, and `pre_hooks`.

### Two kinds of entries

**Base `compile`** (2 entries) — fetch Ray's published locks and compile the shared base locks.
Output paths are **version-stamped**, so a new Ray version writes new files:

```yaml
- name: ray_depset_${RAY_VERSION}_${PYTHON_VERSION}
  operation: compile
  requirements: [/tmp/ray-deps/ray_img_py${PYTHON_SHORT}.lock]
  output: dependencies/depsets/ray_${RAY_VERSION}_img_py${PYTHON_SHORT}.lock
  append_flags: [--python-version=${PYTHON_VERSION}, --python-platform=linux, --unsafe-package ray]
  build_arg_sets: [ray2551_py311_cu128, ray2551_py312_cu128]
  pre_hooks:
    - dependencies/scripts/fetch-ray-depsets.sh ${RAY_VERSION} ${PYTHON_SHORT}
    - dependencies/scripts/fetch-ray-constraints.sh ${RAY_VERSION} ${PYTHON_VERSION}
```

`ray_llm_depset_*` is the same shape for the LLM image → `dependencies/depsets/rayllm_<ver>_*.lock`.

**Per-template `expand`** (the rest) — layer a template's `requirements.txt` on top of a base
depset. Output is **NOT version-stamped** — overwritten in place:

```yaml
- name: <tmpl>_depset_${RAY_VERSION}_${PYTHON_VERSION}
  operation: expand
  source_depset: ray_depset_${RAY_VERSION}_${PYTHON_VERSION}   # or ray_llm_depset_...
  requirements: [templates/<tmpl>/requirements.txt]
  output: templates/<tmpl>/python_depset.lock
  append_flags: [--index https://download.pytorch.org/whl/<cuXXX>, --python-version=${PYTHON_VERSION}, --python-platform=linux, --unsafe-package ray]
  build_arg_sets: [ray2551_py312_cu128]
```

Because `source_depset` interpolates `${RAY_VERSION}`, repointing an entry's `build_arg_sets:` to a
new-version bundle automatically pulls from the new base lock — that's the lever the upgrade uses.

### Pre-hooks
`dependencies/scripts/fetch-ray-{depsets,llm-depsets,constraints}.sh` curl Ray's published locks and
constraints for the target version from `raw.githubusercontent.com/ray-project/ray/ray-<version>/...`
into `/tmp/ray-deps/` before the base compile. They carry fallbacks for older Ray releases that
don't publish `deplocks/`.

## CI gate
`.github/workflows/premerge.yaml` → **`check-depsets`** runs `./update_deps.sh --check`. It
recompiles **all** active entries into a temp dir and unified-diffs against the committed locks;
any drift fails the job.

## Gotchas

- **`check-depsets` couples every active entry to its committed lock, per branch.** To stack or
  split depset PRs, comment out the entries you're not building on that branch (this is why the
  config has large commented-out blocks — staged rollout).
- **Per-template locks are overwritten in place; base locks are version-stamped.** On a version
  bump, delete the stale `dependencies/depsets/ray_<old>_*` files once nothing references them.
- **runtime_env pip hash mismatch** on version-bumped transitive deps: `uv` can emit one
  wrong-interpreter hash. Fix by pinning the offending package to the base-image version.
- **`uv pip install --system` bypasses Anyscale's worker propagation**, and un-pinned depsets float
  `numpy` to 2.x. Pin `numpy`/`pandas`/`pyarrow` to the base image; prefer the `runtime_env` pip
  install for workers.
- **Unpinned `datasets`** resolves to ancient `2.14.4` and breaks on modern `fsspec`. Pin
  `datasets==3.6.0` + `fsspec==2023.12.1` in the template `requirements.txt`.
- **Upstream lag:** Ray's `deplocks/` (and base images) publish a few days after a Ray release; a
  bump can't complete until they're up.
