# Template dependencies (depsets)

How templates' locked Python deps work. System reference for the depset steps in
`../workflows/upgrade-dependencies.md` (whole-repo) and `../workflows/bump-ray-version.md` (per-template).

## What ships, and how it reaches workers

Most templates carry a fully-pinned, hashed **`templates/<name>/python_depset.lock`** next to their
`requirements.txt`. The template installs it in **both** places:

```python
!uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match  # driver
ray.init(runtime_env={"pip": os.path.join(DEMO_ROOT, "python_depset.lock"), ...})                            # workers
```

`--system` covers only the driver; **workers get the deps solely via `runtime_env`** — omit it and they
silently run whatever the image shipped. The lock must also match the Ray version baked into the image; a
bump that moves the image without recompiling the lock desyncs them (`../workflows/bump-ray-version.md`).

## The tool: `raydepsets`

Repo-root **`update_deps.sh`** fetches the pinned `raydepsets` binary (v0.0.1, cached at `/tmp/raydepsets`)
and runs `raydepsets build dependencies/template.depsets.yaml --workspace-dir <root>`, compiling via
`uv pip compile --generate-hashes`. Always go through the wrapper:

```bash
./update_deps.sh                       # build every depset
./update_deps.sh --name <depset-name>  # build one (interpolated name, e.g. ray_depset_2.56.0_3.11)
./update_deps.sh --check               # recompile to a temp dir, diff vs committed (local validation)
```

## Running it

`raydepsets` v0.0.1 ships both `linux-x86_64` and `darwin-arm64` builds (Python zipapps bundling a
per-platform `uv`), so `./update_deps.sh` runs natively on Linux **and** macOS — output is identical
either way (`uv` always resolves for `--python-platform=linux`). `--check` needs all entries (can't
combine with `--name`). Base locks come from Ray's published `deplocks/`, which lag a release by days —
the **image** lock (`deplocks/ray_img/`) usually lands before the **LLM** lock (`deplocks/llm/`), so an
image bump can proceed while the LLM side waits.

## The config: `dependencies/template.depsets.yaml`

Two top-level keys. **`build_arg_sets`** — named `${VAR}` bundles:

```yaml
build_arg_sets:
  ray2551_py311_cu128: {RAY_VERSION: "2.55.1", PYTHON_VERSION: "3.11", PYTHON_SHORT: "311", CUDA_VARIANT: "cu128"}
```

**`depsets`** — entries; each entry's `build_arg_sets:` lists the bundle(s) it builds over, and the tool
emits one concrete depset per bundle, substituting `${VAR}` into every field. Two kinds:

**Base `compile`** (2 entries) — re-emit Ray's published image lock as the shared base lock, **version-stamped**
(new Ray version → new file). The output is Ray's fetched `ray_img` lock recompiled for our target — `ray`
excluded, re-hashed — so it's near-identical in content but reproducible and committable:

```yaml
- name: ray_depset_${RAY_VERSION}_${PYTHON_VERSION}
  operation: compile
  requirements: [/tmp/ray-deps/ray_img_py${PYTHON_SHORT}.lock]   # fetched by the pre_hooks
  output: dependencies/depsets/ray_${RAY_VERSION}_img_py${PYTHON_SHORT}.lock
  append_flags: [--python-version=${PYTHON_VERSION}, --python-platform=linux, --unsafe-package ray]
  build_arg_sets: [ray2551_py311_cu128, ray2551_py312_cu128]
  pre_hooks:
    - dependencies/scripts/fetch-ray-depsets.sh ${RAY_VERSION} ${PYTHON_SHORT}
    - dependencies/scripts/fetch-ray-constraints.sh ${RAY_VERSION} ${PYTHON_VERSION}
```

`ray_llm_depset_*` is the same for the LLM image → `rayllm_<ver>_*.lock`.

**Per-template `expand`** (the rest) — layer a template's `requirements.txt` on a base depset. Output is
**overwritten in place** (not version-stamped):

```yaml
- name: <tmpl>_depset_${RAY_VERSION}_${PYTHON_VERSION}
  operation: expand
  source_depset: ray_depset_${RAY_VERSION}_${PYTHON_VERSION}   # or ray_llm_depset_...
  requirements: [templates/<tmpl>/requirements.txt]
  output: templates/<tmpl>/python_depset.lock
  build_arg_sets: [ray2551_py312_cu128]
```

`source_depset` interpolates `${RAY_VERSION}`, so repointing an entry's `build_arg_sets:` to a new-version
bundle pulls from the new base lock — the lever both upgrade paths use. The `fetch-ray-*.sh` pre-hooks curl
Ray's published locks/constraints into `/tmp/ray-deps/`, with fallbacks for releases predating `deplocks/`.

## Changing a template's dependencies

1. Edit `templates/<name>/requirements.txt`.
2. Regenerate its lock: `./update_deps.sh --name <its-entry>` (see "Running it").
3. Confirm the template installs the regenerated lock on the driver **and** forwards it via `runtime_env`
   (see "What ships") — otherwise workers keep running stale deps.
4. Pin the traps below.

## Gotchas

- **`update_deps.sh` re-resolves against *live* indexes.** A fresh run can shift an untouched lock when
  upstream ships newer deps (ambient drift) or fail on a transient PyPI/PyTorch 503. Refresh drifted locks
  in-passing; treat 503s as infra.
- **`runtime_env` pip hash mismatch** on bumped transitive deps: `uv` can emit one wrong-interpreter hash.
  Pin the offending package to the base-image version.
- **Un-pinned floats.** `--system` installs float `numpy` to 2.x → pin `numpy`/`pandas`/`pyarrow` to the base
  image. Un-pinned `datasets` resolves to ancient `2.14.4` and breaks modern `fsspec` → pin `datasets==3.6.0`
  + `fsspec==2023.12.1` in `requirements.txt`.
- **Upstream lag:** Ray's `deplocks/` and base images publish days after a release; a bump waits on them.
