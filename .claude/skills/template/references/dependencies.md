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
silently run whatever the image shipped. That `ray.init` env reaches Ray Core/Data workers (tasks/actors)
but **not Ray Serve replicas** — serve/LLM templates must declare deps at the Serve app/deployment level
(see "Runtime skew" below). The lock must also match the Ray version baked into the image; a bump that
moves the image without recompiling the lock desyncs them (`../workflows/bump-ray-version.md`).

**Why `runtime_env`, not a bare `pip install`.** In a *workspace*, a raw `pip install` auto-propagates to
workers (a workspace-only convenience); `uv pip install` does **not** — it stays on the head. Since
`/test-template` and the scheduled probe both launch *workspaces* (`rayapp test` / `rayapp probe`), a
template that populates workers via bare `pip install` passes **both** — then fails the moment a customer
runs it as a standalone Service/Job, which has no propagation. So: **worker deps travel by `runtime_env`
(from the `.lock`); `uv pip` is driver-only.** Never lean on a bare `pip install` to reach workers in a
ship-path template (pure workspace tutorials, where auto-propagation *is* the lesson, are the exception).
This includes the **test scripts** (`tests/<name>/tests.sh`): they run under the probe's non-schedulable
head, so their `serve run` / `ray.init` must carry the `.lock` via `runtime_env` — scoped to the apps that
need the added deps (leave an LLM ingress on the `ray-llm` image) — not a head-only `--system` install (#929).

## Runtime skew — right scope, don't move the base framework

The template lock **layers on** the base image's env; it must not *re-resolve down* a package the image's
managed runtime imports (`ray`, `fastapi`, `starlette`, `pydantic`, `vllm`, `torch`). A dep that drags one
**below** the image version passes CI and the publish "Test template" step (there everything co-locates on
the head), then dies on a real multi-node launch, where a Serve replica or actor lands on a worker still on
the image version and cross-node deserialization fails (classic:
`ModuleNotFoundError: fastapi._compat.may_v1`). Only `ray` is `--unsafe-package`-protected today.

Seeding from the image freeze makes this **rare but not impossible**: an unpinned dep now holds at the image
version by default, so skew takes an explicit pin or a hard transitive requirement. Nothing blocks one.

**Author guidance, not a gate — you own your `requirements.txt`.** But heed it:

- **The base framework stack is ground truth.** Leave those packages unpinned and the freeze holds them at
  the image version — that is the default and usually the right answer. Moving one is on you to keep
  consistent cluster-wide.
- **Deliver *added* deps at the right scope.** `ray.init(runtime_env=)` reaches Ray Core/Data workers but
  **not Serve replicas**; declare a serve/LLM template's added deps on the Serve **app-level** `runtime_env`
  (or `@serve.deployment(ray_actor_options={"runtime_env": …})`), never a head-only `--system` install.
- **Secondary configs point at the same `.lock`.** A shipped `service_config.yaml` / `job_config.yaml`
  (the standalone Service/Job path) must source its deps from the template's `python_depset.lock` via
  `runtime_env`, not a hand-maintained pin list — a divergent list drifts silently from the tested lock.
- **A genuine conflict → isolate it.** An added package that hard-pins a clashing version (e.g. `a2a-sdk`
  forcing an old `fastapi`) goes in *its own* deployment's `runtime_env` — the LLM ingress keeps the image
  framework; only that deployment gets the pin.
- **Image = base + system deps only.** The base is `anyscale/ray` or `ray-llm` (`ray-ml` is deprecated —
  don't use it). Bake apt/CUDA/`.so` into a custom image *on that base* only when a system dep needs it
  (custom-GCP case); never add a pip layer there to dodge a lock conflict — that hides the same skew on
  another axis.

## The tool: `raydepsets`

Repo-root **`update_deps.sh`** fetches the pinned `raydepsets` binary (v0.0.1, cached at `/tmp/raydepsets`)
and runs `raydepsets build dependencies/template.depsets.yaml --workspace-dir <root>`, compiling via
`uv pip compile --generate-hashes`. Always go through the wrapper:

```bash
./update_deps.sh                       # build every depset
./update_deps.sh --name <depset-name>  # build one (interpolated name, e.g. mcp_ray_serve_depset_2.56.0_3.12)
./update_deps.sh --check               # recompile to a temp dir, diff vs committed (local validation)
```

## Running it

`raydepsets` v0.0.1 ships both `linux-x86_64` and `darwin-arm64` builds (Python zipapps bundling a
per-platform `uv`), so `./update_deps.sh` runs natively on Linux **and** macOS — output is identical
either way (`uv` always resolves for `--python-platform=linux`). `--check` needs all entries (can't
combine with `--name`). A recompile needs nothing from Ray's repo — only the committed image freeze.

**`--check` leaves the working tree dirty.** The seed pre_hooks write into each lock's own path before uv
runs, so afterwards those files hold the *seed*, not the compiled lock. The check itself is still valid
(it compiles elsewhere and reports "Lock files are up to date") — but `git checkout -- templates/`
afterwards, or you will commit seeds over your locks.

## Image freezes — what a template's lock is built against

A template's lock is seeded from a **`pip freeze` of the published image it runs on**: a lock models a build
*recipe*, a freeze measures the artifact that actually shipped. The
`ray-llm` image is built `--no-deps`, so its installed set is internally inconsistent and **no lock can
reproduce it** — only a freeze describes it. `dependencies/images/<image>.freeze.txt` holds one per tracked
image.

| Script | Does |
|---|---|
| `seed-image-freeze.sh <freeze> <lock>` | writes the seed to the lock's path; each template's `pre_hook` |
| `fetch-image-freeze.sh <image> <dest>` | pulls one image's package list from `docs.anyscale.com/base-images` |
| `refresh-image-freezes.sh <ray-version>` | re-fetches every image in `dependencies/images/tracked-images.txt` |

The seed is the freeze **plus** the previous lock's pins for packages the image doesn't ship — otherwise every
recompile re-floats untouched packages to newest-on-PyPI. uv reads it as **preferences, not constraints**: a
seeded version holds unless a requirement forces otherwise, so a template that genuinely needs a newer package
just pins it in `requirements.txt` and wins. That is the intended way to diverge from the image.

Two limits, both confirmed by testing:

- **A rejected image version floats.** For a package the image *does* ship whose version a requirement rejects
  (`uvicorn` at `0.22.0`, forced past it by `mcp`), the freeze wins the seed so no previous-lock pin is carried
  — then the freeze's version is rejected, leaving no preference at all, and it resolves to newest. Pin it in
  `requirements.txt` when the version matters.
- **Packages outside the freeze never re-resolve.** Carry-over is what stops daily drift, but it also means a
  package the image doesn't ship keeps its locked version until someone edits `requirements.txt` — a regen
  alone will never pick up an upstream security fix. Bump those deliberately.

**A template on an image not listed in `tracked-images.txt` never gets a freeze refreshed for it.** Add the
image there when you introduce it; `seed-image-freeze.sh` exits non-zero on a missing freeze, so the gap
surfaces at the next recompile rather than silently.

## The config: `dependencies/template.depsets.yaml`

Two top-level keys. **`build_arg_sets`** — named `${VAR}` bundles:

```yaml
build_arg_sets:
  ray2551_py311_cu128: {RAY_VERSION: "2.55.1", PYTHON_VERSION: "3.11", PYTHON_SHORT: "311", CUDA_VARIANT: "cu128"}
```

**`depsets`** — entries; each entry's `build_arg_sets:` lists the bundle(s) it builds over, and the tool
emits one concrete depset per bundle, substituting `${VAR}` into every field. Every entry is a per-template
`compile`: resolve a template's `requirements.txt` against a seed of the image it runs on ("Image freezes"
above). Output is **overwritten in place** (not version-stamped):

```yaml
- name: <tmpl>_depset_${RAY_VERSION}_${PYTHON_VERSION}
  operation: compile
  requirements: [templates/<tmpl>/requirements.txt]
  output: templates/<tmpl>/python_depset.lock
  build_arg_sets: [ray2560_py312_cu128]
  pre_hooks:
    - dependencies/scripts/seed-image-freeze.sh dependencies/images/ray-${RAY_VERSION}-py312-cu128.freeze.txt templates/<tmpl>/python_depset.lock
```

The freeze **must name the image that template's `BUILD.yaml` entry actually runs on** — nothing checks the
pairing, so a mismatch silently locks against the wrong environment. Keep `${RAY_VERSION}` (and
`${PYTHON_SHORT}` / `${CUDA_VARIANT}` where they match the tag) interpolated so a version bump follows the
bundle instead of pinning the old image's freeze.

## Changing a template's dependencies

1. Edit `templates/<name>/requirements.txt`.
2. Regenerate its lock: `./update_deps.sh --name <its-entry>` (see "Running it").
3. Confirm the template installs the regenerated lock on the driver **and** forwards it via `runtime_env`
   at the right scope (Serve → app/deployment level; see "What ships" + "Runtime skew") — otherwise
   workers/replicas keep running stale deps.
4. Scan the lock diff for a framework package moving below its image version — the "Runtime skew" trap above.

## Gotchas

- **Don't pin what the image already ships.** The freeze holds `numpy`, `pandas`, `pyarrow`, `fsspec` et al at
  the image version, so `requirements.txt` should list only what the *template* adds. A redundant pin is worse
  than noise: it survives the next image bump and silently blocks the version the new image ships.
- **Do pin what the image doesn't ship.** Only the previous lock holds those, so a package new to the template
  resolves to newest-on-PyPI. The classic: un-pinned `datasets` resolves to ancient `2.14.4` → pin
  `datasets==3.6.0`. (`fsspec` is in the image now; pinning it is no longer needed.)
- **`runtime_env` pip hash mismatch** on bumped transitive deps: `uv` can emit one wrong-interpreter hash.
  Pin the offending package to the base-image version.
- **Torch CUDA must match the build image.** Only `ray-llm` ships `torch`; on plain `anyscale/ray` the template
  supplies it. Write `--index https://download.pytorch.org/whl/${CUDA_VARIANT}` (never a hardcoded `cu121`) so
  the index tracks the image, and **pin** `torch==<ver>` to a version with wheels for that CUDA — an unpinned
  `torch>=…` lets `unsafe-best-match` pull PyPI's newest wheel, a *newer* CUDA than the image, which fails at
  runtime.
- **A transient PyPI/PyTorch 503** fails a recompile; treat as infra and retry.
- **Upstream lag:** base images publish days after a Ray release; a bump waits on the image, and on the freeze
  fetched from it.
