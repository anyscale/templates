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
silently run whatever the image shipped. `ray.init(runtime_env=)` is the standard: use it, and it covers
Ray Core, Data and Train workers, plus Serve replicas that declare no `runtime_env` of their own. The lock
must also match the Ray version baked into the image; a bump that moves the image without recompiling the
lock desyncs them (`../workflows/bump-ray-version.md`).

**One Serve rule: all or nothing.** `serve.run` hands the driver's `runtime_env` to any deployment that
declares none, so the `ray.init` above already reaches those replicas. But declare *any* `runtime_env` on a
deployment and it stops inheriting — only `working_dir` is backfilled and `pip` is dropped
(`ray/serve/_private/build_app.py:_set_default_runtime_env`). So either leave a deployment's `runtime_env`
alone, or give it the lock in full. Adding an `env_vars` to a working deployment silently strips its deps.

**Why `runtime_env`, not a bare `pip install`.** In a *workspace*, a raw `pip install` auto-propagates to
workers (a workspace-only convenience); `uv pip install` does **not** — it stays on the head. Since
`/test-template` and the scheduled probe both launch *workspaces* (`rayapp test` / `rayapp probe` — but see
`testing-template.md`, only the former tests your branch), a
template that populates workers via bare `pip install` passes **both** — then fails the moment a customer
runs it as a standalone Service/Job, which has no propagation. So: **worker deps travel by `runtime_env`
(from the `.lock`); `uv pip` is driver-only.** Never lean on a bare `pip install` to reach workers in a
ship-path template (pure workspace tutorials, where auto-propagation *is* the lesson, are the exception).
This includes the **test scripts** (`tests/<name>/tests.sh`): they run under the probe's non-schedulable
head, so their `serve run` / `ray.init` must carry the `.lock` via `runtime_env` — scoped to the apps that
need the added deps (leave an LLM ingress on the `ray-llm` image) — not a head-only `--system` install (#929).

**And a bare `pip install` in `tests.sh` breaks the template outright.** The propagation above works by the
workspace *tracking* the install; the runtime-env hook then appends it, **unpinned**, to the pip list every
actor receives. A lock full of hashes puts pip in hash-checking mode by itself — nobody passes
`--require-hashes`, it is the mode pip names in the error — and that one unhashed entry then fails **every**
runtime env for that template fails to build — not just the actor that wanted the package. Use
`uv pip install --system` in `tests.sh`, always; `check-dep-delivery.py bare-pip` enforces it for every
template that ships a lock.

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

- **Keep `requirements.txt` minimal — list only what the image doesn't ship.** The lock is seeded from the
  image freeze, so a base-stack package left out already holds at the image version; naming it buys nothing
  and quietly declares you want to *diverge*. A pin equal to the freeze is a no-op until the image moves and
  a downgrade after — that is how 2.57 put `numpy==1.26.4` on a numpy-2.2.6 image in 10 templates.
  **Delete such a line only when something else in the set pulls the package in.** The seed is a
  *preference*, not a requirement: it fixes a version but never adds a package, so deleting a leaf dep your
  own code imports drops it from the lock entirely. `ray` is `--unsafe-package`, which hides Ray's own
  dependencies from the resolver — so anything the image ships *for* Ray (`pyarrow` for Ray Data, `fastapi`
  for Ray Serve) is always leaf-like and its line is permanently load-bearing. When a line has to stay, pin
  it at the **image** version and say what needs it. Moving a package off the image version is on you to
  keep consistent cluster-wide.
- **Deliver *added* deps with `ray.init(runtime_env=)`.** It is the standard and it reaches everything,
  Serve replicas included, as long as those replicas declare no `runtime_env` of their own. Never rely on a
  head-only `--system` install. Reach for a per-deployment `ray_actor_options={"runtime_env": …}` only when
  one deployment genuinely needs something different — and then give it the lock in full (see "all or
  nothing" above). Note `py_modules` cannot take a local directory at deployment scope; Ray uploads
  directories only for `ray.init`.
- **Secondary configs point at the same `.lock`, by absolute path.** A shipped `service_config.yaml` /
  `job_config.yaml` (the standalone Service/Job path) must source its deps from the template's
  `python_depset.lock` via `runtime_env`, not a hand-maintained pin list — a divergent list drifts silently
  from the tested lock. Write `/home/ray/default/python_depset.lock`: a relative `requirements:` resolves
  against **the CLI's working directory**, not the config's `working_dir`, so a config submitted from a
  subdirectory dies with `FileNotFoundError`. `py_modules` entries follow the same rule and must name the
  *importable package* dir (`doggos/doggos`), not the project root — the outer dir yields an empty
  namespace package and an `import_path` under it cannot resolve.
- **A genuine conflict → isolate it.** An added package that hard-pins a clashing version (e.g. `a2a-sdk`
  forcing an old `fastapi`) goes in *its own* deployment's `runtime_env` — the LLM ingress keeps the image
  framework; only that deployment gets the pin.
- **Image = base + system deps only.** The base is `anyscale/ray` or `ray-llm` (`ray-ml` is deprecated —
  don't use it). Bake apt/CUDA/`.so` into a custom image *on that base* only when a system dep needs it
  (custom-GCP case); never add a pip layer there to dodge a lock conflict — that hides the same skew on
  another axis.

## The tool: `raydepsets`

Repo-root **`./scripts/depsets/update_deps.sh`** fetches the pinned `raydepsets` binary (v0.0.1, cached at `/tmp/raydepsets`)
and runs `raydepsets build dependencies/template.depsets.yaml --workspace-dir <root>`, compiling via
`uv pip compile --generate-hashes`. Always go through the wrapper:

```bash
./scripts/depsets/update_deps.sh                       # build every depset
./scripts/depsets/update_deps.sh --name <depset-name>  # build one (interpolated name, e.g. mcp_ray_serve_depset_2.56.0_3.12)
./scripts/depsets/update_deps.sh --check               # recompile to a temp dir, diff vs committed (local validation)
```

## Running it

`raydepsets` v0.0.1 ships both `linux-x86_64` and `darwin-arm64` builds (Python zipapps bundling a
per-platform `uv`), so `./scripts/depsets/update_deps.sh` runs natively on Linux **and** macOS — output is identical
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
| `refresh-image-freezes.py <ray-version> [image ...]` | re-fetches every image in `dependencies/images/tracked-images.txt`, or just the ones named |

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

**Never track an experimental image.** Anyscale ships pre-release variants (py3.14 today) that the console
badges EXPERIMENTAL, and **neither `docs.anyscale.com/base-images` nor its `index.json` carries that flag** —
they look identical to stable there. Only the API distinguishes them, so check before adding a line:

```bash
curl -fsSL -H "Authorization: Bearer $(jq -r .cli_token ~/.anyscale/credentials.json)" \
  "$ANYSCALE_HOST/api/v2/application_templates/supported_base_images" \
  | jq -r --arg i "anyscale/ray:<version>-<tag>" \
      '.result.images[] | select(.docker_image_name==$i) | .is_experimental'
```

`true` → don't track it. Adding an image is the only way one enters the system (the refresh just expands
`{version}` over the existing list), so this check is the whole guard.

**A template on an image not listed in `tracked-images.txt` never gets a freeze refreshed for it.** Add the
image there when you introduce it; `seed-image-freeze.sh` exits non-zero on a missing freeze, so the gap
surfaces at the next recompile rather than silently.

## The config: `dependencies/template.depsets.yaml`

Two top-level keys. **`build_arg_sets`** — named `${VAR}` bundles. They are the version lever: repointing one
entry from `ray2560_*` to `ray2570_*` moves both what uv resolves for and which freeze it seeds from.

```yaml
build_arg_sets:
  ray2560_py312:       {RAY_VERSION: "2.56.0", PYTHON_VERSION: "3.12", PYTHON_SHORT: "312"}
  ray2560_py312_cu130: {RAY_VERSION: "2.56.0", PYTHON_VERSION: "3.12", PYTHON_SHORT: "312", CUDA_VARIANT: "cu130"}
```

`CUDA_VARIANT` belongs in a bundle only where something interpolates it — today just the ray-llm freeze names.
Elsewhere the freeze filename and the torch `--index` carry their CUDA literally, so a `CUDA_VARIANT` on those
bundles would describe nothing and drift from the truth.

**`depsets`** — entries; each entry's `build_arg_sets:` lists the bundle(s) it builds over, and the tool
emits one concrete depset per bundle, substituting `${VAR}` into every field. Every entry is a per-template
`compile`: resolve a template's `requirements.txt` against a seed of the image it runs on ("Image freezes"
above). Output is **overwritten in place** (not version-stamped):

```yaml
- name: <tmpl>_depset_${RAY_VERSION}_${PYTHON_VERSION}
  operation: compile
  requirements: [templates/<tmpl>/requirements.txt]
  output: templates/<tmpl>/python_depset.lock
  build_arg_sets: [ray2560_py312]
  pre_hooks:
    - scripts/depsets/seed-image-freeze.sh dependencies/images/ray-${RAY_VERSION}-py312-cu128.freeze.txt templates/<tmpl>/python_depset.lock
```

The freeze **must name the image that template's `BUILD.yaml` entry actually runs on**, or the lock
silently describes an environment the template never runs in. `check-dep-delivery.py depset-config` enforces
the pairing (see "What CI enforces"). Keep `${RAY_VERSION}` (and
`${PYTHON_SHORT}` / `${CUDA_VARIANT}` where they match the tag) interpolated so a version bump follows the
bundle instead of pinning the old image's freeze.

## Changing a template's dependencies

1. Edit `templates/<name>/requirements.txt`.
2. Regenerate its lock: `./scripts/depsets/update_deps.sh --name <its-entry>` (see "Running it").
3. Confirm the template installs the regenerated lock on the driver **and** forwards it via `runtime_env`
   at the right scope (Serve → app/deployment level; see "What ships" + "Runtime skew") — otherwise
   workers/replicas keep running stale deps.
4. Scan the lock diff for a framework package moving below its image version — the "Runtime skew" trap above.

## How to pin

**Every requirement is `==`, at the newest version that works on that template's image.**

`>=` and bare names are not pins. They tell the resolver "anything newer is fine", so the lock
silently re-resolves to whatever is current every time it is rebuilt — the template's behaviour
changes with nobody editing it, and CI passes because it tests the drift. `<` is worse: an upper
bound with no lower one freezes a template in the past. `text-embeddings` carried `torch<2.5`,
`langchain==0.1.17` and `transformers==4.40.2` for a year that way.

A loose spec needs a **trailing comment saying why** — `check-dep-delivery.py pin-style` fails
without one. Real reasons: upstream publishes no tagged releases; a documented incompatibility;
a range the template deliberately spans. "It worked when I wrote it" is not one.

Two consequences worth spelling out:

- **`==` alone doesn't mean current.** A pin set 18 months ago is exactly as stale as no pin, just
  deterministically so. Freshness is re-checked on every Ray bump — see
  `workflows/launch-ray-bump-wave.md`.
- **For `torch`, "newest that works" is bounded by the image's CUDA index, not PyPI.**
  `download.pytorch.org/whl/cu128` publishes torch from 2.7.0 and `cu129` from 2.8.0. Pin below the
  floor and uv falls back to PyPI, the wheel loses its CUDA tag (`torch==2.7.0+cu128` becomes
  `torch==2.7.0`), and the lock still compiles. Check the index before pinning torch.

Every template pins `==` today. A spec that genuinely can't be pinned needs the trailing comment —
there is no allowlist to add it to.

## Reviewing a template's dependency delivery

Eight questions. Each has produced a real fleet-wide break, and the ones marked ✅ are now caught
automatically — the rest still need eyes.

| # | Question | Failure it catches |
|---|---|---|
| 1 | ✅ Does a user-facing file install the lock, with `-r`? | template ships a lock nothing installs; users run on whatever the image has |
| 2 | Does every off-head unit get those deps *at its own scope*? | `map_batches` actors, `TorchTrainer` workers, `@ray.remote` tasks, Serve replicas that declare a partial `runtime_env` |
| 3 | ✅ Does anything depend on a bare `pip install` reaching workers? | green in a workspace, broken as a Job or Service |
| 4 | Does any hand-typed pin list disagree with the lock? | a notebook pin quietly compensating for a wrong lock is the dangerous shape |
| 5 | Do the shipped Job/Service configs source the lock, by absolute path? | the standalone path nobody tests |
| 6 | Does `tests.sh` install or configure something the template doesn't? | CI green while every user following the README fails |
| 7 | ✅ Does the depset's seed freeze match `BUILD.yaml`'s image? | lock describes an environment the template never runs in |
| 8 | ✅ Is every requirement `==`, at a version that is still current? | the lock re-resolves on every rebuild, or freezes the template years behind |

Dimension 6 is the highest-yield one to check by hand, because CI is what hides it. Read `tests.sh` and the
README side by side and ask what the test does that a user wouldn't.

### What CI enforces

`scripts/hooks/check-dep-delivery.py`, via pre-commit (whole repo, ~0.2s). Run a single check by name:

```bash
python3 scripts/hooks/check-dep-delivery.py                # all four
python3 scripts/hooks/check-dep-delivery.py bare-pip       # one
```

| check | fails when |
|---|---|
| `depset-config` | an entry's seed freeze names an image other than the one `BUILD.yaml` runs, or it omits `include_setuptools: true` |
| `lock-installed` | a template ships a lock nothing installs, or installs it without `-r` |
| `bare-pip` | a lock-bearing template bare-`pip install`s in `tests.sh` |
| `pin-style` | a requirement is neither `==` nor commented |

`include_setuptools` matters because uv drops setuptools from a lock while Ray's runtime env is a
virtualenv seeding its own: a locked package wanting a newer one makes pip collect it unpinned and
unhashed against a hashed lock, and every runtime env for that template then fails to build.

`bare-pip` skips lock-less templates — for a pure workspace tutorial the bare install *is* the lesson —
and they join the gate automatically if they ever gain a lock.

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
