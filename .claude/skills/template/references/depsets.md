# Dependency sets (depsets)

A template's Python deps are frozen into a hash-pinned lock — `templates/<name>/python_depset.lock` — so installs are reproducible at publish and probe time. The lock holds the template's own dependency closure — hash-pinned, and resolved against the base Ray lock (the entry's `source_depset`) so shared packages stay compatible with the image. It does **not** restate Ray's own packages; those already ship in the image. Built by `raydepsets` (a thin wrapper over `uv pip compile`); config lives in `dependencies/`.

## When a template needs one

Only if it adds Python deps **beyond the base Ray image**. A stock-Ray template (`getting-started`, `*-intro`) needs no `requirements.txt`, no lock, no entry — skip this whole file. If it `pip`-installs anything, author the full footprint below.

## Footprint (mirror an existing template, e.g. `text-embeddings`)

| File | What | Authored or generated |
|---|---|---|
| `templates/<name>/requirements.txt` | the deps this template adds, pinned (`==` where you can) | **authored** |
| `dependencies/template.depsets.yaml` entry | registers the lock build (block below) | **authored** |
| `templates/<name>/python_depset.lock` | the resolved hash-pinned lock | **generated** — never hand-edit |
| `tests/<name>/tests.sh` | installs papermill with `uv pip install` (not bare `pip`) | authored |
| template code (notebook / `main.py`) | installs **from the lock** + points Ray `runtime_env` at it (see "Use the lock") | authored |

## The depset entry

Append an `expand` entry to the `depsets:` list in `dependencies/template.depsets.yaml`:

```yaml
- name: <name>_depset_${RAY_VERSION}_${PYTHON_VERSION}
  operation: expand
  source_depset: ray_depset_${RAY_VERSION}_${PYTHON_VERSION}   # ray_llm_depset_${RAY_VERSION}_${PYTHON_VERSION}_${CUDA_VARIANT} for LLM/vLLM
  requirements:
    - templates/<name>/requirements.txt
  output: templates/<name>/python_depset.lock
  append_flags:
    - --index https://download.pytorch.org/whl/cu128   # only if it needs a CUDA torch build; match cuXXX to the image
    - --python-version=${PYTHON_VERSION}
    - --python-platform=linux                          # x86_64-manylinux_2_31 for LLM/GPU-manylinux templates
    - --unsafe-package ray
  build_arg_sets:
    - ray2551_py311_cu128                              # or ray2551_py312_cu128 — pick the template's Python
```

| Field / flag | How to set it |
|---|---|
| `source_depset` | `ray_depset_*` for CPU/most; `ray_llm_depset_*_${CUDA_VARIANT}` for LLM/vLLM. Match a sibling in the same group. |
| `--index …/whl/cuXXX` | add only if the template needs a CUDA torch build; `cuXXX` must match the image's CUDA. Omit for CPU-only. |
| `--python-platform` | `linux` for most; `x86_64-manylinux_2_31` on LLM/GPU-manylinux templates. Copy a sibling. |
| `--unsafe-package ray` | always present — keeps Ray (from the image) out of the lock. |
| `build_arg_sets` | one entry, picking py3.11 vs py3.12 to match the template. |

`operation: compile` entries (the `ray_depset_*` / `ray_llm_depset_*` base locks) are repo-global — don't touch them when adding a template; only on a Ray bump (see "On a Ray-version release" below).

## Compile the lock

`raydepsets` ships **only a linux-x86_64 binary** (it bundles its own `uv`) — no macOS build. On real x86_64 Linux (incl. CI) it just works. On Apple Silicon, run it in a `linux/amd64` glibc container (e.g. `python:3.12-bookworm`) and first shadow `uname` so `-p` returns `x86_64` — QEMU reports `unknown`, which otherwise breaks the bundled-uv lookup (`Unsupported platform/processor: Linux/`).

- **One template (fast — your loop):** `./update_deps.sh --name <name>_depset_2.55.1_3.11`
  `--name` builds that depset and its base. Use the entry's `name:` with build-args substituted (RAY_VERSION→`2.55.1`, PYTHON_VERSION→`3.11`/`3.12`). Unsure of the exact name? Run plain `./update_deps.sh` to rebuild all.
- **CI verification:** `./update_deps.sh --check` recompiles **all** and fails on any drift. `--check` can't combine with `--name`.

Commit the generated `python_depset.lock` alongside the `requirements.txt` and the entry.

## Use the lock in the template

Compiling the lock isn't enough — the template must install **from** it, in up to two places:

- **Driver / notebook env (always).** The install cell uses the lock, not loose pins: `uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match` (`--no-deps`: the lock is already the complete resolved set; `--index-strategy unsafe-best-match`: matches how it was compiled).
- **Ray workers (when the workload runs tasks / actors / Serve replicas).** Point `runtime_env` at the lock so workers get the same pins, not just the driver: `ray.init(runtime_env={"pip": os.path.abspath("python_depset.lock")})`. A driver-only install doesn't reach workers — this worker-reachability is exactly why loose notebook `!pip` was unreliable, and why a depset can replace a BYOD image for getting deps onto workers.

## Enforcement — the trap

Lock freshness is checked **only** by the GitHub CI job `check-depsets` (`./update_deps.sh --check`). It is **not** in local `pre-commit`, and `ci/validate_build_yaml.py` is depset-blind. So if you edit `requirements.txt` and skip the recompile, `pre-commit run --all-files` and `rayapp build` both stay green — and the PR fails on `check-depsets`. **Recompile and commit the lock yourself; nothing local will catch a stale one.**

## On a Ray-version release

A Ray bump is **repo-global** for depsets, not per-template — `build_arg_sets` carries one shared `RAY_VERSION` that every entry inherits. In one PR:

1. bump `RAY_VERSION` in `build_arg_sets` (`dependencies/template.depsets.yaml`);
2. the base `compile` entries refetch Ray's new published locks → new `dependencies/depsets/ray_<ver>_*.lock`;
3. recompile **every** per-template lock: `./update_deps.sh` (no `--name`);
4. verify with `./update_deps.sh --check`, then commit all changed locks.

This is separate from the per-template *image* bump (`../workflows/bump-ray-version.md`), which deliberately leaves locks untouched because it can't bump the shared `build_arg_sets` alone.
