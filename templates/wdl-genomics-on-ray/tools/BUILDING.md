# Building the toolchain

Everything here is driven by [`manifest.toml`](manifest.toml), which pins each tool's version,
source URL and sha256 in one place. These read it; none of them duplicates it:

| Consumer | What it does |
|---|---|
| [`build_tools.sh`](build_tools.sh) | installs the toolchain into a prefix, which is what the `Dockerfile` runs |
| [`build_wheels.sh`](build_wheels.sh) | packages the same builds as pip wheels |
| `wdl_on_ray/envs.py` | derives a per-task `runtime_env` under `--container-runtime native` |

All three verify the same sha256, so every route produces byte-identical payloads.

## Build your own image

The template ships a `Dockerfile`. To build it under your own name:

```bash
cd templates/wdl-genomics-on-ray
anyscale image build -n my-wdl-tools --containerfile Dockerfile
```

The command prints an image URI you can pass to `anyscale job submit --image-uri ...`, set as
`image_uri:` in `job.yaml`, or select when launching a workspace.

To publish it as *this template's* image instead, the path the gallery uses, the name and tag are
not free-form. The BUILD.yaml validator requires the image basename to equal the template name and
the tag to equal `byod.ray_version`:

```bash
../../.claude/skills/template/scripts/push-custom-image-to-gcp.sh . wdl-genomics-on-ray 2.56.0
# -> us-docker.pkg.dev/anyscale-workspace-templates/workspace-templates/wdl-genomics-on-ray:2.56.0
```

### Adding a tool

1. Add a `[tools.<name>]` block to `manifest.toml` with `version`, `kind`, `url`, `sha256`, and
   a `[tools.<name>.provides]` map of PATH-name to path inside the payload.
2. `kind = "binary"` and `kind = "pypi"` need nothing else. `kind = "source"` needs a build
   recipe in `build_one()` in `build_tools.sh`, where there are three to copy from.
3. Add it to *both* verification lists at the bottom of `build_tools.sh` — the `for exe in ...`
   loop that checks PATH and the `--version` calls below it — and to the `RUN set -eux` block in
   the `Dockerfile`. All three exist because every silent failure this toolchain has had was a
   tool that installed successfully and then didn't work.
4. Rebuild the image.

Get the sha256 with `curl -fsSL <url> | sha256sum`.

## The stock-image route, and why it needs a staging step

You can run this template on a stock `anyscale/ray` image instead, with the toolchain delivered
as wheels. It has more moving parts, so read this before choosing it.

```bash
# must run on linux x86_64; easiest inside the template's own image
docker run --rm -v "$PWD:/w" -w /w <image> bash tools/build_wheels.sh /w/wheelhouse
```

That produces `wdl-on-ray-tools-{minimap2,samtools,bcftools,flye,quast}` wheels. On an Anyscale image
their shims land in `/home/ray/anaconda3/bin`, which is already first on every Ray worker's
PATH, so `pip install` alone puts a tool on PATH, with no image build and no `ENV` surgery.

**The wheels are not enough on their own.** The obvious ways to install them both fail:

- A published `--find-links` index over `https://`. pip fetches it *anonymously*, so a
  private bucket is unreachable no matter what IAM role the node carries. pip logs
  `Looking in links: https://…` and then `No matching distribution found`. Only a public index
  works.
- A relative `--find-links ./wheelhouse`. A job's `requirements:` is installed at cluster
  startup, *before* the working directory is staged, so the wheels that travel with the
  submission are absent at the moment they are needed.

What works is staging them to a durable mount first, then pointing `--find-links` at a real
path:

```yaml
# one-off staging job
requirements: []
entrypoint: |
  mkdir -p /mnt/user_storage/wdl-on-ray/wheels
  cp wheelhouse/*.whl /mnt/user_storage/wdl-on-ray/wheels/
```

```yaml
# then the real job
requirements:
  - miniwdl==1.15.0
  - --find-links /mnt/user_storage/wdl-on-ray/wheels
  - wdl-on-ray-tools-flye==2.9.5      # per-tool, not a meta extra
  - wdl-on-ray-tools-minimap2==2.28
  - wdl-on-ray-tools-samtools==1.21
  - wdl-on-ray-tools-quast==5.2.0
```

`/mnt/user_storage` is the right mount for two reasons. It survives cluster termination, unlike
`/mnt/cluster_storage`, which is recreated per job. It is also mounted at node boot, so it exists
before pip runs.

The custom image has none of this: the tools are present at node boot and nothing resolves at
run time, which is why it is the default here.

## A third route: `--container-runtime native`

`native` asks Ray to supply each task's tools through a per-task `runtime_env` derived from the
manifest, so a task gets only what its command actually invokes. It gives the tightest per-task
environment and needs no image, but it still needs a wheel source (`[ray] tool_wheel_dir`), so it
inherits the staging problem above. `wdl_on_ray/envs.py` implements the derivation if you want it.

`none` gets one thing free that `native` does not: with the whole toolchain on every node,
`minimap2` is on PATH for QUAST whatever the quast wheel declares, so the silent
contiguity-only-report failure described in `manifest.toml` cannot happen.

## A fourth route: `--container-runtime ray`, one image per task

Reach for this when a single image stops being reasonable, as with a pipeline of dozens of tools
spanning conflicting runtimes, or when per-task image provenance is a requirement rather than a
preference, as it is in a validated clinical pipeline.

Ray's `runtime_env` accepts an `image_uri`, and Ray then runs the *worker process itself* inside
that image. The nesting is done by the platform rather than by a container CLI this backend
invokes, so it works where `podman run` fails at `container-init exec`. Each task therefore runs
in its own image, and `runtime.docker` describes something that really executed.

### What it costs

The WDL's declared images cannot be used verbatim. A nested image's Ray and Python must match the
cluster's **exactly**, Python to the patch level, so task images have to be built *from* the
cluster's base image. `us.gcr.io/broad-dsp-lrma/lr-flye:2.8.3` is not built that way and will not
start. What you build instead is a small image per task class, from the same base this template's
`Dockerfile` uses, carrying that task's tools from the same `manifest.toml`.

For this template's base, `anyscale/ray:2.56.0-py312`, that target is **Ray 2.56.0 and Python
3.12.12** ([base-image
reference](https://docs.anyscale.com/reference/base-images/ray-2560/py312), or read it off any
candidate base with `docker run --rm <image> python -c 'import ray, sys;
print(ray.__version__, sys.version.split()[0])'`). Note that the image's *Debian* `python3` is
3.10.12 and is not the interpreter Ray runs on; matching that one instead is a plausible and
entirely silent way to get this wrong. `wdl-on-ray doctor` prints the versions to match, and
`wdl-on-ray probe-image` checks a built image against them.

The rest of the constraints come from Ray or the platform:

- An `image_uri` environment cannot also carry `pip`, `conda`, `uv`, `working_dir` or
  `py_modules`; each image must be self-contained. `env_vars` is allowed.
  `wdl_on_ray.envs.validate_runtime_env` rejects the invalid combinations before dispatch.
- The shared run directory has to be visible at the same absolute path inside every task image,
  because miniwdl passes files between tasks by path. Where it is not, the failure is a
  missing-input error on the *second* task naming a path that plainly exists.
- Kubernetes-backed clouds need the ray container running privileged for the nested worker
  container to start.

### Checking before you commit to it

```bash
wdl-on-ray probe-image anyscale/image/my-wdl-flye:1
```

Runs one Ray task in that image and reports whether its Ray and Python versions match the driver's
and whether the shared run directory is readable and writable from inside it. Both are
preconditions, and the second decides whether this mode can work on your cluster at all. Run it
once per image.

The probe needs nothing in the image but Ray. Its code travels inside the task, the same way the
real dispatch path ships `wdl_on_ray.job`, so an image carrying one tool and no Python packages
answers correctly instead of failing to deserialize.

### Wiring it up

Map each `runtime.docker` value to the image that should run it:

miniwdl's config file is INI whose *values* are JSON, not TOML. A bare string, and a
dict on one line:

```ini
# ~/.config/miniwdl.cfg
[ray]
container_runtime = ray
task_image_map = {"us.gcr.io/broad-dsp-lrma/lr-flye:2.8.3": "anyscale/image/my-wdl-flye:1", "us.gcr.io/broad-dsp-lrma/lr-quast:5.2.0": "anyscale/image/my-wdl-quast:1", "us.gcr.io/broad-dsp-lrma/lr-asm:0.1.13": "anyscale/image/my-wdl-asm:1", "us.gcr.io/broad-dsp-lrma/lr-utils:0.1.8": "anyscale/image/my-wdl-base:1", "docker.io/library/ubuntu:20.04": "anyscale/image/my-wdl-base:1"}
```

Two things this snippet is fussy about, both of which fail at config load rather than
at dispatch. The map's braces hold JSON, so `"key": "value"`; TOML's `"key" = "value"`
gives `JSONDecodeError: Expecting ':' delimiter`. And it has to be one line: an indented
continuation is a `configparser.ParsingError`. (Quoting a plain string value, as in
`container_runtime = "ray"`, is harmless — miniwdl unquotes it.) The same map through
the environment instead:

```bash
export MINIWDL__RAY__CONTAINER_RUNTIME=ray
export MINIWDL__RAY__TASK_IMAGE_MAP='{"us.gcr.io/broad-dsp-lrma/lr-flye:2.8.3": "anyscale/image/my-wdl-flye:1"}'
```

A task whose image is not in the map **fails to dispatch**, by design: silently running it in the
cluster image would give that one task the advisory-tag behaviour of `none`, inside a run that
otherwise looks isolated, and arriving at that by accident is worse than choosing it. Both escapes
are explicit: a `"*"` entry catches everything unlisted, and `task_image_fallback = "cluster"` runs
unmapped tasks in the cluster image.

`wdl-on-ray doctor` prints the resolved map and the versions your images must be built against.
