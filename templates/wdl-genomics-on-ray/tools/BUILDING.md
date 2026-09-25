# Building the toolchain

[`manifest.toml`](manifest.toml) pins each tool's version, source URL and sha256 in one place.
Three things read it; none of them duplicates it:

| Consumer | What it does |
|---|---|
| [`build_tools.sh`](build_tools.sh) | installs the toolchain into a prefix, which is what the `Dockerfile` runs |
| [`build_wheels.sh`](build_wheels.sh) | packages the same builds as pip wheels, by running `build_tools.sh` |
| `wdl_on_ray/envs.py` | derives a per-task `runtime_env` under `--container-runtime native` |

`build_tools.sh` checks every download against its sha256, so the image and the wheels carry the
same bytes. `envs.py` reads only the versions and the executables each tool provides.

## Build your own image

The template ships a `Dockerfile`. To build it under your own name:

```bash
cd templates/wdl-genomics-on-ray
anyscale image build -n my-wdl-tools --containerfile Dockerfile
```

The command prints an image URI you can pass to `anyscale job submit --image-uri ...`, set as
`image_uri:` in `job.yaml`, or select when launching a workspace.

### Adding a tool

1. Add a `[tools.<name>]` block to `manifest.toml` with `version`, `kind`, `url`, `sha256`, and
   a `[tools.<name>.provides]` map of PATH name to path inside the payload.
2. `kind = "binary"` and `kind = "pypi"` need nothing else. `kind = "source"` needs a build
   recipe in `build_one()` in `build_tools.sh`, which has three to copy from.
3. Add the tool to both verification lists at the bottom of `build_tools.sh` (the `for exe in ...`
   loop that checks PATH, and the `--version` calls below it) and to the `RUN set -eux` block in
   the `Dockerfile`. They are there to catch a tool that installs and then does not run.
4. Rebuild the image.

Get the sha256 with `curl -fsSL <url> | sha256sum`.

## The stock-image route, and why it needs a staging step

You can run this template on a stock `anyscale/ray` image instead, with the toolchain delivered
as wheels. It has more moving parts, so read this before choosing it.

```bash
# must run on linux x86_64; easiest inside the template's own image
docker run --rm -v "$PWD:/w" -w /w <image> bash tools/build_wheels.sh /w/wheelhouse
```

That produces `wdl-on-ray-tools-{minimap2,samtools,bcftools,flye,quast}` wheels. On an Anyscale
image their shims land in `/home/ray/anaconda3/bin`, which is already first on every Ray worker's
PATH, so `pip install` alone puts a tool on PATH, with no image build and no `ENV` changes.

The two obvious ways for a job to install them both fail:

- A `--find-links` index published over `https://`. pip fetches it anonymously, so a private
  bucket is unreachable whatever IAM role the node carries: pip logs `Looking in links: https://…`
  and then `No matching distribution found`. Only a public index works.
- A relative `--find-links ./wheelhouse`. A job's `requirements:` is installed at cluster
  startup, before the working directory is staged, so the wheels that travel with the submission
  are not there yet.

What works is staging them to a durable mount first, then pointing `--find-links` at that path:

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

`/mnt/user_storage` survives cluster termination, unlike `/mnt/cluster_storage`, which is
recreated per job, and it is mounted at node boot, so it exists before pip runs.

The custom image needs none of this: the tools are present at node boot and nothing resolves at
run time, which is why it is the default here.

## `--container-runtime native`

`native` has Ray supply each task's tools through a per-task `runtime_env` derived from the
manifest, so a task gets only what its command invokes. It needs no image, but it needs a wheel
source (`[ray] tool_wheel_dir`), so it inherits the staging problem above. `wdl_on_ray/envs.py`
implements the derivation.

`none` gets one thing free that `native` does not: with the whole toolchain on every node,
`minimap2` is on PATH for QUAST whatever the quast wheel declares, so the contiguity-only report
described in `manifest.toml` cannot happen.

## `--container-runtime ray`: one image per task

Use this when a single image stops being reasonable, as with a pipeline of dozens of tools across
conflicting runtimes, or when per-task image provenance is a requirement, as in a validated
clinical pipeline.

Ray's `runtime_env` accepts an `image_uri`, and Ray then runs the worker process itself inside
that image. The platform does the nesting, not a container CLI this backend invokes, so it works
where `podman run` fails at `container-init exec`. Each task runs in its own image, and
`runtime.docker` describes something that really executed.

### What it costs

The WDL's declared images cannot be used as they are. A nested image's Ray and Python must match
the cluster's exactly, Python to the patch level, so task images have to be built from the
cluster's base image. `us.gcr.io/broad-dsp-lrma/lr-flye:2.8.3` is not built that way and will not
start. Build a small image per task class instead, from the base this template's `Dockerfile`
uses, carrying that task's tools from the same `manifest.toml`.

For this template's base, `anyscale/ray:2.58.0-py312`, the target is Ray 2.58.0 and Python
3.12.13 ([base-image reference](https://docs.anyscale.com/reference/base-images/ray-2580/py312)).
Read it off any candidate base with
`docker run --rm <image> python -c 'import ray, sys; print(ray.__version__, sys.version.split()[0])'`.
The image's system `python3` is 3.10.12 and is not the interpreter Ray runs on; matching it
instead fails silently. `wdl-on-ray doctor` prints the versions to match, and
`wdl-on-ray probe-image` checks a built image against them.

The other constraints come from Ray or the platform:

- An `image_uri` environment cannot also carry `pip`, `conda`, `uv`, `working_dir` or
  `py_modules`, so each image must be self-contained; `env_vars` is allowed.
  `wdl_on_ray.envs.validate_runtime_env` rejects the invalid combinations before dispatch.
- The shared run directory has to be visible at the same absolute path inside every task image,
  because miniwdl passes files between tasks by path. Where it is not, the second task fails with
  a missing-input error naming a path that exists.
- Kubernetes-backed clouds need the ray container running privileged for the nested worker
  container to start.

### Checking an image before you depend on it

```bash
wdl-on-ray probe-image anyscale/image/my-wdl-flye:1
```

This runs one Ray task in that image and reports whether its Ray and Python versions match the
driver's and whether the shared run directory is readable and writable from inside it. Both are
preconditions, and the second decides whether this mode can work on your cluster at all. Run it
once per image.

The probe needs nothing in the image but Ray. Its code travels inside the task, the same way the
real dispatch path ships `wdl_on_ray.job`, so an image carrying one tool and no Python packages
answers correctly instead of failing to deserialize.

### Worked example: medaka on a GPU

The template ships a Containerfile for medaka,
[`Dockerfile.medaka-gpu`](Dockerfile.medaka-gpu). It has not yet been built or run.

medaka stays out of the cluster image because of its size. Every other tool here is a binary or a
small source build with no Python dependencies, which is what lets them share one environment.
medaka is a PyTorch application: installing it adds 1.2 GB even with CPU-only wheels (measured on
the 2.56.0 base), and GPU polishing needs a CUDA base. On the 2.56.0 base it also needed a numpy
upgrade that broke `cupy`; on 2.58.0, `medaka==2.2.2` resolves against the image's packages without
moving any of them (checked by resolution on 2026-09-24, not by install).

Build it:

```bash
anyscale image build -n wdl-medaka-gpu --containerfile tools/Dockerfile.medaka-gpu
```

Then map the tag `MedakaPolish` already declares, and run every other task in the cluster image:

```ini
[ray]
container_runtime = ray
task_image_map = {"us.gcr.io/broad-dsp-lrma/lr-medaka:0.1.0": "anyscale/image/wdl-medaka-gpu:1"}
task_image_fallback = cluster
```

Run with `medaka_rounds` above 0 and `medaka_use_gpu = true`. The WDL needs no edits:
`MedakaPolish` already declares the image tag, `gpuCount: if use_gpu then 1 else 0` and a
`gpu_type`, and the backend maps those onto Ray's accelerator resources. The cluster needs a GPU
worker group for the request to be satisfiable, or the task waits rather than failing. The default
`gpu_type` is a T4 (`nvidia-tesla-t4`, which on AWS means g4dn); for another GPU, set
`ONTAssembleCohort.assemble.MedakaPolish.gpu_type` in the inputs, in Ray's spelling (`"L4"`) or
GCE's.

The Containerfile builds `FROM anyscale/ray:2.58.0-py312-cu128`, the CUDA variant of the cluster
image's base, so its Ray 2.58.0 and Python 3.12.13 match. It also downloads the model at build
time with `medaka tools download_models`, because medaka otherwise fetches it on first use, which
on a GPU node in a private subnet is a run that hangs rather than one that fails.

### Wiring it up

Map each `runtime.docker` value to the image that should run it. miniwdl's config file is INI
whose *values* are JSON, so a map is a JSON object on one line:

```ini
# ~/.config/miniwdl.cfg
[ray]
container_runtime = ray
task_image_map = {"us.gcr.io/broad-dsp-lrma/lr-flye:2.8.3": "anyscale/image/my-wdl-flye:1", "us.gcr.io/broad-dsp-lrma/lr-quast:5.2.0": "anyscale/image/my-wdl-quast:1", "us.gcr.io/broad-dsp-lrma/lr-asm:0.1.13": "anyscale/image/my-wdl-asm:1", "us.gcr.io/broad-dsp-lrma/lr-utils:0.1.8": "anyscale/image/my-wdl-base:1", "docker.io/library/ubuntu:20.04": "anyscale/image/my-wdl-base:1"}
```

Two mistakes fail at config load rather than at dispatch. TOML's `"key" = "value"` inside the
braces gives `JSONDecodeError: Expecting ':' delimiter`, and an indented continuation line is a
`configparser.ParsingError`. Quoting a plain string value, as in `container_runtime = "ray"`, is
harmless; miniwdl unquotes it. The same map through the environment:

```bash
export MINIWDL__RAY__CONTAINER_RUNTIME=ray
export MINIWDL__RAY__TASK_IMAGE_MAP='{"us.gcr.io/broad-dsp-lrma/lr-flye:2.8.3": "anyscale/image/my-wdl-flye:1"}'
```

A task whose image is not in the map fails to dispatch, by design: silently running it in the
cluster image would give that one task the advisory-tag behaviour of `none`, inside a run that
otherwise looks isolated. Two escapes are explicit: a `"*"` entry maps every unlisted image to one
image URI, and `task_image_fallback = cluster` runs unmapped tasks in the cluster image.

`wdl-on-ray doctor` prints the resolved map and the versions your images must be built against.
