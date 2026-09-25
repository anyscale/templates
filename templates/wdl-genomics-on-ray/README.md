# Run WDL genomics workflows on Ray

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/wdl-genomics-on-ray"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/wdl-genomics-on-ray" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: ~20 min (~10 min at `quick` scale; see Step 1)

This template runs a WDL workflow over a cohort on one autoscaling Ray cluster, with no Cromwell
server and no Terra workspace. miniwdl runs the workflow as a single process on the head node, and
the `wdl_on_ray` backend in this template turns each WDL task into a
[Ray task](https://docs.ray.io/en/latest/ray-core/tasks.html) that lands on a worker that is
already up, instead of on a VM booted for it.

The workflow is a port of [Broad Institute's ONT assembly
pipeline](https://github.com/broadinstitute/long-read-pipelines): Flye assembles each sample de
novo, QUAST evaluates the assembly against GRCh38, and minimap2 and paftools call
assembly-versus-reference variants. The samples are the GIAB Ashkenazi trio (HG002, HG003, HG004),
each assembled independently over a region of chromosome 20.

You finish with a Flye assembly, a QUAST report and a paftools VCF per sample, a timeline of which
node ran each task, and a `job.yaml` that runs the trio over all 64 Mbp of chr20. One measured run
of that job took 2h04m and cost about $10 at on-demand prices (see Time and cost estimates).

Compared with Terra or Cromwell's Google backend, which give each task its own VM:

- small tasks run on nodes that are already up, instead of each paying an instance boot;
- the worker pool grows and shrinks with the cohort's queued tasks and can run on spot, with a
  lost node charged to the task's `preemptible` budget;
- the outputs land on a cluster that runs Python, so downstream analysis can run there too
  (Step 6b).

Cromwell's HPC backends and miniwdl-slurm also pack tasks onto shared nodes; this backend does the
same on a managed, autoscaling cluster.

miniwdl still does the language work: parsing, type checking, scatters, call caching, input
localization and output collection. The backend, selected with
`[scheduler] container_backend = ray`, decides only where each task's command runs:
`runtime { cpu: 30  memory: "32 GiB" }` becomes a Ray request for 30 CPUs and 32 GiB, and nothing
is provisioned per task.

![One Ray task per WDL task: the WDL's tasks, miniwdl's language layer, and the autoscaling Ray cluster they land on](https://raw.githubusercontent.com/anyscale/templates/main/templates/wdl-genomics-on-ray/assets/architecture.png)

### What was ported

Seven of the ten `.wdl` files are adapted from long-read-pipelines 4.0.68 (`02089d9`) under
BSD-3-Clause; the notice is in `wdl/LICENSE`. `ReadStats.wdl`, `ONTAssembleCohort.wdl` and
`smoke.wdl` are new, under Apache-2.0. Most of the diff is portability: `gsutil` calls, GCS-only
paths, `/proc/cpuinfo` core counts. Each file's header lists its own changes, and
[`PIPELINE.md`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/wdl/pipelines/ONT/Assembly/PIPELINE.md)
tabulates them.

Two change what a default run produces, which matters if you compare against a Cromwell run:

- `medaka_rounds` is 0, where upstream runs three rounds of medaka. medaka is not in this
  template's image, so the assemblies below carry only Flye's own polishing round, which is not
  equivalent; see `PIPELINE.md`.
- Flye's read-type flag comes from the reads' declared chemistry instead of upstream's hardcoded
  `--nano-raw`, so R10.4.1 reads get `--nano-hq`, which is what Flye's documentation prescribes
  for R10. `--asm-coverage` is off by default: capping at Flye's documented 40x cut the N50 to a
  third on the full chr20 reads. `flye_impute_params = false` restores upstream's command line
  exactly.

## Where this fits

It fits existing WDL that runs across many samples and whose tasks differ widely in size: here,
ten tasks per sample, from 1 to 30 CPUs. One sample's graph is a chain with one long task in it,
which leaves the scheduler little to pack; a cohort gives it several graphs at once (Step 5). The
backend runs WDL only.

Check the container model first. A validated clinical pipeline needs each task to run in the image
its WDL declares. Ray can run each task in its own image, but that image has to be rebuilt on the
cluster's Ray and Python versions and mapped from the declared tag. The Containers note after
Step 2 compares the modes.

## Scope

What the backend does with each `runtime {}` key:

| key | what the backend does |
|---|---|
| `cpu`, `memory` | requests them from Ray as `num_cpus` and `memory` |
| `gpuCount`, `gpuType`, `acceleratorType` | maps them to Ray GPU and accelerator-type requests |
| `docker` | runs it as declared under `podman`, `docker`, `apptainer` and `singularity`; maps it to a rebuilt image under `ray`; treats it as advisory under `none` |
| `preemptible` | spends it on Ray node or worker loss, which the backend raises as miniwdl's `Interrupted` |
| `maxRetries` | leaves it to miniwdl; Ray-level retries default to 0 so they cannot bypass it |
| `disks` | logs it and otherwise ignores it, unless you set `[ray] disk_resource_name` |

Three limits to know before you port anything:

- A task's ceiling is the largest node that is up when the run starts. miniwdl clamps
  `runtime.cpu` and `runtime.memory` to it and logs a warning. If the big workers are not up yet,
  pass `--max-cpu` (and set `[ray] max_memory_bytes`) to the worker shape.
- Above that ceiling a request waits instead of failing. Ray keeps a request that no node can
  serve pending, so a request larger than every instance type (possible once you raise the
  ceiling by hand), or a GPU request with no GPU worker group, queues with no error.
- Call caching is off by default, as in miniwdl, and miniwdl's default cache directory is
  node-local. `--call-cache DIR` turns it on with DIR as the cache, and DIR must be on shared
  storage. `job.yaml` shows what that does and does not survive.

## Set-up

A workspace launched from this template already has the files. Elsewhere:

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/wdl-genomics-on-ray
```

## Step 1: Check the cluster and the toolchain

`WDL_DEMO_SCALE` sets how much the notebook assembles. `standard`, the default, is a 10 Mbp region
of chromosome 20 per sample; `quick`, which CI runs, is 2 Mbp. Set `WDL_DEMO_SCALE=quick` in the
environment, or change `SCALE` in the cell below. Both run the same pipeline, tools and resource
requests over the same GIAB reads.

Measured as Anyscale Jobs on this template's AWS compute config (`m5.8xlarge` workers, Ray 2.56.0
image), so cluster provisioning and read staging are included:

| scale | notebook, end to end | cohort workflow | slowest sample in that run |
|---|---|---|---|
| `quick` | ~10 min | 8m05s | 7m52s |
| `standard` | ~17 min | 14m00s | 13m32s |

At both scales the cohort takes 1.03x its slowest sample's wall clock, because each assembly gets a
worker of its own once the cluster has scaled to three. A workspace that is already running skips
the provisioning. Over the whole 64 Mbp chromosome (`job.yaml`) the ratio is 1.006.

The cell prints the scale and data location, then runs `wdl-on-ray doctor`, which reports what the
backend would decide without running anything. Check that `none` reads `always available` and
that the default run dir is on shared storage.


```python
import json
import os
import pathlib
import subprocess

# `standard` by default; CI sets `quick`. Only the region size differs, so both scales
# exercise the same pipeline, tools and resource requests.
SCALE = os.getenv("WDL_DEMO_SCALE", "standard")
SCALES = {
    "quick":    {"region": "chr20:1,000,000-3,000,000",  "span": "2 Mbp",  "expect": "~10 min"},
    "standard": {"region": "chr20:1,000,000-11,000,000", "span": "10 Mbp", "expect": "~17 min"},
}
if SCALE not in SCALES:
    raise ValueError(f"WDL_DEMO_SCALE must be one of {sorted(SCALES)}, got {SCALE!r}")
CFG = SCALES[SCALE]

# The GIAB Ashkenazi trio (son, father, mother), assembled independently. Three samples
# because one sample's task graph is a chain (merge, measure, assemble, polish, evaluate),
# which leaves the scheduler nothing to pack.
SAMPLES = ["HG002", "HG003", "HG004"]

TEMPLATE_DIR = pathlib.Path.cwd()
DATA_URI = f"s3://anyscale-public-materials/genomics/giab-trio-chr20/{SCALE}"

# Every node of the cluster can see /mnt/cluster_storage, and the backend requires that: a WDL
# task's working directory has to be readable by whichever node runs the task that consumes its
# output. /mnt/local_storage would fail with "file not found" on the second task.
WORK = pathlib.Path("/mnt/cluster_storage/wdl-on-ray")
DATA_DIR, RUN_DIR, SMOKE_DIR = WORK / "data", WORK / "runs", WORK / "smoke"
for d in (DATA_DIR, RUN_DIR, SMOKE_DIR):
    d.mkdir(parents=True, exist_ok=True)
# Pin Ray's temp root before moving TMPDIR. On Linux, Ray finds a running cluster through
# <TMPDIR>/ray/ray_current_cluster, and a workspace sets no RAY_ADDRESS, so moving TMPDIR alone
# would make `wdl-on-ray run` start a second, empty Ray on the head instead of joining this
# one; the head offers CPU: 0, so every task would wait forever. Jobs get RAY_ADDRESS from the
# platform.
os.environ.setdefault("RAY_TMPDIR", os.environ.get("TMPDIR", "/tmp"))
os.environ["TMPDIR"] = str(WORK / "tmp")
pathlib.Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)


def run(cmd, **kwargs):
    """Run a command, streaming output, and raise if it fails.

    Deliberately not the `!` shell magic: `!` does not raise on a non-zero exit, so a failed
    assembly would leave this notebook green and the CI test passing.
    """
    printable = " ".join(str(c) for c in cmd)
    print(f"$ {printable}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


print(f"scale       {SCALE}  ({CFG['span']} of {CFG['region']}, expect {CFG['expect']})")
print(f"samples     {', '.join(SAMPLES)}")
print(f"data        {DATA_URI}")
print(f"run dir     {RUN_DIR}\n")
run(["wdl-on-ray", "doctor"])
```

## Step 2: Read the workflow you're about to run

Both files are WDL 1.0 with no Ray-specific syntax. `ONTAssembleWithFlye.wdl` is the per-sample
pipeline, adapted from Broad's; its header lists the divergences from upstream.
`ONTAssembleCohort.wdl` scatters it over a sample list and adds no tasks.

![The pipeline's task graph, with the CPU and memory request each task carries into its Ray resource request](https://raw.githubusercontent.com/anyscale/templates/main/templates/wdl-genomics-on-ray/assets/pipeline-dag.png)

The cell type-checks the cohort workflow with miniwdl, lists each file's calls, and prints
`Flye.Assemble`'s `runtime {}` block, which is where Ray gets the task's request. The check prints
the document tree and exits 0; its two `NameCollision` lines are lint warnings about call names
inherited from upstream.


```python
wdl_dir = TEMPLATE_DIR / "wdl/pipelines/ONT/Assembly"
cohort_wdl = wdl_dir / "ONTAssembleCohort.wdl"
sample_wdl = wdl_dir / "ONTAssembleWithFlye.wdl"

# Type-check first. This is miniwdl's own checker, nothing Ray-specific, and it is the
# fastest way to confirm the workflow and its imports are internally consistent.
run(["wdl-on-ray", "check", str(cohort_wdl)])

# The calls, in the order each file declares them.
for path in (cohort_wdl, sample_wdl):
    print(f"\ncalls in {path.stem}:")
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("call "):
            print(f"  {stripped.split('{')[0].strip()}")

# One task's runtime block, as written upstream.
flye_task = (TEMPLATE_DIR / "wdl/tasks/Assembly/Flye.wdl").read_text()
block = flye_task[flye_task.index("runtime {"):]
print("\nFlye.Assemble runtime block:")
print("\n".join(block[: block.index("}") + 1].splitlines()))
```

### Containers

On Cromwell, each task's command runs in the image its `docker` key names. On Anyscale, Ray itself
runs inside a container, and a nested container CLI does not work there: `podman` installs and
pulls images, but `podman run` fails at `container-init exec` under both `crun` and `runc`. Ray can
still give each task its own image, by starting the task's worker process inside it.

| `--container-runtime` | where a task's command runs | `runtime.docker` |
|---|---|---|
| `none` (this notebook) | in the Ray worker's environment, with tools from the cluster image | advisory |
| `ray` | in a Ray worker that Ray starts inside a per-task image | mapped through `[ray] task_image_map` |
| `native` | in the Ray worker, with the task's tools from a Ray `runtime_env` built from the manifest | optionally mapped through `[ray] image_env_map` |
| `podman`, `docker`, `apptainer`, `singularity` | in a nested container the backend starts on the worker, where the platform allows one | run as declared |

`auto`, the default, picks the first of `podman`, `docker`, `apptainer` or `singularity` it finds on
the driver's node, and otherwise falls back to `none`.

Under `none` the declared tag is a label. `Flye.wdl` declares `lr-flye:2.8.3`, but the Flye that
runs is the 2.9.5 pinned in
[`tools/manifest.toml`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/tools/manifest.toml),
and the two are not the same assembler: `--nano-hq` does not exist in 2.8.3. The toolchain is still
pinned, in the manifest instead of the WDL. Tasks get separate working directories but share the
node's environment. Five tools fit in one image; a GATK Best Practices pipeline, with dozens of
tools across Java, Perl and pinned Pythons, does not.

Under `ray`, `runtime.docker` names what actually ran. Each task image's Ray and Python must match
the cluster's exactly, Python to the patch level, so you build task images from the cluster's base
and map the WDL's declared tags to them. `wdl-on-ray probe-image <uri>` runs one task in a
candidate image and checks the versions, and whether the shared run directory is readable and
writable inside it.
[`tools/BUILDING.md`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/tools/BUILDING.md)
covers the rest, including the privileged container that Kubernetes-backed clouds need.

## Step 3: Run a 60-second workflow first

`smoke.wdl` makes some seeds, scatters over them and gathers the results. It needs no genomics
tools and no data, so it confirms that WDL tasks reach Ray workers before any assembly depends on
it.

The backend writes a `ray_placement.json` beside each task's other files, recording the node that
ran it. The second half of the cell tallies those files, and Step 5's timeline reads the same ones.
This run sets `[ray] scheduling_strategy = SPREAD` through an environment variable, with no change
to the WDL. Eight 2-CPU shards fit on one 32-vCPU worker, so until the cluster has a second worker,
expect the tally to read one node.


```python
# SPREAD is best-effort: shards spread over whatever workers exist at dispatch time. Ray's
# default strategy packs before it spreads, which suits the real pipeline, so the override
# lives on this one command rather than in a config file.
run([
    "wdl-on-ray", "run", str(TEMPLATE_DIR / "wdl/smoke/smoke.wdl"),
    "shards=8",
    "--container-runtime", "none",
    "--dir", str(SMOKE_DIR),
], env={**os.environ, "MINIWDL__RAY__SCHEDULING_STRATEGY": "SPREAD"})

# Which node ran each shard, scoped to the run that just finished (miniwdl writes each run
# under its own timestamped directory), so re-running this cell does not count earlier runs.
smoke_run = max((p.parent for p in SMOKE_DIR.glob("*/outputs.json")), key=lambda p: p.stat().st_mtime)
placements = sorted(smoke_run.glob("**/ray_placement.json"))
nodes = {}
for path in placements:
    record = json.loads(path.read_text())
    node = record.get("node_id", record.get("node_ip", "unknown"))
    nodes.setdefault(node, []).append(path.parent.name)

print(f"\n{len(placements)} tasks ran across {len(nodes)} node(s):")
for node, tasks in nodes.items():
    print(f"  {node}: {len(tasks)} task(s)")
```

## Step 4: Stage the reads

The reads are the GIAB Ashkenazi trio, HG002 (son), HG003 (father) and HG004 (mother), from ONT's
public [`s3://ont-open-data`](https://registry.opendata.aws/ont-open-data/) release `giab_2023.05`:
R10.4.1 chemistry, dorado sup v4.1.0 basecalls, aligned to GRCh38. `tools/stage-demo-data.sh`
slices one chromosome-20 region out of those alignments and pairs it with the matching reference
slice, so the derivation is reproducible from public inputs.

Each scale's `MANIFEST.json` records, per sample, the read count, total bases, read N50, coverage
and sha256, plus the chemistry and basecaller. Those two decide Flye's read mode and the correct
medaka model, and these FASTQs do not carry them. The cell stages the files to cluster storage,
prints the manifest and checks each FASTQ against its sha256.

Flye assembles these reads de novo, but they were selected by their alignment to GRCh38, which
makes the problem easier than a real assembly:

- Reads from a divergent haplotype that failed to align are absent, as is anything unmapped.
  Reads mismapped into the region from paralogous sequence elsewhere are present.
- Whole reads are kept, not just the overlapping portion, so coverage tapers over about one read
  length at the edges and the reported coverage runs a percent or two high.
- Secondary and supplementary records are dropped (`-F 0x900`), so a read whose primary
  alignment falls outside the region is absent even if part of it aligns inside, as happens
  across a structural breakpoint at the boundary.

Contiguity and genome fraction are optimistic as a result. Treat the numbers as a check on the
pipeline, not a benchmark of the assembler.

The reference slice is named `chr20:<start>-<end>`, with coordinates numbered from 1 within it, so
the VCFs in Step 6b are slice-local and need lifting back before any comparison with a GIAB truth
set.

miniwdl could localize the `s3://` URIs itself, once per run. Staging here lets the cell check the
checksums before anything runs, and a re-run skips the download.


```python
manifest_path = DATA_DIR / f"MANIFEST.{SCALE}.json"
reference = DATA_DIR / f"reference.{SCALE}.fa"
reads = {s: DATA_DIR / f"{s}.reads.{SCALE}.fastq.gz" for s in SAMPLES}

wanted = [(f"{DATA_URI}/MANIFEST.json", manifest_path), (f"{DATA_URI}/reference.fa", reference)]
wanted += [(f"{DATA_URI}/{s}.reads.fastq.gz", path) for s, path in reads.items()]

for uri, dest in wanted:
    if dest.exists():
        print(f"already staged: {dest.name}")
        continue
    # --no-sign-request because the bucket is public: a *signed* request is evaluated against
    # the node role's policy, so signing can fail where anonymous access succeeds.
    run(["aws", "s3", "cp", "--no-sign-request", "--only-show-errors", uri, str(dest)])

manifest = json.loads(manifest_path.read_text())
print(f"\n{manifest['region']}   {manifest['chemistry']}, {manifest['basecaller']}")
print(f"{'sample':<8}{'reads':>10}{'bases':>14}{'read N50':>10}{'coverage':>10}")
for entry in manifest["samples"]:
    print(f"{entry['sample']:<8}{entry['reads']:>10,}{entry['bases']:>14,}"
          f"{entry['read_n50']:>10,}{entry['coverage']:>9}x")

# Check each FASTQ against the manifest's sha256 before anything runs on it.
import hashlib

for entry in manifest["samples"]:
    path = reads[entry["sample"]]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == entry["sha256"], f"{path.name} does not match the manifest checksum"
print("\nall checksums match the manifest")
```

## Step 5: Assemble the cohort

One workflow runs the ten-task pipeline for all three samples at once: merge the reads, measure
them (two tasks), compute the genome length, assemble with Flye, polish, evaluate and summarize
with QUAST, align to the reference and call variants. Each `Assemble` asks for 30 CPUs and takes a
worker to itself. The 1- to 8-CPU tasks run on nodes that are already up, and only those of 2 CPUs
or fewer fit beside a running assembly. On a VM-per-task backend each small task would pay an
instance boot.

The inputs below set every task's resources. The requests are per task, not per sample, so more
samples means more tasks eligible at once, not bigger nodes. Three settings are worth reading:

- `runtime_attr_flye` asks for 30 CPUs and 32 GiB. One full-chromosome run peaked at 17.6 GiB
  resident, against the 106 GiB that upstream's formula (100 GiB plus 1 GiB per 10 Mbp of genome)
  reserves for chr20. The 30 CPUs fit the 32-vCPU worker in this template's compute config;
  change the two together.
- `read_chemistry` comes from the manifest and selects Flye's read mode.
- `medaka_rounds: 0` makes `MedakaPolish` pass the draft through unchanged, since medaka is not in
  the cluster image (see Turn polishing on). Flye then keeps its own polishing round, because the
  workflow never lets the total number of polishing passes reach zero. That rule guards against a
  silent no-op; it does not make the two polishers equivalent.

While it runs, each task logs `task started on Ray worker` with its queue time and node.


```python
inputs = {
    "ONTAssembleCohort.samples": [
        {"name": s, "fastqs": [str(reads[s])], "ref_fasta": str(reference)} for s in SAMPLES
    ],
    # From the manifest; this is what selects Flye's read mode.
    "ONTAssembleCohort.read_chemistry": manifest["chemistry"],
    "ONTAssembleCohort.flye_num_threads": 30,
    "ONTAssembleCohort.quast_num_threads": 8,
    "ONTAssembleCohort.align_num_threads": 8,
    # 0 does not skip the task: MedakaPolish still runs and copies the draft through
    # unchanged, which is why it appears in the task list. Flye's own polishing round is
    # then the only one applied.
    "ONTAssembleCohort.medaka_rounds": 0,
    "ONTAssembleCohort.medaka_use_gpu": False,
    "ONTAssembleCohort.runtime_attr_fastq_stats":     {"cpu_cores": 1,  "mem_gb": 4,  "disk_gb": 50, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_genome_length":   {"cpu_cores": 2,  "mem_gb": 8,  "disk_gb": 50, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_read_divergence": {"cpu_cores": 4,  "mem_gb": 16, "disk_gb": 100, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_merge_fastqs":    {"cpu_cores": 4,  "mem_gb": 16, "disk_gb": 100, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_flye":            {"cpu_cores": 30, "mem_gb": 32, "disk_gb": 500, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_medaka":          {"cpu_cores": 4,  "mem_gb": 16, "disk_gb": 200, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_quast":           {"cpu_cores": 8,  "mem_gb": 32, "disk_gb": 100, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_quast_summary":   {"cpu_cores": 1,  "mem_gb": 4,  "disk_gb": 20, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_align_paf":       {"cpu_cores": 8,  "mem_gb": 32, "disk_gb": 100, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_paftools":        {"cpu_cores": 2,  "mem_gb": 8,  "disk_gb": 50, "preemptible_tries": 3},
}
inputs_path = WORK / f"inputs.cohort.{SCALE}.json"
inputs_path.write_text(json.dumps(inputs, indent=2))

print(f"assembling {len(SAMPLES)} samples x {CFG['span']} of {CFG['region']}")
print(f"expect about {CFG['expect']}, roughly one sample's wall clock, because they run"
      f" concurrently once the cluster has scaled up\n")
run([
    "wdl-on-ray", "run", str(cohort_wdl),
    "-i", str(inputs_path),
    "--container-runtime", "none",
    "--dir", str(RUN_DIR),
    "--verbose",
])
```

### How the run used the cluster

Each WDL task left two timestamps: the backend writes `ray_placement.json` when the task starts
holding resources on a worker, and miniwdl last writes `task.log` when the task finishes. The cell
draws one bar per task from those, grouped by sample and coloured by node, with no added
instrumentation.

Look for small tasks from one sample sharing a colour with another sample's assembly, which is
tasks packing onto a node that is already up, and compare the total span with one sample's. On a
`quick` run on this template's AWS compute config (Ray 2.56.0 image), the three assemblies queued
32 s, 93 s and 93 s as the backend's `seconds_queued` reports them (it overstates long waits), and
landed on three separate workers; the cohort took 8m05s against 7m52s for its slowest sample. The
93 s is the autoscaler bringing up workers two and three from `min_nodes: 1`. With `max_nodes: 2`,
the third assembly queued 7m37s behind the first and the run took 13m16s.


```python
import matplotlib.pyplot as plt

run_dir = max(RUN_DIR.glob("*/outputs.json"), key=lambda p: p.stat().st_mtime).parent

# One row per task, not per attempt. started = when the backend wrote
# ray_placement.json; finished = miniwdl's last write to task.log. The placement file sits
# in the task directory, and each attempt's Ray task rewrites it (job.execute_and_record),
# so a retried task's bar starts at its last attempt and takes that attempt's node.
#
# miniwdl nests a sub-workflow's calls under the scatter shard that made them, so the
# path from the run directory carries the sample: .../call-assemble/shard-1/call-Flye/...
# The per-sample grouping below reads that path; no instrumentation was added.
rows = []
for placement in run_dir.glob("**/ray_placement.json"):
    task_dir = placement.parent
    parts = placement.relative_to(run_dir).parts
    shard = next((p for p in parts if p.startswith("shard-")), None)
    sample = SAMPLES[int(shard.split("-")[1])] if shard else "cohort"
    rows.append((sample, task_dir.name.removeprefix("call-"),
                 json.loads(placement.read_text()).get("node_id", "?")[-6:],
                 placement.stat().st_mtime, (task_dir / "task.log").stat().st_mtime))

# Group by sample, and by start time within each sample, so the chart reads as three
# pipelines rather than one interleaved list.
rows.sort(key=lambda r: (SAMPLES.index(r[0]) if r[0] in SAMPLES else -1, r[3]))

t0 = min(r[3] for r in rows)
span_min = (max(r[4] for r in rows) - t0) / 60
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
palette = ["#2a78d6", "#eb6834", "#1baf7a", "#a855c7", "#d4a017"]
nodes = list(dict.fromkeys(r[2] for r in rows))
node_color = {n: (palette[i] if i < len(palette) else "#c3c2b7") for i, n in enumerate(nodes)}

fig, ax = plt.subplots(figsize=(10, 0.30 * len(rows) + 1.6), dpi=110)
for i, (sample, name, node, started, finished) in enumerate(rows):
    left, dur = (started - t0) / 60, (finished - started) / 60
    # A pass-through task finishes in under a second; give it a sliver you can still see.
    ax.barh(i, max(dur, span_min * 0.004), left=left, height=0.6, color=node_color[node])
    if dur > span_min * 0.05:
        label = f"{dur:.1f} min" if dur >= 1 else f"{dur * 60:.0f} s"
        ax.text(left + max(dur, span_min * 0.004) + span_min * 0.01, i,
                label, va="center", fontsize=8, color=MUTED)

ax.set_yticks(range(len(rows)), [f"{s}  {n}" for s, n, *_ in rows], fontsize=7.5)
ax.invert_yaxis()
# A rule between samples, so three pipelines are visible as three blocks.
for i in range(1, len(rows)):
    if rows[i][0] != rows[i - 1][0]:
        ax.axhline(i - 0.5, color=GRID, linewidth=1)
ax.set_xlabel("minutes since the first task started", fontsize=9, color=MUTED)
ax.set_title(f"{len(rows)} WDL tasks as Ray tasks: {len(SAMPLES)} samples on {len(nodes)} node(s),"
             f" {span_min:.0f} min wall clock", fontsize=11, color=INK, loc="left")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.grid(axis="x", color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
if len(nodes) > 1:  # one node needs no legend; the title already counts it
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=node_color[n]) for n in nodes],
              labels=[f"node \u2026{n}" for n in nodes], loc="lower right",
              fontsize=8.5, frameon=False, title="colour = worker node",
              title_fontsize=8.5)
plt.tight_layout()
plt.show()
```

## Step 6: Read the assemblies

miniwdl collects every declared workflow output into the run directory and lists them in
`outputs.json`. The cohort emits one array per output, in `samples` order, from the assemblies to
`flye_params` and `read_stats`, which record what each sample measured and chose.

The cell prints the QUAST metrics per sample, the Flye flags each sample ran with, and the read
statistics. With `medaka_rounds = 0` there is one assembly per sample, so QUAST reports one column
each. With polishing on, QUAST evaluates the draft and the polished assembly in one run, against
one reference with one set of thresholds.

The cell also asserts that QUAST produced a `Genome fraction` line. QUAST computes its
reference-based metrics with a minimap2 it builds from source at install time, and when that build
is missing QUAST still exits 0 with a contiguity-only report. `Quast.wdl` fails the task in that
case; the assert is a second check.

Against GRCh38, a different individual's haploid reference, read the columns this way:

- The mismatch rate has a floor. A haplotype-collapsed human assembly carries roughly 85-95 real
  SNVs per 100 kbp against GRCh38 before any assembly error, so the rate compares a sample's draft
  and polished columns; it is not an error rate. It also excludes indels, the ONT error class that
  matters most; read `# indels per 100 kbp` for those.
- QUAST scores any rearrangement against the reference larger than 1 kbp (its default
  `--extensive-mis-size`) as a misassembly and breaks NGA50's aligned blocks there, so both count
  the sample's own structural variants along with assembly errors.
- Genome fraction near 100% is expected here, because the reads were selected by aligning to this
  reference.
- Polishing moves bases, not contig boundaries, so N50 barely responds to it.

One `quick` run on the trio (Ray 2.56.0 image, `--nano-hq --iterations 1` for all three):

| | HG002 | HG003 | HG004 |
|---|---|---|---|
| # contigs | 1 | 1 | 1 |
| N50 | 2,036,301 | 2,081,008 | 2,155,765 |
| NGA50 | 586,366 | 1,122,380 | 583,361 |
| Genome fraction (%) | 98.104 | 99.984 | 98.230 |
| # mismatches per 100 kbp | 112.63 | 104.41 | 113.65 |
| # indels per 100 kbp | 33.08 | 30.37 | 32.03 |
| # misassemblies | 6 | 3 | 6 |

Each sample assembles as one contig spanning its 2 Mbp region, as expected at 60-90x. Expect your
numbers near these rather than identical: Flye's result depends on thread order, so the
alignment-based columns move by a percent or two between runs of the same input, and genome
fraction barely at all. A rerun on the Ray 2.58.0 image (2026-09-24) matched genome fraction
exactly and N50 to within 0.06%.

Coverage and read length do not explain the differences between samples. HG003 has the highest
NGA50 and the fewest misassemblies while sitting in the middle on coverage (75x against 89x and
59x) and last on read N50. HG002 lands within 1% of its mother, HG004, on NGA50, genome fraction,
mismatch rate and misassembly count, which fits the two sharing sequence that differs from GRCh38
in this window. Read these columns as differences from GRCh38, not as a ranking of the assemblies.

`read_stats` also reports `pairwise_divergence`, which depends on the region as much as on the
reads: the same HG002 reads measure 0.0721 over this 2 Mbp window, 0.0986 over 10 Mbp and 0.1532
over all of chr20, because the estimator counts spurious overlaps between repeat copies and the
whole chromosome includes the centromere. `flye_params` flags a value above the `--nano-hq` band;
nothing branches on it.


```python
# One level only: run roots are RUN_DIR's direct children. miniwdl also writes an
# outputs.json inside each nested sub-workflow directory, and one of those can be newer
# than the run root's, so a recursive glob sorted by mtime would pick the wrong file.
outputs_path = max(RUN_DIR.glob("*/outputs.json"), key=lambda p: p.stat().st_mtime)
# miniwdl's outputs.json is the bare name -> value mapping; the {"dir", "outputs"}
# envelope appears only on the CLI's stdout. Accept both, like persist_outputs.py.
report = json.loads(outputs_path.read_text())
outputs = report.get("outputs", report)


def quast_key(name):
    """Map a QUAST metric's display name onto its key in a `quast_summary` map.

    SummarizeQuastReport builds that map by running `sed 's/ /_/g'` and `s/>=/gt/`
    over QUAST's space-aligned report.txt, so "Genome fraction (%)" arrives as
    "Genome_fraction_(%)" and only single-word metrics (N50, NGA50) survive intact.
    Looking up the display names directly would match four keys out of nine and make the
    assert below impossible to satisfy, whatever QUAST produced.
    """
    return name.replace(" ", "_").replace(">=", "gt")


METRICS = (
    "# contigs", "Total length", "N50", "NG50", "NGA50",
    "Genome fraction (%)", "# misassemblies",
    "# mismatches per 100 kbp", "# indels per 100 kbp",
)

names = outputs["ONTAssembleCohort.sample_names"]
summaries = outputs["ONTAssembleCohort.quast_summaries"]

# A QUAST run against a reference that reports no genome fraction has no correctness metrics at
# all, and still exits 0. Fail here rather than let contiguity numbers stand in for a complete
# evaluation.
for name, summary in zip(names, summaries):
    assert quast_key("Genome fraction (%)") in summary, (
        f"{name}: QUAST produced no reference-based metrics; its minimap2 is missing. "
        "See tools/manifest.toml, [tools.quast]."
    )

print(f"{'metric':<28}" + "".join(f"{n:>16}" for n in names))
for metric in METRICS:
    key = quast_key(metric)
    if not any(key in s for s in summaries):
        continue
    print(f"{metric:<28}" + "".join(f"{s.get(key, '-'):>16}" for s in summaries))

print("\nFlye parameters each sample derived from its own reads:")
for name, params in zip(names, outputs["ONTAssembleCohort.flye_params"]):
    print(f"  {name}  {params['read_mode']}  {params['extra_args'] or '(no extra args)'}")

print("\nWhat each read set measured:")
print(f"  {'sample':<8}{'reads':>10}{'read N50':>10}{'coverage':>10}{'divergence':>12}")
for name, stats in zip(names, outputs["ONTAssembleCohort.read_stats"]):
    print(f"  {name:<8}{int(stats['num_reads']):>10,}{int(stats['read_n50']):>10,}"
          f"{float(stats['coverage']):>9.1f}x{float(stats['pairwise_divergence']):>12.4f}")

print("\nassemblies:")
for name, path in zip(names, outputs["ONTAssembleCohort.assemblies"]):
    print(f"  {name}  {path}")
```

The next cell plots Nx (solid) and NGx (dashed) for each assembly, with QUAST's 500 bp contig
floor, so each curve at x = 50 is the N50 or NG50 printed above. The assembly usually runs longer
than the reference slice, because whole reads overhang the region's edges, which puts NGx on or
above Nx. Neither curve uses alignments. At `quick` scale each sample is one contig, so its two
curves coincide as one flat line at the contig's length.


```python
import bisect

import matplotlib.pyplot as plt

# QUAST's default contig floor, so the curve at x = 50 is the N50 QUAST printed above;
# counting every contig would compute a different statistic under the same name.
MIN_CONTIG = 500


def contig_lengths(fasta_path, min_length=MIN_CONTIG):
    lengths, current = [], 0
    with open(fasta_path) as fasta:
        for line in fasta:
            if line.startswith(">"):
                if current:
                    lengths.append(current)
                current = 0
            else:
                current += len(line.strip())
    if current:
        lengths.append(current)
    return sorted((n for n in lengths if n >= min_length), reverse=True)


def nx_points(lengths, denominator):
    """(x, Nx) for x in 1..100: the contig length at which the largest-first running
    total first covers x% of `denominator`. Stops where the assembly stops covering."""
    cums, total = [], 0
    for length in lengths:
        total += length
        cums.append(total)
    pts = []
    for x in range(1, 101):
        idx = bisect.bisect_left(cums, denominator * x / 100)
        if idx >= len(lengths):
            break
        pts.append((x, lengths[idx]))
    return pts


ref_len = sum(contig_lengths(reference, min_length=0))  # the reference slice staged in Step 4
assemblies = {n: contig_lengths(p)
              for n, p in zip(names, outputs["ONTAssembleCohort.assemblies"])}

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
palette = ["#2a78d6", "#eb6834", "#1baf7a"]
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
for (name, lengths), color in zip(assemblies.items(), palette):
    # Solid: Nx, against the assembly's own length. Dashed: NGx, against the reference's;
    # the two separate wherever those lengths differ.
    for pts, style, label in (
        (nx_points(lengths, sum(lengths)), "-", f"{name}  Nx"),
        (nx_points(lengths, ref_len), "--", f"{name}  NGx"),
    ):
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.step(xs, [y / 1e6 for y in ys], where="post", color=color,
                linestyle=style, linewidth=1.8, label=label)

ax.set_xlabel("x (%)", fontsize=9, color=MUTED)
ax.set_ylabel("contig length (Mbp)", fontsize=9, color=MUTED)
ax.set_title(f"{len(assemblies)} assemblies of {ref_len / 1e6:.1f} Mbp"
             "   (solid: Nx, assembly length   dashed: NGx, reference length)",
             fontsize=10.5, color=INK, loc="left")
ax.set_xlim(0, 100)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("bottom", "left"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.grid(axis="y", color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=8.5, frameon=False, ncol=len(assemblies))
plt.tight_layout()
plt.show()
```

### Step 6b: Analyze the outputs in the same cluster

The outputs are files on shared storage, and the cluster that made them is still up, so the next
cell is plain Ray code in this notebook, with no new job and no copy.

It compares the trio's variant calls. HG002 is the son of HG003 and HG004, so most of the variants
called in the child should appear in at least one parent. The cell normalizes each paftools VCF
with `bcftools norm`, reads and parses the three with Ray Data on the workers, and compares them
as `(chrom, pos, ref, alt)` sets. It prints each sample's call count, then the share of HG002's
calls found in either parent. Three effects keep the "in neither parent" share well above zero:

- Allele dropout. Flye collapses haplotypes, so each assembly carries one mosaic haplotype and a
  heterozygous site appears in it roughly at random. A variant the child inherited can be missing
  from the parent that transmitted it. Allele sampling alone puts roughly 25-40% of the child's
  calls in neither parent.
- Representation. paftools places an indel wherever the alignment's `cs` tag put it, which inside
  a homopolymer is arbitrary, so two assemblies can spell one indel two ways. The
  `bcftools norm -f <ref> -m -any` step removes most of that.
- Callable regions. `paftools call -L` calls only inside alignment blocks above its length
  threshold, and the three assemblies break differently, so a call in a block that one sample
  cleared and another did not counts as unshared.

This is a consistency check on three independent assemblies, not a Mendelian error rate, which
would need diploid callsets, benchmark regions and coordinates that are not slice-local. It also
has no unrelated control: much of any genome's variation against GRCh38 is common in the
population, so an unrelated sample would share many of these calls too.


```python
import os

import ray

# The cluster the WDL tasks ran on; nothing new is provisioned here.
vcfs = dict(zip(names, outputs["ONTAssembleCohort.vcfs"]))

# Normalize before comparing: paftools places an indel wherever minimap2's `cs` tag put
# it, so inside a homopolymer two assemblies can spell one indel two ways, and the set
# comparison below would count it twice. `-m -any` also splits multiallelic records, so a
# site with two ALTs compares allele by allele.
NORM_DIR = WORK / "vcf-normalized"
NORM_DIR.mkdir(parents=True, exist_ok=True)
if not pathlib.Path(f"{reference}.fai").exists():
    run(["samtools", "faidx", str(reference)])

normalized = {}
for sample, path in vcfs.items():
    dest = NORM_DIR / f"{sample}.norm.vcf"
    run(["bcftools", "norm", "-f", str(reference), "-m", "-any", "-o", str(dest), str(path)])
    normalized[sample] = dest

# Ray Data reports the source path per row; map basenames back to samples up front so the
# UDF does no searching. Basenames are unique because each sample's prefix is its name.
SAMPLE_BY_FILE = {path.name: sample for sample, path in normalized.items()}
assert len(SAMPLE_BY_FILE) == len(normalized), "VCF basenames are not unique across samples"


def parse_vcf_lines(batch):
    """VCF text -> (sample, chrom, pos, ref, alt), skipping headers.

    Runs as a Ray Data batch UDF, so this is the parallel part: one task per block,
    across the same workers that just ran the assemblies.
    """
    out = {"sample": [], "chrom": [], "pos": [], "ref": [], "alt": []}
    for text, path in zip(batch["text"], batch["path"]):
        text = str(text)
        if text.startswith("#") or not text.strip():
            continue
        fields = text.split("\t")
        if len(fields) < 5:
            continue
        sample = SAMPLE_BY_FILE.get(os.path.basename(str(path)))
        if sample is None:
            continue
        out["sample"].append(sample)
        out["chrom"].append(fields[0])
        out["pos"].append(int(fields[1]))
        out["ref"].append(fields[3])
        out["alt"].append(fields[4])
    return out


variants = (
    ray.data.read_text([str(p) for p in normalized.values()], include_paths=True)
    .map_batches(parse_vcf_lines, batch_format="numpy")
    .to_pandas()
)

if variants.empty:
    # Not an expected outcome at any scale. 2 Mbp of collapsed human assembly should carry
    # on the order of 1,500 SNVs against GRCh38. Empty means the assembly did not align, or
    # every alignment block fell under `paftools call -L` (50 kb by default), which filters
    # silently and exits 0.
    raise AssertionError(
        "no variant calls in any sample. Check the QUAST genome fraction above, then "
        "min_alignment_length_call in CallAssemblyVariants.wdl against this run's NGA50."
    )

print(f"{len(variants):,} normalized calls across {variants['sample'].nunique()} samples")
print(variants.groupby("sample").size().to_string(header=False))

# A call is identified by (chrom, pos, ref, alt). The child's calls that appear in
# neither parent are the ones to look at.
def keyset(sample):
    rows = variants[variants["sample"] == sample]
    return set(zip(rows["chrom"], rows["pos"], rows["ref"], rows["alt"]))


child, father, mother = keyset("HG002"), keyset("HG003"), keyset("HG004")
inherited = child & (father | mother)
unshared = child - father - mother

print(f"\nHG002 calls:                    {len(child):,}")
print(f"  also in HG003 or HG004:       {len(inherited):,}"
      f"  ({100 * len(inherited) / max(len(child), 1):.1f}%)")
print(f"  in neither parent:            {len(unshared):,}"
      f"  ({100 * len(unshared) / max(len(child), 1):.1f}%)")
print("\nAllele dropout from haplotype collapse alone puts the expected 'in neither'")
print("figure around 25-40%, so read the second number against that rather than")
print("against zero. It is a consistency check, not a violation rate.")
```

## Step 7: Persist the outputs

`/mnt/cluster_storage` is shared by the nodes of one cluster and is deleted when that cluster
terminates. In a workspace that is fine, since the cluster stays up. An Anyscale Job terminates its
cluster when it succeeds, so a job that leaves its results there loses them by finishing.

[`persist_outputs.py`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/persist_outputs.py)
copies the declared outputs only, not the run tree and its intermediates, to `WDL_ON_RAY_RESULTS`
or else to the first writable durable mount (`/mnt/user_storage`, then `/mnt/shared_storage`). It
prints one line per output and the total file count.


```python
run(["python", str(TEMPLATE_DIR / "persist_outputs.py"), "--outputs", str(outputs_path)])
```

## Run it as a job

The notebook assembles a 2 or 10 Mbp region so you can watch it.
[`job.yaml`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/job.yaml)
runs the same trio over all 64 Mbp of chromosome 20 per sample, at the coverage the reads came at,
as an Anyscale Job:

```bash
cd templates/wdl-genomics-on-ray
anyscale job submit --config-file job.yaml
```

Submit it from that directory. `working_dir: .` is relative to the shell, not to the config file,
so submitting from the repository root uploads the wrong tree and the job fails within seconds on
a missing WDL.

### Time and cost estimates

One run of that job: Ray 2.56.0 image, three on-demand `m5.8xlarge` workers, 97x, 80x and 63x
coverage, no coverage cap.

| | HG002 | HG003 | HG004 |
|---|---|---|---|
| Flye | 1h55m33s | 1h44m12s | 1h24m04s |
| Sample, end to end | 2h01m15s | 1h48m46s | 1h37m45s |
| # contigs | 62 | 71 | 52 |
| N50 | 33,263,886 | 16,721,982 | 33,233,489 |
| L50 / L90 | 1 / 2 | 2 / 3 | 1 / 2 |
| NGA50 | 2,082,054 | 2,188,745 | 1,960,991 |
| Genome fraction (%) | 95.798 | 95.854 | 95.820 |
| # mismatches per 100 kbp | 154.02 | 142.61 | 149.30 |
| # misassemblies | 121 | 111 | 129 |

The workflow took 2h01m58s and the job 2h03m56s, on three workers plus a head node that runs no
tasks: about 6 worker node-hours, or $10 at us-east-1 on-demand list prices. Carry node-hours over
rather than dollars, since prices vary by region and commitment. The cohort finished in its slowest
sample's time, 2h01m58s against 2h01m15s. A VM-per-task backend would use similar node-hours; the
difference is 30 instance boots, one per task, against three workers here.

Read mode is the largest lever on time. An earlier single-sample run of HG002 used upstream's
`--nano-raw` on the same instance type and reads (97x, uncapped, Flye's one polishing round):

| HG002, full chr20, `m5.8xlarge` | `--nano-raw` (earlier run) | `--nano-hq` (this run) |
|---|---|---|
| Flye, total | 14h44m | 1h55m33s |
| polishing stage | 9h51m | 31m15s |
| everything before polishing | 4h53m | 1h24m |
| N50 | 33,279,582 | 33,263,886 |

`--nano-hq` is 7.6x faster for the same N50. The coverage cap is a contiguity lever rather than a
speed one: capping at 40x cut the N50 to a third to save 14.6 minutes on c6i.16xlarge
(`ONTAssembleWithFlye.wdl`'s header has the four-run grid).

The contigs are chromosome-arm length. HG002 and HG004 hold 90% of the assembly in two contigs, one
per arm: 33.26 and 25.78 Mbp for HG002, 33.23 and 25.85 Mbp for HG004, which is 98% of the q arm's
33.9 Mbp of assemblable sequence and 98% of the p arm's 26.3 Mbp, each stopping at the centromere.
HG002's remaining 5.3 Mbp is in 60 contigs averaging 89 kb. HG003 needs three contigs for the same
90%: its p arm comes out whole at 25.78 Mbp and its q arm splits into 16.72 and 16.54 Mbp, while its
genome fraction and NGA50 are in line with the other two.

NGA50 is about 2 Mbp in every full-chr20 run here: 1.96-2.19 Mbp across the trio, 1.83-2.08 Mbp
in the `--nano-hq` pair of the four-run grid, and 2.1 Mbp in an unpolished run, so polishing is not
what limits it. As in Step 6, QUAST breaks aligned blocks at every rearrangement against GRCh38
over 1 kbp, the sample's own structural variants included, so NGA50 and the misassembly count
measure distance from the reference as much as assembly error.

![The assembly against chromosome 20, one contig per arm stopping at the centromere, and the N50-against-NGA50 gap measured on a separate unpolished run](https://raw.githubusercontent.com/anyscale/templates/main/templates/wdl-genomics-on-ray/assets/chr20-contigs.png)

The first panel is the 14h44m `--nano-raw` run; the current flags reproduce its q-arm contig to
within 0.05% and extend its p-arm contig from 23.65 to 25.78 Mbp. The second panel is the
unpolished run (`--nano-raw --iterations 0 --asm-coverage 30 --genome-size 64444167`): N50 33.25
Mbp, NGA50 2.1 Mbp, genome fraction 94.9%. The polished runs above reach the same NGA50, so its gap
from N50 is not a polishing effect.

A sizing note from the same run: `Assemble` reserves 30 of a worker's 32 cores, so any 4-core
neighbour blocks it. HG004's `Assemble` waited about 587 s from dispatch to Flye's first log line,
with all three workers up inside 100 s, because all three `MeasureDivergence` tasks had landed on
the first worker and HG002's ran for 669.8 s. It cost the cohort nothing: HG004 is the smallest
sample, so its measurements finish first, and it still finished 1410 s before HG002. A fourth node would not
have helped, since HG004's request was passed over twice as the other workers came up. The task
log's `seconds_queued` reads 639.1 s, because it stops when the driver sees `ray_placement.json`
on shared storage.

The worker groups now ask for spot (`market_type: PREFER_SPOT`, in `configs/` and `job.yaml`) and
fall back to on-demand when spot has no capacity; the head stays on demand. The run above was
entirely on demand, and the retry path has been traced in code but has not yet met a real reclaim.
Every task in the inputs files has `preemptible_tries: 3`, so a reclaimed node costs a retry of the
tasks on it, not the run, and a retried task restarts from zero. The spot figures are estimates:

| spot, estimated rather than measured | |
|---|---|
| Spot Advisor discount, `m5.8xlarge`, us-west-2, 2026-09-24 | -69%, taking this run from about $10 to about $3.65 |
| saving, modelled at assumed interruptions of 1.5-15% per node-hour | 58-64% |
| what a resume mechanism would add | $0.05-$0.61 per cohort run, so there is none |
| where restart-from-zero stops paying | roughly `1/rate` hours per assembly: ten hours at 10% per node-hour |

### What that job encodes

- `timeout_s: 86400`, against a measured 2h03m56s, leaves room for a slower instance type, a whole
  genome, or a spot fleet spending retries.
- `WDL_ON_RAY_RESULTS` is where Step 7's copy goes; without it the outputs go with the cluster.
- `--call-cache` runs with `max_retries: 0`. A job-level retry runs on a new cluster, and this
  job's cache is on `/mnt/cluster_storage`, which the platform recreates per job, so a retry would
  find nothing to reuse. The entrypoint's comments show the durable arrangement and its cost.
- `preemptible_tries: 3` comes from the inputs file. The task defaults stay at upstream's values,
  mostly 0, so the WDL still runs unmodified elsewhere. Check what your own pipeline declares
  before relying on spot: a task with `preemptible` and `maxRetries` both at 0 fails the workflow
  on its first reclaim.

## Adding a tool

[`tools/manifest.toml`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/tools/manifest.toml)
pins each tool's version, URL and sha256. The image build, the wheel build and `native` mode's
per-task environments all read it.

Under `none`, the notebook's mode, every tool has to be on every node before the first task
dispatches, so the manifest builds one image that holds all of them. Add a `[tools.<name>]` block
and rebuild from the template directory:

```bash
anyscale image build -n my-wdl-tools --containerfile Dockerfile
```

The fetch, the checksum check and the PATH shim are generic; only `kind = "source"` needs a build
recipe. [`tools/BUILDING.md`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/tools/BUILDING.md)
has the procedure.

When one image stops being reasonable, switch to `--container-runtime ray`: build a small image per
task class from the cluster's base, map each `runtime.docker` value in `[ray] task_image_map`, keep
the map in version control next to the WDL, and run `wdl-on-ray probe-image` against each image
first.

The manifest can also build pip wheels (`tools/build_wheels.sh`) for a stock `anyscale/ray` image,
but a job can only install them after they are staged to `/mnt/user_storage`; `BUILDING.md`
explains why, and the custom image avoids it.

## Next steps

### Point it at your own reads

Start from `wdl/pipelines/ONT/Assembly/inputs.chr20.cohort.json`, which carries the
`runtime_attr_*` sizing, and replace the samples and chemistry:

```json
{
  "ONTAssembleCohort.samples": [
    {"name": "NA12878",
     "fastqs": ["s3://my-bucket/NA12878/flowcell1.fastq.gz",
                "s3://my-bucket/NA12878/flowcell2.fastq.gz"],
     "ref_fasta": "s3://my-bucket/ref/GRCh38.fa"}
  ],
  "ONTAssembleCohort.read_chemistry": "R10.4.1"
}
```

`fastqs` takes one entry per flow cell of one sample; miniwdl localizes `gs://`, `s3://` and
`https://` itself. Four things to check:

- The reference sets the genome size. `ComputeGenomeLength` sums every sequence in `ref_fasta`, and
  the workflow uses the total as Flye's genome size and, when `runtime_attr_flye` is unset, to size
  its memory request. QUAST and paftools also evaluate against the whole reference. For reads from
  one region, pass the matching reference slice, or at least set
  `ONTAssembleCohort.assemble.flye_genome_size`.
- `read_chemistry` selects the read mode: R10 gets `--nano-hq`, and unset keeps upstream's
  `--nano-raw`. Take it from your run metadata; a FASTQ may not record it.
- `runtime_attr_flye` has to fit a node you have. Upstream's sizing for a 3.1 Gbp genome is 16
  cores and ~410 GiB, far above this template's 128 GiB workers, so a whole genome needs a larger
  worker group.
- Above ~100 Mbp, set `ONTAssembleCohort.assemble.quast_is_large` to `true` for QUAST's `--large`.

### Run your own WDL

Start with `wdl-on-ray check <your.wdl>`, miniwdl's type checker, which takes seconds. Then read the
Scope table for what the backend does with each `runtime {}` key; `disks`, for one, is logged and
ignored. On a real port, getting the tools onto the cluster usually takes longer than the WDL;
[`tools/BUILDING.md`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/tools/BUILDING.md) covers the four routes and where each stops working.

### Turn polishing on

`medaka_rounds` is 0 here, the largest divergence from upstream. medaka has its own image,
[`tools/Dockerfile.medaka-gpu`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/tools/Dockerfile.medaka-gpu),
because it would add about 1.2 GB to the cluster image even with CPU-only torch (measured on the
2.56.0 base), and GPU polishing needs a CUDA base. That image has not yet been built or run.

```bash
anyscale image build -n wdl-medaka-gpu --containerfile tools/Dockerfile.medaka-gpu
```

Then add a GPU worker group to the compute config, map the image under `--container-runtime ray`
(`tools/BUILDING.md` has the config), and set `medaka_rounds` above 0 and `medaka_use_gpu` to true.
`MedakaPolish` asks for a T4 unless you set `ONTAssembleCohort.assemble.MedakaPolish.gpu_type`.
Check that `medaka_model` matches your chemistry *and sampling rate* with
`medaka tools list_models`; a mismatched model degrades the consensus without an error. With medaka
on, Flye skips its own polishing round, and QUAST reports the draft and the polished assembly side
by side. For human-scale polishing, ONT now points to `dorado polish` rather than medaka.

### Scale the cohort

Add entries to `samples`. The requests are per task, so a bigger cohort needs more nodes, not
bigger ones: raise `max_nodes`. Pass `--task-concurrency` too. By default the task pool is sized
once, at startup: to the cluster's CPU count at that moment (capped at 200) when `RAY_ADDRESS` is
set, as in a job, and otherwise to the head node's core count, as in a workspace. That caps how
many tasks the autoscaler ever sees queued. Three samples over 64 Mbp took 2h04m on three workers,
against 2h01m15s for the slowest of them.

### Make it cheaper

The workers already run on spot, and every task in the inputs files carries
`preemptible_tries: 3`; Time and cost estimates has the estimate and its assumptions. The saving
shrinks for long assemblies, because Flye does not checkpoint across a retry.

### Read the backend

[`wdl_on_ray/backend.py`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/wdl_on_ray/backend.py) is where the miniwdl contract is written
down; [`wdl_on_ray/resources.py`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/wdl_on_ray/resources.py) maps `runtime {}` onto Ray
resources and is where a scheduling hint of your own would go;
[`wdl_on_ray/runtimes.py`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/wdl_on_ray/runtimes.py) holds the container modes.
