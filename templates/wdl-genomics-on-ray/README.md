# Run WDL genomics workflows on Ray

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/wdl-genomics-on-ray"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/wdl-genomics-on-ray" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: about 2 hours (or ~40 min at `quick` scale; see Step 1)

A miniwdl backend that runs each WDL task as a
[Ray task](https://docs.ray.io/en/latest/ray-core/tasks.html) rather than provisioning a VM for it.
The pipeline it runs here is [Broad Institute's ONT assembly
workflow](https://github.com/broadinstitute/long-read-pipelines): Flye assembles a region of human
chromosome 20 from Oxford Nanopore reads, QUAST evaluates the assembly, and minimap2 and paftools
call variants against the reference. Three samples, the GIAB Ashkenazi trio, run through three
copies of that pipeline on one autoscaling cluster.

[WDL](https://openwdl.org/) is the language of GATK Best Practices, WARP and Terra. On Cromwell's
cloud backends the unit of execution is a VM per task, so a workflow's wall clock carries one
instance boot per task and its dependency graph never reaches a scheduler that could pack it.
Cromwell's HPC backends do pack, and so do miniwdl-slurm and Nextflow on Kubernetes. What none of
them do is put the workflow on the same cluster as the Python, Ray Data or training work
downstream of it, which is what Step 6b uses.

miniwdl discovers container backends through a Python entry point, and this template registers one
called `ray`. A workflow selects it with `[scheduler] container_backend = ray`. miniwdl keeps doing
all the language work (parsing, type checking, scatter expansion, call caching, input localization,
output collection); the backend replaces only the part that decides *where* a task's command runs.

`runtime { cpu: 30  memory: "32 GiB" }` in the WDL becomes `num_cpus=30, memory=32<<30` on the Ray
task. The scheduler places it, and nothing is provisioned per task.

![One Ray task per WDL task: the WDL's tasks, miniwdl's language layer, and the autoscaling Ray cluster they land on](https://raw.githubusercontent.com/anyscale/templates/main/templates/wdl-genomics-on-ray/assets/architecture.png)

### What was ported

Seven of the ten `.wdl` files are Broad's, adapted from long-read-pipelines 4.0.68 (`02089d9`) and
BSD-3-Clause; the notice is in `wdl/LICENSE`. `ReadStats.wdl`, `ONTAssembleCohort.wdl` and
`smoke.wdl` have no upstream counterpart and are Apache-2.0. Most of the diff is portability:
`gsutil` calls, GCS-only paths, `/proc/cpuinfo` core counts. Each file's header lists its own
divergences and
[`PIPELINE.md`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/wdl/pipelines/ONT/Assembly/PIPELINE.md)
tabulates them.

Two of those change what a default run produces, which matters if you plan to compare against a
Cromwell run:

- Upstream polishes with three rounds of medaka. `medaka_rounds` is 0 here, because medaka is not
  in this template's image, so the assemblies below carry Flye's own polishing round and nothing
  more. The two are not equivalent; see `PIPELINE.md`.
- Flye's read-type flag, `--asm-coverage` and `--genome-size` are derived from the reads instead of
  left at a hardcoded `--nano-raw`. For R10.4.1 sup reads that selects `--nano-hq`, which is what
  Flye's documentation prescribes for R10 chemistry. `flye_impute_params = false` restores
  upstream's command line exactly.

## Where this fits

Existing WDL, elastic compute, no rewrite. Tasks whose resource requests differ by an order of
magnitude, where a VM per task wastes both money and minutes. Cohorts, where those savings
multiply: three samples here is a demonstration, and 500 samples is 500 of these graphs sharing one
autoscaling pool, with the small tasks packing onto nodes that are already up. Step 5 shows why the
demo is three samples rather than one.

Per-task container images are the case to check first. A validated clinical pipeline needs the
image its WDL declares, and Ray can give each task its own image, but those images have to be built
against the cluster's own Ray and Python versions rather than pulled as the WDL names them. The
note after Step 2 covers what each container mode costs.

CWL and Nextflow estates have the same scheduling problem. This backend speaks WDL only.

## Scope

What the backend does with a `runtime {}` block, and what it leaves alone:

| | |
|---|---|
| `cpu`, `memory` | become `num_cpus` and `memory` on the Ray task |
| `gpuCount`, `gpuType`, `acceleratorType` | mapped to Ray accelerator resources |
| `docker` | honoured under `podman`, `apptainer`, `singularity` and `ray`; advisory under `none` |
| `preemptible` | Ray node and worker death maps to miniwdl's `Interrupted`, which routes here |
| `maxRetries` | miniwdl's, unchanged. Ray-level retries default to 0 so they cannot bypass it |
| `disks` | parsed, then logged and discarded unless you set `[ray] disk_resource_name` |

Three limits worth knowing before you port anything:

- **A task's ceiling is the largest node, not the cluster.** miniwdl clamps `runtime.cpu` against a
  limit the backend computes once at startup, defaulting to the biggest node then alive. On a
  cluster that has not scaled up yet, pass `--max-cpu` to tell it what is coming.
- **An unsatisfiable request waits rather than failing.** Ray cannot distinguish "no node this big
  exists" from "the autoscaler has not caught up", so a task asking for more CPU than any instance
  type offers hangs with no error.
- **Call caching is off unless you ask for it.** That is miniwdl's default, and its cache directory
  is node-local, so both have to move together: `--call-cache DIR` on shared storage sets all
  three. `job.yaml` shows the arrangement and what it does and does not survive.

## Set-up

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/wdl-genomics-on-ray
```

## Step 1: Check the cluster and the toolchain

One knob controls how much work this notebook does. `standard` is the default and assembles a
10 Mbp region of chromosome 20 for each of three samples in roughly 60-90 minutes, about one
sample's wall clock, because they run concurrently once the cluster has scaled to three workers.
Measured at `quick` scale, that cohort took 8m05s against 7m52s for its own slowest sample, a
factor of 1.03. `standard` has not been measured. `quick` does a 2 Mbp region per sample in about
20 minutes and is what CI runs; set
`WDL_DEMO_SCALE=quick` before launching Jupyter to use it. Only the region size differs; both
scales run the same code over the same GIAB reads.

The reference has to match the region. `ComputeGenomeLength` derives the assembler's genome size
from it, so handing it all of GRCh38 would size the memory request for a 3.1 Gbp assembly and
suppress `--asm-coverage`, because 500 Mbp of reads over 3.1 Gbp is below the coverage threshold
that emits it.

`wdl-on-ray doctor` reports what the backend would decide without running anything, including
which container runtimes this node can use and whether the call cache is on.


```python
import json
import os
import pathlib
import subprocess

# `standard` is what a reader gets; CI sets `quick`. Only the size of the region differs:
# the pipeline, the tools and the resource requests are identical, so a green `quick` run and
# a `standard` run exercise the same code path.
SCALE = os.getenv("WDL_DEMO_SCALE", "standard")
SCALES = {
    "quick":    {"region": "chr20:1,000,000-3,000,000",  "span": "2 Mbp",  "expect": "~20 min"},
    "standard": {"region": "chr20:1,000,000-11,000,000", "span": "10 Mbp", "expect": "60-90 min"},
}
if SCALE not in SCALES:
    raise ValueError(f"WDL_DEMO_SCALE must be one of {sorted(SCALES)}, got {SCALE!r}")
CFG = SCALES[SCALE]

# The GIAB Ashkenazi trio: son, father, mother. Three samples rather than one because a
# single sample's task graph is a chain (merge, measure, assemble, polish, evaluate)
# and a chain gives the scheduler nothing to do. Three at once is what this backend is
# for, and it is how real work arrives.
SAMPLES = ["HG002", "HG003", "HG004"]

TEMPLATE_DIR = pathlib.Path.cwd()
DATA_URI = f"s3://anyscale-public-materials/genomics/giab-trio-chr20/{SCALE}"

# Every node of the cluster can see /mnt/cluster_storage, and the backend requires that: a WDL
# task's working directory has to be readable by whichever node runs the task that consumes its
# output. /mnt/local_storage would silently produce "file not found" on the second task.
WORK = pathlib.Path("/mnt/cluster_storage/wdl-on-ray")
DATA_DIR, RUN_DIR, SMOKE_DIR = WORK / "data", WORK / "runs", WORK / "smoke"
for d in (DATA_DIR, RUN_DIR, SMOKE_DIR):
    d.mkdir(parents=True, exist_ok=True)
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

Read these before they run. They declare tasks, their inputs and outputs, and a `runtime {}` block
per task, and they are the same WDL a Cromwell backend would consume.

`ONTAssembleWithFlye.wdl` is the per-sample pipeline, adapted from Broad's, and its header lists
the divergences from upstream. `ONTAssembleCohort.wdl` scatters it over a sample list: twenty lines
of WDL, no new tasks, and the only reason the cluster has anything to schedule.

The `runtime {}` blocks are where Ray gets its scheduling information. `cpu` becomes `num_cpus`,
`memory` becomes `memory`, and `docker` is read but honoured only under some container runtimes,
which the note below the cell covers.

![The pipeline's task graph: measurement tasks derive Flye's parameters, and every task carries the CPU/memory request that becomes its Ray resource request](https://raw.githubusercontent.com/anyscale/templates/main/templates/wdl-genomics-on-ray/assets/pipeline-dag.png)


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

Every WDL task declares a `docker` image, and on Cromwell that image is what the command runs in.
Reproducing that on Ray is the awkward part of the port. Ray is already running inside a container,
and the obvious way to start another one is unavailable: `podman` installs and pulls images fine,
but `podman run` fails at `container-init exec` under both `crun` and `runc`. That rules out
driving a container CLI from inside the worker. It does not rule out per-task images, which Ray
supplies another way.

Four modes, and `--container-runtime auto` picks the first of podman, docker, apptainer or
singularity that it finds, falling back to `none`.

**`none`, which the notebook runs.** The task command executes in the Ray worker's environment and
the toolchain comes from the cluster image. The declared tag becomes advisory: `Flye.wdl` declares
`lr-flye:2.8.3` while the Flye that runs is the 2.9.5 pinned in
[`tools/manifest.toml`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/tools/manifest.toml).
Those are not the same assembler; `--nano-hq` does not exist in 2.8.3. What runs is still pinned
and reproducible, just pinned in the manifest rather than in the WDL. Tasks get separate working
directories but share the node's environment, so they are not isolated from each other.

This mode has a ceiling. Five tools fit in one image comfortably. A GATK Best Practices pipeline is
dozens of tools spanning Java, Perl and pinned Pythons, and one image that satisfies all of them is
the situation per-task containers exist to avoid.

**`ray`, per-task images.** Ray's `runtime_env` accepts an `image_uri` and runs the *worker
process* inside it, so the platform does the nesting and the `podman` failure never arises.
`runtime.docker` then names something that really ran. The cost is that a task image's Ray and
Python must match the cluster's exactly, Python to the patch level, so images are built from the
cluster's base and the WDL's declared tags are *mapped* through `[ray] task_image_map` rather than
honoured verbatim. `wdl-on-ray probe-image <uri>` runs one task in a candidate image and reports
whether the versions line up and whether the shared run directory is readable and writable from
inside it.
[`tools/BUILDING.md`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/tools/BUILDING.md)
covers the rest: image self-containment, the run directory's path inside each image, and the
privileged container that Kubernetes-backed clouds need.

**`podman`, `docker`, `apptainer`, `singularity`.** miniwdl's own backends, which work wherever
nested containers are allowed, including local development, and give per-task isolation as the WDL
declares it.

**`native`.** Ray supplies each task's tools through a per-task runtime environment derived from
the manifest, with no image at all. `BUILDING.md` covers it.

## Step 3: Run a 60-second workflow first

`smoke.wdl` makes some seeds, scatters over them and gathers the results. It needs no genomics
tools and no data, so it confirms that WDL tasks really are being dispatched to Ray workers before
an hour of assembly depends on the answer.

The backend writes a `ray_placement.json` next to every task's other artifacts, recording the node
that ran it. The second half of the cell reads those, and Step 5's timeline is drawn from the same
files. This run sets `[ray] scheduling_strategy = SPREAD` as an environment override, with nothing
in the WDL changing, so the shards prefer spreading over whatever workers exist. Eight 2-CPU shards
fit comfortably on one 32-vCPU worker, so until the cluster has grown a second worker, expect the
tally to read one node.


```python
# SPREAD is best-effort: shards spread over whatever workers exist at dispatch time. The
# default strategy suits the real pipeline better, keeping data-adjacent tasks together,
# so the override lives on this one command rather than in a config file.
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
R10.4.1 chemistry, dorado sup basecalls, already aligned to GRCh38. `tools/stage-demo-data.sh`
slices one chromosome-20 region out of those alignments and pairs it with the matching reference
slice. Both the source and the script are public, so the derivation is reproducible.

Each scale ships a `MANIFEST.json` recording, per sample, the read count, total bases, read N50,
coverage and sha256, plus the chemistry and basecaller. Those last two decide Flye's read mode and
which medaka model is correct, and neither is recoverable from a FASTQ.

**This is a regional re-assembly of reference-selected reads, not a de novo assembly.** Reads from
a divergent haplotype that failed to align are absent by construction, as is anything unmapped, so
the hardest reads a real assembly must handle are gone. Reads mismapped *into* the region from
paralogous sequence elsewhere are present. Whole reads are emitted rather than the overlapping
portion, so coverage at the edges tapers over about one read length and the reported coverage runs
a percent or two high. Supplementary alignments are dropped (`-F 0x900`), which also drops reads
crossing a structural breakpoint at the boundary. Contiguity and genome fraction are correspondingly
optimistic. Treat it as a demonstration of the pipeline, not a benchmark of the assembler.

The slice also renames the contig to `chr20:<start>-<end>` and numbers coordinates from 1 within
it, so the VCFs in Step 6b are slice-local and are not comparable to a GIAB truth set without
lifting them back.

miniwdl can localize an `s3://` URI for a `File` input by itself, so passing the URIs straight
through would also work. Staging once here is faster, because three tasks per sample read the
reference and would otherwise fetch it nine times.


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

# The manifest's checksums are the point of publishing them: verify rather than assume.
import hashlib

for entry in manifest["samples"]:
    path = reads[entry["sample"]]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == entry["sha256"], f"{path.name} does not match the manifest checksum"
print("\nall checksums match the manifest")
```

## Step 5: Assemble the cohort

Three samples, one workflow, one cluster. Each sample runs the same ten-task pipeline (merge the
reads, measure them, estimate a genome length, assemble with Flye, polish, evaluate with QUAST,
summarize the report, align to the reference with minimap2, call variants with paftools) and all
three graphs are live at once.

That is what gives the scheduler something to do. One sample's pipeline is a chain with a single
long task in the middle: nothing to pack, nothing to autoscale into. Three concurrent chains have
both. Each `Assemble` wants 30 CPUs and takes a worker to itself, while the 1-, 2- and 4-CPU
measurement tasks from the *other* samples fill the gaps on nodes that are already up. On a
per-task-VM backend each of those small tasks pays an instance boot.

`runtime_attr_flye` is measured rather than inherited. A full-chromosome run peaked at 17.6 GiB
resident against the 110 GiB upstream reserves, and that 110 was upstream's whole-genome default
rather than anything derived. 32 GiB is still roughly double the observed peak.

`cpu_cores: 30` fits the 32-vCPU worker in this template's compute config, and the two have to move
together. A request no instance type can satisfy does not fail; Ray cannot tell "no node this big
exists" from "the autoscaler has not caught up yet", so the workflow waits with no error.

The requests are per task, not per sample. Adding samples does not enlarge the cluster this needs,
it enlarges how many tasks are eligible to run at once, and the autoscaler answers that.

`medaka_rounds: 0` skips medaka, which is absent from this template's image: medaka 1.x is pinned
below Python 3.11 while this image runs 3.12, and medaka 2.x is a different major version with a
different model generation and a deep-learning payload. Flye's own polishing round is what the
assemblies below carry. The workflow will not let total polishing passes reach zero, which is a
guard against a silent no-op rather than a claim that the two polishers are equivalent.


```python
inputs = {
    "ONTAssembleCohort.samples": [
        {"name": s, "fastqs": [str(reads[s])], "ref_fasta": str(reference)} for s in SAMPLES
    ],
    "ONTAssembleCohort.flye_num_threads": 30,
    "ONTAssembleCohort.quast_num_threads": 8,
    "ONTAssembleCohort.align_num_threads": 8,
    # Medaka is absent from this template's image, as the cell above explains;
    # tools/BUILDING.md covers adding it. Rounds 0 does not skip the task: MedakaPolish still
    # runs and copies the draft through untouched, which is why it appears in the task list.
    # Flye's own polishing round is then the only one applied, and the workflow's invariant is
    # that total polishing passes never reach zero.
    "ONTAssembleCohort.medaka_rounds": 0,
    "ONTAssembleCohort.medaka_use_gpu": False,
    "ONTAssembleCohort.runtime_attr_fastq_stats":     {"cpu_cores": 1,  "mem_gb": 4,  "disk_gb": 50},
    "ONTAssembleCohort.runtime_attr_genome_length":   {"cpu_cores": 2,  "mem_gb": 8,  "disk_gb": 50},
    "ONTAssembleCohort.runtime_attr_read_divergence": {"cpu_cores": 4,  "mem_gb": 16, "disk_gb": 100},
    "ONTAssembleCohort.runtime_attr_merge_fastqs":    {"cpu_cores": 4,  "mem_gb": 16, "disk_gb": 100},
    "ONTAssembleCohort.runtime_attr_flye":            {"cpu_cores": 30, "mem_gb": 32, "disk_gb": 500},
    "ONTAssembleCohort.runtime_attr_quast":           {"cpu_cores": 8,  "mem_gb": 32, "disk_gb": 100},
    "ONTAssembleCohort.runtime_attr_quast_summary":   {"cpu_cores": 1,  "mem_gb": 4,  "disk_gb": 20},
    "ONTAssembleCohort.runtime_attr_align_paf":       {"cpu_cores": 8,  "mem_gb": 32, "disk_gb": 100},
    "ONTAssembleCohort.runtime_attr_paftools":        {"cpu_cores": 2,  "mem_gb": 8,  "disk_gb": 50},
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

Every WDL task above ran as one Ray task, and each left two timestamps behind. The backend writes
`ray_placement.json` the moment a task starts holding resources on a worker, and miniwdl closes
`task.log` when it finishes. Those two are enough to draw the whole run, including which tasks
overlapped, where the wall clock went and which node each landed on, with no instrumentation added.

Colour is the node, so a task's colour says which worker it landed on. Small tasks from one sample
sharing a colour with another sample's assembly is the bin-packing this approach is for. Then check
the total span against one sample's, and against the queue times, because the cohort is only ever
as wide as the cluster it got. On a `quick` run against this template's compute config the three
assemblies queued 32s, 93s and 93s and landed on three separate workers, giving 8m05s total against
7m52s for the slowest single sample. The 93s is the autoscaler bringing up workers two and three
from `min_nodes: 1`. Cap the same run at two workers and the third assembly queues 7m37s behind the
first instead, and the total goes to 13m16s.


```python
import matplotlib.pyplot as plt

run_dir = max(RUN_DIR.glob("*/outputs.json"), key=lambda p: p.stat().st_mtime).parent

# One row per dispatched task: started = placement file written on the worker;
# finished = miniwdl's last write to that task's log. Retried attempts appear too.
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

miniwdl collects every declared workflow output into the run directory and reports absolute paths
in `outputs.json`. The cohort emits one array per artifact, in `samples` order, from the assemblies
themselves through to the maps recording what each run measured and chose.

QUAST evaluates the draft and the polished consensus in a single run whenever both exist, so the
two columns come from one reference, one set of thresholds and one execution. With
`medaka_rounds = 0` there is only one assembly per sample, and the cell says so rather than
printing a column against itself.

Polishing moves bases and leaves contig boundaries alone, so N50 barely responds; the metrics that
answer "did it help" are the alignment-based ones. Note the floor under the mismatch rate. These
samples are not GRCh38, and a haplotype-collapsed human assembly carries roughly 85-95 real SNVs
per 100 kbp against the reference before any assembly error at all, so QUAST's mismatch rate cannot
separate variants from error. It is a relative measure between arms or between samples, not an
error rate. QUAST also counts mismatches over aligned bases only and excludes indels, which for ONT
is the error class that matters.

The cell asserts that QUAST produced a `Genome fraction` line. QUAST's reference-based metrics need
an aligner it ships only as source, and when that build fails QUAST still exits 0 and writes a
report carrying contiguity metrics only. Two full runs came back missing those numbers before
anyone noticed.

For comparison, one green run at `quick` scale on the trio, with today's derived flags
(`--nano-hq --iterations 1 --asm-coverage 40 --genome-size 2000001`, identical for all three):

| | HG002 | HG003 | HG004 |
|---|---|---|---|
| # contigs | 2 | 1 | 1 |
| N50 | 1,042,796 | 2,081,275 | 2,155,818 |
| NGA50 | 583,002 | 1,122,996 | 583,361 |
| Genome fraction (%) | 98.187 | 99.984 | 98.230 |
| # mismatches per 100 kbp | 111.93 | 102.82 | 113.78 |
| # indels per 100 kbp | 32.56 | 30.56 | 32.78 |
| # misassemblies | 7 | 3 | 6 |

Expect your own numbers to land near these without matching them. Genome fraction reproduced to
three decimal places across two runs of the same input, and HG003 and HG004's contiguity to within
30 bp, but HG002 came out as 4 contigs with a 702 kb N50 on the earlier run and 2 contigs with a
1.04 Mbp N50 on this one. Flye's repeat graph is thread-order sensitive, so a sample sitting near a
resolution boundary can fall either way between runs. Reference-based metrics are the stable ones;
treat a contig count as an observation about one run.

Genome fraction near 100% is close to guaranteed here and is not a quality result: the reads were
selected by aligning to this reference, so covering it is what they were chosen for. The column
worth reading is HG002, which has the most coverage of the three (89x against 75x and 59x) and
still gives the least contiguous assembly. Coverage is the usual explanation for a contiguity
difference and it is the wrong one here; the read stats printed above have the right one, which is
that HG002 also has the highest measured read-to-read divergence, 0.0721 against 0.0370 and 0.0323.
Three samples over one locus is how you notice that the two do not move together.


```python
# One level only: run roots are RUN_DIR's direct children. miniwdl also writes an
# envelope-less outputs.json inside every nested sub-workflow directory, and (measured, on
# the first real cohort run) one of those can carry a newer mtime than the run root's, so
# a recursive glob sorted by mtime picks it and dies on the missing "outputs" key.
outputs_path = max(RUN_DIR.glob("*/outputs.json"), key=lambda p: p.stat().st_mtime)
# miniwdl's outputs.json file is the bare name -> value mapping; the {"dir", "outputs"}
# envelope exists only on the CLI's *stdout*, measured on the first run that ever reached
# this cell. Accept both, like persist_outputs.py.
report = json.loads(outputs_path.read_text())
outputs = report.get("outputs", report)


def quast_key(name):
    """Map a QUAST metric's display name onto its key in a `quast_summary` map.

    SummarizeQuastReport builds that map by running `sed 's/ /_/g'` and `s/>=/gt/`
    over QUAST's space-aligned report.txt, so "Genome fraction (%)" arrives as
    "Genome_fraction_(%)" and only single-word metrics (N50, NGA50) survive intact.
    Looking the display names up directly matched four keys out of nine and left the
    assert below impossible to satisfy, whatever QUAST had produced.
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

The QUAST numbers above summarize contiguity at two points; the whole curve shows more. An Nx curve
reads "contigs of at least this length cover x% of the assembly". NGx changes the denominator to
the reference length and nothing else, so the two curves separate wherever the assembly's total
length differs from the reference's: above when the assembly is longer, which it is here because
the region slice overhangs its boundaries, below when shorter. Neither curve is alignment-aware.
Genome fraction and NGA50 in the table are, and they are the ones that know whether any of this
sequence is in the right place.

At `quick` scale the region is 2 Mbp and a handful of contigs make the steps chunky. The result is
real, just small.

Both curves apply QUAST's 500 bp contig floor, so the N50 annotated on the plot is the N50 printed
above. Counting every contig instead, including the sub-500 bp fragments QUAST discards, computes a
different statistic and labels it with the same name.


```python
import bisect

import matplotlib.pyplot as plt

# QUAST's own default. Applying it here too is the difference between this plot's N50
# and the N50 printed in the cell above being the same number: QUAST reports on contigs
# >= 500 bp, so counting everything computes a different statistic and annotates it with
# the same name.
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
    # Solid: Nx, against the assembly's own length. Dashed: NGx, against the reference;
    # the two separate where the assembly stops covering the region.
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

The assemblies are files on shared storage and the cluster that made them is still up with Ray on
it. No new job, no new cluster, no hand-off: the next cell is Ray code in this process, reading the
workflow's declared outputs.

The example is a trio concordance check. HG002 is the son of HG003 and HG004, so a variant called in
the child is usually present in at least one parent. Ray Data reads the three paftools VCFs, parses
them in parallel, and joins them on position.

Three things bound what that number can mean, and they are worth reading before the output:

- **Allele dropout.** Flye collapses haplotypes, so each sample contributes one mosaic haplotype and
  a heterozygous site appears or does not roughly at random. A variant the child really inherited
  can be absent from the parent that transmitted it. On allele sampling alone, something like 25-40%
  "in neither parent" is the expectation, not a defect.
- **Representation.** The comparison keys on `(chrom, pos, ref, alt)`, and paftools places an indel
  wherever the alignment's `cs` tag put it. Inside a homopolymer that position is arbitrary, so two
  independently aligned assemblies can spell one indel two ways. `bcftools norm -f <ref> -m -any`
  before comparing removes most of it; the cell below does that.
- **Callable regions differ.** `paftools call -L` calls variants only inside alignment blocks above
  its threshold, and the three assemblies do not fragment identically, so a call sitting in a block
  one sample cleared and another did not counts as unshared for mechanical reasons.

So this is a consistency check on three independent assemblies, not a Mendelian violation rate. A
real one needs diploid callsets, a benchmark region list, and coordinates that are not slice-local.
What it does show is that the three outputs agree the way three related genomes should, and that
checking took one cell on the cluster that produced them.


```python
import os

import ray

# Ray is already running: the WDL backend has been submitting tasks to this cluster for
# the whole notebook. Nothing new is provisioned here.
vcfs = dict(zip(names, outputs["ONTAssembleCohort.vcfs"]))

# Normalize before comparing. paftools derives each variant from minimap2's `cs` tag, which
# places an indel wherever the alignment happened to put it; inside a homopolymer that
# position is arbitrary, so two independently aligned assemblies spell one indel two ways
# and the set intersection below counts it as two variants. `-m -any` also splits
# multiallelic records so a site with two ALTs compares allele by allele.
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

`/mnt/cluster_storage` is shared across the nodes of *one* cluster and is deleted when that cluster
terminates. In a workspace that is fine, since the cluster is yours and stays up. As an Anyscale
Job it is a trap, because a job terminates its cluster on success, so a run that finishes correctly
destroys its own results. A 14h44m chromosome-20 assembly completed, reported a 33.3 Mbp N50, and
left nothing behind but the driver log.

[`persist_outputs.py`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/persist_outputs.py)
copies the *declared outputs only*, leaving the run tree and its gigabytes of intermediates behind,
to `WDL_ON_RAY_RESULTS` or to the first writable durable mount (`/mnt/user_storage`, then
`/mnt/shared_storage`).


```python
run(["python", str(TEMPLATE_DIR / "persist_outputs.py"), "--outputs", str(outputs_path)])
```

## Run it as a job

The notebook path above is sized to run while you watch it. A real assembly is a batch job, and
[`job.yaml`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/job.yaml)
is that job: one sample, all 64 Mbp of chromosome 20, at full coverage.

```bash
anyscale job submit --config-file job.yaml
```

### Time and cost estimates

Representative, from a single HG002 chr20 run on the compute config in `job.yaml`. Read the caveat
below before quoting any of it.

| | |
|---|---|
| Wall clock | ~14h 44m, of which the assembly finished at 4h53m and the rest was polishing |
| Contigs | 44, totalling 62,097,887 bp |
| N50 | 33,279,582 (L50 = 1) |
| N90 | 23,650,082 (L90 = 2) |
| GC | 44.00% against the reference's 43.80% |
| Genome fraction, NGA50, misassemblies | not captured on this run |
| Worker node-hours | ~15, one `m5.8xlarge` plus an `m5.2xlarge` head |
| Cost | ~$30 at us-east-1 on-demand list, roughly $23 worker and $6 head |

**This run does not reproduce under today's defaults.** It executed upstream's bare invocation,
`flye --nano-raw <reads> --threads 30`, with no coverage cap and no genome size. Today's inputs
derive `--nano-hq --iterations 1 --asm-coverage 40 --genome-size`, and `--nano-hq` is a different
error model, so treat the wall clock as an order of magnitude and read each run's `flye_params`
output for the flags that produced it. It also reports contiguity only: N50 says how long the
pieces are and nothing about whether they are right, and this run predates the `Genome fraction`
assertion in Step 6 that exists to stop contiguity standing in for a complete evaluation.

Node-hours are the durable unit; instance pricing moves and varies by region and commitment. Spot
suits every task here except the assembly itself, which does not checkpoint.

The cohort is the more interesting comparison, and it has been measured rather than projected. At
`quick` scale on this template's compute config, three samples took 8m05s of workflow time against
7m52s for the slowest of them alone: three assemblies for the wall clock of one, on three workers,
for the same node-hours as running them one after another. The saving is latency, not compute.
Against a per-task-VM backend the node-hours come out the same and what changes is the boot latency
and idle tail on each of the 30 tasks, paid 30 times instead of never. That result depends on
getting the third worker: capped at two, the same run takes 13m16s, because one assembly waits
7m37s for a node.

`L90 = 2` is the line worth reading. Two contigs, 33.3 Mbp and 23.7 Mbp, hold 90% of the assembly:
one per chromosome arm, 98% of the q arm's 33.9 Mbp of assemblable sequence and 90% of the p arm's
26.3 Mbp. Both stop at the centromere. The remaining 5.2 Mbp sits in 42 contigs averaging 123 kb,
about the size of the pericentromeric sequence they came from.

![The assembly against chromosome 20, one contig per arm stopping at the centromere, and the N50-against-NGA50 gap measured on a separate unpolished run](https://raw.githubusercontent.com/anyscale/templates/main/templates/wdl-genomics-on-ray/assets/chr20-contigs.png)

The figure's second panel is a *different* run, and neither panel is today's command line. It
assembled with `--nano-raw --iterations 0 --asm-coverage 30 --genome-size 64444167` to measure what
skipping polishing costs. Contiguity barely moved (N50 33.25 Mbp) while NGA50 came out at 2.1 Mbp
and genome fraction at 94.9%: consensus error dense enough to break QUAST's alignments turns one
33 Mbp contig into blocks with a 2.1 Mbp median. No polished counterpart was ever measured, so the
size of that effect is not established by this data, only that the two statistics disagree sharply
on an unpolished assembly. The workflow now emits both arms' QUAST columns from a single run, so
the paired table is one `medaka_rounds > 0` run away. `PIPELINE.md` has the rest.

### What that job encodes

`timeout_s: 86400`. An earlier attempt at 12h was SIGTERMed at 12h01m, inside polishing.

`WDL_ON_RAY_RESULTS`. Without it the outputs die with the cluster, as Step 7 explains.

`--call-cache`, and `max_retries: 0` despite it. A job-level retry runs on a new cluster, and this
job's cache lives on `/mnt/cluster_storage`, which the platform recreates per job; miniwdl
invalidates a cache entry whose output files have gone, so the retry would find nothing to reuse.
The entrypoint carries the durable arrangement that does survive, and what it costs.

`preemptible_tries` is the WDL's, per task. `CallAssemblyVariants` sets 3; `Flye.Assemble`, the
14-hour task, sets 0 deliberately, because Flye does not checkpoint and a retry buys another
full-length attempt rather than a cheap recovery. Check what your own pipeline declares before
relying on spot.

## Adding a tool

Whichever container mode you pick, the tools have to come from somewhere.
[`tools/manifest.toml`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/tools/manifest.toml)
pins each one's version, URL and sha256, and three consumers read it: the image build, the wheel
build, and `native` mode's per-task environments.

Under `none`, the notebook's mode, every tool has to be on every node before the first task
dispatches, so the manifest builds one image containing all of them:

```bash
anyscale image build -n my-wdl-tools --containerfile Dockerfile
```

Add a `[tools.<name>]` block and rebuild. The fetch, the checksum check and the PATH shim are
generic; only `kind = "source"` needs a build recipe.
[`tools/BUILDING.md`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/tools/BUILDING.md)
has the procedure.

Past the point where one image stops being reasonable, `--container-runtime ray` stops being
optional: build a small image per task class from the cluster's base, map each `runtime.docker`
value in `[ray] task_image_map`, and check the map in next to the WDL. Run `wdl-on-ray probe-image`
against each one before depending on it.

The same manifest also builds pip wheels (`tools/build_wheels.sh`) for a stock `anyscale/ray`
image. Read `BUILDING.md` before choosing that route. The wheels alone are not enough: a job's
`requirements:` is installed at cluster startup *before* the working directory is staged, and pip
fetches an `https://` index anonymously, so neither a wheel shipped with the submission nor one in
a private bucket is reachable. They have to be staged to `/mnt/user_storage` first. The custom
image exists to avoid that.

## Next steps

[`wdl_on_ray/backend.py`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/wdl_on_ray/backend.py)
is the backend, and where the miniwdl contract is written down.
[`wdl_on_ray/resources.py`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/wdl_on_ray/resources.py)
maps `runtime {}` onto Ray resources, and is where a scheduling hint of your own would go.
[`wdl_on_ray/runtimes.py`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/wdl_on_ray/runtimes.py)
holds the container modes, including `ray`'s per-task images.
[`tools/BUILDING.md`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/tools/BUILDING.md)
covers getting your pipeline's tools onto the cluster, which takes the longest on a real port.
