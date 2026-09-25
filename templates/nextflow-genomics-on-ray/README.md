# Run Nextflow genomics pipelines on Ray

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/nextflow-genomics-on-ray"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/nextflow-genomics-on-ray" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: 13m22s for the `standard` pipeline, the default, with the GPU leg on; about 5 minutes at `quick` (4m47s and 4m52s). Prod, 2026-09-25; see Step 1

A Nextflow executor, `executor 'ray'`, that runs each process as a
[Ray task](https://docs.ray.io/en/latest/ray-core/tasks.html) on an autoscaling cluster. The pipeline
it runs here is GATK Best Practices germline short-variant calling, shaped the way
[nf-core/sarek](https://nf-co.re/sarek) shapes it: Illumina reads from the GIAB Ashkenazi trio
(HG002, HG003, HG004) over a region of chromosome 20, aligned, recalibrated, called per sample and
interval, joint-genotyped, and scored against each sample's GIAB v4.2.1 truth set with
`rtg vcfeval`. A GPU process scores the callset with a DNA language model on the same cluster.

[Nextflow](https://www.nextflow.io/) is how most of the field writes pipelines; nf-core alone
curates over a hundred. It already knows how to hand a process to a batch scheduler: write
directives into a job script, submit it with a command, poll a queue, cancel by id. That is the
seam this executor uses. A Nextflow plugin (`nf-ray-plugin/`) subclasses `AbstractGridExecutor`, the
class the SLURM and PBS executors extend, and writes `#RAY` directives where SLURM writes
`#SBATCH`. It submits with `nf-ray submit`, and a long-lived Python daemon (`nf_ray/`) holds the Ray
connection and owns every task. Nothing in `main.nf` or its modules knows it is running on Ray;
`-profile ray` is the diff.

`cpus 6`, `memory 36.GB` and `accelerator 1, type: 'nvidia-l4'` on a process become `num_cpus=6`,
`memory=36<<30`, `num_gpus=1, accelerator_type='L4'` on the Ray task. The scheduler places it, the
autoscaler adds a node when nothing fits, and nothing is provisioned per task.

### What was ported

The processes are adapted from [nf-core/modules](https://github.com/nf-core/modules) (MIT);
[`pipeline/PIPELINE.md`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/pipeline/PIPELINE.md) has the per-process table and every
divergence. Three of them change what a run produces, and matter if you compare against sarek:

- **No `container` directives.** There is no container runtime inside a Ray worker to nest into,
  so the tools come from this template's image instead of per-process biocontainers.
- **Hard filters, not VQSR.** One chromosome is too few variants to train VQSR's model.
- **One known-sites resource** (dbSNP 138) for BQSR, not three.

## Where this fits

Existing Nextflow, elastic compute, no rewrite. Pipelines whose processes ask for an order of
magnitude different resources, where a node per process wastes money and a fixed HPC queue wastes
time. Cohorts: three samples over 24 intervals is 72 HaplotypeCaller tasks with nothing serialising
them but the cluster's size, and 500 samples is the same shape wider.

It is also the case Ray is good at that a batch scheduler is not: the callset is on shared storage
at the end, and Step 7 runs Ray Data over it on the same cluster and the same GPUs, with no job
hand-off in between.

## Scope

What the executor does with a process's directives and the pipeline's files, and what it leaves alone:

| | |
|---|---|
| `cpus`, `memory` | become `num_cpus` and `memory` on the Ray task |
| `accelerator` | `num_gpus`, plus `accelerator_type` from the GCE-style name (`nvidia-l4` becomes `L4`); `nvidia.com/gpu` means any GPU, and `ray.defaultAccelerator` names one |
| `ext.ray_resources` | custom Ray resources, as JSON |
| `ext.image` | `runtime_env={'image_uri': ...}`, only through an explicit `NF_RAY_IMAGE_MAP` |
| `time` | parsed, recorded, and **not enforced**: Ray has no wall-clock limit |
| `errorStrategy`, `maxRetries` | Nextflow's, unchanged. Ray-level retries are 0, so they cannot bypass them |
| `container` | not honoured; see "Executors" below |
| the pipeline's `bin/` | copied under the work directory when the run starts and made executable, because a worker cannot see the project directory; tasks get the copy on `PATH` |

Two limits worth knowing before you port anything:

- **A task's ceiling is the largest node, and what Ray schedules, not what it has.** Ray 2.58.0
  offers about 70% of a node's free memory as `memory`, so a 128 GiB worker schedules roughly
  85 GiB. `conf/base.config` clamps requests to 80 GB and `conf/ray.config` declares the same
  ceiling to the executor, which rejects anything larger at submit time.
- **An unsatisfiable request would wait forever.** Ray cannot tell "no node this big exists" from
  "the autoscaler has not caught up", which is why the executor refuses a request over the declared
  ceiling rather than submitting it.

## Set-up

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/nextflow-genomics-on-ray
```

## Step 1: Check the cluster and the toolchain

One knob controls how much work this notebook does. `standard` is the default and calls a 10 Mbp
region of chromosome 20 for each of three samples; `quick` does 2 Mbp and is what CI runs; set
`NF_DEMO_SCALE=quick` before launching Jupyter to use it. Only the region size differs: same
tools, same DAG, same resource requests.

Measured on prod, 2026-09-25, on the AWS compute config: an m5.2xlarge head with `CPU: 0`,
r6i.4xlarge CPU workers and g6.2xlarge L4 workers.

- `standard`, with the GPU leg on: the Step 5 pipeline ran its 171 tasks in 13m22s, 4.6 CPU hours
  by Nextflow's count, and Step 7's Ray Data pass took 116 s on two L4s.
- `quick`, in two runs with `NF_ANNOTATE=false` and so no GPU node: 81 tasks in 4m47s and 4m52s,
  1.4 CPU hours each.

The Step 3 smoke run took 36 s in all three runs, its eight shards on two nodes.

`NF_ANNOTATE=false` skips the GPU leg, both the Nextflow process and the Ray Data step in Step 7,
for a cluster with no L4 to give it. CI sets it, so a test run never waits on L4 capacity.

`nf-ray doctor` reports what the executor would decide without running anything: the interpreter
and Ray versions it would hand to workers, whether the work directory is on shared storage, and
whether every tool the pipeline calls is on `PATH`.


```python
import csv
import json
import os
import pathlib
import subprocess

# `standard` is what a reader gets; CI sets `quick`. Only the size of the region differs, so a
# green `quick` run and a `standard` run exercise the same code path.
SCALE = os.getenv("NF_DEMO_SCALE", "standard")
SCALES = {
    "quick":    {"region": "chr20:1000000-3000000",  "span": "2 Mbp"},
    "standard": {"region": "chr20:1000000-11000000", "span": "10 Mbp"},
}
if SCALE not in SCALES:
    raise ValueError(f"NF_DEMO_SCALE must be one of {sorted(SCALES)}, got {SCALE!r}")
CFG = SCALES[SCALE]
ANNOTATE = os.getenv("NF_ANNOTATE", "true").lower() in ("1", "true", "yes")

# The GIAB Ashkenazi trio: son, father, mother. Three samples because joint genotyping is the
# point at which the per-sample fan-out converges, and one sample has nothing to converge.
SAMPLES = ["HG002", "HG003", "HG004"]

TEMPLATE_DIR = pathlib.Path.cwd()
PIPELINE = TEMPLATE_DIR / "pipeline"
DATA_URI = f"s3://anyscale-public-materials/genomics/giab-trio-illumina-chr20/{SCALE}"

# Every node sees /mnt/cluster_storage, and the executor requires that: Nextflow stages a task's
# inputs by path, so the next task's node has to read what the last one wrote.
# conf/ray.config puts the work directory at /mnt/cluster_storage/nf-work for the same reason.
WORK = pathlib.Path("/mnt/cluster_storage/nf-genomics")
NF_WORK = "/mnt/cluster_storage/nf-work"
DATA_DIR, RESULTS, SMOKE = WORK / "data" / SCALE, WORK / "results" / SCALE, WORK / "smoke"
for d in (DATA_DIR, RESULTS, SMOKE):
    d.mkdir(parents=True, exist_ok=True)


def run(cmd, **kwargs):
    """Run a command, streaming output, and raise if it fails.

    Deliberately not the `!` shell magic: `!` does not raise on a non-zero exit, so a failed
    pipeline would leave this notebook green and the CI test passing.
    """
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


print(f"scale       {SCALE}  ({CFG['span']} of {CFG['region']})")
print(f"samples     {', '.join(SAMPLES)}")
print(f"annotate    {ANNOTATE}")
print(f"data        {DATA_URI}\n")
run(["nf-ray", "doctor", "--work-dir", NF_WORK])
```

## Step 2: Read the pipeline you're about to run

`main.nf` is an ordinary DSL2 pipeline, and its processes carry nf-core's resource labels,
unchanged: `process_medium` is 6 CPUs and 36 GB here as it is on any nf-core pipeline. The labels
live in `conf/base.config`; everything Ray-specific lives in `conf/ray.config`, which only the
`ray` profile includes.

The cell prints the configuration Nextflow resolves under `-profile ray`, filtered to the parts
that decide scheduling: the label requests, the ceiling they are clamped to, and the executor's
settings.


```python
resolved = subprocess.run(
    ["nextflow", "config", "-flat", "-profile", "ray", str(PIPELINE)],
    check=True, capture_output=True, text=True,
).stdout.splitlines()

# `-flat` quotes selector keys, process.'withLabel:process_medium'.cpus, so that prefix
# carries the quote. Each prefix must match: an empty section would read as "no label requests".
for prefix in ("process.executor", "process.resourceLimits", "process.'withLabel:process_",
               "executor.", "ray.", "workDir"):
    shown = [line for line in resolved
             if line.startswith(prefix) and "errorStrategy" not in line and "time" not in line]
    assert shown, f"`nextflow config -flat` printed nothing under {prefix!r}"
    print("\n".join(shown))
```

### Executors

**Why a grid executor.** Nextflow's engine is a JVM, and Ray does ship a Java API, but it is
documented as experimental and community-supported, its version must match Ray Python exactly, and
`ray-runtime` loads a JNI library that would have to initialise inside Nextflow's plugin
classloader. `AbstractGridExecutor`'s contract (write a header, run a submit command, parse a job
id, poll a status command) has been stable for a decade, and it keeps Ray on the Python side of a
`fork`/`exec`. [`RayExecutor.groovy`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/nf-ray-plugin/src/main/groovy/ai/anyscale/nfray/RayExecutor.groovy)
is under three hundred lines.

**Why a daemon.** A Ray task belongs to the process that submitted it and is cancelled when that
process exits, and `nf-ray submit` lives for milliseconds. So the first submit starts a daemon that
holds the Ray connection for the whole run, and every `nf-ray` call talks to it over a unix socket
on the head node. The daemon also writes `.exitcode` when Ray fails *around* a task (a lost node, a
memory-monitor kill), with a code nf-core's `errorStrategy` already retries, instead of leaving
Nextflow to time out on a file the task never wrote.

**Containers.** Every nf-core module declares a biocontainer, and a Ray worker is already a
container with no runtime to nest another in. This template's modules drop the directive and the
image carries the tools, pinned in [`tools/env.main.yml`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/tools/env.main.yml). For a pipeline
you do not want to edit there are two routes, neither exercised by this notebook: `-profile conda`,
which has Nextflow build each process's declared environment on shared storage, and `ext.image`
mapped through `NF_RAY_IMAGE_MAP` to an image rebuilt on this cluster's base, since Ray refuses a
task image whose Ray and Python do not match the cluster's. `nf-ray probe-image <uri>` checks a
candidate before a pipeline depends on it.

## Step 3: Run a 60-second pipeline first

`smoke.nf` fans out a few shards and gathers them. It calls no genomics tool and reads no data, so
when it fails the failure is the executor's, and that is worth knowing before an hour of alignment
depends on the answer. The gather runs `pipeline/bin/smoke_collect.sh`, so it also checks that the
pipeline's `bin/` reaches a worker, which only the executor's copy on shared storage makes true.

It is also an equivalence oracle. Shard *i* sums the thousand integers from `i*1000+1`, so its
checksum is `1,000,000*i + 500,500` whatever node ran it; a changed number means the environment
changed under the task. The cell checks that, then reads the executor's placement record, which
the daemon writes as each task finishes: one row per Ray task, with the node it ran on.


```python
SHARDS = 8
run(["nextflow", "run", PIPELINE / "smoke.nf", "-profile", "ray",
     "--outdir", SMOKE, "--shards", SHARDS, "--hold", 8])

checksums = [int(x) for x in (SMOKE / "smoke" / "checksums.txt").read_text().split()]
assert checksums == [1_000_000 * i + 500_500 for i in range(SHARDS)], checksums
print(f"\n{SHARDS} shard checksums match the closed form")

with open(pathlib.Path(NF_WORK) / "nf_ray_placement.tsv") as handle:
    placement = list(csv.DictReader(handle, delimiter="\t"))
shard_rows = [row for row in placement if "SHARD" in row["name"]][-SHARDS:]
nodes = {row["node_id"] for row in shard_rows}
print(f"the last {len(shard_rows)} SHARD tasks ran on {len(nodes)} node(s)")
```

## Step 4: Stage the reads

The reads are the GIAB Ashkenazi trio from GIAB's NHGRI Illumina 300x GRCh38 alignments, sliced to
the region and subsampled by read name to about 30x, which keeps mates together.
[`tools/stage-demo-data.sh`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/tools/stage-demo-data.sh)
derives every file from public sources, so the derivation is reproducible: the reads, all of chr20
from the GRCh38 no-alt analysis set, dbSNP 138 over chr20 for BQSR, and each sample's own GIAB
v4.2.1 benchmark VCF and high-confidence BED.

**These reads are reference-selected.** A pair is in the slice because it aligned to the region, so
reads that would mismap *into* chr20 from elsewhere in the genome are absent by construction, and
precision below reads somewhat high against a whole-genome run. Treat the table in Step 6 as a
check that the pipeline is right, not as a benchmark of GATK.

Each scale ships a `MANIFEST.json` with every file's checksum. `make_samplesheet.py` builds the
pipeline's samplesheet from it, verifies the checksums, and gives each sample its own truth set.


```python
# --no-sign-request because the bucket is public: a *signed* request is evaluated against the
# node role's policy, so signing can fail where anonymous access succeeds.
run(["aws", "s3", "sync", "--no-sign-request", "--only-show-errors", f"{DATA_URI}/", DATA_DIR])

SAMPLESHEET = DATA_DIR / "samplesheet.csv"
run(["python", PIPELINE / "bin" / "make_samplesheet.py",
     "--data-dir", DATA_DIR, "--output", SAMPLESHEET])

manifest = json.loads((DATA_DIR / "MANIFEST.json").read_text())
print(f"\n{manifest['region']}   {manifest['platform']}   {manifest['truth_version']}")
print(f"{'sample':<8}{'read pairs':>12}")
for entry in manifest["samples"]:
    print(f"{entry['id']:<8}{entry['read_pairs']:>12,}")
REFERENCE = DATA_DIR / manifest["reference"]["fasta"]
KNOWN_SITES = DATA_DIR / manifest["known_sites"]["vcf"]
```

## Step 5: Call variants across the trio

One command. The pipeline aligns each sample, marks duplicates and recalibrates, then splits the
region into intervals and runs HaplotypeCaller on every sample-interval pair at once: at `standard`,
3 samples x 24 intervals = 72 independent tasks. Each interval's three GVCFs then meet for joint
genotyping, the intervals are merged and hard-filtered, and the joint callset is split back out per
sample and scored.

That fan-out is what gives the autoscaler something to do. HaplotypeCaller asks for 6 CPUs and
36 GB (`process_medium`), so two fit a 16-vCPU, ~85 GiB-schedulable worker, and the compute config
lets the worker group grow to four. The GPU process asks for one L4 and lands on the GPU group,
which starts from zero when the DAG reaches it.

Measured on prod, 2026-09-25, AWS config. At `standard` the chart below drew 170 tasks on 6 nodes,
13.1 minutes from first submit to last finish, with the GPU tasks on 2 nodes of their own. Getting
those two L4s took three `InsufficientInstanceCapacity` errors for g6.2xlarge, over 19 seconds,
before both launched. At `quick`, with the GPU leg off, it drew 80 tasks on 4 nodes in 4.5 minutes.

`--scale` picks the region and interval count from `main.nf`'s presets, which match what the
staging script published. The `ray` profile adds the executor, the plugin, the shared work
directory and the declared node ceiling; nothing else changes.


```python
run(["nextflow", "run", PIPELINE / "main.nf", "-profile", "ray",
     "--scale", SCALE,
     "--samplesheet", SAMPLESHEET,
     "--reference", REFERENCE,
     "--known_sites", KNOWN_SITES,
     "--annotate", str(ANNOTATE).lower(),
     "--outdir", RESULTS])
```

### How the run used the cluster

Every process above ran as one Ray task, and the executor recorded each: when Nextflow submitted
it, when it started holding resources on a worker, when it finished, and on which node. Nextflow's
own `trace.txt` names each task, so joining the two draws the run with no instrumentation added.

The join is on the task's work directory, which both files record, and not on the Ray task id.
Each run starts its own executor daemon and numbers its tasks from 1, and the placement record
accumulates in the shared work directory across runs, so the smoke run's task 3 and this run's
task 3 are both in it.

Colour is the node. A grey lead-in is time spent queued: waiting for a node with room, which on
an autoscaling cluster includes waiting for one to boot. The cell checks the join is complete, since
a trace row with no placement row would mean the executor lost track of a task. The one exception
is `COLLECT_PLACEMENT`, the task that copies the record into the results: it is still running when
it copies, so its own row is not in the copy.


```python
import matplotlib.pyplot as plt

with open(RESULTS / "pipeline_info" / "trace.txt") as handle:
    trace = {row["native_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
with open(RESULTS / "pipeline_info" / "nf_ray_placement.tsv") as handle:
    placed = {row["work_dir"]: row for row in csv.DictReader(handle, delimiter="\t")}

# Keyed on work dir, so earlier runs' rows in the same record cannot match this run's tasks.
#
# One task is exempt: COLLECT_PLACEMENT, which copied this record into the results. The daemon
# appends a task's row when the task finishes, and that one was still running when it made the
# copy, so the copy cannot hold its own row. Every other traced task must have one.
COLLECTOR = "COLLECT_PLACEMENT"
tasks = [t for t in trace.values() if t["process"] != COLLECTOR]
assert len(trace) - len(tasks) == 1, f"expected exactly one {COLLECTOR} task in the trace"
missing = [t["name"] for t in tasks if t["workdir"] not in placed]
assert not missing, f"{len(missing)} traced task(s) have no placement row, e.g. {missing[:3]}"
rows = [(placed[t["workdir"]], t) for t in tasks]
# A task that never started records `started` as 0.000, so it sorts by its submit time.
rows.sort(key=lambda pair: float(pair[0]["started"]) or float(pair[0]["submitted"]))

t0 = min(float(p["submitted"]) for p, _ in rows)
node_ids = sorted({p["node_id"] for p, _ in rows})
colours = {n: plt.cm.tab10(i % 10) for i, n in enumerate(node_ids)}

fig, ax = plt.subplots(figsize=(12, max(4, len(rows) * 0.12)))
for y, (p, _t) in enumerate(rows):
    submitted, finished = float(p["submitted"]) - t0, float(p["finished"]) - t0
    started = float(p["started"]) - t0 if float(p["started"]) else submitted
    ax.barh(y, started - submitted, left=submitted, color="0.85", height=0.8)
    ax.barh(y, finished - started, left=started, color=colours[p["node_id"]], height=0.8)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([t["name"] for _, t in rows], fontsize=5)
ax.invert_yaxis()
ax.set_xlabel("seconds since the first submit")
ax.set_title(f"{len(rows)} tasks on {len(node_ids)} Ray node(s); grey is queued, "
             "colour is the node")
plt.tight_layout()
plt.show()

span = max(float(p["finished"]) for p, _ in rows) - t0
print(f"{len(rows)} tasks, {len(node_ids)} node(s), "
      f"{span / 60:.1f} min from first submit to last finish")
if ANNOTATE:
    gpu_nodes = {p["node_id"] for p, t in rows if t["process"].endswith("ANNOTATE_VARIANTS")}
    cpu_nodes = {p["node_id"] for p, t in rows if t["process"].endswith("HAPLOTYPECALLER")}
    # A GPU task can only have landed on the GPU group, so the two sets cannot overlap.
    assert gpu_nodes and not gpu_nodes & cpu_nodes, (gpu_nodes, cpu_nodes)
    print(f"GPU tasks ran on {len(gpu_nodes)} node(s) of their own")
```

## Step 6: Score the callset

`rtg vcfeval` compares each sample's calls with that sample's GIAB v4.2.1 truth set, matching
variants by the haplotypes they imply rather than by position, so a left-aligned indel and its
right-aligned twin count as the same call. SNPs and indels are scored separately because they fail
for different reasons, and a combined F1 would hide which one moved. The unthresholded row is the
one reported: a threshold chosen to maximise F-measure against the truth set being scored is
fitted to the answer.

Read the numbers with the bounds they come with: one chromosome, hard filters instead of VQSR, one
known-sites resource, reference-selected reads (Step 4), and scoring restricted to the region ∩ the
high-confidence BED. They are not comparable to a published genome-wide benchmark.

Measured on prod, 2026-09-25, on the AWS compute config, the cell below printed these. The reads
are reference-selected, so precision reads somewhat high against a whole-genome run;
[`PIPELINE.md`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/pipeline/PIPELINE.md#bounds-on-the-numbers)
has the bounds.

`standard`, chr20:1,000,000-11,000,000:

| sample | SNP precision | SNP recall | SNP F1 | indel precision | indel recall | indel F1 |
|---|---|---|---|---|---|---|
| HG002 | 0.9965 | 0.9904 | 0.9934 | 0.9590 | 0.9414 | 0.9501 |
| HG003 | 0.9972 | 0.9914 | 0.9943 | 0.9498 | 0.9326 | 0.9411 |
| HG004 | 0.9980 | 0.9884 | 0.9932 | 0.9563 | 0.9439 | 0.9501 |

`quick`, chr20:1,000,000-3,000,000:

| sample | SNP precision | SNP recall | SNP F1 | indel precision | indel recall | indel F1 |
|---|---|---|---|---|---|---|
| HG002 | 0.9966 | 0.9872 | 0.9919 | 0.9561 | 0.9356 | 0.9457 |
| HG003 | 0.9989 | 0.9872 | 0.9930 | 0.9548 | 0.9337 | 0.9441 |
| HG004 | 0.9984 | 0.9707 | 0.9843 | 0.9591 | 0.9437 | 0.9514 |


```python
import pandas as pd

bench = pd.read_csv(RESULTS / "benchmark" / "benchmark.tsv", sep="\t")
# Every sample x type, or the table is quietly missing a comparison.
assert len(bench) == len(SAMPLES) * 2, bench
assert set(bench["sample"]) == set(SAMPLES), bench["sample"].unique()
assert set(bench["caller"]) == {"gatk"}, bench["caller"].unique()

table = bench.pivot_table(index="sample", columns="variant_type",
                          values=["precision", "recall", "f1"])
print(table.round(4).to_string())
```

## Step 7: Annotate at scale with Ray Data, on the same GPUs

`ANNOTATE_VARIANTS` scored the callset in shards, one Nextflow process per shard. The next cell
runs the *same* scorer, the `VariantScorer` class in `pipeline/bin/score_variants.py`, over the
whole callset as one Ray Data job: the same cluster, the same L4s, the same files on shared storage,
no hand-off. One implementation seen two ways, so the two result tables should agree.

Measured at `standard` (prod, 2026-09-25, AWS config, g6.2xlarge L4s): `ANNOTATE_VARIANTS` ran as
8 shards, and the cell below scored the same 25,197 PASS variants with two GPU actors in 116 s. The
two tables' embedding distances agreed to within 2.88e-05.

**What the score is, and is not.** For each variant a nucleotide language model
(`InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`, baked into the image at a pinned
revision) embeds the reference and alternate sequence around it, and the score is the distance
between the two embeddings. That correlates with "this changes the sequence in a way the model
noticed". It is not a pathogenicity score and is not comparable to SpliceAI or CADD.


```python
if not ANNOTATE:
    print("NF_ANNOTATE is off; skipping the GPU leg")
else:
    import sys

    import ray

    sys.path.insert(0, str(PIPELINE / "bin"))
    import score_variants as sv

    # By value, so workers need only what the image already has (torch, transformers) and
    # not this file on their import path.
    ray.cloudpickle.register_pickle_by_value(sv)

    # The PASS set, as SHARD_VCF selected it, written where every node can read it: the head
    # offers CPU: 0, so nothing below runs on this node.
    pass_vcf = WORK / "annotation" / "joint.pass.vcf"
    pass_vcf.parent.mkdir(parents=True, exist_ok=True)
    run(["bcftools", "view", "-f", "PASS,.", "-o", pass_vcf,
         RESULTS / "variants" / "joint.filtered.vcf.gz"])
    rows = [{"chrom": v.chrom, "pos": v.pos, "ref": v.ref, "alt": v.alt}
            for v in sv.read_vcf(str(pass_vcf))]

    scored = ray.data.from_items(rows).map_batches(
        sv.VariantScorer,
        fn_constructor_kwargs={"reference": str(REFERENCE)},
        batch_size=64, batch_format="numpy", num_gpus=1, concurrency=2,
    ).to_pandas()

    nextflow = pd.read_csv(RESULTS / "annotation" / "variant_scores.tsv", sep="\t")
    keys = ["chrom", "pos", "ref", "alt"]
    both = scored.merge(nextflow, on=keys, suffixes=("_raydata", "_nextflow"))
    assert len(both) == len(scored) == len(nextflow), (len(both), len(scored), len(nextflow))
    diff = (both["embedding_distance_raydata"] - both["embedding_distance_nextflow"]).abs()
    print(f"{len(scored):,} variants scored both ways; largest difference {diff.max():.2e}")
    top = scored.nlargest(5, "embedding_distance")[keys + ["embedding_distance"]]
    print(top.to_string(index=False))
```

## Step 8: Persist the outputs

`/mnt/cluster_storage` is shared across the nodes of *one* cluster and is deleted when that cluster
terminates. In a workspace that is fine, since the cluster is yours and stays up. As an Anyscale
Job it is a trap: a job terminates its cluster on success, so a run that finishes correctly
destroys its own results.

Every process that produces a result declares a `publishDir`, so `--outdir` already holds the
declared outputs and none of the intermediates.
[`persist_outputs.py`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/persist_outputs.py)
copies that tree to `NF_RAY_RESULTS`, or to the first writable durable mount (`/mnt/user_storage`,
then `/mnt/shared_storage`).


```python
run(["python", TEMPLATE_DIR / "persist_outputs.py", "--results", RESULTS])
```

## Run it as a job

[`job.yaml`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/job.yaml) is the same pipeline over all of chromosome 20 (`--scale full`: 48
intervals, 144 calling tasks), staged, run and persisted in one entrypoint:

```bash
anyscale job submit --config-file job.yaml
```

It has not been run, so there are no time or cost figures here yet. `max_retries` is 0 because a
job-level retry starts on a new cluster with an empty work directory; retries belong inside the
run, per task, where nf-ray's exit codes put them.

## Next steps

### Point it at your own reads

The samplesheet is the whole change: `sample,fastq_1,fastq_2`, plus `truth_vcf,truth_bed` for a
sample you want scored. A row without truth columns is called and not scored. Three things bite:

- **The reference sets the contig names.** Truth sets, known sites and the reference must agree on
  `chr20` against `20`; `collect_vcfeval.py` says so by name when they do not.
- **`--region` and `--intervals` override the scale presets**, and a region the reads do not cover
  produces an empty table rather than an error at the start.
- **The ceiling must fit a node you have.** `conf/base.config`, `conf/ray.config` and the compute
  config are one decision; `tests/nextflow-genomics-on-ray/test_config_agreement.py` checks them.

### Run your own pipeline

Add the `ray` profile from [`nextflow.config`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/pipeline/nextflow.config) and include
[`conf/ray.config`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/pipeline/conf/ray.config), pinning the plugin as `nf-ray@0.1.0`: without a
version Nextflow looks for nf-ray in the plugin registry, where it is not. Run `smoke.nf` first,
then `nf-ray doctor`. This has been exercised with this template's pipeline only; an nf-core
pipeline's biocontainer directives need one of the two routes under "Executors".

### Read the executor

[`RayExecutor.groovy`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/nf-ray-plugin/src/main/groovy/ai/anyscale/nfray/RayExecutor.groovy) is
the whole Nextflow side; [`nf_ray/resources.py`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/nf_ray/resources.py) maps directives onto Ray
resources; [`nf_ray/daemon.py`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/nf_ray/daemon.py) owns the tasks; and
[`nf_ray/errors.py`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/nf_ray/errors.py) is the table of Ray failures and the exit codes they
become.
