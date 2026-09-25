# Run Nextflow genomics pipelines on Ray

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/nextflow-genomics-on-ray"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/nextflow-genomics-on-ray" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: 20 min at the default `standard` scale, 10 min at `quick`

This template runs a Nextflow pipeline through a Ray executor, `executor 'ray'`, on an Anyscale
cluster that adds CPU and GPU nodes as the DAG needs them. The pipeline is GATK germline
short-variant calling in the shape of [nf-core/sarek](https://nf-co.re/sarek): fastp, BWA-MEM2,
MarkDuplicates and BQSR per sample, HaplotypeCaller per sample and interval, GenomicsDBImport and
GenotypeGVCFs per interval, hard filtering, and `rtg vcfeval` against GIAB v4.2.1. It runs on the
three GIAB Ashkenazi trio samples (HG002, HG003, HG004) over a region of chr20, with one GPU process
in the same DAG.

At the end you have:

- a joint callset, with SNP and indel precision, recall and F1 for each sample;
- a chart of which node ran each task and how long each task waited for one;
- a `ray` profile to add to your own pipeline, and the limits to check before you do.

## Why run it on Ray

Each process's `cpus`, `memory` and `accelerator` directives become a Ray resource request. Ray
places the task on a node with room, and the autoscaler adds a node when none has any, up to the
compute config's `max_nodes`. The `accelerator` directive alone sends a process to the L4 group,
which starts at zero and comes up when the DAG reaches the GPU process. After the pipeline, Step 7
runs a Ray Data job over its outputs on the same L4s.

On an on-premises Slurm cluster a run queues for a fixed partition; this cluster grows and shrinks
with the run. AWS Batch and Seqera Platform scale too; here the CPU and GPU processes share one
cluster, placed by their directives, and the work directory is shared POSIX storage rather than S3.

The executor subclasses Nextflow's `AbstractGridExecutor`, like the SLURM and PBS executors, and
writes `#RAY` directives where SLURM writes `#SBATCH`. Apart from dropping nf-core's `container`
directives, the pipeline has no Ray-specific code; `-profile ray` adds the executor.

## What differs from nf-core/sarek

The processes are adapted from [nf-core/modules](https://github.com/nf-core/modules) (MIT), and
[`pipeline/PIPELINE.md`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/pipeline/PIPELINE.md)
lists every divergence. These change what a run produces:

- Tools come from this template's image, pinned in `tools/env.main.yml`, instead of per-process
  biocontainers.
- The callset is hard-filtered, where sarek's joint-germline mode uses VQSR. GATK recommends VQSR
  for at least one whole genome or about 30 exomes, and a region of chr20 is far smaller.
- The hard filter applies GATK's SNP thresholds to every record. GATK's indel thresholds are looser
  (FS > 200, ReadPosRankSum < -20, no MQ or MQRankSum filter), so indel recall can only be lower
  than GATK's recipe gives.
- BQSR gets one known-sites resource, dbSNP 138, where sarek gives it dbSNP, Mills and a
  known-indels set.

The three samples are a family, but nothing uses the pedigree: they are joint-genotyped without a
pedigree file, nothing checks Mendelian consistency, and each sample is scored against its own truth
set.

## What the executor does with a process

| Directive or file | Under `executor 'ray'` |
|---|---|
| `cpus`, `memory` | `num_cpus` and `memory` on the Ray task, used for placement: Ray does not cap usage. When a node runs out of memory, Ray's memory monitor kills a task there, reported as exit 137 |
| `accelerator` | `num_gpus`, plus `accelerator_type` from the GCE-style name (`nvidia-l4` becomes `L4`). No type, or `nvidia.com/gpu`, means any GPU, and `ray.defaultAccelerator` picks one |
| `ext.ray_resources` | custom Ray resources, as JSON |
| `ext.image` | `runtime_env={'image_uri': ...}`, only through an explicit `NF_RAY_IMAGE_MAP` |
| `time` | recorded and **not enforced**: Ray has no wall-clock limit |
| `errorStrategy`, `maxRetries` | Nextflow's, unchanged. Ray-level retries are 0 |
| `container` | not honoured; see "Containers" below |
| the pipeline's `bin/` | copied under the work directory at the start of the run and made executable, because workers cannot see the project directory |

Two limits to know before you port a pipeline:

- A task's ceiling is what the largest node can schedule, which is less than it has. Ray 2.58.0
  offers about 70% of a node's free memory as `memory`, so a 128 GiB worker schedules roughly 85
  GiB; it took the 72 GB `process_high` tasks in every prod run. `conf/base.config` clamps requests
  to 80 GB, and `conf/ray.config` declares that ceiling to the executor, which refuses anything
  larger at submit time.
- Ray keeps a request that no node type can satisfy pending instead of failing it, so Nextflow would
  show the task queued forever. The declared ceiling is what lets the executor refuse it.

### Containers

A Ray worker is already a container, with no runtime to start another in, so this template's modules
drop nf-core's biocontainer directives and the image carries the tools. A pipeline you don't want to
edit has two routes, neither exercised here: `-profile conda`, which builds each process's declared
environment once on shared storage, or `ext.image` mapped through `NF_RAY_IMAGE_MAP` to an image
rebuilt on this cluster's base. Ray refuses a task image whose Ray and Python versions differ from
the cluster's; `nf-ray probe-image <uri>` checks one.

### The daemon

Ray cancels a task when the process that submitted it exits, so the executor keeps a daemon on the
head node for the length of the run, which owns every task; `nf-ray submit` reaches it over a unix
socket. When Ray fails around a task (a lost node, a memory-monitor kill), the daemon writes the
task's `.exitcode` with a code in nf-core's retry band, so `errorStrategy` retries the task instead
of Nextflow timing out on a file the task never wrote.

## Set-up

If you're not in a workspace created from this template, clone the repository and work from the
template directory:

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/nextflow-genomics-on-ray
```

## Step 1: Check the cluster and the toolchain

`NF_DEMO_SCALE`, set before starting Jupyter, picks how much work the notebook does. The scales run
the same processes with the same tools and resource requests; they differ in region size and in how
many intervals and GPU shards the region is split into.

| Scale | Region | Intervals | Runs as | Measured on prod, 2026-09-25 |
|---|---|---|---|---|
| `quick` | chr20:1,000,000-3,000,000 | 8 | CI, with `NF_ANNOTATE=false` | 81 tasks in 4m47s and 4m52s (two runs), 1.4 CPU h each, on 4 CPU workers |
| `standard` (default) | chr20:1,000,000-11,000,000 | 24 | this notebook | 171 tasks in 13m22s, 4.6 CPU h, on 4 CPU workers and 2 L4 workers |
| `full` | all of chr20 | 48 | `job.yaml` | 299 tasks in 44m07s, 25.1 CPU h, on at most 6 CPU workers and 2 L4 workers at a time |

Times are the pipeline's wall time and CPU hours are Nextflow's count, on the AWS compute config: an
m5.2xlarge head with `CPU: 0`, r6i.4xlarge CPU workers and g6.2xlarge L4 workers.

`NF_ANNOTATE=false` skips the GPU process and Step 7, for a cluster with no L4. CI sets it so that a
test never waits on L4 capacity.

The cell prints the run's settings and runs `nf-ray doctor`, which checks the Python and Ray
versions, that the work directory is on shared storage, and that every tool the pipeline calls is on
`PATH`. Look for `no problems found`. Doctor reads only `NF_RAY_*` environment variables, so its
`declared ceiling` line says `none`; Step 2 shows the ceiling the executor uses.


```python
import csv
import json
import os
import pathlib
import subprocess

# `standard` is what a reader gets; CI sets `quick`. The scales share one code path and differ
# only in region size and how finely it is split, so a green `quick` run covers `standard`.
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

`main.nf` is an ordinary DSL2 pipeline whose processes carry nf-core's resource labels at nf-core's
values: `process_medium` is 6 CPUs and 36 GB. The labels live in `conf/base.config`. Everything
Ray-specific is in `conf/ray.config`, which only the `ray` profile includes.

The cell prints what Nextflow resolves under `-profile ray`, filtered to what decides scheduling:
the label requests, the `resourceLimits` ceiling they are clamped to, the executor's queue settings
and the `ray` scope. `ray.maxNodeMemoryGb` should match the memory in `process.resourceLimits`.


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

## Step 3: Run a one-minute smoke test

`smoke.nf` fans out eight shards and gathers them. It calls no genomics tool and reads no data, so a
failure here belongs to the executor or the cluster, and you find it before any alignment starts.
The gather runs `pipeline/bin/smoke_collect.sh`, so the test also checks that the pipeline's `bin/`
reaches a worker.

Shard *i* sums the integers from `i*1000+1` to `i*1000+1000`, so its checksum is
`1,000,000*i + 500,500` on any node. The cell checks every checksum, then reads the placement record
the daemon appends to as each task finishes, and prints how many nodes the shards ran on. On prod it
took 36 s in each of three runs, with the eight shards on two nodes.


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

The reads come from GIAB's NHGRI Illumina 300x GRCh38 alignments of the three samples: read pairs
whose primary alignment falls in the region, subsampled by read name to about 30x so that mates stay
together. The reference is all of chr20 from the GRCh38 no-alt analysis set, the known sites are
dbSNP 138 over chr20, and each sample has its own GIAB v4.2.1 benchmark VCF and benchmark-regions
BED.
[`tools/stage-demo-data.sh`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/tools/stage-demo-data.sh)
builds every file from public sources. The pairs were chosen by where GIAB's whole-genome alignment
put them and are realigned here against chr20 alone, so they are reference-selected; Step 6 says
what that does to the scores.

Each scale's `MANIFEST.json` lists its files, with checksums for the FASTQs and the reference.
`make_samplesheet.py` builds the samplesheet from it, verifies the FASTQ checksums and gives each
sample its own truth set. The cell prints the region and each sample's read pairs.


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

## Step 5: Call variants for the three samples

The cell runs the whole pipeline. Each sample is aligned, duplicate-marked and recalibrated. The
region is then split into intervals, and HaplotypeCaller is submitted for every sample and interval
at once: 3 samples x 24 intervals = 72 independent tasks at `standard`. Each interval's three GVCFs
go through GenomicsDBImport and GenotypeGVCFs, the intervals are merged and hard-filtered, and the
joint callset is split per sample for scoring.

HaplotypeCaller asks for 6 CPUs and 36 GB (`process_medium`), so two fit on a 16-vCPU worker that
schedules about 85 GiB, and the compute config lets the CPU group grow to four workers: at most
eight calling tasks at a time. The GPU process asks for one L4 and lands on the L4 group, which
starts from zero when the DAG reaches it. On prod at `standard`, g6.2xlarge returned
`InsufficientInstanceCapacity` three times over 19 seconds before both L4 workers launched.

`--scale` takes the region, interval count and GPU shard count from `main.nf`'s presets, which match
the staged data. The `ray` profile adds the executor, the plugin, the shared work directory and the
declared node ceiling.


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

The executor records every task it runs: when Nextflow submitted it, when it started on a worker,
when it finished and on which node. The cell joins that placement record to Nextflow's `trace.txt`
on each task's work directory and draws one bar per task, coloured by node. A grey lead-in is time
spent queued, which on an autoscaling cluster includes waiting for a node to boot. The cell asserts
that every traced task has a placement row and, with the GPU leg on, that no HaplotypeCaller task
ran on a GPU task's node.

On prod the chart drew 170 tasks on 6 nodes at `standard`, 13.1 min from first submit to last
finish, with the GPU tasks on 2 of them; at `quick` it drew 80 tasks on 4 nodes in 4.5 min. It
leaves out `COLLECT_PLACEMENT`, the task that copies the record, which is why it shows one task
fewer than Nextflow counts.


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
    # A GPU task fits only the L4 group, and HaplotypeCaller's 36 GB does not fit an L4 node,
    # so the two sets cannot overlap.
    assert gpu_nodes and not gpu_nodes & cpu_nodes, (gpu_nodes, cpu_nodes)
    print(f"GPU tasks ran on {len(gpu_nodes)} node(s) of their own")
```

## Step 6: Benchmark the callset against GIAB

`rtg vcfeval` compares each sample's calls with that sample's GIAB v4.2.1 benchmark, over the region
and inside the benchmark regions (`--evaluation-regions`), requiring genotypes to match (vcfeval's
default). Calls and truth are both split into SNPs and indels, after splitting multi-allelic
records, before matching. The table reports vcfeval's unthresholded `None` row, because the
best-F-measure threshold is fitted to the truth set being scored.

These numbers check that the pipeline runs end to end and scores each sample against its own truth.
They are not a GATK benchmark: the reads are reference-selected (Step 4), which makes precision read
high against a whole-genome run; the SNP thresholds on indels lower indel recall; and scoring covers
one region of one chromosome, with one known-sites resource for BQSR.
[`PIPELINE.md`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/pipeline/PIPELINE.md#bounds-on-the-numbers)
has the details.

Measured on prod, 2026-09-25, on the AWS compute config; the cell prints the same values.

`standard`, chr20:1,000,000-11,000,000:

| sample | SNP precision | SNP recall | SNP F1 | indel precision | indel recall | indel F1 |
|---|---|---|---|---|---|---|
| HG002 | 0.9965 | 0.9904 | 0.9934 | 0.9590 | 0.9414 | 0.9501 |
| HG003 | 0.9972 | 0.9914 | 0.9943 | 0.9498 | 0.9326 | 0.9411 |
| HG004 | 0.9980 | 0.9884 | 0.9932 | 0.9563 | 0.9439 | 0.9501 |

`full`, all of chr20, from the job ("Run it as a job" below):

| sample | SNP precision | SNP recall | SNP F1 | indel precision | indel recall | indel F1 |
|---|---|---|---|---|---|---|
| HG002 | 0.9947 | 0.9898 | 0.9922 | 0.9584 | 0.9425 | 0.9503 |
| HG003 | 0.9935 | 0.9905 | 0.9920 | 0.9525 | 0.9376 | 0.9450 |
| HG004 | 0.9941 | 0.9900 | 0.9920 | 0.9514 | 0.9378 | 0.9446 |

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

## Step 7: Run the GPU step again as a Ray Data job

`ANNOTATE_VARIANTS`, the pipeline's GPU process, scored the joint callset's PASS variants one shard
per task. The next cell runs the same `VariantScorer` class, from `pipeline/bin/score_variants.py`,
over the whole callset as one Ray Data job on the same L4s. It checks that both paths scored the
same variants, then prints the largest difference between them and the five highest scores.

The score is a zero-shot embedding distance.
[Nucleotide Transformer v2 (50M, multi-species)](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species),
pinned in the image, embeds the 1 kb of reference around each variant with and without the alternate
allele, and the score is the L2 distance between the two mean-pooled embeddings. It is not a
likelihood score, it sees neither reads nor genotypes, and nothing validates it against
pathogenicity, function or call quality. The model reads 6-mer tokens, so an indel whose length is
not a multiple of six re-tokenizes the rest of the window and scores well above SNPs: expect indels
at the top of the printout. The step is here to show a GPU process in the DAG and Ray Data on the
same GPUs. Don't use the score to rank variants.

Measured at `standard` on prod, 2026-09-25: `ANNOTATE_VARIANTS` ran as 8 shards, and the cell scored
the same 25,197 PASS variants with two L4 actors in 116 s; the two paths agreed to within 2.88e-05.
As the actors start, Ray 2.58 logs an error-level advisory ("constructor arguments in the object
store and max_restarts > 0") from Ray Data itself; it does not affect the result.


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

    # Two actors, one L4 each. The error-level "constructor arguments in the object store" line
    # Ray 2.58 logs as they start is Ray Data's own: every actor-pool map gets its transform by
    # ObjectRef, which the dataset holds until it finishes, so a restart can still read it.
    # max_restarts=0 would silence the line and lose restarts.
    scored = ray.data.from_items(rows).map_batches(
        sv.VariantScorer,
        fn_constructor_kwargs={"reference": str(REFERENCE)},
        batch_size=64, batch_format="numpy", num_gpus=1,
        compute=ray.data.ActorPoolStrategy(size=2),
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

`/mnt/cluster_storage` is shared by the nodes of one cluster and is gone once that cluster
terminates. In a workspace that is fine: the cluster is yours and stays up. A job's cluster
terminates when the job ends, so results left there are lost even when the run succeeds.

Every process that produces a result declares a `publishDir`, so `--outdir` holds the declared
outputs and none of the intermediates.
[`persist_outputs.py`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/persist_outputs.py)
copies that tree to `NF_RAY_RESULTS`, or to the first writable durable mount (`/mnt/user_storage`,
then `/mnt/shared_storage`), and prints how many files it copied and where.


```python
run(["python", TEMPLATE_DIR / "persist_outputs.py", "--results", RESULTS])
```

## Run it as a job

[`job.yaml`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/job.yaml)
runs the same pipeline over all of chr20 (`--scale full`: 48 intervals, 144 HaplotypeCaller tasks)
and persists the results, in one entrypoint. Submit it from the template directory:

```bash
anyscale job submit --config-file job.yaml
```

Its inline compute config has the notebook's node types, with up to 8 CPU workers and 2 L4 workers.
On prod, 2026-09-25, the job took about 47 min from submission to persisted results, of which the
pipeline took 44m07s (Step 1 has its task count and CPU hours). g6.2xlarge returned
`InsufficientInstanceCapacity` twice over 13 seconds before launching. `persist_outputs.py` copied
173 files to `/mnt/user_storage/nextflow-genomics-on-ray/results/chr20-trio`, and Step 6 has the
scores. Cost was not measured.

`max_retries` is 0 because a job-level retry starts a new cluster with an empty work directory.
Retries happen per task, inside the run.

## Next steps

### Point it at your own reads

The samplesheet takes `sample,fastq_1,fastq_2`, plus `truth_vcf,truth_bed` for a sample you want
scored; a row without truth columns is called and not scored. Before you run:

- Known sites, truth sets and the reference must agree on contig names (`chr20` or `20`). A
  mismatched known-sites file stops GATK at BQSR, and a mismatched truth set stops `RTG_VCFEVAL`.
- `--region` and `--intervals` override the scale presets. Nothing checks up front that your reads
  cover the region; the benchmark at the end shows it.
- The ceiling must fit a node you have. `conf/base.config`, `conf/ray.config` and the compute config
  are one decision; `tests/nextflow-genomics-on-ray/test_config_agreement.py` checks they agree.

### Run your own pipeline

On this template's image, which carries the plugin, `nf-ray` and the tools, add the `ray` profile
from
[`nextflow.config`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/pipeline/nextflow.config)
and include
[`conf/ray.config`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/pipeline/conf/ray.config),
keeping the plugin pinned as `nf-ray@0.1.0`: without a version, Nextflow looks for nf-ray in the
plugin registry, where it isn't. Run `nf-ray doctor` and `smoke.nf` first. Only this template's
pipeline has run this way; an nf-core pipeline's biocontainer directives need one of the routes
under "Containers".

### Read the executor

[`RayExecutor.groovy`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/nf-ray-plugin/src/main/groovy/ai/anyscale/nfray/RayExecutor.groovy)
is the whole Nextflow side, about 400 lines with comments; it calls the `nf-ray` CLI rather than
Ray's Java API, which Ray documents as experimental.
[`nf_ray/resources.py`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/nf_ray/resources.py)
maps directives onto Ray resources,
[`nf_ray/daemon.py`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/nf_ray/daemon.py)
owns the tasks, and
[`nf_ray/errors.py`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/nf_ray/errors.py)
maps Ray failures to exit codes.
