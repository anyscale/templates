# Run Nextflow genomics pipelines on Ray

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/nextflow-genomics-on-ray"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/nextflow-genomics-on-ray" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: about 20 min at the default `standard` scale, 10 min at `quick` (derived from the measured step times, not timed end to end)

This template runs a Nextflow pipeline on an autoscaling Anyscale cluster through a Ray executor,
`executor 'ray'`. The pipeline is GATK germline short-variant calling in the shape of
[nf-core/sarek](https://nf-co.re/sarek): fastp, BWA-MEM2, MarkDuplicates, BQSR, HaplotypeCaller
scattered by interval, GenomicsDBImport, GenotypeGVCFs and hard filters, scored with `rtg vcfeval`
against GIAB v4.2.1. You end with two callsets benchmarked per sample, the hard-filtered joint
callset and a CNN-filtered one, a chart of which node ran each task and how long it queued, and a
`ray` profile to try on your own pipeline.

The samples are the GIAB Ashkenazi trio, HG002, HG003 and HG004, over a region of chr20, called
jointly as three samples. No pedigree is used: no PED file, no Mendelian check. The only
trio-derived input is GIAB's `_noinconsistent` benchmark-regions BED.

One GPU process shares the DAG. Each sample's GVCFs are also genotyped on their own and scored on
an L4 by GATK's NVScoreVariants, the PyTorch port of CNNScoreVariants, with its 1D model; then
FilterVariantTranches filters them against HapMap, 1000G and Mills, as sarek filters a
single-sample run. vcfeval scores those calls beside the hard-filtered joint callset.

Outside a workspace created from this template, work from `templates/nextflow-genomics-on-ray` in a
clone of [anyscale/templates](https://github.com/anyscale/templates).

## How `executor 'ray'` differs from your executor

`executor 'ray'` is a grid executor, like Nextflow's SLURM and PBS ones: it writes `#RAY` directives
where SLURM writes `#SBATCH`. Ray places each task on a node with room for its request, and the
autoscaler adds nodes, up to `max_nodes`, when none has any; the L4 group starts at zero and comes
up when the DAG reaches the GPU process. AWS Batch and Seqera scale too; here CPU and GPU processes
share one cluster, and the work directory is shared POSIX storage rather than S3.

| Directive | Under `executor 'ray'` |
|---|---|
| `cpus`, `memory` | `num_cpus` and `memory`, for placement only: Ray caps neither. When a node runs out of memory, Ray's memory monitor kills a task on it |
| `accelerator` | `num_gpus` plus `accelerator_type` (`nvidia-l4` becomes `L4`); with no type, `ray.defaultAccelerator` |
| `time` | recorded, not enforced |
| `errorStrategy`, `maxRetries` | Nextflow's, unchanged; Ray itself retries nothing |
| `container` | not honoured; see "Containers" |

The pipeline's `bin/` is copied into the work directory at start-up, since workers can't see the
project directory.

A request that no node type can satisfy stays pending on Ray instead of failing. So
`conf/ray.config` declares the largest request a node can schedule, and the executor refuses
anything bigger at submit time. That is less than the node has: Ray 2.58.0 offers about 70% of free
memory as `memory`, so a 128 GiB worker schedules roughly 85 GiB, and `conf/base.config` clamps
requests to 80 GB.

When Ray fails around a task, the executor writes the task's `.exitcode` in nf-core's retry band:
137 for a memory-monitor kill, 175 for a lost node. Both still count against `maxRetries`.

### Containers

A Ray worker is already a container, with no runtime to start another in, so this template's modules
drop nf-core's `container` directives and the image carries the tools. An unedited pipeline has two
routes, neither tested here: `-profile conda`, which builds each environment once on shared storage,
or `ext.image` mapped through `NF_RAY_IMAGE_MAP` to an image built on this cluster's base. Ray
refuses a task image whose Ray or Python version differs from the cluster's;
`nf-ray probe-image <uri>` checks one.

## Measured runs, and what differs from sarek

| Scale | Region of chr20 | Intervals | Run by | Tasks | Wall time | CPU h | Workers | SNP F1 | Indel F1 |
|---|---|---|---|---|---|---|---|---|---|
| `quick` | 1-3 Mb | 8 | CI, GPU off | 81 | 4m47s, 4m52s (2 runs) | 1.4 | 4 CPU | 0.9843-0.9930 | 0.9441-0.9514 |
| `standard` | 1-11 Mb | 24 | this notebook | 171 | 13m22s | 4.6 | 4 CPU, 2 L4 | 0.9932-0.9943 | 0.9411-0.9501 |
| `full` | all | 48 | `job.yaml` | 299 | 44m07s | 25.1 | up to 6 CPU, 2 L4 | 0.9920-0.9922 | 0.9446-0.9503 |

Measured on Anyscale on AWS, 2026-09-25: an m5.2xlarge head with `CPU: 0`, r6i.4xlarge CPU workers and
g6.2xlarge L4 workers. Wall time is the pipeline's, CPU hours are Nextflow's count, and F1 is the
hard-filtered callset's range over the three samples, each against its own GIAB v4.2.1 benchmark;
Step 6 has both callsets. Cost was not measured. Set `NF_DEMO_SCALE` before Jupyter starts to pick a
scale, and `NF_CNN=false` to skip the CNN arm and its GPU process.

The scores show the pipeline working end to end against GIAB's truth sets, within these limits:

- The reads are reference-selected: pairs whose primary alignment in GIAB's NHGRI Illumina 300x
  GRCh38 BAM falls in the region, subsampled to about 30x and realigned here against chr20 alone.
  Reads a whole-genome run would misplace into chr20 are mostly absent, so precision reads high.
- Hard filters replace sarek's VQSR on the joint callset. VQSR fits a model to the callset's own
  annotations, and GATK recommends it for at least one whole genome or about 30 exomes; a region of
  chr20 has a fraction of that. SNPs and indels are filtered separately, each with GATK's
  thresholds for its type.
- The CNN callset is single-sample calls, because the model was trained on them, so it differs from
  the joint callset in calling mode as well as in filter. Its tranche cutoffs are set by the
  resource sites inside the region, a few hundred indels per sample at `quick`, so they are coarser
  than the ones GATK tuned on whole genomes.
- BQSR gets dbSNP 138 alone, where sarek adds Mills and a known-indels set.
- The callset covers one region of one chromosome, so none of these numbers compares with a
  genome-wide sarek benchmark.

The processes are adapted from [nf-core/modules](https://github.com/nf-core/modules) (MIT);
[`pipeline/PIPELINE.md`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/pipeline/PIPELINE.md)
has each one's provenance, the filter thresholds, the CNN arm's choices and the vcfeval settings.

## Step 1: Check the cluster and the toolchain

The cell runs `nf-ray doctor`, which checks the Python and Ray versions, shared storage for the work
directory, and every tool the pipeline calls: look for `no problems found`. Doctor reads only
`NF_RAY_*` environment variables, so its `declared ceiling` line says `none`; Step 2 prints the
ceiling the executor uses.


```python
import csv
import json
import os
import pathlib
import subprocess

# CI sets `quick`. The scales differ only in region size and how finely it is split.
SCALE = os.getenv("NF_DEMO_SCALE", "standard")
SCALES = {
    "quick":    {"region": "chr20:1000000-3000000",  "span": "2 Mbp"},
    "standard": {"region": "chr20:1000000-11000000", "span": "10 Mbp"},
}
if SCALE not in SCALES:
    raise ValueError(f"NF_DEMO_SCALE must be one of {sorted(SCALES)}, got {SCALE!r}")
CFG = SCALES[SCALE]
# The CNN arm, which is the GPU process. CI turns it off.
CNN = os.getenv("NF_CNN", "true").lower() in ("1", "true", "yes")

# Son, father and mother, called jointly as three samples.
SAMPLES = ["HG002", "HG003", "HG004"]

TEMPLATE_DIR = pathlib.Path.cwd()
PIPELINE = TEMPLATE_DIR / "pipeline"
DATA_URI = f"s3://anyscale-public-materials/genomics/giab-trio-illumina-chr20/{SCALE}"

# Shared storage, because tasks read their inputs by path from whichever node wrote them.
# NF_WORK is conf/ray.config's workDir.
WORK = pathlib.Path("/mnt/cluster_storage/nf-genomics")
NF_WORK = "/mnt/cluster_storage/nf-work"
DATA_DIR, RESULTS, SMOKE = WORK / "data" / SCALE, WORK / "results" / SCALE, WORK / "smoke"
for d in (DATA_DIR, RESULTS, SMOKE):
    d.mkdir(parents=True, exist_ok=True)


def run(cmd, **kwargs):
    """Run a command, streaming output, and raise if it fails, which `!` does not."""
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


print(f"scale       {SCALE}  ({CFG['span']} of {CFG['region']})")
print(f"samples     {', '.join(SAMPLES)}")
print(f"cnn arm     {CNN}")
print(f"data        {DATA_URI}\n")
run(["nf-ray", "doctor", "--work-dir", NF_WORK])
```

## Step 2: Read the resolved config

The cell prints what `-profile ray` resolves for scheduling: nf-core's label requests and the
`resourceLimits` ceiling from `conf/base.config`, and the executor and `ray` settings from
`conf/ray.config`, which holds everything Ray-specific. `ray.maxNodeMemoryGb` should equal the
memory in `process.resourceLimits`.


```python
resolved = subprocess.run(
    ["nextflow", "config", "-flat", "-profile", "ray", str(PIPELINE)],
    check=True, capture_output=True, text=True,
).stdout.splitlines()

# `-flat` quotes selector keys (process.'withLabel:process_medium'.cpus), hence the quote.
# The assert catches a prefix that matches nothing.
for prefix in ("process.executor", "process.resourceLimits", "process.'withLabel:process_",
               "executor.", "ray.", "workDir"):
    shown = [line for line in resolved
             if line.startswith(prefix) and "errorStrategy" not in line and "time" not in line]
    assert shown, f"`nextflow config -flat` printed nothing under {prefix!r}"
    print("\n".join(shown))
```

## Step 3: Run a smoke test

`smoke.nf` fans out eight shards and gathers them through `pipeline/bin/smoke_collect.sh`, calling
no genomics tool, so a failure here is the executor's or the cluster's. The cell checks every
shard's checksum and prints how many nodes the shards used; in each of three measured runs it took
36 s on two nodes.


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

The cell syncs the scale's data from a public bucket, and `make_samplesheet.py` verifies the
checksums in `MANIFEST.json`, gives each sample its own truth set, and lists the tranche resources
the CNN arm filters against. The reference is chr20 of the GRCh38 no-alt analysis set, and
[`tools/stage-demo-data.sh`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/tools/stage-demo-data.sh)
builds every file from public sources.


```python
# Unsigned: the bucket is public, and a signed request can fail on the node role's policy.
run(["aws", "s3", "sync", "--no-sign-request", "--only-show-errors", f"{DATA_URI}/", DATA_DIR])

SAMPLESHEET = DATA_DIR / "samplesheet.csv"
RESOURCE_LIST = DATA_DIR / "tranche_resources.txt"
run(["python", PIPELINE / "bin" / "make_samplesheet.py",
     "--data-dir", DATA_DIR, "--output", SAMPLESHEET, "--resources-output", RESOURCE_LIST])

manifest = json.loads((DATA_DIR / "MANIFEST.json").read_text())
print(f"\n{manifest['region']}   {manifest['platform']}   {manifest['truth_version']}")
print(f"{'sample':<8}{'read pairs':>12}")
for entry in manifest["samples"]:
    print(f"{entry['id']:<8}{entry['read_pairs']:>12,}")
REFERENCE = DATA_DIR / manifest["reference"]["fasta"]
KNOWN_SITES = DATA_DIR / manifest["known_sites"]["vcf"]
# HapMap, 1000G and Mills, comma-separated, as main.nf's --tranche_resources takes them.
TRANCHE_RESOURCES = RESOURCE_LIST.read_text().strip()
if CNN and not TRANCHE_RESOURCES:
    raise RuntimeError(f"{DATA_URI} lists no tranche resources; NF_CNN=false skips the CNN arm")
print("tranche resources:", TRANCHE_RESOURCES.replace(",", ", ") or "none")
```

## Step 5: Call variants for the three samples

The cell runs `main.nf` under `-profile ray` with the scale's preset region and interval count.
HaplotypeCaller runs per sample and interval, 72 tasks at `standard`, and at 6 CPUs and 36 GB two
fit on a worker, so four CPU workers run at most eight at a time. With the CNN arm on, the three
`NVSCOREVARIANTS` tasks, one per sample, start the L4 workers once calling is done.


```python
run(["nextflow", "run", PIPELINE / "main.nf", "-profile", "ray",
     "--scale", SCALE,
     "--samplesheet", SAMPLESHEET,
     "--reference", REFERENCE,
     "--known_sites", KNOWN_SITES,
     "--cnn", str(CNN).lower(),
     "--outdir", RESULTS]
    + (["--tranche_resources", TRANCHE_RESOURCES] if CNN else []))
```

The next cell joins the executor's placement record to `trace.txt` and draws a bar per task,
coloured by node, with queue time, node boot included, in grey. At `standard` it drew 170
tasks on 6 nodes, the GPU tasks on 2 of their own: one fewer than Nextflow's count, because
`COLLECT_PLACEMENT`, which copies the record, is left out.


```python
import matplotlib.pyplot as plt

with open(RESULTS / "pipeline_info" / "trace.txt") as handle:
    trace = {row["native_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
with open(RESULTS / "pipeline_info" / "nf_ray_placement.tsv") as handle:
    placed = {row["work_dir"]: row for row in csv.DictReader(handle, delimiter="\t")}

# Keyed on work dir, so earlier runs' rows cannot match. COLLECT_PLACEMENT copied the record
# while still running, so it has no row; every other traced task must have one.
COLLECTOR = "COLLECT_PLACEMENT"
tasks = [t for t in trace.values() if t["process"] != COLLECTOR]
assert len(trace) - len(tasks) == 1, f"expected exactly one {COLLECTOR} task in the trace"
missing = [t["name"] for t in tasks if t["workdir"] not in placed]
assert not missing, f"{len(missing)} traced task(s) have no placement row, e.g. {missing[:3]}"
rows = [(placed[t["workdir"]], t) for t in tasks]
# A task that never started has `started` 0.000; sort it by submit time.
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
if CNN:
    gpu_nodes = {p["node_id"] for p, t in rows if t["process"].endswith("NVSCOREVARIANTS")}
    cpu_nodes = {p["node_id"] for p, t in rows if t["process"].endswith("HAPLOTYPECALLER")}
    # GPU tasks fit only the L4 group, and HaplotypeCaller's 36 GB fits no L4 node.
    assert gpu_nodes and not gpu_nodes & cpu_nodes, (gpu_nodes, cpu_nodes)
    print(f"GPU tasks ran on {len(gpu_nodes)} node(s) of their own")
```

## Step 6: Benchmark the callsets against GIAB

The cell prints `rtg vcfeval`'s unthresholded precision, recall and F1 for each sample, callset and
variant type: `gatk_hard` is the hard-filtered joint callset and, with the CNN arm on, `gatk_cnn` the
CNN-filtered single-sample calls. They differ in calling mode as well as filter, so not all of a
recall difference is the filter's: joint genotyping keeps sites that one sample's calls alone leave
below QUAL 30. At `standard`, in the measured run:

| sample | SNP precision | SNP recall | SNP F1 | indel precision | indel recall | indel F1 |
|---|---|---|---|---|---|---|
| HG002 | 0.9965 | 0.9904 | 0.9934 | 0.9590 | 0.9414 | 0.9501 |
| HG003 | 0.9972 | 0.9914 | 0.9943 | 0.9498 | 0.9326 | 0.9411 |
| HG004 | 0.9980 | 0.9884 | 0.9932 | 0.9563 | 0.9439 | 0.9501 |


```python
import pandas as pd

bench = pd.read_csv(RESULTS / "benchmark" / "benchmark.tsv", sep="\t")
CALLSETS = {"gatk_hard", "gatk_cnn"} if CNN else {"gatk_hard"}
# One row per sample, callset and variant type.
assert len(bench) == len(SAMPLES) * len(CALLSETS) * 2, bench
assert set(bench["sample"]) == set(SAMPLES), bench["sample"].unique()
assert set(bench["caller"]) == CALLSETS, bench["caller"].unique()

table = bench.pivot_table(index=["sample", "caller"], columns="variant_type",
                          values=["precision", "recall", "f1"])
print(table.round(4).to_string())
```

## Step 7: Persist the outputs

`/mnt/cluster_storage` goes with the cluster, and a job's cluster goes when the job ends. The cell
copies `--outdir`, which holds the published outputs and no intermediates, to `NF_RAY_RESULTS` or
the first writable of `/mnt/user_storage` and `/mnt/shared_storage`.


```python
run(["python", TEMPLATE_DIR / "persist_outputs.py", "--results", RESULTS])
```

## Run it as a job

[`job.yaml`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/job.yaml)
runs the pipeline at `full` and persists the results. Submit it from the template directory:

```bash
anyscale job submit --config-file job.yaml
```

Its inline compute config allows up to 8 CPU and 2 L4 workers. In the measured run the job took about 47 min
from submission to persisted results and copied 173 files to
`/mnt/user_storage/nextflow-genomics-on-ray/results/chr20-trio`. `max_retries` is 0 because a
job-level retry starts a new cluster with an empty work directory; retries happen per task.

## Bring your own reads or pipeline

The samplesheet takes `sample,fastq_1,fastq_2`, plus `truth_vcf,truth_bed` for a sample you want
scored; a row without them is called and not scored. The CNN arm's tranche resources are
`--tranche_resources`, comma-separated VCFs beside their indexes, and `--cnn false` runs without
them. The reference, known sites, tranche resources and truth sets must agree on contig names
(`chr20` or `20`), or GATK stops at BQSR or FilterVariantTranches and `RTG_VCFEVAL` fails. `--region`
and `--intervals` override the scale presets, and nothing checks up front that your reads cover the
region. If you change the compute configs, refit both ceilings to what the largest node schedules;
`test_config_agreement.py` checks that the three agree.

For your own pipeline, on this template's image, which carries the plugin and `nf-ray`, copy
[`conf/ray.config`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/pipeline/conf/ray.config)
and the `ray` profile from
[`nextflow.config`](https://github.com/anyscale/templates/blob/main/templates/nextflow-genomics-on-ray/pipeline/nextflow.config),
keeping the plugin pinned as `nf-ray@0.1.0`: unpinned, Nextflow looks for nf-ray in the plugin
registry, where it isn't. Size the ceiling the same way, and run `nf-ray doctor` and `smoke.nf`
first. Only this template's pipeline has run this way.

The executor is
[`nf-ray-plugin/`](https://github.com/anyscale/templates/tree/main/templates/nextflow-genomics-on-ray/nf-ray-plugin)
on the Nextflow side and
[`nf_ray/`](https://github.com/anyscale/templates/tree/main/templates/nextflow-genomics-on-ray/nf_ray)
on the Ray side, where a daemon on the head node owns every task.
